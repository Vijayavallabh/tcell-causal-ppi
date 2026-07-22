"""Architecture search for the typed graph encoder, scored on a TARGET-GROUPED INNER HOLDOUT of train.

    OMP_NUM_THREADS=4 PYTHONPATH=src uv run python -m tcell_pipeline.arch_search \
        --cells 0,1,2 --epochs 3 --device cuda --out data/results/arch_search

WHY THIS EXISTS. The typed encoder aggregates each relation with an UNNORMALISED sum
(``MessagePassing(aggr="add")``) and then sums the relations together, while ``UntypedGraphEncoder``
uses ``GCNConv``, which applies symmetric ``1/sqrt(d_i d_j)``. On the real graph
``functional_assoc`` is 6,857,702 of 7,980,907 edges (86%) at a median score of 0.228 — below STRING's
own "medium confidence" floor of 0.4. So the least reliable evidence class dominates every node update
by sheer degree, and nothing in the typed model can down-weight it. That predicts the measured ordering
the project has been reading as a biological negative:

    untyped_gnn   +0.0045 vs no-graph   (normalised)      BEST graph variant
    typed_static  -0.0131 vs no-graph   (unnormalised)    WORST, survives Bonferroni

This searches the two knobs that isolate that explanation (per-relation normalisation, edge-confidence
pruning) plus a learnable per-relation scale.

VAL IS NEVER TOUCHED. Selection runs on ``training.inner_split.target_grouped_subsets`` — a
target-grouped holdout carved from TRAIN, because one target spans ~3 rows and a random row split
leaks. The winner is confirmed on val ONCE, with 5 paired seeds, in a separate run. The sealed
challenge split is not opened here or anywhere else.

Writes only under ``--out``; no frozen artifact is touched and ``promoted.json`` is not consulted.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.baselines.graph_baselines import (  # noqa: E402
    AugmentedUntypedEncoder,
    StaticTypedGraphEncoder,
    UntypedGraphEncoder,
)
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder  # noqa: E402
from tcell_pipeline.model import EGIPGModel  # noqa: E402
from tcell_pipeline.screening.screening import collect_truth, screen_config  # noqa: E402
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402
from tcell_pipeline.training.inner_split import target_grouped_subsets  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# THE PRE-REGISTERED GRID. Written down before any cell ran; every cell is reported, including losers.
# ---------------------------------------------------------------------------------------------------
NORMS = ("add", "mean", "gcn")            # add == the current encoder (the control)
THRESHOLDS = (0.0, 0.4, 0.7)              # STRING's own confidence bands: all / medium / high
REL_SCALE = (False,)                      # stage 2 adds True on the stage-1 winner
ARMS = ("condition_gated",)               # stage 2 adds typed_static


# The no-graph and untyped-GCN REFERENCES, run on the SAME inner holdout so the 9 typed cells can be
# read against them. This is the comparison the whole project's negative rests on, measured for the
# FIRST time with LIVE gates (lambda_graph=0): the 5-seed campaign compared these arms to typed models
# whose gates were dead. norm/thr are inert for these arms and fixed only to give a clean cell_id.
REFERENCE_CELLS = (
    {"arm": "expression_only", "norm": "none", "functional_min_score": 0.0, "rel_scale": False},
    {"arm": "untyped_gnn", "norm": "gcn", "functional_min_score": 0.0, "rel_scale": False},
)

# STAGE 2 — improve ON untyped_gnn by adding back the two things its plain GCN discards: the per-edge
# STRING confidence (GATv2 edge-aware attention, or as a GCN edge weight) and learned neighbour
# competition (attention vs fixed 1/sqrt(d_i d_j)). Untyped by construction (no gates), so these are
# graph-PERFORMANCE candidates, not rationale models. `conv`/`layers`/`heads` are read by _encoder.
#
# MEMORY forces the graph size for GAT: on the full graph its multi-head attention over 74k-86k-edge hub
# subgraphs OOMs an 80 GB card in the training loop (donor-invariance re-forwards multiply it). Measured
# worst-case-hub single-forward+backward: heads=4/full = OOM, heads=2/full = 23 GB, heads=2/thr0.4 =
# 12 GB. So GAT runs on the thr=0.4 graph at heads=2, and untyped_gnn_p (plain GCN, same thr=0.4 graph)
# is included as the FAIR baseline — GAT-vs-GCN must be on the same graph to attribute a gain to
# attention. wgcn is cheap and runs on the FULL graph, the clean "untyped_gnn + edge weights" contrast.
STAGE2_CELLS = (
    {"arm": "untyped_wgcn", "conv": "wgcn", "layers": config.GRAPH_LAYERS, "heads": 1,
     "norm": "wgcn", "functional_min_score": 0.0, "rel_scale": False},          # full graph + edge weights
    {"arm": "untyped_gnn_p", "conv": "gcn", "layers": config.GRAPH_LAYERS, "heads": 1,
     "norm": "gcn", "functional_min_score": 0.4, "rel_scale": False},           # fair GCN baseline @0.4
    {"arm": "untyped_gat", "conv": "gat", "layers": config.GRAPH_LAYERS, "heads": 2,
     "norm": "gat", "functional_min_score": 0.4, "rel_scale": False},           # learned attention @0.4
)


def grid() -> list[dict]:
    cells = []
    for arm, norm, thr, scale in itertools.product(ARMS, NORMS, THRESHOLDS, REL_SCALE):
        cells.append({"arm": arm, "norm": norm, "functional_min_score": thr, "rel_scale": scale})
    return cells


def cell_id(c: dict) -> str:
    base = f"{c['arm']}__norm-{c['norm']}__thr-{c['functional_min_score']}__scale-{int(c['rel_scale'])}"
    return base + (f"__L{c['layers']}" if "layers" in c else "")


def _encoder(cell: dict, graph, g2i):
    # The same encoder classes nested_family_factories uses, so the reference arms are built identically
    # to the canonical family. expression_only has no graph encoder (None -> expression-only EGIPGModel).
    arm = cell["arm"]
    if arm in ("untyped_gat", "untyped_wgcn", "untyped_gnn_p"):
        return AugmentedUntypedEncoder(graph, g2i, layers=cell.get("layers", config.GRAPH_LAYERS),
                                       conv=cell["conv"], heads=cell.get("heads", config.GRAPH_N_HEADS))
    if arm == "expression_only":
        return None
    if arm == "untyped_gnn":
        return UntypedGraphEncoder(graph, g2i)
    if arm == "typed_static":
        return StaticTypedGraphEncoder(graph, g2i)
    return TypedGraphEncoder(graph, g2i, norm=cell["norm"], rel_scale=cell["rel_scale"])


def _edge_counts(graph) -> dict:
    from tcell_pipeline.graph.typed_graph_encoder import _PP_RELATIONS
    from tcell_pipeline.graph.graph_builder import PROTEIN
    return {r: int(graph[PROTEIN, r, PROTEIN].edge_index.size(1)) for r in _PP_RELATIONS}


def run_cell(cell: dict, *, epochs: int, device: str, out: Path, seed: int, batch_size: int,
             holdout_frac: float, n_max=None) -> dict:
    t0 = time.perf_counter()
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, g2i = build_hetero_graph(functional_min_score=cell["functional_min_score"])
    edges = _edge_counts(graph)

    train_full = PerturbationDataset("train", n_max=n_max)
    inner_train, inner_hold = target_grouped_subsets(train_full, holdout_frac, seed)
    sym = train_full.pc["hgnc_symbol"].astype(str)
    g_tr = {sym.iloc[i] for i in inner_train.indices}
    g_ho = {sym.iloc[i] for i in inner_hold.indices}
    assert not (g_tr & g_ho), "inner split leaked a target gene across both sides"

    cid = cell_id(cell)
    root = out / cid
    cfg = {"name": cell["arm"], "seed": seed, "n_epochs": epochs, "batch_size": batch_size,
           "lambda_graph": 0.0,  # the penalty annihilates the gates; option C, decided 2026-07-21
           "donor_invariance": config.DONOR_INVARIANCE,
           "model_factory": lambda: EGIPGModel.from_saved_basis(
               gene_names, graph_encoder=_encoder(cell, graph, g2i))}
    print(f"[arch] {cid}: edges={edges} inner_train={len(inner_train):,}/{len(g_tr):,}g "
          f"inner_holdout={len(inner_hold):,}/{len(g_ho):,}g VAL UNTOUCHED", flush=True)

    res = screen_config(cfg, inner_train, inner_hold, collect_truth(inner_train)["delta_z"].mean(0),
                        device=device, split="inner_holdout", predictions_root=root / "pred",
                        screening_root=root, registry_path=None)

    hist = json.loads((root / cell["arm"] / str(seed) / "logs" / "stage_a_history.json").read_text())
    gates = [e["train"]["gate_mean"] for e in hist]
    # expression_only / untyped_gnn emit no edge gates, so gate_mean is None every epoch — None means
    # "no gates", never "collapsed gates". Keep it None in the artifact and render it as a dash.
    g0, gN = gates[0], gates[-1]
    out_row = {**cell, "cell_id": cid, "systema": res["systema"], "pearson": res["pearson"],
               "mae": res["mae"], "epochs_run": res["epochs_run"], "best_val": res["best_val"],
               "gate_mean_first": g0, "gate_mean_last": gN,
               "edges": edges, "n_inner_train": len(inner_train), "n_inner_holdout": len(inner_hold),
               "hours": (time.perf_counter() - t0) / 3600.0}
    (out / f"{cid}.json").write_text(json.dumps(out_row, indent=2, default=float))
    fmt = lambda g: f"{g:.4f}" if g is not None else "  —  "
    print(f"[arch] {cid}: systema={res['systema']:.6f} epochs={res['epochs_run']} "
          f"gate {fmt(g0)}->{fmt(gN)} {out_row['hours']:.2f}h", flush=True)
    return out_row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default=None, help="comma-separated indices into the grid (default: all)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-max", type=int, default=None, help="cap train rows — SMOKE ONLY, not a result")
    ap.add_argument("--out", default="data/results/arch_search")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--refs", action="store_true",
                    help="run the no-graph + untyped-GCN REFERENCE cells (not the grid), so the typed "
                         "cells can be read against a no-graph baseline on the SAME inner holdout")
    ap.add_argument("--stage2", action="store_true",
                    help="run the STAGE-2 cells that improve on untyped_gnn (GATv2 edge-attention, "
                         "score-weighted GCN, deeper GAT) on the same inner holdout")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    torch.set_num_threads(1)
    if a.device.startswith("cuda"):
        # MUST precede the first CUDA allocation (PyTorch reads this at CUDA init). Subgraphs vary in
        # size every step, so without expandable segments the caching allocator fragments and creeps
        # upward until it OOMs hours in. run_screening.run() does this; calling screen_config directly
        # bypassed it, and on 2026-07-22 a cell climbed to 79.7 GB of 81.9 GB before this was found.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cells = list(STAGE2_CELLS) if a.stage2 else list(REFERENCE_CELLS) if a.refs else grid()
    if a.list:
        for i, c in enumerate(cells):
            print(f"{i:3d}  {cell_id(c)}")
        return 0
    chosen = [int(x) for x in a.cells.split(",")] if a.cells else list(range(len(cells)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.n_max is not None:
        print(f"[arch] ** SMOKE: train capped to {a.n_max} rows — NOT a search result **")

    for i in chosen:
        run_cell(cells[i], epochs=a.epochs, device=a.device, out=out, seed=a.seed,
                 batch_size=a.batch_size, holdout_frac=a.holdout_frac, n_max=a.n_max)
    return 0


if __name__ == "__main__":
    sys.exit(main())
