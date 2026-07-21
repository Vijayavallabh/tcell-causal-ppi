"""Stage-1 pilot: does the condition gate survive when L_graph is not ~100x the task?

Phase 1 established that `L_graph` — an unnormalised `sum_{e in E}` over 16k-60k edges — contributes
~100x the response loss and takes ~99.9% of every clipped gradient step, annihilating the edge gates
inside epoch 0 in all 5 seeds (docs/h1-optimization-notes.md). This pilot tests the mechanism directly by
running the SAME arm under three penalty scalings and watching the gates:

  baseline    L_graph / batch_size, lambda_graph=0.01   (exactly what the screening campaign ran)
  zero        lambda_graph=0                            (clean control: no penalty at all)
  normalised  L_graph / |E|,        lambda_graph=0.01   (the spec's formula, per EDGE not per SAMPLE)

KILL CRITERION: if the gates still collapse under `zero`, the Phase-1 diagnosis is wrong and Phase 2
stops. This probe is built to be able to say that.

VAL IS NEVER TOUCHED. Training runs on a TARGET-GROUPED inner split of the train fold
(`training.inner_split`), and every number reported here is a training diagnostic or an inner-holdout
loss. A random row split would leak — one target spans ~3 rows — which is the error that produced a false
+0.0843 on a tabular bar here.

Writes nothing outside its own --out JSON: the epoch loop calls `Trainer._epoch` directly rather than
`Trainer.run()`, so no checkpoint or history is persisted anywhere.

    OMP_NUM_THREADS=4 PYTHONPATH=src uv run python -m tcell_pipeline.pilot_lambda_graph \\
        --mode zero --epochs 2 --device cuda:1 --out zero.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.probe_graph_gradients import (  # noqa: E402
    _first_known_target,
    _neighbourhood_dependence,
)
from tcell_pipeline.screening.screening import CONDITION_GATED, nested_family_factories  # noqa: E402
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402
from tcell_pipeline.training.inner_split import target_grouped_subsets  # noqa: E402
from tcell_pipeline.training.losses import StageALoss  # noqa: E402
from tcell_pipeline.training.trainer import Trainer, seeded_init  # noqa: E402

MODES = ("baseline", "zero", "normalised")


class EdgeNormalisedStageALoss(StageALoss):
    """`L_graph` divided by the EDGE COUNT instead of the batch size.

    `StageALoss._graph` divides the summed gate mass by `n` (batch size), so the penalty's magnitude is
    set by how many edges a target happens to have — 16k for a sparse gene, 60k for a hub, 88% of them
    STRING `functional_assoc`. Dividing by |E| makes `lambda_graph` mean "penalty per edge", so the same
    lambda means the same thing for a hub and a leaf, and the term stops scaling with annotation density.
    Everything else (the sparsity + unsourced-reliance form, the per-edge confidence weighting) is the
    spec's, unchanged."""

    def _graph(self, edge_gates, edge_confidences=None) -> torch.Tensor:
        if not edge_gates:
            return torch.zeros(())
        sparse = torch.zeros(())
        unsrc = torch.zeros(())
        n_edges = 0
        for rel, per_sample in edge_gates.items():
            confs = edge_confidences.get(rel) if edge_confidences else None
            for b, alpha in enumerate(per_sample):
                if alpha.numel() == 0:
                    continue
                n_edges += int(alpha.numel())
                sparse = sparse.to(alpha) + alpha.abs().sum()
                conf = confs[b].to(alpha) if confs is not None else torch.zeros_like(alpha)
                unsrc = unsrc.to(alpha) + ((1.0 - conf) * alpha.pow(2)).sum()
        return (self.graph_lambda_sparse * sparse + self.graph_lambda_unsrc * unsrc) / max(n_edges, 1)


def make_loss(mode: str, model):
    dec = model.decoder
    kw = dict(h_do_dim=dec.h_do_dim)
    if mode == "baseline":
        return StageALoss(dec.gene_dim, dec.program_dim, **kw)                      # lambda_graph = 0.01
    if mode == "zero":
        return StageALoss(dec.gene_dim, dec.program_dim, lambda_graph=0.0, **kw)
    return EdgeNormalisedStageALoss(dec.gene_dim, dec.program_dim, **kw)            # per-edge, 0.01


def gate_stats(model, batch, targets, conditions) -> dict:
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(batch, targets, conditions)
    finally:
        model.train(was)
    alphas = [a for per in (out["edge_gates"] or {}).values() for a in per if a.numel()]
    if not alphas:
        return {"n_edges": 0}
    a = torch.cat(alphas).float()
    return {"n_edges": int(a.numel()), "mean": float(a.mean()), "max": float(a.max()),
            "min": float(a.min()), "frac_below_1e-3": float((a < 1e-3).float().mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8, help="8 = the screening campaign's batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-max", type=int, default=None,
                    help="cap train rows — SMOKE TEST ONLY; a capped run is not a pilot result")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)  # many-core box: tiny per-subgraph GNN ops thrash the pool (README §perf)

    # Fail loudly and immediately on the wrong GPU. torch's default device enumeration is NOT
    # nvidia-smi's: on this box `cuda:4` is the T400 4GB while nvidia-smi calls that card index 3, so a
    # run dispatched to "the free A100 at index 4" silently landed on a 4 GB card and OOMed 6 minutes in,
    # after the dataset build. Export CUDA_DEVICE_ORDER=PCI_BUS_ID to make the two agree, and pin with
    # CUDA_VISIBLE_DEVICES. This check turns a mid-run OOM into a startup error naming the actual card.
    if a.device.startswith("cuda"):
        idx = int(a.device.split(":")[1]) if ":" in a.device else 0
        prop = torch.cuda.get_device_properties(idx)
        gib = prop.total_memory / 2**30
        print(f"[pilot] {a.device} resolves to {prop.name} with {gib:.1f} GiB "
              f"(CUDA_DEVICE_ORDER={__import__('os').environ.get('CUDA_DEVICE_ORDER', 'default')})")
        if gib < 16:
            raise SystemExit(
                f"refusing to run on {prop.name} ({gib:.1f} GiB): the typed encoder needs tens of GB at "
                f"batch {a.batch_size}. torch's device order differs from nvidia-smi's — set "
                f"CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to pin the card you meant.")

    t0 = time.perf_counter()
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, gene_to_idx = build_hetero_graph()
    train_full = PerturbationDataset("train", n_max=a.n_max)
    if a.n_max is not None:
        print(f"[pilot] ** SMOKE TEST: train capped to {a.n_max} rows — NOT a pilot result **")
    inner_train, inner_hold = target_grouped_subsets(train_full, a.holdout_frac, a.seed)
    sym = train_full.pc["hgnc_symbol"].astype(str)
    g_tr = {sym.iloc[i] for i in inner_train.indices}
    g_ho = {sym.iloc[i] for i in inner_hold.indices}
    assert not (g_tr & g_ho), "inner split leaked a target gene across both sides"
    print(f"[pilot] mode={a.mode} device={a.device} epochs={a.epochs} batch={a.batch_size} seed={a.seed}")
    print(f"[pilot] inner_train {len(inner_train):,} rows / {len(g_tr):,} genes | "
          f"inner_holdout {len(inner_hold):,} rows / {len(g_ho):,} genes | overlap 0 | VAL UNTOUCHED")

    with seeded_init(a.seed):
        model = nested_family_factories(gene_names, graph, gene_to_idx)[CONDITION_GATED]()
    loss = make_loss(a.mode, model)
    trainer = Trainer(model, inner_train, inner_hold, loss=loss, batch_size=a.batch_size, seed=a.seed,
                      device=a.device, ckpt_dir=Path(a.out).parent / "_unused",
                      log_dir=Path(a.out).parent / "_unused")
    print(f"[pilot] lambda_graph={loss.lambda_graph}  loss={type(loss).__name__}  "
          f"donor_invariance={trainer._donor_on}  setup {time.perf_counter() - t0:.0f}s")

    diag_batch = PerturbationDataset.collate([inner_train[i] for i in range(a.batch_size)])
    dbatch, dtargets, dconds = diag_batch[0], diag_batch[1], diag_batch[2]
    dtarget = _first_known_target(model, dtargets)
    drow = list(dtargets).index(dtarget) if dtarget else 0

    def sensitivity():
        """delete-all-edges relative change in h_graph, for the CURRENT weights (probe E's measure)."""
        if dtarget is None:
            return None
        with torch.no_grad():
            h_do = model.perturbation_encoder(dbatch)[drow].detach()
        return _neighbourhood_dependence(model, dtarget, dconds[drow], h_do)

    res = {"mode": a.mode, "seed": a.seed, "epochs": a.epochs, "batch_size": a.batch_size,
           "device": a.device, "lambda_graph": float(loss.lambda_graph), "loss_class": type(loss).__name__,
           "n_inner_train": len(inner_train), "n_inner_holdout": len(inner_hold),
           "n_genes_train": len(g_tr), "n_genes_holdout": len(g_ho), "history": []}
    s0 = sensitivity()
    res["at_init"] = {"gates": gate_stats(model, dbatch, dtargets, dconds), "sensitivity": s0}
    print(f"[pilot] AT INIT   gates={res['at_init']['gates']}")
    print(f"[pilot]           neighbourhood rel-change={s0['rel_delta']:.4e} (target {dtarget})")

    for epoch in range(a.epochs):
        te = time.perf_counter()
        train_m = trainer._epoch(trainer.train_loader, train=True)
        hold_m = trainer._epoch(trainer.val_loader, train=False)   # INNER holdout — not the val fold
        g = gate_stats(model, dbatch, dtargets, dconds)
        s = sensitivity()
        row = {"epoch": epoch, "train": train_m, "inner_holdout": hold_m, "gates": g,
               "sensitivity": s, "seconds": time.perf_counter() - te}
        res["history"].append(row)
        print(f"[pilot] epoch {epoch}: train.response={train_m['response']:.4f} "
              f"train.graph={train_m['graph']:.6g} | holdout.response={hold_m['response']:.4f} "
              f"holdout.total={hold_m['total']:.4f}")
        print(f"[pilot]           gate_mean={g['mean']:.6f} gate_max={g['max']:.4f} "
              f"frac<1e-3={g['frac_below_1e-3']:.3f} | neighbourhood rel-change="
              f"{s['rel_delta']:.4e} | {row['seconds'] / 60:.1f} min")
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))  # checkpoint after each epoch

    g_end = res["history"][-1]["gates"]
    survived = g_end["mean"] > 1e-3
    res["gates_survived"] = bool(survived)
    print(f"\n[pilot] {a.mode}: gate mean {res['at_init']['gates']['mean']:.6f} -> {g_end['mean']:.6f}  "
          f"=> gates {'SURVIVED' if survived else 'COLLAPSED'}")
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"[pilot] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
