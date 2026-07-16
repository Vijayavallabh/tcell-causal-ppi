"""Module 7 screening driver: run the §10.6 nested family (+ the untyped-graph diagnostic) on the real
blocked-target-OOD development split and report H2a / H2b on the primary endpoint.

    PYTHONPATH=src python -m tcell_pipeline.screening.run_screening --epochs 2 --device cuda

Builds the real PPI graph + train/val PerturbationDatasets once, screens each family member through the
Stage-A Trainer, writes predictions (common output schema), a per-config metrics table, a summary JSON, and
registers every run in the experiment registry. The graph message passing is CPU-bound per subgraph; the
encoders run on ``--device``.

Memory: the typed encoder's per-edge signed messages are heavy on real DENSE PPI subgraphs (a hub's
512-node neighbourhood carries tens of thousands of STRING functional edges) — it OOMs a single 80 GB A100
at batch 32. So ``--device cuda`` enables ``expandable_segments`` and the default batch is small; a config
that still OOMs is isolated by ``run_screening`` (logged failed, the wave continues) rather than aborting
the run. ``--device cpu`` (1 TB RAM) is the report's home for graph message passing.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.programs.program_basis import zscore_path  # noqa: E402
from tcell_pipeline.screening.screening import (  # noqa: E402
    CONDITION_GATED,
    EXPRESSION_ONLY,
    TYPED_STATIC,
    UNTYPED_GNN,
    nested_family_configs,
    run_screening,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402

_WAVE = [EXPRESSION_ONLY, UNTYPED_GNN, TYPED_STATIC, CONDITION_GATED]  # report §screening first wave
_COLS = ["name", "primary", "pearson", "systema", "centroid", "prog_cos", "mae", "rmse", "topk", "sign"]


def run(epochs: int = 2, batch_size: int = 8, seed: int = config.SPLIT_SEED,
        n_max: int | None = None, device: str = "cpu") -> int:
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        # reduce allocator fragmentation for the dense-subgraph typed encoder (set before the first CUDA
        # allocation, i.e. before any .to(device)); does NOT lift the fundamental per-batch memory need
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    required = [config.BLOCKED_SPLIT_PATH, config.PERTURBATION_CONDITION_PATH, config.DE_OBS_PATH,
                config.DE_VAR_PATH, config.PROGRAM_LOADINGS_PATH, zscore_path(),
                config.PROTEIN_EDGES_PATH, config.COMPLEX_MEMBERSHIP_PATH, config.ID_MAPPING_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[screen] required artifacts absent: {missing} — run run_module0.py, splits, "
              f"run_program_basis, and the PPI graph build first")
        return 1

    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, gene_to_idx = build_hetero_graph()
    train_ds = PerturbationDataset("train", n_max=n_max)
    val_ds = PerturbationDataset("val", n_max=n_max)
    print(f"[screen] {len(train_ds)} train / {len(val_ds)} val; wave={_WAVE}; epochs={epochs}; device={device}")

    configs = nested_family_configs(gene_names, graph, gene_to_idx, epochs, names=_WAVE,
                                    batch_size=batch_size, seed=seed)
    summary = run_screening(configs, train_ds, val_ds, device=device, registry_path=config.REGISTRY_PATH)

    by_name = {r["name"]: r for r in summary["results"]}
    print("\n" + " ".join(f"{c:>16}" if c == "name" else f"{c:>9}" for c in _COLS))
    for n in _WAVE:
        r = by_name.get(n, {"name": n, "status": "missing"})
        if r.get("status") == "completed":
            print(" ".join(f"{r['name']:>16}" if c == "name" else f"{r[c]:>9.4f}" for c in _COLS))
        else:
            print(f"{n:>16}  {r.get('status', '?').upper()}: {r.get('error', '')}")

    for hyp in ("h2a", "h2b"):
        if hyp in summary:
            c = summary[hyp]
            print(f"[screen] {hyp.upper()}: {c['better']} vs {c['worse']} Δsystema={c['delta']:+.4f} "
                  f"supported={c['supported']}")
    print(f"[screen] summary -> {summary['summary_path']}")
    print("[screen] OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="small by default: the typed encoder OOMs 80GB on dense real subgraphs at batch 32 "
                         "(≤8 on one GPU, or --device cpu); a config that still OOMs is isolated, not fatal")
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--n-max", type=int, default=None, help="cap examples per split (quick runs)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (encoders; graph message passing is CPU-bound)")
    a = ap.parse_args()
    sys.exit(run(a.epochs, a.batch_size, a.seed, a.n_max, a.device))
