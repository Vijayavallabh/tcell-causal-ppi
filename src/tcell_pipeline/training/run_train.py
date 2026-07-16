"""Stage A training orchestrator: real-data fit of the EG-IPG H1 predictor (§8.1-8.2).

    PYTHONPATH=src python -m tcell_pipeline.training.run_train --epochs 20 --batch-size 64

Loads the blocked split, builds train/val PerturbationDatasets, assembles the full EGIPGModel on the
real PPI graph with the frozen fold-local basis, and trains Stage A (Module 1 + 2 + 3). Stage B
calibration + the rationale head are fitted separately AFTER this predictor is frozen (§8.1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path for direct runs

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder  # noqa: E402
from tcell_pipeline.model import EGIPGModel  # noqa: E402
from tcell_pipeline.programs.program_basis import zscore_path  # noqa: E402
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402
from tcell_pipeline.training.trainer import Trainer, seeded_init  # noqa: E402


def run(lr=config.LR, epochs=config.MAX_EPOCHS, batch_size=config.BATCH_SIZE, seed=config.SPLIT_SEED,
        n_max=None, expr_only=False, donor_invariance=config.DONOR_INVARIANCE, device="cpu") -> dict | None:
    torch.set_num_threads(1)  # many-core box: tiny per-subgraph GNN ops thrash the default thread pool
    required = [config.BLOCKED_SPLIT_PATH, config.PERTURBATION_CONDITION_PATH, config.DE_OBS_PATH,
                config.DE_VAR_PATH, config.PROGRAM_LOADINGS_PATH, zscore_path()]
    if donor_invariance:  # fail fast rather than silently disabling the donor term when profiles are absent
        required.append(config.CONTROL_DONOR_PROFILES_PATH)
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[train] required artifacts absent: {missing} — run run_module0.py, splits, run_program_basis first")
        return None

    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    with seeded_init(seed):  # reproducible weight init (Trainer's seeded gens cover only data shuffling)
        graph_enc = None if expr_only else TypedGraphEncoder(*build_hetero_graph())
        model = EGIPGModel.from_saved_basis(gene_names, graph_encoder=graph_enc)

    train_ds = PerturbationDataset("train", n_max=n_max)
    val_ds = PerturbationDataset("val", n_max=n_max)
    n_donors = len(train_ds.donor_pool.get("Rest", [])) if train_ds.donor_pool else 0
    donor_active = donor_invariance and n_donors >= 2  # what the Trainer will actually do
    print(f"[train] {len(train_ds)} train / {len(val_ds)} val examples; expr_only={expr_only}; "
          f"donor_invariance={'on' if donor_active else 'off'} ({n_donors} real donors/condition)")
    if donor_invariance and not donor_active:
        print("[train] WARNING donor_invariance requested but <2 donors available — the term is inactive")

    trainer = Trainer(model, train_ds, val_ds, lr=lr, max_epochs=epochs, batch_size=batch_size, seed=seed,
                      donor_invariance=donor_invariance, device=device)
    print(f"[train] device={device}")
    result = trainer.run()
    print(f"[train] {result['epochs_run']} epochs, best_val={result['best_val']:.4f} -> {result['best_ckpt']}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--n-max", type=int, default=None, help="cap examples per split (quick runs)")
    ap.add_argument("--expr-only", action="store_true", help="expression-only nested variant (no graph)")
    ap.add_argument("--no-donor-invariance", action="store_true",
                    help="disable the real per-donor invariance term (skips the extra donor forwards)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (the graph message passing is CPU-bound)")
    a = ap.parse_args()
    sys.exit(0 if run(a.lr, a.epochs, a.batch_size, a.seed, a.n_max, a.expr_only,
                      donor_invariance=not a.no_donor_invariance, device=a.device) else 1)
