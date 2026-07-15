"""Fit the fold-local program basis on real train rows and save B / A (§6.1).

    PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis            # sparse_pca (slow, ~15 min)
    PYTHONPATH=src python -m tcell_pipeline.programs.run_program_basis --method svd  # fast smoke basis

Loads the blocked split + perturbation_condition, keeps *train-role rows only* (fold-locality gate),
slices zscore.npz to them, fits Z_train ~= A @ B^T, and writes gene_program_loadings.parquet (B) and
program_response.parquet (A). The gene axis of B is the full de_var order so it drops straight into
the decoder's frozen buffer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/ on path for direct runs

import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.programs.program_basis import (  # noqa: E402
    fit_program_basis,
    save_program_basis,
    save_program_response,
    train_row_indices,
)


def run(method: str = config.PROGRAM_METHOD, K: int = config.PROGRAM_DIM, max_iter: int = 100) -> bool:
    for p in (config.BLOCKED_SPLIT_PATH, config.PERTURBATION_CONDITION_PATH, config.DE_VAR_PATH):
        if not p.exists():
            print(f"[program-basis] missing {p} — run splits / run_module0.py first")
            return False

    split = pd.read_csv(config.BLOCKED_SPLIT_PATH)
    pc = pd.read_parquet(config.PERTURBATION_CONDITION_PATH, columns=["row_index", "hgnc_symbol"])
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()

    rows = train_row_indices(split, pc)
    assert len(set(rows) & set(train_row_indices(split, pc, role="challenge"))) == 0, "fold leak!"
    Z = sp.load_npz(config.DE_LAYERS_DIR / "zscore.npz").tocsr()[rows].toarray()
    print(f"[program-basis] {method} K={K} on {Z.shape[0]} train rows x {Z.shape[1]} genes")

    t0 = time.time()
    B, A = fit_program_basis(Z, method=method, K=K, max_iter=max_iter)
    print(f"[program-basis] fit in {time.time() - t0:.1f}s -> B {B.shape}, A {A.shape}")

    save_program_basis(B, gene_names, config.PROGRAM_LOADINGS_PATH)
    save_program_response(A, rows, config.PROGRAM_RESPONSE_PATH)
    print(f"[program-basis] wrote {config.PROGRAM_LOADINGS_PATH.name} + {config.PROGRAM_RESPONSE_PATH.name}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=config.PROGRAM_METHOD)
    ap.add_argument("--K", type=int, default=config.PROGRAM_DIM)
    ap.add_argument("--max-iter", type=int, default=100)
    a = ap.parse_args()
    sys.exit(0 if run(a.method, a.K, a.max_iter) else 1)
