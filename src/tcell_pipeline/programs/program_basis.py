"""Fold-local program basis: factor the training-fold DE matrix Z_train ~= A @ B^T (§6.1).

B in R^{G x K} are the frozen gene->program loadings (which genes belong to which program); A in
R^{N_train x K} are the per-perturbation program scores. The basis is a *response-derived* transform,
so it MUST see training rows only — ``train_row_indices`` derives the eligible rows from the blocked
split and the caller slices ``zscore.npz`` to them before fitting. Never pass val/cal/challenge rows.

Methods (§6.5): sparse_pca is the paper default (MiniBatchSparsePCA — the scalable sparse variant),
with nmf/fastica/svd available for the method comparison. B is frozen downstream (decoder buffer).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tcell_pipeline import config


def train_row_indices(split_df: pd.DataFrame, pc_df: pd.DataFrame, role: str = "train") -> np.ndarray:
    """Row indices (into the DE matrix) whose target gene is in ``role`` of the blocked split.

    Fold-locality gate: only these rows may be shown to ``fit_program_basis``. Returned sorted so the
    saved A-scores line up predictably with row_index.
    """
    genes = set(split_df.loc[split_df["role"] == role, "hgnc_symbol"])
    rows = pc_df.loc[pc_df["hgnc_symbol"].isin(genes), "row_index"].to_numpy()
    return np.sort(rows.astype(np.int64))


def _factor(Z: np.ndarray, method: str, K: int, seed: int, max_iter: int):
    """Return (components (K,G), scores (N,K)) for the requested factorisation."""
    if method == "sparse_pca":
        from sklearn.decomposition import MiniBatchSparsePCA

        model = MiniBatchSparsePCA(n_components=K, random_state=seed, max_iter=max_iter, batch_size=256)
        model.fit(Z)
        return model.components_, model.transform(Z)
    if method == "svd":
        from sklearn.decomposition import TruncatedSVD

        model = TruncatedSVD(n_components=K, random_state=seed).fit(Z)
        return model.components_, model.transform(Z)
    if method == "fastica":
        from sklearn.decomposition import FastICA

        model = FastICA(n_components=K, random_state=seed, max_iter=max_iter, whiten="unit-variance")
        scores = model.fit_transform(Z)
        return model.components_, scores
    if method == "nmf":
        from sklearn.decomposition import NMF

        # NMF needs non-negative input; z-scores are signed, so it sees only the up-regulation part.
        # ponytail: positive-part NMF; split into signed +/- channels if down-regulation programs matter.
        model = NMF(n_components=K, random_state=seed, max_iter=max_iter, init="nndsvda")
        scores = model.fit_transform(np.maximum(Z, 0.0))
        return model.components_, scores
    raise ValueError(f"unknown program method {method!r}; valid: sparse_pca, nmf, fastica, svd")


def fit_program_basis(
    zscore_train: np.ndarray,
    method: str = config.PROGRAM_METHOD,
    K: int = config.PROGRAM_DIM,
    seed: int = config.SPLIT_SEED,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Z_train ~= A @ B^T on training rows only. Returns (B (G,K), A (N_train,K)), float32."""
    Z = np.asarray(zscore_train, dtype=np.float32)
    if Z.ndim != 2:
        raise ValueError(f"zscore_train must be 2-D (N,G), got shape {Z.shape}")
    components, scores = _factor(Z, method, K, seed, max_iter)
    return np.ascontiguousarray(components.T, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def _program_cols(K: int) -> list[str]:
    return [f"{config.PROGRAM_COL_PREFIX}{k}" for k in range(K)]


def save_program_basis(B: np.ndarray, gene_names: list[str], path=config.PROGRAM_LOADINGS_PATH) -> None:
    if B.shape[0] != len(gene_names):
        raise ValueError(f"B has {B.shape[0]} gene rows but {len(gene_names)} gene names")
    df = pd.DataFrame(B, columns=_program_cols(B.shape[1]))
    df.insert(0, "gene_name", list(gene_names))
    config.write_parquet_atomic(df, path)


def save_program_response(A: np.ndarray, row_indices: np.ndarray, path=config.PROGRAM_RESPONSE_PATH) -> None:
    if A.shape[0] != len(row_indices):
        raise ValueError(f"A has {A.shape[0]} rows but {len(row_indices)} row indices")
    df = pd.DataFrame(A, columns=_program_cols(A.shape[1]))
    df.insert(0, "row_index", np.asarray(row_indices, dtype=np.int64))
    config.write_parquet_atomic(df, path)


def load_program_basis(
    path=config.PROGRAM_LOADINGS_PATH, gene_order: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Load B as (G,K) float32. If ``gene_order`` is given, reindex B rows to it (0-fill missing genes)
    so B aligns to the model's fixed gene axis regardless of the parquet's row order."""
    df = pd.read_parquet(path)
    prog_cols = [c for c in df.columns if c.startswith(config.PROGRAM_COL_PREFIX)]
    if gene_order is not None:
        df = df.set_index("gene_name").reindex(gene_order).reset_index()
        df[prog_cols] = df[prog_cols].fillna(0.0)
    B = df[prog_cols].to_numpy(dtype=np.float32)
    return B, df["gene_name"].tolist()
