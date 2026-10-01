"""Common prediction store shared by EG-IPG and every baseline (report §Baselines: "common output
schema").

One parquet per (model, split, seed) at ``predictions/<model>/<split>/<seed>.parquet`` with columns
``row_index``, ``delta_z_0..K-1``, ``delta_x_0..G-1``, ``sigma_0..K-1``. A single schema lets the
evaluation harness and the test steward score any model identically, and keeps the challenge split's
scoring code model-agnostic. Baselines that emit no calibrated uncertainty write ``sigma = 0``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tcell_pipeline import config


def prediction_path(model: str, split: str, seed: int,
                    root: Path = config.PREDICTIONS_ROOT) -> Path:
    return Path(root) / model / split / f"{seed}.parquet"


def _matrix(a, n_rows: int, name: str) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != n_rows:
        raise ValueError(f"{name} must be (n_rows, dim) with n_rows={n_rows}, got {a.shape}")
    return a


def predictions_to_frame(row_index, delta_z, delta_x, sigma=None) -> pd.DataFrame:
    ri = np.asarray(row_index).reshape(-1).astype(np.int64)
    n = len(ri)
    dz = _matrix(delta_z, n, "delta_z")
    dx = _matrix(delta_x, n, "delta_x")
    sig = np.zeros_like(dz) if sigma is None else _matrix(sigma, n, "sigma")
    if sig.shape[1] != dz.shape[1]:
        raise ValueError(f"sigma dim {sig.shape[1]} must match delta_z dim {dz.shape[1]}")
    cols: dict = {"row_index": ri}
    cols.update({f"delta_z_{k}": dz[:, k] for k in range(dz.shape[1])})
    cols.update({f"delta_x_{g}": dx[:, g] for g in range(dx.shape[1])})
    cols.update({f"sigma_{k}": sig[:, k] for k in range(sig.shape[1])})
    return pd.DataFrame(cols)


def write_predictions(row_index, delta_z, delta_x, sigma, model: str, split: str, seed: int,
                      root: Path = config.PREDICTIONS_ROOT) -> Path:
    frame = predictions_to_frame(row_index, delta_z, delta_x, sigma)
    final = prediction_path(model, split, seed, root)
    config.write_parquet_atomic(frame, final)
    return final


def _cols(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    names = sorted((c for c in frame.columns if c.startswith(prefix)),
                   key=lambda c: int(c[len(prefix):]))
    return frame[names].to_numpy(dtype=np.float32)


def read_predictions(path: Path) -> dict:
    """Load a prediction parquet back into ``{row_index, delta_z, delta_x, sigma}`` numpy arrays."""
    frame = pd.read_parquet(path)
    return {
        "row_index": frame["row_index"].to_numpy(dtype=np.int64),
        "delta_z": _cols(frame, "delta_z_"),
        "delta_x": _cols(frame, "delta_x_"),
        "sigma": _cols(frame, "sigma_"),
    }
