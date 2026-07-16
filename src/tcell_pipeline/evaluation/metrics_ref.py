"""Independent reference implementation of the core metrics (report line ~1558: "two independent
implementations on a fixed fixture").

Deliberately shares NO code with ``metrics.py``: this module loops per row and leans on
``scipy.stats``/``scipy.spatial``/``sklearn`` where the primary implementation uses hand-vectorised
numpy. Agreement between the two on a fixed synthetic fixture is the cross-check that neither harbours a
silent algebra bug. Covers mae, rmse, pearson, spearman, the Systema perturbation-specific delta, the
centroid accuracy, and the program cosine — the metrics whose formulae are subtle enough to warrant it.

The same zero/constant-vector convention as ``metrics.py`` applies: a row with a zero-variance vector
contributes a correlation of 0.0.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity


def _rows(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _finite(*vs) -> bool:
    return all(np.isfinite(v).all() for v in vs)


def _constant(v: np.ndarray) -> bool:
    # bit-identical row (incl. all-zero) == no signal. `max == min` is exact where `std == 0` is not: at a
    # realistic gene dimension a genuinely-constant row's std underflows to ~1e-16 and slips a std guard.
    return v.max() == v.min()


def _finite_or_zero(r: float) -> float:
    return float(r) if np.isfinite(r) else 0.0


def _pearson_row(x: np.ndarray, y: np.ndarray) -> float:
    if not _finite(x, y) or _constant(x) or _constant(y):
        return 0.0
    return _finite_or_zero(pearsonr(x, y)[0])


def _spearman_row(x: np.ndarray, y: np.ndarray) -> float:
    if not _finite(x, y) or _constant(x) or _constant(y):
        return 0.0
    return _finite_or_zero(spearmanr(x, y).statistic)


def _cosine_row(x: np.ndarray, y: np.ndarray) -> float:
    if not _finite(x, y) or np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 0.0
    return _finite_or_zero(1.0 - cosine_distance(x, y))


def mae(pred, true) -> float:
    p, t = _rows(pred), _rows(true)
    return float(np.mean([np.mean(np.abs(pr - tr)) for pr, tr in zip(p, t)]))


def rmse(pred, true) -> float:
    p, t = _rows(pred), _rows(true)
    return float(np.mean([np.sqrt(np.mean((pr - tr) ** 2)) for pr, tr in zip(p, t)]))


def pearson_corr(pred, true) -> float:
    p, t = _rows(pred), _rows(true)
    return float(np.mean([_pearson_row(pr, tr) for pr, tr in zip(p, t)]))


def spearman_corr(pred, true) -> float:
    p, t = _rows(pred), _rows(true)
    return float(np.mean([_spearman_row(pr, tr) for pr, tr in zip(p, t)]))


def systema_pert_specific_delta(pred, true, train_mean) -> float:
    p, t = _rows(pred), _rows(true)
    m = _rows(train_mean).ravel()
    return float(np.mean([_pearson_row(pr - m, tr - m) for pr, tr in zip(p, t)]))


def centroid_accuracy(pred, true, all_true=None) -> float:
    p, t = _rows(pred), _rows(true)
    bank = t if all_true is None else _rows(all_true)
    # sklearn's finite-check rejects non-finite input; zero them for the similarity (pred_ok drops them)
    sims = cosine_similarity(np.where(np.isfinite(p), p, 0.0), np.where(np.isfinite(bank), bank, 0.0))
    own = np.array([_cosine_row(pr, tr) for pr, tr in zip(p, t)])
    pred_ok = np.array([_finite(pr) and np.linalg.norm(pr) > 0 for pr in p])
    return float(np.mean(pred_ok & (own >= sims.max(axis=1) - 1e-9)))


def program_cosine(pred_z, true_z) -> float:
    p, t = _rows(pred_z), _rows(true_z)
    return float(np.mean([_cosine_row(pr, tr) for pr, tr in zip(p, t)]))
