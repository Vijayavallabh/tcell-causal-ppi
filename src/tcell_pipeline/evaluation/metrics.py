"""Prediction metrics (walkthrough §10.4, report §Evaluation Metrics) — the vectorised implementation.

Every metric is computed PER ROW and then macro-averaged. A row is one perturbation-target-by-condition
response (report: "Observational unit: one perturbation-target by condition response row"), so per-row
IS per-perturbation — this deliberately avoids micro-averaging (pooling all genes across all rows into one
correlation), which the G2-MQ gate treats as a leakage-prone shortcut.

Higher-is-better metrics (correlation, cosine, accuracy) apply a degeneracy convention: a row whose
predicted OR true vector is CONSTANT (bit-identical across genes, ``max == min`` — including all-zero) or
NON-FINITE carries no signal, so it contributes 0.0 (not NaN). This makes a zero predictor score 0 — the
worst — exactly as the metric-qualification gate requires, and lets ``metrics_ref`` reproduce the same
numbers with a completely separate code path. The constant test is deliberately ``max == min`` rather than
``std == 0``: at a realistic gene dimension a genuinely-constant row's variance underflows to ~1e-16 (not
exactly 0), which ``std``-based guards miss. Correlation denominators use ``sqrt(a)*sqrt(b)`` (separate
roots), not ``sqrt(a*b)``, so a tiny-magnitude row doesn't underflow the product to a spurious 0.
``metrics_ref.py`` is a second, independent implementation that must agree with this one on a fixed
fixture AND on degenerate/non-finite rows.

The error metrics ``mae``/``rmse`` are lower-is-better, so the 0.0 convention does NOT apply — a
non-finite prediction propagates to a non-finite error (surfacing the corruption) rather than being
silently rewarded with zero error.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from tcell_pipeline import config
from tcell_pipeline.evaluation._arrays import to_numpy as _np


def _degenerate_rows(*mats: np.ndarray) -> np.ndarray:
    """Boolean per-row mask: True where any given matrix's row is non-finite or constant (max == min)."""
    mask = np.zeros(mats[0].shape[0], dtype=bool)
    with np.errstate(invalid="ignore"):
        for m in mats:
            mask |= ~np.isfinite(m).all(1) | (m.max(1) == m.min(1))
    return mask


def _rowwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        p = pred - pred.mean(1, keepdims=True)
        t = true - true.mean(1, keepdims=True)
        # separate roots (not sqrt of the product) so a tiny-magnitude row doesn't underflow the denominator
        den = np.sqrt((p * p).sum(1)) * np.sqrt((t * t).sum(1))
        r = np.divide((p * t).sum(1), den, out=np.zeros(pred.shape[0]), where=den > 0)
        return np.where(_degenerate_rows(pred, true) | ~np.isfinite(r), 0.0, r)


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.sqrt((a * a).sum(1)) * np.sqrt((b * b).sum(1))
        r = np.divide((a * b).sum(1), den, out=np.zeros(a.shape[0]), where=den > 0)
        finite = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        return np.where(finite & (den > 0) & np.isfinite(r), r, 0.0)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(np.isfinite(a), a, 0.0)  # non-finite entries -> 0 (matches metrics_ref's sanitisation)
        b = np.where(np.isfinite(b), b, 0.0)
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        # proper zero-norm masking (no 1e-12 floor): a tiny-norm row normalises to a real unit vector, a
        # zero-norm row to zeros — so `own` and the bank use one consistent normalisation
        an = np.divide(a, na, out=np.zeros_like(a), where=na > 0)
        bn = np.divide(b, nb, out=np.zeros_like(b), where=nb > 0)
        return an @ bn.T


def mae(pred, true) -> float:
    p, t = _np(pred), _np(true)
    return float(np.abs(p - t).mean(1).mean())


def rmse(pred, true) -> float:
    p, t = _np(pred), _np(true)
    return float(np.sqrt(((p - t) ** 2).mean(1)).mean())


def pearson_corr(pred, true) -> float:
    return float(_rowwise_pearson(_np(pred), _np(true)).mean())


def spearman_corr(pred, true) -> float:
    from scipy.stats import rankdata

    p, t = _np(pred), _np(true)
    # a non-finite row would rank into a finite vector and score spuriously, and a constant raw row has no
    # rank order, so both are forced to 0.0 (on the RAW rows, matching metrics_ref which gates pre-rank)
    degenerate = _degenerate_rows(p, t)
    rp = rankdata(np.where(np.isfinite(p), p, 0.0), axis=1).astype(np.float64)
    rt = rankdata(np.where(np.isfinite(t), t, 0.0), axis=1).astype(np.float64)
    return float(np.where(degenerate, 0.0, _rowwise_pearson(rp, rt)).mean())


def systema_pert_specific_delta(pred, true, train_mean) -> float:
    """Correlation after removing the average training perturbation effect (§10.2 primary endpoint).

    ``rho(i) = corr(dz_hat_i - dz_bar_train, dz_i - dz_bar_train)`` per row, macro-averaged; subtracting
    the generic treatment effect so the model isn't credited for predicting it."""
    p, t = _np(pred), _np(true)
    m = _np(train_mean).reshape(1, -1)
    return float(_rowwise_pearson(p - m, t - m).mean())


def centroid_accuracy(pred, true, all_true=None) -> float:
    """Fraction of rows whose prediction is cosine-closest to its own true centroid (§10.4).

    ``true[i]`` is row i's own perturbation centroid; ``all_true`` (default ``true``) is the bank of
    candidate centroids to beat. Ties count as correct (own is among the closest)."""
    p, t = _np(pred), _np(true)
    bank = t if all_true is None else _np(all_true)
    own = _rowwise_cosine(p, t)
    # a degenerate (zero-norm / non-finite) prediction row is cosine 0 to every centroid; it is a MISS,
    # not a tied-with-everything hit, so a zero predictor scores worst rather than a spurious 1.0
    with np.errstate(invalid="ignore"):
        pred_ok = np.isfinite(p).all(1) & (np.linalg.norm(p, axis=1) > 0)
    return float((pred_ok & (own >= _cosine_matrix(p, bank).max(1) - 1e-9)).mean())


def _topk_indices(mat: np.ndarray, k: int) -> np.ndarray:
    """(N, k) indices of each row's k strongest-magnitude entries, vectorised over rows."""
    g = mat.shape[1]
    return np.argpartition(np.abs(mat), g - k, axis=1)[:, g - k:]


def topk_recall(pred, true, k: int = config.METRICS_TOP_K) -> float:
    """Per-row recall of the k strongest-magnitude (up or down) true genes among the k predicted.

    A non-finite or constant (no strongest genes) prediction row scores 0.0, consistent with the
    higher-is-better degeneracy convention (a diverged/zero predictor must not earn chance recall)."""
    p, t = _np(pred), _np(true)
    k = min(k, p.shape[1])
    if k == 0:
        return 0.0
    t_top, p_top = _topk_indices(t, k), _topk_indices(p, k)
    overlap = np.array([np.intersect1d(tr, pr, assume_unique=True).size
                        for tr, pr in zip(t_top, p_top)], dtype=np.float64) / k
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(p).all(1) & (p.max(1) != p.min(1))
    return float(np.where(ok, overlap, 0.0).mean())


def sign_accuracy(pred, true, top_n: int = config.METRICS_SIGN_TOP_N) -> float:
    """Fraction of correct signs among the ``top_n`` strongest-magnitude true effects, per row. A
    non-finite prediction row scores 0.0 (a diverged predictor is not credited with chance sign hits)."""
    p, t = _np(pred), _np(true)
    n = min(top_n, p.shape[1])
    if n == 0:
        return 0.0
    idx = _topk_indices(t, n)
    rows = np.arange(p.shape[0])[:, None]
    match = (np.sign(p[rows, idx]) == np.sign(t[rows, idx])).mean(1)
    ok = np.isfinite(p).all(1)
    return float(np.where(ok, match, 0.0).mean())


def program_cosine(pred_z, true_z) -> float:
    """Macro-averaged per-row cosine similarity of program-delta vectors (§10.4 program-level quality)."""
    return float(_rowwise_cosine(_np(pred_z), _np(true_z)).mean())


def _split_up_down(a) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(a, dict):
        return _np(a["up"]).ravel(), _np(a["down"]).ravel()
    arr = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-1] != 2:
        raise ValueError("signed-DE arrays need a trailing size-2 (up, down) axis or an {up, down} dict")
    return arr[..., 0].ravel(), arr[..., 1].ravel()


def signed_de_metrics(probs, labels) -> dict:
    """Signed-DE classification quality: macro-F1, per-class precision/recall, and AUPRC (§10.4).

    ``probs`` and ``labels`` are the up/down DE calls, given either as a trailing-(up, down) array or an
    ``{"up", "down"}`` dict. AUPRC is NaN for a class with no positive (or no negative) labels — undefined
    rather than silently zero. AUROC is intentionally omitted; the report only reports it alongside
    prevalence-aware metrics."""
    pu, pdn = _split_up_down(probs)
    lu, ldn = _split_up_down(labels)
    out: dict = {}
    for name, pr, lab in (("up", pu, lu), ("down", pdn, ldn)):
        lab_b = (lab >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            lab_b, (pr >= 0.5).astype(int), average="binary", zero_division=0)
        pos = lab_b.sum()
        auprc = (float(average_precision_score(lab_b, pr))
                 if 0 < pos < len(lab_b) else float("nan"))
        out[name] = {"precision": float(precision), "recall": float(recall),
                     "f1": float(f1), "auprc": auprc}
    out["macro_f1"] = float(np.mean([out["up"]["f1"], out["down"]["f1"]]))
    return out


def response_metric_suite(dz_hat, dx_hat, dz_true, dx_true, train_mean) -> dict:
    """The 8-metric response-prediction block: program-space (Δz) pearson / systema / centroid / cosine and
    gene-space (Δx) mae / rmse / topk / sign; ``systema`` (the primary H1 endpoint) removes the program-space
    training mean. One definition shared by ``screening.compute_all_metrics`` and ``run_module6_smoke._score``
    so the screening scores and the Module-6 headline scores can never silently diverge."""
    return {
        "pearson": pearson_corr(dz_hat, dz_true),
        "systema": systema_pert_specific_delta(dz_hat, dz_true, train_mean),
        "centroid": centroid_accuracy(dz_hat, dz_true),
        "prog_cos": program_cosine(dz_hat, dz_true),
        "mae": mae(dx_hat, dx_true),
        "rmse": rmse(dx_hat, dx_true),
        "topk": topk_recall(dx_hat, dx_true),
        "sign": sign_accuracy(dx_hat, dx_true),
    }
