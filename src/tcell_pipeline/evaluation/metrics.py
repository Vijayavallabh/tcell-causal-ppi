"""Prediction metrics (walkthrough §10.4, report §Evaluation Metrics) — the vectorised implementation.

Every metric is computed PER ROW and then macro-averaged. A row is one perturbation-target-by-condition
response (report: "Observational unit: one perturbation-target by condition response row"), so per-row
IS per-perturbation — this deliberately avoids micro-averaging (pooling all genes across all rows into one
correlation), which the G2-MQ gate treats as a leakage-prone shortcut.

Zero/constant-vector convention: a row whose predicted OR true vector has zero variance carries no
linear/rank information, so its correlation contributes 0.0 (not NaN). This makes a zero predictor score
0 — the worst — exactly as the metric-qualification gate requires, and lets ``metrics_ref`` reproduce the
same numbers with a completely separate code path. ``metrics_ref.py`` is a second, independent
implementation that must agree with this one on a fixed fixture.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from tcell_pipeline import config


def _np(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _rowwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        p = pred - pred.mean(1, keepdims=True)
        t = true - true.mean(1, keepdims=True)
        num = (p * t).sum(1)
        den = np.sqrt((p * p).sum(1) * (t * t).sum(1))
        # a constant OR non-finite row carries no signal -> 0.0 (matches metrics_ref by construction)
        return np.where((den > 0) & np.isfinite(den) & np.isfinite(num), num / den, 0.0)


def _rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        num = (a * b).sum(1)
        den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
        return np.where((den > 0) & np.isfinite(den) & np.isfinite(num), num / den, 0.0)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
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
    # rank first, but a non-finite row would rank into a finite vector and score spuriously, so its
    # contribution is forced to 0.0 (matching metrics_ref, which rejects the raw row before ranking)
    finite = np.isfinite(p).all(1) & np.isfinite(t).all(1)
    rp = rankdata(np.where(np.isfinite(p), p, 0.0), axis=1).astype(np.float64)
    rt = rankdata(np.where(np.isfinite(t), t, 0.0), axis=1).astype(np.float64)
    return float(np.where(finite, _rowwise_pearson(rp, rt), 0.0).mean())


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


def topk_recall(pred, true, k: int = config.METRICS_TOP_K) -> float:
    """Per-row recall of the k strongest-magnitude (up or down) true genes among the k predicted."""
    p, t = _np(pred), _np(true)
    k = min(k, p.shape[1])
    if k == 0:
        return 0.0
    recalls = []
    for pr, tr in zip(p, t):
        ti = set(np.argpartition(np.abs(tr), -k)[-k:])
        pi = set(np.argpartition(np.abs(pr), -k)[-k:])
        recalls.append(len(ti & pi) / k)
    return float(np.mean(recalls))


def sign_accuracy(pred, true, top_n: int = config.METRICS_SIGN_TOP_N) -> float:
    """Fraction of correct signs among the ``top_n`` strongest-magnitude true effects, per row."""
    p, t = _np(pred), _np(true)
    n = min(top_n, p.shape[1])
    if n == 0:
        return 0.0
    accs = []
    for pr, tr in zip(p, t):
        idx = np.argpartition(np.abs(tr), -n)[-n:]
        accs.append(float((np.sign(pr[idx]) == np.sign(tr[idx])).mean()))
    return float(np.mean(accs))


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
