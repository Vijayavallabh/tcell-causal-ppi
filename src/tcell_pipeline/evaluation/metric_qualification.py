"""G2-MQ: model-blind metric qualification (walkthrough §10.1).

A candidate endpoint qualifies only if it orders every negative control strictly below every positive
reference — the report forbids picking a metric post-hoc because it flatters EG-IPG. This module supplies
the standard negative-control constructors (zero, perturbed-mean, label-permutation N1, response-row
shuffle N2) and positive references (oracle; a guide split-half stand-in), plus ``qualify_metric`` which
runs the ordering test.

Constructing negatives/positives with a supplied RNG (never a global seed) keeps the gate reproducible;
the preserved seed is what §10.5 asks for.
"""
from __future__ import annotations

import numpy as np

from tcell_pipeline.evaluation._arrays import to_numpy as _np


def zero_prediction(true) -> np.ndarray:
    """Delta = 0 (§10.1: must score worst)."""
    return np.zeros_like(_np(true))


def perturbed_mean_prediction(true) -> np.ndarray:
    """Systema non-control mean: every row predicts the average training perturbation effect (§10.1: near
    the bottom but above zero — it captures systematic treatment structure only)."""
    t = _np(true)
    return np.broadcast_to(t.mean(0, keepdims=True), t.shape).copy()


def label_permutation(true, rng: np.random.Generator) -> np.ndarray:
    """N1: predictions are the true responses under a permuted row (target) identity — a metric sensitive
    to target identity must collapse to null. A DERANGEMENT (no fixed point) is used: a plain permutation
    leaves ~1 row mapped to itself on average, which scores perfectly and keeps the negative off the null
    floor — for a small fold it could even tie the oracle and spuriously fail the gate."""
    t = _np(true)
    n = t.shape[0]
    idx = np.arange(n)
    if n < 2:
        return t.copy()
    perm = rng.permutation(n)
    for _ in range(16):
        if not np.any(perm == idx):
            break
        perm = rng.permutation(n)
    else:
        perm = np.roll(idx, 1)  # a guaranteed fixed-point-free fallback for n >= 2
    return t[perm]


def row_shuffle(true, rng: np.random.Generator) -> np.ndarray:
    """N2: each row's gene values shuffled within the row, destroying the response pattern while keeping
    its marginal — catches metrics that reward matching only the value distribution."""
    return rng.permuted(_np(true), axis=1)


def oracle_prediction(true) -> np.ndarray:
    """Upper reference: the true response itself (realistic ceiling, not a claim of attainability)."""
    return _np(true).copy()


def guide_split_half(true, rng: np.random.Generator, noise: float = 0.5) -> np.ndarray:
    """Positive reference stand-in for guide-level split-half agreement: the true response plus moderate
    noise, so it lands between the negatives and the oracle (empirical reproducibility reference, not an
    upper bound). Real runs replace this with agreement from guide-level MuData."""
    t = _np(true)
    return t + noise * t.std() * rng.standard_normal(t.shape)


def _score(fn, value) -> float:
    if callable(value):
        return float(value())
    if isinstance(value, tuple):
        return float(fn(*value))
    return float(value)


def qualify_metric(fn, neg_controls: dict, pos_refs: dict) -> dict:
    """Run the G2-MQ ordering test for a single metric.

    ``neg_controls``/``pos_refs`` map a control name to either a pre-computed score, a ``(pred, true)``
    tuple scored with ``fn``, or a zero-arg callable returning a score. The metric passes iff every
    negative scores strictly below every positive (higher metric == better).

    Returns ``{passed, ordering_correct, dynamic_range, neg_scores, pos_scores}`` where ``dynamic_range``
    is the positive/negative separation ``min(pos) - max(neg)`` (negative when the ordering is violated)."""
    neg = {k: _score(fn, v) for k, v in neg_controls.items()}
    pos = {k: _score(fn, v) for k, v in pos_refs.items()}
    scores = list(neg.values()) + list(pos.values())
    if not neg or not pos or not all(np.isfinite(scores)):
        return {"passed": False, "ordering_correct": False, "dynamic_range": float("nan"),
                "neg_scores": neg, "pos_scores": pos}
    max_neg, min_pos = max(neg.values()), min(pos.values())
    ordering_correct = max_neg < min_pos
    return {"passed": ordering_correct, "ordering_correct": ordering_correct,
            "dynamic_range": float(min_pos - max_neg), "neg_scores": neg, "pos_scores": pos}
