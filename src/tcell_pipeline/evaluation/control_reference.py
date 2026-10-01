"""Control-reference safeguards (walkthrough §10.5, report §Control-reference safeguards).

When a delta is built by subtracting an empirical control centroid from both prediction and truth, reusing
ONE control estimate for both sides injects the same control-estimate noise into each, manufacturing a
positive spurious correlation. The corrected estimator subtracts INDEPENDENT control estimates
(``ctrl_a`` from the prediction side, ``ctrl_b`` from the truth side); the shared-control version is kept
only as a bias diagnostic.

The guarantee this module is tested against: ``null_control_predictor`` — a deliberately non-informative
predictor that just re-emits its control — must score approximately null under the independent-control
estimator, yet can look spuriously positive under the shared-control diagnostic.
"""
from __future__ import annotations

import numpy as np

from tcell_pipeline.evaluation._arrays import to_numpy as _np
from tcell_pipeline.evaluation.metrics import pearson_corr


def independent_control_metric(pred, true, ctrl_a, ctrl_b, metric=pearson_corr, **metric_kwargs) -> float:
    """Corrected estimator: subtract independent controls before scoring — ``ctrl_a`` off the prediction,
    ``ctrl_b`` off the truth. This is the version that should be reported. Extra keyword args are forwarded
    to ``metric``, so the 3-arg primary endpoint composes:
    ``independent_control_metric(pred, true, ctrl_a, ctrl_b, metric=systema_pert_specific_delta,
    train_mean=tm)``."""
    return float(metric(_np(pred) - _np(ctrl_a), _np(true) - _np(ctrl_b), **metric_kwargs))


def shared_control_diagnostic(pred, true, shared_ctrl, metric=pearson_corr, **metric_kwargs) -> float:
    """Bias diagnostic ONLY: subtract the same control from both sides. Its gap above the independent
    estimate measures the shared-control inflation, and it must never be reported as the headline number."""
    c = _np(shared_ctrl)
    return float(metric(_np(pred) - c, _np(true) - c, **metric_kwargs))


def null_control_predictor(ctrl) -> np.ndarray:
    """The intentionally non-informative control-derived predictor (§10.5 last bullet): it predicts the
    control itself, so under the independent-control estimator its delta is identically zero and it returns
    ~0. Feed its output as ``pred`` with the SAME array as ``ctrl_a``."""
    return _np(ctrl).copy()
