"""Multi-seed paired robustness aggregation (feat-011 follow-on).

The paired stats are correctness-critical, so the degenerate cases are watched FAILING against a naive
core before hardening (AGENTS.md adversarial-input gate): a non-finite seed metric, a missing seed, a
single seed, and zero variance are each a CONSTRUCTED breaking input, not a hypothetical.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tcell_pipeline.screening.experiment_registry import log_run, register_run
from tcell_pipeline.screening.multiseed import (
    _mean_ci,
    _read_seed_metrics,
    aggregate_seeds,
    paired_delta_summary,
)
from tcell_pipeline.screening.screening import (
    CONDITION_GATED,
    EXPRESSION_ONLY,
    TYPED_STATIC,
    UNTYPED_GNN,
)


# ---- paired_delta_summary: happy path -----------------------------------------------------------
def test_paired_clear_positive_excludes_zero_and_favors_better():
    better = {0: 0.15, 1: 0.16, 2: 0.14, 3: 0.155, 4: 0.145}
    worse = {0: 0.10, 1: 0.10, 2: 0.10, 3: 0.10, 4: 0.10}
    r = paired_delta_summary(better, worse)
    assert r["n"] == 5
    assert r["mean"] == pytest.approx(0.05)
    assert r["ci_low"] > 0 and r["ci_excludes_zero"] is True
    assert "favors" in r["verdict"] or "supports" in r["verdict"]


def test_paired_crossing_zero_is_indistinguishable_not_support():
    # honest frame: a CI that straddles 0 is "indistinguishable", NOT "the graph helps"
    better = {0: 0.10, 1: 0.12, 2: 0.08, 3: 0.11, 4: 0.09}
    worse = {0: 0.10, 1: 0.09, 2: 0.11, 3: 0.10, 4: 0.10}
    r = paired_delta_summary(better, worse)
    assert r["n"] == 5
    assert r["ci_low"] < 0 < r["ci_high"]
    assert r["ci_excludes_zero"] is False
    assert "indistinguishable" in r["verdict"].lower()


# ---- paired_delta_summary: ADVERSARIAL degenerate cases (red against the naive core) ------------
def test_paired_drops_non_finite_seed_and_shrinks_n():
    better = {0: 0.15, 1: float("nan"), 2: 0.14}
    worse = {0: 0.10, 1: 0.10, 2: 0.10}
    r = paired_delta_summary(better, worse)
    assert r["n"] == 2, "a non-finite seed must be dropped, not poison the mean"
    assert 1 in [d["seed"] for d in r["dropped"]]
    assert math.isfinite(r["mean"]) and math.isfinite(r["ci_low"])


def test_paired_drops_numpy_nan_too():
    better = {0: np.float64(0.15), 1: np.float64(np.nan), 2: np.float64(0.14)}
    worse = {0: np.float64(0.10), 1: np.float64(0.10), 2: np.float64(0.10)}
    r = paired_delta_summary(better, worse)
    assert r["n"] == 2 and math.isfinite(r["mean"])


def test_paired_missing_seed_in_one_arm_is_dropped():
    better = {0: 0.15, 1: 0.16}   # seed 1 present here...
    worse = {0: 0.10}             # ...absent here
    r = paired_delta_summary(better, worse)
    assert r["n"] == 1
    assert [d["seed"] for d in r["dropped"]] == [1]


def test_paired_one_seed_has_no_ci():
    r = paired_delta_summary({0: 0.15}, {0: 0.10})
    assert r["n"] == 1
    assert r["mean"] == pytest.approx(0.05)
    assert r["ci_low"] is None and r["ci_high"] is None
    assert r["t"] is None and "n=1" in r["verdict"]


def test_paired_zero_seeds_returns_no_data():
    r = paired_delta_summary({}, {})
    assert r["n"] == 0 and r["mean"] is None and r["ci_low"] is None


def test_paired_identical_deltas_zero_variance_no_crash():
    better = {0: 0.15, 1: 0.15, 2: 0.15}
    worse = {0: 0.10, 1: 0.10, 2: 0.10}
    r = paired_delta_summary(better, worse)
    assert r["n"] == 3
    assert r["sd"] == pytest.approx(0.0)
    assert r["ci_low"] == pytest.approx(0.05) and r["ci_high"] == pytest.approx(0.05)
    assert r["ci_excludes_zero"] is True   # a zero-width CI parked at 0.05 excludes 0


# ---- reader freshness + aggregate wiring --------------------------------------------------------
def _lane(root, name, systema, seed, status="completed"):
    (root / name).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"name": name, "seed": seed, "status": status, "primary": systema, "systema": systema,
                   "gpu_hours": 1.0, "epochs_run": 20, "n_epochs": 20}]
                 ).to_parquet(root / name / f"{seed}.parquet")


def test_read_seed_metrics_reports_missing_nonfinite_and_ok(tmp_path):
    root = tmp_path / "scr"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    _lane(root, TYPED_STATIC, float("nan"), 0)   # completed but non-finite systema
    m, st, _ = _read_seed_metrics([EXPRESSION_ONLY, TYPED_STATIC, CONDITION_GATED], 0,
                                  screening_root=root, registry_path=None)
    assert m == {EXPRESSION_ONLY: pytest.approx(0.10)}
    assert st[TYPED_STATIC] == "non_finite" and st[CONDITION_GATED] == "missing"


def test_read_seed_metrics_freshness_fence_skips_stale(tmp_path):
    root = tmp_path / "scr"
    reg = tmp_path / "registry.yaml"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    _lane(root, TYPED_STATIC, 0.99, 0)           # stale parquet with a winning score
    rid = register_run(EXPRESSION_ONLY, "H2a", "q_pre", "blocked_target_ood", 0, None, path=reg)
    log_run(rid, "completed", {"systema": 0.10}, "b.pt", 1.0, path=reg)
    register_run(TYPED_STATIC, "H2a", "q_pre", "blocked_target_ood", 0, None, path=reg)  # only registered
    m, st, _ = _read_seed_metrics([EXPRESSION_ONLY, TYPED_STATIC], 0,
                                  screening_root=root, registry_path=reg)
    assert TYPED_STATIC not in m and st[TYPED_STATIC] == "stale"
    assert m[EXPRESSION_ONLY] == pytest.approx(0.10)


def test_aggregate_seeds_end_to_end_two_seeds(tmp_path):
    root = tmp_path / "scr"
    data = {  # typed < expr in both seeds (H2a negative); untyped highest
        0: {EXPRESSION_ONLY: 0.086, UNTYPED_GNN: 0.095, TYPED_STATIC: 0.079, CONDITION_GATED: 0.083},
        1: {EXPRESSION_ONLY: 0.088, UNTYPED_GNN: 0.092, TYPED_STATIC: 0.081, CONDITION_GATED: 0.085},
    }
    for s, row in data.items():
        for name, v in row.items():
            _lane(root, name, v, s)
    agg = aggregate_seeds([0, 1], screening_root=root, registry_path=None)
    assert agg["contrasts"]["h2a"]["n"] == 2
    assert agg["contrasts"]["h2a"]["mean"] < 0            # typed − expr negative: graph doesn't help
    assert agg["per_config"][EXPRESSION_ONLY]["n"] == 2
    assert agg["ranking"][0] == UNTYPED_GNN               # highest mean systema


def test_aggregate_flags_fold_mismatch(tmp_path):
    root = tmp_path / "scr"
    reg = tmp_path / "reg.yaml"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    _lane(root, EXPRESSION_ONLY, 0.10, 1)
    r0 = register_run(EXPRESSION_ONLY, "H2a", "q_pre", "blocked_target_ood", 0, None, path=reg)
    log_run(r0, "completed", {"systema": 0.10}, "b.pt", 1.0, path=reg)
    r1 = register_run(EXPRESSION_ONLY, "H2a", "q_pre", "OTHER_FOLD", 1, None, path=reg)
    log_run(r1, "completed", {"systema": 0.10}, "b.pt", 1.0, path=reg)
    agg = aggregate_seeds([0, 1], names=(EXPRESSION_ONLY,), screening_root=root, registry_path=reg)
    assert agg["fold"]["single_frozen_fold"] is False
    assert set(agg["fold"]["observed_splits"]) == {"blocked_target_ood", "OTHER_FOLD"}


def test_mean_ci_one_seed_has_no_interval():
    r = _mean_ci({0: 0.086})
    assert r["n"] == 1 and r["mean"] == pytest.approx(0.086)
    assert r["ci_low"] is None and r["ci_high"] is None
