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
    CONTRASTS,
    _exit_code,
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


def test_paired_identical_deltas_zero_variance_is_undecidable():
    """Zero variance carries NO information about spread: identical deltas across seeds are the
    signature of seeds that never propagated, so publishing p=0.0 / 'CI excludes zero' would report the
    one condition proving the seeds are uninformative as the strongest possible evidence."""
    better = {0: 0.15, 1: 0.15, 2: 0.15}
    worse = {0: 0.10, 1: 0.10, 2: 0.10}
    r = paired_delta_summary(better, worse)
    assert r["n"] == 3
    assert r["sd"] == pytest.approx(0.0)
    assert r["ci_excludes_zero"] is None, "zero variance must be undecidable, not significant"
    assert r["p_value"] is None and r["ci_low"] is None
    assert "degenerate" in r["verdict"].lower()


# ---- reader freshness + aggregate wiring --------------------------------------------------------
def _lane(root, name, systema, seed, status="completed", n_train=None, n_val=None):
    (root / name).mkdir(parents=True, exist_ok=True)
    row = {"name": name, "seed": seed, "status": status, "primary": systema, "systema": systema,
           "gpu_hours": 1.0, "epochs_run": 20, "n_epochs": 20}
    if n_train is not None:
        row["n_train"], row["n_val"] = n_train, n_val
    pd.DataFrame([row]).to_parquet(root / name / f"{seed}.parquet")


# five seeds x four configs with realistic spread (deltas vary, so sd > 0)
_VALS = {
    0: {EXPRESSION_ONLY: 0.0861, UNTYPED_GNN: 0.0951, TYPED_STATIC: 0.0786, CONDITION_GATED: 0.0834},
    1: {EXPRESSION_ONLY: 0.0859, UNTYPED_GNN: 0.0890, TYPED_STATIC: 0.0700, CONDITION_GATED: 0.0838},
    2: {EXPRESSION_ONLY: 0.0852, UNTYPED_GNN: 0.0893, TYPED_STATIC: 0.0742, CONDITION_GATED: 0.0830},
    3: {EXPRESSION_ONLY: 0.0855, UNTYPED_GNN: 0.0900, TYPED_STATIC: 0.0710, CONDITION_GATED: 0.0851},
    4: {EXPRESSION_ONLY: 0.0858, UNTYPED_GNN: 0.0876, TYPED_STATIC: 0.0692, CONDITION_GATED: 0.0860},
}


def _full_family(root, vals=None, **kw):
    for s, row in (vals or _VALS).items():
        for name, v in row.items():
            _lane(root, name, v, s, **kw)


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


# ---- claim-level defects found by the xhigh review -----------------------------------------------
def test_contrasts_include_the_h1_vs_no_graph_pair():
    """The headline claim (frozen H1 vs no-graph) must be a TESTED contrast, not two marginal means
    read off the per-config ranking — that is exactly how an undecidable pair got published as decided."""
    pairs = {(better, worse) for _, better, worse in CONTRASTS}
    assert (CONDITION_GATED, EXPRESSION_ONLY) in pairs


def test_contrasts_carry_family_wise_multiplicity_correction(tmp_path):
    """Several simultaneous contrasts at raw alpha=0.05 inflate the family-wise error rate; each must
    carry a corrected p and an explicit survives/does-not-survive flag."""
    root = tmp_path / "scr"
    _full_family(root)
    agg = aggregate_seeds(range(5), screening_root=root, registry_path=None)
    assert agg["family_size"] == len(agg["contrasts"])
    for key, c in agg["contrasts"].items():
        assert c["p_bonferroni"] is not None, key
        assert c["p_holm"] is not None, key
        assert isinstance(c["survives_family_wise"], bool), key
        assert c["p_bonferroni"] >= c["p_value"], key      # correction can only make p larger


def test_seed_missing_from_both_arms_is_dropped_not_silently_absent():
    """A seed absent from BOTH arms must still be named in `dropped` — otherwise n shrinks with an
    empty drop list and a 4-seed result reads as the intended 5-seed design."""
    r = paired_delta_summary({0: 0.15, 1: 0.16}, {0: 0.10, 1: 0.11}, seeds=[0, 1, 2])
    assert r["n"] == 2
    assert [d["seed"] for d in r["dropped"]] == [2]


def test_absent_fold_evidence_is_not_published_as_proof(tmp_path):
    """`splits <= {FROZEN_SPLIT}` is vacuously true on the empty set — absence of evidence must not be
    reported as positive proof of a single frozen fold."""
    root = tmp_path / "scr"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    agg = aggregate_seeds([0], names=(EXPRESSION_ONLY,), screening_root=root, registry_path=None)
    assert agg["fold"]["single_frozen_fold"] is None      # unknown, NOT True
    assert agg["fold"]["registry_evidence"] is False


def test_fold_sizes_catch_a_capped_fold_the_split_label_cannot(tmp_path):
    """The registry `split` string is a hardcoded literal, so it can never detect a --n-max capped run.
    The recorded row sizes are the real fold signal."""
    root = tmp_path / "scr"
    _lane(root, EXPRESSION_ONLY, 0.10, 0, n_train=21262, n_val=4400)
    _lane(root, EXPRESSION_ONLY, 0.11, 1, n_train=2000, n_val=400)      # capped fold
    agg = aggregate_seeds([0, 1], names=(EXPRESSION_ONLY,), screening_root=root, registry_path=None)
    assert agg["fold"]["fold_sizes_consistent"] is False
    assert len(agg["fold"]["observed_fold_sizes"]) == 2


def test_fold_mismatch_marks_contrasts_not_comparable(tmp_path):
    """A failed fold check must QUALIFY the contrasts (like summarize_vs_h1's fold_comparable gate),
    not just print one stdout line while publishing CIs as if the pairing held."""
    root = tmp_path / "scr"
    _full_family(root)
    _lane(root, EXPRESSION_ONLY, 0.0861, 0, n_train=21262, n_val=4400)
    _lane(root, EXPRESSION_ONLY, 0.0859, 1, n_train=2000, n_val=400)    # different fold
    agg = aggregate_seeds(range(5), screening_root=root, registry_path=None)
    h2a = agg["contrasts"]["h2a"]
    assert h2a["fold_comparable"] is False
    assert "not comparable" in h2a["verdict"].lower()


def test_string_seed_keys_do_not_silently_disable_the_guards(tmp_path):
    """Seeds arriving as strings resolve the parquet path fine but defeat both int-keyed guards at once,
    yielding a fully-numbered report with no staleness and no fold check."""
    root = tmp_path / "scr"
    reg = tmp_path / "reg.yaml"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    rid = register_run(EXPRESSION_ONLY, "H2a", "q_pre", "blocked_target_ood", 0, None, path=reg)
    log_run(rid, "completed", {"systema": 0.10}, "b.pt", 1.0, path=reg)
    agg = aggregate_seeds(["0"], names=(EXPRESSION_ONLY,), screening_root=root, registry_path=reg)
    assert agg["fold"]["observed_splits"] == ["blocked_target_ood"]   # the guard actually engaged
    assert agg["fold"]["registry_evidence"] is True


def test_unformable_contrast_is_recorded_not_silently_absent(tmp_path):
    """A contrast skipped because a member is outside `names` must be reported, so a reader cannot
    confuse 'not computed' with 'not significant'."""
    root = tmp_path / "scr"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    _lane(root, TYPED_STATIC, 0.09, 0)
    agg = aggregate_seeds([0], names=(EXPRESSION_ONLY, TYPED_STATIC), screening_root=root,
                          registry_path=None)
    assert "promotion_margin" in agg["contrasts_skipped"]


def test_unbalanced_per_config_seed_coverage_is_flagged(tmp_path):
    """per_config means are computed over each config's OWN seeds; when those differ the ranking
    compares non-comparable bases and must say so."""
    root = tmp_path / "scr"
    _lane(root, EXPRESSION_ONLY, 0.10, 0)
    _lane(root, EXPRESSION_ONLY, 0.11, 1)
    _lane(root, TYPED_STATIC, 0.09, 0)                       # only one seed
    agg = aggregate_seeds([0, 1], names=(EXPRESSION_ONLY, TYPED_STATIC), screening_root=root,
                          registry_path=None)
    assert agg["balanced"] is False
    assert agg["common_seeds"] == [0]


def test_exit_code_is_nonzero_when_the_report_is_not_comparable(tmp_path):
    """A self-declared incomparable/incomplete report must not exit 0 — an unattended campaign or CI
    gate keyed on exit status would record it green."""
    root = tmp_path / "scr"
    _lane(root, EXPRESSION_ONLY, 0.10, 0, n_train=21262, n_val=4400)
    _lane(root, EXPRESSION_ONLY, 0.11, 1, n_train=2000, n_val=400)
    agg = aggregate_seeds([0, 1], names=(EXPRESSION_ONLY,), screening_root=root, registry_path=None)
    assert _exit_code(agg) != 0
