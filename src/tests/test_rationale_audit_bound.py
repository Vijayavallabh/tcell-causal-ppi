"""A5 tests. The point of the module is that a ratio and an absolute size can disagree, so the tests
make them disagree on purpose and check the module says so."""
from __future__ import annotations

import json

import pytest

from tcell_pipeline.screening.rationale_audit_bound import (
    EDGE_COUNTS,
    TOTAL_PP_EDGES,
    paired_gap,
    per_edge_share,
    run,
)


def test_paired_gap_separates_a_huge_ratio_from_a_negligible_size():
    """The exact situation the audit is in on necessity: the rationale beats its control by half again,
    and both numbers are of order 1e-7. A summary that reported only the fraction would call this a
    strong result."""
    import numpy as np
    rng = np.random.default_rng(0)                       # varying deltas, or the paired t has no variance
    noise = rng.normal(0, 2e-8, size=40)
    cases = [{"v": 3e-7 + n, "r": 2e-7} for n in noise]
    g = paired_gap(cases, "v", "r")
    assert g["gap_as_fraction_of_control"] > 0.4                             # a ~50% "improvement"
    assert abs(g["gap"]) < 1e-6                                             # ...of one part in ten million
    assert g["p_value"] < 1e-6                                              # and it is highly significant
    assert g["ci_low"] > 0 and g["n"] == 40


def test_per_edge_share_can_reverse_the_raw_ranking():
    """A source that is merely large must not out-rank a small informative one after normalisation. If
    the division were dropped, both rankings would be identical and the module would be reporting an
    edge count as though it were evidence quality."""
    deltas = {"string": 0.36, "huri": 0.02}
    counts = {"string": 8_000_000, "huri": 50_000}
    out = per_edge_share(deltas, counts, total=8_050_000)
    assert out["string"]["delta"] > out["huri"]["delta"]                     # raw: string wins
    assert out["huri"]["delta_per_pct_of_edges"] > out["string"]["delta_per_pct_of_edges"]
    assert out["string"]["share_pct"] == pytest.approx(99.38, abs=0.05)


def test_per_edge_share_skips_sources_with_no_edge_count():
    """A source absent from the count table is dropped rather than divided by zero or silently given a
    share of nothing."""
    out = per_edge_share({"string": 0.3, "unknown_source": 0.9}, {"string": 100}, total=100)
    assert set(out) == {"string"}


def test_recorded_edge_counts_are_internally_consistent():
    """The counts are a constant with a provenance comment, so the one thing a test can still catch is
    the constant drifting out of step with its own total."""
    assert sum(EDGE_COUNTS.values()) == pytest.approx(TOTAL_PP_EDGES, rel=0.02)
    assert EDGE_COUNTS["string"] / TOTAL_PP_EDGES > 0.8   # the 86%-functional fact the paper leans on


def test_run_reads_the_landed_audit_and_reports_both_views(tmp_path):
    r = run(out=tmp_path / "bound.json")
    assert r["n_cases"] == 50 and r["gates_live"] is True
    assert r["sufficiency"]["gap"] > 0 and r["sufficiency"]["gap_as_fraction_of_control"] < 0.05
    assert r["necessity"]["gap_as_fraction_of_control"] > 0.2      # big ratio
    assert abs(r["necessity"]["gap"]) < 1e-5                       # negligible size
    assert set(r["source_ablation"]) >= {"string", "bioplex", "huri", "corum"}
    written = json.loads((tmp_path / "bound.json").read_text())
    assert written["n_cases"] == 50
