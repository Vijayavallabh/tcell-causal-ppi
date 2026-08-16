"""A1 report tests. The verdict is a 2x2 over two signed contrasts, so the tests drive each cell and
pin the sign convention Amendment 4b had to correct: both contrasts are (arm - typed_static), so
POSITIVE means the intervention improved on the typed encoder."""
from __future__ import annotations

import pandas as pd
import pytest

from tcell_pipeline.screening.a1_report import _verdict, metric_by_seed, run

SEEDS = (0, 1, 2, 3, 4)


_SEED_SD = 0.003    # about the frozen fold's own per-seed spread, so a "null" lift really is null


def _jitter(arm: str):
    """Per-ARM seed noise. A shared jitter cancels in the paired difference, leaving an exactly
    constant delta, which this pipeline reports as UNDECIDABLE by design rather than as significant —
    so a fixture without per-arm noise tests the degenerate branch instead of the intended one.
    Seeded with crc32 rather than hash(), which is randomised per process and would make the fixture
    non-deterministic across runs."""
    import zlib

    import numpy as np
    return np.random.default_rng(zlib.crc32(arm.encode())).normal(0, _SEED_SD, size=len(SEEDS))


def _arm(root, arm, base, lift=0.0):
    (root / arm).mkdir(parents=True, exist_ok=True)
    for s, j in zip(SEEDS, _jitter(arm)):
        pd.DataFrame([{"name": arm, "seed": s, "status": "completed",
                       "systema": base + lift + float(j)}]).to_parquet(root / arm / f"{s}.parquet")


def _root(tmp_path, shared_lift, permuted_lift, base=0.074):
    """typed_static at `base`; the two diagnostic arms lifted by the given amounts."""
    _arm(tmp_path, "expression_only", 0.0861)
    _arm(tmp_path, "untyped_gnn", 0.0904)
    _arm(tmp_path, "typed_static", base)
    _arm(tmp_path, "typed_shared", base, shared_lift)
    _arm(tmp_path, "typed_permuted", base, permuted_lift)
    return str(tmp_path)


def test_a_positive_contrast_means_the_intervention_improved_on_the_typed_encoder(tmp_path):
    """The sign convention Amendment 4b fixes. Lift the shared arm above typed_static and D1 must come
    out POSITIVE; if the subtraction were the other way, every verdict in the 2x2 would be mirrored."""
    r = run(_root(tmp_path, shared_lift=0.02, permuted_lift=0.0), None)
    assert r["family"]["D1"]["mean"] > 0
    assert r["d1_state"] == "positive"


def test_removing_the_multiplicity_helping_while_a_random_partition_does_not(tmp_path):
    r = run(_root(tmp_path, shared_lift=0.02, permuted_lift=0.0002), None)
    assert (r["d1_state"], r["d2_state"]) == ("positive", "null")
    assert "jointly the route" in r["reading"]
    assert "Amendment 4.3" in r["reading"], "the identifiability caveat must travel with this verdict"


def test_a_random_partition_beating_the_true_one_is_called_actively_misleading(tmp_path):
    r = run(_root(tmp_path, shared_lift=0.0003, permuted_lift=0.02), None)
    assert (r["d1_state"], r["d2_state"]) == ("null", "positive")
    assert "worse than noise" in r["reading"]


def test_neither_intervention_moving_sends_the_search_elsewhere(tmp_path):
    r = run(_root(tmp_path, shared_lift=0.0003, permuted_lift=0.0002), None)
    assert (r["d1_state"], r["d2_state"]) == ("null", "null")
    assert "signed messages" in r["reading"]


def test_a_significantly_negative_contrast_is_not_read_as_a_null(tmp_path):
    """Tying the weights making things WORSE is a finding of its own, not the absence of one."""
    r = run(_root(tmp_path, shared_lift=-0.02, permuted_lift=0.0002), None)
    assert r["d1_state"] == "negative"
    assert "per-relation modules are doing real work" in r["reading"]


def test_a_diagnostic_arm_beating_no_graph_is_sent_to_the_contradiction_stop(tmp_path):
    """Amendment 4.5: a diagnostic arm may not promote a graph claim, and a positive against
    expression_only still trips rail 4."""
    root = _root(tmp_path, shared_lift=0.02, permuted_lift=0.0)
    _arm(tmp_path, "typed_shared", 0.0861, 0.02)      # now clearly above expression_only
    r = run(root, None)
    assert "shared_vs_nograph" in r["contradiction_candidates"]


def test_missing_and_failed_lanes_shrink_n_rather_than_being_used(tmp_path):
    root = _root(tmp_path, shared_lift=0.02, permuted_lift=0.0)
    pd.DataFrame([{"name": "typed_shared", "seed": 1, "status": "failed",
                   "systema": 99.0}]).to_parquet(tmp_path / "typed_shared" / "1.parquet")
    (tmp_path / "typed_shared" / "4.parquet").unlink()
    got = metric_by_seed(tmp_path, "typed_shared")
    assert set(got) == {0, 2, 3} and 99.0 not in got.values()
    r = run(root, None)
    assert r["family"]["D1"]["n"] == 3
    assert any("PRELIMINARY" in n for n in r["notes"])


def test_underpowered_is_its_own_state_not_a_null():
    d1 = {"n": 2, "mean": 0.02, "survives_family_wise": False}
    d2 = {"n": 5, "mean": 0.0, "survives_family_wise": False}
    v = _verdict(d1, d2)
    assert v["d1_state"] == "underpowered" and v["reading"] == "unclassified"
    assert any("rail 5" in n for n in v["notes"])
