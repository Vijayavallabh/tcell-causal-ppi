"""A2(a) report tests. The verdict rule is fixed in Amendment 6.7, so the tests drive it through every
branch it can take on synthetic rungs -- including the two branches designed to REFUSE a number."""
from __future__ import annotations

import pandas as pd
import pytest

from tcell_pipeline.screening.ladder_report import _delta_of, _verdict, collect, run

SEEDS = (0, 1, 2, 3)
# Per-seed jitter, because an EXACTLY constant paired delta is undecidable by design in this pipeline
# (identical deltas are the signature of a seed that never propagated, not of a strong effect), so a
# fixture without it exercises the degenerate branch instead of the one under test.
_JITTER = (0.0004, -0.0003, 0.0002, -0.0002)


def _lift(base, amount):
    return [b + amount + j for b, j in zip(base, _JITTER)]


def _rung(root, name, better, worse, seeds=SEEDS):
    d = root / name
    for arm, vals in (("untyped_gnn", better), ("expression_only", worse)):
        (d / arm).mkdir(parents=True, exist_ok=True)
        for s, v in zip(seeds, vals):
            pd.DataFrame([{"name": arm, "seed": s, "status": "completed", "systema": v}]).to_parquet(
                d / arm / f"{s}.parquet")
    return d


def test_delta_is_read_off_the_directory_name():
    assert _delta_of("d020") == pytest.approx(0.02)
    assert _delta_of("permuted_d400") == pytest.approx(0.40)
    assert _delta_of("not_a_rung") is None


def test_a_clean_monotone_ladder_names_the_smallest_clearing_rung(tmp_path):
    """Rungs 0.02 and 0.05 sit in the noise, 0.10 and up are clearly separated: the floor is 0.10."""
    base = [0.080, 0.081, 0.079, 0.080]
    _rung(tmp_path, "d020", _lift(base, 0.0005), base)
    _rung(tmp_path, "d050", _lift(base, 0.0008), base)
    _rung(tmp_path, "d100", _lift(base, 0.030), base)
    _rung(tmp_path, "d200", _lift(base, 0.060), base)
    _rung(tmp_path, "permuted_d400", _lift(base, 0.0002), base)
    r = run(str(tmp_path), None)
    assert r["floor_status"] == "measured" and r["floor"] == pytest.approx(0.10)
    assert r["control_clears"] is False
    assert r["family_size"] == 5


def test_a_control_that_clears_refuses_to_report_a_floor(tmp_path):
    """If a permuted neighbourhood is recovered too, the ladder measures injected magnitude rather
    than graph structure, and the rule says report NO floor rather than the smallest number."""
    base = [0.080, 0.081, 0.079, 0.080]
    _rung(tmp_path, "d100", _lift(base, 0.030), base)
    _rung(tmp_path, "d200", _lift(base, 0.060), base)
    _rung(tmp_path, "permuted_d400", _lift(base, 0.050), base)
    r = run(str(tmp_path), None)
    assert r["control_clears"] is True and r["floor"] is None
    assert r["floor_status"] == "control_failed"
    assert any("PERMUTED CONTROL" in n for n in r["notes"])


def test_a_non_monotone_ladder_is_a_red_flag_not_a_floor(tmp_path):
    """A small rung clearing while a larger one does not is not a floor; it is a signal that something
    other than the injected size is driving the result."""
    base = [0.080, 0.081, 0.079, 0.080]
    _rung(tmp_path, "d050", _lift(base, 0.030), base)     # clears
    _rung(tmp_path, "d100", _lift(base, 0.0003), base)    # does not
    _rung(tmp_path, "d200", _lift(base, 0.0004), base)    # does not
    r = run(str(tmp_path), None)
    assert r["floor"] is None and r["floor_status"] == "non_monotone"
    assert any("NON-MONOTONE" in n for n in r["notes"])


def test_a_ladder_nothing_clears_reports_the_floor_as_above_the_largest_rung(tmp_path):
    base = [0.080, 0.081, 0.079, 0.080]
    for name in ("d020", "d100", "d400"):
        _rung(tmp_path, name, _lift(base, 0.0002), base)
    r = run(str(tmp_path), None)
    assert r["floor"] is None and r["floor_status"] == "above_ladder"
    assert any("ABOVE the largest rung" in n for n in r["notes"])


def test_a_short_rung_is_flagged_rather_than_quietly_averaged(tmp_path):
    """Rail 5: fewer than four paired seeds is preliminary and must be labelled with its n."""
    base = [0.080, 0.081, 0.079, 0.080]
    _rung(tmp_path, "d100", _lift(base, 0.03), base)
    _rung(tmp_path, "d200", [b + 0.06 for b in base[:2]], base[:2], seeds=(0, 1))
    r = run(str(tmp_path), None)
    assert any("INCOMPLETE" in n for n in r["notes"])
    assert r["contrasts"]["d200"]["n"] == 2


def test_missing_and_uncompleted_lanes_shrink_n_rather_than_being_used(tmp_path):
    d = _rung(tmp_path, "d100", [0.09, 0.09, 0.09, 0.09], [0.08, 0.08, 0.08, 0.08])
    pd.DataFrame([{"name": "untyped_gnn", "seed": 2, "status": "failed", "systema": 99.0}]).to_parquet(
        d / "untyped_gnn" / "2.parquet")
    (d / "untyped_gnn" / "3.parquet").unlink()
    rungs = collect(str(tmp_path))
    assert set(rungs["d100"]["better"]) == {0, 1}
    assert 99.0 not in rungs["d100"]["better"].values()


def test_verdict_needs_a_positive_not_merely_a_significant_rung():
    """A rung where the graph arm is significantly WORSE must not be read as detection."""
    rungs = {"d100": {"delta": 0.10, "permuted": False}, "d200": {"delta": 0.20, "permuted": False}}
    contrasts = {"d100": {"n": 4, "mean": -0.05, "survives_family_wise": True},
                 "d200": {"n": 4, "mean": -0.06, "survives_family_wise": True}}
    v = _verdict(rungs, contrasts)
    assert v["floor"] is None and v["floor_status"] == "above_ladder"
