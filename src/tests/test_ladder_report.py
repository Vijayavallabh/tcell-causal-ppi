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
    _rung(tmp_path, "permuted_d400", base, base)          # control present and NOT clearing
    r = run(str(tmp_path), None)
    assert r["floor"] is None and r["floor_status"] == "non_monotone"
    assert any("NON-MONOTONE" in n for n in r["notes"])


def test_a_ladder_nothing_clears_reports_the_floor_as_above_the_largest_rung(tmp_path):
    base = [0.080, 0.081, 0.079, 0.080]
    for name in ("d020", "d100", "d400"):
        _rung(tmp_path, name, _lift(base, 0.0002), base)
    _rung(tmp_path, "permuted_d400", base, base)          # control present and NOT clearing
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
    rungs = {"d100": {"delta": 0.10, "permuted": False}, "d200": {"delta": 0.20, "permuted": False},
             "permuted_d400": {"delta": 0.40, "permuted": True}}
    contrasts = {"d100": {"n": 4, "mean": -0.05, "survives_family_wise": True},
                 "d200": {"n": 4, "mean": -0.06, "survives_family_wise": True},
                 "permuted_d400": {"n": 4, "mean": 0.0, "survives_family_wise": False}}
    v = _verdict(rungs, contrasts)
    assert v["floor"] is None and v["floor_status"] == "above_ladder"


def test_the_post_hoc_increment_subtracts_the_zero_point_seed_by_seed(tmp_path, monkeypatch):
    """POST-HOC secondary, committed before the last rungs ran. A rung whose gap equals the
    un-injected gap must show an increment of zero: that is the whole point, since the primary tests
    against zero and the un-injected gap on this fold is already about +0.005."""
    import numpy as np

    from tcell_pipeline.screening import ladder_report as lr

    ref = tmp_path / "ref"
    base = [0.080, 0.081, 0.079, 0.080]
    gap0 = [0.004, 0.006, 0.005, 0.005]                        # the zero point, per seed
    _rung(ref, ".", [b + g for b, g in zip(base, gap0)], base)  # reference lanes
    monkeypatch.setattr(lr, "REFERENCE_ROOT", str(ref / "."))

    root = tmp_path / "ladder"
    _rung(root, "d020", [b + g for b, g in zip(base, gap0)], base)          # identical to the zero point
    _rung(root, "d200", [b + g + 0.02 for b, g in zip(base, gap0)], base)   # zero point plus 0.02

    rungs = lr.collect(str(root))
    incr = lr.increment_over_zero(rungs)
    assert incr["d020"]["mean"] == pytest.approx(0.0, abs=1e-9)
    assert incr["d200"]["mean"] == pytest.approx(0.02, abs=1e-9)
    # and the PRIMARY on the same data still reports the pre-existing gap, which is the gap the
    # increment exists to remove
    prim = lr.run(str(root), None)
    assert prim["contrasts"]["d020"]["mean"] == pytest.approx(np.mean(gap0), abs=1e-9)


# --- Amendment 9: the arm and the reference root are parameters ------------------------------------
def _multi_arm_rung(root, name, arms: dict, seeds=SEEDS):
    """One rung carrying several graph arms at once, so a test can tell WHICH one was read."""
    d = root / name
    for arm, vals in arms.items():
        (d / arm).mkdir(parents=True, exist_ok=True)
        for s, v in zip(seeds, vals):
            pd.DataFrame([{"name": arm, "seed": s, "status": "completed", "systema": v}]).to_parquet(
                d / arm / f"{s}.parquet")
    return d


def test_the_arm_parameter_actually_selects_the_arm(tmp_path):
    """Amendment 9 runs a SECOND ladder on `condition_gated`, and 9.11 requires this parameterisation
    to land before its lanes do. The guard is only evidence if it can fail, so the fixture gives the
    two graph arms DIFFERENT numbers on the same rung: if the arm were ignored, both calls would
    return the same contrast and the assertion below would break."""
    base = [0.080, 0.081, 0.079, 0.080]
    _multi_arm_rung(tmp_path, "d020", {
        "expression_only": base,
        "untyped_gnn": _lift(base, 0.010),          # the good detector
        "condition_gated": _lift(base, 0.001),      # the poorer one, an order of magnitude smaller
    })
    untyped = run(str(tmp_path), None, arm="untyped_gnn")
    gated = run(str(tmp_path), None, arm="condition_gated")

    assert untyped["arm"] == "untyped_gnn" and gated["arm"] == "condition_gated"
    assert untyped["contrasts"]["d020"]["mean"] == pytest.approx(0.010, abs=5e-4)
    assert gated["contrasts"]["d020"]["mean"] == pytest.approx(0.001, abs=5e-4)
    # The whole point: the two ladders disagree, so an ignored --arm cannot pass this.
    assert untyped["contrasts"]["d020"]["mean"] > gated["contrasts"]["d020"]["mean"] + 0.005


def test_a_missing_arm_shrinks_n_rather_than_falling_back_to_another_arm(tmp_path):
    """Asking for an arm that was never trained must produce n=0 for that rung, NOT the numbers of
    whichever arm happens to be on disk. Silently reading a different arm is the manufactured-result
    hazard: it returns a plausible number for an experiment that never ran."""
    base = [0.080, 0.081, 0.079, 0.080]
    _multi_arm_rung(tmp_path, "d020", {"expression_only": base, "untyped_gnn": _lift(base, 0.010)})
    gated = run(str(tmp_path), None, arm="condition_gated")
    assert gated["contrasts"]["d020"]["n"] == 0
    assert gated["contrasts"]["d020"]["mean"] is None


def test_the_reference_root_parameter_moves_the_zero_point(tmp_path):
    """Amendment 9.5 reads its zero point from `screening_lambda0` rather than the module default,
    because only there does `condition_gated`'s gate stay live. If --reference-root were ignored the
    two increments below would be identical."""
    from tcell_pipeline.screening import ladder_report as lr

    base = [0.080, 0.081, 0.079, 0.080]
    root = tmp_path / "ladder"
    _multi_arm_rung(root, "d200", {"expression_only": base, "untyped_gnn": _lift(base, 0.020)})

    flat = tmp_path / "ref_flat"
    _multi_arm_rung(flat, ".", {"expression_only": base, "untyped_gnn": base})          # zero point 0
    lifted = tmp_path / "ref_lifted"
    _multi_arm_rung(lifted, ".", {"expression_only": base, "untyped_gnn": _lift(base, 0.005)})

    rungs = lr.collect(str(root))
    a = lr.increment_over_zero(rungs, reference_root=str(flat / "."))
    b = lr.increment_over_zero(rungs, reference_root=str(lifted / "."))
    assert a["d200"]["mean"] == pytest.approx(0.020, abs=1e-3)
    assert b["d200"]["mean"] == pytest.approx(0.015, abs=1e-3)


# --- Amendment 9.2: gate health and lane configuration are REPORTED, not assumed --------------------
def _gated_rung(root, name, better, worse, lam=0.0, seeds=SEEDS):
    """A rung on the one arm with a LIVE gate, carrying the lambda_graph the lane actually ran at."""
    d = root / name
    for arm, vals in (("condition_gated", better), ("expression_only", worse)):
        (d / arm).mkdir(parents=True, exist_ok=True)
        for s, v in zip(seeds, vals):
            pd.DataFrame([{"name": arm, "seed": s, "status": "completed", "systema": v,
                           "lambda_graph": lam}]).to_parquet(d / arm / f"{s}.parquet")
    return d


def _lane_history(root, rung, seed, gate_means, arm="condition_gated"):
    """The trainer's own per-lane history, which is where gate_mean actually lives.

    NOT a log file. `run_screening`'s lane log carries no per-epoch line at all - the "gate mean"
    wording belongs to run_rescreen_lambda0.sh, a RUNNER that post-processes this very file. A first
    version of the gate check scraped logs and would have reported "unavailable" for every lane of a
    perfectly healthy campaign."""
    import json
    d = root / rung / arm / str(seed) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "stage_a_history.json").write_text(json.dumps(
        [{"epoch": i, "train": {"gate_mean": g}, "val": {}} for i, g in enumerate(gate_means)]))


def test_a_collapsed_gate_shrinks_n_instead_of_counting_as_evidence(tmp_path):
    """Amendment 3.4: a lane whose mean edge gate falls to <=1e-3 is an UNDECIDABLE experiment,
    reported as such and NEVER as evidence the graph does not help. So it must leave n smaller, not
    contribute a number. The guard can fail: seed 0's lane below carries a healthy gate in the sibling
    assertion and a dead one here, and only the dead one may shrink n."""
    from tcell_pipeline.screening import ladder_report as lr

    base = [0.080, 0.081, 0.079, 0.080]
    root = tmp_path / "ladder"
    _gated_rung(root, "d200", _lift(base, 0.02), base)
    for s in SEEDS:
        _lane_history(root, "d200", s, [0.7, 0.6, 0.55])        # all healthy
    healthy = lr.run(str(root), None, arm="condition_gated")
    assert healthy["contrasts"]["d200"]["n"] == 4
    assert healthy["gate_health"]["status"] == "live"

    _lane_history(root, "d200", 0, [0.7, 0.0004, 0.0001])      # seed 0's gate dies
    collapsed = lr.run(str(root), None, arm="condition_gated")
    assert collapsed["contrasts"]["d200"]["n"] == 3, "an undecidable lane was counted as evidence"
    assert collapsed["dropped_for_gate_collapse"] == ["d200_condition_gated_s0"]
    assert any("GATE COLLAPSED" in n for n in collapsed["notes"])
    assert collapsed["gate_health"]["status"] == "collapsed"


def test_a_live_gate_arm_run_at_the_wrong_lambda_is_flagged_and_a_pinned_one_is_not(tmp_path):
    """The trap Amendment 9.2 exists to close: run_a2_ladder.sh does not pass --lambda-graph, so the
    config default of 0.01 applies, and at 0.01 the gate is annihilated inside epoch 0. That is fatal
    for condition_gated and harmless for every arm that pins its gate, so the warning must
    discriminate - a check that cries wolf on Amendment 6's landed ladder would be turned off."""
    from tcell_pipeline.screening import ladder_report as lr

    base = [0.080, 0.081, 0.079, 0.080]
    bad = tmp_path / "bad"
    _gated_rung(bad, "d200", _lift(base, 0.02), base, lam=0.01)
    r = lr.run(str(bad), None, arm="condition_gated")
    assert r["lambda_graph_ok"] is False
    assert any("LIVE GATE" in n and "lambda_graph=0" in n for n in r["notes"])

    good = tmp_path / "good"
    _gated_rung(good, "d200", _lift(base, 0.02), base, lam=0.0)
    ok = lr.run(str(good), None, arm="condition_gated")
    assert ok["lambda_graph_ok"] is True
    assert not any("LIVE GATE" in n for n in ok["notes"])


def test_missing_gate_logs_are_a_named_gap_for_a_live_gate_arm_not_a_silent_pass(tmp_path):
    """A skipped check that prints nothing is indistinguishable from a passing one, which is the
    failure mode this project keeps finding in its own harness."""
    from tcell_pipeline.screening import ladder_report as lr

    base = [0.080, 0.081, 0.079, 0.080]
    root = tmp_path / "ladder"
    _gated_rung(root, "d200", _lift(base, 0.02), base)
    r = lr.run(str(root), None, arm="condition_gated", log_dir=str(tmp_path / "absent"))
    assert r["gate_health"]["status"] == "unavailable"
    assert any("gate health UNCHECKED" in n for n in r["notes"])


# --- the two refusals a stopped-early campaign depends on ------------------------------------------
def test_a_missing_permuted_control_refuses_a_floor_rather_than_passing_the_veto(tmp_path):
    """`any()` over an empty list is False, so a ladder whose control rung never landed would sail
    past the veto and name a floor. Amendment 6.7 makes that veto absolute and prior to every other
    reading, and it cannot be applied to a rung that does not exist. The guard can fail: adding the
    control back to the SAME fixture must restore a normal verdict."""
    base = [0.080, 0.081, 0.079, 0.080]
    _rung(tmp_path, "d020", _lift(base, 0.010), base)
    _rung(tmp_path, "d400", _lift(base, 0.030), base)
    r = run(str(tmp_path), None)
    assert r["floor"] is None and r["floor_status"] == "control_missing"
    assert any("NO PERMUTED CONTROL" in n for n in r["notes"])

    _rung(tmp_path, "permuted_d400", base, base)      # a control that does NOT clear
    ok = run(str(tmp_path), None)
    assert ok["floor_status"] != "control_missing", "the guard cannot fail, so it proves nothing"


def test_an_empty_ladder_refuses_instead_of_manufacturing_a_negative(tmp_path):
    """Before this, a ladder with NO landed lanes reported "the floor is ABOVE the largest rung
    tested, which is itself a result: this pipeline cannot see an injected graph signal even at that
    size" - an affirmative negative conjured from zero data. A reader skimming the verdict line would
    conclude the arm is blind. An untested rung is UNKNOWN, not cleared."""
    base = [0.080, 0.081, 0.079, 0.080]
    _rung(tmp_path, "d020", _lift(base, 0.010), base)
    _rung(tmp_path, "permuted_d400", base, base)
    (tmp_path / "d400" / "expression_only").mkdir(parents=True)   # a rung dir with no lanes at all
    (tmp_path / "d400" / "untyped_gnn").mkdir(parents=True)

    r = run(str(tmp_path), None)
    assert r["floor"] is None and r["floor_status"] == "incomplete_ladder"
    assert any("LADDER INCOMPLETE" in n and "n<2" in n for n in r["notes"])
    assert not any("above_ladder" in str(n) for n in r["notes"])


def test_a_reduced_but_complete_ladder_still_gets_a_verdict(tmp_path):
    """Amendment 9.9 expects a stopped campaign to be REPORTED at whatever n landed, labelled
    preliminary. So the refusal above must not swallow a ladder that is merely underpowered: n=2 on
    every rung is thin, but it is a paired contrast and it gets a verdict with its n."""
    base = [0.080, 0.081]
    _rung(tmp_path, "d020", _lift(base, 0.010), base, seeds=(0, 1))
    _rung(tmp_path, "d400", _lift(base, 0.030), base, seeds=(0, 1))
    _rung(tmp_path, "permuted_d400", base, base, seeds=(0, 1))
    r = run(str(tmp_path), None, seeds=(0, 1))
    assert r["floor_status"] not in ("incomplete_ladder", "control_missing")
    assert any("INCOMPLETE at n<2" in n or "preliminary" in n for n in r["notes"]) or r["floor"]
