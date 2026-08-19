"""B1 verdict tests. Every branch of Amendment 7.5 is driven, because the whole value of fixing the
outcomes in advance is lost if the code that reads them only handles the outcome we expect."""
from __future__ import annotations

import pytest

from tcell_pipeline.screening.b1_report import _verdict

GAP = {"mean": 0.0176, "n": 5}


def _c(mean, survives, n=5, arm="typed_gcnnorm", p=0.01):
    return {"mean": mean, "survives_family_wise": survives, "n": n, "arm": arm,
            "p_bonferroni": p, "family_size": 1}


def test_a_clearing_positive_arm_is_named_a_route_with_its_share():
    c = {"D3": _c(0.0088, True)}
    v = _verdict(c, GAP)
    assert v["routes"] == ["D3"]
    assert c["D3"]["recovery_share"] == pytest.approx(0.5, abs=0.01)
    assert "RECOVERS" in " ".join(v["notes"])


def test_a_null_arm_is_not_the_route_and_says_so_with_its_family_size():
    v = _verdict({"D3": _c(0.0004, False, p=1.0)}, GAP)
    assert v["routes"] == []
    joined = " ".join(v["notes"])
    assert "NULL" in joined and "not the route" in joined
    assert "DISTRIBUTED across the message form" in joined


def test_a_significantly_negative_arm_is_reported_as_such_not_as_no_effect():
    """Amendment 7.5. An arm whose removal makes things WORSE is a different finding from an arm that
    does nothing, and folding the two together would hide it."""
    v = _verdict({"D3": _c(-0.0120, True)}, GAP)
    assert v["routes"] == []
    joined = " ".join(v["notes"])
    assert "WORSE" in joined and "NOT folded" in joined
    assert "DISTRIBUTED" not in joined, "a negative result is not 'no component clears'"


def test_an_underpowered_arm_is_preliminary_and_never_a_route():
    v = _verdict({"D3": _c(0.0300, True, n=2)}, GAP)
    assert v["routes"] == [] and "PRELIMINARY" in " ".join(v["notes"])


def test_several_clearing_arms_are_all_reported_and_none_claimed_exclusively():
    v = _verdict({"D3": _c(0.0100, True), "D4": _c(0.0080, True, arm="typed_unsigned")}, GAP)
    assert set(v["routes"]) == {"D3", "D4"}
    assert "NONE is claimed exclusively" in " ".join(v["notes"])


def test_an_undecidable_gap_leaves_the_share_absent_rather_than_dividing_by_zero():
    c = {"D3": _c(0.0088, True)}
    _verdict(c, {"mean": None, "n": 0})
    assert c["D3"]["recovery_share"] is None


def test_an_arm_with_no_landed_lane_is_excluded_from_the_family_not_counted_as_null():
    """Amendment 7.3. Counting an unrun arm would inflate m and weaken every arm that did run."""
    from tcell_pipeline.screening.b1_report import collect
    col = collect(root=__import__("pathlib").Path("/nonexistent"))
    assert col["family_size"] == 0 and col["arms_present"] == []
    assert "typed_gcnnorm" in col["arms_absent"]
