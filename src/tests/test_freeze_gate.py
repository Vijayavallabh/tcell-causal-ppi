"""Near-null-signal freeze gate tests (feat-008 §c) — pure synthetic, no model, no marts.

The gate exists because in this project's near-null-signal regime a Stage-B calibration head or a
rationale head can look good purely by fitting noise. It refuses to freeze a fit whose measured
advantage over its OWN control is indistinguishable from zero, and it must stay refusing on every
degenerate input that a naive significance check would read as a pass:

  - a fit that is reliably WORSE than its control (CI excludes zero, but on the wrong side)
  - zero variance across units (identical deltas -> undecidable, NOT p=0)
  - an empty / missing required contrast (all([]) is vacuously true -> must NOT be a freeze)
  - a raw hit that dies under family-wise correction

Each of those is constructed here as an input that DEFEATS a gate written the obvious way.
"""
from __future__ import annotations

import pytest

from tcell_pipeline.training.freeze_gate import (
    FREEZE,
    REFUSE,
    UNDECIDABLE,
    evaluate_gate,
    exit_code,
    render,
)

# per-unit deltas used to build arms; unit ids are strings so nothing depends on int ordering
_STRONG = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]          # p ~ 2e-6: clears any correction
_MODEST = [0.05, 0.01, 0.04, 0.02, 0.06, 0.00]          # raw p=0.027, Bonferroni x4 = 0.107
_STRADDLE = [0.10, -0.11, 0.09, -0.12, 0.10, -0.11]     # p=0.87: indistinguishable


def _arms(deltas, *, base: float = 1.0, higher_is_better: bool = True) -> dict:
    """Two paired arms whose per-unit ADVANTAGE (oriented by ``higher_is_better``) is ``deltas``."""
    units = [f"u{i}" for i in range(len(deltas))]
    control = {u: base for u in units}
    sign = 1.0 if higher_is_better else -1.0
    fit = {u: base + sign * d for u, d in zip(units, deltas)}
    return {"fit": fit, "control": control, "higher_is_better": higher_is_better}


def test_gate_clears_on_a_consistent_advantage():
    r = evaluate_gate({"suff": _arms(_STRONG)})
    assert r["decision"] == FREEZE
    assert r["contrasts"]["suff"]["decision"] == FREEZE
    assert r["contrasts"]["suff"]["mean"] > 0
    assert exit_code(r) == 0


def test_gate_fires_when_the_advantage_is_indistinguishable():
    r = evaluate_gate({"suff": _arms(_STRADDLE)})
    assert r["decision"] == REFUSE                      # CI crosses zero -> near-null, do not freeze
    assert r["contrasts"]["suff"]["ci_excludes_zero"] is False
    assert exit_code(r) != 0


def test_gate_refuses_a_fit_that_is_reliably_worse_than_its_control():
    """The input that defeats a gate checking only significance: a fit that LOSES to its control has
    ci_excludes_zero=True and survives_family_wise=True — the direction is the only thing separating
    it from a pass."""
    r = evaluate_gate({"suff": _arms([-d for d in _STRONG])})
    c = r["contrasts"]["suff"]
    assert c["ci_excludes_zero"] is True and c["survives_family_wise"] is True   # "significant"...
    assert c["mean"] < 0                                                          # ...on the wrong side
    assert c["decision"] == REFUSE and r["decision"] == REFUSE
    assert exit_code(r) != 0


def test_zero_variance_is_undecidable_not_maximally_significant():
    """Identical deltas prove the units carry no information; reporting p=0 would turn that into the
    strongest possible evidence."""
    r = evaluate_gate({"suff": _arms([0.10] * 6)})
    c = r["contrasts"]["suff"]
    assert c["p_value"] is None and c["ci_excludes_zero"] is None
    assert c["decision"] == UNDECIDABLE and r["decision"] == UNDECIDABLE
    assert exit_code(r) not in (0,)


def test_empty_contrast_set_is_undecidable_not_a_vacuous_pass():
    r = evaluate_gate({})
    assert r["decision"] == UNDECIDABLE                 # all([]) is True — absence of evidence is not a pass
    assert exit_code(r) != 0


def test_missing_required_contrast_is_undecidable():
    r = evaluate_gate({"suff": _arms(_STRONG)}, required=("suff", "nec"))
    assert r["contrasts"]["nec"]["decision"] == UNDECIDABLE
    assert r["contrasts"]["nec"]["verdict"].startswith("not computed")
    assert r["decision"] == UNDECIDABLE
    assert exit_code(r) != 0


def test_single_unit_is_undecidable():
    r = evaluate_gate({"suff": _arms([0.10])})
    assert r["contrasts"]["suff"]["n"] == 1
    assert r["decision"] == UNDECIDABLE                 # n=1 is not a paired result


def test_orientation_decides_which_arm_is_better():
    """sufficiency is lower-is-better: the same numbers must clear one way and refuse the other."""
    lower = evaluate_gate({"suff": _arms(_STRONG, higher_is_better=False)})
    assert lower["decision"] == FREEZE                  # fit's value is BELOW control -> an advantage
    flipped = evaluate_gate({"suff": {**_arms(_STRONG, higher_is_better=False), "higher_is_better": True}})
    assert flipped["decision"] == REFUSE                # same numbers read the wrong way -> a loss
    assert flipped["contrasts"]["suff"]["mean"] < 0


def test_family_wise_correction_is_binding_and_both_methods_reported():
    """A raw hit that dies under Bonferroni must not freeze, and BOTH corrections are on the record so
    the method cannot be chosen after seeing which one rescues the claim."""
    fam = {k: _arms(_MODEST) for k in ("a", "b", "c", "d")}
    r = evaluate_gate(fam)
    c = r["contrasts"]["a"]
    assert c["family_size"] == 4
    assert c["p_value"] < 0.05 < c["p_bonferroni"]      # raw hit, dead after correction
    assert c["p_holm"] is not None
    assert c["survives_family_wise"] is False
    assert c["decision"] == REFUSE and r["decision"] == REFUSE


def test_one_refusal_blocks_a_family_of_otherwise_clean_contrasts():
    r = evaluate_gate({"good": _arms(_STRONG), "bad": _arms(_STRADDLE)})
    assert r["contrasts"]["good"]["decision"] == FREEZE
    assert r["decision"] == REFUSE                      # every REQUIRED contrast must clear


def test_decided_against_outranks_undecidable_in_the_overall_call():
    r = evaluate_gate({"degenerate": _arms([0.1] * 6), "bad": _arms(_STRADDLE)})
    assert r["decision"] == REFUSE                      # a computed negative is a decision, not a gap


def test_unit_missing_from_both_arms_shrinks_n_loudly():
    """Without the requested unit universe a unit absent from BOTH arms vanishes silently and a
    4-unit result reads as the intended 6-unit design."""
    arms = _arms(_STRONG)
    for d in (arms["fit"], arms["control"]):
        d.pop("u0")
    r = evaluate_gate({"suff": {**arms, "units": [f"u{i}" for i in range(6)]}})
    c = r["contrasts"]["suff"]
    assert c["n"] == 5
    assert [d["seed"] for d in c["dropped"]] == ["u0"]


def test_duplicate_unit_ids_are_refused_at_the_gate():
    """Duplicates collapse into one dict entry: n shrinks and the verdict still reads clean."""
    arms = _arms(_STRONG)
    with pytest.raises(ValueError, match="uniquely identified"):
        evaluate_gate({"suff": {**arms, "units": ["u0"] * 6}})


def test_non_finite_unit_is_dropped_not_nan_poisoned():
    arms = _arms(_STRONG)
    arms["fit"]["u0"] = float("nan")
    c = evaluate_gate({"suff": arms})["contrasts"]["suff"]
    assert c["n"] == 5 and c["mean"] is not None and c["decision"] == FREEZE


def test_every_unit_non_finite_is_undecidable():
    arms = _arms(_STRONG)
    arms["fit"] = {u: float("nan") for u in arms["fit"]}
    r = evaluate_gate({"suff": arms})
    assert r["contrasts"]["suff"]["n"] == 0
    assert r["decision"] == UNDECIDABLE                 # no data is not a negative and never a pass


def test_render_puts_the_control_beside_every_headline():
    r = evaluate_gate({"suff": _arms(_STRONG, base=2.0)})
    text = render(r)
    c = r["contrasts"]["suff"]
    assert f"{c['fit_mean']:+.4f}" in text and f"{c['control_mean']:+.4f}" in text
    assert "bonferroni" in text.lower() and "holm" in text.lower()
    assert r["decision"] in text


def test_render_survives_the_degenerate_report_it_exists_for():
    """A `None:+.4f` in the verdict print crashes on exactly the input the guard was written for."""
    r = evaluate_gate({"degenerate": _arms([0.1] * 6)}, required=("degenerate", "never_computed"))
    text = render(r)
    assert "n/a" in text and UNDECIDABLE in text


def test_arm_means_reconcile_under_BOTH_orientations():
    """fit_mean and control_mean are the RAW arms, so the advantage equals fit-minus-control only when
    higher is better; for a lower-is-better contrast it is control-minus-fit. Four of the five real
    Stage-B contrasts are lower-is-better, so an auditor recomputing the advantage from the two arm
    columns needs the orientation, and the report must carry it."""
    for hib in (True, False):
        c = evaluate_gate({"x": _arms(_STRONG, base=3.0, higher_is_better=hib)})["contrasts"]["x"]
        assert c["higher_is_better"] is hib                 # orientation is ON THE RECORD, not implied
        signed = (c["fit_mean"] - c["control_mean"]) if hib else (c["control_mean"] - c["fit_mean"])
        assert signed == pytest.approx(c["mean"], abs=1e-12)
        assert c["mean"] > 0                                # positive always means "the fit won"


def test_render_states_which_direction_is_better_for_each_contrast():
    """Without it the table shows fit=+19.3314, control=+19.4723, advantage=+0.1410 and the reader
    cannot tell whether the advantage column is fit-minus-control or the reverse."""
    r = evaluate_gate({"lower": _arms(_STRONG, higher_is_better=False),
                       "higher": _arms(_STRONG, higher_is_better=True)})
    text = render(r)
    assert "lower is better" in text and "higher is better" in text


def test_arm_means_reconcile_with_the_paired_delta_after_a_unit_is_dropped():
    """The reconciliation only bites when the arms and the delta could disagree: with a dropped unit,
    an arm mean taken over ALL units no longer matches the paired advantage over the USED ones."""
    arms = _arms(_STRONG, base=3.0)
    arms["fit"]["u0"] = float("nan")                    # dropped from the paired comparison
    c = evaluate_gate({"suff": arms})["contrasts"]["suff"]
    assert c["n"] == 5 and len(c["dropped"]) == 1
    assert c["fit_mean"] - c["control_mean"] == pytest.approx(c["mean"], abs=1e-12)


def test_report_is_json_serialisable():
    import json
    json.dumps(evaluate_gate({"suff": _arms(_STRONG), "nec": _arms(_STRADDLE)}))


def test_exit_codes_are_distinct_and_only_freeze_is_zero():
    assert exit_code({"decision": FREEZE}) == 0
    assert exit_code({"decision": REFUSE}) == 1
    assert exit_code({"decision": UNDECIDABLE}) == 2
    assert exit_code({}) != 0                           # a report with no decision is not a pass
