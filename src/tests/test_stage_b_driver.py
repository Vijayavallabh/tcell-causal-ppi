"""Stage-B driver tests (feat-008): the gate's decision must reach the PROCESS EXIT CODE.

This repo has already shipped a campaign that printed "NOT comparable" and then returned 0, so an
unattended run and every exit-status CI gate recorded it green. These pin the mapping and the required
contrast set; the driver's data-path wiring is exercised by the real (approval-gated) Stage-B run.
"""
from __future__ import annotations

from tcell_pipeline.training import run_stage_b
from tcell_pipeline.training.freeze_gate import FREEZE, REFUSE, UNDECIDABLE, evaluate_gate


def _main_with(monkeypatch, report):
    monkeypatch.setattr(run_stage_b, "run", lambda *a, **k: report)
    return run_stage_b.main(["--ckpt", "unused"])


def test_exit_code_is_zero_only_on_a_freeze(monkeypatch):
    assert _main_with(monkeypatch, {"decision": FREEZE}) == 0
    assert _main_with(monkeypatch, {"decision": REFUSE}) == 1
    assert _main_with(monkeypatch, {"decision": UNDECIDABLE}) == 2


def test_a_run_that_produced_nothing_does_not_exit_zero(monkeypatch):
    assert _main_with(monkeypatch, None) != 0          # missing artifacts -> nothing was shown
    assert _main_with(monkeypatch, {}) != 0            # a report with no decision is not a pass


def _clean(names):
    """Contrasts with a consistent, easily-significant advantage (NLL: lower is better)."""
    d = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]
    return {n: {"fit": {i: 1.0 - x for i, x in enumerate(d)},
                "control": {i: 1.0 for i in range(len(d))}, "higher_is_better": False} for n in names}


def test_the_gate_requires_both_families_by_name():
    """The composition the RUN uses must not pass on the calibration contrasts alone. This calls
    run_stage_b.overall_gate — the function run() actually invokes — so dropping its `required=` (the
    vacuous-pass trap the module docstring warns about) turns this red. Asserting on a bare
    evaluate_gate() call instead would re-derive the guarantee and leave that line untested."""
    assert set(run_stage_b.REQUIRED_CONTRASTS) == set(run_stage_b.CALIBRATION_CONTRASTS) | \
        set(run_stage_b.RATIONALE_CONTRASTS)
    cal_only = _clean(run_stage_b.CALIBRATION_CONTRASTS)
    assert evaluate_gate(cal_only)["decision"] == FREEZE                 # they clear on their own...
    r = run_stage_b.overall_gate(cal_only)
    assert r["decision"] == UNDECIDABLE                                  # ...but the run still is not
    assert all(r["contrasts"][n]["decision"] == UNDECIDABLE for n in run_stage_b.RATIONALE_CONTRASTS)


def test_the_gate_clears_only_when_every_required_contrast_is_present():
    """The other side of the same guard: with all five present and clean, the run-level gate freezes —
    so the UNDECIDABLE above comes from the missing family, not from the gate refusing everything."""
    r = run_stage_b.overall_gate(_clean(run_stage_b.REQUIRED_CONTRASTS))
    assert r["decision"] == FREEZE
