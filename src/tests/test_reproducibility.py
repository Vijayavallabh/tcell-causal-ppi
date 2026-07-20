"""Module 8 (Reproducibility Verification, feat-013) tests — fully synthetic. Covers the 11-detector fallacy
scan (complete-coverage clean pass + crafted Simpson / look-elsewhere flags) and the four verify verdicts:
REPRODUCIBLE (all match), NOT_REPRODUCIBLE (critical hash / decision mismatch), CANNOT_VERIFY (missing
checkout / decision / fallacy inputs), PARTIALLY_REPRODUCIBLE (a non-critical schema mismatch).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

from tcell_pipeline import config
from tcell_pipeline.evaluation.output_schema import predictions_to_frame
from tcell_pipeline.reproducibility import FALLACIES, VERDICTS, run_fallacy_scan, verify_reproducibility
from tcell_pipeline.reproducibility import manifest as mf
from tcell_pipeline.reproducibility import run_repro_real
from tcell_pipeline.reproducibility.verify import INDEPENDENT, INDEPENDENT_RERUN, SELF_DERIVED


def _clean_fallacy_inputs() -> dict:
    """A benign kwargs set for all eleven detectors — none should flag, all must be evaluable."""
    return {
        "simpson": {"groups": [([1, 2, 3], [1, 2, 3]), ([10, 11, 12], [20, 21, 22])]},
        "ecological": {"x": [0, 1, 2, 3, 4, 5], "y": [0, 1, 2, 3, 4, 5], "group": [0, 0, 1, 1, 2, 2]},
        "berkson": {"x": [0, 1, 2, 3, 4], "y": [4, 3, 2, 1, 0], "selected": [True] * 5},
        # z is a real (non-constant, non-collinear) covariate that is NOT a collider -> partial ~= marginal.
        # A constant z would make every correlation with it undefined, not "no association".
        "collider": {"x": [1, 2, 3, 4, 5, 6], "y": [2, 4, 6, 8, 10, 12], "z": [1, 0, 1, 0, 1, 0]},
        "base_rate": {"y_true": [1, 1, 0, 0], "y_pred": [1, 1, 0, 0]},
        "regression_to_mean": {"baseline": [0, 1, 2, 3, 4], "followup": [0, 1, 2, 3, 4]},
        "survivorship": {"values": [1, 2, 3, 4], "survived": [True, True, True, True]},
        "look_elsewhere": {"pvalues": [0.001, 0.5, 0.6], "alpha": 0.05},
        "garden_of_forks": {"estimates": [0.5, 0.52, 0.48]},
        "correlation_not_causation": {"corr": 0.8, "has_interventional_support": True},
        "reverse_causation": {"forward_corr": 0.8, "reverse_corr": 0.1},
    }


def test_fallacy_scan_complete_and_clean():
    scan = run_fallacy_scan(_clean_fallacy_inputs())
    assert scan["n_evaluated"] == 11 and scan["complete"] is True
    assert scan["flagged"] == []
    assert set(scan["results"]) == set(FALLACIES)


def test_fallacy_scan_flags_simpson_and_look_elsewhere():
    inputs = _clean_fallacy_inputs()
    inputs["simpson"] = {"groups": [([1, 2, 3], [3, 2, 1]), ([4, 5, 6], [6, 5, 4])]}   # within-neg, pooled-pos
    inputs["look_elsewhere"] = {"pvalues": [0.04, 0.5, 0.6, 0.7, 0.8], "alpha": 0.05}  # raw hit, no Bonferroni
    scan = run_fallacy_scan(inputs)
    assert "simpson" in scan["flagged"] and "look_elsewhere" in scan["flagged"]


def test_fallacy_scan_incomplete_coverage():
    scan = run_fallacy_scan({"simpson": {"groups": [([1, 2, 3], [1, 2, 3]), ([4, 5, 6], [4, 5, 6])]}})
    assert scan["n_evaluated"] == 1 and scan["complete"] is False


def _flagging_inputs() -> dict:
    """A kwargs set that makes EVERY detector fire — locks each detector's positive path."""
    return {
        "simpson": {"groups": [([1, 2, 3], [3, 2, 1]), ([4, 5, 6], [6, 5, 4])]},         # within-neg, pooled-pos
        "ecological": {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8], "y": [20, 10, 0, 24, 14, 4, 28, 18, 8],
                       "group": [0, 0, 0, 1, 1, 1, 2, 2, 2]},                              # indiv~0, aggregate=1
        # full corr 0; selecting the anti-diagonal (3 rows) induces corr -1 -> Berkson
        "berkson": {"x": [0, 0, 3, 3, 1.5, 1.5], "y": [0, 3, 0, 3, 1.5, 1.5],
                    "selected": [False, True, True, False, True, False]},
        "collider": {"x": [0, 0, 3, 3], "y": [0, 3, 0, 3], "z": [0, 3, 3, 6]},             # partial -1 vs marg 0
        "base_rate": {"y_true": [0] * 9 + [1], "y_pred": [0] * 10},                        # acc .9, precision 0
        # genuine regression: the baseline extreme (100) is NOT extreme on retest. (An affine followup like
        # [0,1,2,3,50] must NOT flag — it is a pure rescale — so it cannot serve as the positive case.)
        "regression_to_mean": {"baseline": [0, 1, 2, 3, 100], "followup": [2, 3, 1, 0, 2]},
        "survivorship": {"values": [0, 0, 0, 0, 10, 10, 10, 10],
                         "survived": [False, False, False, False, True, True, True, True]},
        "look_elsewhere": {"pvalues": [0.04, 0.5, 0.6, 0.7, 0.8], "alpha": 0.05},          # raw hit, no Bonferroni
        "garden_of_forks": {"estimates": [0.5, -0.3, 0.1]},                                # sign flip
        "correlation_not_causation": {"corr": 0.8, "has_interventional_support": False},
        "reverse_causation": {"forward_corr": 0.3, "reverse_corr": 0.5},                   # reverse >= forward
    }


def test_every_detector_has_a_working_flag_path():
    scan = run_fallacy_scan(_flagging_inputs())
    assert scan["n_evaluated"] == 11 and set(scan["flagged"]) == set(FALLACIES)  # all eleven fire


def test_ecological_needs_three_groups():
    # 2 groups -> the aggregate corr is degenerately +-1: unevaluable, so it must neither flag (false
    # positive) nor silently pass (certifying a check that never ran) -> errored, coverage drops
    two = run_fallacy_scan({**_flagging_inputs(),
                            "ecological": {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "group": [0, 0, 1, 1]}})
    assert "ecological" not in two["flagged"] and "ecological" in two["errored"]
    assert two["complete"] is False


# --- xhigh review: detectors must not fire on clean/degenerate data, nor pass on unevaluable input ---
def test_regression_to_mean_is_invariant_to_shift_and_scale():
    # ANY affine change of the followup correlates 1.0 with baseline and regresses not at all. Measuring
    # against a pooled grand mean misreads a SHIFT as regression; measuring in raw units against each
    # series' own mean still misreads a RESCALE. Standardised deviations are blind to both.
    rng = np.random.default_rng(0)
    b = rng.normal(100, 10, 500)
    from tcell_pipeline.reproducibility.fallacy_scan import regression_to_mean as r2m
    for name, f in [("identity", b), ("shift-", b - 10.0), ("shift+", b + 1000.0),
                    ("scale-up", b * 2.0), ("scale-down", b * 0.5), ("unit change", b / 10.0),
                    ("affine", b * 0.5 + 50.0)]:
        assert r2m(b, f)["flagged"] is False, f"{name}: affine followup must not read as regression"
    assert r2m(b, b * 2.0)["flagged"] is False and r2m(b * 2.0, b)["flagged"] is False   # symmetric
    # a genuine revert-to-the-mean (followup independent of baseline) still fires
    assert r2m(b, rng.normal(100, 10, 500))["flagged"] is True
    # ...and a partial one (followup half-driven by baseline, half noise)
    assert r2m(b, 0.5 * b + rng.normal(50, 10, 500))["flagged"] is True


def test_berkson_needs_a_defined_within_selection_correlation():
    # a bare row-count guard is not enough: _corr's degenerate cases must not be readable as "the
    # association vanished". A CONSTANT x among the selected rows is undefined, not zero.
    from tcell_pipeline.reproducibility.fallacy_scan import Unevaluable, berkson
    for name, spec in [
        ("1 selected row", {"x": list(range(10)), "y": list(range(10)), "selected": [False] * 9 + [True]}),
        ("constant x in selection", {"x": [1.0, 1.0, 1.0, 1.0, 2, 3, 4, 5, 6, 7],
                                     "y": [1.1, 1.9, 0.8, 1.3, 2, 3.1, 3.9, 5.2, 5.8, 7.1],
                                     "selected": [True] * 4 + [False] * 6}),
        ("constant y in selection", {"x": [1.1, 1.9, 0.8, 1.3, 2, 3.1, 3.9, 5.2, 5.8, 7.1],
                                     "y": [1.0, 1.0, 1.0, 1.0, 2, 3, 4, 5, 6, 7],
                                     "selected": [True] * 4 + [False] * 6}),
    ]:
        with pytest.raises(Unevaluable):
            berkson(**spec)                                    # never a false flag with no collider present
        scan = run_fallacy_scan({**_clean_fallacy_inputs(), "berkson": spec})
        assert "berkson" in scan["errored"] and "berkson" not in scan["flagged"], name


def test_corr_raises_rather_than_returning_a_zero_sentinel():
    # the root cause behind several false flags: a 0.0 return was indistinguishable from a real zero
    from tcell_pipeline.reproducibility.fallacy_scan import Unevaluable, _corr
    with pytest.raises(Unevaluable):
        _corr([1.0], [2.0])                       # <2 pairs
    with pytest.raises(Unevaluable):
        _corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])   # constant series
    assert _corr([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_every_detector_rejects_undefined_input_rather_than_passing_clean():
    # the Unevaluable principle applied to the whole family, not just the three detectors first patched
    from tcell_pipeline.reproducibility.fallacy_scan import FALLACIES
    degenerate = {
        "simpson": {"groups": [([1, 2, 3], [1, 2, 3])]},                       # 1 subgroup: no pooled-vs-within
        "ecological": {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "group": [0, 0, 1, 1]},   # 2 groups
        "berkson": {"x": [1, 2, 3], "y": [1, 2, 3], "selected": [True, False, False]},  # 1 selected
        "collider": {"x": [1.0], "y": [2.0], "z": [3.0]},                      # 1 row
        "base_rate": {"y_true": [0, 0, 0, 0], "y_pred": [0, 0, 0, 0]},         # single-class: no base rate
        "regression_to_mean": {"baseline": [1, 1, 1, 1], "followup": [1, 1, 1, 1]},    # constant
        "survivorship": {"values": [1, 2, 3, 4], "survived": [False] * 4},     # zero survivors
        "look_elsewhere": {"pvalues": []},                                     # no tests
        "garden_of_forks": {"estimates": [0.5]},                               # 1 fork: no spread
        "correlation_not_causation": {"corr": float("nan"), "has_interventional_support": False},
        "reverse_causation": {"forward_corr": float("nan"), "reverse_corr": 0.9},
    }
    scan = run_fallacy_scan(degenerate)
    assert set(scan["errored"]) == set(FALLACIES)   # every one refuses to certify on undefined input
    assert scan["flagged"] == [] and scan["complete"] is False
    assert scan["crashed"] == []                    # all Unevaluable (inadequate input), no detector BUGS


def test_scan_separates_inadequate_input_from_a_detector_bug():
    # a broad `except Exception` that conflated the two would hide a real defect behind "degenerate input"
    scan = run_fallacy_scan({**_clean_fallacy_inputs(),
                             "collider": {"x": [0, 1, 2, 3], "y": [3, 2, 1, 0], "z": [1, 2]}})  # length mismatch
    assert "collider" in scan["errored"] and scan["complete"] is False
    assert scan["crashed"] == []                    # a length mismatch is inadequate input, not a bug
    assert scan["results"]["collider"]["unevaluable"] is True


def test_reverse_causation_floor_is_on_the_stronger_direction():
    from tcell_pipeline.reproducibility.fallacy_scan import reverse_causation
    assert reverse_causation(0.0, 0.0)["flagged"] is False      # null endpoint -> no claim to invalidate
    assert reverse_causation(0.02, 0.03)["flagged"] is False    # both ~null
    assert reverse_causation(0.3, 0.5)["flagged"] is True       # real forward claim, stronger reverse -> flag
    # the archetypal trap: a WEAK claimed forward effect dominated by the reverse association. A floor on the
    # FORWARD correlation would silently unflag exactly the case the detector most needs to catch.
    assert reverse_causation(0.05, 0.95)["flagged"] is True
    assert reverse_causation(0.09, 0.99)["flagged"] is True
    assert reverse_causation(0.7, 0.5)["flagged"] is False      # forward dominates -> direction identified


def test_survivorship_with_zero_survivors_is_unevaluable():
    inputs = {**_clean_fallacy_inputs()}
    inputs["survivorship"] = {"values": [1, 2, 3, 4], "survived": [False] * 4}
    scan = run_fallacy_scan(inputs)
    assert "survivorship" in scan["errored"] and scan["complete"] is False  # never a silent clean pass


def test_nan_input_does_not_crash_detectors():
    from tcell_pipeline.reproducibility.fallacy_scan import ecological, simpson
    nan = float("nan")
    assert simpson(groups=[([1.0, 2.0, nan], [1.0, 2.0, 3.0]), ([10.0, 11.0, 12.0], [20.0, 21.0, 22.0])])
    assert ecological(x=[0, 1, 2, nan, 4, 5], y=[0, 1, 2, 3, 4, 5], group=[0, 0, 1, 1, 2, 2])


def test_errored_detector_is_not_counted_as_clean_coverage():
    inputs = _clean_fallacy_inputs()
    inputs["collider"] = {"x": [0, 1, 2, 3], "y": [3, 2, 1, 0], "z": [1, 2]}   # length-mismatch -> raises
    scan = run_fallacy_scan(inputs)
    assert "collider" in scan["errored"] and scan["complete"] is False         # NOT a silent clean 11/11
    assert scan["n_evaluated"] == 10 and scan["flagged"] == []


def _build_checkout(tmp_path, *, rows=4):
    ck = tmp_path / "checkout"
    ck.mkdir()
    files = {"id_mapping": "id_mapping.txt", "splits": "splits.txt", "de_layers": "de_layers.txt"}
    hashes = {}
    for name, rel in files.items():
        (ck / rel).write_text(f"deterministic-{name}")
        h = hashlib.sha256(f"deterministic-{name}".encode()).hexdigest()
        # INDEPENDENT: these fixtures stand in for an expected hash published by the ORIGINAL run, which is
        # the only kind of match that is a reproduction test (see test_self_derived_hash_* below).
        hashes[name] = {"path": rel, "sha256": h, "provenance": INDEPENDENT}
    frame = predictions_to_frame(np.arange(rows), np.zeros((rows, 3)), np.zeros((rows, 6)))
    frame.to_parquet(ck / "pred.parquet", index=False)
    return ck, hashes


_SNAPSHOT = {"PROGRAM_DIM": 128, "DELTA_PRED": 0.05}


def _config_hash(snapshot: dict) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()


def _manifest(hashes, *, rows=4):
    return {
        "hashes": hashes,
        "predictions": {"egipg_challenge": {"path": "pred.parquet", "n_rows": rows,
                                            "columns_prefixes": ["row_index", "delta_z_", "delta_x_", "sigma_"]}},
        "config_hashes": {"config_snapshot": _config_hash(_SNAPSHOT)},
        "decision": {"h1_confirmed": True, "lcb_95": 0.07, "tolerance": 0.01},
        # the fixture models a genuine RE-RUN, so it attests provenance; without that attestation the
        # decision comparison is `incomplete` (see test_a_decision_compared_against_itself_never_certifies)
        "observed": {"provenance": INDEPENDENT_RERUN,
                     "decision": {"h1_confirmed": True, "lcb_95": 0.068}},
        "fallacy_inputs": _clean_fallacy_inputs(),
    }


def _verify(ck, manifest, tmp_path, **kw):
    kw.setdefault("config_snapshot", _SNAPSHOT)
    return verify_reproducibility(ck, manifest, out_path=tmp_path / "rep.json", **kw)


def test_verify_reproducible(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    report = _verify(ck, _manifest(hashes), tmp_path)
    assert report["verdict"] == "REPRODUCIBLE"
    assert json.loads((tmp_path / "rep.json").read_text())["verdict"] == "REPRODUCIBLE"


def test_verify_not_reproducible_on_hash_mismatch(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    hashes["splits"]["sha256"] = "0" * 64                    # a deterministic (critical) artifact changed
    assert _verify(ck, _manifest(hashes), tmp_path)["verdict"] == "NOT_REPRODUCIBLE"


def test_verify_not_reproducible_on_decision_flip(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["observed"]["decision"]["h1_confirmed"] = False   # the confirmatory call did not reproduce
    assert _verify(ck, manifest, tmp_path)["verdict"] == "NOT_REPRODUCIBLE"


def test_verify_cannot_verify_missing_decision(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest.pop("observed")                                  # no rerun decision to compare
    assert _verify(ck, manifest, tmp_path)["verdict"] == "CANNOT_VERIFY"


def test_verify_cannot_verify_missing_checkout(tmp_path):
    report = verify_reproducibility(tmp_path / "absent", {"hashes": {}}, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "CANNOT_VERIFY" and VERDICTS[3] == "CANNOT_VERIFY"


def test_verify_partial_on_schema_mismatch(tmp_path):
    ck, hashes = _build_checkout(tmp_path, rows=4)
    manifest = _manifest(hashes)
    manifest["predictions"]["egipg_challenge"]["n_rows"] = 999   # non-critical schema mismatch only
    assert _verify(ck, manifest, tmp_path)["verdict"] == "PARTIALLY_REPRODUCIBLE"


def test_verify_partial_when_a_fallacy_detector_errors(tmp_path):
    # a crashed detector -> incomplete 11/11 coverage -> PARTIALLY, never a silent REPRODUCIBLE
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["fallacy_inputs"]["collider"] = {"x": [0, 1, 2, 3], "y": [3, 2, 1, 0], "z": [1, 2]}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "PARTIALLY_REPRODUCIBLE"


# --- xhigh review: the verifier must never certify a check it did not actually perform ------------
def test_verify_cannot_verify_absolute_manifest_path(tmp_path):
    # config.py's roots are absolute, so a manifest built from them would send every hash at the ORIGINAL
    # run's files — certifying a checkout whose contents were never read
    ck, hashes = _build_checkout(tmp_path)
    original = ck / "id_mapping.txt"
    hashes["id_mapping"]["path"] = str(original.resolve())    # absolute -> escapes the checkout contract
    report = _verify(ck, _manifest(hashes), tmp_path)
    assert report["verdict"] == "CANNOT_VERIFY"
    hit = [c for c in report["checks"] if c["check"] == "hash:id_mapping"][0]
    assert hit["status"] == "missing" and "absolute" in hit["reason"]


def test_verify_empty_checkout_is_never_reproducible(tmp_path):
    # end-to-end shape of the same defect: EVERY manifest path points at the original run, so if absolute
    # paths were honoured the verifier would hash that run against itself and certify an EMPTY checkout
    ck, hashes = _build_checkout(tmp_path)
    for name in ("id_mapping", "splits", "de_layers"):
        hashes[name]["path"] = str((ck / f"{name}.txt").resolve())
    manifest = _manifest(hashes)
    manifest["predictions"]["egipg_challenge"]["path"] = str((ck / "pred.parquet").resolve())
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _verify(empty, manifest, tmp_path)["verdict"] == "CANNOT_VERIFY"   # zero files were read


def test_verify_cannot_verify_missing_hashes_block(tmp_path):
    ck, _hashes = _build_checkout(tmp_path)
    manifest = _manifest({})                                  # manifest declares (or misspells) no hashes
    manifest.pop("predictions")
    report = _verify(ck, manifest, tmp_path)
    assert report["verdict"] == "CANNOT_VERIFY"                # not a green pass on zero hashed artifacts
    missing = {c["check"] for c in report["checks"] if c["status"] == "missing"}
    assert {"hash:id_mapping", "hash:splits", "hash:de_layers"} <= missing


def test_verify_cannot_verify_vacuous_decision(tmp_path):
    # both records omit h1_confirmed -> bool(None)==bool(None) must NOT read as "the call reproduced"
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["decision"] = {"foo": 1}
    manifest["observed"] = {"decision": {"bar": 2}}
    report = _verify(ck, manifest, tmp_path)
    assert report["verdict"] == "CANNOT_VERIFY"
    hit = [c for c in report["checks"] if c["check"] == "confirmatory_decision"][0]
    assert hit["status"] == "missing"


def test_verify_rejects_non_boolean_h1_confirmed(tmp_path):
    # presence is not comparison: bool(None)==bool(None) is True, and bool("false") is True. The frozen and
    # observed calls must be REAL booleans or the critical check has compared nothing.
    ck, hashes = _build_checkout(tmp_path)
    for frozen_v, observed_v, why in [(None, None, "bool(None) == bool(None)"),
                                      (True, "false", "bool('false') is True"),
                                      (True, 1, "1 is not a bool"),
                                      ("true", "true", "strings are not booleans")]:
        manifest = _manifest(hashes)
        manifest["decision"] = {"h1_confirmed": frozen_v, "lcb_95": 0.07}
        manifest["observed"] = {"decision": {"h1_confirmed": observed_v, "lcb_95": 0.07}}
        assert _verify(ck, manifest, tmp_path)["verdict"] == "CANNOT_VERIFY", why


def test_verify_caps_a_self_declared_tolerance(tmp_path):
    # the record under test must not be able to set its own bar: tolerance=1e9 would wave through any drift
    ck, hashes = _build_checkout(tmp_path)
    for tol in (1e9, float("inf"), 0.5):
        manifest = _manifest(hashes)
        manifest["decision"] = {"h1_confirmed": True, "lcb_95": 0.07, "tolerance": tol}
        manifest["observed"] = {"decision": {"h1_confirmed": True, "lcb_95": 0.0001}}   # a real drift
        assert _verify(ck, manifest, tmp_path)["verdict"] != "REPRODUCIBLE", f"tolerance {tol}"
    # a tolerance within the credible ceiling still applies
    manifest = _manifest(hashes)
    manifest["decision"] = {"h1_confirmed": True, "lcb_95": 0.07, "tolerance": 0.001}
    manifest["observed"] = {"provenance": INDEPENDENT_RERUN,
                            "decision": {"h1_confirmed": True, "lcb_95": 0.0701}}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "REPRODUCIBLE"


def test_verify_malformed_manifest_yields_a_verdict_not_a_traceback(tmp_path):
    # the module's contract is to RETURN one of VERDICTS; an unattended verification must always get a report
    ck, hashes = _build_checkout(tmp_path)
    for bad in [{"id_mapping": {"path": "id_mapping.txt"}},          # no sha256
                {"id_mapping": "not-a-dict"},
                {"id_mapping": {"path": ".", "sha256": "x"}},        # a directory, not a file
                {"id_mapping": {"path": None, "sha256": "x"}}]:
        manifest = _manifest({**hashes, **bad})
        report = _verify(ck, manifest, tmp_path)
        assert report["verdict"] in VERDICTS and report["verdict"] != "REPRODUCIBLE"


def test_verdict_is_whitelist_shaped(tmp_path):
    # a novel/unexpected status on a critical check must NOT certify (a blacklist let 'skip' through)
    from tcell_pipeline.reproducibility.verify import _verdict
    assert _verdict([{"category": "critical", "status": "pass"}]) == "REPRODUCIBLE"
    for novel in ("skip", "n/a", "unknown", "warn", ""):
        assert _verdict([{"category": "critical", "status": novel}]) == "CANNOT_VERIFY", novel


def test_verify_cannot_verify_without_config_snapshot(tmp_path):
    # a reproduction that changed DELTA_PRED (which alone flips the H1 call) must not certify
    ck, hashes = _build_checkout(tmp_path)
    report = verify_reproducibility(ck, _manifest(hashes), out_path=tmp_path / "rep.json")  # no snapshot
    assert report["verdict"] == "CANNOT_VERIFY"


def test_verify_not_reproducible_on_config_change(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    changed = {**_SNAPSHOT, "DELTA_PRED": 0.01}               # the knob that flips the H1 rule
    assert _verify(ck, _manifest(hashes), tmp_path, config_snapshot=changed)["verdict"] == "NOT_REPRODUCIBLE"


def test_verify_decision_tolerance_defaults_to_float_noise(tmp_path):
    # sealed_eval emits no `tolerance`; a 0.0 default would demand bit-exact floats across machines
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["decision"] = {"h1_confirmed": True, "lcb_95": 0.07}          # no tolerance key
    manifest["observed"] = {"provenance": INDEPENDENT_RERUN,
                            "decision": {"h1_confirmed": True, "lcb_95": 0.07 + 6e-16}}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "REPRODUCIBLE"    # float noise tolerated
    manifest["observed"] = {"decision": {"h1_confirmed": True, "lcb_95": 0.09}}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "NOT_REPRODUCIBLE"  # a real drift is not


# ---------------------------------------------------------------------------------------------------
# feat-013 part 2: a manifest built from THIS checkout's REAL frozen artifacts, and the 11 fallacy
# probes authored from REAL diagnostics. The governing distinction throughout: a hash compared against
# an INDEPENDENTLY frozen record is a reproduction test; a hash compared against itself is not, and must
# never read as a passed check.
# ---------------------------------------------------------------------------------------------------

def test_self_derived_hash_match_reads_as_incomplete_not_pass(tmp_path):
    """Hashing today's file and 'checking' it against itself proves only that the file is READABLE."""
    ck, hashes = _build_checkout(tmp_path)
    hashes["de_layers"]["provenance"] = SELF_DERIVED            # matches, but against its own hash
    report = _verify(ck, _manifest(hashes), tmp_path)
    check = next(c for c in report["checks"] if c["check"] == "hash:de_layers")
    assert check["status"] == "incomplete", check
    assert check["provenance"] == SELF_DERIVED and check["reason"]
    assert report["verdict"] == "PARTIALLY_REPRODUCIBLE"        # never REPRODUCIBLE on a self-hash


def test_unlabelled_hash_provenance_is_not_trusted(tmp_path):
    """An entry that does not SAY where its expected hash came from cannot be known to be independent."""
    ck, hashes = _build_checkout(tmp_path)
    del hashes["splits"]["provenance"]
    report = _verify(ck, _manifest(hashes), tmp_path)
    assert next(c for c in report["checks"] if c["check"] == "hash:splits")["status"] == "incomplete"
    assert report["verdict"] == "PARTIALLY_REPRODUCIBLE"


def test_independently_frozen_hash_match_passes(tmp_path):
    """The downgrade is not blanket: an independently frozen expected hash that matches still certifies."""
    ck, hashes = _build_checkout(tmp_path)
    report = _verify(ck, _manifest(hashes), tmp_path)
    assert all(c["status"] == "pass" for c in report["checks"] if c["check"].startswith("hash:"))
    assert report["verdict"] == "REPRODUCIBLE"


def test_independently_frozen_hash_mismatch_still_fails(tmp_path):
    """The fire path survives the provenance change — a real drift is still NOT_REPRODUCIBLE, not partial."""
    ck, hashes = _build_checkout(tmp_path)
    hashes["splits"]["sha256"] = "0" * 64
    report = _verify(ck, _manifest(hashes), tmp_path)
    assert next(c for c in report["checks"] if c["check"] == "hash:splits")["status"] == "fail"
    assert report["verdict"] == "NOT_REPRODUCIBLE"


# --- the manifest, built from the real checkout -----------------------------------------------------

def test_build_manifest_declares_every_deterministic_artifact_with_a_relative_path():
    m = mf.build_manifest()
    for name in ("id_mapping", "splits", "de_layers"):
        spec = m["hashes"][name]
        assert not os.path.isabs(spec["path"]), spec           # absolute -> would hash the ORIGINAL run
        assert (config.PROJECT_ROOT / spec["path"]).is_file()
        assert len(spec["sha256"]) == 64 and spec["provenance"] in (INDEPENDENT, SELF_DERIVED)


def test_build_manifest_marks_only_the_frozen_split_independent():
    """data/splits/manifest.json published sha256 at freeze time -> splits is a genuine reproduction test.
    Nothing else in this checkout has a frozen record, so nothing else may claim independence."""
    m = mf.build_manifest()
    prov = {k: v["provenance"] for k, v in m["hashes"].items()}
    assert prov["splits"] == INDEPENDENT
    assert prov["id_mapping"] == SELF_DERIVED and prov["de_layers"] == SELF_DERIVED
    assert m["hashes"]["splits"]["source"]                     # names WHERE the expected hash came from


def test_frozen_config_snapshot_matches_todays_config_and_can_still_drift(monkeypatch):
    """The config check is only a check if its input can VARY. Its expected value is read from the
    independently frozen data/splits/manifest.json, so editing config.py makes it disagree."""
    frozen = mf.frozen_config_snapshot()
    assert frozen and mf.live_config_snapshot() == frozen      # today's config still matches the freeze
    monkeypatch.setattr(config, "SPLIT_SEED", config.SPLIT_SEED + 1)
    assert mf.live_config_snapshot() != frozen                 # ...and a drift is visible


def test_config_check_fires_when_live_config_drifts_from_the_freeze(tmp_path, monkeypatch):
    """Trace it all the way to the verdict: a drifted config must FAIL, not silently skip."""
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["config_hashes"] = {"config_snapshot": mf.config_hash(mf.frozen_config_snapshot())}
    assert _verify(ck, manifest, tmp_path,
                   config_snapshot=mf.live_config_snapshot())["verdict"] == "REPRODUCIBLE"
    monkeypatch.setattr(config, "SEQ_SIM_COSINE_THRESHOLD", 0.9)
    report = _verify(ck, manifest, tmp_path, config_snapshot=mf.live_config_snapshot())
    assert next(c for c in report["checks"] if c["check"] == "config_hash")["status"] == "fail"
    assert report["verdict"] == "NOT_REPRODUCIBLE"


def test_manifest_records_the_steward_only_remainder():
    """The one step an agent session cannot take must be NAMED, with who must take it."""
    m = mf.build_manifest()
    assert "decision" not in m and "observed" not in m         # no sealed decision may be invented
    rem = m["unverified"]["confirmatory_decision"]
    assert "steward" in rem["who"].lower() and rem["reason"] and rem["how"]


# --- the eleven probes, authored from real diagnostics ----------------------------------------------

def test_every_fallacy_is_authored_or_explicitly_unevaluable():
    """A probe that is merely ABSENT loses its reason. Every one of the eleven must be accounted for."""
    inputs, unevaluable = mf.build_fallacy_inputs(mf.load_diagnostics())
    assert set(inputs) | set(unevaluable) == set(FALLACIES)
    assert not (set(inputs) & set(unevaluable))
    assert inputs, "no probe could be authored from this checkout's real diagnostics"


def test_unevaluable_reasons_name_what_is_missing():
    _, unevaluable = mf.build_fallacy_inputs(mf.load_diagnostics())
    for name, reason in unevaluable.items():
        assert len(reason) > 40 and name not in ("",), f"{name}: {reason!r}"


def test_authored_probes_run_on_real_diagnostics_without_a_detector_bug():
    """Real inputs must reach the detectors cleanly: no crash, and nothing authored may turn out to be
    Unevaluable at runtime (that would be an input that only LOOKED real)."""
    inputs, unevaluable = mf.build_fallacy_inputs(mf.load_diagnostics())
    scan = run_fallacy_scan(inputs)
    assert scan["crashed"] == [], scan["results"]
    assert scan["errored"] == [], scan["results"]
    assert scan["n_evaluated"] == len(inputs)
    assert scan["complete"] is False and scan["n_evaluated"] == len(FALLACIES) - len(unevaluable)


def test_look_elsewhere_uses_the_whole_preregistered_family():
    """Feeding ONE p-value would make the detector arithmetically incapable of flagging (alpha/1 == alpha)
    — a guard whose input is a constant. It must carry every simultaneously-tested contrast."""
    diag = mf.load_diagnostics()
    inputs, _ = mf.build_fallacy_inputs(diag)
    assert len(inputs["look_elsewhere"]["pvalues"]) == diag["family_size"] >= 2


def test_look_elsewhere_is_dropped_rather_than_understating_the_family():
    """If a contrast is untestable its p is missing; scoring the survivors against a SMALLER m would
    understate the correction. Better no probe than a lenient one."""
    diag = mf.load_diagnostics()
    diag["contrasts"] = dict(list(diag["contrasts"].items())[:-1])     # one contrast lost its p-value
    inputs, unevaluable = mf.build_fallacy_inputs(diag)
    assert "look_elsewhere" not in inputs and "family" in unevaluable["look_elsewhere"]


def test_regression_to_mean_retests_on_seeds_excluded_from_the_selection():
    """The retest must exclude the seed the winner was SELECTED on, or it is not a retest at all."""
    def diag(retest_systema):
        runs = [{"name": n, "seed": s, "systema": v, "pearson": 0.1, "epochs_run": 20.0, "n_epochs": 20.0}
                for n, vals in retest_systema.items() for s, v in enumerate(vals)]
        return {"runs": runs, "contrasts": {}, "family_size": 0, "h1_systema": 0.08,
                "selection_seed": 0, "sources": {}}

    # 'winner' is extreme ONLY on the selection seed -> its retest deviation must shrink -> flagged
    shrinks = diag({"winner": [0.20, 0.09, 0.09, 0.09, 0.09], "a": [0.08, 0.08, 0.08, 0.08, 0.08],
                    "b": [0.07, 0.07, 0.07, 0.07, 0.07], "c": [0.06, 0.06, 0.06, 0.06, 0.06]})
    probe = mf.build_fallacy_inputs(shrinks)[0]["regression_to_mean"]
    assert probe["baseline"][probe["baseline"].index(max(probe["baseline"]))] == 0.20
    assert 0.20 not in probe["followup"]                       # the selection seed is NOT in the retest
    assert run_fallacy_scan({"regression_to_mean": probe})["flagged"] == ["regression_to_mean"]

    # a winner that stays extreme on the retest is NOT regression to the mean
    holds = diag({"winner": [0.20, 0.20, 0.20, 0.20, 0.20], "a": [0.08, 0.08, 0.08, 0.08, 0.08],
                  "b": [0.07, 0.07, 0.07, 0.07, 0.07], "c": [0.06, 0.06, 0.06, 0.06, 0.06]})
    assert run_fallacy_scan({"regression_to_mean": mf.build_fallacy_inputs(holds)[0]
                             ["regression_to_mean"]})["flagged"] == []


def test_survivorship_survivors_are_the_runs_that_used_their_full_budget():
    diag = mf.load_diagnostics()
    probe = mf.build_fallacy_inputs(diag)[0]["survivorship"]
    expected = [r["epochs_run"] == r["n_epochs"] for r in mf.usable_runs(diag["runs"])]
    assert probe["survived"] == expected and any(expected) and not all(expected)


# --- the driver: verdict -> report -> JSON -> exit code ---------------------------------------------

def test_exit_code_is_zero_only_for_the_reproducible_verdict():
    assert run_repro_real.exit_code("REPRODUCIBLE") == 0
    for verdict in [v for v in VERDICTS if v != "REPRODUCIBLE"] + [None, "", "definitely fine"]:
        assert run_repro_real.exit_code(verdict) != 0, verdict


def test_real_run_reports_cannot_verify_on_the_reproduction_axis_and_exits_nonzero(tmp_path):
    """The whole point of the feature: no sealed decision exists, so the REPRODUCTION axis is
    CANNOT_VERIFY — and an unattended run must not exit 0 pretending success."""
    rc = run_repro_real.main(["--out-dir", str(tmp_path)])
    assert rc != 0
    report = json.loads((tmp_path / "repro_real_report.json").read_text())
    assert report["reproduction_verdict"] == "CANNOT_VERIFY"
    assert report["reproduction_cause"]["check"] == "confirmatory_decision"
    assert set(report["fallacy_unevaluable"]) | set(report["fallacy_scan"]["results"]) == set(FALLACIES)
    assert report["unverified"]["confirmatory_decision"]["who"]
    # self-derived hashes must be visible AS such in the written report, not just in the manifest
    assert {c["check"] for c in report["checks"] if c.get("provenance") == SELF_DERIVED}


# --- input classes the code above cannot reach by mutation: constructed by hand ----------------------

def test_garden_of_forks_needs_both_arm_choices_not_one():
    """One estimate has no spread to assess, so a lone fork must be REFUSED, not authored. Real data
    carries both, so no mutation of the builder can surface this — the input has to be constructed."""
    diag = mf.load_diagnostics()
    diag["contrasts"] = {k: v for k, v in diag["contrasts"].items() if k != "h1_vs_no_graph"}
    inputs, unevaluable = mf.build_fallacy_inputs(diag)
    assert "garden_of_forks" not in inputs
    assert "h1_vs_no_graph" in unevaluable["garden_of_forks"]


def test_survivorship_is_refused_when_every_run_survived():
    """With no non-survivors the survivor-only metric IS the full-population metric: the check could only
    ever confirm, which is decoration, not a guard."""
    diag = mf.load_diagnostics()
    for run in diag["runs"]:
        run["epochs_run"] = run["n_epochs"]
    inputs, unevaluable = mf.build_fallacy_inputs(diag)
    assert "survivorship" not in inputs and "could only ever confirm" in unevaluable["survivorship"]


def test_empty_checkout_verifies_nothing_and_claims_nothing(tmp_path):
    """Absence of evidence is never a pass. An empty checkout must declare no hashes, no predictions and no
    config hash — and every critical check must come back `missing`, not vacuously green."""
    m = mf.build_manifest(tmp_path)
    assert m["hashes"] == {} and m["predictions"] == {} and "config_hashes" not in m
    assert m["fallacy_inputs"] == {} and set(m["fallacy_unevaluable"]) == set(FALLACIES)
    report = verify_reproducibility(tmp_path, m, config_snapshot=mf.live_config_snapshot(),
                                    out_path=tmp_path / "r.json")
    assert {c["status"] for c in report["checks"] if c["category"] == "critical"} == {"missing"}
    assert report["verdict"] == "CANNOT_VERIFY"


def test_a_prediction_table_with_no_independent_row_count_is_not_declared(tmp_path):
    """verify reads `n_rows: None` as vacuously satisfied, so declaring the table without a count would
    emit `pass` for a row check that never ran. The count's authority is the FROZEN split; with no split
    to derive it from, the entry must not be declared at all — not declared with an unchecked count."""
    (tmp_path / "data/results/predictions/perturbed_mean/val").mkdir(parents=True)
    (tmp_path / "data/results/predictions/perturbed_mean/val/0.parquet").write_bytes(b"x")
    assert mf._predictions(tmp_path) == {}                      # no frozen split -> no claim
    (tmp_path / "data/results/comparators").mkdir(parents=True)
    (tmp_path / "data/results/comparators/tabular_baselines_vs_h1.json").write_text(
        json.dumps({"feature_coverage": {"val_rows": 4400}}))
    assert mf._predictions(tmp_path) == {}                      # another session's number is NOT authority


def test_manifest_hashes_the_checkout_it_is_pointed_at(tmp_path):
    """A clean-checkout verification must hash the CHECKOUT's files. config's roots are absolute, so a
    builder that passed them through would hash the ORIGINAL run and certify a checkout it never read —
    and it would do so INVISIBLY, because every hash would still match."""
    rels = {name: os.path.relpath(path, config.PROJECT_ROOT) for name, path in
            (("id_mapping", config.ID_MAPPING_PATH), ("splits", config.BLOCKED_SPLIT_PATH),
             ("de_layers", config.DE_LAYERS_DIR / "zscore.npz"))}
    assert not any(r.startswith("..") for r in rels.values()), rels    # else the roots were env-overridden
    for name, rel in rels.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(f"other-checkout-{name}".encode())
    (tmp_path / rels["splits"]).parent.joinpath("manifest.json").write_text(json.dumps(
        {"sha256": {os.path.basename(rels["splits"]): hashlib.sha256(b"other-checkout-splits").hexdigest()}}))

    hashes = mf.build_manifest(tmp_path)["hashes"]
    assert set(hashes) == set(rels)
    for name in rels:                                   # THIS checkout's bytes, not the running repo's
        assert hashes[name]["sha256"] == hashlib.sha256(f"other-checkout-{name}".encode()).hexdigest()
    assert hashes["splits"]["provenance"] == INDEPENDENT           # its own frozen record was honoured
    report = verify_reproducibility(tmp_path, mf.build_manifest(tmp_path),
                                    config_snapshot=mf.live_config_snapshot(), out_path=tmp_path / "r.json")
    assert next(c for c in report["checks"] if c["check"] == "hash:splits")["status"] == "pass"


def test_numpy_scalars_from_parquet_are_not_silently_dropped():
    """Diagnostics are READ FROM PARQUET, so they arrive as numpy scalars. np.float64 subclasses float but
    np.int64 does NOT subclass int — a numeric guard that missed that would empty the run table and report
    every run-level probe as "no data" instead of as a defect (the exact bug screening/multiseed.py records).
    And an np.bool_ mask serialises into the manifest as the STRING "True"."""
    runs = [{"name": f"cfg{i % 2}", "seed": np.int64(i // 2), "systema": np.float64(0.08 + 0.001 * i),
             "epochs_run": np.float64(20 - i), "n_epochs": np.float64(20)} for i in range(6)]
    assert len(mf.usable_runs(runs)) == 6
    # correlation_not_causation is never authored (see
    # test_correlation_not_causation_can_never_be_authored_in_this_study), so h1_systema only has to be a
    # numpy scalar that survives _num
    diag = {"runs": runs, "contrasts": {}, "family_size": 0, "h1_systema": np.float64(0.55),
            "selection_seed": 0, "blocked_target_fold": True, "sources": {}}
    inputs, _ = mf.build_fallacy_inputs(diag)
    assert {"simpson", "survivorship", "regression_to_mean"} <= set(inputs)
    assert run_fallacy_scan(inputs)["crashed"] == []
    # and the probe survives a manifest round-trip: JSON must hold real booleans/numbers, not their reprs
    reloaded = json.loads(json.dumps(inputs, default=str))
    assert reloaded == inputs and run_fallacy_scan(reloaded)["crashed"] == []


# ---------------------------------------------------------------------------------------------------
# Adversarial pass (2026-07-20): inputs the tests above never constructed. Every one of these was a real
# defect found by an agent whose only job was to break this module.
# ---------------------------------------------------------------------------------------------------

def test_zero_checks_is_never_reproducible():
    """S1. `any()` over an empty list is False three times over, so _verdict([]) fell through to the
    certifying return: absence of ALL evidence read as the cleanest possible pass."""
    from tcell_pipeline.reproducibility.verify import _verdict
    assert _verdict([]) == "CANNOT_VERIFY"
    assert _verdict([{"check": "x", "category": "schema", "status": "pass"}]) == "CANNOT_VERIFY"


def test_reproduction_axis_of_a_nonexistent_checkout_is_not_reproducible(tmp_path):
    """S1, end to end: verify short-circuits with `checks: []`, and the driver stamped the JSON that a
    checkout which does not exist had REPRODUCIBLE on the reproduction axis."""
    rc = run_repro_real.main(["--root", str(tmp_path / "absent"), "--out-dir", str(tmp_path)])
    report = json.loads((tmp_path / "repro_real_report.json").read_text())
    assert report["reproduction_verdict"] == "CANNOT_VERIFY" and rc != 0


@pytest.mark.parametrize("manifest", [None, "null"])
def test_a_null_manifest_returns_a_verdict_rather_than_crashing(tmp_path, manifest):
    """S6a. A truncated write landing on the valid JSON document `null` took the SUCCESS path, left
    `report` unbound and crashed — while genuinely unparseable garbage was handled."""
    ck, _ = _build_checkout(tmp_path)
    if manifest == "null":
        (tmp_path / "m.json").write_text("null")
        manifest = tmp_path / "m.json"
    assert _verify(ck, manifest, tmp_path)["verdict"] == "CANNOT_VERIFY"


@pytest.mark.parametrize("field,value", [
    ("config_hashes", [1]), ("observed", [1]), ("fallacy_inputs", [1]), ("fallacy_inputs", "x"),
    ("predictions", {"p": {"path": "pred.parquet", "columns_prefixes": [None]}}),
    ("predictions", {"p": {"path": "pred.parquet", "columns_prefixes": 5}}),
    ("predictions", {"p": {"path": "pred.parquet", "columns_prefixes": []}}),
])
def test_a_malformed_manifest_yields_a_verdict_not_a_traceback(tmp_path, field, value):
    """S6b/S6c/S7. `(x or {}).get(...)` guards a FALSY value, not a wrong-typed one; and an EMPTY
    columns_prefixes made `all(...)` vacuously true, passing a parquet with unrelated columns."""
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes) | {field: value}
    assert _verify(ck, manifest, tmp_path)["verdict"] in VERDICTS      # returns, never raises


def test_an_empty_parquet_does_not_pass_the_schema_check(tmp_path):
    """S7. columns_prefixes: [] over any frame is vacuously satisfied."""
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["predictions"] = {"p": {"path": "pred.parquet", "columns_prefixes": [], "n_rows": 4}}
    report = _verify(ck, manifest, tmp_path)
    assert next(c for c in report["checks"] if c["check"].startswith("schema"))["status"] != "pass"
    assert report["verdict"] != "REPRODUCIBLE"


def test_an_unchecked_row_count_does_not_read_as_a_passed_schema_check(tmp_path):
    """S7. `n_rows is None or ...` skips the row comparison, but the check still said `pass`."""
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    del manifest["predictions"]["egipg_challenge"]["n_rows"]
    report = _verify(ck, manifest, tmp_path)
    assert next(c for c in report["checks"] if c["check"].startswith("schema"))["status"] == "incomplete"
    assert report["verdict"] == "PARTIALLY_REPRODUCIBLE"


def test_a_decision_compared_against_itself_never_certifies(tmp_path):
    """S2. The hash check refuses a self-derived expected value; the CONFIRMATORY DECISION — the most
    load-bearing check in the module — had no such guard, so pointing `observed` at the very same record
    passed. The observed decision must declare that it came from an independent re-run."""
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    same = {"h1_confirmed": True, "lcb_95": 0.07, "tolerance": 0.01}
    manifest["decision"], manifest["observed"] = same, {"decision": same}   # literally the same dict
    report = _verify(ck, manifest, tmp_path)
    check = next(c for c in report["checks"] if c["check"] == "confirmatory_decision")
    assert check["status"] == "incomplete" and check["reason"]
    assert report["verdict"] == "PARTIALLY_REPRODUCIBLE"
    manifest["observed"]["provenance"] = INDEPENDENT_RERUN                 # attested re-run
    assert _verify(ck, manifest, tmp_path)["verdict"] == "REPRODUCIBLE"


def test_a_boolean_observed_endpoint_is_not_a_numeric_comparison(tmp_path):
    """S11. bool was excluded on the frozen side only, so observed lcb_95 == True compared as 1.0."""
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["decision"] = {"h1_confirmed": True, "lcb_95": 1.0, "tolerance": 0.01}
    manifest["observed"] = {"decision": {"h1_confirmed": True, "lcb_95": True}}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "CANNOT_VERIFY"


def test_a_probe_that_dies_inside_the_detector_keeps_its_reason():
    """S4. Probes were authored on a SHAPE precondition (enough arms, enough seeds) but the detectors
    enforce a VARIANCE one. When they disagreed the probe was authored, raised Unevaluable inside the
    scan, and vanished from BOTH lists — the operator saw '2/11 evaluated' with nothing named."""
    runs = [{"name": f"cfg{c}", "seed": s, "systema": 0.08 + 0.001 * c,   # constant epochs_run: no trend
             "epochs_run": 20.0, "n_epochs": 20.0} for c in range(3) for s in range(5)]
    inputs, unevaluable = mf.build_fallacy_inputs(
        {"runs": runs, "contrasts": {}, "family_size": 0, "h1_systema": 0.08,
         "selection_seed": 0, "blocked_target_fold": True, "sources": {}})
    assert set(inputs) | set(unevaluable) == set(FALLACIES)
    assert not (set(inputs) & set(unevaluable))
    for name in ("simpson", "ecological"):
        assert name in unevaluable and "constant" in unevaluable[name]
    assert run_fallacy_scan(inputs)["errored"] == []          # nothing authored is dead on arrival


def test_an_out_of_range_pvalue_does_not_survive_as_an_authored_probe():
    """S4. _num accepts any finite float, so p=1.7 was authored and then rejected by look_elsewhere."""
    diag = mf.load_diagnostics()
    for contrast in diag["contrasts"].values():
        contrast["p_value"] = 1.7
    inputs, unevaluable = mf.build_fallacy_inputs(diag)
    assert "look_elsewhere" not in inputs and "look_elsewhere" in unevaluable


@pytest.mark.parametrize("diag", [{"runs": [5]}, {"contrasts": [1]}, {"contrasts": {"a": 5}},
                                  {"runs": "x"}, {"contrasts": {"a": {"p_value": "0.3"}}}])
def test_build_fallacy_inputs_partitions_even_on_wrong_typed_diagnostics(diag):
    """S8. Documented as pure and callable with a constructed diagnostic set — so it must not raise."""
    inputs, unevaluable = mf.build_fallacy_inputs({"h1_systema": 0.08, "sources": {}} | diag)
    assert set(inputs) | set(unevaluable) == set(FALLACIES)
    assert not (set(inputs) & set(unevaluable))


def test_manifest_never_emits_a_path_that_escapes_the_checkout(tmp_path, monkeypatch):
    """S3a. relative_to() is purely LEXICAL and does not normalise `..`, so an env-overridden root of the
    form <project>/../<project>/data/splits emitted `../<project>/data/splits/...` into the manifest —
    a published record whose hash is of a file outside the declared checkout."""
    sneaky = config.PROJECT_ROOT / ".." / config.PROJECT_ROOT.name / "data" / "splits"
    monkeypatch.setattr(config, "BLOCKED_SPLIT_PATH", sneaky / "blocked_target_ood.csv")
    for spec in mf.build_manifest().get("hashes", {}).values():
        assert not spec["path"].startswith("..") and not os.path.isabs(spec["path"]), spec


def test_cnc_threshold_tracks_the_detector():
    """The refusal below is keyed on the detector's own threshold; pin them so they cannot drift apart."""
    import inspect
    from tcell_pipeline.reproducibility.fallacy_scan import correlation_not_causation
    assert inspect.signature(correlation_not_causation).parameters["threshold"].default == mf.CNC_THRESHOLD


def test_correlation_not_causation_can_never_be_authored_in_this_study():
    """It flags only on |corr| >= 0.3 AND absent interventional support. BOTH conditions are unreachable
    here, so there is no headline association that makes it a live check — an earlier version authored it
    above the threshold, which would have re-created the decoration it was refused for."""
    from tcell_pipeline.reproducibility.fallacy_scan import correlation_not_causation as cnc
    # support is structural: the endpoint IS a target-blocked CRISPR fold, and with support present the
    # detector cannot flag for ANY correlation
    assert not any(cnc(corr=c, has_interventional_support=True)["flagged"]
                   for c in (0.3, 0.55, 0.9, 1.0, -1.0))
    diag = mf.load_diagnostics()
    assert abs(diag["h1_systema"]) < mf.CNC_THRESHOLD
    for h1 in (diag["h1_systema"], 0.55, 0.99, -0.99):     # below AND far above the threshold
        _, unevaluable = mf.build_fallacy_inputs(dict(diag, h1_systema=h1))
        assert "correlation_not_causation" in unevaluable, h1
        assert "cannot fire in this study" in unevaluable["correlation_not_causation"]


def test_a_detector_bug_is_not_laundered_into_inadequate_input(monkeypatch):
    """Demotion must trigger on Unevaluable ONLY. Catching every exception would quietly reclassify a real
    defect in a detector as 'the input was degenerate' — and the scan separates those two deliberately."""
    from tcell_pipeline.reproducibility import fallacy_scan

    def exploding(**_):
        raise RuntimeError("detector defect")

    monkeypatch.setitem(fallacy_scan._DETECTORS, "survivorship", exploding)
    inputs, unevaluable = mf.build_fallacy_inputs(mf.load_diagnostics())
    assert "survivorship" in inputs and "survivorship" not in unevaluable   # stays visible
    scan = run_fallacy_scan(inputs)
    assert scan["crashed"] == ["survivorship"]                              # surfaced AS a bug


def test_an_artifact_outside_the_project_is_dropped_not_emitted_as_an_escape(tmp_path, monkeypatch):
    """An env-overridden root pointing outside the project cannot be attributed to any checkout. Emitting
    it as `../../..` would publish a hash of a file the declared checkout does not contain."""
    outside = tmp_path / "elsewhere" / "id_mapping.parquet"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"a real, readable file that simply is not in this project")   # must EXIST, or
    monkeypatch.setattr(config, "ID_MAPPING_PATH", outside)                            # absence hides it
    hashes = mf.build_manifest()["hashes"]
    assert "id_mapping" not in hashes                        # dropped...
    assert {"splits", "de_layers"} <= set(hashes)            # ...without taking the others with it
    report = verify_reproducibility(config.PROJECT_ROOT, {"hashes": hashes},
                                    out_path=tmp_path / "r.json")
    missing = next(c for c in report["checks"] if c["check"] == "hash:id_mapping")
    assert missing["status"] == "missing"                    # and verify says so explicitly


def test_build_hashes_survives_a_malformed_frozen_split_record(tmp_path):
    """`(x or {}).get(...)` guards a FALSY value, not a wrong-typed one — the same hole `_as_dict` was
    added to close in verify.py. A truncated data/splits/manifest.json crashed build_manifest, so
    run_repro_real.main() produced NO report at all."""
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/splits/blocked_target_ood.csv").write_bytes(b"split")
    for malformed in ("not-an-object", ["a"], 7):
        (tmp_path / "data/splits/manifest.json").write_text(json.dumps({"sha256": malformed}))
        hashes = mf.build_manifest(tmp_path)["hashes"]                # returns rather than raising
        assert hashes["splits"]["provenance"] == SELF_DERIVED         # no frozen record could be read
    report = verify_reproducibility(tmp_path, mf.build_manifest(tmp_path),
                                    config_snapshot=mf.live_config_snapshot(), out_path=tmp_path / "r.json")
    assert report["verdict"] in VERDICTS


def test_cause_names_the_critical_check_that_drove_the_verdict():
    """_cause claims to walk _verdict's OWN precedence. Its fallback scanned every check, so an
    unrecognised status on a CRITICAL check (the case the whitelist verdict exists to survive) got blamed
    on whichever non-critical check happened to come first in the list."""
    from tcell_pipeline.reproducibility.verify import _verdict
    checks = [{"check": "schema:p", "category": "schema", "status": "incomplete"},
              {"check": "config_hash", "category": "critical", "status": "a_status_nobody_defined"}]
    assert _verdict(checks) == "CANNOT_VERIFY"
    assert run_repro_real._cause(checks)["check"] == "config_hash"
    # and the ordinary precedence still holds: a critical fail outranks a critical missing
    ordered = [{"check": "a", "category": "critical", "status": "missing"},
               {"check": "b", "category": "critical", "status": "fail"}]
    assert run_repro_real._cause(ordered)["check"] == "b"


def test_main_ignores_a_parent_process_argv(tmp_path, monkeypatch):
    """main() is now called PROGRAMMATICALLY from run_module8_real.run_repro(), and argparse falls back to
    sys.argv when argv is None — so a bare main() inherited the parent driver's flags and died with
    'unrecognized arguments: --part repro'. Session A's test monkeypatched main, so it stayed green while
    the real command was broken; only running it caught this."""
    monkeypatch.setattr(sys, "argv", ["run_module8_real.py", "--part", "repro", "--device", "cuda:2"])
    monkeypatch.setattr(config, "REPRODUCIBILITY_ROOT", tmp_path)
    assert run_repro_real.main() != 0                 # must not SystemExit(2) on the parent's argv
    assert (tmp_path / "repro_real_report.json").is_file()
    assert run_repro_real.main([]) != 0               # the explicit-empty form A uses today still works


def _fake_checkout_with_split(tmp_path, *, val_rows_in_artifact, n_val=3):
    """A miniature checkout carrying a frozen split, a perturbation table, a prediction parquet, and
    session A's comparator artifact — so the row-count provenance can be exercised end to end."""
    import pandas as pd
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/intermediate").mkdir(parents=True)
    (tmp_path / "data/results/predictions/perturbed_mean/val").mkdir(parents=True)
    (tmp_path / "data/results/comparators").mkdir(parents=True)
    genes = [f"G{i}" for i in range(5)]
    roles = ["val"] * n_val + ["train"] * (5 - n_val)
    pd.DataFrame({"hgnc_symbol": genes, "role": roles}).to_csv(
        tmp_path / "data/splits/blocked_target_ood.csv", index=False)
    pd.DataFrame({"hgnc_symbol": genes}).to_parquet(
        tmp_path / "data/intermediate/perturbation_condition.parquet", index=False)
    frame = predictions_to_frame(np.arange(n_val), np.zeros((n_val, 3)), np.zeros((n_val, 6)))
    frame.to_parquet(tmp_path / "data/results/predictions/perturbed_mean/val/0.parquet", index=False)
    (tmp_path / "data/results/comparators/tabular_baselines_vs_h1.json").write_text(
        json.dumps({"feature_coverage": {"val_rows": val_rows_in_artifact}}))
    return tmp_path


def test_prediction_row_count_comes_from_the_frozen_split_not_another_sessions_artifact(tmp_path):
    """The count used to be read straight out of data/results/comparators/tabular_baselines_vs_h1.json —
    a file another live session rewrites. If its feature handling changed, this probe would have followed
    it SILENTLY: presence is not freshness. The count now comes from the frozen, git-tracked split."""
    ck = _fake_checkout_with_split(tmp_path, val_rows_in_artifact=3, n_val=3)
    spec = mf._predictions(ck)["perturbed_mean_val"]
    assert spec["n_rows"] == 3
    assert "frozen split" in spec["n_rows_source"]
    assert spec["cross_check"]["val_rows"] == 3 and spec["cross_check"]["agrees"] is True


def test_a_row_count_that_disagrees_with_the_frozen_split_is_refused(tmp_path):
    """Two independent sources disagreeing about the fold means we do not KNOW the row count. Refuse the
    entry so verify reports `missing` -> CANNOT_VERIFY, rather than picking a winner."""
    ck = _fake_checkout_with_split(tmp_path, val_rows_in_artifact=999, n_val=3)
    assert mf._predictions(ck) == {}
    report = verify_reproducibility(ck, {"predictions": mf._predictions(ck)}, out_path=tmp_path / "r.json")
    assert next(c for c in report["checks"] if c["check"] == "schema")["status"] == "missing"


def test_the_row_count_survives_the_comparator_artifact_being_absent(tmp_path):
    """The split alone is sufficient — this module must not need another session's results to function."""
    ck = _fake_checkout_with_split(tmp_path, val_rows_in_artifact=3, n_val=3)
    (ck / "data/results/comparators/tabular_baselines_vs_h1.json").unlink()
    spec = mf._predictions(ck)["perturbed_mean_val"]
    assert spec["n_rows"] == 3 and spec["cross_check"]["val_rows"] is None
