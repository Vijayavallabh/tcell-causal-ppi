"""Module 8 (Reproducibility Verification, feat-013) tests — fully synthetic. Covers the 11-detector fallacy
scan (complete-coverage clean pass + crafted Simpson / look-elsewhere flags) and the four verify verdicts:
REPRODUCIBLE (all match), NOT_REPRODUCIBLE (critical hash / decision mismatch), CANNOT_VERIFY (missing
checkout / decision / fallacy inputs), PARTIALLY_REPRODUCIBLE (a non-critical schema mismatch).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tcell_pipeline.evaluation.output_schema import predictions_to_frame
from tcell_pipeline.reproducibility import FALLACIES, VERDICTS, run_fallacy_scan, verify_reproducibility


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
        hashes[name] = {"path": rel, "sha256": h}
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
        "observed": {"decision": {"h1_confirmed": True, "lcb_95": 0.068}},
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
    manifest["observed"] = {"decision": {"h1_confirmed": True, "lcb_95": 0.0701}}
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
    manifest["observed"] = {"decision": {"h1_confirmed": True, "lcb_95": 0.07 + 6e-16}}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "REPRODUCIBLE"    # float noise tolerated
    manifest["observed"] = {"decision": {"h1_confirmed": True, "lcb_95": 0.09}}
    assert _verify(ck, manifest, tmp_path)["verdict"] == "NOT_REPRODUCIBLE"  # a real drift is not
