"""Module 8 (Reproducibility Verification, feat-013) tests — fully synthetic. Covers the 11-detector fallacy
scan (complete-coverage clean pass + crafted Simpson / look-elsewhere flags) and the four verify verdicts:
REPRODUCIBLE (all match), NOT_REPRODUCIBLE (critical hash / decision mismatch), CANNOT_VERIFY (missing
checkout / decision / fallacy inputs), PARTIALLY_REPRODUCIBLE (a non-critical schema mismatch).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from tcell_pipeline.evaluation.output_schema import predictions_to_frame
from tcell_pipeline.reproducibility import FALLACIES, VERDICTS, run_fallacy_scan, verify_reproducibility


def _clean_fallacy_inputs() -> dict:
    """A benign kwargs set for all eleven detectors — none should flag."""
    return {
        "simpson": {"groups": [([1, 2, 3], [1, 2, 3]), ([10, 11, 12], [20, 21, 22])]},
        "ecological": {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "group": [0, 0, 1, 1]},
        "berkson": {"x": [0, 1, 2, 3, 4], "y": [4, 3, 2, 1, 0], "selected": [True] * 5},
        "collider": {"x": [0, 1, 2, 3], "y": [3, 2, 1, 0], "z": [1, 1, 1, 1]},
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
    scan = run_fallacy_scan({"simpson": {"groups": [([1, 2], [1, 2])]}})
    assert scan["n_evaluated"] == 1 and scan["complete"] is False


def _flagging_inputs() -> dict:
    """A kwargs set that makes EVERY detector fire — locks each detector's positive path."""
    return {
        "simpson": {"groups": [([1, 2, 3], [3, 2, 1]), ([4, 5, 6], [6, 5, 4])]},         # within-neg, pooled-pos
        "ecological": {"x": [0, 1, 2, 3, 4, 5, 6, 7, 8], "y": [20, 10, 0, 24, 14, 4, 28, 18, 8],
                       "group": [0, 0, 0, 1, 1, 1, 2, 2, 2]},                              # indiv~0, aggregate=1
        "berkson": {"x": [0, 0, 3, 3], "y": [0, 3, 0, 3], "selected": [False, True, True, False]},  # -1 vs 0
        "collider": {"x": [0, 0, 3, 3], "y": [0, 3, 0, 3], "z": [0, 3, 3, 6]},             # partial -1 vs marg 0
        "base_rate": {"y_true": [0] * 9 + [1], "y_pred": [0] * 10},                        # acc .9, precision 0
        "regression_to_mean": {"baseline": [0, 1, 2, 3, 100], "followup": [0, 1, 2, 3, 50]},
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
    # 2 groups -> the aggregate corr is degenerately +-1, so it must NOT flag (would be a false positive)
    two = run_fallacy_scan({**_flagging_inputs(),
                            "ecological": {"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "group": [0, 0, 1, 1]}})
    assert "ecological" not in two["flagged"]


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


def _manifest(hashes, *, rows=4):
    return {
        "hashes": hashes,
        "predictions": {"egipg_challenge": {"path": "pred.parquet", "n_rows": rows,
                                            "columns_prefixes": ["row_index", "delta_z_", "delta_x_", "sigma_"]}},
        "decision": {"h1_confirmed": True, "lcb_95": 0.07, "tolerance": 0.01},
        "observed": {"decision": {"h1_confirmed": True, "lcb_95": 0.068}},
        "fallacy_inputs": _clean_fallacy_inputs(),
    }


def test_verify_reproducible(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    report = verify_reproducibility(ck, _manifest(hashes), out_path=tmp_path / "rep.json")
    assert report["verdict"] == "REPRODUCIBLE"
    assert json.loads((tmp_path / "rep.json").read_text())["verdict"] == "REPRODUCIBLE"


def test_verify_not_reproducible_on_hash_mismatch(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    hashes["splits"]["sha256"] = "0" * 64                    # a deterministic (critical) artifact changed
    report = verify_reproducibility(ck, _manifest(hashes), out_path=tmp_path / "rep.json")
    assert report["verdict"] == "NOT_REPRODUCIBLE"


def test_verify_not_reproducible_on_decision_flip(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["observed"]["decision"]["h1_confirmed"] = False   # the confirmatory call did not reproduce
    report = verify_reproducibility(ck, manifest, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "NOT_REPRODUCIBLE"


def test_verify_cannot_verify_missing_decision(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest.pop("observed")                                  # no rerun decision to compare
    report = verify_reproducibility(ck, manifest, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "CANNOT_VERIFY"


def test_verify_cannot_verify_missing_checkout(tmp_path):
    report = verify_reproducibility(tmp_path / "absent", {"hashes": {}}, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "CANNOT_VERIFY" and VERDICTS[3] == "CANNOT_VERIFY"


def test_verify_partial_on_schema_mismatch(tmp_path):
    ck, hashes = _build_checkout(tmp_path, rows=4)
    manifest = _manifest(hashes)
    manifest["predictions"]["egipg_challenge"]["n_rows"] = 999   # non-critical schema mismatch only
    report = verify_reproducibility(ck, manifest, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "PARTIALLY_REPRODUCIBLE"


def test_verify_partial_when_a_fallacy_detector_errors(tmp_path):
    # a crashed detector -> incomplete 11/11 coverage -> PARTIALLY, never a silent REPRODUCIBLE
    ck, hashes = _build_checkout(tmp_path)
    manifest = _manifest(hashes)
    manifest["fallacy_inputs"]["collider"] = {"x": [0, 1, 2, 3], "y": [3, 2, 1, 0], "z": [1, 2]}
    report = verify_reproducibility(ck, manifest, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "PARTIALLY_REPRODUCIBLE"


def test_verify_config_hash(tmp_path):
    ck, hashes = _build_checkout(tmp_path)
    snapshot = {"PROGRAM_DIM": 128, "DELTA_PRED": 0.05}
    manifest = _manifest(hashes)
    manifest["config_hashes"] = {
        "config_snapshot": hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()}
    report = verify_reproducibility(ck, manifest, config_snapshot=snapshot, out_path=tmp_path / "rep.json")
    assert report["verdict"] == "REPRODUCIBLE"                # config hash matches -> still reproducible
