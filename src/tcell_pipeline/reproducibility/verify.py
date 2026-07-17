"""Reproducibility verification (feat-013): given a clean ``checkout`` and a frozen ``manifest``, re-derive
the deterministic preprocessing hashes, re-check the prediction schema + row counts + config + checkpoint
provenance, confirm the same confirmatory decision within tolerance, run the 11/11 fallacy scan, and return a
single verdict — REPRODUCIBLE / PARTIALLY_REPRODUCIBLE / NOT_REPRODUCIBLE / CANNOT_VERIFY (report
§reproducibility).

The manifest is the frozen record the original run published; this module re-computes each item against the
checkout and compares. It performs NO training itself — the "rerun the final model + comparators over frozen
seeds" step produces the challenge predictions + sealed decision the manifest carries under ``observed`` (the
sealed evaluator writes them); verify checks that those reproduce the frozen ``decision``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tcell_pipeline import config
from tcell_pipeline.reproducibility.fallacy_scan import run_fallacy_scan

VERDICTS = ("REPRODUCIBLE", "PARTIALLY_REPRODUCIBLE", "NOT_REPRODUCIBLE", "CANNOT_VERIFY")
# deterministic preprocessing artifacts whose hashes MUST reproduce bit-for-bit (report: id_mapping, splits,
# de_layers) — a mismatch here means the frozen pipeline did not reproduce.
_DETERMINISTIC = ("id_mapping", "splits", "de_layers")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(checkout: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else Path(checkout) / rel


def _check_hashes(checkout: Path, entries: dict) -> list[dict]:
    checks = []
    for name, spec in entries.items():
        path = _resolve(checkout, spec["path"])
        actual = _sha256_file(path)
        category = "critical" if name in _DETERMINISTIC else "provenance"
        if actual is None:
            status = "missing"
        elif actual == spec["sha256"]:
            status = "pass"
        else:
            status = "fail"
        checks.append({"check": f"hash:{name}", "category": category, "status": status,
                       "expected": spec["sha256"], "actual": actual})
    return checks


def _check_predictions(checkout: Path, entries: dict) -> list[dict]:
    import pandas as pd
    checks = []
    for name, spec in entries.items():
        path = _resolve(checkout, spec["path"])
        if not path.exists():
            checks.append({"check": f"schema:{name}", "category": "schema", "status": "missing"})
            continue
        frame = pd.read_parquet(path)
        cols = list(frame.columns)
        prefixes = spec.get("columns_prefixes", ["row_index", "delta_z_", "delta_x_", "sigma_"])
        have = all(any(c == p or c.startswith(p) for c in cols) for p in prefixes)
        rows_ok = spec.get("n_rows") is None or len(frame) == spec["n_rows"]
        status = "pass" if (have and rows_ok) else "fail"
        checks.append({"check": f"schema:{name}", "category": "schema", "status": status,
                       "n_rows": len(frame), "expected_rows": spec.get("n_rows"), "columns_ok": have})
    return checks


def _check_config(manifest: dict, config_snapshot: dict | None) -> list[dict]:
    expected = (manifest.get("config_hashes") or {}).get("config_snapshot")
    if expected is None or config_snapshot is None:
        return [{"check": "config_hash", "category": "config", "status": "skip"}]
    actual = hashlib.sha256(json.dumps(config_snapshot, sort_keys=True, default=str).encode()).hexdigest()
    return [{"check": "config_hash", "category": "config", "status": "pass" if actual == expected else "fail",
             "expected": expected, "actual": actual}]


def _check_decision(manifest: dict) -> list[dict]:
    frozen = manifest.get("decision")
    observed = (manifest.get("observed") or {}).get("decision")
    if frozen is None or observed is None:
        return [{"check": "confirmatory_decision", "category": "critical", "status": "missing"}]
    tol = float(frozen.get("tolerance", 0.0))
    same_call = bool(frozen.get("h1_confirmed")) == bool(observed.get("h1_confirmed"))
    within = True
    for key in ("lcb_95", "rho_egipg", "delta_vs_best"):
        if key in frozen and key in observed:
            within = within and abs(float(frozen[key]) - float(observed[key])) <= tol
    status = "pass" if (same_call and within) else "fail"
    return [{"check": "confirmatory_decision", "category": "critical", "status": status,
             "frozen": frozen, "observed": observed, "same_call": same_call, "within_tolerance": within}]


def _check_fallacies(manifest: dict) -> tuple[list[dict], dict]:
    inputs = manifest.get("fallacy_inputs")
    if not inputs:
        return [{"check": "fallacy_scan", "category": "critical", "status": "missing"}], {}
    scan = run_fallacy_scan(inputs)
    if scan["flagged"]:
        status = "fail"          # a detected inference trap invalidates the claim
    elif not scan["complete"]:
        status = "incomplete"    # ran clean but not all 11 covered
    else:
        status = "pass"
    return [{"check": "fallacy_scan", "category": "critical", "status": status,
             "n_evaluated": scan["n_evaluated"], "flagged": scan["flagged"], "complete": scan["complete"],
             "errored": scan.get("errored", [])}], scan


def _verdict(checks: list[dict]) -> str:
    critical = [c for c in checks if c["category"] == "critical"]
    if any(c["status"] == "fail" for c in critical):
        return "NOT_REPRODUCIBLE"
    if any(c["status"] == "missing" for c in critical):
        return "CANNOT_VERIFY"
    non_critical_issue = any(c["status"] in ("fail", "incomplete") for c in checks) \
        or any(c["status"] == "missing" for c in checks)
    return "PARTIALLY_REPRODUCIBLE" if non_critical_issue else "REPRODUCIBLE"


def verify_reproducibility(checkout, manifest, *, config_snapshot: dict | None = None,
                           out_path: Path | None = None) -> dict:
    """Verify ``checkout`` against ``manifest`` (a dict or a path to JSON). Returns
    ``{verdict, checks, fallacy_scan}`` and writes ``reproducibility_report.json``. A missing checkout or an
    empty manifest yields CANNOT_VERIFY."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    checkout = Path(checkout)

    if not checkout.exists():
        report = {"verdict": "CANNOT_VERIFY", "reason": f"checkout {checkout} does not exist", "checks": []}
    elif not manifest:
        report = {"verdict": "CANNOT_VERIFY", "reason": "empty manifest", "checks": []}
    else:
        checks: list[dict] = []
        checks += _check_hashes(checkout, manifest.get("hashes", {}))
        checks += _check_predictions(checkout, manifest.get("predictions", {}))
        checks += _check_config(manifest, config_snapshot)
        checks += _check_decision(manifest)
        fallacy_checks, scan = _check_fallacies(manifest)
        checks += fallacy_checks
        report = {"verdict": _verdict(checks), "checks": checks, "fallacy_scan": scan}

    out_path = Path(out_path) if out_path else config.REPRODUCIBILITY_ROOT / "reproducibility_report.json"
    config.write_text_atomic(json.dumps(report, indent=2, default=str), out_path)
    report["report_path"] = str(out_path)
    return report
