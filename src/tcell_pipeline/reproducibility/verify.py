"""Reproducibility verification (feat-013): given a clean ``checkout`` and a frozen ``manifest``, re-derive
the deterministic preprocessing hashes, re-check the prediction schema + row counts + config + checkpoint
provenance, confirm the same confirmatory decision within tolerance, run the 11/11 fallacy scan, and return a
single verdict — REPRODUCIBLE / PARTIALLY_REPRODUCIBLE / NOT_REPRODUCIBLE / CANNOT_VERIFY (report
§reproducibility).

The manifest is the frozen record the original run published; this module re-computes each item against the
checkout and compares. It performs NO training itself — the "rerun the final model + comparators over frozen
seeds" step produces the challenge predictions + sealed decision the manifest carries under ``observed`` (the
sealed evaluator writes them); verify checks that those reproduce the frozen ``decision``.

**Design rule, learned the hard way: this module must never certify a check it did not actually perform.**
Every review finding against it was a variant of that — absolute manifest paths that hashed the original run
instead of the checkout, an absent `hashes` block that emitted zero checks, a decision comparison that passed
on `bool(None) == bool(None)`, a config check that defaulted to a clean "skip". So:

* ``_verdict`` is **whitelist-shaped**: a critical check certifies only on an explicit ``pass``. Any novel or
  unexpected status degrades the verdict rather than sailing through.
* Required items **must be present**: each ``_DETERMINISTIC`` hash and a ``predictions`` block absent from the
  manifest emit an explicit ``missing`` check.
* Malformed manifest entries yield ``missing`` (→ CANNOT_VERIFY), not a traceback: the module's contract is to
  return a verdict.
* The caller-supplied ``decision.tolerance`` is capped (``MAX_DECISION_TOLERANCE``) — a manifest that
  self-declares a huge tolerance could otherwise wave any drift through its own check.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from tcell_pipeline import config
from tcell_pipeline.reproducibility.fallacy_scan import run_fallacy_scan

VERDICTS = ("REPRODUCIBLE", "PARTIALLY_REPRODUCIBLE", "NOT_REPRODUCIBLE", "CANNOT_VERIFY")
# deterministic preprocessing artifacts whose hashes MUST reproduce bit-for-bit (report: id_mapping, splits,
# de_layers) — a mismatch here means the frozen pipeline did not reproduce.
_DETERMINISTIC = ("id_mapping", "splits", "de_layers")
# The sealed evaluator emits no `tolerance`, and bit-exact float equality across machines/BLAS builds is not a
# realistic reproduction bar, so a manifest that pins no tolerance gets this (tight) float-noise default.
DEFAULT_DECISION_TOLERANCE: float = 1e-6
# Ceiling on a manifest-declared tolerance. The endpoints compared are correlations in [-1, 1] and the H1
# margin is DELTA_PRED (0.05), so a tolerance at or above that scale could wave through a drift large enough
# to flip the confirmatory call — a manifest cannot be trusted to bound its own check.
MAX_DECISION_TOLERANCE: float = 0.01

_PASS, _FAIL, _MISSING, _INCOMPLETE = "pass", "fail", "missing", "incomplete"


def _sha256_file(path: Path) -> str | None:
    """sha256 of a regular file, or None if it is absent / not a file (a directory would raise on open)."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(checkout: Path, rel) -> Path | None:
    """Resolve a manifest path INSIDE ``checkout``, or None if it escapes.

    config.py's roots are absolute by default, so a manifest built from them would send every hash/schema
    check at the ORIGINAL run's files — verifying that run against itself and certifying a checkout whose
    files were never read. Absolute paths and ``..`` escapes are therefore rejected.

    Containment is checked LEXICALLY (``normpath``, no ``resolve()``): a checkout may legitimately symlink a
    large artifact to a shared store (``data/`` is git-ignored and the roots are env-overridable), and
    resolving symlinks would reject that documented layout with a false 'escapes the checkout'."""
    if not isinstance(rel, (str, os.PathLike)) or str(rel) == "":
        return None
    p = Path(rel)
    if p.is_absolute():
        return None
    root = os.path.abspath(checkout)
    joined = os.path.normpath(os.path.join(root, str(p)))
    if joined != root and not joined.startswith(root + os.sep):
        return None
    return Path(joined)


def _check_hashes(checkout: Path, entries) -> list[dict]:
    checks = []
    if not isinstance(entries, dict):
        entries = {}
    # every deterministic artifact must be PRESENT in the manifest — an absent (or misspelled) block would
    # otherwise emit zero checks and sail through as REPRODUCIBLE without hashing anything
    for name in _DETERMINISTIC:
        if name not in entries:
            checks.append({"check": f"hash:{name}", "category": "critical", "status": _MISSING,
                           "reason": "manifest declares no hash for this deterministic artifact"})
    for name, spec in entries.items():
        category = "critical" if name in _DETERMINISTIC else "provenance"
        # a malformed entry must yield a verdict, not a traceback out of the verifier
        if not isinstance(spec, dict) or "path" not in spec or "sha256" not in spec:
            checks.append({"check": f"hash:{name}", "category": category, "status": _MISSING,
                           "reason": "manifest entry is malformed (needs a 'path' and a 'sha256')"})
            continue
        path = _resolve(checkout, spec["path"])
        if path is None:
            checks.append({"check": f"hash:{name}", "category": category, "status": _MISSING,
                           "reason": f"path {spec['path']!r} is absolute, empty or escapes the checkout — "
                                     f"cannot attribute it to this checkout"})
            continue
        actual = _sha256_file(path)
        if actual is None:
            status = _MISSING
        elif actual == spec["sha256"]:
            status = _PASS
        else:
            status = _FAIL
        checks.append({"check": f"hash:{name}", "category": category, "status": status,
                       "expected": spec["sha256"], "actual": actual})
    return checks


def _check_predictions(checkout: Path, entries) -> list[dict]:
    import pandas as pd

    if not isinstance(entries, dict) or not entries:
        # same class as the absent-hashes hole: zero emitted checks would mean the schema/row-count
        # verification the report requires silently never happened
        return [{"check": "schema", "category": "schema", "status": _MISSING,
                 "reason": "manifest declares no predictions to check"}]
    checks = []
    for name, spec in entries.items():
        if not isinstance(spec, dict) or "path" not in spec:
            checks.append({"check": f"schema:{name}", "category": "schema", "status": _MISSING,
                           "reason": "manifest entry is malformed (needs a 'path')"})
            continue
        path = _resolve(checkout, spec["path"])
        if path is None or not path.is_file():
            checks.append({"check": f"schema:{name}", "category": "schema", "status": _MISSING,
                           "reason": "prediction file is absent, or its path escapes the checkout"})
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            checks.append({"check": f"schema:{name}", "category": "schema", "status": _MISSING,
                           "reason": f"unreadable: {type(exc).__name__}: {exc}"})
            continue
        cols = list(frame.columns)
        prefixes = spec.get("columns_prefixes", ["row_index", "delta_z_", "delta_x_", "sigma_"])
        have = all(any(c == p or c.startswith(p) for c in cols) for p in prefixes)
        rows_ok = spec.get("n_rows") is None or len(frame) == spec["n_rows"]
        checks.append({"check": f"schema:{name}", "category": "schema",
                       "status": _PASS if (have and rows_ok) else _FAIL,
                       "n_rows": len(frame), "expected_rows": spec.get("n_rows"), "columns_ok": have})
    return checks


def _check_config(manifest: dict, config_snapshot: dict | None) -> list[dict]:
    """The config the frozen run used is part of what must reproduce (a changed DELTA_PRED alone can flip the
    H1 call), so an unverifiable config is 'missing', never a silent clean skip."""
    expected = (manifest.get("config_hashes") or {}).get("config_snapshot")
    if expected is None:
        return [{"check": "config_hash", "category": "critical", "status": _MISSING,
                 "reason": "manifest declares no config_hashes.config_snapshot"}]
    if config_snapshot is None:
        return [{"check": "config_hash", "category": "critical", "status": _MISSING,
                 "reason": "no config_snapshot supplied to compare against the frozen hash"}]
    actual = hashlib.sha256(json.dumps(config_snapshot, sort_keys=True, default=str).encode()).hexdigest()
    return [{"check": "config_hash", "category": "critical",
             "status": _PASS if actual == expected else _FAIL, "expected": expected, "actual": actual}]


def _check_decision(manifest: dict) -> list[dict]:
    """The confirmatory call must be genuinely COMPARED.

    Presence alone is not a comparison: ``bool(None) == bool(None)`` and ``bool("false") is True``, so
    ``h1_confirmed`` must be a real ``bool`` in BOTH records. At least one numeric field must be shared, and
    the manifest's self-declared tolerance is capped — otherwise the record under test could set its own bar
    (``tolerance: 1e9`` waves through any drift, including a sign flip)."""
    def miss(reason):
        return [{"check": "confirmatory_decision", "category": "critical", "status": _MISSING,
                 "reason": reason}]

    frozen = manifest.get("decision")
    observed = (manifest.get("observed") or {}).get("decision")
    if not isinstance(frozen, dict) or not isinstance(observed, dict):
        return miss("manifest lacks a decision and/or observed.decision object")
    if not isinstance(frozen.get("h1_confirmed"), bool) or not isinstance(observed.get("h1_confirmed"), bool):
        return miss("h1_confirmed must be a boolean in BOTH the frozen and observed decision — a null or "
                    "string value is not a comparison (bool(None)==bool(None), bool('false') is True)")
    compared = [k for k in ("lcb_95", "rho_egipg", "delta_vs_best")
                if isinstance(frozen.get(k), (int, float)) and isinstance(observed.get(k), (int, float))
                and not isinstance(frozen.get(k), bool)]
    if not compared:
        return miss("no numeric decision field (lcb_95/rho_egipg/delta_vs_best) present as a number in both")
    raw_tol = frozen.get("tolerance", DEFAULT_DECISION_TOLERANCE)
    if isinstance(raw_tol, bool) or not isinstance(raw_tol, (int, float)):
        return miss(f"decision.tolerance must be a number (got {raw_tol!r})")
    tol = float(raw_tol)
    if not (0.0 <= tol <= MAX_DECISION_TOLERANCE):
        return miss(f"decision.tolerance {tol} is outside [0, {MAX_DECISION_TOLERANCE}] — a manifest cannot "
                    f"widen its own bar past the scale of the endpoints it certifies")
    same_call = frozen["h1_confirmed"] == observed["h1_confirmed"]
    within = all(abs(float(frozen[k]) - float(observed[k])) <= tol for k in compared)
    return [{"check": "confirmatory_decision", "category": "critical",
             "status": _PASS if (same_call and within) else _FAIL,
             "frozen": frozen, "observed": observed, "same_call": same_call, "within_tolerance": within,
             "compared_fields": compared, "tolerance": tol}]


def _check_fallacies(manifest: dict) -> tuple[list[dict], dict]:
    inputs = manifest.get("fallacy_inputs")
    if not inputs:
        return [{"check": "fallacy_scan", "category": "critical", "status": _MISSING,
                 "reason": "manifest carries no fallacy_inputs"}], {}
    scan = run_fallacy_scan(inputs)
    if scan["flagged"]:
        status = _FAIL           # a detected inference trap invalidates the claim
    elif scan.get("crashed"):
        status = _MISSING        # a detector BUG (not degenerate input) — the scan itself is untrustworthy
    elif not scan["complete"]:
        status = _INCOMPLETE     # ran clean but some probe was inadequate -> partial coverage
    else:
        status = _PASS
    return [{"check": "fallacy_scan", "category": "critical", "status": status,
             "n_evaluated": scan["n_evaluated"], "flagged": scan["flagged"], "complete": scan["complete"],
             "errored": scan.get("errored", []), "crashed": scan.get("crashed", [])}], scan


def _verdict(checks: list[dict]) -> str:
    """Whitelist-shaped: a critical check certifies ONLY on an explicit pass.

    A blacklist ('bad unless the status is in {fail, missing, incomplete}') is what let a 'skip' status
    certify; any status a future check author invents would silently do the same. ``incomplete`` is the one
    non-pass status that still permits PARTIALLY — it means the scan ran but its coverage was short."""
    critical = [c for c in checks if c["category"] == "critical"]
    if any(c["status"] == _FAIL for c in critical):
        return "NOT_REPRODUCIBLE"
    if any(c["status"] not in (_PASS, _INCOMPLETE) for c in critical):
        return "CANNOT_VERIFY"
    if any(c["status"] != _PASS for c in checks):
        return "PARTIALLY_REPRODUCIBLE"
    return "REPRODUCIBLE"


def verify_reproducibility(checkout, manifest, *, config_snapshot: dict | None = None,
                           out_path: Path | None = None) -> dict:
    """Verify ``checkout`` against ``manifest`` (a dict or a path to JSON). Returns
    ``{verdict, checks, fallacy_scan}`` and writes ``reproducibility_report.json``. A missing checkout, an
    empty manifest, or a manifest too malformed to read yields CANNOT_VERIFY — this function returns a
    verdict rather than raising, so an unattended verification always produces a report."""
    if isinstance(manifest, (str, Path)):
        try:
            manifest = json.loads(Path(manifest).read_text())
        except Exception as exc:
            manifest = None
            report = {"verdict": "CANNOT_VERIFY", "reason": f"unreadable manifest: {exc}", "checks": []}
    checkout = Path(checkout)

    if manifest is None:
        pass  # report already set above
    elif not checkout.exists():
        report = {"verdict": "CANNOT_VERIFY", "reason": f"checkout {checkout} does not exist", "checks": []}
    elif not isinstance(manifest, dict) or not manifest:
        report = {"verdict": "CANNOT_VERIFY", "reason": "empty or non-object manifest", "checks": []}
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
