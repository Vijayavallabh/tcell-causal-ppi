"""feat-013 entry point: verify THIS checkout against a manifest built from its own real frozen artifacts,
with the eleven fallacy probes authored from its own real diagnostics.

The expected outcome is CANNOT_VERIFY on the reproduction axis, and that is the CORRECT answer: the
confirmatory decision lives on the sequestered challenge split, which only the test steward may open. This
module exists to make that answer legible instead of silent, so it reports two axes, never one word:

* **reproduction** — the verdict over the hash / schema / config / decision checks. CANNOT_VERIFY here means
  the comparison was never performed, which is emphatically not "verified" and equally not "failed".
* **inference** — the fallacy scan. A flag here is a real trap found in DEVELOPMENT-fold analysis; it is not
  evidence about whether the pipeline reproduces.

``verify_reproducibility``'s single verdict ranks a fallacy flag above an unperformed check, so quoting it
alone would read as "we ran the reproduction and it failed". Both axes are therefore written to the JSON and
printed, and **the process exits non-zero on anything but REPRODUCIBLE** — an unattended run or a CI gate
must not record CANNOT_VERIFY as green.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcell_pipeline import config
from tcell_pipeline.reproducibility import manifest as mf
from tcell_pipeline.reproducibility.verify import SELF_DERIVED, _verdict, verify_reproducibility


def exit_code(verdict) -> int:
    """0 for REPRODUCIBLE and nothing else — including an unrecognised or absent verdict. A whitelist, not
    a blacklist: any status a future check author invents must not be able to exit green."""
    return 0 if verdict == "REPRODUCIBLE" else 1


def _cause(checks: list[dict]) -> dict | None:
    """The check that DRIVES the verdict, walking ``_verdict``'s own precedence: a critical ``fail`` decides
    NOT_REPRODUCIBLE, else a critical non-(pass|incomplete) decides CANNOT_VERIFY, else any non-pass makes it
    PARTIALLY. Reporting merely the FIRST non-passing check would name a self-derived hash as the cause of a
    verdict that a missing decision actually determined."""
    critical = [c for c in checks if c["category"] == "critical"]
    hit = next((c for c in critical if c["status"] == "fail"), None)                       # NOT_REPRODUCIBLE
    # ...else ANY critical status outside {pass, incomplete} -> CANNOT_VERIFY. Matching the named statuses
    # instead would miss a status a future check author invents, and the fallback below — which scans every
    # check, not just critical ones — would then blame whichever non-critical check happened to sort first.
    hit = hit or next((c for c in critical if c["status"] not in ("pass", "incomplete")), None)
    return hit or next((c for c in checks if c["status"] != "pass"), None)                 # PARTIALLY


def reproduction_axis(checks: list[dict]) -> tuple[str, dict | None]:
    """The verdict over everything EXCEPT the fallacy scan — i.e. did the pipeline reproduce, setting aside
    whether its inference was sound. Reuses ``_verdict`` rather than restating its precedence."""
    repro = [c for c in checks if c["check"] != "fallacy_scan"]
    return _verdict(repro), _cause(repro)


def main(argv=()) -> int:
    """Entry point for BOTH the CLI and programmatic callers (``run_module8_real.run_repro()``).

    ``argv`` defaults to empty, NOT to None. ``argparse`` falls back to ``sys.argv`` on None, so a bare
    ``main()`` called from inside another driver inherited THAT driver's flags and died with
    'unrecognized arguments: --part repro' — while the caller's test, which monkeypatched ``main``, stayed
    green. The CLI passes ``sys.argv[1:]`` explicitly at the bottom of this file instead."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=str(config.REPRODUCIBILITY_ROOT))
    ap.add_argument("--root", default=str(config.PROJECT_ROOT))
    args = ap.parse_args(argv)
    root, out_dir = Path(args.root), Path(args.out_dir)

    manifest = mf.build_manifest(root)
    report = verify_reproducibility(root, manifest, config_snapshot=mf.live_config_snapshot(),
                                    out_path=out_dir / "repro_real_report.json")
    repro_verdict, cause = reproduction_axis(report.get("checks") or [])
    scan = report.get("fallacy_scan") or {}
    report |= {"reproduction_verdict": repro_verdict, "reproduction_cause": cause,
               "fallacy_unevaluable": manifest["fallacy_unevaluable"],
               "fallacy_sources": manifest["fallacy_sources"], "unverified": manifest["unverified"],
               "manifest_path": str(out_dir / "manifest_real.json")}
    config.write_text_atomic(json.dumps(manifest, indent=2, default=str), out_dir / "manifest_real.json")
    config.write_text_atomic(json.dumps(report, indent=2, default=str), out_dir / "repro_real_report.json")

    print(f"[repro] verdict (all checks)   = {report['verdict']}")
    print(f"[repro] reproduction axis      = {repro_verdict}"
          + (f"   <- {cause['check']}: {cause['status']}" if cause else ""))
    print(f"[repro] inference axis         = "
          f"{'FLAGGED: ' + ', '.join(scan['flagged']) if scan.get('flagged') else 'no trap detected'}"
          f"   ({scan.get('n_evaluated', 0)}/{scan.get('n_fallacies', 11)} probes evaluated)")
    for check in report.get("checks") or []:
        note = check.get("reason") or ("self-derived" if check.get("provenance") == SELF_DERIVED else "")
        print(f"  {check['status']:10} {check['category']:10} {check['check']}"
              + (f"   ({note})" if note else ""))
    for name, reason in manifest["fallacy_unevaluable"].items():
        print(f"  UNEVALUABLE  {name}: {reason.split('. UNLOCKED BY:')[0]}")
    for name, item in manifest["unverified"].items():
        print(f"[repro] REMAINS ({item['status']}) {name} -> {item['who']}")
    print(f"[repro] report -> {out_dir / 'repro_real_report.json'}  exit={exit_code(report['verdict'])}")
    return exit_code(report["verdict"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))   # explicit: main() itself must never read sys.argv (see its docstring)
