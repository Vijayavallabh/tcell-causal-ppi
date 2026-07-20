"""Reproducibility verification (feat-013): clean-checkout hash/schema/decision checks + the 11/11
statistical-fallacy scan, reported as a single REPRODUCIBLE / PARTIALLY / NOT / CANNOT_VERIFY verdict.

``manifest.build_manifest`` builds the frozen record from a checkout's REAL artifacts (labelling every hash
independent-vs-self-derived) and authors the eleven probes from its REAL diagnostics;
``run_repro_real.main`` is the entry point and returns a non-zero exit code on anything but REPRODUCIBLE.
"""
from tcell_pipeline.reproducibility.fallacy_scan import FALLACIES, run_fallacy_scan
from tcell_pipeline.reproducibility.manifest import build_fallacy_inputs, build_manifest, load_diagnostics
from tcell_pipeline.reproducibility.verify import (
    INDEPENDENT,
    SELF_DERIVED,
    VERDICTS,
    verify_reproducibility,
)

__all__ = ["run_fallacy_scan", "FALLACIES", "verify_reproducibility", "VERDICTS", "build_manifest",
           "build_fallacy_inputs", "load_diagnostics", "INDEPENDENT", "SELF_DERIVED"]
