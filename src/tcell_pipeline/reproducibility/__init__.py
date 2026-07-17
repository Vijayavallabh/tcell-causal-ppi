"""Reproducibility verification (feat-013): clean-checkout hash/schema/decision checks + the 11/11
statistical-fallacy scan, reported as a single REPRODUCIBLE / PARTIALLY / NOT / CANNOT_VERIFY verdict."""
from tcell_pipeline.reproducibility.fallacy_scan import FALLACIES, run_fallacy_scan
from tcell_pipeline.reproducibility.verify import VERDICTS, verify_reproducibility

__all__ = ["run_fallacy_scan", "FALLACIES", "verify_reproducibility", "VERDICTS"]
