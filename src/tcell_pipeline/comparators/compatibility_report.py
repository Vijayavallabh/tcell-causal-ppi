"""Compatibility + license report for each external comparator (feat-010).

The report records what a comparator is allowed to touch so a reviewer can confirm no proprietary data or
weights entered the benchmark: license, exposure class (public-only vs reimplementation), whether an upstream
public package was wrapped, and the checkpoint used (None for the reimplemented public path). Written as
``<COMPARATORS_ROOT>/<family>/compatibility_report.yaml``.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from tcell_pipeline import config


def compatibility(adapter) -> dict:
    """Compatibility metadata for a comparator class OR instance (reads the class-level declarations)."""
    return {
        "name": getattr(adapter, "__name__", type(adapter).__name__),
        "family": adapter.family,
        "license": adapter.LICENSE,
        "exposure_class": adapter.EXPOSURE_CLASS,
        "checkpoint": adapter.CHECKPOINT,
        # explicit declaration, NOT a substring of EXPOSURE_CLASS — "non-public" contains "public"
        "public_only": bool(getattr(adapter, "PUBLIC_ONLY", False)),
        # whether upstream code ACTUALLY runs, not whether it happens to be installed (recorded separately)
        "wrapped_upstream": bool(getattr(adapter, "wrapped", False)),
        "upstream_importable": bool(getattr(adapter, "upstream_importable", False)),
    }


def write_compatibility_report(adapter, root: Path = config.COMPARATORS_ROOT) -> Path:
    doc = compatibility(adapter)
    path = Path(root) / doc["family"] / "compatibility_report.yaml"
    config.write_text_atomic(yaml.safe_dump(doc, sort_keys=False), path)
    return path
