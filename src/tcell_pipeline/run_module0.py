"""Module 0 orchestrator — runs the data-pipeline steps in dependency order.

Each step is independently callable (``python -m tcell_pipeline.<step>``); this driver
just sequences them. Run: ``python src/tcell_pipeline/run_module0.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put src/ on path for direct runs

from tcell_pipeline import (  # noqa: E402
    complex_membership,
    control_profiles,
    de_extraction,
    feature_availability,
    id_mapping,
    perturbation_table,
    ppi_graph,
)

STEPS = [
    ("id_mapping", id_mapping.run),
    ("de_extraction", de_extraction.run),
    ("perturbation_table", perturbation_table.run),
    ("ppi_graph", ppi_graph.run),
    ("complex_membership", complex_membership.run),
    ("control_profiles", control_profiles.run),
    ("feature_availability", feature_availability.run),
]


def run() -> None:
    for i, (name, fn) in enumerate(STEPS, 1):
        print(f"\n=== [{i}/{len(STEPS)}] {name} ===")
        fn()
    print("\n=== Module 0 complete ===")


if __name__ == "__main__":
    run()
