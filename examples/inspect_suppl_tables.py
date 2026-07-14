#!/usr/bin/env python
"""Inspect the 3 CORE experiment-design tables in data/raw/suppl_tables/.

These describe the screen itself (donors, guides, per-perturbation DE summary). The
derived biological analysis tables (signatures, regulator coefficients, clusters,
guide-KD) are covered by inspect_analysis_tables.py; the full present/missing
inventory is in dataset_overview.py.

Covered here:
  - DE_stats.suppl_table.csv          (stale schema: has offtarget_flag /
                                        ontarget_effect_category, LACKS guide/donor
                                        correlation cols -> prefer the .h5ad .obs)
  - sample_metadata.suppl_table.csv   (donor demographics; D1-D4 <-> CE mapping)
  - sgrna_library_metadata.suppl_table.csv  (guide design + target curation)

Key facts verified by this script:
  - The D#<->CE donor mapping is CONSISTENT across runs (D1=CE0008162, D2=CE0010866,
    ...). R1 is a 2-donor batch (D1,D2); R2 is a 4-donor batch (D1-D4). Still key
    donor splits on the physical CE codes — that is what the pseudobulk / donor-pair
    files use as the canonical donor id.
  - ~4.9% of guides have designed_target_gene_id != validated target_gene_id.
    Join perturbations on the validated target_gene_id, not the sgRNA-derived name.

Run:  python examples/inspect_suppl_tables.py
"""
import os
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
TAB = DATA_ROOT / "suppl_tables"

CORE = ["DE_stats.suppl_table.csv", "sample_metadata.suppl_table.csv",
        "sgrna_library_metadata.suppl_table.csv"]


def main() -> None:
    missing = [n for n in CORE if not (TAB / n).exists()]
    assert not missing, f"core tables missing: {missing} (see project README download steps)"

    # --- DE_stats CSV: confirm it's the stale schema ---
    de = pd.read_csv(TAB / "DE_stats.suppl_table.csv", nrows=5, index_col=0)
    print(f"\nDE_stats.suppl_table.csv columns ({len(de.columns)}): {list(de.columns)}")
    print("  note: lacks guide_correlation_* / donor_correlation_* -> use the .h5ad .obs")

    # --- sample_metadata: donor D#<->CE mapping (anchor _D#_ so "CD4i" isn't matched) ---
    sm = pd.read_csv(TAB / "sample_metadata.suppl_table.csv", index_col=0)
    sm["D"] = sm["cell_sample_id"].str.extract(r"_(D\d)_")
    print("\ndonor label mapping (10xrun_id, D#) -> physical CE code:")
    mapping = sm.groupby(["10xrun_id", "D"], observed=True)["donor_id"].first()
    print(mapping.to_string())
    ambiguous = sm.groupby("D")["donor_id"].nunique()
    n_amb = int((ambiguous > 1).sum())
    print(f"\nD# labels mapping to >1 physical CE donor: {n_amb} (0 = mapping is consistent)")

    # --- sgRNA library: target curation mismatch ---
    lib = pd.read_csv(TAB / "sgrna_library_metadata.suppl_table.csv")
    mism = (lib["designed_target_gene_id"].astype(str) != lib["target_gene_id"].astype(str)).sum()
    print(f"\nsgRNA library: {len(lib):,} guides")
    print(f"  designed != validated target_gene_id: {mism} ({100 * mism / len(lib):.1f}%)")

    # --- self-check ---
    assert "guide_correlation_all" not in de.columns, "CSV unexpectedly has q_post corr cols"
    assert n_amb == 0, "donor D#<->CE mapping unexpectedly ambiguous"
    assert mism > 0, "expected some guide target-curation mismatches"
    print("\nOK: stale CSV schema, consistent donor mapping, and curation mismatch confirmed.")


if __name__ == "__main__":
    main()
