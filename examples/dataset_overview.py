#!/usr/bin/env python
"""Dataset provenance + local-vs-expected inventory for the Primary Human CD4+ T
Cell Perturb-seq resource.

Provenance below is from the Virtual Cells Platform dataset card (SOURCE_URL) plus
the local Croissant descriptors. This script does not hit the network — it prints
the canonical identity/license/access info and reports which expected artifacts are
present locally, with the exact S3 URIs / commands to fetch the missing ones.

Run:  python examples/dataset_overview.py
"""
import json
import os
from pathlib import Path

SOURCE_URL = "https://virtualcellmodels.cziscience.com/dataset/genome-scale-tcell-perturb-seq"

PROVENANCE = {
    "name": "Primary Human CD4+ T Cell Perturb-seq",
    "version": "v1.0.0 (processed)",
    "released": "2025-12-22",
    "license": "MIT",
    "citation": "https://www.biorxiv.org/content/10.64898/2025.12.23.696273v1",
    "analysis_repo": "https://github.com/emdann/GWT_perturbseq_analysis_2025",
    "s3_bucket": "s3://genome-scale-tcell-perturb-seq/marson2025_data",
    "vcp_cli": 'vcp data search "Primary Human CD4+ T Cell Perturb-seq" --exact',
    "cells": "~22M",
    "donors": 4,
    "conditions": ["Rest", "Stim8hr", "Stim48hr"],
    "assay": "CRISPRi genome-scale Perturb-seq; Ultima seq; GEMX_flex_v2",
    "pii": "none (per VCP card)",
    "creators": ("Zhu, Dann, Yan, Reyes Retana, Goto, Guitche, Petersen, Ota, "
                 "Pritchard, Marson"),
}

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))

# Aggregate layer — the first-paper footprint (~100 GiB). Present locally.
AGGREGATE = [
    "GWCD4i.DE_stats.h5ad",
    "GWCD4i.pseudobulk_merged.h5ad",
    "GWCD4i.DE_stats.by_guide.h5mu",
    "GWCD4i.DE_stats.by_donors.h5mu",
]

# Supplementary tables described by the card (basename -> role). Most are NOT
# downloaded; several feed the report's program anchors / control weighting.
SUPPL = {
    "DE_stats.suppl_table.csv": "DE .obs (stale schema; prefer the .h5ad)",
    "sample_metadata.suppl_table.csv": "donor demographics + D#<->CE map",
    "sgrna_library_metadata.suppl_table.csv": "guide design + target curation",
    "QC_summaries_per_sample_lane.csv": "per-sample/lane QC",
    "guide_kd_efficiency.suppl_table.csv": "per-guide KD efficiency vs NTC (control weighting)",
    "CD4T_aging_signature_DE_results_full.suppl_table.csv": "aging program anchor",
    "Th2_Th1_polarization_signature_DE_results_full.suppl_table.csv": "Th1/Th2 program anchor",
    "cluster_autoimmune_enrichment_results.suppl_table.csv": "autoimmune enrichment",
    "aging_prediction_condition_comparison_regulator_coefficients.csv": "aging regulator coefs",
    "polarization_prediction_condition_comparison_regulator_coefficients.csv": "polarization regulator coefs",
    "K562_comparison.suppl_table.csv": "cross-cell-type comparison",
    "clustering_downstream_genes.csv": "regulator-cluster downstream genes",
    "Th1Th2_validation_summary.suppl_table.csv": "arrayed CRISPRi validation",
}


def _cell_level_uris():
    """Authoritative S3 URIs for the 12 (donor x condition) cell-level files,
    read from the local Croissant descriptors."""
    uris = {}
    for p in sorted((DATA_ROOT / "metadata").glob("*.jsonld")):
        doc = json.loads(p.read_text())
        uri = doc["distribution"][0]["contentUrl"]
        uris[Path(uri).name] = uri
    return uris


def _status(path: Path):
    if path.exists():
        return f"present  {path.stat().st_size / 1e9:6.1f} GB"
    return "MISSING"


def main() -> None:
    print(f"SOURCE: {SOURCE_URL}\n")
    print("== Provenance ==")
    for k, v in PROVENANCE.items():
        print(f"  {k:12s}: {v}")

    print("\n== Aggregate layer (first-paper footprint) ==")
    for n in AGGREGATE:
        print(f"  [{_status(DATA_ROOT / n)}]  {n}")

    print("\n== Supplementary tables ==")
    missing_suppl = []
    for n, role in SUPPL.items():
        p = DATA_ROOT / "suppl_tables" / n
        if not p.exists():
            missing_suppl.append(n)
        print(f"  [{_status(p):>18s}]  {n}  — {role}")

    print("\n== Cell-level files (~1.58 TiB; storage-blocked, not downloaded) ==")
    cell = _cell_level_uris()
    for n in cell:
        print(f"  [{_status(DATA_ROOT / n):>18s}]  {n}")

    # --- fetch guidance for what's missing ---
    print("\n== How to fetch missing artifacts ==")
    print(f"  VCP CLI : {PROVENANCE['vcp_cli']}")
    print(f"  S3 root : {PROVENANCE['s3_bucket']}/   (aws s3 ls / cp --no-sign-request)")
    print(f"  Suppl   : {PROVENANCE['analysis_repo']}/tree/master/metadata/suppl_tables")
    if missing_suppl:
        print(f"\n  {len(missing_suppl)} suppl tables missing — needed before program anchors "
              "(aging/Th1/Th2) and guide-KD control weighting:")
        for n in missing_suppl:
            print(f"    - {n}")
    print("\n  One cell-level shard (example, streaming target only — do NOT stage all 12):")
    first = next(iter(cell.values()))
    print(f"    aws s3 cp --no-sign-request {first} $SCRATCH_ROOT/")

    # --- self-check ---
    assert all((DATA_ROOT / n).exists() for n in AGGREGATE), "aggregate layer incomplete"
    assert len(cell) == 12, f"expected 12 cell-level descriptors, got {len(cell)}"
    assert PROVENANCE["license"] == "MIT"
    print("\nOK: aggregate layer complete; 12 cell-level URIs resolved; provenance loaded.")


if __name__ == "__main__":
    main()
