#!/usr/bin/env python
"""Inspect the DERIVED analysis supplementary tables (everything beyond the 3 core
experiment-design tables in inspect_suppl_tables.py). These are the paper's ready-made
biological supervision — grouped here by their role in the EG-IPG method.

Sources: 12 of these live in the GitHub analysis repo (not S3). Fetch with the
supplementary-table step in the project README, then run this script.

Roles:
  PROGRAM ANCHORS         curated gene-level signatures for program targets
                          (aging, Th1/Th2 polarization, IL10/IL21 cytokine)
  REGULATOR -> PROGRAM    which regulators drive each signature / downstream genes
                          (regulator-to-program edge supervision)
  COMPLEX / CLUSTER PRIOR regulator clusters mapped to CORUM/STRING/KEGG/Reactome
                          complexes + condition specificity + disease enrichment
  CONTROL WEIGHTING       per-guide knockdown efficiency vs NTC (q_post confidence)
  CROSS-CELL REFERENCE    CD4 vs K562 perturbation-effect correlation

Run:  python examples/inspect_analysis_tables.py
"""
import os
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
TAB = DATA_ROOT / "suppl_tables"


def _load(name):
    p = TAB / name
    if not p.exists():
        print(f"  [MISSING] {name} — run the README supplementary-table fetch step")
        return None
    return pd.read_csv(p)


def _vc(series, dropna=False):
    return dict(series.value_counts(dropna=dropna))


def main() -> None:
    checks = {}

    print("== PROGRAM ANCHORS (gene-level signatures -> program targets) ==")
    aging = _load("CD4T_aging_signature_DE_results_full.suppl_table.csv")
    if aging is not None:
        print(f"  CD4T_aging_signature: {len(aging)} genes | contrasts={_vc(aging['contrast'])}")
        print("    -> aging/exhaustion program anchor; log_fc/zscore per gene")
        checks["aging"] = {"log_fc", "zscore", "gene_name"}.issubset(aging.columns)
    pol = _load("Th2_Th1_polarization_signature_DE_results_full.suppl_table.csv")
    if pol is not None:
        print(f"  Th2_Th1_polarization: {len(pol)} rows | contrasts={_vc(pol['contrast'])}")
        print("    -> Th1/Th2 polarization program anchor (two reference cohorts)")
        checks["pol"] = {"log_fc", "zscore"}.issubset(pol.columns)
    il = _load("IL10IL21bulkRNAseq_DESeq2_results.csv")
    if il is not None:
        print(f"  IL10IL21 bulk RNA-seq: {len(il)} rows | contrasts={il['contrast'].nunique()}")
        print("    -> IL10/IL21 cytokine program anchor (bulk DESeq2)")

    print("\n== REGULATOR -> PROGRAM (edge supervision) ==")
    ag = _load("aging_prediction_condition_comparison_regulator_coefficients.csv")
    if ag is not None:
        print(f"  aging regulator coefs: {len(ag)} | celltype={_vc(ag['celltype'])}")
        print(f"    known_regulators={_vc(ag['known_regulators'])} -> linear-model regulator weights per context")
    po = _load("polarization_prediction_condition_comparison_regulator_coefficients.csv")
    if po is not None:
        print(f"  polarization regulator coefs: {len(po)} | signature={_vc(po['signature'])}")
    dg = _load("clustering_downstream_genes.csv")
    if dg is not None:
        print(f"  clustering_downstream_genes: {len(dg):,} rows | clusters={dg['hdbscan_cluster'].nunique()} "
              f"| condition={_vc(dg['condition'])}")
        print("    -> regulator-cluster -> downstream gene edges; sign_coherence in [-1,1]")
        checks["downstream"] = dg["sign_coherence"].between(-1, 1).all()

    print("\n== COMPLEX / CLUSTER PRIORS + annotation ==")
    ann = _load("clustering_results_and_annotations.csv")
    if ann is not None:
        print(f"  clustering_results_and_annotations: {len(ann)} clusters, {ann.shape[1]} cols")
        print(f"    condition_specificity={_vc(ann['condition_specificity'])}")
        print(f"    CORUM overlap present for {ann['complex_corum'].notna().sum()}/{len(ann)} clusters "
              "(also STRING/KEGG/Reactome)")
        print(f"    manual_annotation e.g. {ann['manual_annotation'].dropna().head(3).tolist()}")
        print("    -> maps regulator clusters to protein complexes -> complex nodes + bio-alignment metrics")
        checks["ann"] = {"complex_corum", "complex_stringdb", "manual_annotation"}.issubset(ann.columns)
    au = _load("cluster_autoimmune_enrichment_results.suppl_table.csv")
    if au is not None:
        print(f"  cluster_autoimmune_enrichment: {len(au)} rows | diseases={au['disease'].nunique()} "
              f"| negative_control rows={int(au['negative_control_disease'].sum())}")

    print("\n== CONTROL WEIGHTING (q_post confidence) ==")
    kd = _load("guide_kd_efficiency.suppl_table.csv")
    if kd is not None:
        print(f"  guide_kd_efficiency: {len(kd):,} (guide x condition) | genes={kd['perturbed_gene_id'].nunique()}")
        print(f"    signif_knockdown={_vc(kd['signif_knockdown'])} | "
              f"high_confidence_no_effect={_vc(kd['high_confidence_no_effect_guides'])}")
        print("    -> per-guide KD strength vs NTC; a q_post weighting / QC source, NOT an H1 input")
        checks["kd"] = {"signif_knockdown", "t_statistic", "culture_condition"}.issubset(kd.columns)

    print("\n== CROSS-CELL REFERENCE ==")
    k562 = _load("K562_comparison.suppl_table.csv")
    if k562 is not None:
        print(f"  K562_comparison: {len(k562)} rows | genes={k562['target_contrast_gene_name'].nunique()} "
              f"x 3 conditions")
        print("    -> CD4-vs-K562 logFC correlation + random controls; cross-cell-type generalization reference")

    print("\n== BONUS (validation / constructs — not modeling inputs) ==")
    for n in ["IL10_IL21_arrayed_validation.csv", "stabl_constructs.csv"]:
        t = _load(n)
        if t is not None:
            print(f"  {n}: {len(t)} rows, cols={list(t.columns)}")

    # --- self-check (only on tables that are present) ---
    assert all(checks.values()), f"schema check failed: {checks}"
    assert checks, "no analysis tables found — fetch them first (see project README)"
    print(f"\nOK: {len(checks)} present analysis tables passed schema checks.")


if __name__ == "__main__":
    main()
