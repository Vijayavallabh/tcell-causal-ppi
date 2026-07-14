#!/usr/bin/env python
"""Inspect GWCD4i.DE_stats.h5ad — the core supervised target for the first paper.

Grain: one row per (perturbed gene, culture condition). Columns = measured genes.

Key facts verified by this script:
  - `.X` is EMPTY. All signal lives in 6 DENSE float64 `.layers`
    (log_fc, zscore, p_value, adj_p_value, baseMean, lfcSE), each 33983x10282.
  - The whole zscore target in float32 is ~1.4 GB -> fits in RAM; no chunking needed.
  - There are NO non-targeting-control rows here (DE is already a per-perturbation
    contrast). Independent control centroids must come from the pseudobulk file.
  - `.obs` carries the q_post leakage fields (ontarget_significant, guide/donor
    correlations, n_downstream, off-target flags) that must be fenced off from H1.
  - `.varm` has per-condition measured_genes_stats with n_regulators per gene —
    a ready-made program/anchor supervision signal.

Run:  python examples/inspect_de_stats.py
"""
import os
from pathlib import Path

import anndata as ad
import h5py

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
PATH = DATA_ROOT / "GWCD4i.DE_stats.h5ad"

# q_post = response-derived; must NOT enter H1 features (leakage). Reference list.
Q_POST_COLS = [
    "ontarget_effect_size", "ontarget_significant", "neighboring_gene_KD",
    "distal_offtarget_flag", "low_target_gex", "n_up_genes", "n_down_genes",
    "n_total_de_genes", "n_downstream", "guide_correlation_signif",
    "guide_correlation_all", "donor_correlation_all_mean", "donor_correlation_hits_mean",
]


def main() -> None:
    print(f"FILE: {PATH}  ({PATH.stat().st_size / 1e9:.1f} GB)\n")

    # --- raw HDF5 view: layers are dense, X is empty ---
    with h5py.File(PATH, "r") as f:
        assert isinstance(f["X"], h5py.Dataset) and f["X"].shape is None, "X should be empty"
        print("X: EMPTY (shape=None) — signal is in .layers only")
        print("\nLayers (all dense float64):")
        for k in f["layers"]:
            d = f["layers"][k]
            print(f"  {k:14s} shape={d.shape} dtype={d.dtype}")
        print("\nvarm (per-condition gene stats):", list(f["varm"].keys()))
        nreg = f["varm"]["measured_genes_stats_Rest"]["n_regulators"][:]
        print(f"  n_regulators/gene (Rest): mean={nreg.mean():.1f} max={nreg.max()}")

    # --- backed AnnData: obs distributions, no matrices loaded ---
    a = ad.read_h5ad(PATH, backed="r")
    obs = a.obs
    print(f"\nshape: {a.shape}  (n_obs=perturbation-condition rows, n_vars=measured genes)")

    print("\nculture_condition:")
    print(obs["culture_condition"].value_counts().to_string())

    n_targets = obs["target_contrast_gene_name"].nunique()
    print(f"\nunique target genes: {n_targets}")
    cond_cov = obs.groupby("target_contrast_gene_name", observed=True)["culture_condition"].nunique()
    print("targets present in N conditions:", dict(cond_cov.value_counts().sort_index()))

    # self/measured overlap — relevant for graph self-loops
    tg = set(obs["target_contrast"].astype(str))
    vg = set(a.var["gene_ids"].astype(str))
    print(f"targets that are themselves measured genes: {len(tg & vg)} / {len(tg)}")

    print("\nq_post leakage fields (DO NOT feed to H1) — value counts:")
    for c in ["ontarget_significant", "low_target_gex", "single_guide_estimate",
              "distal_offtarget_flag", "neighboring_gene_KD"]:
        vc = obs[c].value_counts(dropna=False).to_dict()
        print(f"  {c:22s} {vc}")

    # controls check — \b boundaries so the real gene KNTC1 is not a false positive
    gn = obs["target_contrast_gene_name"].astype(str)
    n_ctrl = int(gn.str.contains(r"\b(?:NTC|non-targeting|control)\b", case=False, regex=True).sum())
    print(f"\nnon-targeting-control rows in DE: {n_ctrl}  (expected 0 — use pseudobulk for controls)")

    # --- self-check ---
    assert a.shape == (33983, 10282), a.shape
    assert set(Q_POST_COLS).issubset(obs.columns), "q_post schema drifted"
    assert n_ctrl == 0, "unexpected control rows in DE"
    print("\nOK: dims and schema match expectations.")


if __name__ == "__main__":
    main()
