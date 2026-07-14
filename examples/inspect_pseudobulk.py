#!/usr/bin/env python
"""Inspect GWCD4i.pseudobulk_merged.h5ad — raw pseudobulk counts and the ONLY
source of non-targeting controls in the aggregate layer.

Grain: one row per (guide, donor, culture condition). Columns = measured genes.

Key facts verified by this script:
  - `.X` is CSR sparse, 278684 x 18129, ~2.78B nnz (~55% dense) — DO NOT densify whole.
  - Carries 11,018 non-targeting-control rows -> this is where independent control
    centroids for shared-control-bias-safe metrics must come from (DE has none).
  - donor_id uses physical CE codes (4 donors); use these, never the D1-D4 filename
    labels, which are batch-relative (see inspect_metadata_jsonld.py).
  - keep_* boolean columns define the DE-eligibility filtering already applied upstream.

Run:  python examples/inspect_pseudobulk.py
"""
import os
from pathlib import Path

import anndata as ad
import h5py

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parent.parent / "data" / "raw"))
PATH = DATA_ROOT / "GWCD4i.pseudobulk_merged.h5ad"


def main() -> None:
    print(f"FILE: {PATH}  ({PATH.stat().st_size / 1e9:.1f} GB)\n")

    with h5py.File(PATH, "r") as f:
        x = f["X"]
        enc = dict(x.attrs).get("encoding-type")
        shape = tuple(dict(x.attrs).get("shape"))
        nnz = x["data"].shape[0]
        density = nnz / (shape[0] * shape[1])
        print(f"X: {enc}  shape={shape}  nnz={nnz:,}  density={density:.1%}")
        print("obs cols:", list(f["obs"].keys()))

    a = ad.read_h5ad(PATH, backed="r")
    obs = a.obs
    print(f"\nshape: {a.shape}  (n_obs=guide x donor x condition)")

    print("\nculture_condition:")
    print(obs["culture_condition"].value_counts().to_string())
    print("\ndonor_id (physical CE codes):")
    print(obs["donor_id"].value_counts().to_string())
    print("\nguide_type:")
    print(obs["guide_type"].value_counts().to_string())

    n_ntc = (obs["guide_type"] == "non-targeting").sum()
    print(f"\nnon-targeting-control rows: {n_ntc}  <- control-centroid source")
    print(f"unique guide_id: {obs['guide_id'].nunique()} | unique perturbed_gene: {obs['perturbed_gene_name'].nunique()}")

    print("\nDE-eligibility keep_* flags:")
    for c in ["keep_min_cells", "keep_total_counts", "keep_effective_guides",
              "keep_test_genes", "keep_for_DE"]:
        print(f"  {c:22s} {obs[c].value_counts(dropna=False).to_dict()}")

    # --- self-check ---
    assert a.shape == (278684, 18129), a.shape
    assert n_ntc > 0, "expected non-targeting controls in pseudobulk"
    assert obs["donor_id"].nunique() == 4, "expected 4 physical donors"
    print("\nOK: dims, controls, and donor count match expectations.")


if __name__ == "__main__":
    main()
