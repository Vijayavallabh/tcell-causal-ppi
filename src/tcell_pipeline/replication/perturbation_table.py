"""L1 stage 2: the per-dataset perturbation-condition table the training path requires.

The reference builder (`tcell_pipeline.perturbation_table.build_perturbation_table`) cannot be used
directly: it joins id_mapping on `ensembl_id`, because in the reference screen `target_contrast` IS an
Ensembl accession. Replication DE matrices are keyed by gene SYMBOL, so the join is on `hgnc_symbol`
here. The PPI degree computation IS reused unchanged, so the graph-derived features are identical in
construction to the reference's -- which matters, since those three scalars are the channel the L5
ablation showed carries measurable signal.

DONOR PCs ARE ZEROED, DELIBERATELY. `ContextEncoder` consumes `config.DONOR_PCA_DIMS` (32) donor PCA
scalars, and no replication dataset has donor structure: these are cell lines or single-subject
screens. Zeroing is defensible because a constant contributes nothing downstream and affects every arm
identically, so it cannot bias the graph-versus-no-graph contrast in either direction. It is done
explicitly and recorded in the provenance rather than arising from a missing column, and it belongs in
the paper next to the PINNACLE and ESM-2 coverage figures.

    PYTHONPATH=src python -m tcell_pipeline.replication.perturbation_table --dataset <name>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from tcell_pipeline import config
from tcell_pipeline.perturbation_table import PPI_DEGREE_COLS, compute_ppi_degrees

OUT_ROOT = "data/intermediate/replication"


def build(dataset: str, out_root: str = OUT_ROOT) -> tuple[str, dict]:
    import anndata as ad

    de_path = f"{out_root}/{dataset}.DE_stats_v2.h5ad"
    a = ad.read_h5ad(de_path, backed="r")
    obs = a.obs.reset_index(drop=True).copy()
    a.file.close()

    t = obs[["target_contrast_gene_name", "culture_condition"]].copy()
    t.insert(0, "row_index", np.arange(len(t), dtype=np.int64))
    t["target_contrast"] = t["target_contrast_gene_name"].astype(str)

    idm = pd.read_parquet(config.ID_MAPPING_PATH)
    keep = ["hgnc_symbol", "ensembl_id", "uniprot_id", "entrez_id", "mapping_status"]
    idm = idm[[c for c in keep if c in idm.columns]].dropna(subset=["hgnc_symbol"])
    idm = idm.drop_duplicates(subset=["hgnc_symbol"], keep="first")
    t = t.merge(idm, left_on="target_contrast_gene_name", right_on="hgnc_symbol", how="left")

    edges_path = config.PROTEIN_EDGES_PATH
    edges = pd.read_parquet(edges_path) if os.path.exists(edges_path) else pd.DataFrame()
    degrees = compute_ppi_degrees(edges) if len(edges) else {}
    key = t["target_contrast_gene_name"].astype(str).str.upper()
    for c in PPI_DEGREE_COLS:
        m = degrees.get(c)
        t[c] = key.map(m).fillna(0).astype(int) if m else 0

    # control_baseline_expr: mean log1p-CPM of the target's OWN transcript in the control cells of its
    # condition. This is a q_pre feature (it is measured before the perturbation's response), and it is
    # what the reference column means.
    t["control_baseline_expr"] = _baseline_expr(dataset, t)

    for i in range(config.DONOR_PCA_DIMS):
        t[f"donor_pc_{i:02d}"] = 0.0

    os.makedirs(out_root, exist_ok=True)
    dest = f"{out_root}/{dataset}.perturbation_condition.parquet"
    t.to_parquet(dest, index=False)
    prov = {
        "dataset": dataset, "rows": len(t), "cols": len(t.columns),
        "mapped_to_uniprot": int(t["uniprot_id"].notna().sum()),
        "ppi_degree_nonzero": {c: int((t[c] > 0).sum()) for c in PPI_DEGREE_COLS},
        "baseline_expr_nonnull": int(pd.notna(t["control_baseline_expr"]).sum()),
        "donor_pcs": f"{config.DONOR_PCA_DIMS} columns ZEROED (no donor structure in this dataset; "
                     f"constant, affects all arms identically, recorded not accidental)",
    }
    json.dump(prov, open(dest.replace(".parquet", ".provenance.json"), "w"), indent=2)
    return dest, prov


def _baseline_expr(dataset: str, t: pd.DataFrame) -> np.ndarray:
    """Mean log1p-CPM of each target's own transcript among the control cells of its condition."""
    import anndata as ad

    from tcell_pipeline.replication.de_amendment2 import CONTROL_LABELS, log1p_cpm

    raw = f"data/raw/scperturb/{dataset}.h5ad"
    if not os.path.exists(raw):
        return np.full(len(t), np.nan)
    a = ad.read_h5ad(raw)
    o = a.obs
    ctrl_src = o["perturbation"] if "perturbation" in o.columns else o.iloc[:, 0]
    is_ctrl = ctrl_src.astype(str).str.strip().str.lower().isin(CONTROL_LABELS).to_numpy()
    gi = {str(g): i for i, g in enumerate(a.var.index)}
    X = log1p_cpm(a.X)
    del a
    out = np.full(len(t), np.nan)
    if not is_ctrl.any():
        return out
    ctrl_mean = np.asarray(X[np.where(is_ctrl)[0]].mean(axis=0)).ravel()
    for r, g in enumerate(t["target_contrast_gene_name"].astype(str)):
        j = gi.get(g)
        if j is not None:
            out[r] = float(ctrl_mean[j])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    dest, prov = build(a.dataset)
    print(f"[ptable] {prov['dataset']}: {prov['rows']} rows x {prov['cols']} cols -> {dest}")
    print(f"[ptable]   uniprot {prov['mapped_to_uniprot']}/{prov['rows']} | "
          f"degrees nonzero {prov['ppi_degree_nonzero']} | "
          f"baseline {prov['baseline_expr_nonnull']}/{prov['rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
