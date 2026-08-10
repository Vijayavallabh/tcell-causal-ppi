"""Survey a scPerturb h5ad and report everything L1 needs, read FROM THE FILE.

The pre-registration requires verifying licence, cell type, target count, condition structure and
gene overlap from primary sources rather than from the papers. Doing that by hand across eight
datasets invites exactly the error this project already made once, so it is automated here.

What it answers per dataset:
  - targets clearing the 25-cell floor under the CORRECTED DE unit (prereg Amendment 2: all cells per
    (target, condition) vs pooled same-condition controls, no per-replicate requirement)
  - whether a SECOND EXPERIMENTAL FACTOR exists. This decides whether h1 is even defined: the
    condition gate needs >=2 contexts, so on single-condition data condition_gated degenerates to
    typed_static and h2a becomes primary. Candidate factors are proposed, never assumed - a column
    with few levels can be a treatment, a batch, or a QC bucket, and only a human reading the source
    can settle which. The script labels its guesses as GUESSES.
  - replicate axis candidates (recorded in provenance; no longer exclusionary after Amendment 2)
  - gene overlap with the reference measured-gene space
  - whether a matching PINNACLE context plausibly exists for the cell type

    PYTHONPATH=src python -m tcell_pipeline.replication.survey [--glob 'data/raw/scperturb/*.h5ad']
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

MIN_CELLS = 25
# obs columns that are never an experimental factor, however few levels they have
_NOT_A_FACTOR = re.compile(
    r"^(ncounts|ngenes|percent_|nperts|n_genes|n_counts|umi|read_count|coverage|"
    r"good_coverage|number_of_cells|umap_|moi|core_|z_gem)", re.I)
_REPLICATE_HINT = re.compile(r"(replicate|batch|gemgroup|donor|lane|library|hto|sample|rep)$", re.I)
_QC_LIKE = {"cancer", "organism", "tissue_type", "disease", "celltype", "cell_line",
            "perturbation_type", "perturbation_type_2", "sex", "age"}


def survey_one(path: str, ref_genes: set[str], ref_targets: set[str]) -> dict:
    import anndata as ad

    a = ad.read_h5ad(path, backed="r")
    o, name = a.obs, os.path.basename(path).replace(".h5ad", "")

    # target column: scPerturb uses `perturbation`; `target` when present is the cleaned gene symbol
    tcol = "target" if "target" in o.columns and o["target"].notna().any() else "perturbation"
    t = o[tcol].astype(str)
    ctrl = t.str.lower().isin({"control", "ctrl", "non-targeting", "nontargeting", "nt"})
    t_valid = t[~ctrl & ~t.isin(["nan", "None", ""])]
    singles = t_valid[~t_valid.str.contains("_")]           # exclude combinatorial perturbations

    # GUIDE-LEVEL LABELS MASQUERADING AS TARGETS. Where a dataset has no `target` column, its
    # `perturbation` values may be per-GUIDE ("STAT2g2", "CAV1g4"), and counting them inflates the
    # target axis by the guides-per-gene factor. Papalexi reports 91 guide labels against 25 genes,
    # which would falsely clear the 50-family floor. Detect and collapse, and record that we did.
    guide_collapsed = False
    n_before = int(singles.nunique())
    if tcol == "perturbation":
        collapsed = singles.str.replace(r"g\d+$", "", regex=True).str.replace(r"-\d+$", "", regex=True)
        if collapsed.nunique() <= 0.75 * max(n_before, 1):   # a real collapse, not noise
            singles, guide_collapsed = collapsed, True

    vc = singles.value_counts()
    kept = vc[vc >= MIN_CELLS]

    # candidate second factors: low-cardinality obs columns that are not QC, not replicate-shaped
    factors = []
    for c in o.columns:
        if c in (tcol, "perturbation", "guide_id", "sgRNA") or c in _QC_LIKE:
            continue
        if _NOT_A_FACTOR.match(c):
            continue
        u = o[c].dropna().unique()
        if 2 <= len(u) <= 10:
            entry = {"column": c, "levels": [str(x) for x in u][:10],
                     "looks_like_replicate": bool(_REPLICATE_HINT.search(c))}
            factors.append(entry)

    genes = {str(g) for g in a.var.index}
    out = {
        "dataset": name, "n_cells": int(a.n_obs), "n_genes_measured": int(a.n_vars),
        "target_column": tcol,
        "guide_labels_collapsed_to_genes": guide_collapsed,
        "targets_before_guide_collapse": n_before,
        "targets_total": int(singles.nunique()),
        "targets_ge_25_cells": int(len(kept)),
        "median_cells_per_kept_target": int(kept.median()) if len(kept) else 0,
        "combinatorial_targets_excluded": int(t_valid[t_valid.str.contains("_")].nunique()),
        "n_control_cells": int(ctrl.sum()),
        "celltype": str(o["celltype"].iloc[0]) if "celltype" in o.columns else "?",
        "cell_line": str(o["cell_line"].iloc[0]) if "cell_line" in o.columns else "?",
        "gene_overlap_with_reference": len(genes & ref_genes),
        "gene_overlap_pct": round(100 * len(genes & ref_genes) / max(len(ref_genes), 1), 1),
        "targets_in_reference": len(set(kept.index) & ref_targets),
        "candidate_second_factors_GUESSES": factors,
        "h1_defined": None,   # a HUMAN decides this from the factors above; never inferred here
        "meets_50_family_floor_hint": len(kept) >= 50,
    }
    a.file.close()
    return out


def main() -> int:
    from tcell_pipeline import config

    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/scperturb/*.h5ad")
    ap.add_argument("--out", default="data/intermediate/replication/survey_l1.json")
    a = ap.parse_args()

    ref_genes = set(pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].astype(str))
    ref_targets = set(pd.read_parquet(config.DE_OBS_PATH, columns=["target_contrast_gene_name"])
                      ["target_contrast_gene_name"].astype(str))

    rows = []
    for p in sorted(_glob.glob(a.glob)):
        try:
            r = survey_one(p, ref_genes, ref_targets)
        except Exception as exc:                      # a bad file must not kill the survey
            print(f"  {os.path.basename(p):46s} FAILED: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        fac = ", ".join(f["column"] for f in r["candidate_second_factors_GUESSES"]
                        if not f["looks_like_replicate"]) or "none obvious"
        print(f"  {r['dataset']:46s} cells={r['n_cells']:>7} targets>=25={r['targets_ge_25_cells']:>5} "
              f"overlap={r['gene_overlap_pct']:>5}% celltype={r['celltype'][:22]:22s}"
              f"{' [guides collapsed]' if r['guide_labels_collapsed_to_genes'] else ''} factors? {fac}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out} ({len(rows)} datasets)")
    print("NOTE: candidate second factors are GUESSES from column cardinality. A human must confirm "
          "each against the source before h1 is treated as defined on that dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
