"""Stage 3b: derive the two q_pre guide-quality scalars for a replication dataset's de_obs.

WHY THIS EXISTS. `QualityEncoder` consumes `n_guides` and `single_guide_estimate` directly - it stacks
them into h_quality with no imputation. The reference DE tables carry both; the Amendment-2 replication
builder does not emit them, so `de_obs.parquet` arrives without them and training dies in the loader.

WHY NOT JUST FILL NaN. NaN propagates straight through the encoder into the loss. Every arm would train
on NaN and the run would produce numbers that mean nothing.

WHY NOT JUST FILL A CONSTANT. A constant is *safe* (identical in all four arms, so it cannot move any
contrast) but it throws away a real covariate the reference screen had. Every one of these datasets
records `guide_id` per cell, so the true count of distinct guides per (target, condition) is derivable
and is what the reference column means. Derive it; fall back to the constant only where a dataset has
no guide column at all, and say so in the provenance.

    PYTHONPATH=src python -m tcell_pipeline.replication.guide_quality --dataset <name>
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

GUIDE_COLS = ("sgRNA", "guide_id", "grna", "protospacer")
# A pooled screen carries a handful of guides per target; the reference screen's own column runs 1-3.
# Some datasets name a per-CELL barcode `guide_id` (Frangieh: 299 distinct values for one target), and
# counting those would put a meaningless covariate in front of the encoder. Reject any candidate whose
# median distinct-count per target exceeds this and try the next one.
MAX_PLAUSIBLE_GUIDES_PER_TARGET = 20
CONTROL_LABELS = {"control", "ctrl", "non-targeting", "nontargeting", "nt", "none", "nan", ""}


def _pick_condition_col(obs: pd.DataFrame, conditions: list[str]) -> str | None:
    """The column whose value set covers the conditions the DE build recorded, or None if single."""
    if len(conditions) <= 1:
        return None
    want = set(map(str, conditions))
    for c in obs.columns:
        if want <= set(obs[c].astype(str).unique()):
            return c
    return None


def build(dataset: str) -> dict:
    import anndata as ad

    from tcell_pipeline import config

    prov = json.load(open(f"data/intermediate/replication/{dataset}.DE_stats_v2.provenance.json"))
    de_obs = pd.read_parquet(config.DE_OBS_PATH)

    a = ad.read_h5ad(f"data/raw/scperturb/{dataset}.h5ad", backed="r")
    obs = a.obs
    tcol = "perturbation" if "perturbation" in obs.columns else obs.columns[0]
    gcol, rejected = None, []
    for c in GUIDE_COLS:
        if c not in obs.columns:
            continue
        med = float(obs.groupby(obs[tcol].astype(str))[c].nunique().median())
        if med > MAX_PLAUSIBLE_GUIDES_PER_TARGET:
            rejected.append(f"{c}(median {med:.0f}/target - looks per-cell)")
            continue
        gcol = c
        break
    ccol = _pick_condition_col(obs, prov.get("conditions", []))

    if gcol is None:
        # No guide column anywhere: fall back to the constant. Recorded, not silent.
        n_guides = pd.Series(1.0, index=de_obs.index)
        source = "constant_1.0_no_usable_guide_column" + (f"; rejected {rejected}" if rejected else "")
    else:
        key = [tcol] + ([ccol] if ccol else [])
        grp = (obs.loc[~obs[tcol].astype(str).str.lower().isin(CONTROL_LABELS), key + [gcol]]
               .astype(str).groupby(key)[gcol].nunique())
        grp = grp.reset_index()
        grp.columns = key + ["n_guides"]
        left_on = ["target_contrast_gene_name"] + (["culture_condition"] if ccol else [])
        merged = de_obs[left_on].astype(str).merge(
            grp.astype({k: str for k in key}), how="left", left_on=left_on, right_on=key)
        n_guides = merged["n_guides"].astype("float64")
        source = (f"distinct {gcol} per ({tcol}{',' + ccol if ccol else ''})"
                  + (f"; rejected {rejected}" if rejected else ""))

    n_guides = n_guides.to_numpy(dtype="float64")
    unmatched = int(np.isnan(n_guides).sum())
    # An unmatched row means the raw label did not join the DE label (guide collapse, symbol drift).
    # Fill with the dataset median so the column has no NaN, and report the count - a large number here
    # means the join key is wrong, not that the dataset is odd.
    if unmatched:
        n_guides = np.where(np.isnan(n_guides), float(np.nanmedian(n_guides)), n_guides)

    de_obs["n_guides"] = n_guides
    de_obs["single_guide_estimate"] = (n_guides == 1).astype("float64")
    config.write_parquet_atomic(de_obs, config.DE_OBS_PATH)
    a.file.close()

    out = {"dataset": dataset, "source": source, "rows": len(de_obs), "unmatched_filled_with_median": unmatched,
           "n_guides_mean": round(float(n_guides.mean()), 4),
           "n_guides_min": float(n_guides.min()), "n_guides_max": float(n_guides.max()),
           "single_guide_frac": round(float((n_guides == 1).mean()), 4)}
    json.dump(out, open(f"data/intermediate/replication/{dataset}.guide_quality.json", "w"), indent=2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    o = build(a.dataset)
    print(f"[guide] {o['dataset']}: n_guides mean {o['n_guides_mean']} "
          f"[{o['n_guides_min']:.0f},{o['n_guides_max']:.0f}], single-guide {o['single_guide_frac']:.1%}, "
          f"unmatched {o['unmatched_filled_with_median']}/{o['rows']} | {o['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
