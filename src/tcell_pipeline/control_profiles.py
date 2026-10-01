"""Non-targeting-control (NTC) baselines and donor profiles from pseudobulk.

The pseudobulk CSR (278684 x 18129) is the ONLY source of non-targeting controls
(DE has none). NTC rows are grouped by (donor_id, culture_condition) into mean
expression; a 32-dim IncrementalPCA is fit on NTC rows only (prediction-time context,
not response-derived, so no fold-local fit is needed) and used to embed each donor x
condition. Group means and the PCA fit share a single streaming pass over the backed
matrix. Both outputs are merged back into perturbation_condition as q_pre features.
"""
from __future__ import annotations

import re

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA

from tcell_pipeline import config

# \bNTC\b so the real gene KNTC1 is NOT flagged as a non-targeting control.
_NTC_RE = re.compile(r"\bNTC\b", flags=re.IGNORECASE)


def ntc_mask(guide_type: pd.Series, perturbed_gene_name: pd.Series) -> np.ndarray:
    is_nt = guide_type.astype(str).str.lower() == "non-targeting"
    # NaN perturbed_gene_name (common on NTC rows) is a float on pandas 3, so guard the type.
    is_ntc_name = perturbed_gene_name.map(lambda s: bool(_NTC_RE.search(s)) if isinstance(s, str) else False)
    return (is_nt | is_ntc_name).to_numpy()


def _pc_columns(n: int) -> list[str]:
    return [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(n)]


def accumulate_ntc(blocks, group_codes: np.ndarray, n_groups: int, n_genes: int,
                   n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Single streaming pass: group-mean expression + IncrementalPCA embedding of each group.

    PCA rank is capped by gene count only (the NTC fit-sample count is large); the embedding
    is padded to n_components so the output schema is stable. Returns (group_means, embedding).
    """
    k = min(n_components, n_genes)
    ipca = IncrementalPCA(n_components=k)
    sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)
    fitted = False
    pos = 0
    for block in blocks:
        m = block.shape[0]
        codes = group_codes[pos:pos + m]
        np.add.at(sums, codes, block)
        counts += np.bincount(codes, minlength=n_groups)
        if m >= k:
            ipca.partial_fit(block)
            fitted = True
        pos += m
    group_means = (sums / np.maximum(counts, 1)[:, None]).astype(np.float32)
    emb = ipca.transform(group_means) if fitted else np.zeros((n_groups, k), dtype=np.float32)
    if emb.shape[1] < n_components:
        emb = np.hstack([emb, np.zeros((n_groups, n_components - emb.shape[1]), dtype=emb.dtype)])
    return group_means, emb


def _iter_dense_rows(a: ad.AnnData, row_idx: np.ndarray, chunk: int):
    for start in range(0, len(row_idx), chunk):
        sub = a.X[row_idx[start:start + chunk]]
        yield np.asarray(sub.todense() if hasattr(sub, "todense") else sub, dtype=np.float32)


def run() -> pd.DataFrame:
    print(f"[control_profiles] opening pseudobulk (backed) {config.PSEUDOBULK_PATH}")
    a = ad.read_h5ad(config.PSEUDOBULK_PATH, backed="r")
    obs = a.obs
    mask = ntc_mask(obs["guide_type"], obs["perturbed_gene_name"])
    ntc_idx = np.flatnonzero(mask)
    print(f"[control_profiles]   {len(ntc_idx)} NTC rows of {len(obs)}")

    groups = obs.loc[mask, ["donor_id", "culture_condition"]].astype(str).reset_index(drop=True)
    group_keys = list(map(tuple, groups.to_numpy()))
    uniq = sorted(set(group_keys))
    gidx = {g: i for i, g in enumerate(uniq)}
    group_codes = np.fromiter((gidx[g] for g in group_keys), dtype=np.int64, count=len(group_keys))

    donors = [g[0] for g in uniq]
    if any(not d.startswith("CE") for d in set(donors)):
        print(f"[control_profiles]   WARNING donor_id not all physical CE codes ({sorted(set(donors))}); "
              "check for batch-relative D1-D4 labels")

    n_genes = a.shape[1]
    group_means, emb = accumulate_ntc(
        _iter_dense_rows(a, ntc_idx, 1000), group_codes, len(uniq), n_genes, config.DONOR_PCA_DIMS)

    var_names = np.asarray(a.var_names, dtype=str)
    conds = [g[1] for g in uniq]

    baseline = pd.DataFrame({
        "donor_id": np.repeat(donors, n_genes),
        "culture_condition": np.repeat(conds, n_genes),
        "ensembl_id": np.tile(var_names, len(uniq)),
        "mean_expr": group_means.reshape(-1),
    })
    profiles = pd.DataFrame(emb, columns=_pc_columns(config.DONOR_PCA_DIMS))
    profiles.insert(0, "culture_condition", conds)
    profiles.insert(0, "donor_id", donors)

    config.write_parquet_atomic(baseline, config.CONTROL_BASELINE_PATH)
    config.write_parquet_atomic(profiles, config.CONTROL_DONOR_PROFILES_PATH)
    print(f"[control_profiles] wrote baseline ({len(baseline)} rows) and "
          f"donor profiles ({len(profiles)} groups)")

    _merge_into_perturbation(baseline, profiles)
    return profiles


def _merge_into_perturbation(baseline: pd.DataFrame, profiles: pd.DataFrame) -> None:
    if not config.PERTURBATION_CONDITION_PATH.exists():
        print("[control_profiles] perturbation_condition.parquet absent; skipping merge")
        return
    table = pd.read_parquet(config.PERTURBATION_CONDITION_PATH)

    # target-gene baseline: average over donors within a condition, keyed by target ensembl.
    per_cond = (baseline.groupby(["culture_condition", "ensembl_id"], as_index=False)["mean_expr"]
                .mean().rename(columns={"mean_expr": "control_baseline_expr"}))
    table = table.drop(columns=["control_baseline_expr"], errors="ignore").merge(
        per_cond, on=["culture_condition", "ensembl_id"], how="left")

    pc_cols = _pc_columns(config.DONOR_PCA_DIMS)
    cond_pcs = profiles.groupby("culture_condition", as_index=False)[pc_cols].mean()
    table = table.drop(columns=pc_cols, errors="ignore").merge(cond_pcs, on="culture_condition", how="left")

    config.write_parquet_atomic(table, config.PERTURBATION_CONDITION_PATH)
    filled = int(table["control_baseline_expr"].notna().sum())
    print(f"[control_profiles] merged baseline+PCA into perturbation_condition "
          f"({filled}/{len(table)} rows have a target baseline)")


def _demo() -> None:
    rng = np.random.default_rng(0)
    gt = pd.Series(["non-targeting", "targeting", "targeting", "non-targeting"])
    gn = pd.Series(["NTC-1", "KNTC1", "A1BG", float("nan")])
    m = ntc_mask(gt, gn)
    assert m.tolist() == [True, False, False, True], m  # KNTC1 must NOT match; NaN must not crash

    n_groups, n_genes = 3, 50
    codes = np.array([0, 1, 2] * 20, dtype=np.int64)  # 60 rows
    blocks = [rng.random((40, n_genes)).astype(np.float32), rng.random((20, n_genes)).astype(np.float32)]
    gm, emb = accumulate_ntc(iter(blocks), codes, n_groups, n_genes, config.DONOR_PCA_DIMS)
    assert gm.shape == (n_groups, n_genes) and emb.shape == (n_groups, config.DONOR_PCA_DIMS), (gm.shape, emb.shape)
    assert not np.allclose(emb, 0.0), "embedding should not be all-zero when the PCA is fit"
    print("[control_profiles] demo OK")


if __name__ == "__main__":
    _demo()
