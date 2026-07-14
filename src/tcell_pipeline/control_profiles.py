"""Non-targeting-control (NTC) baselines and donor profiles from pseudobulk.

The pseudobulk CSR (278684 x 18129) is the ONLY source of non-targeting controls
(DE has none). NTC rows are grouped by (donor_id, culture_condition) into mean
expression; a 32-dim IncrementalPCA is fit on NTC rows only (prediction-time context,
not response-derived, so no fold-local fit is needed) and used to embed each donor x
condition. Both are merged back into perturbation_condition as q_pre features.
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
    is_ntc_name = perturbed_gene_name.astype(str).apply(lambda s: bool(_NTC_RE.search(s)))
    return (is_nt | is_ntc_name).to_numpy()


def _pc_columns(n: int) -> list[str]:
    return [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(n)]


def pca_embed_groups(group_means: np.ndarray, ntc_rows_iter, n_components: int) -> np.ndarray:
    """Fit IncrementalPCA on chunked NTC rows, return the group-mean embeddings."""
    k = min(n_components, group_means.shape[1], group_means.shape[0])
    ipca = IncrementalPCA(n_components=k)
    for chunk in ntc_rows_iter:
        if chunk.shape[0] >= k:
            ipca.partial_fit(chunk)
    emb = ipca.transform(group_means)
    if emb.shape[1] < n_components:  # pad so the schema always has n_components cols
        emb = np.hstack([emb, np.zeros((emb.shape[0], n_components - emb.shape[1]), dtype=emb.dtype)])
    return emb


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
    n_genes = a.shape[1]
    sums = np.zeros((len(uniq), n_genes), dtype=np.float64)
    counts = np.zeros(len(uniq), dtype=np.int64)

    chunk = 1000
    pos = 0
    for block in _iter_dense_rows(a, ntc_idx, chunk):
        for r in range(block.shape[0]):
            gi = gidx[group_keys[pos + r]]
            sums[gi] += block[r]
            counts[gi] += 1
        pos += block.shape[0]
    group_means = (sums / counts[:, None]).astype(np.float32)

    emb = pca_embed_groups(group_means, _iter_dense_rows(a, ntc_idx, chunk), config.DONOR_PCA_DIMS)

    var_names = np.asarray(a.var_names, dtype=str)
    donors = [g[0] for g in uniq]
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
    gn = pd.Series(["NTC-1", "KNTC1", "A1BG", "NTC_2"])
    m = ntc_mask(gt, gn)
    assert m.tolist() == [True, False, False, True], m  # KNTC1 must NOT match
    gm = rng.random((3, 50)).astype(np.float32)
    chunks = [rng.random((40, 50)).astype(np.float32)]
    emb = pca_embed_groups(gm, iter(chunks), config.DONOR_PCA_DIMS)
    assert emb.shape == (3, config.DONOR_PCA_DIMS), emb.shape
    print("[control_profiles] demo OK")


if __name__ == "__main__":
    _demo()
