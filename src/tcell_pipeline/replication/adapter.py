"""scPerturb harmonised ``.h5ad`` -> the per-(perturbation x condition) DE-statistics ``.h5ad``.

The model never sees raw counts. Training reads a DE-statistics matrix with one row per
(perturbation x condition) and one column per gene, carrying the six layers ``log_fc``, ``zscore``,
``p_value``, ``adj_p_value``, ``baseMean``, ``lfcSE`` (see ``config.DE_LAYERS`` and the geometry
assertion at ``de_extraction.py:73``). This module builds that object from a harmonised scPerturb file.

Every methodological choice here is fixed by ``docs/replication-prereg.md`` and its 2026-08-03
amendment, NOT chosen here: pseudobulk by (target, condition, replicate) on summed raw counts, a
moderated t-test against the matched control pseudobulks *within the same condition*, BH adjustment,
a 25-cell floor per pseudobulk, single-gene targets only, and a per-dataset replicate column. Change
the prereg first, or you are choosing a test after seeing its output.

Usage:
    PYTHONPATH=src python -m tcell_pipeline.replication.adapter --dataset frangieh
    PYTHONPATH=src python -m tcell_pipeline.replication.adapter --self-check
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

RAW_ROOT = Path("data/raw/scperturb")
OUT_ROOT = Path("data/intermediate/replication")

MIN_CELLS_PER_PSEUDOBULK = 25   # prereg section 3
CONTROL_LABELS = {"control", "ctrl", "non-targeting", "nontargeting", "nt"}
DE_LAYERS = ("log_fc", "zscore", "p_value", "adj_p_value", "baseMean", "lfcSE")


@dataclass(frozen=True)
class DatasetSpec:
    """One replication dataset. Every field is a decision recorded in the pre-registration."""
    name: str
    filename: str
    target_col: str          # the column holding the perturbed gene
    replicate_col: str       # prereg amendment 2026-08-03 fixes this per dataset
    condition_col: str | None    # None => single-condition; condition_gated degenerates, h2a is primary
    celltype: str
    pinnacle_context: str | None  # None => no matching PINNACLE context; ESM-2-only, logged ablation
    strip_guide_suffix: str | None = None   # e.g. Papalexi's "STAT2g2" -> "STAT2"
    drop_multi_target: bool = True          # Norman's "SET_KLF1" doubles are excluded
    notes: str = ""
    conditions: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, DatasetSpec] = {
    "frangieh": DatasetSpec(
        name="FrangiehIzar2021", filename="FrangiehIzar2021_RNA.h5ad",
        target_col="perturbation", replicate_col="sgRNA", condition_col="perturbation_2",
        celltype="melanocytes", pinnacle_context="melanocyte",
        notes="248 targets x 3 conditions; sgRNA is a WEAKER replicate than a donor or batch, so "
              "read effect size and CI, not the p-value alone (prereg amendment).",
    ),
    "shifrut": DatasetSpec(
        name="ShifrutMarson2018", filename="ShifrutMarson2018.h5ad",
        target_col="target", replicate_col="replicate", condition_col="perturbation_2",
        celltype="primary human T cells", pinnacle_context="cd4-positive helper t cell",
        notes="21 targets: below the 50-family floor, so preliminary/qualitative only (prereg s4).",
    ),
    "datlinger": DatasetSpec(
        name="DatlingerBock2017", filename="DatlingerBock2017.h5ad",
        target_col="target", replicate_col="replicate", condition_col="perturbation_2",
        celltype="Jurkat T cells", pinnacle_context="cd4-positive helper t cell",
        notes="32 targets: below the 50-family floor, preliminary only.",
    ),
    "norman": DatasetSpec(
        name="NormanWeissman2019", filename="NormanWeissman2019_filtered.h5ad",
        target_col="perturbation", replicate_col="gemgroup", condition_col=None,
        celltype="K562 lymphoblasts", pinnacle_context=None,
        notes="single condition -> h2a is primary. NO PINNACLE context exists for lymphoblasts, so "
              "this dataset runs ESM-2-only and is confounded-by-construction on graph quality.",
    ),
    "replogle_rpe1": DatasetSpec(
        name="ReplogleWeissman2022_rpe1", filename="ReplogleWeissman2022_rpe1.h5ad",
        target_col="perturbation", replicate_col="batch", condition_col=None,
        celltype="RPE1 epithelial", pinnacle_context="retinal pigment epithelial cell",
        notes="DROPPED under prereg s3 (measured 2026-08-03). 2,393 targets and the best target "
              "count available, but ~103 cells per target spread over 56 batches is ~1.8 cells per "
              "(target, batch), so only 12 targets clear the 25-cell floor with >=2 replicate "
              "pseudobulks. There is no other replicate axis: guide_id is a dual-guide construct, "
              "~1.1 per target. The prereg says such a dataset is dropped, NOT switched to a "
              "different test, so it is not rescued with random pseudo-replicates.",
    ),
    "papalexi": DatasetSpec(
        name="PapalexiSatija2021", filename="PapalexiSatija2021_eccite_RNA.h5ad",
        target_col="perturbation", replicate_col="hto", condition_col=None,
        celltype="THP-1 monocytes", pinnacle_context="monocyte",
        strip_guide_suffix=r"g\d+$",
        notes="25 targets, single condition; replicate is the hashtag `hto`. There is no "
              "`replicate` column. `hto` looks like it carries a condition (rep2-ctrl vs rep*-tx) "
              "but rep2-ctrl holds 4 cells in total, so it is a replicate axis with 3 usable "
              "levels, not a treatment axis. Stretch candidate only.",
    ),
}


def _is_control(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(CONTROL_LABELS)


def build_pseudobulk(adata, spec: DatasetSpec) -> tuple[np.ndarray, pd.DataFrame]:
    """Sum raw counts within each (target, condition, replicate) cell group.

    Returns (matrix [n_groups x n_genes], group metadata). Groups with fewer than
    MIN_CELLS_PER_PSEUDOBULK cells are dropped and counted, never silently kept: a pseudobulk built
    from a handful of cells is mostly sampling noise and would widen every downstream CI for reasons
    unrelated to the perturbation.
    """
    obs = adata.obs
    tgt = obs[spec.target_col].astype(str)
    if spec.strip_guide_suffix:
        tgt = tgt.str.replace(spec.strip_guide_suffix, "", regex=True)
    # Control identity comes from the harmonised `perturbation` column, NOT from target_col.
    # Datlinger and Shifrut label controls as perturbation == "control" while leaving `target` NaN,
    # so reading controls off target_col drops every one of them and the run dies with "no control
    # pseudobulks" — which reads like a filter working, not a mis-join.
    control = _is_control(obs["perturbation"] if "perturbation" in obs.columns else tgt)
    # A target naming two genes (Norman's "SET_KLF1") is a combinatorial perturbation, not a
    # single-gene target; the split axis and the PPI node lookup are both single-gene.
    multi = tgt.str.contains("_") & ~control
    # a NaN target is only droppable when it is NOT a control (controls legitimately have no target)
    keep = (control | ~tgt.isin(["nan", "None", ""])) & ~(multi if spec.drop_multi_target else False)

    cond = (obs[spec.condition_col].astype(str) if spec.condition_col else
            pd.Series("single", index=obs.index))
    rep = obs[spec.replicate_col].astype(str)

    frame = pd.DataFrame({"target": tgt, "condition": cond, "replicate": rep,
                          "is_control": control}, index=obs.index)[keep]
    # Controls are pooled per (condition, replicate): a control's own "target" label carries no
    # information, and pooling is what gives the reference arm enough replicates to have a variance.
    frame.loc[frame["is_control"], "target"] = "__control__"

    codes, groups = pd.factorize(pd.MultiIndex.from_frame(
        frame[["target", "condition", "replicate"]]))
    counts = np.bincount(codes, minlength=len(groups))
    big = counts >= MIN_CELLS_PER_PSEUDOBULK

    X = adata[frame.index].X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    # one-hot group indicator @ X  ==  per-group column sums, without materialising a dense copy
    ind = sp.csr_matrix((np.ones(len(codes), dtype=np.float32),
                         (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    pb = np.asarray((ind @ X).todense(), dtype=np.float32)

    meta = pd.DataFrame(groups.tolist(), columns=["target", "condition", "replicate"])
    meta["n_cells"] = counts
    meta["is_control"] = meta["target"] == "__control__"
    return pb[big], meta[big].reset_index(drop=True)


def _log_cpm(pb: np.ndarray, prior: float = 1.0) -> np.ndarray:
    lib = pb.sum(1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log2(pb / lib * 1e6 + prior)


def moderated_t(treat: np.ndarray, ctrl: np.ndarray) -> dict[str, np.ndarray]:
    """Two-sample moderated t per gene, variance shrunk toward the across-gene median.

    With 2-3 replicates a raw per-gene variance is wild, and an unmoderated t manufactures both
    spurious hits and spurious nulls. Shrinking each gene's variance halfway to the median (a
    one-parameter stand-in for limma's empirical-Bayes prior, with prior df equal to the residual df)
    is the pre-registered choice; it is stated as an approximation rather than called limma.
    """
    n1, n2 = treat.shape[0], ctrl.shape[0]
    df = n1 + n2 - 2
    m1, m2 = treat.mean(0), ctrl.mean(0)
    lfc = m1 - m2
    ss = ((treat - m1) ** 2).sum(0) + ((ctrl - m2) ** 2).sum(0)
    s2 = ss / df if df > 0 else np.full_like(lfc, np.nan)
    s2_prior = float(np.median(s2[np.isfinite(s2)])) if np.isfinite(s2).any() else 1.0
    s2_mod = (df * s2 + df * s2_prior) / (2 * df) if df > 0 else np.full_like(lfc, s2_prior)
    se = np.sqrt(np.maximum(s2_mod, 1e-12) * (1.0 / n1 + 1.0 / n2))
    t = np.divide(lfc, se, out=np.zeros_like(lfc), where=se > 0)
    df_mod = 2 * df if df > 0 else 1
    p = 2.0 * stats.t.sf(np.abs(t), df=df_mod)
    return {"log_fc": lfc, "zscore": t, "p_value": p, "lfcSE": se}


def _bh(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def build_de_stats(spec: DatasetSpec, raw_root: Path = RAW_ROOT, out_root: Path = OUT_ROOT):
    """h5ad -> DE-stats h5ad. Returns the output path and a provenance dict."""
    import anndata as ad

    path = raw_root / spec.filename
    adata = ad.read_h5ad(path)
    pb, meta = build_pseudobulk(adata, spec)
    genes = np.asarray(adata.var.index, dtype=object)
    del adata

    lcpm = _log_cpm(pb)
    base = lcpm.mean(0)

    rows, layers = [], {k: [] for k in DE_LAYERS}
    skipped = []
    for cond, sub in meta.groupby("condition", sort=True):
        cidx = sub.index[sub["is_control"]].to_numpy()
        if cidx.size < 2:
            skipped.append({"condition": cond, "reason": f"only {cidx.size} control pseudobulks"})
            continue
        ctrl = lcpm[cidx]
        for tgt, tsub in sub[~sub["is_control"]].groupby("target", sort=True):
            tidx = tsub.index.to_numpy()
            if tidx.size < 2:   # prereg s3: <2 replicate pseudobulks -> dropped, not re-tested
                skipped.append({"condition": cond, "target": tgt,
                                "reason": f"only {tidx.size} replicate pseudobulk(s)"})
                continue
            st = moderated_t(lcpm[tidx], ctrl)
            st["adj_p_value"] = _bh(st["p_value"])
            st["baseMean"] = base
            for k in DE_LAYERS:
                layers[k].append(st[k].astype(np.float32))
            rows.append({"target_contrast_gene_name": tgt, "culture_condition": cond,
                         "n_replicates": int(tidx.size), "n_control_replicates": int(cidx.size),
                         "n_cells": int(tsub["n_cells"].sum())})

    if not rows:
        raise RuntimeError(f"{spec.name}: no (target, condition) cell survived the prereg filters")

    obs = pd.DataFrame(rows)
    obs["target_contrast"] = obs["target_contrast_gene_name"]   # symbol space; Ensembl join is Module 0's job
    out = ad.AnnData(X=np.zeros((len(obs), genes.size), dtype=np.float32),
                     obs=obs, var=pd.DataFrame({"gene_name": genes}, index=genes))
    for k in DE_LAYERS:
        out.layers[k] = np.vstack(layers[k])

    prov = {
        "dataset": spec.name, "source_file": str(path),
        "n_rows": int(len(obs)), "n_genes": int(genes.size),
        "n_targets": int(obs["target_contrast_gene_name"].nunique()),
        "conditions": sorted(obs["culture_condition"].unique().tolist()),
        # MISNAMED, and kept that way because renaming it would desync every provenance file
        # already on disk. The value is a COLUMN NAME, not a unit: Shifrut's column is `replicate`
        # and its unit is the donor, while Datlinger's column is ALSO `replicate` for a different
        # unit entirely. The semantic units are pinned in docs/replication-prereg.md's table; read
        # them from there. A check that trusted this key reported the paper's correct table as wrong.
        "replicate_unit": spec.replicate_col,
        "pinnacle_context": spec.pinnacle_context,
        "esm2_only_ablation": spec.pinnacle_context is None,
        "min_cells_per_pseudobulk": MIN_CELLS_PER_PSEUDOBULK,
        "n_skipped": len(skipped), "skipped": skipped[:50],
        "prereg": "docs/replication-prereg.md",
        "notes": spec.notes,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    dest = out_root / f"{spec.name}.DE_stats.h5ad"
    out.write_h5ad(dest)
    (out_root / f"{spec.name}.provenance.json").write_text(json.dumps(prov, indent=2))
    return dest, prov


def _self_check() -> None:
    """Synthetic end-to-end check: two conditions, a planted effect, and the filters actually firing.

    Runs without any downloaded data so it stays runnable on a clean checkout.
    """
    import anndata as ad

    rng = np.random.default_rng(0)
    n_genes = 40
    recs, blocks = [], []
    for cond in ("A", "B"):
        for rep in ("r1", "r2", "r3"):
            for tgt in ("control", "GENE1", "GENE2"):
                n = 30
                mu = np.full(n_genes, 50.0)
                if tgt == "GENE1":          # planted: gene 0 strongly up, only in this target
                    mu[0] = 400.0
                recs += [{"perturbation": tgt, "perturbation_2": cond, "replicate": rep}] * n
                blocks.append(rng.poisson(mu, size=(n, n_genes)))
    # a group that must be dropped by the 25-cell floor
    recs += [{"perturbation": "TINY", "perturbation_2": "A", "replicate": "r1"}] * 5
    blocks.append(rng.poisson(50.0, size=(5, n_genes)))

    obs = pd.DataFrame(recs)
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=sp.csr_matrix(np.vstack(blocks).astype(np.float32)), obs=obs,
                       var=pd.DataFrame(index=[f"G{i}" for i in range(n_genes)]))
    spec = DatasetSpec(name="selfcheck", filename="", target_col="perturbation",
                       replicate_col="replicate", condition_col="perturbation_2",
                       celltype="synthetic", pinnacle_context=None)

    pb, meta = build_pseudobulk(adata, spec)
    assert "TINY" not in set(meta["target"]), "25-cell floor did not drop the small group"
    assert set(meta["condition"]) == {"A", "B"}, meta["condition"].unique()
    assert meta.loc[meta["is_control"], "target"].eq("__control__").all()
    # controls pooled per (condition, replicate): 2 conditions x 3 replicates
    assert int(meta["is_control"].sum()) == 6, meta[meta["is_control"]]

    lcpm = _log_cpm(pb)
    ctrl = lcpm[meta.index[meta["is_control"] & (meta["condition"] == "A")].to_numpy()]
    g1 = lcpm[meta.index[(meta["target"] == "GENE1") & (meta["condition"] == "A")].to_numpy()]
    g2 = lcpm[meta.index[(meta["target"] == "GENE2") & (meta["condition"] == "A")].to_numpy()]
    hit, null = moderated_t(g1, ctrl), moderated_t(g2, ctrl)
    assert hit["log_fc"][0] > 2.0, f"planted effect not recovered: {hit['log_fc'][0]}"
    assert hit["p_value"][0] < 0.01, f"planted effect not significant: {hit['p_value'][0]}"
    assert abs(null["log_fc"][0]) < 1.0, f"unperturbed target moved: {null['log_fc'][0]}"
    # the test must be able to NOT fire: an all-null comparison should not be mostly significant
    assert (null["p_value"] < 0.05).mean() < 0.25, "null comparison is significant everywhere"

    p = np.array([0.001, 0.01, 0.03, 0.5])
    adj = _bh(p)
    assert np.all(adj >= p) and np.all(np.diff(adj) >= -1e-12), adj

    # REGRESSION (2026-08-03, found on real Datlinger data, not by this check as first written):
    # some datasets label controls only in `perturbation` and leave `target` NaN. Reading controls
    # off target_col then silently drops every control, and the run dies claiming no (target,
    # condition) cell survived "the prereg filters" — a mis-join wearing the costume of a filter.
    obs2 = obs.copy()
    obs2["target"] = np.where(obs2["perturbation"].eq("control"), np.nan, obs2["perturbation"])
    adata2 = ad.AnnData(X=adata.X.copy(), obs=obs2, var=adata.var.copy())
    spec2 = DatasetSpec(name="selfcheck_nan_target", filename="", target_col="target",
                        replicate_col="replicate", condition_col="perturbation_2",
                        celltype="synthetic", pinnacle_context=None)
    _, meta2 = build_pseudobulk(adata2, spec2)
    assert int(meta2["is_control"].sum()) == 6, \
        f"controls lost when target is NaN: {int(meta2['is_control'].sum())} control pseudobulks"
    assert {"GENE1", "GENE2"} <= set(meta2["target"]), set(meta2["target"])

    print("[adapter] self-check OK: floor fires, controls pool, planted effect recovered, "
          "null stays null, BH monotone, NaN-target controls survive")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), help="build the DE-stats matrix for one dataset")
    ap.add_argument("--self-check", action="store_true", help="synthetic end-to-end check, no data needed")
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return 0
    if not a.dataset:
        ap.error("pass --dataset NAME or --self-check")
    dest, prov = build_de_stats(DATASETS[a.dataset], a.raw_root, a.out_root)
    print(f"[adapter] {prov['dataset']}: {prov['n_rows']} rows "
          f"({prov['n_targets']} targets x {len(prov['conditions'])} conditions) "
          f"x {prov['n_genes']} genes -> {dest}")
    print(f"[adapter]   replicate unit={prov['replicate_unit']} "
          f"pinnacle={prov['pinnacle_context']} skipped={prov['n_skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
