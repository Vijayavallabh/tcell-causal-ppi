"""Module 8 (Predictive-Rationale Audit, feat-012) tests — fully synthetic (tiny marts + a small typed PPI
graph). Covers the stratified audit end to end: it produces per-case sufficiency/necessity vs matched-random,
a minimality curve, structural-OOD, per-source ablation (BioPlex/HuRI/STRING/CORUM), a GInX-by-sparsity
comparison, and stability; aggregates them; writes audit_report.json; and rejects an expression-only model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

torch.set_num_threads(1)

from tcell_pipeline import config
from tcell_pipeline.encoders import PerturbationEncoder
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.graph import TypedGraphEncoder, build_hetero_graph, sample_subgraph
from tcell_pipeline.model import EGIPGModel
from tcell_pipeline.rationale import RationaleHead, audit_rationale
from tcell_pipeline.rationale.rationale_audit import _SOURCE_INDEX, _source_keep_mask
from tcell_pipeline.training import PerturbationDataset

_G, _K = 6, 3
_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)
_ENC = lambda: PerturbationEncoder(_ZERO_PLM, _ZERO_PIN)


def _edge(src, dst, source, score, phys=0, func=0, cplx=0, direct=0, nsup=1):
    return dict(source_gene=src, target_gene=dst, source=source, evidence_type="x", score=score,
                is_physical=phys, is_functional=func, is_complex=cplx, is_direct_binary=direct,
                n_supporting_sources=nsup)


def _graph():
    edges = pd.DataFrame([
        _edge("G0", "G1", "biogrid", 0.9, phys=1), _edge("G1", "G2", "string", 0.8, func=1),
        _edge("G0", "G2", "string", 0.6, func=1), _edge("G2", "G3", "string", 0.5, func=1),
        _edge("G3", "G4", "biogrid", 0.7, phys=1), _edge("G4", "G5", "corum", 0.6, cplx=1),
    ])
    complexes = pd.DataFrame([
        dict(protein_gene="G0", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="G1", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
    ])
    id_map = pd.DataFrame([dict(hgnc_symbol="G0", uniprot_id="P0"), dict(hgnc_symbol="G1", uniprot_id="P1")])
    baseline = pd.DataFrame([dict(hgnc_symbol="G0", control_baseline_expr=1.0)])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def _write_marts(tmp_path) -> dict:
    genes = [f"G{i}" for i in range(_G)]
    rows = [(0, "G0"), (1, "G1"), (2, "G0"), (3, "G1"), (4, "G3")]  # 0-3 train (G0/G1), 4 val
    n = len(rows)
    rng = np.random.default_rng(0)
    f32 = lambda v: np.full(n, v, np.float32)
    pc = pd.DataFrame({
        "row_index": [r[0] for r in rows], "hgnc_symbol": [r[1] for r in rows],
        "culture_condition": ["Rest", "Stim8hr", "Rest", "Stim48hr", "Rest"],
        "uniprot_id": [f"P{i}" for i in range(n)],
        "ppi_degree_physical": f32(1.0), "ppi_degree_functional": rng.random(n).astype("float32"),
        "ppi_degree_complex": f32(1.0), "control_baseline_expr": f32(0.5),
        **{f"donor_pc_{i:02d}": rng.random(n).astype("float32") for i in range(config.DONOR_PCA_DIMS)},
    })
    obs = pd.DataFrame({"n_guides": np.full(n, 2), "single_guide_estimate": np.zeros(n, bool)})
    B = rng.standard_normal((_G, _K)).astype("float32")
    loadings = pd.DataFrame(B, columns=[f"program_{k}" for k in range(_K)])
    loadings.insert(0, "gene_name", genes)
    z = rng.standard_normal((n, _G)).astype("float32")
    split = pd.DataFrame({"hgnc_symbol": ["G0", "G1", "G3"], "role": ["train", "train", "val"]})
    donor_cols = [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(config.DONOR_PCA_DIMS)]
    prof = [dict(donor_id=f"CE{d}", culture_condition=c,
                 **{cc: float(v) for cc, v in zip(donor_cols, rng.random(config.DONOR_PCA_DIMS))})
            for c in config.CONDITIONS for d in range(3)]
    p = {"split_path": tmp_path / "split.csv", "pc_path": tmp_path / "pc.parquet",
         "obs_path": tmp_path / "obs.parquet", "var_path": tmp_path / "var.parquet",
         "basis_path": tmp_path / "loadings.parquet", "zscore_npz": tmp_path / "zscore.npz",
         "donor_profiles_path": tmp_path / "donor_profiles.parquet"}
    split.to_csv(p["split_path"], index=False)
    pc.to_parquet(p["pc_path"], index=False)
    obs.to_parquet(p["obs_path"], index=False)
    pd.DataFrame({"gene_name": genes}).to_parquet(p["var_path"], index=False)
    loadings.to_parquet(p["basis_path"], index=False)
    sp.save_npz(p["zscore_npz"], sp.csr_matrix(z))
    pd.DataFrame(prof).to_parquet(p["donor_profiles_path"], index=False)
    return p


def _model(tmp_path, graph, g2i, paths, *, with_graph=True):
    genc = TypedGraphEncoder(graph, g2i) if with_graph else None
    return EGIPGModel.from_saved_basis(
        pd.read_parquet(paths["var_path"])["gene_name"].tolist(), path=paths["basis_path"],
        perturbation_encoder=_ENC(), graph_encoder=genc).eval()


def test_audit_rationale_runs_and_aggregates(tmp_path):
    paths = _write_marts(tmp_path)
    graph, g2i = _graph()
    model = _model(tmp_path, graph, g2i, paths)
    ds = PerturbationDataset("train", **paths)
    out = tmp_path / "audit.json"
    report = audit_rationale(model, RationaleHead().eval(), ds, n_cases=3, n_controls=5,
                             sparsities=(0.3, 0.6), out_path=out)

    assert report["n_audited"] >= 1
    agg = report["aggregate"]
    for key in ("frac_sufficiency_below_random", "frac_necessity_above_random", "mean_minimality",
                "mean_stability", "source_ablation_delta_sufficiency", "ginx_by_sparsity"):
        assert key in agg
    assert set(agg["source_ablation_delta_sufficiency"]) == {"bioplex", "huri", "string", "corum"}

    case = report["cases"][0]
    for key in ("sufficiency", "random_sufficiency", "necessity", "random_necessity", "minimality_curve",
                "structural_ood", "source_ablation", "ginx", "stability"):
        assert key in case
    assert np.isfinite(case["sufficiency"]) and np.isfinite(case["necessity"])
    assert set(case["source_ablation"]) == {"bioplex", "huri", "string", "corum"}
    assert 0.0 <= case["stability"] <= 1.0
    assert json.loads(out.read_text())["n_audited"] == report["n_audited"]   # report written to disk


def test_corum_ablation_reaches_complex_membership_edges():
    graph, g2i = _graph()
    sub = sample_subgraph(graph, "G0", gene_to_idx=g2i)          # G0 sits in a CORUM complex -> membership edges
    corum = _source_keep_mask(sub, _SOURCE_INDEX["corum"])
    assert "complex_membership" in corum                         # CORUM ablation reaches the membership edges...
    assert int(corum["complex_membership"].sum()) == 0          # ...and removes all of them (membership is 100% CORUM)
    string = _source_keep_mask(sub, _SOURCE_INDEX["string"])     # STRING is not a membership source
    assert "complex_membership" not in string or int(string["complex_membership"].min()) == 1  # membership kept


def test_stability_is_reproducible_from_the_audit_seed(tmp_path):
    # PyG's dropout_edge draws from the GLOBAL torch RNG, which the audit seed does not otherwise touch, so
    # mean_stability must not depend on ambient process RNG state. top_k=2 + a non-zero scorer makes the
    # selection read node states, so DropEdge genuinely perturbs it (a zero-init head would be trivially 1.0).
    paths = _write_marts(tmp_path)
    graph, g2i = _graph()
    model = _model(tmp_path, graph, g2i, paths)
    ds = PerturbationDataset("train", **paths)
    torch.manual_seed(0)
    head = RationaleHead(top_k=2).eval()
    torch.nn.init.normal_(head.score.weight, std=1.0)     # built ONCE, so only the ambient RNG varies below

    vals = []
    for ambient in (11, 22, 33):
        torch.manual_seed(ambient)
        rep = audit_rationale(model, head, ds, n_cases=2, n_controls=2, sparsities=(0.5,), seed=0,
                              out_path=tmp_path / f"a{ambient}.json")
        vals.append(rep["aggregate"]["mean_stability"])
    assert len(set(vals)) == 1, f"stability not reproducible from the audit seed: {vals}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_audit_runs_on_cuda(tmp_path):
    # the head's scorer consumes the encoder's node states; if it is left on CPU while the encoder runs on
    # cuda, the first case dies with a device mismatch and no report is ever written
    paths = _write_marts(tmp_path)
    graph, g2i = _graph()
    model = _model(tmp_path, graph, g2i, paths)
    ds = PerturbationDataset("train", **paths)
    rep = audit_rationale(model, RationaleHead().eval(), ds, n_cases=1, n_controls=2, sparsities=(0.5,),
                          device="cuda", out_path=tmp_path / "cuda.json")
    assert rep["n_audited"] == 1 and np.isfinite(rep["cases"][0]["sufficiency"])


def test_audit_rejects_expression_only_model(tmp_path):
    paths = _write_marts(tmp_path)
    graph, g2i = _graph()
    model = _model(tmp_path, graph, g2i, paths, with_graph=False)   # no graph encoder -> no rationale
    ds = PerturbationDataset("train", **paths)
    with pytest.raises(ValueError, match="graph model"):
        audit_rationale(model, RationaleHead().eval(), ds, n_cases=2, out_path=tmp_path / "a.json")
