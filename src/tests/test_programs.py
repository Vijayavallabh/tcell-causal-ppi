"""Module 3 (Program Decoder) tests — fully synthetic, no marts or embedding parquets.

Basis fits run on tiny random matrices; the decoder / EGIPGModel run on the same small synthetic
HeteroData used by test_graph.py, with zero-fallback embedding stores so the suite is dataless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from tcell_pipeline import config
from tcell_pipeline.encoders import PerturbationEncoder
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.graph import TypedGraphEncoder, build_hetero_graph
from tcell_pipeline.model import EGIPGModel
from tcell_pipeline.programs import (
    ProgramDecoder,
    fit_program_basis,
    train_row_indices,
)

_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)


# ---- ProgramBasis (fold-local sklearn factorisation) ----

@pytest.mark.parametrize("method", ["sparse_pca", "svd", "nmf", "fastica"])
def test_basis_fit_shapes(method):
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((100, 50)).astype("float32")
    B, A = fit_program_basis(Z, method=method, K=8, max_iter=5)
    assert B.shape == (50, 8)   # (G, K) gene-program loadings
    assert A.shape == (100, 8)  # (N, K) program scores
    assert np.isfinite(B).all() and np.isfinite(A).all()


def test_fit_rejects_non_2d():
    with pytest.raises(ValueError):
        fit_program_basis(np.zeros(10, dtype="float32"), method="svd", K=2)


def test_fold_local_selects_train_rows_only():
    split = pd.DataFrame({
        "hgnc_symbol": [f"g{i}" for i in range(10)],
        "role": ["train"] * 5 + ["val"] * 2 + ["challenge"] * 3,
    })
    pc = pd.DataFrame({"row_index": np.arange(20), "hgnc_symbol": [f"g{i % 10}" for i in range(20)]})
    rows = train_row_indices(split, pc)
    train_genes = {f"g{i}" for i in range(5)}
    picked_genes = set(pc.set_index("row_index").loc[rows, "hgnc_symbol"])
    assert picked_genes <= train_genes                       # never a val/cal/challenge row
    assert set(rows) == set(pc.loc[pc.hgnc_symbol.isin(train_genes), "row_index"])


# ---- ProgramDecoder ----

def _decoder(G=50, K=8):
    torch.manual_seed(0)
    B = torch.randn(G, K)
    return ProgramDecoder(B), B


def test_decoder_output_shapes():
    dec, _ = _decoder()
    h_graph = torch.randn(4, config.GRAPH_HIDDEN_DIM)
    h_do = torch.randn(4, config.H_DO_DIM)
    out = dec(h_do, h_graph)
    assert out["delta_z"].shape == (4, 8)
    assert out["delta_x"].shape == (4, 50)
    assert out["sigma"].shape == (4, 8)
    assert out["lambda"].shape == (4, 1)


def test_lambda_in_unit_interval_and_sigma_positive():
    dec, _ = _decoder()
    out = dec(torch.randn(6, config.H_DO_DIM), torch.randn(6, config.GRAPH_HIDDEN_DIM))
    assert (out["lambda"] >= 0).all() and (out["lambda"] <= 1).all()
    assert (out["sigma"] > 0).all()


def test_gene_decode_uses_frozen_basis_buffer():
    dec, B = _decoder()
    names_p = dict(dec.named_parameters())
    names_b = dict(dec.named_buffers())
    assert "program_basis" in names_b and "program_basis" not in names_p  # B is a buffer, not trainable
    assert "program_basis" not in dec.state_dict()  # non-persistent: never rides in a checkpoint
    h_graph = torch.randn(3, config.GRAPH_HIDDEN_DIM)
    h_do = torch.randn(3, config.H_DO_DIM)
    out = dec(h_do, h_graph)
    expected = out["delta_z"] @ B.T + dec.residual(h_graph)  # delta_x = B @ delta_z^T + r
    assert torch.allclose(out["delta_x"], expected, atol=1e-5)


def test_expression_only_variant_pins_lambda_zero():
    dec, B = _decoder()
    out = dec(torch.randn(4, config.H_DO_DIM), h_graph=None)
    assert (out["lambda"] == 0).all()
    assert out["delta_x"].shape == (4, 50) and torch.isfinite(out["delta_x"]).all()
    # no graph -> no graph-residual intercept: delta_x is exactly the program decode (clean §10.6 ablation)
    assert torch.allclose(out["delta_x"], out["delta_z"] @ B.T, atol=1e-5)


# ---- EGIPGModel (M1 + M2 + M3) on the synthetic graph ----

def _edge(src, dst, source, score, phys=0, func=0, cplx=0):
    return dict(source_gene=src, target_gene=dst, source=source, evidence_type="x", score=score,
                is_physical=phys, is_functional=func, is_complex=cplx, is_direct_binary=0,
                n_supporting_sources=1)


def _synthetic_graph():
    edges = pd.DataFrame([
        _edge("A", "B", "biogrid", 0.9, phys=1), _edge("B", "C", "biogrid", 0.9, phys=1),
        _edge("C", "D", "biogrid", 0.9, phys=1), _edge("A", "E", "string", 0.5, func=1),
        _edge("B", "F", "corum", 0.8, cplx=1),
    ])
    complexes = pd.DataFrame([
        dict(protein_gene="A", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="B", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
    ])
    id_map = pd.DataFrame([dict(hgnc_symbol="A", uniprot_id="P1"), dict(hgnc_symbol="B", uniprot_id="P2")])
    baseline = pd.DataFrame([dict(hgnc_symbol="A", control_baseline_expr=1.0),
                             dict(hgnc_symbol="B", control_baseline_expr=0.5)])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def _batch(n=4):
    return {
        "uniprot_id": ["P1", "P2", None, None][:n],
        "ppi_degree_physical": torch.arange(n).float(),
        "ppi_degree_functional": torch.arange(n).float(),
        "ppi_degree_complex": torch.zeros(n),
        "control_baseline_expr": torch.ones(n),
        "culture_condition": ["Rest", "Stim8hr", "Stim48hr", "Rest"][:n],
        "donor_pc": torch.randn(n, config.DONOR_PCA_DIMS),
        "n_guides": torch.ones(n),
        "single_guide_estimate": torch.zeros(n, dtype=torch.bool),
    }


def _model(G=50, K=8, with_graph=True):
    torch.manual_seed(0)
    enc = PerturbationEncoder(_ZERO_PLM, _ZERO_PIN)
    graph, gene_to_idx = _synthetic_graph()
    genc = TypedGraphEncoder(graph, gene_to_idx) if with_graph else None
    return EGIPGModel(torch.randn(G, K), perturbation_encoder=enc, graph_encoder=genc)


def test_full_model_forward_all_keys():
    model = _model().eval()
    with torch.no_grad():
        out = model(_batch(4), ["A", "B", "C", "D"], ["Rest", "Stim8hr", "Stim48hr", "Rest"])
    assert set(out) >= {"delta_z", "delta_x", "sigma", "lambda", "edge_gates", "h_graph", "h_do"}
    assert out["delta_z"].shape == (4, 8) and out["delta_x"].shape == (4, 50)
    assert out["sigma"].shape == (4, 8) and out["lambda"].shape == (4, 1)
    assert out["h_do"].shape == (4, config.H_DO_DIM) and out["h_graph"].shape == (4, config.GRAPH_HIDDEN_DIM)
    for k in ("delta_z", "delta_x", "sigma", "lambda", "h_do", "h_graph"):
        assert torch.isfinite(out[k]).all()


def test_full_model_expression_only():
    model = _model(with_graph=False).eval()
    with torch.no_grad():
        out = model(_batch(4), ["A", "B", "C", "D"], ["Rest", "Stim8hr", "Stim48hr", "Rest"])
    assert out["h_graph"] is None and out["edge_gates"] is None
    assert (out["lambda"] == 0).all()
    assert out["delta_x"].shape == (4, 50) and torch.isfinite(out["delta_x"]).all()
