"""Module 2 (Typed Graph Encoder) tests on a small synthetic graph.

Kept fully synthetic (no marts, no embedding parquets): a PluggableEmbeddingStore pointed at a
non-existent path returns the zero fallback, so node features are deterministic and the whole
suite runs on a dataless checkout.
"""
from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.graph import (
    COMPLEX,
    PROTEIN,
    TypedGraphEncoder,
    build_hetero_graph,
    sample_subgraph,
    signed_message,
)
from tcell_pipeline.graph.typed_graph_encoder import _MEMBERSHIP

_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "does_not_exist.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "does_not_exist.parquet", config.PINNACLE_EMBED_DIM)


def _edge(src, dst, source, score, phys=0, func=0, cplx=0, direct=0, nsup=1):
    return dict(source_gene=src, target_gene=dst, source=source, evidence_type="x", score=score,
                is_physical=phys, is_functional=func, is_complex=cplx, is_direct_binary=direct,
                n_supporting_sources=nsup)


def _frames():
    edges = pd.DataFrame([
        _edge("A", "B", "biogrid", 0.9, phys=1),
        _edge("B", "C", "biogrid", 0.9, phys=1),
        _edge("C", "D", "biogrid", 0.9, phys=1),
        _edge("A", "E", "string", 0.5, func=1),
        _edge("B", "F", "corum", 0.8, cplx=1),
        _edge("SOLO", "SOLO", "biogrid", 0.7, phys=1),  # isolated self-loop node
    ])
    complexes = pd.DataFrame([
        dict(protein_gene="A", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="B", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="C", complex_id=2, source_database="CORUM", confidence=0.7, is_curated=0),
    ])
    id_map = pd.DataFrame([
        dict(hgnc_symbol="A", uniprot_id="P0001"),
        dict(hgnc_symbol="B", uniprot_id="P0002"),
    ])
    baseline = pd.DataFrame([
        dict(hgnc_symbol="A", control_baseline_expr=1.0),
        dict(hgnc_symbol="B", control_baseline_expr=float("nan")),  # NaN must not poison features
    ])
    return edges, complexes, id_map, baseline


def _graph():
    return build_hetero_graph(*_frames(), plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def test_graph_structure():
    graph, gene_to_idx = _graph()
    assert set(graph.node_types) == {PROTEIN, COMPLEX}
    for rel in ("physical_ppi", "co_complex", "functional_assoc"):
        assert (PROTEIN, rel, PROTEIN) in graph.edge_types
    assert (PROTEIN, "complex_membership", COMPLEX) in graph.edge_types
    assert set(gene_to_idx) == {"A", "B", "C", "D", "E", "F", "SOLO"}
    assert graph[PROTEIN].x.shape == (7, config.PROTEIN_FEATURE_DIM)  # PLM+PINNACLE+3+1 = 1412
    assert graph[COMPLEX].num_nodes == 2
    assert graph[PROTEIN, "physical_ppi", PROTEIN].edge_attr.shape[1] == config.EDGE_FEATURE_DIM
    assert torch.isfinite(graph[PROTEIN].x).all()  # NaN baseline neutralised


def test_two_hop_cap_respected():
    graph, gene_to_idx = _graph()
    sub = sample_subgraph(graph, "A", hops=2, cap=3)
    assert sub[PROTEIN].x.shape[0] <= 3
    assert gene_to_idx["A"] in sub[PROTEIN].orig_idx.tolist()  # seed always kept


def test_condition_gate_differs():
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    h_do = torch.randn(config.GRAPH_HIDDEN_DIM)
    g_rest = enc.encode_one("A", "Rest", h_do)[1]
    g_stim = enc.encode_one("A", "Stim48hr", h_do)[1]
    assert g_rest["physical_ppi"].numel() > 0
    # same edges, different condition -> different gate values
    assert not torch.allclose(g_rest["physical_ppi"], g_stim["physical_ppi"])


def test_signed_message_has_tanh_and_relu():
    torch.manual_seed(0)
    w_sign, w_mag = nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)
    assert torch.allclose(signed_message(torch.zeros(2, 4), w_sign, w_mag), torch.zeros(2, 4))
    h = torch.randn(5, 4)
    out = signed_message(h, w_sign, w_mag)
    expected = torch.tanh(w_sign(h)) * torch.relu(w_mag(h))
    assert torch.allclose(out, expected)  # exact composition is the real invariant
    # sign in [-1, 1] (tanh) times a non-negative magnitude (relu); magnitude is unbounded, so |out|
    # is NOT bounded by 1 — assert only the relu non-negativity, not a false < 1 bound.
    assert (torch.relu(w_mag(h)) >= 0).all()


def test_forward_shape_no_nan():
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    h_do = torch.randn(4, config.GRAPH_HIDDEN_DIM)
    h_graph, _, _ = enc(["A", "B", "C", "D"], ["Rest", "Stim8hr", "Stim48hr", "Rest"], h_do)
    assert h_graph.shape == (4, config.GRAPH_HIDDEN_DIM)
    assert torch.isfinite(h_graph).all()


def test_edge_gates_returned_per_type():
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    _, gates, confs = enc(["A", "B"], ["Rest", "Stim8hr"], torch.randn(2, config.GRAPH_HIDDEN_DIM))
    assert set(gates) == {"physical_ppi", "co_complex", "functional_assoc", _MEMBERSHIP}
    assert set(confs) == set(gates)                       # per-edge source confidence, same relations
    for rel in gates:
        assert len(gates[rel]) == 2 and len(confs[rel]) == 2  # one per-edge tensor per batch sample
        assert all(torch.isfinite(g).all() and (g >= 0).all() and (g <= 1).all() for g in gates[rel])
        for g, c in zip(gates[rel], confs[rel]):
            assert c.shape == g.shape                     # confidence aligned per edge to the gate
            assert (c >= 0).all() and (c <= 1).all()      # score column is clipped to [0,1]


def test_zero_degree_target_works():
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    h_do = torch.randn(2, config.GRAPH_HIDDEN_DIM)
    # SOLO has no protein neighbours (self-loop only); NOTAGENE is absent from the PPI graph.
    h_graph, gates, _ = enc(["SOLO", "NOTAGENE"], ["Rest", "Rest"], h_do)
    assert h_graph.shape == (2, config.GRAPH_HIDDEN_DIM)
    assert torch.isfinite(h_graph).all()
    assert torch.allclose(h_graph[1], torch.zeros(config.GRAPH_HIDDEN_DIM))  # absent target -> zero
    assert gates["functional_assoc"][0].numel() == 0  # SOLO has no functional edges


def test_readout_attention_sums_to_one():
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    _, _, attn = enc.encode_one("A", "Rest", torch.randn(config.GRAPH_HIDDEN_DIM))
    assert torch.allclose(attn.sum(), torch.tensor(1.0), atol=1e-5)


def test_oov_condition_raises():
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    with pytest.raises(ValueError):
        enc.encode_one("A", "NotACondition", torch.randn(config.GRAPH_HIDDEN_DIM))


def test_edge_gates_one_per_original_edge():
    # regression: PP gates are length E (one per original edge), not the 2E symmetrised MP count
    graph, gene_to_idx = _graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    sub = sample_subgraph(graph, "A", gene_to_idx=gene_to_idx)  # sampler is deterministic
    _, gates, _ = enc.encode_one("A", "Rest", torch.randn(config.GRAPH_HIDDEN_DIM))
    for rel in ("physical_ppi", "co_complex", "functional_assoc"):
        assert gates[rel].numel() == sub[PROTEIN, rel, PROTEIN].edge_index.shape[1]
