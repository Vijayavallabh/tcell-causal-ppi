"""Module 7 (Graph Baselines, feat-007) tests — fully synthetic (no marts, no embedding parquets).

Covers the three graph references: network-propagation diffusion (shape, decay from multiple sources,
disconnected/absent -> zero, output schema), the untyped homogeneous GCN encoder (shape, no gates, absent
target -> zero, condition-invariant), and the typed static encoder (condition gate pinned to 1.0, and NOT
all-ones for the dynamic typed encoder). One test forwards both neural encoders inside an EGIPGModel and
round-trips the common output schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

torch.set_num_threads(1)  # many-core box: tiny GNN ops thrash the default thread pool otherwise

from tcell_pipeline import config
from tcell_pipeline.baselines import (
    NetworkPropagationBaseline,
    StaticTypedGraphEncoder,
    UntypedGraphEncoder,
)
from tcell_pipeline.encoders import PerturbationEncoder
from tcell_pipeline.encoders.batch import DONOR_COLS, build_encoder_batch
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.evaluation.output_schema import read_predictions, write_predictions
from tcell_pipeline.graph import TypedGraphEncoder, build_hetero_graph
from tcell_pipeline.model import EGIPGModel

_G, _K = 6, 3
_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)


def _edge(src, dst, source, score, phys=0, func=0, cplx=0, direct=0, nsup=1):
    return dict(source_gene=src, target_gene=dst, source=source, evidence_type="x", score=score,
                is_physical=phys, is_functional=func, is_complex=cplx, is_direct_binary=direct,
                n_supporting_sources=nsup)


def _graph():
    edges = pd.DataFrame([
        _edge("G0", "G1", "biogrid", 0.9, phys=1), _edge("G1", "G2", "biogrid", 0.8, phys=1),
        _edge("G2", "G3", "string", 0.5, func=1), _edge("G3", "G4", "biogrid", 0.7, phys=1),
        _edge("G4", "G5", "corum", 0.6, cplx=1),
    ])
    complexes = pd.DataFrame([
        dict(protein_gene="G0", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="G1", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
    ])
    id_map = pd.DataFrame([dict(hgnc_symbol="G0", uniprot_id="P0"), dict(hgnc_symbol="G1", uniprot_id="P1")])
    baseline = pd.DataFrame([dict(hgnc_symbol="G0", control_baseline_expr=1.0)])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def _batch(genes, conditions):
    n = len(genes)
    pc = pd.DataFrame({
        "uniprot_id": [f"P{i}" for i in range(n)],
        "ppi_degree_physical": np.ones(n, "float32"), "ppi_degree_functional": np.ones(n, "float32"),
        "ppi_degree_complex": np.ones(n, "float32"), "control_baseline_expr": np.full(n, 0.5, "float32"),
        "culture_condition": list(conditions),
        **{c: np.zeros(n, "float32") for c in DONOR_COLS},
    })
    obs = pd.DataFrame({"n_guides": np.full(n, 2), "single_guide_estimate": np.zeros(n, bool)})
    return build_encoder_batch(pc, obs)


# --- network propagation -------------------------------------------------------------------------
def test_network_propagation_decays_from_multiple_sources():
    # path A-B-C; strong signal on the two endpoints, midpoint B should mix both
    a = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float))
    g2i = {"A": 0, "B": 1, "C": 2}
    npb = NetworkPropagationBaseline(a, g2i, basis=np.eye(3)).fit(["A", "C"], np.array([[5.0, 0, 0], [0, 0, 5.0]]))
    dz, dx = npb.predict(["A", "B", "C"])
    assert dz.shape == (3, 3) and np.isfinite(dz).all()
    assert dz[0, 0] > dz[1, 0]           # A's signal strongest at A, weaker at the midpoint
    assert dz[2, 2] > dz[1, 2]           # C's signal strongest at C
    assert dz[1, 0] > 0 and dz[1, 2] > 0  # midpoint genuinely blends both sources
    assert np.allclose(dx, dz @ np.eye(3).T)


def test_network_propagation_disconnected_and_absent_are_zero():
    a = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float))  # node 2 isolated
    g2i = {"A": 0, "B": 1, "ISO": 2}
    npb = NetworkPropagationBaseline(a, g2i, basis=np.eye(3)).fit(["A"], np.array([[3.0, 1.0, 0.0]]))
    dz, _ = npb.predict(["ISO", "NOTINGRAPH"])
    assert np.allclose(dz[0], 0)         # no training presence reaches an isolated node
    assert np.allclose(dz[1], 0)         # gene absent from the graph -> zero fallback


def test_network_propagation_from_hetero_graph_shapes():
    graph, g2i = _graph()
    rng = np.random.default_rng(0)
    B = rng.standard_normal((_G, _K))
    npb = NetworkPropagationBaseline.from_hetero_graph(graph, g2i, basis=B).fit(
        ["G0", "G1"], rng.standard_normal((2, _K)))
    dz, dx = npb.predict(["G2", "G3"])
    assert dz.shape == (2, _K) and dx.shape == (2, _G)
    assert np.allclose(dx, dz @ B.T) and np.isfinite(dz).all()


def test_network_propagation_emits_output_schema(tmp_path):
    a = sp.csr_matrix(np.array([[0, 1], [1, 0]], float))
    npb = NetworkPropagationBaseline(a, {"A": 0, "B": 1}, basis=np.eye(2)).fit(["A"], np.array([[1.0, 2.0]]))
    dz, dx = npb.predict(["A", "B"])
    path = write_predictions([0, 1], dz, dx, None, "network_propagation", "val", 0, root=tmp_path / "pred")
    back = read_predictions(path)
    assert back["delta_z"].shape == (2, 2) and np.array_equal(back["row_index"], [0, 1])


# --- untyped homogeneous GCN ---------------------------------------------------------------------
def test_untyped_gnn_forward_shapes_no_gates_and_condition_invariant():
    graph, g2i = _graph()
    enc = UntypedGraphEncoder(graph, g2i).eval()
    hg, gates, confs = enc(["G0", "G1", "NOPE"], ["Rest", "Stim8hr", "Rest"],
                           torch.randn(3, config.GRAPH_HIDDEN_DIM))
    assert hg.shape == (3, config.GRAPH_HIDDEN_DIM) and torch.isfinite(hg).all()
    assert gates is None and confs is None                                   # untyped: no gates/confidences
    assert torch.allclose(hg[2], torch.zeros(config.GRAPH_HIDDEN_DIM))       # gene absent -> zero h_graph
    h_do = torch.randn(1, config.GRAPH_HIDDEN_DIM)
    r, *_ = enc(["G0"], ["Rest"], h_do)
    s, *_ = enc(["G0"], ["Stim48hr"], h_do)
    assert torch.allclose(r, s)                                             # no condition gate -> invariant


# --- typed static (gate pinned to 1.0) -----------------------------------------------------------
def test_typed_static_gates_are_one_and_condition_invariant():
    graph, g2i = _graph()
    static = StaticTypedGraphEncoder(graph, g2i).eval()
    h_do = torch.randn(config.GRAPH_HIDDEN_DIM)
    _, g_rest, _ = static.encode_one("G0", "Rest", h_do)
    _, g_stim, _ = static.encode_one("G0", "Stim48hr", h_do)
    nonempty = [r for r in g_rest if g_rest[r].numel()]
    assert nonempty                                                          # G0 has at least one edge
    for rel in nonempty:
        assert torch.allclose(g_rest[rel], torch.ones_like(g_rest[rel]))     # gate pinned to 1.0
        assert torch.allclose(g_rest[rel], g_stim[rel])                      # identical across conditions
    dyn = TypedGraphEncoder(graph, g2i).eval()
    _, g_dyn, _ = dyn.encode_one("G0", "Rest", h_do)
    assert any(g_dyn[r].numel() and not torch.allclose(g_dyn[r], torch.ones_like(g_dyn[r])) for r in g_dyn)


# --- both neural encoders wired into EGIPGModel emit the common output schema ---------------------
def test_neural_graph_baselines_emit_output_schema(tmp_path):
    graph, g2i = _graph()
    genes, conds = ["G0", "G1"], ["Rest", "Stim8hr"]
    batch = _batch(genes, conds)
    for enc in (UntypedGraphEncoder(graph, g2i), StaticTypedGraphEncoder(graph, g2i)):
        model = EGIPGModel(torch.randn(_G, _K), perturbation_encoder=PerturbationEncoder(_ZERO_PLM, _ZERO_PIN),
                           graph_encoder=enc).eval()
        with torch.no_grad():
            out = model(batch, genes, conds)
        assert out["delta_z"].shape == (2, _K) and out["delta_x"].shape == (2, _G)
        path = write_predictions([0, 1], out["delta_z"], out["delta_x"], out["sigma"],
                                 type(enc).__name__, "val", 0, root=tmp_path / "pred")
        back = read_predictions(path)
        assert back["delta_z"].shape == (2, _K) and np.array_equal(back["row_index"], [0, 1])


def test_untyped_batched_forward_matches_per_sample_loop():
    """The mini-batched GCN must equal the per-target loop (encode_one is the unbatched oracle).
    eval(): the batched path shares no code with encode_one, so this is a real cross-check."""
    torch.manual_seed(0)
    graph, gene_to_idx = _graph()
    enc = UntypedGraphEncoder(graph, gene_to_idx).eval()
    genes = ["G0", "NOTAGENE", "G3", "G5"]  # an absent target between real ones
    h_do = torch.randn(len(genes), config.GRAPH_HIDDEN_DIM)
    ref = torch.stack([
        torch.zeros(enc.hidden) if g not in gene_to_idx else enc.encode_one(g, h_do[b])
        for b, g in enumerate(genes)
    ])
    got, gates, confs = enc(genes, ["Rest"] * len(genes), h_do)
    assert gates is None and confs is None  # untyped: no provenance, no gate -- by design
    assert torch.allclose(got, ref, atol=1e-5)
    assert torch.allclose(got[1], torch.zeros(enc.hidden))  # absent target -> zero
