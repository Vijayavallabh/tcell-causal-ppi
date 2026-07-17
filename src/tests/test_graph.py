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


# --------------------------------------------------------------------------------------------------
# Sampler equivalence: the CSR neighbour index must reproduce the full-scan sampler EXACTLY.
# The oracle below is deliberately SELF-CONTAINED -- it must not import the sampler's policy
# constants, because a mutation to a shared constant would move both sides and the test would
# pass on a broken sampler.
# --------------------------------------------------------------------------------------------------
_PP_RELATIONS_T = ("physical_ppi", "co_complex", "functional_assoc")
_PRIORITY_BONUS_T = {"physical_ppi": 1e6, "co_complex": 1e6, "functional_assoc": 0.0}
_SCORE_COL_T = 5  # edge_attr layout: source one-hot(5) then score


def _store_key_t(rel: str):
    return (PROTEIN, rel, PROTEIN) if rel in _PP_RELATIONS_T else (PROTEIN, _MEMBERSHIP, COMPLEX)


def _reference_grow(graph, seed, hops, cap):
    """The original O(|E|)-per-hop full-scan growth, frozen here as the equivalence oracle."""
    selected, seen = [seed], {seed}
    frontier = torch.tensor([seed])
    for _ in range(hops):
        nodes, keys = [], []
        for rel in _PP_RELATIONS_T:
            store = graph[PROTEIN, rel, PROTEIN]
            ei, ea = store.edge_index, store.edge_attr
            if ei.numel() == 0:
                continue
            for a, b in ((0, 1), (1, 0)):
                m = torch.isin(ei[a], frontier)
                nodes.append(ei[b][m])
                keys.append(ea[m][:, _SCORE_COL_T] + _PRIORITY_BONUS_T[rel])
        if not nodes:
            break
        nodes, keys = torch.cat(nodes), torch.cat(keys)
        new = []
        for i in torch.argsort(keys, descending=True).tolist():
            n = int(nodes[i])
            if n in seen:
                continue
            seen.add(n)
            new.append(n)
            if len(selected) + len(new) >= cap:
                break
        selected.extend(new)
        if len(selected) >= cap or not new:
            break
        frontier = torch.tensor(new)
    return selected[:cap]


def _reference_sample(graph, target_gene, hops, cap, gene_to_idx):
    """The original sample_subgraph: full-scan grow + full-scan induce. Equivalence oracle."""
    seed = gene_to_idx[target_gene]
    sel = torch.tensor(sorted(_reference_grow(graph, seed, hops, cap)))
    n_protein = graph[PROTEIN].x.shape[0]
    p_remap = torch.full((n_protein,), -1, dtype=torch.long)
    p_remap[sel] = torch.arange(sel.numel())
    memb = graph[PROTEIN, "complex_membership", COMPLEX].edge_index
    sel_complex = (
        torch.unique(memb[1][torch.isin(memb[0], sel)]) if memb.numel() else torch.tensor([], dtype=torch.long)
    )
    c_remap = torch.full((graph[COMPLEX].num_nodes,), -1, dtype=torch.long)
    c_remap[sel_complex] = torch.arange(sel_complex.numel())

    def induce(ei, ea, sr, dr):
        if ei.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long), ea[:0]
        keep = (sr[ei[0]] >= 0) & (dr[ei[1]] >= 0)
        return torch.stack([sr[ei[0][keep]], dr[ei[1][keep]]]), ea[keep]

    out = {"protein_orig": sel, "complex_orig": sel_complex}
    for rel in _PP_RELATIONS_T:
        s = graph[PROTEIN, rel, PROTEIN]
        out[rel] = induce(s.edge_index, s.edge_attr, p_remap, p_remap)
    s = graph[PROTEIN, "complex_membership", COMPLEX]
    out["complex_membership"] = induce(s.edge_index, s.edge_attr, p_remap, c_remap)
    return out


def _dense_random_graph(n_genes=60, seed=0):
    """A denser synthetic graph than _frames(): hubs, self-loops, isolated nodes, multi-hop reach —
    enough structure that a neighbour index and a full scan can actually disagree."""
    rng = torch.Generator().manual_seed(seed)
    names = [f"G{i:03d}" for i in range(n_genes)]
    rows = []
    for _ in range(400):
        i = int(torch.randint(0, n_genes, (1,), generator=rng))
        j = int(torch.randint(0, n_genes, (1,), generator=rng))
        src_kind = int(torch.randint(0, 3, (1,), generator=rng))
        score = float(torch.rand(1, generator=rng))
        kw = [dict(phys=1), dict(func=1), dict(cplx=1)][src_kind]
        source = ["biogrid", "string", "corum"][src_kind]
        rows.append(_edge(names[i], names[j], source, score, **kw))
    for j in range(1, 25):                                   # a hub: G000 wired to many neighbours
        rows.append(_edge("G000", names[j], "biogrid", 0.95, phys=1))
    rows.append(_edge("G059", "G059", "biogrid", 0.6, phys=1))  # self-loop
    edges = pd.DataFrame(rows)
    complexes = pd.DataFrame(
        [dict(protein_gene=names[i], complex_id=i % 7, source_database="CORUM", confidence=0.9, is_curated=1)
         for i in range(0, 40)]
    )
    id_map = pd.DataFrame([dict(hgnc_symbol=n, uniprot_id=f"P{i:04d}") for i, n in enumerate(names)])
    baseline = pd.DataFrame([dict(hgnc_symbol=n, control_baseline_expr=1.0) for n in names])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


@pytest.mark.parametrize("cap", [3, 16, 512])
@pytest.mark.parametrize("hops", [1, 2])
def test_sampler_matches_full_scan_reference(cap, hops):
    """The neighbour-index sampler must be EXACTLY the full-scan sampler — same nodes, same edge
    order, same features. Any divergence changes which subgraph the model sees, i.e. the science.

    Sweeps EVERY gene in the fixture, not a hand-picked few. An earlier version probed only
    (G000, G001, G030, G059) and was blind: deleting _grow's order-restoring sort left the whole
    suite green while the sampler diverged on 35 of these 60 genes. Divergence needs a frontier that
    is BOTH multi-node AND cap-truncated under a priority tie, which no hand-picked handful reliably
    hits — so pick none, and sweep. It costs ~nothing on a 60-node graph.
    """
    graph, gene_to_idx = _dense_random_graph()
    for gene in gene_to_idx:
        ref = _reference_sample(graph, gene, hops, cap, gene_to_idx)
        sub = sample_subgraph(graph, gene, hops=hops, cap=cap, gene_to_idx=gene_to_idx)
        assert torch.equal(sub[PROTEIN].orig_idx, ref["protein_orig"]), f"{gene}: node set differs"
        assert torch.equal(sub[COMPLEX].orig_idx, ref["complex_orig"]), f"{gene}: complex set differs"
        for rel in _PP_RELATIONS_T:
            ei, ea = ref[rel]
            assert torch.equal(sub[PROTEIN, rel, PROTEIN].edge_index, ei), f"{gene}/{rel}: edge_index"
            assert torch.equal(sub[PROTEIN, rel, PROTEIN].edge_attr, ea), f"{gene}/{rel}: edge_attr"
        ei, ea = ref["complex_membership"]
        assert torch.equal(sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_index, ei), f"{gene}: membership ei"
        assert torch.equal(sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_attr, ea), f"{gene}: membership ea"


# --------------------------------------------------------------------------------------------------
# Mini-batch equivalence: one PyG Batch through the encoder must equal the per-sample loop.
# The oracle drives encode_subgraph (the unbatched path Module 4 uses), so the two share no batching
# code. eval() throughout: DropEdge is train-only and random, so it has no per-sample equivalent.
# --------------------------------------------------------------------------------------------------
def _reference_forward(enc, target_genes, conditions, h_do):
    """The original per-sample loop, frozen here as the equivalence oracle."""
    device = enc.proj.weight.device
    h_do = h_do.to(device)
    rels = (*_PP_RELATIONS_T, _MEMBERSHIP)
    h_graphs, gates, confs = [], {r: [] for r in rels}, {r: [] for r in rels}
    for b, (gene, cond) in enumerate(zip(target_genes, conditions)):
        if gene not in enc.gene_to_idx:
            h_graphs.append(torch.zeros(config.GRAPH_HIDDEN_DIM, device=device))
            for r in rels:
                gates[r].append(torch.zeros(0, device=device))
                confs[r].append(torch.zeros(0, device=device))
            continue
        sub = sample_subgraph(enc.graph, gene, gene_to_idx=enc.gene_to_idx)
        e = enc.encode_subgraph(sub, cond, h_do[b])
        h_graphs.append(e["h_graph"])
        for r in rels:
            gates[r].append(e["gates"][r])
            confs[r].append(e["edge_confidences"][r])
    return torch.stack(h_graphs), gates, confs


def _assert_forward_matches(enc, genes, conds, h_do):
    ref_h, ref_g, ref_c = _reference_forward(enc, genes, conds, h_do)
    bat_h, bat_g, bat_c = enc(genes, conds, h_do)
    assert torch.allclose(bat_h, ref_h, atol=1e-5), "h_graph diverges from the per-sample loop"
    for rel in (*_PP_RELATIONS_T, _MEMBERSHIP):
        assert len(bat_g[rel]) == len(ref_g[rel]) == len(genes)  # one per-edge tensor per sample
        for b in range(len(genes)):
            assert bat_g[rel][b].shape == ref_g[rel][b].shape, f"{rel}[{b}] gate shape"
            assert torch.allclose(bat_g[rel][b], ref_g[rel][b], atol=1e-5), f"{rel}[{b}] gates"
            assert torch.allclose(bat_c[rel][b], ref_c[rel][b], atol=1e-5), f"{rel}[{b}] confidences"


def test_batched_forward_matches_per_sample_loop():
    torch.manual_seed(0)
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    # mixed conditions in ONE batch is the load-bearing case: each edge must be gated by its own
    # sample's condition, not the batch's first.
    genes = ["G000", "G001", "G030", "G000", "G059"]
    conds = ["Rest", "Stim48hr", "Stim8hr", "Stim48hr", "Rest"]
    _assert_forward_matches(enc, genes, conds, torch.randn(len(genes), config.GRAPH_HIDDEN_DIM))


def test_batched_forward_matches_loop_with_absent_and_isolated_targets():
    torch.manual_seed(0)
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    genes = ["NOTAGENE", "G000", "NOPE", "G059", "G002"]  # absent targets bracket the real ones
    conds = ["Rest", "Stim48hr", "Rest", "Stim8hr", "Rest"]
    _assert_forward_matches(enc, genes, conds, torch.randn(len(genes), config.GRAPH_HIDDEN_DIM))


def test_batched_forward_all_targets_absent():
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    h, gates, confs = enc(["NOPE", "ALSONOPE"], ["Rest", "Rest"], torch.randn(2, config.GRAPH_HIDDEN_DIM))
    assert torch.allclose(h, torch.zeros(2, config.GRAPH_HIDDEN_DIM))
    for rel in (*_PP_RELATIONS_T, _MEMBERSHIP):
        assert len(gates[rel]) == 2 and all(g.numel() == 0 for g in gates[rel])
        assert len(confs[rel]) == 2


def test_batched_gate_uses_each_samples_own_condition():
    """Same gene twice in one batch under different conditions -> different gates. Guards against a
    scatter that silently broadcasts one sample's condition across the whole batch."""
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    _, gates, _ = enc(["G000", "G000"], ["Rest", "Stim48hr"], torch.randn(2, config.GRAPH_HIDDEN_DIM))
    a, b = gates["physical_ppi"][0], gates["physical_ppi"][1]
    assert a.numel() > 0 and a.shape == b.shape
    assert not torch.allclose(a, b)  # identical subgraph, different condition -> gates must differ


def test_batched_static_encoder_pins_gate_to_one():
    """StaticTypedGraphEncoder overrides only _gate; the batched path must honour that.

    The all-ones assertion is vacuous on an empty gate (``allclose(zeros(0), ones(0))`` is True), so
    the gates are first pinned to the sub-graph's real edge counts. Without that, a forward that
    regressed to returning EMPTY gates for every sample would pass green — and empty gates silently
    zero StageALoss._graph's sparsity + unsourced terms and empty the Module-4 rationale.
    """
    from tcell_pipeline.baselines.graph_baselines import StaticTypedGraphEncoder

    genes, conds = ["G000", "G001"], ["Rest", "Stim48hr"]
    graph, gene_to_idx = _dense_random_graph()
    enc = StaticTypedGraphEncoder(graph, gene_to_idx).eval()
    _, gates, confs = enc(genes, conds, torch.randn(len(genes), config.GRAPH_HIDDEN_DIM))
    subs = [sample_subgraph(graph, g, gene_to_idx=gene_to_idx) for g in genes]  # sampler is deterministic
    for rel in (*_PP_RELATIONS_T, _MEMBERSHIP):
        for b, g in enumerate(gates[rel]):
            assert g.numel() == subs[b][_store_key_t(rel)].edge_index.shape[1]  # one gate per real edge
            assert confs[rel][b].shape == g.shape
            assert torch.allclose(g, torch.ones_like(g))  # gate pinned to 1.0: the H2b isolated variable
    assert sum(g.numel() for rel in _PP_RELATIONS_T for g in gates[rel]) > 0  # the gates are not all empty


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_batched_forward_matches_per_sample_loop_on_cuda():
    """The batched path does device-sensitive work (scatter, dense-batch pad, index_put), so the
    equivalence has to hold on CUDA and not just CPU."""
    torch.manual_seed(0)
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).to("cuda").eval()
    genes = ["G000", "NOTAGENE", "G001", "G030", "G059"]
    conds = ["Rest", "Rest", "Stim48hr", "Stim8hr", "Rest"]
    _assert_forward_matches(enc, genes, conds, torch.randn(len(genes), config.GRAPH_HIDDEN_DIM, device="cuda"))


def test_forward_dtype_follows_the_readout_under_autocast():
    """h_graph must carry whatever dtype the readout produced, as the old torch.stack(h_graphs) did.

    Preallocating the buffer as torch.zeros(...) hardcodes float32, so under autocast the bf16/fp16
    readout is silently cast back up into an fp32 h_graph: the graph pathway then leaves the autocast
    region at a different dtype from every other pathway, and ProgramDecoder concatenates h_do with
    h_graph. The unbatched encode_subgraph — the path Module 4 uses — is the reference dtype.
    """
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    h_do = torch.randn(3, config.GRAPH_HIDDEN_DIM)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        want = enc.encode_subgraph(
            sample_subgraph(graph, "G000", gene_to_idx=gene_to_idx), "Rest", h_do[0]
        )["h_graph"].dtype
        h_graph, _, _ = enc(["G000", "NOTAGENE", "G001"], ["Rest", "Rest", "Stim48hr"], h_do)
        h_absent, _, _ = enc(["NOPE"], ["Rest"], h_do[:1])
    assert want == torch.bfloat16                       # the autocast region really is low-precision
    assert h_graph.dtype == want, f"h_graph is {h_graph.dtype}, unbatched path gives {want}"
    # all targets absent: no readout runs, so there is no computed dtype to follow — the invariant
    # that matters is agreeing with h_do, which ProgramDecoder concatenates h_graph with
    assert h_absent.dtype == h_do.dtype


def test_untyped_forward_dtype_follows_the_readout_under_autocast():
    from tcell_pipeline.baselines.graph_baselines import UntypedGraphEncoder

    graph, gene_to_idx = _dense_random_graph()
    enc = UntypedGraphEncoder(graph, gene_to_idx).eval()
    h_do = torch.randn(2, config.GRAPH_HIDDEN_DIM)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        want = enc.encode_one("G000", h_do[0]).dtype    # the unbatched path is the reference
        h_graph, _, _ = enc(["G000", "NOTAGENE"], ["Rest", "Rest"], h_do)
    assert want == torch.bfloat16
    assert h_graph.dtype == want, f"h_graph is {h_graph.dtype}, unbatched path gives {want}"


def test_per_sample_gates_own_their_storage():
    """torch.split hands out VIEWS into the batch-wide gate tensor; the per-sample loop handed out
    independent tensors. A consumer that edits one sample's gates in place (alpha.clamp_) would
    otherwise raise, and holding one sample's gates would pin the whole batch's storage alive."""
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    with torch.no_grad():
        _, gates, confs = enc(["G000", "G001"], ["Rest", "Stim48hr"], torch.randn(2, config.GRAPH_HIDDEN_DIM))
    for rel in (*_PP_RELATIONS_T, _MEMBERSHIP):
        for g, c in zip(gates[rel], confs[rel]):
            assert g._base is None, f"{rel}: gate is a view into the batched tensor"
            assert c._base is None, f"{rel}: confidence is a view into the batched tensor"
            g.clamp_(0.0, 1.0)  # in-place must not raise on a split view


def test_forward_accepts_any_iterable_of_targets():
    """The batched path indexes target_genes/conditions by position; the old loop only zip()ed them,
    so a generator caller must not silently degrade to an empty batch."""
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    h_do = torch.randn(2, config.GRAPH_HIDDEN_DIM)
    ref, _, _ = enc(["G000", "G001"], ["Rest", "Stim48hr"], h_do)
    got, _, _ = enc(iter(["G000", "G001"]), iter(["Rest", "Stim48hr"]), h_do)
    assert torch.allclose(got, ref)
    assert not torch.allclose(got, torch.zeros_like(got))  # not silently all-absent


def test_sampler_index_rebuilds_when_the_graph_changes():
    """The CSR index is cached on the graph. The full scan it replaced re-read edge_index every call,
    so an edited graph took effect immediately; a stale index would hand an edge-ablation control the
    topology it thought it had removed, silently."""
    graph, gene_to_idx = _dense_random_graph()
    before = sample_subgraph(graph, "G001", hops=1, cap=512, gene_to_idx=gene_to_idx)
    store = graph[PROTEIN, "physical_ppi", PROTEIN]
    seed = gene_to_idx["G001"]
    new = max(set(range(len(gene_to_idx))) - set(before[PROTEIN].orig_idx.tolist()))  # an unreached node
    store.edge_index = torch.cat([store.edge_index, torch.tensor([[seed], [new]])], dim=1)
    store.edge_attr = torch.cat([store.edge_attr, store.edge_attr[:1]], dim=0)
    after = sample_subgraph(graph, "G001", hops=1, cap=512, gene_to_idx=gene_to_idx)
    assert new not in before[PROTEIN].orig_idx.tolist()
    assert new in after[PROTEIN].orig_idx.tolist(), "stale index: the appended edge was never seen"


@pytest.mark.parametrize("chunk", [1, 2, 3, 8, 0])
def test_encode_chunking_does_not_change_results(chunk, monkeypatch):
    """Chunking bounds peak memory; it must not touch the numbers. Every chunk size — including 0
    (disabled) and 1 (fully serial) — must agree with the unchunked batch, edge for edge."""
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    genes = ["G000", "NOTAGENE", "G001", "G030", "G059"]
    conds = ["Rest", "Rest", "Stim48hr", "Stim8hr", "Rest"]
    h_do = torch.randn(len(genes), config.GRAPH_HIDDEN_DIM)
    monkeypatch.setattr(config, "GRAPH_ENCODE_CHUNK", 0)
    ref_h, ref_g, ref_c = enc(genes, conds, h_do)
    monkeypatch.setattr(config, "GRAPH_ENCODE_CHUNK", chunk)
    got_h, got_g, got_c = enc(genes, conds, h_do)
    assert torch.allclose(got_h, ref_h, atol=1e-5)
    for rel in (*_PP_RELATIONS_T, _MEMBERSHIP):
        for b in range(len(genes)):
            assert torch.allclose(got_g[rel][b], ref_g[rel][b], atol=1e-5), f"{rel}[{b}] chunk={chunk}"
            assert torch.allclose(got_c[rel][b], ref_c[rel][b], atol=1e-5)


@pytest.mark.parametrize("chunk", [1, 3, 0])
def test_untyped_encode_chunking_does_not_change_results(chunk, monkeypatch):
    from tcell_pipeline.baselines.graph_baselines import UntypedGraphEncoder

    graph, gene_to_idx = _dense_random_graph()
    enc = UntypedGraphEncoder(graph, gene_to_idx).eval()
    genes = ["G000", "NOTAGENE", "G001", "G030"]
    h_do = torch.randn(len(genes), config.GRAPH_HIDDEN_DIM)
    monkeypatch.setattr(config, "GRAPH_ENCODE_CHUNK", 0)
    ref, _, _ = enc(genes, ["Rest"] * len(genes), h_do)
    monkeypatch.setattr(config, "GRAPH_ENCODE_CHUNK", chunk)
    got, _, _ = enc(genes, ["Rest"] * len(genes), h_do)
    assert torch.allclose(got, ref, atol=1e-5)


def test_chunking_bounds_message_passing_width(monkeypatch):
    """The point of chunking: no more than `chunk` subgraphs are message-passed at once, whatever
    batch the caller passes. Evaluation scores at BATCH_SIZE=64 under no_grad, where the per-row loop
    this replaced held exactly one subgraph — so an unbounded batch is a real memory regression."""
    graph, gene_to_idx = _dense_random_graph()
    enc = TypedGraphEncoder(graph, gene_to_idx).eval()
    widths = []
    real = enc._encode_chunk
    monkeypatch.setattr(enc, "_encode_chunk",
                        lambda part, *a, **k: (widths.append(len(part)), real(part, *a, **k))[1])
    monkeypatch.setattr(config, "GRAPH_ENCODE_CHUNK", 2)
    genes = ["G000", "G001", "G002", "G003", "G004", "G005", "G006"]
    enc(genes, ["Rest"] * len(genes), torch.randn(len(genes), config.GRAPH_HIDDEN_DIM))
    assert widths and max(widths) <= 2, f"chunk width exceeded: {widths}"
    assert sum(widths) == len(genes)  # and every row still got encoded exactly once


def test_incident_rejects_duplicate_nodes():
    """A repeated node re-emits its whole CSR row, so its edges would come back twice and the
    sub-graph would carry duplicates the boolean scan never produced. Guard, don't trust."""
    from tcell_pipeline.graph.neighborhood_sampler import _index_for

    graph, gene_to_idx = _dense_random_graph()
    index = _index_for(graph)
    ok = index.incident("physical_ppi", 0, torch.tensor([0, 1, 2]))
    assert ok.numel() >= 0                                   # duplicate-free is accepted
    with pytest.raises(ValueError, match="duplicate-free"):
        index.incident("physical_ppi", 0, torch.tensor([0, 1, 1]))
