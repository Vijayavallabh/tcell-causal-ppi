"""Module 4 (Sparse Predictive-Rationale Head) tests — fully synthetic, no marts or embeddings.

The graph is a dense random HeteroData (hub + random edges) so the sampled neighbourhood carries far
more than top_k edges, giving the faithfulness / matched-random comparisons something to bite on. The
rationale head is zero-initialised, so an untrained head ranks edges by the frozen condition gate,
which is faithful by construction: keeping the highest-gate edges reproduces dz better than a matched
random subset, and removing them perturbs dz more — the properties tested below.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)  # many-core box: the tiny per-subgraph GNN ops thrash the thread pool otherwise

from tcell_pipeline import config
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.graph import TypedGraphEncoder, build_hetero_graph, sample_subgraph
from tcell_pipeline.programs import ProgramDecoder
from tcell_pipeline.rationale import (
    FaithfulnessTester,
    MatchedRandomSampler,
    RationaleHead,
    RationaleLoss,
    complement,
)

_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)
_KIND_FLAG = {"phys": "is_physical", "func": "is_functional", "cplx": "is_complex"}


def _big_graph(seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(30)]
    sources = list(config.PPI_SOURCES)
    rows = []

    def add(a, b, kind):
        flags = {"is_physical": 0, "is_functional": 0, "is_complex": 0}
        flags[_KIND_FLAG[kind]] = 1
        rows.append(dict(source_gene=a, target_gene=b, source=sources[int(rng.integers(len(sources)))],
                         evidence_type="x", score=float(rng.uniform(0.3, 1.0)),
                         is_direct_binary=int(rng.integers(0, 2)),
                         n_supporting_sources=int(rng.integers(1, 6)), **flags))

    for i in range(1, 13):                          # hub G0 -> G1..G12 (physical), a rich neighbourhood
        add("G0", f"G{i}", "phys")
    kinds = ["phys", "func", "cplx"]
    for _ in range(70):                             # random edges across all relations
        a, b = int(rng.integers(30)), int(rng.integers(30))
        if a != b:
            add(f"G{a}", f"G{b}", kinds[int(rng.integers(3))])

    edges = pd.DataFrame(rows)
    complexes = pd.DataFrame(
        [dict(protein_gene=f"G{i}", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1)
         for i in range(5)]
        + [dict(protein_gene=f"G{i}", complex_id=2, source_database="CORUM", confidence=1.0, is_curated=1)
           for i in range(3, 8)]
    )
    id_map = pd.DataFrame([dict(hgnc_symbol=g, uniprot_id=f"P{i}") for i, g in enumerate(genes)])
    baseline = pd.DataFrame([dict(hgnc_symbol=g, control_baseline_expr=float(rng.uniform(0, 2))) for g in genes])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def _setup(seed=0):
    torch.manual_seed(seed)
    graph, gene_to_idx = _big_graph(seed)
    genc = TypedGraphEncoder(graph, gene_to_idx)
    decoder = ProgramDecoder(torch.randn(20, 8))
    h_do, cond = torch.randn(config.H_DO_DIM), "Rest"
    sub = sample_subgraph(graph, "G0", gene_to_idx=gene_to_idx)
    with torch.no_grad():
        r = genc.encode_subgraph(sub, cond, h_do)
        head = RationaleHead()
        rat = head(r["gates"], r["node_states"], None, sub)
    n_edges = sum(int(g.numel()) for g in r["gates"].values())
    assert n_edges > config.RATIONALE_TOP_K  # neighbourhood must exceed |S| for the tests to be meaningful
    return dict(genc=genc, decoder=decoder, sub=sub, cond=cond, h_do=h_do, r=r, head=head, rat=rat)


def test_importance_in_unit_interval():
    for imp in _setup()["rat"]["importance"].values():
        if imp.numel():
            assert (imp >= 0).all() and (imp <= 1).all()


def test_top_k_selection_sorted():
    rat = _setup()["rat"]
    selected = rat["selected"]
    assert 0 < len(selected) <= config.RATIONALE_TOP_K
    vals = [v for _, _, v in selected]
    assert vals == sorted(vals, reverse=True)               # returned highest-importance first
    for rel, i, _ in selected:
        assert bool(rat["selection_mask"][rel][i])          # mask agrees with the selected list


def test_expression_only_gives_empty_rationale():
    rat = RationaleHead()(None, None, None, None)            # no graph -> nothing to rationalise
    assert rat["selected"] == [] and rat["selection_mask"] == {} and rat["importance"] == {}


def test_output_labeled_predictive_not_causal():
    rat = _setup()["rat"]
    assert rat["label"] == "predictive_rationale"
    assert "causal" not in rat["label"]                     # a predictive rationale, never a causal claim


def test_sufficiency_below_matched_random():
    s = _setup()
    tester = FaithfulnessTester(s["genc"], s["decoder"])
    mask = s["rat"]["selection_mask"]
    suff = tester.sufficiency(s["sub"], s["cond"], s["h_do"], mask)
    controls = MatchedRandomSampler(n_controls=50).sample(mask)
    rand = np.mean([tester.sufficiency(s["sub"], s["cond"], s["h_do"], c) for c in controls])
    assert suff < rand                                      # the rationale reproduces dz better than random


def test_necessity_above_matched_random():
    s = _setup()
    tester = FaithfulnessTester(s["genc"], s["decoder"])
    mask = s["rat"]["selection_mask"]
    nec = tester.necessity(s["sub"], s["cond"], s["h_do"], mask)
    controls = MatchedRandomSampler(n_controls=50).sample(mask)
    rand = np.mean([tester.necessity(s["sub"], s["cond"], s["h_do"], c) for c in controls])
    assert nec > rand                                       # removing the rationale perturbs dz more than random


def test_matched_random_matches_size_and_relations():
    mask = _setup()["rat"]["selection_mask"]
    controls = MatchedRandomSampler(n_controls=10).sample(mask)
    assert len(controls) == 10
    for c in controls:
        assert set(c) == set(mask)                          # same relations
        for rel, m in mask.items():
            assert c[rel].numel() == m.numel()              # same size
            assert int(c[rel].sum()) == int(m.sum())        # same per-relation count (relation composition)


def test_structural_ood_audit_returns_dict():
    s = _setup()
    tester = FaithfulnessTester(s["genc"], s["decoder"])
    audit = tester.structural_ood_audit(s["sub"], s["rat"]["selection_mask"])
    assert set(audit) == {"before", "after"}
    for side in ("before", "after"):
        assert {"degree_dist", "component_count", "sparsity", "hop_distance"} <= set(audit[side])
    assert audit["after"]["sparsity"] >= audit["before"]["sparsity"]  # deletion only removes edges


def test_loss_components_computable_with_gradients():
    s = _setup()
    genc, decoder, sub, cond, h_do = s["genc"], s["decoder"], s["sub"], s["cond"], s["h_do"]

    def dz(weights):
        h = genc.encode_subgraph(sub, cond, h_do, keep_mask=weights)["h_graph"]
        return decoder(h_do.reshape(1, -1), h.reshape(1, -1))["delta_z"].reshape(-1)

    with torch.no_grad():
        dz_full = dz(None)
    head = RationaleHead()                                   # fresh head so its grad is isolated
    rat = head(s["r"]["gates"], s["r"]["node_states"], None, sub)
    imp = rat["importance"]                                  # continuous importance -> soft gate weights
    dz_kept = dz(imp)                                        # differentiable through the head
    dz_removed = dz({rel: 1.0 - v for rel, v in imp.items()})

    loss = RationaleLoss()(imp, dz_full, dz_kept, dz_removed)
    assert all(torch.isfinite(loss[k]).all() for k in loss)
    loss["total"].backward()
    assert head.score.weight.grad is not None and torch.isfinite(head.score.weight.grad).all()
