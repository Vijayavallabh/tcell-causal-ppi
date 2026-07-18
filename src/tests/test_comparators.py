"""Module 8 (External Comparators, feat-010) tests — fully synthetic (a small STRING-typed PPI graph, no
marts). Covers: STRING-only source filtering; the Stable-Shift low-rank + graph-conv predictor (shape,
gene-decode, a covered held-out gene recovers its single neighbour, absent -> zero); the TxPert-public
attention aggregator (single-neighbour attention == that neighbour, absent -> zero); the compatibility
report; and both comparators registering as distinct comparator families under the registry cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_num_threads(1)

from tcell_pipeline import config
from tcell_pipeline.comparators import (
    StableShiftAdapter,
    TxPertPublicAdapter,
    compatibility,
    source_adjacency,
    write_compatibility_report,
)
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.graph import build_hetero_graph
from tcell_pipeline.screening import load_registry, register_run

_G, _K = 6, 3
_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)


def _edge(src, dst, source, score, phys=0, func=0, cplx=0, direct=0, nsup=1):
    return dict(source_gene=src, target_gene=dst, source=source, evidence_type="x", score=score,
                is_physical=phys, is_functional=func, is_complex=cplx, is_direct_binary=direct,
                n_supporting_sources=nsup)


def _graph():
    # a STRING chain G0-G1-G2-G3-G4-G5 plus ONE non-STRING (biogrid) shortcut G0-G5 that STRING filtering
    # must exclude
    edges = pd.DataFrame([
        _edge("G0", "G1", "string", 0.9, func=1), _edge("G1", "G2", "string", 0.8, func=1),
        _edge("G2", "G3", "string", 0.7, func=1), _edge("G3", "G4", "string", 0.6, func=1),
        _edge("G4", "G5", "string", 0.5, func=1), _edge("G0", "G5", "biogrid", 0.95, phys=1),
    ])
    complexes = pd.DataFrame([
        dict(protein_gene="G0", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="G1", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
    ])
    id_map = pd.DataFrame([dict(hgnc_symbol="G0", uniprot_id="P0"), dict(hgnc_symbol="G1", uniprot_id="P1")])
    baseline = pd.DataFrame([dict(hgnc_symbol="G0", control_baseline_expr=1.0)])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def _train_signal(g2i):
    rng = np.random.default_rng(0)
    genes = ["G0", "G2", "G4"]                      # even nodes carry training signal
    z = rng.standard_normal((3, _K))
    return genes, z


def test_source_adjacency_string_only_excludes_other_sources():
    graph, g2i = _graph()
    a_string, _ = source_adjacency(graph, g2i, sources=("string",))
    a_all, _ = source_adjacency(graph, g2i, sources=None)
    i0, i5 = g2i["G0"], g2i["G5"]
    assert a_all[i0, i5] > 0 and a_all[i5, i0] > 0      # the biogrid G0-G5 shortcut is in the full graph
    assert a_string[i0, i5] == 0 and a_string[i5, i0] == 0  # ...but NOT in the STRING-only graph
    assert a_string[g2i["G4"], g2i["G5"]] > 0          # the STRING chain edge survives


def test_stable_shift_recovers_single_covered_neighbour_and_absent_zero():
    graph, g2i = _graph()
    genes, z = _train_signal(g2i)
    B = np.random.default_rng(1).standard_normal((_G, _K))
    adapter = StableShiftAdapter.from_hetero_graph(graph, g2i, basis=B, string_only=True, rank=8).fit(genes, z)
    dz, dx = adapter.predict(["G1", "G5", "NOTINGRAPH"])
    assert dz.shape == (3, _K) and dx.shape == (3, _G) and np.isfinite(dz).all()
    assert np.allclose(dx, dz @ B.T)
    # G5's only STRING neighbour with training signal is G4 -> full-rank low-rank reconstruct recovers G4's dz
    assert np.allclose(dz[1], z[2], atol=1e-6)
    assert np.allclose(dz[2], 0)                        # gene absent from the graph -> zero shift
    assert not np.allclose(dz[0], 0)                    # G1 blends its covered neighbours G0+G2 -> non-zero


def test_txpert_public_attention_over_string_neighbours():
    graph, g2i = _graph()
    genes, z = _train_signal(g2i)
    B = np.eye(_K)
    adapter = TxPertPublicAdapter.from_hetero_graph(graph, g2i, basis=B, string_only=True).fit(genes, z)
    dz, dx = adapter.predict(["G5", "G1", "NOTINGRAPH"])
    assert dz.shape == (3, _K) and np.isfinite(dz).all()
    # G5 attends over its only covered STRING neighbour G4 -> attention weight 1 -> exactly G4's signal
    assert np.allclose(dz[0], z[2], atol=1e-6)
    assert np.allclose(dz[2], 0)                        # absent -> zero
    # G1 attends over covered neighbours {G0, G2}: a convex combination lies within their bounding box
    lo, hi = np.minimum(z[0], z[1]), np.maximum(z[0], z[1])
    assert np.all(dz[1] >= lo - 1e-9) and np.all(dz[1] <= hi + 1e-9)


def test_compatibility_report_records_public_only(tmp_path):
    for cls, family in ((StableShiftAdapter, "stable_shift"), (TxPertPublicAdapter, "txpert_public")):
        path = write_compatibility_report(cls, root=tmp_path / "comp")
        doc = yaml.safe_load(path.read_text())
        assert doc["family"] == family and doc["checkpoint"] is None
        assert doc["public_only"] is True and "license" in doc and "exposure_class" in doc
    assert compatibility(TxPertPublicAdapter)["wrapped_upstream"] is False  # upstream package not installed


def test_public_only_is_explicit_flag_not_substring():
    class _Proprietary:  # "non-public" contains the substring "public"; the explicit flag must win
        __name__ = "_Proprietary"
        family = "fake"
        LICENSE = "proprietary"
        EXPOSURE_CLASS = "non-public (proprietary weights)"
        CHECKPOINT = None
        PUBLIC_ONLY = False
    assert compatibility(_Proprietary)["public_only"] is False


def test_summarize_vs_h1_ranks_margin_and_guards_bad_input():
    from tcell_pipeline.run_module8_real import summarize_vs_h1

    comps = [{"name": "stable_shift", "systema": 0.0217}, {"name": "txpert_public", "systema": 0.0321}]
    h1 = {"name": "condition_gated", "systema": 0.0834}
    s = summarize_vs_h1(comps, h1)
    assert s["strongest_comparator"] == "txpert_public"        # max-systema eligible comparator
    assert abs(s["margin_h1_minus_strongest"] - (0.0834 - 0.0321)) < 1e-12
    assert s["h1_beats_strongest"] is True
    assert [e["name"] for e in s["ranked"]][0] == "condition_gated"   # H1 tops the joint systema ranking

    # CONSTRUCTED breaker: a NaN-systema comparator must NOT be picked as strongest (mirrors promote()'s
    # non-finite fix) and must not crash the ranking.
    s2 = summarize_vs_h1(comps + [{"name": "broken", "systema": float("nan")}], h1)
    assert s2["strongest_comparator"] == "txpert_public"
    assert [e["name"] for e in s2["ranked"]][-1] == "broken"   # non-finite sinks to the bottom, not the top

    # HONEST converging-negative: an H1 BELOW the strongest comparator reports a loss, not a win.
    s3 = summarize_vs_h1(comps, {"name": "weak_h1", "systema": 0.010})
    assert s3["h1_beats_strongest"] is False and s3["margin_h1_minus_strongest"] < 0

    # No frozen H1 on disk (promoted.json absent) -> still ranks comparators, margin is undefined not a crash.
    s4 = summarize_vs_h1(comps, None)
    assert s4["margin_h1_minus_strongest"] is None and s4["h1_beats_strongest"] is False
    assert {e["name"] for e in s4["ranked"]} == {"stable_shift", "txpert_public"}

    # No comparator landed -> nothing to beat, no crash.
    s5 = summarize_vs_h1([], h1)
    assert s5["strongest_comparator"] is None and s5["margin_h1_minus_strongest"] is None


def test_comparators_register_as_distinct_families(tmp_path):
    reg = tmp_path / "registry.yaml"
    r1 = register_run("stable_shift_v1", "H1-comparator", "q_pre", "blocked", 0, None,
                      family=StableShiftAdapter.family, path=reg)
    r2 = register_run("txpert_public_v1", "H1-comparator", "q_pre", "blocked", 0, None,
                      family=TxPertPublicAdapter.family, path=reg)
    assert r1.startswith("run-") and r2.startswith("run-")
    fams = {r["family"] for r in load_registry(reg)}
    assert fams == {"stable_shift", "txpert_public"}    # two distinct comparator families (within the cap of 2)
