"""Module 8 (External Comparators, feat-010) tests — fully synthetic (a small STRING-typed PPI graph, no
marts). Covers: STRING-only source filtering; the Stable-Shift low-rank + graph-conv predictor (shape,
gene-decode, a covered held-out gene recovers its single neighbour, absent -> zero); the TxPert-public
attention aggregator (single-neighbour attention == that neighbour, absent -> zero); the compatibility
report; and both comparators registering as distinct comparator families under the registry cap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
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


def test_summarize_vs_h1_kind_and_basis_are_parameterisable():
    """The tabular baselines reuse this summarizer but are explicitly NOT feat-010 external comparators;
    hardcoding kind='comparator' and the external-comparator basis string mislabels them in the JSON a
    later comparator-family-cap audit would scan."""
    from tcell_pipeline.run_module8_real import summarize_vs_h1
    rows = [{"name": "elastic_net", "systema": 0.0342}]
    out = summarize_vs_h1(rows, {"name": "condition_gated", "systema": 0.0834},
                          kind="baseline", basis="feat-006 tabular baselines; NOT external comparators")
    assert [e["kind"] for e in out["ranked"] if e["name"] == "elastic_net"] == ["baseline"]
    assert out["basis"].startswith("feat-006 tabular baselines")


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

    # No frozen H1 on disk (promoted.json absent) -> still ranks comparators; NO comparison was made, so the
    # verdict is None ("no H1 to compare"), NOT False ("H1 lost").
    s4 = summarize_vs_h1(comps, None)
    assert s4["margin_h1_minus_strongest"] is None and s4["h1_beats_strongest"] is None
    assert {e["name"] for e in s4["ranked"]} == {"stable_shift", "txpert_public"}

    # No comparator landed -> nothing to beat, no crash.
    s5 = summarize_vs_h1([], h1)
    assert s5["strongest_comparator"] is None and s5["margin_h1_minus_strongest"] is None


def test_summarize_vs_h1_beats_none_when_no_eligible_comparator_and_noise_band():
    from tcell_pipeline.run_module8_real import summarize_vs_h1

    h1 = {"name": "h1", "systema": 0.08}
    # every comparator non-finite -> no eligible bar; beats/within_noise are None (undefined), NOT a loss.
    s = summarize_vs_h1([{"name": "a", "systema": float("nan")}], h1)
    assert s["h1_beats_strongest"] is None and s["margin_within_noise"] is None
    assert s["strongest_comparator"] is None
    # a hairline win inside the noise band is FLAGGED, not reported as a decisive beat.
    s2 = summarize_vs_h1([{"name": "c", "systema": 0.0319}], {"name": "h1", "systema": 0.0321}, noise_margin=0.01)
    assert s2["h1_beats_strongest"] is True and s2["margin_within_noise"] is True
    # a real margin sits outside the band.
    s3 = summarize_vs_h1([{"name": "c", "systema": 0.0321}], {"name": "h1", "systema": 0.0834}, noise_margin=0.01)
    assert s3["h1_beats_strongest"] is True and s3["margin_within_noise"] is False


def test_summarize_vs_h1_fold_incomparable_skips_verdict():
    from tcell_pipeline.run_module8_real import summarize_vs_h1

    comps = [{"name": "c", "systema": 0.03}]
    h1 = {"name": "h1", "systema": 0.08}
    s = summarize_vs_h1(comps, h1, fold_comparable=False)   # e.g. a --n-max capped run vs the full-fold H1
    # a frozen H1 exists but the comparators are on a different fold -> NO verdict, and it records why.
    assert s["h1_beats_strongest"] is None and s["h1_systema"] is None and s["frozen_h1"] is None
    assert s["frozen_h1_available"] is True and s["fold_comparable"] is False and "h1_comparison_skipped" in s
    assert {e["name"] for e in s["ranked"]} == {"c"}       # comparators still ranked, no H1 entry


def test_summarize_vs_h1_no_typeerror_on_none_h1_name_and_numpy_finite():
    from tcell_pipeline.run_module8_real import _finite, summarize_vs_h1

    assert _finite(np.float32(0.05)) is True               # numpy scalar counts as finite (mirror _finite_or_none)
    # CONSTRUCTED breaker: partial promoted.json -> h1 dict with no 'name' and non-finite systema, plus a
    # non-finite comparator, so both tie at -inf and the tie-break would compare None < str.
    s = summarize_vs_h1([{"name": "c", "systema": float("nan")}], {"systema": float("nan")})
    assert isinstance(s["ranked"], list)                   # must NOT raise TypeError
    # a numpy-typed finite comparator is eligible and a numpy-typed H1 win is detected.
    s2 = summarize_vs_h1([{"name": "c", "systema": np.float32(0.03)}], {"name": "h1", "systema": np.float32(0.08)})
    assert s2["strongest_comparator"] == "c" and s2["h1_beats_strongest"] is True


def test_load_promoted_final_distinguishes_absent_corrupt_partial_valid(tmp_path):
    from tcell_pipeline.run_module8_real import _load_promoted_final

    p = tmp_path / "promoted.json"
    assert _load_promoted_final(p) == (None, "absent")                       # no file
    p.write_text("{ this is not json")
    h1, st = _load_promoted_final(p)
    assert h1 is None and st.startswith("unreadable")                        # corrupt/truncated
    p.write_text('{"runner_up": {"name": "x"}}')                            # parses, no 'final'
    assert _load_promoted_final(p) == (None, "present but no valid 'final' dict")
    p.write_text('{"final": "condition_gated"}')                            # 'final' is a bare string, not a dict
    assert _load_promoted_final(p) == (None, "present but no valid 'final' dict")
    p.write_text('{"final": {"name": "condition_gated", "systema": 0.0834}}')  # valid
    h1, st = _load_promoted_final(p)
    assert st == "ok" and h1["systema"] == 0.0834


def test_fmt_signed_is_none_safe():
    from tcell_pipeline.run_module8_real import _fmt_signed

    assert _fmt_signed(0.0834) == "+0.0834"
    assert _fmt_signed(None) == "n/a" and _fmt_signed(float("nan")) == "n/a"


def test_comparators_register_as_distinct_families(tmp_path):
    reg = tmp_path / "registry.yaml"
    r1 = register_run("stable_shift_v1", "H1-comparator", "q_pre", "blocked", 0, None,
                      family=StableShiftAdapter.family, path=reg)
    r2 = register_run("txpert_public_v1", "H1-comparator", "q_pre", "blocked", 0, None,
                      family=TxPertPublicAdapter.family, path=reg)
    assert r1.startswith("run-") and r2.startswith("run-")
    fams = {r["family"] for r in load_registry(reg)}
    assert fams == {"stable_shift", "txpert_public"}    # two distinct comparator families (within the cap of 2)


# --------------------------------------------------------------------------------------------------
# feat-006 under-fit gate: a floor the H1 must CLEAR is only as trustworthy as its own fit.
# --------------------------------------------------------------------------------------------------
def test_underfit_bar_makes_the_margin_an_upper_bound():
    """A non-converged bar could only score HIGHER once converged, so it can only SHRINK the H1 margin —
    publishing that margin as a settled number overstates H1. The firing input is real: the shipped
    elastic-net bar reported converged=False with 6.4% non-zero coefficients."""
    from tcell_pipeline.run_module8_real import flag_underfit_bars

    g = flag_underfit_bars({"elastic_net": {"converged": False, "n_iter_max": 2000, "max_iter": 2000}})
    assert g["underfit"] == ["elastic_net"]
    assert g["margin_is_upper_bound"] is True


def test_unknown_convergence_is_not_a_pass():
    """Absence of evidence must read as unknown, never as green (AGENTS.md). A bar that exposes
    diagnostics but cannot say whether it converged still bounds the margin from below."""
    from tcell_pipeline.run_module8_real import flag_underfit_bars

    g = flag_underfit_bars({"gradient_boosting": {"converged": None}})
    assert g["unknown"] == ["gradient_boosting"] and g["underfit"] == []
    assert g["margin_is_upper_bound"] is True


def test_underfit_gate_clears_when_every_iterative_bar_converged():
    """The other half of a real guard: it must be able to go GREEN, or it is decoration that always fires.
    Closed-form bars (ridge/zero/kNN) expose no diagnostics — that is 'no convergence question', not
    'unknown', so they must not pin the gate on forever."""
    from tcell_pipeline.run_module8_real import flag_underfit_bars

    g = flag_underfit_bars({"elastic_net": {"converged": True}, "gradient_boosting": {"converged": True}})
    assert g["underfit"] == [] and g["unknown"] == []
    assert g["margin_is_upper_bound"] is False
    assert flag_underfit_bars({})["margin_is_upper_bound"] is False


def test_tabular_covariate_fence_rejects_a_response_derived_column():
    """The fair-featured tabular bar consumes real perturbation-table columns, so it needs the SAME leakage
    fence H1's encoder has (PerturbationEncoder refuses Q_POST_COLS). CONSTRUCTED breaker: slip a
    response-derived column into the covariate list and the build must refuse, not quietly fit on it."""
    from tcell_pipeline import config
    from tcell_pipeline.run_module8_real import check_qpre

    ok = check_qpre(["culture_condition", "ppi_degree_physical", "donor_pc_00"], ["n_guides"])
    assert ok["q_pre"] == ["culture_condition", "ppi_degree_physical", "donor_pc_00"]

    with pytest.raises(ValueError, match="response-derived"):
        check_qpre(["culture_condition", config.Q_POST_COLS[0]], ["n_guides"])
    with pytest.raises(ValueError, match="response-derived"):
        check_qpre(["culture_condition"], ["n_guides", config.Q_POST_COLS[0]])
    # an UNCLASSIFIED perturbation-table column is not evidence of safety either — metadata is the
    # permissive fall-through, so it must be refused rather than assumed q_pre.
    with pytest.raises(ValueError, match="not declared q_pre"):
        check_qpre(["culture_condition", "row_index"], ["n_guides"])


def test_capped_run_cannot_clobber_the_published_full_fold_artifact():
    """A ``--n-max`` smoke run scores a DIFFERENT fold, so its numbers must never land on the path the
    full-fold result is published to. This bit for real: a 300-row smoke run overwrote
    tabular_baselines_vs_h1.json with capped numbers that ``fold_comparable=False`` labelled honestly but
    that had already destroyed the published full-fold table."""
    from tcell_pipeline.run_module8_real import _artifact_stem

    assert _artifact_stem(None) == "tabular_baselines"
    assert _artifact_stem(300) != _artifact_stem(None)
    assert "300" in _artifact_stem(300)


def test_tabicl_is_restricted_to_the_qpre_feature_set():
    """TabICL costs ~125 s PER PROGRAM per feature set (128 refits, no batching, 41 GiB peak so no
    concurrency). The published H1 margin is measured against the qpre bars, so scoring TabICL on the
    node-only set would burn ~4.4 GPU-h on a feature set no margin is drawn from. Pin the restriction so a
    later edit cannot silently double the GPU bill."""
    from tcell_pipeline.run_module8_real import _QPRE_ONLY, _TABULAR, _TABULAR_GPU

    assert "tabicl" in _QPRE_ONLY and "tabicl" in _TABULAR_GPU
    assert "tabicl" not in _TABULAR, "the GPU bar must not join the default CPU-only run"
    assert not (_QPRE_ONLY & set(_TABULAR)), "a CPU bar must stay on BOTH feature sets"


def test_bar_cache_is_invalidated_by_fold_shape_and_by_scoring_code(tmp_path):
    """The combined run is ~6 h (4.4 of them GPU) on a SHARED box, so a completed bar is checkpointed and
    reused on resume. Presence must not read as freshness: the signature has to change when the FOLD
    changes AND when the scoring or baseline CODE changes. The second half is the one that bites — the
    systema collapse fix silently invalidated every previously cached score without altering any fold
    dimension, so a shape-only key would have resurrected pre-fix numbers as if they were current."""
    from tcell_pipeline.run_module8_real import _bar_signature, load_cached_bar, save_cached_bar

    sig = _bar_signature(n_train=21262, n_val=4400, n_features=1453, k=128)
    assert set(sig) >= {"n_train", "n_val", "n_features", "k", "metrics_sha", "baselines_sha"}

    save_cached_bar(tmp_path, "qpre", "elastic_net", sig, {"systema": 0.0694}, {"converged": True})
    hit = load_cached_bar(tmp_path, "qpre", "elastic_net", sig)
    assert hit is not None and hit["metrics"]["systema"] == 0.0694 and hit["diagnostics"]["converged"]

    assert load_cached_bar(tmp_path, "qpre", "elastic_net", {**sig, "n_val": 4399}) is None
    assert load_cached_bar(tmp_path, "node", "elastic_net", sig) is None      # other feature set
    assert load_cached_bar(tmp_path, "qpre", "ridge", sig) is None            # other bar
    # CONSTRUCTED breaker: same fold, changed scoring code -> the cached score is stale, not a hit.
    assert load_cached_bar(tmp_path, "qpre", "elastic_net", {**sig, "metrics_sha": "different"}) is None
    assert load_cached_bar(tmp_path, "qpre", "elastic_net", {**sig, "baselines_sha": "different"}) is None


def test_bar_signature_is_per_bar_and_covers_feature_construction():
    """Two holes in the first cache key, both found by probing it rather than by it firing.

    (1) It hashed the WHOLE simple_baselines module, so correcting CatBoost's convergence criterion would
    have discarded TabICL's 4.4 GPU-hour cached score. Hash the specific bar's class instead.
    (2) It hashed the metric and the baselines but NOT the feature construction. Changing _qpre_block's
    imputation (median -> mean) leaves n_features identical, so the cache would serve pre-change scores as
    current — the exact 'presence is not freshness' failure the signature exists to prevent."""
    from tcell_pipeline.run_module8_real import _bar_signature

    kw = dict(n_train=21262, n_val=4400, n_features=1453, k=128)
    cat = _bar_signature(**kw, bar="catboost")
    tab = _bar_signature(**kw, bar="tabicl")
    assert "qpre_sha" in cat, "feature construction must be part of the key"
    assert cat["baselines_sha"] != tab["baselines_sha"], \
        "a per-bar key: editing one bar must not invalidate another's expensive cached score"
    assert _bar_signature(**kw, bar="catboost") == cat            # deterministic


def test_repro_part_propagates_a_nonzero_exit_on_cannot_verify(monkeypatch):
    """run_repro() printed 'VERDICT = CANNOT_VERIFY' and then `return 0`, so an unattended run or any
    exit-status CI gate recorded an unverifiable reproduction as success — the same guard-not-honored-to-
    the-exit-code defect this repo already fixed once in the multiseed campaign.

    It also built hash entries with no `provenance` field. verify.py treats unlabelled as self-derived and
    downgrades it from pass to incomplete, so the run reported hash:splits as 'compared against itself'
    while the SAME function printed "independently-checked artifacts: ['splits']" (found by session D).
    Both are fixed by deleting the duplicate builder and delegating to reproducibility.run_repro_real,
    which labels provenance and returns exit_code(verdict)."""
    from tcell_pipeline import run_module8_real
    from tcell_pipeline.reproducibility import run_repro_real

    monkeypatch.setattr(run_repro_real, "main", lambda *a, **k: 1)      # CANNOT_VERIFY
    assert run_module8_real.run_repro() == 1, "an unverifiable reproduction must not exit 0"
    monkeypatch.setattr(run_repro_real, "main", lambda *a, **k: 0)      # REPRODUCIBLE
    assert run_module8_real.run_repro() == 0                            # ...and it must be able to pass
