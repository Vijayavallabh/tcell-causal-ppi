"""Module 4 (Sparse Predictive-Rationale Head) tests — fully synthetic, no marts or embeddings.

The graph is a dense random HeteroData (hub + random edges) so the sampled neighbourhood carries far
more than top_k edges, giving the faithfulness / matched-random comparisons something to bite on. The
rationale head is zero-initialised, so an untrained head ranks edges by the frozen condition gate,
which is faithful by construction: keeping the highest-gate edges reproduces dz better than a matched
random subset, and removing them perturbs dz more — the properties tested below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    edge_index_of,
)

_PP = ("physical_ppi", "co_complex", "functional_assoc")

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
    mask = s["rat"]["selection_mask"]
    audit = tester.structural_ood_audit(s["sub"], mask)
    assert set(audit) == {"before", "after"}
    for side in ("before", "after"):
        assert {"degree_dist", "component_count", "sparsity", "hop_distance"} <= set(audit[side])
    # sparsity is PP-scoped and must equal the INDEPENDENTLY-counted removed-PP fraction (real coverage
    # of the removed-fraction math — not the old `after >= 0` tautology)
    total_pp = sum(int(edge_index_of(s["sub"], rel).shape[1]) for rel in _PP)
    removed_pp = sum(int(mask[rel].sum()) for rel in _PP)
    assert audit["before"]["sparsity"] == 0.0
    assert audit["after"]["sparsity"] == pytest.approx(removed_pp / total_pp)
    # deleting edges can only fragment, never merge, connected components
    assert audit["after"]["component_count"] >= audit["before"]["component_count"]


def test_faithfulness_is_deterministic_under_active_dropout():
    # _setup leaves the encoder in train mode (DropEdge p=0.1 active). The fixed-model contract requires
    # FaithfulnessTester to force eval, so two identical deletion re-runs must match EXACTLY; without the
    # eval-forcing fix, DropEdge randomness makes them differ. Also checks the caller's mode is restored.
    s = _setup()
    assert s["genc"].training  # regression guard: _setup never eval()s the encoder
    tester = FaithfulnessTester(s["genc"], s["decoder"])
    mask = s["rat"]["selection_mask"]
    a = tester.sufficiency(s["sub"], s["cond"], s["h_do"], mask)
    b = tester.sufficiency(s["sub"], s["cond"], s["h_do"], mask)
    assert a == b                    # exact: no DropEdge noise leaks into the frozen re-encode
    assert s["genc"].training        # eval state restored — the tester didn't mutate the caller's encoder


# --------------------------------------------------------------------------------------------------
# Stage B: the rationale-head FIT loop (feat-008 §b) + its matched-random controls
# --------------------------------------------------------------------------------------------------
def _cases(n=4, seed=0):
    """(gene, condition, h_do) cases over the fixture graph — the fit loop samples the subgraph itself."""
    g = torch.Generator().manual_seed(seed)
    return [(f"G{i}", "Rest", torch.randn(config.H_DO_DIM, generator=g)) for i in range(n)]


def _fit_rationale(s, head=None, cases=None, **kw):
    from tcell_pipeline.rationale.rationale_fit import fit_rationale_head
    head = head if head is not None else RationaleHead()
    res = fit_rationale_head(s["genc"], s["decoder"], cases or _cases(), head=head,
                             n_controls=kw.pop("n_controls", 4), max_epochs=kw.pop("max_epochs", 3), **kw)
    return head, res


def test_rationale_fit_moves_only_the_head(tmp_path):
    s = _setup()
    before = {n: p.detach().clone() for m in (s["genc"], s["decoder"]) for n, p in m.named_parameters()}
    head, res = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    frozen = [n for m in (s["genc"], s["decoder"]) for n, p in m.named_parameters()
              if not torch.equal(p.detach(), before[n])]
    assert frozen == []                                       # the backbone did not move at all
    assert head.score.weight.abs().sum() > 0                  # ...and the head did (zero-init -> non-zero)
    assert res["epochs_run"] == 3 and Path(res["best_ckpt"]).exists()


def test_rationale_fit_lowers_its_objective(tmp_path):
    _, res = _fit_rationale(_setup(), max_epochs=8, ckpt_dir=tmp_path, log_dir=tmp_path)
    assert res["history"][-1]["total"] < res["history"][0]["total"]
    assert {"total", "sparsity", "sufficiency", "necessity", "contrastive"} <= set(res["history"][0])


def test_rationale_fit_leaves_the_head_at_the_BEST_epoch_not_the_last(tmp_path):
    """Same trap as the calibration fit, and worse here: this objective PLATEAUS rather than diverging,
    so the loss reads identical to 16 digits while the optimiser keeps stepping — the head drifts by
    ~6.5 in weight space past the checkpointed epoch with no sign of it in the history. A head left at
    the last epoch is not the artifact feat-012's audit would load."""
    s = _setup()
    head, res = _fit_rationale(s, lr=5.0, max_epochs=30, patience=3, ckpt_dir=tmp_path, log_dir=tmp_path)
    hist = [h["total"] for h in res["history"]]
    best_i = min(range(len(hist)), key=lambda i: hist[i])
    assert best_i < res["epochs_run"] - 1                     # the run continued past its best epoch...
    assert res["best_epoch"] == best_i
    ckpt = torch.load(res["best_ckpt"], weights_only=True)
    for k, v in ckpt["head"].items():
        assert torch.equal(v, dict(head.state_dict())[k])     # in-memory head == the saved artifact


def test_rationale_fit_asserts_the_freeze_when_something_moves_the_backbone(tmp_path):
    """The loop's freeze check must be wired: a loss that writes to a decoder weight reaches it."""
    s = _setup()

    class _Saboteur(RationaleLoss):
        def forward(self, *a, **k):
            with torch.no_grad():
                s["decoder"].expr_path.weight += 1e-4
            return super().forward(*a, **k)

    with pytest.raises(RuntimeError, match="not frozen"):
        _fit_rationale(s, loss=_Saboteur(), max_epochs=2, ckpt_dir=tmp_path, log_dir=tmp_path)


def test_rationale_fit_is_deterministic_under_active_dropout(tmp_path):
    """_setup leaves DropEdge on. The fit must force eval (the fixed-model contract the deletion tests
    rely on) and restore the caller's mode, so two fits from the same zero-init head coincide exactly."""
    s = _setup()
    assert s["genc"].training
    a, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    assert s["genc"].training                                 # caller's train mode restored
    b, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    assert torch.equal(a.score.weight, b.score.weight)


def test_rationale_contrasts_report_a_matched_random_control_for_every_headline(tmp_path):
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    s = _setup()
    head, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    c = rationale_contrasts(s["genc"], s["decoder"], head, _cases(), n_controls=8)["contrasts"]
    assert set(c) == {"sufficiency_vs_random", "necessity_vs_random", "sufficiency_vs_untrained"}
    assert c["sufficiency_vs_random"]["higher_is_better"] is False   # sufficiency: closer to dz_full is better
    assert c["necessity_vs_random"]["higher_is_better"] is True      # necessity: removing S should hurt more
    for spec in c.values():
        assert set(spec["fit"]) == set(spec["control"]) == set(spec["units"]) and len(spec["units"]) == 4


def test_untrained_head_is_the_control_the_fit_must_beat(tmp_path):
    """'the fitted head is better' needs the untrained head COMPUTED, not assumed: the zero-init head is
    already faithful by construction, so a fit that only reproduces it has bought nothing."""
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    s = _setup()
    head, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    c = rationale_contrasts(s["genc"], s["decoder"], head, _cases(), n_controls=4)["contrasts"]
    ref = rationale_contrasts(s["genc"], s["decoder"], RationaleHead(), _cases(), n_controls=4)["contrasts"]
    # the control arm of vs_untrained is the ZERO-INIT head's own sufficiency
    assert c["sufficiency_vs_untrained"]["control"] == ref["sufficiency_vs_untrained"]["fit"]


def test_a_fit_that_adds_nothing_cannot_clear_the_gate(tmp_path):
    """max_epochs=0 leaves the zero-init head untouched, so every vs-untrained delta is exactly zero:
    undecidable, never a spurious win."""
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    from tcell_pipeline.training.freeze_gate import FREEZE, evaluate_gate
    s = _setup()
    head, res = _fit_rationale(s, max_epochs=0, ckpt_dir=tmp_path, log_dir=tmp_path)
    assert res["epochs_run"] == 0 and float(head.score.weight.detach().abs().sum()) == 0.0
    c = rationale_contrasts(s["genc"], s["decoder"], head, _cases(), n_controls=4)["contrasts"]
    spec = c["sufficiency_vs_untrained"]
    assert all(spec["fit"][u] == spec["control"][u] for u in spec["units"])
    r = evaluate_gate(c)
    assert r["contrasts"]["sufficiency_vs_untrained"]["ci_excludes_zero"] is None
    assert r["decision"] != FREEZE


def test_contrast_controls_are_wired_to_the_FITTED_rationale(tmp_path):
    """Guards the CALL SITE, not just the helper: rationale_contrasts must score, and match its
    controls to, the mask the FITTED head produces — not the zero-init head's."""
    from tcell_pipeline.rationale.rationale_fit import _case_controls, rationale_contrasts, untrained_like
    s = _setup()
    head, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    cases = _cases(2)
    rep = rationale_contrasts(s["genc"], s["decoder"], head, cases, n_controls=4)

    def edges_of(h, gene, cond, h_do):
        m, _ = _case_controls(s["genc"], s["decoder"], h, gene, cond, h_do, n_controls=1)
        return sorted((rel, i) for rel, v in m.items() for i in v.nonzero().reshape(-1).tolist())

    distinguishable = 0
    for row, (gene, cond, h_do) in zip(rep["per_case"], cases):
        assert row["selected_edges"] == edges_of(head, gene, cond, h_do)      # the FITTED rationale
        distinguishable += edges_of(untrained_like(head), gene, cond, h_do) != row["selected_edges"]
    assert distinguishable                       # ...and on this fixture the two heads really differ


def test_rationale_contrasts_are_deterministic_under_active_dropout(tmp_path):
    """Found a real bug: the contrast builder extracted the rationale with DropEdge still ACTIVE, so the
    selected edges — and every audit number keyed to them — were a fresh random draw per call."""
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    s = _setup()
    assert s["genc"].training                                 # regression guard: _setup never eval()s
    head, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    a = rationale_contrasts(s["genc"], s["decoder"], head, _cases(2), n_controls=4)
    b = rationale_contrasts(s["genc"], s["decoder"], head, _cases(2), n_controls=4)
    assert a["per_case"] == b["per_case"]                     # exact: no DropEdge noise in the audit inputs
    assert s["genc"].training                                 # ...and the caller's mode is restored


def test_duplicate_case_ids_are_refused(tmp_path):
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    s = _setup()
    dup = [(g, c, h, "same-id") for g, c, h in _cases(2)]      # explicit unit ids that collide
    with pytest.raises(ValueError, match="uniquely identified"):
        rationale_contrasts(s["genc"], s["decoder"], RationaleHead(), dup, n_controls=2)


def _graph_blind(decoder):
    """A decoder whose prediction cannot depend on the graph: the graph pathway outputs a constant 0 and
    the mixture gate ignores h_graph, so every edge mask leaves dz EXACTLY unchanged. This is the shape
    the frozen H1 appears to have (deletion distances ~1e-6) — the case the noise floor exists for."""
    with torch.no_grad():
        decoder.graph_path.weight.zero_(), decoder.graph_path.bias.zero_()
        decoder.gate.weight.zero_(), decoder.residual.weight.zero_(), decoder.residual.bias.zero_()
    return decoder


def test_a_case_whose_deletions_move_nothing_is_dropped_as_non_informative():
    """A deletion test on a graph-blind model measures floating-point dust. Dust must not become a data
    point: a paired t on consistently-signed 1e-10 differences can 'clear' a gate on nothing at all."""
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    s = _setup()
    rep = rationale_contrasts(s["genc"], _graph_blind(s["decoder"]), RationaleHead(), _cases(3),
                              n_controls=4)
    assert rep["n_informative"] == 0 and rep["n_cases"] == 3
    for row in rep["per_case"]:
        assert row["informative"] is False
        assert row["suff"] is not None                        # the measured dust stays ON THE RECORD...
    for spec in rep["contrasts"].values():
        assert all(spec["fit"][u] is None for u in spec["units"])   # ...but never enters the statistic


def test_all_cases_non_informative_is_undecidable_not_a_verdict():
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    from tcell_pipeline.training.freeze_gate import UNDECIDABLE, evaluate_gate
    s = _setup()
    c = rationale_contrasts(s["genc"], _graph_blind(s["decoder"]), RationaleHead(), _cases(3),
                            n_controls=4)["contrasts"]
    r = evaluate_gate(c)
    assert r["decision"] == UNDECIDABLE                       # nothing measurable != decided against
    assert all(r["contrasts"][k]["n"] == 0 for k in c)
    assert len(r["contrasts"]["sufficiency_vs_random"]["dropped"]) == 3   # dropped LOUDLY, named


def test_an_informative_case_survives_the_noise_floor():
    """The floor must not swallow real signal: the ordinary fixture model IS graph-sensitive."""
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    s = _setup()
    rep = rationale_contrasts(s["genc"], s["decoder"], RationaleHead(), _cases(3), n_controls=4)
    assert rep["n_informative"] == 3
    assert all(r["informative"] for r in rep["per_case"])


def test_a_model_that_predicts_nothing_is_never_informative():
    """The input that fires the zero-scale fence: ||dz_full||=0 makes the relative floor 0, so without
    it any 1e-9 residue would be promoted to a measurement."""
    from tcell_pipeline.rationale.rationale_fit import _informative
    dust = {"dz_full_norm": 0.0, "suff": 1e-9, "nec": 0.0, "rand_suff": 0.0, "rand_nec": 0.0}
    assert _informative(dust) is False
    assert _informative({**dust, "dz_full_norm": 1.0, "suff": 0.5}) is True     # real movement survives


def test_no_gate_cases_is_undecidable_not_a_crash_or_a_pass():
    from tcell_pipeline.rationale.rationale_fit import rationale_contrasts
    from tcell_pipeline.training.freeze_gate import UNDECIDABLE, evaluate_gate
    s = _setup()
    c = rationale_contrasts(s["genc"], s["decoder"], RationaleHead(), [], n_controls=2)["contrasts"]
    assert all(spec["units"] == [] for spec in c.values())
    assert evaluate_gate(c)["decision"] == UNDECIDABLE


def test_removal_arm_uses_the_complement_of_the_importance():
    """The necessity arm must re-run the model with the rationale REMOVED. Passing the kept-mask twice
    still yields a loss that falls, so only a polarity check catches it."""
    from tcell_pipeline.rationale.rationale_fit import _case_loss, _case_state, _dz
    s = _setup()
    s["genc"].eval()                                          # DropEdge off: the re-encodes must be comparable
    head = RationaleHead()
    st = _case_state(s["genc"], s["decoder"], head, "G0", "Rest", s["h_do"], n_controls=2, seed=0)
    comps = _case_loss(s["genc"], s["decoder"], head, RationaleLoss(), st)
    imp = head(st["r"]["gates"], st["r"]["node_states"], None, st["sub"])["importance"]
    args = (s["genc"], s["decoder"], st["sub"], "Rest", s["h_do"])
    kept = _dz(*args, keep_mask=imp)
    removed = _dz(*args, keep_mask={rel: 1.0 - v for rel, v in imp.items()})
    tot = lambda d: float(d["total"].detach())
    ref = RationaleLoss()(imp, st["dz_full"], kept, removed, st["dz_controls"])
    swapped = RationaleLoss()(imp, st["dz_full"], removed, kept, st["dz_controls"])
    assert tot(comps) == pytest.approx(tot(ref), rel=1e-6)
    assert tot(swapped) != pytest.approx(tot(ref), rel=1e-6)  # the arms are distinguishable


def test_gate_controls_are_matched_to_the_FITTED_selection(tmp_path):
    """The reported control must be matched to the rationale it is compared against — matching the
    untrained head's composition would compare the fit to a control for a different rationale."""
    from tcell_pipeline.rationale.rationale_fit import _case_controls, untrained_like
    s = _setup()
    head, _ = _fit_rationale(s, ckpt_dir=tmp_path, log_dir=tmp_path)
    gene, cond, h_do = _cases(1)[0]
    counts = lambda m: {rel: int(v.sum()) for rel, v in m.items()}
    mask, controls = _case_controls(s["genc"], s["decoder"], head, gene, cond, h_do, n_controls=6)
    assert len(controls) == 6
    for c in controls:
        assert counts(c) == counts(mask)                      # matched to the FITTED mask's composition
    # ...and the fitted mask is not just the zero-init head's mask relabelled
    ref_mask, _ = _case_controls(s["genc"], s["decoder"], untrained_like(head), gene, cond, h_do,
                                 n_controls=1)
    assert any(not torch.equal(mask[rel], ref_mask[rel]) for rel in mask)


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
