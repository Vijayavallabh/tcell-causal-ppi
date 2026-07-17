"""Predictive-rationale audit (feat-012): stress-test the sparse rationale a FROZEN H1 model + head produce,
across a stratified case set, against matched-random controls and structural-OOD / source-ablation /
GInX-style diagnostics (report §Module 4 audit; walkthrough §rationale quality).

The audit runs on the frozen H1 model (feat-011's promoted config) and its fitted rationale head — it never
trains. Per case it re-uses the existing Module-4 machinery unchanged: ``RationaleHead`` to extract the
rationale S, ``FaithfulnessTester`` for the fixed-model sufficiency/necessity deletion tests and the
structural-OOD audit, and ``MatchedRandomSampler`` for the size+relation-matched negative controls.

Per case it reports: |S|, sufficiency vs matched-random, necessity vs matched-random, the structural-OOD
audit, a minimality curve (sufficiency as the top-ranked edges are added back), and stability. Aggregated it
reports: the fraction of cases whose rationale is more sufficient than random, the fraction more necessary
than random, mean minimality, mean stability, per-source ablation (Δ prediction when BioPlex / HuRI / STRING
/ CORUM edges are removed), and a GInX-style informative-vs-random comparison at several sparsities. Written
to ``<RATIONALE_AUDIT_ROOT>/audit_report.json``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tcell_pipeline import config
from tcell_pipeline.encoders.batch import build_encoder_batch
from tcell_pipeline.graph import sample_subgraph
from tcell_pipeline.rationale.faithfulness import FaithfulnessTester
from tcell_pipeline.rationale.matched_random import MatchedRandomSampler
from tcell_pipeline.rationale.rationale_head import _MEMBERSHIP, _PP_RELATIONS, edge_attr_of

_SOURCE_INDEX = {s: i for i, s in enumerate(config.PPI_SOURCES)}
_ABLATION_SOURCES = ("bioplex", "huri", "string", "corum")  # report's four ablated evidence sources
# ablate over PP edges AND the bipartite complex_membership edges — the latter are 100% CORUM-sourced (their
# source one-hot has corum set), so a CORUM ablation that ignored them would leave CORUM's membership routing
# fully active and understate the CORUM delta
_ABLATION_RELATIONS = (*_PP_RELATIONS, _MEMBERSHIP)


# --------------------------------------------------------------------------------------------------
# Stratified case selection
# --------------------------------------------------------------------------------------------------
def _bucket(vals: np.ndarray) -> np.ndarray:
    """Median split -> 0 (low) / 1 (high). A constant column collapses to all-low (one bucket)."""
    med = float(np.median(vals))
    return (vals >= med).astype(int) if np.ptp(vals) > 0 else np.zeros(len(vals), dtype=int)


def _strata(dataset, gene_to_idx: dict) -> list[dict]:
    """Per-row stratification keys: degree, effect size, condition, graph coverage (target in the graph)."""
    genes = dataset.pc["hgnc_symbol"].astype(str).tolist()
    conditions = dataset.pc["culture_condition"].astype(str).tolist()
    degree = dataset.pc.get("ppi_degree_functional")
    degree = np.zeros(len(genes)) if degree is None else np.nan_to_num(degree.to_numpy("float64"))
    Z = dataset._zscore[dataset.row_index].toarray().astype("float64")
    effect = np.linalg.norm(Z, axis=1)
    covered = np.array([g in gene_to_idx for g in genes])
    deg_b, eff_b = _bucket(degree), _bucket(effect)
    return [{"row": i, "gene": genes[i], "condition": conditions[i], "covered": bool(covered[i]),
             "stratum": (int(deg_b[i]), int(eff_b[i]), conditions[i], bool(covered[i]))}
            for i in range(len(genes))]


def _select_cases(strata: list[dict], n_cases: int, seed: int) -> list[dict]:
    """Round-robin across strata buckets so the case set spans degree / effect / condition / coverage
    rather than over-sampling one dominant bucket."""
    gen = np.random.default_rng(seed)
    buckets: dict = {}
    for s in strata:
        buckets.setdefault(s["stratum"], []).append(s)
    for b in buckets.values():
        gen.shuffle(b)
    order = sorted(buckets)
    picked: list[dict] = []
    i = 0
    while len(picked) < min(n_cases, len(strata)):
        b = buckets[order[i % len(order)]]
        if b:
            picked.append(b.pop())
        i += 1
        if all(not b for b in buckets.values()):
            break
    return picked


# --------------------------------------------------------------------------------------------------
# Per-case diagnostics
# --------------------------------------------------------------------------------------------------
def _prefix_mask(selection_mask: dict, selected: list, m: int) -> dict:
    """Keep-mask selecting only the top-``m`` ranked rationale edges (``selected`` is sorted desc)."""
    mask = {rel: torch.zeros_like(t) for rel, t in selection_mask.items()}
    for rel, idx, _ in selected[:m]:
        mask[rel][idx] = True
    return mask


def _empty_mask(selection_mask: dict) -> dict:
    return {rel: torch.zeros_like(t) for rel, t in selection_mask.items()}


def _minimality(tester, sub, cond, h_do, selected, selection_mask, dz_full, suff_full, tol: float = 0.1):
    """Sufficiency as the top-ranked edges are added back, and the fraction of |S| that recovers most of
    the rationale's reconstruction. suff(keep nothing) is the worst; suff(keep S) == ``suff_full`` the best.
    Minimality = smallest prefix fraction whose sufficiency is within ``tol`` of that full improvement."""
    n = len(selected)
    if n == 0:
        return [], None
    suff_empty = tester.sufficiency(sub, cond, h_do, _empty_mask(selection_mask), dz_full=dz_full)
    curve = [tester.sufficiency(sub, cond, h_do, _prefix_mask(selection_mask, selected, m), dz_full=dz_full)
             for m in range(1, n + 1)]
    target = suff_full + tol * (suff_empty - suff_full)  # within tol of the best (lowest) sufficiency
    for m, s in enumerate(curve, start=1):
        if s <= target:
            return curve, m / n
    return curve, 1.0


def _source_keep_mask(sub, source_idx: int):
    """Keep every edge (PP + complex_membership) EXCEPT those carrying ``source_idx`` in their source one-hot,
    so a source's contribution — including CORUM's membership edges — is fully removed."""
    keep = {}
    for rel in _ABLATION_RELATIONS:
        ea = edge_attr_of(sub, rel)
        if ea.numel() and ea.shape[1] > source_idx:
            keep[rel] = (ea[:, source_idx] == 0).to(ea.dtype)
    return keep


def _source_ablation(tester, sub, cond, h_do, dz_full) -> dict:
    """Δ prediction (‖dz(without source) − dz_full‖) when each evidence source's edges are removed."""
    out = {}
    for src in _ABLATION_SOURCES:
        keep = _source_keep_mask(sub, _SOURCE_INDEX[src])
        dz_abl = tester.delta_z(sub, cond, h_do, keep)
        out[src] = float((dz_abl - dz_full).norm())
    return out


def _flatten_importance(importance: dict) -> list:
    """(rel, local_idx, importance) across all relations, sorted desc — the full-edge ranking for GInX."""
    flat = [(rel, i, float(v)) for rel, t in importance.items() for i, v in enumerate(t.tolist())]
    return sorted(flat, key=lambda x: -x[2])


def _keep_from_edges(selection_mask: dict, edges: list) -> dict:
    mask = {rel: torch.zeros_like(t) for rel, t in selection_mask.items()}
    for rel, idx, _ in edges:
        mask[rel][idx] = True
    return mask


def _ginx(tester, sub, cond, h_do, importance, selection_mask, dz_full, sparsities, gen) -> dict:
    """GInX-style check: at each KEEP fraction, does keeping the top-importance edges reconstruct dz better
    than keeping the same number of random edges? Reports rationale vs mean-random sufficiency per sparsity."""
    ranked = _flatten_importance(importance)
    E = len(ranked)
    if E == 0:
        return {}
    out = {}
    for s in sparsities:
        k = max(1, int(round(s * E)))
        rat_suff = tester.sufficiency(sub, cond, h_do, _keep_from_edges(selection_mask, ranked[:k]),
                                      dz_full=dz_full)
        rand = []
        for _ in range(5):  # stochastic edge masks at this sparsity
            pick = [ranked[j] for j in gen.permutation(E)[:k]]
            rand.append(tester.sufficiency(sub, cond, h_do, _keep_from_edges(selection_mask, pick),
                                           dz_full=dz_full))
        out[f"{s:.2f}"] = {"rationale_sufficiency": rat_suff, "random_sufficiency": float(np.mean(rand))}
    return out


def _selected_set(selected: list) -> set:
    return {(rel, idx) for rel, idx, _ in selected}


def _stability(graph_encoder, head, sub, cond, h_do, base_selected: set, n_repeats: int = 3,
               seed: int = 0) -> float | None:
    """Mean Jaccard of the selected-edge set across independent train-mode (DropEdge-on) re-encodes vs the
    eval selection — does the rationale survive stochastic message passing?

    PyG's ``dropout_edge`` draws from the GLOBAL torch RNG, which nothing else in the audit seeds, so the
    stability numbers (and ``mean_stability`` in the report) would otherwise depend on ambient process RNG
    state and not be reproducible from the audit's ``seed``. The global RNG is seeded here and its prior state
    restored, so nothing leaks to the caller — ``torch.manual_seed`` reseeds the CPU **and every CUDA**
    generator, so the CUDA states must be saved/restored too (``torch.random.get/set_rng_state`` cover the CPU
    generator alone). ponytail: DropEdge only; a trained scorer that reads node states will vary more than the
    zero-init head (which ranks by the node-independent gate), so this is a lower bound on instability for an
    untrained head."""
    was_training = graph_encoder.training
    rng_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    graph_encoder.train()
    try:
        torch.manual_seed(seed)
        jac = []
        for _ in range(n_repeats):
            with torch.no_grad():
                r = graph_encoder.encode_subgraph(sub, cond, h_do)
                rat = head(r["gates"], r["node_states"], None, sub)
            s = _selected_set(rat["selected"])
            union = base_selected | s
            jac.append(len(base_selected & s) / len(union) if union else 1.0)
        return float(np.mean(jac))
    finally:
        graph_encoder.train(was_training)
        torch.random.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _audit_one(model, head, dataset, case: dict, tester, n_controls: int, sparsities, gen, seed: int) -> dict:
    gene, cond = case["gene"], case["condition"]
    i = case["row"]
    batch = build_encoder_batch(dataset.pc.iloc[[i]], dataset.obs.iloc[[i]])
    with torch.no_grad():
        h_do = model.perturbation_encoder(batch)[0]
        sub = sample_subgraph(model.graph_encoder.graph, gene, gene_to_idx=model.graph_encoder.gene_to_idx)
        r = model.graph_encoder.encode_subgraph(sub, cond, h_do)
        rat = head(r["gates"], r["node_states"], None, sub)
    mask, selected = rat["selection_mask"], rat["selected"]
    n_edges = int(sum(int(g.numel()) for g in r["gates"].values()))

    dz_full = tester.delta_z(sub, cond, h_do)  # mask-invariant; compute once, reuse across all controls
    suff = tester.sufficiency(sub, cond, h_do, mask, dz_full=dz_full)
    nec = tester.necessity(sub, cond, h_do, mask, dz_full=dz_full)
    controls = MatchedRandomSampler(n_controls=min(n_controls, 30)).sample(mask)
    rand_suff = float(np.mean([tester.sufficiency(sub, cond, h_do, c, dz_full=dz_full) for c in controls]))
    rand_nec = float(np.mean([tester.necessity(sub, cond, h_do, c, dz_full=dz_full) for c in controls]))
    ood = tester.structural_ood_audit(sub, mask)
    curve, minimality = _minimality(tester, sub, cond, h_do, selected, mask, dz_full, suff)
    ginx = _ginx(tester, sub, cond, h_do, rat["importance"], mask, dz_full, sparsities, gen)
    stability = _stability(model.graph_encoder, head, sub, cond, h_do, _selected_set(selected), seed=seed)
    source_ablation = _source_ablation(tester, sub, cond, h_do, dz_full)

    return {
        "gene": gene, "condition": cond, "stratum": case["stratum"], "n_edges": n_edges,
        "rationale_size": len(selected),
        "sufficiency": suff, "random_sufficiency": rand_suff, "sufficiency_below_random": bool(suff < rand_suff),
        "necessity": nec, "random_necessity": rand_nec, "necessity_above_random": bool(nec > rand_nec),
        "minimality": minimality, "minimality_curve": curve, "stability": stability,
        "structural_ood": {"components_before": ood["before"]["component_count"],
                           "components_after": ood["after"]["component_count"],
                           "deleted_fraction": ood["after"]["sparsity"],
                           "hop_before": ood["before"]["hop_distance"], "hop_after": ood["after"]["hop_distance"]},
        "source_ablation": source_ablation, "ginx": ginx,
    }


# --------------------------------------------------------------------------------------------------
# Aggregate + driver
# --------------------------------------------------------------------------------------------------
def _finite_mean(vals) -> float | None:
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def _aggregate(cases: list[dict]) -> dict:
    if not cases:
        return {"n_cases": 0}
    sources = {src: _finite_mean([c["source_ablation"].get(src) for c in cases]) for src in _ABLATION_SOURCES}
    keys = sorted({k for c in cases for k in c["ginx"]})
    ginx = {k: {"rationale": _finite_mean([c["ginx"][k]["rationale_sufficiency"] for c in cases if k in c["ginx"]]),
                "random": _finite_mean([c["ginx"][k]["random_sufficiency"] for c in cases if k in c["ginx"]])}
            for k in keys}
    return {
        "n_cases": len(cases),
        "frac_sufficiency_below_random": _finite_mean([c["sufficiency_below_random"] for c in cases]),
        "frac_necessity_above_random": _finite_mean([c["necessity_above_random"] for c in cases]),
        "mean_minimality": _finite_mean([c["minimality"] for c in cases]),
        "mean_stability": _finite_mean([c["stability"] for c in cases]),
        "source_ablation_delta_sufficiency": sources,
        "ginx_by_sparsity": ginx,
    }


def audit_rationale(model, head, dataset, n_cases: int = config.N_RATIONALE_AUDIT_CASES, *,
                    n_controls: int = config.N_MATCHED_CONTROLS, sparsities=(0.2, 0.4, 0.6),
                    device: str = "cpu", seed: int = 0,
                    out_path: Path | None = None) -> dict:
    """Audit ``model``'s frozen rationale ``head`` over a stratified ``n_cases`` subset of ``dataset``.

    ``model`` must carry a graph encoder (``model.graph_encoder`` with ``encode_subgraph`` + gates) — an
    expression-only model has no rationale to audit. Uncovered targets (absent from the PPI graph) are
    recorded but skipped for faithfulness. Writes ``audit_report.json`` and returns the report dict."""
    if getattr(model, "graph_encoder", None) is None:
        raise ValueError("audit_rationale needs a graph model (model.graph_encoder is None — nothing to audit)")
    model = model.to(device)
    model.eval()
    # the head's scorer consumes the encoder's node states, so it must live on the SAME device — the encoder
    # places its outputs on `device`, and a CPU nn.Linear fed a cuda tensor raises on the first case
    head = head.to(device)
    head.eval()
    tester = FaithfulnessTester(model.graph_encoder, model.decoder)
    gen = np.random.default_rng(seed)
    strata = _strata(dataset, model.graph_encoder.gene_to_idx)
    # uncovered targets (absent from the PPI graph) have no rationale, so they are filtered out BEFORE
    # stratified selection — otherwise they'd win round-robin slots that are then discarded, silently
    # shrinking the audited sample below n_cases even when covered targets remain
    covered = [s for s in strata if s["covered"]]
    n_uncovered = len(strata) - len(covered)
    chosen = _select_cases(covered, n_cases, seed)

    cases = [_audit_one(model, head, dataset, case, tester, n_controls, sparsities, gen, seed)
             for case in chosen]

    report = {"n_requested": n_cases, "n_audited": len(cases), "n_uncovered_in_dataset": n_uncovered,
              "aggregate": _aggregate(cases), "cases": cases}
    out_path = Path(out_path) if out_path else config.RATIONALE_AUDIT_ROOT / "audit_report.json"
    import json
    config.write_text_atomic(json.dumps(report, indent=2, default=float), out_path)
    report["report_path"] = str(out_path)
    return report
