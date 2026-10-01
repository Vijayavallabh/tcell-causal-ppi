"""Phase-1 bug hunt (READ-ONLY): do the typed graph encoder's parameters get gradients, and do they move?

Discriminates "the graph negative is a scientific null" from "the negative was measured on a model whose
graph pathway is broken". Writes NOTHING outside its own stdout (and an optional --out JSON); trains no
checkpoint, touches no shared artifact.

    OMP_NUM_THREADS=4 PYTHONPATH=src uv run python -m tcell_pipeline.probe_graph_gradients --n-max 32

Probes, in the order the question decomposes:

  SELF   a deliberately SEVERED encoder (h_graph detached) must report zero grads. If it doesn't, every
         number below is a lie — this is the probe's own falsification test, run first.
  A      one real Stage-A fwd+bwd; per-parameter grad norm for the graph encoder, DECOMPOSED into the
         response term's contribution and the graph-penalty term's. grad None / ||grad||==0 is the
         severed-path smoking gun; a penalty gradient that dwarfs the response gradient is a different
         defect with the same symptom.
  B      snapshot params at init, run a few real optimiser steps, measure drift + the gate trajectory.
         Zero drift WITH non-zero grads would mean the optimiser never received them.
  C      d(delta_z)/d(h_graph), analytic and by finite difference: is the decoder sensitive to h_graph?
  D      typed_static's constant graph loss: by design or severed? Reports whether the term carries a
         grad_fn at all, then CONSTRUCTS the input that moves it if the path is live.
  E      the decisive one: how much does h_graph depend on the NEIGHBOURHOOD, at init vs under the frozen
         H1 checkpoint — measured by re-running the encoder with every edge gated off (the keep_mask path
         Module 4's faithfulness tests use). Ties the gate trajectory in B to the graph-independence of
         the promoted model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import (  # noqa: E402
    _MEMBERSHIP,
    _PP_RELATIONS,
    _store_key,
)
from tcell_pipeline.screening.screening import (  # noqa: E402
    CONDITION_GATED,
    TYPED_STATIC,
    nested_family_factories,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402
from tcell_pipeline.training.losses import StageALoss  # noqa: E402
from tcell_pipeline.training.trainer import seeded_init  # noqa: E402

_RELS = (*_PP_RELATIONS, _MEMBERSHIP)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _grad_norms(module) -> dict:
    """param name -> ||grad||_2, or None where autograd produced no grad at all (the severed signature)."""
    return {n: (None if p.grad is None else float(p.grad.norm())) for n, p in module.named_parameters()}


def _backward_only(term, model, loss) -> dict:
    """Grad norms on the graph encoder attributable to ONE loss term. Returns {} when the term carries no
    grad_fn — that is itself the answer for a constant (unlearnable) term, not an error to swallow."""
    model.zero_grad(set_to_none=True)
    loss.zero_grad(set_to_none=True)
    if not getattr(term, "requires_grad", False):
        return {}
    term.backward(retain_graph=True)
    return _grad_norms(model.graph_encoder)


def _gate_stats(model, batch, targets, conditions) -> dict:
    """mean / max / fraction-below-1e-3 of the per-edge condition gate over a real batch."""
    with torch.no_grad():
        out = model(batch, targets, conditions)
    alphas = [a for per_sample in (out["edge_gates"] or {}).values() for a in per_sample if a.numel()]
    if not alphas:
        return {"n_edges": 0}
    a = torch.cat(alphas)
    return {"n_edges": int(a.numel()), "mean": float(a.mean()), "max": float(a.max()),
            "frac_below_1e-3": float((a < 1e-3).float().mean())}


def _checkpoint_provenance(ckpt: Path) -> dict:
    """Is this checkpoint the one the promotion registry calls the frozen H1, or just a file sitting at
    the conventional path? "Presence is not freshness" (AGENTS.md): the screening tree accumulates
    checkpoints across runs, so a leftover from a different campaign lives at exactly this path. Returns
    the recorded path and whether it matches — ``is_promoted: None`` when no registry exists, never a
    silent pass."""
    reg = config.SCREENING_ROOT / "promoted.json"
    out = {"path": str(ckpt), "registry": str(reg), "is_promoted": None, "recorded": None,
           "mtime": None, "epoch": None}
    if ckpt.exists():
        out["mtime"] = datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat(timespec="seconds")
        out["epoch"] = torch.load(ckpt, map_location="cpu").get("epoch")
    if not reg.exists():
        return out  # unknown stays unknown
    try:
        recorded = json.loads(reg.read_text())["final"]["checkpoint"]
    except (KeyError, ValueError):
        return out
    out["recorded"] = recorded
    out["is_promoted"] = Path(recorded).resolve() == Path(ckpt).resolve()
    return out


def _first_known_target(model, targets):
    for t in targets:
        if t in model.graph_encoder.gene_to_idx:
            return t
    return None


def _neighbourhood_dependence(model, target, condition, h_do_row) -> dict:
    """||h_graph(full) - h_graph(all edges gated off)|| / ||h_graph(full)||.

    Uses encode_subgraph's keep_mask (Module 4's faithfulness deletion path): a zero weight multiplies the
    gate, so every message dies at every layer while the target's own node features survive. That isolates
    'what the neighbourhood contributes' from 'what the node itself contributes'.

    Forced to eval() for the measurement: config.EDGE_DROPOUT gates ``dropout_edge(training=self.training)``
    in every _RelMessage, and a freshly built nn.Module defaults to training=True — so a train-mode read
    drops ~10% of edges at random and makes this number differ run to run (measured: 1.1935/1.1966/1.1934
    across three train-mode repeats vs 1.1784 exactly in eval). The caller's mode is restored."""
    from tcell_pipeline.graph.neighborhood_sampler import sample_subgraph
    sub = sample_subgraph(model.graph_encoder.graph, target, gene_to_idx=model.graph_encoder.gene_to_idx)
    zero = {r: torch.zeros(int(sub[_store_key(r)].edge_index.shape[1])) for r in _RELS}
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            full = model.graph_encoder.encode_subgraph(sub, condition, h_do_row)
            none_ = model.graph_encoder.encode_subgraph(sub, condition, h_do_row, keep_mask=zero)
    finally:
        model.train(was_training)
    hf, hn = full["h_graph"], none_["h_graph"]
    gates = torch.cat([g for g in full["gates"].values() if g.numel()])
    return {"n_edges": int(sum(int(sub[_store_key(r)].edge_index.shape[1]) for r in _RELS)),
            "gate_mean": float(gates.mean()) if gates.numel() else None,
            "h_graph_norm": float(hf.norm()),
            "abs_delta": float((hf - hn).norm()),
            "rel_delta": float((hf - hn).norm() / hf.norm().clamp_min(1e-12))}


class _SeveredModel(torch.nn.Module):
    """Control for the probe itself: the real model with h_graph DETACHED before the decoder. Every graph
    encoder parameter must then report a zero/None gradient. A probe that cannot see this severance
    cannot be trusted to have seen its absence."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.graph_encoder = inner.graph_encoder
        self.decoder = inner.decoder

    def forward(self, batch, targets, conditions):
        out = self.inner(batch, targets, conditions)
        h_graph = out["h_graph"].detach()
        dec = self.decoder(out["h_do"], h_graph)
        # The gates are forwarded UNDETACHED and on purpose. Returning None here made
        # `StageALoss._graph` short-circuit to a constant `zeros(())` with no grad_fn, so the control's
        # `total` contained only the response/gene/de path — it proved that detaching h_graph kills
        # response-path gradients and nothing at all about the penalty path, while the probe printed
        # "a non-zero reading below is real". The severance under test is h_graph, not the penalty:
        # keeping the penalty live is what makes the control able to FAIL.
        return {**dec, "h_do": out["h_do"], "h_graph": h_graph,
                "edge_gates": out.get("edge_gates"), "edge_confidences": out.get("edge_confidences")}


# --------------------------------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------------------------------
def probe_self_control(build, batch, targets, conditions, dz, dx) -> dict:
    model = _SeveredModel(build())
    loss = StageALoss(model.decoder.gene_dim, model.decoder.program_dim, h_do_dim=model.decoder.h_do_dim)
    out = model(batch, targets, conditions)
    comps = loss(out, dz, dx, edge_confidences=out.get("edge_confidences"))
    model.zero_grad(set_to_none=True)
    comps["total"].backward()
    norms = _grad_norms(model.graph_encoder)
    live = {n: v for n, v in norms.items() if v is not None and v > 0}
    print(f"  severed control: {len(norms)} graph params, "
          f"{sum(v is None for v in norms.values())} with grad=None, {len(live)} with ||grad||>0")
    ok = not live
    print(f"  PROBE VALIDITY: {'PASS' if ok else 'FAIL'} — a severed encoder "
          f"{'reports no gradient, so a non-zero reading below is real' if ok else 'STILL SHOWS GRADIENT; ignore this run'}")
    return {"n_params": len(norms), "n_grad_none": sum(v is None for v in norms.values()),
            "n_grad_nonzero": len(live), "probe_valid": ok}


def probe_a(model, loss, batch, targets, conditions, dz, dx, seed: int = 0) -> dict:
    """One real fwd+bwd; per-param grad norms, decomposed by loss term.

    Stays in TRAIN mode deliberately — the question is what the real training step does to the gates, and
    DropEdge is part of that step. But DropEdge draws from the global RNG at forward time, so the exact
    per-parameter norms move a few percent run to run; the forward is seeded here so the numbers quoted
    downstream are reproducible. (Orders of magnitude were never at risk; 6 significant figures were.)"""
    t0 = time.perf_counter()
    with seeded_init(seed):
        out = model(batch, targets, conditions)
        comps = loss(out, dz, dx, edge_confidences=out.get("edge_confidences"))
    fwd = time.perf_counter() - t0

    print("  loss at init: " + "  ".join(f"{k}={float(v):.4g}" for k, v in comps.items()))
    contrib = {"response": float(comps["response"]),
               "gene": loss.lambda_gene * float(comps["gene"]),
               "de": loss.lambda_de * float(comps["de"]),
               "invariance": loss.lambda_inv * float(comps["invariance"]),
               "graph": loss.lambda_graph * float(comps["graph"])}
    print("  CONTRIBUTION TO TOTAL (lambda x term): "
          + "  ".join(f"{k}={v:.4g}" for k, v in contrib.items()))
    ratio = contrib["graph"] / max(contrib["response"], 1e-12)
    print(f"  graph penalty / response = {ratio:.1f}x")

    g_total = _backward_only(comps["total"], model, loss)
    g_resp = _backward_only(comps["response"], model, loss)
    g_graph = _backward_only(loss.lambda_graph * comps["graph"], model, loss)

    none_or_zero = [n for n, v in g_total.items() if v is None or v == 0.0]
    print(f"  graph-encoder params: {len(g_total)};  grad None or exactly 0 under TOTAL: {len(none_or_zero)}")
    if none_or_zero:
        print(f"    -> {none_or_zero[:12]}{' ...' if len(none_or_zero) > 12 else ''}")
    print(f"  {'param':<34} {'||g_total||':>12} {'||g_response||':>15} {'||g_graphpen||':>15} {'pen/resp':>10}")
    for n in g_total:
        gt, gr, gg = g_total[n], g_resp.get(n), g_graph.get(n)
        r = (gg / gr) if (gr and gg is not None and gr > 0) else float("nan")
        print(f"  {n:<34} {_f(gt):>12} {_f(gr):>15} {_f(gg):>15} {r:>10.1f}")
    model.zero_grad(set_to_none=True)
    loss.zero_grad(set_to_none=True)
    return {"fwd_seconds": fwd, "loss": {k: float(v) for k, v in comps.items()},
            "contribution": contrib, "graph_over_response": ratio,
            "grad_total": g_total, "grad_response": g_resp, "grad_graph_penalty": g_graph,
            "n_params_no_grad": len(none_or_zero)}


def _f(v) -> str:
    return "None" if v is None else f"{v:.4e}"


def probe_b(model, loss, batches, steps: int, lr: float) -> dict:
    """Real AdamW steps (Trainer's optimiser/settings); parameter drift from init + the gate trajectory.

    NOT A TIMING BENCHMARK. This loop does ONE forward per step; the real ``Trainer._epoch`` additionally
    calls ``_donor_variants``, re-forwarding the whole model ``DONOR_INVARIANCE_SAMPLES`` more times per
    training batch (config.DONOR_INVARIANCE is True by default). The per-step seconds recorded here are
    therefore a sub-component cost and must never be quoted as an end-to-end Stage-A step — that is the
    exact substitution AGENTS.md records as having nearly launched a 22.7 h campaign as "~7 h". To time a
    true step, run ``Trainer._epoch`` over a truncated dataset."""
    init = {n: p.detach().clone() for n, p in model.graph_encoder.named_parameters()}
    params = list(model.parameters()) + list(loss.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
    traj, step_times = [], []
    for i in range(steps):
        batch, targets, conditions, dz, dx, _ = batches[i % len(batches)]
        t0 = time.perf_counter()
        out = model(batch, targets, conditions)
        comps = loss(out, dz, dx, edge_confidences=out.get("edge_confidences"))
        opt.zero_grad(set_to_none=True)
        comps["total"].backward()
        # capture the return value: this is the TRUE pre-clip norm over model+loss params, i.e. the
        # denominator clip_grad_norm_ actually divides the whole update by. Summing the graph encoder's
        # own grads gives only a lower bound on it, so without this the crowd-out factor is a derivation
        # rather than a measurement.
        total_norm = float(torch.nn.utils.clip_grad_norm_(params, config.GRAD_CLIP))
        opt.step()
        step_times.append(time.perf_counter() - t0)
        alphas = [a for per in (out["edge_gates"] or {}).values() for a in per if a.numel()]
        a = torch.cat(alphas).detach() if alphas else torch.zeros(1)
        traj.append({"step": i, "total": float(comps["total"]), "response": float(comps["response"]),
                     "graph": float(comps["graph"]), "gate_mean": float(a.mean()),
                     "gate_max": float(a.max()), "grad_norm_preclip": total_norm,
                     "clip_scale": min(1.0, config.GRAD_CLIP / max(total_norm, 1e-12)),
                     "seconds": step_times[-1]})
        print(f"  step {i:2d}  total={float(comps['total']):11.4f}  response={float(comps['response']):.4f}  "
              f"graph={float(comps['graph']):12.4f}  gate_mean={float(a.mean()):.6f}  "
              f"||g||={total_norm:9.2f} -> clip x{min(1.0, config.GRAD_CLIP / max(total_norm, 1e-12)):.3e}  "
              f"({step_times[-1]:.1f}s)")
    drift = {n: float((p.detach() - init[n]).norm()) for n, p in model.graph_encoder.named_parameters()}
    rel = {n: float((p.detach() - init[n]).norm() / init[n].norm().clamp_min(1e-12))
           for n, p in model.graph_encoder.named_parameters()}
    frozen = [n for n, d in drift.items() if d == 0.0]
    print(f"  parameters that did NOT move in {steps} steps: {len(frozen)}/{len(drift)}"
          + (f" -> {frozen}" if frozen else ""))
    for n in sorted(drift, key=lambda k: -rel[k])[:10]:
        print(f"    {n:<34} ||Δ||={drift[n]:.4e}  rel={rel[n]:.4e}")
    mean_s = sum(step_times) / max(len(step_times), 1)
    print(f"  mean {mean_s:.2f}s per single-forward step — NOT an end-to-end Stage-A step "
          f"(donor invariance re-forwards {config.DONOR_INVARIANCE_SAMPLES}x more per real training batch)")
    # deliberately named so it cannot be quoted as a whole-pipeline number by accident
    return {"trajectory": traj, "drift": drift, "relative_drift": rel, "n_frozen": len(frozen),
            "mean_step_seconds_EXCLUDING_donor_invariance": mean_s}


def probe_c(model, batch, targets, conditions) -> dict:
    """d(delta_z)/d(h_graph): analytic grad norm + a finite-difference perturbation."""
    with torch.no_grad():
        out = model(batch, targets, conditions)
    h_do, h_g = out["h_do"].detach(), out["h_graph"].detach()
    hg = h_g.clone().requires_grad_(True)
    dz = model.decoder(h_do, hg)["delta_z"]
    dz.pow(2).sum().backward()
    analytic = float(hg.grad.norm())

    base = model.decoder(h_do, h_g)["delta_z"]
    fd = {}
    for eps in (1e-3, 1e-1, 1.0):
        pert = h_g + eps * torch.randn_like(h_g)
        with torch.no_grad():
            moved = model.decoder(h_do, pert)["delta_z"]
        fd[eps] = float((moved - base).norm() / base.norm().clamp_min(1e-12))
        print(f"  perturb h_graph by eps={eps:<6g} -> relative move in delta_z: {fd[eps]:.4e}")
    print(f"  ||d(sum delta_z^2)/d(h_graph)|| = {analytic:.4e}   "
          f"(||h_graph||={float(h_g.norm()):.4f}, ||delta_z||={float(base.norm()):.4f})")
    return {"analytic_grad_norm": analytic, "finite_difference": fd,
            "h_graph_norm": float(h_g.norm()), "delta_z_norm": float(base.norm())}


def probe_d(build_static, batch, targets, conditions, dz, dx) -> dict:
    """typed_static: is its frozen graph loss by design, or a severed path? Then construct the input that
    moves it if the path is live."""
    model = build_static()
    loss = StageALoss(model.decoder.gene_dim, model.decoder.program_dim, h_do_dim=model.decoder.h_do_dim)
    out = model(batch, targets, conditions)
    comps = loss(out, dz, dx, edge_confidences=out.get("edge_confidences"))
    g = comps["graph"]
    print(f"  typed_static graph term = {float(g):.4f}   requires_grad={bool(g.requires_grad)}   "
          f"grad_fn={type(g.grad_fn).__name__ if g.grad_fn is not None else None}")
    alphas = [a for per in out["edge_gates"].values() for a in per if a.numel()]
    a = torch.cat(alphas)
    print(f"  gates: n={a.numel()}  min={float(a.min()):.6f}  max={float(a.max()):.6f}  "
          f"requires_grad={bool(a.requires_grad)}")
    g_param = _backward_only(comps["graph"], model, loss)
    print(f"  grad on graph-encoder params from the graph term alone: "
          f"{'NONE (term is a constant w.r.t. every parameter)' if not g_param else g_param}")

    # CONSTRUCT the input that moves it: the term is sum_e (alpha + (1-conf)*alpha^2) with alpha == 1, so it
    # is a pure function of the EDGE SET and the confidence column. Change those and it must move; change
    # any parameter and it cannot.
    moved = {}
    gates0 = {r: [t.clone() for t in v] for r, v in out["edge_gates"].items()}
    confs0 = {r: [t.clone() for t in v] for r, v in out["edge_confidences"].items()}
    base = float(loss._graph(gates0, confs0))
    half = {r: [t[: t.numel() // 2] for t in v] for r, v in gates0.items()}
    half_c = {r: [t[: t.numel() // 2] for t in v] for r, v in confs0.items()}
    moved["drop_half_the_edges"] = float(loss._graph(half, half_c))
    all_sourced = {r: [torch.ones_like(t) for t in v] for r, v in confs0.items()}
    moved["set_every_confidence_to_1"] = float(loss._graph(gates0, all_sourced))
    with torch.no_grad():  # perturb every parameter of the encoder: the term must NOT move
        for p in model.graph_encoder.parameters():
            p.add_(torch.randn_like(p))
    out2 = model(batch, targets, conditions)
    moved["randomise_every_encoder_parameter"] = float(
        loss._graph(out2["edge_gates"], out2["edge_confidences"]))
    print(f"  baseline graph term = {base:.4f}")
    for k, v in moved.items():
        print(f"    {k:<36} -> {v:12.4f}   {'MOVED' if abs(v - base) > 1e-6 else 'unchanged'}")
    return {"requires_grad": bool(g.requires_grad), "value": float(g), "gate_min": float(a.min()),
            "gate_max": float(a.max()), "grad_on_params": g_param, "baseline": base, "constructed": moved}


def probe_e(build, ckpt: Path, batch, targets, conditions) -> dict:
    """h_graph's dependence on the neighbourhood, at init vs under the frozen H1 checkpoint."""
    fresh = build()
    target = _first_known_target(fresh, targets)
    if target is None:
        print("  no batch target is in the graph vocabulary — probe E undecidable on this batch")
        return {"status": "undecidable", "reason": "no known target in batch"}
    # h_do, target and condition must all come from the SAME row. _first_known_target skips rows whose
    # gene is out of the graph vocabulary, so it need not return row 0 — taking h_do from row 0 while the
    # subgraph and condition come from row k would silently encode one perturbation against another's
    # neighbourhood, with no error raised.
    row = list(targets).index(target)
    cond = conditions[row]

    def _h_do(model):
        """Each model's OWN perturbation embedding.

        This used to be computed once from `build()` — a THIRD, freshly random-initialised model — and
        then fed to both readings, so the "FROZEN H1" gate and sensitivity numbers were measured at an
        input that model never produces. The headline gate_collapse_factor therefore mixed the real
        collapse with an input-distribution mismatch and was not reproducible from the frozen weights
        alone. The condition gate reads h_do, so this is not a cosmetic difference."""
        with torch.no_grad():
            return model.perturbation_encoder(batch)[row].detach()

    print(f"  target={target!r}  condition={cond!r}  (batch row {row} of {len(list(targets))})")

    out = {"target": target, "condition": str(cond)}
    at_init = _neighbourhood_dependence(fresh, target, cond, _h_do(fresh))
    print(f"  AT INIT           edges={at_init['n_edges']:>7}  gate_mean={at_init['gate_mean']:.6f}  "
          f"||h_graph||={at_init['h_graph_norm']:.4f}  delete-all-edges rel-change={at_init['rel_delta']:.4e}")
    out["at_init"] = at_init

    prov = _checkpoint_provenance(ckpt)
    out["provenance"] = prov
    if not ckpt.exists():
        print(f"  frozen checkpoint absent at {ckpt} — cannot compare against the promoted H1")
        out["frozen"] = None
        return out
    verdict = {True: "IS the promoted H1", False: "** does NOT match promoted.json **",
               None: "UNKNOWN (no usable promotion registry)"}[prov["is_promoted"]]
    print(f"  checkpoint {verdict}; trained to epoch {prov['epoch']}, mtime {prov['mtime']}")
    if prov["is_promoted"] is False:
        print(f"    registry records: {prov['recorded']}")
        print("    -> this is a DIFFERENT checkpoint; anything below is not about the frozen H1")
    trained = build()
    trained.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    tr = _neighbourhood_dependence(trained, target, cond, _h_do(trained))
    print(f"  FROZEN H1         edges={tr['n_edges']:>7}  gate_mean={tr['gate_mean']:.6f}  "
          f"||h_graph||={tr['h_graph_norm']:.4f}  delete-all-edges rel-change={tr['rel_delta']:.4e}")
    out["frozen"] = tr
    out["gate_collapse_factor"] = (at_init["gate_mean"] / tr["gate_mean"]) if tr["gate_mean"] else None
    out["sensitivity_collapse_factor"] = (at_init["rel_delta"] / tr["rel_delta"]) if tr["rel_delta"] else None
    print(f"  gate mean fell {out['gate_collapse_factor']:.3g}x; neighbourhood sensitivity fell "
          f"{out['sensitivity_collapse_factor']:.3g}x from init to the frozen H1")
    return out


# --------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-max", type=int, default=32, help="rows per split (this is a probe, not a run)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6, help="optimiser steps for the drift probe")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional JSON dump of every measurement")
    a = ap.parse_args()
    torch.set_num_threads(1)  # many-core box: tiny per-subgraph GNN ops thrash the pool (README §perf)

    print(f"[probe] building real graph + datasets (n_max={a.n_max}) ...")
    t0 = time.perf_counter()
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, gene_to_idx = build_hetero_graph()
    train_ds = PerturbationDataset("train", n_max=a.n_max)
    factories = nested_family_factories(gene_names, graph, gene_to_idx)
    print(f"[probe] ready in {time.perf_counter() - t0:.1f}s; {len(train_ds)} train rows")

    batches = []
    for s in range(0, min(len(train_ds), a.batch_size * max(a.steps, 1)), a.batch_size):
        idx = range(s, min(s + a.batch_size, len(train_ds)))
        batches.append(PerturbationDataset.collate([train_ds[i] for i in idx]))
    batch, targets, conditions, dz, dx, _ = batches[0]
    print(f"[probe] {len(batches)} batches of <= {a.batch_size}; first batch targets={list(targets)}")

    def build():
        with seeded_init(a.seed):
            return factories[CONDITION_GATED]()

    res = {"n_max": a.n_max, "batch_size": a.batch_size, "steps": a.steps, "seed": a.seed}

    print("\n=== SELF-CONTROL: a deliberately severed encoder must show no gradient ===")
    res["self_control"] = probe_self_control(build, batch, targets, conditions, dz, dx)

    print("\n=== PROBE A: one real Stage-A fwd+bwd, grad norms per graph-encoder parameter ===")
    model = build()
    loss = StageALoss(model.decoder.gene_dim, model.decoder.program_dim, h_do_dim=model.decoder.h_do_dim)
    res["probe_a"] = probe_a(model, loss, batch, targets, conditions, dz, dx, seed=a.seed)

    print("\n=== PROBE C: d(delta_z)/d(h_graph) — is the decoder sensitive to the graph representation? ===")
    res["probe_c"] = probe_c(model, batch, targets, conditions)

    print(f"\n=== PROBE B: {a.steps} real AdamW steps (lr={config.LR}) — parameter drift + gate trajectory ===")
    print("  gates at init: " + str(_gate_stats(model, batch, targets, conditions)))
    res["probe_b"] = probe_b(model, loss, batches, a.steps, config.LR)
    print("  gates after:   " + str(_gate_stats(model, batch, targets, conditions)))
    res["probe_b"]["gates_after"] = _gate_stats(model, batch, targets, conditions)

    print("\n=== PROBE D: typed_static's constant graph loss — by design, or severed? ===")

    def build_static():
        with seeded_init(a.seed):
            return factories[TYPED_STATIC]()
    res["probe_d"] = probe_d(build_static, batch, targets, conditions, dz, dx)

    print("\n=== PROBE E: does h_graph depend on the neighbourhood? init vs the FROZEN H1 ===")
    ckpt = config.SCREENING_ROOT / CONDITION_GATED / str(a.seed) / "ckpt" / "stage_a_best.pt"
    res["probe_e"] = probe_e(build, ckpt, batch, targets, conditions)

    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"\n[probe] measurements -> {a.out}")

    # The self-control decides whether ANY reading above is trustworthy, so it must reach the exit code:
    # printing "ignore this run" and then returning 0 lets an unattended re-run, or any exit-status CI
    # gate, record a self-invalidated probe as green (AGENTS.md: trace the guard to the process exit code).
    if not res["self_control"]["probe_valid"]:
        print("\n[probe] FAILED: the severed-encoder control reported gradients, so this probe cannot "
              "detect a severed path. Every measurement above is void.")
        return 2
    prov = res["probe_e"].get("provenance") or {}
    if prov.get("is_promoted") is False:
        print("\n[probe] FAILED: probe E loaded a checkpoint that promoted.json does not record as the "
              "frozen H1 — its init-vs-frozen comparison is not about the promoted model.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
