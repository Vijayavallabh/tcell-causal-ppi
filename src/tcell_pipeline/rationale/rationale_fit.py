"""Rationale-head FIT loop over a FROZEN backbone (feat-008 §b; Stage B, walkthrough §8.1).

``RationaleHead`` + ``RationaleLoss`` existed as modules with no fit loop; this is that loop, and it is
what feat-012's audit needs — the audit scores a FITTED head and cannot run on one that was never fitted.

    fit_rationale_head(graph_encoder, decoder, cases, head=RationaleHead()) -> history / checkpoint

``cases`` are ``(gene, condition, h_do)`` (optionally a 4th element: the unit id used in the gate); the
loop samples each subgraph itself. Per case the frozen backbone's outputs — node states, condition
gates, ``dz_full`` and the matched-random control predictions — are computed ONCE and reused, because
the encoder/decoder are eval + frozen for the whole fit (the same fixed-model contract
``FaithfulnessTester`` enforces: without eval, DropEdge would make every deletion re-run stochastic).
Only the head's scorer moves, and that is asserted against a pre-fit snapshot, not assumed.

``rationale_contrasts`` builds the paired freeze-gate inputs. In this near-null-signal regime a
rationale head can look good by fitting noise, so every headline is reported against a control:

    sufficiency_vs_random    keeping S reproduces dz better than a SIZE- and RELATION-matched random
                             edge set (``MatchedRandomSampler``), recomputed against the FITTED mask
    necessity_vs_random      removing S perturbs dz more than removing a matched random set
    sufficiency_vs_untrained the fit beats the ZERO-INIT head, which is already faithful by
                             construction (it ranks by the frozen gate). A fit that only reproduces it
                             bought nothing and must not be frozen — freeze the free head instead.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path

import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.graph import sample_subgraph
from tcell_pipeline.rationale.faithfulness import FaithfulnessTester
from tcell_pipeline.rationale.matched_random import MatchedRandomSampler
from tcell_pipeline.rationale.rationale_head import RationaleHead
from tcell_pipeline.rationale.rationale_loss import RationaleLoss
from tcell_pipeline.training.freeze_gate import require_unique_units
from tcell_pipeline.training.stage_b import assert_backbone_frozen, eval_mode, frozen_snapshot

_COMPONENTS = ("total", "sparsity", "sufficiency", "necessity", "contrastive")

# A deletion test is a measurement only if deleting something MOVES the prediction. If keeping the
# rationale, removing it, and every matched-random control all shift dz by less than this fraction of
# ||dz_full||, the frozen model's prediction does not read its graph edges at all and the case measures
# floating-point dust. Dust must not become a data point: a paired t on consistently-signed 1e-10
# differences can "clear" a gate on nothing — the same shape as the collapsed predictor that scored
# +0.0129 on the primary endpoint off numerical residue. Such cases are dropped (None -> the paired core
# drops them LOUDLY and names them) and counted in `n_informative`; all-dust is UNDECIDABLE, never a call.
NOISE_FLOOR_REL = 1e-4


def _backbone(graph_encoder, decoder) -> nn.Module:
    """A throwaway container so the freeze snapshot/assert (shared with the Stage-B calibration fit)
    reports readable parameter names. It holds the SAME tensors — nothing is copied."""
    return nn.ModuleDict({"graph_encoder": graph_encoder, "decoder": decoder})


def _unit(case, i: int):
    return case[3] if len(case) > 3 else i


def _dz(graph_encoder, decoder, sub, condition, h_do, keep_mask=None) -> torch.Tensor:
    """Program delta under an edge keep-mask, WITH gradients — the differentiable twin of
    ``FaithfulnessTester.delta_z`` (which is ``@torch.no_grad`` and so cannot train a head). Callers
    must have put the backbone in eval; this does not touch the mode."""
    h = graph_encoder.encode_subgraph(sub, condition, h_do, keep_mask=keep_mask)["h_graph"]
    return decoder(h_do.reshape(1, -1), h.reshape(1, -1))["delta_z"].reshape(-1)


@torch.no_grad()
def _case_state(graph_encoder, decoder, head, gene: str, condition: str, h_do, *,
                n_controls: int, seed: int) -> dict:
    """Everything about a case that the head cannot change: subgraph, gates, node states, dz_full, and
    the matched-random control predictions the contrastive term measures against."""
    sub = sample_subgraph(graph_encoder.graph, gene, gene_to_idx=graph_encoder.gene_to_idx)
    with eval_mode(graph_encoder, decoder):
        r = graph_encoder.encode_subgraph(sub, condition, h_do)
        dz_full = _dz(graph_encoder, decoder, sub, condition, h_do)
        mask = head(r["gates"], r["node_states"], None, sub)["selection_mask"]
        # ponytail: the in-loop controls are sampled ONCE, against the head's INITIAL selection. |S| is
        # fixed by top_k so their size never drifts; only the relation split can. Re-sample per epoch if
        # a fitted rationale is seen migrating across relations — a moving control makes the objective
        # non-stationary, which is why it is not the default. The GATE's controls are a separate,
        # full-precision set recomputed against the FITTED mask (see rationale_contrasts).
        controls = MatchedRandomSampler(n_controls=n_controls, seed=seed).sample(mask)
        dz_controls = [_dz(graph_encoder, decoder, sub, condition, h_do, c) for c in controls]
    return {"sub": sub, "r": r, "dz_full": dz_full, "dz_controls": dz_controls,
            "condition": condition, "h_do": h_do, "gene": gene}


def fit_rationale_head(graph_encoder, decoder, cases, *, head=None, loss=None,
                       n_controls: int = 8, lr: float = config.LR,
                       weight_decay: float = config.WEIGHT_DECAY, max_epochs: int = config.MAX_EPOCHS,
                       patience: int = config.EARLY_STOP_PATIENCE, seed: int = 0, device: str = "cpu",
                       ckpt_dir: Path = config.CHECKPOINTS_ROOT,
                       log_dir: Path = config.LOGS_ROOT) -> dict:
    """Fit ``head`` on ``cases`` over a frozen backbone. Early stopping keys on the FIT objective — no
    held-out statistic enters the fit, so the fitted head is a pure function of (cases, hyper-params)."""
    cases = [tuple(c) for c in cases]
    if not cases:
        raise ValueError("no rationale cases to fit — check the split role / case selection")
    head = (head if head is not None else RationaleHead()).to(device)
    loss = loss or RationaleLoss()
    graph_encoder, decoder = graph_encoder.to(device), decoder.to(device)
    backbone = _backbone(graph_encoder, decoder)
    snapshot = frozen_snapshot(backbone, list(head.parameters()))
    grad_state = [(p, p.requires_grad) for p in backbone.parameters()]
    history, best, best_path, wait = [], float("inf"), None, 0
    best_epoch, best_state = None, None
    stack = contextlib.ExitStack()          # bound BEFORE the try, so the finally can always close it
    try:
        stack.enter_context(eval_mode(graph_encoder, decoder))  # fixed-model for every re-encode
        for p in backbone.parameters():
            p.requires_grad_(False)
        states = [_case_state(graph_encoder, decoder, head, c[0], c[1], c[2].to(device),
                              n_controls=n_controls, seed=seed) for c in cases]
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(max_epochs):
            opt.zero_grad()
            agg = {k: 0.0 for k in _COMPONENTS}
            for st in states:                           # grads accumulate per case, one step per epoch
                comps = _case_loss(graph_encoder, decoder, head, loss, st)
                comps["total"].backward()
                for k in _COMPONENTS:
                    agg[k] += float(comps[k].detach())
            opt.step()
            agg = {k: v / len(states) for k, v in agg.items()}
            history.append({"epoch": epoch, **agg})
            config.write_text_atomic(json.dumps(history, indent=2),
                                     Path(log_dir) / "stage_b_rationale_history.json")
            if agg["total"] < best - 1e-6:
                best, wait, best_epoch = agg["total"], 0, epoch
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                best_path = _save(head, ckpt_dir, epoch, agg)
            else:
                wait += 1
                if wait >= patience:
                    break
        # The head left in memory must BE the checkpointed artifact. This objective plateaus rather than
        # diverging, so the history reads identical to 16 digits while the optimiser keeps stepping — the
        # scorer drifts far (~6.5 in weight space on the fixture) with no sign of it in the log. Restoring
        # here means every caller gets the artifact, not just the ones that remember to reload.
        if best_state is not None:
            head.load_state_dict(best_state)
    finally:
        for p, req in grad_state:
            p.requires_grad_(req)
        stack.close()
    assert_backbone_frozen(backbone, snapshot)
    return {"history": history, "epochs_run": len(history), "n_cases": len(cases),
            "best_ckpt": str(best_path) if best_path else None, "best_epoch": best_epoch,
            "best_total": best if history else None}


def _case_loss(graph_encoder, decoder, head, loss, st: dict) -> dict:
    """One case's RationaleLoss, differentiable back to the head through the soft gate weights."""
    rat = head(st["r"]["gates"], st["r"]["node_states"], None, st["sub"])
    imp = rat["importance"]
    args = (st["sub"], st["condition"], st["h_do"])
    dz_kept = _dz(graph_encoder, decoder, *args, keep_mask=imp)
    dz_removed = _dz(graph_encoder, decoder, *args, keep_mask={rel: 1.0 - v for rel, v in imp.items()})
    return loss(imp, st["dz_full"], dz_kept, dz_removed, st["dz_controls"])


def _save(head, ckpt_dir: Path, epoch: int, metrics: dict) -> Path:
    config.ensure_dir(Path(ckpt_dir))
    path = Path(ckpt_dir) / "stage_b_rationale_head.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"head": head.state_dict(), "epoch": epoch, "metrics": metrics}, tmp)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------------------
# Freeze-gate inputs: every headline against its control
# --------------------------------------------------------------------------------------------------
def untrained_like(head) -> RationaleHead:
    """A zero-init copy of ``head`` — same architecture, the faithful-by-construction ranking the fit
    has to improve on."""
    ref = copy.deepcopy(head)
    nn.init.zeros_(ref.score.weight)
    nn.init.zeros_(ref.score.bias)
    return ref.eval()


@torch.no_grad()
def _case_controls(graph_encoder, decoder, head, gene: str, condition: str, h_do, *,
                   n_controls: int, seed: int = 0):
    """``(selection_mask, matched-random control masks)`` for a case, matched to the mask ``head``
    actually produces — a control matched to a DIFFERENT rationale is not this rationale's control."""
    sub = sample_subgraph(graph_encoder.graph, gene, gene_to_idx=graph_encoder.gene_to_idx)
    with eval_mode(graph_encoder):
        r = graph_encoder.encode_subgraph(sub, condition, h_do)
    mask = head(r["gates"], r["node_states"], None, sub)["selection_mask"]
    return mask, MatchedRandomSampler(n_controls=n_controls, seed=seed).sample(mask)


@torch.no_grad()
def rationale_contrasts(graph_encoder, decoder, head, cases, *,
                        n_controls: int = config.N_MATCHED_CONTROLS, seed: int = 0,
                        device: str = "cpu") -> dict:
    """Paired freeze-gate inputs for a fitted rationale head, one unit per case.

    Every number here is reported WITH the control it must beat; none of them is a verdict on its own.
    ``FaithfulnessTester`` forces eval internally, so this is the fixed-model contract throughout."""
    cases = [tuple(c) for c in cases]
    units = require_unique_units([_unit(c, i) for i, c in enumerate(cases)], "rationale cases")
    graph_encoder, decoder, head = graph_encoder.to(device), decoder.to(device), head.to(device)
    tester = FaithfulnessTester(graph_encoder, decoder)
    ref = untrained_like(head)
    per_case, arms = [], {k: {} for k in
                          ("suff", "rand_suff", "nec", "rand_nec", "suff_untrained")}
    for unit, (gene, condition, h_do, *_rest) in zip(units, cases):
        h_do = h_do.to(device)
        mask, controls = _case_controls(graph_encoder, decoder, head, gene, condition, h_do,
                                        n_controls=n_controls, seed=seed)
        args = (sample_subgraph(graph_encoder.graph, gene, gene_to_idx=graph_encoder.gene_to_idx),
                condition, h_do)
        dz_full = tester.delta_z(*args)                 # mask-invariant: computed once, reused per control
        ref_mask, _ = _case_controls(graph_encoder, decoder, ref, gene, condition, h_do,
                                     n_controls=1, seed=seed)
        row = {
            "unit": unit, "gene": gene, "condition": condition,
            "rationale_size": int(sum(int(m.sum()) for m in mask.values())),
            # the exact edges scored, so a report can be checked against the mask the controls were
            # matched to — a control matched to a different rationale is not this rationale's control
            "selected_edges": sorted((rel, i) for rel, m in mask.items()
                                     for i in m.nonzero().reshape(-1).tolist()),
            "suff": tester.sufficiency(*args, mask, dz_full=dz_full),
            "nec": tester.necessity(*args, mask, dz_full=dz_full),
            "suff_untrained": tester.sufficiency(*args, ref_mask, dz_full=dz_full),
            "rand_suff": _mean([tester.sufficiency(*args, c, dz_full=dz_full) for c in controls]),
            "rand_nec": _mean([tester.necessity(*args, c, dz_full=dz_full) for c in controls]),
        }
        row["dz_full_norm"] = float(dz_full.norm())
        row["informative"] = _informative(row)
        per_case.append(row)
        for k in arms:
            # the measured value stays on the record in per_case; only an INFORMATIVE case enters the
            # statistic, so a graph-blind model contributes nothing instead of contributing dust
            arms[k][unit] = row[k] if row["informative"] else None

    def spec(fit_key, control_key, higher_is_better):
        return {"fit": arms[fit_key], "control": arms[control_key],
                "higher_is_better": higher_is_better, "units": units}

    n_informative = sum(r["informative"] for r in per_case)
    if per_case and not n_informative:  # loud, not silent: a whole run of dust must be visible in the log
        print(f"[rationale] 0/{len(per_case)} cases moved dz by more than {NOISE_FLOOR_REL:g} x ||dz_full|| — "
              f"the frozen model's prediction does not read its graph edges; every contrast is UNDECIDABLE")
    return {
        "n_cases": len(cases), "n_informative": n_informative, "noise_floor_rel": NOISE_FLOOR_REL,
        "n_controls": n_controls, "per_case": per_case,
        "contrasts": {
            # sufficiency is a DISTANCE from the full prediction: smaller is better
            "sufficiency_vs_random": spec("suff", "rand_suff", False),
            "necessity_vs_random": spec("nec", "rand_nec", True),
            "sufficiency_vs_untrained": spec("suff", "suff_untrained", False),
        },
    }


def _mean(vals):
    """None on an empty control set — an average over nothing is unknown, not zero."""
    return sum(vals) / len(vals) if vals else None


def _informative(row: dict) -> bool:
    """Did ANY deletion actually move the prediction? Scaled by ``||dz_full||`` so the floor tracks the
    model's own output magnitude rather than an absolute guess. A zero-norm prediction has nothing to
    explain, so it is non-informative by definition (and would otherwise make the floor zero, promoting
    every 1e-12 residue to a measurement)."""
    scale = row["dz_full_norm"]
    if not scale > 0:
        return False
    moved = [v for v in (row["suff"], row["nec"], row["rand_suff"], row["rand_nec"]) if v is not None]
    return bool(moved) and max(moved) > NOISE_FLOOR_REL * scale
