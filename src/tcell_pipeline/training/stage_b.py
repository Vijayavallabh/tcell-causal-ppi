"""Stage-B calibration FIT loop over a FROZEN Stage-A model (feat-008 §a; walkthrough §8.1, §8.3).

Stage A optimises the H1 predictor; nothing in ``StageALoss`` reads ``out["sigma"]``, so the decoder's
uncertainty head never receives a gradient there and is still at init when Stage B begins. Stage B
fits exactly that head — and nothing else — with ``StageBCalibrationLoss`` (Gaussian NLL):

    fit_calibration(model, train_ds, val_ds) -> history / best checkpoint / train+val NLL

Two properties are ASSERTED, not intended:

1. STAGE-A STAYS FROZEN. Every parameter outside the calibration head is snapshotted before the fit
   and compared after it; any movement (a stray optimiser group, weight decay, an in-place write)
   raises. ``requires_grad`` is restored to the caller's state afterwards, so the model is not left
   silently frozen.
2. NO VAL STATISTIC ENTERS THE FIT. Early stopping and the best checkpoint key on the TRAIN NLL. The
   val NLL is computed and reported every epoch, but it never selects a parameter or an epoch, so the
   fitted head is a pure function of (train set, hyper-parameters).

The backbone is eval + frozen for the whole fit, so its outputs are CONSTANTS: one forward pass caches
the decoder's inputs and the epoch loop never re-runs the graph encoder again.

``calibration_contrasts`` builds the paired inputs the near-null-signal freeze gate consumes — the
fitted head against two controls it must beat to be worth freezing:

    constant_sigma   the best per-program CONSTANT sigma, fit on TRAIN by the same objective. A head
                     that cannot beat it bought nothing with its input-dependence — this is the
                     collapse-to-a-constant detector.
    permuted_sigma   the fitted sigma with its rows permuted: same marginal distribution, row->input
                     pairing destroyed. The matched-random analogue for a per-row quantity. A collapsed
                     head is IDENTICAL to this control, so its advantage is exactly zero — undecidable,
                     never a spurious win.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tcell_pipeline import config
from tcell_pipeline.training.dataset import PerturbationDataset
from tcell_pipeline.training.losses import StageBCalibrationLoss

_VAR_EPS = 1e-6  # the eps StageBCalibrationLoss floors the variance with; shared so the two agree


@contextlib.contextmanager
def eval_mode(*modules):
    """The fixed-model contract, for every path that re-runs a frozen model: dropout/DropEdge OFF, the
    caller's train/eval state restored. Without it a cached "constant" backbone output is a fresh random
    draw. ``FaithfulnessTester`` enforces the same thing for its deletion tests."""
    was = [(m, m.training) for m in modules]
    try:
        for m, _ in was:
            m.eval()
        yield
    finally:
        for m, t in was:
            m.train(t)


def stage_b_ckpt_dir(seed: int, root: Path = config.CHECKPOINTS_ROOT) -> Path:
    """Seed-namespaced Stage-B checkpoint directory. Both Stage-B heads write FIXED filenames
    (``stage_b_calibration.pt``, ``stage_b_rationale_head.pt``), so without the seed in the directory a
    multi-seed sweep (``config.N_FINAL_SEEDS``) silently overwrites the earlier seeds' artifacts with no
    error. Stage A already learned this — screening namespaces as ``<root>/<name>/<seed>/ckpt``."""
    return Path(root) / "stage_b" / str(int(seed))


def calibration_parameters(model) -> list:
    """The ONLY tensors Stage B may move: the decoder's uncertainty (sigma) head."""
    return list(model.decoder.uncertainty.parameters())


def frozen_snapshot(model, trainable) -> dict:
    """Clone every parameter that must NOT move (identity-matched, so a shared/tied tensor is excluded
    exactly once)."""
    ids = {id(p) for p in trainable}
    return {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in ids}


def assert_backbone_frozen(model, snapshot: dict) -> None:
    """Raise if any snapshotted parameter moved by so much as one bit."""
    moved = [n for n, p in model.named_parameters()
             if n in snapshot and not torch.equal(p.detach().cpu(), snapshot[n].cpu())]
    if moved:
        raise RuntimeError(f"Stage-A weights moved during the Stage-B fit — the backbone was not frozen: "
                           f"{', '.join(moved)}")


def per_row_nll(dz_hat, dz_true, sigma, eps: float = _VAR_EPS) -> torch.Tensor:
    """Per-row Gaussian NLL (mean over programs), (B,). The fit objective is its mean, so the statistic
    the gate tests and the loss the fit minimises are the same quantity — see
    test_per_row_nll_matches_the_fitted_objective."""
    return F.gaussian_nll_loss(dz_hat, dz_true, sigma.pow(2), full=False, eps=eps,
                               reduction="none").mean(dim=1)


@torch.no_grad()
def _frozen_pass(model, loader, device: str) -> list:
    """Cache ``(h_do, h_graph, dz_true, row_ids)`` per batch. Valid only because the backbone is eval
    (DropEdge off) and frozen for the whole fit, which makes these outputs constants."""
    cache = []
    for batch, targets, conditions, dz_true, _dx, rows in loader:
        out = model(batch, targets, conditions)
        h_graph = out["h_graph"]
        cache.append((out["h_do"].detach(), None if h_graph is None else h_graph.detach(),
                      dz_true.to(device), list(rows)))
    return cache


def frozen_caches(model, fit_ds, gate_ds=None, *, batch_size: int = config.BATCH_SIZE,
                  device: str = "cpu") -> tuple:
    """``(fit_cache, gate_cache)`` — the frozen backbone pass, which is the expensive part of Stage B
    (one graph encode per row). Build it ONCE and hand it to both ``fit_calibration`` and
    ``calibration_contrasts``: they need the same constants, and recomputing doubles the run."""
    loader = lambda ds: DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   collate_fn=PerturbationDataset.collate)
    with eval_mode(model):
        return (_frozen_pass(model, loader(fit_ds), device),
                _frozen_pass(model, loader(gate_ds), device) if gate_ds is not None else None)


def _nll_over(model, cache, loss) -> torch.Tensor:
    """Mean calibration NLL over a cached split, differentiable through the sigma head only."""
    totals, n = None, 0
    for h_do, h_graph, dz_true, rows in cache:
        out = model.decoder(h_do, h_graph)
        v = loss(out["delta_z"], dz_true, out["sigma"]) * len(rows)
        totals = v if totals is None else totals + v
        n += len(rows)
    return totals / max(n, 1)


def fit_calibration(model, train_ds, val_ds=None, *, caches=None, loss=None, lr: float = config.LR,
                    weight_decay: float = config.WEIGHT_DECAY, max_epochs: int = config.MAX_EPOCHS,
                    patience: int = config.EARLY_STOP_PATIENCE, batch_size: int = config.BATCH_SIZE,
                    device: str = "cpu", ckpt_dir: Path = config.CHECKPOINTS_ROOT,
                    log_dir: Path = config.LOGS_ROOT) -> dict:
    """Fit the calibration head on TRAIN over a frozen Stage-A model. ``val_ds`` is REPORTED, never fit.

    There is no ``seed``: the backbone is in eval and the cached batches are consumed in a fixed order,
    so the fit is deterministic — a seed knob here would be decoration."""
    if len(train_ds) == 0:
        raise ValueError("calibration training set is empty (0 examples) — check the split role / n_max")
    model = model.to(device)
    loss = (loss or StageBCalibrationLoss()).to(device)
    trainable = calibration_parameters(model)
    snapshot = frozen_snapshot(model, trainable)
    was_training = model.training
    grad_state = [(p, p.requires_grad) for p in model.parameters()]
    collate = PerturbationDataset.collate
    history, best, best_path, wait = [], float("inf"), None, 0
    best_epoch, best_state, best_metrics = None, None, {}
    try:
        model.eval()                                   # DropEdge off: the cached backbone outputs are constants
        for p in model.parameters():
            p.requires_grad_(False)
        for p in trainable:
            p.requires_grad_(True)
        train_cache, val_cache = caches if caches is not None else \
            frozen_caches(model, train_ds, val_ds, batch_size=batch_size, device=device)
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        for epoch in range(max_epochs):
            opt.zero_grad()
            train_nll = _nll_over(model, train_cache, loss)
            train_nll.backward()
            opt.step()
            with torch.no_grad():
                after = float(_nll_over(model, train_cache, loss))
                val_nll = float(_nll_over(model, val_cache, loss)) if val_cache else None
            history.append({"epoch": epoch, "train_nll": after, "val_nll": val_nll})
            config.write_text_atomic(json.dumps(history, indent=2),
                                     Path(log_dir) / "stage_b_calibration_history.json")
            # early stop + best checkpoint on the TRAIN NLL: a val-keyed rule would let the evaluation
            # split pick the parameters, which is the leak this loop exists to avoid
            if after < best - 1e-6:
                best, wait, best_epoch = after, 0, epoch
                best_metrics = {"train_nll": after, "val_nll": val_nll}
                best_state = {k: v.detach().clone() for k, v in model.decoder.uncertainty.state_dict().items()}
                best_path = _save(model, ckpt_dir, epoch, after, val_nll)
            else:
                wait += 1
                if wait >= patience:
                    break
        # Early stopping ENDS on a worse epoch than the one it checkpointed, so without this the model
        # left in memory is not the artifact on disk — and the freeze gate scores the live model. Every
        # caller would have to remember to reload; restoring here makes in-memory == best_ckpt for all
        # of them. (The calibration head is excluded from `snapshot`, so this cannot trip the freeze
        # assert; the backbone is untouched.)
        if best_state is not None:
            model.decoder.uncertainty.load_state_dict(best_state)
    finally:
        for p, req in grad_state:
            p.requires_grad_(req)
        model.train(was_training)
    assert_backbone_frozen(model, snapshot)
    # the reported metrics describe the RESTORED (best) epoch, not the last one that ran
    return {"history": history, "epochs_run": len(history), "best_ckpt": str(best_path) if best_path else None,
            "best_epoch": best_epoch, "train_nll": best_metrics.get("train_nll"),
            "val_nll": best_metrics.get("val_nll"), "last_train_nll": history[-1]["train_nll"] if history else None}


def _save(model, ckpt_dir: Path, epoch: int, train_nll: float, val_nll) -> Path:
    config.ensure_dir(Path(ckpt_dir))
    path = Path(ckpt_dir) / "stage_b_calibration.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"calibration": model.decoder.uncertainty.state_dict(), "epoch": epoch,
                "train_nll": train_nll, "val_nll": val_nll}, tmp)
    tmp.replace(path)
    return path


def _permuted_control(hat, true, sigma, *, n_permutations: int, seed: int) -> torch.Tensor:
    """Per-row NLL with sigma RE-PAIRED to other rows — same marginal distribution, row->input pairing
    destroyed — averaged over ``n_permutations`` draws.

    One draw is itself a random variable, and a control that moves with its own seed is not a property
    of the fitted head: on the real fold a single-draw control moved this contrast's raw p from 0.2494
    to 0.1506 between two runs differing only in the seed. The matched-random control on the rationale
    side already averages over its draws; this is the same discipline."""
    g = torch.Generator().manual_seed(seed)
    n = sigma.shape[0]
    draws = [per_row_nll(hat, true, sigma[torch.randperm(n, generator=g)]) for _ in range(n_permutations)]
    # float64 accumulation: averaging K IDENTICAL float32 values must return that value EXACTLY, or a
    # collapsed head — whose permuted control is by definition its own fit — picks up ~1e-08 of rounding
    # and the contrast stops being exactly degenerate. That would hand a paired t a spread to work with
    # and turn "undecidable" into a p-value computed on float32 noise. Measured: one row of the fixture
    # drifted 5.96e-08 with a float32 mean.
    return torch.stack(draws).to(torch.float64).mean(dim=0)


def calibration_contrasts(model, train_ds, val_ds, *, caches=None, batch_size: int = config.BATCH_SIZE,
                          device: str = "cpu", seed: int = 0, n_permutations: int = 8) -> dict:
    """Paired freeze-gate inputs for a fitted calibration head, evaluated on ``val_ds``.

    Both controls are as blind to the evaluation rows as the head is: the constant sigma is fit on
    TRAIN, and the permutation only reorders the head's own val predictions. Every contrast is keyed by
    dataset row id, so the fit and its control are compared on exactly the same rows."""
    model = model.to(device)
    was_training = model.training
    model.eval()
    collate = PerturbationDataset.collate
    try:
        train_cache, val_cache = caches if caches is not None else \
            frozen_caches(model, train_ds, val_ds, batch_size=batch_size, device=device)
        with torch.no_grad():
            tr_hat, tr_true, _, _, _ = _decode_cache(model, train_cache)
            hat, true, sigma, rows, lam = _decode_cache(model, val_cache)
    finally:
        model.train(was_training)

    if not rows:
        raise ValueError("the evaluation split is empty (0 rows) — there is nothing to gate")
    if len(set(rows)) != len(rows):
        # duplicate row ids would collapse into one dict key and silently shrink n — an underpowered
        # gate that still reports a clean verdict
        raise ValueError(f"evaluation rows are not uniquely identified ({len(rows)} rows, "
                         f"{len(set(rows))} distinct row_index values) — the paired contrast would drop rows")

    # closed-form best CONSTANT sigma per program under the same Gaussian NLL: s_k^2 = mean_i r_ik^2,
    # floored at the loss's own variance eps so a zero-residual program cannot make the control infinite
    const = (tr_hat - tr_true).pow(2).mean(dim=0).clamp_min(_VAR_EPS).sqrt()
    fit = per_row_nll(hat, true, sigma)
    permuted = _permuted_control(hat, true, sigma, n_permutations=n_permutations, seed=seed)

    def by_row(v) -> dict:
        return {r: float(x) for r, x in zip(rows, v)}

    return {
        "n_train": sum(len(c[3]) for c in train_cache), "n_val": len(rows),
        "n_permutations": n_permutations,
        "constant_sigma": [float(s) for s in const],
        # the decoder's graph-vs-expression mixture on the gate rows: lambda ~ 0 means the frozen model
        # ignores its graph, which is the diagnostic that explains a rationale measuring only dust
        "mean_lambda": float(lam.mean()), "min_lambda": float(lam.min()), "max_lambda": float(lam.max()),
        "contrasts": {
            "vs_constant_sigma": {"fit": by_row(fit), "higher_is_better": False, "units": list(rows),
                                  "control": by_row(per_row_nll(hat, true, const.expand_as(sigma)))},
            "vs_permuted_sigma": {"fit": by_row(fit), "higher_is_better": False, "units": list(rows),
                                  "control": by_row(permuted)},
        },
    }


def _decode_cache(model, cache) -> tuple:
    """Flatten a cached split to ``(dz_hat, dz_true, sigma, row_ids, lambda)`` under the CURRENT head."""
    if not cache:  # an empty split has nothing to concatenate; the caller reports it, torch.cat crashes
        return None, None, None, [], None
    hats, trues, sigmas, rows, lams = [], [], [], [], []
    for h_do, h_graph, dz_true, r in cache:
        out = model.decoder(h_do, h_graph)
        hats.append(out["delta_z"])
        sigmas.append(out["sigma"])
        lams.append(out["lambda"])
        trues.append(dz_true)
        rows.extend(r)
    return torch.cat(hats), torch.cat(trues), torch.cat(sigmas), rows, torch.cat(lams)
