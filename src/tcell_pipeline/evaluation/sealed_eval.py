"""Sealed challenge evaluation (Phase 5): the ONE-SHOT, test-steward-only evaluation of the frozen H1
model on the SEQUESTERED challenge split (report §protocol / §10.7 hypothesis rule).

This runs exactly once, after screening + promotion are frozen — the challenge predictions and their metrics
are never shown to the model team before this batch, and the written result is immutable (write-once; a
second run refuses to overwrite unless ``force``). It scores EG-IPG and the supplied baselines/comparators on
the challenge fold and applies the confirmatory H1 rule on the primary endpoint (``systema``):

    H1 confirmed  ⇔  LCB_95%( ρ_EGIPG − ρ_best_baseline ) > DELTA_PRED   AND   ρ_EGIPG > ρ_perturbed_mean

``ρ`` is the per-row systema correlation, macro-averaged. The lower confidence bound is the 2.5th percentile
of a paired row-bootstrap (``N_BOOTSTRAP`` resamples) of the per-row EG-IPG − best-baseline difference.

**The second clause is structurally weak, and the sealed record says so.** The perturbed-mean baseline
predicts ``train_mean`` for every row, and systema subtracts ``train_mean`` from the prediction — so its
prediction becomes an all-zero constant row, which the metric's degeneracy convention maps to exactly 0.0 for
ANY data. ``ρ_perturbed_mean`` is therefore always 0.0 and the clause reduces to ``ρ_EGIPG > 0``. It is kept
because the report specifies it and it pins the treatment-mean reference into the audit trail, but it is not
an independent hurdle; the real bar is the LCB clause. ``perturbed_mean_reference_note`` in the written
result records this so a reader can't mistake it for a check that discriminated.

The best baseline is chosen by point estimate and the bootstrap then resamples rows against that FIXED
baseline. This is deliberate and conservative for a confirmatory test — EG-IPG must clear the strongest
observed baseline — though it does not propagate baseline-selection uncertainty into the interval.
ponytail: re-select the argmax inside each resample if that uncertainty ever needs to be priced in.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tcell_pipeline import config
from tcell_pipeline.evaluation import metrics as M
from tcell_pipeline.evaluation.metrics import _rowwise_pearson
from tcell_pipeline.evaluation.output_schema import write_predictions
from tcell_pipeline.screening.screening import collect_predictions

PERTURBED_MEAN = "perturbed_mean"


def _rowwise_systema(pred: np.ndarray, true: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    """Per-row systema correlation ρ_i = corr(pred_i − train_mean, true_i − train_mean) (the per-row terms
    ``systema_pert_specific_delta`` macro-averages), so the bootstrap can resample rows."""
    m = np.asarray(train_mean, dtype=np.float64).reshape(1, -1)
    return _rowwise_pearson(np.asarray(pred, np.float64) - m, np.asarray(true, np.float64) - m)


def _bootstrap_diffs(diff: np.ndarray, n_boot: int, rng, block: int = 2000) -> np.ndarray:
    """Paired row-bootstrap of the mean per-row difference, in memory-bounded blocks."""
    n = len(diff)
    out = np.empty(n_boot)
    for start in range(0, n_boot, block):
        b = min(block, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        out[start:start + b] = diff[idx].mean(1)
    return out


def _as_dz(value) -> np.ndarray:
    """A baseline entry is either a predictions dict ({'delta_z': ...}) or a bare program-delta array."""
    return np.asarray(value["delta_z"] if isinstance(value, dict) else value, dtype=np.float64)


def fold_fingerprint(dataset) -> str:
    """Stable identity of the sequestered fold: sha256 over its row_index.

    The seal is anchored to THIS, not to the ``split`` label. A label is a caller-supplied string, so keying
    the seal on it left the fold re-openable under any alias ("Challenge", "challenge_rerun", "a/../challenge")
    — the same resample-until-it-confirms hole as keying on the seed."""
    ri = np.asarray(getattr(dataset, "row_index", []), dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(np.sort(ri)).tobytes()).hexdigest()


def _safe_split_label(split: str) -> str:
    """A split label becomes a directory name, so it must be a plain component: `a/../challenge` would both
    escape ``sealed_root`` and dodge a per-directory seal check."""
    s = str(split)
    # NB: `(os.altsep or "") in s` would be `"" in s` on POSIX (altsep is None) — always True, rejecting
    # every label. Only test altsep when the platform actually defines one.
    bad_alt = bool(os.altsep) and os.altsep in s
    if not s or s != Path(s).name or s in (".", "..") or os.sep in s or bad_alt:
        raise ValueError(f"invalid split label {split!r}: must be a single path component (no separators, "
                         f"no '..'), because it names the sealed-result directory")
    return s


class SealedEvaluator:
    """Test-steward harness. Holds the sequestered challenge dataset + the training-set program mean (the
    systema reference); ``evaluate`` forwards the frozen model, scores the baselines, applies the H1 rule,
    and writes the immutable result."""

    def __init__(self, challenge_ds, train_mean, *, device: str = "cpu",
                 sealed_root: Path = config.SEALED_ROOT,
                 predictions_root: Path = config.PREDICTIONS_ROOT) -> None:
        self.challenge_ds = challenge_ds
        self.train_mean = np.asarray(train_mean, dtype=np.float64)
        self.device = device
        self.sealed_root = Path(sealed_root)
        self.predictions_root = Path(predictions_root)

    def _sealed_for_fold(self, fp: str) -> list[Path]:
        """Every prior sealed result under ``sealed_root`` (ANY split label) written for this same fold."""
        if not self.sealed_root.exists():
            return []
        out = []
        for p in sorted(self.sealed_root.rglob("*.json")):
            try:
                if json.loads(p.read_text()).get("fold_fingerprint") == fp:
                    out.append(p)
            except Exception:  # an unreadable neighbour must not block the seal check
                continue
        return out

    def _claim_fold(self, fp: str) -> Path:
        """Atomically claim this fold with an O_EXCL lock keyed on its fingerprint.

        The scan above is a check-then-write and so is racy: two stewards (or two processes) could both pass
        it and both seal. The exclusive create is the actual mutual exclusion, and being keyed on the fold —
        not on split/seed — it is immune to relabelling."""
        lock_dir = self.sealed_root / ".folds"
        config.ensure_dir(lock_dir)
        lock = lock_dir / f"{fp}.lock"
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # raises FileExistsError if held
        return lock

    def evaluate(self, model, baseline_predictions: dict, *, model_name: str = "egipg",
                 split: str = "challenge", seed: int = 0, delta_pred: float = config.DELTA_PRED,
                 n_bootstrap: int = config.N_BOOTSTRAP, min_rows: int = 2, force: bool = False) -> dict:
        """Score ``model`` + ``baseline_predictions`` on the challenge fold and apply the H1 rule, write-once.

        ``baseline_predictions``: ``{name -> {'delta_z','delta_x'} | delta_z_array}`` on the SAME challenge
        rows, in dataset order (what ``collect_predictions`` / the graph scorers emit). MUST include
        ``'perturbed_mean'`` — the reference the second clause checks. Returns the result dict and writes
        ``<sealed_root>/<split>/<seed>.json``.

        **The seal is anchored to the FOLD, not to the label.** Any prior sealed result anywhere under
        ``sealed_root`` whose ``fold_fingerprint`` matches this dataset blocks a second evaluation, so the
        fold cannot be re-opened by renaming the split, changing the seed, or re-casing the label. Pass
        ``force=True`` only to deliberately re-seal.

        Residual, by construction: an operator who points ``sealed_root`` at a fresh directory starts a
        registry with no memory of the prior seal. A filesystem control cannot bind a determined operator —
        the protocol assigns the test-steward role; this control defends against accident and label-gaming."""
        split = _safe_split_label(split)
        result_dir = self.sealed_root / split
        result_path = result_dir / f"{seed}.json"
        fp = fold_fingerprint(self.challenge_ds)
        prior = self._sealed_for_fold(fp)
        if prior and not force:
            raise FileExistsError(
                f"this challenge fold is already sealed ({', '.join(str(p) for p in prior)}) — sealed "
                f"evaluations are write-once PER FOLD (the sequestered split is opened once). The seal is "
                f"keyed on the fold's identity, so re-running under a different seed, split label or casing "
                f"cannot re-open it: that would let the confirmatory decision be resampled until it confirms. "
                f"Pass force=True only to deliberately re-seal.")
        lock = None
        if not force:
            try:
                lock = self._claim_fold(fp)   # atomic: closes the check-then-write race the scan leaves open
            except FileExistsError:
                raise FileExistsError(
                    f"this challenge fold is already claimed (fold {fp[:12]}…) — sealed evaluations are "
                    f"write-once per fold. Pass force=True only to deliberately re-seal.") from None
        try:
            return self._evaluate(model, baseline_predictions, model_name=model_name, split=split, seed=seed,
                                  delta_pred=delta_pred, n_bootstrap=n_bootstrap, min_rows=min_rows,
                                  fp=fp, result_dir=result_dir, result_path=result_path)
        except Exception:
            if lock is not None:  # a failed attempt must not brick the fold forever
                lock.unlink(missing_ok=True)
            raise

    def _evaluate(self, model, baseline_predictions: dict, *, model_name: str, split: str, seed: int,
                  delta_pred: float, n_bootstrap: int, min_rows: int, fp: str,
                  result_dir: Path, result_path: Path) -> dict:
        if PERTURBED_MEAN not in baseline_predictions:
            raise ValueError(f"baseline_predictions must include {PERTURBED_MEAN!r} (the systema reference "
                             f"the H1 rule's second clause checks)")

        pred = collect_predictions(model, self.challenge_ds, self.device)
        dz_true = pred["dz_true"]
        if len(dz_true) < min_rows:  # a 0/1-row fold gives a NaN / zero-width bootstrap CI; refuse to SEAL that
            raise ValueError(f"challenge fold has {len(dz_true)} rows (< min_rows={min_rows}) — refusing to "
                             f"write a degenerate sealed result the split can only be opened once for")
        rho_egipg = _rowwise_systema(pred["delta_z"], dz_true, self.train_mean)

        rho_baselines = {}
        for name, value in baseline_predictions.items():
            dz = _as_dz(value)
            if dz.shape != dz_true.shape:
                raise ValueError(f"baseline {name!r} delta_z {dz.shape} misaligned with challenge truth "
                                 f"{dz_true.shape} — predictions must be on the challenge fold, in dataset order")
            rho_baselines[name] = _rowwise_systema(dz, dz_true, self.train_mean)

        egipg_mean = float(rho_egipg.mean())
        baseline_means = {n: float(r.mean()) for n, r in rho_baselines.items()}
        best_name = max(baseline_means, key=baseline_means.get)  # strongest baseline by point estimate
        diff = rho_egipg - rho_baselines[best_name]              # paired per-row EG-IPG − best baseline
        rng = np.random.default_rng(seed)
        boot = _bootstrap_diffs(diff, n_bootstrap, rng)
        lcb = float(np.percentile(boot, 2.5))
        ucb = float(np.percentile(boot, 97.5))

        beats_margin = lcb > delta_pred
        beats_perturbed = egipg_mean > baseline_means[PERTURBED_MEAN]
        h1_confirmed = bool(beats_margin and beats_perturbed)

        # full metric suite for the record (systema is the endpoint; the rest are secondary)
        metrics = M.response_metric_suite(pred["delta_z"], pred["delta_x"], dz_true, pred["dx_true"],
                                          self.train_mean)
        write_predictions(pred["row_index"], pred["delta_z"], pred["delta_x"], pred["sigma"],
                          model=model_name, split=split, seed=seed, root=self.predictions_root)

        # The note must describe the value actually computed, not a hardcoded assumption: if the caller's
        # 'perturbed_mean' entry is not the train-mean broadcast, rho is NOT 0 and the clause is a real
        # comparison — an unconditional "structurally 0.0" string would then contradict the number beside it.
        rho_pm = baseline_means[PERTURBED_MEAN]
        structural_zero = abs(rho_pm) < 1e-9
        note = (
            "rho_perturbed_mean is structurally 0.0 under systema (the perturbed-mean prediction IS "
            "train_mean, which systema subtracts, leaving a constant row scored 0.0), so the "
            "'beats_perturbed_mean' clause reduces to rho_egipg > 0 and is not an independent hurdle; "
            "the binding constraint is the LCB clause."
            if structural_zero else
            f"WARNING: rho_perturbed_mean is {rho_pm:.6g}, not the structural 0.0 this reference assumes. "
            f"The supplied 'perturbed_mean' predictions are therefore NOT the training-mean broadcast (they "
            f"may have been fit on different rows, or mislabelled), so this clause is a real comparison and "
            f"the reference is not the one the report specifies. Verify the entry before trusting this seal.")
        result = {
            "split": split, "seed": seed, "n_rows": int(len(dz_true)), "model": model_name,
            "fold_fingerprint": fp,  # the seal's key: identifies the sequestered fold across any label
            "primary_metric": "systema", "delta_pred": delta_pred, "n_bootstrap": n_bootstrap,
            "rho_egipg": egipg_mean, "rho_baselines": baseline_means,
            "best_baseline": best_name, "rho_best_baseline": baseline_means[best_name],
            "rho_perturbed_mean": rho_pm,
            "delta_vs_best": float(diff.mean()), "lcb_95": lcb, "ucb_95": ucb,
            "beats_margin": bool(beats_margin), "beats_perturbed_mean": bool(beats_perturbed),
            "h1_confirmed": h1_confirmed, "metrics": metrics,
            "perturbed_mean_is_structural_zero": bool(structural_zero),
            "perturbed_mean_reference_note": note,
        }
        config.write_text_atomic(json.dumps(result, indent=2, default=float), result_path)
        result["result_path"] = str(result_path)
        return result
