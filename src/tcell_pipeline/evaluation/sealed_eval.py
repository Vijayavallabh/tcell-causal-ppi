"""Sealed challenge evaluation (Phase 5): the ONE-SHOT, test-steward-only evaluation of the frozen H1
model on the SEQUESTERED challenge split (report §protocol / §10.7 hypothesis rule).

This runs exactly once, after screening + promotion are frozen — the challenge predictions and their metrics
are never shown to the model team before this batch, and the written result is immutable (write-once; a
second run refuses to overwrite unless ``force``). It scores EG-IPG and the supplied baselines/comparators on
the challenge fold and applies the confirmatory H1 rule on the primary endpoint (``systema``):

    H1 confirmed  ⇔  LCB_95%( ρ_EGIPG − ρ_best_baseline ) > DELTA_PRED   AND   ρ_EGIPG > ρ_perturbed_mean

``ρ`` is the per-row systema correlation, macro-averaged. The lower confidence bound is the 2.5th percentile
of a paired row-bootstrap (``N_BOOTSTRAP`` resamples) of the per-row EG-IPG − best-baseline difference. The
second clause guarantees EG-IPG beats the treatment-mean reference every headline endpoint must clear.
"""
from __future__ import annotations

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

    def evaluate(self, model, baseline_predictions: dict, *, model_name: str = "egipg",
                 split: str = "challenge", seed: int = 0, delta_pred: float = config.DELTA_PRED,
                 n_bootstrap: int = config.N_BOOTSTRAP, min_rows: int = 2, force: bool = False) -> dict:
        """Score ``model`` + ``baseline_predictions`` on the challenge fold and apply the H1 rule, write-once.

        ``baseline_predictions``: ``{name -> {'delta_z','delta_x'} | delta_z_array}`` on the SAME challenge
        rows, in dataset order (what ``collect_predictions`` / the graph scorers emit). MUST include
        ``'perturbed_mean'`` — the reference the second clause checks. Returns the result dict and writes
        ``<sealed_root>/<split>/<seed>.json`` (refuses to overwrite an existing sealed result unless
        ``force``)."""
        result_path = self.sealed_root / split / f"{seed}.json"
        if result_path.exists() and not force:
            raise FileExistsError(
                f"sealed result {result_path} already exists — sealed evaluations are write-once (the "
                f"challenge split is opened once); pass force=True only to deliberately re-seal")
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

        result = {
            "split": split, "seed": seed, "n_rows": int(len(dz_true)), "model": model_name,
            "primary_metric": "systema", "delta_pred": delta_pred, "n_bootstrap": n_bootstrap,
            "rho_egipg": egipg_mean, "rho_baselines": baseline_means,
            "best_baseline": best_name, "rho_best_baseline": baseline_means[best_name],
            "rho_perturbed_mean": baseline_means[PERTURBED_MEAN],
            "delta_vs_best": float(diff.mean()), "lcb_95": lcb, "ucb_95": ucb,
            "beats_margin": bool(beats_margin), "beats_perturbed_mean": bool(beats_perturbed),
            "h1_confirmed": h1_confirmed, "metrics": metrics,
        }
        import json
        config.write_text_atomic(json.dumps(result, indent=2, default=float), result_path)
        result["result_path"] = str(result_path)
        return result
