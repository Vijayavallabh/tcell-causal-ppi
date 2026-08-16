"""A3 driver: re-run the pre-registered contrasts under the endpoints OTHER papers report.

No training. Every arm's per-row predictions for the frozen val fold are already on disk from the
screening campaign, so the same seeds, the same fold and the same paired contrasts can be re-scored
under someone else's metric for the cost of arithmetic. What this can and cannot settle is stated in
``evaluation/external_metrics``: the metric half of the commensurability hedge closes, the split half
does not.

    PYTHONPATH=src python -m tcell_pipeline.screening.rescore_external \
        --out data/results/a3_external/rescored.json

The verdict rule is the campaign's own, unchanged: a paired t on the per-seed deltas, then Bonferroni
AND Holm over the family of four simultaneous contrasts, survival requiring both. Metrics where smaller
is better are ORIENTED first, so "positive delta favours the better-named arm" holds for every endpoint
and no result can be manufactured by a metric that runs backwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tcell_pipeline import config
from tcell_pipeline.evaluation.external_metrics import ORIENTATION, external_metric_suite, oriented
from tcell_pipeline.evaluation.output_schema import prediction_path, read_predictions
from tcell_pipeline.screening.multiseed import CONTRASTS, FAMILY, apply_family_wise, paired_delta_summary
from tcell_pipeline.screening.screening import collect_targets_truth
from tcell_pipeline.training.dataset import PerturbationDataset

SEEDS = (0, 1, 2, 3, 4)


def _align(pred: dict, truth_rows: np.ndarray, truth_dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Line the prediction up with the truth BY ROW INDEX, not by position.

    Both come from the same fold and in practice the same order, but a positional join that happened to
    be right today would silently score arm A's rows against arm B's truth the first time a writer
    reorders. Rows present in one and not the other are dropped and counted by the caller."""
    order = {int(r): i for i, r in enumerate(truth_rows)}
    keep = [(i, order[int(r)]) for i, r in enumerate(pred["row_index"]) if int(r) in order]
    pi = np.array([i for i, _ in keep], dtype=np.int64)
    ti = np.array([j for _, j in keep], dtype=np.int64)
    return pred["delta_x"][pi], truth_dx[ti]


def score_arms(predictions_root: Path, arms=FAMILY, seeds=SEEDS, *, n_max: int | None = None,
               n_sample: int = 800) -> dict:
    """``{arm: {seed: {metric: value}}}`` for every stored prediction found under ``predictions_root``."""
    val = PerturbationDataset("val", n_max=n_max)
    truth = collect_targets_truth(val)
    print(f"[a3] val fold: {len(truth['row_index'])} rows x {truth['delta_x'].shape[1]} genes")
    out: dict = {}
    for arm in arms:
        for seed in seeds:
            path = prediction_path(arm, "val", seed, predictions_root)
            if not Path(path).exists():
                print(f"[a3] {arm} s{seed}: MISSING {path}")
                continue
            pred = read_predictions(path)
            dx_hat, dx_true = _align(pred, truth["row_index"], truth["delta_x"])
            dropped = len(pred["row_index"]) - len(dx_hat)
            met = external_metric_suite(dx_hat, dx_true, n_sample=n_sample, seed=seed)
            out.setdefault(arm, {})[seed] = met
            print(f"[a3] {arm:>16} s{seed} rows={len(dx_hat)}"
                  + (f" (dropped {dropped})" if dropped else "")
                  + "  " + "  ".join(f"{k}={v:+.4f}" for k, v in met.items()))
    return out


def contrast_under(scores: dict, metric: str, alpha: float = 0.05) -> dict:
    """Every pre-registered contrast under ONE metric, corrected as a family exactly as the campaign
    corrects it. Values are oriented to larger-is-better first."""
    contrasts = {}
    for key, better, worse in CONTRASTS:
        b = {s: oriented(metric, v[metric]) for s, v in scores.get(better, {}).items()}
        w = {s: oriented(metric, v[metric]) for s, v in scores.get(worse, {}).items()}
        contrasts[key] = paired_delta_summary(b, w, alpha=alpha, seeds=SEEDS)
        contrasts[key]["better"], contrasts[key]["worse"] = better, worse
    apply_family_wise(contrasts, alpha)
    return contrasts


def run(predictions_root: Path, out: Path | None = None, *, n_max: int | None = None,
        n_sample: int = 800) -> dict:
    scores = score_arms(predictions_root, n_max=n_max, n_sample=n_sample)
    report = {"predictions_root": str(predictions_root), "seeds": list(SEEDS),
              "orientation": ORIENTATION, "per_seed": scores, "contrasts": {}}

    print("\n" + "=" * 104)
    print("A3 — the pre-registered contrasts under externally-reported endpoints "
          "(oriented so + favours the first arm)")
    print("=" * 104)
    hdr = f"{'metric':>20} {'contrast':>17} {'n':>2} {'mean':>10} {'95% CI':>22} {'bonf':>7} {'holm':>7} {'survives':>9}"
    for metric in ORIENTATION:
        print(f"\n{hdr}")
        contrasts = contrast_under(scores, metric)
        report["contrasts"][metric] = contrasts
        for key, c in contrasts.items():
            ci = ("     —" if c["ci_low"] is None
                  else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]")
            mean = "    —" if c["mean"] is None else f"{c['mean']:+.4f}"
            bonf = "  —" if c["p_bonferroni"] is None else f"{c['p_bonferroni']:.4f}"
            holm = "  —" if c["p_holm"] is None else f"{c['p_holm']:.4f}"
            print(f"{metric:>20} {key:>17} {c['n']:>2} {mean:>10} {ci:>22} {bonf:>7} {holm:>7} "
                  f"{str(c['survives_family_wise']):>9}")

    survivors = [(m, k) for m, cs in report["contrasts"].items()
                 for k, c in cs.items() if c.get("survives_family_wise")]
    print("\n[a3] contrasts surviving BOTH corrections: "
          + (", ".join(f"{m}/{k}" for m, k in survivors) if survivors else "NONE"))
    print("[a3] a survivor here is not automatically a graph win — read its SIGN and its arms; the "
          "contradiction stop in the pre-registration applies to a positive, not to a negative.")

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=float))
        print(f"[a3] wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-root", default=str(config.PREDICTIONS_ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-max", type=int, default=None, help="cap val rows (quick check only)")
    ap.add_argument("--n-sample", type=int, default=800,
                    help="rows drawn for the pairwise-distance endpoints (quadratic in this)")
    a = ap.parse_args()
    run(Path(a.predictions_root), Path(a.out) if a.out else None, n_max=a.n_max, n_sample=a.n_sample)
