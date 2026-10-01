"""The second-metric replication: the paired five-seed family re-read on every metric each lane persisted.

RECONSTRUCTED 2026-10-01. data/results/screening_lambda0/second_metric_5seed.json (2026-07-29) was written
by a session script that was never committed. This regenerates it from the per-(arm, seed) parquets
(no re-scoring, no retraining) with the same protocol as multiseed: paired per-seed deltas, a two-sided
t test on n-1 df, Bonferroni and Holm over the four contrasts of each metric.

    PYTHONPATH=src python -m tcell_pipeline.screening.second_metric \\
        --root data/results/screening_lambda0 --out data/results/screening_lambda0/second_metric_5seed.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("systema", "pearson", "prog_cos", "centroid")
ARMS = ("expression_only", "untyped_gnn", "condition_gated", "typed_static")
CONTRASTS = (("typed_static", "expression_only"), ("condition_gated", "typed_static"),
             ("untyped_gnn", "expression_only"), ("condition_gated", "expression_only"))


def contrast(a: np.ndarray, b: np.ndarray, t_crit: float) -> dict:
    d = a - b
    n = len(d)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t = float(d.mean() / se)
    return {"deltas": d.tolist(), "mean": float(d.mean()), "sd": sd, "se": se, "t": t,
            "p_value": float(2 * stats.t.sf(abs(t), n - 1)),
            "ci_low": float(d.mean() - t_crit * se), "ci_high": float(d.mean() + t_crit * se)}


def holm(ps: list[float], alpha: float) -> list[bool]:
    """Holm step-down: the k-th smallest p is tested at alpha/(m-k), and the first failure stops the walk."""
    sig, ok = [False] * len(ps), True
    for k, i in enumerate(sorted(range(len(ps)), key=ps.__getitem__)):
        ok = ok and ps[i] < alpha / (len(ps) - k)
        sig[i] = ok
    return sig


def family(values: dict, seeds, alpha: float) -> dict:
    n = len(seeds)
    t_crit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    cs = []
    for a, b in CONTRASTS:
        c = {"contrast": f"{a} - {b}", "A": a, "B": b, **contrast(values[a], values[b], t_crit)}
        c["bonferroni_p"] = min(1.0, c["p_value"] * len(CONTRASTS))
        c["bonferroni_sig"] = c["bonferroni_p"] < alpha
        cs.append(c)
    for c, sig in zip(cs, holm([c["p_value"] for c in cs], alpha)):
        c["holm_sig"] = sig
    for c in cs:
        c["survives_both"] = c["bonferroni_sig"] and c["holm_sig"]
    return {"per_arm_mean": {a: float(values[a].mean()) for a in ARMS},
            "per_arm_sd": {a: float(values[a].std(ddof=1)) for a in ARMS}, "contrasts": cs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/results/screening_lambda0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    rows = pd.concat([pd.read_parquet(Path(a.root) / arm / f"{s}.parquet") for arm in ARMS for s in seeds])
    metrics = {}
    for m in METRICS:
        values = {arm: rows[rows["name"] == arm].set_index("seed").loc[seeds, m].to_numpy(float) for arm in ARMS}
        metrics[m] = family(values, seeds, a.alpha)
    out = {"description": ("Second-metric paired 5-seed re-analysis of the corrected (lambda_graph=0) screen. "
                           f"Re-aggregated from the per-(arm,seed) metric suite in {Path(a.root).name}/<arm>/"
                           "<seed>.parquet (no re-scoring, no retraining); same paired protocol as "
                           "robustness_5seed.json."),
           "seeds": seeds, "n": len(seeds), "alpha": a.alpha, "family_size": len(CONTRASTS),
           f"t_crit_df{len(seeds) - 1}": float(stats.t.ppf(1 - a.alpha / 2, len(seeds) - 1)),
           "validation_systema_per_arm_mean": metrics["systema"]["per_arm_mean"], "metrics": metrics}
    Path(a.out).write_text(json.dumps(out, indent=2))
    for m in METRICS:
        print(m, " ".join(f"{c['contrast']}: {c['mean']:+.4f} (Bonf {c['bonferroni_p']:.3f})"
                          for c in metrics[m]["contrasts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
