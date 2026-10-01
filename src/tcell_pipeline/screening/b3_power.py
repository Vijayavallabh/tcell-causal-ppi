"""B3: could a second scrambled-control rung settle whether the wrong-prior damage SCALES?

Answer, computed from the landed ladder rather than assumed: no, not at four seeds. This module exists
so the four numbers the paper now quotes in app:floor are re-derivable from the artifact instead of
being a claim in a commit message.

THE QUESTION. A2(a) measured the scrambled control's increment over the zero point at ONE injection
size: -0.0050 at delta=0.40. Two hypotheses fit that single point equally well - the damage is
PROPORTIONAL to the injected magnitude, or it is CONSTANT in it - and they differ by a factor of four at
delta=0.10, which is where the proposed second rung would sit.

WHY IT CANNOT BE RUN CHEAPLY. Proportional damage predicts -0.00125 at delta=0.10. The paired
increment's spread is measured, not guessed: this reads it off the landed rungs' own confidence
intervals, so the minimum detectable effect comes from the same lanes the new rung would join.

    PYTHONPATH=src python -m tcell_pipeline.screening.b3_power --out data/results/a2_ladder/b3_power.json

This is a SENSITIVITY analysis (what could this design detect?), not observed power: it is computed from
the design and the measured variance, never from a p-value the experiment produced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

LADDER_REPORT = "data/results/a2_ladder/floor.json"
CONTROL = "permuted_d400"
TARGET_DELTA = 0.10
POWER = 0.80


def sd_from_ci(mean_ci: dict) -> float | None:
    """Recover the per-seed SD of a paired increment from its reported t interval. The report persists
    the interval, not the raw deltas, and half-width = t_{.975,n-1} * sd / sqrt(n) inverts exactly."""
    if mean_ci.get("ci_low") is None or mean_ci.get("n", 0) < 2:
        return None
    n = mean_ci["n"]
    half = (mean_ci["ci_high"] - mean_ci["ci_low"]) / 2
    return half / stats.t.ppf(0.975, n - 1) * np.sqrt(n)


def mde(sd: float, n: int, alpha: float = 0.05, power: float = POWER) -> float:
    """Two-sided paired-t minimum detectable effect. Uses the t quantile at the design's own df, not z:
    at n=4 the difference is not a rounding detail."""
    df = n - 1
    return (sd / np.sqrt(n)) * (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df))


def seeds_needed(sd: float, target: float, alpha: float = 0.05, power: float = POWER,
                 cap: int = 500) -> int | None:
    return next((n for n in range(3, cap) if mde(sd, n, alpha, power) <= abs(target)), None)


def run(report: str = LADDER_REPORT, out: Path | None = None) -> dict:
    rep = json.load(open(report))
    inc = rep["post_hoc_increment_over_zero"]
    ctrl = inc[CONTROL]
    n = ctrl["n"]
    observed = ctrl["mean"]

    sd_ctrl = sd_from_ci(ctrl)
    real = {k: sd_from_ci(v) for k, v in inc.items() if not v["permuted"]}
    real = {k: v for k, v in real.items() if v is not None}
    sd_real = float(np.median(list(real.values())))

    # The enlarged family: the six existing ladder conditions plus the proposed rung.
    m = len(inc) + 1
    predictions = {"proportional": observed * TARGET_DELTA / ctrl["delta"], "constant": observed}

    res = {"control": CONTROL, "n": n, "observed_increment": observed,
           "observed_delta": ctrl["delta"], "target_delta": TARGET_DELTA,
           "sd_control": sd_ctrl, "sd_real_rungs_median": sd_real,
           "sd_real_rungs": real, "family_size_if_added": m, "power": POWER,
           "predictions_at_target_delta": predictions,
           "mde": {"control_sd_uncorrected": mde(sd_ctrl, n),
                   "real_sd_uncorrected": mde(sd_real, n),
                   "control_sd_bonferroni": mde(sd_ctrl, n, alpha=0.05 / m),
                   "real_sd_bonferroni": mde(sd_real, n, alpha=0.05 / m)},
           "seeds_needed_for_proportional": {
               "control_sd": seeds_needed(sd_ctrl, predictions["proportional"]),
               "real_sd": seeds_needed(sd_real, predictions["proportional"])}}

    print(f"[b3] control {CONTROL}: increment {observed:+.4f} at delta={ctrl['delta']}, n={n}")
    print(f"[b3] measured per-seed sd: control {sd_ctrl:.5f}, injected rungs median {sd_real:.5f}")
    print(f"[b3] at delta={TARGET_DELTA} the two hypotheses predict "
          f"{predictions['proportional']:+.5f} (proportional) vs {predictions['constant']:+.5f} (constant)")
    for k, v in res["mde"].items():
        print(f"[b3]   MDE n={n}, {k:>26}: {v:.5f}")
    print(f"[b3] seeds needed to detect the PROPORTIONAL prediction at {POWER:.0%} power: "
          f"{res['seeds_needed_for_proportional']}")

    worst = max(res["mde"].values())
    if abs(predictions["proportional"]) < min(res["mde"].values()):
        print(f"[b3] VERDICT: a rung at delta={TARGET_DELTA} CANNOT distinguish proportional damage from "
              f"zero at n={n} under ANY measured spread. Do not run it; report the claim at the magnitude "
              f"it was measured.")
        res["verdict"] = "underpowered_do_not_run"
    elif abs(predictions["constant"]) < worst:
        print("[b3] VERDICT: constant damage would be detectable uncorrected but NOT under the "
              "family-wise correction this project requires.")
        res["verdict"] = "underpowered_under_correction"
    else:
        res["verdict"] = "feasible"

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2, default=float))
        print(f"[b3] wrote {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=LADDER_REPORT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.report, Path(a.out) if a.out else None)
