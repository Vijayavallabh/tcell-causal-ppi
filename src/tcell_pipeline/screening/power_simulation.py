"""A2(b): what this pipeline could have detected, simulated over the variance structure it MEASURED.

The paper's detection-floor hedge is currently a modelled MDE from five seeds, which is itself uncertain
over roughly 0.005 to 0.025. L4 replaced the model with measurements: a nested decomposition of the
paired contrast into training-seed, partition-re-draw and difficulty-level components, and the
replication pool measured the between-dataset component as well. This module turns those measurements
into design answers - how many seeds, re-draws, difficulty levels or datasets a benefit of a given size
would need to clear this project's own evidence bar.

WHY SIMULATE rather than use a formula. The bar is not "p < 0.05". It is a paired t on the per-unit
deltas, then BOTH Bonferroni and Holm over a family of four simultaneous contrasts, with survival
requiring both. That rule is code (``multiseed.apply_family_wise``), not a closed form, and the
companion contrasts' p-values enter Holm. So every replicate here builds the WHOLE family - one
contrast at the effect under test and three true nulls - and runs the pipeline's own
``paired_delta_summary`` and ``apply_family_wise`` over it. Nothing is re-derived; if the rule changes,
this follows it.

THREE DESIGNS, because "how many runs" has three different answers:

  seeds     One frozen fold, S training seeds. This is the design every headline in the paper uses.
            Only the seed component enters; level and re-draw are held fixed by the frozen fold.
  nested    Generalising to a NEW partition of the same data: L difficulty levels x R re-draws x S
            seeds, tested over the L level means. Level and re-draw stop being nuisance constants and
            become variance, so the same effect needs far more compute.
  datasets  Generalising across DATASETS: a random-effects pool over k datasets with the measured
            between-dataset tau. This is the one the field actually argues about.

This is a design analysis over a GRID of effect sizes, not observed power. Observed power - plugging in
the effect you just estimated - is a deterministic function of the p-value and says nothing new. What is
reported here is: for each effect size someone might claim, the budget needed to see it.

    PYTHONPATH=src python -m tcell_pipeline.screening.power_simulation --out data/results/l4/power.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tcell_pipeline.screening.multiseed import apply_family_wise, paired_delta_summary

# Measured inputs. Every number here is READ FROM AN ARTIFACT, not assumed:
#   seed / redraw / level  -> data/results/l4/vardecomp_{h2a,h1_vs_no_graph}.json  (L4, closed 2026-08-14)
#   tau, k                 -> data/results/replication/pooled.json                 (7-dataset DL pool)
MEASURED = {
    "h2a": {"sd_seed": 0.01067, "sd_redraw": 0.00034, "sd_level": 0.00329,
            "tau": 0.00316, "k": 7,
            "note": "typed_static - expression_only; tau^2 = 1e-05, I^2 = 39.2%, Q p = 0.13"},
    "h1_vs_no_graph": {"sd_seed": 0.00524, "sd_redraw": 0.00250, "sd_level": 0.00371,
                       "tau": None, "k": 1,
                       "note": "condition_gated - expression_only; only one qualified replication dataset"},
    "promotion_margin": {"sd_seed": None, "sd_redraw": None, "sd_level": None,
                         "tau": 0.02054, "k": 7,
                         "note": "untyped_gnn - expression_only; tau^2 = 0.000422, I^2 = 88.7% - the "
                                 "datasets disagree in SIGN, so tau dwarfs every reported effect"},
}
ALPHA = 0.05
FAMILY_SIZE = 4          # h2a, h2b, promotion_margin, h1_vs_no_graph - the pre-registered family
TARGET_POWER = 0.80
# The ladder of effects. 0.0043 is not a round number: it is the untyped arm's REAL margin on the
# reference screen at n=7, the one positive that survives both corrections, and therefore a known true
# positive to calibrate against rather than a hypothetical.
EFFECTS = (0.0043, 0.005, 0.01, 0.02, 0.03, 0.05)


def _survives(deltas: np.ndarray, companions: list, alpha: float) -> bool:
    """Run ONE replicate through the pipeline's own analysis: paired t on the target's deltas, plus one
    true-null companion per remaining family member, then Bonferroni AND Holm with survival requiring
    both.

    The family size is ``1 + len(companions)`` and nothing else. ``apply_family_wise`` counts the
    contrasts that actually carry a p-value, so a declared family size that is not backed by simulated
    companions would be a label with no effect - the companions ARE the family.

    ``paired_delta_summary`` takes two arms and differences them, so the target's deltas go in as the
    better arm against a zero worse arm: arithmetically the same one-sample t, through the same code
    path the campaign uses."""
    contrasts = {}
    for i, d in enumerate([deltas, *companions]):
        by_seed = {s: float(v) for s, v in enumerate(d)}
        contrasts[f"c{i}"] = paired_delta_summary(by_seed, dict.fromkeys(by_seed, 0.0), alpha=alpha)
    apply_family_wise(contrasts, alpha)
    return bool(contrasts["c0"].get("survives_family_wise"))


def simulate_power(delta: float, sd: float, n: int, *, n_sims: int = 2000, alpha: float = ALPHA,
                   family_size: int = FAMILY_SIZE, seed: int = 0) -> dict:
    """Power to clear BOTH corrections for a true effect ``delta`` with per-unit sd ``sd`` at ``n``
    units. Returns the estimate with its MONTE CARLO 95% CI, so 0.81 against 0.79 is not read as signal.

    ``n < 2`` is powerless BY THE RULE, not by arithmetic: a single unit is not a paired result and the
    pipeline emits no p-value for it. That is returned as 0.0 rather than crashing, because it is the
    honest answer to "what could one seed have detected"."""
    if n < 2:
        return {"n": n, "delta": delta, "sd": sd, "power": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_sims": 0, "note": "n<2 emits no p-value: a single unit is not a paired result"}
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        target = rng.normal(delta, sd, size=n)
        companions = [rng.normal(0.0, sd, size=n) for _ in range(family_size - 1)]
        hits += _survives(target, companions, alpha)
    p = hits / n_sims
    half = 1.96 * math.sqrt(max(p * (1 - p), 1e-12) / n_sims)
    return {"n": n, "delta": delta, "sd": sd, "power": p, "ci_low": max(0.0, p - half),
            "ci_high": min(1.0, p + half), "n_sims": n_sims}


def nested_sd(sd_level: float, sd_redraw: float, sd_seed: float, n_redraws: int, n_seeds: int) -> float:
    """SD of ONE difficulty level's mean when that level is measured with ``n_redraws`` partition
    re-draws and ``n_seeds`` training seeds per re-draw.

    sqrt(sd_level^2 + sd_redraw^2/R + sd_seed^2/(R*S)). The level term does NOT shrink with compute:
    buying more seeds inside a level cannot average away the fact that levels differ. That is the whole
    reason generalising across partitions costs so much more than adding seeds to one."""
    return math.sqrt(sd_level ** 2 + sd_redraw ** 2 / n_redraws + sd_seed ** 2 / (n_redraws * n_seeds))


def meta_sd(tau: float, se_within: float) -> float:
    """Per-DATASET sd entering a random-effects pool: sqrt(tau^2 + se_within^2). Same shape as the
    nested case - tau is the floor no amount of within-dataset compute removes."""
    return math.sqrt(tau ** 2 + se_within ** 2)


def required_n(delta: float, sd: float, *, target: float = TARGET_POWER, n_max: int = 400,
               n_sims: int = 2000, alpha: float = ALPHA, family_size: int = FAMILY_SIZE,
               seed: int = 0) -> int | None:
    """Smallest n whose simulated power reaches ``target``. Doubling search then bisection, so a
    three-figure answer costs ~10 simulated points rather than 400. None means "more than n_max"; that
    is reported as an inequality rather than a number, because extrapolating past the search bound
    would be inventing precision."""
    def pw(n):
        return simulate_power(delta, sd, n, n_sims=n_sims, alpha=alpha,
                              family_size=family_size, seed=seed)["power"]
    lo, hi = 2, 4
    while hi <= n_max and pw(hi) < target:
        lo, hi = hi, hi * 2
    if hi > n_max:
        return None
    while lo + 1 < hi:                       # invariant: power(lo) < target <= power(hi)
        mid = (lo + hi) // 2
        if pw(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def analytic_n(delta: float, sd: float, *, target: float = TARGET_POWER, alpha: float = ALPHA,
               family_size: int = FAMILY_SIZE) -> float:
    """The normal-approximation n, kept only as an independent cross-check on the simulation: a
    simulated answer that disagrees with this by more than the t-vs-z correction is a bug, not a
    finding. Not reported as a result."""
    from scipy import stats
    z = stats.norm.ppf(1 - alpha / family_size / 2) + stats.norm.ppf(target)
    return (z * sd / delta) ** 2


def run(out: Path | None = None, n_sims: int = 2000) -> dict:
    report: dict = {"alpha": ALPHA, "family_size": FAMILY_SIZE, "target_power": TARGET_POWER,
                    "n_sims": n_sims, "effects": list(EFFECTS), "measured": MEASURED, "designs": {}}

    print(f"[power] alpha={ALPHA}, family_size={FAMILY_SIZE} (BOTH Bonferroni and Holm), "
          f"target power={TARGET_POWER:.0%}, {n_sims} sims/point")
    print("[power] every sd below is MEASURED (L4 decomposition / 7-dataset DL pool), not assumed\n")

    # --- design 1: seeds on the frozen fold ------------------------------------------------------
    print("=== SEEDS on one frozen fold (the design every headline in the paper uses) ===")
    print(f"{'contrast':>16} {'sd_seed':>9} {'effect':>9} {'seeds@80%':>10} {'power@5':>9} {'power@7':>9}")
    for name in ("h2a", "h1_vs_no_graph"):
        sd = MEASURED[name]["sd_seed"]
        rows = []
        for d in EFFECTS:
            need = required_n(d, sd, n_sims=n_sims)
            p5 = simulate_power(d, sd, 5, n_sims=n_sims)["power"]
            p7 = simulate_power(d, sd, 7, n_sims=n_sims)["power"]
            rows.append({"effect": d, "seeds_for_80pct": need, "power_at_5": p5, "power_at_7": p7})
            print(f"{name:>16} {sd:>9.5f} {d:>9.4f} {str(need) if need else '>400':>10} "
                  f"{p5:>9.3f} {p7:>9.3f}")
        report["designs"].setdefault("seeds", {})[name] = rows
    print()

    # --- design 2: generalising to a new partition ------------------------------------------------
    print("=== NESTED: generalising to a NEW partition (L levels x R re-draws x S seeds) ===")
    print(f"{'contrast':>16} {'R':>3} {'S':>3} {'sd_level_mean':>14} {'effect':>9} {'levels@80%':>11}")
    for name in ("h2a", "h1_vs_no_graph"):
        m = MEASURED[name]
        rows = []
        for R, S in ((1, 5), (3, 5), (3, 20)):
            sd = nested_sd(m["sd_level"], m["sd_redraw"], m["sd_seed"], R, S)
            for d in (0.0043, 0.01, 0.02, 0.05):
                need = required_n(d, sd, n_sims=n_sims)
                rows.append({"redraws": R, "seeds": S, "sd_level_mean": sd, "effect": d,
                             "levels_for_80pct": need,
                             "lanes_for_80pct": None if need is None else need * R * S * 2})
                print(f"{name:>16} {R:>3} {S:>3} {sd:>14.5f} {d:>9.4f} "
                      f"{str(need) if need else '>400':>11}")
        report["designs"].setdefault("nested", {})[name] = rows
    print()

    # --- design 3: across datasets ----------------------------------------------------------------
    print("=== DATASETS: random-effects pool with the MEASURED between-dataset tau ===")
    print(f"{'contrast':>18} {'tau':>9} {'effect':>9} {'datasets@80%':>13} {'power@7':>9}")
    for name in ("h2a", "promotion_margin"):
        tau = MEASURED[name]["tau"]
        rows = []
        for d in EFFECTS:
            need = required_n(d, tau, n_sims=n_sims)
            p7 = simulate_power(d, tau, 7, n_sims=n_sims)["power"]
            rows.append({"effect": d, "datasets_for_80pct": need, "power_at_7": p7})
            print(f"{name:>18} {tau:>9.5f} {d:>9.4f} {str(need) if need else '>400':>13} {p7:>9.3f}")
        report["designs"].setdefault("datasets", {})[name] = rows

    # tau is the between-dataset sd only; a real pool also carries each dataset's own SE, so these
    # counts are FLOORS. Said out loud so the number is not quoted as exact.
    print("\n[power] the dataset counts ignore each dataset's own standard error, so they are FLOORS: "
          "a real pool needs at least this many and generally more.")
    print("[power] the level component does not shrink with compute — adding seeds inside a partition "
          "cannot average away the fact that partitions differ.")

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"[power] wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    ap.add_argument("--n-sims", type=int, default=2000,
                    help="replicates per simulated point (2000 gives a ~+/-0.018 Monte Carlo CI at 80%%)")
    a = ap.parse_args()
    run(Path(a.out) if a.out else None, a.n_sims)
