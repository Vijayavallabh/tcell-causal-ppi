"""A4: bound the architecture-search claim against the detection floor A2(b) measured.

The paper already says the 14-cell architecture search was the wrong SHAPE: every cell varied how the
typed encoder passes messages and none asked whether the typing belonged, so it excluded its own
premise. This asks the other question. Was it powered to find anything at all?

Each cell is ONE inner-holdout run at ONE seed. This project's own rule emits no p-value at n=1 (a
single seed is not a paired result), so no cell is individually decidable. What can be decided is
whether the search's whole spread is distinguishable from re-seeding a single configuration, and
whether its best cell clears the floor measured in ``power_simulation``.

    PYTHONPATH=src python -m tcell_pipeline.screening.arch_search_power \
        --out data/results/a2_power/arch_search_bound.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from tcell_pipeline.screening.power_simulation import MEASURED, simulate_power

ARCH_ROOT = "data/results/arch_search"
BASELINE_ARM = "expression_only"
# Expected range of n independent standard normals, E[R_n]/sigma. Tabulated (Harter 1960); only the
# entries this report needs. Exact enough for a "the spread is smaller than noise" comparison.
_EXPECTED_RANGE = {5: 2.326, 10: 3.078, 14: 3.407, 15: 3.472, 20: 3.735}


def load_cells(root: str = ARCH_ROOT) -> list[dict]:
    return [json.load(open(f)) for f in sorted(glob.glob(f"{root}/*.json"))]


def expected_range(n: int, sd: float) -> float:
    """What range ``n`` draws of pure seed noise would span, so an observed spread can be compared to
    it. Interpolated between tabulated points; the ratio grows slowly, so this is not delicate."""
    keys = sorted(_EXPECTED_RANGE)
    if n <= keys[0]:
        return _EXPECTED_RANGE[keys[0]] * sd
    if n >= keys[-1]:
        return _EXPECTED_RANGE[keys[-1]] * sd
    lo = max(k for k in keys if k <= n)
    hi = min(k for k in keys if k >= n)
    if lo == hi:
        return _EXPECTED_RANGE[lo] * sd
    w = (n - lo) / (hi - lo)
    return (_EXPECTED_RANGE[lo] * (1 - w) + _EXPECTED_RANGE[hi] * w) * sd


def run(root: str = ARCH_ROOT, out: Path | None = None, n_sims: int = 4000) -> dict:
    cells = load_cells(root)
    base = [c for c in cells if c["arm"] == BASELINE_ARM]
    if not base:
        raise RuntimeError(f"no {BASELINE_ARM} cell under {root}: nothing to measure against")
    b = float(base[0]["systema"])
    deltas = {c["cell_id"]: float(c["systema"]) - b for c in cells}
    best_id = max(deltas, key=deltas.get)
    worst_id = min(deltas, key=deltas.get)
    best, worst = deltas[best_id], deltas[worst_id]
    spread = best - worst

    # The seed sd on the frozen fold, re-derived elsewhere from the landed per-seed parquets. h2a's is
    # used because the search's cells are condition_gated / untyped variants scored on the same fold.
    sd = MEASURED["h2a"]["sd_seed_frozen"]
    exp = expected_range(len(cells), sd)
    seeds_needed = _seeds_for(best, sd, n_sims)

    report = {
        "n_cells": len(cells), "baseline_systema": b, "baseline_cell": base[0]["cell_id"],
        "best_cell": best_id, "best_delta": best, "worst_cell": worst_id, "worst_delta": worst,
        "observed_spread": spread, "seed_sd_frozen": sd,
        "expected_spread_from_seed_noise_alone": exp,
        "spread_within_seed_noise": spread < exp,
        "seeds_that_would_be_needed_for_the_best_cell": seeds_needed,
        "epochs_run": sorted({int(c["epochs_run"]) for c in cells}),
        "seeds_per_cell": 1,
        "deltas": deltas,
    }

    print(f"[a4] {len(cells)} cells, baseline {BASELINE_ARM} systema {b:.4f}, ONE seed per cell")
    print(f"[a4] best  {best_id}  {best:+.4f}")
    print(f"[a4] worst {worst_id}  {worst:+.4f}")
    print(f"[a4] observed spread {spread:.4f}")
    print(f"[a4] spread {len(cells)} draws of PURE SEED NOISE would give (sd {sd}): {exp:.4f}")
    print(f"[a4] -> the search's entire spread is "
          f"{'INSIDE' if spread < exp else 'outside'} what re-seeding one configuration produces")
    print(f"[a4] the best cell ({best:+.4f}) would need {seeds_needed} paired seeds to clear this "
          f"project's correction rule at 80% power; every cell was run at ONE, where the rule emits "
          f"no p-value at all")
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"[a4] wrote {out}")
    return report


def _seeds_for(delta: float, sd: float, n_sims: int) -> int | None:
    if delta <= 0:
        return None
    from tcell_pipeline.screening.power_simulation import required_n
    return required_n(delta, sd, n_sims=n_sims)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ARCH_ROOT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-sims", type=int, default=4000)
    a = ap.parse_args()
    run(a.root, Path(a.out) if a.out else None, a.n_sims)
