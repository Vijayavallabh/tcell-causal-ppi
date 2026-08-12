"""L4: split difficulty vs realisation noise vs seed noise, as a nested variance decomposition.

THE QUESTION. Point estimates of a contrast differ across difficulty levels. Is that a difficulty
effect, or is it the noise you get from re-drawing a partition at the SAME difficulty? The paper
currently declines to read a trend into the ordering; this replaces that hedge with a number.

THE MODEL is nested: delta[i,j,k] = mu + Level_i + Redraw_j(i) + Seed_k(ij).
  Level   difficulty setting (sequence-similarity threshold / family-size cap)
  Redraw  a different SPLIT_SEED at identical threshold and cap - pure partition noise
  Seed    a different training seed on one fixed partition - pure optimisation noise

WHY NOT JUST FIT A MIXED MODEL. With three or four levels and at most three re-draws, a REML fit is
estimating variance components from a handful of groups and will report an interval far tighter than
the design supports. The estimator here is method-of-moments on the nested means, which is transparent
about exactly which cells feed which component, and it REFUSES to report a component no cell can
identify rather than returning a number that looks estimated.

COVERAGE IS NOT SYMMETRIC ACROSS CONTRASTS, and this matters. The 0.80/0.10 re-draws were run with
expression_only + condition_gated, so they identify h1's within-level variance but NOT h2a's. The
0.75/0.15 re-draws were run with expression_only + typed_static and identify h2a's. Each contrast
therefore has within-level spread at one level only, and the decomposition says so instead of pooling
across contrasts as if it did not matter.

    PYTHONPATH=src python -m tcell_pipeline.screening.variance_decomposition --contrast h2a
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (level, redraw_id, results_root). A level is a threshold/cap setting; a redraw is a SPLIT_SEED at
# that identical setting. Roots are read-only here - this module never writes into a results root.
CELLS = [
    ("0.85/0.05", "s0", "data/results/screening_untyped_n7"),
    ("0.85/0.05", "s0alt", "data/results/screening"),          # same partition, earlier n=5 aggregation
    ("0.80/0.10", "s0", "data/results/screening_c080c10_h1"),
    ("0.80/0.10", "s1", "data/results/screening_c080c10_r2"),
    ("0.80/0.10", "s2", "data/results/screening_c080c10_r3"),
    ("0.75/0.15", "s0", "data/results/screening_c075c15_n5"),
    ("0.75/0.15", "s1", "data/results/screening_c075c15_r2"),
    ("0.75/0.15", "s2", "data/results/screening_c075c15_r3"),
    ("0.70/0.05", "s0", "data/results/screening_c070"),
]
# screening_untyped_n7 supersedes screening for the frozen fold (n=7 vs n=5, same partition), so the
# older root is excluded by default rather than double-counting one realisation.
DEFAULT_EXCLUDE = {"s0alt"}


def collect(contrast: str, exclude: set[str] = DEFAULT_EXCLUDE) -> dict:
    """Per-cell paired deltas for `contrast`. A cell with no usable contrast is reported, not dropped
    silently - an absent cell is design information (which arms were run), not missing data."""
    cells, absent = {}, []
    for level, redraw, root in CELLS:
        if redraw in exclude:
            continue
        f = Path(root) / "robustness_5seed.json"
        if not f.exists():
            absent.append((level, redraw, "no aggregation yet")); continue
        c = json.loads(f.read_text()).get("contrasts", {}).get(contrast)
        if not c or not c.get("n"):
            absent.append((level, redraw, "contrast not computable - arms not run here")); continue
        cells[(level, redraw)] = {"deltas": [float(x) for x in c["deltas"]],
                                  "mean": float(c["mean"]), "n": int(c["n"])}
    return {"cells": cells, "absent": absent}


def decompose(cells: dict) -> dict:
    """Method-of-moments nested variance components.

    sigma2_seed   pooled within-cell variance across every cell with n >= 2.
    sigma2_redraw variance of cell means WITHIN a level, over levels having >= 2 re-draws, with the
                  seed contribution subtracted (E[var of means] = sigma2_redraw + sigma2_seed/n).
    sigma2_level  variance of level means, with the redraw contribution subtracted likewise.
    Any component with no identifying cells is returned as None, never as 0.0 - "not estimable" and
    "estimated to be zero" are different claims and only one of them is honest here.
    """
    by_level: dict[str, list] = {}
    for (level, _), v in cells.items():
        by_level.setdefault(level, []).append(v)

    within = [(np.var(v["deltas"], ddof=1), v["n"] - 1) for v in cells.values() if v["n"] >= 2]
    s2_seed = (sum(v * df for v, df in within) / sum(df for _, df in within)) if within else None

    multi = {lv: vs for lv, vs in by_level.items() if len(vs) >= 2}
    s2_redraw, redraw_src = None, []
    if multi:
        parts = []
        for lv, vs in multi.items():
            means = [v["mean"] for v in vs]
            nbar = np.mean([v["n"] for v in vs])
            raw = float(np.var(means, ddof=1))
            parts.append(max(0.0, raw - (s2_seed or 0.0) / nbar))
            redraw_src.append(f"{lv} ({len(vs)} re-draws)")
        s2_redraw = float(np.mean(parts))

    level_means = {lv: float(np.mean([v["mean"] for v in vs])) for lv, vs in by_level.items()}
    s2_level = None
    if len(level_means) >= 2:
        rbar = np.mean([len(vs) for vs in by_level.values()])
        raw = float(np.var(list(level_means.values()), ddof=1))
        s2_level = max(0.0, raw - (s2_redraw if s2_redraw is not None else 0.0) / rbar)

    # Degrees of freedom behind each component. With three levels the level variance has 2 df, so a
    # ratio like "1.5x" is not a stable quantity and must be reported with its df attached.
    df = {"seed": int(sum(d for _, d in within)) if within else 0,
          "redraw": int(sum(len(vs) - 1 for vs in multi.values())) if multi else 0,
          "level": max(0, len(by_level) - 1)}
    return {"sigma2_seed": s2_seed, "sigma2_redraw": s2_redraw, "sigma2_level": s2_level, "df": df,
            "sd_seed": None if s2_seed is None else float(np.sqrt(s2_seed)),
            "sd_redraw": None if s2_redraw is None else float(np.sqrt(s2_redraw)),
            "sd_level": None if s2_level is None else float(np.sqrt(s2_level)),
            "level_means": level_means, "redraw_identified_by": redraw_src,
            "n_levels": len(by_level), "n_cells": len(cells)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", default="h2a", choices=["h2a", "h1_vs_no_graph", "h2b", "promotion_margin"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    got = collect(a.contrast)
    cells, absent = got["cells"], got["absent"]
    print(f"[vardecomp] contrast={a.contrast}: {len(cells)} cells over "
          f"{len({lv for lv, _ in cells})} difficulty levels")
    for (level, redraw), v in sorted(cells.items()):
        print(f"    {level}  redraw {redraw:5s}  n={v['n']}  mean={v['mean']:+.4f}")
    for level, redraw, why in absent:
        print(f"    {level}  redraw {redraw:5s}  ABSENT — {why}")
    if not cells:
        print("[vardecomp] nothing to decompose"); return 1

    d = decompose(cells)
    fmt = lambda x: "not estimable" if x is None else f"{x:.5f}"
    print("\n[vardecomp] variance components (sd in systema units):")
    print(f"    seed   (within re-draw) : sd={fmt(d['sd_seed'])}")
    print(f"    redraw (within level)   : sd={fmt(d['sd_redraw'])}"
          + (f"   identified by {', '.join(d['redraw_identified_by'])}" if d["redraw_identified_by"] else ""))
    print(f"    level  (between levels) : sd={fmt(d['sd_level'])}   over {d['n_levels']} levels")

    print(f"    df behind each: seed={d['df']['seed']}, redraw={d['df']['redraw']}, "
          f"level={d['df']['level']}")

    if d["sd_level"] is None or d["sd_redraw"] is None:
        print("\n[vardecomp] VERDICT WITHHELD: a required component is not estimable from the cells "
              "that exist. Report coverage, not a verdict.")
        return _finish(a, cells, absent, d)

    # Rank all three. Comparing the difficulty effect only against re-draw noise can call an effect
    # "detectable" while the training seed moves the number more than the knob does, which is the
    # practically relevant comparison and the one a reader will make.
    order = sorted((("seed", d["sd_seed"]), ("redraw", d["sd_redraw"]), ("level", d["sd_level"])),
                   key=lambda kv: -(kv[1] or 0))
    print("\n[vardecomp] components ranked: " + " > ".join(f"{k} ({v:.5f})" for k, v in order))
    if d["sd_level"] <= d["sd_redraw"]:
        print("[vardecomp] VERDICT: between-level spread does NOT exceed within-level (re-draw) spread."
              "\n    The difficulty knob has no detectable effect above realisation noise.")
    else:
        print(f"[vardecomp] VERDICT: between-level sd exceeds within-level by "
              f"{d['sd_level'] / d['sd_redraw']:.2f}x.")
    if d["sd_seed"] is not None and d["sd_seed"] >= d["sd_level"]:
        print(f"[vardecomp] AND NOTE: the TRAINING SEED (sd={d['sd_seed']:.5f}) moves this contrast at "
              f"least as much as the difficulty setting does (sd={d['sd_level']:.5f}). Whatever the "
              f"ratio above, a difficulty trend read off single-seed runs is not separable from seed "
              f"noise.")
    if d["df"]["level"] <= 2 or d["df"]["redraw"] <= 2:
        print(f"[vardecomp] CAUTION: {d['df']['level']} df on the level component and "
              f"{d['df']['redraw']} on the re-draw component. These ratios are indicative, not "
              f"precise; do not quote them to two significant figures.")

    return _finish(a, cells, absent, d)


def _finish(a, cells, absent, d) -> int:
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"contrast": a.contrast,
             "cells": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
             "absent": absent, **d}, indent=2))
        print(f"[vardecomp] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
