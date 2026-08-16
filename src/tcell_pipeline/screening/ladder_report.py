"""A2(a) report: read the injected-signal rungs and name the measured detection floor.

The rule is fixed in Amendment 6.7 and is applied here without discretion:

  primary        promotion_margin = untyped_gnn - expression_only on systema, paired per seed, n=4
  family         the six injected conditions together, m=6, Bonferroni AND Holm, survival needs both
  the floor      the smallest delta that clears both AND is cleared by every larger rung
  monotonicity   a rung that clears while a larger one does not is reported as a RED FLAG, not a floor
  the control    if the permuted rung clears, no floor is reported at all - the arms would be
                 responding to injected magnitude rather than to graph structure

The delta=0 row is read off the landed REFERENCE lanes and is the ladder's zero point, not a member of
the family: at delta=0 the data is the untouched screen, where the untyped arm already wins under both
corrections, so it cannot serve as a null (Amendment 6.4).

    PYTHONPATH=src python -m tcell_pipeline.screening.ladder_report \
        --out data/results/a2_ladder/floor.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from tcell_pipeline.screening.multiseed import apply_family_wise, paired_delta_summary
from tcell_pipeline.screening.screening import EXPRESSION_ONLY, PRIMARY_METRIC, UNTYPED_GNN

LADDER_ROOT = "data/results/a2_ladder"
REFERENCE_ROOT = "data/results/screening_untyped_n7"     # the delta=0 zero point, already landed
SEEDS = (0, 1, 2, 3)


def _metric_by_seed(root: Path, arm: str, seeds=SEEDS) -> dict:
    """{seed: systema} for the completed, finite lanes of one arm. A lane that is missing or did not
    complete is absent, which shrinks n through ``paired_delta_summary`` rather than silently."""
    out = {}
    for s in seeds:
        p = Path(root) / arm / f"{s}.parquet"
        if not p.exists():
            continue
        row = pd.read_parquet(p).iloc[0].to_dict()
        if row.get("status") != "completed":
            continue
        v = row.get(PRIMARY_METRIC)
        if v is None or not float(v) == float(v):       # NaN check without importing math
            continue
        out[s] = float(v)
    return out


def _delta_of(name: str) -> float | None:
    """d020 -> 0.02, permuted_d400 -> 0.40. The directory name is the record of which rung this is."""
    m = re.search(r"d(\d{3})$", name)
    return int(m.group(1)) / 1000.0 if m else None


def collect(ladder_root: str = LADDER_ROOT, seeds=SEEDS) -> dict:
    rungs = {}
    for d in sorted(Path(ladder_root).glob("*d[0-9][0-9][0-9]")):
        if not d.is_dir():
            continue
        rungs[d.name] = {"path": str(d), "delta": _delta_of(d.name),
                         "permuted": d.name.startswith("permuted"),
                         "better": _metric_by_seed(d, UNTYPED_GNN, seeds),
                         "worse": _metric_by_seed(d, EXPRESSION_ONLY, seeds)}
    return rungs


def run(ladder_root: str = LADDER_ROOT, out: Path | None = None, seeds=SEEDS,
        alpha: float = 0.05) -> dict:
    rungs = collect(ladder_root, seeds)
    if not rungs:
        print(f"[ladder] no rungs under {ladder_root} — nothing has landed yet")
        return {"rungs": {}, "floor": None, "status": "no lanes"}

    contrasts = {name: paired_delta_summary(r["better"], r["worse"], alpha=alpha, seeds=seeds)
                 for name, r in rungs.items()}
    m = apply_family_wise(contrasts, alpha)

    print(f"[ladder] {len(rungs)} conditions, family size {m} (Bonferroni AND Holm, both required)")
    print(f"{'rung':>16} {'delta':>6} {'n':>2} {'mean':>10} {'95% CI':>22} {'bonf':>7} {'holm':>7} {'survives':>9}")
    zero = _zero_point(seeds, alpha)
    if zero:
        print(f"{'reference':>16} {0.0:>6.3f} {zero['n']:>2} {zero['mean']:>+10.4f} "
              f"[{zero['ci_low']:+.4f}, {zero['ci_high']:+.4f}]".rjust(0)
              + "   (zero point, NOT in the family — Amendment 6.4)")
    for name in sorted(rungs, key=lambda k: (rungs[k]["permuted"], rungs[k]["delta"] or 0)):
        c, r = contrasts[name], rungs[name]
        ci = "     —" if c["ci_low"] is None else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        mean = "    —" if c["mean"] is None else f"{c['mean']:+.4f}"
        bonf = "  —" if c["p_bonferroni"] is None else f"{c['p_bonferroni']:.4f}"
        holm = "  —" if c["p_holm"] is None else f"{c['p_holm']:.4f}"
        print(f"{name:>16} {r['delta']:>6.3f} {c['n']:>2} {mean:>10} {ci:>22} {bonf:>7} {holm:>7} "
              f"{str(c['survives_family_wise']):>9}")

    verdict = _verdict(rungs, contrasts)
    for line in verdict["notes"]:
        print(f"[ladder] {line}")

    report = {"family_size": m, "alpha": alpha, "seeds": list(seeds), "zero_point": zero,
              "contrasts": contrasts, "rungs": {k: {kk: vv for kk, vv in v.items() if kk != "path"}
                                                for k, v in rungs.items()}, **verdict}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=float))
        print(f"[ladder] wrote {out}")
    return report


def _zero_point(seeds, alpha: float) -> dict | None:
    """delta=0 from the LANDED reference lanes. Not re-run, and not a family member."""
    root = Path(REFERENCE_ROOT)
    if not root.exists():
        return None
    s = paired_delta_summary(_metric_by_seed(root, UNTYPED_GNN, seeds),
                             _metric_by_seed(root, EXPRESSION_ONLY, seeds), alpha=alpha, seeds=seeds)
    return s if s["n"] else None


def _verdict(rungs: dict, contrasts: dict) -> dict:
    notes, floor = [], None
    real = sorted((n for n in rungs if not rungs[n]["permuted"]), key=lambda k: rungs[k]["delta"])
    control = [n for n in rungs if rungs[n]["permuted"]]

    incomplete = [n for n in rungs if contrasts[n]["n"] < len(SEEDS)]
    if incomplete:
        notes.append(f"INCOMPLETE at n<{len(SEEDS)}: {incomplete} — read these as preliminary, with "
                     f"their n (rail 5)")

    control_clears = any(contrasts[n].get("survives_family_wise") and (contrasts[n]["mean"] or 0) > 0
                         for n in control)
    if control_clears:
        notes.append("*** THE PERMUTED CONTROL CLEARS CORRECTION. No floor is reported (Amendment "
                     "6.7): the arms are responding to injected magnitude, not to graph structure. ***")
        return {"floor": None, "floor_status": "control_failed", "notes": notes,
                "control_clears": True}

    clears = [n for n in real if contrasts[n].get("survives_family_wise")
              and (contrasts[n]["mean"] or 0) > 0]
    if not clears:
        biggest = real[-1] if real else None
        notes.append(f"No rung clears both corrections. The floor is ABOVE the largest rung tested"
                     + (f" (delta={rungs[biggest]['delta']}), which is itself a result: this pipeline "
                        f"cannot see an injected graph signal even at that size." if biggest else "."))
        return {"floor": None, "floor_status": "above_ladder", "notes": notes,
                "control_clears": False}

    for n in clears:                     # smallest clearing rung with every LARGER rung also clearing
        larger = [x for x in real if rungs[x]["delta"] > rungs[n]["delta"]]
        if all(x in clears for x in larger):
            floor = rungs[n]["delta"]
            break
    if floor is None:
        notes.append("*** NON-MONOTONE: a rung clears while a larger one does not. Reported as a red "
                     "flag, not a floor (Amendment 6.7). ***")
        return {"floor": None, "floor_status": "non_monotone", "notes": notes,
                "control_clears": False}

    notes.append(f"MEASURED FLOOR: delta = {floor} response SDs. Below that, an injected graph signal "
                 f"is not recovered by this pipeline's best graph detector.")
    notes.append("This bounds the PIPELINE's sensitivity via untyped_gnn, not the typed encoder's "
                 "specifically (Amendment 6.5).")
    return {"floor": floor, "floor_status": "measured", "notes": notes, "control_clears": False}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=LADDER_ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.root, Path(a.out) if a.out else None)
