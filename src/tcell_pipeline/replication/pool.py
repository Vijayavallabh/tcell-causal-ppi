"""Pool a contrast across replication datasets: fixed-effect, random-effects, and heterogeneity.

Each dataset contributes ONE paired estimate (mean across its seeds) with its own standard error. The
pooled estimate answers "does the graph help anywhere across these cell types", which no single dataset
can answer alone.

TWO POOLS ARE REPORTED SEPARATELY AND ARE NEVER MERGED (prereg Amendment 3.2):
  h1  condition_gated - expression_only : only datasets with >= 2 conditions can test this, which here
      means Frangieh alone. n=1 is reported as a single dataset, not as a pooled result.
  h2a typed_static - expression_only    : every dataset.

WHY BOTH FE AND RE. Fixed-effect assumes one true effect and weights by precision; it is the tighter
interval and the right one if the datasets really are measuring the same thing. Random-effects admits
between-dataset variance (tau^2) and widens accordingly. Reporting only FE would understate uncertainty
across cell types; reporting only RE would throw away precision if the effects genuinely agree. I^2
says which reading the data support.

A pooled interval that contains zero and is TIGHT is a bounded null - the strongest claim this project
can make. A pooled interval that contains zero and is WIDE is an underpowered experiment and must not
be reported as a null. The two are distinguished here by the half-width, not by the p-value.

    PYTHONPATH=src python -m tcell_pipeline.replication.pool
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path("data/results/replication")
# Directories starting with "_" are snapshots and scratch, not datasets. Globbing them in
# double-counts whichever dataset was snapshotted and silently inflates k.
# Datasets with >= 2 experimental conditions, where condition_gated is not identical to typed_static.
MULTI_CONDITION = {"FrangiehIzar2021_RNA"}
CONTRASTS = {"h1_vs_no_graph": "condition_gated - expression_only",
             "h2a": "typed_static - expression_only",
             "h2b": "condition_gated - typed_static",
             "promotion_margin": "untyped_gnn - expression_only"}


def _load(dataset: str) -> dict:
    p = ROOT / dataset / "robustness_5seed.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _se(c: dict) -> float | None:
    """The aggregator already reports the paired SE across seeds; use it rather than inverting the CI."""
    se = c.get("se")
    return float(se) if se else None


def pool(estimates: list[tuple[str, float, float]]) -> dict:
    """DerSimonian-Laird. `estimates` is [(dataset, effect, se)]."""
    if not estimates:
        return {"k": 0}
    from scipy import stats
    w = [1.0 / (se ** 2) for _, _, se in estimates]
    y = [e for _, e, _ in estimates]
    k = len(estimates)
    fe = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
    fe_se = math.sqrt(1.0 / sum(w))
    Q = sum(wi * (yi - fe) ** 2 for wi, yi in zip(w, y))
    df = k - 1
    C = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0.0, (Q - df) / C) if C > 0 and df > 0 else 0.0
    I2 = max(0.0, (Q - df) / Q) if Q > 0 and df > 0 else 0.0
    wr = [1.0 / (1.0 / wi + tau2) for wi in w]
    re = sum(wi * yi for wi, yi in zip(wr, y)) / sum(wr)
    re_se = math.sqrt(1.0 / sum(wr))
    out = {"k": k, "datasets": [d for d, _, _ in estimates], "effects": [round(e, 5) for _, e, _ in estimates],
           "fixed_effect": round(fe, 5), "fixed_se": round(fe_se, 5),
           "fixed_ci": [round(fe - 1.96 * fe_se, 5), round(fe + 1.96 * fe_se, 5)],
           "random_effect": round(re, 5), "random_se": round(re_se, 5),
           "random_ci": [round(re - 1.96 * re_se, 5), round(re + 1.96 * re_se, 5)],
           "Q": round(Q, 4), "df": df, "tau2": round(tau2, 6), "I2": round(I2, 4)}
    out["Q_p"] = round(float(1 - stats.chi2.cdf(Q, df)), 4) if df > 0 else None
    out["random_p"] = round(float(2 * (1 - stats.norm.cdf(abs(re / re_se)))), 4)
    return out


def run(datasets: list[str], min_seeds: int = 4) -> dict:
    report: dict = {"min_seeds_per_dataset": min_seeds, "per_dataset": {}, "pooled": {}, "excluded": {}}
    for name, label in CONTRASTS.items():
        ests, excluded = [], {}
        for ds in datasets:
            agg = _load(ds)
            if not agg:
                excluded[ds] = "no robustness report"; continue
            # h1/h2b are meaningless where the gate has only one context to gate: condition_gated is
            # then arithmetically typed_static. Excluded by design, recorded so it is not read as
            # attrition.
            if name in ("h1_vs_no_graph", "h2b") and ds not in MULTI_CONDITION:
                excluded[ds] = "single-condition: condition_gated degenerates to typed_static"; continue
            c = agg.get("contrasts", {}).get(name)
            if c is None:
                excluded[ds] = "contrast absent"; continue
            if (c.get("n") or 0) < min_seeds:
                excluded[ds] = f"n={c.get('n')} < {min_seeds} seeds (preliminary, not pooled)"; continue
            se = _se(c)
            if se is None or se == 0:
                excluded[ds] = "no usable standard error"; continue
            ests.append((ds, float(c["mean"]), se))
            report["per_dataset"].setdefault(name, {})[ds] = {
                "mean": round(float(c["mean"]), 5), "se": round(se, 5), "n_seeds": c.get("n"),
                "ci": [c.get("ci_low"), c.get("ci_high")],
                "per_seed": [round(d, 5) for d in c.get("deltas", [])],
                "seeds_used": c.get("seeds_used"), "dropped": c.get("dropped"),
                "survives_family_wise": c.get("survives_family_wise"),
                "family_size": c.get("family_size")}
        report["pooled"][name] = {"label": label, **pool(ests)}
        report["excluded"][name] = excluded
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=sorted(p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")))
    ap.add_argument("--min-seeds", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "pooled.json"))
    a = ap.parse_args()
    rep = run(a.datasets, a.min_seeds)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    for name, p in rep["pooled"].items():
        if not p.get("k"):
            print(f"[pool] {name:16s} NO DATASETS ({rep['excluded'][name]})"); continue
        print(f"[pool] {name:16s} k={p['k']} FE={p['fixed_effect']:+.4f} {p['fixed_ci']} | "
              f"RE={p['random_effect']:+.4f} {p['random_ci']} p={p['random_p']} | "
              f"I2={p['I2']:.1%} tau2={p['tau2']:.5f} Q_p={p['Q_p']}")
        print(f"[pool]   {', '.join(f'{d}={e:+.4f}' for d, e in zip(p['datasets'], p['effects']))}")
    print(f"[pool] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
