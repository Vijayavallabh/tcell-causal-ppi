#!/usr/bin/env python
"""L5 / cause D: does the 'no graph' baseline actually carry graph information?

WRITTEN BEFORE THE LANES LANDED, so the analysis is pre-specified rather than chosen after seeing
the numbers. Contrast, correction and verdict rule are all fixed here.

DESIGN. expression_only receives PINNACLE (learned on a PPI network) and three PPI degree scalars.
Each ablated variant ZEROES a subset; out_dim and parameter count are unchanged, so the paired
difference isolates information, not capacity. Paired by seed against the unablated comparator.

COMPARABILITY CAVEAT, stated up front. The comparator lanes in screening_lambda0/expression_only were
copied from the original campaign and predate a schema change: they lack n_train, n_val and
lambda_graph. So fold identity cannot be verified from the comparator artifact alone. It is verified
indirectly: the ablated runs report n_train=21262 / n_val=4400, which is the frozen fold, and the
comparator was produced on that same frozen fold by the original campaign. lambda_graph is irrelevant
to expression_only, which has no graph encoder and therefore no edge-gate penalty. This is recorded
rather than waved away because 'presence is not comparability'.

FAMILY. Three contrasts (nograph, nopinnacle, nodegree) each vs the unablated baseline. Bonferroni and
Holm over m=3. A contrast counts only if it clears BOTH, matching the rest of the paper.
"""
from __future__ import annotations

import glob
import json
import statistics as st

import pandas as pd
from scipy import stats

BASE = "data/results/screening_lambda0/expression_only"
VARIANTS = {"nograph": "PINNACLE + PPI degrees dropped (the true no-graph floor)",
            "nopinnacle": "PINNACLE dropped only",
            "nodegree": "PPI degrees dropped only"}


def _systema(root: str) -> dict[int, float]:
    out = {}
    for f in glob.glob(f"{root}/[0-4].parquet"):
        d = pd.read_parquet(f).iloc[0]
        out[int(d["seed"])] = float(d["systema"])
    return out


def main() -> int:
    base = _systema(BASE)
    print(f"unablated baseline (frozen fold): n={len(base)} "
          f"mean={st.mean(base.values()):.6f}" if base else "no baseline")
    rows = []
    for v in VARIANTS:
        abl = _systema(f"data/results/ablate_{v}/expression_only")
        seeds = sorted(set(abl) & set(base))
        if len(seeds) < 2:
            print(f"  {v}: n={len(seeds)} — not yet analysable"); continue
        # delta = ablated - full. NEGATIVE means removing the feature HURT, i.e. it carried signal.
        d = [abl[s] - base[s] for s in seeds]
        m, sd, n = st.mean(d), st.stdev(d), len(d)
        se = sd / n ** 0.5
        p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else float("nan")
        rows.append({"variant": v, "n": n, "mean": m, "sd": sd, "p_raw": p,
                     "per_seed": {s: abl[s] - base[s] for s in seeds}})
    m_fam = len(rows)
    for r in rows:
        r["p_bonf"] = min(1.0, r["p_raw"] * m_fam)
    for r, rank in zip(sorted(rows, key=lambda x: x["p_raw"]), range(m_fam)):
        r["p_holm"] = min(1.0, r["p_raw"] * (m_fam - rank))
    run = 0.0
    for r in sorted(rows, key=lambda x: x["p_raw"]):
        run = max(run, r["p_holm"]); r["p_holm"] = run
        r["survives"] = r["p_bonf"] < 0.05 and r["p_holm"] < 0.05

    print(f"\nfamily_size = {m_fam} (Bonferroni and Holm over the three ablation contrasts)\n")
    for r in rows:
        t = stats.t.ppf(0.975, r["n"] - 1)
        se = r["sd"] / r["n"] ** 0.5
        verdict = ("REMOVING IT HURT -> the feature carried signal" if r["survives"] and r["mean"] < 0
                   else "REMOVING IT HELPED -> the feature was noise" if r["survives"]
                   else "indistinguishable at this budget")
        print(f"  {r['variant']:11s} n={r['n']} delta={r['mean']:+.6f} "
              f"CI=[{r['mean']-t*se:+.6f},{r['mean']+t*se:+.6f}] "
              f"raw p={r['p_raw']:.4f} bonf={r['p_bonf']:.4f} holm={r['p_holm']:.4f} -> {verdict}")
        print(f"              per-seed: {', '.join(f'{k}:{v:+.6f}' for k, v in r['per_seed'].items())}")
        print(f"              ({VARIANTS[r['variant']]})")
    json.dump(rows, open("data/results/feature_ablation_report.json", "w"), indent=2, default=str)
    print("\nwrote data/results/feature_ablation_report.json")
    print("\nINTERPRETATION RULE (fixed in advance): a NEGATIVE delta surviving both corrections means")
    print("the ablated channel carried real signal, so the 'no graph' baseline is not graph-free and")
    print("cause D stands. Indistinguishable means the channel is inert and cause D drops to a footnote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
