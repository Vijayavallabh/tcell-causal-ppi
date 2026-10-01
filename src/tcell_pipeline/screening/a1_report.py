"""A1 report: does edge typing hurt through its parameters, through its labels, or through neither?

Reads the diagnostic arms and applies the rule fixed in Amendments 4.4 and 4b, without discretion.

    D1 = typed_shared   - typed_static     one message module tied across all four relations
    D2 = typed_permuted - typed_static     per-relation modules, relation labels randomly reassigned

SIGNS, stated because Amendment 4.4 got its own parenthetical backwards and 4b corrects it. Both
contrasts are (diagnostic arm) MINUS (typed_static), so a POSITIVE value means the intervention IMPROVED
on the typed encoder. D2 positive therefore means a random partition beats the true one: the annotation
is worse than noise at equal capacity. D2 negative means the true typing carries information the random
one lacks.

The two form their own diagnostic family of size two, corrected by Bonferroni AND Holm with survival
requiring both. Contrasts against expression_only and untyped_gnn are reported as CONTEXT and are not in
that family; neither may be used to promote a graph claim (Amendment 4.5).

    PYTHONPATH=src python -m tcell_pipeline.screening.a1_report \
        --out data/results/screening_a1/a1_mechanism.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tcell_pipeline.screening.multiseed import apply_family_wise, paired_delta_summary
from tcell_pipeline.screening.screening import (
    EXPRESSION_ONLY,
    PRIMARY_METRIC,
    TYPED_PERMUTED,
    TYPED_SHARED,
    TYPED_STATIC,
    UNTYPED_GNN,
)

ROOT = "data/results/screening_a1"
SEEDS = (0, 1, 2, 3, 4)
FAMILY = (("D1", TYPED_SHARED, TYPED_STATIC), ("D2", TYPED_PERMUTED, TYPED_STATIC))
CONTEXT = (("shared_vs_nograph", TYPED_SHARED, EXPRESSION_ONLY),
           ("permuted_vs_nograph", TYPED_PERMUTED, EXPRESSION_ONLY),
           ("shared_vs_untyped", TYPED_SHARED, UNTYPED_GNN),
           ("permuted_vs_untyped", TYPED_PERMUTED, UNTYPED_GNN))


def metric_by_seed(root: Path, arm: str, seeds=SEEDS) -> dict:
    out = {}
    for s in seeds:
        p = Path(root) / arm / f"{s}.parquet"
        if not p.exists():
            continue
        row = pd.read_parquet(p).iloc[0].to_dict()
        if row.get("status") != "completed":
            continue
        v = row.get(PRIMARY_METRIC)
        if v is None or not float(v) == float(v):
            continue
        out[s] = float(v)
    return out


def _verdict(d1: dict, d2: dict) -> dict:
    """The 2x2 of Amendment 4.4, with 4b's sign convention. 'Clears' means positive AND surviving both
    corrections; a contrast that is significantly NEGATIVE is called out separately, because it is a
    different finding and not the absence of one."""
    def state(c):
        if c["n"] < 4:
            return "underpowered"
        if not c.get("survives_family_wise"):
            return "null"
        return "positive" if (c["mean"] or 0) > 0 else "negative"

    s1, s2 = state(d1), state(d2)
    notes = []
    if "underpowered" in (s1, s2):
        notes.append(f"n<4 on at least one contrast (D1 n={d1['n']}, D2 n={d2['n']}): PRELIMINARY, "
                     f"and must be labelled with its n (rail 5)")

    key = (s1, s2)
    table = {
        ("null", "null"): "The typed STRUCTURE hurts, and neither its parameter count nor its labels "
                          "is the route. Look at the remaining typed-vs-untyped differences: signed "
                          "messages, the edge-feature term, complex nodes, the residual FFN.",
        ("null", "positive"): "A RANDOM partition beats the true one at equal capacity, while removing "
                              "the partition changes nothing. The evidence typing is worse than noise: "
                              "it is actively misleading, not merely uninformative.",
        ("null", "negative"): "The true typing carries information a random partition lacks, yet "
                              "removing the partition costs nothing. The typing helps relative to noise "
                              "and still leaves the encoder behind the untyped baseline.",
        ("positive", "null"): "Removing the per-relation multiplicity RECOVERS part of the deficit "
                              "while a random partition changes nothing. Capacity and partition are "
                              "jointly the route, and the labels contribute nothing the shared arm "
                              "loses. NOTE both are removed by one intervention under add-aggregation "
                              "(Amendment 4.3), so this cannot be attributed to parameter count alone.",
        ("positive", "positive"): "Both routes are live: tying the weights helps AND a random partition "
                                  "beats the true one. Report both effect sizes and claim neither "
                                  "exclusively.",
        ("positive", "negative"): "Tying the weights helps, and the true partition beats a random one. "
                                  "The typing carries information but costs more than it gives.",
        ("negative", "null"): "Tying the weights HURTS: the per-relation modules are doing real work. "
                              "The deficit is not about capacity.",
        ("negative", "positive"): "Tying hurts and a random partition beats the true one: the encoder "
                                  "needs separate modules but the labels it is given are wrong.",
        ("negative", "negative"): "Both interventions hurt. The typed encoder as configured is the best "
                                  "of the three, and the deficit against no-graph lies elsewhere.",
    }
    reading = table.get(key, "unclassified")
    return {"d1_state": s1, "d2_state": s2, "reading": reading, "notes": notes}


def run(root: str = ROOT, out: Path | None = None, seeds=SEEDS, alpha: float = 0.05) -> dict:
    arms = {a: metric_by_seed(Path(root), a, seeds)
            for a in (EXPRESSION_ONLY, UNTYPED_GNN, TYPED_STATIC, TYPED_SHARED, TYPED_PERMUTED)}
    for a, v in arms.items():
        print(f"[a1] {a:>16}: {len(v)}/{len(seeds)} seeds landed")

    fam = {k: paired_delta_summary(arms.get(b, {}), arms.get(w, {}), alpha=alpha, seeds=seeds)
           for k, b, w in FAMILY}
    m = apply_family_wise(fam, alpha)
    ctx = {k: paired_delta_summary(arms.get(b, {}), arms.get(w, {}), alpha=alpha, seeds=seeds)
           for k, b, w in CONTEXT}

    print(f"\n[a1] DIAGNOSTIC FAMILY (m={m}, Bonferroni AND Holm, both required)")
    print(f"{'contrast':>22} {'n':>2} {'mean':>10} {'95% CI':>22} {'bonf':>7} {'holm':>7} {'survives':>9}")
    for k, b, w in FAMILY:
        c = fam[k]
        ci = "     —" if c["ci_low"] is None else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        mean = "    —" if c["mean"] is None else f"{c['mean']:+.4f}"
        bonf = "  —" if c["p_bonferroni"] is None else f"{c['p_bonferroni']:.4f}"
        holm = "  —" if c["p_holm"] is None else f"{c['p_holm']:.4f}"
        print(f"{k + ' ' + b + '-' + w:>22} {c['n']:>2} {mean:>10} {ci:>22} {bonf:>7} {holm:>7} "
              f"{str(c['survives_family_wise']):>9}")

    print("\n[a1] CONTEXT (not in the family; may not be used to promote a graph claim)")
    for k, _, _ in CONTEXT:
        c = ctx[k]
        mean = "    —" if c["mean"] is None else f"{c['mean']:+.4f}"
        ci = "     —" if c["ci_low"] is None else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        print(f"{k:>22} {c['n']:>2} {mean:>10} {ci:>22}")

    v = _verdict(fam["D1"], fam["D2"])
    print(f"\n[a1] D1 {v['d1_state']}, D2 {v['d2_state']}  ->  {v['reading']}")
    for n in v["notes"]:
        print(f"[a1] {n}")

    # Contradiction stop: a DIAGNOSTIC arm beating no-graph under correction is still a positive.
    stop = [k for k in ("shared_vs_nograph", "permuted_vs_nograph")
            if ctx[k].get("ci_excludes_zero") and (ctx[k]["mean"] or 0) > 0]
    if stop:
        print(f"[a1] *** A diagnostic arm beats expression_only with a CI excluding zero: {stop}. "
              f"Amendment 4.5 sends this to the contradiction stop — snapshot, flag at the top of "
              f"RESULTS_SUMMARY.md, continue. Do NOT rewrite the null around it, and do NOT use a "
              f"diagnostic arm to promote a graph claim. ***")

    report = {"root": root, "seeds": list(seeds), "family_size": m, "arms_landed":
              {a: sorted(v) for a, v in arms.items()}, "family": fam, "context": ctx,
              "contradiction_candidates": stop, **v}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=float))
        print(f"[a1] wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(a.root, Path(a.out) if a.out else None)
