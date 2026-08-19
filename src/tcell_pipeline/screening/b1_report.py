"""B1 report: WHICH component of the message form costs the typed encoder its benefit?

Applies the rule fixed in docs/replication-prereg.md Amendment 7, without discretion.

    D3 = typed_gcnnorm - typed_static      symmetric degree normalisation instead of a plain sum

A1 already eliminated the two obvious routes: tying the message weights (D1, +0.0004) and randomising
every relation label at preserved edge counts (D2, +0.0065), neither clearing correction. So it is
neither the parameter count nor the annotation's information content, and what is left is the FORM of
the message. This reads the arms that walk that form toward ``untyped_gnn`` one component at a time.

SIGNS follow Amendment 4b unchanged: every contrast is (diagnostic arm) MINUS typed_static, so POSITIVE
means the intervention IMPROVED on the typed encoder, i.e. removing that component recovers part of the
deficit.

MULTIPLICITY UNDER STAGING (Amendment 7.3). B1 is staged on purpose - B1a is read before B1b-d are
launched - so ``m`` is the number of B1 arms with landed lanes AT READ TIME, and every contrast is
re-corrected at the larger m as arms are added. That is why ARMS is a list and the family is built from
whichever of them are actually on disk: adding an arm cannot leave an earlier arm's p-value corrected at
the smaller family it was first read under.

    PYTHONPATH=src python -m tcell_pipeline.screening.b1_report \
        --out data/results/screening_b1/b1_message_form.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tcell_pipeline.screening.a1_report import metric_by_seed
from tcell_pipeline.screening.multiseed import apply_family_wise, paired_delta_summary
from tcell_pipeline.screening.screening import (
    EXPRESSION_ONLY,
    TYPED_GCNNORM,
    TYPED_STATIC,
    UNTYPED_GNN,
)

ROOT = "data/results/screening_b1"
SEEDS = (0, 1, 2, 3, 4)
# (label, arm). B1b-d append here as they are implemented and amended; nothing else changes.
ARMS = (("D3", TYPED_GCNNORM),)


def _gap(root: Path, seeds=SEEDS, alpha: float = 0.05) -> dict:
    """untyped_gnn - typed_static on the landed lanes: the deficit B1 is trying to localise. Amendment
    7.4 uses it only as the DENOMINATOR of a descriptive share, never as a test."""
    return paired_delta_summary(metric_by_seed(root, UNTYPED_GNN, seeds),
                                metric_by_seed(root, TYPED_STATIC, seeds), alpha=alpha, seeds=seeds)


def collect(root: Path = Path(ROOT), seeds=SEEDS, alpha: float = 0.05) -> dict:
    """Every B1 arm PRESENT ON DISK, corrected together. An arm with no landed lane is absent from the
    family rather than counted as a null - counting an unrun arm would inflate m and weaken every arm
    that did run."""
    present = [(label, arm) for label, arm in ARMS if metric_by_seed(root, arm, seeds)]
    static = metric_by_seed(root, TYPED_STATIC, seeds)
    contrasts = {label: paired_delta_summary(metric_by_seed(root, arm, seeds), static,
                                             alpha=alpha, seeds=seeds)
                 for label, arm in present}
    for label, arm in present:
        contrasts[label]["arm"] = arm
    m = apply_family_wise(contrasts, alpha) if contrasts else 0
    return {"contrasts": contrasts, "family_size": m, "arms_present": [a for _, a in present],
            "arms_absent": [a for label, a in ARMS if (label, a) not in present]}


def _verdict(contrasts: dict, gap: dict) -> dict:
    """Amendment 7.5, each outcome's meaning fixed before the numbers existed."""
    notes, routes = [], []
    g = gap.get("mean")
    for label, c in contrasts.items():
        if c["n"] < 4:
            notes.append(f"{label} ({c['arm']}): n={c['n']} < 4 — PRELIMINARY, reported with its n (rail 5)")
            continue
        share = (c["mean"] / g) if (g and c["mean"] is not None and g != 0) else None
        c["recovery_share"] = share       # DESCRIPTIVE only (Amendment 7.4): a ratio of two estimates
        if not c.get("survives_family_wise"):
            notes.append(f"{label} ({c['arm']}): NULL at m={c.get('family_size')} "
                         f"({c['mean']:+.4f}, bonf {c['p_bonferroni']:.4f}) — this component is not the route")
        elif (c["mean"] or 0) > 0:
            routes.append(label)
            notes.append(f"{label} ({c['arm']}): RECOVERS {c['mean']:+.4f}"
                         + (f", {share:.0%} of the {g:+.4f} gap" if share is not None else "")
                         + " and clears both corrections")
        else:
            notes.append(f"*** {label} ({c['arm']}): significantly NEGATIVE ({c['mean']:+.4f}). Removing "
                         f"this component makes the typed encoder WORSE. Reported as such, NOT folded "
                         f"into 'no effect' (Amendment 7.5). ***")
    if not contrasts:
        notes.append("no B1 arm has landed a lane yet")
    elif (all(c["n"] >= 4 for c in contrasts.values())
          and all(not c.get("survives_family_wise") for c in contrasts.values())):
        # Only when every arm is genuinely NULL. An arm that clears NEGATIVE is a finding about that
        # component, not an absence of one, and calling it "distributed" would fold it into "no effect"
        # exactly as Amendment 7.5 forbids -- one level up from where the check was written first.
        notes.append("NO component tested clears in either direction: on this evidence the deficit is "
                     "DISTRIBUTED across the message form rather than localised (Amendment 7.5).")
    if len(routes) > 1:
        notes.append(f"{len(routes)} components clear. All are reported and NONE is claimed exclusively: "
                     f"the components are not independent and this design cannot separate interactions "
                     f"(Amendment 7.4).")
    return {"routes": routes, "notes": notes}


def run(root: Path = Path(ROOT), out: Path | None = None, seeds=SEEDS, alpha: float = 0.05) -> dict:
    col = collect(root, seeds, alpha)
    contrasts, m = col["contrasts"], col["family_size"]
    gap = _gap(root, seeds, alpha)
    nograph = paired_delta_summary(metric_by_seed(root, TYPED_STATIC, seeds),
                                   metric_by_seed(root, EXPRESSION_ONLY, seeds), alpha=alpha, seeds=seeds)

    print(f"[b1] root={root}  family size m={m} (Bonferroni AND Holm, both required)")
    if col["arms_absent"]:
        print(f"[b1] arms not yet landed, excluded from m: {col['arms_absent']}")
    print(f"[b1] the deficit under study: untyped_gnn - typed_static = {gap['mean']:+.4f} "
          f"[{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}] at n={gap['n']}" if gap["mean"] is not None
          else "[b1] the deficit under study is UNDECIDABLE on these lanes")
    print(f"[b1] context: typed_static - expression_only = "
          + (f"{nograph['mean']:+.4f} at n={nograph['n']}" if nograph["mean"] is not None else "—"))
    print(f"\n{'arm':>16} {'n':>2} {'mean':>10} {'95% CI':>22} {'bonf':>8} {'holm':>8} {'share':>7} {'survives':>9}")
    for label, c in contrasts.items():
        ci = "     —" if c["ci_low"] is None else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        mean = "    —" if c["mean"] is None else f"{c['mean']:+.4f}"
        bonf = "  —" if c["p_bonferroni"] is None else f"{c['p_bonferroni']:.4f}"
        holm = "  —" if c["p_holm"] is None else f"{c['p_holm']:.4f}"
        sh = c.get("recovery_share")
        print(f"{c['arm']:>16} {c['n']:>2} {mean:>10} {ci:>22} {bonf:>8} {holm:>8} "
              f"{('—' if sh is None else f'{sh:.0%}'):>7} {str(c.get('survives_family_wise')):>9}")

    verdict = _verdict(contrasts, gap)
    for line in verdict["notes"]:
        print(f"[b1] {line}")

    report = {"root": str(root), "seeds": list(seeds), "alpha": alpha, "family_size": m,
              "gap_untyped_minus_typed": gap, "context_typed_vs_nograph": nograph,
              "contrasts": contrasts, "arms_present": col["arms_present"],
              "arms_absent": col["arms_absent"], **verdict}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=float))
        print(f"[b1] wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(Path(a.root), Path(a.out) if a.out else None)
