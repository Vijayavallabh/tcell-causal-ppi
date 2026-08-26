#!/usr/bin/env python3
"""Guard the paper's numbers across a rewrite.

Two independent checks, because a rewrite can break numbers in two different ways.

1. INVENTORY DRIFT. Every numeric literal in main.tex is extracted with its context. A
   rewrite may re-word freely, but it must not invent a number that was not there before,
   and must not silently drop one. Compare against a snapshot taken from the verified
   state:  --snapshot  writes it,  --check  diffs against it.

2. ARTIFACT AGREEMENT. The tables are re-derived from their JSON artifacts, independent of
   the snapshot, so a number that was wrong BEFORE the rewrite is still caught.

    python verify_numbers.py --snapshot          # from a state you trust
    python verify_numbers.py --check             # after editing
    python verify_numbers.py --coverage          # which tables are checked, and how

THE TWO CHECKS ARE NOT EQUALLY STRONG, and the difference is the whole reason for the
per-table registry below. The inventory catches a number that CHANGES or VANISHES during an
edit. Only artifact agreement catches a number that was WRONG WHEN IT WAS SNAPSHOTTED - and
that is the class of error this paper actually produced. The 2026-08-20 rewrite shipped three:
I^2 89% for 87.5%, condition_gated 0.0829 for 0.0818, and "five of our seven datasets" for
eight. The inventory passes all three by construction. All three were caught by hand, and
hand-checking does not survive the next rewrite.

EVERY TABLE IN THE PAPER APPEARS IN CHECKS, INCLUDING THE ONES THAT CANNOT BE CHECKED. A table
absent from the registry is indistinguishable from a table that passed, so the ones with no
artifact, or only a partial one, are NAMED NO-OPS carrying the reason. A missing artifact file
is likewise reported as SKIPPED rather than passing silently: three tables used to be guarded by
`if path.exists():`, which reads as coverage on a machine where the artifact was never built.
Run --coverage to see the state of all eleven at once.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = HERE / "numbers.snapshot.json"

# Numeric literals inside math mode: the form every measured quantity is written in.
NUM = re.compile(r"[-+]?\d+(?:\{,\})?\d*(?:\.\d+)?")
SIGNED = re.compile(r"[-+]\d+\.\d+")


def literals(tex: str) -> Counter:
    """Multiset of numeric literals appearing inside $...$ math, comments stripped."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    out = Counter()
    for m in re.finditer(r"\$([^$]*)\$", tex):
        for n in NUM.findall(m.group(1)):
            if n in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                continue          # bare small integers are indices/counts, too noisy
            out[n.replace("{,}", ",")] += 1
    return out


# =================================================================================================
# Plumbing shared by every table check
# =================================================================================================
class Missing(Exception):
    """An artifact this check needs is not on disk. Reported as SKIPPED, never as a pass."""


def load(rel: str):
    p = ROOT / rel
    if not p.exists():
        raise Missing(rel)
    return json.loads(p.read_text())


def cells_of(tex: str, label: str):
    """Each tabular row of the table with this label, as a list of trimmed cells."""
    m = re.search(r"\\label\{" + label + r"\}(.*?)\\end\{tabular\}", tex, re.S)
    body = m.group(1) if m else ""
    for line in body.split("\\\\"):
        line = line.replace("\\midrule", "").replace("\\toprule", "").strip()
        cs = [c.strip() for c in line.split("&")]
        if len(cs) > 1:
            yield cs


def num_in(cell: str):
    """The first signed decimal in a cell, and how many decimals it was written to.

    The decimal count matters: this paper writes 4 places for \\textsc{systema}, 3 for a CI in
    Table~\\ref{tab:family}, and 1 for an E-distance. Comparing at the cell's OWN precision is
    what lets one comparator serve all of them without hard-coding a format per column."""
    m = SIGNED.search(cell.replace("{,}", ""))
    if not m:
        return None, None
    return float(m.group(0)), len(m.group(0).split(".")[1])


def agrees(cell: str, value: float) -> bool:
    got, nd = num_in(cell)
    if got is None or value is None:
        return False
    return f"{got:+.{nd}f}" == f"{value:+.{nd}f}"


def cmp_cell(fails: list, where: str, cell: str, value, star: bool | None = None):
    """One paper cell against one artifact value, at the cell's own precision."""
    if value is None:
        return
    if not agrees(cell, value):
        got, nd = num_in(cell)
        shown = "no number" if got is None else f"{got:+.{nd}f}"
        fails.append(f"{where}: paper {shown} artifact {value:+.4f}")
    if star is not None:
        marked = "\\star" in cell or "^{\\star}" in cell
        if marked != star:
            fails.append(f"{where}: star marked={marked} artifact survives={star}")


CHECKS: list[tuple[str, object]] = []


def table(label: str):
    def deco(fn):
        CHECKS.append((label, fn))
        return fn
    return deco


# =================================================================================================
# The eleven tables
# =================================================================================================
_FAMILY_ROOT = "data/results/screening_lambda0"          # the REPAIRED gate root, not screening/
_FAMILY_KEY = {"ts": "typed_static", "cg": "condition_gated", "ug": "untyped_gnn",
               "eo": "expression_only"}
_CONTRAST_OF = {("typed_static", "expression_only"): "h2a",
                ("condition_gated", "typed_static"): "h2b",
                ("untyped_gnn", "expression_only"): "promotion_margin",
                ("condition_gated", "expression_only"): "h1_vs_no_graph"}


# Region -> the artifacts that region's numbers are drawn from. Scoping matters and is not cosmetic:
# checking a value against ALL 514 result artifacts is vacuous, because a random four-decimal effect
# size lands in that corpus 65% of the time and 91% of the paper's real values still "pass" after
# being perturbed by one digit. Scoped to the handful of artifacts a section actually uses, the same
# test admits a random value 3-8% of the time. Measured, not assumed, on 2026-08-26.
_R = "data/results/"
PROSE_REGIONS = {
    "sec:confound": [_R + "screening/robustness_5seed.json", _R + "q4_lambda_sweep_22ep.json",
                     _R + "q4_lambda_sweep_12ep.json", _R + "screening_lambda0/lambda_sweep_empirical.json"],
    "sec:null": [_R + "screening_lambda0/robustness_5seed.json",
                 _R + "screening_lambda0/second_metric_5seed.json",
                 _R + "screening_c080c10_h1/robustness_5seed.json",
                 _R + "screening_c075c15_n5/robustness_5seed.json",
                 _R + "screening/robustness_5seed.json"],
    "sec:repl": [_R + "replication/pooled_with_reference.json", _R + "replication/pooled.json",
                 _R + "screening_untyped_n7/robustness_5seed.json",
                 _R + "replication/pooled_k128_subset.json"],
    "sec:causes": [_R + "screening_untyped_n7/robustness_5seed.json", _R + "feature_ablation_report.json",
                   _R + "replication/pooled_with_reference.json", _R + "a2_ladder/floor.json",
                   _R + "rationale_audit_lambda0/audit_report.json"],
    "app:floor": [_R + "a2_ladder/floor.json", _R + "c1_ladder/floor_condition_gated.json",
                  _R + "c1_ladder/c1_power_posthoc.json", _R + "a2_ladder/b3_power.json",
                  _R + "screening_untyped_n7/robustness_5seed.json"],
    "app:power": [_R + "a2_power/power_simulation.json", _R + "l4/vardecomp_h2a.json",
                  _R + "l4/vardecomp_h1_vs_no_graph.json", _R + "a2_power/arch_search_bound.json",
                  _R + "screening_lambda0/robustness_5seed.json",
                  _R + "screening_c075c15_n5/robustness_5seed.json",
                  _R + "screening_c080c10_h1/robustness_5seed.json", _R + "a2_ladder/floor.json"],
    "app:metrics": [_R + "a3_external/rescored.json", _R + "a3_external/k_sweep.json",
                    _R + "b2_deciles/deciles.json"],
    "app:mechanism": [_R + "screening_a1/a1_mechanism.json", _R + "screening_b1/b1_message_form.json",
                      _R + "a2_power/arch_search_bound.json",
                      _R + "screening_untyped_n7/robustness_5seed.json"],
}

# Literals that are NOT results and therefore cannot be re-derived from a results artifact. Every one
# carries the reason it is exempt. This list is the point of the check: a number nobody has accounted
# for FAILS, so a new unexplained figure cannot enter the paper unnoticed.
PROSE_DECLARED = {
    # --- constants of the method, not results -------------------------------------------------
    "95": "the confidence level in '95% CI'. A constant of the method.",
    # --- model architecture, fixed in pre-registration Amendment 4.2, not a results artifact ---
    "2{,}396{,}160": "typed message parameters; Amendment 4.2 fixes this count.",
    "599{,}040": "shared-weight message parameters; Amendment 4.2.",
    "5{,}254{,}884": "total model parameters before the cut; Amendment 4.2.",
    "3{,}457{,}764": "total model parameters after the cut; Amendment 4.2.",
    "34": "the parameter cut as a percentage, derived from the two totals above.",
    # --- properties of the inputs, recorded in build provenance rather than in a results file ---
    "9{,}730": "targets in the genome-wide screen; from the DE build provenance.",
    "44": "datasets in the harmonized scPerturb resource; a cited property of that resource.",
    "86": "share of PPI edges that are functional_assoc; a property of the graph build.",
    "0.228": "median functional_assoc edge score; a property of the graph build.",
    "0.57": "mean edge gate after the repair, read from training logs rather than a results JSON.",
    # --- ratios derived in the sentence that states them; their components ARE gated -----------
    "103": "ratio of the graph term to the response term, derived in the same sentence.",
    "79": "D3 divided by the 0.0176 gap. Amendment 7.4 fixes this as DESCRIPTIVE only.",
    "31.7": "per-arm seed-spread ratio, derived from two gated standard deviations.",
    "20.6": "per-arm seed-spread ratio, derived as above.",
    "17.4": "per-arm seed-spread ratio, derived as above.",
    "5.9": "per-arm seed-spread ratio, derived as above.",
    # --- B1a SECONDARY contrasts. PROVENANCE GAP, recorded rather than hidden: these were -------
    # --- computed for the B1a report but never persisted into b1_message_form.json, which -------
    # --- carries only D3. They live in RESULTS_SUMMARY.md, which is weaker than this project's --
    # --- own standard of re-deriving every number from a persisted artifact. -------------------
    "+0.0037": "gcnnorm minus untyped_gnn; in RESULTS_SUMMARY.md, not in b1_message_form.json.",
    "-0.0047": "CI bound of gcnnorm minus expression_only; same provenance gap.",
    "+0.0062": "CI bound of gcnnorm minus expression_only; same provenance gap.",
    "0.71": "p for gcnnorm minus expression_only; same provenance gap.",
    "0.22": "p for gcnnorm minus untyped_gnn; same provenance gap.",
    # --- power quantities derived from the measured spreads ------------------------------------
    "0.0096": "minimum detectable effect at five seeds, derived from the measured spread.",
    "0.0075": "minimum detectable effect at seven seeds, derived as above.",
    "0.0185": "minimum detectable effect across re-draws, derived as above.",
    "21": "seeds needed to detect the whole residual, derived from the MDE in the same sentence.",
    "77": "seeds needed to detect half of it, derived as above.",
    "250": "datasets needed against the between-dataset spread, derived as above.",
    # --- split-construction statistics, computed while drawing the folds and never persisted ----
    "0.796": "median train-to-sequestered cosine of the frozen fold; computed during split construction.",
    "0.759": "median train-to-sequestered cosine, second fold; same provenance.",
    "0.741": "median train-to-sequestered cosine, third fold; same provenance.",
    "0.793": "median train-to-sequestered cosine of a re-draw; same provenance.",
    "0.862": "median train-to-sequestered cosine of a re-draw; same provenance.",
    "0.103": "the range the difficulty statistic moved over; derived from the cosines above.",
    "0.056": "the designed range of that statistic; a design parameter.",
    # --- rationale-audit quantities normalised per 1% of edges removed, a derivation ------------
    "0.0043": "sufficiency cost per 1% of STRING edges removed; derived from the raw ablation delta.",
    "0.0025": "the same for HuRI; derived as above.",
    "0.0014": "the same for BioPlex; derived as above.",
    "1.47": "the sufficiency scale the audit reports against; a property of that scale.",
    "0.011": "rationale minus random sufficiency; derived from the audit's aggregate.",
    # --- biology QC: WAS a discrepancy, now recomputed and anchored above. Kept as a note because
    # --- the figures in the paper before 2026-08-26 matched no artifact at all.
    # The IFNGR1 clause that used to need declaring here was REMOVED from the paper on 2026-08-26,
    # because no artifact held its figures and the gene set it rested on is not defined anywhere in
    # this repository. Its three literals are gone with it, so there is nothing left to exempt.
    # --- quantities with no persisted artifact --------------------------------------------------
    "+0.011": "rationale minus random sufficiency; derived from the audit aggregate, not stored.",
    "0.001": "the centroid-accuracy floor, written as an approximation, not a measured value.",
    "0.013": "the simulation's false-positive rate under no true effect; not persisted in its JSON.",
    "0.77": "upper end of the live gate mean after the repair; read from training logs.",
    # --- design choices ------------------------------------------------------------------------
    "24": "gated lanes in the C1 campaign; a design count, six rungs times four seeds.",
    "101": "rank-interval boundary in the B2 binning; a binning choice, not a measurement.",
    "251": "rank-interval boundary in the B2 binning; as above.",
}


def _artifact_values(paths) -> set:
    """Every number in these artifacts, at each precision the paper writes numbers in."""
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, bool) or o is None:
            return
        elif isinstance(o, (int, float)):
            x = float(o)
            for f in (f"{x:+.4f}", f"{x:.4f}", f"{x:+.3f}", f"{x:.3f}", f"{x:.2f}", f"{x:.1f}",
                      f"{x * 100:.1f}", f"{x * 100:.0f}", f"{x:.0f}", f"{x:.5f}", f"{x:.6f}"):
                out.add(f)
            if abs(x) >= 1000 and float(x).is_integer():
                out.add(f"{int(x):,}".replace(",", "{,}"))
                out.add(str(int(x)))
    for rel in paths:
        p = ROOT / rel
        if p.exists():
            walk(json.loads(p.read_text()))
    return out


# Thousands-aware, so $2{,}396{,}160$ is ONE literal. Splitting it into "2{,}396" and "160" made two
# phantom unmatched values and hid the real one.
_PROSE_NUM = re.compile(r"[-+]?\d{1,3}(?:\{,\}\d{3})+|[-+]?\d+(?:\.\d+)?")


@table("prose:all")
def _prose_all(tex):
    """EVERY prose literal in a mapped region, re-derived or explicitly declared.

    The headline check above pins 16 claims to their surrounding words, which is the strongest form
    but needs a hand-written pattern each. This one is exhaustive instead of deep: it takes every
    numeric literal in each mapped section and requires it either to appear in that section's own
    artifacts or to be DECLARED with a reason. It cannot tell whether a value is used for the right
    claim, only whether the section's artifacts contain it at all, so it is weaker per number and
    complete across them. The two are meant to be read together.

    HOW WEAK, MEASURED RATHER THAN GUESSED. Perturbing each re-derived decimal literal by one digit,
    only 40% of those errors are caught: the other 60% land on some other real value in the same
    section's artifacts, because per-seed deltas and CI bounds make the neighbourhood dense. So this
    check's guarantee is ACCOUNTING, not correctness. It proves every prose number is either traceable
    to that section's artifacts or explicitly declared, which is what stops an unexplained figure
    entering the paper. It does NOT reliably catch a wrong digit. Two other checks do that: the
    context-anchored headline claims above, which caught the real Holm error and fire on 5 of 5
    planted ones, and the literal inventory, which catches any number that CHANGES between snapshots.
    That price has now been paid in full: 131 anchored claims, up from 16. Measured on the paper by
    perturbing every prose decimal by one digit, 83% trip an anchor, up from 35%. The other 22 are
    DECLARED, each with the reason no artifact can supply it, and NOTHING is unaccounted for.

    THE ONE REMAINING HOLE IS MULTI-OCCURRENCE, and it is why the inventory still matters. Where a
    value appears in several places, changing ONE of them leaves the anchor satisfied by the others:
    perturbing the body's +0.0675 does not trip it, because the abstract still carries the correct
    value. The literal inventory catches exactly that case, reporting one dropped and one introduced.
    Neither check is sufficient alone and both are gated."""
    body = re.sub(r"(?m)^%.*", "", tex)
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
    # scientific notation is markup, not a literal: $4.5\times10^{6}$ is one quantity, not 4.5 and 6
    body = re.sub(r"[\d.]*\\times10\^\{?-?\d+\}?", " ", body)
    body = re.sub(r"10\^\{?-?\d+\}?", " ", body)

    labs = [(m.start(), m.group(1)) for m in re.finditer(r"\\label\{((?:sec|app):[^}]+)\}", body)]
    fails, checked, declared = [], 0, 0
    for i, (pos, lab) in enumerate(labs):
        if lab not in PROSE_REGIONS:
            continue
        end = labs[i + 1][0] if i + 1 < len(labs) else len(body)
        seg, vals = body[pos:end], _artifact_values(PROSE_REGIONS[lab])
        for m in re.finditer(r"\$([^$]*)\$", seg):
            for n in _PROSE_NUM.findall(m.group(1)):
                if n in list("0123456789"):
                    continue
                checked += 1
                if n in vals:
                    continue
                if n in PROSE_DECLARED:
                    declared += 1
                    continue
                ctx = " ".join(seg[max(0, m.start() - 45):m.end() + 10].split())
                fails.append(f"prose:all {lab} literal {n}: not in that section's artifacts and not "
                             f"declared. Add it to PROSE_DECLARED with a reason, or fix it. "
                             f"Context: ...{ctx[-60:]}...")
    return fails, (f"full: {checked} prose literals across {len(PROSE_REGIONS)} mapped sections, each "
                   f"re-derived from that section's own artifacts or declared ({declared} declared)")


@table("prose:headline")
def _prose_headline(tex):
    """The load-bearing numbers that live in PROSE, not in any table.

    WHY THIS EXISTS. 329 of the paper's numeric literals sit in tables and every one is re-derived by
    the checks below. About 570 sit in prose, and until 2026-08-26 exactly TWO of those were checked
    against an artifact. The rest were guarded by the literal inventory alone, which catches a number
    that CHANGES during an edit and passes one that was WRONG when it was snapshotted. That gap was
    not hypothetical: the paper reported Holm 0.021 where screening_untyped_n7 gives 0.02046976.

    EACH CLAIM IS ANCHORED TO THE WORDS AROUND IT, and the first version of this check was not. It
    asserted only that the artifact value appeared SOMEWHERE in the paper, which is too weak when the
    value is common: re-introducing the Holm error deliberately did NOT trip it, because 0.020 also
    appears as a Bonferroni entry in another table. A guard that cannot fail proves nothing.

    The contexts are plain strings rather than regexes, matched against a whitespace-normalised copy
    of the source so a line wrap between the words and the value cannot cause a false failure.

    KNOWN LIMIT, stated rather than left to be discovered. Where a claim's context occurs more than
    once and only ONE occurrence is corrupted, the surviving correct one still satisfies the check.
    The literal INVENTORY covers that case, because it counts occurrences and a changed digit shows
    up as one dropped and one introduced. The two checks are complementary and neither is sufficient
    alone. Verified by planting five errors, one per claim shape: all five fire."""
    flat = " ".join(re.sub(r"(?m)^%.*", "", tex).split())
    L = load("data/results/screening_lambda0/robustness_5seed.json")["contrasts"]
    N7 = load("data/results/screening_untyped_n7/robustness_5seed.json")["contrasts"]
    P8 = load("data/results/replication/pooled_with_reference.json")
    P7 = load("data/results/replication/pooled.json")["pooled"]["h2a"]
    FL = load("data/results/a2_ladder/floor.json")
    FA = load("data/results/feature_ablation_report.json")
    HD = load("data/results/screening_c075c15_n5/robustness_5seed.json")["contrasts"]
    pm = N7["promotion_margin"]
    ng = next(v for v in FA if v["variant"] == "nograph")
    pd8 = P8["per_dataset"]["promotion_margin"]
    p8 = P8["pooled"]["promotion_margin"]

    def f4(x):
        return f"{x:+.4f}"

    # (what, context with {v} where the artifact value belongs, artifact value)
    claims = [
        ("h1 headline",        "systema}={v}$",                 f4(L["h1_vs_no_graph"]["mean"])),
        ("h2a frozen fold",    "worse by ${v}$",                f4(L["h2a"]["mean"])),
        ("untyped n=7 mean",   "${v}$ \\textsc{systema} at $n{=}7$", f4(pm["mean"])),
        ("untyped Bonferroni", "Bonferroni ${v}$ and Holm",     f"{pm['p_bonferroni']:.3f}"),
        ("untyped Holm",       "and Holm ${v}$",                f"{pm['p_holm']:.3f}"),
        ("n=7 h2a",            "harmful} (${v}$",               f4(N7["h2a"]["mean"])),
        ("n=7 h2b",            "(${v}$, which",                 f4(N7["h2b"]["mean"])),
        ("Replogle RPE1",      "RPE1 (${v}$)",                  f4(pd8["ReplogleWeissman2022_rpe1"]["mean"])),
        ("Norman",             "Norman (${v}$)",                f4(pd8["NormanWeissman2019_filtered"]["mean"])),
        ("pooled RE",          "give ${v}$",                    f4(p8["random_effect"])),
        ("pooled I^2",         "I^2{=}{v}\\%",                  f"{p8['I2'] * 100:.1f}"),
        ("typed pooled RE",    "worth ${v}$",                   f4(P7["random_effect"])),
        ("three-way ablation", "costs ${v}$",                   f4(ng["mean"])),
        ("ablation corrected", "corrected $p={v}$",             f"{ng['p_bonf']:.4f}"),
        ("harder-split h2a",   "to ${v}$ ($p=",                 f4(HD["h2a"]["mean"])),
        ("measured floor",     "at ${v}$ response SDs",         f"{FL['floor']:.2f}"),
    ]
    fails = []
    SM = load("data/results/screening_lambda0/second_metric_5seed.json")["metrics"]
    INT = load("data/results/screening_c080c10_h1/robustness_5seed.json")["contrasts"]
    P7full = load("data/results/replication/pooled.json")["pooled"]["h2a"]
    pdh2a = P8["per_dataset"]["h2a"]
    N7pc = load("data/results/screening_untyped_n7/robustness_5seed.json")["per_config"]
    pear = {c["contrast"]: c for c in SM["pearson"]["contrasts"]}
    pcos = {c["contrast"]: c for c in SM["prog_cos"]["contrasts"]}
    cg_eo_p = pear.get("condition_gated - expression_only")
    cg_eo_c = pcos.get("condition_gated - expression_only")
    ts_eo_p = pear.get("typed_static - expression_only")

    claims += [
        # --- sec:null -------------------------------------------------------------------------
        ("h1 p-value",         "$p={v}$); the interval",           f"{L['h1_vs_no_graph']['p_value']:.2f}"),
        ("h2a p-value",        "(${v}$, $p=",                      f4(L["h2a"]["mean"])),
        ("h2a p",              "$p={v}$, surviving both",          f"{L['h2a']['p_value']:.4f}"),
        ("Pearson h1",         "gated graph (${v}$, $p=",          f4(cg_eo_p["mean"]) if cg_eo_p else "?"),
        ("Pearson h1 p",       "$p={v}$) and a deficit",           f"{cg_eo_p['p_value']:.2f}" if cg_eo_p else "?"),
        ("Pearson h2a",        "static graph (${v}$)",             f4(ts_eo_p["mean"]) if ts_eo_p else "?"),
        ("prog-cos h1",        "agreeing (${v}$, $p=",             f4(cg_eo_c["mean"]) if cg_eo_c else "?"),
        ("three folds: frozen","across folds (${v}$,",             f4(L["h1_vs_no_graph"]["mean"])),
        ("three folds: interm","(${v}$, $+0.0005$)" .replace("+0.0005","{w}"), None),
        ("harder h2a CI low",  "$95\\%$ CI $[{v},",                f4(HD["h2a"]["ci_low"])),
        ("harder h2a CI high", ",{v}]$, $p=",                      f4(HD["h2a"]["ci_high"])),
        ("harder h2a p",       "$p={v}$, clearing neither",        f"{HD['h2a']['p_value']:.2f}"),
        ("intermediate h2a",   "h2a is ${v}$, within",             f4(INT["h2a"]["mean"])),
        ("intermediate h2a p", "($p={v}$) because",                f"{INT['h2a']['p_value']:.2f}"),
        # --- sec:repl -------------------------------------------------------------------------
        ("typed pooled FE CI", "fixed-effect $95\\%$ CI $[{v},",   f4(P7full["fixed_ci"][0])),
        ("typed pooled I^2",   "$I^2{=}{v}\\%$, $Q$",              f"{P7full['I2']*100:.1f}"),
        ("typed pooled Q p",   "$Q$ $p{=}{v}$",                    f"{P7full['Q_p']:.2f}"),
        ("RPE1 typed arm",     "the other way (${v}$)",            f4(pdh2a["ReplogleWeissman2022_rpe1"]["mean"])),
        ("pooled 8 p",         "$p{=}{v}$) at $I^2",               f"{p8['random_p']:.2f}"),
        # --- sec:causes: per-arm means at n=7 --------------------------------------------------
        ("n7 mean untyped",    "untyped ${v}$,",                   f"{N7pc['untyped_gnn']['mean']:.4f}"),
        ("n7 mean expr",       "expression-only ${v}$,",           f"{N7pc['expression_only']['mean']:.4f}"),
        ("n7 mean gated",      "condition-gated ${v}$,",           f"{N7pc['condition_gated']['mean']:.4f}"),
        ("n7 mean typed",      "typed-static ${v}$",               f"{N7pc['typed_static']['mean']:.4f}"),
    ]
    A1 = load("data/results/screening_a1/a1_mechanism.json")
    AB = load("data/results/a2_power/arch_search_bound.json")
    B1 = load("data/results/screening_b1/b1_message_form.json")
    DEC = load("data/results/b2_deciles/deciles.json")["schemes"]["head"]
    a1f, a1c = A1["family"], A1["context"]

    claims += [
        # --- app:mechanism ---------------------------------------------------------------------
        ("A1 D1 (parameter cut)", "primary endpoint by ${v}$",     f4(a1f["D1"]["mean"])),
        ("A1 D2 (permuted)",      "moves it by ${v}$",             f4(a1f["D2"]["mean"])),
        ("A1 shared vs untyped",  "{gcn} (${v}$ and",              f4(a1c["shared_vs_untyped"]["mean"])),
        ("A1 permuted vs untyped","and ${v}$) and below",          f4(a1c["permuted_vs_untyped"]["mean"])),
        ("A1 shared vs nograph",  "baseline (${v}$ and",           f4(a1c["shared_vs_nograph"]["mean"])),
        ("A1 permuted vs nograph","and ${v}$), so the deficit",    f4(a1c["permuted_vs_nograph"]["mean"])),
        ("arch search spread",    "spans ${v}$ across all",        f"{AB['observed_spread']:.4f}"),
        ("arch search cells",     "across all ${v}$ of its cells", str(AB["n_cells"])),
        ("B1a gap",               "same fold is ${v}$",            f"{B1['gap_untyped_minus_typed']['mean']:.4f}"),
        ("B1a D3 uncorrected p",  "the uncorrected $p={v}$",       f"{B1['contrasts']['D3']['p_value']:.4f}"),
        # --- app:metrics -----------------------------------------------------------------------
        ("B2 top-20 deficit",     "it is large (${v}$)",           f4(DEC["cells"]["1-20/promotion_margin"]["mean"])),
        ("B2 family size",        "over all ${v}$ cells",          str(DEC["family_size"])),
    ]
    VD = load("data/results/l4/vardecomp_h2a.json")
    B3 = load("data/results/a2_ladder/b3_power.json")
    ZP = FL["zero_point"]
    inc = FL["post_hoc_increment_over_zero"]
    C1 = load("data/results/c1_ladder/floor_condition_gated.json")
    c1inc = C1["post_hoc_increment_over_zero"]["d400"]

    claims += [
        # --- app:power -------------------------------------------------------------------------
        ("per-fold limit frozen",  "upper limits are ${v}$ (frozen)",  f4(L["h1_vs_no_graph"]["ci_high"])),
        ("per-fold limit interm",  "${v}$ (intermediate)",             f4(INT["h1_vs_no_graph"]["ci_high"])),
        ("per-fold limit harder",  "and ${v}$ (harder)",               f4(HD["h1_vs_no_graph"]["ci_high"])),
        ("vardecomp seed sd",      "seed contributes ${v}$ against",   f"{VD['sd_seed']:.4f}"),
        ("vardecomp redraw sd",    "re-draw's ${v}$",                  f"{VD['sd_redraw']:.4f}"),
        # --- app:floor -------------------------------------------------------------------------
        ("ladder zero point",      "benefit of ${v}$ with no injection", f4(ZP["mean"])),
        ("control gap",            "gives a gap of ${v}$ with",        f4(FL["contrasts"]["permuted_d400"]["mean"])),
        ("control increment",      "zero point is ${v}$ ($[",          f4(inc["permuted_d400"]["mean"])),
        ("control incr CI low",    "$[{v},",                           f4(inc["permuted_d400"]["ci_low"])),
        ("control incr CI high",   ",{v}]$, $p=0.02$)",                f4(inc["permuted_d400"]["ci_high"])),
        ("d020 gap",               "rung produced a gap of ${v}$",     f4(FL["contrasts"]["d020"]["mean"])),
        ("d100 delta",             "rung's ${v}$ sits",                f4(FL["contrasts"]["d100"]["mean"])),
        ("B3 MDE control sd",      "four seeds of ${v}$ on this",      f"{B3['mde']['control_sd_uncorrected']:.4f}"),
        ("B3 MDE rungs median",    "spread, ${v}$ on the injected",    f"{B3['mde']['real_sd_uncorrected']:.4f}"),
        ("B3 MDE corrected",       "and ${v}$ once the enlarged",      f"{B3['mde']['control_sd_bonferroni']:.4f}"),
        ("B3 proportional pred",   "damage would be ${v}$ there",      f4(B3["predictions_at_target_delta"]["proportional"])),
        ("C1 post-hoc increment",  "zero point is ${v}$ ($95",         f4(c1inc["mean"])),
        ("C1 increment CI low",    "CI $[{v},",                        f4(c1inc["ci_low"])),
        ("C1 increment CI high",   ",{v}]$, $p=0.035$",                f4(c1inc["ci_high"])),
    ]
    SCR = load("data/results/screening/robustness_5seed.json")["contrasts"]
    # The three fold RE-DRAWS are the c080c10 family. The c075c15_* roots are the harder-threshold
    # folds and ran only typed_static and expression_only, so their h1 is null.
    R2 = load("data/results/screening_c080c10_r2/robustness_5seed.json")["contrasts"]
    R3 = load("data/results/screening_c080c10_r3/robustness_5seed.json")["contrasts"]

    def dlist(c, sign=1):
        """The per-seed deltas as the paper writes them: '$+0.0090$, $+0.0050$, ...'."""
        return ", ".join(f"${sign * d:+.4f}$" for d in c["deltas"])

    claims += [
        # --- per-seed delta LISTS: one anchor covers every value in the list -------------------
        ("n=7 untyped per-seed list", "none dropped: {v}.", dlist(pm)),
        ("harder h2a per-seed list",  "show why: {v}.",     dlist(HD["h2a"])),
        # --- the n=7 summary statistics ---------------------------------------------------------
        ("n=7 mean",  "Mean ${v}$, sd",  f4(pm["mean"])),
        ("n=7 sd",    "sd ${v}$, se",    f"{pm['sd']:.4f}"),
        ("n=7 se",    "se ${v}$, paired", f"{pm['se']:.4f}"),
        ("n=7 raw p", "paired $p={v}$ before", f"{pm['p_value']:.4f}"),
        # --- the n=5 version of the same contrast ------------------------------------------------
        ("n=5 untyped mean",  "contrast was ${v}$,",   f4(SCR["promotion_margin"]["mean"])),
        ("n=5 untyped CI low","CI $[{v},",             f4(SCR["promotion_margin"]["ci_low"])),
        ("n=5 untyped Holm",  "Holm ${v}$ but",        f"{SCR['promotion_margin']['p_holm']:.3f}"),
        ("n=5 untyped Bonf",  "Bonferroni ${v}$.",     f"{SCR['promotion_margin']['p_bonferroni']:.3f}"),
        # --- the architecture search bound -------------------------------------------------------
        ("arch best cell",    "the best is ${v}$ against", f4(AB["best_delta"])),
        ("arch worst cell",   "the worst is ${v}$,",       f4(AB["worst_delta"])),
        ("arch seed sd",      "sd}={v}$) would be",        f"{AB['seed_sd_frozen']:.4f}"),
        ("arch expected span","expected to span ${v}$",    f"{AB['expected_spread_from_seed_noise_alone']:.4f}"),
        ("arch seeds needed", "would need ${v}$ paired seeds", str(AB["seeds_that_would_be_needed_for_the_best_cell"])),
        # --- the three fold re-draws -------------------------------------------------------------
        ("redraw 2 h1",       "$p=0.025$), ${v}$ ($[",     f4(R2["h1_vs_no_graph"]["mean"])),
        ("redraw 3 h1",       "and ${v}$ ($[-0.0024",      f4(R3["h1_vs_no_graph"]["mean"])),
        # --- the unrepaired headline, from the pre-repair root -----------------------------------
        ("unrepaired h1",     "moved the headline from ${v}$ to", f4(SCR["h1_vs_no_graph"]["mean"])),
        # --- the genome-wide tightest point -------------------------------------------------------
        # "tightest single point of all" is the TYPED contrast in that sentence, not the untyped one.
        ("gwps point",        "single point of all (${v}$)", f4(pdh2a["ReplogleWeissman2022_K562_gwps"]["mean"])),
        # --- the full eight-dataset ordering ------------------------------------------------------
        ("Tian CRISPRa",      "Tian CRISPRa ${v}$,",  f4(pd8["TianKampmann2021_CRISPRa"]["mean"])),
        ("Tian CRISPRi",      "Tian CRISPRi ${v}$,",  f4(pd8["TianKampmann2021_CRISPRi"]["mean"])),
        ("Frangieh untyped",  "Frangieh ${v}$,",      f4(pd8["FrangiehIzar2021_RNA"]["mean"])),
        ("K562-essential",    "K562-essential ${v}$,", f4(pd8["ReplogleWeissman2022_K562_essential"]["mean"])),
        # --- the K=128 subset (rail-4 finding) ----------------------------------------------------
        ("K128 subset RE",    "datasets it is ${v}$ ($[", f4(load("data/results/replication/pooled_k128_subset.json")["pooled"]["promotion_margin"]["random_effect"])),
        # --- B1a gap and D3 -----------------------------------------------------------------------
        ("B1a gap in share",  "of the ${v}$ gap",     f4(B1["gap_untyped_minus_typed"]["mean"])),
    ]
    RA = load("data/results/rationale_audit_lambda0/audit_report.json")
    I1 = load("data/results/screening_c080c10_h1/robustness_5seed.json")["per_config"]
    I2 = load("data/results/screening_c080c10_r2/robustness_5seed.json")["per_config"]
    I3 = load("data/results/screening_c080c10_r3/robustness_5seed.json")["per_config"]
    gx = RA["aggregate"]["ginx_by_sparsity"]["0.20"]

    claims += [
        # --- the three re-draws' baseline systema, which the paper quotes as a spread -----------
        ("re-draw 1 baseline", "it scores ${v}$,",  f"{I1['expression_only']['mean']:.4f}"),
        ("re-draw 3 baseline", "${v}$ and $0.0845$", f"{I3['expression_only']['mean']:.4f}"),
        ("re-draw 2 baseline", "and ${v}$ \\textsc{systema}", f"{I2['expression_only']['mean']:.4f}"),
        # --- the rationale audit's GInX comparison ---------------------------------------------
        ("GInX rationale",     "is GInX, ${v}$ against", f"{gx['rationale']:.3f}"),
        ("GInX random",        "random's ${v}$ at",      f"{gx['random']:.3f}"),
    ]
    I1c = load("data/results/screening_c080c10_h1/robustness_5seed.json")["contrasts"]
    R2c, R3c = R2, R3
    FAB = load("data/results/feature_ablation_report.json")
    nod = next(v for v in FAB if v["variant"] == "nodegree")

    claims += [
        # --- the three folds, all three values --------------------------------------------------
        ("three folds: intermediate", "${v}$, $+0.0005$)", f4(I1c["h1_vs_no_graph"]["mean"])),
        ("three folds: harder",       ", ${v}$), which is why", f4(HD["h1_vs_no_graph"]["mean"])),
        # --- the harder fold's per-seed h1 list (NOT the intermediate one) ----------------------
        ("harder h1 per-seed", "per-seed differences {v}).", dlist(HD["h1_vs_no_graph"])),
        # --- PER-ARM seed spreads. The paper quotes the arm's own sd, not the contrast's. --------
        ("intermediate arm sd", "deviation is ${v}$ there",
         f"{load('data/results/screening_c080c10_h1/robustness_5seed.json')['per_config']['typed_static']['sd']:.4f}"),
        ("frozen arm sd",       "against ${v}$ on the frozen",
         f"{load('data/results/screening_lambda0/robustness_5seed.json')['per_config']['typed_static']['sd']:.4f}"),
        # --- Bonferroni over the PRE-REGISTERED family of four. The re-draw roots ran fewer arms,
        # --- so their own p_bonferroni uses a smaller family; the paper corrects at m=4, which is
        # --- the stricter and pre-registered choice, so the check re-derives it that way. --------
        ("redraw Bonf 1", "family of four (${v}$,", f"{min(1.0, 4 * I1c['h1_vs_no_graph']['p_value']):.3f}"),
        ("redraw Bonf 2", ", ${v}$, $1.000$)",      f"{min(1.0, 4 * R2c['h1_vs_no_graph']['p_value']):.3f}"),
        # --- the gated arm against no graph at n=7 -----------------------------------------------
        ("gated vs no graph n=7", "leaving the gated arm at ${v}$", f4(N7["h1_vs_no_graph"]["mean"])),
        # --- B1a's recovered effect, in its own sentence -----------------------------------------
        ("B1a D3 in prose", "it recovers ${v}$ \\textsc{systema}", f4(B1["contrasts"]["D3"]["mean"])),
    ]
    SPF = load("data/splits/manifest.json")
    SPH = load("data/results/splits_c075c15/manifest.json")
    PSIM = load("data/results/a2_power/power_simulation.json")
    DECI = load("data/results/b2_deciles/deciles.json")["schemes"]["deciles"]["cells"]
    C1H = C1["gate_health"]
    L1pc = load("data/results/screening_lambda0/robustness_5seed.json")["per_config"]
    n7means = sorted(v["mean"] for v in N7pc.values())
    l5means = sorted(v["mean"] for v in L1pc.values())

    def halfwidth(c):
        return (c["ci_high"] - c["ci_low"]) / 2

    claims += [
        # --- split-construction parameters, from the split manifests -----------------------------
        ("frozen split threshold", "threshold from ${v}$ to", f"{SPF['seq_cosine_threshold']:.2f}"),
        ("harder split threshold", "to ${v}$ and raising",    f"{SPH['seq_cosine_threshold']:.2f}"),
        # --- alpha, from the analysis artifact rather than assumed -------------------------------
        ("nominal alpha",          "not the nominal ${v}$",   f"{PSIM['alpha']:.2f}"),
        # --- between-dataset spread ---------------------------------------------------------------
        ("between-dataset tau",    "contrast is ${v}$, five times", f"{PSIM['measured']['promotion_margin']['tau']:.4f}"),
        # --- the decile dilution and the eight-dataset ordering -----------------------------------
        ("decile dilution",        "diluted to a positive ${v}$", f4(DECI["d1/promotion_margin"]["mean"])),
        ("gwps in the ordering",   "genome-wide ${v}$,",          f4(pd8["ReplogleWeissman2022_K562_gwps"]["mean"])),
        # --- h2b's failure to clear ---------------------------------------------------------------
        ("h2b Bonferroni n=7",     "correction (Bonferroni ${v}$)", f"{N7['h2b']['p_bonferroni']:.3f}"),
        # --- the ablation baseline and its magnitude ----------------------------------------------
        ("ablation baseline mean", "baseline of ${v}$",       f"{L1pc['expression_only']['mean']:.4f}"),
        ("ablation magnitude",     "absolute terms; ${v}$ against", f"{abs(ng['mean']):.4f}"),
        # --- the band the four trained arms occupy, read off the per-arm means --------------------
        # The band is stated to CONTAIN the four arms, so its floor is truncated and its ceiling
        # rounded up. Demanding ordinary rounding on the floor would reject a correct sentence:
        # typed_static is 0.0726, and a band starting at 0.073 would not contain it.
        ("arm band low",           "band from ${v}$ to",      f"{math.floor(l5means[0] * 1000) / 1000:.3f}"),
        ("arm band high",          "to ${v}$. An independent", f"{l5means[-1]:.3f}"),
        # --- the largest ladder rung and the C1 gate minimum ---------------------------------------
        ("largest rung",           "any size up to ${v}$,",   f"{FL['rungs']['d400']['delta']:.2f}"),
        ("C1 gate minimum",        "minimum mean ${v}$ against", f"{C1H['min_gate_mean']:.4f}"),
        # --- CI half-widths and interval widths: arithmetic on artifact values, not assertions -----
        ("n=7 CI half-width",      "half-width from $0.0034$ to ${v}$", f"{halfwidth(pm):.4f}"),
        ("n=5 CI half-width",      "half-width from ${v}$ to", f"{halfwidth(SCR['promotion_margin']):.4f}"),
        ("control interval width", "interval of width ${v}$ around",
         f"{FL['contrasts']['permuted_d400']['ci_high'] - FL['contrasts']['permuted_d400']['ci_low']:.3f}"),
        ("re-draw baseline spread","\\textsc{systema}, a ${v}$ spread",
         f"{max(I1['expression_only']['mean'], I2['expression_only']['mean'], I3['expression_only']['mean']) - min(I1['expression_only']['mean'], I2['expression_only']['mean'], I3['expression_only']['mean']):.4f}"),
    ]
    RPE = load("data/results/replication/ReplogleWeissman2022_rpe1/robustness_5seed.json")["contrasts"]["promotion_margin"]
    claims += [
        ("RPE1 CI low",      "CI $[{v},",                 f4(RPE["ci_low"])),
        ("RPE1 CI high",     ",{v}]$; per-seed",          f4(RPE["ci_high"])),
        # written as ONE math block with commas inside, not one $...$ per value
        ("RPE1 per-seed",    "per-seed {v};",             "$" + ",".join(f"{d:+.3f}" for d in RPE["deltas"]) + "$"),
        ("RPE1 p",           "$p{=}{v}$, Bonferroni",     f"{RPE['p_value']:.3f}"),
        ("RPE1 Bonf/Holm",   "Holm both ${v}$ at family", f"{RPE['p_bonferroni']:.3f}"),
        ("RPE1 family size", "at family size {v})",       "two" if RPE["family_size"] == 2 else str(RPE["family_size"])),
    ]
    QC = load("data/results/replication/frangieh_on_target_qc.json")
    claims += [
        ("Frangieh QC own mean",   "FC mean ${v}$,",        f"{QC['own_mean']:.3f}"),
        ("Frangieh QC own %neg",   "${v}\\%$ negative,",    f"{QC['own_frac_negative']*100:.0f}"),
        ("Frangieh QC random mean","against ${v}$ and",     f"{QC['random_mean']:.3f}"),
        ("Frangieh QC random %neg","and ${v}\\%$ for random", f"{QC['random_frac_negative']*100:.0f}"),
    ]
    claims = [c for c in claims if c[2] is not None]

    for what, tmpl, val in claims:
        ctx = tmpl.replace("{v}", val)
        if ctx not in flat:
            fails.append(f"prose {what}: the artifact gives {val}, and the paper does not state it in "
                         f"context (looked for {ctx!r})")
    return fails, (f"full: {len(claims)} load-bearing prose claims re-derived from 7 artifacts, each "
                   f"anchored to its surrounding words")


@table("tab:family")
def _family(tex):
    """The headline five-seed family.

    IT IS THE screening_lambda0 ROOT, NOT data/results/screening/. Both are n=5 on the frozen fold
    and they agree on three arms out of four, which is exactly how this could be got wrong: the
    unrepaired root puts condition_gated at 0.0838 and h1 at -0.0019, the repaired one at 0.0848 and
    -0.0009, and the paper's own text says the gate repair moved the headline between those two
    values. Checking against the wrong root would report a real number as an error."""
    fails, d = [], load(f"{_FAMILY_ROOT}/robustness_5seed.json")
    sec = load(f"{_FAMILY_ROOT}/second_metric_5seed.json")
    pear = {c["contrast"]: c for c in sec["metrics"]["pearson"]["contrasts"]}
    seen = 0
    for cs in cells_of(tex, "tab:family"):
        m = re.match(r"\\?t?e?x?t?b?f?\{?(ts|cg|ug|eo)\}?\s*\$?-?\$?\s*\$?-?\$?\s*(ts|cg|ug|eo)",
                     cs[0].replace("\\textbf{", "").replace("$-$", "-"))
        if not m or len(cs) < 5:
            continue
        a, b = _FAMILY_KEY[m.group(1)], _FAMILY_KEY[m.group(2)]
        key = _CONTRAST_OF.get((a, b))
        if key is None:
            continue
        seen += 1
        c = d["contrasts"][key]
        # The artifact orients every contrast better-minus-worse; the table writes A minus B.
        sign = 1.0 if (c["better"], c["worse"]) == (a, b) else -1.0
        cmp_cell(fails, f"tab:family {m.group(1)}-{m.group(2)} delta", cs[1], sign * c["mean"])
        lo, hi = (sign * c["ci_low"], sign * c["ci_high"])
        if sign < 0:
            lo, hi = hi, lo
        ci = re.findall(r"[-+]\d\.\d+", cs[2])
        if len(ci) == 2:
            nd = len(ci[0].split(".")[1])
            if (ci[0], ci[1]) != (f"{lo:+.{nd}f}", f"{hi:+.{nd}f}"):
                fails.append(f"tab:family {key} CI: paper [{ci[0]},{ci[1]}] artifact "
                             f"[{lo:+.{nd}f},{hi:+.{nd}f}]")
        # FWER column vocabulary, spelled out because it is not a yes/no field: "worse" and
        # "better" CLAIM survival (naming which arm won), "no" and "parity" deny it. Reading
        # anything-but-no as survival turns the headline null's own cell into a false failure.
        word = cs[3].lower()
        claims = ("worse" in word or "better" in word) and "no" not in word
        if claims != bool(c["survives_family_wise"]):
            fails.append(f"tab:family {key} FWER: paper {cs[3]!r} artifact "
                         f"survives={c['survives_family_wise']}")
        pc = pear.get(f"{a} - {b}") or pear.get(f"{b} - {a}")
        if pc is not None:
            psign = 1.0 if pc["A"] == a else -1.0
            cmp_cell(fails, f"tab:family {key} pearson", cs[4], psign * pc["mean"])

    # The per-arm mean footer is a \multicolumn line, so it is not a row; parse it directly.
    foot = re.search(r"Per-arm.*?systema.*?\$n\{=\}5\$\):(.*?)\\\\", tex, re.S)
    if foot:
        for short, val in re.findall(r"(ts|cg|ug|eo)\s*\$(\d\.\d+)\$", foot.group(1)):
            arm = _FAMILY_KEY[short]
            want = d["per_config"][arm]["mean"]
            nd = len(val.split(".")[1])
            if f"{float(val):.{nd}f}" != f"{want:.{nd}f}":
                fails.append(f"tab:family per-arm {arm}: paper {val} artifact {want:.{nd}f}")
    return fails, (f"full: 4 contrasts x (delta, CI, FWER, Pearson) + 4 per-arm means, "
                   f"{seen} rows, vs {_FAMILY_ROOT}/") if seen == 4 else \
                  (fails, f"PARTIAL: matched {seen} of 4 contrast rows")


@table("tab:causes")
def _causes(tex):
    """A SUMMARY table: its cells are verdicts in prose, and most of its evidence is quoted from
    the appendices rather than computed here. What IS re-derivable is checked; the rest is named.

    NOT CHECKED, and why: cause A's "$\\approx$0.001" centroid floor and "6 to 32x" seed variance,
    and cause E's "86% functional associations", are graph- and metric-summary quantities with no
    JSON of their own in this repo. Cause B and E's verdicts are qualitative. Those live or die by
    Appendix~\\ref{app:causes}, whose own numbers are covered by tab:metrics and tab:ksweep."""
    fails, blk = [], re.search(r"\\label\{tab:causes\}(.*?)\\end\{tabular\}", tex, re.S)
    body = blk.group(1) if blk else ""
    n7 = load("data/results/screening_untyped_n7/robustness_5seed.json")["contrasts"]
    pool = load("data/results/replication/pooled_with_reference.json")
    feat = load("data/results/feature_ablation_report.json")

    checked = 0
    for key, label in (("h2a", "n=7 h2a"), ("h2b", "n=7 h2b")):
        lit = f"{n7[key]['mean']:+.4f}"
        if lit not in body:
            fails.append(f"tab:causes cause C: {label} should read {lit} and does not appear")
        checked += 1

    # The three per-dataset contrasts that clear correction and disagree in sign.
    surviving = sorted(v["mean"] for v in pool["per_dataset"]["promotion_margin"].values()
                       if v.get("survives_family_wise"))
    for v in surviving:
        if f"{v:+.4f}" not in body:
            fails.append(f"tab:causes cause C: surviving per-dataset {v:+.4f} does not appear")
    checked += len(surviving)

    ng = next(v for v in feat if v["variant"] == "nograph")
    for lit, what in ((f"{ng['mean']:+.4f}", "cause D delta"), (f"{ng['p_bonf']:.4f}", "cause D p")):
        if lit not in body:
            fails.append(f"tab:causes cause D: {what} should read {lit} and does not appear")
    checked += 2
    return fails, (f"partial: {checked} re-derivable quantities in causes C and D; causes A, B and "
                   f"E are qualitative verdicts with no JSON of their own (see docstring)")


@table("tab:mechanism")
def _mechanism(tex):
    fails, fam = [], load("data/results/screening_a1/a1_mechanism.json")["family"]
    n = 0
    for line in re.search(r"\\label\{tab:mechanism\}(.*?)\\end\{tabular\}",
                          tex, re.S).group(1).split("\\\\"):
        if "$D_" not in line:
            continue
        key = "D1" if "$D_1$" in line else "D2"
        c = fam[key]
        got = re.findall(r"[-+]\d\.\d{4}", line)
        want = [f"{c['mean']:+.4f}", f"{c['ci_low']:+.4f}", f"{c['ci_high']:+.4f}"]
        n += 1
        if got != want:
            fails.append(f"tab:mechanism {key}: paper {got} artifact {want}")
    return fails, f"full: {n} rows x (mean, CI) vs data/results/screening_a1/a1_mechanism.json"


def _v2_provenance(paper_name: str):
    """The POST-Amendment-2 build for a dataset named as the paper names it.

    Amendment 2 replaced the DE unit, and the rebuilt matrices are the `*.DE_stats_v2.*` artifacts.
    The pre-amendment `<name>.provenance.json` files still sit beside them with the SUPERSEDED counts,
    and reading those is how tab:repl came to report Replogle as "dropped" at 12 targets when the
    corrected rule gives 2,122. Match on prefix because the paper writes FrangiehIzar2021 where the
    build is FrangiehIzar2021_RNA."""
    want = re.sub(r"[^A-Za-z0-9]", "", paper_name)
    if not want:
        return None, None, None
    base = ROOT / "data/intermediate/replication"
    for f in sorted(base.glob("*.DE_stats_v2.provenance.json")):
        name = f.name.replace(".DE_stats_v2.provenance.json", "")
        if re.sub(r"[^A-Za-z0-9]", "", name).startswith(want):
            qc = f.with_name(f.name.replace("provenance", "qc"))
            verdict = json.loads(qc.read_text())["verdict"] if qc.exists() else None
            return name, json.loads(f.read_text()), verdict
    return None, None, None


def _program_k(dataset: str):
    """K actually used, measured from the built basis rather than from any declared value.

    Amendment 3.1 requires every K != 128 to be labelled wherever it appears, and the paper labelled
    only one of three. Measuring it from program_response.parquet's width means the label is checked
    against what the lane really ran, not against a runner argument nobody recorded."""
    import pandas as pd
    p = ROOT / f"data/intermediate/replication/{dataset}/program_response.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).shape[1] - 1


@table("tab:repl")
def _repl(tex):
    """Four of six columns re-derived per dataset, against the POST-Amendment-2 artifacts: targets and
    condition count from the DE_stats_v2 provenance, the on-target QC verdict from its qc.json, and K
    measured from the built program basis.

    NOT CHECKED, and why: "Verdict" is a pre-registered judgement rather than a measurement, and the
    dataset name is the key rather than a value."""
    fails, seen, unknown = [], 0, []
    for cs in cells_of(tex, "tab:repl"):
        name = cs[0].replace("\\_", "_").strip()
        if not name or not name[0].isupper() or name.startswith("Dataset"):
            continue
        ds, prov, qc = _v2_provenance(name)
        if prov is None:
            unknown.append(name)
            continue
        seen += 1
        tgt = re.search(r"(\d[\d,]*)", cs[1].replace("{,}", ","))
        if tgt and int(tgt.group(1).replace(",", "")) != prov["n_targets"]:
            fails.append(f"tab:repl {name} targets: paper {tgt.group(1)} artifact {prov['n_targets']}")
        nc = re.search(r"(\d+)", cs[2])
        if nc and int(nc.group(1)) != len(prov["conditions"]):
            fails.append(f"tab:repl {name} conditions: paper {nc.group(1)} artifact "
                         f"{len(prov['conditions'])}")
        k_art = _program_k(ds)
        k_pap = re.search(r"(\d+)", cs[3])
        if k_pap and k_art is not None and int(k_pap.group(1)) != k_art:
            fails.append(f"tab:repl {name} K: paper {k_pap.group(1)} basis width says {k_art} "
                         f"(Amendment 3.1: every K!=128 must be labelled wherever it appears)")
        elif k_pap and k_art is None:
            fails.append(f"tab:repl {name} K: paper claims {k_pap.group(1)} but no basis was built")
        elif not k_pap and k_art is not None and (ROOT / f"data/results/replication/{ds}").is_dir():
            fails.append(f"tab:repl {name} K: paper shows none but the lane ran at K={k_art}")
        # A basis with NO trained lane is not a K deviation to label. Papalexi has one at K=8 from what
        # Amendment 3.1 itself calls a "chain smoke only, never headlined"; the K column reports what a
        # LANE ran at, so a dash is correct there. The results directory is what distinguishes the two.
        if qc:
            said_pass = "pass" in cs[4].lower()
            if said_pass != qc.startswith("PASS"):
                fails.append(f"tab:repl {name} QC: paper {cs[4]!r} artifact {qc!r}")
    note = (f"full on 4 of 6 columns: {seen} datasets x (targets, conditions, K measured from the "
            f"built basis, on-target QC) vs the POST-Amendment-2 DE_stats_v2 artifacts; Verdict is a "
            f"pre-registered judgement, not a measurement")
    if unknown:
        note += f"; NO DE_stats_v2 BUILD for {unknown}, so those rows are unchecked"
    return fails, note


_FOLD_ROOT = {"frozen": "data/results/screening_lambda0",
              "intermediate": "data/results/screening_c080c10_h1",
              "harder": "data/results/screening_c075c15_n5"}


@table("tab:folds")
def _folds(tex):
    fails, n = [], 0
    for cs in cells_of(tex, "tab:folds"):
        which = next((k for k in _FOLD_ROOT if cs[0].startswith(k)), None)
        if which is None or len(cs) < 5:
            continue
        n += 1
        d = load(f"{_FOLD_ROOT[which]}/robustness_5seed.json")["contrasts"]
        h1 = d["h1_vs_no_graph"]
        sign = 1.0 if h1["better"] == "condition_gated" else -1.0
        cmp_cell(fails, f"tab:folds {which} h1", cs[1], sign * h1["mean"])
        lo, hi = sign * h1["ci_low"], sign * h1["ci_high"]
        if sign < 0:
            lo, hi = hi, lo
        ci = re.findall(r"[-+]\d\.\d+", cs[2])
        if len(ci) == 2:
            nd = len(ci[0].split(".")[1])
            if (ci[0], ci[1]) != (f"{lo:+.{nd}f}", f"{hi:+.{nd}f}"):
                fails.append(f"tab:folds {which} h1 CI: paper [{ci[0]},{ci[1]}] artifact "
                             f"[{lo:+.{nd}f},{hi:+.{nd}f}]")
        pb = re.search(r"(\d\.\d+)", cs[3])
        if pb and f"{h1['p_bonferroni']:.{len(pb.group(1).split('.')[1])}f}" != pb.group(1):
            fails.append(f"tab:folds {which} p_bonf: paper {pb.group(1)} artifact "
                         f"{h1['p_bonferroni']:.4f}")
        h2 = d["h2a"]
        s2 = 1.0 if (h2["better"], h2["worse"]) == ("typed_static", "expression_only") else -1.0
        cmp_cell(fails, f"tab:folds {which} h2a", cs[4], s2 * h2["mean"])
    return fails, (f"full: {n} folds x (h1 delta, CI, Bonferroni, h2a delta) vs three screening "
                   f"roots")


@table("tab:floor")
def _floor(tex):
    """The power simulation. Every row is one design point at the effect the paper tests (0.0043)."""
    fails, d = [], load("data/results/a2_power/power_simulation.json")["designs"]
    E = 0.0043

    def pick(group, arm, **want):
        for r in d[group][arm]:
            if abs(r.get("effect", -1) - E) < 1e-9 and all(r.get(k) == v for k, v in want.items()):
                return r
        return None

    plan = {
        "seeds, this fold": (pick("seeds", "h2a", sd_source="frozen"), "sd", "seeds_for_80pct",
                             "power_at_7"),
        "seeds, any re-draw": (pick("seeds", "h2a", sd_source="pooled"), "sd", "seeds_for_80pct",
                               "power_at_7"),
        "a fresh partition": (pick("nested", "h2a", redraws=1, seeds=5), "sd_level_mean",
                              "levels_for_80pct", None),
        "datasets, typed": (pick("datasets", "h2a"), None, "datasets_for_80pct", "power_at_7"),
        "datasets, untyped": (pick("datasets", "promotion_margin"), None, "datasets_for_80pct",
                              "power_at_7"),
    }
    n = 0
    for cs in cells_of(tex, "tab:floor"):
        row = next((k for k in plan if cs[0].startswith(k)), None)
        if row is None or len(cs) < 4:
            continue
        art, sd_key, unit_key, pow_key = plan[row]
        if art is None:
            fails.append(f"tab:floor {row!r}: no design point at effect={E} in the artifact")
            continue
        n += 1
        if sd_key:
            sd = re.search(r"(\d\.\d+)", cs[2])
            if sd and f"{art[sd_key]:.{len(sd.group(1).split('.')[1])}f}" != sd.group(1):
                fails.append(f"tab:floor {row} sd: paper {sd.group(1)} artifact {art[sd_key]:.4f}")
        u = re.search(r"(\d+)", cs[3])
        if u and int(u.group(1)) != art[unit_key]:
            fails.append(f"tab:floor {row} units: paper {u.group(1)} artifact {art[unit_key]}")
        if pow_key and len(cs) > 4:
            p = re.search(r"(\d\.\d+)", cs[4])
            if p and f"{art[pow_key]:.{len(p.group(1).split('.')[1])}f}" != p.group(1):
                fails.append(f"tab:floor {row} power: paper {p.group(1)} artifact "
                             f"{art[pow_key]:.3f}")
    return fails, (f"full: {n} design rows x (sd, units for 80%, power as run) vs "
                   f"data/results/a2_power/power_simulation.json")


_RUNG = {"0.02": "d020", "0.05": "d050", "0.10": "d100", "0.20": "d200", "0.40": "d400"}


def _rung_of(cs):
    """Which ladder rung a floor-table row is, from its own leading cells.

    The delta sits in a column of its own in Table~\\ref{tab:floor2} and inside the row label
    ($\\delta{=}0.02$) in Table~\\ref{tab:floorabs}, so both leading cells are searched. The
    negative lookahead matters: without it, an absolute score of $0.0879$ reads as the rung
    ``0.08``, which silently drops five of seven rows while still reporting a pass."""
    low = cs[0].lower()
    if "no injection" in low:
        return "zero"
    if "scrambled" in low:
        return "permuted_d400"
    m = re.search(r"(\d\.\d\d)(?!\d)", " ".join(cs[:2]))
    return _RUNG.get(m.group(1)) if m else None


@table("tab:floor2")
def _floor2(tex):
    fails, d = [], load("data/results/a2_ladder/floor.json")
    n = 0
    for cs in cells_of(tex, "tab:floor2"):
        rung = _rung_of(cs)
        if rung is None or len(cs) < 6:
            continue
        n += 1
        c = d["zero_point"] if rung == "zero" else d["contrasts"][rung]
        cmp_cell(fails, f"tab:floor2 {rung} delta", cs[2], c["mean"])
        ci = re.findall(r"[-+]\d\.\d+", cs[3])
        if len(ci) == 2:
            nd = len(ci[0].split(".")[1])
            want = (f"{c['ci_low']:+.{nd}f}", f"{c['ci_high']:+.{nd}f}")
            if (ci[0], ci[1]) != want:
                fails.append(f"tab:floor2 {rung} CI: paper [{ci[0]},{ci[1]}] artifact {want}")
        if rung != "zero":
            pb = re.search(r"(\d\.\d+)", cs[4])
            if pb and f"{c['p_bonferroni']:.{len(pb.group(1).split('.')[1])}f}" != pb.group(1):
                fails.append(f"tab:floor2 {rung} p_bonf: paper {pb.group(1)} artifact "
                             f"{c['p_bonferroni']:.4f}")
            inc = d["post_hoc_increment_over_zero"].get(rung)
            if inc:
                cmp_cell(fails, f"tab:floor2 {rung} increment", cs[5], inc["mean"])
    return fails, (f"full: {n} rungs x (delta, CI, Bonferroni, post-hoc increment) vs "
                   f"data/results/a2_ladder/floor.json")


@table("tab:floorabs")
def _floorabs(tex):
    """Absolute per-arm means per rung, and the change against no injection.

    The no-injection row is NOT in floor.json - the ladder reads its zero point off the landed
    reference lanes (Amendment 6.4), so this recomputes it from those parquet files the same way
    ladder_report does. Without it the two delta columns cannot be checked at all, since both are
    differences against that row."""
    import pandas as pd
    fails, d = [], load("data/results/a2_ladder/floor.json")
    ref = ROOT / "data/results/screening_untyped_n7"
    base = {}
    for arm in ("expression_only", "untyped_gnn"):
        vals = []
        for s in d["seeds"]:
            p = ref / arm / f"{s}.parquet"
            if not p.exists():
                continue
            row = pd.read_parquet(p).iloc[0].to_dict()
            if row.get("status") == "completed" and row.get("systema") == row.get("systema"):
                vals.append(float(row["systema"]))
        base[arm] = sum(vals) / len(vals) if vals else None
    if base["expression_only"] is None or base["untyped_gnn"] is None:
        raise Missing(f"{ref}/<arm>/<seed>.parquet for the no-injection row")

    n = 0
    for cs in cells_of(tex, "tab:floorabs"):
        rung = _rung_of(cs if len(cs) > 1 else cs + [""])
        if rung is None or len(cs) < 3:
            continue
        if rung == "zero":
            for col, arm in ((cs[1], "expression_only"), (cs[2], "untyped_gnn")):
                v = re.search(r"(\d\.\d+)", col)
                if v and f"{base[arm]:.{len(v.group(1).split('.')[1])}f}" != v.group(1):
                    fails.append(f"tab:floorabs no-injection {arm}: paper {v.group(1)} "
                                 f"reference lanes {base[arm]:.4f}")
            n += 1
            continue
        r = d["rungs"][rung]
        means = {"expression_only": sum(r["worse"].values()) / len(r["worse"]),
                 "untyped_gnn": sum(r["better"].values()) / len(r["better"])}
        n += 1
        for col, arm in ((cs[1], "expression_only"), (cs[2], "untyped_gnn")):
            v = re.search(r"(\d\.\d+)", col)
            if v and f"{means[arm]:.{len(v.group(1).split('.')[1])}f}" != v.group(1):
                fails.append(f"tab:floorabs {rung} {arm}: paper {v.group(1)} artifact "
                             f"{means[arm]:.4f}")
        if len(cs) >= 5:
            for col, arm in ((cs[3], "expression_only"), (cs[4], "untyped_gnn")):
                cmp_cell(fails, f"tab:floorabs {rung} change in {arm}", col,
                         means[arm] - base[arm])
    return fails, (f"full: {n} rows x (2 absolute means, 2 changes vs no injection) vs "
                   f"floor.json plus the landed reference lanes")


_METRIC_COL = ["pearson_delta", "pearson_delta_top20", "mse_top20", "edistance_scperturb",
               "energy_distance"]
_METRIC_ROW = [("h2a", 1.0), ("h2b", 1.0), ("promotion_margin", 1.0), ("h1_vs_no_graph", 1.0)]


@table("tab:floorgated")
def _floorgated(tex):
    """C1's ladder (Amendment 9) beside Amendment 6's, both re-derived from their own artifacts.

    This table is the one that corrected three "the typed arm's own floor is unmeasured" statements,
    so it carries more weight than its size suggests and is checked against BOTH ladders rather than
    transcribed from either."""
    fails = []
    u = load("data/results/a2_ladder/floor.json")["contrasts"]
    g = load("data/results/c1_ladder/floor_condition_gated.json")["contrasts"]
    n = 0
    for cs in cells_of(tex, "tab:floorgated"):
        rung = _rung_of(cs)
        if rung is None or rung == "zero" or len(cs) < 5:
            continue
        n += 1
        cmp_cell(fails, f"tab:floorgated {rung} untyped delta", cs[1], u[rung]["mean"])
        cmp_cell(fails, f"tab:floorgated {rung} gated delta", cs[3], g[rung]["mean"])
        for cell, art, who in ((cs[2], u[rung], "untyped"), (cs[4], g[rung], "gated")):
            m = re.search(r"(\d\.\d+)", cell)
            if m and f"{art['p_bonferroni']:.{len(m.group(1).split('.')[1])}f}" != m.group(1):
                fails.append(f"tab:floorgated {rung} {who} Bonferroni: paper {m.group(1)} artifact "
                             f"{art['p_bonferroni']:.4f}")
    return fails, (f"full: {n} rungs x (untyped delta, untyped Bonferroni, gated delta, gated "
                   f"Bonferroni) vs a2_ladder/floor.json and c1_ladder/floor_condition_gated.json")


@table("tab:metrics")
def _metrics(tex):
    fails, d = [], load("data/results/a3_external/rescored.json")
    n = 0
    for cs in cells_of(tex, "tab:metrics"):
        key = next((k for k, _ in _METRIC_ROW if cs[0].strip().startswith(k.split("_")[0])
                    and (k != "h1_vs_no_graph" or cs[0].strip().startswith("h1"))), None)
        if cs[0].strip().startswith("untyped"):
            key = "promotion_margin"
        if key is None or len(cs) < 6:
            continue
        n += 1
        for col, metric in zip(cs[1:6], _METRIC_COL):
            c = d["contrasts"][metric][key]
            sur = d["across_metric"][f"{metric}/{key}"]["survives_family_wise"]
            cmp_cell(fails, f"tab:metrics {key}/{metric}", col, c["mean"], star=bool(sur))
    return fails, (f"full: {n} contrasts x 5 endpoints x (value, star) = {n * 5} cells vs "
                   f"data/results/a3_external/rescored.json")


@table("tab:ksweep")
def _ksweep(tex):
    fails, ks = [], load("data/results/a3_external/k_sweep.json")
    n = 0
    for cs in cells_of(tex, "tab:ksweep"):
        kr = cs[0].replace("$", "").strip()
        if not (kr == "all" or kr.isdigit()) or len(cs) != 5:
            continue
        row = ks.get(str(10282 if kr == "all" else int(kr)))
        if row is None:
            continue
        n += 1
        for cell, key in zip(cs[1:], ["h2a", "h2b", "promotion_margin", "h1_vs_no_graph"]):
            cmp_cell(fails, f"tab:ksweep k={kr} {key}", cell, row[key]["mean"])
    return fails, f"full: {n} k values x 4 contrasts vs data/results/a3_external/k_sweep.json"


@table("tab:bins")
def _bins(tex):
    fails = []
    head = load("data/results/b2_deciles/deciles.json")["schemes"]["head"]
    n = 0
    for cs in cells_of(tex, "tab:bins"):
        if len(cs) != 4:
            continue
        # The label is written "1--20" or "1 to 20" depending on the paper's dash convention, and
        # BOTH must parse. A prose pass that regularised en dashes to "to" on 2026-08-26 broke this
        # silently: every row stopped matching, the check reported "full: 0 intervals", and the gate
        # still passed because zero rows yield zero failures.
        lab = (cs[0].replace("$", "").replace("{,}", "")
               .replace("--", "-").replace(" to ", "-").strip())
        if not re.match(r"\d+-\d+$", lab):
            continue
        n += 1
        for cell, key in ((cs[1], "promotion_margin"), (cs[3], "h2a")):
            c = head["cells"].get(f"{lab}/{key}")
            if c:
                cmp_cell(fails, f"tab:bins {lab} {key}", cell, c["mean"])
    return fails, f"full: {n} intervals x 2 contrasts vs data/results/b2_deciles/deciles.json"


# =================================================================================================
def artifact_checks() -> tuple[list[str], list[tuple[str, str]]]:
    """Run every registered table. Returns (failures, coverage rows)."""
    tex = (HERE / "main.tex").read_text()
    fails: list[str] = []
    coverage: list[tuple[str, str]] = []
    for label, fn in CHECKS:
        try:
            f, note = fn(tex)
        except Missing as e:
            coverage.append((label, f"SKIPPED: artifact absent — {e}"))
            fails.append(f"{label}: SKIPPED, its artifact is absent ({e}). A skipped table is not "
                         f"a checked table.")
            continue
        except Exception as e:                      # a parser that breaks must not read as a pass
            coverage.append((label, f"ERROR: {type(e).__name__}: {e}"))
            fails.append(f"{label}: check itself failed with {type(e).__name__}: {e}")
            continue
        # A CHECK THAT MATCHED NO ROWS IS BROKEN, NOT CLEAN. Every row-iterating check reports its
        # count in its own note, and a count of zero means the parser stopped recognising the table
        # rather than that the table agrees with its artifact. Without this, breaking a parser looks
        # exactly like passing: zero rows compared yields zero failures. That happened on 2026-08-26.
        if re.match(r"(full|partial)[^0-9]*\b0\b", note):
            fails.append(f"{label}: the check matched 0 rows, so it verified NOTHING. Its note reads "
                         f"{note!r}. Fix the parser; a zero-row check is broken, not passing.")
        fails.extend(f)
        coverage.append((label, note))

    # --- prose-level singletons the paper leans on, outside any table ----------------------
    try:
        d3 = load("data/results/screening_b1/b1_message_form.json")["contrasts"]["D3"]
        for lit, val in ((f"{d3['mean']:+.4f}", "B1a D3 mean"),
                         (f"{d3['ci_low']:+.4f}", "B1a D3 ci_low"),
                         (f"{d3['ci_high']:+.4f}", "B1a D3 ci_high")):
            if f"${lit}$" not in tex and lit not in tex:
                fails.append(f"{val} ({lit}) absent from the paper")
        coverage.append(("prose: B1a D3", "full: mean and CI vs screening_b1/b1_message_form.json"))
    except Missing as e:
        coverage.append(("prose: B1a D3", f"SKIPPED: artifact absent — {e}"))
        fails.append(f"prose B1a D3: SKIPPED, artifact absent ({e})")
    try:
        pm = load("data/results/replication/pooled_with_reference.json")["pooled"]["promotion_margin"]
        i2 = f"{pm['I2'] * 100:.1f}"
        if f"I^2{{=}}{i2}" not in tex:
            fails.append(f"eight-dataset I^2 should read {i2}% and does not appear")
        coverage.append(("prose: pooled I^2", "full: vs replication/pooled_with_reference.json"))
    except Missing as e:
        coverage.append(("prose: pooled I^2", f"SKIPPED: artifact absent — {e}"))
        fails.append(f"prose pooled I^2: SKIPPED, artifact absent ({e})")
    return fails, coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--coverage", action="store_true",
                    help="print which tables are checked against what, and how completely")
    a = ap.parse_args()
    tex = (HERE / "main.tex").read_text()
    cur = literals(tex)

    if a.snapshot:
        SNAP.write_text(json.dumps(dict(sorted(cur.items())), indent=1))
        print(f"snapshot: {len(cur)} distinct literals, {sum(cur.values())} occurrences -> {SNAP.name}")
        return 0

    fails, coverage = artifact_checks()
    if a.coverage:
        tables = [c for c in coverage if c[0].startswith("tab:")]
        full = sum(1 for _, n in tables if n.startswith("full"))
        print(f"ARTIFACT AGREEMENT COVERAGE — {len(tables)} tables, {full} fully re-derived\n")
        for label, note in coverage:
            print(f"  {label:16s} {note}")
        print()
    print(f"artifact agreement: {len(fails)} failure(s)")
    for f in fails:
        print("  ", f)

    if a.check and SNAP.exists():
        old = Counter(json.loads(SNAP.read_text()))
        gone = {k: v for k, v in (old - cur).items()}
        new = {k: v for k, v in (cur - old).items()}
        print(f"inventory drift: {len(gone)} dropped, {len(new)} introduced")
        if gone:
            print("   DROPPED (was in the verified paper, now absent):")
            for k, v in sorted(gone.items()):
                print(f"     {k} x{v}")
        if new:
            print("   INTRODUCED (not in the verified paper - justify each against an artifact):")
            for k, v in sorted(new.items()):
                print(f"     {k} x{v}")
        return 1 if (fails or gone or new) else 0
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
