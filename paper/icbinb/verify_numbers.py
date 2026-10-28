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
Run --coverage to see the state of every check at once.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SNAP = HERE / "numbers.snapshot.json"
GATE_DEAD = 1e-3   # config.GATE_DEAD; duplicated so this script stays importable without src/

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
# The checks, one per table plus the prose checks
# =================================================================================================
_FAMILY_ROOT = "data/results/screening_lambda0"          # the REPAIRED gate root, not screening/
_FAMILY_KEY = {"ts": "typed_static", "cg": "condition_gated", "ug": "untyped_gnn",
               "eo": "expression_only"}
_FAMILY_WORD = {"typed": "typed_static", "gated": "condition_gated", "untyped": "untyped_gnn",
                "no graph": "expression_only"}
_CONTRAST_OF = {("typed_static", "expression_only"): "h2a",
                ("condition_gated", "typed_static"): "h2b",
                ("untyped_gnn", "expression_only"): "promotion_margin",
                ("condition_gated", "expression_only"): "h1_vs_no_graph"}


# Region -> the artifacts that region's numbers are drawn from. Scoping matters and is not cosmetic:
# checking a value against ALL 514 result artifacts is vacuous, because a random four-decimal effect
# size lands in that corpus 65% of the time and 91% of the paper's real values still "pass" after
# being perturbed by one digit. Scoped to the handful of artifacts a section actually uses, the same
# test admits a random value 3-8% of the time. Measured, not assumed, on 2026-08-26.
# Rebuilt 2026-10-28 for the shortened plain-language paper: one entry per section it still has.
_R = "data/results/"
PROSE_REGIONS = {
    # Related Work quotes no number. Mapped to nothing, so a number added to it later fails rather
    # than going unchecked.
    "sec:related": [],
    "sec:setup": [_R + "screening/summary.json", _R + "screening_lambda0/robustness_5seed.json",
                  _R + "camera_ready/facts.json", "data/splits/manifest.json"],
    "sec:approach": [_R + "screening_lambda0/robustness_5seed.json"],
    # sec:outcome carries no numeric literal of its own; mapped anyway so that no section is silently
    # outside the check, which is how 260 literals went unexamined until 2026-08-26.
    "sec:outcome": [_R + "screening_lambda0/robustness_5seed.json"],
    # screening_lambda0 is here because the repair sentence quotes the live headline beside the
    # pre-repair one it replaced.
    "sec:confound": [_R + "screening/robustness_5seed.json", _R + "q4_lambda_sweep_22ep.json",
                     _R + "camera_ready/facts.json", _R + "screening_lambda0/robustness_5seed.json"],
    "sec:null": [_R + "screening_lambda0/robustness_5seed.json",
                 _R + "screening_c080c10_h1/robustness_5seed.json",
                 _R + "screening_c075c15_n5/robustness_5seed.json",
                 _R + "screening_c080c10_r2/robustness_5seed.json",
                 _R + "screening_c080c10_r3/robustness_5seed.json",
                 _R + "screening_n7_live/robustness_5seed.json"],
    "sec:repl": [_R + "replication/pooled.json", _R + "replication/pooled_k128_subset.json",
                 _R + "camera_ready/pooled_k128_subset_hksj.json",
                 _R + "camera_ready/pooled_with_reference_hksj.json",
                 _R + "screening_n7_live/robustness_5seed.json",
                 _R + "replication/pooled_with_reference.json"],
    "sec:causes": [_R + "screening_n7_live/robustness_5seed.json", _R + "feature_ablation_report.json",
                   _R + "screening_lambda0/robustness_5seed.json",
                   _R + "screening_b1/b1_message_form.json"],
    "sec:limits": [_R + "a2_ladder/floor.json", _R + "c1_ladder/floor_condition_gated.json"],
    # The checklist and everything after it up to the appendix (Conclusion, statements). Mapped to
    # nothing: its one number, 103, is declared below.
    "sec:checklist": [],
    "app:arch": [],
    "app:mechanism": [_R + "screening_a1/a1_mechanism.json", _R + "screening_b1/b1_message_form.json",
                      _R + "a2_power/arch_search_bound.json", _R + "camera_ready/facts.json",
                      _R + "screening_n7_live/robustness_5seed.json"],
    "app:repl": [_R + "replication/pooled.json", _R + "replication/pooled_with_reference.json",
                 _R + "replication/pooled_k128_subset.json",
                 _R + "camera_ready/pooled_k128_subset_hksj.json",
                 _R + "camera_ready/pooled_with_reference_hksj.json",
                 _R + "camera_ready/pooled_hksj.json",
                 _R + "screening_lambda0/robustness_5seed.json",
                 _R + "screening_n7_live/robustness_5seed.json"] +
                sorted(str(q.relative_to(ROOT)) for q in
                       (ROOT / "data/results/replication").glob("*/robustness_5seed.json")
                       if not q.parent.name.startswith("_")),
    "app:folds": [_R + "camera_ready/facts.json",
                  _R + "screening_lambda0/robustness_5seed.json",
                  _R + "screening_c080c10_h1/robustness_5seed.json",
                  _R + "screening_c075c15_n5/robustness_5seed.json",
                  _R + "screening_c080c10_r2/robustness_5seed.json",
                  _R + "screening_c080c10_r3/robustness_5seed.json",
                  "data/splits/manifest.json", _R + "splits_c075c15/manifest.json",
                  _R + "splits_c080c10/manifest.json"],
    # Causes A, B, D and E in detail, which since 2026-10-28 also hold what the separate feature-ablation
    # and other-scores appendices used to say.
    "app:causes": [_R + "feature_ablation_report.json", _R + "rationale_audit_lambda0/audit_report.json",
                   _R + "screening_lambda0/second_metric_5seed.json",
                   _R + "screening_lambda0/robustness_5seed.json",
                   _R + "a2_power/rationale_bound.json", _R + "camera_ready/facts.json",
                   _R + "a3_external_live/rescored.json", _R + "b2_deciles_live/deciles.json"],
    "app:floor": [_R + "a2_ladder/floor.json", _R + "c1_ladder/floor_condition_gated.json"],
    # The corrections list quotes both sides of each correction: the live artifacts and the dead-gate
    # ones they replaced.
    "app:repro": [_R + "camera_ready/facts.json",
                  _R + "replication/FrangiehIzar2021_RNA/robustness_5seed.json",
                  _R + "screening_lambda0/robustness_5seed.json",
                  _R + "screening_n7_live/robustness_5seed.json",
                  _R + "screening_untyped_n7/robustness_5seed.json"],
    # The glossary states no result, so it maps to nothing and any number added to it fails.
    "app:glossary": [],
}

# Literals that are NOT results and therefore cannot be re-derived from a results artifact. Every one
# carries the reason it is exempt. This list is the point of the check: a number nobody has accounted
# for FAILS, so a new unexplained figure cannot enter the paper unnoticed. Pruned on 2026-10-28 to the
# literals the shortened paper still prints: an unused exemption is a hole, because it would pass the
# same literal anywhere.
PROSE_DECLARED = {
    # --- constants of the method, not results -------------------------------------------------
    "95": "the confidence level in '95% CI'. A constant of the method.",
    # --- model architecture and run settings ---------------------------------------------------
    "1280": "ESM-2 650M embedding width; a property of the pretrained encoder.",
    "128": "PINNACLE embedding width, the program-basis K, and the validation rows of Figure 1's sweep "
           "(lambda_sweep.py --n-val). Model and run settings.",
    "256": "hidden width of the intervention and graph vectors; a model constant.",
    "512": "the neighbourhood sampling cap, NEIGHBORHOOD_CAP, and the training rows of Figure 1's sweep "
           "(lambda_sweep.py --n-train, recorded in data/logs/campaign/q4b_card*.log); run settings.",
    "22": "epochs in the lambda sweep of Figure 1 (q4_lambda_sweep_22ep.json); a run setting.",
    # --- DATASET SIZES. Properties of the inputs, recorded in build provenance not in results. ---
    "33{,}983": "DE rows in the reference screen.",
    "10{,}282": "genes scored per row in the reference screen.",
    "11{,}526": "distinct gene targets in the reference screen.",
    "1{,}119": "proteins in PINNACLE's CD4 context; a property of that resource.",
    # --- graph properties ------------------------------------------------------------------------
    "85": "share of PPI edges that are functional_assoc; measured at 85.41% in rationale_bound.json.",
    # --- BUILD-TIME FEATURE COVERAGE. Computed while assembling the feature stores, never persisted.
    "96.5": "share with nonzero physical PPI degree; feature-store coverage.",
    "98.4": "share with nonzero functional PPI degree; feature-store coverage.",
    "90.8": "share receiving a ZERO PINNACLE vector; the complement of its 9.2% coverage.",
    # --- ratios derived in the sentence that states them; their components ARE gated -----------
    "103": "ratio of the graph term to the response term, derived in the same sentence.",
    "79": "D3 divided by the 0.0176 gap. Amendment 7.4 fixes this as DESCRIPTIVE only.",
    # --- the per-edge-normalized pilot wrote no JSON; its table is in a tracked doc --------------
    "98": "dead-gate share after TWO epochs with the penalty normalized per edge (98.2%, 'by epoch 1' "
          "0-indexed); the 2026-07-21 pilot table in docs/h1-optimization-notes.md. The pilot JSON was not "
          "persisted. The paper said 'after one epoch' until 2026-10-28; an ML review caught it.",
    "71": "dead-gate share after ONE epoch in the same pilot (71.2%, 'by epoch 0'); same table, same reason.",
    # --- post hoc paired tests, each a difference of two artifacts' per-seed deltas ----------------
    "0.098": "Bonferroni of the degree-scaled variant's p over all four registered message variants "
             "(4 x 0.0245, Amendment 7.3); recomputed by prose:derived.",
    "+0.0101": "upper 95% bound of the features-minus-gated paired difference (Section 5.1); recomputed "
               "by prose:derived.",
    "-0.00427": "95% CI bound of zeroing all graph-derived features; prose:derived recomputes it from the "
                "variant's mean and sd (feature_ablation_report.json stores no interval).",
    "-0.00286": "the other bound of that interval; recomputed the same way.",
    "-0.00403": "95% CI bound of zeroing the counts alone; recomputed the same way.",
    "-0.00290": "the other bound of that interval; recomputed the same way. The accepted paper printed it "
                "as -0.00291; the per-seed values give -0.0029046, and this check caught it on 2026-10-28.",
    "-0.00291": "the superseded value of that bound, quoted only in the corrections list beside -0.00290.",
    "+0.0255": "the superseded Table 3 value for Frangieh's typed result, quoted only in the corrections list "
               "beside +0.0256: pooled_with_reference.json stores the mean rounded to five places (0.02555), "
               "which formats as +0.0255; the unrounded per-dataset mean 0.025553 is +0.0256. tab:repl now "
               "reads the per-dataset files. Caught by an adversarial review on 2026-10-28.",
    # --- the replication DE build under the ORIGINAL replicate rule; docs/replication-dataset-survey.md 7bis
    "12": "Replogle RPE1 targets kept under the original replicate rule (App. C); the DE build table in "
          "docs/replication-dataset-survey.md section 7bis, a build record rather than a result. Also the "
          "upper end of the per-run GPU-hours in App. G, which prose:camera anchors.",
    "2{,}393": "Replogle RPE1 targets in the harmonized file, the denominator of that same table.",
    "-0.0012": "lower 95% bound of the features-minus-gated paired difference (Section 5.1); recomputed "
               "by prose:derived. Elsewhere it is the harder split's typed loss, which those regions' "
               "artifacts cover.",
    # --- the degree-scaled model against no graph (App. B). Not stored as a contrast, but derivable: ---
    # --- prose:derived pairs b1_message_form.json's D3 and typed-vs-no-graph deltas by seed. -------
    "-0.0047": "lower 95% bound of gcnnorm minus expression_only; recomputed by prose:derived.",
    "+0.0062": "upper bound of that interval; recomputed by prose:derived. Also Pearson-delta's gated minus "
               "typed in App. E, which that region's artifact covers.",
    # --- rationale-audit quantities normalised per 1% of edges removed, a derivation ------------
    "0.0043": "prediction change per 1% of STRING edges removed; derived from the raw source-ablation delta.",
    "0.0025": "the same for HuRI; derived as above.",
    "0.0014": "the same for BioPlex; derived as above.",
    # --- quantities with no persisted artifact --------------------------------------------------
    "0.001": "the centroid-accuracy floor, written as an approximation, not a measured value.",
    # --- COMPUTE BUDGET, read off the run registry rather than a results artifact ----------------
    "379": "training lanes behind the paper; a count of runs.",
    # --- design choices ------------------------------------------------------------------------
    "24": "gated lanes in the C1 campaign; a design count, six rungs times four seeds.",
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
                      f"{x * 100:.1f}", f"{x * 100:.0f}", f"{x:.0f}", f"{x:.5f}", f"{x:.6f}",
                      f"{x:+.6f}", f"{x:+.5f}"):
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
        # An unmapped section used to be skipped silently, so a new appendix's numbers went unchecked
        # while the summary still read "nothing unaccounted". Unmapped is a failure, not a pass.
        if lab not in PROSE_REGIONS:
            fails.append(f"prose:all section {lab} has no PROSE_REGIONS entry, so its numbers are unchecked")
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


@table("prose:assertions")
def _prose_assertions(tex):
    """Claims the paper makes in WORDS rather than numbers, which are still checkable.

    "none dropped", "all four of its seeds are negative", "every gate live throughout training" are
    assertions about the data, not rhetoric, and until 2026-08-26 nothing checked any of them. A number
    that goes stale gets caught by the checks above; a sentence that goes stale did not.

    Each entry states the claim as the paper makes it and the artifact condition that must hold."""
    flat = " ".join(re.sub(r"(?m)^%.*", "", tex).split())
    fails = []

    def need(condition, claim, detail):
        if not condition:
            fails.append(f"assertion '{claim}': {detail}")

    def said(phrase):
        """Each check runs only while the paper makes its claim. A reworded sentence used to switch its
        check off silently; now the missing phrase fails, naming the trigger to update."""
        if phrase in flat:
            return True
        fails.append(f"assertion trigger '{phrase}' is no longer in the paper; update the trigger "
                     f"to the new wording, or delete the check if the claim was dropped")
        return False

    def signs(c):
        """Per-seed differences as the paper states them, first model minus second."""
        return [d for d in c["deltas"]]

    n7 = load("data/results/screening_n7_live/robustness_5seed.json")["contrasts"]
    pm = n7["promotion_margin"]
    if said("At seven seeds, none dropped"):
        need(not pm["dropped"] and len(pm["seeds_used"]) == 7, "seeds 0 to 6, none dropped",
             f"seeds_used={pm['seeds_used']} dropped={pm['dropped']}")
    if said("all seven seeds positive"):
        need(len(signs(pm)) == 7 and min(signs(pm)) > 0, "untyped at n=7: all seven seeds positive",
             f"per-seed {[round(d, 4) for d in signs(pm)]}")
    if said("worse in all seven seeds"):
        need(len(signs(n7["h2a"])) == 7 and max(signs(n7["h2a"])) < 0,
             "typed at n=7: worse in all seven seeds", f"per-seed {[round(d, 4) for d in signs(n7['h2a'])]}")

    rp = load("data/results/replication/ReplogleWeissman2022_rpe1/robustness_5seed.json")["contrasts"]
    if said("all four seeds positive"):
        d = signs(rp["promotion_margin"])
        need(len(d) == 4 and min(d) > 0, "RPE1 untyped: all four seeds positive", f"per-seed {d}")

    nm = load("data/results/replication/NormanWeissman2019_filtered/robustness_5seed.json")
    nmc = nm["contrasts"]["promotion_margin"]
    if said("all four of its seeds are negative and none was dropped"):
        neg = sum(1 for d in nmc["deltas"] if d < 0)
        need(neg == len(nmc["deltas"]) and not nmc["dropped"],
             "Norman: four of four seeds negative, none dropped",
             f"{neg}/{len(nmc['deltas'])} negative, dropped={nmc['dropped']}")

    b1 = load("data/results/screening_b1/b1_message_form.json")["contrasts"]["D3"]
    if said("five of five seeds) and so recovers"):
        need(len(b1["deltas"]) == 5 and min(b1["deltas"]) > 0, "degree-scaled variant: five of five seeds",
             f"per-seed {b1['deltas']}")

    # Frangieh is the one gated replication, and the paper says its gates were live throughout.
    if said("every gate live throughout training"):
        mins = []
        for s in range(8):
            h = ROOT / f"data/results/replication/FrangiehIzar2021_RNA/condition_gated/{s}/logs/stage_a_history.json"
            if h.exists():
                g = [e["train"]["gate_mean"] for e in json.loads(h.read_text())
                     if e.get("train", {}).get("gate_mean") is not None]
                if g:
                    mins.append(min(g))
        need(bool(mins) and min(mins) > GATE_DEAD,
             "Frangieh: every gate demonstrably live throughout training",
             f"per-seed minima {[round(x, 4) for x in mins]} against the {GATE_DEAD} threshold")

    # Appendix C: ten candidates built, eight pass the on-target check, the two failures are the
    # T-cell screens, and the seven trained ones are the passes other than Papalexi.
    verdicts = {f.name.split(".")[0]: json.loads(f.read_text())["verdict"]
                for f in sorted((ROOT / "data/intermediate/replication").glob("*.DE_stats_v2.qc.json"))}
    failed = sorted(k for k, v in verdicts.items() if not v.startswith("PASS"))
    if said("We built ten public datasets in our format. Eight passed"):
        need(len(verdicts) == 10 and len(failed) == 2, "ten built, eight pass",
             f"{len(verdicts)} built, {len(verdicts) - len(failed)} pass")
    if said("The two that failed, Datlinger and Shifrut"):
        need([k[:8] for k in failed] == ["Datlinge", "ShifrutM"], "the two failures are Datlinger and Shifrut",
             f"failed {failed}")
    if said("the other seven were trained to completion"):
        trained = sorted(p.parent.name for p in (ROOT / "data/results/replication").glob("*/robustness_5seed.json")
                         if not p.parent.name.startswith("_"))   # "_snapshot_*" holds old copies
        passed = sorted(k for k in verdicts if k not in failed and not k.startswith("Papalexi"))
        need(trained == passed and len(trained) == 7, "the seven trained datasets are the passes but Papalexi",
             f"trained {trained} against passes {passed}")

    return fails, ("full: 11 worded assertions checked against artifacts (seeds dropped and their signs, "
                   "Norman's seeds, Frangieh's live gates, which candidates pass the on-target check "
                   "and which were trained)")


@table("prose:style")
def _prose_style(tex):
    """The style rules this project treats as mandatory but had never gated.

    The post-edit checklist has required "0 em/en dashes, 0 banned words" since the paper began, and
    every pass has enforced it by hand. Nothing checked it. A 2026-08-26 prose pass also found the
    paper mixing British and American spelling in 25 places while its own title says "Regularizer"
    and its abstract "unnormalized"; "center" appeared 17 times and "centre" never, so the convention
    was already decided and simply not followed. Hand-enforcement is how that survives.

    Comments are stripped first: an AUTO note is not prose and does not render."""
    prose = re.sub(r"(?m)^%.*$", "", tex)
    fails = []

    for ch, name in (("\u2014", "em dash"), ("\u2013", "en dash")):
        n = prose.count(ch)
        if n:
            fails.append(f"prose:style {n} {name}(s) in main.tex. The house rule is zero; use a "
                         f"period, comma, colon or parentheses.")

    # Words with no defensible use here. "robust"/"harness" are deliberately absent: this paper uses
    # "robustness report" as an artifact name and "experiment harness" as a noun, both legitimate.
    BANNED = ["delve", "tapestry", "realm", "beacon", "multifacet", "meticulous", "paradigm shift",
              "game.chang", "cutting-edge", "ever-evolving", "supercharge", "transformative",
              "utiliz", "facilitat", "empower", "streamlin", "a testament", "pivotal moment"]
    for w in BANNED:
        if re.search(w, prose, re.I):
            fails.append(f"prose:style banned word matching /{w}/ appears in the paper")

    # The convention is American, set by the title and abstract and by center(17)/centre(0).
    BRITISH = ["neighbour", "normalis", "localis", "labelled", "modelled", "behaviour", "centre",
               "summaris", "analyse", "recognis", "favour", "defence", "generalis", "relabell"]
    for w in BRITISH:
        hits = re.findall(w, prose, re.I)
        if hits:
            fails.append(f"prose:style British spelling /{w}/ x{len(hits)}; this paper's title and "
                         f"abstract set the American convention")

    # A math subscript label is not prose: a 2026-08-26 hyphenation pass rewrote \text{redraw} to
    # \text{re-draw} inside an equation and broke two anchors. Keep the label unhyphenated.
    if re.search(r"\\text\{re-draw\}", prose):
        fails.append("prose:style the variance-component subscript must stay \\text{redraw}; a "
                     "hyphenation pass reached inside the math label once already")

    return fails, ("full: em/en dashes, 18 banned words, 10 British spellings and the redraw "
                   "subscript, over main.tex with comments stripped")


@table("refs:attribution")
def _refs_attribution(tex):
    """Named methods must carry a citation, and every cited key must exist.

    Added 2026-08-26 after a references pass found two methods used by name with no citation while
    their bib entries sat in the file unused: GATv2 in the architecture-search table and GInX in the
    rationale audit. LaTeX warns about a \\cite with no entry and never about an entry with no \\cite,
    and never about a method named in prose that was never cited at all, so nothing caught either.
    The same pass found the paper's own gate mechanism positioned only against graph attention when
    its direct precedent, differentiable edge masking, was in the bib and uncited.

    Each row is a string that appears in the paper only when the named method is being used, and the
    bib key that must then be cited somewhere."""
    flat = " ".join(re.sub(r"(?m)^%.*", "", tex).split())
    fails = []
    cited = set()
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex):
        cited.update(k.strip() for k in m.group(1).split(","))

    NAMED = [
        ("GATv2",         "brody2022gatv2"),
        ("GInX",          "amara2023ginx"),
        ("GNNExplainer",  "ying2019gnnexplainer"),
        ("edge masking",  "schlichtkrull2021graphmask"),
        ("TabPFN",        "hollmann2025tabpfn"),
        ("\\textsc{gears}", "roohani2024gears"),
        ("scPerturb",     "peidli2024scperturb"),
        ("PINNACLE",      "li2024pinnacle"),
        ("ESM-2",         "lin2023esm2"),
        ("STRING",        "szklarczyk2023string"),
        ("BioGRID",       "oughtred2021biogrid"),
        ("CORUM",         "tsitsiridis2023corum"),
        ("BioPlex",       "huttlin2021bioplex"),
        ("HuRI",          "luck2020huri"),
        ("TxPert",        "wenkel2026txpert"),
    ]
    for name, key in NAMED:
        if name.lower() in flat.lower() and key not in cited:
            fails.append(f"refs: '{name}' is used by name in the paper and {key} is never cited. "
                         f"A method named in prose needs an attribution; LaTeX will not tell you.")

    # every cited key must resolve against the two bib files
    defined = set()
    for rel in ("paper/references.bib", "paper/icbinb/extra.bib"):
        f = ROOT / rel
        if f.exists():
            defined.update(m.group(1).strip()
                           for m in re.finditer(r"@\w+\{([^,]+),", f.read_text(encoding="utf-8")))
    for k in sorted(cited - defined):
        fails.append(f"refs: \\cite{{{k}}} has no entry in references.bib or extra.bib")

    return fails, (f"full: {len(NAMED)} named methods checked for attribution, "
                   f"{len(cited)} cited keys resolved against {len(defined)} bib entries")


@table("prose:derived")
def _prose_derived(tex):
    """Quantities the paper DERIVES from two artifacts rather than reading off one, and the words that
    must go with them.

    This is the blind spot prose:all cannot cover. That check passes a literal when the string occurs
    anywhere in the section's artifact values at any of thirteen precisions, so a one or two digit
    number passes for free and a ratio between two artifacts passes whenever either operand happens to
    round to it. Every derivation below is recomputed from the artifacts and matched against the words
    around it, so a reworded sentence fails loudly instead of quietly going unchecked. Rebuilt on
    2026-10-28 for the shortened paper: the derivations behind the sections it removed (the ladder's
    noise ratios, the variance decomposition, the energy-distance cells) went with those sections."""
    flat = " ".join(re.sub(r"(?m)^%.*", "", tex).split())
    fails = []

    def need(claim, value, context):
        """context carries {V}; we accept the paper writing it at 0 to 4 decimals."""
        for txt in (f"{value:.4f}", f"{value:.3f}", f"{value:.2f}", f"{value:.1f}", f"{value:.0f}"):
            if context.replace("{V}", txt) in flat:
                return
        fails.append(f"derived {claim}: recomputed {value:.4f}; the paper does not state it in "
                     f"context (looked for '{context.replace('{V}', f'{value:.4f}')}')")

    # ---- Figure 1: what the caption says about the sweep --------------------------------------------
    runs = load("data/results/q4_lambda_sweep_22ep.json")["runs"]
    z = [r for r in runs if r["lambda_graph"] == 0][0]["trajectory"][-1]
    need("gate at lambda=0 at the end of the sweep", z["val_gate"], r"Without the penalty the gates end at ${V}$")
    if len(runs) != len({r["lambda_graph"] for r in runs}) or "one seed per setting" not in flat.lower():
        fails.append("derived: the lambda sweep now has more than one run per point, or the caption no "
                     "longer says so; Figure 1's caption must match")
    pen = sorted((r for r in runs if r["lambda_graph"] > 0), key=lambda r: r["lambda_graph"])
    count = {8: "eight", 9: "nine", 7: "seven"}
    if f"Across the {count.get(len(pen), len(pen))} penalty strengths we tried" not in flat:
        fails.append(f"derived Figure 1: the sweep has {len(pen)} penalized weights; the caption's count differs")
    ordinal = {1: "second", 2: "third", 3: "fourth", 4: "fifth", 11: "twelfth"}

    def first_dead(r):
        return next((e["epoch"] for e in r["trajectory"] if e["val_gate"] < GATE_DEAD), None)
    fd = {r["lambda_graph"]: first_dead(r) for r in pen}
    hi = {v for k, v in fd.items() if k >= 1e-3}
    if len(hi) != 1 or f"those from $10^{{-3}}$ up are dead from the {ordinal.get(min(hi))} epoch" not in flat:
        fails.append(f"derived Figure 1 crossing: weights >= 1e-3 first dead at 0-indexed epochs "
                     f"{sorted(hi)}; the caption does not say so")
    if pen[0]["lambda_graph"] != 1e-5 or \
            f"the weakest, $10^{{-5}}$, from the {ordinal.get(fd.get(1e-5))}" not in flat:
        fails.append(f"derived Figure 1 crossing: the weakest weight is {pen[0]['lambda_graph']:g}, first dead "
                     f"at 0-indexed epoch {fd.get(pen[0]['lambda_graph'])}; the caption does not say so")
    recovers = [k for k, r in ((r["lambda_graph"], r) for r in pen) if fd[k] is None or any(
        e["val_gate"] >= GATE_DEAD for e in r["trajectory"] if e["epoch"] >= fd[k])]
    if recovers or "none recovers" not in flat:
        fails.append(f"derived Figure 1: the caption says no penalized run recovers; weights {recovers} "
                     f"are live at some epoch after their first dead one")

    # ---- Section 4.1: the main comparison before the repair ----------------------------------------
    pre = load("data/results/screening/robustness_5seed.json")["contrasts"]["h1_vs_no_graph"]["mean"]
    need("pre-repair main comparison", abs(pre), r"barely moved, from $-{V}$ to $-0.0009$")

    # ---- Appendix B: the search's budget, and the degree-scaled variant --------------------------------
    asb = load("data/results/a2_power/arch_search_bound.json")
    if asb["epochs_run"] != [5] or "one seed and five epochs each" not in flat:
        fails.append(f"derived arch-search epochs: artifact says {asb['epochs_run']}, App.~B says five")
    gap = load("data/results/screening_b1/b1_message_form.json")["gap_untyped_minus_typed"]["mean"]
    if not (0.4 < asb["observed_spread"] / gap < 0.6) or "about half the" not in flat:
        fails.append(f"derived: App.~B calls the search's range about half the gap; it is "
                     f"{asb['observed_spread'] / gap:.2f} of it")
    b1 = load("data/results/screening_b1/b1_message_form.json")
    d3 = b1["contrasts"]["D3"]
    need("B1a uncorrected p", d3["p_value"], r"uncorrected $p{=}{V}$")
    need("B1a recovery share", d3["recovery_share"] * 100,
         r"recovers ${V}\%$ of the gap between the typed and untyped models")
    # "still does not beat no graph" needs the degree-scaled model's interval against no graph to reach zero.
    def paired(diffs):
        from scipy import stats
        n = len(diffs)
        m = sum(diffs) / n
        sd = (sum((d - m) ** 2 for d in diffs) / (n - 1)) ** 0.5
        h = stats.t.ppf(0.975, n - 1) * sd / n ** 0.5
        return m, m - h, m + h, 2 * stats.t.sf(abs(m / (sd / n ** 0.5)), n - 1)
    d3s = dict(zip(d3["seeds_used"], d3["deltas"]))                                       # gcnnorm - typed
    tn = dict(zip(b1["context_typed_vs_nograph"]["seeds_used"], b1["context_typed_vs_nograph"]["deltas"]))
    common = sorted(set(d3s) & set(tn))
    m, lo, hi_, pv = paired([d3s[s] + tn[s] for s in common])
    # Section 4.2's heading also says "still does not beat no graph", about the gated model; this anchor
    # must name the repaired model, or the heading would satisfy it after the sentence was deleted.
    if "The repaired model still does not beat no graph" not in flat or lo > 0:
        fails.append(f"derived degree-scaled minus no graph: {m:+.4f} [{lo:+.4f},{hi_:+.4f}]; the paper's "
                     f"'still does not beat no graph' is missing or false")
    if f"still no better than no graph (${m:+.4f}$, $95\\%$ CI $[{lo:+.4f},{hi_:+.4f}]$)" not in flat:
        fails.append(f"derived degree-scaled minus no graph: the paper does not state {m:+.4f} "
                     f"[{lo:+.4f},{hi_:+.4f}] as App.~B's paired difference")

    # ---- Section 4.3, Table 4, Conclusion: which datasets pass correction, and in which direction ------
    pd = load("data/results/replication/pooled_with_reference.json")["per_dataset"]
    ref = "reference_screen_n7"
    rep = [k for k in pd["h2a"] if k != ref]
    typed_rep = [k for k in rep if pd["h2a"][k]["survives_family_wise"]]
    unt = [v["mean"] for v in pd["promotion_margin"].values() if v["survives_family_wise"]]
    if typed_rep or "the typed graph shows no benefit on seven more datasets" not in flat:
        fails.append(f"derived Conclusion: typed survivors on the replication screens {typed_rep}")
    if not (len(unt) == 3 and min(unt) < 0 < max(unt)) or \
            "passes correction on three of the eight datasets, and they disagree in sign" not in flat:
        fails.append(f"derived Section 4.3: untyped survivors over all eight {unt}, expected three in both directions")
    up, down = sum(1 for x in unt if x > 0), sum(1 for x in unt if x < 0)
    for phrase in ("for the better on two datasets and for the worse on one",
                   "helps on two datasets and hurts on one"):
        if (up, down) != (2, 1) or phrase not in flat:
            fails.append(f"derived: '{phrase}' against {up} positive and {down} negative survivors")
    seeds = {k: pd["promotion_margin"][k]["n_seeds"] for k in pd["promotion_margin"]}
    if seeds.pop(ref) != 7 or set(seeds.values()) != {4} or \
            "The T-cell screen has seven seeds per model, the others four" not in flat:
        fails.append(f"derived Table 3 caption: seeds per dataset {seeds}")

    # ---- Section 5.1 D: graph-derived features against the gated model, paired by seed --------------
    def oriented(c, arm):
        sg = 1.0 if c["better"] == arm else -1.0
        return {int(s): sg * d for s, d in zip(c["seeds_used"], c["deltas"])}
    h1s = oriented(load("data/results/screening_lambda0/robustness_5seed.json")["contrasts"]["h1_vs_no_graph"],
                   "condition_gated")
    abl = next(v for v in load("data/results/feature_ablation_report.json") if v["variant"] == "nograph")
    feat = {int(s): -d for s, d in abl["per_seed"].items()}
    common = sorted(set(h1s) & set(feat))
    fdiff = [feat[s] - h1s[s] for s in common]
    m, lo, hi_, pv = paired(fdiff)
    want = (f"the features' effect minus the gated model's is ${m:+.4f}$ ($95\\%$ CI $[{lo:+.4f},{hi_:+.4f}]$, "
            f"$p{{=}}{pv:.3f}$)")
    if len(common) != 5 or want not in flat:
        fails.append(f"derived features minus gated: paired over seeds {common}; the paper does not say '{want}'")
    if "comes out ahead in all five seeds, but its interval crosses zero" not in flat or \
            not (min(fdiff) > 0 and lo < 0 < hi_):
        fails.append(f"derived Section 5.1 D: the reading of the paired test no longer matches "
                     f"(differences {[round(d, 4) for d in fdiff]}, CI [{lo:+.4f},{hi_:+.4f}])")
    # Each place that sets the two effects side by side (the abstract, Table 4 row D) must carry the
    # caveat exactly when the paired interval crosses zero. Checked per place: a phrase-anywhere test
    # let either copy cover for the other's removal (2026-10-01, two planted faults missed).
    for ctx in ("had no detectable effect (the two effects do not differ significantly)",
                "has no detectable effect, and the two do not differ significantly"):
        if (lo < 0 < hi_) != (ctx in flat):
            fails.append(f"derived features minus gated: '{ctx}' is {'present' if ctx in flat else 'missing'}, "
                         f"but the paired interval is [{lo:+.4f},{hi_:+.4f}]")
    if lo <= 0:
        # Not established, so no sentence may assert it, however it is worded: any comparative that
        # reaches the gated model fails unless the sentence frames the comparison as open. A fixed phrase
        # list was beaten by a paraphrase ("matter more than message passing") on 2026-10-01, and the
        # plain-language paper names the model rather than "h1", so the target covers both.
        comparative = re.compile(r"\b(?:more|larger|greater|bigger|stronger|outweighs?|beats?|exceeds?|"
                                 r"surpass\w*|outperform\w*)\b[^.;:]{0,60}?\b(?:message passing|h1\b|"
                                 r"the gated (?:graph|model))", re.I)
        hedged = re.compile(r"\b(?:whether|not settled|not established|no detectable|cannot say)\b", re.I)
        for sentence in re.split(r"(?<=[.;:])\s+", flat):
            hit = comparative.search(sentence)
            if hit and not hedged.search(sentence):
                fails.append(f"derived features minus gated: the paper asserts '{hit.group(0)}', but the "
                             f"paired interval [{lo:+.4f},{hi_:+.4f}] crosses zero")

    # ---- Appendix E, D: the feature-ablation intervals, from each variant's mean and sd --------------
    from scipy import stats
    fa = {v["variant"]: v for v in load("data/results/feature_ablation_report.json")}
    for variant, lead in (("nograph", "the embedding together changes \\textsc{systema} by $"),
                          ("nodegree", "zeroing the counts alone by $")):
        v = fa[variant]
        h = stats.t.ppf(0.975, v["n"] - 1) * v["sd"] / v["n"] ** 0.5
        mid = "$ ($95\\%$ CI $[" if variant == "nograph" else "$ ($["
        want = f"{lead}{v['mean']:+.5f}{mid}{v['mean'] - h:+.5f},{v['mean'] + h:+.5f}]$"
        if want not in flat:
            fails.append(f"derived feature ablation {variant}: the paper does not say '{want}'")

    # ---- Appendix E, B: the four comparisons under Pearson-delta, and program cosine ---------------------
    sm = load("data/results/screening_lambda0/second_metric_5seed.json")["metrics"]
    pear = {c["contrast"]: c for c in sm["pearson"]["contrasts"]}
    surv = []
    for a, b, lab in (("typed_static", "expression_only", "typed $-$ no graph"),
                      ("condition_gated", "typed_static", "gated $-$ typed"),
                      ("untyped_gnn", "expression_only", "untyped $-$ no graph"),
                      ("condition_gated", "expression_only", "gated $-$ no graph")):
        c = pear.get(f"{a} - {b}")
        sign = 1.0
        if c is None:
            c, sign = pear[f"{b} - {a}"], -1.0
        elif c.get("A") not in (None, a):
            sign = -1.0
        if f"${sign * c['mean']:+.4f}$ ({lab}" not in flat:
            fails.append(f"derived Pearson-delta {lab}: the artifact gives {sign * c['mean']:+.4f}, not stated in App.~E")
        surv.append(bool(c.get("survives_both")))
    if surv != [True, True, True, False] or "the first three pass both corrections and the last does not" not in flat:
        fails.append(f"derived Pearson-delta: survival is {surv}; App.~E says the first three pass and the last does not")
    need("Pearson-delta gated p", pear["condition_gated - expression_only"]["p_value"], r"(gated $-$ no graph, $p={V}$)")
    pc = {c["contrast"]: c for c in sm["prog_cos"]["contrasts"]}["condition_gated - expression_only"]
    if "Program cosine similarity" not in flat or f"agrees (${pc['mean']:+.4f}$, $p={pc['p_value']:.2f}$)" not in flat \
            or pc["p_value"] < 0.05:
        fails.append(f"derived program cosine: {pc['mean']:+.4f} p={pc['p_value']:.3f}; App.~E's statement differs")

    # ---- Appendix E, B: other groups' scores, and the gene groups ---------------------------------------
    am = load("data/results/a3_external_live/rescored.json")["across_metric"]

    def survives(o):
        return o.get("p_bonferroni", 1) < 0.05 and o.get("p_holm", 1) < 0.05
    if len(am) != 20 or "correcting over all twenty tests" not in flat:
        fails.append(f"derived: App.~E says twenty tests; the re-scoring has {len(am)}")
    for_graph = [k for k, o in am.items() if k.endswith("/h1_vs_no_graph") and survives(o) and o["mean"] > 0]
    if for_graph or "passes correction in the graph's favor under none of them" not in flat:
        fails.append(f"derived: the gated model passes in the graph's favor under {for_graph}")
    pdx, top = am["pearson_delta/promotion_margin"], am["pearson_delta_top20/promotion_margin"]
    want = (f"positive under TxPert's all-gene Pearson-delta (${pdx['mean']:+.4f}$) and negative under "
            f"\\textsc{{gears}}' correlation over each perturbation's twenty most-changed genes "
            f"(${top['mean']:+.4f}$), both significant")
    if not (survives(pdx) and survives(top) and pdx["mean"] > 0 > top["mean"]) or want not in flat:
        fails.append(f"derived: the untyped sign flip between scores; the paper does not say '{want}'")
    h1top = am["pearson_delta_top20/h1_vs_no_graph"]
    if not (survives(h1top) and h1top["mean"] < 0) or "passes correction over the twenty scores" not in flat:
        fails.append(f"derived: the gated model's top-20 loss over the twenty scores, survives={survives(h1top)}")

    head = load("data/results/b2_deciles_live/deciles.json")["schemes"]["head"]

    def stars(contrast):
        out = []
        for key, o in head["cells"].items():
            iv, _, c = key.rpartition("/")
            if c == contrast and isinstance(o, dict) and o.get("survives_family_wise"):
                out.append((iv, o))
        return out
    s_h1 = stars("h1_vs_no_graph")
    if sorted(i for i, _ in s_h1) != ["2501-5000", "5001-10282"] or any(o["mean"] <= 0 for _, o in s_h1) \
            or "only on the two groups of genes that change least" not in flat:
        fails.append(f"derived gene groups: the gated model passes in {[(i, round(o['mean'], 4)) for i, o in s_h1]}; "
                     f"App.~E says only the two least-changed groups, both positive")
    else:
        by = dict(s_h1)
        need("gene group 2501-5000", by["2501-5000"]["mean"], r"change least ($+{V}$ and")
        need("gene group 5001-10282", by["5001-10282"]["mean"], r"and $+{V}$, five of five seeds each")
        need("gene group 2501-5000 Bonferroni", by["2501-5000"]["p_bonferroni"], r"Bonferroni ${V}$ and $0.024")
        need("gene group 5001-10282 Bonferroni", by["5001-10282"]["p_bonferroni"], r"and ${V}$); over ten equal groups")
        if not all(len(o["deltas"]) == 5 and min(o["deltas"]) > 0 for o in by.values()):
            fails.append("derived gene groups: 'five of five seeds each' no longer holds")
    t20 = head["cells"]["1-20/h1_vs_no_graph"]
    need("gated top-20 loss", t20["mean"], r"change most (${V}$) passes correction")
    need("gated top-20 Bonferroni over the gene groups", t20["p_bonferroni"], r"finer gene groups (Bonferroni ${V}$)")
    if t20.get("survives_family_wise"):
        fails.append("derived: the gated top-20 loss now survives over the gene groups; App.~E says it does not")

    # ---- Section 5.2 and Appendix F: the planted-signal test --------------------------------------------
    u = load("data/results/a2_ladder/floor.json")
    g = load("data/results/c1_ladder/floor_condition_gated.json")
    inj = [k for k in u["contrasts"] if not k.startswith("permuted")]
    if not all(u["contrasts"][k]["survives_family_wise"] for k in inj) or \
            u["contrasts"]["permuted_d400"]["survives_family_wise"] or \
            "passes correction at every planted size" not in flat:
        fails.append("derived planted signal: the untyped model no longer passes at every size and not the control")
    if any(c.get("survives_family_wise") for c in g["contrasts"].values()) or \
            "The gated model's passes at none" not in flat:
        fails.append("derived planted signal: a gated step now passes; the paper says none does")
    gh = g["gate_health"]
    if gh["status"] != "live" or gh["min_gate_mean"] <= GATE_DEAD or len(gh["lanes"]) != 24 or \
            "All $24$ gated runs kept live gates" not in flat:
        fails.append(f"derived planted signal: gate health {gh['status']}, minimum {gh['min_gate_mean']:.4f}, "
                     f"{len(gh['lanes'])} lanes")
    # stated twice, so anchored at each place: one copy cannot cover for the other's corruption
    need("untyped starting gap, Section 5.2", u["zero_point"]["mean"], r"but it was already $+{V}$ ahead of no graph")
    need("untyped starting gap, App. F", u["zero_point"]["mean"], r"the untyped model is already $+{V}$ ahead of no graph")
    if g["zero_point"]["mean"] >= 0:
        fails.append("derived planted signal: the gated model no longer starts behind no graph")
    need("gated starting gap", abs(g["zero_point"]["mean"]), r"the gated model ${V}$ behind")
    ui, gi = u["post_hoc_increment_over_zero"]["d400"], g["post_hoc_increment_over_zero"]["d400"]
    if min(1.0, 6 * ui["p_value"]) <= 0.05 or min(1.0, 6 * gi["p_value"]) <= 0.05 or \
            "and there neither passes correction" not in flat:
        fails.append("derived planted signal: an increment at 0.40 now survives correction over six")
    need("untyped increment at 0.40", ui["mean"], r"the untyped model adds $+{V}$ ($p{=}")
    need("untyped increment p", ui["p_value"], r"($p{=}{V}$) and the gated model")
    need("gated increment at 0.40", gi["mean"], r"and the gated model $+{V}$ ($p{=}")
    need("gated increment p", gi["p_value"], r"($p{=}{V}$), both uncorrected")

    # ---- 2026-10-28 review fixes: claims three reviews made the paper state, each re-derived ----------
    sc = load("data/results/screening_lambda0/robustness_5seed.json")["contrasts"]
    n7c = load("data/results/screening_n7_live/robustness_5seed.json")["contrasts"]
    # the untyped gain needs seven seeds: it fails correction at five and passes at seven
    if sc["promotion_margin"]["survives_family_wise"] or not n7c["promotion_margin"]["survives_family_wise"]:
        fails.append("derived: the paper says the untyped gain fails correction at five seeds and passes at seven")
    for phrase in ("only after we added two runs per model to the first five",
                   "we added two after the five-seed result fell short of correction"):
        if phrase not in flat:
            fails.append(f"derived: '{phrase}' is no longer in the paper; update this check")
    # the pooled typed interval cannot rule out a gain as large as the untyped one
    p7 = load("data/results/replication/pooled.json")["pooled"]["h2a"]
    if not p7["random_ci"][1] > n7c["promotion_margin"]["mean"] or \
            "but not one as large as the untyped model's gain on the T-cell screen" not in flat:
        fails.append(f"derived: the typed pooled upper bound {p7['random_ci'][1]:+.4f} against the untyped "
                     f"gain {n7c['promotion_margin']['mean']:+.4f}; the paper's reading differs")
    # Frangieh is the only replication dataset with a trained gated model
    gated = sorted(q.parent.name for q in (ROOT / "data/results/replication").glob("*/robustness_5seed.json")
                   if not q.parent.name.startswith("_")
                   and json.loads(q.read_text())["contrasts"]["h1_vs_no_graph"].get("n"))
    if gated != ["FrangiehIzar2021_RNA"] or "the only one on which we trained the gated model" not in flat:
        fails.append(f"derived: gated replication lanes {gated}; the paper says only Frangieh")
    # the degree-scaled variant was the first of four registered, and over four it would not pass
    b1p = load("data/results/screening_b1/b1_message_form.json")["contrasts"]["D3"]["p_value"]
    if min(1.0, 4 * b1p) <= 0.05 or "corrected over all four it would not pass" not in flat:
        fails.append(f"derived: degree-scaled p {b1p:.4f} over four is {min(1.0, 4 * b1p):.3f}")
    need("degree-scaled Bonferroni over four", min(1.0, 4 * b1p), r"would not pass Bonferroni over all four (${V}$)")
    # random labels recover about half of the typed shortfall, and neither mechanism model passes
    a1f = load("data/results/screening_a1/a1_mechanism.json")
    share = a1f["family"]["D2"]["mean"] / abs(sc["h2a"]["mean"])
    if not (0.4 < share < 0.6) or any(a1f["family"][k]["survives_family_wise"] for k in ("D1", "D2")) \
            or "random labels recover about half of it" not in flat:
        fails.append(f"derived: random labels recover {share:.2f} of the typed shortfall; App. B says about half")
    if a1f["family_size"] != 2 or "family of two). One shares" not in flat:
        fails.append("derived: App. B corrects the two mechanism models as a family of two")
    # the shared-module interval caps what the extra parameters can explain at about two fifths
    cap = a1f["family"]["D1"]["ci_high"] / abs(sc["h2a"]["mean"])
    if not (0.35 <= cap <= 0.45) or "(at most about two fifths, by the interval)" not in flat:
        fails.append(f"derived: the shared-module bound is {cap:.2f} of the typed shortfall; App. B says two fifths")
    # the K=128 subset leaves out exactly Norman and the two Tian datasets
    k128 = load("data/results/replication/pooled_k128_subset.json")["pooled"]["promotion_margin"]["datasets"]
    left = sorted(set(load("data/results/replication/pooled_with_reference.json")["per_dataset"]
                      ["promotion_margin"]) - set(k128))
    if left != ["NormanWeissman2019_filtered", "TianKampmann2021_CRISPRa", "TianKampmann2021_CRISPRi"] or \
            "which leaves out Norman and the two Tian datasets" not in flat:
        fails.append(f"derived: the K=128 subset leaves out {left}")
    # App. C's amendment sentence: what the ORIGINAL replicate rule kept of Replogle RPE1, read from the
    # public survey's measured-outcome table (a build record, so its two literals are declared above)
    srv = (ROOT / "docs/replication-dataset-survey.md").read_text()
    row = re.search(r"\| ReplogleWeissman2022 \(RPE1\) \| \d+ \| \*\*(\d+)\*\* / ([\d,]+) \|", srv)
    if row is None:
        fails.append("derived App. C: the survey's measured-outcome row for RPE1 is gone; the amendment "
                     "sentence has no source")
    elif (f"kept only ${row.group(1)}$ of Replogle RPE1's ${row.group(2).replace(',', '{,}')}$ targets"
          not in flat):
        fails.append(f"derived App. C: the original rule kept {row.group(1)} of {row.group(2)} RPE1 targets; "
                     f"the paper says otherwise")
    # the live re-scoring: two survivals lost, gene groups gained (corrections list)
    am_dead = load("data/results/a3_external/rescored.json")["across_metric"]
    am_live = load("data/results/a3_external_live/rescored.json")["across_metric"]
    lost = sorted(k for k in am_dead if survives(am_dead[k]) and not survives(am_live[k]))
    hd = load("data/results/b2_deciles/deciles.json")["schemes"]
    hl = load("data/results/b2_deciles_live/deciles.json")["schemes"]
    def h1_pass(sch):
        return sorted(k.split("/")[0] for k, o in sch["cells"].items()
                      if k.endswith("/h1_vs_no_graph") and o.get("survives_family_wise"))
    top_lost = (hd["head"]["cells"]["1-20/h1_vs_no_graph"]["survives_family_wise"]
                and not hl["head"]["cells"]["1-20/h1_vs_no_graph"]["survives_family_wise"])
    if lost != ["energy_distance/h2b"] or not top_lost or len(h1_pass(hl["head"])) != 2 or \
            (len(h1_pass(hd["deciles"])), len(h1_pass(hl["deciles"]))) != (2, 5) or \
            "two significant results disappear" not in flat or "on five of ten, up from two" not in flat:
        fails.append(f"derived corrections: lost {lost}, top-20 lost {top_lost}, deciles "
                     f"{h1_pass(hd['deciles'])} -> {h1_pass(hl['deciles'])}")
    # the typed model peaks at its first epoch in four of five seeds
    tr = load("data/results/camera_ready/facts.json")["training"]["typed_static"]
    first = sum(1 for lane in tr.values() if lane["best_epoch"] == 0)
    for phrase in ("scored after its first epoch in four of five seeds",
                   "peaks at its very first epoch in four of five seeds"):
        if first != 4 or len(tr) != 5 or phrase not in flat:
            fails.append(f"derived: the typed best epoch is the first in {first} of {len(tr)} seeds; '{phrase}'")
    # the feature ablation's p-values are Bonferroni over its three versions
    for v in fa.values():
        if abs(v["p_bonf"] - min(1.0, 3 * v["p_raw"])) > 1e-12:
            fails.append(f"derived: feature ablation {v['variant']} p_bonf is not three times p_raw")
    if len(fa) != 3 or "Corrected over these three versions" not in flat:
        fails.append("derived: App. E says the feature-ablation p-values are corrected over three versions")
    # a rationale alone sits slightly FARTHER from the full prediction than random edges
    suf = load("data/results/a2_power/rationale_bound.json")["sufficiency"]
    if not (suf["gap"] > 0 and suf["p_value"] < 0.05) or ("is slightly farther from the full prediction than "
                                                          "one made from the same number of random edges") not in flat:
        fails.append(f"derived: rationale sufficiency gap {suf['gap']:+.4f} p={suf['p_value']:.2g}; App. E differs")
    # the planted-signal starting gaps are four-seed values
    if u["zero_point"]["n"] != 4 or g["zero_point"]["n"] != 4 or "at $\\delta=0$ (four seeds)" not in flat:
        fails.append("derived planted signal: the starting gaps are not four-seed values, or the paper no longer says so")
    # the other pre-registered grouping: ten equal groups by descending change, a family of 40
    dec = load("data/results/b2_deciles_live/deciles.json")["schemes"]["deciles"]
    names = [b[2] for b in dec["bins"]]
    passed = [nm for nm in names if dec["cells"][f"{nm}/h1_vs_no_graph"].get("survives_family_wise")]
    if dec["family_size"] != 40 or len(names) != 10 or len(passed) != 5 or set(passed) & set(names[:4]) or \
            "over ten equal groups ($40$ tests) it passes on five, none among the four that change most" not in flat:
        fails.append(f"derived deciles: the gated model passes in {passed} of {names} (family {dec['family_size']})")
    if head["family_size"] != 36 or len(head["bins"]) != 9 or "Over nine groups ($36$ tests)" not in flat:
        fails.append(f"derived gene groups: {len(head['bins'])} groups, family {head['family_size']}")

    return fails, ("full: 72 cross-artifact derivations and their wording, recomputed from the sweep, the "
                   "pooled and per-dataset replication files, the feature ablation, the second metric, the "
                   "re-scoring, the gene groups, both planted-signal ladders and b1_message_form.json")


@table("prose:headline")
def _prose_headline(tex):
    """The load-bearing numbers that live in PROSE, not in any table.

    WHY THIS EXISTS. Until 2026-08-26 almost none of the paper's prose numbers were checked against an
    artifact; they were guarded by the literal inventory alone, which catches a number that CHANGES
    during an edit and passes one that was WRONG when it was snapshotted. That gap was not hypothetical:
    the paper reported Holm 0.021 where screening_untyped_n7 gives 0.02046976.

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
    alone. Rebuilt 2026-10-28 for the shortened plain-language paper: one anchor per number it prints."""
    flat = " ".join(re.sub(r"(?m)^%.*", "", tex).split())
    R = "data/results/"
    L = load(R + "screening_lambda0/robustness_5seed.json")
    LC, LP = L["contrasts"], L["per_config"]
    N7 = load(R + "screening_n7_live/robustness_5seed.json")["contrasts"]
    INT = load(R + "screening_c080c10_h1/robustness_5seed.json")
    HD = load(R + "screening_c075c15_n5/robustness_5seed.json")["contrasts"]
    R2 = load(R + "screening_c080c10_r2/robustness_5seed.json")["contrasts"]
    R3 = load(R + "screening_c080c10_r3/robustness_5seed.json")["contrasts"]
    P8 = load(R + "replication/pooled_with_reference.json")
    P7 = load(R + "replication/pooled.json")["pooled"]["h2a"]
    K128 = load(R + "replication/pooled_k128_subset.json")["pooled"]["promotion_margin"]
    FA = {v["variant"]: v for v in load(R + "feature_ablation_report.json")}
    FL = load(R + "a2_ladder/floor.json")
    C1 = load(R + "c1_ladder/floor_condition_gated.json")
    A1 = load(R + "screening_a1/a1_mechanism.json")["family"]
    AB = load(R + "a2_power/arch_search_bound.json")
    B1 = load(R + "screening_b1/b1_message_form.json")
    RPE = load(R + "replication/ReplogleWeissman2022_rpe1/robustness_5seed.json")["contrasts"]["promotion_margin"]
    FRG = load(R + "replication/FrangiehIzar2021_RNA/robustness_5seed.json")["contrasts"]["h1_vs_no_graph"]
    SPF, SPI, SPH = (load("data/splits/manifest.json"), load(R + "splits_c080c10/manifest.json"),
                     load(R + "splits_c075c15/manifest.json"))
    pm7, pm5 = N7["promotion_margin"], LC["promotion_margin"]
    pd8 = P8["per_dataset"]
    ng, nod, nop = FA["nograph"], FA["nodegree"], FA["nopinnacle"]
    d3 = B1["contrasts"]["D3"]
    means5 = sorted(v["mean"] for v in LP.values())

    def f4(x):
        return f"{x:+.4f}"

    def f5(x):
        return f"{x:+.5f}"

    def h1(c):
        return c["h1_vs_no_graph"]

    claims = [
        # --- abstract and Section 4 -------------------------------------------------------------------
        ("main comparison",          "\\textsc{systema}, was ${v}$", f4(h1(LC)["mean"])),
        ("model scores, Methods",    "the four models score between ${v}$ (Table",
         f"{means5[0]:.4f}$ and ${means5[-1]:.4f}"),
        ("graph features, abstract", "lowered its score by ${v}$, a drop that survives", f"{abs(ng['mean']):.4f}"),
        ("intermediate split, body", "the intermediate split: ${v}$, positive in all five seeds", f4(h1(INT["contrasts"])["mean"])),
        ("gated n=7, body",          "With seven seeds it is ${v}$, also not significant", f4(h1(N7)["mean"])),
        ("re-draws, body",           "gave ${v}$, and none of the three passes", f"{f4(h1(R2)['mean'])}$ and ${f4(h1(R3)['mean'])}"),
        ("intermediate split p",     "with uncorrected $p{=}{v}$", f"{h1(INT['contrasts'])['p_value']:.3f}"),
        ("harder split typed loss",  "there the difference is ${v}$, $95\\%$ CI", f4(HD["h2a"]["mean"])),
        ("harder split typed CI",    "CI $[{v}]$, $p=", f"{HD['h2a']['ci_low']:+.4f},{HD['h2a']['ci_high']:+.4f}"),
        ("harder split typed p",     "$p={v}$, and one seed reverses sign", f"{HD['h2a']['p_value']:.2f}"),
        ("typed pooled",             "worth ${v}$ \\textsc{systema}, with a random-effects", f4(P7["random_effect"])),
        ("typed pooled CI",          "random-effects $95\\%$ CI of $[{v}]$", f"{P7['random_ci'][0]:+.4f},{P7['random_ci'][1]:+.4f}"),
        ("untyped n=7",              "(${v}$ \\textsc{systema} at $n{=}7$", f4(pm7["mean"])),
        ("untyped n=7 Bonferroni",   "Bonferroni ${v}$ and Holm", f"{pm7['p_bonferroni']:.3f}"),
        ("untyped n=7 Holm",         "and Holm ${v}$) and on", f"{pm7['p_holm']:.3f}"),
        ("RPE1 untyped",             "RPE1 (${v}$, all four seeds positive)", f4(pd8["promotion_margin"]["ReplogleWeissman2022_rpe1"]["mean"])),
        ("Norman untyped",           "Norman (${v}$; all four of its seeds", f4(pd8["promotion_margin"]["NormanWeissman2019_filtered"]["mean"])),
        ("RPE1 typed",               "scores lower than no graph (${v}$, not significant)", f4(pd8["h2a"]["ReplogleWeissman2022_rpe1"]["mean"])),
        # The K=128 pool stays in the body: reviewers need not read appendices (SUBMISSION.md), and it is
        # the one estimate that speaks to whether Norman's sign is about its smaller K.
        ("untyped all-eight, body",  "Pooled over all eight datasets, the untyped gain is ${v}$ ($p{=}",
         f4(P8["pooled"]["promotion_margin"]["random_effect"])),
        ("untyped all-eight p, body", "the untyped gain is $+0.0091$ ($p{=}{v}$)", f"{P8['pooled']['promotion_margin']['random_p']:.2f}"),
        ("K=128 pool, body",         "and the two Tian datasets, it is ${v}$ ($p{=}", f4(K128["random_effect"])),
        ("K=128 pool p, body",       "it is $+0.0141$ ($p{=}{v}$). Our plan", f"{K128['random_p']:.4f}"),
        # --- Section 5 -----------------------------------------------------------------------------------
        ("graph features, body",     "lowers \\textsc{systema} by ${v}$, a drop that survives correction", f"{abs(ng['mean']):.4f}"),
        ("planted floor",            "down to the smallest we tried, ${v}$, but", f"{FL['floor']:.2f}"),
        ("largest planted size",     "passed correction at no size up to ${v}$ response standard deviations", f"{FL['rungs']['d400']['delta']:.2f}"),
        # --- Appendix B --------------------------------------------------------------------------------
        ("typed loss, n=7",          "the typed model scores ${v}$ against no graph", f4(N7["h2a"]["mean"])),
        ("shared module",            "moves \\textsc{systema} by ${v}$", f4(A1["D1"]["mean"])),
        ("shared module CI",         "against the typed model ($95\\%$ CI $[{v}]$)", f"{A1['D1']['ci_low']:+.4f},{A1['D1']['ci_high']:+.4f}"),
        ("random types",             "moves it by ${v}$", f4(A1["D2"]["mean"])),
        ("random types CI",          "moves it by $+0.0065$ ($[{v}]$)", f"{A1['D2']['ci_low']:+.4f},{A1['D2']['ci_high']:+.4f}"),
        ("search runs",              "Its ${v}$ runs used", str(AB["n_cells"])),
        ("search range",             "The search's whole range, ${v}$, was narrower", f"{AB['observed_spread']:.4f}"),
        ("search seed-noise span",   "scores to differ by about ${v}$.", f"{AB['expected_spread_from_seed_noise_alone']:.4f}"),
        ("untyped minus typed gap",  "about half the ${v}$ gap between", f"{B1['gap_untyped_minus_typed']['mean']:.4f}"),
        ("degree-scaled gain",       "gains ${v}$ \\textsc{systema} over the typed model", f4(d3["mean"])),
        ("degree-scaled CI",         "over the typed model ($95\\%$ CI $[{v}]$", f"{d3['ci_low']:+.4f},{d3['ci_high']:+.4f}"),
        # --- Appendix C --------------------------------------------------------------------------------
        ("typed pooled, App C",      "the typed graph's ${v}$ has a random-effects", f4(P7["random_effect"])),
        ("typed pooled FE CI",       "a fixed-effect CI of $[{v}]$", f"{P7['fixed_ci'][0]:+.4f},{P7['fixed_ci'][1]:+.4f}"),
        ("untyped all-eight pool",   "Over all eight datasets the untyped comparison is ${v}$ ($p{=}", f4(P8["pooled"]["promotion_margin"]["random_effect"])),
        ("untyped all-eight p",      "is $+0.0091$ ($p{=}{v}$)", f"{P8['pooled']['promotion_margin']['random_p']:.2f}"),
        ("K=128 pool, App C",        "with $K{=}128$ it is ${v}$ ($[", f4(K128["random_effect"])),
        ("K=128 pool CI, App C",     "it is $+0.0141$ ($[{v}]$", f"{K128['random_ci'][0]:+.4f},{K128['random_ci'][1]:+.4f}"),
        ("RPE1 CI",                  "untyped gain has $95\\%$ CI $[{v}]$", f"{RPE['ci_low']:+.4f},{RPE['ci_high']:+.4f}"),
        ("RPE1 p",                   "($p{=}{v}$, Bonferroni and Holm both", f"{RPE['p_value']:.3f}"),
        ("RPE1 Bonferroni and Holm", "Holm both ${v}$ in that dataset's family", f"{RPE['p_bonferroni']:.3f}"),
        ("RPE1 family size",         "dataset's family of {v})", "two" if RPE["family_size"] == 2 else str(RPE["family_size"])),
        ("Frangieh gated",           "the gated model against no graph is ${v}$ ($95\\%$ CI", f4(FRG["mean"])),
        ("Frangieh gated CI",        "is $+0.0033$ ($95\\%$ CI $[{v}]$)", f"{FRG['ci_low']:+.4f},{FRG['ci_high']:+.4f}"),
        ("untyped n=5",              "the untyped comparison was ${v}$ ($95\\%$ CI", f4(pm5["mean"])),
        ("untyped n=5 CI",           "was $+0.0045$ ($95\\%$ CI $[{v}]$", f"{pm5['ci_low']:+.4f},{pm5['ci_high']:+.4f}"),
        ("untyped n=5 Holm",         "Holm ${v}$ and Bonferroni", f"{pm5['p_holm']:.3f}"),
        ("untyped n=5 Bonferroni",   "Bonferroni ${v}$, clearing neither", f"{pm5['p_bonferroni']:.3f}"),
        ("untyped n=7, App C",       "the estimate is ${v}$ and it passes both corrections", f4(pm7["mean"])),
        ("gated n=7, App C",         "At seven seeds the gated model against no graph is ${v}; our main test",
         f"{f4(h1(N7)['mean'])}$ ($95\\%$ CI $[{f4(h1(N7)['ci_low'])},{f4(h1(N7)['ci_high'])}]$)"),
        # --- Appendix D --------------------------------------------------------------------------------
        ("main split threshold",     "(centered cosine $\\geq {v}$)", f"{SPF['seq_cosine_threshold']:.2f}"),
        ("main split cap",           "No group may hold more than ${v}\\%$ of targets", f"{SPF['group_size_cap_frac'] * 100:.0f}"),
        ("intermediate split",       "{v} (intermediate)", f"${SPI['seq_cosine_threshold']:.2f}$ with ${SPI['group_size_cap_frac'] * 100:.0f}\\%$"),
        ("harder split",             "and {v} (harder)", f"${SPH['seq_cosine_threshold']:.2f}$ with ${SPH['group_size_cap_frac'] * 100:.0f}\\%$"),
        ("gated on three splits",    "no graph is ${v}$ on the main, intermediate and harder splits",
         f"{f4(h1(LC)['mean'])}$, ${f4(h1(INT['contrasts'])['mean'])}$ and ${f4(h1(HD)['mean'])}"),
        ("Bonferroni on three splits", "none passes correction (Bonferroni ${v}$)",
         f"{h1(LC)['p_bonferroni']:.3f}$, ${h1(INT['contrasts'])['p_bonferroni']:.3f}$ and ${h1(HD)['p_bonferroni']:.3f}"),
        ("typed on three splits",    "The typed model's shortfall is ${v}$.",
         f"{f4(LC['h2a']['mean'])}$, ${f4(INT['contrasts']['h2a']['mean'])}$ and ${f4(HD['h2a']['mean'])}"),
        ("intermediate typed p",     "passes no correction ($p={v}$) because", f"{INT['contrasts']['h2a']['p_value']:.2f}"),
        ("intermediate typed sd",    "spread across seeds is ${v}$ there", f"{INT['per_config']['typed_static']['sd']:.4f}"),
        ("main typed sd",            "against ${v}$ on the main split", f"{LP['typed_static']['sd']:.4f}"),
        ("three re-draws",           "give the gated model ${v}, all uncorrected",
         f"{f4(h1(INT['contrasts'])['mean'])}$ ($p={h1(INT['contrasts'])['p_value']:.3f}$), "
         f"${f4(h1(R2)['mean'])}$ ($p={h1(R2)['p_value']:.3f}$) and ${f4(h1(R3)['mean'])}$ ($p={h1(R3)['p_value']:.3f}$)"),
        # The re-draw roots ran fewer models, so their own p_bonferroni uses a smaller family; the paper
        # corrects at m=4, the pre-registered family, so this re-derives it that way.
        ("re-draws over four",       "over the family of four (${v}$)",
         f"{min(1.0, 4 * h1(INT['contrasts'])['p_value']):.3f}$, ${min(1.0, 4 * h1(R2)['p_value']):.3f}$, "
         f"${min(1.0, 4 * h1(R3)['p_value']):.3f}"),
        # --- Appendix E --------------------------------------------------------------------------------
        ("features zeroed",          "the embedding together changes \\textsc{systema} by ${v}$ ($95\\%$ CI", f5(ng["mean"])),
        ("three ablation p-values",  "the $p$-values are ${v}$. Nearly all",
         f"{ng['p_bonf']:.4f}$, ${nod['p_bonf']:.4f}$ and ${nop['p_bonf']:.2f}"),
        ("counts zeroed",            "zeroing the counts alone by ${v}$ ($[", f5(nod["mean"])),
        ("PINNACLE zeroed",          "zeroing PINNACLE alone by ${v}$. Corrected", f"{nop['mean']:+.6f}"),
        ("no-graph baseline",        "about four percent of the no-graph model's ${v}$;", f"{LP['expression_only']['mean']:.4f}"),
        # --- Appendix F --------------------------------------------------------------------------------
        ("planted 0.02",             "from $0.02$ (${v}$, Bonferroni", f4(FL["contrasts"]["d020"]["mean"])),
        ("planted 0.02 Bonferroni",  "Bonferroni ${v}$), and not in the control", f"{FL['contrasts']['d020']['p_bonferroni']:.3f}"),
        ("planted control",          "not in the control (${v}$)", f4(FL["contrasts"]["permuted_d400"]["mean"])),
        ("gated at 0.40",            "even at $0.40$ it is ${v}$ (Bonferroni", f4(C1["contrasts"]["d400"]["mean"])),
        ("gated at 0.40 Bonferroni", "(Bonferroni ${v}$). All", f"{C1['contrasts']['d400']['p_bonferroni']:.3f}"),
        ("gated minimum gate",       "(minimum mean ${v}$)", f"{C1['gate_health']['min_gate_mean']:.4f}"),
    ]
    fails = []
    # "about four percent" is a ratio the paper derives in words.
    if not 0.035 <= abs(ng["mean"]) / LP["expression_only"]["mean"] < 0.045:
        fails.append("prose: 'about four percent' no longer matches the ablation over the baseline")
    # "positive in all five seeds" (Section 4.2) is a claim about every seed, not the mean.
    _c = h1(INT["contrasts"])
    _sg = 1.0 if _c["better"] == "condition_gated" else -1.0
    if not (len(_c["deltas"]) == 5 and min(_sg * d for d in _c["deltas"]) > 0):
        fails.append(f"prose intermediate split: the paper says positive in all five seeds; per-seed {_c['deltas']}")
    if h1(N7)["survives_family_wise"]:
        fails.append("prose gated n=7: the paper calls it not significant, and it now survives")
    # "not significant" beside RPE1's typed difference is a claim about the dataset's family
    if pd8["h2a"]["ReplogleWeissman2022_rpe1"]["survives_family_wise"]:
        fails.append("prose RPE1 typed: the paper calls it not significant, and it now survives")
    for what, tmpl, val in claims:
        ctx = tmpl.replace("{v}", val)
        if ctx not in flat:
            fails.append(f"prose {what}: the artifact gives {val}, and the paper does not state it in "
                         f"context (looked for {ctx!r})")
    return fails, (f"full: {len(claims)} load-bearing prose claims re-derived from their artifacts, each "
                   f"anchored to its surrounding words")


@table("prose:camera")
def _prose_camera(tex):
    """The numbers the 2026-10-01 camera-ready introduced, each anchored to its surrounding words.

    Most come from camera_ready/facts.json, which persisted quantities the accepted paper had quoted
    from logs, comments or prose and got wrong (training length, live gate range, the paralog
    guarantee); the rest from the Hartung-Knapp pools and the list of corrections, which quotes both the
    live and the replaced value of each."""
    flat = " ".join(re.sub(r"(?m)^%.*", "", tex).split())
    F = load("data/results/camera_ready/facts.json")
    tr, vp = F["training"], F["val_paralog"]
    col = {a: {k: [lane[k] for lane in tr[a].values()] for k in ("best_epoch", "epochs_run")}
           for a in tr}
    gates = [lane["gate_at_best"] for lane in tr["condition_gated"].values()]
    gpu = [lane["gpu_hours"] for a in tr for lane in tr[a].values()]
    eo, ug = col["expression_only"]["best_epoch"], col["untyped_gnn"]["best_epoch"]
    typed_best = max(col["typed_static"]["best_epoch"] + col["condition_gated"]["best_epoch"])
    word = {3: "three", 4: "four", 5: "five"}
    # the paper counts epochs from 1; the artifact stores them from 0
    span = lambda xs: f"${min(xs) + 1}$ to ${max(xs) + 1}$"
    maxep = max(x for a in col for x in col[a]["epochs_run"])
    k128 = load("data/results/camera_ready/pooled_k128_subset_hksj.json")["pooled"]["promotion_margin"]
    all8 = load("data/results/camera_ready/pooled_with_reference_hksj.json")["pooled"]["promotion_margin"]
    typ7 = load("data/results/camera_ready/pooled_hksj.json")["pooled"]["h2a"]
    ci = lambda p: f"$[{p['hksj_ci'][0]:+.4f},{p['hksj_ci'][1]:+.4f}]$"
    live = load("data/results/screening_n7_live/robustness_5seed.json")["contrasts"]
    dead = load("data/results/screening_untyped_n7/robustness_5seed.json")["contrasts"]
    f4 = lambda x: f"{x:+.4f}"
    claims = [
        ("live gate range", f"with a mean between ${min(gates):.2f}$ and ${max(gates):.2f}$ in the checkpoints"),
        ("lane cost range", f"(${min(gpu):.1f}$ to ${max(gpu):.0f}$ GPU-hours each in the main family)"),
        ("val paralog, body", f"This is imperfect: ${vp['val_frac_ge_threshold'] * 100:.0f}\\%$ of validation"),
        ("naive paralog, body", f"against ${vp['naive_every_20th_frac'] * 100:.0f}\\%$ for a simple split that ignores"),
        ("eo and ug best epochs", f"reach theirs at epochs {span(eo)} and {span(ug)}, and all of them run to the "
                                  f"${maxep}$-epoch limit"),
        ("epoch limit, Methods", f"run trains for up to ${maxep}$ epochs, stops early if its validation loss has not "
                                 f"improved for ten epochs"),
        ("typed and gated peak", f"typed and gated runs peak within their first {word.get(typed_best + 1)} epochs"),
        ("HKSJ p both pools, body", f"adjustment for having few datasets ($p{{=}}{all8['hksj_p']:.2f}$ and "
                                    f"${k128['hksj_p']:.2f}$; Appendix"),
        ("HKSJ both untyped pools", f"raises these to $p{{=}}{all8['hksj_p']:.2f}$ and $p{{=}}{k128['hksj_p']:.2f}$"),
        ("HKSJ typed seven", f"widens the typed interval to {ci(typ7)}"),
        ("corrections: n=7 gated", f"is ${f4(live['h1_vs_no_graph']['mean'])}$ ($95\\%$ CI "
                                   f"$[{f4(live['h1_vs_no_graph']['ci_low'])},{f4(live['h1_vs_no_graph']['ci_high'])}]$) "
                                   f"rather than ${f4(dead['h1_vs_no_graph']['mean'])}$"),
        ("corrections: n=7 gates over typed", f"over the typed model is ${f4(live['h2b']['mean'])}$ rather than "
                                              f"${f4(dead['h2b']['mean'])}$"),
    ]
    fails = [f"prose:camera {what}: not stated in context (looked for {ctx!r})"
             for what, ctx in claims if ctx not in flat]
    # "all of them run to the limit" and "stop early ... ten epochs": the no-graph and untyped lanes hit the
    # cap, and every typed and gated lane stopped exactly ten epochs after its best one (patience 10).
    if any(e != maxep for a in ("expression_only", "untyped_gnn") for e in col[a]["epochs_run"]):
        fails.append("prose:camera: a no-graph or untyped lane stopped before the epoch limit")
    for a in ("typed_static", "condition_gated"):
        for b, e in zip(col[a]["best_epoch"], col[a]["epochs_run"]):
            if not (e < maxep and e == b + 11):
                fails.append(f"prose:camera: {a} stopped at {e} epochs with best epoch {b}; not patience ten")
    # The words these anchors cannot see: the dead-gate root must really have had an interval excluding
    # zero, and the live one must not.
    if not (dead["h1_vs_no_graph"]["ci_high"] < 0 < live["h1_vs_no_graph"]["ci_high"]) or \
            "with an interval excluding zero" not in flat:
        fails.append("prose:camera: the corrections say the dead-gate n=7 interval excluded zero and the "
                     "live one does not")
    return fails, f"full: {len(claims)} camera-ready claims anchored to their words, plus 1 word check"


@table("tab:family")
def _family(tex):
    """The headline five-seed family (Table 2).

    IT IS THE screening_lambda0 ROOT, NOT data/results/screening/. Both are n=5 on the frozen fold
    and they agree on three arms out of four, which is exactly how this could be got wrong: the
    unrepaired root puts condition_gated at 0.0838 and h1 at -0.0019, the repaired one at 0.0848 and
    -0.0009, and the paper's own text says the gate repair moved the headline between those two
    values. Checking against the wrong root would report a real number as an error. Since 2026-10-28 the
    table has four columns (comparison, difference, CI, significant) and the per-model means sit in its
    caption; the Pearson-delta column moved to App.~E, where prose:derived checks it."""
    fails, d = [], load(f"{_FAMILY_ROOT}/robustness_5seed.json")
    seen = 0
    for cs in cells_of(tex, "tab:family"):
        # Rows are named in words ("Typed $-$ no graph"); "untyped" must be tried before "typed".
        m = re.match(r"(untyped|typed|gated|no graph)\s*-\s*(untyped|typed|gated|no graph)\b",
                     cs[0].replace("\\textbf{", "").replace("$-$", "-").strip(), re.I)
        if not m or len(cs) < 4:
            continue
        a, b = _FAMILY_WORD[m.group(1).lower()], _FAMILY_WORD[m.group(2).lower()]
        key = _CONTRAST_OF.get((a, b))
        if key is None:
            continue
        seen += 1
        c = d["contrasts"][key]
        # The artifact orients every contrast better-minus-worse; the table writes A minus B.
        sign = 1.0 if (c["better"], c["worse"]) == (a, b) else -1.0
        cmp_cell(fails, f"tab:family {key} delta", cs[1], sign * c["mean"])
        lo, hi = (sign * c["ci_low"], sign * c["ci_high"])
        if sign < 0:
            lo, hi = hi, lo
        ci = re.findall(r"[-+]\d\.\d+", cs[2])
        if len(ci) != 2:
            fails.append(f"tab:family {key} CI: no interval in {cs[2]!r}")
        else:
            nd = len(ci[0].split(".")[1])
            if (ci[0], ci[1]) != (f"{lo:+.{nd}f}", f"{hi:+.{nd}f}"):
                fails.append(f"tab:family {key} CI: paper [{ci[0]},{ci[1]}] artifact "
                             f"[{lo:+.{nd}f},{hi:+.{nd}f}]")
        # "Significant" vocabulary, spelled out because it is not a yes/no field: "yes, worse" and
        # "yes, better" CLAIM survival, "no" denies it.
        word = cs[3].lower().replace("\\textbf{", "")
        claims = word.startswith("yes")
        if claims != bool(c["survives_family_wise"]):
            fails.append(f"tab:family {key} significant: paper {cs[3]!r} artifact "
                         f"survives={c['survives_family_wise']}")
        if claims and (("worse" in word) != (sign * c["mean"] < 0)):
            fails.append(f"tab:family {key}: the paper says {cs[3]!r} but the difference is {sign * c['mean']:+.4f}")

    # The per-model means are in the caption. It must name all four: a footer the regex no longer finds
    # used to be skipped silently, which reads exactly like a pass.
    foot = re.search(r"Mean \\textsc\{systema\} per model\s*\(\$n\{=\}5\$\):(.{0,200})", tex, re.S)
    pairs = re.findall(r"(untyped|typed|gated|no graph)\s*\$(\d\.\d+)\$", foot.group(1)) if foot else []
    if sorted(w for w, _ in pairs) != sorted(_FAMILY_WORD):
        fails.append(f"tab:family per-model means: expected one mean for each of the four models, "
                     f"found {pairs}")
    for word, val in pairs:
        want = d["per_config"][_FAMILY_WORD[word]]["mean"]
        nd = len(val.split(".")[1])
        if f"{float(val):.{nd}f}" != f"{want:.{nd}f}":
            fails.append(f"tab:family per-model {word}: paper {val} artifact {want:.{nd}f}")
    if seen != 4:
        fails.append(f"tab:family: matched {seen} of 4 contrast rows; the rest are unchecked")
        return fails, f"partial: matched {seen} of 4 contrast rows"
    return fails, (f"full: 4 contrasts x (difference, CI, significance and its direction) + 4 per-model "
                   f"means, {seen} rows, vs {_FAMILY_ROOT}/")


@table("tab:causes")
def _causes(tex):
    """A SUMMARY table (Table 4): its cells are verdicts in prose, so what is checked is each
    QUALITATIVE claim causes C and D make, re-derived from the artifacts behind them: a cell that says
    "worse" or "no detectable effect" fails if the artifact stops saying so.

    NOT CHECKED, and why: cause A's "$\\approx$0.001" centroid floor and cause E's "85% functional
    associations" are graph- and metric-summary quantities with no JSON of their own in this repo.
    Causes A, B and E's verdicts are qualitative; App.~E carries their numbers."""
    fails, blk = [], re.search(r"\\label\{tab:causes\}(.*?)\\end\{tabular\}", tex, re.S)
    body = " ".join((blk.group(1) if blk else "").split())
    n7 = load("data/results/screening_n7_live/robustness_5seed.json")["contrasts"]
    pool = load("data/results/replication/pooled_with_reference.json")
    feat = {v["variant"]: v for v in load("data/results/feature_ablation_report.json")}
    h1 = load("data/results/screening_lambda0/robustness_5seed.json")["contrasts"]["h1_vs_no_graph"]
    sm = {c["contrast"]: c for c in load("data/results/screening_lambda0/second_metric_5seed.json")
          ["metrics"]["pearson"]["contrasts"]}["condition_gated - expression_only"]

    def signed(c, arm):
        return c["mean"] if c["better"] == arm else -c["mean"]

    h2a, h2b = signed(n7["h2a"], "typed_static"), signed(n7["h2b"], "condition_gated")
    surviving = [v["mean"] for v in pool["per_dataset"]["promotion_margin"].values() if v.get("survives_family_wise")]
    ratio = feat["nodegree"]["mean"] / feat["nograph"]["mean"]
    claims = [
        # (phrase in the table, does the artifact support it, what was found)
        ("The gated model is still no better than no graph under Pearson-delta", not sm.get("survives_both"),
         f"Pearson-delta gated - no graph {sm['mean']:+.4f}, survives {sm.get('survives_both')}"),
        ("On our main split, with seven seeds, the typed model is \\emph{worse} than no graph",
         h2a < 0 and n7["h2a"]["survives_family_wise"],
         f"n=7 h2a {h2a:+.4f}, survives {n7['h2a']['survives_family_wise']}"),
        ("the gates repair only part of the damage, not significantly",
         0 < h2b < -h2a and not n7["h2b"]["survives_family_wise"],
         f"n=7 h2b {h2b:+.4f} against h2a {h2a:+.4f}, survives {n7['h2b']['survives_family_wise']}"),
        ("passes correction on three datasets, in opposite directions",
         len(surviving) == 3 and min(surviving) < 0 < max(surviving), f"untyped survivors {surviving}"),
        ("graph-derived features to zero lowers \\textsc{systema}, a drop that survives correction",
         feat["nograph"]["mean"] < 0 and feat["nograph"]["survives"] == "True",
         f"nograph {feat['nograph']['mean']:+.4f}, survives {feat['nograph']['survives']}"),
        ("nearly all of it from three counts", 0.9 <= ratio < 1.0, f"counts-only / all = {ratio:.3f}"),
        ("the gated graph has no detectable effect",
         not h1["survives_family_wise"] and h1["ci_low"] < 0 < h1["ci_high"],
         f"h1 {signed(h1, 'condition_gated'):+.4f} [{h1['ci_low']:+.4f},{h1['ci_high']:+.4f}]"),
        ("``no graph'' is not graph-free", feat["nograph"]["survives"] == "True",
         f"nograph survives {feat['nograph']['survives']}"),
    ]
    for phrase, ok, found in claims:
        if phrase not in body:
            fails.append(f"tab:causes: the cell no longer says '{phrase}'; update this check with it")
        elif not ok:
            fails.append(f"tab:causes: '{phrase}' is not what the artifact says ({found})")
    return fails, (f"partial: {len(claims)} qualitative claims in causes B, C and D re-derived from their "
                   f"artifacts; causes A and E have no JSON of their own (see docstring)")


@table("floats")
def _floats(tex):
    """Every figure and table has a caption and a label and is referenced from the running text, not
    only from another float. A float nobody points to is one a reader may never look at."""
    body = re.sub(r"(?m)^%.*", "", tex)
    envs = list(re.finditer(r"\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}", body, re.S))
    text = re.sub(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", " ", body, flags=re.S)
    fails = []
    for env in envs:
        lab = re.search(r"\\label\{([^}]+)\}", env.group(2))
        if "\\caption{" not in env.group(2):
            fails.append(f"floats: a {env.group(1)} has no caption")
        if lab is None:
            fails.append(f"floats: a {env.group(1)} has no label")
        elif not re.search(r"\\ref\{" + re.escape(lab.group(1)) + r"\}", text):
            fails.append(f"floats: {lab.group(1)} is never referenced from the running text")
    return fails, f"full: {len(envs)} floats, each with a caption, a label and a reference in the text"


def _program_k(dataset: str):
    """K actually used, measured from the built basis rather than from any declared value.

    Amendment 3.1 requires every K != 128 to be labelled wherever it appears. Measuring it from
    program_response.parquet's width means the label is checked against what the lane really ran, not
    against a runner argument nobody recorded."""
    import pandas as pd
    p = ROOT / f"data/intermediate/replication/{dataset}/program_response.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).shape[1] - 1


# Table 3's row names, as the paper writes them, and the per-dataset keys of the pooled artifact.
_REPL_ROWS = {"Zhu (main)": "reference_screen_n7",
              "Replogle genome-wide": "ReplogleWeissman2022_K562_gwps",
              "Replogle RPE1": "ReplogleWeissman2022_rpe1",
              "Replogle essential": "ReplogleWeissman2022_K562_essential",
              "Frangieh": "FrangiehIzar2021_RNA",
              "Tian CRISPRi": "TianKampmann2021_CRISPRi",
              "Norman": "NormanWeissman2019_filtered",
              "Tian CRISPRa": "TianKampmann2021_CRISPRa"}


@table("tab:repl")
def _repl(tex):
    """Table 3, one row per dataset, every column but the cell type re-derived: the typed and untyped
    differences from no graph and their bold (survival within the dataset's pre-registered family) from
    replication/pooled_with_reference.json; the target count from the POST-Amendment-2 DE_stats_v2
    provenance; "switched on" against the direction the on-target check inferred; and the K label,
    measured from the built program basis, which Amendment 3.1 requires wherever a K below 128 appears.

    NOT CHECKED: the cell type, a property of the published dataset rather than of any artifact here,
    and "knocked out" against "knocked down", which the on-target check cannot tell apart (both lower
    the target's own expression). The reference row's target count is the Methods' own count."""
    fails, seen = [], set()
    pd_ = load("data/results/replication/pooled_with_reference.json")["per_dataset"]
    cap = re.search(r"\\caption\{What the graph did on each dataset(.*?)\\label\{tab:repl\}", tex, re.S)
    capt = " ".join(cap.group(1).split()) if cap else ""
    base = ROOT / "data/intermediate/replication"
    for cs in cells_of(tex, "tab:repl"):
        name = cs[0].replace("$^{*}$", "").strip()
        key = _REPL_ROWS.get(name)
        if key is None or len(cs) < 6:
            continue
        seen.add(name)
        # The pooled file stores each mean rounded to five places, and rounding that copy again printed
        # Frangieh's +0.025553 as +0.0255 until 2026-10-28. Read the dataset's own robustness file.
        src = ("data/results/screening_n7_live/robustness_5seed.json" if key == "reference_screen_n7"
               else f"data/results/replication/{key}/robustness_5seed.json")
        own = load(src)["contrasts"]
        for col, contrast, arm in ((cs[4], "h2a", "typed_static"), (cs[5], "promotion_margin", "untyped_gnn")):
            v = pd_[contrast][key]
            c = own[contrast]
            mean = c["mean"] if c["better"] == arm else -c["mean"]
            if abs(mean - v["mean"]) > 1e-5:
                fails.append(f"tab:repl {name} {contrast}: the pooled copy {v['mean']} disagrees with {src}")
            cmp_cell(fails, f"tab:repl {name} {contrast}", col, mean)
            bold = "\\mathbf" in col
            if bold != bool(v["survives_family_wise"]):
                fails.append(f"tab:repl {name} {contrast}: bold={bold} artifact "
                             f"survives={v['survives_family_wise']}")
        n = int(re.sub(r"[^\d]", "", cs[3]) or -1)
        if key == "reference_screen_n7":
            if f"covering ${cs[3].strip('$')}$ targets" not in " ".join(tex.split()):
                fails.append(f"tab:repl {name} targets: {cs[3]} is not the count the Methods state")
            continue
        prov = json.loads((base / f"{key}.DE_stats_v2.provenance.json").read_text())
        if n != prov["n_targets"]:
            fails.append(f"tab:repl {name} targets: paper {n} artifact {prov['n_targets']}")
        qc = json.loads((base / f"{key}.DE_stats_v2.qc.json").read_text())
        if ("switched on" in cs[2]) != (qc["inferred_direction"] == "activation"):
            fails.append(f"tab:repl {name}: the paper says {cs[2]!r}; the on-target check infers "
                         f"{qc['inferred_direction']}")
        k = _program_k(key)
        if k is None:
            fails.append(f"tab:repl {name}: no program basis was built, so K cannot be checked")
        elif (k < 128) != ("^{*}" in cs[0]):
            fails.append(f"tab:repl {name}: K={k} but the row is {'' if '^{*}' in cs[0] else 'not '}marked")
        # [^.;$] stops at the next "$K{=}": with ".*" a caption giving one dataset's K to another could
        # still match by running on into the second label.
        elif k < 128 and not re.search(r"\$K\{=\}" + str(k) + r"\$ for [^.;$]*" + re.escape(name), capt):
            fails.append(f"tab:repl {name}: K={k}, and the caption does not give that K for it")
    # The caption gives each dataset's family size; read them rather than trust the sentence.
    fam = {"reference_screen_n7": load("data/results/screening_n7_live/robustness_5seed.json")["contrasts"]
           ["h2a"]["family_size"]}
    for key in _REPL_ROWS.values():
        f = ROOT / f"data/results/replication/{key}/robustness_5seed.json"
        if f.exists():
            fam[key] = json.loads(f.read_text())["contrasts"]["h2a"]["family_size"]
    four = sorted(k for k, v in fam.items() if v == 4)
    if four != ["FrangiehIzar2021_RNA", "reference_screen_n7"] or sorted(set(fam.values())) != [2, 4] or \
            len(fam) != len(_REPL_ROWS) or \
            "four comparisons on Zhu and Frangieh, two on the other six" not in capt:
        fails.append(f"tab:repl caption: family sizes {fam}")
    if len(seen) != len(_REPL_ROWS):
        fails.append(f"tab:repl: matched {len(seen)} of {len(_REPL_ROWS)} rows ({sorted(seen)})")
        return fails, f"partial: matched {len(seen)} of {len(_REPL_ROWS)} rows"
    return fails, (f"full: {len(seen)} datasets x (typed and untyped difference and bold, targets, direction, "
                   f"K label) vs pooled_with_reference.json and the DE_stats_v2 builds")


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
        if re.match(r"(full|partial)[^0-9]*\b0\b", note, re.I):
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
