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


_REPL_PROV = {"FrangiehIzar2021": "FrangiehIzar2021", "NormanWeissman2019": "NormanWeissman2019",
              "PapalexiSatija2021": "PapalexiSatija2021", "ShifrutMarson2018": "ShifrutMarson2018",
              "DatlingerBock2017": "DatlingerBock2017"}


def _prereg_replicate_units() -> dict:
    """{dataset: (replicate unit, column)} from the FROZEN pre-registration's own table.

    DO NOT USE THE PROVENANCE'S `replicate_unit` FIELD FOR THIS. It is misnamed: adapter.py emits
    `spec.replicate_col` under that key, so it holds a COLUMN NAME, not a unit. The two differ for
    exactly the datasets that matter - Shifrut's unit is the donor and its column is `replicate`,
    while Datlinger's column is ALSO `replicate` for a completely different unit - so checking the
    paper against that field reports the correct table as wrong. It did, on this check's first run.

    The paper writes whichever of the two reads better per row (the unit for Shifrut, the column
    for Norman, where they coincide for Frangieh and Replogle). Both are pre-registered and neither
    is wrong, so a cell is accepted if it matches EITHER. That still catches a value that is
    neither, which is the failure worth catching."""
    txt = (ROOT / "docs/replication-prereg.md").read_text()
    out = {}
    for line in txt.splitlines():
        cs = [c.strip() for c in line.split("|")]
        if len(cs) > 4 and cs[3].startswith("`") and cs[3].endswith("`"):
            key = re.sub(r"[^A-Za-z0-9]", "", cs[1].split("(")[0])
            out[key] = (cs[2], cs[3].strip("`"))
    return out


@table("tab:repl")
def _repl(tex):
    """Three of five columns are re-derivable per dataset: targets kept (numerator) and condition
    count from the build's own provenance JSON, replicate unit from the frozen pre-registration.

    NOT CHECKED, and why: the "targets kept" DENOMINATOR is the count present in the source h5ad
    before any rule was applied, which the provenance does not record; and "Verdict" is a
    pre-registered judgement, not a measurement. ReplogleWeissman2022's dropped row has no
    top-level provenance file at all - only the three per-subset builds that replaced it - so that
    row is reported here rather than skipped in silence."""
    fails, rows_seen, missing, unchecked_unit = [], 0, [], []
    for cs in cells_of(tex, "tab:repl"):
        name = re.sub(r"[^A-Za-z0-9]", "", cs[0])
        if name not in _REPL_PROV:
            if name and name[0].isupper() and "Dataset" not in name:
                missing.append(name)
            continue
        try:
            prov = load(f"data/intermediate/replication/{_REPL_PROV[name]}.provenance.json")
        except Missing as e:
            missing.append(f"{name} ({e})")
            continue
        rows_seen += 1
        kept = re.search(r"\$(\d[\d,{}]*)\s*/", cs[1].replace("{,}", ""))
        if kept and int(kept.group(1).replace(",", "")) != prov["n_targets"]:
            fails.append(f"tab:repl {name} targets kept: paper {kept.group(1)} artifact "
                         f"{prov['n_targets']}")
        nc = re.search(r"\$(\d+)\$", cs[2])
        if nc and int(nc.group(1)) != len(prov["conditions"]):
            fails.append(f"tab:repl {name} conditions: paper {nc.group(1)} artifact "
                         f"{len(prov['conditions'])}")
        unit, col = _prereg_replicate_units().get(name, (None, None))
        if unit is None:
            unchecked_unit.append(name)
        elif not (unit.lower() in cs[3].lower() or cs[3].lower() in unit.lower()
                  or col.lower() in cs[3].lower()):
            fails.append(f"tab:repl {name} replicate unit: paper {cs[3]!r}, pre-registration "
                         f"{unit!r} (column {col!r})")
    note = (f"partial: {rows_seen} rows x (targets kept, conditions) vs the build provenance and "
            f"replicate unit vs the frozen prereg; denominator and Verdict are not measurements")
    if unchecked_unit:
        note += f"; replicate unit UNCHECKED for {unchecked_unit} (absent from the prereg table)"
    if missing:
        note += f"; NO PROVENANCE FILE for {missing}, so those rows are unchecked"
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
        lab = cs[0].replace("$", "").replace("{,}", "").replace("--", "-")
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
