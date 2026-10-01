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

THE ARM AND THE REFERENCE ROOT ARE PARAMETERS, and the defaults reproduce Amendment 6's ladder
exactly. Amendment 9 registers a SECOND ladder on `condition_gated`, the arm the paper's headline null
is actually about, and 9.11 requires this parameterisation to land BEFORE its lanes do - so that the
analysis code cannot be shaped by the numbers it will produce. That is the same discipline that makes
``increment_over_zero`` below worth reading: it was committed while three rungs were still unrun.

    PYTHONPATH=src python -m tcell_pipeline.screening.ladder_report \
        --root data/results/c1_ladder --arm condition_gated \
        --reference-root data/results/screening_lambda0 \
        --out data/results/c1_ladder/floor_condition_gated.json

THE REFERENCE ROOT MUST MATCH THE LANES' CONFIGURATION, and for Amendment 9 that is not the default.
Its zero point is read from `screening_lambda0` because those lanes run at `lambda_graph=0`, which is
the only setting under which `condition_gated`'s edge gate stays live (Amendment 9.2). Reading the
zero point from a gate-suppressed root would compare an injected rung against a differently-configured
baseline, which is a confound rather than a zero point.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from tcell_pipeline.config import GATE_DEAD
from tcell_pipeline.screening.multiseed import apply_family_wise, paired_delta_summary
from tcell_pipeline.screening.screening import EXPRESSION_ONLY, PRIMARY_METRIC, UNTYPED_GNN

# The ONLY arm in this project whose edge gate is learned rather than pinned to 1.0. Everything else
# here - untyped_gnn, typed_static, typed_shared, typed_permuted, typed_gcnnorm - fixes the gate by
# construction, so lambda_graph cannot suppress anything and Amendment 3.4's collapse criterion cannot
# bind. Scoping the two Amendment 9.2 checks to this set is what keeps them from crying wolf on
# Amendment 6's landed ladder, which ran at the config default 0.01 quite harmlessly.
LIVE_GATE_ARMS = frozenset({"condition_gated"})

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


def lane_config(ladder_root: str, arm: str, seeds=SEEDS) -> dict:
    """Every lane's recorded ``lambda_graph``, read from the lane's OWN parquet.

    AMENDMENT 9.2 MADE THIS A REPORTED QUANTITY RATHER THAN AN ASSUMPTION, and this is the half of it
    that an artifact can settle. `condition_gated` is the only arm here with a live edge gate, and at
    the config default of 0.01 the gate is annihilated inside epoch 0 - so a ladder accidentally run
    at the default would measure a floor for a model the paper's null is not about, and would look
    exactly like a real result. The lane records what it actually ran at; this reads it back."""
    out = {}
    for d in sorted(Path(ladder_root).glob("*d[0-9][0-9][0-9]")):
        for f in sorted((d / arm).glob("[0-9].parquet")) if (d / arm).is_dir() else []:
            if int(f.stem) not in seeds:
                continue
            row = pd.read_parquet(f).iloc[0].to_dict()
            out[f"{d.name}/{arm}@{f.stem}"] = row.get("lambda_graph")
    return out


def gate_health(ladder_root: str, arm: str, seeds=SEEDS) -> dict:
    """Per-lane minimum gate mean, the other half of Amendment 9.2, read from the LANE'S OWN ARTIFACT.

    THE SOURCE IS `<root>/<arm>/<seed>/logs/stage_a_history.json`, NOT THE LANE LOG. A first version
    of this scraped the runner logs for "gate mean", which is the wording `run_rescreen_lambda0.sh`
    prints - but that is a RUNNER post-processing the history file, and `run_screening`'s own lane log
    contains no per-epoch line at all. Scraping it would have found nothing and reported "unavailable"
    for every lane of a campaign that was in fact perfectly healthy, which is a promise silently
    unkept rather than a check. The history file is written by the trainer, sits inside the results
    root beside the parquet this module already reads, and carries `train.gate_mean` per epoch.

    Reports ``unavailable`` with a reason rather than an empty dict when no history is found: an empty
    result reads as "all healthy" and is exactly the failure this project keeps finding in its harness.
    """
    root = Path(ladder_root)
    if not root.is_dir():
        return {"status": "unavailable", "reason": f"no ladder root at {ladder_root}", "lanes": {}}
    lanes, incomplete, ungated, in_flight = {}, [], [], []
    for d in sorted(root.glob("*d[0-9][0-9][0-9]")):
        for s in seeds:
            h = d / arm / str(s) / "logs" / "stage_a_history.json"
            pq = d / arm / f"{s}.parquet"
            if not h.exists():
                if pq.exists():
                    incomplete.append(f"{d.name}_{arm}_s{s}")   # landed but no history: say so
                continue
            # A HISTORY WITHOUT A PARQUET IS A LANE STILL RUNNING. The trainer appends every epoch, so
            # an in-flight lane has a PARTIAL minimum that can still fall. Counting it would report
            # gate health over lanes that `collect` never counted, so the analysis and the health
            # report would describe different sets. It also misleads a human reading the numbers
            # mid-campaign: on 2026-08-23 I compared a completed lane's gate against a running one's
            # and drew a conclusion from the difference.
            if not pq.exists():
                in_flight.append(f"{d.name}_{arm}_s{s}")
                continue
            try:
                # A NULL gate_mean is not a dead gate, it is an arm with NO LEARNED GATE: the
                # trainer records None wherever the gate is pinned to 1.0 by construction. Treating
                # None as a number makes min() raise; treating it as 0.0 would report every pinned
                # arm as collapsed, which is the same claim as "the graph was switched off" and is
                # false. It is filtered, and a lane with only Nones is reported as ungated below.
                series = [e["train"]["gate_mean"] for e in json.loads(h.read_text())
                          if e.get("train", {}).get("gate_mean") is not None]
            except (json.JSONDecodeError, KeyError, TypeError):
                incomplete.append(f"{d.name}_{arm}_s{s}")
                continue
            if series:
                lanes[f"{d.name}_{arm}_s{s}"] = min(series)
            else:
                ungated.append(f"{d.name}_{arm}_s{s}")
    if not lanes and ungated:
        return {"status": "ungated", "lanes": {}, "ungated_lanes": ungated,
                "reason": f"{len(ungated)} lanes have history but record gate_mean=None throughout: "
                          f"{arm} has no learned gate, so there is nothing for Amendment 3.4 to bind on",
                "history_unreadable": incomplete, "in_flight_lanes": in_flight}
    if not lanes:
        return {"status": "unavailable",
                "reason": f"no stage_a_history.json with train.gate_mean under {ladder_root}/*/{arm}/",
                "lanes": {}, "history_unreadable": incomplete, "in_flight_lanes": in_flight}
    worst = min(lanes.values())
    return {"status": "collapsed" if worst <= GATE_DEAD else "live",
            "min_gate_mean": worst, "gate_dead_threshold": GATE_DEAD,
            "collapsed_lanes": sorted(k for k, v in lanes.items() if v <= GATE_DEAD),
            "history_unreadable": incomplete, "ungated_lanes": ungated,
            "in_flight_lanes": in_flight, "lanes": lanes}


def _delta_of(name: str) -> float | None:
    """d020 -> 0.02, permuted_d400 -> 0.40. The directory name is the record of which rung this is."""
    m = re.search(r"d(\d{3})$", name)
    return int(m.group(1)) / 1000.0 if m else None


def collect(ladder_root: str = LADDER_ROOT, seeds=SEEDS, arm: str = UNTYPED_GNN) -> dict:
    """``arm`` is the GRAPH arm under test; the comparison is always ``expression_only``."""
    rungs = {}
    for d in sorted(Path(ladder_root).glob("*d[0-9][0-9][0-9]")):
        if not d.is_dir():
            continue
        rungs[d.name] = {"path": str(d), "delta": _delta_of(d.name),
                         "permuted": d.name.startswith("permuted"),
                         "better": _metric_by_seed(d, arm, seeds),
                         "worse": _metric_by_seed(d, EXPRESSION_ONLY, seeds)}
    return rungs


def increment_over_zero(rungs: dict, seeds=SEEDS, alpha: float = 0.05,
                        reference_root: str | None = None, arm: str = UNTYPED_GNN) -> dict:
    """POST-HOC, and labelled so wherever it appears. Each rung's gap MINUS the same seed's gap on the
    UN-INJECTED reference lanes.

    WHY IT EXISTS. The pre-registered primary tests each rung's ``untyped - expression_only`` against
    ZERO. On this fold that contrast is already about +0.005 with no injection at all, because a real
    graph benefit exists there (it is the paper's one corrected-significant positive). So a rung can
    clear the primary on the pre-existing benefit alone, and the smallest clearing rung is then a floor
    for "the graph helps", not for "the injected signal was recovered". This subtracts the zero point so
    the remainder is what the INJECTION bought.

    IT IS NOT THE RULE. Amendment 6.7 fixes the primary and this does not replace it; both are reported.
    Written and committed while three rungs were still unrun, so it could not have been shaped by the
    numbers it would produce - which is the only thing that makes a post-hoc analysis worth reading.

    Paired on the SEED: the same seed means the same initialisation and data order, on the same frozen
    fold, so the difference of differences removes that nuisance exactly as the primary does."""
    zero = {}
    root = Path(reference_root if reference_root is not None else REFERENCE_ROOT)
    if root.exists():
        b = _metric_by_seed(root, arm, seeds)
        w = _metric_by_seed(root, EXPRESSION_ONLY, seeds)
        zero = {s: b[s] - w[s] for s in sorted(set(b) & set(w))}
    if not zero:
        return {}
    out = {}
    for name, r in rungs.items():
        gap = {s: r["better"][s] - r["worse"][s]
               for s in sorted(set(r["better"]) & set(r["worse"]))}
        shared = sorted(set(gap) & set(zero))
        if len(shared) < 2:
            continue
        out[name] = paired_delta_summary({s: gap[s] for s in shared},
                                         {s: zero[s] for s in shared}, alpha=alpha, seeds=shared)
        out[name]["delta"] = r["delta"]
        out[name]["permuted"] = r["permuted"]
    return out


def run(ladder_root: str = LADDER_ROOT, out: Path | None = None, seeds=SEEDS,
        alpha: float = 0.05, arm: str = UNTYPED_GNN,
        reference_root: str | None = None, log_dir: str | None = None) -> dict:
    # NONE means "the module default, read NOW". A module global bound as a default argument is
    # frozen at import, which silently defeats monkeypatching it - and the failure is not an error,
    # it is a plausible number computed against the real reference root instead of the fixture.
    reference_root = reference_root if reference_root is not None else REFERENCE_ROOT
    rungs = collect(ladder_root, seeds, arm)

    # AMENDMENT 9.2: gate health and lane configuration are REPORTED, not assumed. Both run BEFORE
    # any contrast is formed, because a collapsed gate makes a lane UNDECIDABLE under Amendment 3.4
    # and an undecidable lane must SHRINK n rather than be counted as evidence about the graph.
    cfg = lane_config(ladder_root, arm, seeds)
    lam = sorted({v for v in cfg.values() if v is not None})
    live_gate = arm in LIVE_GATE_ARMS
    health = (gate_health(log_dir if log_dir is not None else ladder_root, arm, seeds)
              if live_gate else
              {"status": "not_applicable",
               "reason": f"{arm} pins its edge gate to 1.0 by construction, so there is no gate to "
                         f"collapse and Amendment 3.4's criterion cannot bind", "lanes": {}})
    dropped_for_gate = []
    for stem in health.get("collapsed_lanes", []):
        m = re.match(rf"(.+)_{re.escape(arm)}_s(\d+)$", stem)
        if m and m.group(1) in rungs:
            if rungs[m.group(1)]["better"].pop(int(m.group(2)), None) is not None:
                dropped_for_gate.append(stem)

    if not rungs:
        print(f"[ladder] no rungs under {ladder_root} — nothing has landed yet")
        return {"rungs": {}, "floor": None, "status": "no lanes"}

    contrasts = {name: paired_delta_summary(r["better"], r["worse"], alpha=alpha, seeds=seeds)
                 for name, r in rungs.items()}
    m = apply_family_wise(contrasts, alpha)

    print(f"[ladder] ARM = {arm} vs {EXPRESSION_ONLY}; reference root {reference_root}")
    print(f"[ladder] {len(rungs)} conditions, family size {m} (Bonferroni AND Holm, both required)")
    print(f"{'rung':>16} {'delta':>6} {'n':>2} {'mean':>10} {'95% CI':>22} {'bonf':>7} {'holm':>7} {'survives':>9}")
    zero = _zero_point(seeds, alpha, reference_root, arm)
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

    verdict = _verdict(rungs, contrasts, seeds)
    if live_gate and lam and lam != [0.0]:
        verdict["notes"].insert(0, f"*** {arm} HAS A LIVE GATE AND THESE LANES DID NOT ALL RUN AT "
                                   f"lambda_graph=0: observed {lam}. At 0.01 the gate is annihilated "
                                   f"inside epoch 0, so this is not a floor for {arm}, it is a "
                                   f"measurement of a different model (Amendment 9.2). ***")
    elif lam and lam != [0.0]:
        verdict["notes"].append(f"lambda_graph was {lam}, not 0. Harmless for {arm}, which pins its "
                                f"gate to 1.0; recorded so the configuration is on the record.")
    if dropped_for_gate:
        verdict["notes"].insert(0, f"*** GATE COLLAPSED, n SHRUNK: {dropped_for_gate} dropped as "
                                   f"UNDECIDABLE under Amendment 3.4, never counted as evidence. ***")
    if live_gate and health.get("status") == "unavailable":
        verdict["notes"].append(f"gate health UNCHECKED: {health['reason']}. Amendment 9.2 requires "
                                f"it to be reported; this is a NAMED gap, not a pass.")
    for line in verdict["notes"]:
        print(f"[ladder] {line}")

    # POST-HOC, never the rule (see increment_over_zero's docstring). Printed under its own heading so
    # it cannot be mistaken for the pre-registered primary above.
    incr = increment_over_zero(rungs, seeds, alpha, reference_root, arm)
    if incr:
        print("\n[ladder] POST-HOC: each rung's gap MINUS the same seed's gap with NO injection.")
        print("[ladder] Not the pre-registered primary; committed before the last rungs ran.")
        print(f"{'rung':>16} {'delta':>6} {'n':>2} {'increment':>11} {'95% CI':>24} {'p':>8}")
        for name in sorted(incr, key=lambda k: (incr[k]["permuted"], incr[k]["delta"] or 0)):
            c = incr[name]
            ci = "     -" if c["ci_low"] is None else f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
            mean = "    -" if c["mean"] is None else f"{c['mean']:+.4f}"
            pv = "  -" if c["p_value"] is None else f"{c['p_value']:.4f}"
            print(f"{name:>16} {c['delta']:>6.3f} {c['n']:>2} {mean:>11} {ci:>24} {pv:>8}")

    print(f"[ladder] lambda_graph observed across lanes: {lam or 'not recorded'}  (gate: [0.0])")
    print(f"[ladder] gate health: {health.get('status')}"
          + (f", min gate mean {health['min_gate_mean']:.4g} over {len(health['lanes'])} lanes"
             if health.get("lanes") else f" ({health.get('reason', '')})"))
    report = {"arm": arm, "comparison_arm": EXPRESSION_ONLY, "reference_root": reference_root,
              "lane_lambda_graph": cfg, "lambda_graph_ok": (lam == [0.0]) if lam else None,
              "gate_health": health, "dropped_for_gate_collapse": dropped_for_gate,
              "ladder_root": str(ladder_root),
              "family_size": m, "alpha": alpha, "seeds": list(seeds), "zero_point": zero,
              "post_hoc_increment_over_zero": incr,
              "contrasts": contrasts, "rungs": {k: {kk: vv for kk, vv in v.items() if kk != "path"}
                                                for k, v in rungs.items()}, **verdict}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=float))
        print(f"[ladder] wrote {out}")
    return report


def _zero_point(seeds, alpha: float, reference_root: str | None = None,
                arm: str = UNTYPED_GNN) -> dict | None:
    """delta=0 from the LANDED reference lanes. Not re-run, and not a family member."""
    root = Path(reference_root if reference_root is not None else REFERENCE_ROOT)
    if not root.exists():
        return None
    s = paired_delta_summary(_metric_by_seed(root, arm, seeds),
                             _metric_by_seed(root, EXPRESSION_ONLY, seeds), alpha=alpha, seeds=seeds)
    return s if s["n"] else None


def _verdict(rungs: dict, contrasts: dict, seeds=SEEDS) -> dict:
    notes, floor = [], None
    real = sorted((n for n in rungs if not rungs[n]["permuted"]), key=lambda k: rungs[k]["delta"])
    control = [n for n in rungs if rungs[n]["permuted"]]

    incomplete = [n for n in rungs if contrasts[n]["n"] < len(seeds)]
    if incomplete:
        notes.append(f"INCOMPLETE at n<{len(seeds)}: {incomplete} — read these as preliminary, with "
                     f"their n (rail 5)")

    # A MISSING CONTROL IS NOT A PASSED CONTROL. `any()` over an empty list is False, so a ladder
    # whose permuted rung never landed would sail past the veto below and name a floor. Amendment
    # 6.7/9.6 make that veto absolute and prior to every other reading, and it cannot be applied to a
    # rung that does not exist. This bites precisely when a campaign is stopped early - the case
    # Amendment 9.9 plans for - so it is checked before anything else.
    if not control:
        notes.append("*** NO PERMUTED CONTROL RUNG IS PRESENT. No floor is reported: the control's "
                     "veto is absolute (Amendment 6.7) and cannot be applied to a rung that has not "
                     "landed. An absent control is not a passed control. ***")
        return {"floor": None, "floor_status": "control_missing", "notes": notes,
                "control_clears": None}

    # A CONTROL THAT EXISTS BUT CANNOT BE TESTED IS ALSO NOT A PASSED CONTROL. The check above only
    # asks whether the rung is present. A control whose lanes were DROPPED - by gate collapse under
    # Amendment 3.4, or by failing to complete - leaves the rung in place with n=0, whose mean is
    # None, and `(mean or 0) > 0` is then False, so the veto passes silently and a floor is named.
    # This is not hypothetical for THIS ladder: the permuted rung is exactly where the gate sits
    # closest to the dead threshold, because a scrambled neighbourhood gives the gate nothing worth
    # keeping open. Measured on seed 0: 0.024 on the control against 0.57-0.70 on the real rungs.
    control_testable = [n for n in control if contrasts[n]["n"] >= 2]
    if not control_testable:
        ns = {n: contrasts[n]["n"] for n in control}
        notes.append(f"*** THE PERMUTED CONTROL IS PRESENT BUT UNTESTABLE (n={ns}). No floor is "
                     f"reported: the veto is absolute (Amendment 6.7) and a control that cannot be "
                     f"tested has not passed. Check whether its lanes were dropped for gate collapse "
                     f"under Amendment 3.4 - the control is where that is most likely. ***")
        return {"floor": None, "floor_status": "control_untestable", "notes": notes,
                "control_clears": None}

    # A RUNG WITH n<2 IS UNTESTED, NOT CLEARED. The floor is the smallest rung that clears AND is
    # cleared by every LARGER rung, so an untested larger rung leaves every smaller one unreadable.
    # Without this, an EMPTY ladder reports "the floor is above the largest rung tested, which is
    # itself a result" - a manufactured negative from zero data. Reporting at a reduced n is still
    # allowed and expected (Amendment 9.9), but n>=2 is the floor of what forms a paired contrast.
    untested = [n for n in real if contrasts[n]["n"] < 2]
    if untested:
        notes.append(f"*** LADDER INCOMPLETE: {untested} have n<2 and cannot form a paired contrast. "
                     f"No floor is reported in either direction. An untested rung is UNKNOWN, not "
                     f"cleared, and the floor rule reads every larger rung. ***")
        return {"floor": None, "floor_status": "incomplete_ladder", "notes": notes,
                "control_clears": None}

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
        # "NOTHING CLEARED" AND "THE INSTRUMENT IS BLIND" ARE DIFFERENT CLAIMS, and only a COMPLETE
        # ladder can carry the second. At the pre-registered n the above_ladder reading is a result:
        # the pipeline was given an injected signal that large and did not recover it. Below that n it
        # is a statement about the budget, not the instrument - at n=2 the intervals here span +-0.10,
        # so nothing could plausibly clear whatever the truth is, and asserting that the pipeline
        # "cannot see" a signal would be a negative manufactured from missing seeds. Amendment 9.9
        # expects a stopped campaign to be REPORTED at reduced n; it does not license this sentence.
        if incomplete:
            notes.append(f"No rung clears both corrections AT THIS n, and that does NOT establish a "
                         f"floor above the ladder. {len(incomplete)} rung(s) are short of the "
                         f"pre-registered {len(seeds)} seeds, so a null here is a statement about the "
                         f"budget rather than about the instrument. Preliminary, with its n (rail 5).")
            return {"floor": None, "floor_status": "above_ladder_preliminary", "notes": notes,
                    "control_clears": False}
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
    ap.add_argument("--arm", default=UNTYPED_GNN,
                    help="the GRAPH arm under test; the comparison is always expression_only. "
                         "Amendment 6's ladder is untyped_gnn, Amendment 9's is condition_gated")
    ap.add_argument("--reference-root", default=REFERENCE_ROOT,
                    help="where the delta=0 zero point is read from. It MUST match the lanes' "
                         "configuration: Amendment 9's lanes run at lambda_graph=0, so its zero "
                         "point comes from screening_lambda0, not from the default root")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--log-dir", default=None,
                    help="override where per-lane stage_a_history.json files are looked for, for the "
                         "Amendment 9.2 gate-health report. Defaults to --root itself, which is where "
                         "the trainer writes them")
    a = ap.parse_args()
    run(a.root, Path(a.out) if a.out else None, tuple(a.seeds), arm=a.arm,
        reference_root=a.reference_root, log_dir=a.log_dir)
