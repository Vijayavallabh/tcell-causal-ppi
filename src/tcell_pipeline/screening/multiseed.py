"""Multi-seed paired robustness aggregation (feat-011 follow-on; ``config.N_FINAL_SEEDS``).

feat-011 screened the §10.6 family on ONE seed; H2a (typed_static − expression_only) and the promotion
margin (untyped_gnn − expression_only) both landed inside the 0.01 noise band, so "the graph does not
help" is a single-seed coin toss. This re-runs the family under N paired seeds on the SAME frozen fold
and reports each contrast as a mean Δsystema with a paired-t CI — the negative gets error bars, or is
overturned.

PAIRED design: for each seed ``s`` both arms of a contrast were trained under the SAME seed (init + data
order) on the SAME frozen split, so the per-seed difference ``d_s = metric(better,s) − metric(worse,s)``
removes the shared seed/fold nuisance. The statistic is a one-sample t on ``{d_s}`` against 0 — the
paired t-test. ``n<2`` has no CI (a single seed is not a paired result). A non-finite or missing arm
drops that seed from the contrast, LOUDLY, and shrinks n. Honest frame: a CI that crosses zero is
"indistinguishable at this budget", NOT support for the better arm.

This module reads screening parquets + the registry only and writes ``robustness_5seed.{json,md}``. It
NEVER touches ``promoted.json`` — the frozen H1 (condition_gated seed 0) stays the confirmatory model.
"""
from __future__ import annotations

import math
from pathlib import Path

from tcell_pipeline import config
from tcell_pipeline.screening.screening import (
    CONDITION_GATED,
    EXPRESSION_ONLY,
    PRIMARY_METRIC,
    TYPED_STATIC,
    UNTYPED_GNN,
    _config_statuses,
    _finite_or_none,
    _is_stale,
)

FROZEN_SPLIT = "blocked_target_ood"
FAMILY = (EXPRESSION_ONLY, UNTYPED_GNN, TYPED_STATIC, CONDITION_GATED)
# pre-registered contrasts on systema + the promotion margin the seed-0 fold turned on (untyped − expr).
# h1_vs_no_graph is the headline pair the campaign exists to settle: it was ORIGINALLY MISSING here and
# the claim was read off two marginal per-config means instead — an undecidable pair published as decided
# (xhigh review, 2026-07-20). Every claim about a pair must come from a contrast in this tuple.
CONTRASTS = (
    ("h2a", TYPED_STATIC, EXPRESSION_ONLY),          # does typed static beat expression-only?
    ("h2b", CONDITION_GATED, TYPED_STATIC),          # does condition gating beat typed static?
    ("promotion_margin", UNTYPED_GNN, EXPRESSION_ONLY),  # does ANY graph beat no-graph?
    ("h1_vs_no_graph", CONDITION_GATED, EXPRESSION_ONLY),  # does the FROZEN H1 beat no-graph?
)


def _finite(x) -> bool:
    """Reject NaN/Inf/None/bool/str; accept python AND numpy real scalars — parquet reads come back as
    numpy.float64, and a ``_finite`` that rejected them would silently drop every seed."""
    import numpy as np
    return isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) and math.isfinite(x)


def _verdict(mean, ci_excludes_zero, n: int, *, fold_comparable: bool = True) -> str:
    if not fold_comparable:
        return (f"NOT comparable (n={n}): the seeds were not shown to be scored on one frozen fold, so "
                f"the paired assumption is void — no verdict is emitted")
    if ci_excludes_zero is None:
        return (f"degenerate at n={n}: zero variance across seeds (identical deltas) — there is no "
                f"spread to infer from, and this is the signature of seeds that did NOT propagate, "
                f"NOT evidence of an effect")
    if not ci_excludes_zero:
        return (f"indistinguishable at n={n}: the paired CI crosses zero — underpowered / no effect at "
                f"this budget, NOT support for the better arm")
    return f"CI excludes zero (n={n}) — favors the {'better' if mean > 0 else 'worse'} arm by Δ={mean:+.4f}"


def apply_family_wise(contrasts: dict, alpha: float) -> int:
    """Multiplicity control over the SIMULTANEOUS contrasts, in place; returns the family size.

    Both Bonferroni and Holm adjusted p are recorded, and ``survives_family_wise`` requires BOTH — the
    conservative call. Reporting a single method chosen after seeing which one rescues a claim is
    exactly the look-elsewhere effect ``reproducibility/fallacy_scan.py`` exists to catch, so the method
    cannot be shopped here: the numbers for both are always on the record."""
    testable = [(k, c) for k, c in contrasts.items() if c.get("p_value") is not None]
    m = len(testable)
    for c in contrasts.values():
        c["family_size"] = m
        c.setdefault("p_bonferroni", None)
        c.setdefault("p_holm", None)
        c.setdefault("survives_family_wise", None)
    if not m:
        return 0
    for _, c in testable:
        c["p_bonferroni"] = min(1.0, c["p_value"] * m)
    running = 0.0                                     # Holm step-down, monotonic in ascending p
    for i, (_, c) in enumerate(sorted(testable, key=lambda kc: kc[1]["p_value"])):
        running = max(running, min(1.0, c["p_value"] * (m - i)))
        c["p_holm"] = running
    for _, c in testable:
        c["survives_family_wise"] = bool(c["p_holm"] <= alpha and c["p_bonferroni"] <= alpha)
    return m


# Back-compat alias. This began as a private helper and is imported under its private name by
# training/freeze_gate.py — a different session's file, currently FROZEN, so its call sites cannot be
# updated in the same change. A rename is only safe when every caller moves with it; here one cannot, and
# a bare rename would raise ImportError at that module's import time and take its 26 tests down with it.
# Public name is canonical; drop this alias once freeze_gate.py switches over.
_apply_family_wise = apply_family_wise


def paired_delta_summary(better_by_seed, worse_by_seed, *, alpha: float = 0.05, seeds=None) -> dict:
    """One-sample (paired) t on ``d_s = better_s − worse_s`` against 0. Never raises, never NaN-poisons:
    a seed missing from either arm, or non-finite in either, is DROPPED (named in ``dropped``) and
    shrinks n. ``n<2`` -> mean only, no CI (a single seed is not a paired result). Honest verdict: a CI
    crossing zero is "indistinguishable at this budget", not support for the better arm.

    ``seeds`` is the REQUESTED seed set. Pass it: without it the loop only visits the union of the two
    arms, so a seed missing from BOTH (an OOMed or wholly-stale seed) silently vanishes — n shrinks with
    an empty ``dropped`` list and a 4-seed result reads as the intended 5-seed design."""
    universe = sorted(set(better_by_seed) | set(worse_by_seed)) if seeds is None else sorted(set(seeds))
    deltas, used, dropped = [], [], []
    for s in universe:
        b, w = better_by_seed.get(s), worse_by_seed.get(s)
        if _finite(b) and _finite(w):
            deltas.append(float(b) - float(w))
            used.append(s)
        else:
            reasons = ([f"better={b!r}"] if not _finite(b) else []) + \
                      ([f"worse={w!r}"] if not _finite(w) else [])
            dropped.append({"seed": s, "reason": ", ".join(reasons)})
    n = len(deltas)
    out = {"n": n, "seeds_used": used, "dropped": dropped, "deltas": deltas, "alpha": alpha,
           "mean": None, "sd": None, "se": None, "t": None, "p_value": None,
           "ci_low": None, "ci_high": None, "ci_excludes_zero": None, "verdict": None}
    if n == 0:
        out["verdict"] = "no data — every seed dropped from this contrast"
        return out
    mean = sum(deltas) / n
    out["mean"] = mean
    if n == 1:
        out["verdict"] = f"n=1 (seed {used[0]}) — a single seed is not a paired result; no CI"
        return out

    import numpy as np
    from scipy import stats
    d = np.asarray(deltas, dtype=float)
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    out["sd"], out["se"] = sd, se
    if se == 0.0:
        # Zero variance is UNDECIDABLE, not maximally significant. Publishing p=0.0 / "CI excludes zero"
        # here would report the one condition that proves the seeds carry no information (identical
        # deltas -> the seed never propagated, or one parquet backs every seed) as the strongest
        # possible evidence. Leave p/CI/ci_excludes_zero as None and say so.
        out["verdict"] = _verdict(mean, None, n)
        return out
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    t = mean / se
    out["t"] = t
    out["p_value"] = float(2 * stats.t.sf(abs(t), df=n - 1))
    out["ci_low"] = mean - tcrit * se
    out["ci_high"] = mean + tcrit * se
    out["ci_excludes_zero"] = bool(out["ci_low"] > 0 or out["ci_high"] < 0)
    out["verdict"] = _verdict(mean, out["ci_excludes_zero"], n)
    return out


# --------------------------------------------------------------------------------------------------
# Reading the per-(config, seed) rows through the SAME freshness fence as merge/promote
# --------------------------------------------------------------------------------------------------
def _read_seed_metrics(names, seed: int, *, screening_root: Path, registry_path: Path | None,
                       primary: str = PRIMARY_METRIC) -> tuple[dict, dict, dict]:
    """``({name: metric}, {name: status}, {name: run_meta})`` for this seed's fresh+completed+finite rows.

    Freshness is the SAME fence as merge/promote (``_is_stale``): a config whose latest registry run is
    not a fresh ``completed`` is ``stale`` even if a parquet is on disk. A completed-but-non-finite
    primary is ``non_finite``, never silently used. ``registry_path=None`` defers to parquet presence."""
    import pandas as pd
    statuses = _config_statuses(registry_path, seed) if registry_path is not None else None
    metrics, status, meta = {}, {}, {}
    for name in names:
        path = Path(screening_root) / name / f"{seed}.parquet"
        if _is_stale(name, statuses):
            status[name] = "stale"
            continue
        if not path.exists():
            status[name] = "missing"
            continue
        row = pd.read_parquet(path).iloc[0].to_dict()
        if row.get("status") != "completed" or primary not in row:
            status[name] = "incomplete"
            continue
        val = row[primary]
        if not _finite(val):
            status[name] = "non_finite"
            continue
        metrics[name] = float(val)
        status[name] = "ok"
        meta[name] = {"gpu_hours": row.get("gpu_hours"), "epochs_run": row.get("epochs_run"),
                      "n_epochs": row.get("n_epochs"), "mtime": path.stat().st_mtime,
                      "n_train": row.get("n_train"), "n_val": row.get("n_val")}
    return metrics, status, meta


def _splits_by_config_seed(names, seeds, registry_path: Path | None) -> tuple[set, bool]:
    """``(distinct split values, registry_evidence)`` for these (name, seed) COMPLETED runs.

    The second element matters as much as the first: ``load_registry`` degrades a missing, truncated or
    null registry to ``[]``, so an EMPTY split set means "no evidence", never "one fold". Reporting the
    empty case as a pass is the provenance-is-not-comparability failure inverted into a false all-clear."""
    if registry_path is None:
        return set(), False
    from tcell_pipeline.screening.experiment_registry import load_registry
    want_names, want_seeds = set(names), {int(s) for s in seeds}
    matched = [r for r in load_registry(registry_path)
               if r.get("config_id") in want_names and int(r.get("seed", 0)) in want_seeds
               and r.get("status") == "completed"]
    return {r.get("split") for r in matched}, bool(matched)


def _mean_ci(values_by_seed, *, alpha: float = 0.05) -> dict:
    """A config's systema mean and paired-t CI across the seeds it completed on (one-sample t vs a 0
    baseline reuses ``paired_delta_summary``'s guarded core, so n<2 / non-finite are handled once)."""
    r = paired_delta_summary(values_by_seed, {s: 0.0 for s in values_by_seed}, alpha=alpha)
    return {"n": r["n"], "mean": r["mean"], "sd": r["sd"], "se": r["se"],
            "ci_low": r["ci_low"], "ci_high": r["ci_high"], "seeds": r["seeds_used"]}


def aggregate_seeds(seeds, *, names=FAMILY, screening_root: Path = config.SCREENING_ROOT,
                    registry_path: Path | None = config.REGISTRY_PATH,
                    primary: str = PRIMARY_METRIC, alpha: float = 0.05) -> dict:
    """Paired H2a/H2b (+ the promotion margin) across ``seeds`` on ``primary``, plus a per-config
    mean±CI ranking. Reads each (name, seed) parquet through the freshness fence, forms
    ``{name: {seed: metric}}``, and runs ``paired_delta_summary`` per contrast. Every non-ok (name,
    seed) is surfaced in ``coverage`` and named in each affected contrast's ``dropped`` — a dropped seed
    shrinks n, it is never silently ignored."""
    seeds = [int(s) for s in seeds]      # str / numpy seed keys resolve the parquet path but defeat
    by_config = {n: {} for n in names}   # both int-keyed guards, yielding a fully-numbered unguarded run
    coverage = {}                        # seed -> {name: status}
    run_meta = {}                        # "name@seed" -> run meta
    fold_sizes = set()                   # observed (n_train, n_val) — the REAL fold signal
    for s in seeds:
        m, st, mt = _read_seed_metrics(names, s, screening_root=screening_root,
                                       registry_path=registry_path, primary=primary)
        coverage[s] = st
        for n, v in m.items():
            by_config[n][s] = v
        for n, d in mt.items():
            run_meta[f"{n}@{s}"] = d
            if d.get("n_train") is not None and d.get("n_val") is not None:
                fold_sizes.add((int(d["n_train"]), int(d["n_val"])))

    # Fold identity. The registry `split` label is filled from cfg.get("split", "blocked_target_ood") by
    # screen_config while nested_family_configs never sets it, so it is a hardcoded literal that can only
    # ever CONFIRM and never refute (a --n-max capped run still reports blocked_target_ood). The recorded
    # row sizes are the signal that actually catches a capped or redrawn fold. Absence of BOTH is
    # UNKNOWN (None) — never a pass.
    splits, registry_evidence = _splits_by_config_seed(names, seeds, registry_path)
    sizes_consistent = (len(fold_sizes) == 1) if fold_sizes else None
    splits_ok = (bool(splits) and splits <= {FROZEN_SPLIT}) if registry_evidence else None
    if sizes_consistent is False or splits_ok is False:
        single_fold = False
    elif sizes_consistent is True or splits_ok is True:
        single_fold = True
    else:
        single_fold = None
    fold_comparable = single_fold is True

    contrasts, skipped = {}, []
    for key, better, worse in CONTRASTS:
        if better not in by_config or worse not in by_config:
            skipped.append(key)      # recorded, never silent: "not computed" != "not significant"
            continue
        c = paired_delta_summary(by_config[better], by_config[worse], alpha=alpha, seeds=seeds)
        c.update({"better": better, "worse": worse, "fold_comparable": fold_comparable})
        if not fold_comparable:      # qualify like summarize_vs_h1's fold gate, don't publish bare CIs
            c["verdict"] = _verdict(c["mean"], c["ci_excludes_zero"], c["n"], fold_comparable=False)
        contrasts[key] = c
    family_size = apply_family_wise(contrasts, alpha)

    per_config = {n: _mean_ci(by_config[n], alpha=alpha) for n in names}
    covered = [set(by_config[n]) for n in names if by_config[n]]
    common = sorted(set.intersection(*covered)) if covered else []
    balanced = bool(covered) and all(set(by_config[n]) == set(common) for n in names if by_config[n])
    ranked = sorted(per_config.items(),
                    key=lambda kv: (-(kv[1]["mean"] if kv[1]["mean"] is not None else -math.inf), kv[0]))
    return {"seeds": seeds, "primary": primary, "alpha": alpha, "family": list(names),
            "coverage": coverage, "run_meta": run_meta,
            "fold": {"expected_split": FROZEN_SPLIT, "observed_splits": sorted(s for s in splits if s),
                     "registry_evidence": registry_evidence,
                     "observed_fold_sizes": sorted(fold_sizes),
                     "fold_sizes_consistent": sizes_consistent,
                     "single_frozen_fold": single_fold},
            "contrasts": contrasts, "contrasts_skipped": skipped, "family_size": family_size,
            "per_config": per_config, "common_seeds": common, "balanced": balanced,
            "ranking": [n for n, _ in ranked]}


def _exit_code(agg: dict) -> int:
    """Non-zero when the report is not a clean, comparable result.

    ``main()`` previously returned 0 unconditionally — even straight after printing FOLD MISMATCH — so an
    unattended campaign script, or any CI/init.sh gate keyed on exit status, recorded a green run while
    the JSON held CIs the code itself had just declared incomparable."""
    incomplete = any(st != "ok" for cov in agg["coverage"].values() for st in cov.values())
    return 1 if (incomplete or agg["fold"]["single_frozen_fold"] is not True
                 or agg["contrasts_skipped"] or not agg["balanced"]) else 0


# --------------------------------------------------------------------------------------------------
# Report (separate from promoted.json)
# --------------------------------------------------------------------------------------------------
def _render_md(agg: dict) -> str:
    fold = agg["fold"]
    L = [f"# 5-seed robustness — paired {agg['primary']} across seeds {agg['seeds']}", "",
         "_Separate from `promoted.json`: the frozen H1 (condition_gated seed 0) is untouched._", "",
         f"**Fold:** expected `{fold['expected_split']}`, observed splits "
         f"{fold['observed_splits'] or '(none)'} (registry evidence: {fold['registry_evidence']}), "
         f"observed fold sizes {fold['observed_fold_sizes'] or '(none recorded)'} "
         f"(consistent: {fold['fold_sizes_consistent']}) — **single frozen fold: "
         f"{fold['single_frozen_fold']}** (`None` = no evidence either way, NOT a pass)", "",
         f"**Multiplicity:** {agg['family_size']} simultaneous contrasts at alpha={agg['alpha']}; "
         f"`survives_family_wise` requires BOTH Bonferroni and Holm (the conservative call, so the "
         f"correction method cannot be chosen after seeing which one rescues a claim).", "",
         "## Paired contrasts", "",
         "| contrast | better − worse | n | mean Δsystema | 95% CI | raw p | Bonferroni | Holm | "
         "survives FWER | verdict |",
         "|---|---|---:|---:|---|---:|---:|---:|---|---|"]
    for key, c in agg["contrasts"].items():
        ci = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]" if c["ci_low"] is not None else "n/a"
        mean = f"{c['mean']:+.4f}" if c["mean"] is not None else "n/a"
        pr = f"{c['p_value']:.4f}" if c.get("p_value") is not None else "n/a"
        pb = f"{c['p_bonferroni']:.4f}" if c.get("p_bonferroni") is not None else "n/a"
        ph = f"{c['p_holm']:.4f}" if c.get("p_holm") is not None else "n/a"
        L.append(f"| {key} | {c['better']} − {c['worse']} | {c['n']} | {mean} | {ci} | {pr} | {pb} | "
                 f"{ph} | {c.get('survives_family_wise')} | {c['verdict']} |")
    if agg["contrasts_skipped"]:
        L += ["", f"**Contrasts NOT computed** (a member was outside the requested family — this is "
                  f"'not computed', NOT 'not significant'): {agg['contrasts_skipped']}"]
    L += ["", "## Per-config systema (mean ± 95% CI)", "",
          "| rank | config | n | mean systema | 95% CI |", "|---:|---|---:|---:|---|"]
    for i, n in enumerate(agg["ranking"], 1):
        pc = agg["per_config"][n]
        ci = f"[{pc['ci_low']:+.4f}, {pc['ci_high']:+.4f}]" if pc["ci_low"] is not None else "n/a"
        mean = f"{pc['mean']:+.4f}" if pc["mean"] is not None else "n/a"
        L.append(f"| {i} | {n} | {pc['n']} | {mean} | {ci} |")
    L += ["", f"_Ranking bases: common seeds {agg['common_seeds']}, balanced={agg['balanced']}. Each "
              f"per-config mean is over that config's OWN completed seeds, so when `balanced` is False "
              f"the ranking compares different seed bases — read the paired contrasts above instead._"]
    bad = {s: {n: st for n, st in cov.items() if st != "ok"} for s, cov in agg["coverage"].items()}
    bad = {s: v for s, v in bad.items() if v}
    if bad:
        L += ["", f"**Incomplete coverage (seed: {{config: status}}):** {bad}"]
    return "\n".join(L) + "\n"


def write_robustness_report(agg: dict, *, out_dir: Path = config.SCREENING_ROOT) -> dict:
    """Write ``robustness_5seed.json`` (+ ``.md``) NEXT TO promoted.json without touching it. JSON is
    dumped ``ensure_ascii`` default + ``indent=2`` and non-finite floats sanitized to None
    (``allow_nan=False`` as the loud backstop) so the deliverable is always valid JSON."""
    import json
    out_dir = Path(out_dir)
    js = out_dir / "robustness_5seed.json"
    config.write_text_atomic(json.dumps(_finite_or_none(agg), indent=2, default=float, allow_nan=False), js)
    md = out_dir / "robustness_5seed.md"
    config.write_text_atomic(_render_md(agg), md)
    return {"json": str(js), "md": str(md)}


def _print_agg(agg: dict) -> None:
    print(f"[robust] paired {agg['primary']} across seeds {agg['seeds']} "
          f"(fold single={agg['fold']['single_frozen_fold']} splits={agg['fold']['observed_splits']})")
    for key, c in agg["contrasts"].items():
        ci = f"[{c['ci_low']:+.4f},{c['ci_high']:+.4f}]" if c["ci_low"] is not None else "n/a"
        mean = f"{c['mean']:+.4f}" if c["mean"] is not None else "n/a"
        pr = f"{c['p_value']:.4f}" if c.get("p_value") is not None else "n/a"
        ph = f"{c['p_holm']:.4f}" if c.get("p_holm") is not None else "n/a"
        pb = f"{c['p_bonferroni']:.4f}" if c.get("p_bonferroni") is not None else "n/a"
        print(f"[robust] {key:16s} {c['better']} - {c['worse']}: n={c['n']} mean={mean} CI={ci} "
              f"p={pr} bonf={pb} holm={ph} fwer={c.get('survives_family_wise')} :: {c['verdict']}")
    if agg["contrasts_skipped"]:
        print(f"[robust] contrasts NOT computed (member outside the family): {agg['contrasts_skipped']}")
    print("[robust] per-config systema (mean ± CI):")
    for i, n in enumerate(agg["ranking"], 1):
        pc = agg["per_config"][n]
        ci = f"[{pc['ci_low']:+.4f},{pc['ci_high']:+.4f}]" if pc["ci_low"] is not None else "n/a"
        mean = f"{pc['mean']:+.4f}" if pc["mean"] is not None else "n/a"
        print(f"[robust]   {i}. {n:16s} n={pc['n']} mean={mean} CI={ci}")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate the multi-seed paired robustness campaign "
                                             "(feat-011 follow-on) — reads parquets, writes a SEPARATE "
                                             "robustness report, never promoted.json.")
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(config.N_FINAL_SEEDS)),
                    help="comma seeds to aggregate (default 0..N_FINAL_SEEDS-1)")
    ap.add_argument("--alpha", type=float, default=0.05, help="two-sided CI level (default 0.05 -> 95%%)")
    ap.add_argument("--no-write", action="store_true", help="print only; do not write the report")
    ap.add_argument("--no-registry", action="store_true",
                    help="aggregate on PARQUET PRESENCE alone, ignoring the experiment registry. Use ONLY "
                         "when gathering per-seed parquets produced on DIFFERENT machines (no shared "
                         "registry): the staleness fence is off, so every gathered parquet is trusted, and "
                         "fold identity is checked from the recorded n_train/n_val sizes instead of the "
                         "registry split label.")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    reg = None if a.no_registry else config.REGISTRY_PATH
    if a.no_registry:
        print("[robust] --no-registry: staleness fence OFF, trusting gathered parquets; "
              "fold identity from recorded n_train/n_val only")
    agg = aggregate_seeds(seeds, alpha=a.alpha, registry_path=reg)
    _print_agg(agg)
    if not a.no_write:
        paths = write_robustness_report(agg)
        print(f"[robust] report -> {paths['json']} , {paths['md']}")
    incomplete = {s: {n: st for n, st in cov.items() if st != "ok"} for s, cov in agg["coverage"].items()}
    incomplete = {s: v for s, v in incomplete.items() if v}
    if incomplete:
        print(f"[robust] ** INCOMPLETE COVERAGE **: {incomplete}")
    if agg["fold"]["single_frozen_fold"] is False:
        print(f"[robust] ** FOLD MISMATCH **: splits {agg['fold']['observed_splits']} sizes "
              f"{agg['fold']['observed_fold_sizes']} — seeds are NOT comparable")
    elif agg["fold"]["single_frozen_fold"] is None:
        print("[robust] ** NO FOLD EVIDENCE **: neither a registry split nor recorded fold sizes — "
              "comparability is UNVERIFIED (absence of evidence is not a pass)")
    if not agg["balanced"]:
        print(f"[robust] ** UNBALANCED **: per-config seed bases differ; common seeds "
              f"{agg['common_seeds']} — the ranking compares different bases")
    rc = _exit_code(agg)
    if rc:
        print(f"[robust] exiting {rc}: this report is not a clean comparable result")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(main())
