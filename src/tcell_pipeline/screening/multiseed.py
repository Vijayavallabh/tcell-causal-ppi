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
# pre-registered contrasts on systema + the promotion margin the seed-0 fold turned on (untyped − expr):
CONTRASTS = (
    ("h2a", TYPED_STATIC, EXPRESSION_ONLY),          # does typed static beat expression-only?
    ("h2b", CONDITION_GATED, TYPED_STATIC),          # does condition gating beat typed static?
    ("promotion_margin", UNTYPED_GNN, EXPRESSION_ONLY),  # does ANY graph beat no-graph?
)


def _finite(x) -> bool:
    """Reject NaN/Inf/None/bool/str; accept python AND numpy real scalars — parquet reads come back as
    numpy.float64, and a ``_finite`` that rejected them would silently drop every seed."""
    import numpy as np
    return isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) and math.isfinite(x)


def _verdict(mean: float, ci_excludes_zero: bool, n: int) -> str:
    if not ci_excludes_zero:
        return (f"indistinguishable at n={n}: the paired CI crosses zero — underpowered / no effect at "
                f"this budget, NOT support for the better arm")
    return f"CI excludes zero (n={n}) — favors the {'better' if mean > 0 else 'worse'} arm by Δ={mean:+.4f}"


def paired_delta_summary(better_by_seed, worse_by_seed, *, alpha: float = 0.05) -> dict:
    """One-sample (paired) t on ``d_s = better_s − worse_s`` against 0. Never raises, never NaN-poisons:
    a seed missing from either arm, or non-finite in either, is DROPPED (named in ``dropped``) and
    shrinks n. ``n<2`` -> mean only, no CI (a single seed is not a paired result). Honest verdict: a CI
    crossing zero is "indistinguishable at this budget", not support for the better arm."""
    seeds = sorted(set(better_by_seed) | set(worse_by_seed))
    deltas, used, dropped = [], [], []
    for s in seeds:
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
    if se == 0.0:  # identical deltas across seeds — a real edge; no t/p, zero-width CI parked at the mean
        out["ci_low"] = out["ci_high"] = mean
        out["p_value"] = 0.0 if mean != 0.0 else 1.0
        out["ci_excludes_zero"] = bool(mean != 0.0)
    else:
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
                      "n_epochs": row.get("n_epochs"), "mtime": path.stat().st_mtime}
    return metrics, status, meta


def _splits_by_config_seed(names, seeds, registry_path: Path | None) -> set:
    """Distinct ``split`` values the registry recorded for these (name, seed) COMPLETED runs — the
    fold-identity evidence. All must be the one frozen split; a second value means seeds were scored on
    different folds (provenance is not comparability). Empty when no registry is given."""
    if registry_path is None:
        return set()
    from tcell_pipeline.screening.experiment_registry import load_registry
    want_names, want_seeds = set(names), set(seeds)
    return {r.get("split") for r in load_registry(registry_path)
            if r.get("config_id") in want_names and int(r.get("seed", 0)) in want_seeds
            and r.get("status") == "completed"}


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
    seeds = list(seeds)
    by_config = {n: {} for n in names}   # name -> {seed: metric}
    coverage = {}                        # seed -> {name: status}
    run_meta = {}                        # "name@seed" -> run meta
    for s in seeds:
        m, st, mt = _read_seed_metrics(names, s, screening_root=screening_root,
                                       registry_path=registry_path, primary=primary)
        coverage[s] = st
        for n, v in m.items():
            by_config[n][s] = v
        for n, d in mt.items():
            run_meta[f"{n}@{s}"] = d

    contrasts = {}
    for key, better, worse in CONTRASTS:
        if better not in by_config or worse not in by_config:
            continue  # a contrast whose member is outside the requested family cannot be formed
        contrasts[key] = {"better": better, "worse": worse,
                          **paired_delta_summary(by_config[better], by_config[worse], alpha=alpha)}

    per_config = {n: _mean_ci(by_config[n], alpha=alpha) for n in names}
    ranked = sorted(per_config.items(),
                    key=lambda kv: (-(kv[1]["mean"] if kv[1]["mean"] is not None else -math.inf), kv[0]))
    splits = _splits_by_config_seed(names, seeds, registry_path)
    return {"seeds": seeds, "primary": primary, "alpha": alpha, "family": list(names),
            "coverage": coverage, "run_meta": run_meta,
            "fold": {"expected_split": FROZEN_SPLIT, "observed_splits": sorted(s for s in splits if s),
                     "single_frozen_fold": bool(splits <= {FROZEN_SPLIT})},
            "contrasts": contrasts, "per_config": per_config, "ranking": [n for n, _ in ranked]}


# --------------------------------------------------------------------------------------------------
# Report (separate from promoted.json)
# --------------------------------------------------------------------------------------------------
def _render_md(agg: dict) -> str:
    fold = agg["fold"]
    L = [f"# 5-seed robustness — paired {agg['primary']} across seeds {agg['seeds']}", "",
         "_Separate from `promoted.json`: the frozen H1 (condition_gated seed 0) is untouched._", "",
         f"**Fold:** expected `{fold['expected_split']}`, observed "
         f"{fold['observed_splits'] or '(no registry)'} — single frozen fold: "
         f"{fold['single_frozen_fold']}", "",
         "## Paired contrasts", "",
         "| contrast | better − worse | n | mean Δsystema | 95% CI | verdict |",
         "|---|---|---:|---:|---|---|"]
    for key, c in agg["contrasts"].items():
        ci = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]" if c["ci_low"] is not None else "n/a"
        mean = f"{c['mean']:+.4f}" if c["mean"] is not None else "n/a"
        L.append(f"| {key} | {c['better']} − {c['worse']} | {c['n']} | {mean} | {ci} | {c['verdict']} |")
    L += ["", "## Per-config systema (mean ± 95% CI)", "",
          "| rank | config | n | mean systema | 95% CI |", "|---:|---|---:|---:|---|"]
    for i, n in enumerate(agg["ranking"], 1):
        pc = agg["per_config"][n]
        ci = f"[{pc['ci_low']:+.4f}, {pc['ci_high']:+.4f}]" if pc["ci_low"] is not None else "n/a"
        mean = f"{pc['mean']:+.4f}" if pc["mean"] is not None else "n/a"
        L.append(f"| {i} | {n} | {pc['n']} | {mean} | {ci} |")
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
        print(f"[robust] {key:16s} {c['better']} - {c['worse']}: n={c['n']} mean={mean} CI={ci} "
              f":: {c['verdict']}")
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
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip() != ""]
    agg = aggregate_seeds(seeds, alpha=a.alpha)
    _print_agg(agg)
    if not a.no_write:
        paths = write_robustness_report(agg)
        print(f"[robust] report -> {paths['json']} , {paths['md']}")
    incomplete = {s: {n: st for n, st in cov.items() if st != "ok"} for s, cov in agg["coverage"].items()}
    incomplete = {s: v for s, v in incomplete.items() if v}
    if incomplete:
        print(f"[robust] ** INCOMPLETE COVERAGE **: {incomplete}")
    if not agg["fold"]["single_frozen_fold"]:
        print(f"[robust] ** FOLD MISMATCH **: observed splits {agg['fold']['observed_splits']} — "
              f"seeds are NOT comparable")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
