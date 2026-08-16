"""A5: put absolute numbers on the rationale audit's relative claims.

The audit reports comparisons against matched random controls: what fraction of cases beat a random
rationale on sufficiency, on necessity, and how much sufficiency each evidence source carries. Those are
the right controls. They are also RATIOS, and this task sits in a near-null regime where a ratio can be
large while the quantity it divides is negligible. This computes the absolute gaps behind them, so the
paper can state a size rather than a direction.

It also normalises the source ablation by EDGE COUNT, which turns out to matter. STRING is 85% of the
graph, so "removing STRING costs far more sufficiency than removing CORUM" is partly a statement about
how many edges were removed. The per-edge-share figure is the one that speaks to evidence quality.

CPU only, no GPU, no training: everything here is arithmetic on landed artifacts.

    PYTHONPATH=src python -m tcell_pipeline.screening.rationale_audit_bound \
        --out data/results/a2_power/rationale_bound.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

AUDIT = "data/results/rationale_audit_lambda0/audit_report.json"

# Per-source edge counts of the frozen PPI graph. Derived once, on 2026-08-16, by counting the source
# one-hot column of every protein-protein edge_attr:
#   build_hetero_graph(); for each of the three PP relations, (edge_attr[:, :len(PPI_SOURCES)] > 0).sum(0)
# Held as a constant because rebuilding an 8M-edge graph to divide four numbers is not worth two minutes
# on a shared box; pass --recount to re-derive and verify them. A graph rebuild invalidates them, and
# the total below is what a re-count is checked against.
EDGE_COUNTS = {"bioplex": 118_162, "huri": 51_773, "biogrid": 904_881,
               "string": 6_857_702, "corum": 96_778}
TOTAL_PP_EDGES = 8_029_296


def paired_gap(cases: list[dict], value: str, control: str) -> dict:
    """Paired mean, CI and p for one audit quantity against its matched random control, plus the gap as
    a fraction of the control. The fraction is the number the audit reports; the absolute mean is the
    number it omits, and in a near-null regime that is the one that decides whether it matters."""
    x = np.array([c[value] for c in cases], dtype=float)
    y = np.array([c[control] for c in cases], dtype=float)
    d = x - y
    t, p = stats.ttest_rel(x, y)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    denom = abs(float(y.mean()))
    return {"n": int(len(d)), "mean_value": float(x.mean()), "mean_control": float(y.mean()),
            "gap": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p),
            "gap_as_fraction_of_control": (float(d.mean()) / denom) if denom > 0 else None}


def per_edge_share(deltas: dict, counts: dict = EDGE_COUNTS, total: int = TOTAL_PP_EDGES) -> dict:
    """Each source's sufficiency delta divided by its share of the graph's edges, so a source is not
    credited for merely being large."""
    out = {}
    for src, delta in deltas.items():
        n = counts.get(src)
        if not n:
            continue
        share_pct = 100.0 * n / total
        out[src] = {"delta": float(delta), "edges": int(n), "share_pct": share_pct,
                    "delta_per_pct_of_edges": float(delta) / share_pct}
    return out


def recount_edges() -> tuple[dict, int]:
    """Re-derive EDGE_COUNTS from the graph. Slow (loads 8M edges); used to verify, not by default."""
    from tcell_pipeline import config
    from tcell_pipeline.graph import PROTEIN, build_hetero_graph
    graph, _ = build_hetero_graph()
    counts = {s: 0 for s in config.PPI_SOURCES}
    total = 0
    for rel in ("physical_ppi", "co_complex", "functional_assoc"):
        ea = graph[PROTEIN, rel, PROTEIN].edge_attr
        if ea.numel() == 0:
            continue
        total += int(ea.shape[0])
        onehot = ea[:, :len(config.PPI_SOURCES)]
        for i, s in enumerate(config.PPI_SOURCES):
            counts[s] += int((onehot[:, i] > 0).sum())
    return counts, total


def run(audit: str = AUDIT, out: Path | None = None, recount: bool = False) -> dict:
    d = json.loads(Path(audit).read_text())
    cases, agg = d["cases"], d["aggregate"]
    counts, total = (recount_edges() if recount else (EDGE_COUNTS, TOTAL_PP_EDGES))
    if recount:
        drift = {k: (counts[k], EDGE_COUNTS.get(k)) for k in counts if counts[k] != EDGE_COUNTS.get(k)}
        print(f"[a5] re-counted edges: total {total:,}" + (f"  DRIFT {drift}" if drift else "  (matches)"))

    report = {
        "audit": audit, "n_cases": len(cases), "gate_mean": agg.get("gate_mean", d.get("gate_mean")),
        "gates_live": d.get("gates_live"),
        "sufficiency": paired_gap(cases, "sufficiency", "random_sufficiency"),
        "necessity": paired_gap(cases, "necessity", "random_necessity"),
        "frac_necessity_above_random": agg.get("frac_necessity_above_random"),
        "frac_sufficiency_below_random": agg.get("frac_sufficiency_below_random"),
        "ginx_by_sparsity": agg.get("ginx_by_sparsity"),
        "source_ablation": per_edge_share(agg["source_ablation_delta_sufficiency"], counts, total),
    }

    print(f"[a5] {report['n_cases']} audited cases, gate mean {d.get('gate_mean'):.3f}, "
          f"gates live {d.get('gates_live')}")
    for k in ("sufficiency", "necessity"):
        s = report[k]
        frac = s["gap_as_fraction_of_control"]
        print(f"[a5] {k:>11}: rationale {s['mean_value']:.3e} vs random {s['mean_control']:.3e}; "
              f"gap {s['gap']:+.3e} [{s['ci_low']:+.3e},{s['ci_high']:+.3e}] p={s['p_value']:.1e}"
              + (f"  = {100*frac:+.1f}% of control" if frac is not None else ""))
    print("[a5] source ablation, raw and per 1% of the graph's edges:")
    for src, s in sorted(report["source_ablation"].items(), key=lambda kv: -kv[1]["delta"]):
        print(f"[a5]   {src:>8}: delta {s['delta']:.5f}  edges {s['edges']:>9,} ({s['share_pct']:5.2f}%)"
              f"  -> {s['delta_per_pct_of_edges']:.5f} per 1% of edges")
    ranked = sorted(report["source_ablation"].items(), key=lambda kv: -kv[1]["delta_per_pct_of_edges"])
    print(f"[a5] per-edge ranking: {' > '.join(s for s, _ in ranked)}")

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"[a5] wrote {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=AUDIT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--recount", action="store_true", help="re-derive the per-source edge counts")
    a = ap.parse_args()
    run(a.audit, Path(a.out) if a.out else None, a.recount)
