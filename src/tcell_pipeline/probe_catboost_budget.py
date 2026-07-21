"""Does a CONVERGED CatBoost close H1's comparator margin? (feat-006 Phase 1 follow-up)

``catboost_qpre`` scored +0.0657 at a 1000-iteration budget and reported ``tree_count_ == max_iter``, i.e.
it was still improving when the budget ran out — so that score is a LOWER bound, and it sits only 0.0037
below ``elastic_net_qpre`` (+0.0694), the bar H1's +0.0140 margin is measured against. An under-fit bar
inflates the very margin it exists to bound, so the question has to be computed, not argued.

This reuses the driver's OWN feature construction (``_qpre_block``), fold loading and metric suite, so the
number it produces is directly comparable to the published table rather than a parallel reimplementation.
It writes to a SCRATCH path and never touches the run cache, the predictions store, or the artifact.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.baselines import CatBoostBaseline  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.run_module8_real import _qpre_block  # noqa: E402
from tcell_pipeline.screening.screening import (  # noqa: E402
    collect_targets_truth,
    compute_all_metrics,
    dataset_delta_z,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402

CURVE = "--curve" in sys.argv
GROUPED = "--grouped" in sys.argv
_FINE = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 400]
_POS = [a for a in sys.argv[1:] if not a.startswith("--")]   # flags must not be parsed as positionals
ITERATIONS = int(_POS[0]) if len(_POS) > 0 else 4000
OD_WAIT = int(_POS[1]) if len(_POS) > 1 else 100
THREADS = int(_POS[2]) if len(_POS) > 2 else 48
OUT = Path("data/results/comparators/"
           + ("_probe_catboost_curve.json" if CURVE else "_probe_catboost_budget.json"))


def _grouped(Xtr, Xva, tr, va, train_mean, B) -> int:
    """CatBoost's boosting depth chosen VAL-BLIND, on a TARGET-GROUPED holdout carved from train.

    The val-selected curve peaked at 25 trees with systema +0.0843 — above the frozen H1 — but a depth
    picked by looking at the evaluation fold is not a bar, it is an upper bound (and a best-of-18 maximum
    at that). CatBoost's own built-in early stopping is no good either: it holds out a RANDOM slice of
    train, and because one target gene spans many rows that slice shares targets with the rows being fitted,
    so it keeps rewarding depth long after blocked-target-OOD generalisation has decayed — which is exactly
    why the 4000-iteration "converged" fit (0.0553) scored WORSE than an arbitrary 1000-tree cut (0.0657).

    So select on a holdout that has the same structure as the real evaluation: hold out whole TARGET GENES.
    Fit on 90% of targets, read the depth off the disjoint 10%, then refit on 100% of train at that depth
    and touch val exactly ONCE. That number is a legitimate published bar."""
    from catboost import CatBoostRegressor

    genes = np.asarray(tr["genes"])
    uniq = np.unique(genes)
    rng = np.random.default_rng(0)
    held = set(uniq[rng.permutation(len(uniq))[: max(1, int(round(0.10 * len(uniq))))]].tolist())
    m_hold = np.array([g in held for g in genes])
    print(f"[grp] {len(uniq)} train targets -> {len(held)} held out; rows {int(m_hold.sum())} holdout / "
          f"{int((~m_hold).sum())} fit; target overlap = "
          f"{len(set(genes[m_hold]) & set(genes[~m_hold]))} (must be 0)", flush=True)

    z = tr["delta_z"]
    fit_mean = z[~m_hold].mean(0)                 # the mean a model trained on this subset would subtract
    t0 = time.time()
    a = CatBoostRegressor(loss_function="MultiRMSE", iterations=max(_FINE), depth=6, learning_rate=0.1,
                          verbose=False, random_seed=0, thread_count=THREADS)
    a.fit(Xtr[~m_hold], z[~m_hold])
    print(f"[grp] selection fit ({max(_FINE)} trees) in {time.time()-t0:.0f}s", flush=True)

    sel = []
    for k in _FINE:
        dzh = np.asarray(a.predict(Xtr[m_hold], ntree_end=k), dtype=np.float64)
        s = compute_all_metrics(dzh, dzh @ B.T, z[m_hold], tr["delta_x"][m_hold], fit_mean)["systema"]
        sel.append({"n_trees": k, "holdout_systema": s})
        print(f"[grp] n_trees={k:4d} holdout systema={s:+.4f}", flush=True)
    k_star = max(sel, key=lambda r: r["holdout_systema"])["n_trees"]
    print(f"[grp] VAL-BLIND selected depth = {k_star} trees", flush=True)

    t1 = time.time()
    full = CatBoostRegressor(loss_function="MultiRMSE", iterations=k_star, depth=6, learning_rate=0.1,
                             verbose=False, random_seed=0, thread_count=THREADS)
    full.fit(Xtr, z)                                          # refit on 100% of train at the chosen depth
    dz = np.asarray(full.predict(Xva), dtype=np.float64)
    metrics = compute_all_metrics(dz, dz @ B.T, va["delta_z"], va["delta_x"], train_mean)
    print(f"[grp] refit {k_star} trees on 100% train in {time.time()-t1:.0f}s", flush=True)

    doc = {"mode": "target_grouped_valblind_selection", "selection_grid": _FINE,
           "selection_curve": sel, "selected_n_trees": k_star, "val_metrics": metrics,
           "val_systema": metrics["systema"],
           "reference": {"elastic_net_qpre": 0.0694, "frozen_h1": 0.0834, "expression_only_no_graph": 0.0861,
                         "catboost_val_selected_25trees": 0.0843},
           "note": ("depth chosen on a holdout of DISJOINT target genes carved from train; val touched once "
                    "at the selected depth. This is a legitimate bar, unlike the val-selected 0.0843.")}
    out = Path("data/results/comparators/_probe_catboost_grouped.json")
    config.ensure_dir(out.parent)
    config.write_text_atomic(json.dumps(doc, indent=2, allow_nan=False, default=float), out)
    h1 = 0.0834
    print(f"[grp] VAL-BLIND CatBoost systema={metrics['systema']:+.4f} at {k_star} trees | "
          f"vs elastic_net_qpre 0.0694: {'BEATS' if metrics['systema'] > 0.0694 else 'below'} | "
          f"vs frozen H1 {h1}: {'BEATS H1' if metrics['systema'] > h1 else 'below H1'}", flush=True)
    print(f"[grp] -> {out}", flush=True)
    return 0


def _curve(Xtr, Xva, tr, va, train_mean, B) -> int:
    """The OOD generalisation curve vs boosting depth, from ONE fit.

    The 4000-iteration fit CONVERGED by its own early stopping (3613) and scored WORSE on val (0.0553) than
    an arbitrary 1000-iteration truncation (0.0657). The stopping rule is the problem: CatBoost holds out a
    RANDOM slice of train, but a target gene occupies many rows, so that slice shares targets with the rows
    it is fitted on — while the real val fold is blocked-target OOD with DISJOINT targets. The internal
    signal therefore keeps rewarding depth long after OOD generalisation has started to decay, so the true
    peak lies BELOW 1000 and neither number so far is CatBoost's honest best.

    So: fit once WITHOUT early stopping (no internal holdout, so the bar also trains on 100% of train) and
    score every prefix of the tree sequence. Picking the best point ON VAL is deliberately OPTIMISTIC for
    the bar — it is a val-SELECTED upper bound, not a clean generalisation estimate. That is the right
    direction for a floor H1 must clear: if even a val-tuned CatBoost cannot reach elastic_net_qpre, no
    reasonable CatBoost does."""
    from catboost import CatBoostRegressor

    grid = [25, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1250, 1500]
    t0 = time.time()
    cb = CatBoostRegressor(loss_function="MultiRMSE", iterations=max(grid), depth=6, learning_rate=0.1,
                           verbose=False, random_seed=0, thread_count=THREADS)
    cb.fit(Xtr, tr["delta_z"])                      # no eval_set -> full tree sequence, 100% of train
    fit_s = time.time() - t0
    print(f"[curve] fitted {max(grid)} trees on 100% of train in {fit_s:.0f}s", flush=True)

    rows = []
    for k in grid:
        dz = np.asarray(cb.predict(Xva, ntree_end=k), dtype=np.float64)
        m = compute_all_metrics(dz, dz @ B.T, va["delta_z"], va["delta_x"], train_mean)
        rows.append({"n_trees": k, "systema": m["systema"], "pearson": m["pearson"]})
        print(f"[curve] n_trees={k:5d}  systema={m['systema']:+.4f}  pearson={m['pearson']:+.4f}", flush=True)

    best = max(rows, key=lambda r: r["systema"])
    doc = {"mode": "val_selected_depth_curve", "grid": grid, "fit_seconds": round(fit_s, 1), "rows": rows,
           "best": best, "beats_elastic_net_qpre": bool(best["systema"] > 0.0694),
           "reference": {"catboost_qpre_1000iter": 0.0657, "catboost_qpre_4000iter_converged": 0.0553,
                         "elastic_net_qpre": 0.0694, "frozen_h1": 0.0834},
           "note": ("depth selected ON VAL, so this OVERSTATES CatBoost — an optimistic upper bound for the "
                    "bar, which is the conservative direction for a floor H1 must clear. One fit, no early "
                    "stopping, scored at every tree prefix.")}
    config.ensure_dir(OUT.parent)
    config.write_text_atomic(json.dumps(doc, indent=2, allow_nan=False, default=float), OUT)
    print(f"[curve] BEST n_trees={best['n_trees']} systema={best['systema']:+.4f} -> "
          f"{'OVERTAKES elastic_net_qpre 0.0694' if doc['beats_elastic_net_qpre'] else 'still below 0.0694'}",
          flush=True)
    print(f"[curve] -> {OUT}", flush=True)
    return 0


def main() -> int:
    graph, g2i = build_hetero_graph()
    train, val = PerturbationDataset("train"), PerturbationDataset("val")
    tr, va = collect_targets_truth(train), collect_targets_truth(val)
    train_mean = dataset_delta_z(train).mean(0)
    B = train.B.numpy()
    Xg = graph["protein"].x.numpy()

    def node(genes):
        F = np.zeros((len(genes), Xg.shape[1]), dtype=np.float64)
        for i, g in enumerate(genes):
            j = g2i.get(g)
            if j is not None:
                F[i] = Xg[j]
        return F

    Ctr, Cva, names = _qpre_block(train, val)          # the SAME q_pre block the published bars used
    Xtr = np.hstack([node(tr["genes"]), Ctr])
    Xva = np.hstack([node(va["genes"]), Cva])
    print(f"[probe] Xtr={Xtr.shape} Xva={Xva.shape} K={B.shape[1]}; iterations={ITERATIONS} "
          f"od_wait={OD_WAIT} threads={THREADS}", flush=True)

    if GROUPED:
        return _grouped(Xtr, Xva, tr, va, train_mean, B)
    if CURVE:
        return _curve(Xtr, Xva, tr, va, train_mean, B)

    t0 = time.time()
    model = CatBoostBaseline(basis=B, iterations=ITERATIONS, od_wait=OD_WAIT, thread_count=THREADS)
    model.fit(Xtr, tr["delta_z"])
    dz, dx = model.predict(Xva)
    metrics = compute_all_metrics(dz, dx, va["delta_z"], va["delta_x"], train_mean)
    diag = model.fit_diagnostics()
    # the SHIPPED converged flag is too lenient in the near-miss band; record the corrected reading too
    corrected = bool(diag["n_iter_max"] + OD_WAIT <= ITERATIONS)
    doc = {"iterations": ITERATIONS, "od_wait": OD_WAIT, "wall_seconds": round(time.time() - t0, 1),
           "metrics": metrics, "fit_diagnostics": diag, "converged_corrected_criterion": corrected,
           "reference": {"catboost_qpre_1000iter": 0.0657, "elastic_net_qpre": 0.0694, "frozen_h1": 0.0834},
           "note": "scratch probe; same fold / q_pre block / metric suite as the published table"}
    config.ensure_dir(OUT.parent)
    config.write_text_atomic(json.dumps(doc, indent=2, allow_nan=False, default=float), OUT)
    print(f"[probe] systema={metrics['systema']:+.4f} pearson={metrics['pearson']:+.4f} "
          f"diag={diag} converged_corrected={corrected} in {doc['wall_seconds']}s", flush=True)
    print(f"[probe] vs elastic_net_qpre 0.0694 -> "
          f"{'OVERTAKES (H1 margin shrinks)' if metrics['systema'] > 0.0694 else 'still below'}", flush=True)
    print(f"[probe] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
