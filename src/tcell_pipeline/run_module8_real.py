"""Module 8 real-data driver (feat-010 comparators / feat-012 rationale audit / feat-013 reproducibility).

    PYTHONPATH=src python -m tcell_pipeline.run_module8_real --part comparators --device cuda

Parts:
  comparators  fit Stable-Shift + TxPert-public on the REAL train fold (STRING topology, train responses
               only) and score the REAL val fold; writes predictions in the common output schema, a metrics
               table, a compatibility report per family, and registers each family in the experiment
               registry. Pure numpy/scipy — no GPU, and it runs on the FULL fold.
  audit        run the rationale audit over the REAL PPI graph. NOTE: feat-012's campaign requires the FROZEN
               PROMOTED H1 model, which does not exist yet (feat-011's screening campaign is blocked on the
               graph mini-batch refactor). With --untrained this exercises the audit end-to-end at real
               scale on an UNTRAINED graph model: the faithfulness numbers are then a machinery check, NOT a
               scientific result, and the report says so.
  repro        verify the frozen preprocessing artifacts against this checkout with a manifest built from
               the real files. With no sealed decision on disk the confirmatory check is 'missing', so the
               honest verdict is CANNOT_VERIFY — the per-check table is the useful output.

The sealed challenge evaluation is deliberately NOT here: it is write-once on the SEQUESTERED split and must
be run once, by the test steward, against the promoted final model (report §Phase 5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.comparators import (  # noqa: E402
    StableShiftAdapter,
    TxPertPublicAdapter,
    write_compatibility_report,
)
from tcell_pipeline.evaluation.output_schema import write_predictions  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.programs.program_basis import zscore_path  # noqa: E402
from tcell_pipeline.screening.experiment_registry import log_run, register_run  # noqa: E402
from tcell_pipeline.screening.screening import (  # noqa: E402
    collect_targets_truth,
    compute_all_metrics,
    dataset_delta_z,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402

_COLS = ["name", "systema", "pearson", "centroid", "prog_cos", "mae", "rmse", "topk", "sign"]
_COMPARATORS = (StableShiftAdapter, TxPertPublicAdapter)


def _folds(n_max=None):
    train, val = PerturbationDataset("train", n_max=n_max), PerturbationDataset("val", n_max=n_max)
    return train, val


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def summarize_vs_h1(comparator_rows: list[dict], h1: dict | None) -> dict:
    """H1-vs-comparators on the primary endpoint (report §Comparators: H1 must beat "the strongest eligible
    comparator"). Ranks the external comparators + the frozen H1 on ``systema`` and records H1's margin over
    the strongest ELIGIBLE comparator — a finite-systema one, so a NaN/Inf metric (a near-null-signal
    degeneracy) cannot masquerade as the bar to clear (the same guard ``promote()`` grew). ``h1`` is read
    from ``promoted.json`` (NOT retrained); ``None`` when no frozen H1 exists yet — the comparators still
    rank, margin is undefined rather than a crash. A negative margin is reported as a LOSS, not hidden: an
    H1 that also loses to a public comparator is a valid converging negative."""
    finite = [r for r in comparator_rows if _finite(r.get("systema"))]
    strongest = sorted(finite, key=lambda r: (-r["systema"], r["name"]))[0] if finite else None
    h1_sys = h1.get("systema") if h1 else None
    margin = (h1_sys - strongest["systema"]) if (strongest and _finite(h1_sys)) else None

    entries = [{"name": r["name"], "systema": r.get("systema"), "kind": "comparator"} for r in comparator_rows]
    if h1:
        entries.append({"name": h1.get("name"), "systema": h1_sys, "kind": "frozen_h1"})
    # non-finite systema sinks to the bottom of the ranking rather than sorting as a huge/small number
    ranked = sorted(entries, key=lambda e: (-(e["systema"] if _finite(e["systema"]) else -math.inf), e["name"]))
    return {
        "primary_metric": "systema",
        "frozen_h1": h1.get("name") if h1 else None,
        "h1_systema": h1_sys,
        "strongest_comparator": strongest["name"] if strongest else None,
        "strongest_comparator_systema": strongest["systema"] if strongest else None,
        "margin_h1_minus_strongest": margin,
        "h1_beats_strongest": bool(margin is not None and margin > 0),
        "ranked": ranked,
        "basis": "development val fold (blocked_target_ood); single-seed; H1 systema read from promoted.json "
                 "(frozen campaign artifact, NOT retrained). A comparator win does NOT rescue the graph "
                 "premise — the no-graph expression_only model beats these comparators too (see campaign spec).",
    }


# --------------------------------------------------------------------------------------------------
def run_comparators(n_max=None, seed: int = 0, register: bool = True) -> int:
    """feat-010 on the real fold: fit on TRAIN responses over STRING topology, score VAL."""
    graph, g2i = build_hetero_graph()
    train, val = _folds(n_max)
    tr, va = collect_targets_truth(train), collect_targets_truth(val)
    train_mean = dataset_delta_z(train).mean(0)
    B = train.B.numpy()
    print(f"[m8-comp] {len(tr['genes'])} train / {len(va['genes'])} val rows; graph "
          f"{graph['protein'].x.shape[0]} proteins; K={B.shape[1]} G={B.shape[0]}", flush=True)

    rows = []
    for cls in _COMPARATORS:
        run_id = None
        if register:
            run_id = register_run(f"{cls.family}_v1", "H1-comparator", "q_pre", "blocked_target_ood", seed,
                                  None, family=cls.family, path=config.REGISTRY_PATH)
        try:
            model = cls.from_hetero_graph(graph, g2i, basis=B, string_only=True)
            model.fit(tr["genes"], tr["delta_z"])          # TRAIN responses only — the leakage fence
            dz, dx = model.predict(va["genes"])
            metrics = compute_all_metrics(dz, dx, va["delta_z"], va["delta_x"], train_mean)
            write_predictions(va["row_index"], dz, dx, None, model=cls.family, split="val", seed=seed,
                              root=config.PREDICTIONS_ROOT)
            report = write_compatibility_report(cls, root=config.COMPARATORS_ROOT)
            rows.append({"name": cls.family, **metrics})
            covered = int((dz != 0).any(1).sum())
            print(f"[m8-comp] {cls.family}: systema={metrics['systema']:+.4f} pearson={metrics['pearson']:+.4f} "
                  f"covered={covered}/{len(dz)} rows; compat -> {report}", flush=True)
            if run_id:
                log_run(run_id, "completed", metrics, None, path=config.REGISTRY_PATH)
        except Exception as exc:
            print(f"[m8-comp] {cls.family} FAILED: {type(exc).__name__}: {exc}", flush=True)
            if run_id:
                log_run(run_id, "failed", {"error": str(exc)}, None, path=config.REGISTRY_PATH)
            raise

    out = Path(config.COMPARATORS_ROOT) / "comparators_val.parquet"
    config.write_parquet_atomic(pd.DataFrame(rows), out)
    print("\n" + " ".join(f"{c:>16}" if c == "name" else f"{c:>9}" for c in _COLS))
    for r in rows:
        print(" ".join(f"{r['name']:>16}" if c == "name" else f"{r[c]:>9.4f}" for c in _COLS))
    print(f"[m8-comp] table -> {out}")

    # H1-vs-comparators on systema. The frozen H1 systema is read from promoted.json (this run scores only
    # the comparators; the H1 was scored by the campaign on the SAME fold — do not retrain it).
    from tcell_pipeline.screening.screening import _finite_or_none
    promo_path = Path(config.SCREENING_ROOT) / "promoted.json"
    h1 = json.loads(promo_path.read_text()).get("final") if promo_path.exists() else None
    summary = summarize_vs_h1(rows, h1)
    summary["comparators_val_parquet"] = str(out)
    summary["promoted_json"] = str(promo_path) if h1 else None
    summ_path = Path(config.COMPARATORS_ROOT) / "comparators_vs_h1.json"
    config.write_text_atomic(json.dumps(_finite_or_none(summary), indent=2, allow_nan=False), summ_path)
    if h1:
        verdict = ("beats" if summary["h1_beats_strongest"] else "does NOT beat")
        print(f"[m8-comp] H1 {summary['frozen_h1']} systema={summary['h1_systema']:+.4f} {verdict} strongest "
              f"comparator {summary['strongest_comparator']} systema="
              f"{summary['strongest_comparator_systema']:+.4f} (margin {summary['margin_h1_minus_strongest']:+.4f})")
    else:
        print("[m8-comp] no promoted.json — comparators ranked without a frozen H1 to compare against")
    print(f"[m8-comp] H1-vs-comparators summary -> {summ_path}")
    return 0


# --------------------------------------------------------------------------------------------------
def run_audit(n_cases: int, n_controls: int, device: str, n_max=None, untrained: bool = True) -> int:
    """feat-012 machinery over the REAL graph. Honest framing: without the frozen promoted H1 the numbers
    are a machinery check, not the feat-012 result."""
    from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder
    from tcell_pipeline.model import EGIPGModel
    from tcell_pipeline.rationale import RationaleHead, audit_rationale

    if not untrained:
        print("[m8-audit] a trained graph checkpoint is not available (Stage-A's real run is expr-only and "
              "the graph model cannot converge until the mini-batch refactor lands) — rerun with --untrained")
        return 1
    torch.set_num_threads(1)
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, g2i = build_hetero_graph()
    ds = PerturbationDataset("val", n_max=n_max)
    model = EGIPGModel.from_saved_basis(gene_names, graph_encoder=TypedGraphEncoder(graph, g2i)).eval()
    head = RationaleHead().eval()
    print(f"[m8-audit] UNTRAINED graph model over the real PPI graph; {len(ds)} val rows; "
          f"n_cases={n_cases} n_controls={n_controls} device={device}", flush=True)
    report = audit_rationale(model, head, ds, n_cases=n_cases, n_controls=n_controls, device=device, seed=0)
    agg = report["aggregate"]
    print(f"[m8-audit] audited={report['n_audited']} uncovered_in_fold={report['n_uncovered_in_dataset']}")
    print(f"[m8-audit] frac sufficiency<random = {agg['frac_sufficiency_below_random']}")
    print(f"[m8-audit] frac necessity>random   = {agg['frac_necessity_above_random']}")
    print(f"[m8-audit] mean minimality={agg['mean_minimality']} mean stability={agg['mean_stability']}")
    print(f"[m8-audit] source ablation Δ: {agg['source_ablation_delta_sufficiency']}")
    print(f"[m8-audit] report -> {report['report_path']}")
    print("[m8-audit] NOTE: model is UNTRAINED — this validates the audit path at real scale; the feat-012 "
          "campaign needs the frozen promoted H1.")
    return 0


# --------------------------------------------------------------------------------------------------
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_repro() -> int:
    """feat-013 against this checkout, with a manifest built from the REAL frozen artifacts."""
    from tcell_pipeline.reproducibility import verify_reproducibility

    root = config.PROJECT_ROOT
    artifacts = {"id_mapping": config.ID_MAPPING_PATH, "splits": config.BLOCKED_SPLIT_PATH,
                 "de_layers": zscore_path()}
    # Prefer an INDEPENDENT expected hash — one recorded when the artifact was frozen. splits/manifest.json
    # published sha256 at freeze time, so checking today's file against it is a genuine reproduction test.
    # Where no frozen record exists we can only self-hash, which proves the file is readable but compares it
    # to itself; those entries are labelled so the report is not read as more than it is.
    frozen_split = {}
    if config.SPLIT_MANIFEST_PATH.exists():
        frozen_split = (json.loads(config.SPLIT_MANIFEST_PATH.read_text()).get("sha256") or {})
    hashes, independent = {}, []
    for name, p in artifacts.items():
        rel = str(Path(p).relative_to(root))          # MUST be relative to the checkout
        expected = frozen_split.get(Path(p).name)     # an independently frozen record, if one was published
        hashes[name] = {"path": rel, "sha256": expected or _sha(Path(p))}
        src = "frozen manifest" if expected else "self-derived (no frozen record)"
        if expected:
            independent.append(name)
        print(f"[m8-repro] {name:11} {rel}  expected={hashes[name]['sha256'][:16]}…  [{src}]", flush=True)

    snapshot = {k: getattr(config, k) for k in
                ("PROGRAM_DIM", "DE_N_VARS", "DELTA_PRED", "N_BOOTSTRAP", "SPLIT_SEED", "SPLIT_FRACTIONS",
                 "SEQ_SIM_COSINE_THRESHOLD", "GROUP_SIZE_CAP", "MAX_EGIPG_TRIALS", "RATIONALE_TOP_K")}
    manifest = {
        "hashes": hashes,
        "config_hashes": {"config_snapshot":
                          hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()},
    }
    pred = Path(config.PREDICTIONS_ROOT) / "perturbed_mean" / "val" / "0.parquet"
    if pred.exists():
        manifest["predictions"] = {"perturbed_mean_val": {"path": str(pred.relative_to(root))}}
    # decision / observed / fallacy_inputs are deliberately absent: no sealed decision exists yet and the
    # fallacy probes must be authored by the analyst from real diagnostics. The verifier is expected to say
    # CANNOT_VERIFY — that is the correct answer, not a failure of the run.
    report = verify_reproducibility(root, manifest, config_snapshot=snapshot)
    print(f"\n[m8-repro] VERDICT = {report['verdict']}")
    for c in report["checks"]:
        print(f"  {c['status']:10} {c['category']:10} {c['check']}"
              + (f"   ({c['reason']})" if c.get("reason") else ""))
    print(f"[m8-repro] report -> {report['report_path']}")
    print(f"[m8-repro] independently-checked artifacts (expected hash came from a frozen record, so a MATCH "
          f"is a real reproduction test): {independent or 'NONE'}")
    print("[m8-repro] the remaining hash entries are self-derived — they prove the artifact is readable and "
          "stable within this checkout, but compare it to itself, so they are NOT a reproduction test. A true "
          "clean-checkout run needs the original run's published manifest.")
    print("[m8-repro] CANNOT_VERIFY is the correct verdict here: no sealed confirmatory decision exists to "
          "reproduce, and the 11 fallacy probes must be authored by the analyst from real diagnostics.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["comparators", "audit", "repro", "all"])
    ap.add_argument("--n-max", type=int, default=None, help="cap rows per split (quick runs)")
    ap.add_argument("--n-cases", type=int, default=config.N_RATIONALE_AUDIT_CASES)
    ap.add_argument("--n-controls", type=int, default=10, help="matched-random controls per audited case")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--untrained", action="store_true", default=True)
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.part in ("comparators", "all"):
        rc |= run_comparators(a.n_max, register=not a.no_register)
    if a.part in ("audit", "all"):
        rc |= run_audit(a.n_cases, a.n_controls, a.device, a.n_max, a.untrained)
    if a.part in ("repro", "all"):
        rc |= run_repro()
    return rc


if __name__ == "__main__":
    sys.exit(main())
