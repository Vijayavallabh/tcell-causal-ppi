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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.comparators import (  # noqa: E402
    StableShiftAdapter,
    TxPertPublicAdapter,
    write_compatibility_report,
)
from tcell_pipeline.evaluation.output_schema import prediction_path, write_predictions  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.programs.program_basis import zscore_path  # noqa: E402
from tcell_pipeline.screening.experiment_registry import log_run, register_run  # noqa: E402
from tcell_pipeline.screening.screening import (  # noqa: E402
    _finite_or_none,
    collect_targets_truth,
    compute_all_metrics,
    dataset_delta_z,
)
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402

_COLS = ["name", "systema", "pearson", "centroid", "prog_cos", "mae", "rmse", "topk", "sign"]
_COMPARATORS = (StableShiftAdapter, TxPertPublicAdapter)
# The 0.01 systema noise band the single-seed campaign flags on this same primary metric (mirrors the
# --noise-margin the screening promotion uses); a within-band H1-vs-comparator margin is a coin toss, not a win.
_NOISE_MARGIN = 0.01


def _folds(n_max=None):
    train, val = PerturbationDataset("train", n_max=n_max), PerturbationDataset("val", n_max=n_max)
    return train, val


def _finite(x) -> bool:
    # accept numpy scalars too (np.float32 is NOT a Python-float subclass) — mirrors screening._finite_or_none,
    # so a numpy-typed systema is not misclassified non-finite and silently dropped from the ranking
    return isinstance(x, (int, float, np.floating)) and math.isfinite(x)


def _fmt_signed(x) -> str:
    """None/NaN-safe signed formatter for the verdict print — the summary fields are legitimately None when
    there is no eligible comparator or no frozen H1, and ``None:+.4f`` would raise TypeError."""
    return f"{x:+.4f}" if _finite(x) else "n/a"


def _load_promoted_final(promo_path: Path) -> tuple[dict | None, str]:
    """Load ``promoted.json``'s ``final`` block robustly. Returns ``(final_dict | None, status)`` where status
    is 'absent' (no file), 'unreadable (<err>)' (corrupt/truncated JSON or an IO error), 'present but no valid
    ''final'' dict' (parses but ``final`` is missing / not a dict, e.g. a bare name string), or 'ok'. A
    present-but-unusable file is NOT silently reported as 'absent' — the status carries the distinction so
    provenance stays honest and a bad file skips the H1 verdict loudly instead of crashing the run after the
    comparator table is already written."""
    if not promo_path.exists():
        return None, "absent"
    try:
        doc = json.loads(promo_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"unreadable ({type(exc).__name__})"
    final = doc.get("final") if isinstance(doc, dict) else None
    if not isinstance(final, dict):
        return None, "present but no valid 'final' dict"
    return final, "ok"


_EXTERNAL_BASIS = (
    "development val fold (blocked_target_ood); single-seed; H1 systema read from promoted.json "
    "(frozen campaign artifact, NOT retrained). A comparator win does NOT rescue the graph premise — "
    "the no-graph expression_only model beats these comparators too (see campaign spec).")


def summarize_vs_h1(comparator_rows: list[dict], h1: dict | None, *, noise_margin: float = 0.0,
                    fold_comparable: bool = True, kind: str = "comparator",
                    basis: str | None = None) -> dict:
    """H1-vs-comparators on the primary endpoint (report §Comparators: H1 must beat "the strongest eligible
    comparator"). Ranks the external comparators + the frozen H1 on ``systema`` and records H1's margin over
    the strongest ELIGIBLE comparator — a finite-systema one, so a NaN/Inf metric (a near-null-signal
    degeneracy) cannot masquerade as the bar to clear (the same guard ``promote()`` grew).

    ``h1`` is the ``promoted.json`` ``final`` dict (NOT retrained), or ``None`` when no frozen H1 exists.
    ``fold_comparable=False`` (e.g. a ``--n-max`` capped run scored against the full-fold H1) suppresses the
    verdict entirely — the comparators still rank among themselves, but no H1 margin is emitted because the
    two systema values are on different rows. ``noise_margin`` flags a within-band margin as a coin toss.

    Undefined outcomes are reported as ``None``, never conflated: ``h1_beats_strongest`` is ``None`` when
    there is no eligible comparator OR no comparable H1 (that is "nothing to compare", NOT "H1 lost"); a real
    negative margin still reports ``False`` (a loss H1 owns, a valid converging negative)."""
    finite = [r for r in comparator_rows if _finite(r.get("systema"))]
    strongest = sorted(finite, key=lambda r: (-r["systema"], str(r["name"])))[0] if finite else None

    compare = bool(h1) and fold_comparable                 # only emit an H1 verdict on a comparable fold
    h1_sys = h1.get("systema") if compare else None
    margin = (h1_sys - strongest["systema"]) if (strongest and _finite(h1_sys)) else None
    beats = None if margin is None else bool(margin > 0)   # None = nothing to compare, NOT a loss
    within_noise = None if margin is None else bool(abs(margin) <= noise_margin)

    # `kind`/`basis` are parameterised: the feat-006 tabular baselines reuse this summarizer but are
    # explicitly NOT external comparators, and hardcoding both mislabelled them in the emitted JSON that
    # a later comparator-family-cap audit would scan.
    entries = [{"name": r["name"], "systema": r.get("systema"), "kind": kind} for r in comparator_rows]
    if compare:
        entries.append({"name": h1.get("name"), "systema": h1_sys, "kind": "frozen_h1"})
    # non-finite systema sinks to the bottom; str() on the name tie-break so a None H1 name never hits None < str
    ranked = sorted(entries, key=lambda e: (-(e["systema"] if _finite(e["systema"]) else -math.inf),
                                            str(e["name"])))
    out = {
        "primary_metric": "systema",
        "fold_comparable": bool(fold_comparable),
        "noise_margin": float(noise_margin),
        "frozen_h1_available": bool(h1),
        "frozen_h1": h1.get("name") if compare else None,
        "h1_systema": h1_sys,
        "strongest_comparator": strongest["name"] if strongest else None,
        "strongest_comparator_systema": strongest["systema"] if strongest else None,
        "margin_h1_minus_strongest": margin,
        "h1_beats_strongest": beats,
        "margin_within_noise": within_noise,
        "ranked": ranked,
        "kind": kind,
        "basis": basis if basis is not None else _EXTERNAL_BASIS,
    }
    if bool(h1) and not fold_comparable:
        out["h1_comparison_skipped"] = ("comparators scored on a subsampled/other fold; the frozen H1 systema "
                                        "is from a different fold and is NOT comparable — no verdict emitted")
    return out


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
    # the comparators; the H1 was scored by the campaign on the SAME fold — do not retrain it). A --n-max run
    # scores the comparators on a CAPPED fold, so its systema is not comparable to the full-fold H1: guard it.
    fold_comparable = n_max is None
    promo_path = Path(config.SCREENING_ROOT) / "promoted.json"
    h1, promo_status = _load_promoted_final(promo_path)
    if promo_status not in ("ok", "absent"):
        print(f"[m8-comp] WARNING: promoted.json {promo_status} — H1 verdict skipped", flush=True)
    if h1 and not fold_comparable:
        print(f"[m8-comp] WARNING: --n-max={n_max} subsamples the fold; the frozen H1 systema is full-fold, "
              f"so the H1-vs-comparator verdict is SKIPPED (not comparable)", flush=True)
    summary = summarize_vs_h1(rows, h1, noise_margin=_NOISE_MARGIN, fold_comparable=fold_comparable)
    summary["comparators_val_parquet"] = str(out)
    summary["promoted_json"] = str(promo_path) if promo_path.exists() else None  # keyed on the FILE, not on h1
    summary["promoted_json_status"] = promo_status
    summ_path = Path(config.COMPARATORS_ROOT) / "comparators_vs_h1.json"
    config.write_text_atomic(json.dumps(_finite_or_none(summary), indent=2, allow_nan=False), summ_path)
    if summary["h1_beats_strongest"] is not None:
        verdict = "beats" if summary["h1_beats_strongest"] else "does NOT beat"
        noise = " [WITHIN NOISE — single-seed, treat as a tie]" if summary["margin_within_noise"] else ""
        print(f"[m8-comp] H1 {summary['frozen_h1']} systema={_fmt_signed(summary['h1_systema'])} {verdict} "
              f"strongest comparator {summary['strongest_comparator']} "
              f"systema={_fmt_signed(summary['strongest_comparator_systema'])} "
              f"(margin {_fmt_signed(summary['margin_h1_minus_strongest'])}){noise}")
    elif h1 and fold_comparable:
        print("[m8-comp] frozen H1 present but no eligible (finite-systema) comparator to compare against")
    else:
        print("[m8-comp] no H1 verdict emitted — comparators ranked without a comparable frozen H1")
    print(f"[m8-comp] H1-vs-comparators summary -> {summ_path}")
    return 0


_TABULAR = ("zero", "perturbed_mean", "ridge", "elastic_net", "gradient_boosting", "catboost",
            "nearest_neighbor", "low_rank")
# TabICL is opt-in: it is a GPU transformer refit once PER PROGRAM (~125 s x K here), so the default
# CPU-only run must not silently become a multi-hour GPU job.
_TABULAR_GPU = ("tabicl",)
# ...and qpre-ONLY. The published H1 margin is measured against the qpre bars, so scoring the GPU bar on
# the handicapped node-only set would spend another ~4.4 GPU-h on a feature set no margin is drawn from.
_QPRE_ONLY = frozenset({"tabicl"})


def flag_underfit_bars(diagnostics: dict) -> dict:
    """Turn per-bar convergence evidence into a verdict on the H1 margin (AGENTS.md: a bar must be fit well
    enough to bound).

    Direction matters. For a COMPETITOR, under-fitting is safe — a weaker rival cannot manufacture a win.
    For a FLOOR the result must CLEAR it is the opposite: a bar stopped mid-descent scores lower than the
    model family actually reaches, so the published margin is an UPPER bound on H1's advantage, and a
    converged bar could only shrink it. That distinction was missed once here (the elastic-net bar shipped
    with ``n_iter_max == max_iter`` and 6.4% non-zero coefficients while the margin was reported as settled).

    Only iterative bars answer this question: ridge / kNN / low-rank / the mean baselines are closed-form and
    expose no ``fit_diagnostics`` at all, which is "no convergence question", not "unknown". A bar that DOES
    expose diagnostics but cannot report ``converged`` is ``unknown`` and counts against the margin —
    absence of evidence must never read as a pass."""
    underfit = sorted(n for n, d in diagnostics.items() if d.get("converged") is False)
    unknown = sorted(n for n, d in diagnostics.items() if d.get("converged") is None)
    return {
        "underfit": underfit,
        "unknown": unknown,
        "margin_is_upper_bound": bool(underfit or unknown),
        "note": ("no iterative bar is under-fit: the margin is bounded by bars that reached their own "
                 "optimum" if not (underfit or unknown) else
                 "at least one bar did not demonstrably converge, so it scores BELOW what its model family "
                 "reaches — the H1 margin is an UPPER BOUND and can only shrink"),
    }


# The q_pre covariates H1's own perturbation encoder consumes (encoders/batch.build_encoder_batch), minus
# ``uniprot_id`` — the 1412-d static graph node feature is the tabular bar's protein representation. Giving
# the bar the SAME prediction-time inputs is what makes it a fair floor: the first version saw only the
# target's node feature, so it could not distinguish Rest from Stim48hr at all while H1 could, and that
# handicap inflates the H1 margin exactly the way an under-fit bar does.
_QPRE_NUMERIC = ("ppi_degree_physical", "ppi_degree_functional", "ppi_degree_complex",
                 "control_baseline_expr")
_QPRE_CATEGORICAL = ("culture_condition",)
_QPRE_OBS = ("n_guides", "single_guide_estimate")       # from the DE obs table, not the perturbation table


def check_qpre(pc_columns, obs_columns) -> dict:
    """Refuse to build a baseline feature block out of anything response-derived.

    ``PerturbationEncoder`` already refuses ``Q_POST_COLS``; a tabular bar reading the same table straight
    from pandas has no such fence, and a q_post covariate would let a baseline peek at the response it is
    being scored on. Perturbation-table columns must be classified q_pre by the project's own fence
    (``feature_availability.classify_columns``) — ``metadata`` is that classifier's permissive fall-through,
    so an unclassified column is refused too rather than assumed safe. The DE obs columns are not in the
    perturbation table and so are not classified there; they are checked against ``Q_POST_COLS`` directly."""
    from tcell_pipeline.feature_availability import classify_columns

    pc_columns, obs_columns = list(pc_columns), list(obs_columns)
    tagged = classify_columns(pc_columns)
    if tagged["q_post"]:
        raise ValueError(f"leakage fence: {tagged['q_post']} are response-derived (q_post) — refusing to "
                         f"feed them to a baseline")
    leaked_obs = [c for c in obs_columns if c in config.Q_POST_COLS]
    if leaked_obs:
        raise ValueError(f"leakage fence: obs columns {leaked_obs} are response-derived (q_post)")
    if tagged["metadata"]:
        raise ValueError(f"leakage fence: {tagged['metadata']} are not declared q_pre (unclassified columns "
                         f"fall through to metadata, which is not evidence they are safe)")
    return {"q_pre": tagged["q_pre"], "obs": obs_columns}


def _bar_source(bar: str | None) -> str:
    """The source a bar's score actually depends on: its own class PLUS the shared base.

    ``inspect.getsource`` on a subclass returns only that subclass's block, so hashing the leaf alone
    left ``BaseBaseline`` — which owns ``_decode_genes`` (the ``dz @ B.T`` decode every bar's gene-space
    metrics run through), ``_features`` and the fit contract — outside the key. Editing it changed every
    bar's mae/rmse/topk/sign while ``baselines_sha`` sat still."""
    import inspect

    from tcell_pipeline.baselines import BASELINES, simple_baselines
    from tcell_pipeline.baselines.simple_baselines import BaseBaseline

    if bar is None:
        return inspect.getsource(simple_baselines)
    shared = inspect.getsource(BaseBaseline) + inspect.getsource(simple_baselines._np)
    return inspect.getsource(BASELINES[bar]) + shared


def _array_sha(a) -> str:
    """Content fingerprint of a float array, so a REFIT at identical shape is a different key."""
    import numpy as _np_mod
    if a is None:
        return "none"
    arr = _np_mod.ascontiguousarray(_np_mod.asarray(a, dtype=_np_mod.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _bar_signature(*, n_train: int, n_val: int, n_features: int, k: int, bar: str | None = None,
                   basis=None, features=None, seed: int = 0) -> dict:
    """What a cached bar score is only valid FOR.

    Fold dimensions are the obvious half. The half that actually bites is the CODE: the systema collapse
    fix changed every score without moving a single fold dimension, so a shape-only key would have served
    pre-fix numbers as current. Hashing source makes any edit — a metric definition, a hyperparameter
    default — invalidate the affected cache automatically, rather than relying on someone remembering to
    bump a version constant.

    Two corrections, both found by probing this key rather than by it firing:
    - PER-BAR, not per-module. Hashing all of ``simple_baselines`` meant correcting CatBoost's convergence
      criterion would have discarded TabICL's 4.4 GPU-hour cached score. Only the bar's own class matters.
    - The FEATURE CONSTRUCTION counts too. ``_qpre_block`` was invisible to the key, so swapping its
      imputation (median -> mean) would leave ``n_features`` identical and the cache would serve pre-change
      scores as current — the precise "presence is not freshness" failure the key exists to prevent."""
    import inspect

    from tcell_pipeline.evaluation import metrics as _M

    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    qpre_src = inspect.getsource(_qpre_block) + repr((_QPRE_NUMERIC, _QPRE_CATEGORICAL, _QPRE_OBS))
    return {"n_train": int(n_train), "n_val": int(n_val), "n_features": int(n_features), "k": int(k),
            "seed": int(seed),
            "metrics_sha": _sha(inspect.getsource(_M)), "baselines_sha": _sha(_bar_source(bar)),
            "qpre_sha": _sha(qpre_src),
            # CONTENT, not shape. A refit basis at the same PROGRAM_DIM, or a rebuilt STRING graph whose
            # node features change while the 1412 dim holds, leaves every other field byte-identical —
            # so every bar would hit the cache and republish pre-change scores while printing [cached].
            "basis_sha": _array_sha(basis), "features_sha": _array_sha(features)}


# Distinct exit BITS, one per part. They used to share a single `rc |= <0 or 1>`, which made two very
# different states indistinguishable: an under-fit tabular bar (the published H1 margin is only an upper
# bound — a finding) and feat-013's CANNOT_VERIFY (the correct and PERMANENT answer while the sealed
# split stays sequestered — not a finding). Under `--part all` the run therefore exited 1 on a healthy
# checkout every time, so any CI gate on it was permanently red and duly ignored, taking the under-fit
# guard down with it. Separate bits keep every guard honored to the exit code AND readable.
RC_COMPARATORS = 1 << 0
RC_BASELINES = 1 << 1     # under-fit bar: the H1 margin it bounds is an UPPER bound
RC_AUDIT = 1 << 2
RC_REPRO = 1 << 3         # NOT reproducible / cannot verify — expected while the split is sealed

_RC_NAMES = {RC_COMPARATORS: "comparators", RC_BASELINES: "baselines(under-fit)",
             RC_AUDIT: "audit", RC_REPRO: "repro(not-reproducible)"}


def _rc_legend(rc: int) -> str:
    return "+".join(name for bit, name in _RC_NAMES.items() if rc & bit) or "ok"


def _gpu_bar_device(device: str | None, explicit: bool) -> str | None:
    """Device for a GPU-only bar: None (let it auto-select) unless the user asked for one.

    ``--device`` is shared with the audit and defaults to cpu, so ``--with-tabicl`` alone constructed
    ``TabICLBaseline(device="cpu")`` and turned a ~4.4 GPU-h bar into a multi-day CPU job on a shared
    box — silently, since the flag's own help text says GPU."""
    return device if explicit else None


def _bar_cache_file(root, fs: str, name: str) -> Path:
    return Path(root) / f"{fs}__{name}.json"


def load_cached_bar(root, fs: str, name: str, signature: dict, predictions: Path | None = None) -> dict | None:
    """A completed bar's score, or None when absent, stale, incomplete, or missing its artifact.

    Three ways this returns a miss rather than a hit, each a real failure that reached the published
    table once:

    * signature mismatch — the score is not valid for this fold/code;
    * a metric recorded as NON-FINITE — ``_finite_or_none`` maps a non-finite mae to JSON null, and the
      resumed run then formats that None with ``:>9.4f`` and dies AFTER the parquet is written but
      BEFORE the H1 summary and the under-fit exit gate. A cache entry that cannot be printed is not a
      completed bar. The keys are recorded explicitly at save time rather than inferred from ``is
      None``, because None is also this project's encoding for "no evidence": keying the miss on None
      would make an honestly-undecidable metric recompute forever, silently, on every run;
    * ``predictions`` given but absent on disk — ``write_predictions`` only runs on a miss, so a cleared
      ``data/results/predictions`` would otherwise republish the whole comparator table with every
      per-bar prediction file missing."""
    path = _bar_cache_file(root, fs, name)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None                                     # unreadable cache is a miss, never a silent pass
    if doc.get("signature") != signature:
        return None
    if doc.get("nonfinite_metrics"):
        return None
    if predictions is not None and not Path(predictions).exists():
        return None
    return doc


def _nonfinite_keys(metrics: dict) -> list[str]:
    import math as _math
    out = []
    for k, v in (metrics or {}).items():
        if isinstance(v, (float, np.floating)) and not _math.isfinite(float(v)):
            out.append(k)
    return sorted(out)


def save_cached_bar(root, fs: str, name: str, signature: dict, metrics: dict,
                    diagnostics: dict | None) -> None:
    config.ensure_dir(Path(root))
    config.write_text_atomic(
        json.dumps({"signature": signature, "metrics": _finite_or_none(metrics),
                    # Which metrics were non-finite, recorded at the only point where it is still
                    # knowable — after _finite_or_none they are indistinguishable from an honest None.
                    "nonfinite_metrics": _nonfinite_keys(metrics),
                    "diagnostics": _finite_or_none(diagnostics) if diagnostics else None}, indent=2,
                   allow_nan=False),
        _bar_cache_file(root, fs, name))


def _artifact_stem(n_max) -> str:
    """Published-artifact stem. A ``--n-max`` run scores a DIFFERENT (subsampled) fold, so it gets its own
    stem and can never overwrite the full-fold result — ``fold_comparable=False`` labels a capped run
    honestly but does not undo having destroyed the published table it landed on."""
    return "tabular_baselines" if n_max is None else f"tabular_baselines_nmax{int(n_max)}"


def _qpre_block(train_ds, val_ds) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(train, val, names) covariate blocks. Every fitted statistic — the one-hot vocabulary and the NaN
    impute value — comes from TRAIN only, so no val information reaches the fit."""
    from sklearn.preprocessing import OneHotEncoder

    from tcell_pipeline.encoders.batch import DONOR_COLS

    num = list(_QPRE_NUMERIC) + list(DONOR_COLS)
    check_qpre(num + list(_QPRE_CATEGORICAL), list(_QPRE_OBS))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(train_ds.pc[list(_QPRE_CATEGORICAL)])
    med = train_ds.pc[num].median()                       # control_baseline_expr has NaNs; impute on TRAIN

    def block(ds):
        return np.hstack([ds.pc[num].fillna(med).to_numpy(dtype="float64"),
                          enc.transform(ds.pc[list(_QPRE_CATEGORICAL)]),
                          ds.obs[list(_QPRE_OBS)].to_numpy(dtype="float64")])

    names = num + list(enc.get_feature_names_out(list(_QPRE_CATEGORICAL))) + list(_QPRE_OBS)
    return block(train_ds), block(val_ds), names


def run_tabular_baselines(n_max=None, seed: int = 0, with_tabicl: bool = False,
                          device: str | None = None, device_explicit: bool = True) -> int:
    """feat-006 tabular baselines as an ADDITIONAL H1 comparator bar (like feat-010, CPU/minutes): fit each
    simple baseline on the REAL train fold — predicting Δz from the target gene's static graph node feature —
    score the REAL val fold through the SAME response_metric_suite, and rank vs the frozen H1 on systema.

    Same leakage fence as feat-010: models see TRAIN (feature, response) only; val features are the FROZEN
    static node features (no val response leaks); ``train_mean`` is the train perturbed mean. blocked-target
    OOD means val targets are DISJOINT from train, so this is a genuine generalisation bar, not memorisation.
    These are NOT external comparators: they consume no feat-010 comparator-family cap and are not registered
    in the experiment registry — the deliverable is predictions + the H1-vs-baselines report."""
    from tcell_pipeline.baselines import BASELINES

    graph, g2i = build_hetero_graph()
    train, val = _folds(n_max)
    tr, va = collect_targets_truth(train), collect_targets_truth(val)
    train_mean = dataset_delta_z(train).mean(0)
    B = train.B.numpy()
    Xg = graph["protein"].x.numpy()

    def feats(genes):
        F = np.zeros((len(genes), Xg.shape[1]), dtype=np.float64)
        cov = 0
        for i, g in enumerate(genes):
            j = g2i.get(g)
            if j is not None:
                F[i] = Xg[j]
                cov += 1
        return F, cov

    Xtr_node, ctr = feats(tr["genes"])
    Xva_node, cva = feats(va["genes"])
    Ctr, Cva, cov_names = _qpre_block(train, val)
    feature_sets = {"node": (Xtr_node, Xva_node),
                    "qpre": (np.hstack([Xtr_node, Ctr]), np.hstack([Xva_node, Cva]))}
    print(f"[m8-base] {len(tr['genes'])} train / {len(va['genes'])} val rows; target-in-graph "
          f"{ctr}/{len(Xtr_node)} train, {cva}/{len(Xva_node)} val; node_dim={Xg.shape[1]} "
          f"+{len(cov_names)} q_pre covariates; K={B.shape[1]} G={B.shape[0]}", flush=True)

    bars = _TABULAR + (_TABULAR_GPU if with_tabicl else ())
    cache_root = Path(config.COMPARATORS_ROOT) / f"_bars_{_artifact_stem(n_max)}"
    rows, diagnostics = [], {}
    for fs, (Xtr, Xva) in feature_sets.items():
        for name in bars:
            if fs != "node" and not BASELINES[name].requires_features:
                continue          # X-independent by construction — a second identical row, not a second bar
            if name in _QPRE_ONLY and fs != "qpre":
                continue          # too expensive to score on a feature set no margin is published from
            label = name if fs == "node" else f"{name}_qpre"
            model_name = (f"baseline_{label}" if n_max is None
                          else f"baseline_{label}_nmax{int(n_max)}")
            pred_path = prediction_path(model_name, "val", seed, config.PREDICTIONS_ROOT)
            sig = _bar_signature(n_train=Xtr.shape[0], n_val=Xva.shape[0], n_features=Xtr.shape[1],
                                 k=B.shape[1], bar=name, basis=B, features=Xtr, seed=seed)
            hit = load_cached_bar(cache_root, fs, name, sig, predictions=pred_path)
            if hit is not None:   # resume: this bar already scored under THIS fold and THIS scoring code
                metrics, diag = hit["metrics"], hit["diagnostics"]
                print(f"[m8-base] {label:22s} systema={metrics['systema']:+.4f} [cached]", flush=True)
            else:
                kw = {"device": _gpu_bar_device(device, device_explicit)} if name in _TABULAR_GPU else {}
                model = BASELINES[name](basis=B, **kw)
                # groups = the per-row target symbol: bars that hold rows out internally must move whole
                # target genes, or their selection split shares genes with the rows they fit on.
                model.fit(Xtr, tr["delta_z"], groups=tr["genes"])
                dz, dx = model.predict(Xva)
                metrics = compute_all_metrics(dz, dx, va["delta_z"], va["delta_x"], train_mean)
                write_predictions(va["row_index"], dz, dx, None, split="val", seed=seed,
                                  model=model_name, root=config.PREDICTIONS_ROOT)
                diag = model.fit_diagnostics() if hasattr(model, "fit_diagnostics") else None
                save_cached_bar(cache_root, fs, name, sig, metrics, diag)
                if diag:      # an under-fit bar INFLATES the H1 margin — record it
                    print(f"[m8-base] {label} fit: {diag}", flush=True)
                print(f"[m8-base] {label:22s} systema={metrics['systema']:+.4f} "
                      f"pearson={metrics['pearson']:+.4f}", flush=True)
            rows.append({"name": label, "features": fs, **metrics})
            if diag:
                diagnostics[label] = diag

    stem = _artifact_stem(n_max)
    out = Path(config.COMPARATORS_ROOT) / f"{stem}_val.parquet"
    config.write_parquet_atomic(pd.DataFrame(rows), out)
    print("\n" + " ".join(f"{c:>22}" if c == "name" else f"{c:>9}" for c in _COLS))
    for r in sorted(rows, key=lambda r: -r["systema"]):
        print(" ".join(f"{r['name']:>22}" if c == "name" else f"{r[c]:>9.4f}" for c in _COLS))
    print(f"[m8-base] table -> {out}")

    fold_comparable = n_max is None
    promo_path = Path(config.SCREENING_ROOT) / "promoted.json"
    h1, promo_status = _load_promoted_final(promo_path)
    if promo_status not in ("ok", "absent"):
        print(f"[m8-base] WARNING: promoted.json {promo_status} — H1 verdict skipped", flush=True)
    if h1 and not fold_comparable:
        print(f"[m8-base] WARNING: --n-max={n_max} subsamples the fold; the H1-vs-baseline verdict is "
              f"SKIPPED (not comparable to the full-fold H1)", flush=True)
    summary = summarize_vs_h1(
        rows, h1, noise_margin=_NOISE_MARGIN, fold_comparable=fold_comparable, kind="baseline",
        basis="feat-006 tabular baselines on the development val fold (blocked_target_ood); single-seed; "
              "H1 systema read from promoted.json (frozen, NOT retrained). Two feature sets are scored: "
              "'node' = the target's STATIC graph node feature only (the original bar, which cannot see the "
              "culture condition at all), and '*_qpre' = that plus the q_pre covariates H1's own "
              "perturbation encoder consumes (condition one-hot, donor PCs, PPI degrees, control baseline "
              "expression, guide count). These are NOT feat-010 external comparators and consume no "
              "comparator-family cap. Beating this bar is a trained-predictor win, NOT graph value — the "
              "no-graph expression_only model beats it too.")
    summary["tabular_baselines_val_parquet"] = str(out)
    summary["promoted_json"] = str(promo_path) if promo_path.exists() else None
    summary["promoted_json_status"] = promo_status
    summary["fit_diagnostics"] = diagnostics
    summary["underfit_gate"] = flag_underfit_bars(diagnostics)
    summary["qpre_covariates"] = cov_names
    # off-graph targets get an all-zero feature row indistinguishable from real data; every such val row
    # gets the SAME constant prediction, depressing the feature-regressing baselines and inflating the H1
    # margin. Persist the counts so the margin can be audited from the artifact, not a lost stdout line.
    summary["feature_coverage"] = {
        "train_rows": int(len(Xtr_node)), "train_in_graph": int(ctr),
        "train_off_graph": int(len(Xtr_node) - ctr),
        "val_rows": int(len(Xva_node)), "val_in_graph": int(cva),
        "val_off_graph": int(len(Xva_node) - cva),
        "distinct_train_node_rows": int(len(np.unique(Xtr_node, axis=0))),
        "note": "off-graph targets receive an all-zero node-feature vector (no node features available). "
                "The node block is a function of the TARGET only, so it repeats across conditions — the "
                "'node' bars cannot separate Rest/Stim8hr/Stim48hr at all, which is what the '_qpre' bars "
                "fix."}
    summ_path = Path(config.COMPARATORS_ROOT) / f"{stem}_vs_h1.json"
    config.write_text_atomic(json.dumps(_finite_or_none(summary), indent=2, allow_nan=False), summ_path)
    if summary["h1_beats_strongest"] is not None:
        verdict = "beats" if summary["h1_beats_strongest"] else "does NOT beat"
        noise = " [WITHIN NOISE — single-seed, treat as a tie]" if summary["margin_within_noise"] else ""
        print(f"[m8-base] H1 {summary['frozen_h1']} systema={_fmt_signed(summary['h1_systema'])} {verdict} "
              f"strongest baseline {summary['strongest_comparator']} "
              f"systema={_fmt_signed(summary['strongest_comparator_systema'])} "
              f"(margin {_fmt_signed(summary['margin_h1_minus_strongest'])}){noise}")
    elif h1 and fold_comparable:
        print("[m8-base] frozen H1 present but no eligible (finite-systema) baseline to compare against")
    else:
        print("[m8-base] no H1 verdict emitted — baselines ranked without a comparable frozen H1")
    print(f"[m8-base] H1-vs-tabular-baselines summary -> {summ_path}")

    # The gate is honored all the way to the EXIT CODE: an under-fit floor means the published margin is an
    # upper bound, and an unattended run (or a CI status check) must not record that as a settled result.
    gate = summary["underfit_gate"]
    if gate["margin_is_upper_bound"]:
        print(f"[m8-base] *** MARGIN IS AN UPPER BOUND *** under-fit={gate['underfit'] or 'none'} "
              f"unknown-convergence={gate['unknown'] or 'none'} — {gate['note']}", flush=True)
        return 1
    print("[m8-base] under-fit gate CLEAR: every iterative bar converged, so the margin is not inflated by "
          "a truncated fit.")
    return 0


# --------------------------------------------------------------------------------------------------
def run_audit(n_cases: int, n_controls: int, device: str, n_max=None, untrained: bool = True,
              ckpt=None, head_ckpt=None, seed: int = 0) -> int:
    """feat-012 over the REAL graph.

    With ``--ckpt`` this is the feat-012 campaign: a TRAINED Stage-A model, and ``--head-ckpt`` adds the
    Stage-B-fitted rationale head (without it the head is zero-init, which ranks purely by the frozen
    gate — faithful by construction, but weaker than a fitted head and reported as such). Without
    ``--ckpt`` the model is untrained and the numbers are a machinery check, NOT the feat-012 result.

    ``audit_rationale`` refuses a model whose gates have collapsed — importance is gate * sigmoid(scorer),
    so at ~1e-07 every deletion is a float32 no-op and every contrast is undecidable BY CONSTRUCTION.
    That refusal is honored HERE to the exit code: an undecidable audit must never exit 0."""
    from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder
    from tcell_pipeline.model import EGIPGModel
    from tcell_pipeline.rationale import RationaleHead, audit_rationale

    if not untrained and ckpt is None:
        print("[m8-audit] --no-untrained requires --ckpt naming a trained Stage-A checkpoint")
        return 1
    torch.set_num_threads(1)
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    graph, g2i = build_hetero_graph()
    ds = PerturbationDataset("val", n_max=n_max)
    model = EGIPGModel.from_saved_basis(gene_names, graph_encoder=TypedGraphEncoder(graph, g2i)).eval()
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval()
    head = RationaleHead().eval()
    if head_ckpt is not None:
        head.load_state_dict(torch.load(head_ckpt, map_location=device, weights_only=True)["head"])
        head.eval()
    provenance = ("TRAINED " + str(ckpt)) if ckpt else "UNTRAINED"
    print(f"[m8-audit] {provenance} graph model over the real PPI graph; {len(ds)} val rows; "
          f"head={'fitted ' + str(head_ckpt) if head_ckpt else 'zero-init (ranks by gate)'}; "
          f"n_cases={n_cases} n_controls={n_controls} device={device}", flush=True)
    report = audit_rationale(model, head, ds, n_cases=n_cases, n_controls=n_controls, device=device,
                             seed=seed)
    print(f"[m8-audit] gate mean {report['gate_mean']} -> gates "
          f"{'LIVE' if report['gates_live'] else 'COLLAPSED'}")
    if report["undecidable_by_construction"]:
        print("[m8-audit] UNDECIDABLE BY CONSTRUCTION: the gates have collapsed, so every deletion is a "
              "float32 no-op and no faithfulness contrast can be formed. This is NOT a failed audit and "
              "NOT a negative result — nothing was measured. Audit a checkpoint with live gates.")
        print(f"[m8-audit] report -> {report['report_path']}")
        return 1
    agg = report["aggregate"]
    print(f"[m8-audit] audited={report['n_audited']} uncovered_in_fold={report['n_uncovered_in_dataset']}")
    print(f"[m8-audit] frac sufficiency<random = {agg['frac_sufficiency_below_random']}")
    print(f"[m8-audit] frac necessity>random   = {agg['frac_necessity_above_random']}")
    print(f"[m8-audit] mean minimality={agg['mean_minimality']} mean stability={agg['mean_stability']}")
    print(f"[m8-audit] source ablation Δ: {agg['source_ablation_delta_sufficiency']}")
    print(f"[m8-audit] report -> {report['report_path']}")
    if ckpt is None:
        print("[m8-audit] NOTE: model is UNTRAINED — this validates the audit path at real scale; the "
              "feat-012 campaign needs a trained checkpoint whose gates are alive.")
    return 0


# --------------------------------------------------------------------------------------------------
def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_repro() -> int:
    """feat-013 against this checkout — delegated to ``reproducibility.run_repro_real``.

    This function used to build the manifest itself, and that duplicate had two defects (the first found
    by the feat-013 session, the second by the test that now pins it):

    * Its hash entries carried no ``provenance`` field. ``verify.py`` treats an unlabelled entry as
      self-derived and downgrades it from pass to incomplete — so the run reported ``hash:splits`` as
      "compared against itself" while this same function printed "independently-checked artifacts:
      ['splits']". One invocation, two contradictory claims.
    * It printed ``VERDICT = CANNOT_VERIFY`` and then ``return 0``, so an unattended run or an exit-status
      CI gate recorded an unverifiable reproduction as success — the guard-not-honored-to-the-exit-code
      failure this repo already fixed once in the multiseed campaign.

    ``run_repro_real`` owns the manifest, labels provenance correctly, and returns ``exit_code(verdict)``
    (0 only for REPRODUCIBLE). Deleting the copy is the fix: two implementations of one contract is how the
    two drifted apart in the first place."""
    from tcell_pipeline.reproducibility import run_repro_real

    # Pass argv explicitly. This call originally read `main()`, which back then defaulted to argv=None and
    # so let argparse fall through to sys.argv — still carrying THIS driver's `--part repro`, which died
    # with "unrecognized arguments". The feat-013 session fixed it at the source (`main(argv=())`, with
    # `sys.argv[1:]` passed only from its own `__main__`), so a bare call is safe now; the explicit empty
    # list is kept because this caller genuinely wants that module's DEFAULTS, not this process's flags.
    return run_repro_real.main([])

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["comparators", "baselines", "audit", "repro", "all"])
    ap.add_argument("--n-max", type=int, default=None, help="cap rows per split (quick runs)")
    ap.add_argument("--n-cases", type=int, default=config.N_RATIONALE_AUDIT_CASES)
    ap.add_argument("--n-controls", type=int, default=10, help="matched-random controls per audited case")
    ap.add_argument("--device", default=None,
                    help="torch device for GPU-capable parts; default cpu, except GPU-only bars which "
                         "auto-select when this is not given explicitly")
    ap.add_argument("--untrained", action=argparse.BooleanOptionalAction, default=True,
                    help="allow auditing an UNTRAINED model (machinery check at real scale). Pass "
                         "--no-untrained to REQUIRE a trained checkpoint via --ckpt (the guard then fires "
                         "if none is given). Default lets a --ckpt-less run proceed as an untrained probe.")
    ap.add_argument("--ckpt", default=None,
                    help="trained Stage-A checkpoint to audit (feat-012 campaign). Without it the audited "
                         "model is untrained and the run is only a machinery check.")
    ap.add_argument("--head-ckpt", default=None,
                    help="Stage-B-fitted rationale head (stage_b_rationale_head.pt). Without it the head "
                         "is zero-init, which ranks edges purely by the frozen gate.")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--with-tabicl", action="store_true",
                    help="add the TabICL in-context bar (GPU; refit once per program)")
    a = ap.parse_args()
    device_explicit = a.device is not None
    device = a.device if device_explicit else "cpu"
    rc = 0
    if a.part in ("comparators", "all"):
        rc |= RC_COMPARATORS if run_comparators(a.n_max, register=not a.no_register) else 0
    if a.part in ("baselines", "all"):
        rc |= RC_BASELINES if run_tabular_baselines(
            a.n_max, with_tabicl=a.with_tabicl, device=device,
            device_explicit=device_explicit) else 0
    if a.part in ("audit", "all"):
        rc |= RC_AUDIT if run_audit(a.n_cases, a.n_controls, device, a.n_max, a.untrained,
                                    ckpt=a.ckpt, head_ckpt=a.head_ckpt) else 0
    if a.part in ("repro", "all"):
        rc |= RC_REPRO if run_repro() else 0
    if rc:
        print(f"[m8] exit {rc} = {_rc_legend(rc)}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
