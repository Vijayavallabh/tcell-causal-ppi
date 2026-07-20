"""Build a feat-013 reproducibility manifest from THIS checkout's REAL frozen artifacts, and author the
eleven fallacy probes from THIS project's REAL diagnostics (report §reproducibility).

Two things this module refuses to do, because doing either would make the verifier certify a check nobody
performed:

1. **It never presents a self-derived hash as a reproduction test.** Only ``data/splits/manifest.json``
   published a sha256 at freeze time, so only ``splits`` carries an INDEPENDENT expected hash; hashing
   ``id_mapping`` / ``de_layers`` today and comparing them to themselves shows the files are readable and
   nothing more. Those entries are labelled ``self-derived`` and ``verify._check_hashes`` downgrades them to
   ``incomplete`` — see ``verify.INDEPENDENT``.
2. **It never invents a probe input.** A detector with no real input in this checkout is returned in
   ``unevaluable`` WITH ITS REASON rather than fed a toy array or dropped silently (dropping loses the
   reason; a toy array manufactures a verdict). Four of the eleven land there, and the reasons name exactly
   what artifact would unlock them.

The same rule governs the config check. Hashing today's config and comparing it to itself always passes — a
guard whose input is a constant can only confirm. The expected value is therefore taken from the *frozen*
``data/splits/manifest.json``, which recorded the split seed, fractions, sequence-similarity threshold and
group-size cap when the split was cut; editing any of those in ``config.py`` now makes the check FAIL.

``decision`` / ``observed`` are deliberately absent: the confirmatory decision lives on the sequestered
challenge split and only the test steward may produce it. ``unverified`` records that, so the gap is a named
remainder rather than a silence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean

import os

from tcell_pipeline import config
from tcell_pipeline.reproducibility.verify import INDEPENDENT, SELF_DERIVED, _as_dict, _sha256_file

# the seed the single-seed screening promoted from (promoted.json ``final.seed``); the retest is every OTHER
# seed, which is what makes the regression-to-the-mean probe a retest rather than a re-read of the selection
SELECTION_SEED = 0
# the two pre-registered contrasts that estimate the SAME quantity — "does the graph beat no-graph?" — under
# two admissible choices of which arm counts as "the graph". A sign flip between them is the fork.
_FORK_CONTRASTS = ("promotion_margin", "h1_vs_no_graph")
# mirrors ``fallacy_scan.correlation_not_causation``'s own ``threshold`` default; pinned equal by
# ``test_cnc_threshold_tracks_the_detector`` so the refusal below cannot silently drift away from it
CNC_THRESHOLD = 0.3

# Probes with no real input in this checkout. Stated as facts about what is missing, not as excuses: each
# names the artifact that would make it evaluable. An honest "not applicable here" beats a fabricated input.
UNEVALUABLE_REASONS: dict[str, str] = {
    "base_rate":
        "NOT a lack of evidence — the blocker is the detector's SHAPE, and the probe does fire. Val truth "
        "is on disk: data/intermediate/de_layers/zscore.npz is (33,983 x 10,282) keyed by the same "
        "row_index the prediction parquets carry, and training/dataset.py already slices it that way. This "
        "session performed the join and validated it — it reproduces condition_gated seed 0's recorded "
        "topk (0.012807), sign (0.507814) and mae (0.815549) EXACTLY — and base_rate on the study's own "
        "top-20 DE call then FLAGS: accuracy 0.9962, precision 0.012807 (identical to the published topk), "
        "prevalence 0.0019. It is not authored here only because base_rate takes ARRAYS and the label "
        "array is 4,400 x 10,282 ~ 45M entries, which cannot be carried as literal kwargs in a frozen "
        "manifest. The result is recorded in docs/feat013-repro-notes.md rather than dropped. UNLOCKED BY: "
        "a confusion-matrix entry point on base_rate, or a manifest that may reference a derived artifact.",
    "berkson":
        "DEFERRED, not impossible — do not read this as 'no evidence exists'. Both per-row inputs are "
        "derivable: the off-graph mask from data/graphs/protein_edges.parquet (the published 385 train / "
        "91 val counts in comparators/tabular_baselines_vs_h1.json:feature_coverage come from exactly that "
        "derivation, run_module8_real.py), and a per-row metric from evaluation.metrics._rowwise_pearson. "
        "At 4,400 rows they would even fit in a manifest. What is missing is the derived artifact itself, "
        "which this session did not compute. The RUN-level selections that do exist cannot substitute: "
        "full-epoch-budget completion is a deterministic function of one of the variables (range "
        "restriction, not conditioning on a common effect), and registry completion excludes 3 runs whose "
        "metrics are empty, so they carry no y at all. UNLOCKED BY: a per-row table joining the off-graph "
        "mask to a row-level metric.",
    "collider":
        "no run-level variable here is a plausible COMMON EFFECT of training budget and accuracy that is "
        "measured independently of both. best_val — the model-selection criterion, and the natural "
        "candidate — is recorded on two incommensurable scales (~3.47 for expression_only / untyped_gnn / "
        "condition_gated vs ~490 for typed_static), and sub-setting to the comparable arms after seeing "
        "the numbers would itself be a fork. The other run-level numerics ARE on one scale (pearson, "
        "prog_cos, mae, rmse, topk, sign, centroid, gpu_hours — an earlier version of this reason wrongly "
        "said best_val was the only candidate), but they are not independent of the outcome: pearson and "
        "prog_cos correlate 0.973 with systema, so conditioning on them conditions on the outcome itself. "
        "UNLOCKED BY: a val loss on one scale across all arms, or a per-row covariate.",
    "reverse_causation":
        "needs a cross-lagged pair — corr(x_t -> y_t+1) against corr(y_t -> x_t+1). This is a "
        "single-timepoint design: the three culture conditions (Rest / Stim8hr / Stim48hr) are "
        "experimental arms, not repeated measures on the same units, and no lagged association is computed "
        "anywhere in the pipeline. UNLOCKED BY: a longitudinal panel, which this dataset does not contain.",
}


def _num(v) -> bool:
    """A real, finite number — rejecting bool (``isinstance(True, int)``) and NaN, either of which would ride
    into a detector and produce a statistic with no meaning.

    Numpy scalars are accepted deliberately. ``np.float64`` subclasses ``float`` but ``np.int64`` does NOT
    subclass ``int``, and parquet reads come back as numpy scalars — ``screening/multiseed.py::_finite``
    carries the same note because a guard that rejected them once silently dropped every seed. Here it would
    have emptied the run table and reported the run-level probes as "no data" rather than as a defect."""
    import numpy as np
    return (isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, (bool, np.bool_)) and v == v and abs(v) != float("inf"))


def config_hash(snapshot: dict) -> str:
    """Exactly ``verify._check_config``'s hashing, so the manifest and the verifier cannot drift apart."""
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest()


def _frozen_split_record(root: Path | None = None) -> dict:
    path = (Path(root) / "data" / "splits" / "manifest.json") if root else config.SPLIT_MANIFEST_PATH
    try:
        record = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return record if isinstance(record, dict) else {}


def frozen_config_snapshot(root: Path | None = None) -> dict | None:
    """The config values that were INDEPENDENTLY recorded when the split was frozen, or None if that record
    is unreadable. These four are the only config knobs in this checkout with an external frozen witness, so
    they are the only ones a config check can genuinely reproduce."""
    rec = _frozen_split_record(root)
    try:
        return {"SPLIT_SEED": int(rec["seed"]),
                "SPLIT_FRACTIONS": {k: float(v) for k, v in rec["fractions_target"].items()},
                "SEQ_SIM_COSINE_THRESHOLD": float(rec["seq_cosine_threshold"]),
                "GROUP_SIZE_CAP": float(rec["group_size_cap_frac"])}
    except Exception:
        return None


def live_config_snapshot() -> dict:
    """Today's values for the same four knobs. Read off ``config`` at CALL time so a drift is visible."""
    return {"SPLIT_SEED": int(config.SPLIT_SEED),
            "SPLIT_FRACTIONS": {k: float(v) for k, v in config.SPLIT_FRACTIONS.items()},
            "SEQ_SIM_COSINE_THRESHOLD": float(config.SEQ_SIM_COSINE_THRESHOLD),
            "GROUP_SIZE_CAP": float(config.GROUP_SIZE_CAP)}


# --------------------------------------------------------------------------------------------------------
# real diagnostics
# --------------------------------------------------------------------------------------------------------

def load_diagnostics(root: Path | None = None) -> dict:
    """Read the REAL diagnostic artifacts this checkout published. Nothing is computed or simulated here —
    every number is read off disk. Missing artifacts leave their slot empty and the probes that need them
    become Unevaluable; they are never defaulted to a value."""
    import pandas as pd

    root = Path(root) if root else config.PROJECT_ROOT
    screening, sources = root / "data" / "results" / "screening", {}

    runs: list[dict] = []
    for parquet in sorted(screening.glob("*/*.parquet")):
        try:
            frame = pd.read_parquet(parquet)
        except Exception:
            continue
        for row in frame.to_dict("records"):
            runs.append({k: (float(row[k]) if _num(row.get(k)) else None)
                         for k in ("systema", "pearson", "epochs_run", "n_epochs")}
                        | {"name": str(row.get("name")),
                           "seed": int(row["seed"]) if _num(row.get("seed")) else None})
    if runs:
        sources["runs"] = str(screening.relative_to(root) / "*" / "*.parquet")

    contrasts, family_size = {}, 0
    robustness = screening / "robustness_5seed.json"
    if robustness.is_file():
        try:
            record = json.loads(robustness.read_text())
            contrasts = record.get("contrasts") or {}
            family_size = max((c.get("family_size") or 0 for c in contrasts.values()), default=0)
            sources["contrasts"] = str(robustness.relative_to(root))
        except Exception:
            contrasts, family_size = {}, 0

    h1_systema, promoted = None, screening / "promoted.json"
    if promoted.is_file():
        try:
            final = (json.loads(promoted.read_text()) or {}).get("final") or {}
            h1_systema = float(final["systema"]) if _num(final.get("systema")) else None
            sources["h1"] = str(promoted.relative_to(root))
        except Exception:
            h1_systema = None

    return {"runs": sorted(runs, key=lambda r: (r["name"], r["seed"] if r["seed"] is not None else -1)),
            "contrasts": contrasts, "family_size": int(family_size), "h1_systema": h1_systema,
            "selection_seed": SELECTION_SEED,
            # the endpoint is scored on a TARGET-BLOCKED fold, i.e. on CRISPR perturbations of genes the
            # model never trained on — a real, checkable fact about the frozen split, not an assertion
            "blocked_target_fold": config.BLOCKED_SPLIT_PATH.name == "blocked_target_ood.csv",
            "sources": sources}


_RUN_FIELDS = ("systema", "epochs_run", "n_epochs")


def usable_runs(runs) -> list[dict]:
    """Runs carrying every field the run-level probes read, coerced to Python natives. Drops e.g.
    ``network_propagation``, a scored baseline with no training budget recorded — including it would put a
    None into an epochs array.

    The coercion is defence in depth, NOT the load-bearing guard — do not rely on it. The hazard it targets
    (a numpy scalar making ``epochs_run == n_epochs`` an ``np.bool_``, which ``json.dumps(default=str)``
    writes to the manifest as the STRING ``"True"``) is actually prevented by the explicit ``bool()`` at the
    one site that computes it; and every other value that leaks through uncoerced is ``np.float64``, which
    subclasses ``float`` and serialises correctly. Removing this coercion is therefore unobservable today —
    it exists so the "python natives everywhere" invariant holds at one choke point for fields added later.
    The guard that is load-bearing is ``_num`` accepting ``np.integer``: without it every run is dropped."""
    runs = runs if isinstance(runs, (list, tuple)) else []
    return [r | {"seed": int(r["seed"]), **{k: float(r[k]) for k in _RUN_FIELDS}}
            for r in runs if isinstance(r, dict)
            and _num(r.get("seed")) and all(_num(r.get(k)) for k in _RUN_FIELDS)]


def build_fallacy_inputs(diag: dict) -> tuple[dict, dict]:
    """``(inputs, unevaluable)`` — the eleven probes, authored from ``diag``. Pure: no disk, no globals, so
    a caller can hand it a constructed diagnostic set and watch each probe fire or refuse.

    Every one of the eleven ends up in exactly one of the two dicts. A probe whose real input would make the
    detector arithmetically incapable of flagging is refused, not authored: that is decoration, not a check.
    """
    inputs: dict[str, dict] = {}
    unevaluable: dict[str, str] = dict(UNEVALUABLE_REASONS)
    runs = usable_runs(diag.get("runs") or [])
    # wrong-typed diagnostics must partition, not raise: this function is documented as pure and callable
    # with a CONSTRUCTED diagnostic set, which is exactly how an adversarial caller reaches it
    contrasts = {k: v for k, v in _as_dict(diag.get("contrasts")).items() if isinstance(v, dict)}
    family_size = int(diag["family_size"]) if _num(diag.get("family_size")) else 0

    # --- look-elsewhere: the campaign's SIMULTANEOUS family, entire ---------------------------------------
    pvalues = [float(c["p_value"]) for c in contrasts.values() if _num(c.get("p_value"))]
    if family_size >= 2 and len(pvalues) == family_size:
        alphas = {float(c["alpha"]) for c in contrasts.values() if _num(c.get("alpha"))}
        inputs["look_elsewhere"] = {"pvalues": pvalues,
                                    "alpha": alphas.pop() if len(alphas) == 1 else 0.05}
    else:
        unevaluable["look_elsewhere"] = (
            f"the recorded family size is {family_size} but only {len(pvalues)} contrast(s) carry a "
            f"p-value. Scoring the survivors against a SMALLER m would understate the very correction this "
            f"probe exists to apply, and a single p-value makes it arithmetically incapable of flagging "
            f"(alpha/1 == alpha). UNLOCKED BY: a p-value for every simultaneously-tested contrast.")

    # --- garden of forks: one estimand, two admissible arm choices ----------------------------------------
    estimates = [float(contrasts[k]["mean"]) for k in _FORK_CONTRASTS
                 if k in contrasts and _num(contrasts[k].get("mean"))]
    if len(estimates) == len(_FORK_CONTRASTS):
        inputs["garden_of_forks"] = {"estimates": estimates}
    else:
        unevaluable["garden_of_forks"] = (
            f"needs both graph-vs-no-graph contrasts {_FORK_CONTRASTS} to compare the same estimand across "
            f"analysis choices; only {len(estimates)} carries a mean. A single estimate has no spread to "
            f"assess. UNLOCKED BY: running both contrasts in the multi-seed campaign.")

    # --- regression to the mean: select on the screening seed, retest on the others -----------------------
    selection_seed = diag.get("selection_seed", SELECTION_SEED)
    baseline = {r["name"]: r["systema"] for r in runs if r["seed"] == selection_seed}
    retest: dict[str, list[float]] = {}
    for r in runs:
        if r["seed"] != selection_seed:
            retest.setdefault(r["name"], []).append(r["systema"])
    paired = sorted(n for n in baseline if retest.get(n))
    if len(paired) >= 2:
        inputs["regression_to_mean"] = {"baseline": [baseline[n] for n in paired],
                                        "followup": [fmean(retest[n]) for n in paired]}
    else:
        unevaluable["regression_to_mean"] = (
            f"needs >=2 configs scored BOTH on the selection seed ({selection_seed}) and on at least one "
            f"other seed, so the retest is independent of the selection; {len(paired)} qualify. UNLOCKED "
            f"BY: multi-seed runs of the screened family.")

    # --- Simpson + ecological: does the pooled budget->accuracy trend survive within arms? -----------------
    # The typed-edge arms (typed_static, condition_gated) early-stop at 11-13 of 20 epochs while
    # expression_only and untyped_gnn run all 20. Simpson asks whether pooling REVERSES the within-arm
    # trend, ecological whether aggregating inflates it.
    # READ THE RESULT NARROWLY. `epochs_run` is 98.8% explained by the arm label (between-arm SS 361.4 of
    # 365.8) and is CONSTANT within two of the four arms, so those two contribute no within-trend at all and
    # the "aggregate" correlation is 4 points on a tied x. It is also the OUTCOME of early stopping on val
    # loss, i.e. a descendant of model quality, not a treatment. A clean non-reversal here is weak evidence,
    # and it does not survive a change of budget proxy: under gpu_hours the pooled trend is NEGATIVE
    # (-0.534) with all four within-arm trends negative. Neither probe flags under either proxy.
    # Do NOT read this as "the graph arms simply trained less": untyped_gnn IS a graph arm, it used the full
    # budget, and it posts the highest 5-seed systema (0.0902). expression_only is the only no-graph arm.
    grouped: dict[str, tuple[list, list]] = {}
    for r in runs:
        epochs, systema = grouped.setdefault(r["name"], ([], []))
        epochs.append(r["epochs_run"])
        systema.append(r["systema"])
    groups = [[epochs, systema] for _, (epochs, systema) in sorted(grouped.items()) if len(epochs) >= 2]
    if len(groups) >= 2:
        inputs["simpson"] = {"groups": groups}
    else:
        unevaluable["simpson"] = (
            f"the paradox IS a pooled-vs-within disagreement, so it needs >=2 arms with >=2 runs each; "
            f"{len(groups)} qualify. UNLOCKED BY: multi-seed runs of the screened family.")
    if len(grouped) >= 3:
        inputs["ecological"] = {"x": [r["epochs_run"] for r in runs], "y": [r["systema"] for r in runs],
                                "group": [r["name"] for r in runs]}
    else:
        unevaluable["ecological"] = (
            f"the correlation of <=2 group means is degenerately +-1, so this needs >=3 arms; "
            f"{len(grouped)} present. UNLOCKED BY: scoring the whole screened family.")

    # --- survivorship: the metric over full-budget runs vs over every run ---------------------------------
    # `survived` is COLLINEAR with the arm label here (zero within-arm variance: two arms always finish,
    # two always early-stop), so a flag says "the full-budget subset reports a higher mean", NOT "the
    # budget caused it". That is still a real reporting hazard — survivorship asks whether restricting the
    # sample CHANGES the reported number, which collinearity does not invalidate. The same collinearity IS
    # fatal to berkson, which asks a causal question (does selection INDUCE association?), and that is why
    # it is refused above rather than authored on the same mask.
    survived = [bool(r["epochs_run"] == r["n_epochs"]) for r in runs]
    if any(survived) and not all(survived):
        inputs["survivorship"] = {"values": [r["systema"] for r in runs], "survived": survived}
    else:
        unevaluable["survivorship"] = (
            f"needs both survivors and non-survivors: {sum(survived)} of {len(survived)} runs used their "
            f"full epoch budget. With every run on one side of the split the survivor-only metric IS the "
            f"full-population metric, so the check could only ever confirm. UNLOCKED BY: a run set mixing "
            f"early-stopped and full-budget runs.")

    # --- correlation != causation -------------------------------------------------------------------------
    # ALWAYS REFUSED here, by the same rule that refuses a one-p-value look_elsewhere. The detector flags
    # only when |corr| >= threshold AND interventional support is ABSENT, and BOTH conditions are out of
    # reach in this study — so there is no headline association that would make it a live check.
    # An earlier version refused it only below the threshold and authored it above; that authored branch
    # would have re-created the very decoration the refusal exists to avoid, because `support` stays True.
    support = bool(diag.get("blocked_target_fold"))
    if not _num(diag.get("h1_systema")):
        unevaluable["correlation_not_causation"] = (
            "no frozen headline association to test: promoted.json carries no finite systema for the "
            "confirmatory model.")
    else:
        unevaluable["correlation_not_causation"] = (
            f"the probe cannot fire in this study, so authoring it would be decoration. Two independent "
            f"reasons: (1) the headline association is {float(diag['h1_systema']):.4f} (frozen H1 systema; "
            f"its Pearson is 0.113), well below the detector's {CNC_THRESHOLD} threshold; and (2) "
            f"has_interventional_support is {support} as a STRUCTURAL fact, not a reading of an artifact — "
            f"the endpoint is scored on a target-blocked fold of a CRISPR screen, and the derivation "
            f"available here (config.BLOCKED_SPLIT_PATH.name) is a hardcoded literal, so no run of this "
            f"pipeline can make it False. With support present the detector returns 'not flagged' for "
            f"EVERY correlation. UNLOCKED BY: an endpoint measured WITHOUT interventional support — i.e. "
            f"this detector is for observational claims, and this study is interventional by construction. "
            f"The PPI edges themselves ARE observational, so an edge-level causal claim would need the "
            f"rationale audit (feat-009), not this probe.")

    for name in inputs:
        unevaluable.pop(name, None)
    return _demote_dead_probes(inputs, unevaluable)


def _demote_dead_probes(inputs: dict, unevaluable: dict) -> tuple[dict, dict]:
    """Run each authored probe and move any that cannot actually evaluate into ``unevaluable``, carrying the
    detector's own reason.

    The guards above are SHAPE preconditions (enough arms, enough seeds, a p-value per contrast); the
    detectors enforce VARIANCE ones (a series must not be constant, a p must lie in [0, 1]). When they
    disagreed the probe was authored, raised ``Unevaluable`` inside the scan, and vanished from BOTH lists —
    it is in ``inputs``, so it had already been popped from ``unevaluable``. A checkout where nobody
    early-stopped (constant ``epochs_run``) printed "no trap detected (2/11)" with nothing naming simpson or
    ecological as unexamined. Enumerating every such disagreement by hand is the losing game; asking the
    detector is the cheap, total answer.

    Only ``Unevaluable`` demotes. Any other exception is a detector BUG and must stay visible in the scan's
    ``crashed`` list rather than being laundered here into 'inadequate input'."""
    from tcell_pipeline.reproducibility.fallacy_scan import _DETECTORS, Unevaluable

    for name in list(inputs):
        try:
            _DETECTORS[name](**inputs[name])
        except Unevaluable as exc:
            unevaluable[name] = (f"authored from real diagnostics, but the detector cannot evaluate it: "
                                 f"{exc}. UNLOCKED BY: diagnostics with the variation this statistic needs.")
            inputs.pop(name)
        except Exception:
            pass
    return inputs, unevaluable


# --------------------------------------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------------------------------------

def _artifacts(root: Path) -> dict[str, Path]:
    """The frozen DETERMINISTIC preprocessing artifacts, resolved INSIDE ``root``. Live checkpoints under
    data/checkpoints/ are deliberately excluded — they are written during training, so hashing one races a
    half-written file.

    ``config``'s roots are absolute and env-overridable, so each path is re-anchored on ``root``: using them
    as-is would hash the ORIGINAL run's files while claiming to have verified a different checkout, which is
    the whole failure mode ``verify._resolve`` exists to block. A path that cannot be expressed relative to
    the project (an env override pointing outside it) is dropped, and verify then reports it ``missing``."""
    artifacts = {}
    for name, path in (("id_mapping", config.ID_MAPPING_PATH), ("splits", config.BLOCKED_SPLIT_PATH),
                       ("de_layers", config.DE_LAYERS_DIR / "zscore.npz")):
        # normpath FIRST: relative_to is purely lexical, so an env root of the form <project>/../<project>/…
        # satisfies it and carries the `..` straight through into the published manifest path
        rel = os.path.relpath(os.path.normpath(path), config.PROJECT_ROOT)
        if rel.startswith(os.pardir) or os.path.isabs(rel):
            continue
        artifacts[name] = Path(root) / rel
    return artifacts


def build_hashes(root: Path) -> dict:
    """Hash each frozen artifact and label WHERE its expected value came from. Only ``data/splits/
    manifest.json`` published hashes at freeze time, so only artifacts named there can be independent."""
    # _as_dict, not `or {}`: the latter guards a FALSY sha256 field but not a wrong-typed one, so a
    # truncated data/splits/manifest.json raised AttributeError out of build_manifest and run_repro_real
    # produced no report at all — the same hole _as_dict exists to close in verify.py
    frozen = _as_dict(_frozen_split_record(root).get("sha256"))
    hashes = {}
    for name, path in _artifacts(root).items():
        path = Path(path)
        try:
            rel = str(path.relative_to(root))     # MUST be relative: an absolute path would send the
        except ValueError:                        # verifier at the ORIGINAL run's files, not the checkout's
            continue
        expected = frozen.get(path.name) if isinstance(frozen.get(path.name), str) else None
        actual = _sha256_file(path)
        if expected is None and actual is None:
            continue                              # nothing to declare; verify emits an explicit `missing`
        hashes[name] = {"path": rel, "sha256": expected or actual,
                        "provenance": INDEPENDENT if expected else SELF_DERIVED,
                        "source": ("data/splits/manifest.json, published when the split was frozen"
                                   if expected else
                                   "hashed from this checkout now — no frozen record exists for it")}
    return hashes


def _val_rows_from_frozen_split(root: Path) -> int | None:
    """Number of val rows, derived from the FROZEN split — the git-tracked ``blocked_target_ood.csv`` joined
    to the perturbation table. None if either is unreadable.

    This is the authority for the row count. It used to be read out of
    ``data/results/comparators/tabular_baselines_vs_h1.json``, which a concurrently-running session
    rewrites: if that run's feature handling changed, the expected count would have moved with it and this
    probe would have followed SILENTLY. Presence is not freshness, and a "check" that tracks whatever
    another session last wrote is not a check."""
    import pandas as pd

    try:
        table = pd.read_parquet(root / "data" / "intermediate" / "perturbation_condition.parquet",
                                columns=["hgnc_symbol"])
        split = pd.read_csv(root / "data" / "splits" / "blocked_target_ood.csv")
        roles = table["hgnc_symbol"].map(dict(zip(split["hgnc_symbol"], split["role"])))
        return int((roles == "val").sum())
    except Exception:
        return None


def _predictions(root: Path) -> dict:
    """One real prediction table, row-count-checked against the FROZEN split.

    Declared only when that count can be established. ``verify`` treats ``n_rows: None`` as vacuously
    satisfied, so declaring the table without a count would emit a `pass` for a row check that never ran —
    the same vacuous-truth hole as ``set() <= {expected}``.

    Session A's ``feature_coverage.val_rows`` is still read, but only as a CROSS-CHECK that must agree.
    Two independent sources disagreeing about the fold means the row count is unknown, so the entry is
    refused (-> verify says `missing` -> CANNOT_VERIFY) rather than picking a winner."""
    path = root / "data" / "results" / "predictions" / "perturbed_mean" / "val" / "0.parquet"
    n_rows = _val_rows_from_frozen_split(root)
    if not path.is_file() or not isinstance(n_rows, int) or n_rows <= 0:
        return {}
    coverage = root / "data" / "results" / "comparators" / "tabular_baselines_vs_h1.json"
    recorded = None
    if coverage.is_file():
        try:
            recorded = _as_dict(json.loads(coverage.read_text()).get("feature_coverage")).get("val_rows")
        except Exception:
            recorded = None
        if not isinstance(recorded, int) or isinstance(recorded, bool):
            recorded = None
        elif recorded != n_rows:
            return {}                     # the two sources disagree -> we do not know the row count
    return {"perturbed_mean_val": {
        "path": str(path.relative_to(root)), "n_rows": n_rows,
        "n_rows_source": "derived from the frozen split (data/splits/blocked_target_ood.csv joined to "
                         "data/intermediate/perturbation_condition.parquet) — git-tracked and frozen, so "
                         "it cannot drift with another session's results",
        "cross_check": {"artifact": "data/results/comparators/tabular_baselines_vs_h1.json:"
                                    "feature_coverage.val_rows",
                        "val_rows": recorded, "agrees": recorded == n_rows if recorded is not None else None,
                        "mtime": coverage.stat().st_mtime if coverage.is_file() else None}}}


UNVERIFIED = {
    "confirmatory_decision": {
        "status": "absent",
        "reason": "the confirmatory H1 decision is defined on the sequestered challenge split (5,608 rows), "
                  "which is UNOPENED. No decision record exists to reproduce, so the reproduction verdict "
                  "is CANNOT_VERIFY — not a failure, and not something an agent session may resolve.",
        "who": "the test steward — opening the sealed split is steward-only, and every artifact in this "
               "repository is DEVELOPMENT-fold evidence.",
        "how": "run evaluation/sealed_eval.py ONCE on the sealed challenge split, publish its decision as "
               "manifest.decision (h1_confirmed plus at least one of lcb_95 / rho_egipg / delta_vs_best) "
               "and the re-run's as manifest.observed.decision, then re-run this verifier.",
    },
    "independent_artifact_hashes": {
        "status": "partial",
        "reason": "only data/splits/manifest.json published sha256 at freeze time, so only `splits` is "
                  "checked against an independent record. id_mapping and de_layers are self-derived: they "
                  "are shown readable and internally stable, NOT reproduced.",
        "who": "whoever cuts the next freeze",
        "how": "publish a sha256 for every deterministic preprocessing artifact at the moment it is frozen, "
               "then a later checkout's hashes become genuine reproduction tests.",
    },
}


def build_manifest(root: Path | None = None) -> dict:
    """The manifest for THIS checkout. ``decision`` / ``observed`` are absent by design — see UNVERIFIED."""
    root = Path(root) if root else config.PROJECT_ROOT
    diag = load_diagnostics(root)
    inputs, unevaluable = build_fallacy_inputs(diag)
    frozen_config = frozen_config_snapshot(root)
    manifest = {
        "hashes": build_hashes(root),
        "predictions": _predictions(root),
        "fallacy_inputs": inputs,
        "fallacy_unevaluable": unevaluable,
        "fallacy_sources": diag["sources"],
        "unverified": UNVERIFIED,
    }
    if frozen_config is not None:
        manifest["config_hashes"] = {"config_snapshot": config_hash(frozen_config),
                                     "source": "data/splits/manifest.json, recorded when the split was cut "
                                               "— an independent witness, so a config drift FAILS this"}
    return manifest
