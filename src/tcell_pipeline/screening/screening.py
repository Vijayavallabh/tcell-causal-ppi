"""Screening harness (feat-011): train, evaluate, and compare the §10.6 nested confirmatory family on a
development split, all through the existing Stage-A ``Trainer`` + ``PerturbationDataset`` + evaluation
metrics (report §screening, walkthrough §10.6-10.7).

Nested family (each adds exactly one architectural element):

  1. expression_only   EGIPGModel(graph_encoder=None)             — no graph
  2. typed_static      EGIPGModel(StaticTypedGraphEncoder)        — typed graph, condition gate pinned to 1
  3. condition_gated   EGIPGModel(TypedGraphEncoder)              — + condition gating (== full EG-IPG here)

``screen_config`` trains one config, reloads its best checkpoint, scores the val fold, writes predictions in
the common output schema, and persists a one-row metrics table. ``run_screening`` runs several configs and
reports the two key-secondary contrasts on the primary endpoint (``systema_pert_specific_delta``):

  H2a — does typed static beat expression-only?      (member 2 vs 1)
  H2b — does condition gating beat typed static?     (member 3 vs 2)

The untyped-graph diagnostic (``UntypedGraphEncoder``) is not a member of the H2a/H2b *confirmatory* nested
family, so the driver runs it in the wave but excludes it from the H2a/H2b comparison. It IS still an
internal EG-IPG ablation, so it registers under ``family='egipg'`` and counts against the 32-trial EG-IPG
budget (report §1291: "32 across the entire EG-IPG family, not per ablation"). The separate 16-per-family
comparator budget is for EXTERNAL trainable comparators (TxPert-public / Stable-Shift — feat-010), not for
these in-family ablations.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcell_pipeline import config
from tcell_pipeline.baselines.graph_baselines import StaticTypedGraphEncoder, UntypedGraphEncoder
from tcell_pipeline.evaluation import metrics as M
from tcell_pipeline.evaluation.output_schema import write_predictions
from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder
from tcell_pipeline.model import EGIPGModel
from tcell_pipeline.screening.experiment_registry import log_run, register_run
from tcell_pipeline.training.dataset import PerturbationDataset
from tcell_pipeline.training.trainer import Trainer, seeded_init

EXPRESSION_ONLY = "expression_only"
TYPED_STATIC = "typed_static"
CONDITION_GATED = "condition_gated"
UNTYPED_GNN = "untyped_gnn"
NETWORK_PROP = "network_propagation"
PRIMARY_METRIC = "systema"  # systema_pert_specific_delta — the locked H1 primary endpoint


# --------------------------------------------------------------------------------------------------
# Collecting predictions / truth and scoring (metric spaces mirror run_module6_smoke._score)
# --------------------------------------------------------------------------------------------------
def _loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=PerturbationDataset.collate)


def collect_predictions(model, dataset, device: str = "cpu", batch_size: int = config.BATCH_SIZE) -> dict:
    """Forward ``model`` over ``dataset`` (eval mode, no grad) → stacked numpy Δz / Δx / σ / row_index."""
    model = model.to(device)
    model.eval()
    dz, dx, sig, ri = [], [], [], []
    with torch.no_grad():
        for batch, targets, conditions, _dz, _dx, rows in _loader(dataset, batch_size):
            out = model(batch, targets, conditions)
            dz.append(out["delta_z"].detach().cpu().numpy())
            dx.append(out["delta_x"].detach().cpu().numpy())
            sig.append(out["sigma"].detach().cpu().numpy())
            ri.extend(rows)
    return {"row_index": np.asarray(ri), "delta_z": np.concatenate(dz),
            "delta_x": np.concatenate(dx), "sigma": np.concatenate(sig)}


def collect_truth(dataset, batch_size: int = config.BATCH_SIZE) -> dict:
    """Stack the dataset's supervised Δz_true (=z@B) / Δx_true (=z) / row_index — no model forward needed."""
    dz, dx, ri = [], [], []
    for _batch, _targets, _conditions, dz_true, dx_true, rows in _loader(dataset, batch_size):
        dz.append(dz_true.numpy())
        dx.append(dx_true.numpy())
        ri.extend(rows)
    return {"row_index": np.asarray(ri), "delta_z": np.concatenate(dz), "delta_x": np.concatenate(dx)}


def collect_targets_truth(dataset, batch_size: int = config.BATCH_SIZE) -> dict:
    """Per-row target symbol + supervised Δz_true/Δx_true/row_index, in dataset order — what the non-neural
    NetworkPropagationBaseline needs (it fits on target genes, not a forwarded model)."""
    genes, dz, dx, ri = [], [], [], []
    for _batch, targets, _conditions, dz_true, dx_true, rows in _loader(dataset, batch_size):
        genes.extend(targets)
        dz.append(dz_true.numpy())
        dx.append(dx_true.numpy())
        ri.extend(rows)
    return {"genes": genes, "delta_z": np.concatenate(dz), "delta_x": np.concatenate(dx),
            "row_index": np.asarray(ri)}


def compute_all_metrics(dz_hat, dx_hat, dz_true, dx_true, train_mean) -> dict:
    """The evaluation suite on both spaces: program-space (Δz) correlation/centroid/cosine and gene-space
    (Δx) error/recall/sign. ``systema`` (the primary endpoint) removes the program-space training mean."""
    return {
        "pearson": M.pearson_corr(dz_hat, dz_true),
        "systema": M.systema_pert_specific_delta(dz_hat, dz_true, train_mean),
        "centroid": M.centroid_accuracy(dz_hat, dz_true),
        "prog_cos": M.program_cosine(dz_hat, dz_true),
        "mae": M.mae(dx_hat, dx_true),
        "rmse": M.rmse(dx_hat, dx_true),
        "topk": M.topk_recall(dx_hat, dx_true),
        "sign": M.sign_accuracy(dx_hat, dx_true),
    }


# --------------------------------------------------------------------------------------------------
# Nested-family model factories (a fresh model per call, so screening never reuses trained weights)
# --------------------------------------------------------------------------------------------------
def _egipg(gene_names, graph_encoder, basis_path, perturbation_encoder):
    return EGIPGModel.from_saved_basis(gene_names, path=basis_path,
                                       perturbation_encoder=perturbation_encoder,
                                       graph_encoder=graph_encoder)


def nested_family_factories(gene_names, graph, gene_to_idx, *, basis_path=None,
                            perturbation_encoder_factory=None) -> dict:
    """name -> zero-arg factory for the nested family + the untyped-graph diagnostic. Each factory builds a
    FRESH model (fresh graph encoder AND a fresh perturbation encoder), so two configs in one screening run
    never share or co-train weights. ``perturbation_encoder_factory`` is a callable returning a fresh encoder
    (tests inject zero-embedding stores); None lets EGIPGModel build the default real-embedding encoder."""
    def enc():
        return perturbation_encoder_factory() if perturbation_encoder_factory is not None else None
    return {
        EXPRESSION_ONLY: lambda: _egipg(gene_names, None, basis_path, enc()),
        TYPED_STATIC: lambda: _egipg(gene_names, StaticTypedGraphEncoder(graph, gene_to_idx), basis_path, enc()),
        CONDITION_GATED: lambda: _egipg(gene_names, TypedGraphEncoder(graph, gene_to_idx), basis_path, enc()),
        UNTYPED_GNN: lambda: _egipg(gene_names, UntypedGraphEncoder(graph, gene_to_idx), basis_path, enc()),
    }


def nested_family_configs(gene_names, graph, gene_to_idx, n_epochs: int, *, names=None, basis_path=None,
                          perturbation_encoder_factory=None, lr: float = config.LR,
                          batch_size: int = config.BATCH_SIZE, seed: int = 0) -> list[dict]:
    """Build screening configs for the named members (default: the three nested-family members)."""
    factories = nested_family_factories(gene_names, graph, gene_to_idx, basis_path=basis_path,
                                        perturbation_encoder_factory=perturbation_encoder_factory)
    if names is None:  # an explicit [] means "no configs", not "use the defaults"
        names = [EXPRESSION_ONLY, TYPED_STATIC, CONDITION_GATED]
    return [{"name": n, "model_factory": factories[n], "n_epochs": n_epochs, "lr": lr,
             "batch_size": batch_size, "seed": seed} for n in names]


# --------------------------------------------------------------------------------------------------
# Train → evaluate → write one config
# --------------------------------------------------------------------------------------------------
def screen_config(cfg: dict, train_ds, val_ds, train_mean, *, device: str = "cpu", split: str = "val",
                  predictions_root: Path = config.PREDICTIONS_ROOT,
                  screening_root: Path = config.SCREENING_ROOT,
                  ckpt_dir: Path | None = None, log_dir: Path | None = None,
                  registry_path: Path | None = None) -> dict:
    """Train ``cfg`` on ``train_ds``, reload its best checkpoint, score ``val_ds``, write predictions +
    a one-row metrics table, and return the results dict. If ``registry_path`` is given the run is
    registered before training and logged after (completed OR failed — a failure re-raises after logging)."""
    name = cfg["name"]
    seed = int(cfg.get("seed", 0))
    bs = int(cfg.get("batch_size", config.BATCH_SIZE))
    # seed-namespaced so a multi-seed sweep of one config name doesn't overwrite stage_a_best.pt (the
    # predictions + metrics parquet are already per-seed)
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else Path(screening_root) / name / str(seed) / "ckpt"
    log_dir = Path(log_dir) if log_dir else Path(screening_root) / name / str(seed) / "logs"
    donor_invariance = cfg.get("donor_invariance", config.DONOR_INVARIANCE)

    run_id = None
    if registry_path is not None:
        run_id = register_run(name, cfg.get("hypothesis", "screening"), cfg.get("inputs", "q_pre"),
                              cfg.get("split", "blocked_target_ood"), seed, cfg.get("budget"),
                              family=cfg.get("family", "egipg"), path=registry_path)
    start = time.perf_counter()  # wall time as a single-lane GPU-hour proxy for the registry audit field
    try:
        with seeded_init(seed):  # weight init from the config seed too, so the whole run is reproducible
            model = cfg["model_factory"]()
        trainer = Trainer(model, train_ds, val_ds, lr=cfg.get("lr", config.LR), max_epochs=cfg["n_epochs"],
                          batch_size=bs, seed=seed, device=device, ckpt_dir=ckpt_dir, log_dir=log_dir,
                          donor_invariance=donor_invariance)
        result = trainer.run()
        if result["best_ckpt"]:  # score the best-validation weights, not the last epoch's
            model.load_state_dict(torch.load(result["best_ckpt"], map_location=device)["model"])
        pred = collect_predictions(model, val_ds, device, bs)
        truth = collect_truth(val_ds, bs)
        metrics = compute_all_metrics(pred["delta_z"], pred["delta_x"], truth["delta_z"], truth["delta_x"],
                                      train_mean)
        write_predictions(truth["row_index"], pred["delta_z"], pred["delta_x"], pred["sigma"],
                          model=name, split=split, seed=seed, root=predictions_root)
        gpu_hours = (time.perf_counter() - start) / 3600.0
        results = {"name": name, "seed": seed, "n_epochs": cfg["n_epochs"], "status": "completed",
                   "best_val": result["best_val"], "epochs_run": result["epochs_run"],
                   "gpu_hours": gpu_hours, "primary": metrics[PRIMARY_METRIC], **metrics}
        _write_result_row(results, screening_root, name, seed)
        if run_id is not None:
            log_run(run_id, "completed", metrics, result["best_ckpt"], gpu_hours=gpu_hours, path=registry_path)
        return results
    except Exception as exc:  # log the failure (with elapsed GPU time), then re-raise — every run is accounted for
        if run_id is not None:
            log_run(run_id, "failed", {"error": str(exc)},
                    gpu_hours=(time.perf_counter() - start) / 3600.0, path=registry_path)
        raise


def _write_result_row(results: dict, screening_root: Path, name: str, seed: int) -> None:
    import pandas as pd
    final = Path(screening_root) / name / f"{seed}.parquet"
    config.write_parquet_atomic(pd.DataFrame([results]), final)


def score_network_propagation(train_ds, val_ds, train_mean, *, graph, gene_to_idx, basis,
                              seed: int = 0, split: str = "val", batch_size: int = config.BATCH_SIZE,
                              predictions_root: Path = config.PREDICTIONS_ROOT,
                              screening_root: Path = config.SCREENING_ROOT) -> dict:
    """Fit + score the non-neural ``NetworkPropagationBaseline`` on the same fold as the neural wave, so the
    topology-diffusion reference (feat-007's third graph baseline) lands in the screening table + the common
    output schema alongside the trained members. It diffuses train responses over the PPI graph on CPU, so it
    needs no Trainer/device. Returns a results row shaped like ``screen_config``'s (name/seed/status/primary
    + the metric suite)."""
    from tcell_pipeline.baselines.graph_baselines import NetworkPropagationBaseline
    tr = collect_targets_truth(train_ds, batch_size)
    va = collect_targets_truth(val_ds, batch_size)
    model = NetworkPropagationBaseline.from_hetero_graph(graph, gene_to_idx, basis=basis)
    model.fit(tr["genes"], tr["delta_z"])
    dz_hat, dx_hat = model.predict(va["genes"])
    metrics = compute_all_metrics(dz_hat, dx_hat, va["delta_z"], va["delta_x"], train_mean)
    write_predictions(va["row_index"], dz_hat, dx_hat, None, model=NETWORK_PROP, split=split, seed=seed,
                      root=predictions_root)
    results = {"name": NETWORK_PROP, "seed": seed, "status": "completed",
               "primary": metrics[PRIMARY_METRIC], **metrics}
    _write_result_row(results, screening_root, NETWORK_PROP, seed)
    return results


def _finite_or_none(obj):
    """Recursively replace non-finite floats (NaN / ±Inf — e.g. a diverged metric, or ``best_val`` left at
    +inf when val never improved) with None, so summary.json stays RFC-8259 valid: strict parsers (JS
    JSON.parse, most non-Python libraries) reject bare NaN/Infinity tokens that json.dumps would otherwise
    emit under its default ``allow_nan=True``."""
    if isinstance(obj, dict):
        return {k: _finite_or_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(v) for v in obj]
    if isinstance(obj, (float, np.floating)):  # np.float64 is a float subclass; np.float32 caught by np.floating
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


# --------------------------------------------------------------------------------------------------
# Run several configs → nested-family comparison
# --------------------------------------------------------------------------------------------------
def run_screening(configs: list[dict], train_ds, val_ds, *, device: str = "cpu", split: str = "val",
                  predictions_root: Path = config.PREDICTIONS_ROOT,
                  screening_root: Path = config.SCREENING_ROOT,
                  registry_path: Path | None = None, resilient: bool = True, extra_scorers=None) -> dict:
    """Screen every config on the same train/val split and report H2a / H2b on the primary endpoint.
    Writes ``<screening_root>/summary.json`` with the per-config table and the two contrasts.

    ``resilient`` (default) isolates a config's failure — an OOM / crash is caught, recorded as a failed
    result (and, if a registry is used, already logged failed by ``screen_config``), and the remaining
    configs still run, so one lane going down doesn't lose the others' results (report §screening: four
    independent lanes for cleaner failure isolation). ``resilient=False`` re-raises the first failure.

    ``extra_scorers`` are non-neural reference scorers (e.g. ``score_network_propagation``) run after the
    trained configs on the same fold, each called as ``scorer(train_ds, val_ds, train_mean,
    predictions_root=, screening_root=, split=)`` and returning a results row; they share the same failure
    isolation but never enter the H2a/H2b nested comparison."""
    train_mean = collect_truth(train_ds)["delta_z"].mean(0)
    results = []
    for cfg in configs:
        try:
            results.append(screen_config(cfg, train_ds, val_ds, train_mean, device=device, split=split,
                                         predictions_root=predictions_root, screening_root=screening_root,
                                         registry_path=registry_path))
        except Exception as exc:
            if not resilient:
                raise
            print(f"[screen] config {cfg['name']!r} FAILED ({type(exc).__name__}: {exc}); continuing wave")
            results.append({"name": cfg["name"], "seed": int(cfg.get("seed", 0)),
                            "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    for scorer in (extra_scorers or []):
        try:
            results.append(scorer(train_ds, val_ds, train_mean, predictions_root=predictions_root,
                                  screening_root=screening_root, split=split))
        except Exception as exc:
            if not resilient:
                raise
            name = getattr(scorer, "screen_name", getattr(scorer, "__name__", "extra_baseline"))
            print(f"[screen] extra baseline {name!r} FAILED ({type(exc).__name__}: {exc}); continuing wave")
            results.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    by_name = {r["name"]: r for r in results}
    summary = {"results": results, **_nested_comparison(by_name)}
    summary_path = Path(screening_root) / "summary.json"
    # sanitize non-finite -> None (and allow_nan=False as a loud backstop) so the deliverable is valid JSON
    config.write_text_atomic(
        json.dumps(_finite_or_none(summary), indent=2, default=float, allow_nan=False), summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def _nested_comparison(by_name: dict) -> dict:
    """H2a (typed static > expr-only) and H2b (condition-gated > typed static) on ``systema``; each present
    only when both of its members were screened."""
    def contrast(better: str, worse: str) -> dict | None:
        if all(n in by_name and PRIMARY_METRIC in by_name[n] for n in (better, worse)):  # both succeeded
            b, w = by_name[better][PRIMARY_METRIC], by_name[worse][PRIMARY_METRIC]
            return {"better": better, "worse": worse, "delta": float(b - w), "supported": bool(b > w)}
        return None
    out = {}
    if (h2a := contrast(TYPED_STATIC, EXPRESSION_ONLY)) is not None:
        out["h2a"] = h2a
    if (h2b := contrast(CONDITION_GATED, TYPED_STATIC)) is not None:
        out["h2b"] = h2b
    return out
