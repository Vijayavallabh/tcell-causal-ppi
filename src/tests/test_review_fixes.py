"""Regression tests for the 15 findings of the 2026-07-21 max-effort review.

Every test here was watched FAILING before its fix, and each one encodes a breaking INPUT rather than
the shape of the repair — the review's own failure scenarios, turned into code. Named by the defect they
pin, not by the function they call, so a future reader learns what went wrong.
"""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# --------------------------------------------------------------------------------------------------
# [0] + [7] basis-study sweep: a capped cell must not destroy a measured one, nor leave a live .npy
# --------------------------------------------------------------------------------------------------


def _measured_cell(path: Path, method="sparse_pca", K=512) -> dict:
    doc = {
        "method": method, "K": K, "fit_seconds": 5780.0,
        "convergence": {"n_iter": 6, "max_iter": 100, "converged": True},
        "recon": {"recon_mae": 0.6499, "zero_baseline_mae": 0.8174, "explained_frac": 0.2048},
        "recon_native": {"target": "centred", "recon_mae": 0.6499},
        "sparsity": {"zero_frac": 0.5112, "n_dead": 0},
        "has_stability": False,
    }
    path.write_text(json.dumps(doc, indent=2))
    return doc


def test_capped_cell_does_not_destroy_a_completed_measurement(tmp_path):
    """A >90-minute measured cell must survive a later timeout on the same cell."""
    from tcell_pipeline.programs import run_basis_study as R

    cell = tmp_path / "sparse_pca_K512.json"
    before = _measured_cell(cell)

    R.write_not_measured(cell, "sparse_pca", 512, "timeout", 5400.0)

    after = json.loads(cell.read_text())
    assert after["recon"]["recon_mae"] == before["recon"]["recon_mae"], (
        "write_not_measured clobbered a completed cell: a 90-minute fit is now an all-None stub")


def test_capped_cell_removes_its_orphan_row_mae(tmp_path):
    """A not-measured cell must not leave a row_mae array behind for build_contrasts to publish."""
    from tcell_pipeline.programs import run_basis_study as R

    cells, row_mae = tmp_path / "cells", tmp_path / "row_mae"
    cells.mkdir(), row_mae.mkdir()
    npy = row_mae / "nmf_K256.npy"
    np.save(npy, np.zeros(8))

    R.write_not_measured(cells / "nmf_K256.json", "nmf", 256, "timeout", 5400.0, row_mae_dir=row_mae)

    assert not npy.exists(), (
        "the orphan row_mae array survives, so build_contrasts publishes a p-value and consumes a "
        "Bonferroni slot for a cell the table reports as not measured")


# --------------------------------------------------------------------------------------------------
# [3] held-out reconstruction must not rank NMF on a target it never modelled
# --------------------------------------------------------------------------------------------------


def test_heldout_records_the_native_target_for_nmf():
    """The in-sample path has recon_native for this exact reason; the held-out path must too."""
    from tcell_pipeline.programs import run_basis_study as R

    src = inspect.getsource(R.stability_and_heldout)
    assert "_native_target" in src, (
        "held-out scores NMF on the signed centred target it never modelled, while sparse_pca/fastica "
        "are scored on theirs — and the study's headline is a method x K ranking over those columns")


# --------------------------------------------------------------------------------------------------
# [6] inner split: the ACHIEVED fraction is what the caller gets, so it is what must be validated
# --------------------------------------------------------------------------------------------------


def test_group_partition_warns_when_the_holdout_becomes_the_majority():
    """74/26 achieved against a requested 50% is silent today: |0.74-0.5| = 0.24 < 0.5*0.5."""
    from tcell_pipeline.training.inner_split import group_partition

    labels = ["A"] * 74 + ["B"] * 26
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tr, ho = group_partition(labels, holdout_frac=0.5)

    assert len(ho) / len(labels) > 0.5
    assert caught, (
        f"achieved holdout is {len(ho) / len(labels):.0%} of rows against a requested 50% and NOTHING "
        f"warned — the caller trains on {len(tr)} rows and selects on {len(ho)}")


# --------------------------------------------------------------------------------------------------
# [10] absence of convergence evidence is None, never the verdict "did not converge"
# --------------------------------------------------------------------------------------------------


def test_no_iteration_evidence_reports_none_not_a_negative_verdict():
    from tcell_pipeline.baselines.simple_baselines import _converged_from_iters

    assert _converged_from_iters([], max_iter=100) is None, (
        "an empty iteration list is NO EVIDENCE; reporting False publishes 'under-fit' as a measured "
        "fact and that entry is what a reader uses to call the H1 margin an upper bound")
    assert _converged_from_iters([50], max_iter=100) is True
    assert _converged_from_iters([100], max_iter=100) is False


# --------------------------------------------------------------------------------------------------
# [1] + [2] + [12] the resume cache key must cover everything a score depends on
# --------------------------------------------------------------------------------------------------


def test_bar_signature_changes_when_the_basis_content_changes():
    """Same shapes, different basis: a refit basis must not serve pre-refit scores as current."""
    from tcell_pipeline.run_module8_real import _bar_signature

    rng = np.random.default_rng(0)
    b1 = rng.normal(size=(64, 8))
    b2 = b1.copy()
    b2[0, 0] += 1.0
    kw = dict(n_train=100, n_val=20, n_features=12, k=8, bar="ridge", seed=0)

    assert _bar_signature(basis=b1, **kw) != _bar_signature(basis=b2, **kw), (
        "a rebuilt basis at the same PROGRAM_DIM leaves every signature field byte-identical, so every "
        "bar hits the cache and republishes pre-change scores while printing [cached]")


def test_bar_signature_covers_the_shared_base_class():
    """inspect.getsource on a subclass returns only its own block — BaseBaseline fell out of the key."""
    from tcell_pipeline.run_module8_real import _bar_source

    src = _bar_source("ridge")
    assert "_decode_genes" in src, (
        "editing BaseBaseline._decode_genes (the dz @ B.T decode every bar shares) changes every score "
        "without moving baselines_sha")


def test_bar_signature_includes_the_seed():
    """Predictions are written to val/{seed}.parquet, so seed is part of what a score is valid for."""
    from tcell_pipeline.run_module8_real import _bar_signature

    kw = dict(n_train=100, n_val=20, n_features=12, k=8, bar="ridge", basis=np.zeros((4, 8)))
    assert _bar_signature(seed=0, **kw) != _bar_signature(seed=1, **kw), (
        "seed-1 hits the seed-0 cache, so val/1.parquet is never written and the seed-1 table is "
        "silently identical to seed 0")


def test_cache_hit_requires_its_prediction_artifact(tmp_path):
    """write_predictions moved into the miss branch; a hit with no parquet must not count as done."""
    from tcell_pipeline.run_module8_real import load_cached_bar, save_cached_bar

    sig = {"n_train": 1}
    save_cached_bar(tmp_path, "node", "ridge", sig, {"systema": 0.1}, None)
    missing = tmp_path / "does_not_exist.parquet"

    assert load_cached_bar(tmp_path, "node", "ridge", sig, predictions=missing) is None, (
        "clearing data/results/predictions and re-running republishes the whole comparator table while "
        "every per-bar prediction file stays missing, with no warning")


# --------------------------------------------------------------------------------------------------
# [4] a non-finite metric must not round-trip to null and kill the resumed run
# --------------------------------------------------------------------------------------------------


def test_non_finite_metric_is_never_served_from_cache(tmp_path):
    from tcell_pipeline.run_module8_real import load_cached_bar, save_cached_bar

    sig = {"n_train": 1}
    save_cached_bar(tmp_path, "node", "low_rank", sig, {"systema": 0.1, "mae": float("inf")}, None)

    hit = load_cached_bar(tmp_path, "node", "low_rank", sig)
    assert hit is None, (
        "a non-finite mae is stored as JSON null, and the resumed run formats that None with :>9.4f and "
        "dies AFTER the parquet is written but BEFORE the H1 summary and the under-fit exit-code gate")


def test_a_legitimately_none_metric_is_still_cacheable(tmp_path):
    """SELF-REVIEW: rejecting on ``value is None`` conflated two different things.

    ``_finite_or_none`` maps a non-finite float to null, but None is also this project's standing
    encoding for 'no evidence' (an undecidable metric). Keying the miss on None would make such a bar
    recompute forever, silently, every run. Record WHICH keys were non-finite instead."""
    from tcell_pipeline.run_module8_real import load_cached_bar, save_cached_bar

    sig = {"n_train": 1}
    save_cached_bar(tmp_path, "node", "ridge", sig, {"systema": 0.1, "undecidable_metric": None}, None)

    hit = load_cached_bar(tmp_path, "node", "ridge", sig)
    assert hit is not None, "an honestly-undecidable metric must not make the bar uncacheable forever"
    assert hit["metrics"]["systema"] == 0.1


# --------------------------------------------------------------------------------------------------
# [11] the expected repro verdict must stay distinguishable from a real gate failure
# --------------------------------------------------------------------------------------------------


def test_repro_verdict_and_underfit_gate_use_distinct_exit_bits():
    from tcell_pipeline import run_module8_real as M

    assert M.RC_REPRO != M.RC_BASELINES, "both signal through the same bit"
    assert M.RC_REPRO & M.RC_BASELINES == 0, (
        "an under-fit bar (which makes the published H1 margin an upper bound) is indistinguishable "
        "from CANNOT_VERIFY, the correct and permanent repro answer for this project")


# --------------------------------------------------------------------------------------------------
# [13] the BLAS thread cap must be set before numpy is imported, not after
# --------------------------------------------------------------------------------------------------


def _import_env(repo: Path) -> dict:
    return {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo / "src")}


def test_importing_the_programs_package_does_not_cap_threads_globally():
    """SELF-REVIEW of the first fix for [13], which had a blast radius its own test could not see.

    Setting the caps in ``programs/__init__.py`` did make them precede numpy — but that package is
    imported by model.py, training/dataset.py, run_train.py, run_stage_b.py and run_screening.py, so it
    silently pinned OMP_NUM_THREADS=4 on every training and screening run, including the 20-epoch GPU
    campaigns. A one-cell fix must not throttle the whole project."""
    repo = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c",
         "import os; import tcell_pipeline.programs; print(os.environ.get('OMP_NUM_THREADS'))"],
        capture_output=True, text=True, cwd=repo, env=_import_env(repo))
    assert out.stdout.strip() == "None", (
        f"importing the programs package set OMP_NUM_THREADS={out.stdout.strip()!r}, which every "
        f"trainer inherits. stderr={out.stderr[-300:]}")


def test_basis_study_reports_whether_its_thread_cap_actually_took_effect():
    """The driver must not print OMP=4 when numpy was already loaded and the cap could not apply."""
    from tcell_pipeline.programs import run_basis_study as R

    assert R.thread_cap_effective() is False, (
        "numpy is imported long before this test runs, so the cap cannot have taken effect here — a "
        "guard that cannot report that is the silent-lie this finding was about")


# --------------------------------------------------------------------------------------------------
# [14] the GPU-only bar must not be pinned to CPU by a shared --device default
# --------------------------------------------------------------------------------------------------


def test_tabicl_is_not_pinned_to_cpu_by_the_shared_device_default():
    from tcell_pipeline.run_module8_real import _gpu_bar_device

    assert _gpu_bar_device("cpu", explicit=False) is None, (
        "--with-tabicl with no --device constructs TabICLBaseline(device='cpu'), turning a ~4.4 GPU-h "
        "bar into a multi-day CPU job on a shared box")
    assert _gpu_bar_device("cpu", explicit=True) == "cpu", "an explicit --device cpu must be honoured"


# --------------------------------------------------------------------------------------------------
# [5] the boosted bars must select depth on a target-grouped split, not random rows
# --------------------------------------------------------------------------------------------------


def test_catboost_holds_out_whole_target_groups():
    """The eval_set that drives early stopping must share no target gene with the fitted rows."""
    from tcell_pipeline.baselines.simple_baselines import CatBoostBaseline

    bar = CatBoostBaseline()
    bar._groups = [f"G{i // 3}" for i in range(60)]        # one gene spans 3 rows, as in the real mart
    keep, hold = bar._internal_split(60, frac=0.2, seed=0)

    shared = {bar._groups[i] for i in keep} & {bar._groups[i] for i in hold}
    assert not shared, f"target genes {sorted(shared)} appear on BOTH sides of the selection split"


def test_gradient_boosting_publishes_no_leaked_convergence_verdict():
    """sklearn HGB can only early-stop on random rows, so it must report unknown, not a verdict."""
    from tcell_pipeline.baselines.simple_baselines import GradientBoostingBaseline

    bar = GradientBoostingBaseline()
    assert bar._model.estimator.early_stopping is False, (
        "early stopping is selecting depth on a random-row split that shares target genes with the "
        "fitted rows, and that verdict feeds the under-fit gate")


def test_internal_split_survives_a_fold_with_one_target_gene():
    """SELF-REVIEW: group_partition RAISES when no grouped split exists, and the boosted bar swallowed
    no such case before — so the grouped-split fix turned a working (if leaky) fit into a crash."""
    from tcell_pipeline.baselines.simple_baselines import CatBoostBaseline

    bar = CatBoostBaseline()
    bar._groups = ["ONLY"] * 50                      # degenerate but real for a tiny or filtered fold
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keep, hold = bar._internal_split(50, frac=0.1, seed=0)
    assert len(keep) and len(hold), "a fold with one target gene must still fit, not raise"
    assert caught, "and it must say the verdict is contaminated, since no grouped split was possible"


def test_internal_split_warns_when_it_has_to_fall_back_to_random_rows():
    from tcell_pipeline.baselines.simple_baselines import CatBoostBaseline

    bar = CatBoostBaseline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bar._internal_split(60, frac=0.2, seed=0)
    assert caught, "a silent fallback to random rows reads exactly like a grouped split"


# --------------------------------------------------------------------------------------------------
# [8] + [9] the gradient probe's own controls must be able to fail
# --------------------------------------------------------------------------------------------------


def test_severed_control_forwards_the_edge_gates():
    """Behavioural: the wrapper must hand the inner model's real gates through, not None."""
    import torch

    from tcell_pipeline.probe_graph_gradients import _SeveredModel

    gates = {"r": torch.rand(4, requires_grad=True)}

    class _Inner(torch.nn.Module):
        graph_encoder = torch.nn.Linear(2, 2)

        def __init__(self):
            super().__init__()
            self.decoder = lambda h_do, h_graph: {"delta_z": h_do + h_graph}

        def forward(self, batch, targets, conditions):
            return {"h_do": torch.zeros(1, 2), "h_graph": torch.zeros(1, 2, requires_grad=True),
                    "edge_gates": gates, "edge_confidences": {"r": torch.ones(4)}}

    out = _SeveredModel(_Inner())(None, None, None)
    assert out["edge_gates"] is not None, (
        "_SeveredModel returns edge_gates=None, so StageALoss._graph short-circuits to a constant with "
        "no grad_fn and the self-control never exercises the penalty-gradient half it claims to license")
    assert out["edge_confidences"] is not None


def test_frozen_gate_readings_use_the_trained_model_for_h_do():
    from tcell_pipeline import probe_graph_gradients as P

    src = inspect.getsource(P.probe_e)
    assert "build().perturbation_encoder" not in src, (
        "h_do comes from a freshly built RANDOM-INIT model and is fed to the frozen-H1 encoder, so the "
        "headline gate_collapse_factor mixes the real collapse with an input-distribution mismatch")
