"""Module 8 (Sealed Challenge Eval, Phase 5) tests — fully synthetic (tiny marts). Covers the H1 rule end to
end with a deterministic echo model: H1 confirmed when EG-IPG beats the baselines by more than the margin AND
beats the perturbed mean; NOT confirmed when a baseline matches it; the write-once seal (a second evaluate
refuses, force overrides); and the input guards (perturbed_mean required, aligned shapes).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from torch import nn

torch.set_num_threads(1)

from tcell_pipeline import config
from tcell_pipeline.evaluation.sealed_eval import SealedEvaluator, _bootstrap_diffs
from tcell_pipeline.screening.screening import dataset_delta_z
from tcell_pipeline.training import PerturbationDataset

_G, _K = 6, 3


def _write_marts(tmp_path) -> dict:
    genes = [f"G{i}" for i in range(_G)]
    # 4 train rows (G0/G1) + 4 challenge rows (G3/G4); duplicate-gene challenge rows share z so an echo model
    # keyed by gene reproduces the per-row truth exactly (row-level ρ == 1)
    rows = [(0, "G0"), (1, "G1"), (2, "G0"), (3, "G1"), (4, "G3"), (5, "G3"), (6, "G4"), (7, "G4")]
    n = len(rows)
    rng = np.random.default_rng(0)
    f32 = lambda v: np.full(n, v, np.float32)
    pc = pd.DataFrame({
        "row_index": [r[0] for r in rows], "hgnc_symbol": [r[1] for r in rows],
        "culture_condition": ["Rest"] * n, "uniprot_id": [f"P{i}" for i in range(n)],
        "ppi_degree_physical": f32(1.0), "ppi_degree_functional": f32(1.0),
        "ppi_degree_complex": f32(1.0), "control_baseline_expr": f32(0.5),
        **{f"donor_pc_{i:02d}": rng.random(n).astype("float32") for i in range(config.DONOR_PCA_DIMS)},
    })
    obs = pd.DataFrame({"n_guides": np.full(n, 2), "single_guide_estimate": np.zeros(n, bool)})
    B = rng.standard_normal((_G, _K)).astype("float32")
    loadings = pd.DataFrame(B, columns=[f"program_{k}" for k in range(_K)])
    loadings.insert(0, "gene_name", genes)
    z = rng.standard_normal((n, _G)).astype("float32")
    z[5] = z[4]      # both G3 challenge rows identical
    z[7] = z[6]      # both G4 challenge rows identical
    split = pd.DataFrame({"hgnc_symbol": ["G0", "G1", "G3", "G4"],
                          "role": ["train", "train", "challenge", "challenge"]})
    donor_cols = [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(config.DONOR_PCA_DIMS)]
    prof = [dict(donor_id=f"CE{d}", culture_condition=c,
                 **{cc: float(v) for cc, v in zip(donor_cols, rng.random(config.DONOR_PCA_DIMS))})
            for c in config.CONDITIONS for d in range(3)]
    p = {"split_path": tmp_path / "split.csv", "pc_path": tmp_path / "pc.parquet",
         "obs_path": tmp_path / "obs.parquet", "var_path": tmp_path / "var.parquet",
         "basis_path": tmp_path / "loadings.parquet", "zscore_npz": tmp_path / "zscore.npz",
         "donor_profiles_path": tmp_path / "donor_profiles.parquet"}
    split.to_csv(p["split_path"], index=False)
    pc.to_parquet(p["pc_path"], index=False)
    obs.to_parquet(p["obs_path"], index=False)
    pd.DataFrame({"gene_name": genes}).to_parquet(p["var_path"], index=False)
    loadings.to_parquet(p["basis_path"], index=False)
    sp.save_npz(p["zscore_npz"], sp.csr_matrix(z))
    pd.DataFrame(prof).to_parquet(p["donor_profiles_path"], index=False)
    return p


class _EchoModel(nn.Module):
    """A test model whose forward returns a fixed per-gene program-delta — set it to the challenge truth to
    get ρ==1, or leave a gene out to weaken it."""

    def __init__(self, gene_dz: dict, k: int, g: int) -> None:
        super().__init__()
        self.gene_dz, self.k, self.g = gene_dz, k, g

    def forward(self, batch, targets, conditions):
        dz = torch.stack([self.gene_dz[t] for t in targets])
        return {"delta_z": dz, "delta_x": torch.zeros(len(targets), self.g), "sigma": torch.zeros(len(targets), self.k)}


def _setup(tmp_path):
    paths = _write_marts(tmp_path)
    train_ds = PerturbationDataset("train", **paths)
    challenge_ds = PerturbationDataset("challenge", **paths)
    train_mean = dataset_delta_z(train_ds).mean(0)
    truth = dataset_delta_z(challenge_ds)                     # (N, K) in dataset order == collect_predictions order
    genes = challenge_ds.pc["hgnc_symbol"].astype(str).tolist()
    gene_dz = {g: torch.tensor(truth[i], dtype=torch.float32) for i, g in enumerate(genes)}
    perturbed_mean = np.broadcast_to(train_mean, truth.shape).copy()
    ev = SealedEvaluator(challenge_ds, train_mean, sealed_root=tmp_path / "sealed",
                         predictions_root=tmp_path / "pred")
    return ev, _EchoModel(gene_dz, _K, _G), truth, perturbed_mean


def test_h1_confirmed_when_egipg_beats_baselines(tmp_path):
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    res = ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    assert res["h1_confirmed"] is True
    assert res["rho_egipg"] > 0.99 and res["lcb_95"] > config.DELTA_PRED
    assert res["beats_perturbed_mean"] is True and res["rho_perturbed_mean"] == pytest.approx(0.0, abs=1e-9)
    assert json.loads((tmp_path / "sealed" / "challenge" / "0.json").read_text())["h1_confirmed"] is True


def test_h1_not_confirmed_when_a_baseline_matches(tmp_path):
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    res = ev.evaluate(echo, {"perturbed_mean": perturbed_mean, "oracle": {"delta_z": truth, "delta_x": truth}},
                      split="challenge", seed=1)
    assert res["best_baseline"] == "oracle"                   # the oracle ties EG-IPG
    assert res["lcb_95"] == pytest.approx(0.0, abs=1e-9) and res["h1_confirmed"] is False


def test_sealed_result_is_write_once(tmp_path):
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    with pytest.raises(FileExistsError, match="write-once"):
        ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    res = ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0, force=True)
    assert res["h1_confirmed"] is True                        # force re-seals


def test_seal_is_per_split_not_per_seed(tmp_path):
    # `seed` only redraws the bootstrap RNG; if the seal were keyed on it, a steward who got lcb just under
    # the margin could re-run at seed=1 and re-open the sequestered fold until the decision confirmed
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    with pytest.raises(FileExistsError, match="already sealed"):
        ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=1)
    assert not (tmp_path / "sealed" / "challenge" / "1.json").exists()   # no second sealed decision
    # a DIFFERENT split is still sealable (the per-split seal is not a global lock)
    ev2 = SealedEvaluator(ev.challenge_ds, ev.train_mean, sealed_root=tmp_path / "sealed",
                          predictions_root=tmp_path / "pred")
    assert ev2.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="calibration", seed=0)["n_rows"] == 4


def test_perturbed_mean_reference_is_structurally_zero(tmp_path):
    # the H1 rule's second clause reduces to rho_egipg > 0 under systema; the sealed record must say so
    # rather than let a reader mistake it for a check that discriminated
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    res = ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    assert res["rho_perturbed_mean"] == pytest.approx(0.0, abs=1e-12)
    assert "structurally 0.0" in res["perturbed_mean_reference_note"]


def test_evaluate_guards(tmp_path):
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    with pytest.raises(ValueError, match="perturbed_mean"):
        ev.evaluate(echo, {"zero": np.zeros_like(truth)}, split="challenge", seed=2)
    with pytest.raises(ValueError, match="misaligned"):
        ev.evaluate(echo, {"perturbed_mean": np.zeros((truth.shape[0] + 1, _K))}, split="challenge", seed=3)


def test_evaluate_refuses_degenerate_fold(tmp_path):
    ev, echo, truth, perturbed_mean = _setup(tmp_path)   # 4-row challenge fold
    with pytest.raises(ValueError, match="min_rows"):    # a too-small fold must not be sealed write-once
        ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=4, min_rows=5)
    assert not (tmp_path / "sealed" / "challenge" / "4.json").exists()   # nothing sealed


def test_bootstrap_lcb_on_nonconstant_diffs():
    diff = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])     # a genuinely varying per-row difference
    rng = np.random.default_rng(0)
    boot = _bootstrap_diffs(diff, 3000, rng)
    assert boot.shape == (3000,)
    assert diff.min() <= boot.min() and boot.max() <= diff.max()      # resampled means stay within the range
    assert abs(boot.mean() - diff.mean()) < 0.05                      # unbiased around the sample mean
    lcb = float(np.percentile(boot, 2.5))
    assert diff.min() < lcb < diff.mean()                            # a real (non-degenerate) lower bound
