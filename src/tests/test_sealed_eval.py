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


def test_seal_is_bound_to_the_fold_not_the_label(tmp_path):
    # The seal must key on the sequestered FOLD. Keying it on `seed` let a steward resample the decision;
    # keying it on the `split` LABEL merely moved the hole — a label is caller-supplied, so any alias
    # ("Challenge", "challenge_rerun") re-opened the same fold. Every alias below must be refused.
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    B = {"perturbed_mean": perturbed_mean}
    ev.evaluate(echo, B, split="challenge", seed=0)
    for label, seed in [("challenge", 1), ("Challenge", 1), ("challenge_rerun", 7), ("CHALLENGE", 2)]:
        with pytest.raises(FileExistsError, match="already"):
            ev.evaluate(echo, B, split=label, seed=seed)
        assert not (tmp_path / "sealed" / label / f"{seed}.json").exists()
    # a path-traversal label must be rejected outright (it would escape sealed_root AND dodge the seal)
    for bad in ["calib/../challenge", "a/b", "..", "."]:
        with pytest.raises(ValueError, match="invalid split label"):
            ev.evaluate(echo, B, split=bad, seed=9)
    # only ONE sealed decision exists for this fold
    assert len(list((tmp_path / "sealed").rglob("*.json"))) == 1


def test_seal_permits_a_genuinely_different_fold(tmp_path):
    # the fold-keyed seal must not become a global lock: a DIFFERENT sequestered fold is still sealable
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    m2 = tmp_path / "m2"
    m2.mkdir()
    other = PerturbationDataset("train", **_write_marts(m2))   # different rows -> a different fold
    ev2 = SealedEvaluator(other, ev.train_mean, sealed_root=tmp_path / "sealed",
                          predictions_root=tmp_path / "pred")
    truth2 = dataset_delta_z(other)
    genes2 = other.pc["hgnc_symbol"].astype(str).tolist()
    echo2 = _EchoModel({g: torch.tensor(truth2[i], dtype=torch.float32) for i, g in enumerate(genes2)}, _K, _G)
    pm2 = np.broadcast_to(ev.train_mean, truth2.shape).copy()
    assert ev2.evaluate(echo2, {"perturbed_mean": pm2}, split="train_fold", seed=0)["n_rows"] == len(truth2)


def test_failed_attempt_does_not_brick_the_fold(tmp_path):
    # the atomic fold claim must be released when the evaluation itself fails, or a typo would permanently
    # prevent the fold from ever being sealed
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    with pytest.raises(ValueError, match="perturbed_mean"):
        ev.evaluate(echo, {"zero": np.zeros_like(truth)}, split="challenge", seed=0)   # bad call
    res = ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)  # now succeeds
    assert res["h1_confirmed"] is True


def test_perturbed_mean_note_matches_the_computed_value(tmp_path):
    # the H1 rule's second clause reduces to rho_egipg > 0 under systema; the sealed record must say so
    ev, echo, truth, perturbed_mean = _setup(tmp_path)
    res = ev.evaluate(echo, {"perturbed_mean": perturbed_mean}, split="challenge", seed=0)
    assert res["rho_perturbed_mean"] == pytest.approx(0.0, abs=1e-12)
    assert res["perturbed_mean_is_structural_zero"] is True
    assert "structurally 0.0" in res["perturbed_mean_reference_note"]

    # ...but the note must NOT be a hardcoded claim: if the caller supplies something that is not the
    # train-mean broadcast, rho is not 0 and the record must say THAT instead of contradicting itself
    ev2 = SealedEvaluator(ev.challenge_ds, ev.train_mean, sealed_root=tmp_path / "sealed2",
                          predictions_root=tmp_path / "pred2")
    bogus = truth * 0.5 + np.random.default_rng(7).standard_normal(truth.shape) * 0.3
    res2 = ev2.evaluate(echo, {"perturbed_mean": bogus}, split="challenge", seed=0)
    assert abs(res2["rho_perturbed_mean"]) > 1e-6
    assert res2["perturbed_mean_is_structural_zero"] is False
    assert "WARNING" in res2["perturbed_mean_reference_note"]


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
