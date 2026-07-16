"""Module 5 (Loss + Training) tests — fully synthetic: tiny fixture marts + zero embedding stores.

Stage A trains Module 1+2+3; Stage B is a loss module only. Covered: loss component shapes + gradient
flow, the DE head probability range, the learnable lambda mixture, the graph-gate penalty, the dataset
contract (correct keys, the q_post leakage fence, program_response vs out-of-fold projection), and a
2-epoch expression-only training run that actually writes checkpoints and moves parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

torch.set_num_threads(1)  # many-core box: tiny GNN/linear ops thrash the default thread pool otherwise

from tcell_pipeline import config
from tcell_pipeline.encoders import PerturbationEncoder
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.model import EGIPGModel
from tcell_pipeline.programs import ProgramDecoder
from tcell_pipeline.training import (
    DEHead,
    PerturbationDataset,
    StageALoss,
    StageBCalibrationLoss,
    Trainer,
)

_G, _K = 6, 3
_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)


def _write_fixtures(tmp_path) -> dict:
    genes = [f"G{i}" for i in range(_G)]
    rows = [(0, "G0"), (1, "G1"), (2, "G0"), (3, "G3"), (4, "G4")]  # 0-2 train, 3 val, 4 challenge
    n = len(rows)
    rng = np.random.default_rng(0)
    f32 = lambda v: np.full(n, v, dtype=np.float32)
    pc = pd.DataFrame({
        "row_index": [r[0] for r in rows],
        "hgnc_symbol": [r[1] for r in rows],
        "culture_condition": ["Rest"] * n,
        "uniprot_id": [f"P{i}" for i in range(n)],
        "ppi_degree_physical": f32(1.0), "ppi_degree_functional": f32(1.0),
        "ppi_degree_complex": f32(1.0), "control_baseline_expr": f32(0.5),
        **{f"donor_pc_{i:02d}": rng.random(n).astype(np.float32) for i in range(config.DONOR_PCA_DIMS)},
    })
    obs = pd.DataFrame({"n_guides": np.full(n, 2), "single_guide_estimate": np.zeros(n, dtype=bool)})
    B = rng.standard_normal((_G, _K)).astype(np.float32)
    loadings = pd.DataFrame(B, columns=[f"program_{k}" for k in range(_K)])
    loadings.insert(0, "gene_name", genes)
    A = rng.standard_normal((3, _K)).astype(np.float32)
    resp = pd.DataFrame(A, columns=[f"program_{k}" for k in range(_K)])
    resp.insert(0, "row_index", [0, 1, 2])
    z = rng.standard_normal((n, _G)).astype(np.float32)
    split = pd.DataFrame({"hgnc_symbol": ["G0", "G1", "G3", "G4"],
                          "role": ["train", "train", "val", "challenge"]})
    # 3 distinct real "donors" per condition (control_donor_profiles); the mart's donor_pc is their mean
    donor_cols = [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(config.DONOR_PCA_DIMS)]
    prof_rows = [dict(donor_id=f"CE{d}", culture_condition=cond,
                      **{c: float(v) for c, v in zip(donor_cols, rng.random(config.DONOR_PCA_DIMS))})
                 for cond in config.CONDITIONS for d in range(3)]
    donor_profiles = pd.DataFrame(prof_rows)

    p = {
        "split_path": tmp_path / "split.csv", "pc_path": tmp_path / "pc.parquet",
        "obs_path": tmp_path / "obs.parquet", "var_path": tmp_path / "var.parquet",
        "basis_path": tmp_path / "loadings.parquet", "response_path": tmp_path / "resp.parquet",
        "zscore_npz": tmp_path / "zscore.npz", "donor_profiles_path": tmp_path / "donor_profiles.parquet",
    }
    split.to_csv(p["split_path"], index=False)
    pc.to_parquet(p["pc_path"], index=False)
    obs.to_parquet(p["obs_path"], index=False)
    pd.DataFrame({"gene_name": genes}).to_parquet(p["var_path"], index=False)
    loadings.to_parquet(p["basis_path"], index=False)
    resp.to_parquet(p["response_path"], index=False)
    sp.save_npz(p["zscore_npz"], sp.csr_matrix(z))
    donor_profiles.to_parquet(p["donor_profiles_path"], index=False)
    return p


def _expr_model(paths) -> EGIPGModel:
    gene_names = pd.read_parquet(paths["var_path"])["gene_name"].tolist()
    return EGIPGModel.from_saved_basis(
        gene_names, path=paths["basis_path"],
        perturbation_encoder=PerturbationEncoder(_ZERO_PLM, _ZERO_PIN), graph_encoder=None,
    )


def _synthetic_out(batch=4):
    return {
        "delta_z": torch.randn(batch, _K, requires_grad=True),
        "delta_x": torch.randn(batch, _G, requires_grad=True) * 3.0,  # some |z| beyond the DE call
        "h_do": torch.randn(batch, config.H_DO_DIM, requires_grad=True),
        "edge_gates": None,
    }


def test_stage_a_components_and_gradients():
    torch.manual_seed(0)
    out = _synthetic_out()
    loss = StageALoss(gene_dim=_G, program_dim=_K)
    # two donor-variant Delta z (distinct real donors) drive a real, non-zero invariance signal
    dz_variants = [torch.randn(4, _K, requires_grad=True), torch.randn(4, _K, requires_grad=True)]
    comps = loss(out, torch.randn(4, _K), out["delta_x"].detach(), dz_variants=dz_variants)
    assert set(comps) == {"total", "response", "gene", "de", "invariance", "graph"}
    assert all(torch.isfinite(v).all() for v in comps.values())
    assert comps["invariance"] > 0                       # variance of f_shared across the two donors
    comps["total"].backward()
    assert out["h_do"].grad is not None                  # gradient reaches the encoder input
    assert loss.de_head.head.weight.grad is not None     # ...the DE head
    assert loss.f_shared.weight.grad is not None         # ...the shared extractor (via the donor variants)
    assert dz_variants[0].grad is not None               # ...and back to the donor-variant predictions


def test_invariance_zero_without_donor_variants():
    loss = StageALoss(gene_dim=_G, program_dim=_K)
    out = _synthetic_out()
    comps = loss(out, torch.randn(4, _K), out["delta_x"].detach(), dz_variants=None)
    assert float(comps["invariance"]) == 0.0             # resampling off / no pool -> no-op, not spurious


def test_donor_pool_and_variants(tmp_path):
    paths = _write_fixtures(tmp_path)
    ds = PerturbationDataset("train", **paths)
    assert set(ds.donor_pool) == set(config.CONDITIONS)          # real per-condition donor pool loaded
    assert ds.donor_pool["Rest"].shape == (3, config.DONOR_PCA_DIMS)
    from tcell_pipeline.training.dataset import sample_donor_variants
    gen = torch.Generator().manual_seed(0)
    variants = sample_donor_variants(ds.donor_pool, ds.donor_mean, ["Rest", "Rest", "Rest"], 2, gen)
    assert len(variants) == 2 and all(v.shape == (3, config.DONOR_PCA_DIMS) for v in variants)
    assert not torch.allclose(variants[0], variants[1])          # distinct real donors -> real variation


def test_graph_loss_penalizes_gates_and_confidence_lowers_it():
    loss = StageALoss(gene_dim=_G, program_dim=_K)
    gate = torch.rand(5, requires_grad=True)
    g = loss._graph({"physical_ppi": [gate]})
    assert g > 0
    g.backward()
    assert gate.grad is not None
    gates = {"physical_ppi": [torch.rand(5)]}
    unsourced = loss._graph(gates)                                       # conf = 0 -> full L2
    sourced = loss._graph(gates, {"physical_ppi": [torch.ones(5)]})     # conf = 1 -> unsourced term zeroed
    assert sourced < unsourced


def test_stage_b_calibration_nll_and_gradient():
    loss = StageBCalibrationLoss()
    dz_hat = torch.randn(4, _K, requires_grad=True)
    val = loss(dz_hat, torch.randn(4, _K), torch.rand(4, _K) + 0.5)
    assert torch.isfinite(val)
    val.backward()
    assert dz_hat.grad is not None


def test_de_head_probs_in_unit_interval():
    head = DEHead(gene_dim=_G)
    up, down = head.probs(torch.randn(4, config.H_DO_DIM))
    for p in (up, down):
        assert p.shape == (4, _G) and (p >= 0).all() and (p <= 1).all()


def test_lambda_mixture_learnable():
    torch.manual_seed(0)
    dec = ProgramDecoder(torch.randn(_G, _K))
    out = dec(torch.randn(4, config.H_DO_DIM), torch.randn(4, config.GRAPH_HIDDEN_DIM))
    lam = out["lambda"]
    assert lam.shape == (4, 1) and (lam >= 0).all() and (lam <= 1).all()
    out["delta_z"].sum().backward()
    assert dec.gate.weight.grad is not None and torch.isfinite(dec.gate.weight.grad).all()


def test_dataset_keys_no_qpost_and_dz_sources(tmp_path):
    paths = _write_fixtures(tmp_path)
    ds = PerturbationDataset("train", **paths)
    assert len(ds) == 3                                  # rows 0,1,2 (targets G0,G1,G0)
    batch, target, cond, dz, dx, ri = ds[0]
    assert dz.shape == (_K,) and dx.shape == (_G,)
    assert target == "G0" and cond == "Rest" and ri == 0
    assert set(batch) & set(config.Q_POST_COLS) == set()  # leakage fence: q_post never enters features
    val_ds = PerturbationDataset("val", **paths)
    _, _, _, dz_val, dx_val, _ = val_ds[0]
    assert torch.allclose(dz_val, dx_val @ val_ds.B)     # out-of-fold role projects z onto frozen B


def test_trainer_runs_two_epochs_and_checkpoints(tmp_path):
    paths = _write_fixtures(tmp_path)
    train_ds = PerturbationDataset("train", **paths)
    val_ds = PerturbationDataset("val", **paths)
    trainer = Trainer(_expr_model(paths), train_ds, val_ds, max_epochs=2, patience=10, batch_size=2,
                      ckpt_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
    res = trainer.run()
    assert res["epochs_run"] == 2
    assert (tmp_path / "ckpt" / "stage_a_last.pt").exists()
    assert (tmp_path / "ckpt" / "stage_a_best.pt").exists()
    assert (tmp_path / "logs" / "stage_a_history.json").exists()
    assert res["best_ckpt"] is not None


def test_expr_only_training_updates_params(tmp_path):
    paths = _write_fixtures(tmp_path)
    model = _expr_model(paths)
    before = model.perturbation_encoder.fusion.weight.detach().clone()
    trainer = Trainer(model, PerturbationDataset("train", **paths), max_epochs=3, patience=10,
                      batch_size=2, ckpt_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
    trainer.run()
    assert not torch.allclose(before, model.perturbation_encoder.fusion.weight)  # optimisation moved params


def test_donor_invariance_produces_real_training_signal(tmp_path):
    paths = _write_fixtures(tmp_path)
    train_ds = PerturbationDataset("train", **paths)
    assert train_ds.donor_pool                                  # fixture donor profiles loaded
    trainer = Trainer(_expr_model(paths), train_ds, max_epochs=2, patience=10, batch_size=2,
                      ckpt_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", donor_invariance=True)
    assert trainer._donor_on                                    # real per-donor resampling is active
    before = trainer.loss.f_shared.weight.detach().clone()
    res = trainer.run()
    # the donor-variance term is non-zero on real donor draws (not the old inert group term) and trains f_shared
    assert any(e["train"]["invariance"] > 0 for e in res["history"])
    assert not torch.allclose(before, trainer.loss.f_shared.weight)

    off = Trainer(_expr_model(paths), PerturbationDataset("train", **paths), max_epochs=1, patience=10,
                  batch_size=2, ckpt_dir=tmp_path / "ckpt2", log_dir=tmp_path / "logs2",
                  donor_invariance=False)
    assert not off._donor_on
    assert off.run()["history"][0]["train"]["invariance"] == 0.0  # cleanly off, not spuriously firing
