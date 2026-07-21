"""Module 5 (Loss + Training) tests — fully synthetic: tiny fixture marts + zero embedding stores.

Stage A trains Module 1+2+3; Stage B is a loss module only. Covered: loss component shapes + gradient
flow, the DE head probability range, the learnable lambda mixture, the graph-gate penalty, the dataset
contract (correct keys, the q_post leakage fence, program_response vs out-of-fold projection), and a
2-epoch expression-only training run that actually writes checkpoints and moves parameters.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
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
    seeded_init,
)

_G, _K = 6, 3
_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)


def _write_fixtures(tmp_path, extra_val: int = 0) -> dict:
    genes = [f"G{i}" for i in range(_G)]
    rows = [(0, "G0"), (1, "G1"), (2, "G0"), (3, "G3"), (4, "G4")]  # 0-2 train, 3 val, 4 challenge
    rows += [(5 + i, "G3") for i in range(extra_val)]  # more val rows: the Stage-B gate needs n>1 units
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
        "basis_path": tmp_path / "loadings.parquet",
        "zscore_npz": tmp_path / "zscore.npz", "donor_profiles_path": tmp_path / "donor_profiles.parquet",
    }
    split.to_csv(p["split_path"], index=False)
    pc.to_parquet(p["pc_path"], index=False)
    obs.to_parquet(p["obs_path"], index=False)
    pd.DataFrame({"gene_name": genes}).to_parquet(p["var_path"], index=False)
    loadings.to_parquet(p["basis_path"], index=False)
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
    assert comps["invariance"] > 0                       # variance of Δz across the two donors
    comps["total"].backward()
    assert out["h_do"].grad is not None                  # gradient reaches the encoder input
    assert loss.de_head.head.weight.grad is not None     # ...the DE head
    assert dz_variants[0].grad is not None               # invariance flows to the donor-variant Δz (the encoder)
    assert not hasattr(loss, "f_shared")                 # no collapsible free projection to zero out


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


def test_dataset_keys_no_qpost_and_consistent_dz(tmp_path):
    paths = _write_fixtures(tmp_path)
    ds = PerturbationDataset("train", **paths)
    assert len(ds) == 3                                  # rows 0,1,2 (targets G0,G1,G0)
    batch, target, cond, dz, dx, ri = ds[0]
    assert dz.shape == (_K,) and dx.shape == (_G,)
    assert target == "G0" and cond == "Rest" and ri == 0
    assert set(batch) & set(config.Q_POST_COLS) == set()  # leakage fence: q_post never enters features
    assert torch.allclose(dz, dx @ ds.B)                 # train dz = z@B ...
    val_ds = PerturbationDataset("val", **paths)
    _, _, _, dz_val, dx_val, _ = val_ds[0]
    assert torch.allclose(dz_val, dx_val @ val_ds.B)     # ...same z@B definition out of fold (was A vs z@B)


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
    val_ds = PerturbationDataset("val", **paths)
    assert train_ds.donor_pool                                  # fixture donor profiles loaded
    trainer = Trainer(_expr_model(paths), train_ds, val_ds, max_epochs=2, patience=10, batch_size=2,
                      ckpt_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs", donor_invariance=True)
    assert trainer._donor_on                                    # real per-donor resampling is active
    res = trainer.run()
    # the donor-variance term (variance of Δz across real donors) is non-zero in TRAIN...
    assert any(e["train"]["invariance"] > 0 for e in res["history"])
    # ...but zero in VAL: the stochastic resample is train-only, so val_total stays deterministic
    assert all(e["val"]["invariance"] == 0.0 for e in res["history"])

    off = Trainer(_expr_model(paths), PerturbationDataset("train", **paths), max_epochs=1, patience=10,
                  batch_size=2, ckpt_dir=tmp_path / "ckpt2", log_dir=tmp_path / "logs2",
                  donor_invariance=False)
    assert not off._donor_on
    assert off.run()["history"][0]["train"]["invariance"] == 0.0  # cleanly off, not spuriously firing


def test_graph_penalty_is_batch_size_normalized():
    loss = StageALoss(gene_dim=_G, program_dim=_K)
    one = loss._graph({"physical_ppi": [torch.full((4,), 0.5)]})
    two = loss._graph({"physical_ppi": [torch.full((4,), 0.5), torch.full((4,), 0.5)]})
    assert torch.allclose(one, two)                      # mean over the batch, so bs doesn't scale the penalty


def test_seeded_init_reproducible_and_restores_global_rng():
    with seeded_init(123):
        a = torch.randn(4)
    with seeded_init(123):
        b = torch.randn(4)
    assert torch.allclose(a, b)                       # same seed -> identical weight-init draw
    before = torch.random.get_rng_state()             # ...and the context leaks nothing to the caller's RNG
    x = torch.randn(4)
    torch.random.set_rng_state(before)
    with seeded_init(999):
        torch.randn(10)                               # consume RNG inside the context
    assert torch.allclose(x, torch.randn(4))          # outer stream continues as if the context never ran


# --------------------------------------------------------------------------------------------------
# Stage B: the calibration FIT loop (feat-008 §a) + its controls
# --------------------------------------------------------------------------------------------------
class _ScaledTargets(torch.utils.data.Dataset):
    """A val set whose targets are wildly off-scale, so its NLL RISES while the train NLL falls. A loop
    that early-stopped on val would halt on this set at a different epoch than on a well-behaved one —
    which is what makes the 'no val statistic enters the fit' test able to fail."""

    def __init__(self, ds, k: float) -> None:
        self.ds, self.k = ds, k

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i):
        b, t, c, dz, dx, r = self.ds[i]
        return b, t, c, dz * self.k, dx, r


def _fit(paths, model=None, val=None, **kw):
    from tcell_pipeline.training.stage_b import fit_calibration
    if model is None:
        with seeded_init(0):  # identical starting weights across runs, else "same fit" compares two inits
            model = _expr_model(paths)
    res = fit_calibration(model, PerturbationDataset("train", **paths), val, batch_size=2,
                          ckpt_dir=kw.pop("ckpt_dir", None) or (paths["split_path"].parent / "ckpt"),
                          log_dir=paths["split_path"].parent / "logs", **kw)
    return model, res


def test_calibration_fit_moves_only_the_sigma_head(tmp_path):
    paths = _write_fixtures(tmp_path)
    model = _expr_model(paths)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _, res = _fit(paths, model=model, max_epochs=3)
    after = dict(model.named_parameters())
    moved = {n for n, p in after.items() if not torch.equal(p.detach(), before[n])}
    assert moved == {"decoder.uncertainty.weight", "decoder.uncertainty.bias"}   # and nothing else
    assert res["epochs_run"] == 3


def test_calibration_refuses_to_return_if_a_stage_a_weight_moved(tmp_path):
    """The freeze is ASSERTED, not intended: mutate a backbone weight against the snapshot and the
    check must raise. Constructed input — nothing in the fit loop can produce it."""
    from tcell_pipeline.training.stage_b import assert_backbone_frozen, calibration_parameters, frozen_snapshot
    paths = _write_fixtures(tmp_path)
    model = _expr_model(paths)
    snap = frozen_snapshot(model, calibration_parameters(model))
    assert_backbone_frozen(model, snap)                                  # clean model: silent
    with torch.no_grad():
        model.perturbation_encoder.fusion.weight[0, 0] += 1e-6           # one Stage-A weight, one ulp-ish
    with pytest.raises(RuntimeError, match="fusion.weight"):
        assert_backbone_frozen(model, snap)


def test_calibration_snapshot_excludes_only_the_calibration_head(tmp_path):
    from tcell_pipeline.training.stage_b import calibration_parameters, frozen_snapshot
    model = _expr_model(_write_fixtures(tmp_path))
    snap = frozen_snapshot(model, calibration_parameters(model))
    assert "decoder.uncertainty.weight" not in snap and "decoder.uncertainty.bias" not in snap
    assert len(snap) == len(list(model.named_parameters())) - 2


def test_no_val_statistic_enters_the_calibration_fit(tmp_path):
    """Fitted weights must be a pure function of (train, seed, hyper-params). Three different val sets —
    including none at all — must land on bit-identical calibration weights."""
    paths = _write_fixtures(tmp_path, extra_val=6)
    val = PerturbationDataset("val", **paths)
    fits = []
    for v in (None, val, _ScaledTargets(val, 50.0)):
        m, res = _fit(paths, val=v, max_epochs=6, patience=2)
        fits.append((m.decoder.uncertainty.weight.detach().clone(), res["epochs_run"]))
        assert res["val_nll"] is None or torch.isfinite(torch.tensor(res["val_nll"]))
    for w, e in fits[1:]:
        assert torch.equal(w, fits[0][0]) and e == fits[0][1]
    assert _fit(paths, val=val, max_epochs=6, patience=2)[1]["val_nll"] is not None  # val IS reported


def test_calibration_lowers_the_train_nll(tmp_path):
    paths = _write_fixtures(tmp_path)
    _, res = _fit(paths, max_epochs=8)
    assert res["history"][-1]["train_nll"] < res["history"][0]["train_nll"]
    assert res["best_ckpt"] is not None and Path(res["best_ckpt"]).exists()


def test_stage_b_checkpoints_are_seed_namespaced(tmp_path):
    """Both Stage-B heads write fixed filenames, so without a seed in the DIRECTORY a 5-seed sweep
    (config.N_FINAL_SEEDS) silently overwrites earlier seeds' artifacts. Stage A already learned this
    (commit 32fb473, screening.py: <root>/<name>/<seed>/ckpt)."""
    from tcell_pipeline.training.stage_b import fit_calibration, stage_b_ckpt_dir
    assert stage_b_ckpt_dir(0, tmp_path) != stage_b_ckpt_dir(1, tmp_path)
    assert "0" in stage_b_ckpt_dir(0, tmp_path).parts and "1" in stage_b_ckpt_dir(1, tmp_path).parts
    paths = _write_fixtures(tmp_path, extra_val=6)
    written = []
    for seed in (0, 1):
        with seeded_init(seed):
            model = _expr_model(paths)
        res = fit_calibration(model, PerturbationDataset("train", **paths), max_epochs=2, batch_size=2,
                              ckpt_dir=stage_b_ckpt_dir(seed, tmp_path), log_dir=tmp_path / "logs")
        written.append(Path(res["best_ckpt"]))
    assert written[0] != written[1]                      # two seeds -> two artifacts...
    assert all(p.exists() for p in written)              # ...and the first was not clobbered
    a, b = (torch.load(p, weights_only=True)["calibration"]["weight"] for p in written)
    assert not torch.equal(a, b)                         # they really are different fits


def test_calibration_leaves_the_model_at_the_BEST_epoch_not_the_last(tmp_path):
    """Early stopping means the run ENDS on a worse epoch than the one it checkpointed. The freeze gate
    scores the live model, so if the fit left the last epoch's weights in memory the verdict would
    describe a head that was never saved. lr=5.0 makes the fit overshoot so later epochs really are
    worse — without that, best and last coincide and the test proves nothing."""
    from tcell_pipeline.training.stage_b import fit_calibration
    paths = _write_fixtures(tmp_path, extra_val=6)
    with seeded_init(0):
        model = _expr_model(paths)
    res = fit_calibration(model, PerturbationDataset("train", **paths),
                          PerturbationDataset("val", **paths), lr=5.0, max_epochs=30, patience=3,
                          batch_size=2, ckpt_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
    hist = [h["train_nll"] for h in res["history"]]
    assert res["epochs_run"] > 1 and hist[-1] > min(hist)     # the run really did end on a worse epoch
    best_i = min(range(len(hist)), key=lambda i: hist[i])
    assert res["best_epoch"] == best_i
    assert res["train_nll"] == pytest.approx(hist[best_i])   # reported metric describes the ARTIFACT...
    ckpt = torch.load(res["best_ckpt"], weights_only=True)
    for k, v in ckpt["calibration"].items():                  # ...and so do the in-memory weights
        assert torch.equal(v, dict(model.decoder.uncertainty.state_dict())[k])


def test_calibration_checkpoint_reloads_the_fitted_head(tmp_path):
    paths = _write_fixtures(tmp_path)
    model, res = _fit(paths, max_epochs=3)
    fresh = _expr_model(paths)
    ckpt = torch.load(res["best_ckpt"], weights_only=True)
    fresh.decoder.uncertainty.load_state_dict(ckpt["calibration"])
    assert torch.equal(fresh.decoder.uncertainty.weight, model.decoder.uncertainty.weight)


def test_constant_sigma_control_is_fit_on_train_only(tmp_path):
    """The control must be as blind to val as the head is; a control fit on the evaluation rows would
    flatter or damn the head for the wrong reason."""
    from tcell_pipeline.training.stage_b import calibration_contrasts
    paths = _write_fixtures(tmp_path, extra_val=6)
    model = _expr_model(paths)
    train, val = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    a = calibration_contrasts(model, train, val, batch_size=4)
    b = calibration_contrasts(model, train, _ScaledTargets(val, 50.0), batch_size=4)
    assert a["constant_sigma"] == b["constant_sigma"]                     # per-program sigma, train-derived
    assert len(a["constant_sigma"]) == _K


def test_calibration_contrasts_are_paired_on_the_same_rows(tmp_path):
    from tcell_pipeline.training.stage_b import calibration_contrasts
    paths = _write_fixtures(tmp_path, extra_val=6)
    c = calibration_contrasts(_expr_model(paths), PerturbationDataset("train", **paths),
                              PerturbationDataset("val", **paths), batch_size=4)
    for name in ("vs_constant_sigma", "vs_permuted_sigma"):
        spec = c["contrasts"][name]
        assert set(spec["fit"]) == set(spec["control"]) == set(spec["units"])
        assert len(spec["units"]) == 7                                    # 1 + 6 val rows, none lost
        assert spec["higher_is_better"] is False                          # NLL: lower is better
    # the permuted control must actually BE permuted: an un-permuted one is the fit itself, which
    # would hand every head a free 0.0 advantage against a control that is not a control
    perm = c["contrasts"]["vs_permuted_sigma"]
    assert any(perm["control"][u] != perm["fit"][u] for u in perm["units"])
    assert sorted(perm["control"].values()) != sorted(perm["fit"].values())  # ...and re-paired, not resorted


def test_permuted_control_is_averaged_over_draws_not_a_single_permutation(tmp_path):
    """One permutation IS a random variable: on the real fold a single-draw control moved this
    contrast's raw p from 0.2494 to 0.1506 between two runs differing only in the seed. Averaging over
    draws is what makes the control a property of the fitted head rather than of the seed."""
    from tcell_pipeline.training.stage_b import calibration_contrasts
    paths = _write_fixtures(tmp_path, extra_val=20)
    with seeded_init(0):
        model = _expr_model(paths)
    train, val = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)

    def spread(**kw):
        means = []
        for seed in range(5):
            c = calibration_contrasts(model, train, val, batch_size=8, seed=seed, **kw)
            assert c["n_permutations"] == kw.get("n_permutations", c["n_permutations"])
            ctl = c["contrasts"]["vs_permuted_sigma"]["control"]
            means.append(sum(ctl.values()) / len(ctl))
        return max(means) - min(means)

    single = spread(n_permutations=1)
    averaged = spread()
    assert single > 0                                    # a single draw really does depend on the seed
    assert averaged < single / 2                         # averaging damps it


def test_a_collapsed_calibration_head_cannot_clear_the_gate(tmp_path):
    """Collapse-to-a-constant is the failure mode a calibration head hits here. A collapsed head is
    IDENTICAL to its own permuted control, so the paired advantage is exactly zero — undecidable, not
    a spurious win (the same trap that scored numerical dust at +0.0129 on the primary endpoint)."""
    from tcell_pipeline.training.freeze_gate import FREEZE, evaluate_gate
    from tcell_pipeline.training.stage_b import calibration_contrasts
    paths = _write_fixtures(tmp_path, extra_val=6)
    model = _expr_model(paths)
    with torch.no_grad():
        model.decoder.uncertainty.weight.zero_()                          # sigma ignores its input
    c = calibration_contrasts(model, PerturbationDataset("train", **paths),
                              PerturbationDataset("val", **paths), batch_size=4)
    perm = c["contrasts"]["vs_permuted_sigma"]
    assert all(perm["fit"][u] == perm["control"][u] for u in perm["units"])
    r = evaluate_gate(c["contrasts"])
    assert r["decision"] != FREEZE
    assert r["contrasts"]["vs_permuted_sigma"]["ci_excludes_zero"] is None  # zero spread -> undecidable


def test_calibration_fit_asserts_the_freeze_when_something_moves_the_backbone(tmp_path):
    """The loop's freeze check must be WIRED, not just available: a loss that writes to a Stage-A weight
    mid-fit is a constructed input that reaches it."""
    paths = _write_fixtures(tmp_path)
    model = _expr_model(paths)

    class _Saboteur(StageBCalibrationLoss):
        def forward(self, dz_hat, dz_true, sigma):
            with torch.no_grad():
                model.perturbation_encoder.fusion.weight += 1e-4
            return super().forward(dz_hat, dz_true, sigma)

    with pytest.raises(RuntimeError, match="not frozen"):
        _fit(paths, model=model, val=None, loss=_Saboteur(), max_epochs=2)
    assert model.perturbation_encoder.fusion.weight.requires_grad   # ...and the caller's model is restored


def test_calibration_forces_eval_and_restores_the_callers_mode(tmp_path):
    """The cached backbone outputs are only constants if the backbone runs in eval (DropEdge off), and
    the caller's train/eval + requires_grad state must survive the fit unchanged."""
    paths = _write_fixtures(tmp_path)
    with seeded_init(0):
        model = _expr_model(paths)
    model.train()
    seen, orig = [], model.forward
    model.forward = lambda *a, **k: (seen.append(model.training), orig(*a, **k))[1]
    _fit(paths, model=model, max_epochs=2)
    assert seen and not any(seen)                       # every frozen forward ran in eval
    assert model.training                               # ...and train mode came back
    assert all(p.requires_grad for p in model.parameters())


def test_duplicate_row_ids_are_refused_not_silently_dropped(tmp_path):
    """Row ids key the paired contrast; duplicates would collapse into one dict entry and shrink n while
    the gate still printed a clean verdict."""
    from tcell_pipeline.training.stage_b import calibration_contrasts
    paths = _write_fixtures(tmp_path, extra_val=6)
    val = PerturbationDataset("val", **paths)
    val.row_index = np.full(len(val), 3)                # the mart handed us a non-unique row_index
    with pytest.raises(ValueError, match="uniquely identified"):
        calibration_contrasts(_expr_model(paths), PerturbationDataset("train", **paths), val, batch_size=4)


def test_an_empty_gate_split_is_refused_not_crashed_through(tmp_path):
    from tcell_pipeline.training.stage_b import calibration_contrasts
    paths = _write_fixtures(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        calibration_contrasts(_expr_model(paths), PerturbationDataset("train", **paths),
                              PerturbationDataset("val", n_max=0, **paths), batch_size=4)


def test_per_row_nll_matches_the_fitted_objective():
    """The statistic the gate tests must be the loss the fit minimises, not a lookalike."""
    from tcell_pipeline.training.stage_b import per_row_nll
    torch.manual_seed(0)
    a, b, s = torch.randn(7, _K), torch.randn(7, _K), torch.rand(7, _K) + 0.5
    assert float(per_row_nll(a, b, s).mean()) == pytest.approx(float(StageBCalibrationLoss()(a, b, s)), rel=1e-6)


def test_gate_clears_when_the_sigma_really_tracks_the_residual():
    """The clear path through the real per-row NLL: an oracle sigma (= |residual|) genuinely beats a
    constant one, so the gate must not be refusing everything by construction."""
    from tcell_pipeline.training.freeze_gate import FREEZE, evaluate_gate
    from tcell_pipeline.training.stage_b import per_row_nll
    torch.manual_seed(0)
    n = 24
    dz_true = torch.randn(n, _K) * torch.linspace(0.1, 5.0, n).unsqueeze(1)   # real heteroscedasticity
    dz_hat = torch.zeros(n, _K)
    oracle = dz_true.abs().clamp_min(1e-3)
    const = dz_true.pow(2).mean(0, keepdim=True).sqrt().expand(n, _K)
    fit = per_row_nll(dz_hat, dz_true, oracle)
    ctl = per_row_nll(dz_hat, dz_true, const)
    assert fit.shape == (n,)
    spec = {"fit": {i: float(fit[i]) for i in range(n)}, "control": {i: float(ctl[i]) for i in range(n)},
            "higher_is_better": False}
    assert evaluate_gate({"vs_constant_sigma": spec})["decision"] == FREEZE


def test_empty_train_split_raises_clear_error(tmp_path):
    paths = _write_fixtures(tmp_path)
    empty = PerturbationDataset("train", n_max=0, **paths)
    assert len(empty) == 0
    with pytest.raises(ValueError, match="empty"):       # clear message, not an opaque RandomSampler error
        Trainer(_expr_model(paths), empty, max_epochs=1, patience=10, batch_size=2,
                ckpt_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
