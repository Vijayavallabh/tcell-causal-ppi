"""Module 7 (Screening + Registry, feat-011) tests — fully synthetic (tiny marts + a small PPI graph).

Covers: a single config trained one epoch that evaluates, writes predictions + a metrics row, and returns
the metric suite; a two-config screening run that reports the H2a contrast and a summary JSON; the registry
registering/logging a run; the 32-trial EG-IPG cap; and that a failing config is still logged (status
``failed``) before re-raising.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

torch.set_num_threads(1)  # many-core box: tiny GNN/linear ops thrash the default thread pool otherwise

from tcell_pipeline import config
from tcell_pipeline.encoders import PerturbationEncoder
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.evaluation.output_schema import prediction_path, read_predictions
from tcell_pipeline.graph import build_hetero_graph
from tcell_pipeline.screening import (
    EXPRESSION_ONLY,
    NETWORK_PROP,
    TYPED_STATIC,
    collect_predictions,
    collect_truth,
    load_registry,
    log_run,
    nested_family_configs,
    register_run,
    run_screening,
    score_network_propagation,
    screen_config,
)
from tcell_pipeline.screening.screening import _finite_or_none
from tcell_pipeline.training import PerturbationDataset

_G, _K = 6, 3
_ZERO_PLM = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PLM_EMBED_DIM)
_ZERO_PIN = PluggableEmbeddingStore(config.INTERMEDIATE_ROOT / "nope.parquet", config.PINNACLE_EMBED_DIM)
_ENC = lambda: PerturbationEncoder(_ZERO_PLM, _ZERO_PIN)  # a fresh zero-embedding encoder per model


def _edge(src, dst, source, score, phys=0, func=0, cplx=0, direct=0, nsup=1):
    return dict(source_gene=src, target_gene=dst, source=source, evidence_type="x", score=score,
                is_physical=phys, is_functional=func, is_complex=cplx, is_direct_binary=direct,
                n_supporting_sources=nsup)


def _graph():
    edges = pd.DataFrame([
        _edge("G0", "G1", "biogrid", 0.9, phys=1), _edge("G1", "G2", "biogrid", 0.8, phys=1),
        _edge("G2", "G3", "string", 0.5, func=1), _edge("G3", "G4", "biogrid", 0.7, phys=1),
        _edge("G4", "G5", "corum", 0.6, cplx=1),
    ])
    complexes = pd.DataFrame([
        dict(protein_gene="G0", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
        dict(protein_gene="G1", complex_id=1, source_database="CORUM", confidence=1.0, is_curated=1),
    ])
    id_map = pd.DataFrame([dict(hgnc_symbol="G0", uniprot_id="P0"), dict(hgnc_symbol="G1", uniprot_id="P1")])
    baseline = pd.DataFrame([dict(hgnc_symbol="G0", control_baseline_expr=1.0)])
    return build_hetero_graph(edges, complexes, id_map, baseline, plm_store=_ZERO_PLM, pinnacle_store=_ZERO_PIN)


def _write_marts(tmp_path) -> dict:
    genes = [f"G{i}" for i in range(_G)]
    rows = [(0, "G0"), (1, "G1"), (2, "G0"), (3, "G3"), (4, "G4")]  # 0-2 train, 3 val, 4 challenge
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
    split = pd.DataFrame({"hgnc_symbol": ["G0", "G1", "G3", "G4"], "role": ["train", "train", "val", "challenge"]})
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


def _fixture(tmp_path):
    paths = _write_marts(tmp_path)
    graph, g2i = _graph()
    gene_names = pd.read_parquet(paths["var_path"])["gene_name"].tolist()
    return paths, graph, g2i, gene_names


def _configs(tmp_path, names, epochs=1):
    paths, graph, g2i, gene_names = _fixture(tmp_path)
    cfgs = nested_family_configs(gene_names, graph, g2i, epochs, names=names, basis_path=paths["basis_path"],
                                 perturbation_encoder_factory=_ENC, batch_size=2)
    for c in cfgs:
        c["donor_invariance"] = False  # keep the synthetic run fast; donor term is Module-5-tested
    return paths, cfgs


def test_screen_config_one_epoch_trains_evaluates_and_writes(tmp_path):
    paths, cfgs = _configs(tmp_path, [TYPED_STATIC])
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    train_mean = collect_truth(train_ds)["delta_z"].mean(0)
    res = screen_config(cfgs[0], train_ds, val_ds, train_mean, ckpt_dir=tmp_path / "ck",
                        log_dir=tmp_path / "lg", predictions_root=tmp_path / "pred",
                        screening_root=tmp_path / "scr")
    assert {"systema", "pearson", "mae", "topk", "primary"} <= set(res)
    assert np.isfinite(res["systema"]) and res["primary"] == res["systema"]
    back = read_predictions(prediction_path(TYPED_STATIC, "val", 0, root=tmp_path / "pred"))
    assert back["delta_z"].shape == (len(val_ds), _K)
    assert (tmp_path / "scr" / TYPED_STATIC / "0.parquet").exists()   # per-config metrics table


def test_run_screening_two_configs_reports_h2a(tmp_path):
    paths, cfgs = _configs(tmp_path, [EXPRESSION_ONLY, TYPED_STATIC])
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    summary = run_screening(cfgs, train_ds, val_ds, predictions_root=tmp_path / "pred",
                            screening_root=tmp_path / "scr")
    assert len(summary["results"]) == 2
    assert "h2a" in summary                                          # both nested members present
    # pin the contrast to the actual per-config systema values, so an inverted sign/direction fails here
    by = {r["name"]: r for r in summary["results"]}
    expected_delta = by[TYPED_STATIC]["systema"] - by[EXPRESSION_ONLY]["systema"]
    assert summary["h2a"]["better"] == TYPED_STATIC and summary["h2a"]["worse"] == EXPRESSION_ONLY
    assert summary["h2a"]["delta"] == pytest.approx(expected_delta)
    assert summary["h2a"]["supported"] == (by[TYPED_STATIC]["systema"] > by[EXPRESSION_ONLY]["systema"])
    from pathlib import Path
    assert Path(summary["summary_path"]).exists()                    # summary JSON written


def test_run_screening_isolates_failed_config(tmp_path):
    paths, cfgs = _configs(tmp_path, [EXPRESSION_ONLY])          # one good config
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)

    def boom():
        raise RuntimeError("lane down")

    bad = {"name": TYPED_STATIC, "model_factory": boom, "n_epochs": 1, "seed": 0}
    summary = run_screening([cfgs[0], bad], train_ds, val_ds, predictions_root=tmp_path / "pred",
                            screening_root=tmp_path / "scr")
    by = {r["name"]: r for r in summary["results"]}
    assert by[EXPRESSION_ONLY]["status"] == "completed"          # good lane still ran
    assert by[TYPED_STATIC]["status"] == "failed" and "lane down" in by[TYPED_STATIC]["error"]
    assert "h2a" not in summary                                  # failed member -> contrast omitted, not a crash


def test_run_screening_scores_network_propagation(tmp_path):
    paths, graph, g2i, gene_names = _fixture(tmp_path)
    cfgs = nested_family_configs(gene_names, graph, g2i, 1, names=[EXPRESSION_ONLY],
                                 basis_path=paths["basis_path"], perturbation_encoder_factory=_ENC, batch_size=2)
    for c in cfgs:
        c["donor_invariance"] = False
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    B = train_ds.B.numpy()

    def netprop(tr, va, tm, *, predictions_root, screening_root, split):
        return score_network_propagation(tr, va, tm, graph=graph, gene_to_idx=g2i, basis=B, seed=0,
                                         batch_size=2, predictions_root=predictions_root,
                                         screening_root=screening_root, split=split)
    netprop.screen_name = NETWORK_PROP

    summary = run_screening(cfgs, train_ds, val_ds, predictions_root=tmp_path / "pred",
                            screening_root=tmp_path / "scr", extra_scorers=[netprop])
    by = {r["name"]: r for r in summary["results"]}
    assert by[NETWORK_PROP]["status"] == "completed"                  # the topology-diffusion reference ran
    assert np.isfinite(by[NETWORK_PROP]["systema"]) and "mae" in by[NETWORK_PROP]
    back = read_predictions(prediction_path(NETWORK_PROP, "val", 0, root=tmp_path / "pred"))
    assert back["delta_z"].shape == (len(val_ds), _K)                # emitted the common output schema
    assert "h2a" not in summary or summary["h2a"]["better"] != NETWORK_PROP  # not in the nested comparison


def test_summary_json_sanitizes_non_finite(tmp_path):
    import json
    dirty = {"results": [{"name": "x", "mae": float("nan"), "rmse": float("inf"),
                          "best_val": float("-inf"), "systema": 0.5, "seed": 0, "status": "completed"}]}
    clean = _finite_or_none(dirty)
    row = clean["results"][0]
    assert row["mae"] is None and row["rmse"] is None and row["best_val"] is None  # non-finite -> None
    assert row["systema"] == 0.5 and row["seed"] == 0                              # finite values untouched
    assert json.loads(json.dumps(clean, allow_nan=False))["results"][0]["mae"] is None  # valid RFC-8259 JSON
    assert _finite_or_none(np.float32("nan")) is None and _finite_or_none(np.float64("inf")) is None


def test_registry_registers_and_logs(tmp_path):
    reg = tmp_path / "registry.yaml"
    rid = register_run("expression_only", "H2a", "q_pre", "blocked_target_ood", 0, {"gpu_hours": 2}, path=reg)
    assert rid == "run-0001"
    entry = log_run(rid, "completed", {"systema": 0.12}, "best.pt", 1.5, path=reg)
    assert entry["status"] == "completed" and entry["metrics"]["systema"] == 0.12
    runs = load_registry(reg)
    assert len(runs) == 1 and runs[0]["run_id"] == rid and runs[0]["checkpoint"] == "best.pt"


def test_registry_load_tolerates_null_and_missing(tmp_path):
    assert load_registry(tmp_path / "absent.yaml") == []      # missing file -> empty, not crash
    null_reg = tmp_path / "null.yaml"
    null_reg.write_text("runs:\n")                            # present-but-null runs key (truncated file)
    assert load_registry(null_reg) == []
    register_run("cfg", "H1", "q_pre", "blocked", 0, None, path=null_reg)  # register on top must not crash
    assert len(load_registry(null_reg)) == 1


def test_registry_enforces_egipg_cap(tmp_path):
    reg = tmp_path / "registry.yaml"
    for i in range(config.MAX_EGIPG_TRIALS):
        register_run(f"cfg{i}", "H1", "q_pre", "blocked", 0, None, path=reg)
    with pytest.raises(ValueError, match="cap reached"):
        register_run("one_too_many", "H1", "q_pre", "blocked", 0, None, path=reg)
    # a comparator family keeps its own separate (smaller) budget, unaffected by the EG-IPG count
    rid = register_run("comp", "H1", "q_pre", "blocked", 0, None, family="comparator_a", path=reg)
    assert rid.startswith("run-")


def test_registry_cap_counts_distinct_configs(tmp_path):
    reg = tmp_path / "registry.yaml"
    # re-registering the SAME config_id (dev re-runs / retries) never grows the distinct count or trips the cap
    for _ in range(config.MAX_EGIPG_TRIALS + 5):
        register_run("same_cfg", "H1", "q_pre", "blocked", 0, None, path=reg)
    runs = load_registry(reg)
    assert len(runs) == config.MAX_EGIPG_TRIALS + 5          # every execution still logged (complete audit trail)
    assert len({r["config_id"] for r in runs}) == 1          # ...but only one DISTINCT config counts toward the cap
    assert register_run("other_cfg", "H1", "q_pre", "blocked", 0, None, path=reg).startswith("run-")  # new config OK


def test_screen_config_logs_failure_then_reraises(tmp_path):
    paths, _cfgs = _configs(tmp_path, [EXPRESSION_ONLY])
    reg = tmp_path / "registry.yaml"
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)

    def boom():
        raise RuntimeError("factory boom")

    cfg = {"name": EXPRESSION_ONLY, "model_factory": boom, "n_epochs": 1, "seed": 0}
    with pytest.raises(RuntimeError, match="boom"):
        screen_config(cfg, train_ds, val_ds, np.zeros(_K), registry_path=reg,
                      screening_root=tmp_path / "scr", predictions_root=tmp_path / "pred")
    runs = load_registry(reg)
    assert runs[-1]["status"] == "failed" and "boom" in runs[-1]["metrics"]["error"]


# --- Tier 2 fixes ---------------------------------------------------------------------------------
def test_screen_config_seed_namespaced_ckpt_and_gpu_hours(tmp_path):
    paths, cfgs = _configs(tmp_path, [EXPRESSION_ONLY])
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    train_mean = collect_truth(train_ds)["delta_z"].mean(0)
    reg, scr = tmp_path / "registry.yaml", tmp_path / "scr"
    for seed in (0, 1):                                            # two seeds of the SAME config name
        cfg = dict(cfgs[0], seed=seed)
        res = screen_config(cfg, train_ds, val_ds, train_mean, screening_root=scr,
                            predictions_root=tmp_path / "pred", registry_path=reg)
        assert res["gpu_hours"] >= 0 and np.isfinite(res["gpu_hours"])
    # both seeds' best checkpoints coexist — seed 1 did NOT overwrite seed 0
    assert (scr / EXPRESSION_ONLY / "0" / "ckpt" / "stage_a_best.pt").exists()
    assert (scr / EXPRESSION_ONLY / "1" / "ckpt" / "stage_a_best.pt").exists()
    runs = load_registry(reg)
    assert len(runs) == 2 and all(r["gpu_hours"] is not None for r in runs)  # gpu_hours audit field populated


def test_nested_family_configs_empty_names_gives_no_configs(tmp_path):
    paths, graph, g2i, gene_names = _fixture(tmp_path)
    assert nested_family_configs(gene_names, graph, g2i, 1, names=[], basis_path=paths["basis_path"]) == []
    assert len(nested_family_configs(gene_names, graph, g2i, 1, names=None,
                                     basis_path=paths["basis_path"])) == 3  # None -> the three nested members


def test_registry_caps_comparator_families(tmp_path):
    reg = tmp_path / "registry.yaml"
    register_run("c1", "H1", "q_pre", "blocked", 0, None, family="comparator_a", path=reg)
    register_run("c2", "H1", "q_pre", "blocked", 0, None, family="comparator_b", path=reg)  # 2nd family OK
    with pytest.raises(ValueError, match="comparator-family cap"):
        register_run("c3", "H1", "q_pre", "blocked", 0, None, family="comparator_c", path=reg)  # 3rd rejected
    # an EXISTING comparator family still accepts more configs, and egipg is unaffected by this cap
    assert register_run("c1b", "H1", "q_pre", "blocked", 0, None, family="comparator_a", path=reg).startswith("run-")
    assert register_run("e1", "H1", "q_pre", "blocked", 0, None, path=reg).startswith("run-")


# --- Tier 3 test-strength gaps --------------------------------------------------------------------
def _forward_val_dz(factory, ckpt_path, val_ds):
    m = factory()
    m.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    return collect_predictions(m, val_ds, "cpu", 2)["delta_z"]


def test_screen_config_scores_best_not_last_checkpoint(tmp_path):
    # No external seeding: screen_config now seeds weight init from the config seed (via seeded_init), so the
    # run is deterministic regardless of test order. Under seed 0, lr=0.5 / 5 epochs gives best-val at epoch
    # 1 and last at epoch 4, so best != last and the "score the best checkpoint, not the last" contract is
    # genuinely exercised — a regression scoring last-epoch weights fails here.
    paths, cfgs = _configs(tmp_path, [EXPRESSION_ONLY], epochs=5)
    cfg = dict(cfgs[0], lr=0.5)                                   # cfg["seed"] == 0 (from nested_family_configs)
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    train_mean = collect_truth(train_ds)["delta_z"].mean(0)
    ck = tmp_path / "ck"
    screen_config(cfg, train_ds, val_ds, train_mean, ckpt_dir=ck, log_dir=tmp_path / "lg",
                  predictions_root=tmp_path / "pred", screening_root=tmp_path / "scr")
    written = read_predictions(prediction_path(EXPRESSION_ONLY, "val", 0, root=tmp_path / "pred"))["delta_z"]
    factory = cfgs[0]["model_factory"]
    dz_best = _forward_val_dz(factory, ck / "stage_a_best.pt", val_ds)
    dz_last = _forward_val_dz(factory, ck / "stage_a_last.pt", val_ds)
    assert not np.allclose(dz_best, dz_last)                              # non-vacuous: best epoch != last epoch
    assert np.allclose(written, dz_best.astype("float32"), atol=1e-4)     # scored the BEST-val weights...
    assert not np.allclose(written, dz_last.astype("float32"), atol=1e-4)  # ...not the last epoch's


def test_screen_config_logs_completed_run(tmp_path):
    paths, cfgs = _configs(tmp_path, [EXPRESSION_ONLY])
    train_ds, val_ds = PerturbationDataset("train", **paths), PerturbationDataset("val", **paths)
    train_mean = collect_truth(train_ds)["delta_z"].mean(0)
    reg = tmp_path / "registry.yaml"
    res = screen_config(cfgs[0], train_ds, val_ds, train_mean, registry_path=reg,
                        screening_root=tmp_path / "scr", predictions_root=tmp_path / "pred")
    r = load_registry(reg)[0]
    assert r["status"] == "completed" and r["config_id"] == EXPRESSION_ONLY   # flipped from 'registered'
    assert r["metrics"]["systema"] == pytest.approx(float(res["systema"]))     # metrics logged in the right arg slot
    assert r["checkpoint"] and "stage_a_best.pt" in r["checkpoint"]            # checkpoint path (not metrics) logged
    assert r["gpu_hours"] is not None and r["gpu_hours"] >= 0                  # completed-path gpu_hours recorded
