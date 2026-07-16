"""Module 6 real-data smoke: score the simple baselines (and the trained Stage-A model, if a checkpoint
exists) on the real val fold with the evaluation metrics + the G2-MQ gate + the §10.5 control safeguards.

    PYTHONPATH=src python -m tcell_pipeline.run_module6_smoke --device cuda

Metrics/baselines are numpy/sklearn (CPU) — GPU applies only to the model forward (the encoders), which is
run on ``--device`` when a Stage-A checkpoint is present. Δz_true = z@B and the features are pulled straight
from the dataset's CSR (no 21k per-row __getitem__).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.baselines import BASELINES  # noqa: E402
from tcell_pipeline.encoders.batch import DONOR_COLS  # noqa: E402
from tcell_pipeline.evaluation import control_reference as cr  # noqa: E402
from tcell_pipeline.evaluation import metric_qualification as mq  # noqa: E402
from tcell_pipeline.evaluation import metrics as M  # noqa: E402
from tcell_pipeline.evaluation.output_schema import read_predictions, write_predictions  # noqa: E402
from tcell_pipeline.model import EGIPGModel  # noqa: E402
from tcell_pipeline.training.dataset import PerturbationDataset  # noqa: E402

_FEAT = ["ppi_degree_physical", "ppi_degree_functional", "ppi_degree_complex", "control_baseline_expr"]


def _fold(ds: PerturbationDataset):
    """(X features, Δz_true=z@B, Δx_true=zscore, conditions, row_index) straight from the CSR — vectorised."""
    Z = ds._zscore[ds.row_index].toarray().astype("float32")   # (N, G) zscore
    dz = Z @ ds.B.numpy()                                       # (N, K) z@B
    # control_baseline_expr is NaN for ~1.5k real rows (the encoder imputes internally); 0-fill for the
    # sklearn baselines, which reject NaN in X
    X = np.nan_to_num(ds.pc[_FEAT + DONOR_COLS].to_numpy("float32"))
    return X, dz, Z, ds.pc["culture_condition"].tolist(), ds.row_index


def _model_predict(ds: PerturbationDataset, device: str):
    """Forward the trained expr-only Stage-A model over the val fold on `device`; None if no checkpoint."""
    ckpt = config.CHECKPOINTS_ROOT / "stage_a_best.pt"
    if not ckpt.exists():
        return None
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()
    model = EGIPGModel.from_saved_basis(gene_names, graph_encoder=None).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.eval()
    dz, dx = [], []
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=PerturbationDataset.collate)
    with torch.no_grad():
        for batch, targets, conditions, *_ in loader:
            out = model(batch, targets, conditions)
            dz.append(out["delta_z"].detach().cpu().numpy())
            dx.append(out["delta_x"].detach().cpu().numpy())
    return np.concatenate(dz), np.concatenate(dx)


def _score(name, dz_hat, dx_hat, dz_true, dx_true, train_mean) -> dict:
    return {
        "model": name,
        "pearson": M.pearson_corr(dz_hat, dz_true),
        "systema": M.systema_pert_specific_delta(dz_hat, dz_true, train_mean),
        "centroid": M.centroid_accuracy(dz_hat, dz_true),
        "prog_cos": M.program_cosine(dz_hat, dz_true),
        "mae": M.mae(dx_hat, dx_true),
        "rmse": M.rmse(dx_hat, dx_true),
        "topk": M.topk_recall(dx_hat, dx_true),
        "sign": M.sign_accuracy(dx_hat, dx_true),
    }


def run(device: str = "cpu", seed: int = 0) -> int:
    torch.set_num_threads(1)
    train, val = PerturbationDataset("train"), PerturbationDataset("val")
    Xtr, ztr, _, ctr, _ = _fold(train)
    Xva, zva, dxva, cva, ri = _fold(val)
    train_mean = ztr.mean(0)
    B = train.B.numpy()
    print(f"[m6] train {len(ztr)} / val {len(zva)} rows; K={ztr.shape[1]} G={dxva.shape[1]}; device={device}")

    rows = []
    mp = _model_predict(val, device)
    if mp is not None:
        rows.append(_score("egipg (trained)", mp[0], mp[1], zva, dxva, train_mean))
        print(f"[m6] scored trained model on {device}")
    else:
        print("[m6] no Stage-A checkpoint — baselines only (run run_train --expr-only --device cuda first)")

    for key, cls in BASELINES.items():
        model = cls(basis=B).fit(Xtr, ztr, conditions=ctr)
        dz_hat, dx_hat = model.predict(Xva, conditions=cva)
        rows.append(_score(key, dz_hat, dx_hat, zva, dxva, train_mean))

    cols = ["model", "pearson", "systema", "centroid", "prog_cos", "mae", "rmse", "topk", "sign"]
    print("\n" + " ".join(f"{c:>16}" if c == "model" else f"{c:>9}" for c in cols))
    for r in rows:
        print(" ".join(f"{r['model']:>16}" if c == "model" else f"{r[c]:>9.4f}" for c in cols))

    # --- G2-MQ gate on the real val fold (candidate metric = the primary systema endpoint) ---
    rng = np.random.default_rng(seed)
    fn = lambda p, t: M.systema_pert_specific_delta(p, t, train_mean)
    neg = {"zero": (mq.zero_prediction(zva), zva), "perturbed_mean": (mq.perturbed_mean_prediction(zva), zva),
           "label_perm": (mq.label_permutation(zva, rng), zva), "row_shuffle": (mq.row_shuffle(zva, rng), zva)}
    pos = {"guide_split_half": (mq.guide_split_half(zva, rng, 0.3), zva), "oracle": (mq.oracle_prediction(zva), zva)}
    g2 = mq.qualify_metric(fn, neg, pos)
    print(f"\n[m6] G2-MQ systema: passed={g2['passed']} range={g2['dynamic_range']:.4f} "
          f"neg={ {k: round(v,4) for k,v in g2['neg_scores'].items()} } pos={ {k: round(v,4) for k,v in g2['pos_scores'].items()} }")
    assert g2["passed"], "primary endpoint failed the model-blind G2-MQ ordering on real data"

    # --- §10.5 control-reference: null predictor ~0 under independent controls ---
    ctrl_a, ctrl_b = rng.standard_normal(zva.shape), rng.standard_normal(zva.shape)
    null = cr.null_control_predictor(ctrl_a)
    indep = cr.independent_control_metric(null, zva, ctrl_a, ctrl_b)
    print(f"[m6] control-reference null-predictor under independent controls: {indep:.2e} (~0 expected)")
    assert abs(indep) < 1e-6

    # --- common output schema on real shapes (roundtrip) ---
    pm = BASELINES["perturbed_mean"](basis=B).fit(Xtr, ztr)
    dz_hat, dx_hat = pm.predict(Xva, conditions=cva)
    path = write_predictions(ri, dz_hat, dx_hat, None, "perturbed_mean", "val", seed)
    back = read_predictions(path)
    assert np.allclose(back["delta_z"], dz_hat.astype("float32")) and np.array_equal(back["row_index"], ri)
    print(f"[m6] wrote + read {len(ri)} predictions -> {path}")
    print("[m6] OK")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu | cuda (model forward only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sys.exit(run(a.device, a.seed))
