"""Module 3 real-data smoke: fit a fold-local basis on real train rows, run M1->M2->M3 end to end.

Fits a fast SVD basis on the real train-split rows (the paper default sparse_pca is a deliberate
~15-min run via run_program_basis; SVD keeps the smoke to seconds while still exercising the real
fold-local extraction), builds the full EGIPGModel on the real PPI graph, and forwards 4 real
perturbations — checking every output key's shape, lambda in [0,1], sigma > 0, and finiteness. Also
forwards the expression-only nested variant (graph_encoder=None, lambda pinned to 0).

    python src/tcell_pipeline/run_module3_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path for direct runs

import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.graph import build_hetero_graph  # noqa: E402
from tcell_pipeline.graph.typed_graph_encoder import TypedGraphEncoder  # noqa: E402
from tcell_pipeline.model import EGIPGModel  # noqa: E402
from tcell_pipeline.programs.program_basis import fit_program_basis, train_row_indices  # noqa: E402

DONOR_COLS = [f"{config.DONOR_PC_PREFIX}{i:02d}" for i in range(config.DONOR_PCA_DIMS)]


def _batch(pc: pd.DataFrame, obs: pd.DataFrame) -> dict:
    return {
        "uniprot_id": [None if pd.isna(x) else str(x) for x in pc["uniprot_id"]],
        "ppi_degree_physical": torch.tensor(pc["ppi_degree_physical"].to_numpy()),
        "ppi_degree_functional": torch.tensor(pc["ppi_degree_functional"].to_numpy()),
        "ppi_degree_complex": torch.tensor(pc["ppi_degree_complex"].to_numpy()),
        "control_baseline_expr": torch.tensor(pc["control_baseline_expr"].to_numpy()),
        "culture_condition": pc["culture_condition"].tolist(),
        "donor_pc": torch.tensor(pc[DONOR_COLS].to_numpy(dtype="float32")),
        "n_guides": torch.tensor(obs["n_guides"].to_numpy()),
        "single_guide_estimate": torch.tensor(obs["single_guide_estimate"].to_numpy(dtype=bool)),
    }


def run() -> bool:
    if not config.PROTEIN_EDGES_PATH.exists() or not config.BLOCKED_SPLIT_PATH.exists():
        print("[module3-smoke] marts/splits absent — run run_module0.py + splits first")
        return False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = pd.read_csv(config.BLOCKED_SPLIT_PATH)
    pc_all = pd.read_parquet(config.PERTURBATION_CONDITION_PATH)
    obs_all = pd.read_parquet(config.DE_OBS_PATH, columns=["n_guides", "single_guide_estimate"])
    gene_names = pd.read_parquet(config.DE_VAR_PATH, columns=["gene_name"])["gene_name"].tolist()

    rows = train_row_indices(split, pc_all[["row_index", "hgnc_symbol"]])
    t0 = time.time()
    Z = sp.load_npz(config.DE_LAYERS_DIR / "zscore.npz").tocsr()[rows].toarray()
    B, A = fit_program_basis(Z, method="svd", K=config.PROGRAM_DIM)
    print(f"[module3-smoke] fold-local SVD basis on {Z.shape[0]} train rows in {time.time()-t0:.1f}s "
          f"-> B {B.shape}, A {A.shape}")

    graph, gene_to_idx = build_hetero_graph()
    graph_enc = TypedGraphEncoder(graph, gene_to_idx)
    model = EGIPGModel(torch.from_numpy(B), graph_encoder=graph_enc).eval().to(device)

    in_graph = pc_all["hgnc_symbol"].isin(gene_to_idx).to_numpy()
    picked = list(pc_all.index[in_graph][:4])
    pc, obs = pc_all.loc[picked].reset_index(drop=True), obs_all.loc[picked]
    targets, conditions = pc["hgnc_symbol"].tolist(), pc["culture_condition"].tolist()

    with torch.no_grad():
        out = model(_batch(pc, obs), targets, conditions)
    K, G, n = config.PROGRAM_DIM, len(gene_names), len(picked)
    shapes_ok = (
        out["delta_z"].shape == (n, K) and out["delta_x"].shape == (n, G)
        and out["sigma"].shape == (n, K) and out["lambda"].shape == (n, 1)
    )
    lam = out["lambda"]
    finite = all(bool(torch.isfinite(out[k]).all()) for k in ("delta_z", "delta_x", "sigma", "lambda"))
    lam_ok = bool((lam >= 0).all() and (lam <= 1).all())
    sigma_ok = bool((out["sigma"] > 0).all())
    print(f"[module3-smoke] targets {targets} under {conditions}")
    print(f"  shapes_ok={shapes_ok}  finite={finite}  lambda in [{lam.min():.2f},{lam.max():.2f}]  sigma>0={sigma_ok}")

    expr_only = EGIPGModel(torch.from_numpy(B), perturbation_encoder=model.perturbation_encoder).eval().to(device)
    with torch.no_grad():
        out2 = expr_only(_batch(pc, obs), targets, conditions)
    expr_ok = bool((out2["lambda"] == 0).all()) and out2["h_graph"] is None and out2["delta_x"].shape == (n, G)
    print(f"  expr-only variant: lambda==0 & no graph = {expr_ok}")

    ok = shapes_ok and finite and lam_ok and sigma_ok and expr_ok
    print(f"\n=== Module 3 real-data smoke {'PASSED' if ok else 'FAILED'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
