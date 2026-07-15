"""Module 2 real-data smoke — build the typed graph, sample a real neighbourhood, encode it.

The Module 2 analogue of ``run_module1_smoke.py``: builds the full HeteroData from the real PPI
marts, samples the CD3E neighbourhood (a canonical CD4+ T-cell receptor-complex gene), runs the
real Module 1 PerturbationEncoder to produce h_do for a handful of real perturbations, then pushes
them through the TypedGraphEncoder — checking h_graph is finite, edge gates are returned per
relation, readout attention sums to 1, and the SAME edge is gated differently across conditions.

    python src/tcell_pipeline/graph/run_module2_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # put src/ on path for direct runs

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from tcell_pipeline import config  # noqa: E402
from tcell_pipeline.encoders import PerturbationEncoder, build_encoder_batch  # noqa: E402
from tcell_pipeline.graph import COMPLEX, PROTEIN, TypedGraphEncoder, build_hetero_graph, sample_subgraph  # noqa: E402

DE_OBS_PATH = config.INTERMEDIATE_ROOT / "de_obs.parquet"


def _module1_h_do(pc: pd.DataFrame, obs: pd.DataFrame, device: str) -> torch.Tensor:
    with torch.no_grad():
        return PerturbationEncoder().eval().to(device)(build_encoder_batch(pc, obs))


def run() -> bool:
    if not config.PROTEIN_EDGES_PATH.exists() or not config.PERTURBATION_CONDITION_PATH.exists():
        print("[module2-smoke] marts absent — run run_module0.py first")
        return False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    graph, gene_to_idx = build_hetero_graph()
    print(f"Built graph in {time.time() - t0:.1f}s on real data:")
    print(f"  protein nodes: {graph[PROTEIN].x.shape[0]}   complex nodes: {graph[COMPLEX].num_nodes}")
    for et in graph.edge_types:
        print(f"  edge {et}: {graph[et].edge_index.shape[1]} edges, attr {tuple(graph[et].edge_attr.shape)}")

    seed_gene = "CD3E" if "CD3E" in gene_to_idx else next(iter(gene_to_idx))
    sub = sample_subgraph(graph, seed_gene)
    print(f"\n{seed_gene} neighbourhood: {sub[PROTEIN].x.shape[0]} proteins (cap {config.NEIGHBORHOOD_CAP}), "
          f"{sub[COMPLEX].num_nodes} complexes")
    for rel in ("physical_ppi", "co_complex", "functional_assoc", "complex_membership"):
        et = (PROTEIN, rel, COMPLEX) if rel == "complex_membership" else (PROTEIN, rel, PROTEIN)
        print(f"    {rel}: {sub[et].edge_index.shape[1]} edges")

    # real Module 1 h_do for the first few real perturbations, filtered to genes in the graph
    pc_all = pd.read_parquet(config.PERTURBATION_CONDITION_PATH)
    obs_all = pd.read_parquet(DE_OBS_PATH, columns=["n_guides", "single_guide_estimate"])
    in_graph = pc_all["hgnc_symbol"].isin(gene_to_idx).to_numpy()
    rows = list(pc_all.index[in_graph][:4])
    pc, obs = pc_all.loc[rows].reset_index(drop=True), obs_all.loc[rows]
    h_do = _module1_h_do(pc, obs, device)
    targets, conditions = pc["hgnc_symbol"].tolist(), pc["culture_condition"].tolist()
    print(f"\nModule 1 -> h_do {tuple(h_do.shape)} for targets {targets} under {conditions}")

    enc = TypedGraphEncoder(graph, gene_to_idx).eval().to(device)
    with torch.no_grad():
        h_graph, edge_gates = enc(targets, conditions, h_do)
    finite = bool(torch.isfinite(h_graph).all())
    print(f"\nh_graph {tuple(h_graph.shape)} on {device}: finite={finite}")
    for rel, gs in edge_gates.items():
        counts = [int(g.numel()) for g in gs]
        rng = [f"{g.min():.2f}-{g.max():.2f}" for g in gs if g.numel()]
        print(f"  edge_gates[{rel}]: per-sample edge counts {counts}, ranges {rng[:2]}")

    # readout attention sums to 1, and the same target is gated differently across conditions
    _, _, attn = enc.encode_one(targets[0], "Rest", h_do[0])
    attn_ok = torch.allclose(attn.sum(), torch.tensor(1.0, device=device), atol=1e-4)
    g_rest = enc.encode_one(targets[0], "Rest", h_do[0])[1]
    g_stim = enc.encode_one(targets[0], "Stim48hr", h_do[0])[1]
    rel = next((r for r in g_rest if g_rest[r].numel()), None)
    cond_differ = rel is not None and not torch.allclose(g_rest[rel], g_stim[rel])
    print(f"\nReadout attention sums to 1: {attn_ok}")
    print(f"Condition changes the gate on {rel!r} edges: {cond_differ}")

    ok = finite and attn_ok and cond_differ
    print(f"\n=== Module 2 real-data smoke {'PASSED' if ok else 'FAILED'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
