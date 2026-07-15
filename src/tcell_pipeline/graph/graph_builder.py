"""GraphBuilder: assemble the typed protein interaction graph as a PyG HeteroData.

Two node types — ``protein`` (keyed by upper-case HGNC symbol) and ``complex`` (protein
complexes from CORUM) — and four relations: physical_ppi / co_complex / functional_assoc
(protein-protein, split by the ``is_*`` evidence flags) and complex_membership (the bipartite
protein->complex table). Protein nodes carry the same frozen 1412-d descriptor Module 1's
TargetEncoder builds (PLM + PINNACLE + 3 PPI degrees + control baseline expression), so the
graph and the perturbation encoder describe a protein the same way. Complex nodes carry only
an index; their learnable vector lives in the encoder's nn.Embedding.

Degrees are computed from the graph itself (edge incidence per relation), so every node has a
degree regardless of whether it was a perturbation target. Everything not covered — a protein
with no UniProt, a gene with no baseline — falls back to zero, matching the encoder's frozen
zero-fallback convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from tcell_pipeline import config
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore

PROTEIN = "protein"
COMPLEX = "complex"
_SOURCE_INDEX = {s: i for i, s in enumerate(config.PPI_SOURCES)}
_CORUM_ONEHOT = np.eye(len(config.PPI_SOURCES), dtype=np.float32)[_SOURCE_INDEX["corum"]]
_RELATION_FLAG = {
    "physical_ppi": "is_physical",
    "co_complex": "is_complex",
    "functional_assoc": "is_functional",
}


def _edge_features(sub: pd.DataFrame) -> torch.Tensor:
    """(E, 8) = source one-hot(5) | score | is_direct_binary | n_supporting_sources."""
    idx = sub["source"].map(_SOURCE_INDEX)
    if idx.isna().any():  # fail fast (clear message) instead of an IndexError in the one-hot assign
        raise ValueError(f"unknown PPI source(s) not in config.PPI_SOURCES: {sorted(set(sub.loc[idx.isna(), 'source']))}")
    onehot = np.zeros((len(sub), len(config.PPI_SOURCES)), dtype=np.float32)
    onehot[np.arange(len(sub)), idx.to_numpy(dtype=int)] = 1.0
    extra = sub[["score", "is_direct_binary", "n_supporting_sources"]].to_numpy(dtype=np.float32)
    return torch.from_numpy(np.nan_to_num(np.concatenate([onehot, extra], axis=1)))


def _protein_features(
    genes: list[str],
    edge_index: dict[str, torch.Tensor],
    n_protein: int,
    id_map: pd.DataFrame,
    baseline: pd.DataFrame,
    plm_store: PluggableEmbeddingStore,
    pinnacle_store: PluggableEmbeddingStore,
) -> torch.Tensor:
    uni = id_map.dropna(subset=["uniprot_id"]).drop_duplicates("hgnc_symbol")
    gene_to_uni = dict(zip(uni["hgnc_symbol"].astype(str), uni["uniprot_id"].astype(str)))
    uniprot_ids = [gene_to_uni.get(g) for g in genes]
    plm = plm_store.lookup(uniprot_ids)
    pinnacle = pinnacle_store.lookup(uniprot_ids)

    degrees = torch.zeros((n_protein, 3), dtype=torch.float32)
    # order MUST match Module 1's TargetEncoder.TARGET_SCALAR_KEYS = [physical, functional, complex]
    for col, rel in enumerate(("physical_ppi", "functional_assoc", "co_complex")):
        ei = edge_index[rel]
        if ei.numel():
            inc = torch.cat([ei[0], ei[1]])  # undirected: both endpoints count
            degrees[:, col] = torch.bincount(inc, minlength=n_protein).to(torch.float32)

    base = baseline.dropna(subset=["control_baseline_expr"]).drop_duplicates("hgnc_symbol")
    gene_to_base = dict(zip(base["hgnc_symbol"].astype(str), base["control_baseline_expr"].astype(float)))
    baseline_vec = torch.tensor([[gene_to_base.get(g, 0.0)] for g in genes], dtype=torch.float32)
    return torch.cat([plm, pinnacle, degrees, torch.nan_to_num(baseline_vec)], dim=1)


def _load_default_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edges = pd.read_parquet(config.PROTEIN_EDGES_PATH)
    complexes = pd.read_parquet(config.COMPLEX_MEMBERSHIP_PATH)
    id_map = pd.read_parquet(config.ID_MAPPING_PATH, columns=["hgnc_symbol", "uniprot_id"])
    baseline = pd.read_parquet(
        config.PERTURBATION_CONDITION_PATH, columns=["hgnc_symbol", "control_baseline_expr"]
    )
    return edges, complexes, id_map, baseline


def build_hetero_graph(
    edges: pd.DataFrame | None = None,
    complexes: pd.DataFrame | None = None,
    id_map: pd.DataFrame | None = None,
    baseline: pd.DataFrame | None = None,
    plm_store: PluggableEmbeddingStore | None = None,
    pinnacle_store: PluggableEmbeddingStore | None = None,
) -> tuple[HeteroData, dict[str, int]]:
    """Build the typed protein graph. Any argument left None is loaded from ``config`` paths.

    Returns the HeteroData plus the ``gene_to_idx`` map (also attached as ``graph.gene_to_idx``
    so the sampler can seed from a gene symbol). Complex ids are attached as ``graph.complex_ids``.
    """
    if edges is None:
        edges, complexes, id_map, baseline = _load_default_frames()
    if plm_store is None:
        plm_store = PluggableEmbeddingStore(config.PLM_EMBEDDINGS_PATH, config.PLM_EMBED_DIM)
    if pinnacle_store is None:
        pinnacle_store = PluggableEmbeddingStore(config.PINNACLE_EMBEDDINGS_PATH, config.PINNACLE_EMBED_DIM)

    genes = sorted(set(edges["source_gene"].dropna()) | set(edges["target_gene"].dropna()))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_protein = len(genes)
    src_idx = edges["source_gene"].map(gene_to_idx).to_numpy()
    dst_idx = edges["target_gene"].map(gene_to_idx).to_numpy()

    data = HeteroData()
    edge_index: dict[str, torch.Tensor] = {}
    for rel, flag in _RELATION_FLAG.items():
        mask = edges[flag].to_numpy() == 1
        ei = torch.tensor(np.stack([src_idx[mask], dst_idx[mask]]), dtype=torch.long)
        edge_index[rel] = ei
        data[PROTEIN, rel, PROTEIN].edge_index = ei
        data[PROTEIN, rel, PROTEIN].edge_attr = _edge_features(edges.loc[mask])

    # bipartite complex membership (drop members that aren't protein nodes -- ponytail: no isolated
    # complex-only proteins, keeps the node set == the PPI gene set)
    complex_ids = sorted(complexes["complex_id"].unique())
    complex_to_idx = {c: i for i, c in enumerate(complex_ids)}
    memb = complexes[complexes["protein_gene"].isin(gene_to_idx)]
    m_src = torch.tensor(memb["protein_gene"].map(gene_to_idx).to_numpy(), dtype=torch.long)
    m_dst = torch.tensor(memb["complex_id"].map(complex_to_idx).to_numpy(), dtype=torch.long)
    data[PROTEIN, "complex_membership", COMPLEX].edge_index = torch.stack([m_src, m_dst])
    m_extra = memb[["confidence", "is_curated"]].to_numpy(dtype=np.float32)
    m_onehot = np.tile(_CORUM_ONEHOT, (len(memb), 1))
    data[PROTEIN, "complex_membership", COMPLEX].edge_attr = torch.from_numpy(
        np.nan_to_num(np.concatenate([m_onehot, m_extra, np.ones((len(memb), 1), dtype=np.float32)], axis=1))
    )

    data[PROTEIN].x = _protein_features(
        genes, edge_index, n_protein, id_map, baseline, plm_store, pinnacle_store
    )
    data[PROTEIN].node_id = torch.arange(n_protein)
    data[COMPLEX].num_nodes = len(complex_ids)
    data[COMPLEX].node_id = torch.arange(len(complex_ids))

    data.gene_to_idx = gene_to_idx
    data.complex_ids = complex_ids
    return data, gene_to_idx
