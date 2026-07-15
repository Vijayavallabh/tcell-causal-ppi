"""NeighborhoodSampler: cut a bounded subgraph around one perturbation target.

Message passing runs on a target's local neighbourhood, not the whole 25k-node graph. We grow
outward ``hops`` steps, taking physical/co-complex neighbours first and then filling by edge
score, capped at ``cap`` protein nodes, then pull in every complex the selected proteins belong
to. The returned HeteroData preserves each node's original index (``orig_idx``) so the encoder
can look up the right complex embedding and hand stable ids to Module 4.
"""
from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from tcell_pipeline import config
from tcell_pipeline.graph.graph_builder import COMPLEX, PROTEIN

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_PRIORITY_BONUS = {"physical_ppi": 1e6, "co_complex": 1e6, "functional_assoc": 0.0}
_SCORE_COL = len(config.PPI_SOURCES)  # edge_attr layout: onehot(5) then score at index 5


def _grow(graph: HeteroData, seed: int, hops: int, cap: int) -> list[int]:
    selected, seen = [seed], {seed}
    frontier = torch.tensor([seed])
    for _ in range(hops):
        nodes, keys = [], []
        for rel in _PP_RELATIONS:
            store = graph[PROTEIN, rel, PROTEIN]
            ei, ea = store.edge_index, store.edge_attr
            if ei.numel() == 0:
                continue
            for a, b in ((0, 1), (1, 0)):  # undirected traversal
                m = torch.isin(ei[a], frontier)
                nodes.append(ei[b][m])
                keys.append(ea[m][:, _SCORE_COL] + _PRIORITY_BONUS[rel])
        if not nodes:
            break
        nodes, keys = torch.cat(nodes), torch.cat(keys)
        new = []
        for i in torch.argsort(keys, descending=True).tolist():  # priority then score
            n = int(nodes[i])
            if n in seen:
                continue
            seen.add(n)
            new.append(n)
            if len(selected) + len(new) >= cap:
                break
        selected.extend(new)
        if len(selected) >= cap or not new:
            break
        frontier = torch.tensor(new)
    return selected[:cap]


def _induce(ei: torch.Tensor, ea: torch.Tensor, src_remap: torch.Tensor, dst_remap: torch.Tensor):
    if ei.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long), ea[:0]
    keep = (src_remap[ei[0]] >= 0) & (dst_remap[ei[1]] >= 0)
    sub_ei = torch.stack([src_remap[ei[0][keep]], dst_remap[ei[1][keep]]])
    return sub_ei, ea[keep]


def sample_subgraph(
    graph: HeteroData,
    target_gene: str,
    hops: int = config.GRAPH_HOPS,
    cap: int = config.NEIGHBORHOOD_CAP,
    gene_to_idx: dict[str, int] | None = None,
) -> HeteroData:
    """Return the induced ≤``cap``-node subgraph around ``target_gene`` (KeyError if unknown)."""
    gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
    seed = gene_to_idx[target_gene]
    selected = _grow(graph, seed, hops, cap)
    sel = torch.tensor(sorted(selected))

    n_protein = graph[PROTEIN].x.shape[0]
    p_remap = torch.full((n_protein,), -1, dtype=torch.long)
    p_remap[sel] = torch.arange(sel.numel())

    memb = graph[PROTEIN, "complex_membership", COMPLEX].edge_index
    sel_complex = (
        torch.unique(memb[1][torch.isin(memb[0], sel)]) if memb.numel() else torch.tensor([], dtype=torch.long)
    )
    n_complex = graph[COMPLEX].num_nodes
    c_remap = torch.full((n_complex,), -1, dtype=torch.long)
    c_remap[sel_complex] = torch.arange(sel_complex.numel())

    sub = HeteroData()
    sub[PROTEIN].x = graph[PROTEIN].x[sel]
    sub[PROTEIN].orig_idx = sel
    sub[COMPLEX].num_nodes = int(sel_complex.numel())
    sub[COMPLEX].orig_idx = sel_complex
    for rel in _PP_RELATIONS:
        store = graph[PROTEIN, rel, PROTEIN]
        ei, ea = _induce(store.edge_index, store.edge_attr, p_remap, p_remap)
        sub[PROTEIN, rel, PROTEIN].edge_index = ei
        sub[PROTEIN, rel, PROTEIN].edge_attr = ea
    m_ei, m_ea = _induce(
        graph[PROTEIN, "complex_membership", COMPLEX].edge_index,
        graph[PROTEIN, "complex_membership", COMPLEX].edge_attr,
        p_remap,
        c_remap,
    )
    sub[PROTEIN, "complex_membership", COMPLEX].edge_index = m_ei
    sub[PROTEIN, "complex_membership", COMPLEX].edge_attr = m_ea
    return sub
