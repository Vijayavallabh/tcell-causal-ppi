"""NeighborhoodSampler: cut a bounded subgraph around one perturbation target.

Message passing runs on a target's local neighbourhood, not the whole 25k-node graph. We grow
outward ``hops`` steps, taking physical/co-complex neighbours first and then filling by edge
score, capped at ``cap`` protein nodes, then pull in every complex the selected proteins belong
to. The returned HeteroData preserves each node's original index (``orig_idx``) so the encoder
can look up the right complex embedding and hand stable ids to Module 4.

Neighbour lookup goes through a CSR index built ONCE per graph (``_NeighborIndex``). Growing and
inducing both need "the edges incident on this node set"; done as a boolean scan over the full
edge_index that costs O(|E|) *per row* -- ~8M edges swept to find a few thousand -- which made
sampling 95% of graph-encode wall-clock on GPU. The index answers the same question in O(sum of
the node set's degree) and returns edge ids in original edge order, so the sampled subgraph is
bit-identical to the scan (see test_sampler_matches_full_scan_reference).
"""
from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from tcell_pipeline import config
from tcell_pipeline.graph.graph_builder import COMPLEX, PROTEIN

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_MEMBERSHIP = "complex_membership"
_PRIORITY_BONUS = {"physical_ppi": 1e6, "co_complex": 1e6, "functional_assoc": 0.0}
_SCORE_COL = len(config.PPI_SOURCES)  # edge_attr layout: onehot(5) then score at index 5


def _build_csr(col: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Group edge ids by the node in ``col`` (one endpoint column of an edge_index).

    Returns ``(indptr, order)`` where ``order[indptr[v]:indptr[v+1]]`` are the ids of the edges whose
    ``col`` endpoint is node v. Stable sort so ids stay ascending within a node.
    """
    order = torch.argsort(col, stable=True)
    counts = torch.bincount(col, minlength=n)
    indptr = torch.cat([col.new_zeros(1), torch.cumsum(counts, dim=0)])
    return indptr, order


def _gather(indptr: torch.Tensor, order: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    """Concatenate the CSR rows of ``nodes`` -> the edge ids incident on that node set."""
    starts = indptr[nodes]
    counts = indptr[nodes + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return order.new_zeros(0)
    # ragged range: for each node, emit its [start, start+count) slice, vectorised
    base = torch.repeat_interleave(starts, counts)
    within = torch.arange(total) - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    return order[base + within]


class _NeighborIndex:
    """Per-relation CSR over the full graph, built once and cached on the graph object.

    Derived purely from each relation's ``edge_index``; the graph is built once and treated as
    immutable, so rebuilding the graph is what invalidates this. Costs ~130 MB on the real graph
    (two int64 orderings of the 6.9M functional_assoc edges dominate).
    """

    def __init__(self, graph: HeteroData) -> None:
        n_protein = graph[PROTEIN].x.shape[0]
        self._csr: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for rel in _PP_RELATIONS:
            ei = graph[PROTEIN, rel, PROTEIN].edge_index
            for key in (0, 1):  # growth traverses PP edges in both directions
                self._csr[(rel, key)] = _build_csr(ei[key], n_protein)
        memb = graph[PROTEIN, _MEMBERSHIP, COMPLEX].edge_index
        self._csr[(_MEMBERSHIP, 0)] = _build_csr(memb[0], n_protein)  # keyed on the protein endpoint

    def incident(self, rel: str, key: int, nodes: torch.Tensor) -> torch.Tensor:
        """Ids of the edges whose ``key`` endpoint lies in ``nodes`` -- exactly the set
        ``torch.isin(edge_index[key], nodes)`` masks, without the |E| scan.

        Grouped by node, NOT in original edge order: sorting here would sort every candidate, while
        callers that need original order only need their surviving subset sorted (a ~5x smaller sort
        for induction). ``nodes`` must be duplicate-free, else an edge whose other endpoint is also
        in ``nodes`` would come back twice.
        """
        indptr, order = self._csr[(rel, key)]
        return _gather(indptr, order, nodes)


def _index_for(graph: HeteroData) -> _NeighborIndex:
    index = getattr(graph, "_neighbor_index", None)
    if index is None:
        index = _NeighborIndex(graph)
        graph._neighbor_index = index
    return index


def _grow(graph: HeteroData, seed: int, hops: int, cap: int, index: _NeighborIndex) -> list[int]:
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
                # ascending ids == original edge order, which is what the score ranking below ties on
                eids = torch.sort(index.incident(rel, a, frontier)).values
                nodes.append(ei[b][eids])
                keys.append(ea[eids][:, _SCORE_COL] + _PRIORITY_BONUS[rel])
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


def _induce(ei: torch.Tensor, ea: torch.Tensor, src_remap: torch.Tensor, dst_remap: torch.Tensor,
            eids: torch.Tensor):
    """Induce ``eids`` (the edges already known to have their source in the selection) onto the
    sub-graph's local node ids, keeping those whose destination is also selected.

    The kept ids are sorted so the sub-graph's edges stay in ORIGINAL edge order -- Module 4 hands
    out (relation, edge position) pairs, so the ordering is part of the contract.
    """
    if eids.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long), ea[:0]
    keep = (src_remap[ei[0][eids]] >= 0) & (dst_remap[ei[1][eids]] >= 0)
    kept = torch.sort(eids[keep]).values
    sub_ei = torch.stack([src_remap[ei[0][kept]], dst_remap[ei[1][kept]]])
    return sub_ei, ea[kept]


def sample_subgraph(
    graph: HeteroData,
    target_gene: str,
    hops: int = config.GRAPH_HOPS,
    cap: int = config.NEIGHBORHOOD_CAP,
    gene_to_idx: dict[str, int] | None = None,
) -> HeteroData:
    """Return the induced ≤``cap``-node subgraph around ``target_gene`` (KeyError if unknown)."""
    gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
    index = _index_for(graph)
    seed = gene_to_idx[target_gene]
    selected = _grow(graph, seed, hops, cap, index)
    sel = torch.tensor(sorted(selected))

    n_protein = graph[PROTEIN].x.shape[0]
    p_remap = torch.full((n_protein,), -1, dtype=torch.long)
    p_remap[sel] = torch.arange(sel.numel())

    memb_store = graph[PROTEIN, _MEMBERSHIP, COMPLEX]
    memb = memb_store.edge_index
    memb_eids = index.incident(_MEMBERSHIP, 0, sel)
    sel_complex = (
        torch.unique(memb[1][memb_eids]) if memb.numel() else torch.tensor([], dtype=torch.long)
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
        ei, ea = _induce(store.edge_index, store.edge_attr, p_remap, p_remap,
                         index.incident(rel, 0, sel))
        sub[PROTEIN, rel, PROTEIN].edge_index = ei
        sub[PROTEIN, rel, PROTEIN].edge_attr = ea
    m_ei, m_ea = _induce(memb_store.edge_index, memb_store.edge_attr, p_remap, c_remap, memb_eids)
    sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_index = m_ei
    sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_attr = m_ea
    return sub
