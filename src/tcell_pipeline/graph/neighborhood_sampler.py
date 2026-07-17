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
the node set's degree), so the sampled subgraph is bit-identical to the scan (pinned by
test_sampler_matches_full_scan_reference).

``incident()`` returns edge ids GROUPED BY NODE, not in original edge order -- each caller sorts
the subset it actually keeps. Both call sites then sort, and those sorts are LOAD-BEARING, not
leftovers: original edge order is what the growth ranking ties on and what Module 4's (relation,
edge position) pairs address. _PRIORITY_BONUS adds 1e6 to a float32 score, whose spacing at 1e6 is
0.0625, so scores quantise into ties in bulk and the tie-break order decides which neighbours
survive the cap. Delete either sort and the sampler silently returns a different neighbourhood.
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
    span = torch.arange(total, device=order.device)  # follow the graph's device, not the default one
    within = span - torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
    return order[base + within]


def _edge_stores(graph: HeteroData):
    """Every (relation, key-column) the index is derived from, in build order."""
    for rel in _PP_RELATIONS:
        for key in (0, 1):  # growth traverses PP edges in both directions
            yield rel, key, graph[PROTEIN, rel, PROTEIN].edge_index
    yield _MEMBERSHIP, 0, graph[PROTEIN, _MEMBERSHIP, COMPLEX].edge_index  # keyed on the protein endpoint


def _fingerprint(graph: HeteroData) -> tuple:
    """Cheap identity of the edge_index tensors the index was built from: storage address, shape, and
    autograd's in-place version counter. Catches a relation being reassigned, resized, or edited in
    place -- the ways a graph actually changes."""
    return tuple((ei.data_ptr(), tuple(ei.shape), ei._version) for _, _, ei in _edge_stores(graph))


class _NeighborIndex:
    """Per-relation CSR over the full graph, built once and cached on the graph object.

    Derived purely from each relation's ``edge_index``. Costs ~130 MB on the real graph (two int64
    orderings of the 6.9M functional_assoc edges dominate), so it is cached rather than rebuilt --
    but the cache is fingerprinted, because the full scan it replaced re-read ``edge_index`` on every
    call and so an edited graph took effect immediately. Silently sampling a stale topology would
    hand an edge-ablation or rewired-network control the neighbourhood it thought it had removed.
    """

    def __init__(self, graph: HeteroData) -> None:
        n_protein = graph[PROTEIN].x.shape[0]
        self._csr: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {
            (rel, key): _build_csr(ei[key], n_protein) for rel, key, ei in _edge_stores(graph)
        }
        self.fingerprint = _fingerprint(graph)

    def incident(self, rel: str, key: int, nodes: torch.Tensor) -> torch.Tensor:
        """Ids of the edges whose ``key`` endpoint lies in ``nodes`` -- exactly the set
        ``torch.isin(edge_index[key], nodes)`` masks, without the |E| scan.

        Grouped by node, NOT in original edge order: sorting here would sort every candidate, while
        callers that need original order only need their surviving subset sorted (a ~5x smaller sort
        for induction). ``nodes`` must be duplicate-free -- a repeated node emits its whole CSR row
        again, so every one of its edges would come back once per repeat.
        """
        indptr, order = self._csr[(rel, key)]
        return _gather(indptr, order, nodes)


def _index_for(graph: HeteroData) -> _NeighborIndex:
    """The graph's cached CSR index, rebuilt if its edges changed since the index was built."""
    index = getattr(graph, "_neighbor_index", None)
    if index is None or index.fingerprint != _fingerprint(graph):
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
                # LOAD-BEARING sort, not a leftover: ascending ids == original edge order, and the
                # ranking below ties in bulk (see the module docstring), so this decides selection.
                eids = torch.sort(index.incident(rel, a, frontier)).values
                nodes.append(ei[b][eids])
                keys.append(ea[eids, _SCORE_COL] + _PRIORITY_BONUS[rel])  # one column, not all of ea
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
    """Induce ``eids`` onto the sub-graph's local node ids, keeping those whose destination is also
    selected. ``eids`` MUST come from ``incident(rel, 0, <the selection>)``, so every edge's source is
    already in the selection by construction and only the destination needs testing.

    The kept ids are sorted so the sub-graph's edges stay in ORIGINAL edge order -- Module 4 hands
    out (relation, edge position) pairs, so the ordering is part of the contract.
    """
    if eids.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long), ea[:0]
    kept = torch.sort(eids[dst_remap[ei[1][eids]] >= 0]).values
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
