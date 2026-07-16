"""FaithfulnessTester: fixed-model deletion tests for a predictive rationale (Module 4 §C).

These are *fixed-model perturbation tests* on the FROZEN encoder+decoder (report line 499), NOT
causal interventions: we re-run the model with the rationale's edges kept or removed (their condition
gate zeroed) and measure how the program delta ``dz`` moves.

    sufficiency(sub, cond, h_do, mask) = || dz(keep only S)   - dz_full ||_2   (small => S suffices)
    necessity  (sub, cond, h_do, mask) = || dz(remove S)      - dz_full ||_2   (large => S is needed)

``structural_ood_audit`` reports how deleting S distorts the graph (degree, components, sparsity, hop
distance) before vs after, so a "faithful" rationale that merely fragments the graph out of
distribution can be caught.
"""
from __future__ import annotations

from collections import deque

import torch

from tcell_pipeline.graph.graph_builder import PROTEIN
from tcell_pipeline.rationale.rationale_head import RELATIONS, complement, edge_index_of

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")


class FaithfulnessTester:
    def __init__(self, graph_encoder, decoder) -> None:
        self.graph_encoder = graph_encoder
        self.decoder = decoder

    @torch.no_grad()
    def _dz(self, sub, condition, h_do, keep_mask=None) -> torch.Tensor:
        r = self.graph_encoder.encode_subgraph(sub, condition, h_do, keep_mask=keep_mask)
        out = self.decoder(h_do.reshape(1, -1), r["h_graph"].reshape(1, -1))
        return out["delta_z"].reshape(-1)

    def sufficiency(self, sub, condition, h_do, mask: dict) -> float:
        dz_full = self._dz(sub, condition, h_do)
        dz_kept = self._dz(sub, condition, h_do, mask)               # keep only the rationale edges
        return float((dz_kept - dz_full).norm())

    def necessity(self, sub, condition, h_do, mask: dict) -> float:
        dz_full = self._dz(sub, condition, h_do)
        dz_removed = self._dz(sub, condition, h_do, complement(mask))  # keep everything except S
        return float((dz_removed - dz_full).norm())

    def structural_ood_audit(self, sub, mask: dict) -> dict:
        """Graph-structure distortion from deleting S, before vs after (protein-protein connectivity)."""
        n = int(sub[PROTEIN].x.shape[0])
        total = sum(int(edge_index_of(sub, rel).shape[1]) for rel in RELATIONS)
        removed_frac = (self._removed_count(mask) / total) if total else 0.0
        return {
            "before": _structure(n, self._pp_edges(sub, None), removed_frac=0.0),
            "after": _structure(n, self._pp_edges(sub, mask), removed_frac=removed_frac),
        }

    @staticmethod
    def _removed_count(mask: dict) -> int:
        return int(sum(int(m.sum()) for m in mask.values())) if mask else 0

    @staticmethod
    def _pp_edges(sub, mask) -> list[tuple[int, int]]:
        """Undirected protein-protein edge list, optionally dropping the masked (rationale) edges."""
        edges = []
        for rel in _PP_RELATIONS:
            ei = edge_index_of(sub, rel)
            drop = mask.get(rel) if mask else None
            for i in range(ei.shape[1]):
                if drop is not None and bool(drop[i]):
                    continue
                edges.append((int(ei[0, i]), int(ei[1, i])))
        return edges


def _structure(n: int, edges: list[tuple[int, int]], removed_frac: float) -> dict:
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return {
        "degree_dist": deg,
        "component_count": _components(n, edges),
        "sparsity": removed_frac,          # fraction of edges deleted
        "hop_distance": _eccentricity(n, edges),
    }


def _components(n: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in edges:
        parent[find(u)] = find(v)
    return len({find(i) for i in range(n)})


def _eccentricity(n: int, edges: list[tuple[int, int]]) -> int:
    """Max hop distance from anchor node 0 over its component (0 if the graph is empty).

    ponytail: the audit signature carries no seed, so node 0 is a stable anchor; before/after use the
    same anchor, so the delta reflects deletion, which is what the OOD check compares."""
    if n == 0:
        return 0
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    dist = {0: 0}
    q = deque([0])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if y not in dist:
                dist[y] = dist[x] + 1
                q.append(y)
    return max(dist.values())
