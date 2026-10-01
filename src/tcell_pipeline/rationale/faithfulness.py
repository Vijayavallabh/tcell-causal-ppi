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
from tcell_pipeline.rationale.rationale_head import _PP_RELATIONS, complement, edge_index_of


class FaithfulnessTester:
    def __init__(self, graph_encoder, decoder) -> None:
        self.graph_encoder = graph_encoder
        self.decoder = decoder

    @torch.no_grad()
    def delta_z(self, sub, condition, h_do, keep_mask=None) -> torch.Tensor:
        """Program delta under an optional edge keep-mask. Forces the encoder/decoder into eval so the
        'fixed-model' contract holds: ``@torch.no_grad`` suppresses gradients but NOT DropEdge, so
        without eval each re-encode would randomly drop edges and make the deletion scores stochastic.
        The prior train/eval state is restored, so this never mutates the caller's model."""
        enc_train, dec_train = self.graph_encoder.training, self.decoder.training
        self.graph_encoder.eval()
        self.decoder.eval()
        try:
            r = self.graph_encoder.encode_subgraph(sub, condition, h_do, keep_mask=keep_mask)
            return self.decoder(h_do.reshape(1, -1), r["h_graph"].reshape(1, -1))["delta_z"].reshape(-1)
        finally:
            self.graph_encoder.train(enc_train)
            self.decoder.train(dec_train)

    def sufficiency(self, sub, condition, h_do, mask: dict, dz_full: torch.Tensor | None = None) -> float:
        if dz_full is None:  # mask-invariant; pass it in to skip the recompute across matched-random controls
            dz_full = self.delta_z(sub, condition, h_do)
        dz_kept = self.delta_z(sub, condition, h_do, mask)               # keep only the rationale edges
        return float((dz_kept - dz_full).norm())

    def necessity(self, sub, condition, h_do, mask: dict, dz_full: torch.Tensor | None = None) -> float:
        if dz_full is None:
            dz_full = self.delta_z(sub, condition, h_do)
        dz_removed = self.delta_z(sub, condition, h_do, complement(mask))  # keep everything except S
        return float((dz_removed - dz_full).norm())

    def structural_ood_audit(self, sub, mask: dict) -> dict:
        """Graph-structure distortion from deleting S, before vs after. Everything reported here —
        degree, components, hop-distance AND the deleted-fraction ``sparsity`` — is scoped to the
        protein-protein edges, so the sparsity signal is consistent with the connectivity signals
        (membership edges never enter the PP structure graph)."""
        n = int(sub[PROTEIN].x.shape[0])
        total = sum(int(edge_index_of(sub, rel).shape[1]) for rel in _PP_RELATIONS)
        removed = sum(int(mask[rel].sum()) for rel in _PP_RELATIONS if rel in mask) if mask else 0
        removed_frac = (removed / total) if total else 0.0
        return {
            "before": _structure(n, self._pp_edges(sub, None), removed_frac=0.0),
            "after": _structure(n, self._pp_edges(sub, mask), removed_frac=removed_frac),
        }

    @staticmethod
    def _pp_edges(sub, mask) -> list[tuple[int, int]]:
        """Undirected protein-protein edge list, optionally dropping the masked (rationale) edges."""
        edges = []
        for rel in _PP_RELATIONS:
            drop = mask.get(rel) if mask else None
            for i, (u, v) in enumerate(edge_index_of(sub, rel).t().tolist()):
                if drop is None or not bool(drop[i]):
                    edges.append((u, v))
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
