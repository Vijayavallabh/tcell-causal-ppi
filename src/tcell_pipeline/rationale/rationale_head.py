"""RationaleHead: sparse PREDICTIVE-rationale extraction over the typed graph (Module 4, Stage B).

Fitted AFTER the H1 predictor is frozen. It scores each evidence edge by how much the frozen model
leans on it and keeps the top-k as the rationale ``S``. This is a *predictive* rationale — which
edges explain the model's prediction — NOT a causal mechanism; the downstream deletion scores are
fixed-model perturbation tests, not interventions (report Module 4). Hence the output is labelled
``predictive_rationale`` and never ``causal``.

Per edge (u, v) with condition gate alpha_bar and edge feature f_e:
    s   = sigmoid(Linear([h_u || h_v || f_e]))    -- learned edge relevance in [0, 1]
    imp = alpha_bar * s                            -- importance in [0, 1] (gate * relevance)
The scorer is zero-initialised, so an untrained head has s == 0.5 everywhere and importance ranks
purely by the frozen gate (already faithful by construction); training moves s to refine that.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.graph.graph_builder import COMPLEX, PROTEIN

RATIONALE_LABEL = "predictive_rationale"  # never "causal" — see module docstring

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_MEMBERSHIP = "complex_membership"
RELATIONS = (*_PP_RELATIONS, _MEMBERSHIP)


def _store_key(rel: str):
    return (PROTEIN, rel, PROTEIN) if rel in _PP_RELATIONS else (PROTEIN, _MEMBERSHIP, COMPLEX)


def edge_index_of(sub, rel: str) -> torch.Tensor:
    return sub[_store_key(rel)].edge_index


def edge_attr_of(sub, rel: str) -> torch.Tensor:
    return sub[_store_key(rel)].edge_attr


def complement(mask: dict) -> dict:
    """Boolean complement of a selection mask (the edges NOT in the rationale)."""
    return {rel: ~m for rel, m in mask.items()}


class RationaleHead(nn.Module):
    def __init__(
        self,
        node_dim: int = config.GRAPH_HIDDEN_DIM,
        edge_dim: int = config.EDGE_FEATURE_DIM,
        top_k: int = config.RATIONALE_TOP_K,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.score = nn.Linear(2 * node_dim + edge_dim, 1)
        nn.init.zeros_(self.score.weight)  # untrained head ranks by the frozen gate (faithful init)
        nn.init.zeros_(self.score.bias)

    def forward(self, edge_gates, node_states, edge_attrs, subgraph) -> dict:
        if edge_gates is None:  # expression-only member (no graph) -> empty rationale
            return {"label": RATIONALE_LABEL, "importance": {}, "selection_mask": {},
                    "selected": [], "subgraph_edges": {}}

        h_p, h_c = node_states[PROTEIN], node_states[COMPLEX]
        importance, edges = {}, {}
        for rel in RELATIONS:
            ei = edge_index_of(subgraph, rel)
            edges[rel] = ei
            gate = edge_gates.get(rel)
            if gate is None or gate.numel() == 0:
                importance[rel] = h_p.new_zeros(0)
                continue
            ea = edge_attrs[rel] if edge_attrs is not None else edge_attr_of(subgraph, rel)
            h_u = h_p[ei[0]]
            h_v = h_p[ei[1]] if rel in _PP_RELATIONS else h_c[ei[1]]
            s = torch.sigmoid(self.score(torch.cat([h_u, h_v, ea], dim=1))).squeeze(-1)  # (E,) in [0,1]
            importance[rel] = gate * s  # imp in [0,1]: gate in [0,1] * s in [0,1]

        selection_mask, selected = self._select(importance)
        return {"label": RATIONALE_LABEL, "importance": importance,
                "selection_mask": selection_mask, "selected": selected, "subgraph_edges": edges}

    def _select(self, importance: dict):
        """Top-k edges by importance across ALL relations -> (selection_mask, S sorted desc)."""
        mask = {rel: torch.zeros(t.numel(), dtype=torch.bool) for rel, t in importance.items()}
        rels = [rel for rel, t in importance.items() if t.numel()]
        if not rels:
            return mask, []
        # torch.topk over the pooled edges; ranking uses detached values, the differentiable
        # importance stays in the dict. rel_id/local map each pooled position back to (relation, edge).
        flat = torch.cat([importance[rel].detach() for rel in rels])
        rel_id = torch.cat([torch.full((importance[rel].numel(),), j, dtype=torch.long) for j, rel in enumerate(rels)])
        local = torch.cat([torch.arange(importance[rel].numel()) for rel in rels])
        vals, idx = torch.topk(flat, min(self.top_k, flat.numel()))  # sorted highest-first
        selected = []
        for v, j, i in zip(vals.tolist(), rel_id[idx].tolist(), local[idx].tolist()):
            rel = rels[j]
            mask[rel][i] = True
            selected.append((rel, i, v))  # (relation, edge index, importance), highest first
        return mask, selected
