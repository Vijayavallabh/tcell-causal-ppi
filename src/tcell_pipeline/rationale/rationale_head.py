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
        scored = []  # ranking values are detached; the differentiable importance stays in the dict
        for rel, tensor in importance.items():
            vals = tensor.detach()
            scored.extend((float(vals[i]), rel, i) for i in range(vals.numel()))
        scored.sort(key=lambda t: t[0], reverse=True)
        mask = {rel: torch.zeros(t.numel(), dtype=torch.bool) for rel, t in importance.items()}
        selected = []
        for val, rel, i in scored[: self.top_k]:
            mask[rel][i] = True
            selected.append((rel, i, val))  # (relation, edge index, importance), highest first
        return mask, selected
