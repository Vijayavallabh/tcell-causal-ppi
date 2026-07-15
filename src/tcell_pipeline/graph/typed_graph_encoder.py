"""TypedGraphEncoder: 3-layer relational GNN with condition-gated signed message passing.

For each perturbation target we sample its neighbourhood, then run message passing where every
message is *signed* and *condition-gated*:

  signed message   m = tanh(W_sign h_u) * relu(W_mag h_u)   -- sign = activation/inhibition,
                                                                magnitude = strength
  condition gate   alpha = sigmoid(w_gate[h_cond || f_e])    -- the SAME edge is weighted
                                                                differently under Rest vs Stim
  aggregate        h_v = LayerNorm(h_v + FFN(sum_r sum_u alpha * (W_r m + U_r f_e)))

The gate depends only on the culture condition and the edge features (not on h_u), so it is
computed once and reused across all 3 layers, and returned as ``edge_gates`` for Module 4's
mechanism attribution. Readout cross-attends h_do over the final node states -> h_graph in R^256.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_edge

from tcell_pipeline import config
from tcell_pipeline.graph.graph_builder import COMPLEX, PROTEIN, build_hetero_graph
from tcell_pipeline.graph.graph_readout import GraphReadout
from tcell_pipeline.graph.neighborhood_sampler import sample_subgraph

_PP_RELATIONS = ("physical_ppi", "co_complex", "functional_assoc")
_MEMBERSHIP = "complex_membership"
_COND_INDEX = {c: i for i, c in enumerate(config.CONDITIONS)}


def signed_message(h: torch.Tensor, w_sign: nn.Linear, w_mag: nn.Linear) -> torch.Tensor:
    """m = tanh(W_sign h) * relu(W_mag h): tanh carries the sign, relu the (non-negative) magnitude."""
    return torch.tanh(w_sign(h)) * torch.relu(w_mag(h))


class _RelMessage(MessagePassing):
    """One relation's message + aggregation for one layer (custom message on PyG MessagePassing)."""

    def __init__(self, hidden: int, edge_dim: int, edge_dropout: float) -> None:
        super().__init__(aggr="add")
        self.w_r = nn.Linear(hidden, hidden)
        self.u_r = nn.Linear(edge_dim, hidden)
        self.w_sign = nn.Linear(hidden, hidden)
        self.w_mag = nn.Linear(hidden, hidden)
        self.p = edge_dropout

    def forward(self, x_src, x_dst, edge_index, edge_attr, alpha):
        ei, mask = dropout_edge(edge_index, p=self.p, training=self.training)
        return self.propagate(
            ei, x=(x_src, x_dst), edge_attr=edge_attr[mask], alpha=alpha[mask],
            size=(x_src.size(0), x_dst.size(0)),
        )

    def message(self, x_j, edge_attr, alpha):
        m = self.w_r(signed_message(x_j, self.w_sign, self.w_mag)) + self.u_r(edge_attr)
        return alpha * m


class _FFN(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        return self.norm(h + self.net(agg))


class _GraphLayer(nn.Module):
    """One relational layer: message over 4 relations, residual FFN+LayerNorm per node type."""

    def __init__(self, hidden: int, edge_dim: int, edge_dropout: float) -> None:
        super().__init__()
        self.rel = nn.ModuleDict(
            {r: _RelMessage(hidden, edge_dim, edge_dropout) for r in (*_PP_RELATIONS, _MEMBERSHIP)}
        )
        self.ffn_protein = _FFN(hidden)
        self.ffn_complex = _FFN(hidden)

    def forward(self, h_p, h_c, edges):
        agg_p = torch.zeros_like(h_p)
        agg_c = torch.zeros_like(h_c)
        for rel in _PP_RELATIONS:
            ei, ea, al = edges[rel]
            if ei.numel():
                agg_p = agg_p + self.rel[rel](h_p, h_p, ei, ea, al)
        ei, ea, al = edges[_MEMBERSHIP]
        if ei.numel():
            agg_c = agg_c + self.rel[_MEMBERSHIP](h_p, h_c, ei, ea, al)          # protein -> complex
            agg_p = agg_p + self.rel[_MEMBERSHIP](h_c, h_p, ei.flip(0), ea, al)  # complex -> protein
        return self.ffn_protein(h_p, agg_p), self.ffn_complex(h_c, agg_c)


class TypedGraphEncoder(nn.Module):
    def __init__(self, graph=None, gene_to_idx: dict[str, int] | None = None) -> None:
        super().__init__()
        if graph is None:
            graph, gene_to_idx = build_hetero_graph()
        self.graph = graph
        self.gene_to_idx = gene_to_idx if gene_to_idx is not None else graph.gene_to_idx
        hidden, edge_dim = config.GRAPH_HIDDEN_DIM, config.EDGE_FEATURE_DIM

        self.proj = nn.Linear(config.PROTEIN_FEATURE_DIM, hidden)
        self.complex_embed = nn.Embedding(max(int(graph[COMPLEX].num_nodes), 1), config.COMPLEX_EMBED_DIM)
        self.condition = nn.Embedding(len(config.CONDITIONS), config.CONDITION_EMBED_DIM)
        # one condition gate per relation (alpha is layer-independent -> computed once, reused)
        self.gate = nn.ModuleDict(
            {r: nn.Linear(config.CONDITION_EMBED_DIM + edge_dim, 1) for r in (*_PP_RELATIONS, _MEMBERSHIP)}
        )
        self.layers = nn.ModuleList(
            [_GraphLayer(hidden, edge_dim, config.EDGE_DROPOUT) for _ in range(config.GRAPH_LAYERS)]
        )
        self.readout = GraphReadout(hidden, config.GRAPH_N_HEADS)

    def _edges_with_gates(self, sub, h_cond, device):
        """Symmetrise protein-protein edges, compute the per-edge condition gate once per relation."""
        edges, gates = {}, {}
        for rel in _PP_RELATIONS:
            ei = sub[PROTEIN, rel, PROTEIN].edge_index
            ea = sub[PROTEIN, rel, PROTEIN].edge_attr
            ei = torch.cat([ei, ei.flip(0)], dim=1)   # undirected message passing
            ea = torch.cat([ea, ea], dim=0)
            alpha = self._gate(rel, ea, h_cond)
            edges[rel] = (ei, ea, alpha)
            gates[rel] = alpha.squeeze(-1)
        ei = sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_index
        ea = sub[PROTEIN, _MEMBERSHIP, COMPLEX].edge_attr
        alpha = self._gate(_MEMBERSHIP, ea, h_cond)
        edges[_MEMBERSHIP] = (ei, ea, alpha)
        gates[_MEMBERSHIP] = alpha.squeeze(-1)
        return edges, gates

    def _gate(self, rel: str, edge_attr: torch.Tensor, h_cond: torch.Tensor) -> torch.Tensor:
        if edge_attr.numel() == 0:
            return edge_attr.new_zeros((0, 1))
        cond = h_cond.expand(edge_attr.size(0), -1)
        return torch.sigmoid(self.gate[rel](torch.cat([cond, edge_attr], dim=1)))

    def encode_one(self, target_gene: str, condition, h_do: torch.Tensor):
        """Encode a single (target, condition) -> (h_graph (dim,), edge_gates dict, attn (N,))."""
        device = self.proj.weight.device
        sub = sample_subgraph(self.graph, target_gene, gene_to_idx=self.gene_to_idx).to(device)
        cond_idx = _COND_INDEX[condition] if isinstance(condition, str) else int(condition)
        h_cond = self.condition(torch.tensor([cond_idx], device=device))  # (1, 64)

        edges, gates = self._edges_with_gates(sub, h_cond, device)
        h_p = self.proj(sub[PROTEIN].x)
        c_idx = sub[COMPLEX].orig_idx if sub[COMPLEX].num_nodes else torch.zeros(0, dtype=torch.long, device=device)
        h_c = self.complex_embed(c_idx)
        for layer in self.layers:
            h_p, h_c = layer(h_p, h_c, edges)

        node_states = torch.cat([h_p, h_c], dim=0)
        h_graph, weights = self.readout(h_do.unsqueeze(0), node_states)
        return h_graph.squeeze(0), gates, weights.squeeze(0)

    def forward(self, target_genes: list[str], conditions: list[str], h_do: torch.Tensor):
        """(B targets, B conditions, h_do (B, dim)) -> (h_graph (B, dim), edge_gates dict-of-lists).

        Each target has its own subgraph, so ``edge_gates[relation]`` is a list over the batch
        (one gate tensor per sample) — the per-sample structure Module 4 needs. Unknown target
        genes fall back to a zero h_graph (never crash a batch).  ponytail: per-sample loop; swap
        for PyG mini-batching if graph-encode throughput becomes the bottleneck in Module 3.
        """
        device = self.proj.weight.device
        h_do = h_do.to(device)
        h_graphs, edge_gates = [], {r: [] for r in (*_PP_RELATIONS, _MEMBERSHIP)}
        for b, (gene, cond) in enumerate(zip(target_genes, conditions)):
            if gene not in self.gene_to_idx:
                h_graphs.append(torch.zeros(config.GRAPH_HIDDEN_DIM, device=device))
                for r in edge_gates:
                    edge_gates[r].append(torch.zeros(0, device=device))
                continue
            h_graph, gates, _ = self.encode_one(gene, cond, h_do[b])
            h_graphs.append(h_graph)
            for r, g in gates.items():
                edge_gates[r].append(g)
        return torch.stack(h_graphs), edge_gates
