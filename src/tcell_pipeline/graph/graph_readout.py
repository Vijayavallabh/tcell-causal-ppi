"""GraphReadout: pool a subgraph's node states into one h_graph via cross-attention.

The perturbation embedding h_do is the query; the message-passed node states are keys/values.
Multi-head attention lets the readout weight neighbours by relevance to *this* perturbation
rather than mean-pooling everything equally. Attention weights over nodes sum to 1.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config


class GraphReadout(nn.Module):
    def __init__(self, dim: int = config.GRAPH_HIDDEN_DIM, n_heads: int = config.GRAPH_N_HEADS) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, h_do: torch.Tensor, node_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h_do (B, dim) query over node_states (N, dim) -> (h_graph (B, dim), weights (B, N))."""
        q = h_do.unsqueeze(1)                      # (B, 1, dim)
        kv = node_states.unsqueeze(0).expand(q.shape[0], -1, -1)  # (B, N, dim)
        out, weights = self.attn(q, kv, kv, need_weights=True)
        return out.squeeze(1), weights.squeeze(1)  # (B, dim), (B, N)
