"""ProgramDecoder: [h_graph || h_do] -> program deltas, gene-level deltas, uncertainty (§6.1-6.4).

Two prediction pathways mixed by a learned gate: the graph path sees both representations, the
expression-only path sees h_do alone. lambda in [0,1] weights them, so the model can lean on the
graph for well-connected hubs and fall back to expression for low-degree genes. Gene-level deltas
decode through the FROZEN loading matrix B (registered_buffer, not a Parameter) plus a graph-derived
residual. Passing ``h_graph=None`` gives the expression-only nested variant (lambda pinned to 0).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tcell_pipeline import config

_SIGMA_EPS = 1e-12  # keeps sqrt(softplus(.)) strictly positive when softplus underflows in float32


class ProgramDecoder(nn.Module):
    def __init__(
        self,
        program_basis: torch.Tensor,
        h_graph_dim: int = config.GRAPH_HIDDEN_DIM,
        h_do_dim: int = config.H_DO_DIM,
    ) -> None:
        super().__init__()
        B = torch.as_tensor(program_basis, dtype=torch.float32)
        if B.ndim != 2:
            raise ValueError(f"program_basis must be (G,K), got shape {tuple(B.shape)}")
        # persistent=False: B is fold-local and reloaded from the loadings parquet (from_saved_basis),
        # so it must NOT ride in state_dict where a stale checkpoint could clobber the gene-aligned basis.
        self.register_buffer("program_basis", B, persistent=False)  # frozen: not a Parameter, follows .to(device)
        self.gene_dim, self.program_dim = int(B.shape[0]), int(B.shape[1])
        self.h_graph_dim, self.h_do_dim = h_graph_dim, h_do_dim
        joint = h_graph_dim + h_do_dim

        self.graph_path = nn.Linear(joint, self.program_dim)
        self.expr_path = nn.Linear(h_do_dim, self.program_dim)
        self.gate = nn.Linear(joint, 1)
        self.uncertainty = nn.Linear(joint, self.program_dim)
        self.residual = nn.Linear(h_graph_dim, self.gene_dim)

    def forward(self, h_do: torch.Tensor, h_graph: torch.Tensor | None = None) -> dict:
        n = h_do.shape[0]
        device, dtype = h_do.device, h_do.dtype
        delta_z_expr = self.expr_path(h_do)
        expr_only = h_graph is None

        if expr_only:  # expression-only nested variant: no graph pathway, lambda == 0
            h_graph = torch.zeros(n, self.h_graph_dim, device=device, dtype=dtype)
            lam = torch.zeros(n, 1, device=device, dtype=dtype)
            delta_z = delta_z_expr
        else:
            h_graph = h_graph.to(device)

        joint = torch.cat([h_graph, h_do], dim=1)  # built once, reused by gate / graph / uncertainty
        if not expr_only:
            lam = torch.sigmoid(self.gate(joint))
            delta_z = lam * self.graph_path(joint) + (1.0 - lam) * delta_z_expr

        # eps floor so softplus can't underflow to 0 in float32 -> sigma stays strictly positive
        sigma = torch.sqrt(F.softplus(self.uncertainty(joint)) + _SIGMA_EPS)
        # the residual is a graph-derived per-gene correction; the pure expression-only member must
        # NOT carry residual.bias, else the §10.6 nested comparison is confounded by a graph-head intercept
        delta_x = delta_z @ self.program_basis.T
        if not expr_only:
            delta_x = delta_x + self.residual(h_graph)
        return {"delta_z": delta_z, "delta_x": delta_x, "sigma": sigma, "lambda": lam}
