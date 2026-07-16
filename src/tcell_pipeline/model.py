"""EGIPGModel: end-to-end Module 1 + Module 2 + Module 3 (§10.6 nested family top member).

PerturbationEncoder -> h_do, TypedGraphEncoder -> h_graph, ProgramDecoder -> program/gene deltas.
``graph_encoder=None`` yields the expression-only ablation from the nested confirmatory family: the
decoder runs its expression-only pathway (lambda == 0, no edge gates). The loading matrix B is frozen
inside the decoder as a buffer, so the whole model is a single ``.to(device)`` away from CPU or CUDA.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.encoders import PerturbationEncoder
from tcell_pipeline.graph import TypedGraphEncoder
from tcell_pipeline.programs.program_basis import load_program_basis
from tcell_pipeline.programs.program_decoder import ProgramDecoder


class EGIPGModel(nn.Module):
    def __init__(
        self,
        program_basis: torch.Tensor,
        perturbation_encoder: PerturbationEncoder | None = None,
        graph_encoder: TypedGraphEncoder | None = None,
        h_graph_dim: int = config.GRAPH_HIDDEN_DIM,
        h_do_dim: int = config.H_DO_DIM,
    ) -> None:
        super().__init__()
        self.perturbation_encoder = perturbation_encoder or PerturbationEncoder()
        self.graph_encoder = graph_encoder  # None == expression-only nested variant
        # dims are explicit + overridable so a reduced-width encoder ablation sizes the decoder to the
        # wrapped encoders' real output, not a hardcoded config assumption
        self.decoder = ProgramDecoder(program_basis, h_graph_dim=h_graph_dim, h_do_dim=h_do_dim)

    @classmethod
    def from_saved_basis(cls, gene_order, path=None, **kw) -> "EGIPGModel":
        """Build with B loaded from the fold-local loadings parquet, aligned to ``gene_order``."""
        B, _ = load_program_basis(path or config.PROGRAM_LOADINGS_PATH, gene_order=gene_order)
        return cls(torch.from_numpy(B), **kw)

    def forward(self, batch: dict, target_genes: list[str], conditions: list[str]) -> dict:
        h_do = self.perturbation_encoder(batch)
        if self.graph_encoder is not None:
            h_graph, edge_gates, edge_confidences = self.graph_encoder(target_genes, conditions, h_do)
        else:
            h_graph, edge_gates, edge_confidences = None, None, None
        out = self.decoder(h_do, h_graph)
        out["h_do"] = h_do
        out["h_graph"] = h_graph
        out["edge_gates"] = edge_gates
        out["edge_confidences"] = edge_confidences  # per-edge source confidence for L_graph's unsourced term
        return out
