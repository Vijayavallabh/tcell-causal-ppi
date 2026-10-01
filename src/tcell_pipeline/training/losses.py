"""Stage A / Stage B training losses for the EG-IPG H1 predictor (walkthrough §8.2-8.3).

Stage A (``StageALoss``) trains Module 1 + 2 + 3 jointly:

    L_pred = L_response + lambda_gene * L_gene
           + lambda_DE * L_DE + lambda_inv * L_invariance + lambda_graph * L_graph

with Huber response reconstruction (program + gene level), a focal-BCE DE up/down head, a donor
invariance penalty on the shared program component, and an edge-gate sparsity / unsourced-reliance
regulariser. Stage B (``StageBCalibrationLoss``) is the Gaussian-NLL calibration objective, fitted on
the calibration partition AFTER the H1 predictor is frozen — a loss module only, no training loop
here (see §8.1). The rationale objective is Module 4's ``RationaleLoss``; not reimplemented.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from tcell_pipeline import config


class DEHead(nn.Module):
    """Per-gene up/down differential-expression head over h_do -> (up_logits, down_logits), each (B, G)."""

    def __init__(self, gene_dim: int, h_do_dim: int = config.H_DO_DIM) -> None:
        super().__init__()
        self.gene_dim = gene_dim
        self.head = nn.Linear(h_do_dim, 2 * gene_dim)

    def forward(self, h_do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        up, down = self.head(h_do).split(self.gene_dim, dim=1)
        return up, down

    def probs(self, h_do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        up, down = self.forward(h_do)
        return torch.sigmoid(up), torch.sigmoid(down)


def _focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    """Focal binary cross-entropy: BCE modulated by (1 - p_t)^gamma so abundant easy calls (the
    overwhelming majority of non-DE genes) stop dominating the gradient (§8.2 item 3)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    return ((1.0 - p_t).pow(gamma) * bce).mean()


class StageALoss(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        program_dim: int,
        h_do_dim: int = config.H_DO_DIM,
        huber_delta: float = config.HUBER_DELTA,
        focal_gamma: float = config.FOCAL_GAMMA,
        de_call_z: float = config.DE_CALL_ZSCORE,
        lambda_gene: float = config.LAMBDA_GENE,
        lambda_de: float = config.LAMBDA_DE,
        lambda_inv: float = config.LAMBDA_INV,
        lambda_graph: float = config.LAMBDA_GRAPH,
        graph_lambda_sparse: float = 1.0,
        graph_lambda_unsrc: float = 1.0,
    ) -> None:
        super().__init__()
        self.de_head = DEHead(gene_dim, h_do_dim)
        self.program_dim = program_dim
        self.huber_delta, self.focal_gamma, self.de_call_z = huber_delta, focal_gamma, de_call_z
        self.lambda_gene, self.lambda_de = lambda_gene, lambda_de
        self.lambda_inv, self.lambda_graph = lambda_inv, lambda_graph
        self.graph_lambda_sparse, self.graph_lambda_unsrc = graph_lambda_sparse, graph_lambda_unsrc

    def _invariance(self, dz_variants) -> torch.Tensor:
        """Donor-invariance penalty (§8.2 item 4): the model's program prediction Δz must not depend on
        WHICH donor's control profile conditions the encoder. ``dz_variants`` are Δz predicted under
        distinct REAL per-donor PC vectors (control_donor_profiles — the mart's donor_pc is only their
        mean) for the same (target, condition); we penalise the per-example VARIANCE of Δz across those
        donors DIRECTLY.

        Penalising Δz itself, not a learnable projection f_shared(Δz), is deliberate: a free f_shared is
        trivially minimised by collapsing its weights to 0 (weight decay + the variance objective both
        drive W→0), so the term would decay back to inert without ever pressuring the encoder — the exact
        degeneracy an earlier version had. Raw-Δz variance has no such trivial solution: driving it down
        forces the encoder to emit donor-invariant predictions, consistent with the donor-averaged
        response target. (A shared/nuisance split Δz = Δz_shared + Δz_nuisance(d) that preserves
        legitimate donor-specific signal would need a paired nuisance head — deferred; for a single-output
        predictor evaluated on donor-averaged data, full Δz invariance is the right target.) <2 draws → 0."""
        if not dz_variants or len(dz_variants) < 2:
            return dz_variants[0].new_zeros(()) if dz_variants else torch.zeros(())
        reps = torch.stack(dz_variants)                               # (S, B, K) — Δz directly
        return (reps - reps.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean()

    def _graph(self, edge_gates, edge_confidences=None) -> torch.Tensor:
        """Per-sample edge-gate sparsity + an unsourced-reliance L2 that a per-edge source confidence
        (in [0,1]) down-weights; with no confidence supplied every edge is treated as unsupported
        (conf = 0), so the term is a plain L2 on the gates (§8.2 item 5). Averaged over the batch so its
        strength doesn't scale with batch size (the other loss terms are all mean-reduced)."""
        if not edge_gates:
            return torch.zeros(())
        n = max(len(next(iter(edge_gates.values()))), 1)  # batch size (one gate tensor per sample)
        sparse = torch.zeros(())
        unsrc = torch.zeros(())
        for rel, per_sample in edge_gates.items():
            confs = edge_confidences.get(rel) if edge_confidences else None
            for b, alpha in enumerate(per_sample):
                if alpha.numel() == 0:
                    continue
                sparse = sparse.to(alpha) + alpha.abs().sum()
                conf = confs[b].to(alpha) if confs is not None else torch.zeros_like(alpha)
                unsrc = unsrc.to(alpha) + ((1.0 - conf) * alpha.pow(2)).sum()
        return (self.graph_lambda_sparse * sparse + self.graph_lambda_unsrc * unsrc) / n

    def forward(self, out: dict, dz_true, dx_true, dz_variants=None, edge_confidences=None) -> dict:
        response = F.huber_loss(out["delta_z"], dz_true, delta=self.huber_delta)
        gene = F.huber_loss(out["delta_x"], dx_true, delta=self.huber_delta)

        up_logits, down_logits = self.de_head(out["h_do"])
        y_up = (dx_true >= self.de_call_z).to(up_logits.dtype)
        y_down = (dx_true <= -self.de_call_z).to(down_logits.dtype)
        de = _focal_bce(up_logits, y_up, self.focal_gamma) + _focal_bce(down_logits, y_down, self.focal_gamma)

        invariance = self._invariance(dz_variants).to(response)
        graph = self._graph(out.get("edge_gates"), edge_confidences).to(response)

        total = (response + self.lambda_gene * gene + self.lambda_de * de
                 + self.lambda_inv * invariance + self.lambda_graph * graph)
        return {"total": total, "response": response, "gene": gene,
                "de": de, "invariance": invariance, "graph": graph}


class StageBCalibrationLoss(nn.Module):
    """Gaussian negative log-likelihood over the frozen H1 program deltas (§8.3 item 6). Fitted on the
    calibration partition after the predictor freeze — a loss module only, never a training loop."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, dz_hat, dz_true, sigma) -> torch.Tensor:
        return F.gaussian_nll_loss(dz_hat, dz_true, sigma.pow(2), full=False, eps=self.eps, reduction="mean")
