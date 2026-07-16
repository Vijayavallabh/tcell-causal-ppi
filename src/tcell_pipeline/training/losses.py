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
        self.f_shared = nn.Linear(program_dim, program_dim)  # donor-invariant component extractor
        self.huber_delta, self.focal_gamma, self.de_call_z = huber_delta, focal_gamma, de_call_z
        self.lambda_gene, self.lambda_de = lambda_gene, lambda_de
        self.lambda_inv, self.lambda_graph = lambda_inv, lambda_graph
        self.graph_lambda_sparse, self.graph_lambda_unsrc = graph_lambda_sparse, graph_lambda_unsrc

    def _invariance(self, dz_hat: torch.Tensor, target_genes, conditions) -> torch.Tensor:
        """Donor-invariance penalty (§8.2 item 4): pull the shared program component together across
        rows sharing the (target, condition) key — the same perturbation seen under different donors.
        KNOWN CEILING (verified by review, 2026-07-16): Module 0 aggregates donor PCs to condition-level
        means (control_profiles), so the marts carry NO per-donor rows. The donor-invariance objective is
        therefore *vacuously satisfied* — there is no donor variation left to be invariant to — and this
        term is correctly inert on real data, activating only if per-donor examples are reintroduced. The
        (target, condition) key is the correct donor-invariance grouping; the one artefact is an upstream
        id_mapping quirk where two paralogues share an HGNC symbol (GPR89A/GPR89B -> 'GPHRA'), the only
        groups that fire today (6 of 33,983 rows, negligible at lambda_inv). That collision is a feat-002
        id_mapping concern, not a loss defect — fixing it here would paper over the wrong module.
        ponytail: term inert until the pipeline exposes a per-donor axis; upgrade = re-emit donor-resolved
        rows in Module 0 + group on (target, condition, donor). Kept: spec-required, forward-compatible,
        unit-tested on synthetic donor groups. Zero when every (target, condition) group is a singleton."""
        groups: dict[tuple, list[int]] = {}
        for i, key in enumerate(zip(target_genes, conditions)):
            groups.setdefault(key, []).append(i)
        shared = self.f_shared(dz_hat)
        total = dz_hat.new_zeros(())
        for idx in groups.values():
            if len(idx) > 1:
                g = shared[idx]
                total = total + (g - g.mean(dim=0, keepdim=True)).pow(2).sum()
        return total / dz_hat.shape[0]

    def _graph(self, edge_gates, edge_confidences=None) -> torch.Tensor:
        """Sparsity on the condition gates plus an unsourced-reliance L2 that a per-edge source
        confidence (in [0,1]) down-weights; with no confidence supplied every edge is treated as
        unsupported (conf = 0), so the term is a plain L2 on the gates (§8.2 item 5)."""
        if not edge_gates:
            return torch.zeros(())
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
        return self.graph_lambda_sparse * sparse + self.graph_lambda_unsrc * unsrc

    def forward(self, out: dict, dz_true, dx_true, target_genes, conditions, edge_confidences=None) -> dict:
        response = F.huber_loss(out["delta_z"], dz_true, delta=self.huber_delta)
        gene = F.huber_loss(out["delta_x"], dx_true, delta=self.huber_delta)

        up_logits, down_logits = self.de_head(out["h_do"])
        y_up = (dx_true >= self.de_call_z).to(up_logits.dtype)
        y_down = (dx_true <= -self.de_call_z).to(down_logits.dtype)
        de = _focal_bce(up_logits, y_up, self.focal_gamma) + _focal_bce(down_logits, y_down, self.focal_gamma)

        invariance = self._invariance(out["delta_z"], target_genes, conditions)
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
