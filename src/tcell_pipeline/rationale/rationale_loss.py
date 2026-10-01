"""RationaleLoss: sparsity + sufficiency + necessity + contrastive (Module 4 §Module 4).

    L = lambda_sp  * |S|
      + lambda_suff * ||dz_S     - dz_full||^2          (rationale reproduces the prediction)
      + lambda_nec  * relu(delta_nec - ||dz_\\S - dz_full||)^2   (removing it changes the prediction)
      + lambda_contrast * relu(margin + ||dz_S - dz_full|| - mean ||dz_rand - dz_full||)

The module is a pure function of pre-computed program deltas plus the head's importance. The caller
supplies dz_full, dz_S (rationale kept) and dz_\\S (rationale removed); pass those computed with the
head's continuous importance as soft gate weights (via ``TypedGraphEncoder.encode_subgraph``) and the
whole objective is differentiable back to the head. ``|S|`` is the summed importance mass (a soft L0
surrogate), so the sparsity term also flows gradient to the scorer.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config


class RationaleLoss(nn.Module):
    def __init__(
        self,
        lambda_sparse: float = config.LAMBDA_SPARSE,
        lambda_suff: float = config.LAMBDA_SUFF,
        lambda_nec: float = config.LAMBDA_NEC,
        lambda_contrast: float = config.LAMBDA_CONTRAST,
        delta_nec: float = config.RATIONALE_TAU,
        margin: float = config.RATIONALE_TAU,
    ) -> None:
        super().__init__()
        self.lambda_sparse, self.lambda_suff = lambda_sparse, lambda_suff
        self.lambda_nec, self.lambda_contrast = lambda_nec, lambda_contrast
        self.delta_nec, self.margin = delta_nec, margin

    def forward(self, importance: dict, dz_full, dz_kept, dz_removed, dz_controls=None) -> dict:
        parts = [v.reshape(-1) for v in importance.values()] if importance else []
        sparsity = torch.cat(parts).sum() if parts else dz_full.new_zeros(())

        sufficiency = (dz_kept - dz_full).pow(2).sum()
        nec_dist = (dz_removed - dz_full).norm()
        necessity = torch.clamp(self.delta_nec - nec_dist, min=0.0).pow(2)

        if dz_controls:
            kept_dist = (dz_kept - dz_full).norm()
            rand_dist = torch.stack([(c - dz_full).norm() for c in dz_controls]).mean()
            contrastive = torch.clamp(self.margin + kept_dist - rand_dist, min=0.0)
        else:
            contrastive = dz_full.new_zeros(())

        total = (self.lambda_sparse * sparsity + self.lambda_suff * sufficiency
                 + self.lambda_nec * necessity + self.lambda_contrast * contrastive)
        return {"total": total, "sparsity": sparsity, "sufficiency": sufficiency,
                "necessity": necessity, "contrastive": contrastive}
