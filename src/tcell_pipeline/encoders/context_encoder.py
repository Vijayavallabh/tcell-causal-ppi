"""ContextEncoder: culture condition + donor context into h_context.

Culture condition (Rest / Stim8hr / Stim48hr) gets a small trainable embedding — there are
only three, and they recur across donors. Donor is deliberately NOT a free per-donor
embedding: that would be ineligible under leave-one-donor-out evaluation (an unseen donor has
no learned vector), so the 32 fixed donor PCA scalars pass through a single Linear projection.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from tcell_pipeline import config

CONDITION_EMBED_DIM = 64
DONOR_PROJ_DIM = 32

_COND_INDEX = {c: i for i, c in enumerate(config.CONDITIONS)}


def _condition_indices(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(torch.long).reshape(-1)
    idx = [_COND_INDEX[v] if isinstance(v, str) else int(v) for v in value]
    return torch.tensor(idx, dtype=torch.long)


def _donor_matrix(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        t = value.to(torch.float32)
    else:
        t = torch.as_tensor(np.asarray(value, dtype=np.float32))
    return t.unsqueeze(0) if t.ndim == 1 else t


class ContextEncoder(nn.Module):
    out_dim = CONDITION_EMBED_DIM + DONOR_PROJ_DIM

    def __init__(self) -> None:
        super().__init__()
        self.condition = nn.Embedding(len(config.CONDITIONS), CONDITION_EMBED_DIM)
        self.donor = nn.Linear(config.DONOR_PCA_DIMS, DONOR_PROJ_DIM)

    def forward(self, batch: dict) -> torch.Tensor:
        device = self.condition.weight.device  # follow the module's device (CPU or GPU)
        cond = self.condition(_condition_indices(batch["culture_condition"]).to(device))
        donor = self.donor(_donor_matrix(batch["donor_pc"]).to(device))
        return torch.cat([cond, donor], dim=1)
