"""QualityEncoder: perturbation quality features into h_quality.

n_guides and single_guide_estimate are q_pre guide-quality scalars from the DE tables.
guide_seq_embed is a zero placeholder (dim GUIDE_SEQ_EMBED_DIM) reserved for a future
guide-sequence embedding — the guide sequences are not present in the DE tables yet, so it
stays zeros, keeping h_quality's geometry fixed so the encoder is drop-in ready later.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.encoders._tensor import as_float_vector

QUALITY_SCALAR_KEYS = ("n_guides", "single_guide_estimate")


class QualityEncoder(nn.Module):
    out_dim = len(QUALITY_SCALAR_KEYS) + config.GUIDE_SEQ_EMBED_DIM

    def forward(self, batch: dict) -> torch.Tensor:
        scalars = torch.stack([as_float_vector(batch[k]) for k in QUALITY_SCALAR_KEYS], dim=1)
        guide_seq = torch.zeros((scalars.shape[0], config.GUIDE_SEQ_EMBED_DIM), dtype=torch.float32)
        return torch.cat([scalars, guide_seq], dim=1)
