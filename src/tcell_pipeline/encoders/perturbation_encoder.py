"""PerturbationEncoder: fuse target + context + quality into the h_do embedding in R^256.

Enforces the leakage fence at the module boundary: any q_post (response-derived) column in
the incoming batch raises immediately, so a prohibited feature can never reach the model as an
input. Fusion is a single Linear over the concatenated sub-encoder outputs followed by
LayerNorm — the only place the frozen and trainable parts are learnably mixed.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.encoders.context_encoder import ContextEncoder
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.encoders.quality_encoder import QualityEncoder
from tcell_pipeline.encoders.target_encoder import TargetEncoder


class PerturbationEncoder(nn.Module):
    def __init__(
        self,
        plm_store: PluggableEmbeddingStore | None = None,
        pinnacle_store: PluggableEmbeddingStore | None = None,
    ) -> None:
        super().__init__()
        self.target = TargetEncoder(plm_store, pinnacle_store)
        self.context = ContextEncoder()
        self.quality = QualityEncoder()
        fusion_in = self.target.out_dim + self.context.out_dim + self.quality.out_dim
        self.fusion = nn.Linear(fusion_in, config.H_DO_DIM)
        self.norm = nn.LayerNorm(config.H_DO_DIM)

    def forward(self, batch: dict) -> torch.Tensor:
        forbidden = set(batch) & set(config.Q_POST_COLS)
        if forbidden:
            raise ValueError(f"q_post columns are prohibited as encoder input: {sorted(forbidden)}")
        h = torch.cat([self.target(batch), self.context(batch), self.quality(batch)], dim=1)
        return self.norm(self.fusion(h))
