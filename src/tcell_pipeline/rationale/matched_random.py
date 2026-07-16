"""MatchedRandomSampler: matched-random edge rationales as negative controls (Module 4 §D).

A rationale must beat a *matched* random baseline, not a naive one. Each control keeps the SAME
number of edges as the rationale in EACH relation, so it matches on size and relation-type
composition (and therefore total sparsity). The controls feed the contrastive loss term and the
faithfulness sufficiency<random / necessity>random comparison.

ponytail: matches per-relation edge count (-> size + relation composition + sparsity). Matching the
report's fuller criteria (endpoint degree, connectivity, target-hop distance) is a refinement for the
final rationale-quality analysis; deferred until that analysis is run.
"""
from __future__ import annotations

import torch

from tcell_pipeline import config


class MatchedRandomSampler:
    def __init__(self, n_controls: int = config.N_MATCHED_CONTROLS, seed: int = 0) -> None:
        self.n_controls = n_controls
        self.gen = torch.Generator().manual_seed(seed)

    def sample(self, selection_mask: dict) -> list[dict]:
        """Return ``n_controls`` boolean masks, each matching ``selection_mask``'s per-relation count."""
        counts = {rel: int(m.sum()) for rel, m in selection_mask.items()}
        sizes = {rel: int(m.numel()) for rel, m in selection_mask.items()}
        return [self._one(counts, sizes) for _ in range(self.n_controls)]

    def _one(self, counts: dict, sizes: dict) -> dict:
        ctrl = {}
        for rel, e in sizes.items():
            m = torch.zeros(e, dtype=torch.bool)
            k = counts[rel]
            if k and e:
                m[torch.randperm(e, generator=self.gen)[:k]] = True
            ctrl[rel] = m
        return ctrl
