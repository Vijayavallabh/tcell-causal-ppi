"""Frozen, pluggable embedding lookup (PLM / PINNACLE) keyed by UniProt accession.

NOT an nn.Module: these are pretrained, frozen feature vectors loaded as data, never a
trainable parameter. Populate the parquets with tcell_pipeline.embeddings_plm /
embeddings_pinnacle. Any protein without a stored vector — a store whose parquet is absent,
or an id outside a store's coverage (e.g. proteins outside PINNACLE's cell-type context) —
falls back to a zero vector, so Module 1 trains and runs unchanged regardless of coverage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PluggableEmbeddingStore:
    def __init__(self, path: Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self._cache: dict[str, np.ndarray] | None = None

    @property
    def available(self) -> bool:
        return self.path.exists()

    def _load(self) -> dict[str, np.ndarray]:
        if self._cache is not None:
            return self._cache
        cache: dict[str, np.ndarray] = {}
        if self.path.exists():
            import pandas as pd

            df = pd.read_parquet(self.path)
            for uid, emb in zip(df["uniprot_id"], df["embedding"]):
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                if vec.shape != (self.dim,):
                    raise ValueError(
                        f"{self.path}: embedding for {uid} has {vec.shape[0]} dims, expected {self.dim}"
                    )
                cache[str(uid)] = vec
        self._cache = cache
        return cache

    def lookup(self, uniprot_ids: list[str]) -> torch.Tensor:
        """Return an (N, dim) float32 tensor; missing or null ids fall back to a zero vector."""
        cache = self._load()
        zero = np.zeros(self.dim, dtype=np.float32)
        rows = [cache.get(str(u), zero) if u is not None else zero for u in uniprot_ids]
        if not rows:
            return torch.zeros((0, self.dim), dtype=torch.float32)
        return torch.from_numpy(np.stack(rows))
