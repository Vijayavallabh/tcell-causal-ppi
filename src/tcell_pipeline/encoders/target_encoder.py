"""TargetEncoder: assembles the perturbed gene's frozen feature vector h_target.

NO trainable gene-ID embedding — H1 forbids the model from memorising a per-gene identity
vector, so the target is described only by transferable features: the frozen protein-language-
model and PINNACLE embeddings (looked up by UniProt accession) plus PPI degree and control
baseline expression. All learnable mixing happens downstream in the fusion layer.
"""
from __future__ import annotations

import torch
from torch import nn

from tcell_pipeline import config
from tcell_pipeline.encoders._tensor import as_float_vector
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore

TARGET_SCALAR_KEYS = (
    "ppi_degree_physical",
    "ppi_degree_functional",
    "ppi_degree_complex",
    "control_baseline_expr",
)


class TargetEncoder(nn.Module):
    out_dim = config.PLM_EMBED_DIM + config.PINNACLE_EMBED_DIM + len(TARGET_SCALAR_KEYS)

    def __init__(
        self,
        plm_store: PluggableEmbeddingStore | None = None,
        pinnacle_store: PluggableEmbeddingStore | None = None,
    ) -> None:
        super().__init__()
        self.plm = plm_store or PluggableEmbeddingStore(config.PLM_EMBEDDINGS_PATH, config.PLM_EMBED_DIM)
        self.pinnacle = pinnacle_store or PluggableEmbeddingStore(
            config.PINNACLE_EMBEDDINGS_PATH, config.PINNACLE_EMBED_DIM
        )

    def forward(self, batch: dict) -> torch.Tensor:
        uniprot_ids = list(batch["uniprot_id"])
        plm = self.plm.lookup(uniprot_ids)
        pinnacle = self.pinnacle.lookup(uniprot_ids)
        scalars = torch.stack([as_float_vector(batch[k]) for k in TARGET_SCALAR_KEYS], dim=1)
        return torch.cat([plm, pinnacle, scalars], dim=1)
