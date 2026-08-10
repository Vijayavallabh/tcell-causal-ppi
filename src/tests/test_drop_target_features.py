"""The graph-feature ablation must remove INFORMATION without changing CAPACITY.

The `expression_only` arm is the paper's "no graph" baseline but receives PINNACLE (learned on a PPI
network) and three PPI degree scalars, so h1 measures message passing over a graph SUMMARY rather
than graph vs no graph. This ablation supplies the missing arm. Zeroing rather than removing is the
whole point: if out_dim changed, the contrast would confound information with model capacity.
"""
from __future__ import annotations

import numpy as np
import torch

from tcell_pipeline import config
from tcell_pipeline.encoders.embedding_store import PluggableEmbeddingStore
from tcell_pipeline.encoders.target_encoder import TARGET_SCALAR_KEYS, TargetEncoder


class _FakeStore(PluggableEmbeddingStore):
    """Returns a constant non-zero vector for every id, so zeroing is unambiguous."""

    def __init__(self, dim: int, fill: float) -> None:
        self.dim, self._fill = dim, fill

    def lookup(self, uniprot_ids):  # noqa: D102
        return torch.full((len(uniprot_ids), self.dim), self._fill, dtype=torch.float32)


def _batch(n: int = 3) -> dict:
    b = {"uniprot_id": [f"P{i:05d}" for i in range(n)]}
    for j, k in enumerate(TARGET_SCALAR_KEYS):
        b[k] = np.full(n, float(j + 1), dtype=np.float32)
    return b


def _encode(monkeypatch, drop: tuple[str, ...]) -> torch.Tensor:
    monkeypatch.setattr(config, "DROP_TARGET_FEATURES", drop)
    enc = TargetEncoder(plm_store=_FakeStore(config.PLM_EMBED_DIM, 0.5),
                        pinnacle_store=_FakeStore(config.PINNACLE_EMBED_DIM, 0.25))
    return enc(_batch())


def test_ablation_preserves_dimension_so_capacity_is_not_confounded(monkeypatch):
    full = _encode(monkeypatch, ())
    both = _encode(monkeypatch, ("pinnacle", "ppi_degree"))
    assert full.shape == both.shape, "ablation changed out_dim: the contrast now confounds capacity"
    assert both.shape[1] == TargetEncoder.out_dim


def test_pinnacle_ablation_zeros_only_pinnacle(monkeypatch):
    p0 = config.PLM_EMBED_DIM
    p1 = p0 + config.PINNACLE_EMBED_DIM
    out = _encode(monkeypatch, ("pinnacle",))
    assert torch.all(out[:, p0:p1] == 0.0), "PINNACLE block not zeroed"
    assert torch.all(out[:, :p0] == 0.5), "ESM-2 must be untouched: it is sequence, not graph"
    # every scalar, including the ppi degrees, survives when only pinnacle is dropped
    for j, _ in enumerate(TARGET_SCALAR_KEYS):
        assert torch.all(out[:, p1 + j] == float(j + 1))


def test_degree_ablation_zeros_degrees_but_keeps_baseline_expression(monkeypatch):
    p1 = config.PLM_EMBED_DIM + config.PINNACLE_EMBED_DIM
    out = _encode(monkeypatch, ("ppi_degree",))
    for j, k in enumerate(TARGET_SCALAR_KEYS):
        col = out[:, p1 + j]
        if k.startswith("ppi_degree"):
            assert torch.all(col == 0.0), f"{k} not zeroed"
        else:
            assert torch.all(col == float(j + 1)), f"{k} is not graph-derived and must survive"
    assert torch.all(out[:, p1 - 1] == 0.25), "PINNACLE must survive a degree-only ablation"


def test_default_is_a_true_no_op(monkeypatch):
    """An unset env must reproduce the frozen behaviour exactly, or every prior result is invalid."""
    assert config.DROP_TARGET_FEATURES == () or isinstance(config.DROP_TARGET_FEATURES, tuple)
    full = _encode(monkeypatch, ())
    assert torch.all(full[:, :config.PLM_EMBED_DIM] == 0.5)
    p0, p1 = config.PLM_EMBED_DIM, config.PLM_EMBED_DIM + config.PINNACLE_EMBED_DIM
    assert torch.all(full[:, p0:p1] == 0.25)
    for j, _ in enumerate(TARGET_SCALAR_KEYS):
        assert torch.all(full[:, p1 + j] == float(j + 1))
