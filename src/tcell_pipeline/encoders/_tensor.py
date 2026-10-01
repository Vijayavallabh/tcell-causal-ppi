"""Batch-value coercion shared by the sub-encoders."""
from __future__ import annotations

import numpy as np
import torch


def as_float_vector(value) -> torch.Tensor:
    """Coerce a batch column (tensor / list / ndarray / bool) to a 1-D float32 tensor.

    Missing values (a real gene lacks a control baseline in ~5% of rows; some rows lack a
    guide count) are neutralised to 0 so one NaN scalar cannot propagate through LayerNorm and
    poison the entire h_do embedding.
    """
    if isinstance(value, torch.Tensor):
        t = value.to(torch.float32).reshape(-1)
    else:
        t = torch.as_tensor(np.asarray(value, dtype=np.float32)).reshape(-1)
    # ponytail: NaN/inf -> 0 fill; upgrade to fold-fit mean imputation in the Module 3 loader.
    return torch.nan_to_num(t)
