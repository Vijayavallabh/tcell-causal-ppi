"""Shared torch/array -> 2-D float64 numpy conversion for the evaluation modules.

One definition instead of a copy pasted into each metric module, so a change to the detach/dtype/reshape
behaviour can't silently diverge one module's numbers from another's. ``metrics_ref`` keeps its OWN
converter on purpose — it is the independent second implementation and must share no code with metrics.py.
"""
from __future__ import annotations

import numpy as np


def to_numpy(a) -> np.ndarray:
    """Detach a torch tensor if needed, cast to float64, and promote a 1-D vector to a single row."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(1, -1) if a.ndim == 1 else a
