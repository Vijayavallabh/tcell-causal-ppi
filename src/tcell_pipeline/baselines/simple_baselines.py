"""Simple baselines with a common fit/predict contract (report §Baselines).

Every baseline: ``fit(X, z, conditions=None) -> self`` then ``predict(X, conditions=None) -> (delta_z,
delta_x)``. Baselines predict in PROGRAM space (delta_z, shape (M, K)); gene-space delta_x is decoded
through the same frozen fold-local basis B the decoder uses (``delta_x = delta_z @ B.T`` — the decoder's
program pathway ``B @ delta_z``). ``basis=None`` -> ``delta_x`` is returned as zeros, so a baseline stays
usable in program-only evaluation.

These are the mandatory simple references the report requires in every headline table (never compare only
to weak deep-learning baselines). Graph baselines (feat-007) and external comparators (feat-010) are out
of scope here. X is treated as an opaque feature matrix — the evaluation harness decides whether it holds
q_pre context features (ridge) or a target profile (kNN); the baselines stay agnostic.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from tcell_pipeline import config


def _np(a, dtype=np.float64) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=dtype)


class BaseBaseline:
    """Common protocol; subclasses implement ``_fit``/``_predict_z`` and inherit gene decoding."""

    def __init__(self, basis=None) -> None:
        self.B = None if basis is None else _np(basis)  # (G, K) frozen fold-local loadings
        self._k: int | None = None

    def fit(self, X, z, conditions=None) -> "BaseBaseline":
        z = _np(z)
        self._k = z.shape[1]
        self._fit(None if X is None else _np(X), z, conditions)
        return self

    def predict(self, X, conditions=None) -> tuple[np.ndarray, np.ndarray]:
        dz = self._predict_z(None if X is None else _np(X), conditions)
        return dz, self._decode_genes(dz)

    def _decode_genes(self, dz: np.ndarray) -> np.ndarray:
        if self.B is None:
            return np.zeros((dz.shape[0], 0), dtype=np.float64)
        return dz @ self.B.T

    def _fit(self, X, z, conditions) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _predict_z(self, X, conditions) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError


class ZeroBaseline(BaseBaseline):
    """No-effect baseline: predict zero delta (§10.1 worst-case control)."""

    def _fit(self, X, z, conditions) -> None:
        pass

    def _predict_z(self, X, conditions) -> np.ndarray:
        n = _n_rows(X, conditions)
        return np.zeros((n, self._k), dtype=np.float64)


class PerturbedMeanBaseline(BaseBaseline):
    """Systema non-control / perturbed mean: predict the average training perturbation effect for every
    row — the reference every headline endpoint must clear."""

    def _fit(self, X, z, conditions) -> None:
        self._mean = z.mean(0)

    def _predict_z(self, X, conditions) -> np.ndarray:
        n = _n_rows(X, conditions)
        return np.broadcast_to(self._mean, (n, self._k)).copy()


class ConditionMeanBaseline(BaseBaseline):
    """Per-condition mean perturbation effect; unseen conditions fall back to the global perturbed mean."""

    def _fit(self, X, z, conditions) -> None:
        self._global = z.mean(0)
        self._by_cond: dict = {}
        if conditions is not None:
            conditions = list(conditions)
            for c in set(conditions):
                mask = np.array([x == c for x in conditions])
                self._by_cond[c] = z[mask].mean(0)

    def _predict_z(self, X, conditions) -> np.ndarray:
        if conditions is None:
            raise ValueError("ConditionMeanBaseline.predict needs the per-row conditions")
        return np.stack([self._by_cond.get(c, self._global) for c in conditions])


class RidgeBaseline(BaseBaseline):
    """Ridge regression from context features X to program delta z (multi-output)."""

    def __init__(self, basis=None, alpha: float = 1.0) -> None:
        super().__init__(basis)
        self._model = Ridge(alpha=alpha)

    def _fit(self, X, z, conditions) -> None:
        self._model.fit(X, z)

    def _predict_z(self, X, conditions) -> np.ndarray:
        return _as_columns(self._model.predict(X))


class NearestNeighborBaseline(BaseBaseline):
    """kNN by target/context profile: predict the mean program delta of the k nearest training rows."""

    def __init__(self, basis=None, k: int = 1) -> None:
        super().__init__(basis)
        self._k_neighbors = k

    def _fit(self, X, z, conditions) -> None:
        self._z_train = z
        self._nn = NearestNeighbors(n_neighbors=min(self._k_neighbors, X.shape[0])).fit(X)

    def _predict_z(self, X, conditions) -> np.ndarray:
        idx = self._nn.kneighbors(X, return_distance=False)
        return self._z_train[idx].mean(axis=1)


class LowRankBaseline(BaseBaseline):
    """Low-rank matrix factorisation: regress X onto a truncated-SVD program subspace, then reconstruct.

    Fits the top-``rank`` right singular directions of the centred training responses, learns a ridge map
    from features into those reduced coordinates, and decodes back — a denoised linear predictor that
    cannot chase program directions unsupported by the training responses."""

    def __init__(self, basis=None, rank: int = 8, alpha: float = 1.0) -> None:
        super().__init__(basis)
        self._rank = rank
        self._model = Ridge(alpha=alpha)

    def _fit(self, X, z, conditions) -> None:
        self._mean = z.mean(0)
        r = min(self._rank, z.shape[0], z.shape[1])
        _, _, vt = np.linalg.svd(z - self._mean, full_matrices=False)
        self._components = vt[:r]                     # (r, K)
        self._model.fit(X, (z - self._mean) @ self._components.T)

    def _predict_z(self, X, conditions) -> np.ndarray:
        reduced = _as_columns(self._model.predict(X))
        return reduced @ self._components + self._mean


def _as_columns(pred: np.ndarray) -> np.ndarray:
    """sklearn returns a 1-D (M,) vector for a single target; keep it a column (M, 1), never atleast_2d's
    row (1, M) which would transpose a K==1 or rank==1 prediction."""
    return pred if pred.ndim == 2 else pred[:, None]


def _n_rows(X, conditions) -> int:
    if X is not None:
        return X.shape[0]
    if conditions is not None:
        return len(conditions)
    raise ValueError("cannot infer the number of rows to predict without X or conditions")


BASELINES: dict = {
    "zero": ZeroBaseline,
    "perturbed_mean": PerturbedMeanBaseline,
    "condition_mean": ConditionMeanBaseline,
    "ridge": RidgeBaseline,
    "nearest_neighbor": NearestNeighborBaseline,
    "low_rank": LowRankBaseline,
}
