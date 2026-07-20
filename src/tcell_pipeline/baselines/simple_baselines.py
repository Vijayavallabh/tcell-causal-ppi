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
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tcell_pipeline import config


def _np(a, dtype=np.float64) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=dtype)


class BaseBaseline:
    """Common protocol; subclasses implement ``_fit``/``_predict_z`` and inherit gene decoding.

    ``requires_features`` marks the baselines that regress on X (ridge, kNN, low-rank): calling them with
    ``X=None`` raises a clear error rather than an opaque sklearn crash, while the feature-free baselines
    (zero, perturbed-mean, condition-mean) accept ``X=None`` — a uniform, honest fit/predict contract."""

    requires_features: bool = False

    def __init__(self, basis=None) -> None:
        self.B = None if basis is None else _np(basis)  # (G, K) frozen fold-local loadings
        self._k: int | None = None

    def _features(self, X):
        if X is None:
            if self.requires_features:
                raise ValueError(f"{type(self).__name__} requires a feature matrix X (got None)")
            return None
        return _np(X)

    def fit(self, X, z, conditions=None) -> "BaseBaseline":
        z = _np(z)
        self._k = z.shape[1]
        self._fit(self._features(X), z, conditions)
        return self

    def predict(self, X, conditions=None) -> tuple[np.ndarray, np.ndarray]:
        dz = self._predict_z(self._features(X), conditions)
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
        if conditions is None:  # no condition info -> degrade to the global perturbed mean (uniform contract)
            n = _n_rows(X, conditions)
            return np.broadcast_to(self._global, (n, self._k)).copy()
        return np.stack([self._by_cond.get(c, self._global) for c in conditions])


class RidgeBaseline(BaseBaseline):
    """Ridge regression from context features X to program delta z (multi-output)."""

    requires_features = True

    def __init__(self, basis=None, alpha: float = 1.0) -> None:
        super().__init__(basis)
        self._model = Ridge(alpha=alpha)

    def _fit(self, X, z, conditions) -> None:
        self._model.fit(X, z)

    def _predict_z(self, X, conditions) -> np.ndarray:
        return _as_columns(self._model.predict(X))


class ElasticNetBaseline(BaseBaseline):
    """Per-output ElasticNet (L1+L2) from context features X to program Δz, one fit per program in parallel.

    Standardises X on the TRAINING fold before the coordinate-descent fit — an L1 penalty is scale-sensitive,
    so unstandardised features let large-scale columns dominate the sparsity. The scaler is fit on train only
    (inside the pipeline), so no val statistics leak. Uses ``MultiOutputRegressor`` (K independent single-task
    fits across cores) rather than ``MultiTaskElasticNet``: the coupled-L21 multitask descent is
    O(iters × features × samples × K) and grinds for many minutes on 1412 features × 128 programs, while the
    parallel per-output form converges in minutes. ``alpha``/``tol`` are set for CONVERGENCE, not the score —
    more regularisation only weakens the linear model, so it cannot flatter elastic-net vs H1."""

    requires_features = True

    def __init__(self, basis=None, alpha: float = 0.1, l1_ratio: float = 0.5, max_iter: int = 2000,
                 tol: float = 1e-3, n_jobs: int = -1) -> None:
        super().__init__(basis)
        self._model = make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol),
                                 n_jobs=n_jobs))

    def _fit(self, X, z, conditions) -> None:
        self._model.fit(X, z)

    def _predict_z(self, X, conditions) -> np.ndarray:
        return _as_columns(self._model.predict(X))

    def fit_diagnostics(self) -> dict:
        """Convergence + sparsity evidence for the fitted bar.

        This baseline is published as a floor the H1 must CLEAR, so the dangerous direction is an
        UNDER-fit: a non-converged or all-zero-coefficient solution silently inflates the H1 margin it
        is supposed to bound. Regularisation-makes-it-weaker is the safe direction for a competitor, not
        for a bar — so the evidence is recorded rather than assumed."""
        head = self._model[-1]
        est = getattr(head, "estimators_", [])
        iters = [int(e.n_iter_) for e in est if getattr(e, "n_iter_", None) is not None]
        nonzero = sum(int((e.coef_ != 0).sum()) for e in est)
        total = sum(int(e.coef_.size) for e in est)
        max_iter = int(head.estimator.max_iter)
        return {"n_outputs": len(est), "n_iter_max": max(iters) if iters else None,
                "max_iter": max_iter, "converged": bool(iters and max(iters) < max_iter),
                "nonzero_coef_frac": (nonzero / total) if total else 0.0}


class NearestNeighborBaseline(BaseBaseline):
    """kNN by target/context profile: predict the mean program delta of the k nearest training rows."""

    requires_features = True

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

    requires_features = True

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
    "elastic_net": ElasticNetBaseline,
    "nearest_neighbor": NearestNeighborBaseline,
    "low_rank": LowRankBaseline,
}
