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

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
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

    def fit(self, X, z, conditions=None, groups=None) -> "BaseBaseline":
        """``groups`` are the per-row target symbols, used by bars that hold rows out INTERNALLY.

        One target gene spans ~3 rows here, so a random-row internal holdout shares targets with the
        rows it selects on and keeps rewarding capacity long after blocked-target-OOD generalisation has
        decayed. Bars that do not split internally ignore it."""
        z = _np(z)
        self._k = z.shape[1]
        self._groups = None if groups is None else [str(g) for g in groups]
        self._fit(self._features(X), z, conditions)
        return self

    def _internal_split(self, n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """(keep, hold) row indices for an internal holdout — target-grouped when groups are known.

        Falls back to random rows with a WARNING rather than silently, because the fallback is the exact
        leak this method exists to remove and a silent fallback would read as a grouped split."""
        groups = getattr(self, "_groups", None)
        if groups is not None and len(groups) == n:
            from tcell_pipeline.training.inner_split import group_partition
            keep, hold = group_partition(groups, holdout_frac=frac, seed=seed)
            return np.asarray(keep, dtype=int), np.asarray(hold, dtype=int)
        warnings.warn(
            f"{type(self).__name__}: no target groups supplied, falling back to a RANDOM ROW internal "
            f"holdout. One target spans several rows, so the resulting convergence verdict is "
            f"contaminated and must not be read as evidence about generalisation.",
            stacklevel=3,
        )
        idx = np.random.default_rng(seed).permutation(n)
        n_hold = max(1, int(round(n * frac)))
        return idx[n_hold:], idx[:n_hold]

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
    parallel per-output form converges in minutes.

    ``selection="random"`` is what makes the bar actually CONVERGE. The first shipped config
    (max_iter=2000, tol=1e-3, cyclic) stopped at its iteration cap on the real fold with only 6.4% non-zero
    coefficients — an UNDER-fit floor, which inflates the very H1 margin it exists to bound. Randomised
    coordinate descent reaches the same optimum (identical support) in ~70-100 sweeps instead of thousands,
    so ``max_iter=20000, tol=1e-4`` now converges in ~1.4 s/output and the support rises to ~30%."""

    requires_features = True

    def __init__(self, basis=None, alpha: float = 0.1, l1_ratio: float = 0.5, max_iter: int = 20000,
                 tol: float = 1e-4, n_jobs: int = -1, selection: str = "random",
                 random_state: int = 0) -> None:
        super().__init__(basis)
        self._model = make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=tol,
                                            selection=selection, random_state=random_state),
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
                "max_iter": max_iter, "converged": _converged_from_iters(iters, max_iter),
                "nonzero_coef_frac": (nonzero / total) if total else 0.0}


class GradientBoostingBaseline(BaseBaseline):
    """Histogram gradient-boosted trees, one booster per program (the report's CatBoost slot).

    The report calls this a near-null-signal regime in which strong TABULAR models are the real threat to
    H1's comparator clause, and every other bar here is linear or an average — this is the only one that can
    fit curvature and interactions. ``HistGradientBoostingRegressor`` is sklearn's LightGBM-style binned
    booster, so the bar needs no new dependency (a CatBoost/LightGBM build would add one for the same
    algorithm family).

    ``early_stopping`` is ON with a TRAIN-internal ``validation_fraction`` — never the evaluation fold, so
    the leakage fence holds. That is what gives the bar an honest convergence signal: stopping before the
    cap means the boosting loss plateaued, while ``n_iter_ == max_iter`` means the budget ran out with the
    model still improving, i.e. an under-fit floor that would inflate the H1 margin."""

    requires_features = True

    def __init__(self, basis=None, max_iter: int = 500, learning_rate: float = 0.1,
                 max_leaf_nodes: int = 31, validation_fraction: float = 0.1,
                 n_iter_no_change: int = 20, random_state: int = 0, n_jobs: int = 8) -> None:
        super().__init__(basis)
        # early_stopping is OFF, deliberately. sklearn's HGB can only carve its validation set out by
        # RANDOM ROWS (`validation_fraction`) and offers no eval-set hook, so its early stopping selects
        # depth on rows whose target genes are also in the fitted rows — the exact leak
        # `training/inner_split.py` exists to remove, measured on the real fold in
        # `probe_catboost_budget.py`: the fit its own early stopping called converged scored 0.0553 on
        # val, WORSE than an arbitrary 1000-tree cut at 0.0657. A fixed budget with NO convergence claim
        # is honest; a leaked verdict feeding the under-fit gate is not. CatBoostBaseline keeps a real
        # verdict because it accepts an explicit, target-grouped eval_set.
        self._n_iter_no_change = n_iter_no_change
        self._model = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                          max_leaf_nodes=max_leaf_nodes, early_stopping=False,
                                          n_iter_no_change=n_iter_no_change, random_state=random_state),
            n_jobs=n_jobs)

    def _fit(self, X, z, conditions) -> None:
        self._model.fit(X, z)

    def _predict_z(self, X, conditions) -> np.ndarray:
        return _as_columns(self._model.predict(X))

    def fit_diagnostics(self) -> dict:
        """Fixed-budget bar: iteration counts are reported, a convergence VERDICT is not.

        ``converged`` is None because the only selection signal sklearn offers here is a random-row
        holdout, and a verdict measured on leaked rows is not evidence about generalisation.
        ``flag_underfit_bars`` routes None to ``unknown``, which is the truthful bucket."""
        est = getattr(self._model, "estimators_", [])
        iters = [int(e.n_iter_) for e in est if getattr(e, "n_iter_", None) is not None]
        max_iter = int(self._model.estimator.max_iter)
        return {"n_outputs": len(est), "n_iter_max": max(iters) if iters else None,
                "n_iter_mean": (sum(iters) / len(iters)) if iters else None, "max_iter": max_iter,
                "converged": None,
                "converged_unknown_reason": "fixed budget; sklearn HGB can only early-stop on a "
                                            "random-row split, which shares target genes with the "
                                            "fitted rows"}


def _converged_from_iters(iters, max_iter: int) -> bool | None:
    """Convergence from per-output iteration counts, or None when there is NO evidence either way.

    The bug this replaces was ``bool(iters and max(iters) < max_iter)``: an empty ``iters`` collapses to
    ``False``, which is not "we could not tell" but the positive claim "it did not converge". That claim
    is then published as ``under-fit=[...]`` in ``tabular_baselines_vs_h1.json`` and is exactly what a
    reader uses to decide the H1 margin is an UPPER bound — a fabricated fact doing load-bearing work.
    Absence of evidence is None; ``flag_underfit_bars`` already routes None to ``unknown``."""
    iters = list(iters)
    if not iters:
        return None
    return max(iters) < max_iter


def _boost_converged(used: int, max_iter: int, od_wait: int) -> bool:
    """Did early stopping genuinely fire, or did the budget simply run out?

    ``used < max_iter`` is too lenient. CatBoost sets ``tree_count_ = best_iteration + 1``, and overfitting
    detection only fires after ``od_wait`` consecutive non-improving rounds — so it needs a full patience
    window inside the budget. The real feat-006 run returned ``tree_count_=999`` of ``max_iter=1000`` with
    ``od_wait=50``: the naive test called that converged, but the best iteration was 998 and the window
    never had room to close. The bar was still improving when the budget ended, which makes its score a
    LOWER bound — the dangerous direction for a floor H1 must clear.

    Verified against catboost 1.2.10: a generous budget (2000 iters, od_wait=10) stopped at tree_count_=99
    (best 98), and a starved one (5 iters, od_wait=100) ran to 5 — both branches exercised."""
    return bool(used + od_wait <= max_iter)


class CatBoostBaseline(BaseBaseline):
    """CatBoost with ``MultiRMSE`` — the one bar here that models all K programs in a SINGLE model.

    Every other learner in this module fits each program independently (``MultiOutputRegressor``), so it
    cannot borrow strength across programs. ``MultiRMSE`` grows one tree ensemble whose leaves carry a
    K-vector, sharing the split structure across all outputs. In a near-null-signal regime that shared
    structure is a real denoiser — the same reason ``LowRankBaseline`` helps — so it is a genuinely
    different inductive bias, not a second copy of the histogram booster.

    Early stopping (``od_type='Iter'``) runs against a TRAIN-internal holdout, never the evaluation fold."""

    requires_features = True

    def __init__(self, basis=None, iterations: int = 1000, depth: int = 6, learning_rate: float = 0.1,
                 od_wait: int = 50, validation_fraction: float = 0.1, random_state: int = 0,
                 thread_count: int = -1) -> None:
        super().__init__(basis)
        self._iterations = iterations
        self._validation_fraction = validation_fraction
        self._kw = dict(loss_function="MultiRMSE", iterations=iterations, depth=depth,
                        learning_rate=learning_rate, od_type="Iter", od_wait=od_wait,
                        random_seed=random_state, verbose=False, thread_count=thread_count)
        self._model = None

    def _fit(self, X, z, conditions) -> None:
        from catboost import CatBoostRegressor

        # TARGET-GROUPED, not random rows: one gene spans ~3 rows, so a permutation split puts the same
        # gene on both sides and early stopping keeps rewarding depth long after blocked-target-OOD
        # generalisation has decayed. Measured on the real fold (probe_catboost_budget.py): the 4000-iter
        # fit CatBoost's own early stopping called converged scored 0.0553 on val, WORSE than an
        # arbitrary 1000-tree cut at 0.0657.
        keep, hold = self._internal_split(X.shape[0], self._validation_fraction,
                                          int(self._kw["random_seed"]))
        self._model = CatBoostRegressor(**self._kw)
        self._model.fit(X[keep], z[keep], eval_set=(X[hold], z[hold]))

    def _predict_z(self, X, conditions) -> np.ndarray:
        return _as_columns(np.asarray(self._model.predict(X), dtype=np.float64))

    def fit_diagnostics(self) -> dict:
        """``tree_count_`` is what the ensemble actually kept. Equal to the cap means early stopping never
        fired and the model was still improving when the budget ran out — an under-fit floor."""
        used = int(self._model.tree_count_)
        od_wait = int(self._kw["od_wait"])
        return {"n_outputs": self._k, "n_iter_max": used, "max_iter": int(self._iterations),
                "od_wait": od_wait, "converged": _boost_converged(used, self._iterations, od_wait)}


class TabICLBaseline(BaseBaseline):
    """TabICL v2 — a tabular in-context foundation model, one target at a time.

    "Fitting" an in-context model is just holding the training context, so the K per-program regressors are
    built lazily inside ``_predict_z`` and released one at a time: each one peaks around 41 GiB of GPU
    memory on this fold, so keeping all K alive would not fit on any single device.

    HONESTY NOTE (see ``diagnostics_for``): this fold has ~14x more columns than TabICL's documented
    pre-training range, so it is being used outside the regime it was trained for. That is a reason its
    score may UNDER-represent the model family, which is the dangerous direction for a bar H1 must clear —
    hence ``converged=None`` (unknown), never ``True``."""

    requires_features = True
    PRETRAIN_MAX_FEATURES = 100        # documented pre-training range is 2-100 columns

    def __init__(self, basis=None, device: str | None = None, random_state: int = 0) -> None:
        super().__init__(basis)
        self._device = device
        self._random_state = random_state
        self._n_features: int | None = None

    def _fit(self, X, z, conditions) -> None:
        self._Xtr, self._ztr = X, z    # in-context: the training set IS the model
        self._n_features = int(X.shape[1])

    def _predict_z(self, X, conditions) -> np.ndarray:
        from tabicl import TabICLRegressor

        out = np.empty((X.shape[0], self._ztr.shape[1]), dtype=np.float64)
        for k in range(self._ztr.shape[1]):
            reg = TabICLRegressor(device=self._device, random_state=self._random_state)
            reg.fit(self._Xtr, self._ztr[:, k])
            out[:, k] = reg.predict(X)
            del reg
            _free_cuda()
        return out

    @classmethod
    def diagnostics_for(cls, n_features: int, n_outputs: int) -> dict:
        return {
            "n_outputs": int(n_outputs), "n_features": int(n_features), "converged": None,
            "pretraining_feature_range": "2-100 columns documented; benchmarked to ~2000",
            "out_of_pretraining_range": bool(n_features > cls.PRETRAIN_MAX_FEATURES),
            "note": ("amortized in-context model: there is no iterative fit to converge, so convergence is "
                     "UNKNOWN, not True. Used outside its pre-training column range here, so its score may "
                     "under-represent the family — the wrong direction for a bar H1 must clear."),
        }

    def fit_diagnostics(self) -> dict:
        return self.diagnostics_for(self._n_features, self._ztr.shape[1])


def _free_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a hard dep here, but the bar must not need it
        pass


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
    "gradient_boosting": GradientBoostingBaseline,
    "catboost": CatBoostBaseline,
    "tabicl": TabICLBaseline,
    "nearest_neighbor": NearestNeighborBaseline,
    "low_rank": LowRankBaseline,
}
