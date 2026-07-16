"""Module 6 (Simple Baselines) tests — fully synthetic.

Covers the common fit/predict contract: output shapes (program + decoded gene space), the zero baseline,
per-condition grouping, that ridge/low-rank actually learn a linear signal, and that k=1 nearest-neighbor
returns its neighbour's stored response.
"""
from __future__ import annotations

import numpy as np
import pytest

from tcell_pipeline.baselines import (
    BASELINES,
    ConditionMeanBaseline,
    LowRankBaseline,
    NearestNeighborBaseline,
    PerturbedMeanBaseline,
    RidgeBaseline,
    ZeroBaseline,
)

_N, _M, _D, _K, _G = 40, 10, 6, 8, 30


def _data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((_N, _D))
    W = rng.standard_normal((_D, _K))
    z = X @ W + 0.05 * rng.standard_normal((_N, _K))          # a learnable linear signal
    B = rng.standard_normal((_G, _K))                          # frozen fold-local loadings
    Xt = rng.standard_normal((_M, _D))
    conds = rng.choice(["Rest", "Stim8hr", "Stim48hr"], size=_N).tolist()
    conds_t = rng.choice(["Rest", "Stim8hr", "Stim48hr"], size=_M).tolist()
    return X, z, B, Xt, conds, conds_t


@pytest.mark.parametrize("name", list(BASELINES))
def test_each_baseline_fit_predict_shapes(name):
    X, z, B, Xt, conds, conds_t = _data()
    model = BASELINES[name](basis=B)
    model.fit(X, z, conditions=conds)
    dz, dx = model.predict(Xt, conditions=conds_t)
    assert dz.shape == (_M, _K)
    assert dx.shape == (_M, _G)
    assert np.isfinite(dz).all() and np.isfinite(dx).all()


def test_zero_baseline_returns_zeros():
    X, z, B, Xt, conds, conds_t = _data()
    dz, dx = ZeroBaseline(basis=B).fit(X, z).predict(Xt, conditions=conds_t)
    assert not dz.any() and not dx.any()


def test_decoded_genes_match_program_projection():
    X, z, B, Xt, conds, conds_t = _data()
    dz, dx = PerturbedMeanBaseline(basis=B).fit(X, z).predict(Xt, conditions=conds_t)
    assert np.allclose(dx, dz @ B.T)                          # gene space is the decoder's B @ delta_z


def test_basis_none_gives_empty_gene_block():
    X, z, _, Xt, conds, conds_t = _data()
    dz, dx = PerturbedMeanBaseline().fit(X, z).predict(Xt, conditions=conds_t)
    assert dz.shape == (_M, _K) and dx.shape == (_M, 0)       # program-only evaluation stays usable


def test_condition_mean_groups_by_condition():
    X, z, B, Xt, conds, conds_t = _data()
    model = ConditionMeanBaseline(basis=B).fit(X, z, conditions=conds)
    dz, _ = model.predict(Xt, conditions=conds_t)
    for i, c in enumerate(conds_t):
        expected = z[np.array([cc == c for cc in conds])].mean(0)
        assert np.allclose(dz[i], expected)


def test_condition_mean_unseen_condition_falls_back_to_global():
    X, z, B, _, conds, _ = _data()
    model = ConditionMeanBaseline(basis=B).fit(X, z, conditions=conds)
    dz, _ = model.predict(None, conditions=["NoSuchCondition"])
    assert np.allclose(dz[0], z.mean(0))


def test_perturbed_mean_is_training_average():
    X, z, B, Xt, conds, conds_t = _data()
    dz, _ = PerturbedMeanBaseline(basis=B).fit(X, z).predict(Xt, conditions=conds_t)
    assert np.allclose(dz, np.broadcast_to(z.mean(0), dz.shape))


def test_ridge_learns_linear_signal():
    X, z, B, Xt, conds, conds_t = _data()
    ridge = RidgeBaseline(basis=B, alpha=0.1).fit(X, z)
    zero = ZeroBaseline(basis=B).fit(X, z)
    dz_ridge, _ = ridge.predict(X)
    dz_zero, _ = zero.predict(X)
    err_ridge = np.mean((dz_ridge - z) ** 2)
    err_zero = np.mean((dz_zero - z) ** 2)
    assert err_ridge < 0.25 * err_zero


def test_low_rank_learns_and_respects_rank():
    X, z, B, Xt, conds, conds_t = _data()
    model = LowRankBaseline(basis=B, rank=3, alpha=0.1).fit(X, z)
    assert model._components.shape == (3, _K)
    dz, _ = model.predict(X)
    err = np.mean((dz - z) ** 2)
    assert err < np.mean((z - z.mean(0)) ** 2)                # better than the mean predictor


def test_nearest_neighbor_returns_neighbor_values():
    X, z, B, _, conds, conds_t = _data()
    model = NearestNeighborBaseline(basis=B, k=1).fit(X, z)
    dz, _ = model.predict(X)                                  # each row's nearest neighbour is itself
    assert np.allclose(dz, z)


@pytest.mark.parametrize("name", ["ridge", "nearest_neighbor", "low_rank"])
def test_feature_requiring_baselines_reject_none_features(name):
    X, z, B, _, conds, conds_t = _data()
    with pytest.raises(ValueError, match="requires a feature matrix X"):
        BASELINES[name](basis=B).fit(None, z)                 # clear error, not an opaque sklearn crash


@pytest.mark.parametrize("name", ["zero", "perturbed_mean", "condition_mean"])
def test_feature_free_baselines_accept_none_features(name):
    X, z, B, _, conds, conds_t = _data()
    dz, _ = BASELINES[name](basis=B).fit(None, z, conditions=conds).predict(None, conditions=conds_t)
    assert dz.shape == (_M, _K)                                # feature-free lane works uniformly


def test_condition_mean_without_conditions_falls_back_to_global():
    X, z, B, Xt, conds, conds_t = _data()
    model = ConditionMeanBaseline(basis=B).fit(X, z, conditions=conds)
    dz, _ = model.predict(Xt)                                  # no conditions -> global perturbed mean
    assert np.allclose(dz, np.broadcast_to(z.mean(0), dz.shape))


def test_ridge_and_low_rank_handle_single_program():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((20, _D))
    z = X @ rng.standard_normal((_D, 1))                      # K == 1 (single program)
    B = rng.standard_normal((_G, 1))
    Xt = rng.standard_normal((5, _D))
    for model in (RidgeBaseline(basis=B, alpha=0.1), LowRankBaseline(basis=B, rank=1, alpha=0.1)):
        dz, dx = model.fit(X, z).predict(Xt)
        assert dz.shape == (5, 1)                             # column, not atleast_2d's (1, M) transpose
        assert dx.shape == (5, _G)
