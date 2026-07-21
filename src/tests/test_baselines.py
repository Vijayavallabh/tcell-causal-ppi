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
    ElasticNetBaseline,
    LowRankBaseline,
    NearestNeighborBaseline,
    PerturbedMeanBaseline,
    RidgeBaseline,
    ZeroBaseline,
)
from tcell_pipeline.run_module8_real import flag_underfit_bars

_N, _M, _D, _K, _G = 40, 10, 6, 8, 30


def _groups(n: int, span: int = 3) -> list[str]:
    """Per-row target symbols, one gene spanning ``span`` rows as in the real mart.

    Bars that hold rows out internally need these; without them they fall back to a random-row split
    and warn, which is correct behaviour but not the path these tests mean to exercise."""
    return [f"GENE{i // span}" for i in range(n)]


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
    model.fit(X, z, conditions=conds, groups=_groups(len(X)))
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


def test_elastic_net_learns_linear_signal():
    X, z, B, Xt, conds, conds_t = _data()
    en = ElasticNetBaseline(basis=B).fit(X, z)
    zero = ZeroBaseline(basis=B).fit(X, z)
    dz_en, _ = en.predict(X)
    dz_zero, _ = zero.predict(X)
    err_en = np.mean((dz_en - z) ** 2)
    err_zero = np.mean((dz_zero - z) ** 2)
    assert err_en < 0.5 * err_zero          # standardised elastic-net recovers most of the linear signal


def test_elastic_net_records_fit_diagnostics():
    """elastic_net is published as THE strongest tabular baseline backing an H1 margin, so an under-fit
    (non-converged / all-zero-coefficient) bar would INFLATE that margin. The fit must expose evidence."""
    X, z, B, Xt, conds, conds_t = _data()
    m = ElasticNetBaseline(basis=B).fit(X, z)
    d = m.fit_diagnostics()
    assert d["n_iter_max"] is not None and d["n_iter_max"] >= 1
    assert 0.0 <= d["nonzero_coef_frac"] <= 1.0
    assert d["nonzero_coef_frac"] > 0.0, "an all-zero-coefficient fit is a degenerate bar, not a baseline"
    assert isinstance(d["converged"], bool)


@pytest.mark.parametrize("name", ["ridge", "elastic_net", "nearest_neighbor", "low_rank"])
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


def test_elastic_net_diagnostics_detect_a_truncated_fit():
    """CONSTRUCTED breaker for the convergence guard: the previous test only asserted
    ``isinstance(converged, bool)``, which no input can falsify — a guard whose input is a constant can
    only confirm (AGENTS.md). Force the solver to stop at the iteration cap and the flag must go False;
    give it room and it must go True, or the flag is decoration."""
    X, z, B, _, _, _ = _data()
    truncated = ElasticNetBaseline(basis=B, alpha=0.01, max_iter=1, tol=1e-12).fit(X, z)
    d = truncated.fit_diagnostics()
    assert d["converged"] is False, "a fit stopped at max_iter is NOT converged"
    assert d["n_iter_max"] == d["max_iter"] == 1

    shipped = ElasticNetBaseline(basis=B).fit(X, z)             # the DEFAULTS published as the H1 floor
    assert shipped.fit_diagnostics()["converged"] is True, "the shipped bar must converge, or it under-bounds"


def test_gradient_boosting_learns_a_nonlinear_signal_the_linear_bars_cannot():
    """The real tabular threat to H1 is a non-linear learner, not another linear one: the report calls this
    a near-null-signal regime where strong TABULAR models are what could break the comparator clause. Pin
    that the bar actually captures curvature ridge cannot."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((300, 4))
    z = np.stack([np.sin(3 * X[:, 0]) + X[:, 1] ** 2, np.abs(X[:, 2]) * X[:, 3]], 1)   # purely non-linear
    Xt = rng.standard_normal((80, 4))
    zt = np.stack([np.sin(3 * Xt[:, 0]) + Xt[:, 1] ** 2, np.abs(Xt[:, 2]) * Xt[:, 3]], 1)
    B = rng.standard_normal((_G, 2))

    gb_err = np.mean((BASELINES["gradient_boosting"](basis=B).fit(X, z).predict(Xt)[0] - zt) ** 2)
    ridge_err = np.mean((RidgeBaseline(basis=B).fit(X, z).predict(Xt)[0] - zt) ** 2)
    assert gb_err < 0.5 * ridge_err, f"gradient boosting {gb_err:.4f} must beat ridge {ridge_err:.4f} here"


def test_gradient_boosting_reports_unknown_convergence_with_a_reason():
    """CONTRACT CHANGED 2026-07-21 (review finding 5). This bar used to report a convergence VERDICT
    from sklearn's ``early_stopping``, whose validation set can only be carved out by RANDOM ROWS. One
    target gene spans ~3 rows, so those rows shared genes with the fitted rows and the verdict kept
    rewarding depth past the point of blocked-target-OOD generalisation — measured on the real fold in
    ``probe_catboost_budget.py``: the fit its own early stopping called converged scored 0.0553 on val,
    WORSE than an arbitrary 1000-tree cut at 0.0657.

    sklearn HGB offers no eval-set hook, so there is no way to give it a grouped split. The bar is now
    fixed-budget and reports ``converged=None`` WITH a reason. That is not a downgrade of the gate:
    ``flag_underfit_bars`` routes None to ``unknown`` and still sets ``margin_is_upper_bound``, so the
    margin stays conservatively bounded — it is now bounded by an honest absence of evidence rather
    than by a verdict measured on leaked rows."""
    X, z, B, _, _, _ = _data()
    d = BASELINES["gradient_boosting"](basis=B).fit(X, z).fit_diagnostics()

    assert d["converged"] is None, "a verdict here could only come from a leaking random-row split"
    assert d["converged_unknown_reason"], "unknown without a stated reason is indistinguishable from a bug"
    assert d["n_outputs"] == _K and d["n_iter_max"] >= 1   # evidence is still reported, just not a verdict
    assert flag_underfit_bars({"gradient_boosting": d})["margin_is_upper_bound"] is True


def test_catboost_learns_a_nonlinear_signal_and_reports_convergence():
    """CatBoost is the bar feat-006's description actually names, and unlike the per-output boosters it
    fits all K programs in ONE model (MultiRMSE), sharing tree structure across them. CONSTRUCTED breaker:
    starve the iteration budget and ``converged`` must go False."""
    pytest.importorskip("catboost")
    rng = np.random.default_rng(11)
    X = rng.standard_normal((300, 4))
    z = np.stack([np.sin(3 * X[:, 0]) + X[:, 1] ** 2, np.abs(X[:, 2]) * X[:, 3]], 1)
    Xt = rng.standard_normal((80, 4))
    zt = np.stack([np.sin(3 * Xt[:, 0]) + Xt[:, 1] ** 2, np.abs(Xt[:, 2]) * Xt[:, 3]], 1)
    B = rng.standard_normal((_G, 2))

    m = BASELINES["catboost"](basis=B, iterations=300).fit(X, z, groups=_groups(len(X)))
    cat_err = np.mean((m.predict(Xt)[0] - zt) ** 2)
    ridge_err = np.mean((RidgeBaseline(basis=B).fit(X, z).predict(Xt)[0] - zt) ** 2)
    assert cat_err < ridge_err, f"catboost {cat_err:.4f} must beat ridge {ridge_err:.4f} on curvature"

    # od_wait > iterations so early stopping CANNOT fire: the cap is the only thing that can stop it
    starved = BASELINES["catboost"](basis=B, iterations=3, od_wait=100).fit(X, z, groups=_groups(len(X))).fit_diagnostics()
    assert starved["converged"] is False, "a bar stopped by its iteration cap is still improving"


def test_tabicl_declares_unknown_convergence_so_the_gate_cannot_read_it_as_a_pass():
    """TabICL is an amortized in-context model: there is no iterative fit to converge, so ``converged`` is
    genuinely UNKNOWN, not True. That distinction is load-bearing — this fold has ~14x more columns than
    TabICL's pre-training range, so its score may under-represent the family, and a bar we cannot certify
    was fit well enough to bound must push the H1 margin to an upper bound rather than silently pass."""
    pytest.importorskip("tabicl")
    from tcell_pipeline.baselines import TabICLBaseline
    from tcell_pipeline.run_module8_real import flag_underfit_bars

    d = TabICLBaseline.diagnostics_for(n_features=1453, n_outputs=128)
    assert d["converged"] is None                       # NOT True — absence of evidence is not a pass
    assert d["n_features"] == 1453
    assert d["out_of_pretraining_range"] is True
    assert flag_underfit_bars({"tabicl": d})["margin_is_upper_bound"] is True


def test_boost_converged_requires_a_full_patience_window():
    """CONSTRUCTED breaker from the real run. `tree_count_ < max_iter` reads True at 999/1000, but CatBoost
    sets tree_count_ = best_iteration + 1, so best was iteration 998 and overfitting detection never had its
    od_wait=50 window to fire in — the bar was still improving when the budget ran out. Early stopping
    genuinely fired only if `tree_count_ + od_wait <= max_iter`. Verified against catboost 1.2.10: a
    generous budget (2000 iters, od_wait=10) stopped at tree_count_=99, and a starved one (5 iters,
    od_wait=100) ran to 5."""
    from tcell_pipeline.baselines.simple_baselines import _boost_converged

    assert _boost_converged(999, 1000, 50) is False       # THE REAL RUN — shipped criterion said True
    assert _boost_converged(1000, 1000, 50) is False      # ran the whole budget
    assert _boost_converged(99, 2000, 10) is True         # measured genuine early stop
    assert _boost_converged(5, 5, 100) is False           # starved: cap, not a plateau
    assert _boost_converged(950, 1000, 50) is True        # exactly a full patience window fits
    assert _boost_converged(951, 1000, 50) is False       # one short of it
