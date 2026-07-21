"""Module 6 (Evaluation Metrics) tests — fully synthetic.

Covers: each metric's known-answer behaviour, the two-independent-implementations agreement on a fixed
fixture, the G2-MQ ordering gate, the §10.5 control-reference safeguards (independent vs shared control,
and the null control predictor scoring ~0), and the common prediction-schema roundtrip.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tcell_pipeline.evaluation import metric_qualification as mq
from tcell_pipeline.evaluation import control_reference as cr
from tcell_pipeline.evaluation import metrics, metrics_ref
from tcell_pipeline.evaluation.output_schema import (
    prediction_path,
    read_predictions,
    write_predictions,
)

_N, _G, _K = 12, 40, 8


def _fixture(seed: int = 0):
    rng = np.random.default_rng(seed)
    true = rng.standard_normal((_N, _G))
    pred = 0.8 * true + 0.4 * rng.standard_normal((_N, _G))  # correlated but imperfect
    return pred, true


def test_mae_rmse_known_answer():
    pred = np.array([[1.0, 2.0, 3.0]])
    true = np.array([[1.0, 0.0, 3.0]])
    assert metrics.mae(pred, true) == pytest.approx(2.0 / 3.0)
    assert metrics.rmse(pred, true) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_pearson_and_spearman_bounds_and_perfect():
    pred, true = _fixture()
    assert metrics.pearson_corr(true, true) == pytest.approx(1.0)
    assert metrics.spearman_corr(true, true) == pytest.approx(1.0)
    r = metrics.pearson_corr(pred, true)
    assert 0.3 < r < 1.0                                       # correlated but imperfect


def test_pearson_handles_zero_vector():
    true = np.random.default_rng(1).standard_normal((3, _G))
    zero = np.zeros((3, _G))
    assert metrics.pearson_corr(zero, true) == 0.0            # constant row -> 0, the worst score
    assert metrics.spearman_corr(zero, true) == 0.0


def test_systema_pert_specific_removes_treatment_effect():
    pred, true = _fixture()
    train_mean = true.mean(0)
    # a predictor that only emits the treatment mean has ~0 perturbation-specific correlation
    only_mean = np.broadcast_to(train_mean, true.shape)
    assert abs(metrics.systema_pert_specific_delta(only_mean, true, train_mean)) < 1e-6
    assert metrics.systema_pert_specific_delta(true, true, train_mean) == pytest.approx(1.0)


def test_centroid_accuracy_perfect_and_shuffled():
    _, true = _fixture()
    assert metrics.centroid_accuracy(true, true) == pytest.approx(1.0)   # own centroid closest
    shuffled = true[np.roll(np.arange(_N), 1)]
    assert metrics.centroid_accuracy(shuffled, true) < 0.5


def test_topk_recall_and_sign_accuracy():
    _, true = _fixture()
    assert metrics.topk_recall(true, true, k=5) == pytest.approx(1.0)
    assert metrics.sign_accuracy(true, true, top_n=10) == pytest.approx(1.0)
    assert metrics.sign_accuracy(-true, true, top_n=10) == pytest.approx(0.0)


def test_topk_recall_selects_strongest_by_magnitude():
    true = np.zeros((1, 10))
    true[0, [0, 1, 2, 5]] = [5.0, 4.0, 3.0, -6.0]      # strongest-|mag| top-3 = {5, 0, 1}
    pred = np.zeros((1, 10))
    pred[0, [5, 0, 7]] = [10.0, 9.0, 8.0]              # predicted top-3 = {5, 0, 7}
    assert metrics.topk_recall(pred, true, k=3) == pytest.approx(2 / 3)  # overlap {5, 0}


def test_sign_accuracy_scores_signs_of_strongest_only():
    true = np.array([[-5.0, 4.0, 3.0, 0.1, 0.05]])     # top-3 by |mag| = idx 0, 1, 2 (signs -, +, +)
    pred = np.array([[-1.0, -1.0, 1.0, 9.0, 9.0]])     # signs at those idx: -, -, + -> 2/3 correct
    assert metrics.sign_accuracy(pred, true, top_n=3) == pytest.approx(2 / 3)


def test_centroid_accuracy_penalizes_degenerate_predictor():
    _, true = _fixture()
    zero = np.zeros_like(true)
    assert metrics.centroid_accuracy(zero, true) == 0.0        # zero predictor is worst, not a spurious 1.0
    assert metrics_ref.centroid_accuracy(zero, true) == 0.0    # both impls agree on the degenerate case


def test_program_cosine():
    _, z = _fixture()
    assert metrics.program_cosine(z, z) == pytest.approx(1.0)
    assert metrics.program_cosine(-z, z) == pytest.approx(-1.0)


def test_signed_de_metrics_perfect_and_ranges():
    rng = np.random.default_rng(2)
    labels_up = (rng.random((_N, _G)) > 0.7).astype(float)
    labels_dn = (rng.random((_N, _G)) > 0.7).astype(float)
    labels = np.stack([labels_up, labels_dn], axis=-1)
    out = metrics.signed_de_metrics(labels.copy(), labels)     # perfect probs == labels
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["up"]["auprc"] == pytest.approx(1.0)
    for cls in ("up", "down"):
        for key in ("precision", "recall", "f1"):
            assert 0.0 <= out[cls][key] <= 1.0


def test_signed_de_accepts_dict_and_torch():
    up = torch.rand(4, _G)
    dn = torch.rand(4, _G)
    lab = {"up": (up > 0.5).float(), "down": (dn > 0.5).float()}
    out = metrics.signed_de_metrics({"up": up, "down": dn}, lab)
    assert set(out) == {"up", "down", "macro_f1"}


def test_signed_de_metrics_imperfect_probs_discriminate():
    # up: labels [1,1,0,0], probs give one FN (idx1) + one FP (idx2) -> P = R = 0.5
    up_p, up_l = np.array([0.9, 0.2, 0.8, 0.1]), np.array([1, 1, 0, 0])
    # down: labels [1,1,1,0], all predicted positive -> precision 0.75, recall 1.0 (P != R)
    dn_p, dn_l = np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 1, 1, 0])
    out = metrics.signed_de_metrics({"up": up_p, "down": dn_p}, {"up": up_l, "down": dn_l})
    assert out["up"]["precision"] == pytest.approx(0.5)
    assert out["up"]["recall"] == pytest.approx(0.5)
    assert out["up"]["auprc"] < 1.0                            # 0.8/label-0 outranks 0.2/label-1
    assert out["down"]["precision"] == pytest.approx(0.75)
    assert out["down"]["recall"] == pytest.approx(1.0)
    assert 0.0 < out["macro_f1"] < 1.0


def test_two_implementations_agree_on_fixture():
    pred, true = _fixture(7)
    train_mean = true.mean(0)
    for name, args in (
        ("mae", (pred, true)),
        ("rmse", (pred, true)),
        ("pearson_corr", (pred, true)),
        ("spearman_corr", (pred, true)),
        ("program_cosine", (pred, true)),
        ("centroid_accuracy", (pred, true)),
    ):
        a = getattr(metrics, name)(*args)
        b = getattr(metrics_ref, name)(*args)
        assert a == pytest.approx(b, abs=1e-9), f"{name}: {a} vs {b}"
    a = metrics.systema_pert_specific_delta(pred, true, train_mean)
    b = metrics_ref.systema_pert_specific_delta(pred, true, train_mean)
    assert a == pytest.approx(b, abs=1e-9)


def test_two_implementations_agree_on_degenerate_rows():
    rng = np.random.default_rng(9)
    pred = rng.standard_normal((5, _G))
    true = rng.standard_normal((5, _G))
    pred[0] = 0.0                                              # zero-norm prediction row
    true[1] = 3.0                                             # constant (zero-variance) truth row
    train_mean = true.mean(0)
    for name in ("mae", "rmse", "pearson_corr", "spearman_corr", "program_cosine", "centroid_accuracy"):
        a = getattr(metrics, name)(pred, true)
        b = getattr(metrics_ref, name)(pred, true)
        assert a == pytest.approx(b, abs=1e-9), f"{name}: {a} vs {b}"
    a = metrics.systema_pert_specific_delta(pred, true, train_mean)
    b = metrics_ref.systema_pert_specific_delta(pred, true, train_mean)
    assert a == pytest.approx(b, abs=1e-9)


def test_two_implementations_agree_on_non_finite_rows():
    rng = np.random.default_rng(13)
    pred = rng.standard_normal((5, _G))
    true = rng.standard_normal((5, _G))
    pred[0, 0] = np.inf
    pred[1, 0] = np.nan                                        # corrupted prediction rows
    train_mean = true.mean(0)
    for name in ("pearson_corr", "spearman_corr", "program_cosine", "centroid_accuracy"):
        a = getattr(metrics, name)(pred, true)
        b = getattr(metrics_ref, name)(pred, true)
        assert np.isfinite(a) and a == pytest.approx(b, abs=1e-9), f"{name}: {a} vs {b}"
    a = metrics.systema_pert_specific_delta(pred, true, train_mean)
    b = metrics_ref.systema_pert_specific_delta(pred, true, train_mean)
    assert np.isfinite(a) and a == pytest.approx(b, abs=1e-9)


def test_centroid_accuracy_agrees_when_true_has_non_finite():
    true = np.eye(3)
    true[2, 0] = np.inf                                        # a single non-finite in the bank
    pred = np.eye(3)
    a, b = metrics.centroid_accuracy(pred, true), metrics_ref.centroid_accuracy(pred, true)
    assert a == pytest.approx(b)                               # no whole-fold collapse; both sanitise the bank
    assert a > 0.0                                             # the two finite rows still resolve


def test_two_impls_agree_on_high_dimensional_constant_row():
    rng = np.random.default_rng(21)
    pred = np.full((1, 2000), 0.1)                             # std underflows to ~1e-16 (not exactly 0)
    true = rng.standard_normal((1, 2000))
    assert metrics.pearson_corr(pred, true) == 0.0 == metrics_ref.pearson_corr(pred, true)
    assert metrics.spearman_corr(pred, true) == 0.0 == metrics_ref.spearman_corr(pred, true)


def test_centroid_accuracy_tiny_norm_wrong_direction_is_miss():
    u0 = np.array([1.0, 0.0])
    v = np.array([0.5, np.sqrt(0.75)])                         # cos(v, u0) = 0.5
    true = np.stack([u0, v])
    pred = np.stack([1e-13 * v, u0])                           # row 0: near-zero, points at the WRONG centroid
    a, b = metrics.centroid_accuracy(pred, true), metrics_ref.centroid_accuracy(pred, true)
    assert a == pytest.approx(b)                               # consistent normalisation (no 1e-12 floor)
    assert a == pytest.approx(0.0)                             # not a spurious perfect hit


def test_pearson_is_scale_robust_no_underflow_collapse():
    rng = np.random.default_rng(22)
    base_p = rng.standard_normal((3, 60))
    base_t = 0.7 * base_p + 0.5 * rng.standard_normal((3, 60))
    ref = metrics.pearson_corr(base_p, base_t)
    for scale in (1.0, 1e-160):                                # correlation is scale-invariant
        a = metrics.pearson_corr(base_p * scale, base_t * scale)
        b = metrics_ref.pearson_corr(base_p * scale, base_t * scale)
        assert abs(a - ref) < 1e-3 and abs(b - ref) < 1e-3    # neither collapses to a spurious 0.0
        assert abs(a - b) < 1e-3


def test_topk_and_sign_ignore_degenerate_prediction_rows():
    rng = np.random.default_rng(23)
    true = rng.standard_normal((2, 30))
    nan_pred = true.copy()
    nan_pred[0] = np.nan                                       # a diverged row must not earn chance recall
    assert metrics.topk_recall(nan_pred, true, k=5) == pytest.approx(0.5 * metrics.topk_recall(true, true, k=5))
    assert metrics.topk_recall(np.zeros((2, 30)), true, k=5) == 0.0
    assert metrics.sign_accuracy(np.zeros((2, 30)), true, top_n=5) == 0.0


def test_independent_control_metric_forwards_kwargs_to_primary_endpoint():
    rng = np.random.default_rng(24)
    pred, true = rng.standard_normal((4, 10)), rng.standard_normal((4, 10))
    ctrl_a, ctrl_b = rng.standard_normal((4, 10)), rng.standard_normal((4, 10))
    val = cr.independent_control_metric(pred, true, ctrl_a, ctrl_b,
                                        metric=metrics.systema_pert_specific_delta, train_mean=true.mean(0))
    assert np.isfinite(val)                                    # 3-arg primary endpoint composes, no TypeError


def test_row_shuffle_permutes_within_each_row():
    true = np.arange(12).reshape(3, 4).astype(float)
    shuffled = mq.row_shuffle(true, np.random.default_rng(1))
    assert np.array_equal(np.sort(shuffled, axis=1), np.sort(true, axis=1))  # marginal preserved per row
    assert not np.array_equal(shuffled, true)


def test_g2mq_gate_orders_controls():
    pred, true = _fixture(3)
    rng = np.random.default_rng(3)
    train_mean = true.mean(0)
    fn = lambda p, t: metrics.systema_pert_specific_delta(p, t, train_mean)
    neg = {
        "zero": (mq.zero_prediction(true), true),
        "perturbed_mean": (mq.perturbed_mean_prediction(true), true),
        "label_perm": (mq.label_permutation(true, rng), true),
        "row_shuffle": (mq.row_shuffle(true, rng), true),
    }
    pos = {
        "guide_split_half": (mq.guide_split_half(true, rng, noise=0.3), true),
        "oracle": (mq.oracle_prediction(true), true),
    }
    res = mq.qualify_metric(fn, neg, pos)
    assert res["passed"] and res["ordering_correct"]
    assert res["dynamic_range"] > 0
    assert res["neg_scores"]["zero"] < res["pos_scores"]["oracle"]


def test_g2mq_gate_fails_when_ordering_violated():
    res = mq.qualify_metric(lambda p, t: 0.0, neg_controls={"a": 0.9}, pos_refs={"b": 0.1})
    assert not res["passed"]
    assert res["dynamic_range"] < 0


def test_label_permutation_is_a_derangement():
    true = np.arange(2 * 5).reshape(2, 5).astype(float)      # small fold where plain perm often == identity
    for s in range(50):
        perm = mq.label_permutation(true, np.random.default_rng(s))
        assert not np.array_equal(perm, true)                # no row keeps its own target identity (N1 null)


def test_null_control_predictor_is_neutral_under_independent_control():
    rng = np.random.default_rng(11)
    true = rng.standard_normal((_N, _G))
    ctrl_a = rng.standard_normal((_N, _G))
    ctrl_b = rng.standard_normal((_N, _G))
    null = cr.null_control_predictor(ctrl_a)                   # predicts its own control -> zero delta
    assert abs(cr.independent_control_metric(null, true, ctrl_a, ctrl_b)) < 1e-6


def test_shared_control_manufactures_spurious_correlation():
    rng = np.random.default_rng(12)
    ctrl_true = rng.standard_normal((_N, _G))
    signal_true = rng.standard_normal((_N, _G))               # what truth's perturbation actually is
    signal_pred = rng.standard_normal((_N, _G))               # prediction signal, independent of truth
    true = signal_true + ctrl_true
    pred = signal_pred + ctrl_true
    ctrl_a = ctrl_true + rng.standard_normal((_N, _G))        # independent noisy estimates
    ctrl_b = ctrl_true + rng.standard_normal((_N, _G))
    ctrl_shared = ctrl_true + 3.0 * rng.standard_normal((_N, _G))  # one shared, noisier estimate
    indep = cr.independent_control_metric(pred, true, ctrl_a, ctrl_b)
    shared = cr.shared_control_diagnostic(pred, true, ctrl_shared)
    assert abs(indep) < 0.25                                  # corrected: uncorrelated signals stay ~0
    assert shared > indep + 0.2                               # shared control injects spurious positive corr


def test_output_schema_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    ri = np.arange(_N)
    dz = rng.standard_normal((_N, _K)).astype(np.float32)
    dx = rng.standard_normal((_N, _G)).astype(np.float32)
    sigma = np.abs(rng.standard_normal((_N, _K))).astype(np.float32)
    path = write_predictions(ri, dz, dx, sigma, "egipg", "challenge", 0, root=tmp_path)
    assert path == prediction_path("egipg", "challenge", 0, root=tmp_path)
    got = read_predictions(path)
    assert np.array_equal(got["row_index"], ri)
    assert np.allclose(got["delta_z"], dz) and np.allclose(got["delta_x"], dx)
    assert np.allclose(got["sigma"], sigma)


def test_output_schema_defaults_sigma_to_zero(tmp_path):
    ri = np.arange(3)
    dz = np.ones((3, _K), dtype=np.float32)
    dx = np.ones((3, _G), dtype=np.float32)
    path = write_predictions(ri, dz, dx, None, "zero", "val", 1, root=tmp_path)
    assert np.count_nonzero(read_predictions(path)["sigma"]) == 0


def test_systema_scores_a_collapsed_predictor_zero_however_it_rounds():
    """CONSTRUCTED breaker for the primary endpoint.

    ``test_systema_pert_specific_removes_treatment_effect`` feeds a BIT-EXACT copy of train_mean, which is
    the one input the existing degeneracy guard already handles — no real pipeline ever produces it. In the
    live pipeline the perturbed-mean baseline built its mean in float64 while the scorer subtracted a
    float32 mean of the SAME array, leaving a relative 2.4e-06 residue; because Pearson is SCALE-invariant
    it read only that residue's DIRECTION and scored +0.0129 on the real val fold — above the 0.01 noise
    band, and above three genuine bars.

    Collapse-to-the-training-mean is the expected failure mode in a near-null-signal regime, so a collapsed
    predictor must score 0 no matter which way its floating-point dust points, and must not depend on the
    residue's magnitude (it does not: 1e-12 and 1e-5 scored identically)."""
    pred, true = _fixture()
    train_mean = true.mean(0)
    rng = np.random.default_rng(0)
    for eps in (1e-12, 1e-8, 1e-5):
        for s in range(5):
            r = rng.standard_normal(train_mean.shape)
            r /= np.linalg.norm(r)
            collapsed = np.broadcast_to(train_mean + eps * np.linalg.norm(train_mean) * r, true.shape)
            got = metrics.systema_pert_specific_delta(collapsed, true, train_mean)
            assert abs(got) < 1e-9, f"collapsed predictor scored {got:+.4f} (eps={eps})"


def test_systema_still_scores_a_genuine_predictor_unchanged():
    """The other half: the collapse guard must not swallow real signal. The nearest genuine predictor on the
    real fold sits ~0.46 x ||train_mean|| away, four orders above the 1e-4 collapse threshold, so no real
    result may move."""
    pred, true = _fixture()
    train_mean = true.mean(0)
    assert metrics.systema_pert_specific_delta(true, true, train_mean) == pytest.approx(1.0)
    # a predictor a realistic distance from the mean keeps its exact score
    near = np.broadcast_to(train_mean, true.shape) + 0.4 * np.linalg.norm(train_mean) * (true - train_mean)
    direct = metrics._rowwise_pearson(near - train_mean, true - train_mean).mean()
    assert metrics.systema_pert_specific_delta(near, true, train_mean) == pytest.approx(direct)


def test_both_systema_impls_agree_on_a_collapsed_predictor():
    """The cross-check suite is the project's second opinion on the primary endpoint, and its convention is
    that both implementations agree on the DEGENERATE cases too (see the centroid zero-predictor check).
    Pin the collapse case here so the reference cannot silently keep the old scale-invariant reading."""
    pred, true = _fixture()
    train_mean = true.mean(0)
    r = np.random.default_rng(3).standard_normal(train_mean.shape)
    r /= np.linalg.norm(r)
    collapsed = np.broadcast_to(train_mean + 1e-9 * np.linalg.norm(train_mean) * r, true.shape)
    a = metrics.systema_pert_specific_delta(collapsed, true, train_mean)
    b = metrics_ref.systema_pert_specific_delta(collapsed, true, train_mean)
    assert a == pytest.approx(0.0, abs=1e-9)
    assert b == pytest.approx(a, abs=1e-9), f"impls disagree on a collapsed predictor: {a:+.4f} vs {b:+.4f}"
