"""feat-005 basis-study metric tests — tiny, synthetic, no marts.

The correctness-critical function is ``matched_stability``: a basis is identified only up to
permutation and sign, so these tests construct inputs that DEFEAT the naive alternatives
(elementwise correlation, greedy nearest-neighbour matching) rather than merely exercising the
happy path. Dead components are the degenerate case that must read as UNDECIDABLE, never as a pass.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tcell_pipeline.programs.basis_study import (
    fit_vae_basis,
    matched_stability,
    multiplicity_adjust,
    paired_recon_contrast,
    recon_metrics,
    sparsity_metrics,
)
from tcell_pipeline.programs.program_basis import fit_program_basis


def _rand_basis(G=40, K=6, seed=0):
    return np.random.default_rng(seed).standard_normal((G, K)).astype(np.float32)


# ---- matched_stability: identifiability up to permutation and sign ----

def test_identical_bases_are_perfectly_stable():
    B = _rand_basis()
    assert matched_stability(B, B)["mean_abs_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_stability_is_invariant_to_permutation_and_sign_flip():
    """The claim the whole metric rests on. A naive elementwise correlation scores this far below 1."""
    B = _rand_basis(K=6)
    perm = np.array([3, 0, 5, 1, 4, 2])
    signs = np.array([1, -1, -1, 1, -1, 1], dtype=np.float32)
    B2 = B[:, perm] * signs
    assert matched_stability(B, B2)["mean_abs_cosine"] == pytest.approx(1.0, abs=1e-6)


def test_matching_is_a_true_assignment_not_greedy():
    """B1 has two near-identical components; B2 has only one of them plus an unrelated one.

    Greedy nearest-neighbour lets BOTH B1 components claim B2's column 0, double-counting a 1.0 and
    reporting perfect stability (measured: 1.0000, columns used [0,0,2,3]). A true assignment must
    spend each component once, forcing the duplicate onto the unrelated column (measured: 0.7761).
    """
    rng = np.random.default_rng(1)
    G = 60
    v, x, y, w = (rng.standard_normal(G) for _ in range(4))
    B1 = np.stack([v, v + 1e-3 * rng.standard_normal(G), x, y], 1).astype(np.float32)
    B2 = np.stack([v, w, x, y], 1).astype(np.float32)
    assert matched_stability(B1, B2)["mean_abs_cosine"] < 0.9, "greedy double-matching reports 1.0 here"


def test_unrelated_bases_are_not_reported_as_stable():
    """Guards a matcher that always returns 1.0. Random G=200 columns are near-orthogonal."""
    out = matched_stability(_rand_basis(G=200, K=8, seed=3), _rand_basis(G=200, K=8, seed=4))
    assert out["mean_abs_cosine"] < 0.4


def test_dead_component_is_counted_and_never_nans():
    """An all-zero column has undefined cosine (0/0). It must not poison the number silently."""
    B1 = _rand_basis(K=5, seed=5)
    B2 = B1.copy()
    B2[:, 2] = 0.0
    out = matched_stability(B1, B2)
    assert out["n_dead"] == 1
    assert np.isfinite(out["mean_abs_cosine"])
    assert out["mean_abs_cosine"] < 1.0


def test_all_dead_basis_is_undecidable_not_zero():
    """Nothing to match => None (unknown), never a number that reads as a real low-stability result."""
    B1 = _rand_basis(K=4, seed=6)
    out = matched_stability(B1, np.zeros_like(B1))
    assert out["mean_abs_cosine"] is None
    assert out["n_dead"] == 4


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        matched_stability(_rand_basis(G=40, K=4), _rand_basis(G=40, K=6))
    with pytest.raises(ValueError):
        matched_stability(_rand_basis(G=40, K=4), _rand_basis(G=30, K=4))


# ---- reconstruction ----

def test_zero_reconstruction_exactly_equals_the_zero_baseline():
    """Pins the baseline's meaning: predicting zero must score exactly the baseline, ratio 1, explained 0."""
    Zc = np.random.default_rng(7).standard_normal((20, 15)).astype(np.float32)
    Zc -= Zc.mean(0)
    A = np.zeros((20, 3), dtype=np.float32)
    B = _rand_basis(G=15, K=3, seed=8)
    m = recon_metrics(Zc, A, B)
    assert m["recon_mae"] == pytest.approx(m["zero_baseline_mae"])
    assert m["explained_frac"] == pytest.approx(0.0)


def test_recon_mae_is_mean_absolute_residual_against_the_given_target():
    Zc = np.array([[1.0, -3.0], [2.0, 0.0]], dtype=np.float32)
    A = np.array([[1.0], [1.0]], dtype=np.float32)
    B = np.array([[1.0], [-1.0]], dtype=np.float32)  # A @ B.T = [[1,-1],[1,-1]]
    m = recon_metrics(Zc, A, B)
    assert m["recon_mae"] == pytest.approx((0 + 2 + 1 + 1) / 4)
    assert m["zero_baseline_mae"] == pytest.approx((1 + 3 + 2 + 0) / 4)


def test_all_zero_target_explains_nothing_decidable():
    """A target with no signal has nothing to explain: 1 - 0/0. Must be None, not 0.0 or 1.0."""
    Zc = np.zeros((5, 4), dtype=np.float32)
    m = recon_metrics(Zc, np.zeros((5, 2), dtype=np.float32), _rand_basis(G=4, K=2, seed=20))
    assert m["explained_frac"] is None
    assert m["zero_baseline_mae"] == 0.0


def test_per_row_mae_is_returned_and_averages_to_the_scalar():
    """The paired cell-vs-cell contrast needs per-row residuals. Without persisting them, deciding
    the statistic after the sweep would force a full re-fit of every cell."""
    Zc = np.random.default_rng(18).standard_normal((17, 9)).astype(np.float32)
    A = np.random.default_rng(19).standard_normal((17, 3)).astype(np.float32)
    B = _rand_basis(G=9, K=3, seed=21)
    m = recon_metrics(Zc, A, B)
    assert m["row_mae"].shape == (17,)
    assert m["row_mae"].mean() == pytest.approx(m["recon_mae"])
    assert m["row_mae"][0] == pytest.approx(np.abs(Zc[0] - A[0] @ B.T).mean())


def test_per_row_mae_spans_chunk_boundaries():
    """Chunked accumulation must not drop or duplicate rows when N exceeds the chunk size."""
    from tcell_pipeline.programs import basis_study

    n = basis_study._ROW_CHUNK * 2 + 5
    Zc = np.random.default_rng(22).standard_normal((n, 4)).astype(np.float32)
    m = recon_metrics(Zc, np.zeros((n, 2), dtype=np.float32), _rand_basis(G=4, K=2, seed=23))
    assert m["row_mae"].shape == (n,)
    assert m["row_mae"][-1] == pytest.approx(np.abs(Zc[-1]).mean())


def test_perfect_reconstruction_explains_everything():
    Zc = np.random.default_rng(9).standard_normal((12, 8)).astype(np.float32)
    B = _rand_basis(G=8, K=8, seed=10)
    A = (Zc @ np.linalg.pinv(B).T).astype(np.float32)  # exact: B is square and full rank
    m = recon_metrics(Zc, A, B)
    assert m["explained_frac"] == pytest.approx(1.0, abs=1e-4)


# ---- sparsity ----

def test_sparsity_counts_exact_zeros_and_dead_programs():
    B = np.array([[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    m = sparsity_metrics(B)
    assert m["zero_frac"] == pytest.approx(6 / 9)
    assert m["n_dead"] == 1  # only column 2 is entirely zero


def test_near_zero_is_not_counted_as_zero():
    """Exact-zero is the recorded definition (22.7% for the frozen fit); a tolerance would inflate it.

    One exact zero, one merely tiny value: a tolerance-based count would report 1.0 here.
    """
    B = np.array([[1e-12, 0.0]], dtype=np.float32)
    assert sparsity_metrics(B)["zero_frac"] == pytest.approx(0.5)


# ---- frozen-basis safety: the study must be incapable of overwriting the production basis ----

def test_study_never_references_the_frozen_basis_write_path():
    """The project-destroying failure mode. gene_program_loadings.parquet is the coordinate system
    for every result in the repo; a stray save_program_basis() call in the sweep silently replaces
    it with an incompatible basis. This fires the moment such a call is added."""
    import ast
    import inspect

    from tcell_pipeline.programs import basis_study, run_basis_study

    forbidden = {"save_program_basis", "save_program_response", "PROGRAM_LOADINGS_PATH",
                 "PROGRAM_RESPONSE_PATH", "write_parquet_atomic"}
    for mod in (basis_study, run_basis_study):
        tree = ast.parse(inspect.getsource(mod))
        # AST, not text: these names appear in the modules' own docstrings explaining why they are
        # never called. A guard that fires on prose is one you learn to silence.
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not (used & forbidden), f"{mod.__name__} references {sorted(used & forbidden)}"


def test_a_timing_probe_cell_is_not_reused_as_a_complete_cell(tmp_path):
    """Presence is not freshness. A --no-stability probe writes the same JSON path as a full cell;
    without a completeness marker the sweep skips it and the table silently reports blank stability
    for a cell that was never resampled."""
    from tcell_pipeline.programs.run_basis_study import is_cached

    probe = {"method": "nmf", "K": 64, "has_stability": False}
    full = {"method": "nmf", "K": 64, "has_stability": True}
    p = tmp_path / "nmf_K64.json"

    p.write_text(json.dumps(probe))
    assert is_cached(p, want_stability=True) is False   # must re-run: the probe has no resamples
    assert is_cached(p, want_stability=False) is True

    p.write_text(json.dumps(full))
    assert is_cached(p, want_stability=True) is True
    assert is_cached(p, want_stability=False) is True   # a full cell is a superset of a probe

    assert is_cached(tmp_path / "absent.json", want_stability=False) is False


def test_reference_cell_is_excluded_from_its_own_contrast_family(tmp_path):
    """Self-contrast is degenerate (all differences exactly 0) AND it would consume a slot in the
    Bonferroni/Holm family, weakening every real comparison for a test that says nothing."""
    from tcell_pipeline.programs.run_basis_study import build_contrasts

    rng = np.random.default_rng(40)
    ref = rng.random(300) + 1.0
    np.save(tmp_path / "sparse_pca_K128.npy", ref.astype(np.float32))
    np.save(tmp_path / "svd_K128.npy", (ref - 0.05).astype(np.float32))
    np.save(tmp_path / "nmf_K64.npy", (ref + 0.05).astype(np.float32))

    df = build_contrasts(tmp_path)
    assert len(df) == 2 and "sparse_pca" not in set(df["method"])
    assert df["family_size"].unique().tolist() == [2]


def test_capped_methods_get_a_convergence_budget_only_at_the_frozen_k():
    """Decision 2026-07-20: capped at 100 everywhere is undecidable everywhere; 500 everywhere costs
    ~6h for nmf. Buy one decidable point per capped method at the K matching the frozen basis.

    Both nmf (100/100 at K=64) and fastica (100/100 at K=128) were MEASURED hitting the cap, so both
    are re-budgeted. sparse_pca converges in 2 iters and svd has no cap, so neither needs it.
    """
    from tcell_pipeline.programs.run_basis_study import MAX_ITER, cell_max_iter

    assert cell_max_iter("nmf", 128) == 500
    assert cell_max_iter("fastica", 128) == 500
    assert cell_max_iter("nmf", 64) == MAX_ITER
    assert cell_max_iter("fastica", 512) == MAX_ITER
    assert cell_max_iter("sparse_pca", 128) == MAX_ITER  # converges at 2 iters; nothing to buy
    assert cell_max_iter("svd", 128) == MAX_ITER


def test_sweep_runs_cheapest_cells_first():
    """Under a total budget the expensive tail is what gets dropped, so cheap cells must land first.

    Ordering uses the MEASURED K-exponents, which are not all linear: fastica's cost is ~90% a
    K-independent whitening SVD (564s -> 596s for 2x K), so fastica K=512 is genuinely cheaper than
    sparse_pca K=256 (171s -> 361s for 2x K, linear). Assuming linear-in-K for every method would
    order these backwards and needlessly strand the fastica cells at the end of the budget.
    """
    from tcell_pipeline.programs.run_basis_study import sweep_order

    order = sweep_order([("fastica", 512), ("svd", 64), ("sparse_pca", 256), ("svd", 512)])
    assert order[0] == ("svd", 64)
    assert order.index(("svd", 512)) < order.index(("sparse_pca", 256))
    assert order.index(("fastica", 512)) < order.index(("sparse_pca", 256))


def test_cell_too_expensive_for_resamples_degrades_instead_of_returning_blank():
    """A cell whose 4 fits blow the cap would burn the whole 90 min and produce nothing. One fit
    still fits, so run it and mark stability not-computed — recon+sparsity beats an empty row."""
    from tcell_pipeline.programs.run_basis_study import wants_stability

    assert wants_stability("svd", 64, cap_seconds=5400) is True
    assert wants_stability("sparse_pca", 128, cap_seconds=5400) is True
    # fastica K=512 IS affordable once the measured flat K-exponent is used (~11 min/fit, not 75).
    # A linear-in-K assumption would strand it on fit+score only for no reason.
    assert wants_stability("fastica", 512, cap_seconds=5400) is True
    # A genuinely unaffordable cell still degrades rather than burning the whole cap for nothing.
    assert wants_stability("sparse_pca", 512, cap_seconds=600) is False


def test_unmeasured_cell_is_recorded_with_a_reason_not_left_blank(tmp_path):
    """A cell that timed out must be an explicit value in the table. A missing row reads as 'not
    run'; a blank metric reads as 'ran and found nothing' — both misreport a bounded sweep."""
    from tcell_pipeline.programs.run_basis_study import write_not_measured

    p = tmp_path / "fastica_K512.json"
    write_not_measured(p, "fastica", 512, "timeout", 5400)
    c = json.loads(p.read_text())
    assert c["not_measured"] == "timeout"
    assert c["has_stability"] is False
    assert c["recon"]["recon_mae"] is None and c["sparsity"]["zero_frac"] is None


def test_an_unmeasured_stub_is_not_reused_as_a_finished_cell(tmp_path):
    """Raising the cap on a later run must retry the cell, not skip it as already done."""
    from tcell_pipeline.programs.run_basis_study import is_cached, write_not_measured

    p = tmp_path / "fastica_K512.json"
    write_not_measured(p, "fastica", 512, "timeout", 5400)
    assert is_cached(p, want_stability=True) is False
    assert is_cached(p, want_stability=False) is False


def test_stability_merges_into_an_existing_cell_without_disturbing_its_fit():
    """Backfilling stability must not re-run the 96-min K=512 fit or silently restate its numbers."""
    from tcell_pipeline.programs.run_basis_study import merge_stability

    cell = {"method": "sparse_pca", "K": 512, "fit_seconds": 5780.0,
            "convergence": {"n_iter": 6, "converged": True},
            "recon": {"recon_mae": 0.649989, "explained_frac": 0.204818},
            "sparsity": {"zero_frac": 0.511257, "n_dead": 0}, "has_stability": False}
    out = merge_stability(dict(cell), {"mean_abs_cosine": 0.21}, {"recon_mae": 0.66})

    assert out["has_stability"] is True
    assert out["stability"]["mean_abs_cosine"] == 0.21 and out["heldout"]["recon_mae"] == 0.66
    for k in ("fit_seconds", "convergence", "recon", "sparsity"):
        assert out[k] == cell[k], f"{k} was disturbed by the backfill"


def test_stability_cannot_be_merged_into_an_unmeasured_stub():
    """A stub has no fit at all. Attaching stability would fabricate a half-real cell that reads as
    measured — worse than the honest not_measured it replaced."""
    from tcell_pipeline.programs.run_basis_study import merge_stability

    with pytest.raises(ValueError):
        merge_stability({"method": "sparse_pca", "K": 512, "not_measured": "timeout"},
                        {"mean_abs_cosine": 0.21}, {"recon_mae": 0.66})


def test_study_output_dir_is_confined_to_the_results_tree():
    from tcell_pipeline import config
    from tcell_pipeline.programs.run_basis_study import OUT_DIR

    assert OUT_DIR.resolve() == (config.DATA_DIR / "results" / "basis_study").resolve()
    assert config.INTERMEDIATE_ROOT.resolve() not in OUT_DIR.resolve().parents


# ---- convergence evidence (a capped fit is UNDECIDABLE, not evidence the method is worse) ----

@pytest.mark.parametrize("method", ["sparse_pca", "nmf", "fastica"])
def test_iterative_fits_report_hitting_the_iteration_cap(method):
    """max_iter=1 forces the cap. _factor silences ConvergenceWarning internally, so an outer
    warnings catcher sees nothing — without an explicit channel this is silently unobservable."""
    Z = np.abs(np.random.default_rng(15).standard_normal((60, 25))).astype(np.float32) + 0.1
    info = {}
    fit_program_basis(Z, method=method, K=4, max_iter=1, info=info)
    assert info["converged"] is False
    assert info["n_iter"] == 1 and info["max_iter"] == 1


def test_converged_fit_reports_true():
    Z = np.abs(np.random.default_rng(16).standard_normal((60, 25))).astype(np.float32) + 0.1
    info = {}
    fit_program_basis(Z, method="nmf", K=4, max_iter=2000, info=info)
    assert info["converged"] is True and info["n_iter"] < 2000


def test_non_iterative_method_reports_convergence_as_unknown_not_true():
    """TruncatedSVD has no iteration cap. Claiming converged=True invents evidence it never produced."""
    Z = np.random.default_rng(17).standard_normal((60, 25)).astype(np.float32)
    info = {}
    fit_program_basis(Z, method="svd", K=4, info=info)
    assert info["converged"] is None and info["n_iter"] is None


# ---- paired contrast + multiplicity ----

def test_paired_contrast_detects_a_real_improvement():
    rng = np.random.default_rng(30)
    ref = rng.random(500) + 1.0
    cand = ref - 0.1 + 0.01 * rng.standard_normal(500)  # candidate reconstructs better
    out = paired_recon_contrast(ref, cand)
    assert out["mean_diff"] == pytest.approx(-0.1, abs=0.01)
    assert out["p_raw"] < 1e-10
    assert out["ci_low"] < out["mean_diff"] < out["ci_high"] and out["ci_high"] < 0


def test_identical_inputs_are_undecidable_not_significant():
    """AGENTS.md names this failure by name: zero variance was once reported as p=0.0 with
    'CI excludes zero' — the one condition proving the inputs carry NO information turned into the
    strongest possible evidence. Zero-variance differences must be None, never a p-value."""
    ref = np.random.default_rng(31).random(500) + 1.0
    out = paired_recon_contrast(ref, ref.copy())
    assert out["p_raw"] is None
    assert out["mean_diff"] == 0.0


def test_constant_nonzero_difference_is_also_undecidable():
    """A constant shift has zero variance too: t = d/0 = inf. An infinite t is not evidence."""
    ref = np.random.default_rng(32).random(500) + 1.0
    out = paired_recon_contrast(ref, ref + 0.05)
    assert out["p_raw"] is None
    assert out["mean_diff"] == pytest.approx(0.05)


def test_p_value_underflow_is_flagged_not_reported_as_exactly_zero():
    """At n=21,262 the real sweep underflows to p == 0.0 even for a 0.13%-of-baseline difference.
    An unflagged 0.0 reads as infinitely strong evidence for a practically negligible effect."""
    rng = np.random.default_rng(33)
    ref = rng.random(5000) + 1.0
    out = paired_recon_contrast(ref, ref - 0.1 + 1e-4 * rng.standard_normal(5000))
    assert out["p_raw"] == 0.0
    assert out["p_underflow"] is True


def test_ordinary_p_value_is_not_flagged_as_underflow():
    rng = np.random.default_rng(34)
    ref = rng.random(200) + 1.0
    out = paired_recon_contrast(ref, ref + 0.05 * rng.standard_normal(200))
    assert out["p_raw"] > 0.0 and out["p_underflow"] is False


def test_contrast_reports_effect_size_in_explained_fraction_points(tmp_path):
    """Significance is near-automatic at this n, so the conclusion must rest on effect size. The
    reported delta must equal the difference of the cells' own explained_frac values."""
    from tcell_pipeline.programs.run_basis_study import build_contrasts

    ref = np.full(400, 0.70)
    cand = np.full(400, 0.65) + 1e-3 * np.random.default_rng(35).standard_normal(400)
    np.save(tmp_path / "sparse_pca_K128.npy", ref.astype(np.float32))
    np.save(tmp_path / "svd_K256.npy", cand.astype(np.float32))

    df = build_contrasts(tmp_path, zero_baseline=0.8174)
    # explained_frac = 1 - recon/base, so the delta is -(mean_diff)/base
    assert df["delta_explained_frac"].iloc[0] == pytest.approx(
        (1 - cand.mean() / 0.8174) - (1 - 0.70 / 0.8174), abs=1e-6)


def test_bonferroni_and_holm_are_both_reported():
    ps = [0.001, 0.02, 0.04]
    adj = multiplicity_adjust(ps)
    assert adj["bonferroni"] == pytest.approx([0.003, 0.06, 0.12])
    # Holm: 3*0.001=0.003, 2*0.02=0.04, 1*0.04=0.04 (monotone non-decreasing)
    assert adj["holm"] == pytest.approx([0.003, 0.04, 0.04])


def test_holm_enforces_monotonicity():
    """Raw 2*0.03=0.06 exceeds the next step's 1*0.04, so Holm must carry 0.06 forward, not drop."""
    adj = multiplicity_adjust([0.001, 0.03, 0.04])
    assert adj["holm"] == pytest.approx([0.003, 0.06, 0.06])


def test_adjusted_p_values_are_capped_at_one():
    adj = multiplicity_adjust([0.5, 0.6])
    assert max(adj["bonferroni"]) <= 1.0 and max(adj["holm"]) <= 1.0


def test_undecidable_p_values_survive_adjustment_as_none():
    """A None must not be silently coerced to 1.0 (a pass) or 0.0 (a hit), and must not consume a
    slot in the correction — the family size is the number of tests actually run."""
    adj = multiplicity_adjust([0.01, None, 0.02])
    assert adj["bonferroni"][1] is None and adj["holm"][1] is None
    assert adj["bonferroni"][0] == pytest.approx(0.02)  # family size 2, not 3


# ---- shallow VAE ----

def test_vae_basis_has_basis_shape_and_is_finite():
    Zc = np.random.default_rng(11).standard_normal((64, 20)).astype(np.float32)
    Zc -= Zc.mean(0)
    B, A, info = fit_vae_basis(Zc, K=4, seed=0, epochs=2)
    assert B.shape == (20, 4) and A.shape == (64, 4)
    assert np.isfinite(B).all() and np.isfinite(A).all()
    assert info["epochs_run"] == 2 and np.isfinite(info["final_loss"])


def test_vae_basis_is_deterministic_for_a_fixed_seed():
    Zc = np.random.default_rng(12).standard_normal((64, 20)).astype(np.float32)
    B1, _, _ = fit_vae_basis(Zc, K=4, seed=3, epochs=2)
    B2, _, _ = fit_vae_basis(Zc, K=4, seed=3, epochs=2)
    assert np.array_equal(B1, B2)


def test_vae_seed_actually_reaches_the_fit():
    """Determinism alone is satisfied by ignoring ``seed`` entirely — torch.Generator() has a fixed
    default seed. Different seeds must give different bases, or the stability resamples are a sham."""
    Zc = np.random.default_rng(13).standard_normal((64, 20)).astype(np.float32)
    B1, _, _ = fit_vae_basis(Zc, K=4, seed=3, epochs=2)
    B2, _, _ = fit_vae_basis(Zc, K=4, seed=4, epochs=2)
    assert not np.array_equal(B1, B2)


def test_vae_survives_a_large_magnitude_target():
    """Constructed to defeat the fit: without the logvar clamp, exp(0.5*logvar) overflows and B
    comes back all-NaN (verified: `B finite=False` unclamped, True clamped)."""
    Zc = (np.random.default_rng(14).standard_normal((256, 30)) * 1e4).astype(np.float32)
    B, A, info = fit_vae_basis(Zc, K=4, seed=0, epochs=3, batch_size=64, lr=1e-1)
    assert np.isfinite(B).all() and np.isfinite(A).all()
