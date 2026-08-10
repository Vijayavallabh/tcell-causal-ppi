"""DE under prereg Amendment 2: all cells per (target, condition) vs pooled same-condition controls.

WHY THIS REPLACES THE REPLICATE-PSEUDOBULK PATH. Amendment 2 (2026-08-10) records that the original
rule -- two or more replicate pseudobulks per (target, condition) -- was never applied to the REFERENCE
dataset, whose supervised target is an artifact published with the source screen and computed over all
cells per (perturbation, condition) against matched controls. Holding replications to a stricter
standard than the reference discarded the best-powered candidate available (Replogle RPE1: 12 of 2,393
targets survived). This module implements the reference's own standard.

WHERE UNCERTAINTY COMES FROM. Without replicate pseudobulks there is no replicate-level t. Uncertainty
is therefore cell-level: a per-gene Welch t on log1p-CPM, target cells against the pooled control cells
of the SAME condition. Welch rather than Student because group sizes and variances differ by orders of
magnitude (a target may have 30 cells against 20,000 controls). This yields exactly the six layers the
pipeline consumes: log_fc, zscore (the t statistic), p_value, adj_p_value (BH), baseMean, lfcSE.

HONEST LIMITATION, recorded rather than buried. Treating cells as independent replicates overstates
precision: cells from one transduction share technical and biological structure, so p-values here are
anti-conservative relative to a replicate-level design. This is the standard trade in single-cell DE
and it is why the analysis leans on effect size and the paired cross-arm contrast rather than on these
per-gene p-values, which enter the pipeline only as features.

Computed in one sparse pass per group via an indicator matrix (sums and sums-of-squares), so it scales
to the 1.99M-cell genome-wide screen without densifying.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import stats

MIN_CELLS = 25


def log1p_cpm(X) -> sp.csr_matrix:
    """Row-normalise to counts-per-million then log1p, staying sparse throughout."""
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    X = sp.diags(1e6 / lib) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def _group_moments(X: sp.csr_matrix, codes: np.ndarray, n_groups: int):
    """Per-group n, mean and variance for every gene, in two sparse products."""
    ind = sp.csr_matrix((np.ones(codes.size, dtype=np.float64),
                         (codes, np.arange(codes.size))), shape=(n_groups, codes.size))
    n = np.asarray(ind.sum(axis=1)).ravel()
    s = np.asarray((ind @ X).todense())
    sq = np.asarray((ind @ X.multiply(X)).todense())
    n_safe = np.maximum(n, 1)[:, None]
    mean = s / n_safe
    # unbiased variance; groups of one get zero variance and are excluded upstream by MIN_CELLS
    var = np.maximum((sq - n_safe * mean ** 2) / np.maximum(n_safe - 1, 1), 0.0)
    return n, mean, var


def welch(mean_t, var_t, n_t, mean_c, var_c, n_c):
    """Per-gene Welch t. Returns (log_fc, t, p, se, df)."""
    se2_t, se2_c = var_t / max(n_t, 1), var_c / max(n_c, 1)
    se = np.sqrt(se2_t + se2_c)
    lfc = mean_t - mean_c
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, lfc / se, 0.0)
        num = (se2_t + se2_c) ** 2
        den = (se2_t ** 2 / max(n_t - 1, 1)) + (se2_c ** 2 / max(n_c - 1, 1))
        df = np.where(den > 0, num / den, 1.0)
    p = 2.0 * stats.t.sf(np.abs(t), np.maximum(df, 1.0))
    return lfc, t, p, se, df


def bh(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _self_check() -> None:
    """Synthetic checks, including the compositional artifact CPM normalisation creates.

    Note the gene count: 600, not 30. The first draft used 30 and the null genes all came out
    significant, which was NOT a coding error - it is real compositional bias. Elevating one gene
    six-fold inflates the library size of the target cells, so CPM deflates every other gene and they
    read as coordinately down-regulated. With 30 genes one gene is 3% of the library and the artifact
    dominates; with thousands, as in any real screen, it is negligible. The effect is tested for
    explicitly below so it stays a known property rather than a future surprise.
    """
    rng = np.random.default_rng(0)
    n_genes = 600
    ctrl = rng.poisson(50.0, size=(400, n_genes)).astype(np.float64)
    mu = np.full(n_genes, 50.0); mu[0] = 300.0
    tgt = rng.poisson(mu, size=(120, n_genes)).astype(np.float64)
    X = log1p_cpm(sp.csr_matrix(np.vstack([ctrl, tgt])))
    codes = np.concatenate([np.zeros(400, int), np.ones(120, int)])
    n, mean, var = _group_moments(X, codes, 2)
    assert n.tolist() == [400, 120]
    lfc, t, p, se, df = welch(mean[1], var[1], n[1], mean[0], var[0], n[0])
    assert lfc[0] > 1.0, f"planted effect not recovered: {lfc[0]}"
    assert p[0] < 1e-6, f"planted effect not significant: {p[0]}"
    assert abs(lfc[1]) < 0.3, f"unperturbed gene moved: {lfc[1]}"
    assert (p[1:] < 0.05).mean() < 0.25, f"null genes significant everywhere: {(p[1:] < 0.05).mean():.2f}"

    # WELCH vs STUDENT. An earlier version of this test inflated the control variance and checked
    # that |t| shrank -- which a pooled-variance (Student) implementation also passes, so it
    # discriminated nothing. Mutation testing caught that. The discriminating case is the one Welch
    # exists for: a SMALL group with LARGE variance against a LARGE group with SMALL variance.
    # Pooling averages the two variances and badly understates the standard error, inflating |t|;
    # Welch weights each by its own n. Analytic values below, so the assertion is exact.
    m_t = np.array([1.0]); v_t = np.array([4.0]); k_t = 5      # small n, large variance
    m_c = np.array([0.0]); v_c = np.array([0.04]); k_c = 500   # large n, small variance
    _, t_w, _, se_w, df_w = welch(m_t, v_t, k_t, m_c, v_c, k_c)
    se_pooled = np.sqrt(((v_t + v_c) / 2) * (1 / k_t + 1 / k_c))
    # Welch here gives 0.8945 against a pooled 0.6388, a ratio of 1.40; Student would give exactly
    # 1.00. The threshold sits between them. It was first set at 1.5, which failed on CORRECT code -
    # a reminder to derive a bound from the analytic values rather than pick a round number.
    assert se_w[0] > 1.25 * se_pooled[0], (
        f"standard error {se_w[0]:.4f} is close to the pooled {se_pooled[0]:.4f}: this is Student, "
        f"not Welch, and it understates uncertainty when the small group is the noisy one")
    assert df_w[0] < 10, f"Welch-Satterthwaite df {df_w[0]:.1f} should track the SMALL group, not n_c"

    # UNBIASED VARIANCE. At n in the hundreds, n vs n-1 differs by under a percent and no assertion
    # above would notice -- mutation testing caught that too. Assert it where it bites: n=5.
    tiny = np.array([[0.0], [2.0], [4.0], [6.0], [8.0]])
    _, _, v_tiny = _group_moments(sp.csr_matrix(tiny), np.zeros(5, int), 1)
    assert abs(v_tiny[0, 0] - 10.0) < 1e-9, (
        f"variance {v_tiny[0,0]:.4f} != 10.0 (unbiased, n-1); 8.0 would mean the biased n divisor")

    # COMPOSITIONAL BIAS, asserted so it is documented rather than discovered later: the same planted
    # effect in a 30-gene matrix drives the null genes significant purely through library size.
    small = 30
    c2 = rng.poisson(50.0, size=(400, small)).astype(np.float64)
    m2 = np.full(small, 50.0); m2[0] = 300.0
    t2 = rng.poisson(m2, size=(120, small)).astype(np.float64)
    X2 = log1p_cpm(sp.csr_matrix(np.vstack([c2, t2])))
    n2, mean2, var2 = _group_moments(X2, codes, 2)
    _, _, p2, _, _ = welch(mean2[1], var2[1], n2[1], mean2[0], var2[0], n2[0])
    assert (p2[1:] < 0.05).mean() > 0.5, "compositional artifact did not appear where it should"

    q = bh(np.array([0.001, 0.01, 0.03, 0.5]))
    assert np.all(np.diff(q) >= -1e-12) and np.all(q >= np.array([0.001, 0.01, 0.03, 0.5]))
    print("[de2] self-check OK: effect recovered, null stays null at realistic gene count, "
          "Welch honours unequal variance, compositional bias reproduced at low gene count, BH monotone")


if __name__ == "__main__":
    _self_check()
