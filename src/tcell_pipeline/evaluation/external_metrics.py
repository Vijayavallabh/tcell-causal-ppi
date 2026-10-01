"""A3: re-score the STORED predictions under the endpoints other perturbation papers actually report.

The paper's commensurability hedge says outside gains use different splits AND different metrics, so our
bound constrains this task under this protocol rather than adjudicating anyone's claim. The splits part
is irreducible. The metrics part is not: every arm's per-row predicted and true responses are already on
disk under ``data/results/*/predictions/``, so the same folds can be re-scored under someone else's
endpoint for the cost of arithmetic.

WHICH ENDPOINTS, VERIFIED RATHER THAN ASSUMED (checked 2026-08-16):

  ``pearson_delta``          TxPert's headline. Pearson between predicted and observed response over ALL
                             genes, per perturbation, macro-averaged. TxPert reports Pearson-delta across
                             cell lines precisely to stop a model scoring well on the generic stress
                             response. https://www.valencelabs.com/wp-content/uploads/2025/05/TxPert.pdf
  ``pearson_delta_top20``    GEARS reports Pearson on gene subsets, including the top-20 differentially
                             expressed genes of each perturbation.
  ``mse_top20``              GEARS' headline error: MSE restricted to each perturbation's top-20 DE genes,
                             chosen so a model cannot score well by predicting no effect at all.
                             https://www.biorxiv.org/content/10.1101/2022.07.12.499735v2.full
  ``edistance_scperturb``    scPerturb's E-distance, as scPerturb computes it: SQUARED euclidean pairwise
                             distances. See the warning on that function - in that form the statistic is
                             algebraically a mean difference, not a distributional one.
  ``energy_distance``        Szekely's energy distance with plain euclidean distances: zero if and only if
                             the two distributions match. This is the genuinely distributional endpoint,
                             and the one that speaks to the paper's "gene-space and full single-cell
                             distribution conclusions can still differ" hedge.

WHAT THE DISTRIBUTIONAL ENDPOINTS ARE COMPUTED OVER, stated because it is easy to overclaim. This
pipeline predicts one pseudobulk response per (target, condition); it does not predict per-cell
distributions. The distance here is therefore between the DISTRIBUTION OF PREDICTED RESPONSES and the
DISTRIBUTION OF OBSERVED RESPONSES across the held-out perturbations - not between cell populations,
which is what scPerturb's E-distance compares. It answers "does the model reproduce the spread and shape
of the response population" rather than "does it reproduce a single perturbation's cell cloud".

ORIENTATION. Correlations are better when larger, errors and distances when smaller. ``ORIENTATION``
carries the sign that turns each metric into "larger is better", so a paired contrast reads the same way
for every endpoint and a graph arm cannot look good merely because a metric runs backwards.
"""
from __future__ import annotations

import numpy as np

from tcell_pipeline.evaluation._arrays import to_numpy as _np
from tcell_pipeline.evaluation.metrics import _rowwise_pearson

TOP_DE_K = 20          # GEARS' subset size
E_SUBSAMPLE = 800      # rows drawn for the pairwise-distance endpoints (N^2 in the row count)

ORIENTATION = {
    "pearson_delta": +1,
    "pearson_delta_top20": +1,
    "mse_top20": -1,
    "edistance_scperturb": -1,
    "energy_distance": -1,
}


def top_de_index(true: np.ndarray, k: int = TOP_DE_K) -> np.ndarray:
    """Per-row column indices of the ``k`` most differentially expressed genes, ranked by |observed
    response|. Taken from the OBSERVED response, never the prediction: ranking on the prediction would
    let a model pick the genes it happens to be confident about and score itself on those."""
    k = min(int(k), true.shape[1])
    return np.argpartition(-np.abs(true), kth=k - 1, axis=1)[:, :k]


def _gather(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.take_along_axis(mat, idx, axis=1)


def pearson_delta(pred, true) -> float:
    """TxPert's Pearson-delta: per-perturbation correlation over all genes, macro-averaged."""
    return float(_rowwise_pearson(_np(pred), _np(true)).mean())


def pearson_delta_top_de(pred, true, k: int = TOP_DE_K) -> float:
    """GEARS' Pearson restricted to each perturbation's top-k observed DE genes."""
    p, t = _np(pred), _np(true)
    idx = top_de_index(t, k)
    return float(_rowwise_pearson(_gather(p, idx), _gather(t, idx)).mean())


def mse_top_de(pred, true, k: int = TOP_DE_K) -> float:
    """GEARS' headline: mean squared error over each perturbation's top-k observed DE genes."""
    p, t = _np(pred), _np(true)
    idx = top_de_index(t, k)
    d = _gather(p, idx) - _gather(t, idx)
    return float(np.mean(d * d))


def _subsample(n: int, m: int, seed: int) -> np.ndarray:
    if n <= m:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))


def edistance_scperturb(pred, true) -> float:
    """scPerturb's E-distance: ``2*mean||x-y||^2 - mean||x-x'||^2 - mean||y-y'||^2``.

    WARNING, and the reason this is reported separately from ``energy_distance``: with SQUARED euclidean
    distances that expression collapses algebraically to ``2*||mean(X) - mean(Y)||^2``. It is a
    difference of means wearing a distributional name, and it cannot distinguish two populations with the
    same mean and different spread. It is computed here for commensurability with what scPerturb reports,
    not as evidence about distributions - that is what ``energy_distance`` is for. Computed from the means
    directly, which is exact and skips the N^2 work the definition implies."""
    p, t = _np(pred), _np(true)
    diff = p.mean(0) - t.mean(0)
    return float(2.0 * np.dot(diff, diff))


def energy_distance(pred, true, *, n_sample: int = E_SUBSAMPLE, seed: int = 0) -> float:
    """Szekely's energy distance with PLAIN euclidean distances: zero if and only if the distributions
    agree, so unlike the squared form it does see spread and shape.

    Subsampled to ``n_sample`` rows per side because the estimator is quadratic in the row count; the draw
    is seeded, so the number is reproducible. Rows are the held-out perturbations' response vectors."""
    from scipy.spatial.distance import cdist
    p, t = _np(pred), _np(true)
    ip = _subsample(p.shape[0], n_sample, seed)
    it = _subsample(t.shape[0], n_sample, seed + 1)
    a, b = p[ip], t[it]
    cross = cdist(a, b).mean()
    within_a = cdist(a, a).sum() / (a.shape[0] * (a.shape[0] - 1)) if a.shape[0] > 1 else 0.0
    within_b = cdist(b, b).sum() / (b.shape[0] * (b.shape[0] - 1)) if b.shape[0] > 1 else 0.0
    return float(2.0 * cross - within_a - within_b)


def external_metric_suite(dx_hat, dx_true, *, k: int = TOP_DE_K, n_sample: int = E_SUBSAMPLE,
                          seed: int = 0) -> dict:
    """Every externally-reported endpoint in one pass over one arm's gene-space predictions."""
    return {
        "pearson_delta": pearson_delta(dx_hat, dx_true),
        "pearson_delta_top20": pearson_delta_top_de(dx_hat, dx_true, k),
        "mse_top20": mse_top_de(dx_hat, dx_true, k),
        "edistance_scperturb": edistance_scperturb(dx_hat, dx_true),
        "energy_distance": energy_distance(dx_hat, dx_true, n_sample=n_sample, seed=seed),
    }


def oriented(name: str, value: float) -> float:
    """The metric signed so that LARGER IS BETTER, so one contrast direction works for all of them."""
    return ORIENTATION[name] * float(value)


__all__ = ["ORIENTATION", "TOP_DE_K", "edistance_scperturb", "energy_distance",
           "external_metric_suite", "mse_top_de", "oriented", "pearson_delta",
           "pearson_delta_top_de", "top_de_index"]
