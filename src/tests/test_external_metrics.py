"""A3 external-metric tests.

Each assertion is picked to fail under a specific plausible bug, and says which:

  DE subset taken from the prediction rather than the observation  -> test_top_de_ranks_the_observation
  the top-k restriction silently not applied                       -> test_top20_metrics_really_restrict
  scPerturb's squared form quietly used AS a distributional claim  -> test_scperturb_edistance_is_a_mean_difference
  an error metric contrasted without flipping its sign             -> test_orientation_turns_errors_into_larger_is_better
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from tcell_pipeline.evaluation.external_metrics import (
    ORIENTATION,
    edistance_scperturb,
    energy_distance,
    external_metric_suite,
    mse_top_de,
    oriented,
    pearson_delta,
    pearson_delta_top_de,
    top_de_index,
)


def test_top_de_ranks_the_observation_not_the_prediction():
    """Ranking on the prediction would let a model choose the genes it is confident about and be scored
    on those; GEARS ranks on the observed effect."""
    true = np.array([[0.0, 5.0, 0.0, -4.0, 1.0]])
    pred = np.array([[9.0, 0.0, 8.0, 0.0, 0.0]])          # confident on exactly the wrong genes
    assert sorted(top_de_index(true, k=2)[0]) == [1, 3]
    assert sorted(top_de_index(pred, k=2)[0]) == [0, 2]   # the mutation this guards against


def test_top20_metrics_really_restrict_to_the_de_subset():
    """Perfect on the top-k genes and wrong everywhere else: the restricted MSE must be 0 while the
    unrestricted error is large, and the restricted correlation must be 1."""
    rng = np.random.default_rng(0)
    true = rng.normal(size=(12, 60))
    true[:, :5] *= 40.0                                   # these are unambiguously the top-5 DE genes
    pred = np.zeros_like(true)
    pred[:, :5] = true[:, :5]
    assert mse_top_de(pred, true, k=5) == pytest.approx(0.0, abs=1e-12)
    assert np.mean((pred - true) ** 2) > 0.5              # the unrestricted error is nowhere near 0
    assert pearson_delta_top_de(pred, true, k=5) == pytest.approx(1.0, abs=1e-6)
    assert pearson_delta(pred, true) < 0.999              # all-gene correlation is dragged down


def test_scperturb_edistance_is_a_mean_difference_and_the_energy_distance_is_not():
    """The claim the module's warning rests on. Same mean, different spread: scPerturb's squared form
    reads ~0 (it is algebraically 2||mean_X - mean_Y||^2), while the plain-euclidean energy distance is
    clearly positive. Using the squared form to close a hedge about DISTRIBUTIONS would be wrong, and
    this is the test that says so."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=(400, 8))
    b = rng.normal(0.0, 4.0, size=(400, 8))               # same mean, 4x the spread
    a -= a.mean(0)
    b -= b.mean(0)                                        # means identical by construction
    assert edistance_scperturb(a, b) == pytest.approx(0.0, abs=1e-10)
    assert energy_distance(a, b, n_sample=400) > 0.5


def test_scperturb_edistance_matches_its_pairwise_definition():
    """The closed form is an optimisation, so it has to agree with the definition it replaces:
    2*mean||x-y||^2 - mean||x-x'||^2 - mean||y-y'||^2 over SQUARED euclidean distances."""
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(50, 6)), rng.normal(1.0, 2.0, size=(70, 6))
    literal = (2 * cdist(a, b, "sqeuclidean").mean()
               - cdist(a, a, "sqeuclidean").mean() - cdist(b, b, "sqeuclidean").mean())
    assert edistance_scperturb(a, b) == pytest.approx(literal, rel=1e-9)


def test_energy_distance_is_zero_for_one_distribution_and_grows_with_separation():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 5))
    y = rng.normal(size=(300, 5))
    same = energy_distance(x, y, n_sample=300)
    far = energy_distance(x, y + 3.0, n_sample=300)
    assert abs(same) < 0.15 and far > same + 2.0


def test_orientation_turns_errors_into_larger_is_better():
    assert ORIENTATION["pearson_delta"] == 1 and ORIENTATION["mse_top20"] == -1
    assert oriented("mse_top20", 0.4) < oriented("mse_top20", 0.1)      # less error scores higher
    assert oriented("energy_distance", 2.0) < oriented("energy_distance", 1.0)
    assert oriented("pearson_delta", 0.1) < oriented("pearson_delta", 0.4)


def test_suite_returns_every_oriented_metric_and_nothing_else():
    rng = np.random.default_rng(4)
    true = rng.normal(size=(40, 30))
    pred = true + rng.normal(0, 0.5, size=true.shape)
    out = external_metric_suite(pred, true, k=5, n_sample=40)
    assert set(out) == set(ORIENTATION)
    assert all(np.isfinite(v) for v in out.values())
