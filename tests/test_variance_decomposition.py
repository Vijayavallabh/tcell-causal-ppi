"""Does the nested decomposition recover variance components it was given?

The estimator is method-of-moments on nested means, which is easy to get subtly wrong in the direction
that flatters the result: forget to subtract the seed contribution from the re-draw term and the
re-draw variance absorbs seed noise, inflating it and making any real difficulty effect look small by
comparison. These tests plant known components and check they come back.
"""
from __future__ import annotations

import numpy as np
import pytest

from tcell_pipeline.screening.variance_decomposition import decompose


def _cells(level_effects, redraws_per_level, n_seeds, sd_redraw, sd_seed, seed=0):
    """Synthesise cells from a known nested model."""
    rng = np.random.default_rng(seed)
    cells = {}
    for li, mu in enumerate(level_effects):
        for r in range(redraws_per_level[li]):
            cell_mean = mu + rng.normal(0, sd_redraw)
            deltas = list(cell_mean + rng.normal(0, sd_seed, n_seeds))
            cells[(f"L{li}", f"s{r}")] = {"deltas": deltas,
                                          "mean": float(np.mean(deltas)), "n": n_seeds}
    return cells


def test_recovers_seed_variance():
    """With one level and many re-draws, the seed component should match what was planted."""
    cells = _cells([0.0], [40], n_seeds=8, sd_redraw=0.0, sd_seed=0.02, seed=1)
    d = decompose(cells)
    assert d["sd_seed"] == pytest.approx(0.02, rel=0.15)


def test_seed_noise_does_not_leak_into_redraw():
    """THE failure this guards. Plant zero re-draw variance and large seed noise; the re-draw
    component must stay near zero rather than absorbing the seed spread."""
    cells = _cells([0.0], [30], n_seeds=6, sd_redraw=0.0, sd_seed=0.05, seed=2)
    d = decompose(cells)
    assert d["sd_seed"] == pytest.approx(0.05, rel=0.15)
    # Without subtracting sd_seed/sqrt(n) the naive estimate would be ~0.05/sqrt(6) = 0.020.
    assert d["sd_redraw"] < 0.010, f"seed noise leaked into re-draw: {d['sd_redraw']}"


def test_seed_variance_is_unbiased_not_ddof0():
    """Discriminates ddof=1 from ddof=0, which a loose tolerance will not.

    With only 3 seeds per cell the biased estimator underestimates the variance by a factor (n-1)/n
    = 2/3, i.e. the sd by sqrt(2/3) = 0.816 - an 18% error. Many cells make the pooled estimate
    precise enough that a 10% tolerance separates the two. This is load-bearing: the seed component is
    compared against the level component to decide whether a difficulty trend is real, so a systematic
    underestimate of seed noise biases that comparison toward the more interesting answer.
    """
    cells = _cells([0.0], [200], n_seeds=3, sd_redraw=0.0, sd_seed=0.04, seed=11)
    d = decompose(cells)
    assert d["sd_seed"] == pytest.approx(0.04, rel=0.10), d["sd_seed"]
    assert d["sd_seed"] > 0.04 * 0.90, f"looks like a biased (ddof=0) estimator: {d['sd_seed']}"


def test_recovers_redraw_variance():
    cells = _cells([0.0], [40], n_seeds=8, sd_redraw=0.03, sd_seed=0.01, seed=3)
    d = decompose(cells)
    assert d["sd_redraw"] == pytest.approx(0.03, rel=0.25)


def test_detects_a_real_level_effect():
    """Widely separated levels with little other noise must give sd_level >> sd_redraw."""
    cells = _cells([-0.10, 0.0, 0.10], [8, 8, 8], n_seeds=6, sd_redraw=0.005, sd_seed=0.005, seed=4)
    d = decompose(cells)
    assert d["sd_level"] > 5 * d["sd_redraw"]


def test_no_level_effect_is_not_manufactured():
    """Identical levels must not produce a level component larger than the re-draw component."""
    cells = _cells([0.0, 0.0, 0.0], [8, 8, 8], n_seeds=6, sd_redraw=0.02, sd_seed=0.01, seed=5)
    d = decompose(cells)
    assert d["sd_level"] <= d["sd_redraw"], (d["sd_level"], d["sd_redraw"])


def test_unidentified_components_are_none_not_zero():
    """One re-draw per level cannot identify within-level variance. It must come back None, because
    'not estimable' and 'estimated as zero' would license opposite conclusions."""
    cells = _cells([0.0, 0.05], [1, 1], n_seeds=5, sd_redraw=0.0, sd_seed=0.01, seed=6)
    d = decompose(cells)
    assert d["sd_redraw"] is None
    assert d["df"]["redraw"] == 0


def test_single_seed_cells_give_no_seed_component():
    cells = {("L0", "s0"): {"deltas": [0.01], "mean": 0.01, "n": 1},
             ("L1", "s0"): {"deltas": [0.02], "mean": 0.02, "n": 1}}
    d = decompose(cells)
    assert d["sd_seed"] is None


def test_degrees_of_freedom_are_reported():
    cells = _cells([0.0, 0.1], [3, 3], n_seeds=5, sd_redraw=0.01, sd_seed=0.01, seed=7)
    d = decompose(cells)
    assert d["df"]["level"] == 1                 # 2 levels
    assert d["df"]["redraw"] == 4                # (3-1) + (3-1)
    assert d["df"]["seed"] == 6 * (5 - 1)        # 6 cells x 4


def test_unbalanced_cells_use_mean_of_inverse_n():
    """Unequal seeds per cell: the seed correction must be mean(sigma2/n_j), not sigma2/mean(n_j).

    Plant zero re-draw variance with very unequal n. Jensen's inequality makes mean(1/n) strictly
    larger than 1/mean(n) whenever n varies, so the wrong form under-subtracts and leaves a re-draw
    component that is not there. That component is the verdict's denominator, so the error would push
    the conclusion toward "difficulty does not matter" for free.
    """
    rng = np.random.default_rng(21)
    cells = {}
    for i, n in enumerate([1, 1, 2, 3, 5, 8] * 12):
        deltas = list(rng.normal(0.0, 0.04, n))
        cells[("L0", f"s{i}")] = {"deltas": deltas, "mean": float(np.mean(deltas)), "n": n}
    d = decompose(cells)
    assert d["sd_redraw"] < 0.012, f"under-subtracted; phantom re-draw variance: {d['sd_redraw']}"
