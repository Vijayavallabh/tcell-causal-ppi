"""A2(b) power-simulation tests.

This project has shipped two tests that passed against buggy code, so each assertion below is chosen to
FAIL under a specific plausible mutation, and each says which one:

  - null calibration      fails if the family-wise correction is dropped (0.0125 -> 0.05)
  - correction bites      fails if family_size stops reaching the survival rule
  - analytic cross-check  fails if the simulated statistic drifts from the planned test
  - level floor           fails if the level component is treated as shrinkable by compute
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tcell_pipeline.screening.power_simulation import (
    MEASURED,
    analytic_n,
    meta_sd,
    nested_sd,
    required_n,
    simulate_power,
)


def test_null_effect_survives_at_the_corrected_rate_not_the_nominal_one():
    """The delta=0 negative control. With no true effect the survival rate must sit at alpha/m =
    0.0125, NOT at alpha = 0.05: that gap is the entire contribution of the family-wise rule, and a
    simulation that reported 0.05 here would be quietly running an uncorrected test."""
    r = simulate_power(0.0, MEASURED["h2a"]["sd_seed"], 5, n_sims=1500, seed=11)
    assert 0.003 <= r["power"] <= 0.030, f"false-positive rate {r['power']:.4f} is not alpha/4"
    assert r["ci_low"] <= 0.0125 <= r["ci_high"] or r["power"] < 0.0125


def test_the_correction_costs_real_power():
    """Same effect, same n, family of 4 against family of 1. If these came out equal the family_size
    argument would not be reaching the survival rule at all, and every count this module reports would
    be an uncorrected count wearing a corrected label."""
    kw = dict(n_sims=800, seed=5)
    corrected = simulate_power(0.02, 0.01067, 6, family_size=4, **kw)["power"]
    uncorrected = simulate_power(0.02, 0.01067, 6, family_size=1, **kw)["power"]
    assert uncorrected > corrected + 0.05, f"correction did not bite: {uncorrected} vs {corrected}"


@pytest.mark.parametrize("delta,sd", [(0.02, 0.01067), (0.01, 0.00524)])
def test_simulated_sample_size_tracks_the_normal_approximation(delta, sd):
    """Independent cross-check. The simulation uses a t-test and the closed form uses z, so the
    simulated n must be a little LARGER and within a modest factor. Order-of-magnitude disagreement
    means the replicate is not running the test the module claims it runs."""
    sim = required_n(delta, sd, n_sims=600, seed=3)
    approx = analytic_n(delta, sd)
    assert sim is not None
    assert approx <= sim <= approx + 6, f"simulated n={sim} against normal approximation {approx:.1f}"


def test_required_n_falls_as_the_effect_grows():
    sd = MEASURED["h2a"]["sd_seed"]
    ns = [required_n(d, sd, n_sims=500, seed=1) for d in (0.01, 0.02, 0.05)]
    assert all(a is not None for a in ns)
    assert ns[0] > ns[1] > ns[2]


def test_one_unit_is_powerless_by_the_rule_rather_than_by_arithmetic():
    """n=1 emits no p-value in this pipeline (a single seed is not a paired result), so the honest
    answer is zero power, not a crash and not a number borrowed from a formula."""
    r = simulate_power(0.5, 0.01, 1, n_sims=10)
    assert r["power"] == 0.0 and r["n_sims"] == 0 and "paired" in r["note"]


def test_the_level_component_is_a_floor_that_compute_cannot_buy_down():
    """sd of a level mean must converge to the level sd itself as re-draws and seeds grow, never to
    zero. A version that averaged the level term away would make 'generalises to a new partition' look
    reachable by adding seeds, which is the claim L4 exists to refute."""
    m = MEASURED["h2a"]
    huge = nested_sd(m["sd_level"], m["sd_redraw"], m["sd_seed"], 1000, 1000)
    assert huge > m["sd_level"]
    assert math.isclose(huge, m["sd_level"], rel_tol=1e-3)
    assert nested_sd(m["sd_level"], m["sd_redraw"], m["sd_seed"], 1, 5) > huge
    assert meta_sd(0.02, 0.0) == pytest.approx(0.02)
    assert meta_sd(0.02, 0.02) == pytest.approx(0.02 * math.sqrt(2))


def test_simulation_is_reproducible_from_its_seed():
    a = simulate_power(0.02, 0.01, 5, n_sims=200, seed=42)
    b = simulate_power(0.02, 0.01, 5, n_sims=200, seed=42)
    c = simulate_power(0.02, 0.01, 5, n_sims=200, seed=43)
    assert a["power"] == b["power"]
    assert isinstance(np.float64(a["power"]).item(), float)
    assert a != c or a["power"] == c["power"]  # a different seed may coincide, but must not crash
