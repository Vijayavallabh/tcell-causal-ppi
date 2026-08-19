"""B3 sensitivity-analysis tests. The load-bearing step is recovering the per-seed SD from a REPORTED
confidence interval - the ladder report persists the interval, not the raw deltas - so if that inversion
is wrong every minimum detectable effect in app:floor is wrong with it."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from tcell_pipeline.screening.b3_power import mde, sd_from_ci, seeds_needed


def test_sd_is_recovered_exactly_from_an_interval_built_from_a_known_sample():
    """Round trip against scipy: build a real paired sample, form its t interval the way the ladder
    report does, then invert it. Anything but the sample's own ddof=1 SD is a bug."""
    x = np.array([-0.0061, -0.0044, -0.0038, -0.0057])
    n = len(x)
    sd = x.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    ci = {"n": n, "ci_low": x.mean() - half, "ci_high": x.mean() + half}
    assert sd_from_ci(ci) == pytest.approx(sd, rel=1e-9)


def test_an_undecidable_or_single_seed_interval_returns_none_rather_than_a_number():
    assert sd_from_ci({"n": 4, "ci_low": None, "ci_high": None}) is None
    assert sd_from_ci({"n": 1, "ci_low": 0.0, "ci_high": 0.0}) is None


def test_mde_uses_the_t_quantile_at_the_designs_own_df_not_z():
    """At n=4 the t/z difference is not a rounding detail: using z would understate the MDE by a third
    and could turn 'cannot detect' into 'can'."""
    sd = 0.00212
    t_based = mde(sd, 4)
    z_based = (sd / 2) * (stats.norm.ppf(0.975) + stats.norm.ppf(0.80))
    assert t_based > z_based * 1.2
    assert mde(sd, 4) > mde(sd, 8) > mde(sd, 25)      # more seeds detect smaller effects


def test_seeds_needed_returns_the_smallest_n_that_actually_reaches_the_target():
    sd = 0.00212
    n = seeds_needed(sd, -0.00124)
    assert n is not None and mde(sd, n) <= 0.00124 and mde(sd, n - 1) > 0.00124


def test_the_verdict_is_driven_by_the_artifact_and_says_do_not_run():
    """The whole point of this module is that it reached a decision. If the landed ladder ever changes
    enough to make the rung feasible, this test fails and the decision gets re-made rather than
    inherited."""
    import json
    from pathlib import Path

    from tcell_pipeline.screening.b3_power import run
    art = Path("data/results/a2_ladder/b3_power.json")
    if not art.exists():
        pytest.skip("b3_power.json not built")
    res = json.loads(art.read_text())
    assert res["verdict"] == "underpowered_do_not_run"
    assert abs(res["predictions_at_target_delta"]["proportional"]) < min(res["mde"].values())
    assert run(out=None)["verdict"] == res["verdict"]
