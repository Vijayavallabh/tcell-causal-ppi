"""B2 rank-bin tests. The bins must be DISJOINT and COVERING - the whole point of the analysis is that
it decomposes A3's cumulative sweep, and a binning that overlapped or left a gap would not."""
from __future__ import annotations

import numpy as np

from tcell_pipeline.screening.rank_deciles import HEAD_EDGES, _bins, _crossover


def test_bins_are_disjoint_and_cover_every_gene():
    for scheme in ("deciles", "head"):
        bins = _bins(10282, scheme)
        seen = np.zeros(10282, dtype=int)
        for lo, hi, _ in bins:
            seen[lo:hi] += 1
        assert (seen == 1).all(), f"{scheme} bins overlap or leave a gap"
        assert bins[0][0] == 0 and bins[-1][1] == 10282


def test_head_bins_are_exactly_the_increments_of_a3s_cumulative_sweep():
    """A head bin must be the difference between two consecutive k-sweep points, or the two analyses do
    not decompose each other and the crossover cannot be read off against A3's."""
    from tcell_pipeline.screening.rescore_external import K_SWEEP
    bins = _bins(10282, "head")
    assert [lo for lo, _, _ in bins][1:] == [k for k in K_SWEEP[:-1]]
    assert [hi for _, hi, _ in bins] == list(K_SWEEP)
    assert HEAD_EDGES[-1] == 10282


def test_bins_degrade_sanely_on_a_smaller_gene_panel():
    bins = _bins(120, "head")
    assert bins[0][0] == 0 and bins[-1][1] == 120
    seen = np.zeros(120, dtype=int)
    for lo, hi, _ in bins:
        seen[lo:hi] += 1
    assert (seen == 1).all()


def _cells(means, survives=True):
    return {f"{l}/promotion_margin": {"mean": m, "survives_family_wise": survives}
            for l, m in means.items()}


def test_crossover_finds_the_first_sign_change_walking_from_the_most_moved_genes():
    bins = [(0, 1, "a"), (1, 2, "b"), (2, 3, "c"), (3, 4, "d")]
    cr = _crossover(_cells({"a": -0.02, "b": -0.01, "c": +0.01, "d": +0.03}), bins, "promotion_margin")
    fs = cr["first_sign_change"]
    assert (fs["from_bin"], fs["to_bin"]) == ("b", "c") and fs["both_survive"] is True


def test_crossover_reports_no_change_when_the_sign_holds_and_flags_an_unresolved_one():
    bins = [(0, 1, "a"), (1, 2, "b")]
    assert _crossover(_cells({"a": +0.02, "b": +0.01}), bins, "promotion_margin")["first_sign_change"] is None
    # Amendment 8.4: a flip whose bins do not both clear correction is NOT a located crossover
    cr = _crossover(_cells({"a": -0.02, "b": +0.01}, survives=False), bins, "promotion_margin")
    assert cr["first_sign_change"]["both_survive"] is False


def test_an_undecidable_bin_is_skipped_rather_than_read_as_a_sign():
    """A bin whose contrast is None (no finite paired deltas) must not be treated as sign zero and must
    not silently become a crossover boundary."""
    bins = [(0, 1, "a"), (1, 2, "b"), (2, 3, "c")]
    cells = _cells({"a": -0.02, "c": +0.01})
    cells["b/promotion_margin"] = {"mean": None, "survives_family_wise": None}
    cr = _crossover(cells, bins, "promotion_margin")
    assert [s["bin"] for s in cr["sequence"]] == ["a", "c"]
    assert cr["first_sign_change"]["from_bin"] == "a" and cr["first_sign_change"]["to_bin"] == "c"
