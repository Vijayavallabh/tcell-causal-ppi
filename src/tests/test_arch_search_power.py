"""A4 tests. The claim this module makes is a comparison, so the tests pin the comparison rather than
the numbers: an architecture search's spread is inside seed noise, or it is not, and the answer must
flip when the inputs say it should.
"""
from __future__ import annotations

import json

import pytest

from tcell_pipeline.screening.arch_search_power import expected_range, load_cells, run


def test_expected_range_grows_with_draws_and_clamps_outside_the_table():
    """More draws span more, and the ratio is bounded at both ends of the tabulated set rather than
    extrapolated. A version that used a flat 1 sigma would understate the spread noise produces and
    would call a noisy search a real difference."""
    assert expected_range(5, 1.0) < expected_range(14, 1.0) < expected_range(20, 1.0)
    assert expected_range(2, 1.0) == expected_range(5, 1.0)         # clamped low
    assert expected_range(500, 1.0) == expected_range(20, 1.0)      # clamped high
    assert expected_range(14, 2.0) == pytest.approx(2 * expected_range(14, 1.0))
    assert expected_range(14, 1.0) > 3.0                            # NOT 1 sigma, the mutation to catch


def _write_cells(tmp_path, systemas):
    """A synthetic arch-search root: one baseline plus one cell per given systema."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "expression_only__norm-none.json").write_text(json.dumps(
        {"arm": "expression_only", "cell_id": "expression_only__norm-none",
         "systema": 0.0862, "epochs_run": 5}))
    for i, s in enumerate(systemas):
        (tmp_path / f"cell{i}.json").write_text(json.dumps(
            {"arm": "condition_gated", "cell_id": f"cell{i}", "systema": s, "epochs_run": 5}))
    return str(tmp_path)


def test_a_search_inside_seed_noise_is_called_inside_and_a_wide_one_is_not(tmp_path):
    """The verdict has to be able to go both ways on real inputs, or it is not a test of anything. The
    frozen-fold seed sd is 0.00431, so 14 draws of pure noise span about 0.0147: a search whose cells
    sit within a few thousandths is noise, and one spanning 0.06 is not."""
    narrow = _write_cells(tmp_path / "narrow", [0.0862 + d for d in
                                                (0.005, 0.003, 0.001, -0.001, -0.003)])
    wide = _write_cells(tmp_path / "wide", [0.0862 + d for d in (0.05, 0.02, 0.0, -0.01)])
    r_narrow = run(narrow, None, n_sims=200)
    r_wide = run(wide, None, n_sims=200)
    assert r_narrow["spread_within_seed_noise"] is True
    assert r_wide["spread_within_seed_noise"] is False
    assert r_wide["best_delta"] > r_narrow["best_delta"]


def test_a_search_with_no_baseline_cell_fails_loudly(tmp_path):
    """Without the no-graph cell every delta would be measured against an arbitrary member of the
    search, which is how an architecture search flatters itself."""
    d = tmp_path / "nobase"
    d.mkdir()
    (d / "cell0.json").write_text(json.dumps(
        {"arm": "condition_gated", "cell_id": "cell0", "systema": 0.09, "epochs_run": 5}))
    with pytest.raises(RuntimeError, match="expression_only"):
        run(str(d), None, n_sims=50)


def test_the_real_search_is_read_and_carries_one_seed_per_cell():
    """Guards the premise the whole bound rests on: if the landed search ever gains seeds, the n=1
    argument stops applying and this test should be the thing that notices."""
    cells = load_cells()
    assert len(cells) == 14
    assert all(c["epochs_run"] == 5 for c in cells)
