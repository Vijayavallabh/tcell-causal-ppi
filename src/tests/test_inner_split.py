"""Target-grouped train-internal holdout: the fence that keeps hyperparameter tuning off the val fold.

The load-bearing claim is ONE invariant: no target gene may appear on both sides. A random row split
satisfies every other property here (partition, determinism, sizes) and still leaks, because one target
spans ~3 rows in this mart — so `test_no_target_spans_both_sides` is the test that has to be able to
fail, and it is written to defeat a naive row-shuffling implementation specifically.
"""
from __future__ import annotations

import pytest

from tcell_pipeline.training.inner_split import group_partition


def _labels(spec: dict[str, int]) -> list[str]:
    """{gene: n_rows} -> a row-label list, one contiguous block per gene.

    Block order is immaterial to what these tests assert (``group_partition`` keys on label VALUE, never
    on row position), so no interleaving is needed — but say what it does, in case someone later builds a
    contiguous-slice baseline test on top of it."""
    rows: list[str] = []
    for gene, n in spec.items():
        rows.extend([gene] * n)
    return rows


def test_no_target_spans_both_sides():
    """THE invariant. 20 genes x 8 rows each: a random row split puts nearly every gene on both sides."""
    labels = _labels({f"GENE{i}": 8 for i in range(20)})
    tr, ho = group_partition(labels, holdout_frac=0.25, seed=0)
    tr_genes = {labels[i] for i in tr}
    ho_genes = {labels[i] for i in ho}
    assert tr_genes & ho_genes == set(), (
        f"{len(tr_genes & ho_genes)} target genes appear on BOTH sides — the holdout leaks: "
        f"{sorted(tr_genes & ho_genes)[:5]}"
    )


def test_every_row_used_exactly_once():
    labels = _labels({f"GENE{i}": 3 for i in range(30)})
    tr, ho = group_partition(labels, holdout_frac=0.2, seed=0)
    assert sorted(tr + ho) == list(range(len(labels)))


def test_holdout_fraction_is_approximately_respected():
    labels = _labels({f"GENE{i}": 4 for i in range(100)})
    _, ho = group_partition(labels, holdout_frac=0.2, seed=0)
    assert 0.1 <= len(ho) / len(labels) <= 0.3


def test_deterministic_for_a_seed_and_varies_across_seeds():
    labels = _labels({f"GENE{i}": 3 for i in range(40)})
    assert group_partition(labels, seed=0) == group_partition(labels, seed=0)
    assert group_partition(labels, seed=0) != group_partition(labels, seed=1)


def test_one_dominant_gene_cannot_be_split():
    """A single target owning most rows must land wholly on one side, even though that wrecks the
    requested fraction. Silently splitting it to hit 20% is the leak this exists to prevent — and the
    wrecked fraction must be announced, not swallowed."""
    labels = _labels({"HUB": 100, "A": 2, "B": 2})
    with pytest.warns(UserWarning, match="not the requested"):
        tr, ho = group_partition(labels, holdout_frac=0.2, seed=0)
    hub_sides = {"train" if i in set(tr) else "holdout" for i, g in enumerate(labels) if g == "HUB"}
    assert len(hub_sides) == 1, "HUB's rows were split across both sides"


def test_unreachable_fraction_warns_instead_of_silently_returning_a_different_split():
    """The `not held` fallback: when EVERY group individually overshoots the target, the split cannot be
    near the requested fraction. It must still be a valid split, and it must say so — a caller that
    believes it held out 20% while holding out 50% reports the wrong number."""
    labels = _labels({"A": 50, "B": 50})
    with pytest.warns(UserWarning, match=r"50\.0% of rows, not the requested 20\.0%"):
        tr, ho = group_partition(labels, holdout_frac=0.2, seed=0)
    assert len(ho) == 50 and len(tr) == 50
    assert {labels[i] for i in tr} & {labels[i] for i in ho} == set()


def test_a_reachable_fraction_does_not_warn():
    """The warning must be able to STAY SILENT, or it carries no information when it fires."""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning here becomes a test failure
        group_partition(_labels({f"GENE{i}": 4 for i in range(100)}), holdout_frac=0.2, seed=0)


@pytest.mark.parametrize("frac", [0.0, -0.1, 0.51, 0.9, 1.0])
def test_holdout_frac_outside_the_valid_range_raises(frac):
    """Above 0.5 a 'holdout' is the majority of the data, and it is the only regime where the admission
    loop could sweep every group into the holdout and leave the train side empty. Rejected at the door,
    which is what lets group_partition carry no repair branch for that case."""
    with pytest.raises(ValueError, match=r"holdout_frac must be in \(0, 0.5\]"):
        group_partition(_labels({"A": 10, "B": 10}), holdout_frac=frac, seed=0)


def test_holdout_frac_of_exactly_one_half_is_allowed():
    tr, ho = group_partition(_labels({f"GENE{i}": 2 for i in range(10)}), holdout_frac=0.5, seed=0)
    assert tr and ho


def test_degenerate_single_gene_raises_rather_than_returning_an_empty_side():
    """Absence of a usable holdout must be loud, not an empty Subset that silently tunes on nothing."""
    with pytest.raises(ValueError, match="one target gene"):
        group_partition(["ONLY"] * 10, holdout_frac=0.2, seed=0)


def test_empty_input_raises():
    with pytest.raises(ValueError):
        group_partition([], holdout_frac=0.2, seed=0)
