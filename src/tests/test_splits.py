"""feat-003 leakage-safe split tests on synthetic data (no marts, no embedding parquets)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tcell_pipeline import config, splits


def _vec(seed: int, d: int = 16) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(d).astype(np.float32)


def _synthetic():
    # paralog family P1..P3 (identical vectors) + complex pair C1,C2 + distinct singletons S1..S5
    genes = ["P1", "P2", "P3", "C1", "C2", "S1", "S2", "S3", "S4", "S5"]
    pv = _vec(100)
    gene_vec = {"P1": pv, "P2": pv.copy(), "P3": pv.copy()}
    for i, g in enumerate(["C1", "C2", "S1", "S2", "S3", "S4", "S5"]):
        gene_vec[g] = _vec(i + 1)
    complex_df = pd.DataFrame([{"protein_gene": "C1", "complex_id": 1},
                               {"protein_gene": "C2", "complex_id": 1}])
    edges_df = pd.DataFrame([{"source_gene": "P1", "target_gene": "S1", "is_physical": 1}])
    return genes, gene_vec, complex_df, edges_df


def test_paralogs_and_complex_grouped():
    genes, gene_vec, complex_df, _ = _synthetic()
    labels, _ = splits.family_groups(genes, gene_vec, complex_df, cap_frac=0.5, seq_threshold=0.85)
    assert labels["P1"] == labels["P2"] == labels["P3"]      # paralog family blocked together
    assert labels["C1"] == labels["C2"]                       # complex co-members blocked together
    assert labels["P1"] != labels["C1"]                       # distinct families are distinct groups
    assert len({labels["S1"], labels["S2"], labels["S3"]}) == 3  # singletons stay separate


def test_blocking_no_family_split_across_roles():
    genes, gene_vec, complex_df, _ = _synthetic()
    labels, _ = splits.family_groups(genes, gene_vec, complex_df, cap_frac=0.5, seq_threshold=0.85)
    role_of = splits.assign_partitions(labels, genes, seed=0)
    by_group: dict[int, set] = {}
    for g in genes:
        by_group.setdefault(labels[g], set()).add(role_of[g])
    assert all(len(rs) == 1 for rs in by_group.values())      # no group spans >1 role


def test_size_cap_refuses_giant_merge():
    # one complex with 8 members, cap=4 -> some merges refused, no group exceeds the cap
    members = [f"M{i}" for i in range(8)]
    complex_df = pd.DataFrame([{"protein_gene": m, "complex_id": 1} for m in members])
    labels, refused = splits.family_groups(members, {}, complex_df, cap_frac=0.5)  # cap = ceil(0.5*8)=4
    assert refused > 0
    assert max(np.bincount(list(labels.values()))) <= 4


def test_determinism_by_seed():
    genes = [f"G{i}" for i in range(40)]
    labels = {g: i for i, g in enumerate(genes)}  # all singletons
    a = splits.assign_partitions(labels, genes, seed=0)
    b = splits.assign_partitions(labels, genes, seed=0)
    c = splits.assign_partitions(labels, genes, seed=1)
    assert a == b                                             # same seed -> identical
    assert a != c                                             # different seed -> different


def test_singletons_distribute_across_roles():
    genes = [f"G{i}" for i in range(40)]
    labels = {g: i for i, g in enumerate(genes)}
    role_of = splits.assign_partitions(labels, genes, seed=0)
    assert set(role_of.values()) == set(config.SPLIT_ROLES)  # every role populated


def test_leakage_audit_passes_and_fails_closed():
    genes, gene_vec, complex_df, edges_df = _synthetic()
    labels, _ = splits.family_groups(genes, gene_vec, complex_df, cap_frac=0.5, seq_threshold=0.85)
    role_of = splits.assign_partitions(labels, genes, seed=0)
    splits.audit_leakage(role_of, labels, genes, gene_vec, complex_df, edges_df)  # valid -> no raise
    # hand-inject a leak: split the paralog family across roles
    bad = dict(role_of)
    bad["P1"], bad["P2"] = "train", "challenge"
    with pytest.raises(ValueError):
        splits.audit_leakage(bad, labels, genes, gene_vec, complex_df, edges_df)


def test_random_split_fractions():
    items = list(range(1000))
    role_of = splits.assign_random(items, seed=0)
    counts = pd.Series(list(role_of.values())).value_counts(normalize=True)
    for r, frac in config.SPLIT_FRACTIONS.items():
        assert abs(counts[r] - frac) < 0.02


def test_random_split_covers_all_items_small_n():
    # cumulative-boundary allocation: every item placed exactly once, no truncated tail (regression for
    # banker's-rounding over-allocation that dropped the last role's slice at small N)
    for n in (10, 13, 27, 101):
        role_of = splits.assign_random(list(range(n)), seed=0)
        assert set(role_of) == set(range(n))                               # each item assigned once
        assert sum(1 for _ in role_of) == n                                # sizes partition n exactly


def test_grouping_ignores_non_qpre_columns():
    # family grouping must consult only q_pre structure; an extra (q_post-named) column changes nothing
    genes, gene_vec, complex_df, _ = _synthetic()
    base, _ = splits.family_groups(genes, gene_vec, complex_df, cap_frac=0.5, seq_threshold=0.85)
    poisoned = complex_df.assign(ontarget_effect_size=1.0)
    other, _ = splits.family_groups(genes, gene_vec, poisoned, cap_frac=0.5, seq_threshold=0.85)
    assert base == other
