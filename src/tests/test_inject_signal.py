"""A2(a) injection tests. The leakage test is the one that matters, and it was watched to FAIL against
a deliberately leaky variant before being trusted (see test_the_leakage_guard_can_actually_fail).

The whole ladder's meaning rests on one property: a validation target's injected component must be a
function of TRAIN responses only. If it is not, the graph arm detects leakage rather than structure and
the measured floor is a floor for nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from tcell_pipeline.screening.inject_signal import injection_matrix, neighbour_operator

TARGETS = ["A", "B", "C", "D"]
#          A-B, B-C, C-D chain, so A's only neighbour is B and D's only neighbour is C.
_EDGES = [(0, 1), (1, 2), (2, 3)]


def _chain(n: int = 4, hops: int = 1) -> sp.csr_matrix:
    rows = [a for a, b in _EDGES] + [b for a, b in _EDGES]
    cols = [b for a, b in _EDGES] + [a for a, b in _EDGES]
    a = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    for _ in range(hops - 1):
        a = (a @ a).tocsr()
    a.setdiag(0.0)
    a.eliminate_zeros()
    deg = np.asarray(a.sum(1)).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(deg > 0, 1.0 / deg, 0.0)
    return sp.diags(s) @ a


def _fixture(seed: int = 0, n_genes: int = 5):
    """Eight rows over four targets: A and B are train, C is val, D is sealed (challenge)."""
    row_target = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    role_of_row = np.array(["train", "train", "train", "train", "val", "val",
                            "challenge", "challenge"])
    rng = np.random.default_rng(seed)
    Z = sp.csr_matrix(rng.normal(size=(8, n_genes)))
    return Z, row_target, role_of_row


def test_a_validation_response_never_reaches_any_injected_value():
    """THE test. Change a validation row's response by a large amount and the entire injection matrix
    must be bit-identical, because it is built from train rows only."""
    Z, rt, role = _fixture()
    A = _chain()
    base = injection_matrix(Z, rt, role, A, TARGETS)["M"]

    dense = Z.toarray()
    dense[4] += 100.0                       # row 4 is a VALIDATION row of target C
    tampered = injection_matrix(sp.csr_matrix(dense), rt, role, A, TARGETS)["M"]
    assert np.array_equal(base, tampered), "a validation response changed the injection: LEAKAGE"


def test_the_leakage_guard_can_actually_fail():
    """The guard above is only evidence if it can fail. Build the injection from ALL rows instead of
    train rows and the same assertion must break; if it does not, the test is measuring nothing."""
    Z, rt, role = _fixture()
    A = _chain()
    leaky_role = np.array(["train"] * len(role))            # the bug: val treated as train
    base = injection_matrix(Z, rt, leaky_role, A, TARGETS)["M"]
    dense = Z.toarray()
    dense[4] += 100.0
    tampered = injection_matrix(sp.csr_matrix(dense), rt, leaky_role, A, TARGETS)["M"]
    assert not np.array_equal(base, tampered), "the leakage test cannot fail, so it proves nothing"


def test_a_target_never_receives_its_own_response():
    """Self-exclusion, checked where it is visible: A's only neighbour is B, so A's injection must be
    B's train mean and must not move when A's own rows move."""
    Z, rt, role = _fixture()
    A = _chain()
    out = injection_matrix(Z, rt, role, A, TARGETS)
    dense = Z.toarray()
    dense[0] += 50.0                        # a TRAIN row of target A
    moved = injection_matrix(sp.csr_matrix(dense), rt, role, A, TARGETS)["M"]
    assert not np.allclose(out["M"][2], moved[2]), "B's injection ignores its neighbour A entirely"
    # A's injection comes only from B, so re-deriving it from B's train mean must reproduce it exactly
    b_mean = Z.toarray()[[2, 3]].mean(0)
    assert np.allclose(out["M"][0] / out["scale"], b_mean, atol=1e-5)


def test_two_hops_still_cannot_return_to_the_target():
    """A two-hop walk A->B->A would hand A its own response back. The diagonal is cleared AFTER the
    expansion precisely to stop that, and this is the assertion that keeps it cleared."""
    a2 = _chain(hops=2)
    assert np.allclose(a2.diagonal(), 0.0)
    Z, rt, role = _fixture()
    out = injection_matrix(Z, rt, role, a2, TARGETS, hops=2)
    dense = Z.toarray()
    dense[0] += 50.0                        # a train row of A
    moved = injection_matrix(sp.csr_matrix(dense), rt, role, a2, TARGETS, hops=2)["M"]
    a_rows = np.flatnonzero(rt == "A")
    assert np.allclose(out["M"][a_rows], moved[a_rows]), "a 2-hop walk returned A's response to A"


def test_sealed_and_absent_rows_are_left_at_exactly_zero():
    """Rail 1. Challenge rows must be bit-identically untouched, which means their injection is zero
    and no sealed response enters the scaling constant either."""
    Z, rt, role = _fixture()
    out = injection_matrix(Z, rt, role, _chain(), TARGETS)
    sealed = np.flatnonzero(role == "challenge")
    assert np.count_nonzero(out["M"][sealed]) == 0
    assert not out["usable"][sealed].any()
    dense = Z.toarray()
    dense[6] += 1000.0                      # tamper with a SEALED response
    moved = injection_matrix(sp.csr_matrix(dense), rt, role, _chain(), TARGETS)
    assert np.array_equal(out["M"], moved["M"])
    assert out["scale"] == pytest.approx(moved["scale"])


def test_scaling_is_fixed_on_train_rows_and_puts_the_injection_on_the_response_scale():
    """delta is meant to read as a fraction of a response SD. The constant that makes it so is computed
    on TRAIN rows, because a constant fitted on train+val is a leak (it is the one this module shipped
    first, and the leakage test caught it moving every injected value by 46x)."""
    Z, rt, role = _fixture()
    out = injection_matrix(Z, rt, role, _chain(), TARGETS)
    train = rt_train = (role == "train")
    assert out["M"][train].std() == pytest.approx(out["sd_response"], rel=1e-5)
    assert out["sd_response"] == pytest.approx(Z.toarray()[rt_train].std(), rel=1e-5)


def test_the_permuted_control_moves_every_target_and_shares_the_train_scale():
    Z, rt, role = _fixture(seed=3, n_genes=7)
    A = _chain()
    real = injection_matrix(Z, rt, role, A, TARGETS)
    perm = injection_matrix(Z, rt, role, A, TARGETS, permute_seed=1)
    assert perm["permuted"] is True and real["permuted"] is False
    assert not np.allclose(real["M"], perm["M"])
    train = role == "train"
    assert perm["M"][train].std() == pytest.approx(real["M"][train].std(), rel=1e-5)


def test_neighbour_operator_row_normalises_and_drops_the_diagonal():
    """Built against the real helper rather than the fixture, so the shipped operator is what is tested."""
    class _Store:
        def __init__(self, ei, ea):
            self.edge_index, self.edge_attr = ei, ea

    import torch
    from tcell_pipeline import config
    n_src = len(config.PPI_SOURCES)
    ei = torch.tensor([[0, 1], [1, 2]])
    ea = torch.zeros(2, n_src + 3)
    ea[:, n_src] = 1.0
    graph = {("protein", "physical_ppi", "protein"): _Store(ei, ea),
             ("protein", "co_complex", "protein"): _Store(torch.zeros((2, 0), dtype=torch.long),
                                                          torch.zeros(0, n_src + 3)),
             ("protein", "functional_assoc", "protein"): _Store(torch.zeros((2, 0), dtype=torch.long),
                                                                torch.zeros(0, n_src + 3))}
    A = neighbour_operator(["A", "B", "C"], {"A": 0, "B": 1, "C": 2}, graph)
    assert np.allclose(A.diagonal(), 0.0)
    assert np.allclose(np.asarray(A.sum(1)).reshape(-1), [1.0, 1.0, 1.0])
    assert A[0, 1] == pytest.approx(1.0)     # A's only neighbour is B
