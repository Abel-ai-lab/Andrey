"""GFCI's separating-set search takes the canonical query path with unchanged results.

For FisherZ and StudentT, ``_derive_sepsets`` answers the size-0 queries of all the pairs in one
canonical query array and evaluates each nonempty size's subsets of all the pairs still unseparated
in stacks of them. It asks the queries of the scalar loop, which a subclass of either test takes,
and records the same separating sets; a singular query raises the scalar loop's error, after the
tests of the other pairs.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from andrey.constraint import _skeleton
from andrey.core import TAIL
from andrey.core.ci import FisherZ, SingularCorrelationError, StudentT
from andrey.search import gfci
from andrey.search.gfci import _derive_sepsets

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


class _ScalarFisherZ(FisherZ):
    """A FisherZ subclass, which ``_derive_sepsets`` queries one call at a time."""


class _ScalarStudentT(StudentT):
    """A StudentT subclass, which ``_derive_sepsets`` queries one call at a time."""


TESTS = [(FisherZ, _ScalarFisherZ), (StudentT, _ScalarStudentT)]


def _case(seed: int, d: int = 12, n: int = 300, p: float = 0.35):
    """A random CPDAG adjacency over a sample with a few linear dependencies."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.7 * X[:, 1]
    X[:, 3] += 0.6 * X[:, 0] + 0.5 * X[:, 2]
    X[:, 4] += 0.5 * X[:, 3]
    upper = np.triu(rng.random((d, d)) < p, 1)
    cpdag = np.where(upper | upper.T, np.int8(TAIL), np.int8(0))
    return X, cpdag


def _logged(test):
    """``test`` with every query it answers appended to the returned log, as ``(x, y, set)``, and
    the conditioning-set size of each canonical call to the returned list of sizes."""
    log: list[tuple[int, int, tuple[int, ...]]] = []
    batches: list[int] = []
    scalar, canonical = test._partial_correlation, test._canonical_pvalues

    def partial_correlation(x, y, cond):
        log.append((x, y, tuple(cond)))
        return scalar(x, y, cond)

    def canonical_pvalues(xs, ys, conditions):
        values = canonical(xs, ys, conditions)
        log.extend(zip(xs.tolist(), ys.tolist(), map(tuple, conditions.tolist())))
        batches.append(conditions.shape[1])
        return values

    test._partial_correlation = partial_correlation
    test._canonical_pvalues = canonical_pvalues
    return test, log, batches


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
@pytest.mark.parametrize("chunk", [2, None])
@pytest.mark.parametrize("max_depth", [None, 1])
@pytest.mark.parametrize("in_column_order", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_canonical_queries_match_the_scalar_loop(
    monkeypatch: pytest.MonkeyPatch, canonical, scalar, chunk, max_depth, in_column_order, seed
) -> None:
    if chunk is not None:
        monkeypatch.setattr(_skeleton, "CHUNK", chunk)
    X, cpdag = _case(seed)
    results = []
    for kind in (canonical, scalar):
        test, log, _ = _logged(kind(X))
        sepsets = _derive_sepsets(
            cpdag, test, ALPHA, max_depth=max_depth, in_column_order=in_column_order
        )
        results.append((sepsets, sorted(log)))
    assert results[0] == results[1]
    assert any(len(cond) >= 2 for _, _, cond in results[0][1]) or max_depth == 1


@pytest.mark.parametrize("seed", range(3))
def test_the_size_zero_queries_come_in_one_canonical_call(seed) -> None:
    X, cpdag = _case(seed)
    test, log, batches = _logged(FisherZ(X))
    _derive_sepsets(cpdag, test, ALPHA)
    assert batches.count(0) == 1
    assert sum(cond == () for _, _, cond in log) == np.count_nonzero(np.triu(cpdag == 0, 1))


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
def test_a_singular_query_raises_the_scalar_error(canonical, scalar) -> None:
    d = 8
    X = np.random.default_rng(4).standard_normal((200, d))
    X[:, 5] = X[:, 0] + X[:, 1]  # (3, 4) given {0, 1, 5}, after {0, 1, 2}, is singular
    cpdag = np.zeros((d, d), dtype=np.int8)
    for u in (0, 1, 2, 5):
        for v in (3, 4):
            cpdag[u, v] = cpdag[v, u] = TAIL
    errors, logs = [], []
    for kind in (canonical, scalar):
        test, log, _ = _logged(kind(X))
        with pytest.raises(SingularCorrelationError) as raised:
            _derive_sepsets(cpdag, test, 1.0)
        errors.append((type(raised.value), str(raised.value)))
        logs.append(log)
    assert errors[0] == errors[1]
    assert logs[1], "the singular query must come after other queries"
    # The stacked search asks the other pairs' queries of the sizes below the singular query's
    # (size 3) before the stack that holds it raises, and the pair-by-pair search then asks the
    # scalar loop's queries again; no larger query is asked.
    singular = 3
    extra = Counter(logs[0]) - Counter(logs[1])
    assert not Counter(logs[1]) - Counter(logs[0])
    assert all(len(cond) < singular for _, _, cond in extra)
    assert any(cond and (x, y) != (3, 4) for x, y, cond in extra)


def _refuse_pair_by_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the pair-by-pair search raise, so a search that completes ran on the stacked route."""

    def refuse(*args, **kwargs):
        raise AssertionError("the pair-by-pair search ran")

    monkeypatch.setattr(gfci, "separating_subsets", refuse)


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
@pytest.mark.parametrize(("stack_bytes", "chunk"), [(1, 2), (2048, 5), (None, None)])
@pytest.mark.parametrize("max_depth", [None, 1, 2])
@pytest.mark.parametrize("seed", range(3))
def test_the_stacked_search_records_the_pair_by_pair_sets_in_the_same_order(
    monkeypatch: pytest.MonkeyPatch, canonical, scalar, stack_bytes, chunk, max_depth, seed
) -> None:
    """Every cap on a stack and on a chunk gives the separating sets of the pair-by-pair search,
    the keys in its order."""
    X, cpdag = _case(seed)
    want = _derive_sepsets(cpdag, scalar(X), ALPHA, max_depth=max_depth)
    if stack_bytes is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", stack_bytes)
        monkeypatch.setattr(_skeleton, "CHUNK", chunk)
    _refuse_pair_by_pair(monkeypatch)
    got = _derive_sepsets(cpdag, canonical(X), ALPHA, max_depth=max_depth)
    assert list(got.items()) == list(want.items())
    assert any(len(found) >= 2 for found in want.values()) or max_depth == 1


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
def test_every_stacked_p_value_is_the_scalar_one(
    monkeypatch: pytest.MonkeyPatch, canonical, scalar
) -> None:
    X, cpdag = _case(1)
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 4096)
    test = canonical(X)
    inner = test._canonical_pvalues
    asked: list[tuple[int, int, tuple[int, ...], float]] = []

    def values(xs, ys, conditions):
        found = inner(xs, ys, conditions)
        asked.extend(zip(xs.tolist(), ys.tolist(), map(tuple, conditions.tolist()), found.tolist()))
        return found

    test._canonical_pvalues = values
    _refuse_pair_by_pair(monkeypatch)
    _derive_sepsets(cpdag, test, ALPHA)
    reference = scalar(X)
    assert any(len(cond) >= 2 for _, _, cond, _ in asked)
    assert all(p == reference(x, y, cond) for x, y, cond, p in asked)


@pytest.mark.parametrize(("canonical", "scalar"), TESTS)
def test_a_search_whose_stacks_raise_searches_pair_by_pair(canonical, scalar) -> None:
    """A test whose canonical call raises for the sets of a stack gives the scalar loop's sets."""
    X, cpdag = _case(0)
    want = _derive_sepsets(cpdag, scalar(X), ALPHA)
    test = canonical(X)
    inner = test._canonical_pvalues

    def refuse_nonempty(xs, ys, conditions):
        if conditions.shape[1]:
            raise FloatingPointError("stack")
        return inner(xs, ys, conditions)

    test._canonical_pvalues = refuse_nonempty
    got = _derive_sepsets(cpdag, test, ALPHA)
    assert list(got.items()) == list(want.items())
    assert any(len(found) for found in want.values())
