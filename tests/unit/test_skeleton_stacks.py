"""The skeleton scan stacks the chunks of many pairs into one evaluation, with unchanged results.

For FisherZ, StudentT, RCoT, and the degenerate Gaussian test, ``scan_pairs`` evaluates the
chunks of consecutive pairs together, up to ``_STACK_BYTES`` of correlation submatrices. Every
cap, from one chunk per stack to the default, gives the removals, separating sets, largest
p-values, and trace of the per-pair scan, which a subclass of each test takes; a query that raises
raises the per-pair scan's error after the same trace; a test with only the scalar call gives the
degenerate Gaussian test's results too. The size-1 pass, ``first_order_removals``, stacks the same
way and prepares its queries only as the stacks need them; without a trace, a test that takes
canonical query arrays gets its queries from arrays in blocks of pairs, with the removals, p-values,
and errors of the pair-by-pair pass bit for bit. ``subset_rows`` yields a neighborhood that fits in
one chunk as one table lookup, with the combinations of the block-by-block path.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from andrey.constraint import _skeleton
from andrey.constraint._skeleton import (
    _combination_table,
    _members,
    canonical_capable,
    first_order_removals,
    pair_queries,
    scan_pairs,
    subset_rows,
)
from andrey.constraint.pc import adjacency_search
from andrey.core.ci import (
    DGTest,
    FisherZ,
    NormalScores,
    SingularCorrelationError,
    StudentT,
    make_indep_test,
)
from andrey.core.ci_kernel import RCoT

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("skip_within", [None, {2, 5, 9}, {2, 5, 9, 11, 12}, {2}])
def test_subset_rows_are_the_combinations_in_order_in_full_chunks(
    monkeypatch: pytest.MonkeyPatch, size: int, skip_within: set[int] | None
) -> None:
    monkeypatch.setattr(_skeleton, "CHUNK", 7)
    candidates = [2, 3, 5, 8, 9, 11, 12, 15]
    want = list(combinations(candidates, size))
    if skip_within is not None and len(skip_within) >= size:
        want = [s for s in want if not skip_within.issuperset(s)]
    chunks = list(subset_rows(candidates, size, skip_within))
    assert all(c.shape == (7, size) for c in chunks[:-1])
    assert all(len(c) > 0 for c in chunks)
    assert [tuple(row) for c in chunks for row in c.tolist()] == want


class _PerPairFisherZ(FisherZ):
    """A FisherZ subclass, which ``scan_pairs`` scans pair by pair."""


class _PerPairStudentT(StudentT):
    """A StudentT subclass, which ``scan_pairs`` scans pair by pair."""


class _PerPairRCoT(RCoT):
    """An RCoT subclass, which ``scan_pairs`` scans pair by pair."""


TESTS = [(FisherZ, _PerPairFisherZ), (StudentT, _PerPairStudentT)]


def _case(seed: int, d: int = 12, n: int = 300, p: float = 0.6):
    """A dense random adjacency over a sample with a few linear dependencies."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.7 * X[:, 1]
    X[:, 3] += 0.6 * X[:, 0] + 0.5 * X[:, 2]
    upper = np.triu(rng.random((d, d)) < p, 1)
    adj = upper | upper.T
    neighbors = [np.flatnonzero(adj[i]) for i in range(d)]
    pairs = [(x, y) for x in range(d) for y in range(x + 1, d) if adj[x, y]]
    return X, pairs, neighbors


def _scan(test, pairs, neighbors, size, record_pvalues=True):
    trace: list[dict] = []
    removals = scan_pairs(test, pairs, neighbors, size, ALPHA, trace, record_pvalues)
    return removals, trace


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("cap", [1, 4000, None])
@pytest.mark.parametrize("chunk", [2, None])
@pytest.mark.parametrize("size", [0, 1, 2, 3])
@pytest.mark.parametrize("seed", range(3))
def test_stacks_match_the_per_pair_scan(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, cap, chunk, size, seed
) -> None:
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    if chunk is not None:
        monkeypatch.setattr(_skeleton, "CHUNK", chunk)
    X, pairs, neighbors = _case(seed)
    for record_pvalues in (True, False):
        got = _scan(stacked(X), pairs, neighbors, size, record_pvalues)
        want = _scan(per_pair(X), pairs, neighbors, size, record_pvalues)
        assert got == want
    assert got[0], "the case must remove something"


def test_a_stack_holds_many_pairs_within_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    size, limit = 2, 200
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * (size + 2) ** 2)
    X, pairs, neighbors = _case(0)
    test = FisherZ(X)
    rows: list[int] = []
    canonical = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return canonical(x, y, conditions)

    test._canonical_pvalues = spy
    scan_pairs(test, pairs, neighbors, size, ALPHA)
    assert sum(rows) == sum(pair_queries(neighbors, x, y, size) for x, y in pairs)
    assert max(rows) <= limit
    assert len(rows) < len(pairs), "a stack must hold the chunks of several pairs"


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("cap", [1, None])
def test_a_singular_stack_raises_the_per_pair_error(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, cap
) -> None:
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    d = 10
    X = np.random.default_rng(4).standard_normal((200, d))
    X[:, -1] = X[:, -3] + X[:, -2]  # a query over the last three columns is singular
    pairs = [(x, y) for x in range(d) for y in range(x + 1, d)]
    neighbors = [np.array([j for j in range(d) if j != i]) for i in range(d)]
    errors, traces = [], []
    for test in (stacked(X), per_pair(X)):
        trace: list[dict] = []
        with pytest.raises(SingularCorrelationError) as raised:
            scan_pairs(test, pairs, neighbors, 2, ALPHA, trace)
        errors.append((type(raised.value), str(raised.value)))
        traces.append(trace)
    assert errors[0] == errors[1]
    assert traces[0] == traces[1]
    assert traces[0], "the singular query must come after other pairs' tests"


def _first_order(test, pairs, neighbors, record_pvalues=True):
    trace: list[dict] = []
    removals = first_order_removals(test, pairs, neighbors, ALPHA, trace, record_pvalues)
    return removals, trace


def _candidate_rows(pairs, neighbors):
    """The first-order queries of ``pairs``: either endpoint's neighbors, less the pair."""
    return sum(len((set(neighbors[x]) | set(neighbors[y])) - {x, y}) for x, y in pairs)


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("cap", [1, 4000, None])
@pytest.mark.parametrize("seed", range(3))
def test_first_order_stacks_match_the_per_pair_pass(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, cap, seed
) -> None:
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, pairs, neighbors = _case(seed)
    for record_pvalues in (True, False):
        got = _first_order(stacked(X), pairs, neighbors, record_pvalues)
        want = _first_order(per_pair(X), pairs, neighbors, record_pvalues)
        assert got == want
    assert got[0], "the case must remove something"
    assert len(got[1]) == _candidate_rows(pairs, neighbors)


def test_a_first_order_stack_holds_many_pairs_within_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 100
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * 3**2)
    X, pairs, neighbors = _case(0)
    test = FisherZ(X)
    rows: list[int] = []
    canonical = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return canonical(x, y, conditions)

    test._canonical_pvalues = spy
    first_order_removals(test, pairs, neighbors, ALPHA)
    assert sum(rows) == _candidate_rows(pairs, neighbors)
    assert max(rows) <= limit
    assert len(rows) < len(pairs), "a stack must hold the queries of several pairs"


class _Watched(FisherZ):
    """A FisherZ subclass whose first-order calls report how many rows were prepared before each."""

    def __init__(self, X, prepared):
        super().__init__(X)
        self.prepared = prepared
        self.waiting: list[int] = []
        self.evaluated = 0

    def first_order_pvalues(self, x, y, zs):
        zs = list(zs)
        self.waiting.append(self.prepared() - self.evaluated)
        self.evaluated += len(zs)
        return super().first_order_pvalues(x, y, zs)


@pytest.mark.parametrize("stacked", [True, False])
def test_first_order_queries_are_prepared_as_they_are_evaluated(
    monkeypatch: pytest.MonkeyPatch, stacked: bool
) -> None:
    """Rows built but not yet evaluated stay within one stack and one pair, whatever the pairs."""
    limit, d = 100, 30
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * 3**2)
    prepared = 0
    real = _skeleton._first_order_queries

    def counting(pairs, neighbors, test=None):
        nonlocal prepared
        for query in real(pairs, neighbors, test):
            prepared += len(query[2])
            yield query

    monkeypatch.setattr(_skeleton, "_first_order_queries", counting)
    X, pairs, neighbors = _case(1, d=d, p=1.0)
    total = _candidate_rows(pairs, neighbors)
    assert total > 20 * limit
    if stacked:
        test = FisherZ(X)
        rows: list[int] = []
        canonical = test._canonical_pvalues
        waiting: list[int] = []

        def spy(x, y, conditions):
            waiting.append(prepared - sum(rows))
            rows.append(len(x))
            return canonical(x, y, conditions)

        test._canonical_pvalues = spy
        # A trace keeps the pair-by-pair pass, which builds its queries as it evaluates them.
        first_order_removals(test, pairs, neighbors, ALPHA, [])
        assert prepared == sum(rows) == total
        assert max(waiting) <= limit + d - 2
    else:
        test = _Watched(X, lambda: prepared)
        first_order_removals(test, pairs, neighbors, ALPHA)
        assert test.evaluated == total
        assert max(test.waiting) <= d - 2


def test_canonical_capable_is_the_exact_type_rule() -> None:
    X, _, _ = _case(0)

    class Scalar:
        def __call__(self, x, y, condition_set=()):
            return 1.0

    assert canonical_capable(FisherZ(X)) and canonical_capable(StudentT(X))
    assert not canonical_capable(_PerPairFisherZ(X))
    assert not canonical_capable(_PerPairStudentT(X))
    assert not canonical_capable(Scalar())  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("cap", [1, None])
@pytest.mark.parametrize("size", [0, 1, 2])
def test_rcot_stacks_match_the_per_pair_scan(monkeypatch: pytest.MonkeyPatch, cap, size) -> None:
    """RCoT's stacks run each conditioning set's queries together; every p-value, removal, and
    trace record is the per-pair scan's, bit for bit."""
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, pairs, neighbors = _case(0, d=8, n=500)
    test, rows = RCoT(X), []
    grouped = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return grouped(x, y, conditions)

    test._canonical_pvalues = spy
    got = _scan(test, pairs, neighbors, size)
    want = _scan(_PerPairRCoT(X), pairs, neighbors, size)
    assert got == want
    assert got[0], "the case must remove something"
    assert sum(rows) == sum(pair_queries(neighbors, x, y, size) for x, y in pairs)


def test_a_failing_rcot_stack_raises_the_per_pair_error(monkeypatch: pytest.MonkeyPatch) -> None:
    X, pairs, neighbors = _case(0, d=8, n=500)
    failing = pairs[len(pairs) // 2]
    call = RCoT.__call__

    def raising(test, x, y, condition_set=()):
        if (min(x, y), max(x, y)) == failing:
            raise FloatingPointError(f"query {x}, {y}, {tuple(condition_set)}")
        return call(test, x, y, condition_set)

    monkeypatch.setattr(RCoT, "__call__", raising)
    errors, traces = [], []
    for test in (RCoT(X), _PerPairRCoT(X)):
        trace: list[dict] = []
        with pytest.raises(FloatingPointError) as raised:
            scan_pairs(test, pairs, neighbors, 2, ALPHA, trace)
        errors.append(str(raised.value))
        traces.append(trace)
    assert errors[0] == errors[1]
    assert traces[0] == traces[1]
    assert traces[0], "the failing query must come after other pairs' tests"


class _PerPairDGTest(DGTest):
    """A degenerate Gaussian subclass, which ``scan_pairs`` scans pair by pair."""


class _ScalarOnly:
    """A test with only the scalar call, which ``scan_pairs`` evaluates one query at a time."""

    def __init__(self, test) -> None:
        self._test = test

    def __call__(self, x, y, condition_set=()):
        return DGTest.__call__(self._test, x, y, condition_set)


def _mixed_case(seed: int = 0):
    """``_case`` with every third column cut into two to four categories."""
    X, pairs, neighbors = _case(seed, d=10, n=400)
    mask = np.zeros(X.shape[1], dtype=bool)
    for j in range(0, X.shape[1], 3):
        X[:, j] = np.digitize(X[:, j], np.quantile(X[:, j], np.linspace(0, 1, 2 + j % 3)[1:-1]))
        mask[j] = True
    return X, mask, pairs, neighbors


@pytest.mark.parametrize("cap", [1, None])
@pytest.mark.parametrize("size", [0, 1, 2, 3])
def test_dg_stacks_match_the_per_pair_and_the_scalar_scans(
    monkeypatch: pytest.MonkeyPatch, cap, size
) -> None:
    """The degenerate Gaussian test's stacks give every p-value, removal, and trace record of the
    per-pair scan, which calls its batched path pair by pair, and of a test with only the scalar
    call, bit for bit."""
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, mask, pairs, neighbors = _mixed_case()
    test, rows = DGTest(X, mask), []
    grouped = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return grouped(x, y, conditions)

    test._canonical_pvalues = spy
    got = _scan(test, pairs, neighbors, size)
    assert got == _scan(_PerPairDGTest(X, mask), pairs, neighbors, size)
    assert got == _scan(_ScalarOnly(DGTest(X, mask)), pairs, neighbors, size)
    assert got[0], "the case must remove something"
    assert sum(rows) == sum(pair_queries(neighbors, x, y, size) for x, y in pairs)
    if cap is None:
        assert len(rows) < len(pairs), "a stack must hold the chunks of several pairs"


def test_a_failing_dg_stack_raises_the_per_pair_error(monkeypatch: pytest.MonkeyPatch) -> None:
    X, mask, pairs, neighbors = _mixed_case()
    failing = pairs[len(pairs) // 2]
    call, stacked = DGTest.__call__, DGTest._stacked_pvalues

    def raising(test, x, y, condition_set=()):
        if (min(x, y), max(x, y)) == failing:
            raise FloatingPointError(f"query {x}, {y}, {tuple(condition_set)}")
        return call(test, x, y, condition_set)

    def failing_stack(test, xs, ys, conds):
        if failing in zip(xs, ys):
            raise FloatingPointError("stack")
        return stacked(test, xs, ys, conds)

    monkeypatch.setattr(DGTest, "__call__", raising)
    monkeypatch.setattr(DGTest, "_stacked_pvalues", failing_stack)
    errors, traces = [], []
    for test in (DGTest(X, mask), _ScalarOnly(DGTest(X, mask))):
        trace: list[dict] = []
        with pytest.raises(FloatingPointError) as raised:
            scan_pairs(test, pairs, neighbors, 2, ALPHA, trace)
        errors.append(str(raised.value))
        traces.append(trace)
    assert errors[0] == errors[1]
    assert traces[0] == traces[1]
    assert traces[0], "the failing query must come after other pairs' tests"


class _PerPairNormalScores(NormalScores):
    """A NormalScores subclass, which the size-1 pass takes pair by pair."""


ARRAY_TESTS = [*TESTS, (NormalScores, _PerPairNormalScores)]


def _sparse_case(seed: int, d: int = 40, n: int = 400, degree: float = 2.0, lopsided: bool = False):
    """``_case`` over a wide, sparse adjacency, so some pairs have no candidate. Some adjacent pairs
    are left out of ``pairs``, and still condition the others; with ``lopsided``, each neighbor
    list also loses entries that the other endpoint keeps."""
    rng = np.random.default_rng(seed)
    X, pairs, neighbors = _case(seed, d=d, n=n, p=degree / d)
    if lopsided:
        neighbors = [nb[rng.random(len(nb)) < 0.7] for nb in neighbors]
    tested = rng.random(len(pairs)) < 0.8
    return X, [pair for pair, keep in zip(pairs, tested) if keep], neighbors


@pytest.fixture
def array_pass(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """For each call of the array-built size-1 pass, whether it answered (``False``: it fell back,
    answering ``None``)."""
    answered: list[bool] = []
    real = _skeleton._array_first_order

    def watching(*args, **kwargs):
        found = real(*args, **kwargs)
        answered.append(found is not None)
        return found

    monkeypatch.setattr(_skeleton, "_array_first_order", watching)
    return answered


@pytest.mark.parametrize(("stacked", "per_pair"), ARRAY_TESTS)
@pytest.mark.parametrize("cap", [1, 4000, None])
@pytest.mark.parametrize(
    ("entries", "factor"),
    [(1, None), (200, 0), (200, 1 << 40), (None, None)],
    ids=["one-pair-blocks", "listed", "mask", "default"],
)
@pytest.mark.parametrize("case", ["dense", "sparse", "lopsided"])
def test_the_array_first_order_pass_matches_the_pair_by_pair_pass(
    monkeypatch: pytest.MonkeyPatch,
    array_pass: list[bool],
    stacked,
    per_pair,
    cap,
    entries,
    factor,
    case,
) -> None:
    """Whatever the block size, the stack size, and the candidate builder, the removals, their
    separating sets, and their largest p-values equal the pair-by-pair passes': the stacked one,
    which a trace keeps, and one ``first_order_pvalues`` call per pair."""
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    if entries is not None:
        monkeypatch.setattr(_skeleton, "_FIRST_ORDER_MASK_ENTRIES", entries)
    if factor is not None:
        monkeypatch.setattr(_skeleton, "_LISTED_FACTOR", factor)
    for seed in range(2):
        if case == "dense":
            X, pairs, neighbors = _case(seed)
        else:
            X, pairs, neighbors = _sparse_case(seed, lopsided=case == "lopsided")
        for record_pvalues in (True, False):
            got = first_order_removals(stacked(X), pairs, neighbors, ALPHA, None, record_pvalues)
            by_stack, _ = _first_order(stacked(X), pairs, neighbors, record_pvalues)
            by_pair = first_order_removals(
                per_pair(X), pairs, neighbors, ALPHA, None, record_pvalues
            )
            assert got == by_stack == by_pair
            assert got, "the case must remove something"
    assert array_pass and all(array_pass)


@pytest.mark.parametrize("cap", [1, None])
@pytest.mark.parametrize("record_pvalues", [True, False])
def test_the_array_first_order_pass_matches_for_a_canonical_test_without_the_prescreen(
    monkeypatch: pytest.MonkeyPatch, array_pass: list[bool], cap, record_pvalues
) -> None:
    """The degenerate Gaussian test takes canonical query arrays but has no first-order call, so
    the skeleton search never gives it this pass; called directly, the array pass returns the
    stacked pass's removals."""
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, mask, pairs, neighbors = _mixed_case()
    got = first_order_removals(DGTest(X, mask), pairs, neighbors, ALPHA, None, record_pvalues)
    want, _ = _first_order(DGTest(X, mask), pairs, neighbors, record_pvalues)
    assert got == want
    assert got, "the case must remove something"
    assert array_pass == [True]


@pytest.mark.parametrize("builder", ["listed", "mask"])
def test_a_block_takes_the_neighbor_lists_when_they_are_short_against_the_mask(
    monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    """A block lists its candidates from the neighbor lists when eight times their entries are
    fewer than the mask's, and from the mask otherwise; the removals are the same."""
    used: list[str] = []
    for name in ("_listed_candidates", "_mask_candidates"):
        real = getattr(_skeleton, name)

        def watching(*args, _name=name, _real=real):
            used.append(_name.split("_")[1])
            return _real(*args)

        monkeypatch.setattr(_skeleton, name, watching)
    if builder == "listed":
        X, pairs, neighbors = _sparse_case(0, d=120, n=300, degree=1.5)
    else:
        X, pairs, neighbors = _case(0)
    got = first_order_removals(FisherZ(X), pairs, neighbors, ALPHA)
    assert got == first_order_removals(_PerPairFisherZ(X), pairs, neighbors, ALPHA)
    assert set(used) == {builder}


def test_the_two_candidate_builders_list_the_same_queries() -> None:
    """Pair by pair and candidates ascending, each pair's candidates its two endpoints' neighbors
    less the pair, whatever the lists, including lists that lose entries the other keeps."""
    d = 50
    _, pairs, neighbors = _sparse_case(3, d=d, degree=4.0, lopsided=True)
    adjacent = np.zeros((d, d), dtype=bool)
    for i, nb in enumerate(neighbors):
        adjacent[i, nb] = True
    sizes = np.array([nb.size for nb in neighbors])
    flat = np.concatenate(neighbors)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    both = np.array(pairs)
    assert len(both) > 20
    for block in (slice(None), slice(0, 7), slice(0, 1)):
        xs, ys = both[block, 0], both[block, 1]
        want = [
            (i, z)
            for i, (x, y) in enumerate(zip(xs.tolist(), ys.tolist()))
            for z in sorted((set(neighbors[x]) | set(neighbors[y])) - {x, y})
        ]
        assert want or block != slice(None)
        for which, zs in (
            _skeleton._listed_candidates(flat, offsets, sizes, xs, ys, d),
            _skeleton._mask_candidates(adjacent, xs, ys),
        ):
            assert which.dtype == zs.dtype == np.intp
            assert list(zip(which.tolist(), zs.tolist())) == want
    none = np.zeros(d, dtype=np.intp)
    xs, ys = both[:3, 0], both[:3, 1]
    for which, zs in (
        _skeleton._listed_candidates(np.empty(0, dtype=np.intp), none, none, xs, ys, d),
        _skeleton._mask_candidates(np.zeros((d, d), dtype=bool), xs, ys),
    ):
        assert not which.size
        assert not zs.size


def test_the_array_first_order_pass_holds_one_block_and_one_stack(
    monkeypatch: pytest.MonkeyPatch, array_pass: list[bool]
) -> None:
    """Whatever the pairs, a block's mask holds at most the entries the block size allows, and one
    evaluation at most one stack of rows."""
    limit, d, entries = 100, 30, 8 * 30
    monkeypatch.setattr(_skeleton, "_STACK_BYTES", 8 * limit * 3**2)
    monkeypatch.setattr(_skeleton, "_FIRST_ORDER_MASK_ENTRIES", entries)
    masks: list[int] = []
    real = _skeleton._mask_candidates

    def counting(adjacent, xs, ys):
        masks.append(len(xs) * adjacent.shape[1])
        return real(adjacent, xs, ys)

    monkeypatch.setattr(_skeleton, "_mask_candidates", counting)
    X, pairs, neighbors = _case(1, d=d, p=1.0)
    total = _candidate_rows(pairs, neighbors)
    assert total > 20 * limit
    test = FisherZ(X)
    rows: list[int] = []
    canonical = test._canonical_pvalues

    def spy(x, y, conditions):
        rows.append(len(x))
        return canonical(x, y, conditions)

    test._canonical_pvalues = spy
    first_order_removals(test, pairs, neighbors, ALPHA)
    assert sum(rows) == total
    assert max(rows) == limit
    assert len(masks) == -(-len(pairs) // (entries // d))
    assert max(masks) <= entries
    assert array_pass == [True]


@pytest.mark.parametrize("record_pvalues", [True, False])
def test_only_a_p_value_above_alpha_separates_and_a_not_tested_query_never_does(
    array_pass: list[bool], record_pvalues: bool
) -> None:
    """A p-value of exactly ``alpha`` and a NaN, a query not tested, leave the pair connected, and
    the largest p-value of a pair is read from its separating queries only."""
    X, pairs, neighbors = _case(0)
    results = []
    for trace in (None, []):
        test = FisherZ(X)
        canonical = test._canonical_pvalues

        def patched(x, y, conditions, _canonical=canonical):
            values = _canonical(x, y, conditions)
            values[(x + y + conditions[:, 0]) % 5 == 0] = ALPHA
            values[(x + y + conditions[:, 0]) % 7 == 0] = np.nan
            return values

        test._canonical_pvalues = patched
        results.append(first_order_removals(test, pairs, neighbors, ALPHA, trace, record_pvalues))
    assert results[0] == results[1]
    assert results[0], "the case must remove something"
    assert all(p is None or p > ALPHA for *_, p in results[0])
    assert array_pass == [True]


@pytest.mark.parametrize("guard", ["trace", "subclass", "deletion"])
def test_a_pass_that_needs_the_pair_by_pair_route_builds_no_arrays(
    monkeypatch: pytest.MonkeyPatch, guard: str
) -> None:
    """A trace records every test pair by pair, a subclass keeps what it overrides, and a test that
    deletes rows per test asks only the sets some row observes."""
    calls: list[int] = []
    real = _skeleton._array_first_order

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_skeleton, "_array_first_order", counting)
    X, pairs, neighbors = _case(0)
    trace: list[dict] | None = None
    if guard == "trace":
        test, trace = FisherZ(X), []
    elif guard == "subclass":
        test = _PerPairFisherZ(X)
    else:
        missing = X.copy()
        missing[np.random.default_rng(1).random(X.shape) < 0.05] = np.nan
        test = make_indep_test("fisherz", missing)
    first_order_removals(test, pairs, neighbors, ALPHA, trace)
    assert not calls
    first_order_removals(FisherZ(X), pairs, neighbors, ALPHA)
    assert len(calls) == 1


@pytest.mark.parametrize("cap", [1 << 12, None])
def test_an_error_only_a_large_stack_raises_is_left_to_the_pair_by_pair_pass(
    monkeypatch: pytest.MonkeyPatch, array_pass: list[bool], cap
) -> None:
    """The array pass answers ``None`` when an evaluation raises. The pair-by-pair pass asks each
    pair alone after a stack fails, so it returns what a test that never fails returns."""
    if cap is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", cap)
    X, pairs, neighbors = _case(0)
    test = FisherZ(X)
    canonical = test._canonical_pvalues

    def failing(x, y, conditions):
        if len(x) > len(neighbors):
            raise FloatingPointError(f"a stack of {len(x)} queries")
        return canonical(x, y, conditions)

    test._canonical_pvalues = failing
    got = first_order_removals(test, pairs, neighbors, ALPHA)
    assert array_pass == [False]
    assert got
    want, _ = _first_order(FisherZ(X), pairs, neighbors)
    assert got == want


@pytest.mark.parametrize(("stacked", "per_pair"), ARRAY_TESTS)
def test_a_pass_with_no_degrees_of_freedom_raises_the_pair_by_pair_error(
    array_pass: list[bool], stacked, per_pair
) -> None:
    """Three observations leave a size-1 test no degree of freedom: the array pass falls back, and
    the error is the pair-by-pair pass's."""
    X, pairs, neighbors = _case(0)
    corr = np.corrcoef(X, rowvar=False)
    errors = []
    for trace in (None, []):
        with pytest.raises(ValueError, match="degrees of freedom") as raised:
            first_order_removals(stacked.from_corr(corr, 3), pairs, neighbors, ALPHA, trace)
        errors.append(str(raised.value))
    assert errors[0] == errors[1]
    assert array_pass == [False]


def test_a_query_that_raises_raises_the_pair_by_pair_error(array_pass: list[bool]) -> None:
    X, pairs, neighbors = _case(0)
    failing = pairs[len(pairs) // 2]
    errors = []
    for trace in (None, []):
        test = FisherZ(X)
        canonical = test._canonical_pvalues

        def raising(x, y, conditions, _canonical=canonical):
            if failing in zip(x.tolist(), y.tolist()):
                raise FloatingPointError(f"query {failing}")
            return _canonical(x, y, conditions)

        test._canonical_pvalues = raising
        with pytest.raises(FloatingPointError) as raised:
            first_order_removals(test, pairs, neighbors, ALPHA, trace)
        errors.append(str(raised.value))
    assert errors[0] == errors[1]
    assert array_pass == [False]


@pytest.mark.parametrize("name", ["fisherz", "student_t", "normal_scores"])
@pytest.mark.parametrize("record_pvalues", [True, False])
def test_the_skeleton_search_is_the_same_with_and_without_the_array_pass(
    monkeypatch: pytest.MonkeyPatch, array_pass: list[bool], name: str, record_pvalues: bool
) -> None:
    """The whole search, size by size, returns the marks, separating sets, and p-values, in the
    same order, that it returns when its size-1 pass goes pair by pair."""
    d = 30
    X = np.random.default_rng(5).standard_normal((600, d))
    for j in range(1, d):
        X[:, j] += 0.7 * X[:, int(j * 0.6)]
    keep = np.zeros((d, d), dtype=bool)
    keep[2, 3] = keep[3, 2] = True
    marks, sepsets, pvalues = adjacency_search(
        make_indep_test(name, X), d, 0.05, keep=keep, record_pvalues=record_pvalues
    )
    assert array_pass == [True]
    monkeypatch.setattr(_skeleton, "_array_first_order", lambda *args, **kwargs: None)
    want_marks, want_sepsets, want_pvalues = adjacency_search(
        make_indep_test(name, X), d, 0.05, keep=keep, record_pvalues=record_pvalues
    )
    assert sepsets, "the search must remove something"
    assert np.array_equal(marks, want_marks)
    assert list(sepsets.items()) == list(want_sepsets.items())
    assert list(pvalues.items()) == list(want_pvalues.items())
    assert bool(pvalues) == record_pvalues


def _want_rows(candidates, size, skip_within):
    rows = combinations([int(c) for c in candidates], size)
    if skip_within is None or len(skip_within) < size:
        return list(rows)
    return [row for row in rows if not all(c in skip_within for c in row)]


def test_subset_rows_in_one_chunk_are_the_combinations_in_order() -> None:
    """With the default chunk, subsets that fit in it are one array, equal to ``combinations`` less
    those inside the skip set, whatever the skip members are: outside the candidates, NumPy
    integers, all the candidates, or too few to skip."""
    rng = np.random.default_rng(12345)
    one_chunk = 0
    for _ in range(600):
        n, size = int(rng.integers(0, 30)), int(rng.integers(0, 8))
        pool = int(rng.integers(max(n, 1), 400))
        candidates = sorted(rng.choice(pool, n, replace=False).tolist())
        members = rng.choice(pool + 10, int(rng.integers(0, n + 6)), replace=False).tolist()
        inside = (
            set(rng.choice(candidates, int(rng.integers(0, n + 1)), replace=False).tolist())
            if n
            else set()
        )
        skip_within = [
            None,
            inside,
            set(members),
            {np.int64(c) for c in members},
            set(candidates),
            set(candidates[: max(size - 1, 0)]),
        ][int(rng.integers(0, 6))]
        given = np.asarray(candidates, dtype=np.int64) if rng.random() < 0.3 else candidates
        chunks = list(subset_rows(given, size, skip_within))
        want = _want_rows(candidates, size, skip_within)
        assert [tuple(row) for chunk in chunks for row in chunk.tolist()] == want
        for chunk in chunks:
            assert chunk.shape[1] == size
            assert len(chunk)
            assert chunk.dtype == np.intp
            assert chunk.flags.c_contiguous
        if 1 < size <= n and comb(n, size) <= _skeleton.CHUNK:
            assert len(chunks) == (1 if want else 0)
            one_chunk += 1
    assert one_chunk > 100, "most cases must take the one-chunk path"


@pytest.mark.parametrize("fewer", [1, 0, -1])
def test_subset_rows_split_where_the_subsets_pass_the_chunk_size(
    monkeypatch: pytest.MonkeyPatch, fewer: int
) -> None:
    """Exactly a chunk of subsets is one array; one more is a full chunk and one row."""
    n, size = 9, 3
    total = comb(n, size)
    monkeypatch.setattr(_skeleton, "CHUNK", total - fewer)
    chunks = list(subset_rows(list(range(n)), size, None))
    assert [len(chunk) for chunk in chunks] == ([total - 1, 1] if fewer == 1 else [total])
    assert [tuple(row) for chunk in chunks for row in chunk.tolist()] == list(
        combinations(range(n), size)
    )


@pytest.mark.parametrize("size", [2, 3, 4])
def test_writing_into_a_subset_chunk_does_not_reach_the_cached_table(size: int) -> None:
    candidates = [4, 6, 9, 10, 13, 17]
    table = _combination_table(len(candidates), size).copy()
    for skip_within in (None, {4, 6, 9, 10}):
        for chunk in subset_rows(candidates, size, skip_within):
            chunk[...] = -7
    assert np.array_equal(_combination_table(len(candidates), size), table)
    assert [
        tuple(row) for chunk in subset_rows(candidates, size, None) for row in chunk.tolist()
    ] == list(combinations(candidates, size))


@pytest.mark.parametrize("size", [0, 1, 5, 6, 7])
def test_subset_rows_at_the_edges_of_the_candidate_count(size: int) -> None:
    candidates = [3, 5, 8, 9, 12, 14]
    for skip_within in (None, set(candidates), {3, 5}):
        got = [
            tuple(row)
            for chunk in subset_rows(candidates, size, skip_within)
            for row in chunk.tolist()
        ]
        assert got == _want_rows(candidates, size, skip_within)


@pytest.mark.parametrize("n", [1, 299, 300, 301, 700])
def test_a_skip_set_lookup_reads_the_same_members_either_side_of_the_switch(
    monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    rng = np.random.default_rng(n)
    labels = np.sort(rng.choice(5 * n, n, replace=False)).astype(np.intp)
    members = {int(c) for c in rng.choice(labels, n // 2, replace=False)} | {10**6, -3}
    members |= {np.int64(c) for c in rng.choice(labels, n // 4, replace=False)}
    want = np.array([c in members for c in labels.tolist()])
    for switch in (0, 300, 10**6):
        monkeypatch.setattr(_skeleton, "_SET_LOOKUP_LABELS", switch)
        got = _members(labels, members)
        assert got.dtype == bool
        assert np.array_equal(got, want)
