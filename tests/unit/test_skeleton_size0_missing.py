"""The size-0 scan of the skeleton search under test-wise deletion reads ``together()`` once.

A pair that no row observes together is counted as one query not tested, and every other pair
yields the empty set as its one chunk. These tests pin the chunks, the counter, the removals, and
the error against the generic per-pair chunk generator (:func:`_candidate_chunks`, which every other
size and test uses), on data with blocks of columns that never meet and pairs observed together on
two or three rows, at every stack size.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import andrey
from andrey.constraint import _skeleton
from andrey.constraint._skeleton import scan_pairs
from andrey.core.ci import CITest, FisherZ, make_indep_test
from andrey.core.ci_missing import GaussianDeletion, reading_flags, rows_tested
from andrey.data import sample_scm

ALPHA = 0.05
CAPS = [1, 64, 96, 1000, None]


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _blocks(seed: int, d: int = 14, n: int = 400) -> np.ndarray:
    """A linear-Gaussian table where columns 0-2 and 3-5 are never observed together, columns 8 and
    9 are observed together on two rows and 10 and 11 on three, and 3% of the other entries are
    missing."""
    x = sample_scm(d=d, n=n, seed=seed, density=1.5).data.copy()
    half = n // 2
    x[:half, 0:3] = np.nan
    x[half:, 3:6] = np.nan
    x[half + 2 :, 8] = np.nan
    x[:half, 9] = np.nan
    x[half + 3 :, 10] = np.nan
    x[:half, 11] = np.nan
    scattered = np.random.default_rng(seed).random(x.shape) < 0.03
    scattered[:, 8:12] = False
    x[scattered] = np.nan
    return x


def _complete_graph(d: int, seed: int | None = None) -> tuple[list[tuple[int, int]], list]:
    """Every pair of ``d`` columns, ascending, or in a seeded order with a part of them reversed,
    and the neighbors of the complete graph."""
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    if seed is not None:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.permutation(len(pairs))]
        pairs = [(y, x) if rng.random() < 0.5 else (x, y) for x, y in pairs]
    return pairs, [np.delete(np.arange(d), i) for i in range(d)]


def _candidate_chunks(
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    size: int,
    test: CITest | None = None,
) -> Iterator[_skeleton._Chunk]:
    """The chunks of the per-pair scan, as :func:`_skeleton._pair_chunks` yields them for every
    size and test but size 0 under test-wise deletion: each side's candidates less what
    :func:`_skeleton.kept_apart` drops, subset by subset."""
    for i, (a, b) in enumerate(pairs):
        x, y = min(a, b), max(a, b)
        for side_a, side_b, candidates, seen in _skeleton._sides(neighbors, a, b, size):
            if test is not None:
                kept, _ = _skeleton.kept_apart(test, (side_a, side_b), [(candidates, seen)], size)
                if kept is None:
                    continue
                [(candidates, seen)] = kept
            for chunk in _skeleton.subset_rows(candidates, size, seen):
                yield i, x, y, chunk


def _state(counter) -> tuple:
    """Everything the running counter holds, each mapping in insertion order."""
    return (
        sorted(counter.items()),
        counter.p_one,
        list(counter.not_tested.items()),
        list(counter.largest.items()),
        [(pair, sorted(sorted(s) for s in sets)) for pair, sets in counter.untested_sets.items()],
        counter.not_tested_elsewhere,
    )


def _unordered(state: tuple) -> tuple:
    """:func:`_state` with each mapping sorted, for two scans that read the pairs in an order of
    their own."""
    counts, p_one, not_tested, largest, sets, elsewhere = state
    return counts, p_one, sorted(not_tested), sorted(largest), sorted(sets), elsewhere


@pytest.fixture
def marginal_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The pair counts of the calls to :func:`_skeleton._marginal_chunks`."""
    calls: list[int] = []
    original = _skeleton._marginal_chunks

    def watching(pairs, test):
        calls.append(len(pairs))
        return original(pairs, test)

    monkeypatch.setattr(_skeleton, "_marginal_chunks", watching)
    return calls


def _scan_size_0(
    monkeypatch: pytest.MonkeyPatch,
    x: np.ndarray,
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    *,
    generic: bool,
    cap: int | None = None,
    record_pvalues: bool = True,
    name: str = "fisherz",
) -> tuple[list, tuple]:
    """The removals of the size-0 scan of ``pairs`` and the counter it leaves, with the chunks of
    :func:`_candidate_chunks` when ``generic``."""
    with monkeypatch.context() as patch:
        if cap is not None:
            patch.setattr(_skeleton, "_STACK_BYTES", cap)
        if generic:
            patch.setattr(_skeleton, "_pair_chunks", _candidate_chunks)
        test = make_indep_test(name, x)
        assert isinstance(test, GaussianDeletion)
        with rows_tested() as counter, reading_flags():
            removals = scan_pairs(test, pairs, neighbors, 0, ALPHA, record_pvalues=record_pvalues)
    return removals, _state(counter)


@pytest.mark.parametrize("order", [None, 0, 1])
@pytest.mark.parametrize("seed", range(3))
def test_the_chunks_and_the_counts_interleave_as_the_per_pair_generator_does(seed, order):
    x = _blocks(seed)
    pairs, neighbors = _complete_graph(x.shape[1], order)
    steps = []
    for chunks in (_skeleton._pair_chunks, _candidate_chunks):
        test = make_indep_test("fisherz", x)
        with rows_tested() as counter, reading_flags():
            pulled = []
            for i, a, b, rows in chunks(pairs, neighbors, 0, test):
                pulled.append(((i, a, b), rows.tolist(), rows.shape, rows.dtype, _state(counter)))
            steps.append((pulled, _state(counter)))
    assert steps[0] == steps[1]
    pulled, final = steps[0]
    assert pulled
    assert final[2]  # some pair no row observes together was counted
    # A pair is counted between two chunks, so the order of the two is part of what was compared.
    counted = [len(state[2]) for *_, state in pulled]
    assert any(after > before for before, after in zip(counted, counted[1:]))


@pytest.mark.parametrize("cap", CAPS)
@pytest.mark.parametrize("order", [None, 0])
@pytest.mark.parametrize("seed", range(3))
def test_the_size_0_scan_is_the_generic_scans_at_every_stack_size(
    monkeypatch, marginal_calls, seed, order, cap
):
    x = _blocks(seed)
    pairs, neighbors = _complete_graph(x.shape[1], order)
    got = _scan_size_0(monkeypatch, x, pairs, neighbors, generic=False, cap=cap)
    assert marginal_calls == [len(pairs)]
    want = _scan_size_0(monkeypatch, x, pairs, neighbors, generic=True, cap=cap)
    assert marginal_calls == [len(pairs)]  # the generic scan never took the branch
    assert got == want
    removals, state = got
    assert removals  # some pair is separated by the empty set
    assert state[2]  # some pair no row observes together
    # The pairs observed together on two and on three rows are asked and answered not tested.
    assert {(8, 9), (10, 11)} <= dict(state[2]).keys()


@pytest.mark.parametrize("name", ["fisherz", "student_t"])
@pytest.mark.parametrize("record_pvalues", [True, False])
def test_the_size_0_scan_is_the_generic_scans_for_each_gaussian_test(
    monkeypatch, name, record_pvalues
):
    x = _blocks(5)
    pairs, neighbors = _complete_graph(x.shape[1])
    got = _scan_size_0(
        monkeypatch, x, pairs, neighbors, generic=False, record_pvalues=record_pvalues, name=name
    )
    want = _scan_size_0(
        monkeypatch, x, pairs, neighbors, generic=True, record_pvalues=record_pvalues, name=name
    )
    assert got == want
    assert (got[0][0][3] is None) == (not record_pvalues)


@pytest.mark.parametrize("cap", [64, None])
@pytest.mark.parametrize("seed", range(3))
def test_the_size_0_scan_matches_the_scan_one_pair_at_a_time(monkeypatch, seed, cap):
    x = _blocks(seed)
    pairs, neighbors = _complete_graph(x.shape[1])
    got = _scan_size_0(monkeypatch, x, pairs, neighbors, generic=False, cap=cap)
    with monkeypatch.context() as patch:
        patch.setattr(_skeleton, "_stacked_route", lambda test: None)
        want = _scan_size_0(monkeypatch, x, pairs, neighbors, generic=False, cap=cap)
    assert got[0] == want[0]
    assert _unordered(got[1]) == _unordered(want[1])


def test_a_pair_no_row_observes_together_is_one_query_not_tested():
    x = _blocks(0)
    pairs, neighbors = _complete_graph(x.shape[1])
    test = make_indep_test("fisherz", x)
    together = test.together()
    never = {pair for pair in pairs if not together[pair]}
    assert {(0, 3), (2, 5)} <= never
    with rows_tested() as counter, reading_flags():
        chunks = list(_skeleton._pair_chunks(pairs, neighbors, 0, test))
    assert {(a, b) for _, a, b, _ in chunks} == set(pairs) - never
    assert dict(counter.not_tested) == {pair: 1 for pair in never}
    assert counter.untested_sets == {pair: {frozenset()} for pair in never}
    assert all(rows.shape == (1, 0) and rows.dtype == np.intp for *_, rows in chunks)
    assert not any(rows.flags.writeable for *_, rows in chunks)


def test_no_pairs_make_no_chunks_and_read_nothing(monkeypatch, marginal_calls):
    def read(self):
        raise AssertionError("together() read")

    monkeypatch.setattr(GaussianDeletion, "together", read)
    test = make_indep_test("fisherz", _blocks(0))
    assert list(_skeleton._pair_chunks([], [], 0, test)) == []
    assert marginal_calls == [0]


@pytest.mark.parametrize("guard", ["trace", "size one", "complete data"])
def test_a_pass_that_is_not_size_0_test_wise_deletion_keeps_the_generic_chunks(
    marginal_calls, guard
):
    x = _blocks(0)
    pairs, neighbors = _complete_graph(x.shape[1])
    if guard == "complete data":
        test = FisherZ(sample_scm(d=x.shape[1], n=400, seed=0, density=1.5).data)
    else:
        test = make_indep_test("fisherz", x)
    with rows_tested(), reading_flags():
        scan_pairs(
            test,
            pairs,
            neighbors,
            1 if guard == "size one" else 0,
            ALPHA,
            [] if guard == "trace" else None,
        )
    assert marginal_calls == []


def _constant_on_a_block(seed: int) -> np.ndarray:
    """:func:`_blocks` with column 12 constant where it is observed, so a test of column 12 with
    column 13 reads rows where one of its columns does not vary."""
    x = _blocks(seed)
    x[:, 12] = np.where(np.isnan(x[:, 12]), np.nan, 1.5)
    return x


@pytest.mark.parametrize("cap", [1, 64, None])
def test_a_column_constant_on_a_block_raises_the_generic_error_after_the_same_queries(
    monkeypatch, cap
):
    x = _constant_on_a_block(0)
    pairs, neighbors = _complete_graph(x.shape[1])
    outcomes = []
    for generic in (False, True):
        with monkeypatch.context() as patch:
            patch.setattr(_skeleton, "_STACK_BYTES", cap or _skeleton._STACK_BYTES)
            if generic:
                patch.setattr(_skeleton, "_pair_chunks", _candidate_chunks)
            test = make_indep_test("fisherz", x)
            with rows_tested() as counter, reading_flags():
                with pytest.raises(ValueError, match="constant") as raised:
                    scan_pairs(test, pairs, neighbors, 0, ALPHA)
            outcomes.append((str(raised.value), _state(counter)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1][2]  # pairs were counted before the error


@pytest.mark.parametrize("cap", [64, None])
@pytest.mark.parametrize(
    ("method", "options"),
    [
        ("pc", {}),
        ("pc", {"indep_test": "student_t"}),
        ("pc", {"collider_rule": "majority", "possibly_separable": "ambiguous"}),
        ("pc", {"knowledge": "apart"}),
        ("fci", {}),
        ("fci", {"knowledge": "apart"}),
        ("cdnod", {"env": "halves"}),
    ],
)
def test_the_missing_value_searches_return_what_the_generic_chunks_return(
    monkeypatch, method, options, cap
):
    x = _blocks(1)
    options = dict(options)
    if options.get("knowledge") == "apart":
        # Both directions forbidden: a pair the search never tests, which separate_apart does.
        pairs = [(0, 4), (4, 0), (1, 8), (8, 1), (7, 10), (10, 7), (2, 12), (12, 2)]
        options["knowledge"] = andrey.Knowledge(forbidden=pairs)
    if options.get("env") == "halves":
        options["env"] = np.arange(len(x)) % 2
    fits = []
    for generic in (False, True):
        with monkeypatch.context() as patch:
            if cap is not None:
                patch.setattr(_skeleton, "_STACK_BYTES", cap)
            if generic:
                patch.setattr(_skeleton, "_pair_chunks", _candidate_chunks)
            fit = getattr(andrey, method)(x, missing="testwise", **options)
        fits.append((fit.structure.to_numpy().tobytes(), repr(fit.structure), repr(fit.metadata)))
    assert fits[0] == fits[1]
