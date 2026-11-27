"""The shared Markov-blanket engine: IAMB, Inter-IAMB, and MMPC, and the local search over them.

Under a d-separation oracle each search returns the true set, each recorded separating set
separates, and the spouse step completes the parents and children into the Markov blanket. With a
Fisher-Z test, IAMB returns, bit for bit, the blankets of the Fisher-Z IAMB that CALM's moral graph
is built from. A memoized run is the run made afresh, and the local search gives the sets the
methods return, whatever the column order.
"""

from __future__ import annotations

from itertools import combinations, islice

import numpy as np
import pytest

import andrey
from andrey import GraphStructure, data
from andrey.constraint import _skeleton, blanket
from andrey.constraint._skeleton import kept_apart, scan_conditioning_sets
from andrey.constraint.blanket import (
    LEARNERS,
    Learner,
    LocalSearch,
    blanket_colliders,
    blanket_graph,
    local_search,
    markov_blanket,
    mmpc_sets,
    parents_children,
    reach,
    table_size,
)
from andrey.core.ci import ChiSquare, FisherZ, PartialCorrelationTest, make_indep_test
from andrey.core.ci_missing import not_tested, rows_tested
from andrey.core.knowledge import Constraints
from andrey.core.orient import dag2cpdag
from andrey.core.ties import KEY_DP, tie_ranks
from andrey.search.calm import moral_graph
from tests.recovery.helpers.oracle import DSeparation


def _fisherz_iamb(test: FisherZ, target: int, d: int, alpha: float, ranks: list[int]) -> list[int]:
    """Fisher-Z IAMB as CALM's moral graph is defined: grow by the largest absolute partial
    correlation among candidates with a p-value at or below ``alpha``, ties by rank, then shrink
    once in joining order."""
    blanket: list[int] = []
    while True:
        best_key, best_x = None, None
        for x in range(d):
            if x == target or x in blanket:
                continue
            r, p = test.statistic_pvalue(target, x, blanket)
            key = (-round(abs(r), KEY_DP), ranks[x])
            if p <= alpha and (best_key is None or key < best_key):
                best_key, best_x = key, x
        if best_x is None:
            break
        blanket.append(best_x)
    for x in list(blanket):
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)
    return blanket


@pytest.mark.parametrize(
    ("d", "n", "density", "seed"),
    [(6, 300, 1.0, 0), (10, 500, 2.0, 1), (15, 1000, 3.0, 2), (20, 2000, 2.0, 3), (12, 80, 4.0, 4)],
)
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
def test_iamb_with_fisherz_is_calms_moral_graph_bit_for_bit(d, n, density, seed, alpha):
    x = data.sample_scm(graph="erdos_renyi", d=d, n=n, seed=seed, density=density).data
    test = FisherZ(x)
    ranks = tie_ranks(x).tolist()
    reference = np.zeros((d, d), dtype=bool)
    for i in range(d):
        blanket = markov_blanket(test, i, d, alpha, ranks)
        assert blanket == _fisherz_iamb(test, i, d, alpha, ranks)
        reference[i, blanket] = True
    want = (reference & reference.T).astype(np.float64)
    got = moral_graph(x, alpha=alpha)
    assert got.dtype == want.dtype and got.tobytes() == want.tobytes()


def _oracle(seed: int, d: int = 12, density: float = 3.0) -> DSeparation:
    return DSeparation(
        data.sample_scm(graph="erdos_renyi", d=d, n=2, seed=seed, density=density).graph
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("interleaved", [False, True], ids=["iamb", "inter_iamb"])
def test_iamb_finds_the_true_blanket_under_an_oracle(seed, interleaved):
    oracle = _oracle(seed)
    ranks = list(range(oracle.d))
    for v in range(oracle.d):
        got = markov_blanket(oracle, v, oracle.d, 0.5, ranks, interleaved=interleaved)
        assert set(got) == oracle.markov_blanket(v), v
    graph = blanket_graph(oracle, oracle.d, 0.5, ranks, interleaved=interleaved)
    np.testing.assert_array_equal(graph, oracle.moral_graph())


class _PValues:
    """A test of target 0 that answers each variable with its own p-value, whatever the set."""

    def __init__(self, pvalues: dict[int, float]) -> None:
        self.pvalues = pvalues

    def __call__(self, x, y, cond=()):
        return self.pvalues[y if x == 0 else x]


@pytest.mark.parametrize("interleaved", [False, True], ids=["iamb", "inter_iamb"])
def test_a_grow_step_admits_the_smallest_p_value_even_far_below_the_key_precision(interleaved):
    """IAMB and Inter-IAMB order the dependent candidates of a grow step by their p-values to
    ``KEY_DP`` significant digits, so p-values below ``10**-KEY_DP`` keep the order of their
    sizes; the rank decides only among equal p-values, exactly zero ones included."""
    ranks = [0, 5, 4, 3, 2, 1]
    tiny = _PValues({1: 1e-12, 2: 1e-70, 3: 1e-41, 4: 1e-70, 5: 0.5})
    assert markov_blanket(tiny, 0, 6, 0.05, ranks, interleaved=interleaved) == [4, 2, 3, 1]
    zero = _PValues({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.5})
    assert markov_blanket(zero, 0, 6, 0.05, ranks, interleaved=interleaved) == [4, 3, 2, 1]


def _chisq_iamb(test, target, d, alpha, ranks, decimals):
    """IAMB written out for a p-value test: grow by the smallest p-value among candidates at or
    below ``alpha``, compared to ``KEY_DP`` significant digits (``decimals`` False) or decimals,
    ties by rank, then shrink once in joining order."""
    blanket = []
    while True:
        best_key, best_x = None, None
        for x in range(d):
            if x == target or x in blanket:
                continue
            p = test(target, x, blanket)
            key = (round(p, KEY_DP) if decimals else float(f"{p:.{KEY_DP}g}"), ranks[x])
            if p <= alpha and (best_key is None or key < best_key):
                best_key, best_x = key, x
        if best_x is None:
            break
        blanket.append(best_x)
    for x in list(blanket):
        rest = [z for z in blanket if z != x]
        if test(target, x, rest) > alpha:
            blanket.remove(x)
    return blanket


def test_iamb_on_discrete_data_orders_by_p_value_to_significant_digits():
    """The engine's blankets equal the written-out IAMB's, which a key of decimals would not give
    for every target."""
    x, test = _search_data("chisq")
    ranks = tie_ranks(x).tolist()
    by_decimals = 0
    for target in range(15):
        want = _chisq_iamb(test, target, 15, 0.05, ranks, decimals=False)
        assert markov_blanket(test, target, 15, 0.05, ranks) == want, target
        by_decimals += _chisq_iamb(test, target, 15, 0.05, ranks, decimals=True) != want
    assert by_decimals


@pytest.mark.parametrize("seed", range(4))
def test_mmpc_finds_the_true_parents_and_children_under_an_oracle(seed):
    oracle = _oracle(seed)
    ranks = list(range(oracle.d))
    sets = mmpc_sets(oracle, oracle.d, 0.5, ranks)
    np.testing.assert_array_equal(sets & sets.T, oracle.skeleton())
    for v in range(oracle.d):
        # Without the symmetry check, a set may also hold descendants it cannot separate.
        assert set(parents_children(oracle, v, oracle.d, 0.5, ranks)) >= oracle.parents_children(v)


def test_mmpc_symmetry_check_drops_a_descendant_the_target_cannot_separate():
    # t -> c <- b, b -> x, c -> x: only {c, b} separates t and x, and b, independent of t, never
    # joins t's set, so t's set keeps x; x's set separates t, so the skeleton drops the pair.
    t, c, b, x = range(4)
    marks = np.zeros((4, 4), dtype=np.int8)
    for i, j in [(t, c), (b, c), (b, x), (c, x)]:
        marks[i, j], marks[j, i] = 1, 2
    oracle = DSeparation(GraphStructure.from_numpy(marks, kind="dag"))
    ranks = list(range(4))
    assert set(parents_children(oracle, t, 4, 0.5, ranks)) == {c, x}
    sets = mmpc_sets(oracle, 4, 0.5, ranks)
    np.testing.assert_array_equal(sets & sets.T, oracle.skeleton())


def test_independent_columns_have_empty_sets():
    x = np.random.default_rng(0).standard_normal((500, 6))
    test = FisherZ(x)
    ranks = tie_ranks(x).tolist()
    assert not blanket_graph(test, 6, 0.01, ranks).any()
    sets = mmpc_sets(test, 6, 0.01, ranks)
    assert not (sets & sets.T).any()


def _collider_sets(sides, size, middle):
    """The conditioning sets of a collider test, each once, as sorted tuples: ``middle`` with every
    size-``size`` subset of ``sides[0]``, then of the other side, less the subsets the first side
    holds."""
    first = set(sides[0])
    for n, side in enumerate(sides):
        for rest in combinations(side, size):
            if n == 0 or not first.issuperset(rest):
                yield tuple(sorted((*rest, middle)))


def _colliders_on_every_set(test, skeleton, blankets, alpha):
    """Each unshielded triple tested on every set of each size, by increasing size, up to the first
    size with a separating set: the separating sets and the collider ranks."""
    sepsets, ranks = {}, {}
    for b in range(skeleton.shape[0]):
        for a, c in combinations(np.flatnonzero(skeleton[b]).tolist(), 2):
            if skeleton[a, c]:
                continue
            sepsets.setdefault((a, c), ())
            sides = blanket._smaller(blankets, a, c, (b,))
            p, strength = -np.inf, np.inf
            for size in range(max(len(side) for side in sides) + 1):
                sets = list(_collider_sets(sides, size, b))
                [(p_size, strength_size)] = blanket._weakest(test, a, [c], sets)
                p, strength = max(p, p_size), min(strength, strength_size)
                if p > alpha:
                    sepsets[a, c] = (*sepsets[a, c], b)
                    break
            else:
                ranks[a, b, c] = strength
    return sepsets, ranks


def _search_data(kind):
    """Data and a test of 15 variables: Gaussian with Fisher-z, discrete with the chi-square
    test, and discrete or Gaussian with a twentieth of the values missing, whose test deletes rows
    per test."""
    if kind == "fisherz":
        x = data.sample_scm(graph="erdos_renyi", d=15, n=1000, seed=2, density=4.0).data
        return x, FisherZ(x)
    if kind == "missing_fisherz":
        x = data.sample_scm(graph="erdos_renyi", d=15, n=1000, seed=2, density=4.0).data.copy()
        x[np.random.default_rng(0).random(x.shape) < 0.05] = np.nan
        return x, make_indep_test("fisherz", x)
    x = data.sample_discrete(d=15, n=3000, seed=2, density=4.0).data
    if kind == "chisq":
        return x, ChiSquare(x)
    x = x.copy()
    x[np.random.default_rng(0).random(x.shape) < 0.05] = np.nan
    return x, make_indep_test("chisq", x)


@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing"])
@pytest.mark.parametrize(("first", "chunk"), [(1, 4096), (1, 2), (4, 8)])
def test_a_non_collider_stops_at_its_first_separating_chunk(monkeypatch, kind, first, chunk):
    """Every triple is decided, and every collider ranked, as testing every set of each size up to
    the first size that separates decides and ranks it, with fewer sets tested; a test that
    deletes rows per test tests the same sets, so the rows it reports are the same."""
    monkeypatch.setattr(blanket, "_FIRST_CHUNK", first)
    monkeypatch.setattr(blanket, "CHUNK", chunk)
    x, test = _search_data(kind)
    sets = mmpc_sets(test, 15, 0.05, tie_ranks(x).tolist())
    skeleton, blankets = sets & sets.T, reach(sets)
    tested = []
    weakest = blanket._weakest

    def counted(test, target, xs, conds, *args, **kwargs):
        tested.append(len(conds))
        return weakest(test, target, xs, conds, *args, **kwargs)

    monkeypatch.setattr(blanket, "_weakest", counted)
    with rows_tested() as rows:
        sepsets, ranks, ambiguous = blanket_colliders(test, skeleton, blankets, 0.05)
    found = sum(tested)
    tested.clear()
    with rows_tested() as want_rows:
        want_sepsets, want_ranks = _colliders_on_every_set(test, skeleton, blankets, 0.05)
    assert sepsets == want_sepsets
    assert {t: ranks[t] for t in want_ranks} == want_ranks
    assert not ambiguous
    assert rows == want_rows
    assert found == sum(tested) if kind == "missing" else found < sum(tested)


def _tuple_chunks(sides, size, middle, first, chunk):
    """The chunks of the tuples of :func:`_collider_sets`: ``first`` sets, then twice as many each
    time, at most ``chunk``."""
    sets = _collider_sets(sides, size, middle)
    out, width = [], first
    while rows := list(islice(sets, width)):
        out.append(rows)
        width = min(2 * width, chunk)
    return out


def _side_draws(rng):
    """One side of candidates, ascending, or two of the same size that overlap, as a blanket
    pair's do; the columns a test that deletes rows per test finds no row observes with the pair;
    and a middle node outside the sides."""
    n = int(rng.integers(0, 10))
    sides = [sorted(rng.choice(14, size=n, replace=False).tolist())]
    if rng.random() < 0.6:
        sides.append(sorted(rng.choice(14, size=n, replace=False).tolist()))
    dropped = set(rng.choice(14, size=int(rng.integers(0, 4)), replace=False).tolist())
    free = sorted(set(range(15)) - set(sides[0]) - set(sides[-1]))
    return sides, dropped, int(rng.choice(free))


@pytest.mark.parametrize(("first", "chunk", "block"), [(32, 4096, 4096), (1, 3, 8), (2, 8, 5)])
def test_collider_rows_are_the_combinations_in_the_same_chunks(monkeypatch, first, chunk, block):
    """Each size's sets, as the arrays of :func:`_collider_rows`, are the tuples of the
    ``combinations`` reference, in its order, cut into the same chunks, whatever the chunk
    ``subset_rows`` builds them in, for sizes from none to one past the side, one side or two that
    overlap, and sides cut by the columns a test that deletes rows per test drops."""
    monkeypatch.setattr(blanket, "CHUNK", chunk)
    monkeypatch.setattr(_skeleton, "CHUNK", block)
    rng = np.random.default_rng(first * 100 + chunk)
    multi = 0
    for _ in range(60):
        sides, dropped, middle = _side_draws(rng)
        kept = [
            ([c for c in side if c not in dropped], skip)
            for side, skip in blanket._side_sets(sides)
        ]
        for size in range(max(len(side) for side in sides) + 2):
            want = _tuple_chunks([s for s, _ in kept], size, middle, first, chunk)
            got = list(blanket._collider_rows(kept, size, middle, first))
            assert [[tuple(row) for row in part.tolist()] for part in got] == want
            assert all(part.dtype == np.intp and part.shape[1] == size + 1 for part in got)
            multi += len(want) > 1
    assert multi > 20


def test_collider_rows_cross_sides_and_chunks_at_the_default_widths():
    """Two overlapping sides of sixteen candidates give more sets of size six than one chunk holds,
    and the runs cross from the first side into the second."""
    first, second = list(range(16)), list(range(8, 24))
    kept = blanket._side_sets([first, second])
    want = _tuple_chunks([first, second], 6, 30, blanket._FIRST_CHUNK, blanket.CHUNK)
    got = list(blanket._collider_rows(kept, 6, 30, blanket._FIRST_CHUNK))
    assert len(want) > 8
    assert [[tuple(row) for row in part.tolist()] for part in got] == want


class _Table:
    """A test whose ``statistic_pvalues`` answers from fixed arrays and keeps what it was asked."""

    def __init__(self, r, p):
        self.r, self.p, self.asked = np.asarray(r), np.asarray(p), []

    def statistic_pvalues(self, x, y, cond):
        self.asked.append((x.tolist(), y.tolist(), cond.tolist(), x.dtype, y.dtype, cond.dtype))
        return self.r[: len(x)], self.p[: len(x)]


_NAN = float("nan")
_INF = float("inf")
_TABLES = {
    "tested": ([0.5, -0.2, 0.1, -0.7, 0.3, 0.2], [0.3, 0.01, 0.5, 0.04, 0.2, 0.0]),
    "some_not_tested_above_alpha": (
        [0.5, _NAN, 0.1, 0.4, 0.3, 0.2],
        [0.3, _NAN, 0.5, 0.9, 0.2, 0.1],
    ),
    "some_not_tested_below_alpha": (
        [0.5, _NAN, 0.1, 0.4, 0.3, 0.2],
        [0.01, _NAN, 0.02, 0.0, 0.03, 0.04],
    ),
    "none_tested": ([_NAN] * 6, [_NAN] * 6),
    "p_minus_inf": ([0.5, 0.1, 0.1, 0.4, 0.3, 0.2], [-_INF] * 6),
    "p_plus_inf": ([0.5, 0.1, 0.1, 0.4, 0.3, 0.2], [_INF, 0.2, 0.1, 0.4, 0.3, 0.2]),
    "nan_strengths": ([_NAN, _NAN, 0.1, _NAN, 0.3, _NAN], [0.3, 0.01, 0.5, 0.04, 0.2, 0.7]),
}


def _same_bits(got, want):
    assert [(float(p).hex(), float(r).hex()) for p, r in got] == [
        (float(p).hex(), float(r).hex()) for p, r in want
    ]


@pytest.mark.parametrize("alpha", [None, 0.05])
@pytest.mark.parametrize("table", _TABLES)
def test_weakest_reads_the_sets_of_an_array_as_it_reads_their_tuples(table, alpha):
    """One candidate against the sets of an int array gets the answer, and puts the test the
    queries, that the same sets as a list of tuples do: the largest p-value, the smallest strength,
    and a verdict that needs every set not tested when a set was not tested, a p-value was
    infinite, or none was tested."""
    rows = np.array([[0, 1, 5], [0, 2, 5], [1, 2, 5], [0, 3, 5], [2, 3, 5], [3, 4, 5]])
    r, p = _TABLES[table]
    by_array, by_tuples = _Table(r, p), _Table(r, p)
    got = blanket._weakest(by_array, 7, [8], rows, alpha=alpha)
    want = blanket._weakest(by_tuples, 7, [8], [tuple(row) for row in rows.tolist()], alpha=alpha)
    _same_bits(got, want)
    assert by_array.asked == by_tuples.asked
    assert len(by_array.asked) == 1


def test_weakest_marks_a_verdict_that_needs_a_set_not_tested():
    """The not-tested guard of the array read fires where the tuple read fires it: with ``alpha``,
    a p-value that does not pass ``alpha`` beside a set not tested is not tested."""
    rows = np.array([[0, 5], [1, 5], [2, 5]])
    cases = {
        (0.01, _NAN, 0.02): True,
        (0.9, _NAN, 0.02): False,
        (_NAN, _NAN, _NAN): True,
        (0.01, 0.2, 0.02): False,
    }
    for ps, marked in cases.items():
        [(p, _)] = blanket._weakest(_Table([0.1] * 3, ps), 3, [4], rows, alpha=0.05)
        assert bool(not_tested(p)) is marked


def test_weakest_of_an_array_beyond_the_fast_path_is_the_tuples_answer():
    """Several candidates, ``sets`` to fill, a test without ``statistic_pvalues``, and no row each
    answer for an array as for tuples, including the sets appended and the order of the queries."""
    continuous = data.sample_scm(graph="erdos_renyi", d=8, n=400, seed=1, density=3.0).data
    discrete = data.sample_discrete(d=8, n=800, seed=1, density=3.0).data
    rows = np.array([[1, 2, 5], [1, 3, 5], [2, 3, 5], [1, 4, 5]])
    tuples = [tuple(row) for row in rows.tolist()]
    ranks = list(range(8))
    for test in (FisherZ(continuous), ChiSquare(discrete)):
        for xs in ([6], [6, 7], [7, 6]):
            for alpha in (None, 0.05):
                by_array, by_tuples = [], []
                got = blanket._weakest(test, 0, xs, rows, by_array, ranks, alpha)
                want = blanket._weakest(test, 0, xs, tuples, by_tuples, ranks, alpha)
                _same_bits(got, want)
                assert by_array == by_tuples
                _same_bits(
                    blanket._weakest(test, 0, xs, rows, None, ranks, alpha),
                    blanket._weakest(test, 0, xs, tuples, None, ranks, alpha),
                )
    empty = np.empty((0, 2), dtype=np.intp)
    _same_bits(
        blanket._weakest(FisherZ(continuous), 0, [6], empty, alpha=0.05),
        blanket._weakest(FisherZ(continuous), 0, [6], [], alpha=0.05),
    )


def _colliders_by_tuples(test, skeleton, blankets, alpha, max_depth=None):
    """:func:`blanket_colliders` as it tested the sets before it built them as arrays: each size's
    sets as sorted tuples in chunks, each chunk one :func:`blanket._weakest` call on the tuples."""
    sepsets, strengths, ambiguous = {}, {}, set()
    every = blanket._tests_every_set(test)
    for b in range(skeleton.shape[0]):
        for a, c in combinations(np.flatnonzero(skeleton[b]).tolist(), 2):
            if skeleton[a, c]:
                continue
            sepsets.setdefault((a, c), ())
            sides = blanket._smaller(blankets, a, c, (b,))
            top = max(len(side) for side in sides)
            if max_depth is not None:
                top = min(top, max_depth - 1)
            if top < 0:
                ambiguous.add((a, b, c))
                continue
            p, strength = -np.inf, np.inf
            for size in range(top + 1):
                kept, left = kept_apart(test, (a, c, b), blanket._side_sets(sides), size)
                conds = iter(()) if kept is None else _collider_sets([s for s, _ in kept], size, b)
                width = blanket.CHUNK if every else min(blanket._FIRST_CHUNK, blanket.CHUNK)
                while (every or p <= alpha) and (chunk := list(islice(conds, width))):
                    [(p_chunk, strength_chunk)] = blanket._weakest(test, a, [c], chunk, alpha=alpha)
                    skipped = bool(not_tested([p, p_chunk]).any())
                    p = blanket._weaker(float(np.fmax(p, p_chunk)), skipped, alpha)
                    strength = float(np.fmin(strength, strength_chunk))
                    width = min(2 * width, blanket.CHUNK)
                if left:
                    p = blanket._weaker(p, True, alpha)
                if p > alpha:
                    sepsets[a, c] = (*sepsets[a, c], b)
                    break
            if not_tested(p):
                ambiguous.add((a, b, c))
                continue
            strengths[a, b, c] = strength
    return sepsets, strengths, frozenset(ambiguous)


def _recorded(test):
    """``test``'s ``statistic_pvalues`` queries, kept as they are asked, in a list."""
    asked = []
    ask = getattr(test, "statistic_pvalues", None)
    if ask is not None:

        def recording(x, y, cond, *args, **kwargs):
            asked.append((x.tolist(), y.tolist(), cond.tolist()))
            return ask(x, y, cond, *args, **kwargs)

        test.statistic_pvalues = recording
    return asked


@pytest.mark.parametrize("max_depth", [None, 2])
@pytest.mark.parametrize(("first", "chunk"), [(32, 4096), (1, 2), (4, 8)])
@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing", "missing_fisherz"])
def test_the_collider_test_on_arrays_is_the_collider_test_on_tuples(
    monkeypatch, kind, first, chunk, max_depth
):
    """The decided triples, their ranks, the ambiguous triples, the rows each test read, and the
    queries a test with ``statistic_pvalues`` was asked, in order, are those of the collider test on
    tuples, for a test that answers a set at a time, one that deletes rows per test, and one that
    does both, at chunk widths that cut every size and at the default ones."""
    monkeypatch.setattr(blanket, "_FIRST_CHUNK", first)
    monkeypatch.setattr(blanket, "CHUNK", chunk)
    monkeypatch.setattr(_skeleton, "CHUNK", chunk)
    x, test = _search_data(kind)
    ranks = tie_ranks(x).tolist()
    sets = mmpc_sets(test, 15, 0.05, ranks)
    skeleton, blankets = sets & sets.T, reach(sets)
    _, want_test = _search_data(kind)
    mmpc_sets(want_test, 15, 0.05, ranks)
    asked, want_asked = _recorded(test), _recorded(want_test)
    with rows_tested() as rows:
        got = blanket_colliders(test, skeleton, blankets, 0.05, max_depth)
    with rows_tested() as want_rows:
        want = _colliders_by_tuples(want_test, skeleton, blankets, 0.05, max_depth)
    assert got[0] == want[0]
    assert {t: s.hex() for t, s in got[1].items()} == {t: s.hex() for t, s in want[1].items()}
    assert got[2] == want[2]
    assert rows == want_rows
    assert asked == want_asked
    assert got[1] or got[2]
    assert asked or kind in ("chisq", "missing")


def _mmpc_on_every_set(test, target, d, alpha, ranks):
    """MMPC with every remaining candidate tested against every set at each addition, then the
    backward phase of :func:`parents_children`."""
    weakest = {x: blanket._weakest(test, target, [x], [()])[0] for x in range(d) if x != target}
    members: list[int] = []
    while open_ := {x: w for x, w in weakest.items() if w[0] <= alpha}:
        best = min(
            open_,
            key=lambda x: (round(open_[x][0], KEY_DP), -round(open_[x][1], KEY_DP), ranks[x]),
        )
        del open_[best]
        conds = [
            tuple(sorted((*rest, best)))
            for size in range(len(members) + 1)
            for rest in combinations(members, size)
        ]
        for x, (p, strength) in open_.items():
            [(p_x, strength_x)] = blanket._weakest(test, target, [x], conds)
            open_[x] = (max(p, p_x), min(strength, strength_x))
        members.append(best)
        weakest = open_
    for x in list(members):
        before = set(members[: members.index(x)])
        rest = sorted(z for z in members if z != x)
        for size in range(1, len(rest) + 1):
            if scan_conditioning_sets(test, target, x, rest, size, alpha, set(), before):
                members.remove(x)
                break
    return members


@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing"])
def test_mmpc_stops_testing_a_candidate_once_a_set_separates_it(kind):
    """Each variable's MMPC set, in joining order, is the one found by testing every remaining
    candidate against every set at each addition; a test that deletes rows per test tests the
    same sets, so the rows it reports are the same."""
    x, test = _search_data(kind)
    ranks = tie_ranks(x).tolist()
    for target in range(15):
        with rows_tested() as want_rows:
            want = _mmpc_on_every_set(test, target, 15, 0.05, ranks)
        with rows_tested() as rows:
            assert parents_children(test, target, 15, 0.05, ranks) == want, target
        assert rows == want_rows, target


def _fresh(method, test, target, d, alpha, ranks):
    """``target``'s run by ``method`` made anew: the set and the separating sets."""
    sepsets: dict[int, tuple[int, ...]] = {}
    if method == "mmpc":
        found = parents_children(test, target, d, alpha, ranks, sepsets=sepsets)
    else:
        interleaved = method == "inter_iamb"
        found = markov_blanket(
            test, target, d, alpha, ranks, interleaved=interleaved, sepsets=sepsets
        )
    return found, sepsets


class _NoTests:
    """A test that fails on any use: a memoized answer must not reach it."""

    def __getattr__(self, name):
        raise AssertionError(f"test.{name} used")

    def __call__(self, *args):
        raise AssertionError("test called")


@pytest.mark.parametrize("kind", ["fisherz", "chisq", "missing"])
def test_a_memoized_run_is_the_run_made_afresh_and_tests_nothing_again(kind):
    """Each run of a local search, asked for every target in a shuffled order, is the run made
    afresh; the parents and children are the members whose own run holds the target; and asking
    again tests nothing."""
    x, test = _search_data(kind)
    ranks = tie_ranks(x).tolist()
    search = LocalSearch(test, ranks, 0.05)
    order = np.random.default_rng(0).permutation(15).tolist()
    want = {}
    for method in ("iamb", "inter_iamb", "mmpc"):
        for t in order:
            want[method, t] = search.local(method, t)
            assert search.run(method, t) == _fresh(method, test, t, 15, 0.05, ranks), (method, t)
        for t in order:
            members, _ = _fresh(method, test, t, 15, 0.05, ranks)
            if method == "mmpc":
                symmetric = [v for v in members if t in _fresh(method, test, v, 15, 0.05, ranks)[0]]
                assert want[method, t][0] == symmetric, t
            else:
                assert want[method, t][1] == members, t
    search.test = _NoTests()
    for (method, t), local in want.items():
        assert search.local(method, t) == local


def _check_sepsets(oracle: DSeparation, t: int, learned: list[int], sepsets) -> None:
    """Every other variable outside ``learned`` has a recorded set, and each set separates."""
    assert set(sepsets) == set(range(oracle.d)) - set(learned) - {t}, t
    for v, given in sepsets.items():
        assert oracle(t, v, given) == 1.0, (t, v, given)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_each_recorded_separating_set_separates_under_an_oracle(method, seed):
    oracle = _oracle(seed)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for t in range(oracle.d):
        pc, mb, sepsets = search.local(method, t)
        learned = mb if pc is None else pc
        assert learned is not None
        _check_sepsets(oracle, t, learned, sepsets)
        members, own = search.run(method, t)
        assert set(own) <= set(range(oracle.d)) - set(members) - {t}
        assert all(oracle(t, v, given) == 1.0 for v, given in own.items())


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_each_recorded_separating_set_is_a_test_that_separated(method, kind):
    """On data, the test given a recorded set has a p-value above alpha."""
    x, test = _search_data(kind)
    search = LocalSearch(test, tie_ranks(x).tolist(), 0.05)
    for t in range(15):
        for v, given in search.local(method, t)[2].items():
            assert test(t, v, given) > 0.05, (t, v, given)


@pytest.fixture
def mmpc_spouses(monkeypatch):
    """A parents-and-children-then-spouses learner over MMPC, registered for the test."""
    monkeypatch.setitem(LEARNERS, "mmpc_spouses", Learner("pc_spouses", pc="mmpc"))
    return "mmpc_spouses"


@pytest.mark.parametrize("seed", range(4))
def test_the_spouse_step_completes_the_blanket_under_an_oracle(mmpc_spouses, seed):
    dag = data.sample_scm(graph="erdos_renyi", d=12, n=2, seed=seed, density=3.0).graph
    oracle = DSeparation(dag)
    search = LocalSearch(oracle, list(range(oracle.d)), 0.5)
    for t in range(oracle.d):
        pc, mb, sepsets = search.local(mmpc_spouses, t)
        assert pc is not None and mb is not None
        assert set(pc) == oracle.parents_children(t), t
        assert set(mb) == oracle.markov_blanket(t) and mb[: len(pc)] == pc, t
        _check_sepsets(oracle, t, pc, sepsets)
    np.testing.assert_array_equal(
        search.structure(mmpc_spouses, "moral") != 0, oracle.moral_graph()
    )
    np.testing.assert_array_equal(
        search.structure(mmpc_spouses, "skeleton") != 0, oracle.skeleton()
    )
    truth = dag2cpdag(dag.to_numpy())
    np.testing.assert_array_equal(search.structure(mmpc_spouses, "cpdag"), truth)
    search.test = _NoTests()
    assert search.local(mmpc_spouses, 0)[1] is not None


def test_the_spouse_step_pairs_a_candidate_only_with_its_neighbor(mmpc_spouses):
    """On P -> Q -> T -> S <- R <- P, T's blanket is {Q, S, R}: P, separated from T by {Q}, is
    dependent on T given {Q, S}, but S is not adjacent to P, so that test is never run (Pena et
    al. 2007, DAG (b))."""
    p, q, t, s, r = range(5)
    marks = np.zeros((5, 5), dtype=np.int8)
    for i, j in [(p, q), (q, t), (t, s), (r, s), (p, r)]:
        marks[i, j], marks[j, i] = 1, 2
    oracle = DSeparation(GraphStructure.from_numpy(marks, kind="dag"))
    search = LocalSearch(oracle, list(range(5)), 0.5)
    pc, mb, sepsets = search.local(mmpc_spouses, t)
    assert pc == [q, s] or pc == [s, q]
    assert mb is not None and set(mb) == {q, s, r}
    assert oracle(t, p, (q, s)) == 0.0 and q in sepsets[p]


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_the_local_entry_gives_the_set_the_method_returns(method, kind):
    x, _ = _search_data(kind)
    indep_test = "fisherz" if kind == "fisherz" else "chisq"
    search = local_search(x, alpha=0.05, indep_test=indep_test)
    fit = getattr(andrey, method)
    key = "parents_and_children" if method == "mmpc" else "markov_blanket"
    for t in range(15):
        pc, mb, _ = search.local(method, t)
        assert (pc if method == "mmpc" else mb) == fit(x, target=t, indep_test=indep_test).metadata[
            key
        ], t
        assert (mb if method == "mmpc" else pc) is None


@pytest.mark.parametrize("kind", ["fisherz", "chisq"])
@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_a_column_shuffle_leaves_every_recorded_set(method, kind):
    x, _ = _search_data(kind)
    indep_test = "fisherz" if kind == "fisherz" else "chisq"
    perm = np.random.default_rng(1).permutation(15)
    search = local_search(x, alpha=0.05, indep_test=indep_test)
    shuffled = local_search(x[:, perm], alpha=0.05, indep_test=indep_test)
    where = np.argsort(perm)  # where[v]: the shuffled position of column v

    def back(found):
        return None if found is None else [int(perm[v]) for v in found]

    for t in range(15):
        pc, mb, sepsets = search.local(method, t)
        spc, smb, ssepsets = shuffled.local(method, int(where[t]))
        assert (back(spc), back(smb)) == (pc, mb), t
        moved = {int(perm[v]): tuple(sorted(int(perm[u]) for u in g)) for v, g in ssepsets.items()}
        assert moved == sepsets, t


def test_table_size_counts_cells_and_degrees_of_freedom():
    levels = [2, 3, 4, 0]
    assert table_size(levels, 0, 1, [2]) == 24
    assert table_size(levels, 0, 1, [2], df=True) == 8
    assert table_size(levels, 1, 2, []) == 12
    assert table_size(levels, 0, 1, [3]) is None
    x = data.sample_discrete(d=4, n=500, seed=0).data
    want = [np.unique(c).size for c in x.T]
    assert local_search(x, alpha=0.05, indep_test="chisq").levels.tolist() == want
    assert local_search(x, alpha=0.05, indep_test="fisherz").levels is None


@pytest.mark.parametrize(
    "row",
    [
        {"kind": "pc_spouses"},
        {"kind": "pc_spouses", "pc": "mmpc", "search": blanket._mmpc},
        {"kind": "pc"},
        {"kind": "blanket", "search": blanket._mmpc, "pc": "mmpc"},
    ],
)
def test_a_learner_row_names_one_search_or_one_pc_learner(row):
    with pytest.raises(ValueError, match="pc_spouses"):
        Learner(**row)


def test_the_recorded_set_is_the_tested_separator_beside_a_set_not_tested():
    pvalues = np.array([[np.nan, 0.3, 0.1], [np.nan, np.nan, np.nan]])
    conds = [(1,), (2,), (3,)]
    assert blanket._largest_sets(pvalues, conds, [0, 1, 2, 3]) == [(2,), (1,)]


def _scanned(scan, target, xs, cond):
    """The ``(p-value, key)`` rows of a grow scan, one per candidate."""
    pvalues, key = scan(target, xs, cond)
    return np.array([(pvalues[i], key(i)) for i in range(len(xs))]).reshape(len(xs), 2)


@pytest.mark.parametrize("kind", ["fisherz", "student_t", "normal_scores"])
def test_a_large_grow_scan_is_answered_from_one_factor(kind):
    """A grow scan of enough candidates given two or more variables goes to the test's one-factor
    path and agrees with each candidate asked alone to rounding (the p-value to 1e-9 relative, on
    the same side of alpha); one candidate keeps the per-query answer bit for bit."""
    x = data.sample_scm(graph="erdos_renyi", d=40, n=400, seed=3, density=3.0).data
    test = make_indep_test(kind, x)
    factored = []
    factor = test._schur_partial_correlations
    test._schur_partial_correlations = lambda *args: factored.append(1) or factor(*args)
    scan = blanket._dependence_scan(test, significant=True)
    one = blanket._dependence(test, significant=True)
    cond = [2, 9, 17]
    xs = [v for v in range(40) if v != 4 and v not in cond]
    got = _scanned(scan, 4, xs, cond)
    assert factored == [1]
    want = np.array([one(4, v, cond) for v in xs])
    np.testing.assert_allclose(got[:, 0], want[:, 0], rtol=1e-9, atol=1e-300)
    assert ((got[:, 0] <= 0.05) == (want[:, 0] <= 0.05)).all()
    np.testing.assert_allclose(got[:, 1], want[:, 1], rtol=0.0, atol=10.0**-KEY_DP)
    assert _scanned(scan, 4, xs[:1], cond).tolist() == [list(one(4, xs[0], cond))]


class _Recorded:
    """A test that records every query it answers, in order, and the scans it answers at once."""

    def __init__(self) -> None:
        self.queries: list[tuple[int, int, tuple[int, ...]]] = []
        self.scans = 0

    def _answer(self, target: int, x: int, cond: tuple[int, ...]) -> tuple[float, float]:
        """The statistic and p-value: coarse, so that many candidates tie on both exactly or after
        ``KEY_DP`` digits, and NaN for a fixed rule of queries, and for every query of variable 7,
        which a test that does not test some queries answers."""
        if getattr(self, "blank", False) and (x == 7 or (3 * x + len(cond)) % 7 == 2):
            return float("nan"), float("nan")
        statistic = ((7 * x + 3 * len(cond)) % 4 + 1) / 10 - x * 1e-13
        pvalue = (0.001, 0.02, 0.05 + x * 1e-13, 0.3, 0.9)[(3 * x + 2 * len(cond) + sum(cond)) % 5]
        return statistic, pvalue


class _FactorTest(_Recorded):
    """A test with ``statistic_pvalue`` and ``statistic_pvalues_given`` over :meth:`_answer`."""

    def __init__(self, blank: bool) -> None:
        super().__init__()
        self.blank = blank

    def __call__(self, target, x, cond=()):
        return self.statistic_pvalue(target, x, cond)[1]

    def statistic_pvalue(self, target, x, cond=()):
        cond = tuple(sorted(cond))
        self.queries.append((target, x, cond))
        return self._answer(target, x, cond)

    def statistic_pvalues_given(self, target, xs, cond):
        self.scans += 1
        cond = tuple(sorted(cond))
        self.queries.extend((target, x, cond) for x in xs)
        pairs = [self._answer(target, x, cond) for x in xs]
        return np.array([r for r, _ in pairs]), np.array([p for _, p in pairs])


class _FactorOneByOne(_FactorTest):
    """The same answers without ``statistic_pvalues_given``: one query at a time."""

    statistic_pvalues_given = None  # ty: ignore[invalid-assignment]


class _ScanTest(_Recorded):
    """A test with ``pvalues_given`` over :meth:`_answer`, and no ``statistic_pvalue``."""

    def __init__(self, blank: bool) -> None:
        super().__init__()
        self.blank = blank

    def __call__(self, target, x, cond=()):
        cond = tuple(sorted(cond))
        self.queries.append((target, x, cond))
        return self._answer(target, x, cond)[1]

    def pvalues_given(self, target, xs, cond=()):
        self.scans += 1
        cond = tuple(sorted(cond))
        self.queries.extend((target, x, cond) for x in xs)
        return np.array([self._answer(target, x, cond)[1] for x in xs], dtype=np.float64)


class _ScanOneByOne(_ScanTest):
    """The same answers without ``pvalues_given``: one query at a time."""

    pvalues_given = None  # ty: ignore[invalid-assignment]


def _blankets(test, d, alpha, interleaved, ranks=None, **options):
    """Every target's blanket and its recorded separating sets in the order they were set."""
    ranks = list(range(d))[::-1] if ranks is None else ranks
    runs = []
    for target in range(d):
        sepsets: dict[int, tuple[int, ...]] = {}
        found = markov_blanket(
            test, target, d, alpha, ranks, interleaved=interleaved, sepsets=sepsets, **options
        )
        runs.append((found, list(sepsets.items())))
    return runs


@pytest.mark.parametrize("blank", [False, True], ids=["tested", "some_not_tested"])
@pytest.mark.parametrize("interleaved", [False, True], ids=["iamb", "inter_iamb"])
@pytest.mark.parametrize(
    ("d", "alpha", "max_depth", "required"),
    [
        (12, 0.05, None, ()),
        (12, 0.02, None, ()),
        (12, 0.3, None, ()),
        (12, 0.5, None, ()),
        (12, 0.5, 3, ()),
        (12, 0.05, None, ((0, 1), (2, 3))),
        (12, 0.5, 2, ((4, 5),)),
        (3, 0.5, None, ()),
    ],
)
@pytest.mark.parametrize(
    ("scanned", "one_by_one"),
    [(_FactorTest, _FactorOneByOne), (_ScanTest, _ScanOneByOne)],
    ids=["statistic_pvalues_given", "pvalues_given"],
)
def test_a_grow_scan_read_as_arrays_is_the_per_candidate_loop_bit_for_bit(
    scanned, one_by_one, d, alpha, max_depth, required, interleaved, blank
):
    """A grow step that takes the p-values of a scan as one array gives every target the blanket,
    the order the members joined in, the separating sets in the order they were set, and the
    queries of the loop that asks one candidate at a time: with keys that tie exactly, so the rank
    decides, queries not tested, required members, a depth cap, and scans of two, one, or no
    candidates."""
    constraints = None
    if required:
        edges = np.zeros((d, d), dtype=bool)
        for i, j in required:
            edges[i, j] = True
        constraints = Constraints(np.zeros((d, d), dtype=bool), edges, np.full(d, -1))
    options = {"max_depth": max_depth, "constraints": constraints}
    array_test, loop_test = scanned(blank), one_by_one(blank)
    assert blanket._dependence_scan(array_test) is not None
    assert blanket._dependence_scan(loop_test) is None
    got = _blankets(array_test, d, alpha, interleaved, **options)
    want = _blankets(loop_test, d, alpha, interleaved, **options)
    assert got == want
    assert array_test.queries == loop_test.queries
    assert array_test.scans > 0
    assert loop_test.scans == 0


class _OneByOne:
    """A test that answers only through ``__call__`` and the ``statistic_pvalue`` of the test it
    wraps, if it has one, so a grow scan asks one candidate at a time."""

    def __init__(self, test) -> None:
        self._test = test
        if hasattr(test, "statistic_pvalue"):
            self.statistic_pvalue = test.statistic_pvalue

    def __call__(self, x, y, cond=()):
        return self._test(x, y, cond)


@pytest.mark.parametrize("kind", ["fisherz", "student_t", "normal_scores", "chisq", "gsq"])
@pytest.mark.parametrize("interleaved", [False, True], ids=["iamb", "inter_iamb"])
def test_a_grow_scan_of_a_registered_test_is_the_per_candidate_loop_bit_for_bit(
    kind, interleaved, monkeypatch
):
    """With the scans of a partial-correlation test answered by its stacked queries, which match
    each query asked alone bit for bit, and with the scans of a table test, which do too, every
    target's blanket and separating sets equal those of the loop over the test's own queries."""
    monkeypatch.setenv("ANDREY_SHARED_SCAN_MIN_WORK", "1000000000")
    if kind in ("chisq", "gsq"):
        x = data.sample_discrete(d=8, n=1500, seed=2, density=2.0).data
    else:
        x = data.sample_scm(graph="erdos_renyi", d=14, n=300, seed=2, density=3.0).data
    test = make_indep_test(kind, x)
    assert blanket._dependence_scan(test) is not None
    assert blanket._dependence_scan(_OneByOne(test)) is None
    ranks = tie_ranks(x).tolist()
    for alpha in (0.05, 0.3):
        got = _blankets(test, x.shape[1], alpha, interleaved, ranks)
        want = _blankets(_OneByOne(test), x.shape[1], alpha, interleaved, ranks)
        assert got == want


@pytest.mark.parametrize("interleaved", [False, True])
def test_the_shared_factor_leaves_every_blanket_unchanged(interleaved, monkeypatch):
    """IAMB and Inter-IAMB find the same blanket at every target whether every grow scan is
    factored or none is."""
    x = data.sample_scm(graph="erdos_renyi", d=30, n=500, seed=1, density=3.0).data
    ranks = tie_ranks(x).tolist()

    def blankets(min_work: str) -> list[list[int]]:
        monkeypatch.setenv("ANDREY_SHARED_SCAN_MIN_WORK", min_work)
        test = FisherZ(x)
        return [
            markov_blanket(test, t, 30, 0.05, ranks, interleaved=interleaved) for t in range(30)
        ]

    assert blankets("0") == blankets("1000000000")


@pytest.mark.parametrize("kind", ["fisherz", "student_t", "normal_scores"])
def test_a_small_grow_scan_keeps_the_per_query_bits(kind):
    """A grow scan below the factor's work, or given fewer than two variables, gives each candidate
    the p-value and key of that candidate asked alone, bit for bit."""
    x = data.sample_scm(graph="erdos_renyi", d=12, n=400, seed=3, density=3.0).data
    test = make_indep_test(kind, x)
    scan = blanket._dependence_scan(test, significant=True)
    one = blanket._dependence(test, significant=True)
    for cond in [[], [7], [2, 9], [0, 3, 8]]:
        xs = [v for v in range(12) if v != 4 and v not in cond]
        for some in (xs, xs[:3]):
            if len(cond) >= 2 and len(some) * (len(cond) + 2) >= 128:
                continue
            got = _scanned(scan, 4, some, cond).tolist()
            assert got == [list(one(4, v, cond)) for v in some], (cond, some)


def _loop_shrink(test, target, alpha, members, keep=(), sepsets=None):
    """The shrink that asks one member at a time: the reference for :func:`blanket._shrink`."""
    dropped = []
    for x in list(members):
        if x in keep:
            continue
        rest = [z for z in members if z != x]
        if test(target, x, rest) > alpha:
            members.remove(x)
            dropped.append(x)
            if sepsets is not None:
                sepsets[x] = tuple(sorted(rest))
    return dropped


class _Spied:
    """Each stacked shrink call of a test: its number of queries and of p-values returned."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[tuple[int, int]] = []
        original = PartialCorrelationTest.pvalues_until_separated

        def spied(test, x, y, conditions, alpha):
            found = original(test, x, y, conditions, alpha)
            self.calls.append((len(x), len(found)))
            return found

        monkeypatch.setattr(PartialCorrelationTest, "pvalues_until_separated", spied)


@pytest.mark.parametrize("kind", ["fisherz", "student_t", "normal_scores"])
@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.7])
def test_a_stacked_shrink_is_the_one_at_a_time_loop_bit_for_bit(kind, alpha, monkeypatch):
    """The shrink that asks the members not yet visited in one call drops the members, in the order
    it drops them, and sets the separating sets, in the order it sets them, as the loop that asks
    one member at a time does, from blankets of two to nine members in a random order, with members
    kept untested and without. It drops a member with others after it, so a stack is asked again,
    and it finds none to drop."""
    x = data.sample_scm(graph="erdos_renyi", d=14, n=300, seed=5, density=3.0).data
    test = make_indep_test(kind, x)
    spy = _Spied(monkeypatch)
    rng = np.random.default_rng(0)
    for _ in range(120):
        target = int(rng.integers(14))
        others = [v for v in range(14) if v != target]
        members = [int(v) for v in rng.permutation(others)[: int(rng.integers(2, 10))]]
        keep = [m for m in members if rng.random() < 0.25]
        got, want = list(members), list(members)
        got_sepsets: dict[int, tuple[int, ...]] = {}
        want_sepsets: dict[int, tuple[int, ...]] = {}
        dropped = blanket._shrink(test, target, alpha, got, keep, got_sepsets)
        assert dropped == _loop_shrink(test, target, alpha, want, keep, want_sepsets)
        assert got == want
        assert list(got_sepsets.items()) == list(want_sepsets.items())
    assert any(found < stack for stack, found in spy.calls)  # a drop with members after it
    assert any(found == stack for stack, found in spy.calls)  # a pass that drops none
    assert any(stack == 2 for stack, _ in spy.calls)


def test_a_shrink_that_cannot_stack_asks_one_member_at_a_time(monkeypatch):
    """A test that is not exactly a Fisher-z, Student-t, or normal-scores test takes the loop, and a
    shrink with fewer than two members to ask takes the scalar call."""
    x = data.sample_scm(graph="erdos_renyi", d=10, n=300, seed=5, density=3.0).data
    spy = _Spied(monkeypatch)

    class Mine(FisherZ):
        pass

    for test in (Mine(x), _OneByOne(FisherZ(x))):
        blanket._shrink(test, 0, 0.5, [1, 2, 3, 4])
    blanket._shrink(FisherZ(x), 0, 0.5, [1, 2, 3], keep=[1, 2])
    blanket._shrink(FisherZ(x), 0, 0.5, [1])
    assert spy.calls == []
    blanket._shrink(FisherZ(x), 0, 0.5, [1, 2, 3], keep=[1])
    assert len(spy.calls) == 1


@pytest.mark.parametrize("kind", ["fisherz", "student_t", "normal_scores"])
@pytest.mark.parametrize("interleaved", [False, True], ids=["iamb", "inter_iamb"])
def test_a_stacked_shrink_leaves_every_blanket_and_separating_set_as_the_loop_does(
    kind, interleaved, monkeypatch
):
    """IAMB and Inter-IAMB, with required members and a depth cap too, give every target the blanket
    and the separating sets that they give with the shrink asking one member at a time, the grow
    step the same in both."""
    monkeypatch.setenv("ANDREY_SHARED_SCAN_MIN_WORK", "1000000000")
    x = data.sample_scm(graph="erdos_renyi", d=16, n=200, seed=7, density=4.0).data
    test = make_indep_test(kind, x)
    ranks = tie_ranks(x).tolist()
    edges = np.zeros((16, 16), dtype=bool)
    edges[2, 3] = edges[5, 6] = True
    required = Constraints(np.zeros((16, 16), dtype=bool), edges, np.full(16, -1))
    spy = _Spied(monkeypatch)
    for alpha, options in (
        (0.05, {}),
        (0.4, {}),
        (0.4, {"max_depth": 3}),
        (0.4, {"constraints": required}),
    ):
        got = _blankets(test, 16, alpha, interleaved, ranks, **options)
        with monkeypatch.context() as one_at_a_time:
            one_at_a_time.setattr(blanket, "canonical_capable", lambda test: False)
            want = _blankets(test, 16, alpha, interleaved, ranks, **options)
        assert got == want
    assert any(found < stack for stack, found in spy.calls)
