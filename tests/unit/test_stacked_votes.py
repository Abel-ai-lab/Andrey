"""The collider vote takes the stacked canonical route with unchanged results.

For FisherZ, StudentT, NormalScores, and the degenerate Gaussian test, ``vote_triples`` evaluates
the subsets of every pair as canonical query arrays in stacks and decides each triple from the
counts of the separating subsets. A subclass of each test takes the pair-by-pair vote, the
reference: every cap on a stack and on a chunk gives its decisions, separating sets (their key
order too), p-values, and queries, each pair's in the order of the pair-by-pair vote. A test that
deletes rows per test, and a test that raises, take the pair-by-pair vote, a raising test with the
same error and the same separating sets rewritten before it. The facades give the same graphs with
every route.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint import _skeleton, pc
from andrey.constraint._skeleton import (
    _separating_votes,
    _stacked_counts,
    _unshielded_triples,
    collider_verdict,
    count_verdict,
    vote_triples,
)
from andrey.constraint.mvpc import PermutationCorrection
from andrey.core import CIRCLE
from andrey.core.ci import (
    DGTest,
    FisherZ,
    NormalScores,
    SingularCorrelationError,
    StudentT,
    make_indep_test,
)
from andrey.core.ci_missing import GaussianDeletion

ALPHA = 0.2


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


class _PerPairFisherZ(FisherZ):
    """A FisherZ subclass, which ``vote_triples`` votes on pair by pair."""


class _PerPairStudentT(StudentT):
    """A StudentT subclass, which ``vote_triples`` votes on pair by pair."""


class _PerPairNormalScores(NormalScores):
    """A NormalScores subclass, which ``vote_triples`` votes on pair by pair."""


class _PerPairDGTest(DGTest):
    """A degenerate Gaussian subclass, which ``vote_triples`` votes on pair by pair."""


TESTS = [
    (FisherZ, _PerPairFisherZ),
    (StudentT, _PerPairStudentT),
    (NormalScores, _PerPairNormalScores),
    (DGTest, _PerPairDGTest),
]


def _case(seed: int, d: int = 14, n: int = 300, p: float = 0.22, mixed: bool = False):
    """A random skeleton, the data of a linear model on it, and the categorical mask."""
    rng = np.random.default_rng(seed)
    upper = np.triu(rng.random((d, d)) < p, 1)
    X = rng.standard_normal((n, d))
    for j in range(d):
        for i in np.flatnonzero(upper[:, j]):
            X[:, j] += 0.8 * X[:, i]
    mask = np.zeros(d, dtype=bool)
    if mixed:
        for j in range(0, d, 3):
            X[:, j] = np.digitize(X[:, j], np.quantile(X[:, j], np.linspace(0, 1, 2 + j % 3)[1:-1]))
            mask[j] = True
    marks = np.where(upper | upper.T, np.int8(CIRCLE), np.int8(0))
    return X, marks, mask


def _build(kind, X, mask):
    return kind(X, mask) if issubclass(kind, DGTest) else kind(X)


def _sepsets(marks: np.ndarray, seed: int) -> dict[tuple[int, int], tuple[int, ...]]:
    """Separating sets recorded before the vote, under one key order or both for some pairs."""
    rng = np.random.default_rng(seed)
    adjacent = marks != 0
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {}
    for a, c in combinations(range(len(marks)), 2):
        if adjacent[a, c] or rng.random() < 0.4:
            continue
        around = np.flatnonzero(adjacent[a] | adjacent[c])
        held = tuple(sorted(rng.choice(around, size=min(2, len(around)), replace=False).tolist()))
        keys = [(a, c), (c, a)] if rng.random() < 0.5 else [(c, a)]
        sepsets.update(dict.fromkeys(keys, held))
    return sepsets


def _vote(test, marks, rule, max_depth=None, skip=frozenset(), seed=0):
    sepsets = _sepsets(marks, seed)
    ambiguous = vote_triples(
        marks.copy(), test, ALPHA, sepsets, rule, max_depth=max_depth, skip=skip
    )
    return ambiguous, list(sepsets.items())


def _no_pair_by_pair_vote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the pair-by-pair vote raise, so a vote that completes ran on the stacked route."""

    def refuse(*args, **kwargs):
        raise AssertionError("the pair-by-pair vote ran")

    monkeypatch.setattr(_skeleton, "_separating_votes", refuse)


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("rule", ["conservative", "majority"])
@pytest.mark.parametrize("max_depth", [None, 1])
@pytest.mark.parametrize(("stack_bytes", "chunk"), [(1, 3), (4096, 7), (None, None)])
@pytest.mark.parametrize("seed", range(2))
def test_the_stacked_vote_equals_the_pair_by_pair_vote(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, rule, max_depth, stack_bytes, chunk, seed
) -> None:
    """Every cap on a stack and on a chunk gives the decisions, the separating sets in their key
    order, and the ambiguous triples of the pair-by-pair vote."""
    X, marks, mask = _case(seed, mixed=issubclass(stacked, DGTest))
    want = _vote(_build(per_pair, X, mask), marks, rule, max_depth, seed=seed)
    if stack_bytes is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", stack_bytes)
        monkeypatch.setattr(_skeleton, "CHUNK", chunk)
    _no_pair_by_pair_vote(monkeypatch)
    got = _vote(_build(stacked, X, mask), marks, rule, max_depth, seed=seed)
    assert got == want
    assert want[1] != list(_sepsets(marks, seed).items()), "the case must decide some triples"


@pytest.mark.parametrize("seed", range(3))
def test_a_skipped_triple_is_neither_voted_nor_tested_for_its_own_sake(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    X, marks, mask = _case(seed)
    triples = list(zip(*(c.tolist() for c in _unshielded_triples(marks != 0, frozenset()))))
    rng = np.random.default_rng(seed)
    skip = frozenset(t for t in triples if rng.random() < 0.5)
    want = _vote(_PerPairFisherZ(X), marks, "majority", skip=skip, seed=seed)
    _no_pair_by_pair_vote(monkeypatch)
    got = _vote(FisherZ(X), marks, "majority", skip=skip, seed=seed)
    assert got == want
    assert not skip & got[0]


def test_every_triple_skipped_asks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    X, marks, _ = _case(0)
    triples = frozenset(zip(*(c.tolist() for c in _unshielded_triples(marks != 0, frozenset()))))
    test = FisherZ(X)
    asked = []
    inner = test._canonical_pvalues
    test._canonical_pvalues = lambda *args: asked.append(1) or inner(*args)
    sepsets: dict = {}
    assert vote_triples(marks, test, ALPHA, sepsets, "majority", skip=triples) == frozenset()
    assert not asked
    assert not sepsets


def _logged(test):
    """``test`` with the queries of its canonical calls appended, in order, to the returned log as
    ``(x, y, set, p)``."""
    log: list[tuple[int, int, tuple[int, ...], float]] = []
    inner = test._canonical_pvalues

    def canonical(xs, ys, conditions):
        values = inner(xs, ys, conditions)
        log.extend(zip(xs.tolist(), ys.tolist(), map(tuple, conditions.tolist()), values.tolist()))
        return values

    test._canonical_pvalues = canonical
    return test, log


def _pair_order(marks: np.ndarray, max_depth: int | None):
    """The subsets each unshielded pair is tested on, in the order of the pair-by-pair vote."""
    adjacent = marks != 0
    neighbors = [np.flatnonzero(row).tolist() for row in adjacent]
    order = {}
    for b in range(len(marks)):
        for a, c in combinations(neighbors[b], 2):
            if not adjacent[a, c] and (a, c) not in order:
                order[a, c] = [
                    subset
                    for end in (a, c)
                    for size in range(
                        len(neighbors[end]) + 1
                        if max_depth is None
                        else min(len(neighbors[end]), max_depth) + 1
                    )
                    for subset in combinations(neighbors[end], size)
                ]
    return order


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS)
@pytest.mark.parametrize("max_depth", [None, 2])
@pytest.mark.parametrize("stack_bytes", [1, None])
def test_each_pair_is_asked_its_subsets_once_in_the_pair_by_pair_order_with_the_same_p_values(
    monkeypatch: pytest.MonkeyPatch, stacked, per_pair, max_depth, stack_bytes
) -> None:
    """The stacked vote asks each pair's subsets of ``a``'s neighbors, then ``c``'s, by increasing
    size, a subset both offer twice, and every p-value is the one the pair-by-pair vote's chunks
    get, bit for bit."""
    X, marks, mask = _case(1, mixed=issubclass(stacked, DGTest))
    order = _pair_order(marks, max_depth)
    reference, _ = _logged(_build(per_pair, X, mask))
    reference_log = _chunk_p_values(reference, order)
    if stack_bytes is not None:
        monkeypatch.setattr(_skeleton, "_STACK_BYTES", stack_bytes)
    test, log = _logged(_build(stacked, X, mask))
    vote_triples(marks, test, ALPHA, {}, "conservative", max_depth=max_depth)
    for (a, c), subsets in order.items():
        asked = [(cond, p) for x, y, cond, p in log if (x, y) == (a, c)]
        assert [cond for cond, _ in asked] == subsets
        assert [p for _, p in asked] == reference_log[a, c]
    assert len(log) == sum(len(subsets) for subsets in order.values())


def _chunk_p_values(test, order):
    """The p-values the pair-by-pair vote's chunks of ``_VOTE_CHUNK`` subsets get, by pair."""
    values = {}
    for (a, c), subsets in order.items():
        found = []
        for start in range(0, len(subsets), _skeleton._VOTE_CHUNK):
            found.extend(test.batched_call(a, c, subsets[start : start + _skeleton._VOTE_CHUNK]))
        values[a, c] = found
    return values


def _reference_verdict(separating, b, rule, skipped=()):
    """The decision of ``collider_verdict`` from lists of subsets, as it was written before it took
    counts."""

    def verdict(sets):
        if not sets:
            return "ambiguous"
        share = sum(b in subset for subset in sets) / len(sets)
        if rule == "conservative":
            return "collider" if share == 0 else "non-collider" if share == 1 else "ambiguous"
        return "collider" if share < 0.5 else "non-collider" if share > 0.5 else "ambiguous"

    decided = verdict(separating)
    if not skipped or decided == "ambiguous":
        return decided
    holding = [subset for subset in skipped if b in subset]
    lacking = [subset for subset in skipped if b not in subset]
    for extra in (holding, lacking):
        if verdict([*separating, *extra]) != decided:
            return "ambiguous"
    return decided


def test_the_count_verdict_is_the_verdict_of_the_subsets() -> None:
    """Every share, with and without untested subsets, under both rules: ``count_verdict`` and
    ``collider_verdict`` are the earlier list form's decisions."""
    sets = [(0,), (0, 1), (1,), (0, 2), (), (2, 1)]
    seen = set()
    rng = np.random.default_rng(0)
    for rule, n, m in product(["conservative", "majority"], range(7), range(5)):
        for _ in range(20):
            separating = [sets[i] for i in rng.integers(0, len(sets), n)]
            skipped = [sets[i] for i in rng.integers(0, len(sets), m)]
            want = _reference_verdict(separating, 0, rule, skipped)
            assert collider_verdict(separating, 0, rule, skipped) == want
            holding = sum(0 in subset for subset in separating)
            skipped_holding = sum(0 in subset for subset in skipped)
            assert count_verdict(n, holding, rule, m, skipped_holding) == want
            seen.add(want)
    assert seen == {"collider", "non-collider", "ambiguous"}


def test_the_counts_are_the_pair_by_pair_lists_with_untested_subsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For every triple the counts of the stacked route are those of the lists the pair-by-pair
    vote keeps, with some queries not tested (NaN) and a subset both ends offer counted twice."""
    X, marks, _ = _case(2)
    test = FisherZ(X)

    def gap(x, y, cond) -> bool:
        return (x + 3 * y + sum(cond) * 7) % 5 == 0

    def scalar(x, y, cond=()):
        return float("nan") if gap(x, y, cond) else test(x, y, cond)

    def canonical(xs, ys, conds):
        values = test._canonical_pvalues(xs, ys, conds)
        return np.where([gap(x, y, c) for x, y, c in zip(xs, ys, conds.tolist())], np.nan, values)

    adjacent = marks != 0
    triples = _unshielded_triples(adjacent, frozenset())
    neighbors = [np.flatnonzero(row) for row in adjacent]
    for max_depth in (None, 1):
        got = _stacked_counts(canonical, neighbors, triples, ALPHA, max_depth)
        assert got is not None
        lists = [row.tolist() for row in neighbors]
        votes = {}
        want = []
        for a, b, c in zip(*(column.tolist() for column in triples)):
            if (a, c) not in votes:
                votes[a, c] = _separating_votes(scalar, a, c, lists, ALPHA, max_depth)
            found, skipped = votes[a, c]
            want.append(
                (
                    len(found),
                    sum(b in s for s in found),
                    len(skipped),
                    sum(b in s for s in skipped),
                )
            )
        assert list(zip(*(count.tolist() for count in got))) == want
        assert sum(skipped for *_, skipped, _ in want) > 0
        assert any(separating and holding for separating, holding, *_ in want)


def _singular_case():
    """Two unshielded triples, the second over a pair whose size-3 subset of its neighbors is
    singular: column 8 is the sum of 6 and 7."""
    d = 9
    X = np.random.default_rng(4).standard_normal((200, d))
    X[:, 1] += X[:, 0] + X[:, 2]  # a collider at 1, which the first pair's vote removes from (0, 2)
    X[:, 8] = X[:, 6] + X[:, 7]
    marks = np.zeros((d, d), dtype=np.int8)
    for u, v in [(0, 1), (1, 2), (3, 5), (4, 5), (3, 6), (3, 7), (3, 8)]:
        marks[u, v] = marks[v, u] = CIRCLE
    return X, marks


@pytest.mark.parametrize(("stacked", "per_pair"), TESTS[:2])
def test_a_singular_query_raises_the_pair_by_pair_error_after_the_same_rewrites(
    stacked, per_pair
) -> None:
    X, marks = _singular_case()
    results = []
    for kind in (stacked, per_pair):
        sepsets = {(0, 2): (1,), (3, 4): (5, 6)}
        with pytest.raises(SingularCorrelationError) as raised:
            vote_triples(marks, kind(X), ALPHA, sepsets, "majority")
        results.append((str(raised.value), sepsets))
    assert results[0] == results[1]
    assert results[0][1][0, 2] == (), "the first pair is decided before the singular one raises"


def test_a_stacked_vote_that_cannot_evaluate_votes_pair_by_pair() -> None:
    """A test whose canonical call raises takes the pair-by-pair vote: its answers are the scalar
    ones, not an error."""
    X, marks, _ = _case(3)
    test = FisherZ(X)

    def refuse(xs, ys, conditions):
        raise RuntimeError("no canonical call")

    test._canonical_pvalues = refuse
    # batched_call reads the canonical call for same-size batches, so only the scalar path stays.
    test.batched_call = lambda a, c, subsets: [FisherZ.__call__(test, a, c, s) for s in subsets]
    want = _vote(_PerPairFisherZ(X), marks, "majority")
    assert _vote(test, marks, "majority") == want


def test_a_test_that_deletes_rows_per_test_votes_pair_by_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, marks, _ = _case(0)
    X = X.copy()
    X[np.random.default_rng(0).random(X.shape) < 0.05] = np.nan
    test = make_indep_test("fisherz", X)
    assert isinstance(test, GaussianDeletion)

    def refuse(*args, **kwargs):
        raise AssertionError("the stacked vote ran")

    monkeypatch.setattr(_skeleton, "_stacked_counts", refuse)
    assert vote_triples(marks, test, ALPHA, {}, "majority") is not None


def test_a_test_with_no_canonical_route_votes_pair_by_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    X, marks, _ = _case(0)

    inner = FisherZ(X)

    class ScalarCall:
        def __call__(self, a, c, subset=()):
            return inner(a, c, subset)

    def refuse(*args, **kwargs):
        raise AssertionError("the stacked vote ran")

    monkeypatch.setattr(_skeleton, "_stacked_counts", refuse)
    want = _vote(_PerPairFisherZ(X), marks, "majority")
    assert _vote(ScalarCall(), marks, "majority") == want


def _missing_pc_data():
    d = data.sample_scm(d=8, n=500, density=2.0, seed=2, scale="raw").data
    d = d.copy()
    d[np.random.default_rng(2).random(d.shape) < 0.1] = np.nan
    return d


@pytest.mark.parametrize("rule", ["conservative", "majority"])
def test_the_mvpc_vote_runs_the_correction_pair_by_pair(
    monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    """``pc(missing="mvpc")`` votes with the correction object of its skeleton search, which has no
    canonical route, so the stacked vote does not see it."""
    X = _missing_pc_data()
    tests = []
    vote = pc.vote_triples

    def spy(marks, test, *args, **kwargs):
        tests.append(type(test))
        return vote(marks, test, *args, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("the stacked vote ran")

    monkeypatch.setattr(pc, "vote_triples", spy)
    monkeypatch.setattr(_skeleton, "_stacked_counts", refuse)
    andrey.pc(X, missing="mvpc", collider_rule=rule)
    assert tests == [PermutationCorrection]


def _fit(engine: str, X, **options):
    result = getattr(andrey, engine)(X, **options)
    return result.structure.to_numpy(), result.metadata


@pytest.mark.parametrize(
    "engine",
    ["pc", "fci", "gfci", "fci_jci"],
)
@pytest.mark.parametrize("rule", ["conservative", "majority"])
@pytest.mark.parametrize("indep_test", ["fisherz", "student_t", "normal_scores"])
def test_the_facades_give_the_same_results_on_every_route(
    monkeypatch: pytest.MonkeyPatch, engine, rule, indep_test
) -> None:
    """With the canonical route off for every test, the fit takes the pair-by-pair vote, GFCI's
    pair-by-pair separating sets, and the pair-by-pair scans, and returns the graph and the metadata
    of the stacked fit."""
    if engine == "fci_jci" and indep_test == "normal_scores":
        pytest.skip("fci_jci does not offer normal_scores")
    sample = data.sample_scm(d=14, n=500, density=2.5, seed=5, scale="raw")
    X = sample.data
    options = {"collider_rule": rule, "indep_test": indep_test}
    if engine == "fci_jci":
        options["env"] = np.random.default_rng(0).integers(0, 3, len(X))
    calls = Counter()
    votes = _skeleton._separating_votes

    def counted(*args, **kwargs):
        calls["pair by pair"] += 1
        return votes(*args, **kwargs)

    monkeypatch.setattr(_skeleton, "_separating_votes", counted)
    stacked = _fit(engine, X, **options)
    assert not calls
    monkeypatch.setattr(_skeleton, "_canonical_route", lambda test: None)
    reference = _fit(engine, X, **options)
    assert calls, "the reference must vote pair by pair"
    np.testing.assert_array_equal(stacked[0], reference[0])
    assert stacked[1] == reference[1]


def test_the_stacked_vote_asks_the_queries_of_every_pair_once() -> None:
    X, marks, _ = _case(1)
    test, log = _logged(FisherZ(X))
    vote_triples(marks, test, ALPHA, {}, "majority")
    assert len(log) == sum(len(subsets) for subsets in _pair_order(marks, None).values())
