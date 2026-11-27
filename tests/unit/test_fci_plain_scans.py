"""FCI's orientation rules and possible-d-sep give what a plain scan of the marks gives.

R4, R5, R6, and the possible-d-sep step read the marks through column and row reads, ascending
``o-o`` lists, and node sets. The functions below are the same searches as loops over every node
and every mark. ``plain_scans`` puts them in place of the ones in ``andrey.constraint.fci``, and
each test compares the two on the same input, bit for bit.
"""

from __future__ import annotations

import importlib
import pickle
import random
import zlib
from collections import Counter, deque
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from itertools import combinations

import numpy as np
import pytest

import andrey
from andrey.constraint._parallel_skeleton import SkeletonWorkerPool
from andrey.core import ARROW, CIRCLE, NULL, TAIL
from andrey.core.ci import CITest
from andrey.data import sample_scm

fci = importlib.import_module("andrey.constraint.fci")


# ---- the plain scans ----------------------------------------------------------------------------


def parents(marks: np.ndarray, b: int) -> set[int]:
    return {p for p in range(marks.shape[0]) if fci._is_parent_of(marks, p, b)}


def rule_r4(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: Mapping[tuple[int, int], tuple[int, ...]],
    change: bool,
    *,
    max_depth: int | None = None,
    out: fci._Implied | None = None,
    flagged: frozenset[tuple[int, int]] = frozenset(),
) -> bool:
    n = marks.shape[0]
    for b in range(n):
        poss_a = [a for a in range(n) if marks[a, b] == ARROW]  # b *-> a
        poss_c = fci._into(marks, b, CIRCLE)  # c *-o b
        for a in poss_a:
            for c in poss_c:
                if not fci._is_parent_of(marks, a, c):
                    continue
                if marks[c, b] != ARROW:
                    continue
                if flagged and (
                    (min(a, b), max(a, b)) in flagged or (min(b, c), max(b, c)) in flagged
                ):
                    continue
                if out is None:
                    change = fci._ddp_orient(
                        marks, a, b, c, test, alpha, sepsets, change, max_depth=max_depth
                    )
                    continue
                for paths in fci._discriminating_paths(marks, a, b, c):
                    decided = False
                    for path in paths:
                        previous = dict(zip(path, path[1:]))
                        oriented, _ = fci._do_ddp(
                            marks,
                            path[0],
                            a,
                            b,
                            c,
                            previous,
                            test,
                            alpha,
                            sepsets,
                            False,
                            max_depth=max_depth,
                            out=out,
                        )
                        decided = oriented or decided
                    if decided:
                        change = True
                        break
    return change


def circle_reaches(marks: np.ndarray, target: int, avoid: int) -> set[int]:
    reachable = {target}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for other in range(marks.shape[0]):
            if (
                other != avoid
                and other not in reachable
                and fci._traverse_circle(marks, node, other)
            ):
                reachable.add(other)
                queue.append(other)
    return reachable


def circle_walk_closes(
    marks: np.ndarray,
    a: int,
    b: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> bool:
    n = marks.shape[0]
    start = [
        (a, c)
        for c in range(n)
        if c != b
        and fci._traverse_circle(marks, a, c)
        and not fci._adjacent(marks, b, c)
        and fci._triple(b, a, c) not in ambiguous
    ]
    seen = set(start)
    queue = deque(start)
    while queue:
        previous, current = queue.popleft()
        for node in range(n):
            if node in (previous, a) or not fci._traverse_circle(marks, current, node):
                continue
            if (
                fci._adjacent(marks, previous, node)
                or fci._triple(previous, current, node) in ambiguous
            ):
                continue
            if node == b:
                if (
                    not fci._adjacent(marks, a, current)
                    and fci._triple(current, b, a) not in ambiguous
                ):
                    return True
            elif (current, node) not in seen:
                seen.add((current, node))
                queue.append((current, node))
    return False


def uncovered_circle_cycle(
    marks: np.ndarray,
    a: int,
    b: int,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
) -> list[int] | None:
    if not circle_walk_closes(marks, a, b, ambiguous):
        return None
    reachable = circle_reaches(marks, b, a)
    stack = [
        [a, c]
        for c in range(marks.shape[0])
        if c != b
        and c in reachable
        and fci._traverse_circle(marks, a, c)
        and not fci._adjacent(marks, b, c)
        and fci._triple(b, a, c) not in ambiguous
    ]
    for _ in range(fci._R5_PARTIAL_PATHS):
        if not stack:
            return None
        path = stack.pop()
        last = path[-1]
        for node in range(marks.shape[0]):
            if node not in reachable or node in path or not fci._traverse_circle(marks, last, node):
                continue
            if (
                fci._adjacent(marks, path[-2], node)
                or fci._triple(path[-2], last, node) in ambiguous
            ):
                continue
            if node != b:
                stack.append([*path, node])
            elif not fci._adjacent(marks, a, last) and fci._triple(last, b, a) not in ambiguous:
                return path[1:]
    return None


def rule_r5(
    marks: np.ndarray,
    change: bool,
    ambiguous: frozenset[tuple[int, int, int]] = frozenset(),
    *,
    out: np.ndarray | fci._Implied | None = None,
) -> bool:
    n = marks.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            if not fci._traverse_circle(marks, a, b):
                continue
            path = uncovered_circle_cycle(marks, a, b, ambiguous)
            if path is not None:
                fci._orient_tails(marks if out is None else out, a, b, path)
                change = True
    return change


def rule_r6(
    marks: np.ndarray, change: bool, *, out: np.ndarray | fci._Implied | None = None
) -> bool:
    out = marks if out is None else out
    n = marks.shape[0]
    for b in range(n):
        if not any(marks[b, a] == TAIL and marks[a, b] == TAIL for a in range(n)):
            continue
        for c in fci._into(marks, b, CIRCLE):
            out[b, c] = TAIL
            change = True
    return change


def possible_dsep(
    marks: np.ndarray,
    x: int,
    y: int,
    neighbors: list[list[int]],
    reach: Callable[[int], int],
) -> list[int]:
    dsep: set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    visited: set[tuple[int, int]] = set()
    previous: dict[int, set[int]] = {}
    first: tuple[int, int] | None = None
    distance = 0

    for b in neighbors[x]:
        if b == y:
            continue
        edge = (x, b)
        if first is None:
            first = edge
        queue.append(edge)
        visited.add(edge)
        previous.setdefault(x, set()).add(b)
        dsep.add(b)

    while queue:
        t = queue.popleft()
        if first == t:
            first = None
            distance += 1
            if distance > fci._UNBOUNDED_PATH:
                break
        a, b = t
        if fci._exists_possible_parent_path(previous, b, x, reach):
            dsep.add(b)
        for c in neighbors[b]:
            if c in (a, x, y):
                continue
            previous.setdefault(c, set()).add(b)
            if fci._is_def_collider(marks, a, b, c) or fci._adjacent(marks, a, c):
                u = (a, c)
                if u in visited:
                    continue
                visited.add(u)
                queue.append(u)
                if first is None:
                    first = u

    dsep.discard(x)
    dsep.discard(y)
    # Sort by descending labels "X{i+1}". Lexicographic order differs from numeric order
    # at two digits; this order decides which separating subset is found and recorded first.
    return sorted(dsep, key=lambda v: f"X{v + 1}", reverse=True)


def iter_dsep_choices(
    dsep: list[int],
    adj_a: set[int],
    adj_b: set[int],
    max_depth: int | None = None,
    tested: Collection[int] = (),
) -> Iterator[tuple[int, ...]]:
    within = (adj_a, adj_b, tested) if tested else (adj_a, adj_b)
    for size in range(1, fci._largest_size(len(dsep), max_depth) + 1):
        yield from (c for c in combinations(dsep, size) if outside(c, within))


def outside(values: tuple[int, ...], within: Iterable[Collection[int]]) -> bool:
    return not any(all(v in nodes for v in values) for nodes in within)


def remove_in_node_order(
    marks: np.ndarray,
    test: CITest,
    alpha: float,
    sepsets: dict[tuple[int, int], tuple[int, ...]],
    keep: np.ndarray | None,
    max_depth: int | None,
    pool: SkeletonWorkerPool | None = None,
) -> None:
    n = marks.shape[0]
    edges = [fci._edge_order(marks, i, j) for i in range(n) for j in range(i + 1, n) if marks[i, j]]
    # The lists and the reach read the marks, which only a removal changes.
    neighbors, reach = fci._neighbor_lists(marks), fci._semidirected_reach(marks)
    for na, nb in edges:
        if not fci._adjacent(marks, na, nb) or (keep is not None and keep[na, nb]):
            continue
        adj_a, adj_b = set(neighbors[na]), set(neighbors[nb])
        tested: set[int] = set()
        for a, b in ((na, nb), (nb, na)):
            dsep = possible_dsep(marks, a, b, neighbors, reach)
            if fci._try_remove_by_dsep(
                marks, na, nb, dsep, adj_a, adj_b, test, alpha, sepsets, max_depth, pool, tested
            ):
                neighbors, reach = fci._neighbor_lists(marks), fci._semidirected_reach(marks)
                break
            tested = set(dsep)


@contextmanager
def plain_scans():
    """Run the code under test with the plain scans in place of the ones that read faster."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fci, "_parents", parents)
        patch.setattr(fci, "_rule_r4", rule_r4)
        patch.setattr(fci, "_rule_r5", rule_r5)
        patch.setattr(fci, "_rule_r6", rule_r6)
        patch.setattr(fci, "_iter_dsep_choices", iter_dsep_choices)
        patch.setattr(fci, "_remove_in_node_order", remove_in_node_order)
        yield


def both(call):
    """What ``call()`` returns as written, then under the plain scans."""
    got = call()
    with plain_scans():
        return got, call()


# ---- random marks -------------------------------------------------------------------------------


def hashed_test(tag):
    """A CI test whose p-value of each query is fixed by the query, uniform on [0, 1)."""

    def test(x, y, condition_set=()):
        key = f"{tag}|{min(x, y)}|{max(x, y)}|{sorted(set(condition_set))}".encode()
        return (zlib.crc32(key) % 10_000) / 10_000.0

    return test


def random_marks(seed):
    """Marks of a ring or a grid with chords, relabeled, with circle, arrowhead, and tail
    endpoints; some planted discriminating paths; random separating sets, ``ambiguous`` triples,
    and ``flagged`` pairs. Chordless rings and grids make overlapping circle cycles, so one R5
    orientation changes what a later pair can use."""
    rng = random.Random(seed)
    if rng.random() < 0.4:
        d = rng.randint(5, 16)
        edges = {(i, (i + 1) % d) for i in range(d)}
    else:
        rows, cols = rng.randint(2, 4), rng.randint(3, 5)
        d = rows * cols
        edges = {(i * cols + j, i * cols + j + 1) for i in range(rows) for j in range(cols - 1)}
        edges |= {(i * cols + j, (i + 1) * cols + j) for i in range(rows - 1) for j in range(cols)}
    edges |= {tuple(rng.sample(range(d), 2)) for _ in range(rng.randint(0, d // 2))}
    order = rng.sample(range(d), d)
    marks = np.zeros((d, d), dtype=np.int8)
    circle = rng.choice([1.0, 0.9, 0.7, 0.5])
    for u, v in edges:
        u, v = order[u], order[v]
        for x, y in ((u, v), (v, u)):
            q = rng.random()
            marks[x, y] = (
                CIRCLE if q < circle else ARROW if q < circle + (1 - circle) * 0.6 else TAIL
            )
    if rng.random() < 0.5:
        for _ in range(rng.randint(1, 3)):
            k = rng.randint(1, 3)
            if d < k + 3:
                break
            first, *inner, b, c = rng.sample(range(d), k + 3)
            path = [first, *inner[::-1], b]
            for x, y in zip(path, path[1:]):
                marks[y, x] = ARROW
                marks[x, y] = ARROW if x != first else rng.choice([CIRCLE, ARROW, TAIL])
            for a in inner:
                marks[c, a], marks[a, c] = ARROW, TAIL
            marks[b, c], marks[c, b] = CIRCLE, ARROW
            marks[first, c] = marks[c, first] = NULL
    sepsets = {}
    for a in range(d):
        for b in range(a + 1, d):
            if marks[a, b] == NULL and rng.random() < 0.8:
                others = [v for v in range(d) if v not in (a, b)]
                sepsets[(a, b)] = tuple(sorted(rng.sample(others, rng.randint(0, min(3, d - 2)))))
    ambiguous, flagged = set(), set()
    for b in range(d):
        nodes = np.flatnonzero(marks[b] != NULL).tolist()
        ambiguous |= {(a, b, c) for a, c in combinations(nodes, 2) if rng.random() < 0.05}
        flagged |= {(min(a, b), max(a, b)) for a in nodes if rng.random() < 0.05}
    return marks, sepsets, frozenset(ambiguous), frozenset(flagged)


def implied(call):
    """``call(out)`` with an ``_Implied`` as ``out``: the marks it records, in order, and its
    return value."""
    out = fci._Implied()
    change = call(out)
    return [(key, frozenset(marks)) for key, marks in out.marks.items()], change


def in_place(marks, call):
    """``call(marks_copy)``: the marks it leaves and its return value."""
    marks = marks.copy()
    change = call(marks)
    return marks, change


def same(got, want):
    if isinstance(got, tuple):
        return len(got) == len(want) and all(same(g, w) for g, w in zip(got, want))
    return np.array_equal(got, want) if isinstance(got, np.ndarray) else got == want


# ---- R4, R5, R6, and fci_orient -----------------------------------------------------------------


def compare_rules(seed):
    """Check R4, R5, and R6 in place and implied, and ``fci_orient`` in rounds and in column order
    with and without selection bias, on the random marks of ``seed``; return which rules fired."""
    marks, sepsets, ambiguous, flagged = random_marks(seed)
    test = hashed_test(seed)
    max_depth = (None, 1)[seed % 2]
    orientations = [0]
    orient_tails = fci._orient_tails

    def counting(*args):
        orientations[0] += 1
        return orient_tails(*args)

    def r4(m, out=None):
        args = (test, 0.05, sepsets, False)
        return fci._rule_r4(m, *args, max_depth=max_depth, out=out, flagged=flagged)

    calls = {
        "r4 in place": lambda: in_place(marks, lambda m: r4(m)),
        "r4 implied": lambda: implied(lambda out: r4(marks, out)),
        "r5 in place": lambda: in_place(marks, lambda m: fci._rule_r5(m, False, ambiguous)),
        "r5 implied": lambda: implied(lambda out: fci._rule_r5(marks, False, ambiguous, out=out)),
        "r6 in place": lambda: in_place(marks, lambda m: fci._rule_r6(m, False)),
        "r6 implied": lambda: implied(lambda out: fci._rule_r6(marks, False, out=out)),
    }
    for column_order in (False, True):
        for selection_bias in (True, False):
            calls[f"fci_orient {column_order} {selection_bias}"] = (
                lambda c=column_order, s=selection_bias: in_place(
                    marks,
                    lambda m: fci.fci_orient(
                        m,
                        test,
                        0.05,
                        sepsets,
                        ambiguous,
                        max_depth=max_depth,
                        in_column_order=c,
                        selection_bias=s,
                        flagged=flagged,
                    ),
                )
            )
    fired = dict.fromkeys(["r4", "r5", "r5 twice", "r6"], False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fci, "_orient_tails", counting)
        for name, call in calls.items():
            orientations[0] = 0
            got, want = both(call)
            assert same(got, want), (name, seed)
            if name in ("r4 in place", "r5 in place", "r6 in place"):
                fired[name[:2]] = bool(want[1])
            if name == "r5 in place":
                fired["r5 twice"] = orientations[0] >= 4  # two or more in each of the two runs
    return fired


@pytest.mark.parametrize("bounds", [(10_000, 10_000), (3, 2)], ids=["default", "small bounds"])
def test_the_rules_equal_the_plain_scans(monkeypatch, bounds):
    """R4, R5, and R6 in place and implied, and ``fci_orient`` in rounds and in column order with
    and without selection bias, on 150 random marks. With the small bounds, R5's and R4's path
    searches stop at their bound, which the plain scans do as well."""
    monkeypatch.setattr(fci, "_R5_PARTIAL_PATHS", bounds[0])
    monkeypatch.setattr(fci, "_R4_PARTIAL_PATHS", bounds[1])
    fired = Counter()
    for seed in range(150):
        fired.update(rule for rule, did in compare_rules(seed).items() if did)
    assert all(fired[rule] >= 10 for rule in ("r4", "r5", "r5 twice", "r6")), fired


def test_the_circle_walk_helpers_equal_the_plain_scans_with_and_without_lists():
    """``_circle_reaches``, ``_circle_walk_closes``, and ``_uncovered_circle_cycle`` build the
    ``o-o`` lists themselves when none is passed, and read the ones passed."""
    checked = 0
    for seed in range(60):
        marks, _, ambiguous, _ = random_marks(seed)
        circles = fci._circle_lists(marks)
        for a, b in zip(*np.nonzero(np.triu((marks == CIRCLE) & (marks.T == CIRCLE), 1))):
            a, b = int(a), int(b)
            for given in (None, circles):
                assert fci._circle_reaches(marks, b, a, given) == circle_reaches(marks, b, a)
                assert fci._circle_walk_closes(marks, a, b, ambiguous, given) == (
                    circle_walk_closes(marks, a, b, ambiguous)
                )
                assert fci._uncovered_circle_cycle(marks, a, b, ambiguous, given) == (
                    uncovered_circle_cycle(marks, a, b, ambiguous)
                )
            checked += 1
    assert checked > 100


def test_the_circle_lists_are_ascending():
    marks, *_ = random_marks(3)
    for node, nodes in enumerate(fci._circle_lists(marks)):
        assert nodes == [v for v in range(len(marks)) if fci._traverse_circle(marks, node, v)]


def test_r5_in_column_order_reads_the_lists_after_each_orientation():
    """Two chordless four-cycles share the edge ``1-4`` of a 2 x 3 grid. R5 orients the first
    cycle ``0, 1, 4, 3`` and its edge ``1-4`` becomes a tail-tail edge, so the second cycle
    ``1, 2, 5, 4`` is no longer a circle cycle and stays. Lists built before the first
    orientation would still hold the edge ``1-4`` and orient the second cycle as well."""
    marks = np.zeros((6, 6), dtype=np.int8)
    for u, v in [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]:
        marks[u, v] = marks[v, u] = CIRCLE
    oriented = marks.copy()
    assert fci._rule_r5(oriented, False)
    tails = {(u, v) for u in range(6) for v in range(u + 1, 6) if oriented[u, v] == TAIL}
    assert tails == {(0, 1), (0, 3), (1, 4), (3, 4)}
    assert all(oriented[u, v] == oriented[v, u] == CIRCLE for u, v in [(1, 2), (2, 5), (4, 5)])
    plain = marks.copy()
    with plain_scans():
        assert fci._rule_r5(plain, False)
    np.testing.assert_array_equal(oriented, plain)
    implied_marks = fci._Implied()
    assert fci._rule_r5(marks, False, out=implied_marks)
    assert len(implied_marks.marks) == 14  # in rounds every circle cycle orients: all seven edges


# ---- possible-d-sep -----------------------------------------------------------------------------


def compare_possible_dsep(seed):
    """Check the possible-d-sep search of every adjacent pair of the random marks of ``seed``
    against the plain search; return the number of pairs."""
    marks, *_ = random_marks(seed)
    neighbors, reach = fci._neighbor_lists(marks), fci._semidirected_reach(marks)
    into = [set(fci._into(marks, b, ARROW)) for b in range(len(marks))]
    near = [set(nodes) for nodes in neighbors]
    pairs = 0
    for x in range(len(marks)):
        for y in neighbors[x]:
            got = fci._possible_dsep(x, y, neighbors, reach, into, near)
            assert got == possible_dsep(marks, x, y, neighbors, reach), (seed, x, y)
            pairs += 1
    return pairs


def test_possible_dsep_equals_the_plain_search():
    """The search over the sets ``into`` and ``near`` gives the nodes of the search over the marks,
    in the same order, for every adjacent pair of 150 random marks."""
    assert sum(compare_possible_dsep(seed) for seed in range(150)) > 1000


def test_the_subsets_of_a_possible_dsep_set_equal_the_plain_filter():
    """The subsets inside neither adjacency nor ``tested``, by size, equal the plain filter's."""
    for seed in range(200):
        rng = random.Random(seed)
        nodes = rng.sample(range(14), rng.randint(2, 9))
        adj_a, adj_b, tested = (set(rng.sample(range(14), rng.randint(0, 7))) for _ in range(3))
        max_depth = rng.choice([None, 1, 2, 3])
        want = list(iter_dsep_choices(nodes, adj_a, adj_b, max_depth, tested))
        assert list(fci._iter_dsep_choices(nodes, adj_a, adj_b, max_depth, tested)) == want
    assert list(fci._iter_dsep_choices([4, 2], {2, 4}, set())) == []
    assert list(fci._iter_dsep_choices([4, 2], {4}, set())) == [(2,), (4, 2)]


def compare_removals(seed, max_depth):
    """Check the removals of ``_remove_in_node_order`` on the random marks of ``seed``, with a test
    that separates one pair in four, against the plain order; return the number of edges removed."""
    marks = random_marks(seed)[0]
    test = hashed_test(seed)

    def run():
        m, sepsets = marks.copy(), {}
        fci._remove_in_node_order(
            m, lambda x, y, s: 0.5 * (test(x, y, s) < 0.25) + 0.01, 0.05, sepsets, None, max_depth
        )
        return m, list(sepsets.items())

    got, want = both(run)
    assert same(got, want), seed
    return (np.count_nonzero(marks) - np.count_nonzero(want[0])) // 2


@pytest.mark.parametrize("max_depth", [None, 2])
def test_removals_in_node_order_equal_the_plain_order(max_depth):
    """The removals, their separating sets, and the marks after them equal the plain scans' on 120
    random marks, most of which remove two or more edges, so a set not updated after a removal
    would show."""
    removed = [compare_removals(seed, max_depth) for seed in range(120)]
    assert sum(count >= 2 for count in removed) >= 60


# ---- the methods --------------------------------------------------------------------------------


def _draw(seed, d=24, n=600, latents=2):
    return sample_scm(d=d, n=n, seed=seed, density=3.0, latents=latents).data


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("tie_break", ["rank", "position"])
def test_fci_equals_the_plain_scans(seed, tie_break):
    """FCI in rounds (``rank``) and in column order (``position``), with each of its
    ``possible_dsep`` values, returns the same output bytes."""
    x = _draw(seed)
    for possible_dsep in ("order_free", "fast"):

        def fit():
            with andrey.config(tie_break=tie_break):
                return pickle.dumps(andrey.fci(x, possible_dsep=possible_dsep))

        got, want = both(fit)
        assert got == want


@pytest.mark.parametrize("seed", range(2))
def test_the_hybrids_equal_the_plain_scans(seed):
    x = _draw(seed, d=16, n=500)
    cohort = np.random.default_rng(seed).integers(0, 3, size=len(x))
    fits = {
        "gfci": lambda: andrey.gfci(x),
        "gfci fast": lambda: andrey.gfci(x, possible_dsep="fast"),
        "fci_jci": lambda: andrey.fci_jci(x, cohort),
        "boss_fci": lambda: andrey.boss_fci(x, seed=0),
        "grasp_fci": lambda: andrey.grasp_fci(x, seed=0),
    }
    for name, fit in fits.items():
        got, want = both(lambda fit=fit: pickle.dumps(fit()))
        assert got == want, name


def test_the_truth_pag_equals_the_plain_scans():
    """The PAG of a DAG of 24 nodes with 4 hidden and 2 selection variables, on 12 DAGs. R4 decides
    on some, and R5 and R6 fire on others."""
    undirected = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        dag = sample_scm(d=24, n=50, seed=seed, density=4.0).graph
        marks = np.asarray(dag.to_numpy())
        parents = ((marks.T == ARROW) & (marks == TAIL)).sum(axis=0)  # directed parents of a node
        latent = rng.choice(24, size=4, replace=False).tolist()
        rest = [v for v in range(24) if v not in latent]
        selection = rng.choice([v for v in rest if parents[v] >= 2] or rest, size=2, replace=False)

        def pag():
            return dag.to_pag(latent=latent, selection=selection.tolist()).to_numpy()

        got, want = both(pag)
        np.testing.assert_array_equal(got, want)
        undirected += int(((want == TAIL) & (want.T == TAIL)).any())
    assert undirected >= 4
