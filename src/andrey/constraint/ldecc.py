"""LDECC [1]_: the local structure around a treatment by eager collider checks.

:func:`ldecc` learns the treatment ``x``'s Markov blanket with a learner of
:data:`~andrey.constraint.blanket.LEARNERS` (Fig. 5, line 1), its neighbors among the blanket
(:func:`neighbors`, line 2), and the children that form an unshielded collider with a spouse
(:func:`uc_children`, line 3). It then runs PC's tests over the pairs of the other variables (Fig.
12) and reads each independence ``a _||_ b | S`` as it is found (lines 6-22): two neighbors
separated without ``x`` that turn dependent given ``S`` and ``x`` are parents, as are the neighbors
in ``S``; two neighbors separated by a set holding ``x`` form a non-collider at ``x``, and an
undirected neighbor that makes them dependent again is a child; otherwise, when ``a`` and ``b`` turn
dependent given ``S`` and ``x``, ``x`` is a collider or a descendant of one between them, and the
minimal neighbor separators of ``a`` and ``b`` (:func:`minimal_separator`) are parents
(:func:`ecc_parents`, Prop. 4). A neighbor joined to a parent by a non-collider is a child. The
tests stop once every neighbor is directed.

Five steps follow the authors' implementation of [1]_ where its text reads them more broadly. Fig. 5
marks two neighbors separated without ``x`` as parents; here only when they turn dependent given
``S`` and ``x``. Fig. 4's UCChildren tests a neighbor against a spouse given the one set that
separated the spouse from ``x``; :func:`uc_children` tests it given every subset of the neighbors
that separates them, and marks a child only when all agree. Lines 10-14 test every neighbor outside
``S`` as a child; here only the neighbors still undirected are tested. ECCParents with one neighbor
marks it and the other's minimal neighbor separator; here only when that separator holds the
neighbor and one more variable, the parent at which the collider lies (:func:`ecc_parents`). Lines
24-28 keep a parent set that holds no non-collider pair; here the pairs are checked among the
undirected neighbors added to the parents. With exact tests they gave the text's answers on every
graph tested; with finite samples they mark fewer parents as children and fewer children as
parents, the errors that most often separate a fit from the treatment's true parents.

The neighbors are the blanket members that no subset of the rest of the blanket separates from
``x`` ([2]_ Lemma A.2): each member is tested against the subsets of ``MB(x)`` without it, never
against ``x``'s remaining neighbors, the rule [2]_ (App. A.1.1) gives for the LocalPC step of [1]_
(Fig. 2), so a spouse that descends from ``x`` is not kept as a neighbor.

Every query is asked once (:class:`Tests`). Variables, neighbors, and subsets are visited in the
order of ``search.ranks`` (:func:`~andrey.core.ties.tie_ranks`), the treatment's neighbors first
and ``x`` first among the members of a conditioning set, the order of the authors' code; the first
subset that separates a pair is its separating set. So the answer does not depend on the column
order, and does not depend on the columns' units when the test does not.

References
----------
.. [1] Gupta, S., Childers, D., and Lipton, Z. C. (2023). Local causal discovery for estimating
   causal effects. Proceedings of the Second Conference on Causal Learning and Reasoning,
   Proceedings of Machine Learning Research 213, 408-447.
.. [2] Schubert, M., Claassen, T., and Magliacane, S. (2026). Local causal discovery for
   statistically efficient causal inference. Proceedings of the 29th International Conference
   on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research 300.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from andrey.constraint.blanket import LocalSearch

#: A query: the lower and the higher variable and the sorted conditioning set.
Query = tuple[int, int, tuple[int, ...]]
#: The step that finds the treatment's neighbors from its blanket: ``(tests, x, mb)`` to the
#: neighbors and the separating set of each blanket member left out.
Adjacency = Callable[["Tests", int, Sequence[int]], tuple[list[int], dict[int, tuple[int, ...]]]]


class Tests:
    """The p-values of the queries asked of ``search``, each asked once.

    ``allowed`` says whether a conditioning set of a given size is within the search's
    ``max_depth``; a query beyond it is never asked.
    """

    def __init__(self, search: LocalSearch) -> None:
        self.search = search
        self.alpha = search.alpha
        self.ranks = search.ranks
        self.max_depth = search.max_depth
        self.kept: dict[Query, float] = {}

    @staticmethod
    def key(a: int, b: int, cond: Iterable[int]) -> Query:
        return (min(a, b), max(a, b), tuple(sorted(cond)))

    def allowed(self, size: int) -> bool:
        return self.max_depth is None or size <= self.max_depth

    def ask(self, a: int, bs: Sequence[int], cond: Sequence[int] = ()) -> None:
        """Keep the p-value of ``a`` against each of ``bs`` given ``cond``, asking the test
        together for those not kept yet."""
        new = [b for b in bs if self.key(a, b, cond) not in self.kept]
        if new:
            _, pvalues = self.search.scan(a, new, tuple(cond))
            for b, p in zip(new, pvalues, strict=True):
                self.kept[self.key(a, b, cond)] = float(p)

    def independent(self, a: int, b: int, cond: Sequence[int] = ()) -> bool:
        """Whether ``a`` and ``b`` test independent given ``cond``: a p-value above ``alpha``."""
        key = self.key(a, b, cond)
        if key not in self.kept:
            low, high = (a, b) if self.ranks[a] < self.ranks[b] else (b, a)
            self.ask(low, [high], cond)
        return self.kept[key] > self.alpha

    def by_rank(self, nodes: Iterable[int]) -> list[int]:
        return sorted(nodes, key=lambda v: self.ranks[v])

    def first_separating(
        self, a: int, b: int, among: Sequence[int], size: int
    ) -> tuple[int, ...] | None:
        """The first subset of ``size`` members of ``among`` (in its order) that separates ``a``
        and ``b``, sorted, or ``None``."""
        for cond in combinations(among, size):
            if self.independent(a, b, cond):
                return tuple(sorted(cond))
        return None


@dataclass(frozen=True)
class Result:
    """LDECC's answer at the treatment.

    ``blanket`` is the Markov blanket learned, ``neighbors`` the members adjacent to the
    treatment, and ``parents``, ``children``, and ``unoriented`` split the neighbors (a neighbor
    marked both a parent and a child is a child, as in the authors' code). ``non_colliders`` holds
    the pairs of neighbors marked as a non-collider at the treatment, and ``parent_sets`` the
    locally valid parent sets: the parents with each subset of the unoriented neighbors that holds
    no marked pair (Fig. 5, lines 24-28), each sorted, the sets in the order of their sizes.
    """

    blanket: tuple[int, ...]
    neighbors: tuple[int, ...]
    parents: frozenset[int]
    children: frozenset[int]
    unoriented: frozenset[int]
    non_colliders: frozenset[frozenset[int]]
    parent_sets: tuple[tuple[int, ...], ...]


def neighbors(
    tests: Tests, x: int, mb: Sequence[int]
) -> tuple[list[int], dict[int, tuple[int, ...]]]:
    """The members of ``mb`` adjacent to ``x``, and the separating set of each one left out.

    Fig. 2's LocalPC on ``MB+(x)`` with ``MB(x)`` in place of the remaining neighbors at its
    lines 3 and 5 ([2]_ App. A.1.1): at each size ``s`` from 0 while ``|MB(x)| > s``, each
    remaining member ``v`` is tested against the subsets of size ``s`` of ``MB(x) \\ {v}``, and the
    first that separates it removes it. No set exceeds the search's ``max_depth``.
    """
    blanket = tests.by_rank(mb)
    adjacent = list(blanket)
    sepsets: dict[int, tuple[int, ...]] = {}
    size = 0
    while len(blanket) > size and tests.allowed(size):
        if size == 0:
            tests.ask(x, adjacent)
        for v in list(adjacent):
            found = tests.first_separating(x, v, [u for u in blanket if u != v], size)
            if found is not None:
                adjacent.remove(v)
                sepsets[v] = found
        size += 1
    return adjacent, sepsets


def uc_children(tests: Tests, x: int, mb: Sequence[int], ne: Sequence[int]) -> set[int]:
    """Fig. 4's UCChildren as the authors' code reads it: a neighbor ``c`` is a child
    (``x -> c <- d``) when, for some spouse ``d``, a subset of the neighbors ``ne`` separates ``d``
    from ``x``, and every such subset leaves ``c`` out and leaves ``c`` dependent on ``d``.

    The subsets run from size 0 to ``|ne|`` in rank order; a spouse that no subset separates
    marks no child.
    """
    pool = tests.by_rank(ne)
    spouses = tests.by_rank(set(mb) - set(ne))
    sizes = [size for size in range(len(pool) + 1) if tests.allowed(size)]
    children: set[int] = set()
    for c in pool:
        for d in spouses:
            separators = (
                cond
                for size in sizes
                for cond in combinations(pool, size)
                if tests.independent(x, d, cond)
            )
            marked = False
            for cond in separators:
                marked = c not in cond and not tests.independent(c, d, cond)
                if not marked:
                    break
            if marked:
                children.add(c)
                break
    return children


def minimal_separator(
    tests: Tests,
    x: int,
    v: int,
    ne: Sequence[int],
    cache: dict[int, tuple[int, ...] | None],
) -> tuple[int, ...] | None:
    """Fig. 4's GetMNS: the smallest subset of the neighbors ``ne`` that separates ``v`` from
    ``x``, the first of its size in rank order, or ``None`` when none does; kept in ``cache``.

    Sizes run from 0 to ``|ne|``, the full set included, as in the authors' code.
    """
    if v not in cache:
        found = None
        pool = tests.by_rank(ne)
        for size in range(len(pool) + 1):
            if not tests.allowed(size):
                break
            found = tests.first_separating(x, v, pool, size)
            if found is not None:
                break
        cache[v] = found
    return cache[v]


def ecc_parents(
    tests: Tests,
    x: int,
    a: int,
    b: int,
    ne: Sequence[int],
    check: bool,
    cache: dict[int, tuple[int, ...] | None],
) -> set[int]:
    """Fig. 4's ECCParents: the parents an eager collider check between ``a`` and ``b`` marks.

    With one of them a neighbor, the other's minimal neighbor separator, when it holds that
    neighbor and at least one more variable, as the authors' code reads the check: the collider it
    finds is ``nbr -> v <- other`` at a parent ``v`` of ``x``. With neither a neighbor, the minimal
    neighbor separators of both; with ``check``, only when they are equal. When a separator is not
    found, none (as the authors' code).
    """
    if (a in ne) != (b in ne):
        nbr, other = (a, b) if a in ne else (b, a)
        found = minimal_separator(tests, x, other, ne, cache)
        if found is None or len(found) < 2 or nbr not in found:
            return set()
        return set(found)
    sep_a = minimal_separator(tests, x, a, ne, cache)
    sep_b = minimal_separator(tests, x, b, ne, cache)
    if sep_a is None or sep_b is None or (check and sep_a != sep_b):
        return set()
    return set(sep_a) | set(sep_b)


def ldecc(
    search: LocalSearch,
    x: int,
    *,
    blanket: str = "iamb",
    check: bool = True,
    adjacency: Adjacency | None = None,
) -> Result:
    """Fig. 5 of [1]_ for the treatment ``x``; see the module docstring.

    ``blanket`` names the Markov blanket learner and ``check`` is ECCParents' ``check``.
    ``adjacency`` replaces :func:`neighbors` as the step of line 2.
    """
    tests = Tests(search)
    _, mb, _ = search.local(blanket, x)
    mb = [int(v) for v in (mb or [])]
    ne, _ = (adjacency or neighbors)(tests, x, mb)
    ne_set = set(ne)
    children = uc_children(tests, x, mb, ne)
    parents: set[int] = set()
    marked: set[frozenset[int]] = set()
    cache: dict[int, tuple[int, ...] | None] = {}

    def unoriented() -> set[int]:
        return ne_set - parents - children

    def read(a: int, b: int, cond: tuple[int, ...]) -> None:
        """Lines 7-20 for the independence ``a _||_ b | cond``."""
        held = set(cond)
        if a in ne_set and b in ne_set and x not in held:
            wider = (*cond, x)
            if tests.allowed(len(wider)) and not tests.independent(a, b, wider):
                parents.update((a, b), held & ne_set)
        elif a in ne_set and b in ne_set:
            marked.add(frozenset((a, b)))
            for v in tests.by_rank(unoriented() - held - {a, b}):
                wider = (*cond, v)
                if tests.allowed(len(wider)) and not tests.independent(a, b, wider):
                    children.add(v)
        elif x not in held:
            wider = (*cond, x)
            if tests.allowed(len(wider)) and not tests.independent(a, b, wider):
                parents.update(ecc_parents(tests, x, a, b, ne, check, cache))
        open_ = unoriented()
        for pair in marked:
            for p, c in (tuple(pair), tuple(pair)[::-1]):
                if p in parents and c in open_:
                    children.add(c)

    d = search.d
    adjacent = np.ones((d, d), dtype=bool)
    np.fill_diagonal(adjacent, False)
    adjacent[x, :] = adjacent[:, x] = False
    adjacent[x, ne] = adjacent[ne, x] = True
    others = [v for v in range(d) if v != x]
    nodes = [*tests.by_rank(set(others) & ne_set), *tests.by_rank(set(others) - ne_set)]
    place = {v: i for i, v in enumerate(nodes)}
    size = 0
    while unoriented() and tests.allowed(size):
        testable = False
        for a in nodes:
            if not unoriented():
                break
            if np.count_nonzero(adjacent[a]) - 1 < size:
                continue
            pairs = sorted(
                (b for b in np.flatnonzero(adjacent[a]).tolist() if b != x), key=place.__getitem__
            )
            if size == 0:  # the tests read no neighbor, so asking them together changes nothing
                tests.ask(a, pairs)
            for b in pairs:
                if not adjacent[a, b]:
                    continue
                rest = [v for v in np.flatnonzero(adjacent[a]).tolist() if v not in (b, x)]
                pool = ([x] if adjacent[a, x] else []) + tests.by_rank(rest)
                if len(pool) < size:
                    continue
                testable = True
                found = tests.first_separating(a, b, pool, size)
                if found is not None:
                    adjacent[a, b] = adjacent[b, a] = False
                    read(a, b, found)
                    if not unoriented():
                        break
        if not testable:
            break
        size += 1
    parents -= children
    open_ = unoriented()
    sets = []
    for k in range(len(open_) + 1):
        for extra in combinations(tests.by_rank(open_), k):
            if not any(pair <= set(extra) for pair in marked):
                sets.append(tuple(sorted(parents | set(extra))))
    return Result(
        blanket=tuple(tests.by_rank(mb)),
        neighbors=tuple(ne),
        parents=frozenset(parents),
        children=frozenset(children),
        unoriented=frozenset(open_),
        non_colliders=frozenset(marked),
        parent_sets=tuple(sets),
    )
