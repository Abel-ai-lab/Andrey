"""PCMCI+: the lagged and contemporaneous causal links of a multivariate time series.

PCMCI+ extends PCMCI to links within one time step (lag 0). It works on the copies
``X^i_{t-tau}`` of the variables, ``tau = 0 .. tau_max``, in four phases:

1. **Lagged skeleton.** :func:`~andrey.temporal.pcmci.pc1` selects the lagged parents ``B(j)``
   of each variable ``X^j_t`` among the lags ``1 .. tau_max``.
2. **Contemporaneous skeleton.** The graph starts with the links from ``B(j)`` and a link between
   every two variables at lag 0. Round ``p = 0, 1, ...`` tests each linked pair
   ``(X^i_{t-tau}, X^j_t)`` given each subset ``S`` of ``p`` lag-0 neighbors of ``X^j_t``, with
   ``B(j)`` (without ``X^i_{t-tau}``) and ``B(i)`` shifted ``tau`` steps back added to every
   conditioning set. The first p-value above ``pc_alpha`` removes the link and stores ``S`` as its
   separating set. A lag-0 pair is tested from both ends, first from the end with the higher
   :func:`~andrey.core.ties.tie_ranks`, given that end's neighbors. The neighbor sets stay fixed
   during a round and are re-sorted after it, strongest link first, by the smallest absolute
   statistic each link has had; statistics equal to ``KEY_DP`` decimals are ordered by the
   neighbors' :func:`~andrey.core.ties.tie_ranks`. The rounds stop when no linked pair has ``p``
   neighbors left to condition on.
3. **Colliders.** Each unshielded triple ``X^i_{t-tau} - X^k_t - X^j_t`` with a lag-0 link
   ``X^k_t - X^j_t`` is a collider, ``X^i_{t-tau} -> X^k_t <- X^j_t``, by ``collider_rule``:
   ``"sepsets"`` when ``X^k_t`` is not in the separating set; ``"majority"`` and ``"conservative"``
   test ``X^i_{t-tau}`` and ``X^j_t`` again given every subset of the lag-0 neighbors of
   ``X^j_t`` (and, at lag 0, of ``X^i_t``), with the phase-2 conditions added, and count the
   separating subsets that hold ``X^k_t``. ``"majority"`` calls the triple a collider when under
   half of them hold it, a non-collider when over half do, and ambiguous at exactly half;
   ``"conservative"`` requires none and all. A triple with no separating subset is ambiguous.
4. **Rules.** Meek's rules orient more lag-0 links: R1 (``X^i_{t-tau} -> X^k_t - X^j_t``, an
   unambiguous unshielded triple, gives ``X^k_t -> X^j_t``), R2 (``X^i_t -> X^k_t -> X^j_t``
   and ``X^i_t - X^j_t`` give ``X^i_t -> X^j_t``), and R3 (``X^i_t - X^k_t -> X^j_t`` and
   ``X^i_t - X^l_t -> X^j_t`` with ``X^k_t``, ``X^l_t`` not linked, ``X^i_t - X^j_t``, and the
   triple ``X^k_t - X^i_t - X^l_t`` unambiguous give ``X^i_t -> X^j_t``), in turn, until none
   applies.

Lagged links always point forward in time. Phases 3 and 4 collect the orientations of a pass from
the graph as it stood before the pass, then apply them together; a link that gets both orientations
is marked as a conflict and takes no part in later rules. Every test reads the lagged design of
:func:`~andrey.temporal.pcmci.lagged_design`, as in PCMCI, built from the variables sorted by
:func:`~andrey.core.ties.tie_ranks` (:func:`~andrey.temporal.pcmci.ranked_design_test`), so a test
gives the same bits under any order of the variables.

References
----------
Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated
nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial
Intelligence (UAI), PMLR 124, 1388-1397. https://proceedings.mlr.press/v124/runge20a.html
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import check_collider_rule, collider_verdict
from andrey.constraint.pc import check_knowledge_cycle
from andrey.core import backend
from andrey.core.ci_missing import (
    NOT_TESTED,
    counts_its_tests,
    keeps_no_untested_sets,
    reads_flags,
    running_counter,
)
from andrey.core.fdr import check_fdr_method, corrected_pvalues
from andrey.core.knowledge import orient as orient_known
from andrey.core.structure import ARROW, NULL, TAIL
from andrey.core.ties import KEY_DP, tie_ranks
from andrey.temporal.pcmci import (
    Link,
    LinkTest,
    conditionable,
    count_not_tested,
    deletes_rows,
    flagged_links,
    pc1,
    ranked_design_test,
)

if TYPE_CHECKING:
    from andrey.core.knowledge import LaggedConstraints

# A triple ``(i, lag, k, j)``: ``X^i_{t-lag} - X^k_t - X^j_t``.
Triple = tuple[int, int, int, int]


@dataclass(frozen=True)
class PCMCIPlusResult:
    """The phases' results of one PCMCI+ run over ``n`` variables.

    ``parents[j]`` holds the lagged ``(i, lag)`` parents phase 1 selected for variable ``j``,
    strongest first. ``adjacency``, ``p_values``, and ``statistics`` have shape
    ``(tau_max + 1, n, n)``, indexed ``[lag, i, j]`` for ``X^i_{t-lag}`` and ``X^j_t``; lag 0 is
    symmetric. ``adjacency`` holds the links after phase 2. A link's p-value is the largest of its
    phase-2 tests, and its statistic that test's statistic; a lagged link phase 1
    removed reports its largest phase-1 test instead, and the lag-0 diagonal is ``1`` and ``0``,
    as is a link the knowledge keeps apart; a required link reads ``0`` and ``1``.
    ``marks`` holds the lag-0 endpoint marks: ``marks[i, j]`` is the mark at ``i`` on the link
    ``i - j``, ``TAIL``/``ARROW`` for ``i -> j``, ``TAIL`` at both ends for an unoriented link, and
    ``ARROW`` at both ends for a conflict. ``sepsets[(i, lag, j)]`` is the lag-0 separating set
    phase 2 stored for a removed link, under both orders at lag 0. ``undecided``, shaped as
    ``adjacency``, marks the links no test with rows decided
    (:data:`~andrey.core.ci_missing.NOT_TESTED`): ``adjacency`` leaves them out, and they read
    ``NaN`` in ``p_values``. ``ambiguous_triples`` lists the
    ``(i, lag, k, j)`` triples phase 3 left ambiguous, a lag-0 triple in both orders.
    """

    parents: tuple[tuple[Link, ...], ...]
    adjacency: np.ndarray
    marks: np.ndarray
    p_values: np.ndarray
    statistics: np.ndarray
    sepsets: dict[tuple[int, int, int], tuple[int, ...]]
    ambiguous_triples: tuple[Triple, ...]
    undecided: np.ndarray | None = None

    @property
    def conflicts(self) -> tuple[tuple[int, int], ...]:
        """The lag-0 links with conflicting orientations, as ``(i, j)`` pairs with ``i < j``."""
        both = (self.marks == ARROW) & (self.marks.T == ARROW)
        return tuple((int(i), int(j)) for i, j in np.argwhere(np.triu(both)))


class Recorder:
    """Passes queries to ``test`` and keeps each pair's largest tested p-value and its statistic."""

    def __init__(self, test: LinkTest) -> None:
        self.test = test
        self.largest: dict[tuple[int, int], tuple[float, float]] = {}

    def conditionable(
        self, x: int, y: int, ordered: Iterable[int], limit: int | None = None
    ) -> list[int]:
        """:func:`~andrey.temporal.pcmci.conditionable` of ``test``."""
        return conditionable(self.test, x, y, ordered, limit)

    def count_not_tested(self, x: int, y: int, queries: int) -> None:
        """:func:`~andrey.temporal.pcmci.count_not_tested` of ``test``."""
        count_not_tested(self.test, x, y, queries)

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        statistics, p_values = self.test.statistic_pvalues(x, y, conditions)
        for pair, statistic, p_value in zip(
            zip(np.asarray(x).tolist(), np.asarray(y).tolist()),
            statistics.tolist(),
            p_values.tolist(),
        ):
            if math.isnan(p_value):  # not tested: no p-value to keep
                continue
            best = self.largest.get(pair)
            if best is None or p_value > best[1]:
                self.largest[pair] = (statistic, p_value)
        return statistics, p_values


def mark_undecided(
    adjacency: np.ndarray,
    p_values: np.ndarray,
    statistics: np.ndarray,
    undecided: np.ndarray,
    untested: list[list[Link]],
) -> None:
    """Mark in place the untested lagged links in ``untested[j]`` missed by phase 2's ``undecided``,
    removing every undecided link from ``adjacency`` because none has a tested result. Each gets
    :data:`~andrey.core.ci_missing.NOT_TESTED` as its p-value and statistic."""
    for j, links in enumerate(untested):
        for i, lag in links:
            undecided[lag, i, j] = True
    adjacency[undecided] = False
    p_values[undecided] = NOT_TESTED
    statistics[np.isnan(p_values)] = NOT_TESTED


def lagged_parents(
    test: LinkTest,
    n_vars: int,
    tau_max: int,
    pc_alpha: float,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
    undecided: list[list[Link]] | None = None,
) -> tuple[tuple[Link, ...], ...]:
    """Phase 1: the PC1 lagged parents of every variable, strongest first; ``max_depth`` caps
    PC1's conditioning sets. With ``undecided``, entry ``j`` gets the candidates of variable
    ``j`` with no tested result (:func:`~andrey.temporal.pcmci.pc1`)."""
    lists: list[list[Link]] = [[] for _ in range(n_vars)]
    parents = tuple(
        pc1(test, n_vars, j, tau_max, pc_alpha, ranks, constraints, max_depth, lists[j])
        for j in range(n_vars)
    )
    if undecided is not None:
        undecided.extend(lists)
    return parents


def _conditions(
    parents: tuple[tuple[Link, ...], ...],
    n_vars: int,
    i: int,
    lag: int,
    j: int,
    subset: Iterable[int],
) -> list[int]:
    """Design columns of ``S``, ``B(j)`` without ``X^i_{t-lag}``, and ``B(i)`` shifted ``lag``."""
    columns = set(subset)  # the lag-0 column of variable k is k
    columns.update(s * n_vars + k for k, s in parents[j] if (k, s) != (i, lag))
    columns.update((s + lag) * n_vars + k for k, s in parents[i])
    return sorted(columns)


def _testable(
    test: LinkTest, n_vars: int, i: int, lag: int, j: int, pool: Iterable[int]
) -> list[int]:
    """Lag-0 variables from ``pool`` individually testable as conditions of ``X^i_{t-lag}`` and
    ``X^j_t`` (:func:`~andrey.temporal.pcmci.conditionable`): on missing data, a variable leaving
    too few jointly observed rows is left out; all are kept on complete data."""
    pool = list(pool)
    if not deletes_rows(test):
        return pool
    x = lag * n_vars + i
    return [k for k in pool if conditionable(test, x, j, [k])]


def _parent_columns(
    parents: tuple[tuple[Link, ...], ...], n_vars: int, i: int, lag: int, j: int
) -> list[int]:
    """Design columns of ``B(j)`` without ``X^i_{t-lag}``, then of ``B(i)`` shifted ``lag``, each
    strongest first, once each."""
    columns = [s * n_vars + k for k, s in parents[j] if (k, s) != (i, lag)]
    shifted = ((s + lag) * n_vars + k for k, s in parents[i])
    return columns + [c for c in shifted if c not in columns]


def _test_subsets(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    i: int,
    lag: int,
    j: int,
    subsets: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    """Test ``X^i_{t-lag}`` against ``X^j_t`` given each subset, all one size, in one call.

    With missing values, each test conditions on its subset and the strongest parents keeping its
    block testable (:meth:`~andrey.core.ci_missing.TestwiseDeletion.conditionable`), excluding
    columns never observed together; omitted parents cause the full set to count as untested
    (:func:`~andrey.temporal.pcmci.count_not_tested`), with one call per condition count.
    """
    x_column = lag * n_vars + i
    if not deletes_rows(test):
        rows = [_conditions(parents, n_vars, i, lag, j, subset) for subset in subsets]
    else:
        defaults = _parent_columns(parents, n_vars, i, lag, j)
        rows = []
        for subset in subsets:
            kept = conditionable(test, x_column, j, [*subset, *defaults])
            whole = kept[: len(subset)] == list(subset)
            if whole and len(kept) < len(subset) + len(defaults):
                count_not_tested(test, x_column, j)
            rows.append(sorted(kept) if whole else _conditions(parents, n_vars, i, lag, j, subset))
    statistics = np.empty(len(rows))
    p_values = np.empty(len(rows))
    for size in sorted({len(row) for row in rows}):
        at = [k for k, row in enumerate(rows) if len(row) == size]
        conditions = np.array([rows[k] for k in at], dtype=np.intp).reshape(len(at), size)
        x = np.full(len(at), x_column, dtype=np.intp)
        found = test.statistic_pvalues(x, np.full_like(x, j), conditions)
        if len(at) == len(rows):
            return found
        statistics[at], p_values[at] = found
    return statistics, p_values


def skeleton(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    tau_max: int,
    pc_alpha: float,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
    max_p_global: int | None = None,
    pool: np.ndarray | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int, int], tuple[int, ...]], np.ndarray
]:
    """Phase 2: return ``(adjacency, largest p-values, their statistics, sepsets, undecided)``.

    ``undecided`` marks adjacent links with untested queries
    (:data:`~andrey.core.ci_missing.NOT_TESTED`) and none tested; untested queries neither remove
    links nor set strength or largest p-value, and lag-0 neighbors still undecided after a round are
    excluded from later conditioning subsets.

    The p-value array holds ``-1`` for a link phase 2 never tested. ``ranks`` holds each
    variable's :func:`~andrey.core.ties.tie_ranks`: a lag-0 pair is tested first from its end of
    higher rank, and neighbors of equal strength are ordered by rank, so the tests a link has, and
    the p-value it reports, do not depend on the order of the variables. With ``constraints``, the
    lag-0 links start from the pairs they allow, and no required link is tested or removed. Every
    test conditions on the entries of ``parents`` for both ends and on a subset of the lag-0
    neighbors; ``max_p_global`` caps the size of that subset, the conditions beyond the parents.
    ``pool``, a mask over the variables, limits the neighbors a subset draws from.
    """
    shape = (tau_max + 1, n_vars, n_vars)
    adjacency = np.zeros(shape, dtype=bool)
    for j, links in enumerate(parents):
        for i, lag in links:
            adjacency[lag, i, j] = True
    adjacency[0] = ~np.eye(n_vars, dtype=bool)
    kept = np.zeros(shape, dtype=bool)  # the required links, both orders at lag 0
    if constraints is not None:
        adjacency[0] &= constraints.contemporaneous().support()
        kept = constraints.required.copy()
        kept[0] |= kept[0].T
    p_values = np.full(shape, -1.0)
    statistics = np.zeros(shape)
    strength = np.full(shape, np.inf)
    skipped = np.zeros(shape, dtype=bool)
    sepsets: dict[tuple[int, int, int], tuple[int, ...]] = {}
    pool = np.ones(n_vars, dtype=bool) if pool is None else pool
    order = sorted(range(n_vars), key=lambda k: (ranks[k], k))
    neighbors = [tuple(np.flatnonzero(adjacency[0, :, j] & pool).tolist()) for j in range(n_vars)]
    p = 0
    while max_p_global is None or p <= max_p_global:
        pairs = [
            (i, lag, j)
            for j in range(n_vars)
            for i, lag in parents[j]
            if adjacency[lag, i, j] and not kept[lag, i, j] and len(neighbors[j]) >= p
        ] + [
            (i, 0, j)  # by rank: a pair first meets the neighbors of its higher-ranked end
            for i in order
            for j in order
            if adjacency[0, i, j] and not kept[0, i, j] and len(neighbors[j]) - int(pool[i]) >= p
        ]
        if not pairs:
            break
        for i, lag, j in pairs:
            if not adjacency[lag, i, j]:
                continue  # removed from its other end in this round
            full = [k for k in neighbors[j] if lag or k != i]
            pool_j = _testable(test, n_vars, i, lag, j, full)
            if lost := math.comb(len(full), p) - math.comb(len(pool_j), p):
                # The subsets with a neighbor left out for want of rows are not tested.
                count_not_tested(test, lag * n_vars + i, j, lost)
            subsets = list(combinations(pool_j, p))
            if not subsets:
                continue  # fewer than p neighbors observed with both ends: no test
            tested_statistics, tested_p_values = _test_subsets(
                test, n_vars, parents, i, lag, j, subsets
            )
            # The subsets count as tested in order up to the first that separates.
            separating = np.flatnonzero(tested_p_values > pc_alpha)
            tested = separating[0] + 1 if separating.size else len(subsets)
            # A subset not tested (NaN) sets neither the strength nor the largest p-value.
            counted = np.flatnonzero(~np.isnan(tested_p_values[:tested]))
            ends = ((lag, i, j), (0, j, i)) if lag == 0 else ((lag, i, j),)
            for index in ends:
                skipped[index] |= counted.size < tested
                if not counted.size:
                    continue
                largest = int(counted[np.argmax(tested_p_values[counted])])
                strength[index] = min(strength[index], np.abs(tested_statistics[counted]).min())
                if tested_p_values[largest] > p_values[index]:
                    p_values[index] = tested_p_values[largest]
                    statistics[index] = tested_statistics[largest]
            if separating.size:
                for index in ends:
                    adjacency[index] = False
                sepsets[(i, lag, j)] = subsets[separating[0]]
                if lag == 0:
                    sepsets[(j, 0, i)] = subsets[separating[0]]
        p += 1
        # A lag-0 neighbor whose link has no tested result conditions no test.
        conditioning = adjacency[0] & ~(skipped[0] & (p_values[0] < 0))
        neighbors = [
            tuple(
                sorted(
                    np.flatnonzero(conditioning[:, j] & pool).tolist(),
                    key=lambda k: (-round(float(strength[0, k, j]), KEY_DP), ranks[k], k),
                )
            )
            for j in range(n_vars)
        ]
    return adjacency, p_values, statistics, sepsets, adjacency & skipped & (p_values < 0)


def _neighbors(adjacency: np.ndarray, j: int) -> list[int]:
    """The lag-0 neighbors of variable ``j``."""
    return np.flatnonzero(adjacency[0, j]).tolist()


def _separating_subsets(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    adjacency: np.ndarray,
    pc_alpha: float,
    i: int,
    lag: int,
    j: int,
    max_p_global: int | None = None,
    middles: Sequence[int] | None = None,
) -> tuple[list[frozenset[int]], list[frozenset[int]]]:
    """Every distinct subset of the lag-0 neighbors of ``j`` (and of ``i`` at lag 0) that
    separates, and every one not tested for want of rows.

    A subset separates ``X^i_{t-lag}`` and ``X^j_t`` when its test with phase-2 conditions gives a
    p-value above ``pc_alpha``. ``max_p_global`` caps subset size; ``middles``, when given, limits
    candidates. Subsets with a neighbor individually untestable as a condition (:func:`_testable`)
    count as untested without being asked.
    """
    families = [_neighbors(adjacency, j)] + ([_neighbors(adjacency, i)] if lag == 0 else [])
    if middles is not None:
        families = [[k for k in family if k in middles] for family in families]
    testable = set(_testable(test, n_vars, i, lag, j, {k for f in families for k in f}))
    largest = max(len(family) for family in families)
    separating = []
    skipped = []
    for size in range(largest + 1 if max_p_global is None else min(largest, max_p_global) + 1):
        subsets = list(
            dict.fromkeys(
                tuple(sorted(subset))
                for family in families
                for subset in combinations(family, size)
            )
        )
        asked = [subset for subset in subsets if testable.issuperset(subset)]
        left = [frozenset(subset) for subset in subsets if not testable.issuperset(subset)]
        if left:
            count_not_tested(test, lag * n_vars + i, j, len(left))
            skipped += left
        if asked:
            _, p_values = _test_subsets(test, n_vars, parents, i, lag, j, asked)
            separating += [frozenset(asked[k]) for k in np.flatnonzero(p_values > pc_alpha)]
            skipped += [frozenset(asked[k]) for k in np.flatnonzero(np.isnan(p_values))]
    return separating, skipped


def apply_orientations(marks: np.ndarray, requests: Iterable[tuple[int, int]]) -> bool:
    """Orient each requested ``(source, target)`` lag-0 link; a link asked both ways is a conflict.

    Return whether any link changed.
    """
    wanted: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for source, target in requests:
        wanted.setdefault((min(source, target), max(source, target)), set()).add((source, target))
    for (a, b), directions in wanted.items():
        if len(directions) == 2:
            marks[a, b] = marks[b, a] = ARROW
        else:
            ((source, target),) = directions
            marks[source, target], marks[target, source] = TAIL, ARROW
    return bool(wanted)


def colliders(
    test: LinkTest,
    n_vars: int,
    parents: tuple[tuple[Link, ...], ...],
    adjacency: np.ndarray,
    sepsets: dict[tuple[int, int, int], tuple[int, ...]],
    pc_alpha: float,
    collider_rule: str,
    constraints: LaggedConstraints | None = None,
    max_p_global: int | None = None,
    middles: Sequence[int] | None = None,
    undecided: np.ndarray | None = None,
    flagged: np.ndarray | None = None,
) -> tuple[np.ndarray, frozenset[Triple]]:
    """Phase 3: return the lag-0 marks with the colliders oriented, and the ambiguous triples.

    Triples over an ``undecided`` link, shaped as ``adjacency`` and removed in phase 2
    (:func:`mark_undecided`), are ambiguous under every rule because the link may be absent. So are
    triples with a side marked by ``flagged``, shaped as ``adjacency``: an untested conditioning set
    might have separated it (:func:`~andrey.temporal.pcmci.flagged_links`).

    With ``constraints``, the lag-0 links they decide are directed first, and no collider reverses
    one. With ``collider_rule="sepsets"``, a triple over a pair they keep apart has no separating
    set to read and is ambiguous. ``max_p_global`` caps the subsets the majority and conservative
    votes read. With ``middles``, a triple's last two variables are among them, and so are the
    subsets the votes read; any variable can open a triple.
    """
    tau_max = adjacency.shape[0] - 1
    marks = np.where(adjacency[0], TAIL, NULL).astype(np.int8)
    apart = np.zeros(adjacency.shape, dtype=bool)
    if constraints is not None:
        lag0 = constraints.contemporaneous()
        marks = orient_known(marks, lag0)
        check_knowledge_cycle((marks == TAIL) & (marks.T == ARROW))
        apart = constraints.forbidden.copy()
        apart[0] = ~lag0.support()
    cache: dict[tuple[int, int, int], tuple[list[frozenset[int]], list[frozenset[int]]]] = {}
    requests: list[tuple[int, int]] = []
    ambiguous: set[Triple] = set()
    inner = range(n_vars) if middles is None else middles
    for k in inner:
        for j in (j for j in _neighbors(adjacency, k) if j in inner):
            far = [
                (i, 0)
                for i in _neighbors(adjacency, k)
                if (i < j or i not in inner) and not adjacency[0, i, j]
            ]
            far += [
                (i, lag)
                for lag in range(1, tau_max + 1)
                for i in range(n_vars)
                if adjacency[lag, i, k] and not adjacency[lag, i, j]
            ]
            for i, lag in far:
                if undecided is not None and undecided[lag, i, j]:
                    verdict = "ambiguous"
                elif collider_rule == "sepsets" and apart[lag, i, j]:
                    verdict = "ambiguous"
                elif collider_rule == "sepsets":
                    verdict = "non-collider" if k in sepsets.get((i, lag, j), ()) else "collider"
                else:
                    key = (min(i, j), 0, max(i, j)) if lag == 0 else (i, lag, j)
                    if key not in cache:
                        cache[key] = _separating_subsets(
                            test, n_vars, parents, adjacency, pc_alpha, *key, max_p_global, middles
                        )
                    separating, skipped = cache[key]
                    verdict = collider_verdict(separating, k, collider_rule, skipped)
                if flagged is not None and (flagged[lag, i, k] or flagged[0, k, j]):
                    verdict = "ambiguous"  # the votes ask the same tests either way
                if verdict == "collider":
                    requests.append((j, k))
                    if lag == 0:
                        requests.append((i, k))
                elif verdict == "ambiguous":
                    ambiguous.add((i, lag, k, j))
                    if lag == 0:
                        ambiguous.add((j, 0, k, i))
    if constraints is not None:
        directed = (marks == TAIL) & (marks.T == ARROW)
        requests = [(source, target) for source, target in requests if not directed[target, source]]
    apply_orientations(marks, requests)
    return marks, frozenset(ambiguous)


def _directed(marks: np.ndarray, a: int, b: int) -> bool:
    return bool(marks[a, b] == TAIL and marks[b, a] == ARROW)


def _undirected(marks: np.ndarray, a: int, b: int) -> bool:
    return bool(marks[a, b] == TAIL and marks[b, a] == TAIL)


def _rule_1(
    marks: np.ndarray, adjacency: np.ndarray, ambiguous: frozenset[Triple]
) -> list[tuple[int, int]]:
    """R1 orientations: ``X^i_{t-lag} -> X^k_t - X^j_t``, unshielded and unambiguous."""
    n_vars, tau_max = marks.shape[0], adjacency.shape[0] - 1
    requests = []
    for k in range(n_vars):
        for j in range(n_vars):
            if not _undirected(marks, k, j):
                continue
            into = [(i, 0) for i in range(n_vars) if _directed(marks, i, k)]
            into += [
                (i, lag)
                for lag in range(1, tau_max + 1)
                for i in range(n_vars)
                if adjacency[lag, i, k]
            ]
            if any(not adjacency[lag, i, j] and (i, lag, k, j) not in ambiguous for i, lag in into):
                requests.append((k, j))
    return requests


def _rule_2(marks: np.ndarray) -> list[tuple[int, int]]:
    """R2 orientations: ``X^i_t -> X^k_t -> X^j_t`` with ``X^i_t - X^j_t``."""
    n_vars = marks.shape[0]
    return [
        (i, j)
        for i in range(n_vars)
        for j in range(n_vars)
        if _undirected(marks, i, j)
        and any(_directed(marks, i, k) and _directed(marks, k, j) for k in range(n_vars))
    ]


def _rule_3(
    marks: np.ndarray, adjacency: np.ndarray, ambiguous: frozenset[Triple]
) -> list[tuple[int, int]]:
    """R3 orientations: ``X^i_t - X^k_t -> X^j_t`` and ``X^i_t - X^m_t -> X^j_t``, k, m apart."""
    n_vars = marks.shape[0]
    requests = []
    for i in range(n_vars):
        for j in range(n_vars):
            if not _undirected(marks, i, j):
                continue
            middles = [
                k for k in range(n_vars) if _undirected(marks, i, k) and _directed(marks, k, j)
            ]
            if any(
                not adjacency[0, k, m] and (k, 0, i, m) not in ambiguous
                for k, m in combinations(middles, 2)
            ):
                requests.append((i, j))
    return requests


def orient(marks: np.ndarray, adjacency: np.ndarray, ambiguous: frozenset[Triple]) -> None:
    """Phase 4: apply R1, R2, and R3 in turn to ``marks`` in place until no rule orients a link."""
    changed = True
    while changed:
        changed = apply_orientations(marks, _rule_1(marks, adjacency, ambiguous))
        changed |= apply_orientations(marks, _rule_2(marks))
        changed |= apply_orientations(marks, _rule_3(marks, adjacency, ambiguous))


def pcmciplus(
    data: np.ndarray,
    *,
    tau_max: int = 1,
    pc_alpha: float = 0.01,
    fdr_method: str = "none",
    collider_rule: str = "majority",
    indep_test: str = "student_t",
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    categorical: np.ndarray | None = None,
    seed: int | None = None,
    degrees_of_freedom: str = "mean",
) -> PCMCIPlusResult:
    """Run the four PCMCI+ phases on a ``(T, n)`` series with the ``indep_test`` test.

    ``fdr_method`` corrects the reported p-values of the lagged links
    (:func:`~andrey.core.fdr.corrected_pvalues`); the links themselves are decided by the
    phases at ``pc_alpha``, so ``adjacency`` and ``marks`` do not depend on it.

    ``constraints``, background knowledge over lags ``0`` to ``tau_max``, reach every phase
    (:func:`run_phases`). ``max_depth`` caps phase 1's conditioning sets and ``max_p_global`` the
    conditions phases 2 and 3 take beyond the lagged parents. ``categorical``, ``seed``, and
    ``degrees_of_freedom`` reach :func:`~andrey.temporal.pcmci.ranked_design_test`, which raises
    on an ``indep_test`` it cannot build. Raises :class:`ValueError` when ``collider_rule`` is not
    one of
    :data:`~andrey.constraint._skeleton.COLLIDER_RULES`.
    """
    check_collider_rule(collider_rule)
    check_fdr_method(fdr_method)
    data = np.asarray(data, dtype=np.float64)
    ranks = tie_ranks(data, categorical).tolist()
    test = ranked_design_test(
        indep_test,
        data,
        tau_max,
        ranks,
        categorical=categorical,
        seed=seed,
        degrees_of_freedom=degrees_of_freedom,
    )
    with backend.one_blas_thread():  # the stacked inversions then give the same bits on any host
        return run_phases(
            test,
            data.shape[1],
            tau_max,
            pc_alpha,
            collider_rule,
            ranks,
            constraints,
            max_depth,
            max_p_global,
            fdr_method,
        )


@counts_its_tests
@keeps_no_untested_sets
def run_phases(
    test: LinkTest,
    n_vars: int,
    tau_max: int,
    pc_alpha: float,
    collider_rule: str,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    fdr_method: str = "none",
) -> PCMCIPlusResult:
    """Run the four phases with ``test`` over the columns of the lagged design.

    ``ranks`` holds each variable's :func:`~andrey.core.ties.tie_ranks`, which orders PC1's equal
    strengths and the ends and equal neighbors of phase 2 (:func:`skeleton`). With
    ``constraints``, no phase tests a forbidden or a required link: phase 1 leaves out the
    forbidden lagged links and keeps the required ones, phase 2 starts the lag-0 links from the
    allowed pairs and removes no required link, and phase 3 directs the lag-0 links the knowledge
    decides before the colliders, which do not reverse them. A link no phase tested
    reports p-value 1 and statistic 0, or 0 and 1 when it is required. ``max_depth`` caps phase 1's
    conditioning sets; ``max_p_global`` caps the subsets of lag-0 neighbors that phase 2 conditions
    on beyond the lagged parents of both ends, and the subsets the collider votes of phase 3 read.
    ``fdr_method`` corrects the lagged ``p_values`` once the phases have recorded them, and only
    them: lag 0 is left as tested, and the corrected values decide no link.
    """
    recorder = Recorder(test)
    untested: list[list[Link]] = []
    parents = lagged_parents(
        recorder, n_vars, tau_max, pc_alpha, ranks, constraints, max_depth, untested
    )
    adjacency, p_values, statistics, sepsets, undecided = skeleton(
        test, n_vars, parents, tau_max, pc_alpha, ranks, constraints, max_p_global
    )
    mark_undecided(adjacency, p_values, statistics, undecided, untested)
    for (x, j), (statistic, p_value) in recorder.largest.items():
        lag, i = divmod(x, n_vars)
        if p_values[lag, i, j] < 0:  # removed in phase 1
            p_values[lag, i, j], statistics[lag, i, j] = p_value, statistic
    p_values[p_values < 0] = 1.0  # the lag-0 diagonal, and the links the knowledge forbids
    if constraints is not None:
        required = constraints.required.copy()
        required[0] |= required[0].T
        p_values[required], statistics[required] = 0.0, 1.0
    if fdr_method != "none":
        p_values[1:] = corrected_pvalues(
            p_values[1:], fdr_method, None if constraints is None else constraints.forbidden[1:]
        )
    required = None if constraints is None else constraints.required
    marks, ambiguous = colliders(
        test,
        n_vars,
        parents,
        adjacency,
        sepsets,
        pc_alpha,
        collider_rule,
        constraints,
        max_p_global,
        undecided=undecided,
        flagged=flagged_links(running_counter(), adjacency, undecided, required)
        if reads_flags()
        else None,
    )
    orient(marks, adjacency, ambiguous)
    return PCMCIPlusResult(
        parents=parents,
        adjacency=adjacency,
        marks=marks,
        p_values=p_values,
        statistics=statistics,
        sepsets=sepsets,
        ambiguous_triples=tuple(sorted(ambiguous)),
        undecided=undecided,
    )
