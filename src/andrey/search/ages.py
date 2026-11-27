"""Aggregative greedy equivalence search, the engine behind :func:`andrey.ages`.

GES with the linear-Gaussian BIC charges every edge one penalty, so the best Insert or Delete on a
graph is the same under every penalty, and only the point where a phase stops moves with it. The
CPDAGs GES returns over all penalties at or above a minimum, its solution path, therefore come
from one forward phase at the minimum penalty and one backward phase per distinct forward result
(:func:`solution_path`). The GES engine (:mod:`andrey.search._parallel_ges`) scans the forward
phase. The covariance inverses, the remove ratios, and the Delete candidates of a backward phase
read no penalty, so the phases share them. :func:`aggregate` overlays the orientations of the
path's CPDAGs on the CPDAG of the minimum penalty. The algorithm, its settings, and its reference
are described on :func:`andrey.ages`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np

from andrey.core import GraphStructure, backend
from andrey.core.orient import meek, pdag2dag, to_structure
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore, _InvCache, remove_ratio
from andrey.core.structure import ARROW, NULL, TAIL
from andrey.core.ties import EPS_IMPROVE, stable_key, tie_mask, tie_ranks
from andrey.search import operators as ops
from andrey.search._parallel_ges import _ForwardScan, _touched, move_key, neighborhood_state
from andrey.search.ges import recompute_objective


def _forward(
    score: BICScore, n: int, ranks: list[int], inverses: _InvCache
) -> tuple[list[np.ndarray], list[float], list[float]]:
    """The forward phase of GES under ``score`` from the empty graph, move by move.

    Returns the CPDAG after 0, 1, ... Inserts, the total score of each, and each Insert's limit:
    the ``penalty_discount`` below which the Insert lowers the score. No gate limits a node's
    directed parents, unlike :func:`andrey.search.ges.ges` without ``max_parents``.
    """
    scan = _ForwardScan(range(n), DeltaBICScore(score, inv_cov_cache=inverses), n, None, ranks)
    graphs = [np.zeros((n, n), dtype=np.int8)]
    totals = [recompute_objective(graphs[0], score, n)]
    limits: list[float] = []
    while (best := scan.best(graphs[-1])) is not None:
        _, x, y, T = best
        graphs.append(ops.insert_cpdag(graphs[-1].copy(), x, y, T))
        totals.append(recompute_objective(graphs[-1], score, n))
        limits.append(score.penalty_discount + (totals[-2] - totals[-1]) / score.log_n)
    return graphs, totals, limits


class _Deletes:
    """What the backward phases of one solution path share, which reads no penalty.

    ``inverses`` holds ``inv(cov[S, S])``, which reads the covariance alone. ``ratios`` holds the
    :func:`~andrey.core.score_delta.remove_ratio` of each ``(y, base, x)``, so a Delete's delta
    under any penalty is its ratio minus ``log_n * penalty_discount``. ``of[y]`` holds the Delete
    candidates ``(x, H, base)`` of target ``y`` on the graph ``adj``, in the order
    :func:`~andrey.search._parallel_ges.backward_chunk_best` visits them.

    A target's candidates read the marks at the target, which of its undirected neighbors each node
    adjacent to it is adjacent to, and which of those neighbors are adjacent to each other. Between
    two graphs a candidate of ``y`` can differ only if a mark at ``y`` changed, or a mark changed
    between two nodes of which one is an undirected neighbor of ``y``. In the second case no mark at
    ``y`` changed, so ``y`` has the same undirected neighbors in both graphs, and it is an
    undirected neighbor in the later graph of a node with a changed mark. Either way ``y`` is one of
    the targets :func:`~andrey.search._parallel_ges._touched` marks, the nodes with a changed mark
    and their undirected neighbors in the later graph, and only those are enumerated again.
    """

    def __init__(self, n: int, inverses: _InvCache) -> None:
        self.inverses = inverses
        self.ratios: dict[tuple[int, tuple[int, ...], int], float | None] = {}
        self.adj: np.ndarray | None = None
        self.of: list[list[tuple[int, tuple[int, ...], tuple[int, ...]]]] = [[] for _ in range(n)]

    def refresh(self, adj: np.ndarray) -> None:
        """Enumerate the candidates of each target that can differ from those kept, on ``adj``."""
        touched = np.flatnonzero(_touched(self.adj, adj)).tolist()
        if touched:
            adj_sets, par_of, nbr = neighborhood_state(adj)
            for y in touched:
                pa_set = set(par_of[y])
                found = []
                for x in sorted(adj_sets[y]):
                    directed = adj[y, x] == ops.ARROW  # x -> y
                    undirected = adj[y, x] == ops.TAIL and adj[x, y] == ops.TAIL  # x -- y
                    if not (directed or undirected):
                        continue
                    h0 = sorted(nbr[y] & adj_sets[x])
                    h0_set = set(h0)
                    for _h_index, H in ops.clique_residuals(adj, h0):
                        base = tuple(sorted((h0_set - set(H)) | pa_set | {x}))
                        found.append((x, H, base))
                self.of[y] = found
        self.adj = adj.copy()


def _backward(
    adj: np.ndarray, score: BICScore, n: int, ranks: list[int], deletes: _Deletes
) -> Iterator[tuple[np.ndarray, float]]:
    """The backward phase of GES under ``score`` from the CPDAG ``adj``, move by move.

    Yields the CPDAG after each Delete and the Delete's limit: the ``penalty_discount`` above which
    the Delete lowers the score. Each Delete is the one
    :func:`~andrey.search._parallel_ges.backward_chunk_best` selects over every target, from the
    candidates and remove ratios of ``deletes``. A Delete whose ratio is ``None`` is scored by the
    two full scores of its move, which read the penalty, so they are kept for this phase only.
    """
    engine = DeltaBICScore(score, inv_cov_cache=deletes.inverses)
    cost = score.log_n * score.penalty_discount
    ratios = deletes.ratios
    fallback: dict[tuple[int, tuple[int, ...], int], float] = {}
    total = recompute_objective(adj, score, n)
    while True:
        deletes.refresh(adj)
        best: tuple[int, int, tuple[int, ...]] | None = None
        best_key: tuple | None = None
        for y in range(n):
            for x, H, base in deletes.of[y]:
                key = (y, base, x)
                if key not in ratios:
                    ratios[key] = remove_ratio(score, y, base, x, deletes.inverses)
                ratio = ratios[key]
                if ratio is not None:
                    delta = float(ratio - cost)
                elif (delta := fallback.get(key)) is None:
                    delta = fallback[key] = engine.delta_many_with_base(y, base, [("remove", x)])[0]
                if delta < -EPS_IMPROVE:
                    if best_key is not None and stable_key(delta, n) > best_key[0]:
                        continue
                    move = move_key(delta, n, x, y, H, ranks)
                    if best_key is None or move < best_key:
                        best_key, best = move, (x, y, H)
        if best is None:
            return
        x, y, H = best
        adj = ops.delete_cpdag(adj.copy(), x, y, H)
        after = recompute_objective(adj, score, n)
        yield adj, score.penalty_discount + (after - total) / score.log_n
        total = after


def solution_path(
    data: np.ndarray, *, penalty_discount: float = 1.0
) -> list[tuple[float, np.ndarray]]:
    """The CPDAGs GES returns under every penalty from ``penalty_discount`` up, by rising penalty.

    Each entry is ``(penalty, cpdag)``: the CPDAG of the forward then backward phase under the
    standard linear-Gaussian BIC (``ebic_gamma=0.0``) with that ``penalty_discount`` and no gate
    on a node's directed parents, the lowest one at or above the minimum that gives the CPDAG.
    The first entry is the CPDAG of the minimum
    penalty; a CPDAG equal to the entry before it is left out.

    The forward phase runs once, at the minimum penalty: under a higher penalty it makes the same
    Inserts and stops at the first whose limit the penalty reaches, so the forward results are the
    graphs before an Insert whose limit is below every earlier one, and the last graph. From each,
    the backward phase runs at the lowest penalty that stops the forward phase there, then at the
    highest: a graph after a Delete is a result when the next Delete's limit is above every
    earlier one. Two limits within :data:`~andrey.core.ties.EPS_IMPROVE` score units are equal.
    """
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    base = BICScore(X, penalty_discount=penalty_discount)
    ranks = tie_ranks(X, tie_mask("bic", None, n)).tolist()
    tol = EPS_IMPROVE / base.log_n

    def at(penalty: float) -> BICScore:
        return BICScore.from_cov(base.cov, base.n, penalty_discount=penalty)

    path: list[tuple[float, np.ndarray]] = []

    def add(penalty: float, cpdag: np.ndarray) -> None:
        if not path or not np.array_equal(path[-1][1], cpdag):
            path.append((penalty, cpdag))

    with backend.one_blas_thread():
        # The inverses, remove ratios, and Delete candidates read no penalty, so each backward phase
        # of the path reuses them.
        inverses = _InvCache()
        deletes = _Deletes(n, inverses)
        graphs, _, limits = _forward(base, n, ranks, inverses)
        for k in range(len(limits), -1, -1):
            low = limits[k] if k < len(limits) else base.penalty_discount
            high = min(limits[:k], default=math.inf)
            if not low < high - tol:
                continue  # no penalty stops the forward phase after k Inserts
            cpdag = graphs[k]
            for cpdag, _ in _backward(cpdag, at(low), n, ranks, deletes):  # noqa: B007
                pass
            add(low, cpdag)
            if math.isinf(high):
                continue  # the empty graph
            moves = list(_backward(cpdag, at(high), n, ranks, deletes))
            top = low
            for j, (after, limit) in enumerate(moves):
                top = max(top, limit)
                if j + 1 == len(moves) or moves[j + 1][1] > top + tol:
                    add(top, after)
    return path


def aggregate(cpdags: Sequence[np.ndarray]) -> tuple[np.ndarray, list[bool]]:
    """Overlay the orientations of ``cpdags`` on the first and close the result under Meek's rules.

    ``cpdags`` are a solution path, by rising penalty. The graph starts as the first CPDAG. Each
    later CPDAG whose skeleton lies within the first's directs, as it does, every edge it directs
    that the graph leaves undirected, when the graph so directed has a consistent DAG extension
    (:func:`~andrey.core.orient.pdag2dag`); otherwise none of its orientations is taken. Meek's
    rules R1 to R4 then direct what follows. Returns the aggregated graph and, per CPDAG, whether
    its orientations were taken.
    """
    graph = np.array(cpdags[0], dtype=np.int8)
    skeleton = graph != NULL
    taken = [True]
    for cpdag in cpdags[1:]:
        fits = not ((cpdag != NULL) & ~skeleton).any()
        if fits:
            directed = (cpdag == TAIL) & (cpdag.T == ARROW)  # [i, j]: i -> j
            new = directed & (graph == TAIL) & (graph.T == TAIL)
            merged = graph.copy()
            merged.T[new] = ARROW
            try:
                pdag2dag(merged)
            except ValueError:
                fits = False
            else:
                graph = merged
        taken.append(fits)
    return meek(graph, rule4=True), taken


def ages(
    data: np.ndarray, *, penalty_discount: float = 1.0
) -> tuple[GraphStructure, list[float], list[bool]]:
    """Aggregate the GES solution path of ``data`` from ``penalty_discount`` up.

    Returns the aggregated graph (``kind="cpdag"``, a maximally oriented partially directed graph
    with the skeleton of the CPDAG of the minimum penalty), the penalties of the path's CPDAGs
    (:func:`solution_path`), and whether each CPDAG's orientations were taken (:func:`aggregate`).
    """
    path = solution_path(data, penalty_discount=penalty_discount)
    graph, taken = aggregate([cpdag for _, cpdag in path])
    return to_structure(graph, kind="cpdag"), [penalty for penalty, _ in path], taken
