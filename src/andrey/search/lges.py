"""Less greedy equivalence search over a decomposable score, the engine behind :func:`andrey.lges`.

The Insert phase is less greedy than that of GES: at each step it reads one DAG ``G`` of the
current class and leaves out every pair ``(x, y)`` for which ``G`` scores better than ``G`` with
``x -> y`` added (SafeInsert); under ``insertion="conservative"`` it also leaves out a pair when
any valid ``Insert(x, y, T)`` worsens the score (ConservativeInsert). The Delete phase is that of
GES (:func:`~andrey.search._parallel_ges.backward_chunk_best`) and the turning phase that of GIES
on observational data (:func:`~andrey.search.gies._turning_best`). Two schedules run them: LGES
(``schedule="lges"``) deletes, then turns, until neither improves the score, before each single
Insert, and stops when no move improves it; LGES-0 in the phase order of the released code
(``schedule="lges0_released"``) runs Insert, Delete, then turning once, each until no move
improves the score. The algorithm, its settings, and its reference are described on
:func:`andrey.lges`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np

from andrey.core import GraphStructure, _bitset, backend
from andrey.core.orient import pdag2dag, to_structure
from andrey.core.score import BICScore, LocalScore, make_score
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta
from andrey.core.ties import EPS_IMPROVE, stable_key, tie_mask, tie_ranks
from andrey.search import operators as ops
from andrey.search._parallel_ges import backward_chunk_best, move_key, neighborhood_state
from andrey.search.ges import recompute_objective
from andrey.search.gies import FamilyScore, _turn, _turning_best

#: The insertion strategies of :func:`lges`.
INSERTIONS: tuple[str, ...] = ("safe", "conservative")

#: The schedules of :func:`lges`: the paper's Algorithm 1, and LGES-0 in the released code's order.
SCHEDULES: tuple[str, ...] = ("lges", "lges0_released")


def extension(adj: np.ndarray, ranks: Sequence[int]) -> np.ndarray:
    """The DAG of the class of the CPDAG ``adj`` that SafeInsert reads.

    :func:`~andrey.core.orient.pdag2dag` first removes the lowest-numbered sink with adjacent
    neighbors; numbering nodes by ``ranks`` makes the DAG follow column tie ranks rather than
    positions.
    """
    order = np.argsort(np.asarray(ranks), kind="stable")
    dag = pdag2dag(adj[np.ix_(order, order)])
    back = np.argsort(order, kind="stable")
    return dag[np.ix_(back, back)]


def insert_moves(
    adj: np.ndarray,
    score: LocalScore,
    n: int,
    ranks: Sequence[int],
    conservative: bool,
    verdicts: dict[tuple[int, tuple[int, ...]], int] | None = None,
) -> Iterator[tuple[float, int, int, tuple[int, ...]]]:
    """Yield every valid ``Insert(x, y, T)`` the less greedy rule keeps, as ``(delta, x, y, T)``.

    A pair ``(x, y)`` of non-adjacent nodes is left out when ``score(y, Pa_G(y) + x)`` is worse than
    ``score(y, Pa_G(y))`` for the DAG ``G`` of :func:`extension` (SafeInsert). Otherwise every
    valid ``Insert(x, y, T)`` is scored, ``T`` a subset of the neighbors of ``y`` not adjacent to
    ``x`` with ``NA_yx + T`` a clique that blocks every semi-directed path from ``y`` to ``x``;
    with ``conservative``, a pair with a valid Insert that worsens the score is left out whole
    (ConservativeInsert). ``delta`` is the change in score; lower is better.

    ``verdicts``, when given, keeps the SafeInsert verdict of each target by ``(y, Pa_G(y))``: the
    nodes ``x`` for which ``G`` does not score better without ``x -> y``, as a bitmask
    (:func:`~andrey.core._bitset.bitmask`). A verdict depends on the score alone, so one dict
    serves every state of a search with the same score.
    """
    dag = extension(adj, ranks)
    adj_sets, par_of, nbr = neighborhood_state(adj)
    succ = _bitset.row_masks(adj == ops.TAIL)
    adjacent = _bitset.row_masks(adj != ops.NULL)
    dag_parents = [np.flatnonzero(row).tolist() for row in (dag == ops.ARROW)]
    for y in range(n):
        dag_pa_list = dag_parents[y]
        key = (y, tuple(dag_pa_list))
        safe = None if verdicts is None else verdicts.get(key)
        if safe is None:
            dag_pa = set(dag_pa_list)
            dag_base = score.score(y, dag_pa_list)
            safe = _bitset.bitmask(
                x
                for x in range(n)
                if x != y
                and x not in dag_pa
                and not score.score(y, sorted(dag_pa | {x})) > dag_base
            )
            if verdicts is not None:
                verdicts[key] = safe
        pa = {int(p) for p in par_of[y]}
        nbr_y = nbr[y]
        extensions: dict[frozenset[int], list[tuple[tuple[int, ...], int, list[int]]]] = {}
        reach: dict[int, int] = {}
        base_score: dict[tuple[int, ...], float] = {}
        for x in _bitset.nodes(safe & ~adjacent[y]):
            na = frozenset(nbr_y & adj_sets[x])
            ext = extensions.get(na)
            if ext is None:
                ext = []
                for _index, T in ops.clique_extensions(adj, na, sorted(nbr_y - na)):
                    clique = na.union(T)
                    ext.append((T, _bitset.bitmask(clique), sorted(clique | pa)))
                extensions[na] = ext
            moves: list[tuple[float, int, int, tuple[int, ...]]] = []
            for T, clique_mask, kept in ext:
                seen = reach.get(clique_mask)
                if seen is None:
                    seen = reach[clique_mask] = _bitset.semidirected_reach_set(succ, y, clique_mask)
                if (seen >> x) & 1:
                    continue
                base = tuple(kept)
                s_kept = base_score.get(base)
                if s_kept is None:
                    s_kept = base_score[base] = score.score(y, kept)
                delta = score.score(y, sorted([*kept, x])) - s_kept
                if conservative and delta > 0:
                    moves = []
                    break  # ConservativeInsert: some Insert(x, y, T) worsens the score
                moves.append((delta, x, y, T))
            yield from moves


def insert_best(
    adj: np.ndarray,
    score: LocalScore,
    n: int,
    ranks: Sequence[int],
    conservative: bool,
    verdicts: dict[tuple[int, tuple[int, ...]], int] | None = None,
) -> tuple[float, int, int, tuple[int, ...]] | None:
    """The best of :func:`insert_moves` by :func:`~andrey.search._parallel_ges.move_key`, or None
    when none lowers the score by more than :data:`~andrey.core.ties.EPS_IMPROVE`; ``verdicts``
    is that function's."""
    best: tuple[float, int, int, tuple[int, ...]] | None = None
    best_key: tuple | None = None
    for move in insert_moves(adj, score, n, ranks, conservative, verdicts):
        delta = move[0]
        if not delta < -EPS_IMPROVE:
            continue
        if best_key is not None and stable_key(delta, n) > best_key[0]:
            continue
        key = move_key(delta, n, move[1], move[2], move[3], ranks)
        if best_key is None or key < best_key:
            best, best_key = move, key
    return best


def lges(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 0.0,
    equivalent_sample_size: float = 1.0,
    insertion: str = "safe",
    schedule: str = "lges",
    categorical: np.ndarray | None = None,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by less greedy equivalence search over a decomposable score.

    Starts from the empty graph. ``schedule="lges"`` repeats Delete moves until none improves,
    turning moves until none improves, then the best Insert of :func:`insert_moves`; it stops when
    no move improves, meaning no improving Insert follows a pass without a turning move.
    ``schedule="lges0_released"`` runs Insert, Delete, then turning, each until no move improves.
    Improvement means lowering the score by more than :data:`~andrey.core.ties.EPS_IMPROVE`; each
    move re-forms the CPDAG. Returns the CPDAG and its recomputed total score, lower being better
    (:func:`~andrey.search.ges.recompute_objective`).

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score, as for :func:`~andrey.search.ges.ges`.
    penalty_discount, ebic_gamma, equivalent_sample_size : float
        The score's settings, as for :func:`~andrey.search.ges.ges`.
    insertion : {"safe", "conservative"}
        The rule that leaves pairs out of the Insert phase (:func:`insert_moves`).
    schedule : {"lges", "lges0_released"}
        The order of the phases, as above.
    categorical : ndarray of bool or None
        Marks the categorical columns for a mixed-data score; the other scores ignore it.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and its total score.
    """
    if insertion not in INSERTIONS:
        raise ValueError(f"unsupported insertion {insertion!r}; use one of {INSERTIONS}")
    if schedule not in SCHEDULES:
        raise ValueError(f"unsupported schedule {schedule!r}; use one of {SCHEDULES}")
    X = np.asarray(data, dtype=np.float64)
    n = X.shape[1]
    score = make_score(
        score_func,
        X,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        categorical=categorical,
    )
    ranks = tie_ranks(X, tie_mask(score_func, categorical, n)).tolist()
    conservative = insertion == "conservative"
    adj = np.zeros((n, n), dtype=np.int8)
    cut = np.zeros((n, n), dtype=bool)
    with backend.one_blas_thread():
        engine = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
        # The turning scan reads only the score's local scores, not its batched additions.
        family = cast(FamilyScore, score)
        kept: dict[tuple[int, tuple[int, ...], int], float] = {}
        verdicts: dict[tuple[int, tuple[int, ...]], int] = {}

        def deletes(adj: np.ndarray) -> np.ndarray:
            while (
                found := backward_chunk_best(
                    range(n), adj, engine, n, neighborhood_state(adj), ranks, kept
                )
            ) is not None:
                _, x, y, H = found
                adj = ops.delete_cpdag(adj, x, y, H)
            return adj

        def turns(adj: np.ndarray) -> tuple[np.ndarray, bool]:
            turned = False
            while (turn := _turning_best(adj, family, n, ranks)) is not None:
                u, v, C, essential = turn
                adj, turned = _turn(adj, u, v, C, essential, cut), True
            return adj, turned

        if schedule == "lges":
            while True:
                adj, turned = turns(deletes(adj))
                if (move := insert_best(adj, score, n, ranks, conservative, verdicts)) is not None:
                    _, x, y, T = move
                    adj = ops.insert_cpdag(adj, x, y, T)
                elif not turned:
                    break  # no Delete, turning move, or Insert improves
        else:
            while (move := insert_best(adj, score, n, ranks, conservative, verdicts)) is not None:
                _, x, y, T = move
                adj = ops.insert_cpdag(adj, x, y, T)
            adj, _ = turns(deletes(adj))
        total = recompute_objective(adj, score, n)
    return to_structure(adj, kind="cpdag"), total
