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

from andrey.core import GraphStructure, backend
from andrey.core.orient import pdag2dag, to_structure
from andrey.core.score import BICScore, LocalScore, make_score
from andrey.core.score_delta import DeltaBICScore, FullScoreDelta
from andrey.core.ties import EPS_IMPROVE, tie_mask, tie_ranks
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

    :func:`~andrey.core.orient.pdag2dag` removes the lowest-numbered sink whose neighbors are
    adjacent first; here the nodes are numbered by ``ranks``, so the DAG follows the columns' tie
    ranks and not their positions.
    """
    order = np.argsort(np.asarray(ranks), kind="stable")
    dag = pdag2dag(adj[np.ix_(order, order)])
    back = np.argsort(order, kind="stable")
    return dag[np.ix_(back, back)]


def insert_moves(
    adj: np.ndarray, score: LocalScore, n: int, ranks: Sequence[int], conservative: bool
) -> Iterator[tuple[float, int, int, tuple[int, ...]]]:
    """Yield every valid ``Insert(x, y, T)`` the less greedy rule keeps, as ``(delta, x, y, T)``.

    A pair ``(x, y)`` of non-adjacent nodes is left out when ``score(y, Pa_G(y) + x)`` is worse than
    ``score(y, Pa_G(y))`` for the DAG ``G`` of :func:`extension` (SafeInsert). Otherwise every
    valid ``Insert(x, y, T)`` is scored, ``T`` a subset of the neighbors of ``y`` not adjacent to
    ``x`` with ``NA_yx + T`` a clique that blocks every semi-directed path from ``y`` to ``x``;
    with ``conservative``, a pair with a valid Insert that worsens the score is left out whole
    (ConservativeInsert). ``delta`` is the change in score; lower is better.
    """
    dag = extension(adj, ranks)
    adj_sets, par_of, nbr = neighborhood_state(adj)
    succ = ops.tail_successor_lists(adj)
    for y in range(n):
        dag_pa = set(ops.parents(dag, y).tolist())
        dag_base = score.score(y, sorted(dag_pa))
        pa = {int(p) for p in par_of[y]}
        for x in range(n):
            if x == y or x in adj_sets[y]:
                continue
            if score.score(y, sorted(dag_pa | {x})) > dag_base:
                continue  # SafeInsert: G scores better without x -> y
            na = nbr[y] & adj_sets[x]
            t0 = sorted(nbr[y] - adj_sets[x])
            moves: list[tuple[float, int, int, tuple[int, ...]]] = []
            for _index, T in ops.clique_extensions(adj, na, t0):
                clique = na | set(T)
                if not ops.blocks_semi_directed_paths(succ, y, x, clique):
                    continue
                kept = clique | pa
                delta = score.score(y, sorted(kept | {x})) - score.score(y, sorted(kept))
                if conservative and delta > 0:
                    moves = []
                    break  # ConservativeInsert: some Insert(x, y, T) worsens the score
                moves.append((delta, x, y, T))
            yield from moves


def insert_best(
    adj: np.ndarray, score: LocalScore, n: int, ranks: Sequence[int], conservative: bool
) -> tuple[float, int, int, tuple[int, ...]] | None:
    """The best of :func:`insert_moves` by :func:`~andrey.search._parallel_ges.move_key`, or None
    when none lowers the score by more than :data:`~andrey.core.ties.EPS_IMPROVE`."""
    improving = (m for m in insert_moves(adj, score, n, ranks, conservative) if m[0] < -EPS_IMPROVE)
    return min(improving, key=lambda m: move_key(m[0], n, m[1], m[2], m[3], ranks), default=None)


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

    Starts from the empty graph. Under ``schedule="lges"`` it repeats: Delete moves until none
    improves, turning moves until none improves, then the best Insert of :func:`insert_moves`,
    stopping when no move of any kind improves: when no Insert improves after a pass without a
    turning move. Under ``schedule="lges0_released"`` it runs the Insert phase,
    then the Delete phase, then the turning phase, each until no move improves. A move improves when
    it lowers the score by more than :data:`~andrey.core.ties.EPS_IMPROVE`; the CPDAG is re-formed
    after every move. Returns the CPDAG and its total score (lower is better), recomputed from it
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
                if (move := insert_best(adj, score, n, ranks, conservative)) is not None:
                    _, x, y, T = move
                    adj = ops.insert_cpdag(adj, x, y, T)
                elif not turned:
                    break  # no Delete, turning move, or Insert improves
        else:
            while (move := insert_best(adj, score, n, ranks, conservative)) is not None:
                _, x, y, T = move
                adj = ops.insert_cpdag(adj, x, y, T)
            adj, _ = turns(deletes(adj))
        total = recompute_objective(adj, score, n)
    return to_structure(adj, kind="cpdag"), total
