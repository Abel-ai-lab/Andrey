"""RESIT: regression with subsequent independence test (Peters, Mooij, Janzing, Scholkopf 2014).

RESIT (Algorithm 1 of the paper) assumes an additive noise model, ``x_j = f_j(parents) + e_j``
with each noise independent of the variable's non-descendants, and runs two phases:

1. **Causal order.** Each remaining variable is regressed on the other remaining ones, and its
   residual is tested for independence of them jointly. The variable whose residual has the largest
   p-value is the next sink: it leaves the set, and the variables still in it are its possible
   parents. The causal order is the sinks in reverse.
2. **Edge removal.** Each variable's possible parents are tried one at a time: a parent goes when
   the residual of the fit on the other current parents is independent of all the variable's
   predecessors in the order, at a p-value above ``alpha``. With one parent left, the residual
   without it is the variable itself.

The settings are those of the paper's experiments: the regression is an additive model of thin
plate regression splines with ten basis functions per term and GCV smoothing,
:func:`andrey.fcm.gam.fit_gam`, which lowers the rank with fewer than 30 rows per term; the test is
the gamma HSIC test of :func:`andrey.core.independence.hsic_gamma_log_p`, with Gaussian kernels on
each side's joint squared distances, the bandwidth set by the median heuristic, and the null
moments of Gretton et al. (2007) in their product form; and ``alpha`` is ``0.05``.

The choices Algorithm 1 leaves open:

- The kernels read the columns centered and divided by their standard deviations, so neither a
  column's units nor its offset changes a test.
- The bandwidth is the median over every pair of rows with a positive distance.
- p-values are compared as their logarithms, computed directly so that a p-value below the
  smallest double keeps its place, rounded to ``KEY_DP`` decimal places, as :func:`andrey.hc`
  compares its score changes; a tie goes to the earlier column.
- The parents are tried in column order, and :func:`andrey.resit` passes the columns in their rank
  order, so they are tried by rank. Each removal changes the fits after it, so another order can
  keep other parents.

The fit draws no random numbers.

References
----------
Peters, Mooij, Janzing, Scholkopf. "Causal discovery with continuous additive noise models."
JMLR 15:2009-2053, 2014.

Gretton, Fukumizu, Teo, Song, Scholkopf, Smola. "A kernel statistical test of independence."
NIPS 20, 2007.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from andrey.core.independence import hsic_gram, hsic_log_p
from andrey.core.ties import tie_key
from andrey.core.units import unit_factors
from andrey.fcm.gam import BasisCache, fit_gam

if TYPE_CHECKING:
    from andrey.fcm._parallel import ResitPool

_log = logging.getLogger(__name__)

#: Significance level of the edge-removal tests.
ALPHA = 0.05
#: Basis functions per term of each regression.
K = 10


def _residual(bases: BasisCache, child: int, parents: list[int]) -> np.ndarray:
    """The residual of ``child``'s additive fit on ``parents``; with none, ``child`` centered."""
    if not parents:
        y = bases.data[:, child]
        return y - y.mean()
    return fit_gam(bases, child, parents, k=K).residuals


def _sink_log_p(
    bases: BasisCache, unit: np.ndarray, allowed: np.ndarray, remaining: list[int], k: int
) -> float:
    """The log p-value of the sink test of ``k`` among the ``remaining`` variables: its residual on
    the remaining variables that may be its parents against all the other remaining ones.

    The operations of the loop in :func:`causal_order`, for a worker. The loop has its own copy: its
    arrays live until the next step's are made, which takes less system time in one process than
    allocating and freeing them on each call."""
    others = [i for i in remaining if i != k]
    parents = [i for i in others if allowed[i, k]]
    side = hsic_gram(unit[:, others])
    return hsic_log_p(side, hsic_gram(_residual(bases, k, parents)), "resit")


def causal_order(
    bases: BasisCache,
    unit: np.ndarray,
    allowed: np.ndarray,
    precedence: np.ndarray | None = None,
    *,
    pool: ResitPool | None = None,
) -> np.ndarray:
    """Find a causal order from the sink up, the first phase.

    Each step regresses every candidate on the remaining variables it may have as parents and tests
    its residual against all the other remaining ones; the candidate of largest p-value is the next
    sink. The log p-values are compared by :func:`~andrey.core.ties.tie_key`, and a tie goes to the
    earlier column. A ``pool`` made from the same ``bases``, ``unit``, and ``allowed`` fits the
    candidates of a step on its workers; the sink is chosen here from the same log p-values.

    Parameters
    ----------
    bases : BasisCache
        Bases of the data's columns.
    unit : np.ndarray
        ``(n, d)`` columns the kernels read.
    allowed : np.ndarray
        ``(d, d)`` bool; ``allowed[i, j]`` lets ``i`` be a parent of ``j``.
    precedence : np.ndarray or None
        ``(d, d)`` bool; ``precedence[a, b]`` puts ``a`` before ``b``, so ``a`` is a candidate
        only once ``b`` has left.
    pool : ResitPool or None
        Worker processes that hold the same ``bases``, ``unit``, and ``allowed``, or ``None`` to
        fit in this process.

    Returns
    -------
    np.ndarray
        The order, causes first.
    """
    remaining = list(range(unit.shape[1]))
    sinks: list[int] = []
    while len(remaining) > 1:
        candidates = [
            k for k in remaining if precedence is None or not precedence[k, remaining].any()
        ]
        pooled = None if pool is None else pool.sink_log_ps(remaining, candidates)
        keys = {}
        for index, k in enumerate(candidates):
            if pooled is None:
                others = [i for i in remaining if i != k]
                parents = [i for i in others if allowed[i, k]]
                side = hsic_gram(unit[:, others])
                log_p = hsic_log_p(side, hsic_gram(_residual(bases, k, parents)), "resit")
            else:
                log_p = pooled[index]
            _log.debug("%d remaining: sink %d, log p-value %.9g", len(remaining), k, log_p)
            keys[k] = tie_key(log_p)
        sink = min(candidates, key=lambda k: (-keys[k], k))
        remaining.remove(sink)
        sinks.append(sink)
    return np.array([*remaining, *reversed(sinks)], dtype=np.int64)


def _kept_parents(
    bases: BasisCache,
    unit: np.ndarray,
    order: np.ndarray,
    position: int,
    alpha: float,
    allowed: np.ndarray,
    required: np.ndarray,
) -> tuple[list[int], list[tuple[int, float]]]:
    """The parents the variable at ``position`` of ``order`` keeps, and the log p-value of each
    removal test, as ``(parent, log_p)`` in the order the tests ran.

    The operations of the loop in :func:`remove_edges`, for a worker, which has its own copy as
    :func:`_sink_log_p` explains."""
    child = int(order[position])
    gram = hsic_gram(unit[:, order[:position]])
    parents = sorted(int(i) for i in order[:position] if allowed[i, child])
    tests = []
    for parent in list(parents):
        if required[parent, child]:
            continue
        rest = [i for i in parents if i != parent]
        log_p = hsic_log_p(gram, hsic_gram(_residual(bases, child, rest)), "resit")
        tests.append((parent, log_p))
        if log_p > math.log(alpha):
            parents.remove(parent)
    return parents, tests


def remove_edges(
    bases: BasisCache,
    unit: np.ndarray,
    order: np.ndarray,
    *,
    alpha: float = ALPHA,
    allowed: np.ndarray,
    required: np.ndarray,
    pool: ResitPool | None = None,
) -> np.ndarray:
    """Remove the superfluous edges of the complete DAG of ``order``, the second phase.

    Each variable's possible parents are its allowed predecessors, tried in column order; a parent
    that is not ``required`` goes when the residual of the fit on the other current parents has a
    p-value above ``alpha`` against all the variable's predecessors. The variables are independent
    of each other, so a ``pool`` made from the same ``bases``, ``unit``, ``allowed``, and
    ``required`` removes the edges into each on its workers.

    Returns
    -------
    np.ndarray
        ``(d, d)`` bool DAG; ``out[i, j]`` is the edge ``i -> j``.
    """
    d = unit.shape[1]
    dag = np.zeros((d, d), dtype=bool)
    pooled = None if pool is None else pool.kept_parents(order, alpha)
    for position in range(1, d):
        child = int(order[position])
        if pooled is None:
            gram = hsic_gram(unit[:, order[:position]])
            parents = sorted(int(i) for i in order[:position] if allowed[i, child])
            for parent in list(parents):
                if required[parent, child]:
                    continue
                rest = [i for i in parents if i != parent]
                log_p = hsic_log_p(gram, hsic_gram(_residual(bases, child, rest)), "resit")
                _log.debug("edge %d -> %d: log p-value without it %.9g", parent, child, log_p)
                if log_p > math.log(alpha):
                    parents.remove(parent)
        else:
            parents, tests = pooled[position - 1]
            for parent, log_p in tests:
                _log.debug("edge %d -> %d: log p-value without it %.9g", parent, child, log_p)
        dag[parents, child] = True
    return dag


@dataclass(frozen=True)
class RESITResult:
    """A RESIT fit: the DAG after the edge removal and the causal order, causes first."""

    dag: np.ndarray
    order: np.ndarray


def resit(
    X: np.ndarray,
    *,
    alpha: float = ALPHA,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> RESITResult:
    """Fit RESIT to the ``(n, d)`` matrix ``X``.

    A ``forbidden`` edge is in no regression and never in the result, a ``required`` one is never
    removed, and ``precedence`` goes to :func:`causal_order`. Ties go to the earlier column
    throughout, so a caller that orders the columns by rank breaks them by rank.
    """
    d = X.shape[1]
    allowed = ~np.eye(d, dtype=bool)
    if forbidden is not None:
        allowed &= ~forbidden
    required = np.zeros((d, d), dtype=bool) if required is None else required
    from andrey.fcm._parallel import resit_pool

    bases = BasisCache(X)
    unit = (X - X.mean(axis=0)) * unit_factors(X, standardize=True)
    with resit_pool(X, unit, allowed, required) as pool:
        order = causal_order(bases, unit, allowed, precedence, pool=pool)
        dag = remove_edges(
            bases, unit, order, alpha=alpha, allowed=allowed, required=required, pool=pool
        )
    return RESITResult(dag=dag, order=order)
