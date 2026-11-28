"""Structural expectation-maximization (EM) for searches whose fit reads only a covariance.

On data with missing values (``NaN``), :func:`structural_em` runs the alternating structural EM of
Friedman (1997) over a linear-Gaussian model of the columns: a mean ``mu``, coefficients ``B``
(``B[i, j]`` the weight of edge ``i -> j``), and noise variances ``omega``. It is made of three
steps:

- **E-step** (:func:`expected_moments`): under the current model, each row's missing values have a
  Gaussian distribution given its observed ones. Their conditional means fill the row, and their
  conditional covariances add to the second moments, which give the expected mean and the expected
  covariance ``S`` about it.
- **Parameter step**: the graph stays, and ``B`` and ``omega`` are the least-squares fit on ``S``
  of a DAG of it (:func:`fit_dag`), the parameters that maximize the expected log-likelihood of
  that graph. Every DAG of a CPDAG's class gives the same fitted covariance.
- **Structure step**: the method's own search runs on ``(S, n)``, with ``n`` every row, and a
  parameter step fits the graph it returns on the same ``S``.

The model starts from ``B = 0`` and each column's mean and variance over its observed values (the
available cases; Little and Rubin 2019), which are the maximum-likelihood parameters of the graph
with no edge, so the first round is an E-step and a structure step. Each later round runs an
E-step, then :data:`PARAMETER_STEPS` times a parameter step and an E-step, then a structure step.
:data:`ITERATIONS` rounds run, as in Gao et al. (2022), unless the facades' ``n_em_iterations``
overrides the count.

Friedman (1997) repeats the parameter steps "until convergence or" a largest count and the rounds
"until convergence", and leaves the count open. The first search reads the expected covariance
under the model with no edge, where every covariance is shrunk toward zero by the share of rows
that observe both columns. Measured with ``ges``, ``boss``, and ``grasp`` at 20 and 50 variables
and 2,000 rows with 30% of the values missing, that search returned many extra edges; with one
parameter fit per round the later rounds removed them slowly and the graph was still changing
after 10 rounds, and with :data:`PARAMETER_STEPS` parameter steps before each search the last
round changed the graph in about 1 fit in 10. At 100 rows it did in about 4 in 10. The record
tells whether each round changed the graph, and a fit whose last round did logs a warning.

The rows are grouped by which columns they observe once per fit (:class:`Patterns`). Each E-step
solves a group of several rows with one solve, and the groups of one row that miss the same number
of columns together, in stacks, with the arithmetic each group has alone. A group of one row that
observes one column or misses one stays alone: a stack of it multiplies in dot products, which
some BLAS sum by each slice's memory alignment. The E-step runs on the columns in order of their
:func:`~andrey.core.ties.tie_ranks`, with BLAS on one thread, so a new column order or more threads
move no bit of ``S``.

The missingness must be ignorable: missing completely at random or at random given observed
values. Under values missing not at random the expected covariance is biased, and nothing in the
data can show it.

References
----------
- Friedman, N. (1997). Learning belief networks in the presence of missing values and hidden
  variables. Proceedings of the 14th International Conference on Machine Learning, 125-133.
- Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H. (2022).
  MissDAG: Causal discovery in the presence of missing data with continuous additive noise models.
  Advances in Neural Information Processing Systems 35.
- Little, R. J. A., and Rubin, D. B. (2019). Statistical Analysis with Missing Data, 3rd edition.
  Wiley.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from andrey.core import _checkpoint, backend
from andrey.core.orient import pdag2dag
from andrey.core.score import copy_groups
from andrey.core.structure import ARROW, TAIL
from andrey.core.ties import tie_ranks

if TYPE_CHECKING:
    from collections.abc import Callable

    from andrey.core.structure import GraphStructure

__all__ = [
    "ITERATIONS",
    "PARAMETER_STEPS",
    "PLAN_CELLS",
    "STACK_LIMIT",
    "Patterns",
    "expected_moments",
    "fit_dag",
    "observed_copies",
    "shared_rows",
    "structural_em",
]

_log = logging.getLogger(__name__)

#: The rounds, each with one structure step, as Gao et al. (2022) run them; the default of the
#: facades' ``n_em_iterations``.
ITERATIONS = 10
#: The parameter steps on the current graph before each structure step after the first. Friedman
#: (1997) leaves the count open ("until convergence or" a largest count, or "some specified number
#: of steps"). With 5, the graph had settled within the 10 rounds in about 9 of 10 fits measured at
#: 2,000 rows; with none it was still changing in most. Stopping the steps earlier, once a step
#: gains less than 0.01 or 0.0001 in log-likelihood, returned another graph in 8 and 2 of 360 fits.
PARAMETER_STEPS = 5
#: The most patterns one stack solves in a call.
STACK_LIMIT = 256
#: The most entries each array of a stack holds, so a stack over wide columns has fewer patterns,
#: and the most entries of covariance shares a plan indexes. When the shares of all patterns would
#: hold more, each pattern is solved alone and adds its own share.
PLAN_CELLS = 2**22

# A node's noise variance is at least this share of its expected variance, so the precision stays
# finite when a fitted node is a near copy of its parents.
_VARIANCE_FLOOR = 1e-12

R = TypeVar("R")


class Patterns:
    """The rows of ``X`` grouped by which columns they observe, built once for a fit's E-steps.

    Patterns come in the order of :func:`numpy.unique` over the observed masks. The patterns that
    hold one row, observe at least two columns, and miss the same number of at least two columns are
    solved as stacks of at most :data:`STACK_LIMIT` patterns, fewer when :data:`PLAN_CELLS` bounds
    their arrays; every other pattern is solved alone, and so is every pattern when the shares of
    ``K_mm^-1`` of all of them hold more than :data:`PLAN_CELLS` entries. Either way each pattern's
    arithmetic is the one it has alone. A stack of patterns that observe one column or miss one
    would multiply a row by a column, a dot product, on each slice in place, and some BLAS (Apple's
    Accelerate) sum a dot product by the memory alignment of its second operand, which a slice of
    odd size does not share with an array of its own.
    """

    def __init__(self, X: np.ndarray) -> None:
        n, d = X.shape
        observed = ~np.isnan(X)
        patterns, group = np.unique(observed, axis=0, return_inverse=True)
        group = group.ravel()
        order = np.argsort(group, kind="stable")
        bounds = np.searchsorted(group[order], np.arange(len(patterns) + 1))
        self.shape = (n, d)
        self.n_patterns = len(patterns)
        self.filled = np.where(observed, X, 0.0)
        # The shares of K_mm^-1 add into the expected covariance in pattern order. Over the budget
        # each pattern adds its own in the loop; otherwise one np.add.at adds them all.
        shares = int(((d - patterns.sum(axis=1, dtype=np.int64)) ** 2).sum())
        self.direct_adds = shares > PLAN_CELLS
        stack_limit = max(1, min(STACK_LIMIT, PLAN_CELLS // max(d * d, 1)))
        self.alone = []  # (k, rows, o, m, X block, (rows, m), (m, m), (o, o))
        by_size: dict[int, list[tuple]] = {}
        for k, pattern in enumerate(patterns):
            rows = order[bounds[k] : bounds[k + 1]]
            o, m = np.flatnonzero(pattern), np.flatnonzero(~pattern)
            if rows.size == 1 and m.size > 1 and o.size > 1 and not self.direct_adds:
                by_size.setdefault(int(m.size), []).append((k, int(rows[0]), o, m))
            else:
                self.alone.append(
                    (k, rows, o, m, X[np.ix_(rows, o)], np.ix_(rows, m), np.ix_(m, m), np.ix_(o, o))
                )
        self.stacks = []
        for all_items in by_size.values():
            for start in range(0, len(all_items), stack_limit):
                items = all_items[start : start + stack_limit]
                ks = np.array([it[0] for it in items])
                rows = np.array([it[1] for it in items])
                os_ = np.stack([it[2] for it in items])
                ms = np.stack([it[3] for it in items])
                block = X[rows[:, None], os_][:, None, :]  # (K, 1, |o|)
                self.stacks.append((ks, rows, os_, ms, block))
        self.add_order = None
        if self.direct_adds or not shares:
            return
        # Where each pattern's K_mm^-1 adds into the expected covariance, as a flat cell index: the
        # stacks' patterns, then the others', reordered to pattern order so that every cell sums
        # its shares in that order.
        cell_type = np.int32 if d * d < 2**31 else np.int64
        keys, cells = [], []
        for ks, _, _, ms, _ in self.stacks:
            keys.append(np.repeat(ks, ms.shape[1] ** 2))
            cells.append((ms[:, :, None] * d + ms[:, None, :]).astype(cell_type).ravel())
        for k, _, _, m, *_ in self.alone:
            if m.size:
                keys.append(np.full(m.size * m.size, k))
                cells.append((np.repeat(m, m.size) * d + np.tile(m, m.size)).astype(cell_type))
        self.add_order = np.argsort(np.concatenate(keys), kind="stable")
        self.add_cells = np.concatenate(cells)[self.add_order]


def expected_moments(
    X: np.ndarray | Patterns, mean: np.ndarray, precision: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """The expected mean, the expected covariance, and the observed-data log-likelihood.

    ``X`` holds ``NaN`` where a value is missing, or is its :class:`Patterns`; ``mean`` and
    ``precision`` (the inverse covariance) are the model's. A row that observes the columns ``o``
    and misses ``m`` has ``x_m`` given ``x_o`` Gaussian with mean
    ``mean_m - K_mm^-1 K_mo (x_o - mean_o)`` and covariance ``K_mm^-1``, ``K`` the precision. The
    covariance is about the expected mean and divided by the number of rows. The log-likelihood
    is the sum over rows of the log density of each row's observed values under the model; a row
    that observes nothing adds zero.
    """
    plan = X if isinstance(X, Patterns) else Patterns(X)
    n, d = plan.shape
    filled = plan.filled.copy()
    log_det = np.linalg.slogdet(precision)[1]
    log_2pi = np.log(2 * np.pi)
    terms = np.zeros(plan.n_patterns)  # each pattern's share of -2 * loglik, in pattern order
    has_term = np.zeros(plan.n_patterns, dtype=bool)
    extra = np.zeros((d, d))
    added: list[np.ndarray] = []  # each K_mm^-1 share, stacks first, as Patterns lays them out
    for ks, rows, os_, ms, block in plan.stacks:
        resid = block - mean[os_][:, None, :]
        k_mm = precision[ms[:, :, None], ms[:, None, :]]
        k_mo = precision[ms[:, :, None], os_[:, None, :]]
        factor = np.linalg.cholesky(k_mm)
        solved = np.linalg.solve(factor.transpose(0, 2, 1), np.linalg.solve(factor, k_mo))
        filled[rows[:, None], ms] = (mean[ms][:, None, :] - resid @ solved.transpose(0, 2, 1))[:, 0]
        inverse = np.linalg.inv(factor)
        cov_m = 1 * (inverse.transpose(0, 2, 1) @ inverse)
        added.append(cov_m.ravel())
        inner = precision[os_[:, :, None], os_[:, None, :]] - k_mo.transpose(0, 2, 1) @ solved
        log_det_o = log_det - 2.0 * np.log(np.diagonal(factor, axis1=1, axis2=2)).sum(axis=1)
        if os_.shape[1] == 2:  # the stacked einsum sums two observed columns in another order
            quadratic = np.array([np.einsum("ij,jk,ik->", r, q, r) for r, q in zip(resid, inner)])
        else:
            quadratic = np.einsum("pij,pjk,pik->p", resid, inner, resid)
        terms[ks] = quadratic + 1 * (os_.shape[1] * log_2pi - log_det_o)
        has_term[ks] = True
    for k, rows, o, m, block, rows_m, mm, oo in plan.alone:
        resid = block - mean[o]
        if m.size == 0:
            inner = precision
            log_det_o = log_det
        else:
            k_mm = precision[mm]
            k_mo = precision[np.ix_(m, o)]
            factor = np.linalg.cholesky(k_mm)
            solved = np.linalg.solve(factor.T, np.linalg.solve(factor, k_mo))  # K_mm^-1 K_mo
            filled[rows_m] = mean[m] - resid @ solved.T
            inverse = np.linalg.inv(factor)
            cov_m = rows.size * (inverse.T @ inverse)  # rows * K_mm^-1
            if plan.direct_adds:
                extra[mm] += cov_m
            else:
                added.append(cov_m.ravel())
            inner = precision[oo] - k_mo.T @ solved
            log_det_o = log_det - 2.0 * float(np.log(np.diagonal(factor)).sum())
        if o.size:
            quadratic = float(np.einsum("ij,jk,ik->", resid, inner, resid))
            terms[k] = quadratic + rows.size * (o.size * log_2pi - log_det_o)
            has_term[k] = True
    if plan.add_order is not None:  # each pattern's share in pattern order
        np.add.at(extra.reshape(-1), plan.add_cells, np.concatenate(added)[plan.add_order])
    loglik = 0.0
    for term in terms[has_term]:
        loglik -= 0.5 * term
    expected_mean = filled.mean(axis=0)
    centered = filled - expected_mean
    return expected_mean, (centered.T @ centered + extra) / n, loglik


def shared_rows(X: np.ndarray) -> np.ndarray:
    """The ``(d, d)`` counts of the rows of ``X`` that observe both columns of each pair.

    Columns sharing no row have no observed covariance: EM would derive it entirely from the model,
    so :func:`~andrey.api._adapt.fit_missing` refuses such data.
    """
    both = (~np.isnan(X)).astype(np.float64)
    return both.T @ both


def observed_copies(X: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """The groups of columns of ``X`` that repeat one another up to scale and shift where observed.

    Two columns are compared on the rows that observe both, and are copies when their correlation
    there passes :func:`~andrey.core.score.copy_groups`; a pair observed together on at most two
    rows, or constant on them, is no copy. The expected covariance blurs a copy whose values go
    missing on other rows than the original's, so the linear-Gaussian BIC checks the observed
    values for copies before EM, as it checks complete data.
    """
    observed = ~np.isnan(X)
    centered = np.where(observed, X - np.nanmean(X, axis=0), 0.0)
    both = observed.astype(np.float64)
    count = shared_rows(X)  # count[i, j]: the rows that observe columns i and j
    sums = centered.T @ both  # sums[i, j]: column i summed over those rows
    squares = (centered**2).T @ both
    rows = np.maximum(count, 1.0)
    cross = centered.T @ centered / rows - (sums / rows) * (sums.T / rows)
    spread = (squares / rows - (sums / rows) ** 2) * (squares.T / rows - (sums.T / rows) ** 2)
    usable = (count > 2) & (spread > 0.0)
    corr = np.where(usable, cross / np.sqrt(np.where(usable, spread, 1.0)), 0.0)
    return copy_groups(corr, X.shape[0])


def fit_dag(adjacency: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The coefficients ``B`` and noise variances ``omega`` of the DAG ``adjacency`` fit on ``cov``.

    ``adjacency[i, j]`` marks edge ``i -> j``. Each node is regressed on its parents by least
    squares on ``cov``, and its noise variance is the residual variance, at least
    ``1e-12`` of its own variance.
    """
    d = cov.shape[0]
    B = np.zeros((d, d))
    omega = np.diagonal(cov).copy()
    for j in range(d):
        parents = np.flatnonzero(adjacency[:, j])
        if parents.size:
            weights = np.linalg.lstsq(cov[np.ix_(parents, parents)], cov[parents, j], rcond=None)[0]
            B[parents, j] = weights
            omega[j] = cov[j, j] - cov[j, parents] @ weights
    return B, np.maximum(omega, _VARIANCE_FLOOR * np.diagonal(cov))


def _precision(B: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """The inverse covariance ``(I - B) diag(1 / omega) (I - B)^T`` of the model ``(B, omega)``."""
    lower = np.eye(B.shape[0]) - B
    return (lower / omega) @ lower.T


def _dag_adjacency(structure: GraphStructure) -> np.ndarray:
    """A DAG of ``structure``'s class as a boolean ``i -> j`` matrix (:func:`pdag2dag`)."""
    marks = pdag2dag(structure.to_numpy())
    return (marks == TAIL) & (marks.T == ARROW)


def structural_em(
    X: np.ndarray,
    m_step: Callable[[np.ndarray], tuple[R, GraphStructure]],
    *,
    iterations: int = ITERATIONS,
) -> tuple[R, dict[str, object]]:
    """Run alternating structural EM on ``X``, which holds ``NaN`` where a value is missing.

    ``m_step(cov)`` runs the method's search on the expected covariance ``cov`` (in the columns of
    ``X``, divided by the number of rows) and returns its result and the graph it learned, a DAG,
    CPDAG, or a CPDAG narrowed by knowledge. ``iterations`` is the number of rounds, each with one
    such structure step; every round after the first runs :data:`PARAMETER_STEPS` parameter steps
    on its graph before it. Returns the last structure step's result and the record:

    - ``"iterations"``: the rounds;
    - ``"parameter_steps"``: the parameter steps before each structure step after the first;
    - ``"log_likelihood"``: the observed-data log-likelihood of the model after each round's
      structure step. A parameter step never lowers it; a structure step can, as the search
      weighs the expected log-likelihood against the graph's size;
    - ``"graph_changed"``: for each round, whether its structure step returned another graph than
      the round before, the first round against the graph with no edge.

    A last round that changed the graph logs a warning: the rounds ended before EM settled on a
    graph, and the warning names ``n_em_iterations``, the facades' argument for ``iterations``. A
    row that observes no value counts in the number of rows and adds the model's
    covariance to the expected covariance, so the facades drop such rows first
    (:func:`~andrey.api._adapt.data_matrix`).
    """
    if iterations < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations}")
    order = np.argsort(tie_ranks(X))  # the E-step's column k is the data's column order[k]
    back = np.argsort(order)
    ranked = np.ascontiguousarray(X[:, order])
    plan = Patterns(ranked)
    mean = np.nanmean(ranked, axis=0)
    precision = np.diag(1.0 / np.nanvar(ranked, axis=0))
    adjacency = np.zeros((X.shape[1], X.shape[1]), dtype=bool)
    marks = np.zeros_like(adjacency, dtype=np.int8)
    loglik: list[float] = []
    changed: list[bool] = []
    for k in range(iterations):
        with backend.one_blas_thread():
            mean, cov, value = expected_moments(plan, mean, precision)
            if k:
                loglik.append(value)  # the model after the previous structure step
                for _ in range(PARAMETER_STEPS):
                    precision = _precision(*fit_dag(adjacency, cov))
                    mean, cov, _ = expected_moments(plan, mean, precision)
        result, graph = m_step(cov[np.ix_(back, back)])
        found = np.asarray(graph.to_numpy())
        changed.append(not np.array_equal(found, marks))
        marks = found
        adjacency = _dag_adjacency(graph)[np.ix_(order, order)]
        precision = _precision(*fit_dag(adjacency, cov))
        if (take := _checkpoint.TAKE.get()) is not None:
            take("em.round", k, mean=mean, cov=cov, marks=marks, precision=precision)
    with backend.one_blas_thread():
        loglik.append(expected_moments(plan, mean, precision)[2])
    if (take := _checkpoint.TAKE.get()) is not None:
        take("em.loglik", log_likelihood=np.asarray(loglik))
    if changed[-1]:
        _log.warning(
            "structural EM: the graph still changed in the last of %d rounds, so EM has not "
            "settled on a graph; the result is the last round's, and n_em_iterations above %d "
            "runs more rounds",
            iterations,
            iterations,
        )
    return result, {
        "iterations": iterations,
        "parameter_steps": PARAMETER_STEPS,
        "log_likelihood": loglik,
        "graph_changed": changed,
    }
