"""Regime-PCMCI: the regimes of a time series and the lagged causal links within each regime.

The series ``x_t``, ``t = 0 .. T - 1``, is assumed to switch between ``K`` persistent regimes, each
with its own lagged causal graph. A regime-assigning process ``Gamma``, a ``(K, T)`` matrix,
weights the regimes at each step. Each start draws its first ``Gamma`` entry by entry, uniformly in
``[0, 1)`` and not normalized, and alternates two steps up to ``max_iter`` times:

1. **Causal discovery and model estimation.** For each regime ``k``, PCMCI runs on the steps ``t``
   with ``gamma_k(t) >= switch_threshold``, each weight rounded to :data:`~andrey.core.ties.KEY_DP`
   decimals first: every test reads those steps as its effect's time and the lagged values of
   every earlier step (:func:`~andrey.temporal.pcmci.pcmci` with ``rows``). Each variable is then
   regressed by least squares, with an intercept, on its MCI parents over the same steps from
   ``tau_max`` on, and the fit predicts every step from ``tau_max`` on.
2. **Regime learning.** ``c_k(t)`` is the squared residual of that prediction summed over the
   variables, each residual divided by the regime's standard deviation of its variable. The new
   ``Gamma``, whose columns lie on the probability simplex, solves the linear program

   ``min sum_k sum_t gamma_k(t) c_k(t)`` subject to ``sum_k gamma_k(t) = 1`` for every ``t``,
   ``0 <= gamma_k(t) <= 1``, and ``sum_t |gamma_k(t + 1) - gamma_k(t)| <= max_transitions`` for
   every ``k``,

   written with one slack ``eta_k(t) >= |gamma_k(t + 1) - gamma_k(t)|`` per step and solved by
   the dual simplex method (:func:`scipy.optimize.linprog`, HiGHS), which returns an optimal
   vertex. The first ``tau_max`` steps have no prediction and cost ``0`` in every regime.

A start stops when ``Gamma`` no longer changes, or from its sixth iteration on when the summed
absolute change is at most ``2 * K * T // 100``. A start fails when a regime keeps fewer than
:data:`FEWEST_STEPS` steps, holds a lagged value constant over its steps, or has a test that cannot
run on them. The start with the lowest objective is returned, the first of objectives equal per step
to :data:`~andrey.core.ties.KEY_DP` decimals (:func:`~andrey.core.ties.stable_key`): its ``Gamma``
from the last linear program and the PCMCI results of its last discovery step.

References
----------
Saggioro, E., de Wiljes, J., Kretschmer, M., and Runge, J. (2020). Reconstructing
regime-dependent causal relationships from observational time series. Chaos 30(11), 113115.
https://doi.org/10.1063/5.0020538
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from andrey.core import backend
from andrey.core.seeding import start_generator
from andrey.core.ties import KEY_DP, stable_key, tie_ranks

from .pcmci import PCMCIResult, lagged_design, pcmci

#: The iteration from which a start also stops on a small change of ``Gamma``, and the share of
#: ``K * T`` (in percent, doubled) that counts as small: the authors' code's values.
SMALL_CHANGE_FROM = 5
SMALL_CHANGE_PERCENT = 2


class RegimeFitError(ValueError):
    """Raised when no start of :func:`rpcmci` kept every regime fit."""


@dataclass(frozen=True)
class RPCMCIResult:
    """The returned start of one Regime-PCMCI run.

    ``weights`` is the ``(K, T)`` regime-assigning process; ``results[k]`` is PCMCI's result for
    regime ``k``. ``objective`` is the linear program's optimal value, ``n_iterations`` the
    iterations the start ran, ``converged`` whether it stopped on its own rule rather than on
    ``max_iter``, ``changes`` the summed absolute change of ``Gamma`` at each iteration,
    ``start`` the index of the start, and ``n_failed`` the number of starts that failed.
    """

    weights: np.ndarray
    results: tuple[PCMCIResult, ...]
    objective: float
    n_iterations: int
    converged: bool
    changes: tuple[float, ...]
    start: int
    n_failed: int


#: The fewest time steps a regime keeps in a start that does not fail: the authors' code's limit.
FEWEST_STEPS = 6


def regime_costs(
    data: np.ndarray,
    parents: tuple[tuple[tuple[int, int], ...], ...],
    selected: np.ndarray,
    tau_max: int,
    ranks: np.ndarray,
) -> np.ndarray:
    """The ``(T,)`` cost of one regime: its standardized squared residuals summed over variables.

    Each variable ``j`` is regressed with an intercept on the lagged ``(i, lag)`` values in
    ``parents[j]`` over the steps ``t >= tau_max`` that ``selected`` marks, and predicted at every
    step from ``tau_max`` on. The regression runs on the values standardized over the selected
    steps (``ddof=0``; a constant one keeps its scale), so a residual is in units of variable
    ``j``'s standard deviation over those steps, and neither the columns' units nor their offsets
    reach the fit. Parents enter in the order of their variables' ``ranks``, then lag, and the
    variables are summed in rank order, so the cost does not depend on the column order.
    """
    n_steps = data.shape[0]
    fit = selected[tau_max:]
    cost = np.zeros(n_steps, dtype=np.float64)
    for j in np.argsort(ranks, kind="stable"):
        links = sorted(parents[j], key=lambda link: (ranks[link[0]], link[1]))
        columns = [data[tau_max:, j]] + [data[tau_max - lag : n_steps - lag, i] for i, lag in links]
        Z = np.column_stack(columns)
        scale = Z[fit].std(axis=0)
        Z = (Z - Z[fit].mean(axis=0)) / np.where(scale > 0, scale, 1.0)
        coef = np.linalg.lstsq(Z[fit, 1:], Z[fit, 0], rcond=None)[0]
        residual = Z[:, 0] - Z[:, 1:] @ coef
        cost[tau_max:] += residual * residual
    return cost


def regime_lp(costs: np.ndarray, max_transitions: int) -> tuple[np.ndarray, float]:
    """Solve the regime-learning linear program for the ``(K, T)`` ``costs``.

    Returns the optimal ``(K, T)`` ``Gamma`` and the objective. The variables are ``Gamma`` row by
    row, then the slacks ``eta_k(t)``, ``t = 0 .. T - 2``, row by row.
    """
    n_regimes, n_steps = costs.shape
    n_gamma = n_regimes * n_steps
    n_eta = n_regimes * (n_steps - 1)
    gamma = np.arange(n_gamma).reshape(n_regimes, n_steps)
    eta = n_gamma + np.arange(n_eta).reshape(n_regimes, n_steps - 1)
    # gamma_k(t + 1) - gamma_k(t) - eta_k(t) <= 0 and its mirror, then sum_t eta_k(t) <= budget.
    pairs = np.arange(n_eta)
    rows = np.concatenate([pairs, pairs, pairs, n_eta + pairs, n_eta + pairs, n_eta + pairs])
    cols = np.concatenate([gamma[:, 1:].ravel(), gamma[:, :-1].ravel(), eta.ravel()] * 2)
    vals = np.concatenate([np.ones(n_eta), -np.ones(n_eta), -np.ones(n_eta)] * 2)
    vals[n_eta * 3 : n_eta * 5] *= -1
    budget = 2 * n_eta + np.repeat(np.arange(n_regimes), n_steps - 1)
    A_ub = sp.csr_array(
        (
            np.concatenate([vals, np.ones(n_eta)]),
            (np.concatenate([rows, budget]), np.concatenate([cols, eta.ravel()])),
        ),
        shape=(2 * n_eta + n_regimes, n_gamma + n_eta),
    )
    b_ub = np.concatenate([np.zeros(2 * n_eta), np.full(n_regimes, float(max_transitions))])
    A_eq = sp.csr_array(
        (np.ones(n_gamma), (np.tile(np.arange(n_steps), n_regimes), gamma.ravel())),
        shape=(n_steps, n_gamma + n_eta),
    )
    bounds = np.array([(0.0, 1.0)] * n_gamma + [(0.0, None)] * n_eta, dtype=object)
    solution = linprog(
        np.concatenate([costs.ravel(), np.zeros(n_eta)]),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=np.ones(n_steps),
        bounds=bounds.tolist(),
        method="highs-ds",
    )
    if solution.status != 0:  # the program is feasible and bounded; a failure is the solver's
        raise RuntimeError(f"the regime linear program did not solve: {solution.message}")
    weights = np.clip(solution.x[:n_gamma], 0.0, 1.0)  # a vertex's last bits can leave [0, 1]
    return weights.reshape(n_regimes, n_steps), float(solution.fun)


def _constant_column(data: np.ndarray, tau_max: int, rows: np.ndarray) -> bool:
    """Whether a lagged value is constant over the design ``rows``, so no test can read it."""
    design = lagged_design(data, 2 * tau_max)[rows]
    return bool((design.max(axis=0, initial=-np.inf) == design.min(axis=0, initial=np.inf)).any())


def _start(
    data: np.ndarray,
    weights: np.ndarray,
    *,
    max_transitions: int,
    switch_threshold: float,
    max_iter: int,
    tau_max: int,
    pcmci_args: dict,
    ranks: np.ndarray,
) -> RPCMCIResult | None:
    """Run one start from ``weights``; ``None`` when a regime cannot be fit."""
    n_regimes, n_steps = weights.shape
    changes: list[float] = []
    objective = 0.0
    results: tuple[PCMCIResult, ...] = ()
    converged = False
    for q in range(max_iter):
        selected = np.round(weights, KEY_DP) >= switch_threshold
        fits: list[PCMCIResult] = []
        costs = np.empty((n_regimes, n_steps), dtype=np.float64)
        for k in range(n_regimes):
            rows = selected[k, 2 * tau_max :]
            if selected[k].sum() < FEWEST_STEPS or _constant_column(data, tau_max, rows):
                return None
            try:
                result = pcmci(data, tau_max=tau_max, rows=rows, **pcmci_args)
            except ValueError:  # a test without a degree of freedom, or copies on these steps
                return None
            parents = tuple(
                tuple((int(i), int(lag) + 1) for lag, i in np.argwhere(result.graph[:, :, j]))
                for j in range(data.shape[1])
            )
            costs[k] = regime_costs(data, parents, selected[k], tau_max, ranks)
            fits.append(result)
        results = tuple(fits)
        previous = weights
        weights, objective = regime_lp(costs, max_transitions)
        changes.append(float(np.abs(weights - previous).sum()))
        small = n_regimes * n_steps * SMALL_CHANGE_PERCENT // 100
        if changes[-1] == 0 or (q >= SMALL_CHANGE_FROM and changes[-1] <= small):
            converged = True
            break
    return RPCMCIResult(
        weights=weights,
        results=results,
        objective=objective,
        n_iterations=len(changes),
        converged=converged,
        changes=tuple(changes),
        start=0,
        n_failed=0,
    )


def rpcmci(
    data: np.ndarray,
    *,
    n_regimes: int = 2,
    max_transitions: int = 1,
    switch_threshold: float = 0.5,
    n_starts: int = 50,
    max_iter: int = 20,
    tau_max: int = 1,
    pc_alpha: float = 0.2,
    alpha_level: float = 0.01,
    fdr_method: str = "none",
    indep_test: str = "student_t",
    seed: int = 0,
) -> RPCMCIResult:
    """Run Regime-PCMCI on a ``(T, n)`` series: ``n_starts`` starts, the best one returned.

    Start ``a`` draws its first ``Gamma`` from :func:`~andrey.core.seeding.start_generator`,
    uniform in ``[0, 1)``. ``pc_alpha``, ``alpha_level``, ``fdr_method``, and ``indep_test`` reach
    each regime's PCMCI. Raises :class:`RegimeFitError` when every start fails.
    """
    data = np.asarray(data, dtype=np.float64)
    ranks = tie_ranks(data)
    pcmci_args = {
        "pc_alpha": pc_alpha,
        "alpha_level": alpha_level,
        "fdr_method": fdr_method,
        "indep_test": indep_test,
    }
    best: RPCMCIResult | None = None
    n_failed = 0
    n_steps = data.shape[0]
    with backend.one_blas_thread():  # the regression costs then give the same bits on any host
        for a in range(n_starts):
            weights = start_generator(seed, a).uniform(0, 1, size=(n_regimes, n_steps))
            run = _start(
                data,
                weights,
                max_transitions=max_transitions,
                switch_threshold=switch_threshold,
                max_iter=max_iter,
                tau_max=tau_max,
                pcmci_args=pcmci_args,
                ranks=ranks,
            )
            if run is None:
                n_failed += 1
            elif best is None or stable_key(run.objective, n_steps) < stable_key(
                best.objective, n_steps
            ):
                best = replace(run, start=a)
    if best is None:
        raise RegimeFitError(
            f"rpcmci: every one of the {n_starts} starts left a regime with fewer than "
            f"{FEWEST_STEPS} time steps or with a test that cannot run on its steps; use fewer "
            "regimes, a smaller tau_max, more starts, or a longer series"
        )
    return replace(best, n_failed=n_failed)
