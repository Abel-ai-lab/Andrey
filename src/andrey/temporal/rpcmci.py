"""Regime-PCMCI: the regimes of a time series and the lagged causal links within each regime.

The series ``x_t``, ``t = 0 .. T - 1``, is assumed to switch between ``K`` persistent regimes, each
with its own lagged causal graph. A regime-assigning process ``Gamma``, a ``(K, T)`` matrix,
weights the regimes at each step. Each start draws its first ``Gamma`` entry by entry, uniformly in
``[0, 1)`` and not normalized, and alternates two steps up to ``max_iter`` times:

1. **Causal discovery and model estimation.** For each regime ``k``, PCMCI runs on the steps ``t``
   with ``gamma_k(t) > switch_threshold``, each weight rounded to :data:`~andrey.core.ties.KEY_DP`
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
   HiGHS's dual simplex method (:func:`regime_lp`), which returns an optimal vertex. The first
   ``tau_max`` steps have no prediction and cost ``0`` in every regime.

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

import importlib
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import lru_cache
from types import ModuleType

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from andrey.core import backend
from andrey.core.seeding import start_generator
from andrey.core.ties import KEY_DP, stable_key, tie_ranks

from ._parallel_rpcmci import pooled_starts
from .pcmci import PCMCIResult, lagged_design, pcmci

#: The iteration enabling early stopping on small ``Gamma`` changes, and the small-change share of
#: ``K * T`` (percent, doubled): the authors' code's values.
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

    Each variable ``j`` is regressed with an intercept on lagged ``(i, lag)`` values in
    ``parents[j]`` over ``selected`` steps ``t >= tau_max``, then predicted from ``tau_max`` onward.
    Values are standardized over selected steps (``ddof=0``; constants keep their scale), expressing
    residuals in variable ``j``'s standard deviations and making fits independent of column units
    and offsets. Parents follow variable ``ranks``, then lag; variables are summed in rank order,
    making cost independent of column order.
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


@dataclass(frozen=True, eq=False)
class _Program:
    """The regime linear program of one shape, which the costs do not change.

    ``A_ub``, ``b_ub``, ``A_eq``, ``b_eq``, and ``bounds`` are the arguments of ``linprog``. The
    rest is the same model in HiGHS's form, as ``linprog`` hands it over: the stacked constraint
    matrix in compressed-column form with 32-bit indices, the row bounds, and the column bounds.
    """

    n_gamma: int
    n_eta: int
    A_ub: sp.csr_array
    b_ub: np.ndarray
    A_eq: sp.csr_array
    b_eq: np.ndarray
    bounds: list[tuple[float, float | None]]
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    row_lower: np.ndarray
    row_upper: np.ndarray
    col_lower: np.ndarray
    col_upper: np.ndarray


def _build_program(n_regimes: int, n_steps: int, max_transitions: int) -> _Program:
    """The :class:`_Program` of ``n_regimes`` regimes and ``n_steps`` steps.

    The variables are ``Gamma`` row by row, then the slacks ``eta_k(t)``, ``t = 0 .. T - 2``, row by
    row.
    """
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
    b_eq = np.ones(n_steps)
    bounds = np.array([(0.0, 1.0)] * n_gamma + [(0.0, None)] * n_eta, dtype=object).tolist()
    stacked = sp.csc_array(sp.vstack((A_ub, A_eq)))
    return _Program(
        n_gamma=n_gamma,
        n_eta=n_eta,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        indptr=stacked.indptr.astype(np.int32),
        indices=stacked.indices.astype(np.int32),
        data=stacked.data,
        row_lower=np.concatenate([np.full(b_ub.size, -np.inf), b_eq]),
        row_upper=np.concatenate([b_ub, b_eq]),
        col_lower=np.zeros(n_gamma + n_eta),
        col_upper=np.concatenate([np.ones(n_gamma), np.full(n_eta, np.inf)]),
    )


#: The program of the last shapes fitted: a fit solves one shape at every iteration of every start.
_program = lru_cache(maxsize=2)(_build_program)


def _solve_linprog(program: _Program, objective: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve ``program`` for the cost vector ``objective`` through ``scipy.optimize.linprog``.

    Returns the optimal point and value, or raises ``RuntimeError`` when the solver does not
    finish, and the errors of ``linprog`` on an input it rejects, such as a cost that is not finite.
    """
    solution = linprog(
        objective,
        A_ub=program.A_ub,
        b_ub=program.b_ub,
        A_eq=program.A_eq,
        b_eq=program.b_eq,
        bounds=program.bounds,
        method="highs-ds",
    )
    if solution.status != 0:  # the program is feasible and bounded; a failure is the solver's
        raise RuntimeError(f"the regime linear program did not solve: {solution.message}")
    return solution.x, float(solution.fun)


#: The options of the direct call, which are the ones ``scipy.optimize.linprog`` sets for
#: ``method="highs-ds"``: presolve on, the simplex solver with the dual strategy (1), no debug
#: checks (0), and no output.
_HIGHS_OPTIONS = (
    ("presolve", "on"),
    ("solver", "simplex"),
    ("simplex_strategy", 1),
    ("highs_debug_level", 0),
    ("log_to_console", False),
    ("output_flag", False),
)


def _solve_highs(
    core: ModuleType, program: _Program, objective: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """Solve ``program`` for the cost vector ``objective`` in a new HiGHS object of ``core``.

    The model and the options are those ``scipy.optimize.linprog`` passes for ``method="highs-ds"``:
    presolve on, the simplex solver with its dual strategy, and no output. Returns the optimal point
    and value, or ``None`` when HiGHS does not report an optimal solution, so that
    :func:`_solve_linprog` solves it and raises its error.
    """
    highs = core._Highs()
    for name, value in _HIGHS_OPTIONS:
        if highs.setOptionValue(name, value) != core.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejects the option {name}")
    model = core.HighsLp()
    n_columns, n_rows = objective.size, program.row_upper.size
    model.num_col_ = n_columns
    model.num_row_ = n_rows
    model.a_matrix_.num_col_ = n_columns
    model.a_matrix_.num_row_ = n_rows
    model.a_matrix_.format_ = core.MatrixFormat.kColwise
    model.col_cost_ = objective
    model.col_lower_ = program.col_lower
    model.col_upper_ = program.col_upper
    model.row_lower_ = program.row_lower
    model.row_upper_ = program.row_upper
    model.a_matrix_.start_ = program.indptr
    model.a_matrix_.index_ = program.indices
    model.a_matrix_.value_ = program.data
    if highs.passModel(model) == core.HighsStatus.kError:
        return None
    if highs.run() == core.HighsStatus.kError:
        return None
    if highs.getModelStatus() != core.HighsModelStatus.kOptimal:
        return None
    return np.array(highs.getSolution().col_value), float(highs.getInfo().objective_function_value)


#: The programs of the self-check, each a ``(costs, max_transitions)``: two with a vertex whose
#: ``Gamma`` has fractional entries, for two and for three regimes, and one with three regimes whose
#: ``Gamma`` is 0 or 1 and whose budget of two switches binds.
_SELF_CHECK = (
    (
        np.array(
            [
                [2.0, 3.75, 0.5, 3.75, 1.25, 1.75, 3.25, 1.75],
                [2.25, 0.0, 3.0, 2.25, 1.25, 3.25, 1.25, 1.75],
            ]
        ),
        1,
    ),
    (
        np.array(
            [
                [2.5, 3.0, 2.75, 3.75, 3.5, 0.75, 2.75, 3.25],
                [3.0, 3.25, 2.75, 3.25, 2.75, 2.0, 1.25, 3.5],
                [0.25, 2.0, 0.75, 0.25, 1.0, 1.75, 3.5, 2.5],
            ]
        ),
        1,
    ),
    (
        np.array(
            [
                [0.5, 2.5, 0.25, 3.0, 1.5, 0.75, 2.0, 0.5, 1.0],
                [1.5, 0.5, 2.0, 0.25, 1.0, 2.5, 0.5, 1.5, 3.0],
                [2.0, 1.0, 1.0, 1.5, 0.5, 1.0, 1.5, 2.0, 0.25],
            ]
        ),
        2,
    ),
)


def _load_highs() -> ModuleType | None:
    """The HiGHS bindings of SciPy 1.15 or later, when :func:`_solve_highs` reproduces ``linprog``.

    The bindings are private to SciPy and have changed between its versions. They are used only
    when the solution of each program of ``_SELF_CHECK``, point and value, has the bytes that
    ``linprog`` returns for it, so any other SciPy, and any failure of the import or of a call,
    leaves ``linprog``.
    """
    try:
        core = importlib.import_module("scipy.optimize._highspy._core")
        for costs, max_transitions in _SELF_CHECK:
            program = _build_program(*costs.shape, max_transitions)
            objective = np.concatenate([costs.ravel(), np.zeros(program.n_eta)])
            direct = _solve_highs(core, program, objective)
            point, value = _solve_linprog(program, objective)
            if (
                direct is None
                or direct[0].tobytes() != point.tobytes()
                or float(direct[1]).hex() != float(value).hex()
            ):
                return None
        return core
    except Exception:
        return None


_HIGHS = _load_highs()


def regime_lp(costs: np.ndarray, max_transitions: int) -> tuple[np.ndarray, float]:
    """Solve the regime-learning linear program for the ``(K, T)`` ``costs``.

    Returns the optimal ``(K, T)`` ``Gamma`` and the objective. The variables are ``Gamma`` row by
    row, then the slacks ``eta_k(t)``, ``t = 0 .. T - 2``, row by row.

    The program goes to HiGHS through the bindings SciPy bundles, in a new object with the options
    ``scipy.optimize.linprog`` passes for ``method="highs-ds"``, so the point is the one ``linprog``
    returns. ``linprog`` solves it instead where those bindings are missing or fail
    :func:`_load_highs`'s check, when a cost is not finite, and when HiGHS finishes without an
    optimal solution, which raises ``linprog``'s error.
    """
    n_regimes, n_steps = costs.shape
    program = _program(n_regimes, n_steps, max_transitions)
    objective = np.concatenate([costs.ravel(), np.zeros(program.n_eta)])
    solution = None
    if _HIGHS is not None and np.isfinite(objective).all():
        solution = _solve_highs(_HIGHS, program, objective)
    if solution is None:
        solution = _solve_linprog(program, objective)
    point, value = solution
    weights = np.clip(point[: program.n_gamma], 0.0, 1.0)  # a vertex's last bits can leave [0, 1]
    return weights.reshape(n_regimes, n_steps), value


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
        selected = np.round(weights, KEY_DP) > switch_threshold
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


@dataclass(frozen=True)
class Starts:
    """What every start of one fit shares: the series, its variables' ``ranks``, the ``seed`` of the
    starts' first weights, and the settings :func:`_start` takes.

    A worker process rebuilds the fit's starts from one, so a start runs the same in any process.
    """

    data: np.ndarray
    ranks: np.ndarray
    seed: int
    n_regimes: int
    max_transitions: int
    switch_threshold: float
    max_iter: int
    tau_max: int
    pcmci_args: dict


def run_start(starts: Starts, a: int) -> RPCMCIResult | None:
    """Start ``a`` of a fit: its first weights from :func:`~andrey.core.seeding.start_generator`,
    uniform in ``[0, 1)``, run by :func:`_start`; ``None`` when it fails."""
    weights = start_generator(starts.seed, a).uniform(
        0, 1, size=(starts.n_regimes, starts.data.shape[0])
    )
    return _start(
        starts.data,
        weights,
        max_transitions=starts.max_transitions,
        switch_threshold=starts.switch_threshold,
        max_iter=starts.max_iter,
        tau_max=starts.tau_max,
        pcmci_args=starts.pcmci_args,
        ranks=starts.ranks,
    )


def best_start(
    runs: Iterable[RPCMCIResult | None], n_steps: int
) -> tuple[RPCMCIResult | None, int]:
    """The best of the starts ``runs``, given in start order, and the number that failed.

    The best has the lowest objective, per step to :data:`~andrey.core.ties.KEY_DP` decimals, the
    first start of objectives that tie, and holds its index as ``start``; ``None`` when every start
    failed. ``runs`` are read once, in order, so they come from this process or from workers alike.
    """
    best: RPCMCIResult | None = None
    n_failed = 0
    for a, run in enumerate(runs):
        if run is None:
            n_failed += 1
        elif best is None or stable_key(run.objective, n_steps) < stable_key(
            best.objective, n_steps
        ):
            best = replace(run, start=a)
    return best, n_failed


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

    With more than one worker (``andrey.config.n_jobs``) and enough work, the starts run on worker
    processes (:func:`~andrey.temporal._parallel_rpcmci.pooled_starts`) and come back in start
    order, so the result is the one of a fit in this process.
    """
    data = np.asarray(data, dtype=np.float64)
    starts = Starts(
        data=data,
        ranks=tie_ranks(data),
        seed=seed,
        n_regimes=n_regimes,
        max_transitions=max_transitions,
        switch_threshold=switch_threshold,
        max_iter=max_iter,
        tau_max=tau_max,
        pcmci_args={
            "pc_alpha": pc_alpha,
            "alpha_level": alpha_level,
            "fdr_method": fdr_method,
            "indep_test": indep_test,
        },
    )
    runs = pooled_starts(starts, n_starts)
    with backend.one_blas_thread():  # the regression costs then give the same bits on any host
        if runs is None:
            runs = (run_start(starts, a) for a in range(n_starts))
        best, n_failed = best_start(runs, data.shape[0])
    if best is None:
        raise RegimeFitError(
            f"rpcmci: every one of the {n_starts} starts left a regime with fewer than "
            f"{FEWEST_STEPS} time steps or with a test that cannot run on its steps; use fewer "
            "regimes, a smaller tau_max, more starts, or a longer series"
        )
    return replace(best, n_failed=n_failed)
