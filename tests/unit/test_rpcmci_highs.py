"""Regime-PCMCI's regime program, through SciPy's HiGHS bindings, gives the bits of ``linprog``."""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
import scipy
import scipy.sparse as sp
from scipy.optimize import linprog

from andrey.temporal import rpcmci
from andrey.temporal.rpcmci import RegimeFitError, regime_lp
from tests.recovery.helpers import datagen

SCIPY_HAS_HIGHS = tuple(int(part) for part in scipy.__version__.split(".")[:2]) >= (1, 15)
needs_highs = pytest.mark.skipif(not SCIPY_HAS_HIGHS, reason="SciPy before 1.15")

Solution = tuple[np.ndarray, float]


def linprog_regime_lp(costs: np.ndarray, max_transitions: int) -> Solution:
    """The regime program through ``linprog``, with its arrays built from nothing for each call."""
    n_regimes, n_steps = costs.shape
    n_gamma = n_regimes * n_steps
    n_eta = n_regimes * (n_steps - 1)
    gamma = np.arange(n_gamma).reshape(n_regimes, n_steps)
    eta = n_gamma + np.arange(n_eta).reshape(n_regimes, n_steps - 1)
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
    assert solution.status == 0
    weights = np.clip(solution.x[:n_gamma], 0.0, 1.0)
    return weights.reshape(n_regimes, n_steps), float(solution.fun)


def same_solution(a: Solution, b: Solution) -> bool:
    """Equal in shape, dtype, and bytes, and in the bits of the objective."""
    return (
        a[0].shape == b[0].shape
        and a[0].dtype == b[0].dtype
        and a[0].tobytes() == b[0].tobytes()
        and float(a[1]).hex() == float(b[1]).hex()
    )


# --------------------------------------------------------------------------------------------
# The programs of real fits and synthetic ones
# --------------------------------------------------------------------------------------------


def switching_series(seed: int, n_steps: int, n_regimes: int) -> np.ndarray:
    """Two variables, the first driving the second at lag 1 with a coefficient whose sign changes
    with the regime, in runs of 70 to 100 steps that cycle through ``n_regimes`` regimes.
    """
    rng = np.random.default_rng(seed)
    path: list[int] = []
    while len(path) < n_steps:
        path += [0 if not path else (path[-1] + 1) % n_regimes] * int(rng.integers(70, 101))
    sign = np.where(np.array(path[:n_steps]) % 2 == 0, 0.8, -0.8)
    x = rng.standard_normal((n_steps, 2))
    for t in range(1, n_steps):
        x[t, 0] += 0.2 * x[t - 1, 0]
        x[t, 1] += sign[t] * x[t - 1, 0] + 0.2 * x[t - 1, 1]
    return x


def fits() -> Iterator[tuple[np.ndarray, dict]]:
    """Series with the arguments of the fits whose programs are recorded: two regimes of
    ``regime_var_2v``, two regimes of 300 and of 400 steps, and three regimes, every start of which
    fails after solving its programs.
    """
    yield datagen.generate("regime_var_2v", shuffle=False), {"max_transitions": 7, "n_starts": 8}
    yield switching_series(1, 300, 2), {"max_transitions": 3, "n_starts": 6, "tau_max": 1}
    yield switching_series(5, 400, 2), {"max_transitions": 3, "n_starts": 4, "tau_max": 2}
    yield switching_series(2, 600, 3), {"max_transitions": 5, "n_regimes": 3, "n_starts": 4}


@pytest.fixture(scope="module")
def recorded() -> list[tuple[np.ndarray, int]]:
    """The ``(costs, max_transitions)`` of every program the fits of :func:`fits` solve."""
    programs: list[tuple[np.ndarray, int]] = []
    real = rpcmci.regime_lp

    def record(costs: np.ndarray, max_transitions: int) -> Solution:
        programs.append((costs.copy(), max_transitions))
        return real(costs, max_transitions)

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch.setattr(rpcmci, "regime_lp", record)
        for x, settings in fits():
            try:
                rpcmci.rpcmci(x, seed=0, **settings)
            except RegimeFitError:  # a start that fails still solved its programs
                pass
    return programs


def synthetic() -> Iterator[tuple[str, np.ndarray, int]]:
    """Costs of tied, constant, scaled, and fractional programs, with their budgets."""
    rng = np.random.default_rng(0)
    for n_regimes, n_steps, max_transitions in ((2, 6, 1), (2, 40, 3), (3, 40, 2), (4, 120, 5)):
        label = f"{n_regimes}x{n_steps}-{max_transitions}"
        yield f"uniform {label}", rng.uniform(0.0, 4.0, (n_regimes, n_steps)), max_transitions
        yield f"integers {label}", rng.integers(0, 3, (n_regimes, n_steps)) * 1.0, max_transitions
        yield f"quarters {label}", rng.integers(0, 17, (n_regimes, n_steps)) / 4.0, max_transitions
        blocks = np.repeat(rng.integers(0, 3, (n_regimes, n_steps // 6 + 1)), 6, axis=1)
        yield f"blocks {label}", blocks[:, :n_steps] * 1.0, max_transitions
    yield "zero", np.zeros((2, 30)), 2
    yield "equal", np.ones((3, 30)), 2
    yield "large", rng.uniform(0.0, 4.0, (2, 60)) * 1e6, 3
    yield "small", rng.uniform(0.0, 4.0, (2, 60)) * 1e-6, 3


SYNTHETIC = list(synthetic())


def test_the_recorded_programs_cover_two_and_three_regimes_and_fractional_vertices(
    recorded,
) -> None:
    assert len(recorded) > 50
    assert {costs.shape for costs, _ in recorded} == {(2, 600), (2, 300), (2, 400), (3, 600)}
    fractional = sum(
        bool(((w > 0) & (w < 1)).any()) for w in (linprog_regime_lp(c, m)[0] for c, m in recorded)
    )
    assert 0 < fractional < len(recorded)


def test_a_program_of_a_real_fit_gives_the_bits_of_linprog(recorded) -> None:
    for costs, max_transitions in recorded:
        assert same_solution(
            regime_lp(costs, max_transitions), linprog_regime_lp(costs, max_transitions)
        )


@pytest.mark.parametrize(
    ("costs", "max_transitions"),
    [case[1:] for case in SYNTHETIC],
    ids=[case[0] for case in SYNTHETIC],
)
def test_a_synthetic_program_gives_the_bits_of_linprog(costs, max_transitions) -> None:
    assert same_solution(
        regime_lp(costs, max_transitions), linprog_regime_lp(costs, max_transitions)
    )


def test_a_fit_gives_the_bits_it_gives_through_linprog(monkeypatch) -> None:
    x = switching_series(3, 400, 2)
    kwargs = {"max_transitions": 3, "n_starts": 3, "seed": 1, "tau_max": 2}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = rpcmci.rpcmci(x, **kwargs)
        monkeypatch.setattr(rpcmci, "regime_lp", lambda *args: linprog_regime_lp(*args))
        reference = rpcmci.rpcmci(x, **kwargs)
    assert direct.weights.tobytes() == reference.weights.tobytes()
    assert direct.objective.hex() == reference.objective.hex()
    assert direct.changes == reference.changes
    assert (direct.start, direct.n_failed, direct.n_iterations, direct.converged) == (
        reference.start,
        reference.n_failed,
        reference.n_iterations,
        reference.converged,
    )
    for ours, theirs in zip(direct.results, reference.results, strict=True):
        assert ours.p_values.tobytes() == theirs.p_values.tobytes()
        assert ours.graph.tobytes() == theirs.graph.tobytes()


# --------------------------------------------------------------------------------------------
# Where the bindings are used, and where linprog is
# --------------------------------------------------------------------------------------------


def test_the_bindings_are_in_use_from_scipy_1_15_and_not_before() -> None:
    assert (rpcmci._HIGHS is not None) == SCIPY_HAS_HIGHS


@needs_highs
def test_the_bindings_have_what_the_direct_call_reads() -> None:
    core = rpcmci._HIGHS
    for name in ("setOptionValue", "passModel", "run", "getModelStatus", "getSolution", "getInfo"):
        assert callable(getattr(core._Highs, name))
    lp = core.HighsLp()
    for name in ("num_col_", "num_row_", "col_cost_", "col_lower_", "col_upper_", "row_lower_"):
        assert hasattr(lp, name)
    for name in ("row_upper_", "a_matrix_"):
        assert hasattr(lp, name)
    for name in ("num_col_", "num_row_", "format_", "start_", "index_", "value_"):
        assert hasattr(lp.a_matrix_, name)
    assert int(core.simplex_constants.SimplexStrategy.kSimplexStrategyDual) == 1
    assert int(core.HighsDebugLevel.kHighsDebugLevelNone) == 0
    assert {core.HighsStatus.kOk, core.HighsStatus.kError}
    assert core.HighsModelStatus.kOptimal
    assert core.MatrixFormat.kColwise


@needs_highs
def test_the_direct_call_solves_the_programs_and_linprog_does_not(monkeypatch) -> None:
    direct = []
    real = rpcmci._solve_highs
    monkeypatch.setattr(rpcmci, "_solve_highs", lambda *args: direct.append(1) or real(*args))
    monkeypatch.setattr(rpcmci, "linprog", lambda *args, **kwargs: pytest.fail("linprog called"))
    for costs, max_transitions in [case[1:] for case in SYNTHETIC[:4]]:
        regime_lp(costs, max_transitions)
    assert len(direct) == 4


@needs_highs
def test_the_model_is_the_one_linprog_hands_to_highs(monkeypatch) -> None:
    """Each array the direct call reads equals, in dtype and bytes, the argument ``linprog`` passes
    to SciPy's HiGHS wrapper (the two index arrays, in values), and the options it sets are the
    wrapper's.
    """
    module = pytest.importorskip("scipy.optimize._linprog_highs")
    seen: list[tuple] = []
    real = module._highs_wrapper
    monkeypatch.setattr(module, "_highs_wrapper", lambda *args: seen.append(args) or real(*args))
    for costs, max_transitions in [case[1:] for case in SYNTHETIC[::5]]:
        seen.clear()
        program = rpcmci._build_program(*costs.shape, max_transitions)
        objective = np.concatenate([costs.ravel(), np.zeros(program.n_eta)])
        rpcmci._solve_linprog(program, objective)
        c, indptr, indices, data, lhs, rhs, lb, ub, _, options = seen[0]
        mine = (
            objective,
            program.indptr,
            program.indices,
            program.data,
            program.row_lower,
            program.row_upper,
            program.col_lower,
            program.col_upper,
        )
        for ours, theirs in zip(mine, (c, indptr, indices, data, lhs, rhs, lb, ub), strict=True):
            if (
                ours.dtype.kind == "i"
            ):  # SciPy's index width depends on its version; HiGHS reads 32 bits
                assert np.array_equal(ours, theirs)
            else:
                assert (ours.dtype, ours.shape) == (theirs.dtype, theirs.shape)
                assert ours.tobytes() == theirs.tobytes()
        passed = {
            key: (("on" if value else "off") if key == "presolve" else value)
            for key, value in options.items()
            if value is not None and key != "sense"
        }
        assert passed == dict(rpcmci._HIGHS_OPTIONS)


def same_program(a: rpcmci._Program, b: rpcmci._Program) -> bool:
    """Equal in every array, in dtype and bytes, and in the entries of the sparse matrices."""
    arrays = ("b_ub", "b_eq", "indptr", "indices", "data", "row_lower", "row_upper", "col_lower")
    arrays += ("col_upper",)
    return (
        (a.n_gamma, a.n_eta) == (b.n_gamma, b.n_eta)
        and a.bounds == b.bounds
        and all(
            getattr(a, name).dtype == getattr(b, name).dtype
            and getattr(a, name).tobytes() == getattr(b, name).tobytes()
            for name in arrays
        )
        and all(
            getattr(a, name).data.tobytes() == getattr(b, name).data.tobytes()
            and (getattr(a, name) != getattr(b, name)).nnz == 0
            for name in ("A_ub", "A_eq")
        )
    )


def test_the_arrays_of_a_shape_are_built_once_and_solves_leave_them_as_they_were() -> None:
    first = rpcmci._program(2, 40, 3)
    assert rpcmci._program(2, 40, 3) is first
    assert rpcmci._program(2, 41, 3) is not first
    rng = np.random.default_rng(0)
    for _ in range(3):
        regime_lp(rng.uniform(0.0, 4.0, (2, 40)), 3)
    assert same_program(rpcmci._program(2, 40, 3), rpcmci._build_program(2, 40, 3))


def test_the_program_cache_misses_when_any_argument_changes() -> None:
    rpcmci._program.cache_clear()
    rpcmci._program(2, 40, 3)
    for args in ((3, 40, 3), (2, 41, 3), (2, 40, 4)):
        misses = rpcmci._program.cache_info().misses
        found = rpcmci._program(*args)
        assert rpcmci._program.cache_info().misses == misses + 1
        assert same_program(found, rpcmci._build_program(*args))
    rpcmci._program.cache_clear()


def test_without_the_bindings_linprog_solves_every_program(monkeypatch, recorded) -> None:
    calls = []
    real = rpcmci.linprog
    monkeypatch.setattr(rpcmci, "_HIGHS", None)
    monkeypatch.setattr(rpcmci, "_solve_highs", lambda *args: pytest.fail("direct call"))
    monkeypatch.setattr(
        rpcmci, "linprog", lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs)
    )
    for costs, max_transitions in recorded[:20]:
        assert same_solution(
            regime_lp(costs, max_transitions), linprog_regime_lp(costs, max_transitions)
        )
    assert len(calls) == 20


def test_a_fit_without_the_bindings_gives_the_bits_of_a_fit_with_them(monkeypatch) -> None:
    x = switching_series(4, 300, 2)
    kwargs = {"max_transitions": 3, "n_starts": 3, "seed": 2, "tau_max": 1}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_bindings = rpcmci.rpcmci(x, **kwargs)
        monkeypatch.setattr(rpcmci, "_HIGHS", None)
        without = rpcmci.rpcmci(x, **kwargs)
    assert with_bindings.weights.tobytes() == without.weights.tobytes()
    assert with_bindings.objective.hex() == without.objective.hex()
    assert with_bindings.changes == without.changes
    assert with_bindings.start == without.start


def test_a_program_that_highs_does_not_solve_goes_to_linprog(monkeypatch) -> None:
    calls = []
    real = rpcmci.linprog
    monkeypatch.setattr(rpcmci, "_solve_highs", lambda *args: None)
    monkeypatch.setattr(
        rpcmci, "linprog", lambda *args, **kwargs: calls.append(1) or real(*args, **kwargs)
    )
    costs, max_transitions = SYNTHETIC[0][1:]
    assert same_solution(
        regime_lp(costs, max_transitions), linprog_regime_lp(costs, max_transitions)
    )
    assert calls == [1]


def test_a_cost_that_is_not_finite_goes_to_linprog_and_its_error(monkeypatch) -> None:
    monkeypatch.setattr(rpcmci, "_solve_highs", lambda *args: pytest.fail("direct call"))
    for bad in (np.nan, np.inf, -np.inf):
        costs = np.ones((2, 10))
        costs[1, 3] = bad
        with pytest.raises(ValueError, match="c must not contain"):
            regime_lp(costs, 1)


def test_a_program_that_linprog_does_not_solve_raises_with_its_message(monkeypatch) -> None:
    monkeypatch.setattr(rpcmci, "_HIGHS", None)
    failed = SimpleNamespace(status=4, message="the solver stopped")
    monkeypatch.setattr(rpcmci, "linprog", lambda *args, **kwargs: failed)
    with pytest.raises(RuntimeError, match="did not solve: the solver stopped"):
        regime_lp(np.ones((2, 10)), 1)


# --------------------------------------------------------------------------------------------
# The self-check
# --------------------------------------------------------------------------------------------


@needs_highs
def test_the_self_check_programs_have_fractional_and_integral_vertices_at_a_binding_budget() -> (
    None
):
    fractional = []
    for costs, max_transitions in rpcmci._SELF_CHECK:
        program = rpcmci._build_program(*costs.shape, max_transitions)
        point = rpcmci._solve_linprog(
            program, np.concatenate([costs.ravel(), np.zeros(program.n_eta)])
        )[0]
        switches = point[program.n_gamma :].reshape(costs.shape[0], -1).sum(axis=1)
        np.testing.assert_allclose(switches, max_transitions)
        gamma = point[: program.n_gamma]
        fractional.append(bool(((gamma > 1e-9) & (gamma < 1 - 1e-9)).any()))
    assert fractional == [True, True, False]


@needs_highs
def test_the_self_check_passes_where_the_bindings_are_there() -> None:
    assert rpcmci._load_highs() is rpcmci._HIGHS


def test_bindings_that_cannot_be_imported_leave_linprog(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "scipy.optimize._highspy._core", None)
    assert rpcmci._load_highs() is None


DOCTORED = ("a point one step off", "a value one step off", "no solution", "an error")


def doctored_solver(kind: str):
    """``rpcmci._solve_highs`` with one change to its answer."""
    real = rpcmci._solve_highs

    def solve(core, program, objective):
        if kind == "an error":
            raise RuntimeError("the bindings changed")
        result = real(core, program, objective)
        if kind == "no solution":
            return None
        point, value = result
        if kind == "a point one step off":
            return np.nextafter(point, np.inf), value
        return point, float(np.nextafter(value, np.inf))

    return solve


@needs_highs
@pytest.mark.parametrize("kind", DOCTORED)
def test_a_direct_call_that_differs_from_linprog_fails_the_self_check(monkeypatch, kind) -> None:
    monkeypatch.setattr(rpcmci, "_solve_highs", doctored_solver(kind))
    assert rpcmci._load_highs() is None
    monkeypatch.undo()
    assert rpcmci._load_highs() is rpcmci._HIGHS


@needs_highs
def test_options_the_bindings_reject_fail_the_self_check(monkeypatch) -> None:
    core = rpcmci._HIGHS

    class Rejecting:
        def __init__(self) -> None:
            self.real = core._Highs()

        def setOptionValue(self, name, value):  # noqa: N802
            return core.HighsStatus.kError

        def __getattr__(self, name):
            return getattr(self.real, name)

    monkeypatch.setattr(core, "_Highs", Rejecting)
    assert rpcmci._load_highs() is None
