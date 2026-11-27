"""The engine's gradient-only steps and its matrix exponential give the bits of the full calls."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import scipy
import scipy.linalg

import andrey
from andrey.search import continuous
from andrey.search.continuous import (
    Canonical,
    L1Penalty,
    QuasiMCP,
    adam_augmented_lagrangian,
    central_path,
    gaussian_likelihood,
    l1_penalty,
    least_squares,
    loss_gradient,
    penalty_gradient,
    quasi_mcp,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning"),
    pytest.mark.filterwarnings(
        "ignore:(overflow|invalid value) encountered in matmul:RuntimeWarning"
    ),
]


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal in shape, dtype, and bytes, so ``-0.0`` and each ``NaN`` count."""
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def covariance(d: int, seed: int, *, duplicate: bool = False) -> np.ndarray:
    """The covariance of 200 rows of ``d`` dependent columns; with ``duplicate``, the last column
    repeats the second, so the covariance has rank ``d - 1``.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((200, d)) @ np.triu(rng.uniform(-1, 1, (d, d)))
    if duplicate:
        x[:, -1] = x[:, 1]
    x -= x.mean(axis=0)
    return x.T @ x / 200


def canonical(cov: np.ndarray, forbidden: tuple | None = None, required: tuple | None = None):
    d = cov.shape[0]
    masks = []
    for cells in (forbidden, required):
        mask = np.zeros((d, d), dtype=bool)
        for i, j in cells or ():
            mask[i, j] = True
        masks.append(mask)
    order = np.arange(d)
    return Canonical(cov, order, order, *masks)


# --------------------------------------------------------------------------------------------
# Gradient-only companions
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("loss", [least_squares, gaussian_likelihood])
@pytest.mark.parametrize("d", [1, 3, 8])
def test_a_loss_gradient_equals_the_gradient_of_the_full_loss(loss, d) -> None:
    cov = covariance(d, seed=d)
    gradient = loss_gradient(loss, cov)
    rng = np.random.default_rng(d)
    for _ in range(5):
        weights = rng.uniform(-1, 1, (d, d))  # a diagonal and cycles included
        assert same(gradient(weights), loss(cov, weights)[1])


def test_the_likelihood_gradient_keeps_the_variance_floor() -> None:
    """Column 4 repeats column 1, so with ``W[1, 4] = 1`` its residual variance is rounding noise,
    far below the floor, which then sets the divisor of that column.
    """
    cov = covariance(5, seed=0, duplicate=True)
    weights = np.zeros((5, 5))
    weights[1, 4] = 1.0
    residual = np.eye(5) - weights
    variances = np.einsum("ij,ij->j", residual, cov @ residual)
    floor = continuous._VARIANCE_FLOOR * np.diagonal(cov)
    assert variances[4] < floor[4]
    assert same(
        loss_gradient(gaussian_likelihood, cov)(weights), gaussian_likelihood(cov, weights)[1]
    )


def reference_l1(lambda1: float) -> Callable:
    """The penalty as a closure of its value and gradient, as it was written before the objects."""

    def penalty(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return lambda1 * np.abs(weights), lambda1 * np.sign(weights)

    return penalty


def reference_quasi_mcp(lam: float, delta: float) -> Callable:
    """The quasi-MCP penalty as a closure of its value and gradient, as it was written before the
    objects.
    """

    def penalty(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inside = np.abs(weights) <= delta
        value = lam * np.where(inside, np.abs(weights) - weights**2 / (2.0 * delta), delta / 2.0)
        grad = lam * np.where(inside, np.sign(weights) - weights / delta, 0.0)
        return value, grad

    return penalty


#: Weights with a zero, both signs, an entry on ``delta = 0.2`` and one just below it, and entries
#: beyond it.
PENALTY_WEIGHTS = np.array(
    [
        [0.0, 0.2, -0.2, 0.5],
        [-1.3, 0.05, 0.0, 0.19999999999999998],
        [0.2000000000000001, -0.0, 7.0, 0.1],
    ]
)


@pytest.mark.parametrize(
    ("penalty", "reference"),
    [
        (l1_penalty(0.03), reference_l1(0.03)),
        (quasi_mcp(0.4, 0.2), reference_quasi_mcp(0.4, 0.2)),
    ],
    ids=["l1", "quasi_mcp"],
)
def test_a_penalty_object_equals_its_closure_in_value_and_gradient(penalty, reference) -> None:
    rng = np.random.default_rng(0)
    for weights in (PENALTY_WEIGHTS, rng.uniform(-1, 1, (6, 6)), np.zeros((3, 3))):
        value, grad = penalty(weights)
        ref_value, ref_grad = reference(weights)
        assert same(value, ref_value)
        assert same(grad, ref_grad)
        assert same(penalty_gradient(penalty)(weights), ref_grad)


def test_the_penalties_are_objects_that_the_gradient_helpers_recognize() -> None:
    assert isinstance(l1_penalty(0.1), L1Penalty)
    assert isinstance(quasi_mcp(0.4, 0.2), QuasiMCP)


def test_another_callable_gives_its_own_gradient() -> None:
    """A loss or a penalty that is not one of the engine's is called for both parts and keeps the
    gradient.
    """
    cov = covariance(4, seed=1)
    weights = np.random.default_rng(1).uniform(-1, 1, (4, 4))
    calls = []

    def loss(c: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
        calls.append("loss")
        return least_squares(c, w)

    def penalty(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append("penalty")
        return 0.5 * np.abs(w), 0.5 * np.sign(w)

    assert same(loss_gradient(loss, cov)(weights), least_squares(cov, weights)[1])
    assert same(penalty_gradient(penalty)(weights), 0.5 * np.sign(weights))
    assert calls == ["loss", "penalty"]


def plain(loss: Callable, penalty: Callable) -> tuple[Callable, Callable]:
    """A loss and a penalty as plain functions, which the solvers call for value and gradient."""
    return (lambda cov, weights: loss(cov, weights)), (lambda weights: penalty(weights))


def count_undone_steps(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the inverses with a negative entry, which send DAGMA's stage to its undo branch."""
    count = [0]
    inverse = np.linalg.inv

    def counting(a: np.ndarray) -> np.ndarray:
        result = inverse(a)
        count[0] += int(((result + 1e-16) < 0).any())
        return result

    monkeypatch.setattr(continuous.np.linalg, "inv", counting)
    return count


def central(cov, loss, penalty, *, lr, free_diagonal, forbidden=None, required=None):
    d = cov.shape[0]
    rng = np.random.default_rng(d)
    weights = rng.uniform(-1, 1, (d, d)) * ~np.eye(d, dtype=bool)
    radius = np.max(np.abs(np.linalg.eigvals(weights * weights)))
    return central_path(
        canonical(cov, forbidden, required),
        loss=loss,
        penalty=penalty,
        start=weights * np.sqrt(0.97 / radius),
        stages=3,
        warm_iter=1500,
        max_iter=2500,
        lr=lr,
        free_diagonal=free_diagonal,
    )


CENTRAL_CASES = [
    pytest.param(10, 0, 3e-4, False, id="d10-small-rate"),
    pytest.param(6, 1, 0.2, False, id="d6-large-rate"),
    pytest.param(10, 2, 0.5, True, id="d10-larger-rate-duplicate"),
    pytest.param(6, 3, 0.05, False, id="d6-masks-duplicate"),
]


@pytest.mark.parametrize("likelihood", [False, True], ids=["least_squares", "likelihood"])
@pytest.mark.parametrize(("d", "seed", "rate", "duplicate"), CENTRAL_CASES)
def test_the_central_path_gives_the_bits_of_calling_the_full_functions(
    d, seed, rate, duplicate, likelihood, monkeypatch
) -> None:
    cov = covariance(d, seed, duplicate=duplicate)
    masks = {"forbidden": ((2, 3), (3, 2)), "required": ((0, 1),)} if seed == 3 else {}
    if likelihood:
        loss, penalty = gaussian_likelihood, quasi_mcp(0.4, 0.2)
    else:
        loss, penalty = least_squares, l1_penalty(0.03)
    undone = count_undone_steps(monkeypatch)
    fast = central(cov, loss, penalty, lr=rate, free_diagonal=not likelihood, **masks)
    undone_by_fast = undone[0]
    plain_loss, plain_penalty = plain(loss, penalty)
    reference = central(
        cov, plain_loss, plain_penalty, lr=rate, free_diagonal=not likelihood, **masks
    )
    assert same(fast, reference)
    assert undone[0] == 2 * undone_by_fast  # both runs took the same steps


def test_the_central_path_enters_the_undo_branch_in_the_cases_above(monkeypatch) -> None:
    undone = count_undone_steps(monkeypatch)
    for d, seed, rate, duplicate in (case.values for case in CENTRAL_CASES):
        central(
            covariance(d, seed, duplicate=duplicate),
            least_squares,
            l1_penalty(0.03),
            lr=rate,
            free_diagonal=True,
        )
    assert undone[0] > 0


def adam(cov, loss, penalty, *, num_steps, forbidden=None, required=None):
    return adam_augmented_lagrangian(
        canonical(cov, forbidden, required),
        loss=loss,
        penalty=penalty,
        start=np.zeros_like(cov),
        max_iter=3,
        h_tol=1e-8,
        rho_max=1e8,
        lr=3e-4,
        num_steps=num_steps,
    )


@pytest.mark.parametrize(
    ("d", "seed", "duplicate", "num_steps", "masks"),
    [
        pytest.param(5, 0, False, 1200, {}, id="d5"),
        pytest.param(6, 1, True, 400, {}, id="d6-duplicate"),
        pytest.param(
            6, 2, False, 400, {"forbidden": ((2, 3),), "required": ((0, 1),)}, id="d6-masks"
        ),
    ],
)
def test_the_adam_solver_gives_the_bits_of_calling_the_full_functions(
    d, seed, duplicate, num_steps, masks
) -> None:
    cov = covariance(d, seed, duplicate=duplicate)
    penalty = quasi_mcp(0.4, 0.2)
    fast = adam(cov, gaussian_likelihood, penalty, num_steps=num_steps, **masks)
    plain_loss, plain_penalty = plain(gaussian_likelihood, penalty)
    reference = adam(cov, plain_loss, plain_penalty, num_steps=num_steps, **masks)
    assert same(fast, reference)
    assert np.abs(fast).max() > 0


# --------------------------------------------------------------------------------------------
# SciPy's kernels of the matrix exponential
# --------------------------------------------------------------------------------------------

SCIPY_HAS_KERNELS = tuple(int(part) for part in scipy.__version__.split(".")[:2]) >= (1, 15)
needs_kernels = pytest.mark.skipif(not SCIPY_HAS_KERNELS, reason="SciPy before 1.15")


def test_the_kernels_are_in_use_from_scipy_1_15_and_not_before() -> None:
    assert (continuous._KERNELS is not None) == SCIPY_HAS_KERNELS


def squarings(a: np.ndarray) -> int:
    """How many times ``scipy.linalg.expm`` squares its Pade approximant of ``a``."""
    kernels = continuous._KERNELS
    assert kernels is not None
    work = np.empty((5, *a.shape))
    work[0] = a
    return kernels.pick_pade_structure(work)[1]


@needs_kernels
def test_the_self_check_matrices_cover_the_exponential_with_and_without_squaring() -> None:
    assert [squarings(a) > 0 for a in continuous._SELF_CHECK] == [True, True, False]


def generic_matrices() -> list[np.ndarray]:
    """``W * W`` of random weights at sizes 2 to 40, with entries from rounding noise to large, and
    some of them zero.
    """
    rng = np.random.default_rng(12345)
    out = []
    for d in (2, 3, 5, 10, 20, 40):
        for _ in range(20):
            weights = rng.standard_normal((d, d)) * 10.0 ** rng.uniform(-8, 1.5)
            weights[rng.random((d, d)) < rng.random()] = 0.0
            np.fill_diagonal(weights, 0.0)
            out.append(weights * weights)
    return out


@needs_kernels
def test_the_kernels_give_the_bits_of_scipys_expm_with_and_without_squaring() -> None:
    kernels = continuous._KERNELS
    assert kernels is not None
    seen = set()
    for a in generic_matrices():
        result = continuous._kernel_expm(a, kernels)
        if result is None:  # a triangular matrix, which a zero entry in each row of ``a`` can make
            assert 0 in kernels.bandwidth(a)
            continue
        assert same(result, scipy.linalg.expm(a))
        seen.add(min(squarings(a), 1))
    assert seen == {0, 1}


def special_matrices() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    dense = rng.uniform(0.0, 1.0, (6, 6))
    infinite, undefined = dense.copy(), dense.copy()
    infinite[0, 1], undefined[1, 0] = np.inf, np.nan
    return {
        "upper": np.triu(4 * dense),
        "lower": np.tril(4 * dense),
        "diagonal": np.diag(np.arange(1.0, 7.0)),
        "zero": np.zeros((6, 6)),
        "scalar": np.array([[2.0]]),
        "empty": np.zeros((0, 0)),
        "huge": dense * 1e3,
        "tiny": dense * 1e-160,
        "infinite": infinite,
        "undefined": undefined,
        "float32": dense.astype(np.float32),
        "integer": np.arange(9).reshape(3, 3),
        "fortran": np.asfortranarray(dense),
        "strided": np.tile(dense, (2, 2))[::2, ::2],
        "batch": np.stack([dense, 2 * dense]),
    }


@pytest.mark.parametrize("name", list(special_matrices()))
def test_the_exponential_equals_scipys_on_any_array(name) -> None:
    a = special_matrices()[name]
    assert same(continuous._expm(a), scipy.linalg.expm(a))


@needs_kernels
@pytest.mark.parametrize(
    "name", ["upper", "lower", "diagonal", "zero", "scalar", "empty", "float32", "integer", "batch"]
)
def test_the_kernels_leave_every_other_input_to_scipy(name) -> None:
    assert continuous._KERNELS is not None
    assert continuous._kernel_expm(special_matrices()[name], continuous._KERNELS) is None


def test_the_exponential_of_a_non_square_array_raises_scipys_error() -> None:
    with pytest.raises(scipy.linalg.LinAlgError):
        continuous._expm(np.ones((3, 4)))


@needs_kernels
def test_a_kernel_error_code_leaves_the_exponential_to_scipy() -> None:
    kernels = continuous._KERNELS
    assert kernels is not None
    a = continuous._SELF_CHECK[0]
    no_memory = kernels._replace(pick_pade_structure=lambda work: (-1, 0))
    assert continuous._kernel_expm(a, no_memory) is None
    lapack_error = kernels._replace(pade_uv_calc=lambda work, order: -7)
    assert continuous._kernel_expm(a, lapack_error) is None


@needs_kernels
def test_the_trace_exponential_runs_the_kernels_and_not_scipys_expm(monkeypatch) -> None:
    calls = []
    real = scipy.linalg.expm
    monkeypatch.setattr(scipy.linalg, "expm", lambda a: calls.append(1) or real(a))
    weights = np.random.default_rng(0).uniform(-1, 1, (6, 6))
    continuous.trace_exponential(weights)
    assert calls == []


def test_forcing_the_fallback_gives_the_same_bits_through_scipys_expm(monkeypatch) -> None:
    weights = np.random.default_rng(1).uniform(-2, 2, (8, 8))
    x = np.random.default_rng(2).standard_normal((200, 5))
    x[:, 1] += x[:, 0]
    x[:, 3] -= x[:, 1]
    series = np.random.default_rng(3).standard_normal((200, 4))
    series[1:, 2] += 0.7 * series[:-1, 0]
    value, grad = continuous.trace_exponential(weights)
    notears = andrey.notears(x).weighted_adjacency
    dynotears = andrey.dynotears(series).structure.lag_weights
    calls = []
    real = scipy.linalg.expm
    monkeypatch.setattr(continuous, "_KERNELS", None)
    monkeypatch.setattr(scipy.linalg, "expm", lambda a: calls.append(1) or real(a))
    forced_value, forced_grad = continuous.trace_exponential(weights)
    assert calls == [1]
    assert forced_value.hex() == value.hex()
    assert same(forced_grad, grad)
    calls.clear()
    assert same(andrey.notears(x).weighted_adjacency, notears)
    assert same(andrey.dynotears(series).structure.lag_weights, dynotears)
    assert calls


def test_scipy_before_1_15_keeps_scipys_expm(monkeypatch) -> None:
    monkeypatch.setattr(scipy, "__version__", "1.14.1")
    assert continuous._load_kernels() is None
    monkeypatch.setattr(scipy, "__version__", "unreleased")
    assert continuous._load_kernels() is None


def test_kernels_with_other_arguments_fail_the_self_check(monkeypatch) -> None:
    """SciPy 1.13 and 1.14 take ``pade_UV_calc(Am, n, m)`` and scale in Python: used as 1.15's,
    the call raises, and the kernels stay off even when the version says otherwise.
    """
    module = pytest.importorskip("scipy.linalg._matfuncs_expm")

    def old_pade_uv_calc(work: np.ndarray, size: int, order: int) -> None:
        raise AssertionError("not reached with two arguments")

    monkeypatch.setattr(scipy, "__version__", "1.99.0")
    monkeypatch.setattr(module, "pade_UV_calc", old_pade_uv_calc)
    assert continuous._load_kernels() is None


@needs_kernels
def test_kernels_that_differ_from_scipys_expm_fail_the_self_check(monkeypatch) -> None:
    module = pytest.importorskip("scipy.linalg._matfuncs_expm")
    real = module.pick_pade_structure

    def one_squaring_more(work: np.ndarray) -> tuple[int, int]:
        order, squarings = real(work)
        return order, squarings + 1

    monkeypatch.setattr(module, "pick_pade_structure", one_squaring_more)
    assert continuous._load_kernels() is None
    monkeypatch.undo()
    assert continuous._load_kernels() is not None
