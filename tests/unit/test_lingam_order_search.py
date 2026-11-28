"""The LiNGAM order search's skew signs and residualization return the bits of the expressions they
replace.

``_skew_signs`` gives ``np.where(np.mean(x**3, axis=0) < 0, -1.0, 1.0)`` from the mean of
``x * x * x``, and ``residualize`` gives the loop of :func:`~andrey.lingam.pwling.compute_residual`
over the remaining columns. Each is pinned against its expression on inputs that reach the guard of
the signs: columns whose third moment is zero or near it, where the sign of a rounded mean is
decided by the rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.lingam import direct, multi_group, pwling


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether ``a`` and ``b`` have the same shape and the same bytes, NaN and signed zeros too."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _signs_by_the_expression(x: np.ndarray) -> np.ndarray:
    return np.where(np.mean(x**3, axis=0) < 0, -1.0, 1.0)


def _residualize_by_the_loop(X: np.ndarray, remaining: list[int], m: int) -> None:
    for i in remaining:
        X[:, i] = pwling.compute_residual(X[:, i], X[:, m])


def _linear_sem(seed: int, d: int, n: int, noise: str) -> np.ndarray:
    """A linear model on a random DAG with ``d`` variables, whose columns are shuffled."""
    rng = np.random.default_rng(seed)
    weights = np.tril((rng.random((d, d)) < 0.3) * rng.uniform(0.5, 1.5, (d, d)), -1)
    weights *= rng.choice([-1.0, 1.0], (d, d))
    if noise == "exponential":
        errors = rng.exponential(1.0, (n, d)) * rng.choice([-1.0, 1.0], d)
    elif noise == "uniform":
        errors = rng.uniform(-1.0, 1.0, (n, d))
    else:
        errors = rng.laplace(0.0, 1.0, (n, d))
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = x @ weights[j] + errors[:, j]
    return x[:, rng.permutation(d)]


def _third_moment_zero_columns(seed: int, n: int = 600) -> dict[str, np.ndarray]:
    """Inputs whose columns have a third moment of zero or near it, as the signs' guard decides."""
    rng = np.random.default_rng(seed)
    half = n // 2
    magnitudes = rng.exponential(1.0, half)
    mirrored = np.concatenate([magnitudes, -magnitudes])
    shuffled = rng.permutation(mirrored)
    binary = rng.permutation(np.repeat([-1.0, 1.0], half))
    three_levels = rng.permutation(np.resize(np.array([-1.0, 0.0, 1.0]), n))
    levels = np.arange(-3, 4, dtype=np.float64).repeat(n // 7 + 1)[:half]
    integers = rng.permutation(np.concatenate([levels, -levels]))
    skewed = rng.exponential(1.0, n) - 1.0
    uniform = rng.uniform(-1.0, 1.0, half)
    child = 0.7 * shuffled + rng.permutation(np.concatenate([uniform, -uniform]))
    laplace = rng.laplace(0.0, 1.0, half)
    symmetric_child = 0.6 * mirrored + np.concatenate([laplace, -laplace])
    return {
        "mirrored": np.column_stack(
            [mirrored, shuffled, skewed, 0.8 * skewed + rng.exponential(1.0, n), child]
        ),
        "discrete": np.column_stack(
            [
                binary,
                three_levels,
                integers,
                skewed,
                0.5 * binary + 0.4 * skewed + rng.uniform(-1.0, 1.0, n),
            ]
        ),
        "symmetric_child": np.column_stack(
            [mirrored, symmetric_child, 0.5 * symmetric_child + rng.exponential(1.0, n), skewed]
        ),
        "copies": np.column_stack(
            [
                skewed,
                2.0 * skewed,
                -skewed,
                mirrored,
                -mirrored,
                rng.exponential(1.0, n) + 0.3 * mirrored,
            ]
        ),
    }


def _standardized(x: np.ndarray) -> np.ndarray:
    return pwling._standardized(x, list(range(x.shape[1])))


def test_signs_are_the_expressions_on_inputs_near_a_zero_third_moment() -> None:
    """On every input, mirrored or discrete columns included, the signs are the original ones."""
    differ_without_the_guard = 0
    for seed in range(20):
        for name, raw in _third_moment_zero_columns(seed).items():
            x = _standardized(raw)
            want = _signs_by_the_expression(x)
            assert _same_bits(pwling._skew_signs(x), want), (seed, name)
            unguarded = np.where(np.mean(x * x * x, axis=0) < 0, -1.0, 1.0)
            differ_without_the_guard += int(np.sum(unguarded != want))
    # The guard is reached: the mean of x * x * x alone gives another sign on some columns.
    assert differ_without_the_guard > 0


@pytest.mark.parametrize("noise", ["exponential", "uniform", "laplace"])
def test_signs_are_the_expressions_on_random_columns(noise: str) -> None:
    for seed in range(5):
        x = _standardized(_linear_sem(seed, 12, 400, noise))
        assert _same_bits(pwling._skew_signs(x), _signs_by_the_expression(x))
        assert _same_bits(pwling._skew_signs(x[:, ::-1]), _signs_by_the_expression(x[:, ::-1]))
        c_ordered = np.ascontiguousarray(x)
        assert _same_bits(pwling._skew_signs(c_ordered), _signs_by_the_expression(c_ordered))


def test_a_nan_column_takes_the_expression() -> None:
    x = _standardized(_linear_sem(0, 6, 300, "exponential"))
    x[3, 2] = np.nan
    want = _signs_by_the_expression(x)
    assert _same_bits(pwling._skew_signs(x), want)
    assert want[2] == 1.0


@pytest.mark.parametrize("n", [20, 50, 1000])
def test_residualize_is_the_residual_loop(n: int) -> None:
    """Every step of the search leaves the residualized columns with the loop's bits."""
    rng = np.random.default_rng(n)
    for order_of_memory in ("C", "F"):
        raw = _linear_sem(n, 9, n, "exponential") if n > 9 else rng.normal(size=(n, 9))
        X = np.array(raw, dtype=np.float64, order=order_of_memory)
        want = X.copy(order=order_of_memory)
        remaining = list(range(9))
        while len(remaining) > 1:
            m = remaining.pop(int(rng.integers(len(remaining))))
            pwling.residualize(X, remaining, m)
            _residualize_by_the_loop(want, remaining, m)
            assert _same_bits(X, want), (n, order_of_memory, m)


@pytest.mark.parametrize("n", [7, 300, 1000])
def test_center_columns_has_the_bits_of_each_columns_own_mean(n: int) -> None:
    """Whatever the memory layout of the input, each column loses the mean of its contiguous copy,
    where ``X.mean(axis=0)`` of a C-ordered array sums the rows one after the other."""
    rng = np.random.default_rng(n)
    raw = rng.standard_normal((n, 9)) * 3 + rng.choice([0.0, 1e3, 0.37], 9)
    want = np.column_stack([c - np.ascontiguousarray(c).mean() for c in raw.T])
    layouts = {
        "C": np.ascontiguousarray(raw),
        "F": np.asfortranarray(raw),
        "fancy index": raw[:, list(range(9))],
    }
    for name, X in layouts.items():
        pwling.center_columns(X)
        assert _same_bits(X, want), name


def test_residualize_computes_nothing_when_no_column_remains() -> None:
    """The last pick has nothing to residualize, as the empty loop did no arithmetic."""
    X = np.array([[0.0, 1.0], [1e308, 2.0]])
    before = X.copy()
    with np.errstate(all="raise"):
        pwling.residualize(X, [], 0)
    assert _same_bits(X, before)


def test_residualize_leaves_the_other_columns_alone() -> None:
    X = _linear_sem(1, 6, 200, "uniform")
    before = X.copy()
    pwling.residualize(X, [0, 2, 3], 4)
    assert _same_bits(X[:, [1, 4, 5]], before[:, [1, 4, 5]])


class _Calls:
    """Counts the calls of the search's two replaced steps, each checked against its expression."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.signs = 0
        self.residualizations = 0
        signs, residualize = pwling._skew_signs, pwling.residualize

        def checked_signs(x: np.ndarray) -> np.ndarray:
            self.signs += 1
            got = signs(x)
            assert _same_bits(got, _signs_by_the_expression(x))
            return got

        def checked_residualize(X: np.ndarray, remaining: list[int], m: int) -> None:
            self.residualizations += 1
            want = X.copy(order="K")
            _residualize_by_the_loop(want, remaining, m)
            residualize(X, remaining, m)
            assert _same_bits(X, want)

        monkeypatch.setattr(pwling, "_skew_signs", checked_signs)
        monkeypatch.setattr(pwling, "residualize", checked_residualize)


def _fits(measure: str) -> list[tuple[list[int], list[np.ndarray]]]:
    """Fits of both searches on linear models and on inputs with a zero third moment."""
    inputs = [
        _linear_sem(s, d, n, noise)
        for s in range(2)
        for d, n in ((8, 300), (20, 600))
        for noise in ("exponential", "uniform", "laplace")
    ]
    inputs += [raw for s in range(2) for raw in _third_moment_zero_columns(s).values()]
    out = []
    for X in inputs:
        order, weights = direct.direct_lingam(X, measure=measure)
        out.append((order, [weights]))
    for s in range(2):
        for noise in ("exponential", "laplace"):
            groups = [_linear_sem(s, 10, 400, noise), _linear_sem(s + 100, 10, 250, noise)]
            out.append(multi_group.multi_group_direct_lingam(groups, measure=measure))
    return out


@pytest.mark.parametrize("measure", ["skew", "tanh", "pwling"])
def test_every_step_of_the_order_searches_has_the_expressions_bits(
    measure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each sign and residualization of both searches, single and multi-group, is checked."""
    calls = _Calls(monkeypatch)
    _fits(measure)
    assert calls.residualizations > 0
    assert (calls.signs > 0) == (measure == "skew")


@pytest.mark.parametrize("measure", ["skew", "tanh", "pwling"])
def test_the_searches_return_the_orders_of_the_original_expressions(
    measure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With both steps put back to their expressions, the same orders and weights come out."""
    fast = _fits(measure)
    monkeypatch.setattr(pwling, "_skew_signs", _signs_by_the_expression)
    monkeypatch.setattr(pwling, "residualize", _residualize_by_the_loop)
    original = _fits(measure)
    for (order, weights), (want_order, want_weights) in zip(fast, original, strict=True):
        assert order == want_order
        assert all(_same_bits(w, v) for w, v in zip(weights, want_weights, strict=True))
