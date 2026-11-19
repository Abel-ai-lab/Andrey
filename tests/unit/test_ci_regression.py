"""The regression-based tests GCM, wGCM, and PCM: statistics on fixed inputs, size and power on
fixed cases, and p-values that depend on neither the row, column, nor argument order."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr

from andrey.core import ci
from andrey.core import ci_regression as cr

N = 300


def _chain(n: int = N, seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with nonlinear links, and an independent ``x3``."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = x0 + np.sin(2 * x0) + 0.3 * rng.standard_normal(n)
    x2 = np.tanh(x1) + 0.3 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2, rng.standard_normal(n)])


def _sign_changing(n: int = 400, seed: int = 0) -> np.ndarray:
    """``x`` and ``y`` share noise with the sign of ``z``: a conditional covariance that averages
    to zero over ``z``, which the GCM cannot see and a weight on ``z`` can."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    shared = rng.standard_normal(n)
    x = np.sign(z) * shared + 0.5 * rng.standard_normal(n)
    y = shared + 0.5 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


# Linear regressions keep these fast; the default regressions are checked further down.
LINEAR = {
    "gcm": {"regression": "linear"},
    "wgcm": {"regression": "linear"},
    "wgcm_fix": {"regression": "linear", "variant": "fix"},
    "pcm": {"regression": "linear"},
}
CLASSES = {"gcm": cr.GCM, "wgcm": cr.WGCM, "wgcm_fix": cr.WGCM, "pcm": cr.PCM}


def _test(name: str, data: np.ndarray, **options) -> cr._RegressionTest:
    return CLASSES[name](data, **{**LINEAR[name], **options})


def test_the_tests_are_registered_and_seeded():
    for name in ("gcm", "wgcm", "pcm"):
        assert name in ci.SHIPPED_INDEP_TESTS
        assert name in ci.STOCHASTIC_INDEP_TESTS
        assert name in ci.CONTINUOUS_INDEP_TESTS
        assert name in ci.EXPERIMENTAL_INDEP_TESTS
    assert isinstance(ci.make_indep_test("wgcm", _chain(60), variant="fix"), cr.WGCM)


def test_gcm_statistic_on_fixed_residuals():
    rx = np.array([1.0, -1.0, 2.0, 0.5])
    ry = np.array([0.5, -2.0, 1.0, 1.0])
    products = rx * ry  # 0.5, 2, 2, 0.5: mean 1.25, variance 0.5625
    statistic, p = cr._gcm_answer(rx, ry)
    assert statistic == pytest.approx(2.0 * 1.25 / 0.75)
    assert p == pytest.approx(2.0 * ndtr(-statistic))
    assert products.mean() == 1.25


def test_a_covariance_without_noise_is_a_dependence():
    x = np.repeat([-1.0, 1.0], 30)
    data = np.column_stack([x, x, np.random.default_rng(0).standard_normal(60)])
    for test in (cr.GCM(data, regression="linear"), cr.WGCM(data, regression="linear")):
        assert test(0, 1, []) == 0.0
    statistic, p = cr._fixed_answer(np.ones(8), np.arange(8.0)[:, None], np.ones((8, 9)))
    assert statistic == np.inf
    assert p == 0.1


def test_pcm_needs_rows_for_a_split_in_each_half():
    with pytest.raises(ValueError, match="10"):
        cr.PCM(_chain(9))
    assert 0.0 <= cr.PCM(_chain(10), n_splits=1)(0, 1, []) <= 1.0


def test_wgcm_est_needs_rows_for_a_split_in_its_estimation_part():
    with pytest.raises(ValueError, match="14"):
        cr.WGCM(_chain(13))
    assert 0.0 <= cr.WGCM(_chain(14), regression="linear")(0, 1, [2]) <= 1.0
    assert 0.0 <= cr.WGCM(_chain(4), regression="linear", variant="fix")(0, 1, [2]) <= 1.0


def test_one_sided_statistics_report_zero_below_the_null():
    statistic, p = cr._estimated_answer(np.array([1.0, -2.0, -1.0, 0.5]), np.ones(4))
    assert statistic == 0.0
    assert p > 0.5


def test_fixed_weights_on_fixed_input():
    products = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    z = np.arange(8.0)[:, None]
    # With every multiplier 2, each draw is twice the statistic, so all 9 draws exceed it.
    statistic, p = cr._fixed_answer(products, z, np.full((8, 9), 2.0))
    assert statistic > 0.0
    assert p == 1.0


@pytest.mark.parametrize("name", LINEAR)
def test_size_and_power_on_the_chain(name):
    test = _test(name, _chain())
    assert test(0, 2, [1]) > 0.05  # x0 and x2 are independent given x1
    assert test(0, 3, []) > 0.05  # x3 is independent of everything
    assert test(0, 1, []) < 0.01
    assert test(0, 2, []) < 0.01


@pytest.mark.parametrize("name", LINEAR)
def test_p_values_ignore_row_column_and_argument_order(name):
    x = _chain(120)
    rows = np.random.default_rng(1).permutation(len(x))
    cols = [2, 0, 3, 1]  # data column j sits at position cols.index(j)
    test, shuffled = _test(name, x, seed=3), _test(name, x[rows][:, cols], seed=3)
    at = cols.index
    for a, b, s in [(0, 2, [1]), (1, 3, [0, 2]), (0, 1, [])]:
        p = test(a, b, s)
        assert test(b, a, s[::-1]) == p
        assert shuffled(at(a), at(b), [at(c) for c in s]) == p


@pytest.mark.parametrize("name", LINEAR)
def test_p_values_ignore_the_units(name):
    x = _chain(120)
    scaled = x * np.array([4.0, 0.25, 8.0, 2.0]) + np.array([3.0, -1.0, 0.0, 5.0])
    test, other = _test(name, x, seed=3), _test(name, scaled, seed=3)
    for query in [(0, 2, [1]), (1, 3, [0, 2])]:
        assert other(*query) == pytest.approx(test(*query), abs=1e-9)


@pytest.mark.parametrize("name", LINEAR)
def test_a_constant_variable_is_independent(name):
    x = _chain(60)
    x[:, 3] = 2.0
    assert _test(name, x)(0, 3, [1]) == 1.0
    assert _test(name, x)(3, 2, []) == 1.0


def test_wgcm_detects_a_covariance_whose_sign_follows_the_conditioning_set():
    x = _sign_changing()
    assert cr.GCM(x, regression="linear")(0, 1, [2]) > 0.05
    assert cr.WGCM(x, regression="linear")(0, 1, [2]) < 0.01
    assert cr.WGCM(x, regression="linear", variant="fix")(0, 1, [2]) < 0.01


def test_pcm_detects_an_interaction_the_gcm_misses():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = x * z + 0.5 * rng.standard_normal(n)  # E[y | x, z] depends on x, E[xy | z] = 0
    data = np.column_stack([x, y, z])
    assert cr.GCM(data, regression="linear")(0, 1, [2]) > 0.05
    assert cr.PCM(data, regression="forest", n_splits=1)(0, 1, [2]) < 0.01


def test_default_regressions_on_the_chain():
    x = _chain(150)
    for test in (cr.GCM(x), cr.WGCM(x, variant="fix")):
        assert test(0, 2, [1]) > 0.05
        assert test(0, 1, [2]) < 0.01


def test_early_stopping_keeps_ten_rounds_past_the_best():
    curve = np.array([5.0, 4.0, 3.0, *([3.5] * 20)])
    cut = cr._early_stopped(curve)
    assert np.isfinite(cut[:13]).all()
    assert np.isinf(cut[13:]).all()
    assert np.array_equal(cr._early_stopped(curve[:12]), curve[:12])


def test_variance_shift_brings_the_mean_ratio_to_one():
    squares = np.array([1.0, 4.0, 9.0])
    fitted = np.array([0.0, 1.0, 1.0])
    shift = cr._variance_shift(squares, fitted)
    assert np.mean(squares / (fitted + shift)) == pytest.approx(1.0)
    assert cr._variance_shift(squares, squares * 2) == 0.0


@pytest.mark.parametrize(
    ("cls", "options", "match"),
    [
        (cr.GCM, {"regression": "spline"}, "regression"),
        (cr.WGCM, {"variant": "both"}, "variant"),
        (cr.PCM, {"n_splits": 0}, "n_splits"),
    ],
)
def test_invalid_settings_raise(cls, options, match):
    with pytest.raises(ValueError, match=match):
        cls(_chain(20), **options)
