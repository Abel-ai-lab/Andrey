"""``least_squares`` returns the bits of ``LinearRegression`` on the unit-scale predictors.

For a finite float64 design with a matching 1-D or 2-D ``y``, ``least_squares`` solves with the
steps of ``LinearRegression.fit`` and never builds the estimator. It is pinned to the estimator in
the coefficients and the intercept, down to their shapes, dtypes, and memory layout, on the cases
the LiNGAM methods and RCD reach and on the ones that exercise its rank cutoff, and every other
input keeps the estimator's error.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import lstsq
from sklearn.linear_model import LinearRegression

from andrey.core.units import _RANK_CUTOFF, least_squares, unit_factors


def reference(x, y):
    """The fit ``least_squares`` replaces: ``LinearRegression`` on the unit-scale predictors."""
    factors = unit_factors(x)
    fit = LinearRegression().fit(x * factors, y)
    return fit.coef_ * factors, np.asarray(fit.intercept_)


def assert_same(x, y) -> None:
    got, want = least_squares(x, y), reference(x, y)
    for name, a, b in zip(("coef", "intercept"), got, want, strict=True):
        assert a.shape == b.shape, name
        assert a.dtype == b.dtype, name
        assert a.tobytes() == b.tobytes(), name
        assert (a.flags.c_contiguous, a.flags.f_contiguous) == (
            b.flags.c_contiguous,
            b.flags.f_contiguous,
        ), name


def data(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d))


def test_the_rank_cutoff_is_linear_regressions_tol() -> None:
    assert LinearRegression().tol == _RANK_CUTOFF


@pytest.mark.parametrize("n", [1, 2, 5, 9, 60, 500])
@pytest.mark.parametrize("p", [1, 2, 5, 8])
@pytest.mark.parametrize("y_columns", [None, 1, 3])
def test_one_and_two_dimensional_targets(n: int, p: int, y_columns: int | None) -> None:
    """A 1-D ``y`` gives a vector and a scalar intercept, a 2-D one a row of ``coef`` per column."""
    X = data(n, 12, seed=n + p)
    y = X[:, 11] if y_columns is None else X[:, 8 : 8 + y_columns]
    assert_same(X[:, :p], y)


def test_a_sample_smaller_than_the_design() -> None:
    """An under-determined fit takes the minimum-norm solution both ways."""
    X = data(4, 9, seed=3)
    assert_same(X[:, :7], X[:, 8])


@pytest.mark.parametrize("seed", range(6))
def test_columns_on_different_scales(seed: int) -> None:
    """Raw scales span ten orders of magnitude, among them exact powers of two."""
    X = data(300, 6, seed=seed)
    scales = np.array([2.0**20, 1.0, 2.0**-30, 1e5, 2e-4, 7.0])
    assert_same(X[:, :5] * scales[:5], X[:, 5] * scales[5])
    assert_same(X[:, :5] * scales[:5], X[:, 5:] * scales[5:])


@pytest.mark.parametrize("gap", [1e-3, 1e-5, 1e-7, 1e-9, 1e-12, 0.0])
def test_a_design_below_the_rank_cutoff(gap: float) -> None:
    """A near-copy column, the singular-value ratio from above to far below ``tol``, drops alike."""
    X = data(400, 5, seed=11)
    X[:, 2] = X[:, 0] + gap * X[:, 1]
    assert_same(X[:, :3], X[:, 3])
    assert_same(X[:, :3], X[:, 3:5])


def test_the_near_copy_case_exercises_the_cutoff() -> None:
    """Its smallest singular value lies below ``tol`` and ``cond=None`` would solve differently."""
    X = data(400, 5, seed=11)
    X[:, 2] = X[:, 0] + 1e-9 * X[:, 1]
    x, y = X[:, :3] * unit_factors(X[:, :3]), X[:, 3]
    centered = x - x.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    assert singular[-1] / singular[0] < _RANK_CUTOFF
    dropped = lstsq(centered, y - y.mean(), cond=_RANK_CUTOFF)[0]
    kept = lstsq(centered, y - y.mean(), cond=None)[0]
    assert not np.array_equal(dropped, kept)
    assert_same(X[:, :3], X[:, 3])


def test_integer_input() -> None:
    rng = np.random.default_rng(5)
    X = rng.integers(-40, 40, size=(200, 5))
    assert_same(X[:, :4], X[:, 4])
    assert_same(X[:, :4], X[:, 3:5])
    assert_same(X[:, :4].astype(np.float64), X[:, 4])


def test_float32_input() -> None:
    X = data(150, 5, seed=2).astype(np.float32)
    assert_same(X[:, :4], X[:, 4])
    assert_same(X[:, :4].astype(np.float64), X[:, 4])


@pytest.mark.parametrize("order", ["C", "F"])
def test_fancy_indexed_columns_and_strided_views(order: str) -> None:
    """Index lists, slices, and either memory order of the data give the estimator's bits."""
    X = np.asarray(data(250, 12, seed=7), order=order)
    assert_same(X[:, [4, 1, 6, 0]], X[:, 9])
    assert_same(X[:, [4, 1, 6, 0]], X[:, [9, 10]])
    assert_same(X[:, ::3], X[:, 1])
    assert_same(X[::2, :4], X[::2, 5:8])
    assert_same(X[np.random.default_rng(0).permutation(250), :][:, [2, 3]], X[:, 11])


def test_the_coefficients_map_back_to_the_data_units() -> None:
    """The fit predicts ``y`` from the raw columns: ``x @ coef.T + intercept``."""
    X = data(200, 4, seed=9) * np.array([1e4, 1.0, 1e-3, 3.0])
    y = X @ np.array([2e-4, -1.5, 300.0, 0.25]) + 7.0
    coef, intercept = least_squares(X, y)
    assert np.allclose(X @ coef + intercept, y, rtol=1e-9)


def test_a_plain_input_does_not_build_the_estimator(monkeypatch) -> None:
    """The direct path never calls ``LinearRegression``; any other input still does."""
    calls = []
    fit = LinearRegression.fit

    def counting(self, X, y, sample_weight=None):
        calls.append(np.shape(X))
        return fit(self, X, y, sample_weight)

    monkeypatch.setattr(LinearRegression, "fit", counting)
    X = data(50, 4)
    least_squares(X[:, :3], X[:, 3])
    least_squares(X[:, :3], X[:, 2:])
    assert calls == []
    with_nan = X[:, :3].copy()
    with_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        least_squares(with_nan, X[:, 3])
    assert len(calls) == 1


def refusal(function, *args) -> tuple[type, str]:
    with pytest.raises(Exception) as caught:
        function(*args)
    return type(caught.value), str(caught.value)


def invalid_inputs():
    X = data(30, 4, seed=1)
    with_nan = X[:, :3].copy()
    with_nan[7, 1] = np.nan
    with_inf = X[:, :3].copy()
    with_inf[3, 0] = np.inf
    y_nan = X[:, 3].copy()
    y_nan[5] = np.nan
    y_inf = X[:, 3:].copy()
    y_inf[0, 0] = -np.inf
    huge = np.full((30, 2), 1e308)
    huge[:, 1] = X[:, 0]
    return {
        "nan in x": (with_nan, X[:, 3]),
        "inf in x": (with_inf, X[:, 3]),
        "nan in y": (X[:, :3], y_nan),
        "inf in y": (X[:, :3], y_inf),
        "no columns": (X[:, :0], X[:, 3]),
        "no rows": (X[:0, :3], X[:0, 3]),
        "rows differ": (X[:, :3], X[:20, 3]),
        "three-dimensional y": (X[:, :3], X[:, 2:].reshape(30, 1, 2)),
        "zero-column y": (X[:, :3], X[:, :0]),
        "overflow on centering": (huge, X[:, 3]),
        "object y": (X[:, :3], X[:, 3].astype(object)),
        "complex y": (X[:, :3], X[:, 3].astype(complex)),
        "string y": (X[:, :3], np.array(["a"] * 30)),
        "masked y": (X[:, :3], np.ma.masked_array(X[:, 3], mask=np.arange(30) == 4)),
        "list y": (X[:, :3], X[:, 3].tolist()),
    }


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("name", list(invalid_inputs()))
def test_other_inputs_keep_the_estimators_errors_and_values(name: str) -> None:
    x, y = invalid_inputs()[name]
    try:
        want = reference(x, y)
    except Exception:
        assert refusal(least_squares, x, y) == refusal(reference, x, y)
    else:
        got = least_squares(x, y)
        assert all(a.tobytes() == b.tobytes() for a, b in zip(got, want, strict=True))


def test_a_boolean_target_is_the_estimators() -> None:
    X = data(40, 3, seed=4)
    assert_same(X[:, :2], X[:, 2] > 0)
