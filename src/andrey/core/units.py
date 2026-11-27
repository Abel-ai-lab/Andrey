"""Unit-scale columns for the numerical steps that would otherwise read the data's units.

A method's result depends on the data's units only where its model does. A few numerical steps
reach the units through a shortcut instead: a rank cutoff relative to the largest singular value
(least squares, a pseudo-inverse), an optimizer's start and tolerances, or an estimator defined only
up to the columns' scale. Each such step runs on unit-scale columns, the data's columns multiplied
by their :func:`unit_factors`, and its results are mapped back to the data's units: a coefficient
on a scaled predictor is multiplied by the predictor's factor, and a value in a scaled column's
units is divided by the column's factor.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq

#: ``LinearRegression``'s default ``tol``, which it passes to ``scipy.linalg.lstsq`` as ``cond``
#: (scikit-learn 1.9 and later) so that singular values below it, relative to the largest, drop.
_RANK_CUTOFF = 1e-6


def unit_factors(x: np.ndarray, *, standardize: bool = False) -> np.ndarray:
    """The factor of each column of ``x`` that puts the column on unit scale.

    By default each factor is the power of two that puts the column's standard deviation in
    ``[2**-0.5, 2**0.5)``. A product with a power of two is exact, so results map back exactly; a
    column already in that range, such as a standardized one, keeps the factor ``1`` and its bits;
    and a column rescaled by any power of two gives the same scaled column. With
    ``standardize=True`` each factor is one over the standard deviation, for a step that still
    reads the scale within a factor of two, such as an optimizer: the scaled column is then the
    same, up to rounding, for any positive unit. A constant column keeps the factor ``1``.
    """
    sd = np.std(x, axis=0)
    usable = np.isfinite(sd) & (sd > 0)
    if standardize:
        return np.where(usable, 1.0 / np.where(usable, sd, 1.0), 1.0)
    mantissa, exponent = np.frexp(sd)
    exponent = np.where(mantissa < 2.0**-0.5, exponent - 1, exponent)
    return np.where(usable, np.ldexp(1.0, -exponent), 1.0)


def least_squares(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary least squares of ``y`` on ``x`` with an intercept: ``(coef, intercept)``.

    scikit-learn's ``LinearRegression`` drops the directions whose singular value lies below its
    ``tol`` relative to the largest, so on raw columns it can drop a predictor whose scale lies far
    from the others'. The fit runs on the unit-scale predictors (:func:`unit_factors`) and the
    coefficients are mapped back. A ``y`` with several columns gives one row of ``coef`` and one
    ``intercept`` per column. The fitted values are ``x @ coef.T + intercept``.

    A finite float64 design with at least one row and one column, and a ``y`` of one or two
    dimensions with the same rows, is solved with the steps of ``LinearRegression.fit``: centering
    by the means, ``scipy.linalg.lstsq`` with ``cond`` set to its ``tol``, and
    ``intercept = y_offset - x_offset @ coef``, which gives its bits without its input checks.
    Any other input goes through ``LinearRegression``, so its errors stay.
    """
    factors = unit_factors(x)
    scaled = x * factors
    target = _float_target(scaled, y)
    if target is None:
        from sklearn.linear_model import LinearRegression

        fit = LinearRegression().fit(scaled, y)
        return fit.coef_ * factors, np.asarray(fit.intercept_)
    x_offset = np.average(scaled, axis=0)
    y_offset = np.average(target, axis=0)
    coef = lstsq(scaled - x_offset, target - y_offset, cond=_RANK_CUTOFF)[0].T
    if target.ndim == 1:
        return coef * factors, np.asarray(y_offset - x_offset @ coef)
    return coef * factors, np.asarray(y_offset - x_offset @ coef.T)


def _float_target(scaled: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """``y`` as float64 when :func:`least_squares` solves the pair directly, else ``None``.

    That is a finite float64 ``scaled`` with at least one row and one column, and a real-number
    ``y`` of one or two dimensions with the rows of ``scaled`` and, when two-dimensional, at least
    one column, whose float64 values are finite.
    """
    if not (
        isinstance(scaled, np.ndarray)
        and scaled.dtype == np.float64
        and scaled.ndim == 2
        and scaled.shape[0] >= 1
        and scaled.shape[1] >= 1
    ):
        return None
    y = np.asarray(y)
    if (
        y.dtype.kind not in "fiu"
        or y.ndim not in (1, 2)
        or y.shape[0] != scaled.shape[0]
        or y.size == 0
    ):
        return None
    target = np.array(y, dtype=np.float64)
    if not (np.isfinite(scaled).all() and np.isfinite(target).all()):
        return None
    return target
