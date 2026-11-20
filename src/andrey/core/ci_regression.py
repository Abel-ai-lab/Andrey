"""Regression-based conditional-independence tests: GCM, weighted GCM, and PCM.

Selected through ``indep_test="gcm"``, ``"wgcm"``, and ``"pcm"``. Each test regresses the two
variables, or one of them, on the conditioning set and tests the covariance of what the
regressions leave. The tests follow the :class:`~andrey.core.ci.CITest` protocol: built once over
the data and called as ``test(x, y, condition_set)``, returning a p-value where a larger value means
more independent.

The three share one data layout and one set of regressions. Each column is standardized, rounded
to 9 decimal places, and given the sign its sorted values fix, so the values the fits read do not
change with the units or the sign of a column. Each fit reads only the columns it uses: their
:func:`~andrey.core.ties.tie_ranks` order within that block gives the order of the features, and
the rows in lexicographic order of the block's values give the order of the rows. The sample
splits, cross-validation folds, and multiplier draws are drawn from ``seed`` over that row order,
so a query's p-value depends only on the values of its own columns: not on the row, the column, or
the argument order, not on the other columns, and not on which queries came before it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from andrey.core.ties import tie_ranks

from . import stats
from .ci import _canonical, per_query
from .seeding import resolve_seed

__all__ = ["GCM", "PCM", "WGCM"]

#: The regressions the tests offer: boosted regression trees with a cross-validated depth and
#: number of rounds, a random forest, and ordinary least squares.
REGRESSIONS: tuple[str, ...] = ("boosting", "forest", "linear")

# Boosted trees as the GCM authors' default: learning rate 0.3, 10-fold cross-validation over the
# tree depths 1, 3, 4, 5, 6 and up to 50 rounds, a fold's rounds cut 10 rounds after its best.
_BOOST_RATE = 0.3
_BOOST_DEPTHS = (1, 3, 4, 5, 6)
_BOOST_ROUNDS = 50
_BOOST_FOLDS = 10
_BOOST_PATIENCE = 10
# A random forest as the PCM authors': 500 trees, every predictor tried at each split, and nodes of
# 5 rows or fewer left unsplit. A node's size counts the distinct rows in it, where the authors'
# forest counts a row its bootstrap draws twice as two.
_FOREST_TREES = 500
_FOREST_MIN_SPLIT = 6
# The decimal places the standardized values keep, so that copies of a column in other units, whose
# standardized values differ in their last bits, give the same fits.
_DECIMALS = 9

Predictor = Callable[[np.ndarray], np.ndarray]


def _constant(value: float) -> Predictor:
    """A predictor that returns ``value`` for every row."""
    return lambda features: np.full(features.shape[0], value)


def _early_stopped(curve: np.ndarray) -> np.ndarray:
    """``curve`` with every round after the stop set to infinity.

    The rounds stop once ``_BOOST_PATIENCE`` rounds in a row have not lowered the best error.
    """
    best = 0
    for i in range(1, curve.size):
        if curve[i] < curve[best]:
            best = i
        elif i - best >= _BOOST_PATIENCE:
            out = curve.copy()
            out[i + 1 :] = np.inf
            return out
    return curve


def _boosting(depth: int, rounds: int) -> GradientBoostingRegressor:
    """Boosted regression trees of ``depth`` over ``rounds`` rounds."""
    return GradientBoostingRegressor(
        learning_rate=_BOOST_RATE,
        n_estimators=rounds,
        max_depth=depth,
        min_samples_leaf=1,
        random_state=0,  # orders the candidate features at each split; the columns are canonical
    )


def _fit_boosting(features: np.ndarray, target: np.ndarray, folds: np.ndarray) -> Predictor:
    """Boosted trees whose depth and number of rounds minimize the cross-validated error."""
    n_folds = int(folds.max()) + 1
    error = np.zeros((_BOOST_ROUNDS, len(_BOOST_DEPTHS)))
    for j, depth in enumerate(_BOOST_DEPTHS):
        for fold in range(n_folds):
            held = folds == fold
            model = _boosting(depth, _BOOST_ROUNDS).fit(features[~held], target[~held])
            curve = np.array(
                [
                    np.sqrt(np.mean((target[held] - p) ** 2))
                    for p in model.staged_predict(features[held])
                ]
            )
            error[:, j] += _early_stopped(curve)
    # The first minimum over the depths in turn, each over its rounds.
    j, rounds = divmod(int(np.argmin(error.T)), _BOOST_ROUNDS)
    return _boosting(_BOOST_DEPTHS[j], rounds + 1).fit(features, target).predict


def _fit_forest(features: np.ndarray, target: np.ndarray, seed: int) -> Predictor:
    """A random forest of ``_FOREST_TREES`` trees."""
    model = RandomForestRegressor(
        n_estimators=_FOREST_TREES,
        max_features=1.0,
        min_samples_split=_FOREST_MIN_SPLIT,
        random_state=seed,
    )
    return model.fit(features, target).predict


def _fit_linear(features: np.ndarray, target: np.ndarray) -> Predictor:
    """Ordinary least squares with an intercept."""
    design = np.column_stack([np.ones(len(target)), features])
    coef = np.linalg.lstsq(design, target, rcond=None)[0]
    return lambda new: coef[0] + new @ coef[1:]


def _canonical_columns(arr: np.ndarray) -> np.ndarray:
    """Each column of ``arr`` standardized, rounded to ``_DECIMALS`` places, and signed.

    The mean and the standard deviation are sums over the sorted values, so they do not change with
    the row order. Of a column and its negation, the one whose sorted values are larger at the first
    place they differ is kept; a column equal to its negation once sorted keeps its sign.
    """
    ordered = np.sort(arr, axis=0)
    with np.errstate(over="ignore", under="ignore"):  # either one raises below
        mean = ordered.mean(axis=0)
        std = np.sqrt(np.mean((ordered - mean) ** 2, axis=0))
    if not np.isfinite(std).all():
        raise ValueError("data values too large to standardize; rescale the columns")
    std[np.ptp(arr, axis=0) == 0] = np.inf  # identical values; their std is rounding, not 0
    if (std == 0).any():
        raise ValueError("data values too small to standardize; rescale the columns")
    z = np.round((arr - mean) / std, _DECIMALS)
    for j in range(z.shape[1]):
        low = np.sort(z[:, j])
        flipped = -low[::-1]
        differ = np.flatnonzero(low != flipped)
        if differ.size and flipped[differ[0]] > low[differ[0]]:
            z[:, j] = -z[:, j]
    return z


class _RegressionTest:
    """The data layout and the regressions the three tests share."""

    pin_blas = True  # read by pins_blas: the skeleton scans run it on one BLAS thread

    @staticmethod
    def least_rows(size: int, **options: object) -> int:
        """The fewest rows a query given ``size`` conditioning variables is tested on: ``4``, the
        rows each column needs to be standardized and regressed."""
        return 4

    def __init__(
        self, data: np.ndarray, *, regression: str, seed: int | None, name: str, min_rows: int = 4
    ) -> None:
        if regression not in REGRESSIONS:
            raise ValueError(f"regression must be one of {REGRESSIONS}, got {regression!r}")
        arr = stats.checked_matrix(data, f"run a {name} test", min_rows=min_rows)
        self._data = _canonical_columns(arr)
        self._n = arr.shape[0]
        self._regression = regression
        self._seed = resolve_seed(seed)
        self._layouts: dict[tuple[int, ...], tuple[np.ndarray, list[int]]] = {}
        self._residuals: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``."""
        return self._statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The statistics and p-values of many queries, one per row.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets.
        """
        return per_query(self._statistic_pvalue, x, y, conditions)

    def _statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int]
    ) -> tuple[float, float]:
        """The statistic and p-value of one query."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        if not (np.ptp(self._data[:, x_int]) > 0.0 and np.ptp(self._data[:, y_int]) > 0.0):
            return 0.0, 1.0  # a constant variable: no variation to depend on
        return self._answer(x_int, y_int, cond)

    def _answer(self, x: int, y: int, cond: list[int]) -> tuple[float, float]:
        raise NotImplementedError

    def _layout(self, columns: Iterable[int]) -> tuple[np.ndarray, list[int]]:
        """The row order and the column order in which a fit reads a block of columns.

        The columns are in their :func:`~andrey.core.ties.tie_ranks` order within the block, and
        the rows in lexicographic order of the block's values in that column order, so the layout
        depends on the block's values alone.
        """
        key = tuple(sorted(set(columns)))
        layout = self._layouts.get(key)
        if layout is None:
            cols = [key[i] for i in np.argsort(tie_ranks(self._data[:, key]), kind="stable")]
            layout = (np.lexsort(self._data[:, cols].T[::-1]), cols)
            self._layouts[key] = layout
        return layout

    def _full_residual(self, column: int, cond: list[int]) -> np.ndarray:
        """``column`` minus its regression on ``cond`` over every row, in the data's row order.

        The fit reads the layout of ``column`` and ``cond``, and the residual is kept for later
        queries.
        """
        key = (column, tuple(cond))
        residual = self._residuals.get(key)
        if residual is None:
            rows, cols = self._layout([column, *cond])
            block = self._data[rows]
            features = [c for c in cols if c != column]
            residual = np.empty(self._n)
            residual[rows] = self._residual(block[:, column], block[:, features])
            self._residuals[key] = residual
        return residual

    def _fit(self, features: np.ndarray, target: np.ndarray) -> Predictor:
        """The regression of ``target`` on ``features``; no features gives the mean."""
        m = len(target)
        if features.shape[1] == 0 or m < 4:
            return _constant(float(np.mean(target)))
        if self._regression == "linear":
            return _fit_linear(features, target)
        if self._regression == "forest":
            return _fit_forest(features, target, self._seed % 2**32)
        n_folds = min(_BOOST_FOLDS, m)
        perm = np.random.default_rng([self._seed, m]).permutation(m)
        return _fit_boosting(features, target, (np.arange(m) % n_folds)[perm])

    def _residual(self, target: np.ndarray, features: np.ndarray) -> np.ndarray:
        """``target`` minus its in-sample regression on ``features``."""
        return target - self._fit(features, target)(features)


def _normalized_mean(products: np.ndarray) -> float:
    """``sqrt(n)`` times the mean of ``products`` over their standard deviation.

    The standard deviation divides by ``n``. Constant products give ``nan`` when they are zero, no
    covariance to detect, and an infinity of their sign otherwise, a covariance without noise.
    """
    mean = float(np.mean(products))
    var = float(np.mean(products * products)) - mean * mean
    if not var > 1e-300:
        return float("nan") if mean == 0.0 else math.copysign(math.inf, mean)
    return float(np.sqrt(products.size) * mean / np.sqrt(var))


def _gcm_answer(rx: np.ndarray, ry: np.ndarray) -> tuple[float, float]:
    """The GCM statistic of the residuals ``rx`` and ``ry``, with its two-sided normal p-value."""
    statistic = _normalized_mean(rx * ry)
    if np.isnan(statistic):
        return 0.0, 1.0  # a constant product: no covariance to detect
    return statistic, float(2.0 * ndtr(-abs(statistic)))


class GCM(_RegressionTest):
    """The generalized covariance measure (GCM): the normalized covariance of two residuals.

    Each query regresses ``x`` and ``y`` on the conditioning set, multiplies the two residuals, and
    tests whether the mean of the products is zero, which it is when ``x`` and ``y`` are
    conditionally independent and the regressions are accurate enough.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is standardized to unit variance.
    regression : {"boosting", "forest", "linear"}, default="boosting"
        The regression of each variable on the conditioning set. ``"boosting"`` is boosted
        regression trees whose depth (1, 3, 4, 5, or 6) and number of rounds (up to 50, learning
        rate 0.3) minimize the 10-fold cross-validated error, the authors' default grid; the trees'
        leaf values carry no penalty, where the authors' boosting shrinks them, so the fits can
        leave smaller residuals. ``"forest"`` is a random forest of 500 trees; ``"linear"``
        ordinary least squares.
    seed : int or None, default=None
        Seed for the cross-validation folds of ``"boosting"`` and the trees of ``"forest"``.
        ``None`` uses ``ANDREY_SEED``, or ``0`` when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has fewer than 4 rows, contains NaN or infinity, or has values too
        large or too small to standardize, or ``regression`` is not one of the choices.

    Notes
    -----
    **When to use.** For continuous data whose variables depend on the conditioning set through a
    regression the chosen method can learn, such as a smooth nonlinear function plus noise, at
    hundreds to thousands of rows [1]_. The test keeps its level whenever the product of the two
    regressions' errors shrinks faster than one over the number of rows, whatever the noise's
    distribution [1]_. It detects a conditional covariance of ``x`` and ``y``; a dependence that
    leaves their conditional covariance zero at every value of the conditioning set is the target
    of ``"wgcm"``.

    **How it works.** For a conditioning set ``z``, ``x`` and ``y`` are each regressed on ``z`` and
    ``R_i`` is the product of their residuals at row ``i``. The statistic is ``sqrt(n)`` times the
    mean of ``R`` over its standard deviation (dividing by ``n``), and the p-value is twice the
    standard normal tail beyond its absolute value [1]_. An empty conditioning set uses ``x`` and
    ``y`` minus their means. A residual depends only on the variable and the conditioning set, so
    it is kept on the test object and reused by every later query on the same pair of them. Each
    column is standardized and rounded to 9 decimal places, and the folds and the trees are drawn
    over the rows in lexicographic order of the variable and the conditioning set alone, so the
    p-value does not change with the row or column order, with columns outside the query, or with
    a column's units or sign. The exceptions: a change of units that moves a value across the
    rounding, and a sign flip of a column whose sorted values equal their negation's, such as a
    balanced ``-1``/``1`` column, which no value of the column orients.
    Products that are all zero give the statistic ``0`` and the p-value ``1``; products that are
    one nonzero constant, a covariance without noise, give an infinite statistic and the p-value
    ``0``.

    **Limitations.** Not benchmarked yet. The tests run it on up to 4 variables and 400 rows. A
    ``"boosting"`` regression fits 51 boosted models, once per variable and conditioning set. Its
    level depends on the regressions: a conditional mean the method cannot learn, such as a function
    of several variables of ``z`` jointly under ``"linear"``, leaves a covariance in the residuals
    and reads as dependence. At a few hundred rows, where the conditional means are hard to learn,
    the test can reject a true independence more often than its level, as the weighted GCM's
    paper reports for it at 100 and 200 rows [2]_. With an empty conditioning set the test is a
    test of correlation: a dependence without one, such as ``y = x**2`` for a symmetric ``x``, goes
    unseen.

    References
    ----------
    .. [1] Shah, R. D., and Peters, J. (2020). The hardness of conditional independence testing and
       the generalised covariance measure. Annals of Statistics 48(3), 1514-1538.
       https://doi.org/10.1214/19-AOS1857
    .. [2] Scheidegger, C., Hoerrmann, J., and Buehlmann, P. (2022). The weighted generalised
       covariance measure. Journal of Machine Learning Research 23(273), 1-68.
       https://jmlr.org/papers/v23/21-1328.html
    """

    def __init__(
        self, data: np.ndarray, *, regression: str = "boosting", seed: int | None = None
    ) -> None:
        super().__init__(data, regression=regression, seed=seed, name="GCM")

    def _answer(self, x: int, y: int, cond: list[int]) -> tuple[float, float]:
        return _gcm_pair(self, x, y, cond)


def _gcm_pair(test: _RegressionTest, x: int, y: int, cond: list[int]) -> tuple[float, float]:
    """The GCM answer of a query from residuals over every row, in the query's row order."""
    rows, _ = test._layout([x, y, *cond])
    return _gcm_answer(test._full_residual(x, cond)[rows], test._full_residual(y, cond)[rows])


# The weighted GCM as its paper runs it: 30% of the rows estimate the weight function, the fixed
# weights are the signs of each conditioning variable about its 7 octiles plus a constant, and the
# multiplier null has 499 draws.
WGCM_VARIANTS: tuple[str, ...] = ("est", "fix")
_WEIGHT_FRACTION = 0.3
_FIXED_WEIGHTS = 7
_MULTIPLIER_DRAWS = 499


class WGCM(_RegressionTest):
    """The weighted generalized covariance measure (wGCM): the GCM of weighted residual products.

    The GCM's products of residuals are multiplied by a function of the conditioning set before
    their mean is tested, so that a conditional covariance of ``x`` and ``y`` that changes sign
    over the conditioning set, and averages to zero, is detected.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is standardized to unit variance.
    variant : {"est", "fix"}, default="est"
        ``"est"`` estimates one weight function on 30% of the rows and tests on the rest;
        ``"fix"`` tests a fixed set of weight functions on every row and takes the largest
        statistic.
    regression : {"boosting", "forest", "linear"}, default="boosting"
        The regression of each variable, and of the products of residuals, on the conditioning set,
        as for :class:`~andrey.core.ci_regression.GCM`.
    seed : int or None, default=None
        Seed for the sample split of ``"est"``, the multiplier draws of ``"fix"``, the
        cross-validation folds of ``"boosting"``, and the trees of ``"forest"``. ``None`` uses
        ``ANDREY_SEED``, or ``0`` when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has fewer than 17 rows (4 for ``"fix"``), contains NaN or
        infinity, or has values too large or too small to standardize, or ``variant`` or
        ``regression`` is not one of the choices.

    Notes
    -----
    **When to use.** For continuous data where ``x`` and ``y`` may depend on each other given the
    conditioning set with a conditional covariance that changes sign, which the GCM does not
    detect, at hundreds to thousands of rows [1]_. Where the dependence is a covariance of one sign
    or the rows are few, ``"gcm"`` has more power [1]_.

    **How it works.** For a conditioning set ``z``, the residuals of ``x`` and ``y`` on ``z`` give
    the products ``R_i`` as for the GCM [2]_. ``"est"`` (Method 1 of [1]_) splits the rows at
    random, 30% to estimate and 70% to test. On the first part it regresses the products on ``z``
    and takes the weight ``w(z)``, the sign of that fit; on the second part, with residuals from
    regressions on that part, the statistic is the GCM statistic of ``R_i w(z_i)``, and the
    p-value is the standard normal tail above it, one-sided because the weight points the
    covariance up. ``"fix"`` takes the constant weight and the signs of each variable of ``z`` about
    its empirical 1/8, ..., 7/8 quantiles, normalizes each weighted product by its standard
    deviation, and the statistic is ``sqrt(n)`` times the largest absolute mean; its p-value is
    ``(b + 1) / 500``, where ``b`` of 499 Gaussian multiplier draws of the same maximum over the
    centered columns, whose covariance is the products' sample correlation, are at least as large
    [1]_. An empty conditioning set has no weight to estimate, and both variants give the
    GCM's p-value of ``x`` and ``y`` minus their means. The statistic reported for ``"est"`` is the
    statistic or ``0``, whichever is larger, so a larger statistic always means a smaller p-value.
    The weight is the sign of the fit, as in the paper; with the fit itself as the weight, as in
    the authors' later code, in reduced versions of the paper's designs with boosted trees the test
    rejected more often under both alternatives and nulls, and no gain held at the level.
    The draws of ``"fix"`` use the centered columns, as in the paper; the authors' code's uncentered
    columns gave less power at the same level. The split and the multiplier draws are drawn from
    ``seed`` over the rows in lexicographic order of the query's own columns, so they do not depend
    on other columns.

    **Limitations.** Not benchmarked yet. The tests run it on up to 4 variables and 400 rows.
    ``"est"`` tests on 70% of the rows, and its p-value changes with the seed. A ``"boosting"``
    regression fits 51 boosted models; ``"est"`` fits five per query, its split being the query's
    own. ``"fix"`` uses 7 weights per conditioning variable, so with many conditioning variables
    the maximum spreads its power thin. At a few hundred rows, where the conditional means are hard
    to learn, both variants can reject a true independence more often than their level [1]_; with
    ``"boosting"`` they did at 400 rows on the paper's one-dimensional design.

    References
    ----------
    .. [1] Scheidegger, C., Hoerrmann, J., and Buehlmann, P. (2022). The weighted generalised
       covariance measure. Journal of Machine Learning Research 23(273), 1-68.
       https://jmlr.org/papers/v23/21-1328.html
    .. [2] Shah, R. D., and Peters, J. (2020). The hardness of conditional independence testing and
       the generalised covariance measure. Annals of Statistics 48(3), 1514-1538.
       https://doi.org/10.1214/19-AOS1857
    """

    @staticmethod
    def least_rows(size: int, *, variant: object = "est", **options: object) -> int:
        """The fewest rows a query given ``size`` conditioning variables is tested on: ``17`` for
        ``variant="est"``, whose 30% part then holds the 6 rows a forest splits, else ``4``."""
        return 17 if variant == "est" else 4

    def __init__(
        self,
        data: np.ndarray,
        *,
        variant: str = "est",
        regression: str = "boosting",
        seed: int | None = None,
    ) -> None:
        if variant not in WGCM_VARIANTS:
            raise ValueError(f"variant must be one of {WGCM_VARIANTS}, got {variant!r}")
        min_rows = self.least_rows(0, variant=variant)
        super().__init__(data, regression=regression, seed=seed, name="wGCM", min_rows=min_rows)
        self._variant = variant
        n = self._n
        n_estimate = int(np.ceil(_WEIGHT_FRACTION * n))
        perm = np.random.default_rng([self._seed, n, 1]).permutation(n)
        # The positions of the two parts in a query's row order.
        self._parts = (np.sort(perm[:n_estimate]), np.sort(perm[n_estimate:]))
        self._multipliers: np.ndarray | None = None

    def _answer(self, x: int, y: int, cond: list[int]) -> tuple[float, float]:
        if not cond:
            return _gcm_pair(self, x, y, cond)
        rows, cols = self._layout([x, y, *cond])
        block, features = self._data[rows], [c for c in cols if c not in (x, y)]
        if self._variant == "fix":
            products = (self._full_residual(x, cond) * self._full_residual(y, cond))[rows]
            return _fixed_answer(products, block[:, features], self._multiplier_draws())
        estimate, test = (block[part] for part in self._parts)

        def products(rows: np.ndarray) -> np.ndarray:
            z = rows[:, features]
            return self._residual(rows[:, x], z) * self._residual(rows[:, y], z)

        weight = np.sign(self._fit(estimate[:, features], products(estimate))(test[:, features]))
        return _estimated_answer(products(test), weight)

    def _multiplier_draws(self) -> np.ndarray:
        """The ``(n, 499)`` standard normal multipliers by position in a query's row order."""
        if self._multipliers is None:
            rng = np.random.default_rng([self._seed, self._n, 3])
            self._multipliers = rng.standard_normal((self._n, _MULTIPLIER_DRAWS))
        return self._multipliers


def _estimated_answer(products: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """The one-sided statistic and p-value of ``products`` weighted by ``weight``."""
    statistic = _normalized_mean(products * weight)
    if np.isnan(statistic):
        return 0.0, 1.0
    return max(statistic, 0.0), float(ndtr(-statistic))


def _fixed_answer(
    products: np.ndarray, z: np.ndarray, multipliers: np.ndarray
) -> tuple[float, float]:
    """The maximum statistic of ``products`` under the fixed sign weights of ``z``, and its
    multiplier p-value over the columns of ``multipliers``."""
    n = len(products)
    levels = np.quantile(z, np.arange(1, _FIXED_WEIGHTS + 1) / (_FIXED_WEIGHTS + 1), axis=0)
    weights = np.column_stack(
        [np.ones(n), *(np.sign(z[:, j, None] - levels[None, :, j]) for j in range(z.shape[1]))]
    )
    weighted = products[:, None] * weights
    std = weighted.std(axis=0)
    keep = std > 1e-300
    if (~keep & (weighted.mean(axis=0) != 0.0)).any():
        return math.inf, 1.0 / (multipliers.shape[1] + 1)  # a covariance without noise
    if not keep.any():
        return 0.0, 1.0
    normalized = weighted[:, keep] / std[keep]
    root_n = np.sqrt(n)
    means = normalized.mean(axis=0)
    statistic = float(root_n * np.abs(means).max())
    # Centered columns give the draws the sample correlation of the weighted products as their
    # covariance, the null of the paper's maximum.
    draws = np.abs((normalized - means).T @ multipliers).max(axis=0) / root_n
    exceed = np.count_nonzero(draws >= statistic)
    return statistic, (exceed + 1) / (multipliers.shape[1] + 1)


# The PCM as its paper recommends it: 6 sample splits, their statistics averaged. The smallest
# conditional variance is 0.01 of the standardized response, the authors' code's floor, which keeps
# the projection finite where the variance fit is zero.
_PCM_SPLITS = 6
_VARIANCE_FLOOR = 0.01


class PCM(_RegressionTest):
    """The projected covariance measure (PCM): whether one variable improves the prediction of
    another given the conditioning set.

    One half of the rows learns how ``y`` depends on ``x`` beyond the conditioning set, as a
    function of both; the other half tests the covariance of that function with ``y``, both after
    regressing out the conditioning set. The statistic is averaged over several random splits.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_variables)
        Observed data, one row per sample. Each column is standardized to unit variance.
    regression : {"forest", "boosting", "linear"}, default="forest"
        The regressions of the test. ``"forest"`` is a random forest of 500 trees that tries every
        predictor at each split and leaves nodes of 5 rows or fewer unsplit, as the authors' code
        does for its random-forest experiments; ``"boosting"`` and ``"linear"`` are as for
        :class:`~andrey.core.ci_regression.GCM`.
    n_splits : int, default=6
        The number of random sample splits whose statistics are averaged, at least ``1``.
    seed : int or None, default=None
        Seed for the sample splits, the cross-validation folds of ``"boosting"``, and the trees of
        ``"forest"``. ``None`` uses ``ANDREY_SEED``, or ``0`` when it is unset.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, has fewer than 12 rows, contains NaN or infinity, or has values too
        large or too small to standardize, ``regression`` is not one of the choices, or
        ``n_splits`` is below 1.

    Notes
    -----
    **When to use.** For continuous data where the conditional mean of one variable may depend on
    the other given the conditioning set in any form, interactions included, at hundreds to
    thousands of rows [1]_. It tests conditional mean independence, which conditional independence
    implies, and keeps its level whenever the regressions on the conditioning set are accurate
    enough, whatever the regression that learns the dependence [1]_.

    **How it works.** Of a query's two variables, the one with the larger
    :func:`~andrey.core.ties.tie_ranks` rank within the query's columns is the response ``y`` and
    the other the predictor ``x``, so the direction depends on the values of those columns and not
    on the column order, the argument order, or the other columns.
    Each split puts half of the rows, at random, in the first part [1]_ (Algorithm 1). On the first
    part, ``y`` is regressed on ``(x, z)`` to give ``g``, ``g`` on ``z`` to give ``m``, and
    ``h = g - m`` takes the sign of the mean of ``(y - m) h``; the squared residuals of ``g`` are
    regressed on ``(x, z)`` and the fit's positive part, plus the smallest constant ``c`` that
    brings the mean of the squared residuals over it to at most 1, is the conditional variance
    ``v``, at least 0.01. On the second part, ``f = h / v`` and ``y`` are each regressed on ``z``,
    and the statistic is the GCM statistic of the products of their residuals. The ``n_splits``
    statistics are averaged and the p-value is the standard normal tail above the average, the
    one-sided test of Algorithm 1DR with its threshold at the normal quantile [1]_. The reported
    statistic is the average or ``0``, whichever is larger. A split whose products are all zero adds
    ``0``, and one whose products are one nonzero constant an infinity of its sign. The splits are
    drawn from ``seed`` over the rows in lexicographic order of the query's own columns, so they
    do not depend on other columns. The variance adds ``c`` to the fit's positive part, as in the
    paper; the authors' code adds it first, the same for a forest, whose fit is never negative, and
    less powerful with least squares. The standard deviation divides by the second part's row count
    ``m``, as in the paper; the authors' code's ``m - 1`` shrinks each split's statistic by the
    factor ``sqrt((m - 1) / m)``, which in the measurements turned a few rejections near the level
    into acceptances and changed no other decision.

    **Limitations.** Not benchmarked yet. The tests run it on up to 4 variables and 400 rows. Each
    query fits 30 regressions with the default 6 splits, each of 500 trees under ``"forest"``. The
    variance floor of 0.01, in units of the standardized response, is the authors' code's value;
    the paper sets none. The test detects a dependence of the mean of ``y`` on ``x``; one that
    changes only the spread of ``y`` goes unseen, as does a dependence of the mean of ``x`` on
    ``y`` that leaves the mean of ``y`` unchanged. Which variable is the response follows a rule on
    the data that has nothing to do with the causal question, so swapping the roles can change the
    answer. At a few hundred rows, where the conditional means are hard to learn, the test can
    reject a true independence more often than its level.

    References
    ----------
    .. [1] Lundborg, A. R., Kim, I., Shah, R. D., and Samworth, R. J. (2024). The projected
       covariance measure for assumption-lean variable significance testing. Annals of Statistics
       52(6), 2851-2878. https://doi.org/10.1214/24-AOS2447
    """

    @staticmethod
    def least_rows(size: int, **options: object) -> int:
        """The fewest rows a query given ``size`` conditioning variables is tested on: ``12``, so
        each half of a split holds the 6 rows a forest splits and the learned function can
        vary."""
        return 2 * _FOREST_MIN_SPLIT

    def __init__(
        self,
        data: np.ndarray,
        *,
        regression: str = "forest",
        n_splits: int = _PCM_SPLITS,
        seed: int | None = None,
    ) -> None:
        if isinstance(n_splits, bool) or not isinstance(n_splits, int | np.integer) or n_splits < 1:
            raise ValueError(f"n_splits must be a whole number of at least 1, got {n_splits!r}")
        super().__init__(
            data, regression=regression, seed=seed, name="PCM", min_rows=self.least_rows(0)
        )
        n = self._n
        rng = np.random.default_rng([self._seed, n, 2])
        # The positions of each split's two halves in a query's row order.
        self._splits = []
        for _ in range(int(n_splits)):
            perm = rng.permutation(n)
            self._splits.append((np.sort(perm[: n // 2]), np.sort(perm[n // 2 :])))

    def _answer(self, x: int, y: int, cond: list[int]) -> tuple[float, float]:
        rows, cols = self._layout([x, y, *cond])
        block, features = self._data[rows], [c for c in cols if c not in (x, y)]
        response = max(x, y, key=cols.index)  # the later of the two in the block's order
        predictor = x + y - response
        mean = float(
            np.mean(
                [
                    self._split_statistic(block, split, response, predictor, features)
                    for split in self._splits
                ]
            )
        )
        if np.isnan(mean):  # splits of opposite noiseless signs: no direction to report
            mean = 0.0
        return max(mean, 0.0), float(ndtr(-mean))

    def _split_statistic(
        self,
        block: np.ndarray,
        split: tuple[np.ndarray, np.ndarray],
        response: int,
        predictor: int,
        features: list[int],
    ) -> float:
        """The statistic of one sample split of ``block``, ``response`` regressed on ``predictor``
        and the conditioning ``features``."""
        learn, test = split
        y = block[:, response]
        xz = block[:, [predictor, *features]]
        z = block[:, features]
        g = self._fit(xz[learn], y[learn])
        g_learn = g(xz[learn])
        m = self._fit(z[learn], g_learn)
        m_learn = m(z[learn])
        h_learn = g_learn - m_learn
        sign = -1.0 if np.mean((y[learn] - m_learn) * h_learn) < 0 else 1.0
        squares = (y[learn] - g_learn) ** 2
        v = self._fit(xz[learn], squares)
        shift = _variance_shift(squares, np.maximum(v(xz[learn]), 0.0))
        variance = _conditional_variance(v(xz[test]), shift)
        f = sign * (g(xz[test]) - m(z[test])) / variance
        statistic = _normalized_mean(self._residual(y[test], z[test]) * self._residual(f, z[test]))
        return 0.0 if np.isnan(statistic) else statistic


def _conditional_variance(fitted: np.ndarray, shift: float) -> np.ndarray:
    """The PCM's conditional variance: the positive part of ``fitted`` plus ``shift``, at least
    ``_VARIANCE_FLOOR``."""
    return np.maximum(np.maximum(fitted, 0.0) + shift, _VARIANCE_FLOOR)


def _variance_shift(squares: np.ndarray, fitted: np.ndarray) -> float:
    """The smallest ``c >= 0`` with ``mean(squares / (fitted + c)) <= 1``."""

    def excess(c: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(squares > 0, squares / (fitted + c), 0.0)
        return float(np.mean(ratio)) - 1.0

    if excess(0.0) <= 0:
        return 0.0
    high = max(float(np.mean(squares)), 1e-12)
    while excess(high) > 0:
        high *= 2.0
    low = high
    while excess(low) <= 0:
        low /= 2.0
    return float(brentq(excess, low, high, xtol=1e-14, rtol=1e-14))
