"""Conditionally parametric causal models (CPCM): residual fits, the score, and the searches.

In a CPCM (Bodik and Chavez-Demoulin 2025) each variable given its parents follows a known
parametric family whose parameters are smooth functions of the parents. Fitting the family and
mapping each value through its fitted distribution function, ``eps = F(x_j; theta(parents))``,
gives a residual that is uniform and independent of the parents in the causal direction only.

Each parameter is a linear predictor ``eta = c + sum_k f_k(x_k)`` through the family's link, every
``f_k`` a rank-10 thin plate regression spline (:func:`~andrey.fcm.gam.tp_basis`) with its own
smoothing parameter, or indicators for a parent of at most 9 distinct values. The coefficients
maximize the penalized log-likelihood by Newton's method, and the smoothing parameters maximize
its Laplace approximate marginal likelihood (Wood, Pya, and Safken 2016). A count family's
distribution function is randomized between its steps, which needs the ``seed``.

:func:`pair_tests` runs Algorithm 1's two independence tests; :func:`search` minimizes the score
``-log p + edge_penalty * edges`` over DAGs, ``p`` the gamma dHSIC test of the residuals' joint
independence, exactly for at most 4 variables and by Algorithm 2 (RESIT's order, then greedy edge
removal) above.

References
----------
Bodik and Chavez-Demoulin (2025). Identifiability of causal graphs under non-additive conditionally
parametric causal models. JMLR 26(264).

Wood, Pya, and Safken (2016). Smoothing parameter and model selection for general smooth models.
JASA 111(516), 1548-1563.
"""

from __future__ import annotations

import itertools
import math
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from andrey.core.independence import dhsic_gram, dhsic_log_p, within_one
from andrey.core.ties import EPS_IMPROVE, tie_key
from andrey.fcm.gam import tp_basis

#: Basis rank of every smooth before its sum-to-zero constraint, mgcv's default for ``s(x)``.
RANK = 10
#: A parent with at most this many distinct values enters as indicators, as in the authors' code.
MAX_LEVELS = 9
#: Largest number of variables the exact search enumerates (543 DAGs on 4 variables).
EXACT_MAX = 4
#: Half-width of the log smoothing parameters' range around their start, which follows the units.
_LOG_SP_RANGE = 20.0
#: The search over the log smoothing parameters moves by a step that halves from the first of
#: these to the second, so every point it reads lies on one lattice around the start.
_SP_STEPS = (4.0, 1.0 / 16.0)
#: A point of that search replaces the best one only if its LAML is lower by more than this share.
_LAML_GAIN = 1e-6
_NEWTON_TOL = 1e-11
_MAX_NEWTON = 100
#: Lower bound of a fitted log scale that varies with the parents, on the column the fit reads:
#: the Gaussian standard deviation stays above 0.01 of the column's and the Gumbel scale above
#: ``exp(-7)`` of it, and the gamma dispersion above ``exp(-7)``. These are the values of the
#: location-scale families in the paper's software; without a bound the likelihood has no maximum,
#: since it grows without limit as the scale shrinks at a row the location fits exactly.
LOG_SCALE_FLOOR = {"gaussian": math.log(0.01), "gumbel": -7.0, "gamma": -7.0}
#: Bounds of the upper bound of the negative binomial family's log size, which follows the counts
#: (:func:`_log_size_bound`): at least 12 (a size of about 163,000), and at most 20, from where
#: the likelihood's terms lose their digits to cancellation.
NB_LOG_SIZE_MAX = (12.0, 20.0)
#: The bound is the log of the largest count plus this, so a size at the bound is ``exp(5)`` times
#: every count or more and the family is within 1 part in 148 of its Poisson limit's variance.
NB_LOG_SIZE_OVER_MAX = 5.0
#: A Hessian is singular to working precision if, with its diagonal brought to 1, an eigenvalue is
#: below this. The coefficients and the log determinant at such a point follow the table's last
#: digits (their rounding error is the machine's divided by that eigenvalue), so its LAML is not a
#: number two runs agree on, and the point has no fit.
_SINGULAR = 1e-8
#: A residual, a probability, is the same on every row if its values lie within this of each other.
_CONSTANT = 1e-9
#: A point is a minimum only if a Newton step from it lowers the objective by less than this share.
_UNFINISHED = 1e-7

Derivs = tuple[float, list[np.ndarray], list[list[np.ndarray]]]


@dataclass(frozen=True)
class Family:
    """One conditional family: its linear predictors and their likelihood derivatives.

    ``predictors`` is 1, or 2 with ``second`` ``"smooth"`` (a function of the parents) or
    ``"constant"`` (one fitted value for all rows). ``derivs(y, etas)`` returns the negative
    log-likelihood's sum, its first derivatives per predictor, and its second derivatives per
    pair of predictors, each per row. ``cdf(y, etas, u)`` is the fitted distribution function at
    ``y``, randomized by the uniforms ``u`` for a count family. ``rescale``, set for a
    location-scale or scale family, maps ``y`` to the column the fit and the distribution function
    read: the family's probability transform is the same on it, so the result does not depend on
    the units and the fit does not depend on their size. ``upper(y)`` bounds a ``"constant"``
    second predictor from above.
    """

    name: str
    support: str
    predictors: int
    second: str
    start: Callable[[np.ndarray], list[float]]
    derivs: Callable[[np.ndarray, list[np.ndarray]], Derivs]
    cdf: Callable[[np.ndarray, list[np.ndarray], np.ndarray | None], np.ndarray]
    outside: Callable[[np.ndarray], np.ndarray]
    counts: bool = False
    rescale: Callable[[np.ndarray], np.ndarray] | None = None
    upper: Callable[[np.ndarray], float] | None = None

    @property
    def reads_units(self) -> bool:
        """Whether the column's units are part of the family: it fixes a scale (counts step by
        1, the Pareto law starts at 1), so the same column in other units is another model."""
        return self.rescale is None


def _gaussian_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    mu, log_s = etas
    inv = np.exp(-log_s)
    r = (y - mu) * inv
    nll = float(np.sum(log_s + 0.5 * r * r))
    return nll, [-r * inv, 1.0 - r * r], [[inv * inv, 2.0 * r * inv], [2.0 * r * inv, 2.0 * r * r]]


def _gumbel_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    mu, log_s = etas
    inv = np.exp(-log_s)
    z = (y - mu) * inv
    ez = np.exp(-z)
    nll = float(np.sum(log_s + z + ez))
    h12 = (z * ez + 1.0 - ez) * inv
    return (
        nll,
        [-(1.0 - ez) * inv, 1.0 - z * (1.0 - ez)],
        [[ez * inv * inv, h12], [h12, z * (1.0 - ez + z * ez)]],
    )


def _floored(derivs: Callable[[np.ndarray, list[np.ndarray]], Derivs], floor: float) -> Callable:
    """``derivs`` with its log scale read as ``log(exp(eta) + exp(floor))``, which stays above
    ``floor`` and nears ``eta`` away from it; the derivatives follow by the chain rule."""

    def floored(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
        nll, g, h = derivs(y, [etas[0], np.logaddexp(etas[1], floor)])
        d1 = special.expit(etas[1] - floor)
        h12 = h[0][1] * d1
        return (
            nll,
            [g[0], g[1] * d1],
            [[h[0][0], h12], [h12, h[1][1] * d1 * d1 + g[1] * d1 * (1 - d1)]],
        )

    return floored


def _exponential_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    (eta,) = etas
    u = y * np.exp(-eta)
    return float(np.sum(eta + u)), [1.0 - u], [[u]]


def _pareto_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    # log(y) is exponential with rate exp(-eta); the Jacobian 1 / y does not depend on eta.
    nll, grad, hess = _exponential_derivs(np.log(y), etas)
    return nll + float(np.sum(np.log(y))), grad, hess


def _poisson_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    (eta,) = etas
    lam = np.exp(eta)
    return float(np.sum(lam - y * eta + special.gammaln(y + 1.0))), [lam - y], [[lam]]


def _gamma_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    log_m, log_phi = etas
    k = np.exp(-log_phi)  # shape; the mean is exp(log_m)
    u = y * np.exp(-log_m)
    nll = float(
        np.sum(special.gammaln(k) - k * np.log(k) + k * log_m - (k - 1.0) * np.log(y) + k * u)
    )
    a = special.digamma(k) - np.log(k) - 1.0 - np.log(u) + u
    h22 = k * a + k * k * special.polygamma(1, k) - k
    return nll, [k * (1.0 - u), -k * a], [[k * u, -k * (1.0 - u)], [-k * (1.0 - u), h22]]


def _negative_binomial_derivs(y: np.ndarray, etas: list[np.ndarray]) -> Derivs:
    log_m, log_size = etas
    m, t = np.exp(log_m), np.exp(log_size)
    tm = t + m
    nll = -float(
        np.sum(
            special.gammaln(y + t)
            - special.gammaln(t)
            - special.gammaln(y + 1.0)
            + t * (log_size - np.log(tm))
            + y * (log_m - np.log(tm))
        )
    )
    b = special.digamma(y + t) - special.digamma(t) + log_size - np.log(tm) + (m - y) / tm
    db = (
        special.polygamma(1, y + t) - special.polygamma(1, t) + 1.0 / t - 1.0 / tm - (m - y) / tm**2
    )
    h12 = t * m * (m - y) / tm**2
    return (
        nll,
        [(t + y) * m / tm - y, -t * b],
        [[(t + y) * m * t / tm**2, h12], [h12, -t * b - t * t * db]],
    )


def _randomized(lower: np.ndarray, mass: np.ndarray, u: np.ndarray | None) -> np.ndarray:
    if u is None:
        raise ValueError("a count family's distribution function needs uniform draws")
    return lower + u * mass


def _poisson_cdf(y: np.ndarray, etas: list[np.ndarray], u: np.ndarray | None) -> np.ndarray:
    lam = np.exp(etas[0])
    return _randomized(stats.poisson.cdf(y - 1.0, lam), stats.poisson.pmf(y, lam), u)


def _negative_binomial_cdf(
    y: np.ndarray, etas: list[np.ndarray], u: np.ndarray | None
) -> np.ndarray:
    if etas[1][0] >= _log_size_bound(y):
        return _poisson_cdf(y, etas[:1], u)
    m, t = np.exp(etas[0]), np.exp(etas[1])
    p = t / (t + m)
    return _randomized(stats.nbinom.cdf(y - 1.0, t, p), stats.nbinom.pmf(y, t, p), u)


def _log_size_bound(y: np.ndarray) -> float:
    """Upper bound of the negative binomial log size fitted to the counts ``y``.

    The likelihood is flat in the size only once the size is far above the counts: the variance
    is ``1 + mean / size`` times the Poisson's, so a fixed bound would read overdispersed counts
    with large means as Poisson. The bound therefore follows the largest count, and a fit that
    ends on it is read as the Poisson limit, which it then is to within that margin.
    """
    low, high = NB_LOG_SIZE_MAX
    return min(high, max(low, math.log(max(float(np.max(y)), 1.0)) + NB_LOG_SIZE_OVER_MAX))


def _not_finite(y: np.ndarray) -> np.ndarray:
    return ~np.isfinite(y)


def _not_whole(y: np.ndarray) -> np.ndarray:
    return ~np.isfinite(y) | (y < 0) | (y != np.floor(y))


def _standardized(y: np.ndarray) -> np.ndarray:
    """``y`` centered and divided by its standard deviation (location-scale families)."""
    y = within_one(y)[0]
    spread = float(np.std(y))
    return (y - np.mean(y)) / (spread if spread > 0 else 1.0)


def _over_mean(y: np.ndarray) -> np.ndarray:
    """``y`` divided by its mean (scale families)."""
    y = within_one(y)[0]
    mean = float(np.mean(y))
    return y / (mean if mean > 0 else 1.0)


def _scale_start(y: np.ndarray) -> float:
    return math.log(max(float(np.std(y)), 1e-8))


def _count_start(y: np.ndarray) -> list[float]:
    mean = max(float(np.mean(y)), 1e-3)
    excess = float(np.var(y)) - mean
    size = mean * mean / excess if excess > 1e-8 else 1e3
    return [math.log(mean), math.log(min(max(size, 1e-3), 1e3))]


def _gamma_start(y: np.ndarray) -> list[float]:
    mean = float(np.mean(y))
    return [math.log(mean), math.log(max(float(np.var(y)) / mean**2, 1e-6))]


def _gumbel_start(y: np.ndarray) -> list[float]:
    scale = max(float(np.std(y)) * math.sqrt(6.0) / math.pi, 1e-8)
    return [float(np.mean(y)) - np.euler_gamma * scale, math.log(scale)]


def _families() -> dict[str, Family]:
    gaussian_cdf = lambda y, e, u: stats.norm.cdf((y - e[0]) * np.exp(-e[1]))  # noqa: E731
    gumbel_cdf = lambda y, e, u: np.exp(-np.exp(-(y - e[0]) * np.exp(-e[1])))  # noqa: E731
    exponential_cdf = lambda y, e, u: -np.expm1(-y * np.exp(-e[0]))  # noqa: E731
    pareto_cdf = lambda y, e, u: -np.expm1(-np.log(y) * np.exp(-e[0]))  # noqa: E731
    gamma_cdf = lambda y, e, u: special.gammainc(  # noqa: E731
        np.exp(-e[1]), y * np.exp(-e[0] - e[1])
    )
    gaussian_start = lambda y: [float(np.mean(y)), _scale_start(y)]  # noqa: E731
    positive = lambda y: ~np.isfinite(y) | (y <= 0)  # noqa: E731
    nonnegative = lambda y: ~np.isfinite(y) | (y < 0)  # noqa: E731
    at_least_one = lambda y: ~np.isfinite(y) | (y < 1)  # noqa: E731
    real = "real numbers"

    def floored(
        name: str, start: Callable, derivs: Callable, cdf: Callable
    ) -> tuple[Callable, Callable, Callable]:
        floor = LOG_SCALE_FLOOR[name]

        def floored_start(y: np.ndarray) -> list[float]:
            first, log_scale = start(y)
            return [first, math.log(max(math.exp(log_scale) - math.exp(floor), math.exp(floor)))]

        return (
            floored_start,
            _floored(derivs, floor),
            lambda y, e, u: cdf(y, [e[0], np.logaddexp(e[1], floor)], u),
        )

    out = [
        Family("gaussian", real, 2, "smooth",
               *floored("gaussian", gaussian_start, _gaussian_derivs, gaussian_cdf),
               _not_finite, rescale=_standardized),
        Family("gaussian_fixed_scale", real, 2, "constant", gaussian_start, _gaussian_derivs,
               gaussian_cdf, _not_finite, rescale=_standardized),
        Family("exponential", "non-negative numbers", 1, "", lambda y: [math.log(np.mean(y))],
               _exponential_derivs, exponential_cdf, nonnegative, rescale=_over_mean),
        Family("pareto", "numbers of at least 1 (a Pareto law of scale 1)", 1, "",
               lambda y: [math.log(max(float(np.mean(np.log(y))), 1e-8))], _pareto_derivs,
               pareto_cdf, at_least_one),
        Family("poisson", "counts (whole numbers of at least 0)", 1, "",
               lambda y: _count_start(y)[:1], _poisson_derivs, _poisson_cdf, _not_whole,
               counts=True),
        Family("gamma", "positive numbers", 2, "smooth",
               *floored("gamma", _gamma_start, _gamma_derivs, gamma_cdf),
               positive, rescale=_over_mean),
        Family("gamma_fixed_shape", "positive numbers", 2, "constant", _gamma_start,
               _gamma_derivs, gamma_cdf, positive, rescale=_over_mean),
        Family("gumbel", real, 2, "smooth",
               *floored("gumbel", _gumbel_start, _gumbel_derivs, gumbel_cdf),
               _not_finite, rescale=_standardized),
        Family("gumbel_fixed_scale", real, 2, "constant", _gumbel_start, _gumbel_derivs,
               gumbel_cdf, _not_finite, rescale=_standardized),
        Family("negative_binomial", "counts (whole numbers of at least 0)", 2, "constant",
               _count_start, _negative_binomial_derivs, _negative_binomial_cdf, _not_whole,
               counts=True, upper=_log_size_bound),
    ]  # fmt: skip
    return {f.name: f for f in out}


#: Every conditional family, by name.
FAMILIES: dict[str, Family] = _families()


@dataclass(frozen=True)
class _Term:
    """One parent's columns in a design: a constrained smooth with its penalty, or indicators."""

    X: np.ndarray
    S: np.ndarray | None
    rank: int


def _term(x: np.ndarray) -> _Term:
    levels = np.unique(x)
    if levels.size <= MAX_LEVELS:
        return _Term((x[:, None] == levels[None, 1:]).astype(np.float64), None, 0)
    basis = tp_basis(x, RANK)
    rank = int(np.sum(np.linalg.eigvalsh(basis.S) > 1e-8 * np.abs(basis.S).max()))
    return _Term(basis.X, basis.S, rank)


class Residuals:
    """The residuals of one table under per-column families, cached by variable and parent set.

    ``X`` is the ``(n, d)`` table and ``families`` one :class:`Family` per column. A residual is
    the probability transform of a column under its family fitted on a parent set; a variable
    without parents is its own residual. A count family's uniforms are drawn once per column from
    ``seed`` and the column's values, so the draws follow the column whatever its position.
    ``method`` and ``name(j)`` name the method and column ``j`` in messages. A fit that fails, or
    a residual that is the same on every row, raises ``ValueError`` with both.
    """

    def __init__(
        self,
        X: np.ndarray,
        families: list[Family],
        seed: int,
        *,
        method: str = "cpcm",
        name: Callable[[int], str] = str,
    ) -> None:
        self.X = X
        self.families = families
        self.seed = seed
        self.method, self.name = method, name
        self._terms: dict[int, _Term] = {}
        self._residuals: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
        self._grams: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}

    def term(self, j: int) -> _Term:
        if j not in self._terms:
            self._terms[j] = _term(self.X[:, j])
        return self._terms[j]

    def uniforms(self, j: int) -> np.ndarray:
        column = np.ascontiguousarray(self.X[:, j])
        rng = np.random.default_rng([self.seed, zlib.crc32(column.tobytes())])
        return rng.random(column.size)

    def residual(self, j: int, parents: tuple[int, ...]) -> np.ndarray:
        key = (j, tuple(sorted(parents)))
        if key not in self._residuals:
            if not parents:
                self._residuals[key] = self.X[:, j]
            else:
                family = self.families[j]
                y = self.X[:, j] if family.rescale is None else family.rescale(self.X[:, j])
                on = f"{self.method}: column {self.name(j)} on " + ", ".join(
                    self.name(k) for k in key[1]
                )
                try:
                    etas = fit(y, [self.term(k) for k in key[1]], family)
                except ValueError as error:
                    raise ValueError(f"{on}: {error}") from None
                u = self.uniforms(j) if family.counts else None
                residual = family.cdf(y, etas, u)
                if not np.ptp(residual) > _CONSTANT:
                    raise ValueError(
                        f"{on}: under family {family.name!r} the residual is the same on every "
                        "row, as when the column is an exact function of those columns, so no "
                        "independence test can read it; drop one of the columns"
                    )
                self._residuals[key] = residual
        return self._residuals[key]

    def gram(self, j: int, parents: tuple[int, ...]) -> np.ndarray:
        key = (j, tuple(sorted(parents)))
        if key not in self._grams:
            self._grams[key] = dhsic_gram(self.residual(j, parents))
        return self._grams[key]

    def log_p(self, parents: list[tuple[int, ...]]) -> float:
        """Log p-value of the dHSIC test of the residuals' joint independence under ``parents``."""
        return dhsic_log_p([self.gram(j, pa) for j, pa in enumerate(parents)])[1]


def fit(y: np.ndarray, terms: list[_Term], family: Family) -> list[np.ndarray]:
    """Fit ``family`` to ``y`` on the parents' ``terms``; return each linear predictor per row.

    Each smooth linear predictor has an intercept and every term's columns; a constant one has the
    intercept alone. Beside a smooth parent, a second predictor leaves out the indicator of a
    parent value on whose rows ``y`` holds one value (:func:`_varying`). The coefficients
    minimize the penalized negative log-likelihood, and the log
    smoothing parameters minimize its Laplace approximate marginal likelihood (Wood, Pya, and
    Safken 2016) over a lattice around their start (:func:`_search`).

    Raises ``ValueError`` where the fit has no maximum to find: as many unpenalized coefficients
    as rows, or a search that ends away from a minimum of the penalized likelihood.
    """
    n = y.size
    free = 1 + sum(t.X.shape[1] for t in terms if t.S is None)
    if free >= n:
        raise ValueError(
            f"a fit on the parents has {free} unpenalized coefficients for {n} rows: a parent of "
            f"at most {MAX_LEVELS} values adds one for each value but the first, so use more rows "
            "or fewer columns"
        )
    layouts: list[list[_Term]] = [terms]
    if family.predictors == 2:
        layouts.append([])
        if family.second == "smooth":
            smooth = any(t.S is not None for t in terms)
            layouts[1] = [_Term(_varying(t.X, y), None, 0) if smooth and t.S is None else t
                          for t in terms]  # fmt: skip
    designs = [np.hstack([np.ones((n, 1)), *(t.X for t in layout)]) for layout in layouts]
    sizes = [B.shape[1] for B in designs]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    penalties: list[tuple[slice, np.ndarray, int]] = []
    for j, layout in enumerate(layouts):
        col = offsets[j] + 1
        for t in layout:
            width = t.X.shape[1]
            if t.S is not None:
                penalties.append((slice(col, col + width), t.S, t.rank))
            col += width
    beta = np.zeros(offsets[-1])
    for j, value in enumerate(family.start(y)[: family.predictors]):
        beta[offsets[j]] = value
    model = _Model(y, designs, offsets, penalties, family, beta)
    fitted = _search(model) if penalties else model.laml(np.zeros(0))[1]
    if fitted is None:
        raise ValueError(
            f"the fit of family {family.name!r} on the parents stopped away from a minimum of its "
            "penalized likelihood; use more rows, or a family with fewer fitted parameters"
        )
    return model.etas(fitted)


def _varying(indicators: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The indicator columns of a second predictor, a scale or shape, beside a smooth parent.

    ``indicators`` holds one column per value of a parent but its first. The rows of a value on
    which ``y`` holds one value, such as a value seen once, say nothing about a scale: the location
    fits them exactly, and a scale of their own would shrink without limit. Those values take the
    scale of the first value on whose rows ``y`` varies, which is the reference of the columns
    returned; when ``y`` varies on every value's rows the columns are ``indicators``.
    """
    level = indicators @ np.arange(1.0, indicators.shape[1] + 1)
    varying = [v for v in np.unique(level) if np.ptp(y[level == v]) > 0]
    return (level[:, None] == np.array(varying[1:])[None, :]).astype(np.float64)


def _search(model: _Model) -> np.ndarray | None:
    """The coefficients at the log smoothing parameters of lowest LAML, by a compass search.

    The search starts where each penalty is on the scale of its Hessian block, which moves with
    the units, and tries one step up and down in each parameter in turn, taking a point that
    lowers the LAML and halving the step once none does, from 4 down to 1/16 and within 20 of
    the start. Every LAML is fitted from the same starting coefficients, so it depends on its
    point alone, and the points lie on a fixed lattice around the start: a change of the table
    within rounding (a column in other units) moves the result within rounding too, where a
    gradient search on the LAML's narrow dips, which form where the Hessian is close to
    singular, does not. ``None`` where no point has a fit.
    """
    start = model.start_rho(model.start)
    unit = _SP_STEPS[1]
    reach = round(_LOG_SP_RANGE / unit)
    seen: dict[tuple[int, ...], tuple[float, np.ndarray | None]] = {}

    def laml(point: tuple[int, ...]) -> float:
        if point not in seen:
            seen[point] = model.laml(start + unit * np.array(point))
        return seen[point][0]

    point = (0,) * start.size
    best = laml(point)
    step = round(_SP_STEPS[0] / unit)
    while step >= 1:
        moved = False
        for i in range(start.size):
            for move in (step, -step):
                trial = (*point[:i], point[i] + move, *point[i + 1 :])
                gain = _LAML_GAIN * (1.0 + abs(best)) if math.isfinite(best) else 0.0
                if abs(trial[i]) <= reach and laml(trial) < best - gain:
                    point, best, moved = trial, laml(trial), True
        if not moved:
            step //= 2
    return seen[point][1]


class _Model:
    """A penalized likelihood over per-predictor designs, for :func:`fit`."""

    def __init__(
        self,
        y: np.ndarray,
        designs: list[np.ndarray],
        offsets: np.ndarray,
        penalties: list[tuple[slice, np.ndarray, int]],
        family: Family,
        start: np.ndarray,
    ) -> None:
        self.y, self.designs, self.offsets, self.start = y, designs, offsets, start
        self.penalties, self.family = penalties, family
        self.upper = np.full(offsets[-1], math.inf)
        if family.upper is not None:
            self.upper[offsets[1]] = family.upper(y)

    def etas(self, beta: np.ndarray) -> list[np.ndarray]:
        o = self.offsets
        return [B @ beta[o[j] : o[j + 1]] for j, B in enumerate(self.designs)]

    def penalty(self, rho: np.ndarray) -> np.ndarray:
        S = np.zeros((self.offsets[-1], self.offsets[-1]))
        for (cols, block, _), r in zip(self.penalties, rho, strict=True):
            S[cols, cols] += math.exp(r) * block
        return S

    def evaluate(self, beta: np.ndarray, S: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """The penalized objective, its gradient, and its Hessian at ``beta``."""
        nll, g, h = self.family.derivs(self.y, self.etas(beta))
        o, Bs = self.offsets, self.designs
        grad = np.concatenate([B.T @ gj for B, gj in zip(Bs, g, strict=True)]) + S @ beta
        H = S.copy()
        for j, Bj in enumerate(Bs):
            for k in range(j, len(Bs)):
                block = Bj.T @ (h[j][k][:, None] * Bs[k])
                H[o[j] : o[j + 1], o[k] : o[k + 1]] += block
                if k != j:
                    H[o[k] : o[k + 1], o[j] : o[j + 1]] += block.T
        return nll + 0.5 * float(beta @ S @ beta), grad, H

    def newton(self, beta: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Minimize the penalized objective from ``beta`` by damped Newton steps.

        A step stops at a coefficient's upper bound, and a coefficient at its bound stays there
        while the objective still falls beyond it. A trial point counts only with a finite value,
        gradient, and Hessian. The steps end with the second in a row that lowers the objective by
        less than the tolerance, which brings the coefficients to rounding at a minimum.
        """
        S = self.penalty(rho)
        value, grad, H = self.evaluate(beta, S)
        settled = False
        for _ in range(_MAX_NEWTON):
            free = self.free(beta, grad)
            step = np.zeros(beta.size)
            step[free] = -_solve_pd(H[np.ix_(free, free)], grad[free])
            for _ in range(60):
                trial = np.minimum(beta + step, self.upper)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    new_value, new_grad, new_H = self.evaluate(trial, S)
                finite = np.isfinite(new_value) and np.isfinite(new_grad).all()
                if finite and np.isfinite(new_H).all() and new_value <= value + 1e-12 * abs(value):
                    break
                step *= 0.5
            else:
                break
            done = value - new_value <= _NEWTON_TOL * (1.0 + abs(value))
            beta, value, grad, H = trial, new_value, new_grad, new_H
            if done and settled:
                break
            settled = done
        return beta

    def laml(self, rho: np.ndarray) -> tuple[float, np.ndarray | None]:
        """The negative Laplace approximate marginal likelihood at ``rho``, up to a constant, and
        the coefficients fitted there from the starting ones; infinite where they are no minimum.
        """
        found = self.minimum(rho)
        if found is None:
            return math.inf, None
        fitted, value, logdet = found
        ranks = np.array([rank for _, _, rank in self.penalties], dtype=np.float64)
        return value - 0.5 * float(ranks @ rho) + 0.5 * logdet, fitted

    def free(self, beta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """The coefficients that move: all but one at its upper bound that the gradient holds
        there. A held coefficient is a constant of the fit, outside its Laplace determinant."""
        return ~((beta >= self.upper) & (grad < 0))

    def minimum(self, rho: np.ndarray) -> tuple[np.ndarray, float, float] | None:
        """A minimum of the penalized objective: coefficients, value, Hessian log determinant.

        :meth:`newton` runs from the fit's starting coefficients. ``None`` where it stops away
        from a minimum that two runs agree on (a Hessian that is singular to working precision or
        has a negative eigenvalue, or a Newton step that would still lower the objective) or where
        the objective is not finite. The log determinant is taken from the Hessian with its
        diagonal brought to 1, plus the diagonal's logarithms: the penalties of two smooths can
        differ by many orders of magnitude, and the small eigenvalues of the Hessian as it stands
        would carry the rounding error of its largest entries.
        """
        fitted = self.newton(self.start, rho)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value, grad, H = self.evaluate(fitted, self.penalty(rho))
        if not (np.isfinite(value) and np.isfinite(grad).all() and np.isfinite(H).all()):
            return None
        free = self.free(fitted, grad)
        H, grad = H[np.ix_(free, free)], grad[free]
        unit = 1.0 / np.sqrt(_diagonal(H))
        left = 0.5 * float(grad @ _solve_pd(H, grad))
        eigenvalues = np.linalg.eigvalsh(H * unit[:, None] * unit)
        if eigenvalues[0] < _SINGULAR or left > _UNFINISHED * (1.0 + abs(value)):
            return None
        return fitted, value, float(np.sum(np.log(eigenvalues)) - 2.0 * np.sum(np.log(unit)))

    def start_rho(self, beta: np.ndarray) -> np.ndarray:
        """Log smoothing parameters that put each penalty on the scale of its Hessian block."""
        _, _, H = self.evaluate(beta, np.zeros((self.offsets[-1], self.offsets[-1])))
        rho = [
            math.log(max(np.trace(H[cols, cols]), 1e-8) / np.trace(block))
            for cols, block, _ in self.penalties
        ]
        return np.array(rho)


def _diagonal(H: np.ndarray) -> np.ndarray:
    """The size of each diagonal entry of ``H``, at least ``1e-12`` of the largest."""
    size = np.abs(np.diag(H))
    return np.maximum(size, max(1e-12 * float(size.max()), 1e-300))


def _solve_pd(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Solve ``H x = g``, adding a multiple of ``H``'s diagonal size until it is positive definite.

    The shift follows each coefficient's own scale, so a large penalty on one block does not hold
    back a coefficient outside it.
    """
    shift = 0.0
    diagonal = np.diag(_diagonal(H))
    for _ in range(40):
        try:
            factor = np.linalg.cholesky(H + shift * diagonal)
        except np.linalg.LinAlgError:
            shift = max(2.0 * shift, 1e-10)
            continue
        return np.linalg.solve(factor.T, np.linalg.solve(factor, g))
    return np.linalg.lstsq(H, g, rcond=None)[0]


def pair_tests(res: Residuals) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Algorithm 1's tests on a two-column table: the forward and backward residual tests.

    Returns the log p-values of ``x -> y`` (column 1's residual on column 0 against column 0) and
    of ``y -> x``, their statistics, and the log p-value of the two columns' independence.
    """
    forward = dhsic_log_p([res.gram(0, ()), res.gram(1, (0,))])
    backward = dhsic_log_p([res.gram(1, ()), res.gram(0, (1,))])
    independence = res.log_p([(), ()])
    return (forward[1], backward[1]), (forward[0], backward[0]), independence


def score(res: Residuals, parents: list[tuple[int, ...]], edge_penalty: float) -> float:
    """The score (13) of the DAG ``parents``: ``-log p + edge_penalty * edges``."""
    return -res.log_p(parents) + edge_penalty * sum(len(pa) for pa in parents)


def all_dags(d: int) -> list[list[tuple[int, ...]]]:
    """Every DAG on ``d`` nodes as parent tuples, fewest edges first, then by adjacency bits."""
    pairs = [(i, j) for i in range(d) for j in range(d) if i != j]
    found = []
    for bits in itertools.product((0, 1), repeat=len(pairs)):
        parents: list[list[int]] = [[] for _ in range(d)]
        for (i, j), on in zip(pairs, bits, strict=True):
            if on:
                parents[j].append(i)
        if _acyclic(parents):
            found.append((sum(bits), [tuple(pa) for pa in parents]))
    found.sort(key=lambda item: item[0])
    return [dag for _, dag in found]


def _acyclic(parents: list[list[int]]) -> bool:
    remaining = set(range(len(parents)))
    while remaining:
        leaves = [v for v in remaining if not any(p in remaining for p in parents[v])]
        if not leaves:
            return False
        remaining -= set(leaves)
    return True


def search(
    res: Residuals, edge_penalty: float
) -> tuple[list[tuple[int, ...]], list[int] | None, float]:
    """Minimize the score over DAGs; return the parents, the causal order, and the log p-value.

    Up to :data:`EXACT_MAX` variables every DAG is scored and the first lowest score wins (fewest
    edges first). Above, Algorithm 2: RESIT's order from the sink up, each sink the variable whose
    residual on the others is most independent of them, then the complete DAG of the order loses,
    one at a time, the edge whose removal lowers the score most, while one does. The causal order
    is ``None`` for the exact search.
    """
    d = res.X.shape[1]
    if d <= EXACT_MAX:
        best, best_score = None, math.inf
        for dag in all_dags(d):
            value = score(res, dag, edge_penalty)
            if tie_key(value) < tie_key(best_score):
                best, best_score = dag, value
        assert best is not None
        return best, None, res.log_p(best)
    order = _resit_order(res)
    parents = [tuple(order[: order.index(v)]) for v in range(d)]
    current = score(res, parents, edge_penalty)
    while True:
        trials = []
        for v in range(d):
            for p in parents[v]:
                trial = list(parents)
                trial[v] = tuple(q for q in parents[v] if q != p)
                trials.append((score(res, trial, edge_penalty), trial))
        if not trials:
            break
        value, trial = min(trials, key=lambda item: tie_key(item[0]))
        if value >= current - EPS_IMPROVE:
            break
        parents, current = trial, value
    return parents, order, res.log_p(parents)


def _resit_order(res: Residuals) -> list[int]:
    """RESIT's causal order: repeatedly remove the variable whose residual on the rest is the most
    independent of the rest (the largest gamma HSIC p-value, the rest read jointly with each
    column at unit standard deviation), then put it before the variables removed earlier."""
    remaining = list(range(res.X.shape[1]))
    order: list[int] = []
    while len(remaining) > 1:
        best, best_log_p = remaining[0], -math.inf
        for v in remaining:
            rest = tuple(u for u in remaining if u != v)
            grams = [res.gram(v, rest), dhsic_gram(res.X[:, rest], standardize=True)]
            log_p = dhsic_log_p(grams)[1]
            if tie_key(log_p) > tie_key(best_log_p):
                best, best_log_p = v, log_p
        order.insert(0, best)
        remaining.remove(best)
    return remaining + order


def resolve_families(
    family: str | list[str] | tuple[str, ...], X: np.ndarray, method: str, name: Callable
) -> list[Family]:
    """One :class:`Family` per column of ``X`` from ``family``; check each column's support.

    ``family`` is one name for every column or a list of one name per column. ``name(j)`` names
    column ``j`` in messages. Raises ``TypeError`` for a value that is neither a name nor a list
    of names, and ``ValueError`` for an unknown name, a list of the wrong length, a column with a
    value outside its family's support, or a count column whose mean is 0.
    """
    if not isinstance(family, str) and not (
        isinstance(family, (list, tuple)) and all(isinstance(f, str) for f in family)
    ):
        raise TypeError(f"{method}: family must be a name or a list of names, got {family!r}")
    names = [family] * X.shape[1] if isinstance(family, str) else list(family)
    if len(names) != X.shape[1]:
        raise ValueError(
            f"{method}: family must be one name or one name per column, got {len(names)} names "
            f"for {X.shape[1]} columns"
        )
    out = []
    for j, label in enumerate(names):
        if label not in FAMILIES:
            raise ValueError(
                f"{method}: family must be one of {sorted(FAMILIES)} or a list of them, "
                f"got {label!r}"
            )
        fam = FAMILIES[label]
        bad = np.flatnonzero(fam.outside(X[:, j]))
        if bad.size:
            raise ValueError(
                f"{method}: family {label!r} models {fam.support}, but column {name(j)} holds "
                f"{float(X[bad[0], j])!r} in row {bad[0]}; choose a family whose support holds the "
                "column, or transform the column"
            )
        if fam.counts and not np.any(X[:, j] > 0):
            raise ValueError(f"{method}: column {name(j)} holds only zeros, which {label!r} "
                             "cannot fit")  # fmt: skip
        out.append(fam)
    return out
