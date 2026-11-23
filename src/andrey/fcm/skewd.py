"""Skew-normal location-scale fit of SkewD: one effect given its parents.

SkewD (Klippert and Marx 2026) models an effect ``y`` given its parents ``x`` as skew-normal,
``y | x ~ SN(f(x), g(x), lambda)``, with the density ``2 / g phi(r) Phi(lambda r)`` at
``r = (y - f(x)) / g(x)``. The location ``f = N psi`` and the log variance ``log g^2 = Z rho`` are
additive B-splines of the parents (Eq. 3, Appendix A.2); one shape ``lambda`` holds for every row,
and ``lambda = 0`` is the Gaussian. The estimate maximizes the log-likelihood less second-order
difference penalties on each parent's spline coefficients, ``alpha_t`` on the location's and
``kappa_t`` on the log variance's (Eq. 4), by Algorithm 1:

1. Bayesian optimization chooses the penalties: each candidate's score is the held-out
   log-likelihood, without penalty, of a k-fold cross-validation (Appendix B.1).
2. CMA-ES (Hansen 2016) maximizes the penalized log-likelihood at the chosen penalties from
   several starts: least squares with ``rho = 0`` and ``lambda`` the effect's sample skewness, and
   each fold's estimate at those penalties. With several parents it reads the location
   coefficients as those whose parents' blocks have equal sums, as ECM does: a constant moved
   from one block to another changes no fitted value.
3. Expectation conditional maximization (ECM, Section 4) refines the best of them: the E-step's
   moments of the latent truncated normal, a closed-form step in ``(psi, lambda)``, then the
   maximum over ``rho`` of the expected penalized log-likelihood. That function is concave in
   ``rho``, ``-sum z_i' rho - sum c_i exp(-z_i' rho) / 2 - rho' M rho / 2`` with
   ``c_i = E[(V_i - lambda a_i)^2] + a_i^2 >= 0``, so L-BFGS-B within the bounds reaches its
   maximum; the authors' code takes CMA-ES steps there.

References
----------
Klippert and Marx (2026). Skewness-robust causal discovery in location-scale noise models. ICML.

Hansen (2016). The CMA evolution strategy: a tutorial. arXiv:1604.00772.

Eilers and Marx (1996). Flexible smoothing with B-splines and penalties. Statistical Science 11(2).

Ferreira, Zeller, and Garcia (2023). Heteroscedastic partially linear model under skew-normal
distribution. Journal of Applied Statistics 50(6).

Tarjan and Yannakakis (1984). Simple linear-time algorithms to test chordality of graphs, test
acyclicity of hypergraphs, and selectively reduce acyclic hypergraphs. SIAM Journal on Computing
13(3).
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer

from andrey.core.structure import has_cycle

#: The scale spline's knots and both splines' degree (the authors' code: 5 knots, degree 3, so
#: ``p~ = 7`` columns per parent, Appendix B.1). Each spline extends linearly past the data's range.
SCALE_KNOTS = 5
DEGREE = 3
#: The bounds on each log-variance coefficient and on the shape (Appendix B.1).
RHO_BOUNDS = (-10.0, 5.0)
SHAPE_BOUND = 25.0
#: The variance of a standardized effect under which its rows count as constant.
CONSTANT_VARIANCE = 1e-12
#: The range of each log penalty that the Bayesian optimization searches (the authors' code).
LOG_PENALTY_BOUNDS = (-4.0, 4.0)
#: CMA-ES: the population, the first step size, and the most generations (Appendix B.1).
POPULATION = 100
STEP_SIZE = 1.0
GENERATIONS = 5000
#: Both CMA-ES and ECM stop once the parameters move less than ``TOLERANCE`` in L2 norm over
#: ``WINDOW`` iterations (Appendix B.1, Section 4).
WINDOW = 25
TOLERANCE = 1e-6
#: The most DAGs one undirected component may list while ``max_fits`` caps the conditional models: a
#: sparse component with conflicting colliders has about ``2^edges`` acyclic orientations for a few
#: models per edge, so the cap on models alone does not stop its listing.
LISTED_DAGS = 2**16
#: The expected-improvement margin and the random candidates the acquisition is maximized over.
EI_MARGIN = 0.01
EI_CANDIDATES = 10_000

_LOG_NORM = math.log(2.0) - 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True)
class Budget:
    """The settings of one fit: the location spline's knots, the cross-validation folds, the
    penalty settings the Bayesian optimization scores and how many of them start it at Latin
    hypercube points, and the most ECM iterations."""

    knots: int
    folds: int
    tuning_iter: int
    initial: int
    max_iter: int


@dataclass(frozen=True)
class Fit:
    """One fitted skew-normal model: its mean log-likelihood over the rows, its scaled residual
    ``(y - f(x)) / g(x)``, and whether ECM stopped on its own rule."""

    log_likelihood: float
    residual: np.ndarray
    converged: bool


@dataclass(frozen=True)
class _Design:
    location: np.ndarray  # N, (n, q)
    scale: np.ndarray  # Z, (n, p)
    location_penalty: np.ndarray  # second-order difference penalty of one parent's location block
    scale_penalty: np.ndarray
    parents: int

    def penalties(self, log_penalties: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The block-diagonal penalty matrices ``K_alpha`` and ``M_kappa`` of Appendix A.2."""
        alpha, kappa = np.exp(log_penalties[: self.parents]), np.exp(log_penalties[self.parents :])
        return (
            np.kron(np.diag(alpha), self.location_penalty),
            np.kron(np.diag(kappa), self.scale_penalty),
        )


def _difference_penalty(columns: int) -> np.ndarray:
    difference = np.diff(np.eye(columns), n=2, axis=0)
    return difference.T @ difference


def _design(parents: np.ndarray, knots: int) -> _Design:
    def basis(n_knots: int) -> np.ndarray:
        with _FILTERS:
            return np.hstack(
                [
                    SplineTransformer(
                        n_knots=n_knots, degree=DEGREE, extrapolation="linear"
                    ).fit_transform(parents[:, [t]])
                    for t in range(parents.shape[1])
                ]
            )

    location, scale = basis(knots), basis(SCALE_KNOTS)
    s = parents.shape[1]
    return _Design(
        location,
        scale,
        _difference_penalty(location.shape[1] // s),
        _difference_penalty(scale.shape[1] // s),
        s,
    )


def _row_log_likelihoods(
    theta: np.ndarray, location: np.ndarray, scale: np.ndarray, effect: np.ndarray
) -> np.ndarray:
    """Each row's skew-normal log density, one column per parameter vector in ``theta`` (m, d)."""
    q, p = location.shape[1], scale.shape[1]
    # The operations of ``C - v / 2 - r^2 / 2 + log Phi(lambda r)``, ``r = (y - N psi) e^(-v / 2)``,
    # in that order, written into three arrays instead of a new one per operation.
    log_variance = scale @ theta[:, q : q + p].T
    r = location @ theta[:, :q].T
    np.subtract(effect[:, None], r, out=r)
    other = np.multiply(log_variance, -0.5)
    np.exp(other, out=other)
    np.multiply(r, other, out=r)
    np.multiply(theta[:, -1], r, out=other)
    log_ndtr(other, out=other)
    out = np.multiply(log_variance, 0.5, out=log_variance)
    np.subtract(_LOG_NORM, out, out=out)
    np.square(r, out=r)
    np.multiply(r, 0.5, out=r)
    np.subtract(out, r, out=out)
    return np.add(out, other, out=out)


def _penalized_nll(
    theta: np.ndarray,
    location: np.ndarray,
    scale: np.ndarray,
    effect: np.ndarray,
    k_alpha: np.ndarray,
    m_kappa: np.ndarray,
) -> np.ndarray:
    """The negative log-likelihood plus the penalties (Eq. 4) of each row of ``theta`` (m, d)."""
    q, p = location.shape[1], scale.shape[1]
    psi, rho = theta[:, :q], theta[:, q : q + p]
    value = -_row_log_likelihoods(theta, location, scale, effect).sum(axis=0)
    value += 0.5 * np.einsum("mi,ij,mj->m", psi, k_alpha, psi)
    value += 0.5 * np.einsum("mi,ij,mj->m", rho, m_kappa, rho)
    return np.where(np.isfinite(value), value, np.inf)


def _bounds(q: int, p: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate([np.full(q, -np.inf), np.full(p, RHO_BOUNDS[0]), [-SHAPE_BOUND]])
    upper = np.concatenate([np.full(q, np.inf), np.full(p, RHO_BOUNDS[1]), [SHAPE_BOUND]])
    return lower, upper


def _cma_es(
    objective, start: np.ndarray, lower: np.ndarray, upper: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Minimize ``objective`` (rows of candidates -> values) by CMA-ES from ``start``.

    The (mu/mu_w, lambda) strategy with the default weights and learning rates of Hansen (2016),
    the population :data:`POPULATION`, and the step size :data:`STEP_SIZE`. A candidate is the mean
    plus the step size times the covariance's symmetric square root applied to a standard normal
    draw: the root, unlike the eigenvectors it is built from, has no signs to choose, so a
    covariance that differs in its last digits gives the same candidates up to rounding. A
    candidate outside the bounds is moved to the nearest point inside them, and the update reads
    the moved candidate.
    The learning rates stay at those defaults for the whole search; the paper's runs adapt them
    (Appendix B.1). It returns the best candidate of every generation and stops after
    :data:`GENERATIONS` generations, once the distribution's mean moved less than :data:`TOLERANCE`
    over :data:`WINDOW` generations, or once the steps fall below rounding. The window reads the
    mean, not the best candidate, which stays where it is for as long as no candidate beats it,
    however wide the search still is.
    """
    dim = start.size
    mu = POPULATION // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / float(weights @ weights)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
    mean = np.clip(start, lower, upper)
    sigma = STEP_SIZE
    cov, basis, scales = np.eye(dim), np.eye(dim), np.ones(dim)
    path_c, path_s = np.zeros(dim), np.zeros(dim)
    best, best_value = mean.copy(), float(objective(mean[None, :])[0])
    history = [mean]
    for generation in range(GENERATIONS):
        steps = rng.standard_normal((POPULATION, dim)) @ ((basis * scales) @ basis.T)
        candidates = np.clip(mean + sigma * steps, lower, upper)
        steps = (candidates - mean) / sigma
        values = objective(candidates)
        order = np.argsort(values, kind="stable")
        if values[order[0]] < best_value:
            best, best_value = candidates[order[0]].copy(), float(values[order[0]])
        chosen = steps[order[:mu]]
        step = weights @ chosen
        mean = mean + sigma * step
        inverse_root = (basis / scales) @ basis.T
        path_s = (1 - cs) * path_s + math.sqrt(cs * (2 - cs) * mueff) * (inverse_root @ step)
        norm_s = float(np.linalg.norm(path_s))
        hsig = norm_s / math.sqrt(1 - (1 - cs) ** (2 * (generation + 1))) / chi < 1.4 + 2 / (
            dim + 1
        )
        path_c = (1 - cc) * path_c + hsig * math.sqrt(cc * (2 - cc) * mueff) * step
        cov = (
            (1 - c1 - cmu) * cov
            + c1 * (np.outer(path_c, path_c) + (1 - hsig) * cc * (2 - cc) * cov)
            + cmu * (chosen.T * weights) @ chosen
        )
        try:
            sigma *= math.exp((cs / damps) * (norm_s / chi - 1))
        except OverflowError:  # the covariance lost a direction, so the step size means nothing
            break
        cov = (cov + cov.T) / 2
        eigen, basis = np.linalg.eigh(cov)
        scales = np.sqrt(np.maximum(eigen, 1e-300))
        history.append(mean)
        if generation >= WINDOW and np.linalg.norm(mean - history[-1 - WINDOW]) < TOLERANCE:
            break
        if not math.isfinite(sigma) or sigma * scales.max() < 1e-12:
            break
    return best, best_value


def _start(location: np.ndarray, scale: np.ndarray, effect: np.ndarray) -> np.ndarray:
    """Least squares for the location, ``rho = 0``, and the effect's sample skewness as the shape
    (the authors' code), or a shape of 0 when the effect is constant, a cross-validation fold whose
    training rows hold no variation (the callers standardize, so a variance under
    :data:`CONSTANT_VARIANCE` is rounding)."""
    psi = np.linalg.lstsq(location, effect, rcond=None)[0]
    centered = effect - effect.mean()
    variance = float(np.mean(centered**2))
    skewness = float(np.mean(centered**3) / variance**1.5) if variance > CONSTANT_VARIANCE else 0.0
    return np.concatenate([psi, np.zeros(scale.shape[1]), [np.clip(skewness, -25, 25)]])


def _identified(design: _Design) -> np.ndarray:
    """An orthonormal basis, one column per coordinate, of the location coefficients whose
    parents' blocks have equal sums, for a fit with several parents.

    Each parent's location basis sums to one and the difference penalty ignores constants, so a
    constant moved from one parent's block of ``psi`` to another's changes neither a fitted value
    nor the penalized likelihood. The basis holds each block's centered part (Helmert contrasts)
    and one constant shared by every block: the vectors ECM's location step keeps.
    """
    s, q = design.parents, design.location.shape[1]
    block = q // s
    helmert = np.zeros((block, block - 1))
    for k in range(1, block):
        helmert[:k, k - 1] = 1.0 / math.sqrt(k * (k + 1))
        helmert[k, k - 1] = -k / math.sqrt(k * (k + 1))
    return np.hstack([np.kron(np.eye(s), helmert), np.full((q, 1), 1.0 / math.sqrt(q))])


@dataclass(frozen=True)
class _Search:
    """One CMA-ES search: the design and effect with the rows it reads, the penalties, the start (in
    the coordinates of ``frame`` when it is set), the bounds, and the stream the search draws from.
    The rows are read when the search runs, so a list of searches holds no copy of the data."""

    design: _Design
    effect: np.ndarray
    rows: np.ndarray
    k_alpha: np.ndarray
    m_kappa: np.ndarray
    frame: np.ndarray | None
    start: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    stream: np.random.Generator


#: Runs a list of searches and returns their results in order.
Run = Callable[[list[_Search]], list[tuple[np.ndarray, float]]]


def _run_search(search: _Search) -> tuple[np.ndarray, float]:
    """Run one search; return its best parameters, in the raw coordinates, and its value."""
    frame, rows = search.frame, search.rows
    design, y = search.design, search.effect[rows]
    location, scale = design.location[rows], design.scale[rows]

    def objective(theta: np.ndarray) -> np.ndarray:
        if frame is not None:
            theta = np.hstack([theta[:, : frame.shape[1]] @ frame.T, theta[:, frame.shape[1] :]])
        return _penalized_nll(theta, location, scale, y, search.k_alpha, search.m_kappa)

    best, value = _cma_es(objective, search.start, search.lower, search.upper, search.stream)
    if frame is not None:
        best = np.concatenate([frame @ best[: frame.shape[1]], best[frame.shape[1] :]])
    return best, value


def _serial(searches: list[_Search]) -> list[tuple[np.ndarray, float]]:
    return [_run_search(search) for search in searches]


def _searches(
    design: _Design,
    rows: np.ndarray,
    effect: np.ndarray,
    log_penalties: np.ndarray,
    starts: list[np.ndarray],
    rng: np.random.Generator,
) -> list[_Search]:
    """The CMA-ES searches on ``rows`` from the least-squares start and ``starts``, each with its
    own stream spawned from ``rng``, so the draws of a search do not depend on how many
    generations the searches before it ran, nor on the order or the place they run in.

    With several parents the searches read ``psi`` in the coordinates of :func:`_identified`: along
    a constant moved between the parents' blocks the likelihood is flat, so the distribution's mean
    would wander along it and the stop rule on the mean would not hold. The same constant in
    ``rho`` stays, as its bounds keep it from wandering far.
    """
    location, scale, y = design.location[rows], design.scale[rows], effect[rows]
    k_alpha, m_kappa = design.penalties(log_penalties)
    lower, upper = _bounds(location.shape[1], scale.shape[1])
    points = [_start(location, scale, y), *starts]
    frame = None
    if design.parents > 1:
        q = location.shape[1]
        frame = _identified(design)
        free = np.full(frame.shape[1], np.inf)  # psi has no bounds
        lower, upper = np.concatenate([-free, lower[q:]]), np.concatenate([free, upper[q:]])
        points = [np.concatenate([frame.T @ point[:q], point[q:]]) for point in points]
    return [
        _Search(design, effect, rows, k_alpha, m_kappa, frame, point, lower, upper, stream)
        for point, stream in zip(points, rng.spawn(len(points)), strict=True)
    ]


def _heuristic(
    design: _Design,
    rows: np.ndarray,
    effect: np.ndarray,
    log_penalties: np.ndarray,
    starts: list[np.ndarray],
    rng: np.random.Generator,
    run: Run = _serial,
) -> tuple[np.ndarray, float]:
    """The best CMA-ES estimate on ``rows`` over ``starts`` (the least-squares start first)."""
    return _best(run(_searches(design, rows, effect, log_penalties, starts, rng)))


@contextmanager
def search_pool(workers: int) -> Generator[Run]:
    """A ``run`` function for :func:`fit` that spreads searches over ``workers`` threads; with one
    worker, the searches run in this thread. A search spends its time in NumPy and SciPy array
    operations, which release the interpreter lock. Every search draws from its own stream, so the
    results do not depend on the number of workers."""
    if workers <= 1:
        yield _serial
        return
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield lambda searches: list(pool.map(_run_search, searches))


def _best(found: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
    """The search result with the lowest value, the first on a tie."""
    return min(found, key=lambda f: f[1])


#: Held around every scikit-learn call that changes the warning filters on a thread that runs a
#: fit (the spline bases, and the Gaussian process's fit and prediction): scikit-learn and the
#: penalty search enter :func:`warnings.catch_warnings`, which saves and restores the filters of
#: the whole process, so two such blocks in concurrent threads must not overlap.
_FILTERS = threading.Lock()


def _reset_after_fork() -> None:
    """Give a forked child an unlocked :data:`_FILTERS`: a child forked while another thread holds
    it inherits it locked, with no thread left to release it."""
    global _FILTERS
    _FILTERS = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _tune(
    design: _Design,
    effect: np.ndarray,
    budget: Budget,
    rng: np.random.Generator,
    run: Run = _serial,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The log penalties with the best cross-validated held-out log-likelihood, and each fold's
    estimate at them.

    The first ``budget.initial`` candidates are Latin hypercube points over
    :data:`LOG_PENALTY_BOUNDS`; each later one maximizes the expected improvement over a Gaussian
    process of the scores so far, among :data:`EI_CANDIDATES` uniform points. Every candidate
    reads the same folds. ``run`` runs a list of searches and returns their results in order; the
    Latin hypercube candidates' searches go to it together, and each later candidate's folds.
    """
    dim = 2 * design.parents
    low, high = LOG_PENALTY_BOUNDS
    folds = list(
        KFold(budget.folds, shuffle=True, random_state=int(rng.integers(2**31))).split(effect)
    )
    points: list[np.ndarray] = []
    scores: list[float] = []
    estimates: list[list[np.ndarray]] = []

    def score(candidates: np.ndarray) -> None:
        searches = [
            search
            for log_penalties in candidates
            for train, _ in folds
            for search in _searches(design, train, effect, log_penalties, [], rng)
        ]
        found = iter(run(searches))
        for log_penalties in candidates:
            held_out, fold_estimates = 0.0, []
            for _, test in folds:
                theta, _ = next(found)
                rows = _row_log_likelihoods(
                    theta[None, :], design.location[test], design.scale[test], effect[test]
                )
                held_out -= float(rows.sum())
                fold_estimates.append(theta)
            points.append(log_penalties)
            scores.append(held_out / len(folds))
            estimates.append(fold_estimates)

    sampler = qmc.LatinHypercube(d=dim, seed=int(rng.integers(2**31)))
    score(qmc.scale(sampler.random(budget.initial), [low] * dim, [high] * dim))
    kernel = (
        ConstantKernel(1.0, (0.01, 1000.0)) * Matern(np.ones(dim), (0.01, 100.0), nu=2.5)
        + WhiteKernel()
    )
    for _ in range(budget.tuning_iter - budget.initial):
        process = GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=2, random_state=int(rng.integers(2**31))
        )
        values = np.array(scores)
        finite = np.isfinite(values)
        candidates = rng.uniform(low, high, size=(EI_CANDIDATES, dim))
        with _FILTERS, warnings.catch_warnings():
            # A kernel parameter at its bound only shapes the acquisition, not the fit.
            warnings.simplefilter("ignore", ConvergenceWarning)
            process.fit(np.array(points)[finite], values[finite])
            mean, sd = process.predict(candidates, return_std=True)
        gain = values[finite].min() - mean - EI_MARGIN
        z = gain / np.maximum(sd, 1e-12)
        improvement = gain * _ndtr(z) + sd * np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        score(candidates[[int(np.argmax(improvement))]])
    best = int(np.argmin(scores))
    return points[best], estimates[best]


def _ndtr(z: np.ndarray) -> np.ndarray:
    return np.exp(log_ndtr(z))


def _ecm(
    design: _Design,
    effect: np.ndarray,
    theta: np.ndarray,
    log_penalties: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, bool]:
    """Refine ``theta`` by ECM (Section 4) for at most ``max_iter`` iterations; return it and
    whether it stopped on the paper's rule.

    Each parent's location basis sums to one and the difference penalty ignores constants, so with
    several parents the likelihood leaves free a constant moved from one parent's coefficients to
    another's. The location step then takes the solution whose blocks have equal sums, which sets
    ``psi`` without changing a fitted value, so the rule on the parameters can be met.
    """
    location, scale = design.location, design.scale
    q, p = location.shape[1], scale.shape[1]
    k_alpha, m_kappa = design.penalties(log_penalties)
    bounds = [RHO_BOUNDS] * p
    psi, rho, shape = theta[:q].copy(), theta[q : q + p].copy(), float(theta[-1])
    block = q // design.parents
    ties = np.zeros((design.parents - 1, q))  # block 0's sum less each other block's
    ties[:, :block] = 1.0
    for t in range(1, design.parents):
        ties[t - 1, t * block : (t + 1) * block] = -1.0
    if design.parents > 1:
        psi -= ties.T @ np.linalg.solve(ties @ ties.T, ties @ psi)
    history = [np.concatenate([psi, rho, [shape]])]
    for _ in range(max_iter):
        residual = effect - location @ psi
        variance = np.exp(scale @ rho)
        sd = np.sqrt(variance)
        u = shape * residual / sd
        ratio = np.exp(-0.5 * u**2 - 0.5 * math.log(2 * math.pi) - log_ndtr(u))  # phi / Phi
        v = shape * residual + sd * ratio
        v2 = (shape * residual) ** 2 + variance + shape * sd * residual * ratio
        weight = 1.0 / variance
        shrink = 1.0 + shape**2
        lhs = location.T @ (location * weight[:, None]) + k_alpha / shrink
        rhs = location.T @ (weight * (effect - shape / shrink * v))
        if design.parents > 1:
            lhs += np.trace(lhs) / q * (ties.T @ ties)
        try:
            psi_next = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            psi_next = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        # CM-step 1 updates psi and the shape together, the shape from the residual at the
        # current psi (Section 4); only the scale's step reads the new psi.
        shape = float((residual * weight) @ v / ((residual * weight) @ residual))
        shape = float(np.clip(shape, -SHAPE_BOUND, SHAPE_BOUND))
        psi = psi_next
        a = effect - location @ psi
        c = v2 - 2 * shape * a * v + (1 + shape**2) * a**2

        def negative_q(r: np.ndarray, c=c) -> tuple[float, np.ndarray]:
            eta = scale @ r
            spread = c * np.exp(-eta)
            value = eta.sum() + 0.5 * spread.sum() + 0.5 * r @ m_kappa @ r
            gradient = scale.T @ (1.0 - 0.5 * spread) + m_kappa @ r
            return float(value), gradient

        rho = minimize(negative_q, rho, jac=True, method="L-BFGS-B", bounds=bounds).x
        history.append(np.concatenate([psi, rho, [shape]]))
        if len(history) > WINDOW and np.linalg.norm(history[-1] - history[-1 - WINDOW]) < TOLERANCE:
            return history[-1], True
    return history[-1], False


def fit(
    parents: np.ndarray, effect: np.ndarray, budget: Budget, seed: int, run: Run = _serial
) -> Fit:
    """Fit the skew-normal location-scale model of ``effect`` given ``parents`` by Algorithm 1.

    ``parents`` is ``(n, s)`` and ``effect`` ``(n,)``, both standardized by the caller. The fit
    draws its random numbers from ``seed`` alone, so the same data and seed give the same fit,
    whatever ``run`` (a function that runs a list of searches and returns their results in order)
    runs the searches on.
    """
    rng = np.random.default_rng(seed)
    design = _design(parents, budget.knots)
    log_penalties, fold_estimates = _tune(design, effect, budget, rng, run)
    every = np.arange(effect.size)
    theta, _ = _best(run(_searches(design, every, effect, log_penalties, fold_estimates, rng)))
    theta, converged = _ecm(design, effect, theta, log_penalties, budget.max_iter)
    q, p = design.location.shape[1], design.scale.shape[1]
    rows = _row_log_likelihoods(theta[None, :], design.location, design.scale, effect)[:, 0]
    residual = (effect - design.location @ theta[:q]) * np.exp(
        -0.5 * design.scale @ theta[q : q + p]
    )
    return Fit(float(rows.mean()), residual, converged)


def class_members(marks: np.ndarray, *, collider_check: bool = True) -> Iterator[np.ndarray]:
    """Yield each DAG that orients the edges a graph leaves without one direction, as 0/1
    adjacencies.

    ``marks`` holds Andrey's endpoint marks (``marks[i, j]`` the mark at ``i`` on the edge
    ``i-j``): a directed edge keeps its direction, and an undirected or bidirected one takes either.
    A DAG qualifies when it is acyclic and, with ``collider_check``, makes no collider ``a -> c <-
    b`` of nonadjacent ``a`` and ``b`` that the graph does not already hold, so for a CPDAG the DAGs
    are the members of its Markov equivalence class. The edges are oriented one at a time and a
    partial orientation that already breaks a condition is dropped, so the DAGs come one at a time
    and a caller that stops reading stops the search. The edges are taken vertex by vertex in a
    maximum cardinality search order (Tarjan and Yannakakis 1984), each vertex's edges to the
    vertices before it: in a chordal component those vertices are adjacent to each other, so the
    vertex can always be made their common child, and an orientation of the vertices so far
    always extends to a DAG. A dead end then costs at most the orientations of one vertex's
    edges, and the work follows the number of DAGs whatever the column order.
    """
    d = marks.shape[0]
    adjacent = (marks != 0) & (marks.T != 0)
    fixed = (marks == 1) & (marks.T == 2)  # tail at i, arrow at j: i -> j
    undirected = adjacent & ~fixed & ~fixed.T
    position = np.full(d, -1)
    visited = np.zeros(d, dtype=int)  # each vertex's neighbors already placed
    for k in range(d):
        weight = np.where(position < 0, visited, -1)
        node = int(np.argmax(weight))  # the most placed neighbors, then the first column
        position[node] = k
        visited += undirected[node]
    free = sorted(
        ((i, j) for i in range(d) for j in range(i + 1, d) if undirected[i, j]),
        key=lambda edge: (max(position[list(edge)]), min(position[list(edge)])),
    )

    def reaches(dag: np.ndarray, source: int, target: int) -> bool:
        seen, stack = {source}, [source]
        while stack:
            for nxt in np.flatnonzero(dag[stack.pop()]):
                if nxt == target:
                    return True
                if nxt not in seen:
                    seen.add(int(nxt))
                    stack.append(int(nxt))
        return False

    def new_collider(dag: np.ndarray, head: int, tail: int) -> bool:
        others = np.flatnonzero(dag[:, head])
        return any(o != tail and not adjacent[o, tail] for o in others)

    def extend(k: int, dag: np.ndarray) -> Iterator[np.ndarray]:
        if k == len(free):
            yield dag.copy()
            return
        for tail, head in (free[k], free[k][::-1]):
            if reaches(dag, head, tail) or (collider_check and new_collider(dag, head, tail)):
                continue
            dag[tail, head] = 1
            yield from extend(k + 1, dag)
            dag[tail, head] = 0

    # Each free edge is checked against every parent its head has when it is oriented, so a
    # collider of two fixed edges, the graph's own, is the only one never checked.
    yield from extend(0, fixed.astype(np.int8))


def _acyclic(adjacency: np.ndarray) -> bool:
    reach = adjacency.astype(bool)
    for k in range(len(reach)):
        reach |= reach[:, k, None] & reach[None, k, :]
    return not np.diagonal(reach).any()


def class_candidates(
    marks: np.ndarray, max_fits: int | None
) -> tuple[list[tuple[list[int], list[np.ndarray]]], np.ndarray, list[tuple[int, tuple[int, ...]]]]:
    """The class's directed part, its undirected components with their DAGs, and the conditional
    models they need.

    The undirected edges of ``marks`` fall into connected components, and the class's DAGs are every
    combination of one orientation of each component, so each component is listed on its own: its
    nodes and the DAGs (0/1 adjacencies that hold the directed edges too) that orient its edges
    with no directed cycle and no new collider, or, when none does, as conflicting colliders can
    leave, every acyclic orientation of them. The first array holds the directed edges alone, the
    DAG of a class with no undirected edge. A conditional model is a node of a component with at
    least one parent, fitted given a parent set. A component's DAGs come sorted by the directions
    of its edges in column order, each edge from its lower column first, so their order, which
    decides ties, does not depend on the order of the listing. The DAGs of a component are listed
    one at a time, and the listing stops with a ``ValueError`` as soon as the components together
    need more than ``max_fits`` conditional models or one component lists more than
    :data:`LISTED_DAGS` DAGs (``None`` sets neither cap), so a class too large to score is never
    listed in full. A directed cycle among the directed edges also raises ``ValueError``.
    """
    d = marks.shape[0]
    adjacent = (marks != 0) & (marks.T != 0)
    fixed = (marks == 1) & (marks.T == 2)
    base = fixed.astype(np.int8)
    if not _acyclic(base):
        raise ValueError("skewd_mv: PC's graph orients a directed cycle, so no DAG keeps its edges")
    undirected = adjacent & ~fixed & ~fixed.T
    seen = np.zeros(d, dtype=bool)
    groups: list[list[int]] = []
    for start in range(d):
        if seen[start] or not undirected[start].any():
            continue
        stack, group = [start], []
        seen[start] = True
        while stack:
            node = stack.pop()
            group.append(node)
            for other in np.flatnonzero(undirected[node] & ~seen):
                seen[other] = True
                stack.append(int(other))
        groups.append(sorted(group))
    fits: set[tuple[int, tuple[int, ...]]] = set()
    components: list[tuple[list[int], list[np.ndarray]]] = []
    for group in groups:
        elsewhere = undirected.copy()  # the other components' edges leave this listing
        elsewhere[np.ix_(group, group)] = False
        part = marks.copy()
        part[elsewhere] = 0
        for collider_check in (True, False):
            members: list[np.ndarray] = []
            for dag in class_members(part, collider_check=collider_check):
                if max_fits is not None and len(members) == LISTED_DAGS:
                    raise ValueError(
                        f"skewd_mv: one undirected component of PC's graph ({len(group)} "
                        f"variables) has more than {LISTED_DAGS} orientations to score; set "
                        "max_fits to None to list them all"
                    )
                members.append(dag)
                for y in group:
                    parents = tuple(np.flatnonzero(dag[:, y]).tolist())
                    if parents:
                        fits.add((y, parents))
                if max_fits is not None and len(fits) > max_fits:
                    raise ValueError(
                        f"skewd_mv: the equivalence class of the data needs more than {max_fits} "
                        f"conditional models (the listing stopped after {len(members)} DAGs of "
                        "one component); raise max_fits, or set it to None, to fit them all"
                    )
            if members:
                break
        if not members:
            raise ValueError(
                "skewd_mv: PC's graph orients a directed cycle, so no DAG keeps its edges"
            )
        edges = np.argwhere(np.triu(undirected[np.ix_(group, group)]))
        tails, heads = np.array(group)[edges[:, 0]], np.array(group)[edges[:, 1]]
        members.sort(key=lambda dag: dag[heads, tails].tobytes())
        components.append((group, members))
    return components, base, sorted(fits)


def _coupled_groups(
    base: np.ndarray, components: list[tuple[list[int], list[np.ndarray]]]
) -> list[list[int]]:
    """Group the components that can lie on one directed cycle: those that touch the same
    strongly connected set of the graph of every edge some orientation can hold."""
    reach = base.astype(bool)
    for _, members in components:
        for member in members:
            reach = reach | member.astype(bool)
    reach = reach | np.eye(len(reach), dtype=bool)
    for k in range(len(reach)):
        reach |= reach[:, k, None] & reach[None, k, :]
    label = np.argmax(reach & reach.T, axis=1)  # a node's smallest mutually reachable node
    parent = list(range(len(components)))

    def find(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    holder: dict[int, int] = {}
    for k, (nodes, _) in enumerate(components):
        for node_label in set(label[nodes].tolist()):
            parent[find(k)] = find(holder.setdefault(node_label, k))
    groups: dict[int, list[int]] = {}
    for k in range(len(components)):
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def _best_acyclic(
    base: np.ndarray, members: list[list[np.ndarray]], scores: list[list[float]]
) -> list[int] | None:
    """The orientation of each component, one index per component, that keeps ``base`` acyclic and
    has the highest total score (the first found on a tie), or ``None`` when none is acyclic.

    A depth-first search takes each component's orientations from the best score down and drops a
    partial choice as soon as it closes a cycle (more edges never open one) or its score plus the
    best of every remaining component cannot beat the best total found, so it lists no combination
    that one of these rules rules out.
    """
    order = [sorted(range(len(row)), key=lambda i: -row[i]) for row in scores]
    after = [0.0] * (len(scores) + 1)  # after[k]: the best total of components k and later
    for k in range(len(scores) - 1, -1, -1):
        after[k] = after[k + 1] + scores[k][order[k][0]]
    best: list[int] | None = None
    best_total = -math.inf
    choice = [0] * len(scores)

    def search(k: int, graph: np.ndarray, total: float) -> None:
        nonlocal best, best_total
        if k == len(scores):
            if total > best_total:
                best, best_total = choice.copy(), total
            return
        for i in order[k]:
            if total + scores[k][i] + after[k + 1] <= best_total:
                break  # the later orientations score no higher
            merged = np.maximum(graph, members[k][i])
            if not has_cycle(merged):
                choice[k] = i
                search(k + 1, merged, total + scores[k][i])

    search(0, base, 0.0)
    return best


def class_size(
    base: np.ndarray,
    components: list[tuple[list[int], list[np.ndarray]]],
    limit: int | None = None,
) -> int:
    """The number of DAGs that hold ``base`` and one orientation of each component.

    It is the product of the components' counts, except that components on a common cycle
    (:func:`_coupled_groups`) contribute their acyclic combinations, counted one at a time: with
    ``limit``, such a group with more than ``limit`` of them raises ``ValueError``. A count of 0
    says the graph leaves no DAG.
    """

    def acyclic(members: list[list[np.ndarray]], k: int, graph: np.ndarray) -> int:
        if k == len(members):
            return 1
        count = 0
        for member in members[k]:
            merged = np.maximum(graph, member)
            if not has_cycle(merged):
                count += acyclic(members, k + 1, merged)
                if limit is not None and count > limit:
                    raise ValueError(
                        f"skewd_mv: {len(members)} undirected components of PC's graph that "
                        f"share a cycle have more than {limit} acyclic combinations to count; "
                        "set max_fits to None to count them all"
                    )
        return count

    size = 1
    for group in _coupled_groups(base, components):
        members = [components[k][1] for k in group]
        size *= len(members[0]) if len(group) == 1 else acyclic(members, 0, base)
    return size


def merge_components(
    base: np.ndarray,
    components: list[tuple[list[int], list[np.ndarray]]],
    scores: list[list[float]],
) -> np.ndarray:
    """The DAG that holds the directed edges ``base`` and the highest-scoring orientation of each
    component (the first on a tie), as 0/1 adjacency.

    ``scores[k][i]`` is the log-likelihood the ``i``-th DAG of ``components[k]`` gives its nodes.
    The components are independent, so the best of each combine into the best DAG, unless a graph
    with conflicting orientations lets them close a directed cycle across its directed edges. Only
    components on a common cycle depend on each other, so the components are grouped by the
    strongly connected sets of every edge they could hold and each group takes its acyclic
    combination with the highest total, exactly (:func:`_best_acyclic`). The search stays short
    while the best orientations of a group are nearly acyclic and grows with the number of its
    components that conflict. A group with no acyclic combination raises ``ValueError``.
    """
    out = base.copy()
    for group in _coupled_groups(base, components):
        members = [components[k][1] for k in group]
        choice = _best_acyclic(base, members, [scores[k] for k in group])
        if choice is None:
            raise ValueError(
                "skewd_mv: PC's graph leaves no DAG that keeps its directed edges and adds no "
                "collider"
            )
        for k, i in zip(group, choice, strict=True):
            out = np.maximum(out, components[k][1][i])
    return out
