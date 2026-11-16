"""The pairwise likelihood-ratio exogeneity measures for LiNGAM ordering.

Scores each candidate variable by how plausibly exogenous it is and peels the most-exogenous
variable one at a time. Three approximations of the pairwise likelihood ratios of Hyvarinen & Smith
(2013) build the score (:data:`MEASURES`): ``"pwling"``, their maximum-entropy approximation (Eqs.
2-3), evaluated in batch over a stack of residuals; and ``"tanh"`` and ``"skew"``, their
first-order approximations by a nonlinear correlation of the standardized columns (Eqs. 5, 10, 13,
and 18), which need one matrix product per step. Shared by the LiNGAM ordering methods
(DirectLiNGAM, MultiGroupDirectLiNGAM, the temporal LiNGAMs).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from andrey.core import stats
from andrey.core.ties import tie_ranks

#: The measures a DirectLiNGAM order search takes, in the order the facades list them.
MEASURES = ("pwling", "skew", "tanh")

# A score is ``-sum_j min(0, diff_j)^2``, so ``sqrt(-score)`` is the norm of the negative entropy
# differences, and rounding moves it by at most the norm of their rounding errors. Mathematically
# tied candidates measure < 1e-12 apart in this norm and distinct ones > 5e-10 apart. The
# nonlinear correlations of ``"skew"`` and ``"tanh"`` are sums of products whose rounding errors
# are near 1e-16, well inside it.
_TIE_TOL = 1e-10
# Two columns are copies when one leaves at most this fraction of the other's variance
# unexplained (1 - corr^2): their pairwise residuals are then zero or rounding noise, whose
# entropy differs by CPU. Copies, exact or perturbed by 1e-12, measure < 2e-15 at every step;
# rank-transformed near-copies with |corr| up to 0.999999 measure > 7e-6, and scoring both columns
# of a pair gave orders that differed between CPUs only below 1e-14.
COPY_TOL = 1e-12
# :func:`pruned_scores` skips the residual entropies a pick cannot need, which is exact only when
# every skipped one is finite: a NaN score would rank highest in :func:`pick`. It is finite when the
# residual of two standardized columns keeps at least this fraction of their variance (``1 -
# corr^2``), so its standard deviation exceeds its rounding error by over nine orders of magnitude,
# and when there are fewer than ``_MAX_ROWS`` rows: a standardized column has ``|u| <= sqrt(n)``,
# and ``cosh(u)`` overflows only past ``|u| = 710``.
_SAFE_RESIDUAL = 1e-6
_MAX_ROWS = 500_000


def compute_residual(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Least-squares residual of ``a`` regressed on ``b``: ``a - (cov(a, b) / var(b)) * b``.

    Covariance and variance share ``ddof=0``, so the residual is uncorrelated with ``b`` in the
    sample, as the order search assumes.
    """
    return a - (np.cov(a, b, ddof=0)[0, 1] / np.var(b)) * b


def score_candidates(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Score how plausibly exogenous each column in ``U`` of ``X`` is (higher = more exogenous).

    Returns an array aligned with ``U``: ``score(i) = -sum_j min(0, diff_mi(i, j))^2`` with
    ``diff_mi(i, j) = (H(x_j) + H(r_{i|j})) - (H(x_i) + H(r_{j|i}))`` on standardized columns.
    Each ``H(r_{i|j})`` is computed once and serves both ``i`` and ``j``, batched over the residuals
    of one column on the others; :func:`pick` chooses the best, MultiGroup sums across groups
    first. Order-exactness needs the float64 entropy path -- on a float32 device (atol 1e-3) a
    near-tie could flip the pick, so pin ``ANDREY_DEVICE=numpy`` for exact order.
    """
    if len(U) < 2:
        return np.zeros(len(U), dtype=np.float64)  # a lone candidate has no pair to score
    x_std = _standardized(X, U)  # (n, m) standardized candidate columns
    ent = stats.entropy(x_std)  # (m,) batched candidate entropy
    cov = np.cov(x_std, rowvar=False, ddof=0)  # (m, m), same ddof as var: least-squares slopes
    var = np.var(x_std, axis=0)  # (m,), ddof=0
    n, m = x_std.shape
    # h[i, j] = H(r_{i|j}), computed once per ordered pair: candidate i reads its H(r_{i|j}) terms
    # from row i and its H(r_{j|i}) terms from column i.
    h = np.zeros((m, m), dtype=np.float64)
    r, work = np.empty((n, m - 1), order="F"), np.empty((n, m - 1), order="F")
    for a in range(m):
        others = np.arange(m) != a
        np.multiply(cov[a, others] / var[others], x_std[:, others], out=r)
        np.subtract(x_std[:, a][:, None], r, out=r)  # residual of i on each j
        r /= np.std(r, axis=0)
        h[a, others] = stats.entropy(r, work=work)
    return np.array([_score_of(ent, h, a) for a in range(m)], dtype=np.float64)


def _score_of(ent: np.ndarray, h: np.ndarray, a: int) -> np.float64:
    """Candidate ``a``'s score from the entropies ``ent`` and ``h[i, j] = H(r_{i|j})``."""
    others = np.arange(len(ent)) != a
    diff = (ent[others] + h[a, others]) - (ent[a] + h[others, a])
    return -np.sum(np.minimum(0.0, diff) ** 2)


def pruned_scores(
    groups: list[np.ndarray],
    U: list[int],
    *,
    weights: list[float] | None = None,
    allowed: np.ndarray | None = None,
    hint: dict[int, float] | None = None,
) -> np.ndarray:
    """The :func:`score_candidates` scores :func:`pick` reads, computing no residual entropy more.

    With one group and no ``weights``, the scores are ``score_candidates(groups[0], U)``; with
    ``weights``, ``sum_g weights[g] * score_candidates(groups[g], U)``, added in group order. Only
    the candidates in ``allowed`` (a mask aligned with ``U``, all by default) are scored; the others
    get ``-inf``. Each score is ``-sum_j t_j`` with every term ``t_j >= 0``, so a candidate whose
    partial sum, less a rounding margin, already puts its ``sqrt(-score)`` more than ``_TIE_TOL``
    above the best complete candidate's can neither be picked nor tie. Such a candidate stops there
    and gets ``-`` that lower bound, which lies at or above its score and still out of the tie. A
    candidate that completes gets the bits of the full computation, so the pick is unchanged.

    The first candidate is completed; then each round adds one pair to every candidate still open,
    in batches as wide as one row of :func:`score_candidates`. Candidates and their pairs go in
    descending ``hint`` (a column's score at the last step; unknown columns first), so the likely
    pick completes first; the order changes how many entropies are computed, never the result. A
    step falls back to the full computation when its entropy batches would run on a torch device,
    a pair of columns is within ``_SAFE_RESIDUAL`` of a copy, an entropy is not finite, or a group
    has ``_MAX_ROWS`` rows or more.
    """
    m = len(U)
    width = (m - 1) // 2  # pairs per entropy call, as wide as one row of score_candidates
    if width < 1 or any(stats.entropy_device(len(X) * (m - 1)) != "numpy" for X in groups):
        return _full_scores(groups, U, weights)
    parts = []
    for X in groups:
        x_std = _standardized(X, U)
        ent = stats.entropy(x_std)
        cov = np.cov(x_std, rowvar=False, ddof=0)
        var = np.var(x_std, axis=0)
        near = cov * cov.T >= (1.0 - _SAFE_RESIDUAL) * np.outer(var, var)
        np.fill_diagonal(near, False)
        finite = np.isfinite(ent).all() and np.isfinite(cov).all()
        if len(X) >= _MAX_ROWS or near.any() or not finite:
            return _full_scores(groups, U, weights)
        parts.append((x_std, ent, cov, var))
    w = [1.0] * len(groups) if weights is None else weights
    # The terms are the full computation's floats. Summing and weighting them rounds the partial sum
    # and the full score fewer than 3 * groups * m times in all, each by at most eps / 2 of the
    # nonnegative value, so the full score exceeds the partial sum less this margin.
    margin = 4.0 * np.finfo(np.float64).eps * len(groups) * m
    h = [np.zeros((m, m), dtype=np.float64) for _ in groups]
    known = np.eye(m, dtype=bool)
    partial = np.zeros(m, dtype=np.float64)
    visit = np.array(sorted(range(m), key=lambda a: -hint.get(U[a], 0.0)) if hint else range(m))
    open_ = visit if allowed is None else visit[allowed[visit]]
    scores = np.full(m, -np.inf, dtype=np.float64)
    best = np.inf
    while len(open_):
        done = known[open_].all(axis=1)
        for a in open_[done]:
            total = None
            for g, (_, ent, _, _) in enumerate(parts):
                s = _score_of(ent, h[g], a)
                total = s if weights is None else (0.0 if total is None else total) + w[g] * s
            scores[a] = total
            best = min(best, np.sqrt(-total))
        bound = partial[open_] * (1.0 - margin)
        out = ~done & (np.sqrt(bound) > best + _TIE_TOL)
        scores[open_[out]] = -bound[out]
        open_ = open_[~done & ~out]
        if not len(open_):
            break
        if best == np.inf:  # complete the likeliest pick first: its score bounds the others
            J = visit[~known[open_[0], visit]]
            A = np.full(len(J), open_[0])
        else:  # one more pair for every open candidate: its first unknown one in visit order
            first = np.argmax(~known[np.ix_(open_, visit)], axis=1)
            pairs = np.sort(np.column_stack([open_, visit[first]]), axis=1)
            A, J = np.unique(pairs, axis=0).T
        for k in range(0, len(A), width):
            a, j = A[k : k + width], J[k : k + width]
            for g, (x_std, ent, cov, var) in enumerate(parts):
                h[g][a, j], h[g][j, a] = _pair_entropies(x_std, cov, var, a, j)
                diff = (ent[j] + h[g][a, j]) - (ent[a] + h[g][j, a])
                np.add.at(partial, a, w[g] * np.minimum(0.0, diff) ** 2)
                np.add.at(partial, j, w[g] * np.minimum(0.0, -diff) ** 2)
            known[a, j] = known[j, a] = True
    return scores


def _full_scores(groups: list[np.ndarray], U: list[int], weights: list[float] | None) -> np.ndarray:
    """Every candidate's score, as :func:`pruned_scores` defines it, with no pruning."""
    if weights is None:
        return score_candidates(groups[0], U)
    total = np.zeros(len(U), dtype=np.float64)
    for X, w in zip(groups, weights, strict=True):
        total += w * score_candidates(X, U)
    return total


def _pair_entropies(
    x: np.ndarray, cov: np.ndarray, var: np.ndarray, A: np.ndarray, J: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``H(r_{A|J})`` and ``H(r_{J|A})`` for the column pairs ``(A[k], J[k])`` of ``x``.

    The same operations as one row of :func:`score_candidates` on Fortran-ordered columns, whose
    sums run down each column alone, so every value has the full computation's bits.
    """
    p = len(A)
    r = np.empty((len(x), 2 * p), dtype=np.float64, order="F")
    np.multiply(cov[A, J] / var[J], x[:, J], out=r[:, :p])
    np.subtract(x[:, A], r[:, :p], out=r[:, :p])
    np.multiply(cov[J, A] / var[A], x[:, A], out=r[:, p:])
    np.subtract(x[:, J], r[:, p:], out=r[:, p:])
    r /= np.std(r, axis=0)
    e = stats.entropy(r)
    return e[:p], e[p:]


def _standardized(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Columns ``U`` of ``X`` with mean 0 and standard deviation 1 (``ddof=0``)."""
    sub = X[:, np.asarray(U)]
    return (sub - sub.mean(axis=0)) / sub.std(axis=0)


def nonlinear_correlations(x: np.ndarray, g: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """The matrix ``M`` of Eq. 18 for standardized columns ``x``, shape ``(n, m)``.

    ``M[i, j] = rho_ij * (E[x_i g(x_j)] - E[g(x_i) x_j])`` with ``rho`` the correlation matrix of
    the columns and ``g`` the nonlinearity. ``M`` is antisymmetric with a zero diagonal. A positive
    entry says ``i`` precedes ``j``, and the row of an exogenous variable is non-negative.
    """
    n = x.shape[0]
    a = x.T @ g(x) / n  # a[i, j] = E[x_i g(x_j)]
    return (x.T @ x / n) * (a - a.T)


def _violations(x: np.ndarray, g: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Score each column by ``-sum_j min(0, M[i, j])^2`` (Eq. 19, Sec. 3.2.2): its violations."""
    return -np.sum(np.minimum(0.0, nonlinear_correlations(x, g)) ** 2, axis=1)


def _negative_square(u: np.ndarray) -> np.ndarray:
    """The skew measure's ``g(u) = -u^2`` (Sec. 3.2.1)."""
    return -(u * u)


def score_tanh(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Score how plausibly exogenous each column in ``U`` of ``X`` is by the tanh measure.

    The entries are ``rho * E[x_i tanh(x_j) - tanh(x_i) x_j]`` of the standardized columns (Eq.
    5), positive when ``i`` precedes ``j`` for super-Gaussian (positive-kurtosis) variables; for
    sub-Gaussian ones the sign reverses (Eq. 7 and Theorem 1). Returns an array aligned with ``U``,
    higher = more exogenous, as :func:`score_candidates` does.
    """
    if len(U) < 2:
        return np.zeros(len(U), dtype=np.float64)  # a lone candidate has no pair to score
    return _violations(_standardized(X, U), np.tanh)


def score_skew(X: np.ndarray, U: list[int]) -> np.ndarray:
    """Score how plausibly exogenous each column in ``U`` of ``X`` is by the skew measure.

    The entries are ``rho * E[x_i^2 x_j - x_i x_j^2]`` (Eq. 10) of the standardized columns, each
    first multiplied by the sign of its skewness (Eq. 13, ``+1`` for a skewness of ``0``) so that
    ``rho`` is the correlation of the multiplied columns; the entry is then positive when ``i``
    precedes ``j`` (Theorem 2), whatever the sign of the skewness. A column of symmetric noise has
    skewness near ``0``, and the measure then says little about it. Returns an array aligned with
    ``U``, higher = more exogenous, as :func:`score_candidates` does.
    """
    if len(U) < 2:
        return np.zeros(len(U), dtype=np.float64)
    x = _standardized(X, U)
    x = x * np.where(np.mean(x**3, axis=0) < 0, -1.0, 1.0)
    return _violations(x, _negative_square)


def scorer(measure: str) -> Callable[[np.ndarray, list[int]], np.ndarray]:
    """The function that scores the candidates of ``measure``, one of :data:`MEASURES`.

    Called as ``scorer(measure)(X, U)``, it returns :func:`score_candidates`-style scores of the
    columns ``U`` of ``X``. Raises ``ValueError`` for any other value.
    """
    if measure not in MEASURES:
        options = ", ".join(repr(m) for m in MEASURES)
        raise ValueError(f"measure must be one of {options}, got {measure!r}")
    return {"pwling": score_candidates, "skew": score_skew, "tanh": score_tanh}[measure]


def pick(scores: np.ndarray, candidates: list[int], ranks: np.ndarray) -> int:
    """The candidate with the highest score; the lowest ``ranks`` entry among tied candidates.

    ``scores`` are aligned with ``candidates`` and ``<= 0`` or NaN; two tie when their
    ``sqrt(-score)`` values both lie within ``_TIE_TOL`` of the best one. A NaN score ranks highest,
    as under ``np.argmax``; all NaN candidates then tie. ``ranks`` is indexed by column, as
    :func:`~andrey.core.ties.tie_ranks` returns for the input data.
    """
    norm = np.sqrt(-np.asarray(scores, dtype=np.float64))
    nan = np.isnan(norm)
    tied = np.flatnonzero(nan) if nan.any() else np.flatnonzero(norm <= norm.min() + _TIE_TOL)
    return min((candidates[t] for t in tied), key=lambda c: ranks[c])


def copy_groups(groups: list[np.ndarray], U: list[int], ranks: np.ndarray) -> dict[int, list[int]]:
    """Each candidate in ``U`` that represents its copies, mapped to those copies.

    Two columns are copies when, in any of ``groups``, one leaves at most ``COPY_TOL`` of the
    other's variance unexplained; a copy group is a connected set of such pairs. Its
    representative is the column that ranks first by ``ranks`` (as :func:`pick` uses), and its
    copies follow in ``ranks`` order. Representatives keep their order in ``U``; a column without
    copies represents itself.
    """
    k = len(U)
    linked = np.zeros((k, k), dtype=bool)
    for X in groups:
        cov = np.cov(X[:, U], rowvar=False, ddof=0).reshape(k, k)
        var = np.diag(cov)
        linked |= cov * cov.T >= (1.0 - COPY_TOL) * np.outer(var, var)
    np.fill_diagonal(linked, False)
    if not linked.any():
        return {u: [] for u in U}
    copies: dict[int, list[int]] = {}
    seen = np.zeros(k, dtype=bool)
    for start in range(k):
        if seen[start]:
            continue
        component, frontier = [start], [start]
        seen[start] = True
        while frontier:
            new = np.flatnonzero(linked[frontier].any(axis=0) & ~seen)
            seen[new] = True
            component += new.tolist()
            frontier = new.tolist()
        members = sorted((U[a] for a in component), key=lambda c: ranks[c])
        copies[members[0]] = members[1:]
    return {u: copies[u] for u in U if u in copies}


def eligible(
    precedence: np.ndarray, forbidden_paths: np.ndarray | None, candidates: list[int]
) -> np.ndarray:
    """The ``candidates`` an order search may place next, as a mask aligned with them.

    A candidate is eligible when no other candidate must precede it (``precedence[a, b]``: ``a``
    before ``b``). Step 2a-1 of DirectLiNGAM with prior knowledge (Shimizu et al. 2011, Section
    3.4) also reads ``forbidden_paths[a, b]``, the claim that ``a`` is no ancestor of ``b``: when
    the claims show that no other candidate is an ancestor of some eligible candidates (their Lemma
    3), only those are exogenous among the candidates, and the pick is among them. A candidate with
    a claimed ancestor among the others is never exogenous (their Lemma 4); ``precedence`` holds
    those claims.
    """
    block = np.ix_(candidates, candidates)
    allowed = ~precedence[block].any(axis=0).astype(bool)
    if forbidden_paths is not None:
        exogenous = (forbidden_paths[block] | np.eye(len(candidates), dtype=bool)).all(axis=0)
        if (allowed & exogenous).any():
            return allowed & exogenous
    return allowed


def find_causal_order(
    X: np.ndarray,
    *,
    precedence: np.ndarray | None = None,
    forbidden_paths: np.ndarray | None = None,
    measure: str = "pwling",
) -> list[int]:
    """Peel exogenous variables (:func:`pick` of the ``measure`` scores) one at a time.

    ``measure`` is one of :data:`MEASURES`; :func:`scorer` gives its scoring function, and
    ``"pwling"`` scores with :func:`pruned_scores`, which gives the pick of
    :func:`score_candidates` from fewer entropies.

    Each pick is residualized out of the rest. Only one column of each copy group
    (:func:`copy_groups`) is scored; when it is picked, its copies follow it directly in the order.
    Ties and copy representatives are decided by :func:`~andrey.core.ties.tie_ranks` of the input
    columns, which are identical on every machine and keep their ranks when the columns are
    reordered; residualized columns differ in the last bits across machines. Single-dataset;
    MultiGroupDirectLiNGAM runs its own group loop (per-group residualization), reusing
    :func:`copy_groups`, :func:`scorer`, and :func:`pick` directly. With ``precedence``
    (``precedence[a, b]``: ``a`` comes before ``b``, acyclic), no copy group is formed, since
    placing one at once could put a column before one that must precede it: every remaining
    variable is scored, and the pick is the best of :func:`eligible`.
    """
    score = scorer(measure)
    remaining = list(range(X.shape[1]))
    order: list[int] = []
    residualized = np.array(X, dtype=np.float64)  # copy + float64: the write-back must not truncate
    ranks = tie_ranks(residualized)
    hint: dict[int, float] = {}
    while remaining:
        if precedence is None:
            copies = copy_groups([residualized], remaining, ranks)
        else:
            copies = {u: [] for u in remaining}
        candidates = list(copies)
        # A lone candidate needs no scoring (and has no pair to score).
        if len(candidates) == 1:
            m = candidates[0]
        else:
            allowed = (
                None if precedence is None else eligible(precedence, forbidden_paths, candidates)
            )
            if measure == "pwling":
                values = pruned_scores([residualized], candidates, allowed=allowed, hint=hint)
                hint = dict(zip(candidates, values.tolist(), strict=True))
            else:
                values = score(residualized, candidates)
            if allowed is None:
                m = pick(values, candidates, ranks)
            else:
                m = pick(values[allowed], [c for c, ok in zip(candidates, allowed) if ok], ranks)
        placed = [m, *copies[m]]
        remaining = [u for u in remaining if u not in placed]
        for i in remaining:
            residualized[:, i] = compute_residual(residualized[:, i], residualized[:, m])
        order += placed
    return order
