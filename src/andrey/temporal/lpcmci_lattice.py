"""Partial correlations of one pair given every subset of a candidate list, a level at a time.

LPCMCI tests a link ``x - y`` given each ``p``-subset ``S`` of a search list, in the order of
:func:`itertools.combinations`, with the same fixed set ``F`` in every query. A ``p``-subset
extends its parent, the ``(p-1)``-subset without its last element, by one candidate, so its partial
correlation follows from the parent's factor by one row, where an inversion starts from the
``[x, y, *S, *F]`` correlation submatrix again.

For the subset ``S`` in search order, after ``F``, a :class:`SubsetLattice` keeps
``W = L^-1 R[F + S, (x, y, *candidates)]``, with ``L`` the Cholesky factor of ``R[F + S]``, and the
Schur complement of ``(x, y)`` given ``F + S``: its entries ``pxx``, ``pxy``, ``pyy``. Adding the
candidate ``t`` appends the row ``(R[t] - l_t W) / sqrt(R[t, t] - |l_t|^2)``, with ``l_t`` the
``t`` column of ``W``, and takes that row's ``x`` and ``y`` entries off the Schur complement; the
partial correlation is ``pxy / sqrt(pxx * pyy)``. A level holds every subset of its size, so the
next level is read from it and the search goes on from there in the next round.

The values are a screen, not the answer. The partial correlations differ from the inversion's in
their last digits, so :meth:`SubsetLattice.prefix` asks the test again for the subsets whose value
the search reads at the edge of a decision: those within a band of ``alpha``, of the prefix's
smallest absolute statistic, and of its largest p-value. The search then records the test's own
values for them, and the other subsets cannot change what it records.
"""

from __future__ import annotations

import math

import numpy as np

from andrey.core.ci import BOUND_MARGIN, ONE_MINUS_EPS, SINGULAR_RTOL, PartialCorrelationTest
from andrey.temporal.pcmci import DesignTest

#: ``MIN_SUBSETS`` and ``MAX_BYTES`` are fixed cutoffs of the algorithm, measured on one Linux CPU,
#: not dispatch gates: they choose between two ways to compute the same values, so neither is read
#: from the environment or recorded by ``andrey calibrate``.
#:
#: The fewest subsets of one size for which the lattice replaces the test. Below about 2,000 a
#: level's fixed cost (about 0.7 ms) and the levels built beneath it exceed the inversions' time;
#: from 4,096 on the lattice was faster in every case measured (candidates 12 to 16, 0 or 3 fixed).
MIN_SUBSETS = 1 << 12
#: The most bytes the factors of the live levels hold, in all the lattices of a fit, 1 GiB. A
#: search that runs in one of a pool's workers holds its share of it,
#: ``MAX_BYTES // backend.pool_workers()``, so the workers of a pool hold ``MAX_BYTES`` in all.
MAX_BYTES = 1 << 30
#: The subsets of a level answered in one call, after the doubling from :data:`FIRST_CHUNK`.
CHUNK = 1 << 14
FIRST_CHUNK = 16
#: How far from ``alpha`` (relative), from the prefix's smallest absolute statistic (absolute), and
#: from its largest p-value (relative) a subset's value is asked of the test again. Each exceeds by
#: orders of magnitude the largest difference between the two paths that a state passing the
#: certificate shows (:class:`SubsetLattice`: 2e-12 of a partial correlation, 3e-10 relative of a
#: p-value at ``n = 1000``).
ALPHA_BAND = 1e-6
STATISTIC_BAND = 1e-8
PVALUE_BAND = 1e-6


def level_bytes(m: int, f: int, q: int) -> int:
    """The bytes of the ``q``-subsets of ``m`` candidates, given ``f`` fixed columns."""
    return math.comb(m, q) * ((f + q) * (m + 2) + 4) * 8


def live_bytes(m: int, f: int, p: int) -> int:
    """The most bytes a lattice holds while it reads level ``p`` from level ``p - 1``, on its way up
    from level 0."""
    return max(level_bytes(m, f, q - 1) + level_bytes(m, f, q) for q in range(1, p + 1))


def rebuild_pays(m: int, p: int) -> bool:
    """Whether building levels ``0 .. p - 1`` from scratch costs at most twice the tests of level
    ``p``: the lattice is built again only when that is so."""
    return sum(math.comb(m, q) for q in range(p)) <= 2 * math.comb(m, p)


def unrank(m: int, q: int, ranks: np.ndarray) -> np.ndarray:
    """The ``q``-subsets of ``range(m)`` at ``ranks`` in the order of
    :func:`itertools.combinations`, as an ``(len(ranks), q)`` array."""
    left = np.array(ranks, dtype=np.int64)
    out = np.empty((len(left), q), dtype=np.intp)
    first = np.zeros(len(left), dtype=np.intp)
    for i in range(q):
        chosen = np.full(len(left), -1, dtype=np.intp)
        for v in range(m):
            size = math.comb(m - v - 1, q - i - 1)
            open_ = (chosen < 0) & (first <= v)
            take = open_ & (left < size)
            chosen[take] = v
            left[open_ & ~take] -= size
        out[:, i] = chosen
        first = chosen + 1
    return out


class SubsetLattice:
    """The partial correlations of ``x`` and ``y`` given each subset of ``columns``, with ``fixed``.

    :meth:`build` makes the level-0 lattice, and it climbs a level at a time: :meth:`grow` computes
    levels in full, and :meth:`prefix` reads the next level in chunks of its subsets in search
    order. ``columns`` are the candidates in search order and ``fixed`` the sorted columns every
    query holds. :meth:`rows` gives the columns of queries, as the test is asked.

    A lattice is built only when the correlation block of ``(x, y, *columns, *fixed)`` passes the
    eigenvalue certificate that the inversion path's singularity test implies: its smallest
    eigenvalue is above ``BOUND_MARGIN * SINGULAR_RTOL`` times its size, an upper bound of its
    largest, and by interlacing so is every principal submatrix's. No query of the lattice is then
    one the test calls singular, and no step divides by a pivot near zero.
    """

    def __init__(
        self,
        design: DesignTest,
        test: PartialCorrelationTest,
        x: int,
        y: int,
        columns: np.ndarray,
        fixed: np.ndarray,
    ) -> None:
        corr, _ = test.to_seed()
        self.design, self.test = design, test
        self.x, self.y, self.columns, self.fixed = x, y, columns, fixed
        self.m, self.f = len(columns), len(fixed)
        at = np.concatenate([[x, y], columns]).astype(np.intp)
        self.R = corr[np.ix_(at, at)]
        self.diagonal = np.diagonal(self.R).copy()
        full = np.concatenate([at, fixed]).astype(np.intp)
        smallest = np.linalg.eigvalsh(corr[np.ix_(full, full)])[0]
        self.certified = bool(smallest > BOUND_MARGIN * SINGULAR_RTOL * len(full))
        self.level = 0
        self.upcoming: tuple[np.ndarray, ...] | None = None
        self.done = 0
        self.counts = self.cum = np.zeros(0, dtype=np.intp)
        w = np.zeros((self.f, len(at)))
        if self.certified and self.f:
            factor = np.linalg.cholesky(corr[np.ix_(fixed, fixed)])
            w = np.linalg.solve(factor, corr[np.ix_(fixed, at)])
        self.last = np.array([-1], dtype=np.intp)
        self.w = w[None]
        self.pxy = np.array([self.R[0, 1] - w[:, 0] @ w[:, 1]])
        self.pxx = np.array([self.R[0, 0] - w[:, 0] @ w[:, 0]])
        self.pyy = np.array([self.R[1, 1] - w[:, 1] @ w[:, 1]])

    @classmethod
    def build(
        cls, design: DesignTest, x: int, y: int, columns: np.ndarray, fixed: np.ndarray
    ) -> SubsetLattice | None:
        """The level-0 lattice of ``design``'s test, or ``None`` when its test is not a partial
        correlation of complete data or the state fails the certificate."""
        test = design.test
        if not isinstance(test, PartialCorrelationTest):
            return None
        lattice = cls(design, test, x, y, columns, np.sort(fixed))
        return lattice if lattice.certified else None

    @property
    def nbytes(self) -> int:
        """The bytes the lattice holds: its level and the next one's arrays once allocated."""
        held = (self.last, self.w, self.pxy, self.pxx, self.pyy, *(self.upcoming or ()))
        return sum(a.nbytes for a in held)

    def _start_level(self, store: bool) -> None:
        """Count the children of each subset of the level, and allocate the next level's arrays
        when ``store``."""
        self.counts = self.m - 1 - self.last
        self.cum = np.cumsum(self.counts)
        size = int(self.cum[-1]) if len(self.cum) else 0
        self.upcoming = None
        if store and size:
            self.upcoming = (
                np.empty(size, dtype=np.intp),
                np.empty((size, self.f + self.level + 1, self.m + 2)),
                np.empty(size),
                np.empty(size),
                np.empty(size),
            )

    def _extend(self, start: int, stop: int, store: bool) -> np.ndarray:
        """The partial correlations, clipped, of the next level's subsets ``start .. stop - 1``;
        their rows are kept when ``store``."""
        if start == 0:
            self._start_level(store)
        ranks = np.arange(start, stop)
        parent = np.searchsorted(self.cum, ranks, side="right")
        first = self.cum[parent] - self.counts[parent]
        t = self.last[parent] + 1 + (ranks - first)
        w = self.w[parent]  # (n, f + level, m + 2)
        l_t = w[np.arange(len(t)), :, 2 + t]
        pivot = np.sqrt(self.diagonal[2 + t] - np.einsum("ci,ci->c", l_t, l_t))
        row = (self.R[2 + t] - np.matmul(l_t[:, None, :], w)[:, 0, :]) / pivot[:, None]
        pxy = self.pxy[parent] - row[:, 0] * row[:, 1]
        pxx = self.pxx[parent] - row[:, 0] * row[:, 0]
        pyy = self.pyy[parent] - row[:, 1] * row[:, 1]
        if self.upcoming is not None:
            last, rows, kept_xy, kept_xx, kept_yy = self.upcoming
            last[start:stop] = t
            rows[start:stop, :-1] = w
            rows[start:stop, -1] = row
            kept_xy[start:stop], kept_xx[start:stop], kept_yy[start:stop] = pxy, pxx, pyy
        self.done = stop
        return np.clip(pxy / np.sqrt(pxx * pyy), -ONE_MINUS_EPS, ONE_MINUS_EPS)

    def advance(self) -> None:
        """Make the next level, once :meth:`prefix` has read every chunk of it, the current one."""
        assert self.upcoming is not None
        self.last, self.w, self.pxy, self.pxx, self.pyy = self.upcoming
        self.level += 1
        self.upcoming = None

    def grow(self, level: int) -> None:
        """Compute levels up to ``level``, each in full."""
        while self.level < level:
            total = math.comb(self.m, self.level + 1)
            for start in range(0, total, CHUNK):
                self._extend(start, min(start + CHUNK, total), store=True)
            self.advance()

    def rows(self, ranks: np.ndarray) -> np.ndarray:
        """The sorted columns of the queries of the next level's subsets at ``ranks``, one row
        each."""
        picks = unrank(self.m, self.level + 1, ranks)
        rows = np.concatenate(
            [self.columns[picks], np.broadcast_to(self.fixed, (len(ranks), self.f))], axis=1
        )
        rows.sort(axis=1)
        return rows

    def subset(self, rank: int) -> tuple[int, ...]:
        """The indices into ``columns`` of the next level's subset at ``rank``."""
        return tuple(unrank(self.m, self.level + 1, np.array([rank]))[0].tolist())

    def _ask(
        self, at: np.ndarray, start: int, statistics: np.ndarray, p_values: np.ndarray
    ) -> None:
        """Replace the values at positions ``at`` of the chunk by the test's own."""
        if len(at):
            rows = self.rows(start + at)
            x = np.full(len(at), self.x, dtype=np.intp)
            statistics[at], p_values[at] = self.design.statistic_pvalues(
                x, np.full_like(x, self.y), rows
            )

    def prefix(self, start: int, stop: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Statistics and p-values of the next level's subsets ``start .. stop - 1``, up to and
        including the first with ``p > alpha``.

        Chunks come in order from ``start = 0``. The values of the subsets that decide what the
        search records are the test's own: the first above ``alpha``, the one with the smallest
        absolute statistic, and the one with the largest p-value, each with the subsets within a
        band of it (:data:`ALPHA_BAND`, :data:`STATISTIC_BAND`, :data:`PVALUE_BAND`). After the
        last chunk of the level, with none above ``alpha``, :meth:`advance` keeps the level for the
        next round.
        """
        store = self.level + 1 < self.m
        statistics = self._extend(start, stop, store)
        df = self.test._degrees_of_freedom(self.level + 1 + self.f)
        p_values = np.asarray(self.test._pvalue(statistics, df))
        asked = np.zeros(len(p_values), dtype=bool)
        # Every subset before the first confidently above alpha is decided by the test's own
        # value when it lies within the band.
        confident = np.flatnonzero(p_values > alpha * (1 + ALPHA_BAND))
        end = int(confident[0]) if len(confident) else len(p_values)
        near = np.flatnonzero(np.abs(p_values[:end] - alpha) <= ALPHA_BAND * alpha)
        self._ask(near, start, statistics, p_values)
        asked[near] = True
        above = np.flatnonzero(p_values[: end + 1] > alpha)
        made = int(above[0]) + 1 if len(above) else len(p_values)
        statistics, p_values, asked = statistics[:made], p_values[:made], asked[:made]
        # The statistic and p-value the search keeps of the tests made: the smallest absolute
        # statistic and the largest p-value.
        magnitude = np.abs(statistics)
        low = np.flatnonzero(magnitude <= magnitude.min() + STATISTIC_BAND)
        top = p_values.max()
        if top > 0:
            high = np.flatnonzero(p_values >= top * (1 - PVALUE_BAND))
        else:
            high = np.array([int(np.argmax(p_values))])
        pick = np.union1d(low, high)
        self._ask(pick[~asked[pick]], start, statistics, p_values)
        return statistics, p_values
