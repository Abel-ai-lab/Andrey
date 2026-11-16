"""The sample correlation matrix of a data set's columns, computed block by block, whole or lazily.

Every entry comes from one fixed computation, so the whole matrix and a row computed alone agree
bit for bit. :func:`centered` centers each column and finds its Euclidean norm, on a contiguous
copy whose rows are the columns, so the result does not depend on the data's memory layout. The
columns fall into blocks of ``BLOCK`` consecutive positions, and the entries between blocks
``i <= j`` are one product of their centered columns, ``C_i @ C_j.T``, divided by the products of
the norms and clipped to ``[-1, 1]``; the entries of blocks ``j < i`` are the transpose of block
``(j, i)``, and a block with itself keeps its upper triangle, mirrored, so the matrix is exactly
symmetric. Each product runs on one BLAS thread, and as many Python threads as the BLAS has run
different products at once. A product that the BLAS itself splits among threads can differ in its
last bits (OpenBLAS's does at a few hundred rows or fewer), so this way the entries are the same on
any number of threads. Dividing after the sums keeps an exact sum exact: on data whose sums of
products round nowhere, such as small integers in a power-of-two number of rows, two columns that
hold the same values in another row order correlate equally with a third column. The diagonal is
exactly ``1``, or ``NaN`` for a column whose values are all equal. :func:`correlation_matrix` gives
the whole matrix; :class:`LazyCorrelation` computes the ``BLOCK`` rows of one block when a row of it
is first read and keeps them, the same products in the same order, so its rows equal those of
:func:`correlation_matrix` bit for bit.

A search that reads few rows, as a local causal discovery method does, then costs time and memory
in proportion to ``n * d`` per block of rows it reads instead of ``n * d**2`` time and ``d**2``
memory up front.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt

from . import backend
from .score import ZERO_RESIDUAL, copies_error

__all__ = ["BLOCK", "LazyCorrelation", "centered", "correlation_matrix"]

#: The number of consecutive columns in one block, enough that each product is worth a thread. A
#: lazily read row computes the ``BLOCK`` rows of its block: ``BLOCK * d * 8`` bytes, 33 MB at
#: 16,000 columns.
BLOCK = 256


def centered(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The columns of the ``(n, d)`` ``data`` less their means, and their Euclidean norms.

    Returns the ``(d, n)`` C-contiguous array whose row ``j`` is column ``j`` less its mean, and
    the ``(d,)`` norms of those rows. Each row's sums run along that contiguous row, so the values
    depend only on the column's own values.
    """
    c = np.array(np.asarray(data, dtype=np.float64).T, order="C")
    d, n = c.shape
    c -= (c.sum(axis=1) / n)[:, None]
    norm = np.empty(d)
    for start in range(0, d, BLOCK):
        rows = c[start : start + BLOCK]
        norm[start : start + BLOCK] = np.sqrt((rows * rows).sum(axis=1))
    return c, norm


def _block(c: np.ndarray, norm: np.ndarray, i: int, j: int) -> np.ndarray:
    """The correlations between blocks ``i <= j`` of the centered columns ``c`` with norms
    ``norm``, clipped to ``[-1, 1]``; block ``i`` with itself is its upper triangle mirrored, with
    the unit diagonal (``NaN`` for a zero norm)."""
    span_i, span_j = slice(i * BLOCK, (i + 1) * BLOCK), slice(j * BLOCK, (j + 1) * BLOCK)
    rows = c[span_i]
    if i == j:
        product = rows @ rows.T
        product = np.triu(product, 1)
        product += product.T
    else:
        product = rows @ c[span_j].T
    product /= np.multiply.outer(norm[span_i], norm[span_j])
    if i == j:
        np.fill_diagonal(product, np.where(norm[span_i] > 0.0, 1.0, np.nan))
    return np.clip(product, -1.0, 1.0, out=product)


def _run(work: Callable[[int, int], None], jobs: Sequence[tuple[int, int]], threads: int) -> None:
    """``work(i, j)`` for each of ``jobs``, on up to ``threads`` Python threads with the BLAS held
    to one thread, so each product is the same whatever ``threads`` is."""
    with backend.one_blas_thread():
        if threads <= 1 or len(jobs) <= 1:
            for i, j in jobs:
                work(i, j)
            return
        with ThreadPoolExecutor(min(threads, len(jobs))) as pool:
            list(pool.map(lambda job: work(*job), jobs))


def _matrix(c: np.ndarray, norm: np.ndarray, threads: int) -> np.ndarray:
    """The whole correlation matrix of the centered columns ``c`` with norms ``norm``, its blocks
    computed on up to ``threads`` threads."""
    d = c.shape[0]
    blocks = -(-d // BLOCK)
    out = np.empty((d, d))

    def fill(i: int, j: int) -> None:
        product = _block(c, norm, i, j)
        out[i * BLOCK : (i + 1) * BLOCK, j * BLOCK : (j + 1) * BLOCK] = product
        if j > i:
            out[j * BLOCK : (j + 1) * BLOCK, i * BLOCK : (i + 1) * BLOCK] = product.T

    _run(fill, [(i, j) for i in range(blocks) for j in range(i, blocks)], threads)
    return out


def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """The ``(d, d)`` sample correlation matrix of the columns of the ``(n, d)`` ``data``.

    Computed block by block from :func:`centered` columns as the module describes: exactly
    symmetric, its diagonal ``1`` (``NaN`` for a constant column), its entries clipped to
    ``[-1, 1]``, and each row equal bit for bit to the row :class:`LazyCorrelation` computes.
    """
    return _matrix(*centered(data), backend.blas_threads())


def _groups(pairs: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    """The connected groups of the copied pairs ``pairs``, each ascending, ordered by first."""
    parent: dict[int, int] = {}

    def root(v: int) -> int:
        while parent.setdefault(v, v) != v:
            v = parent[v]
        return v

    for a, b in pairs:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    members: dict[int, list[int]] = {}
    for v in sorted(parent):
        members.setdefault(root(v), []).append(v)
    return tuple(sorted(tuple(group) for group in members.values() if len(group) > 1))


class LazyCorrelation:
    """The correlation matrix of the columns of ``data``, a block of rows computed when one of its
    rows is first read and kept for the life of the object.

    The rows equal :func:`correlation_matrix`'s bit for bit, and the matrix is exactly symmetric, so
    an entry ``[a, b]`` is read from whichever of the rows ``a`` and ``b`` is computed.
    :meth:`entry` and :meth:`cover` compute one when neither is: the column asked about more
    often, so the row of a search's target serves its candidates. The centered columns, as large as
    the data, are made at the first read.

    Each computed block is checked for columns that repeat another up to scale and shift, as
    :func:`~andrey.core.score.copy_groups` defines them: a copy shown by a computed row raises
    :class:`~andrey.core.score.CopiedColumnsError` naming its group, on that read and every later
    read of the block. Copies among columns whose rows are never computed do not raise.

    The products of a block run on as many threads as the BLAS has when the object is made, wherever
    a row is first read, so a first read inside a search's one-thread block still uses them, with
    the same result.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data: np.ndarray | None = data
        self.n, self.d = data.shape
        self._z = np.empty((0, 0))
        self._norm = np.empty(0)
        self.unit = np.empty(0)
        self._slot = np.full(self.d, -1, dtype=np.intp)  # each column's row in _rows, -1 if none
        self._slots = [-1] * self.d  # the same, for scalar reads
        self._rows = np.empty((0, self.d))
        self._count = 0
        self._asked = np.zeros(self.d, dtype=np.intp)  # reads that found neither row computed
        self._threads = backend.blas_threads()

    @property
    def computed(self) -> int:
        """The number of rows computed so far."""
        return self._count

    def _center(self) -> None:
        if self._data is not None:
            self._z, self._norm = centered(self._data)
            self.unit = np.where(self._norm > 0.0, 1.0, np.nan)
            self._data = None

    def matrix(self) -> np.ndarray:
        """The whole matrix, as :func:`correlation_matrix` computes it, checked for copies: the
        computed rows in column order once every row is computed, else computed whole."""
        if self._count == self.d:
            return self._rows[self._slot]
        self._center()
        full = _matrix(self._z, self._norm, self._threads)
        self._check(full, np.arange(self.d))
        return full

    def _check(self, rows: np.ndarray, ids: np.ndarray) -> None:
        """Raise the copies error when ``rows``, the rows of columns ``ids``, show a copy."""
        if self.n <= 2 or self.d < 2:
            return
        alike = np.abs(rows)
        alike[np.arange(len(ids)), ids] = 0.0
        found = np.argwhere(1.0 - alike * alike <= ZERO_RESIDUAL)
        if len(found):
            raise copies_error(_groups([(int(ids[k]), int(b)) for k, b in found]))

    def load(self, cols: npt.ArrayLike) -> None:
        """Compute the rows of the blocks holding the columns ``cols``, those not computed yet."""
        cols = np.asarray(cols, dtype=np.intp).reshape(-1)
        missing = cols[self._slot[cols] < 0]
        if not len(missing):
            return
        self._center()
        blocks = -(-self.d // BLOCK)
        for i in np.unique(missing // BLOCK).tolist():
            ids = np.arange(i * BLOCK, min(self.d, (i + 1) * BLOCK))
            rows = np.empty((len(ids), self.d))

            def fill(i: int, j: int, rows: np.ndarray = rows) -> None:
                block = _block(self._z, self._norm, *((i, j) if j >= i else (j, i)))
                rows[:, j * BLOCK : (j + 1) * BLOCK] = block if j >= i else block.T

            _run(fill, [(i, j) for j in range(blocks)], self._threads)
            self._check(rows, ids)
            self._store(rows, ids)

    def _store(self, rows: np.ndarray, ids: np.ndarray) -> None:
        if self._count + len(ids) > len(self._rows):
            size = max(min(2 * len(self._rows), self.d), self._count + len(ids))
            grown = np.empty((size, self.d))
            grown[: self._count] = self._rows[: self._count]
            self._rows = grown
        slots = np.arange(self._count, self._count + len(ids))
        self._rows[slots] = rows
        self._slot[ids] = slots
        for v, s in zip(ids.tolist(), slots.tolist()):
            self._slots[v] = s
        self._count += len(ids)

    def need(self, cols: Sequence[int]) -> None:
        """:meth:`load` for a short list of columns, checked one at a time."""
        slots = self._slots
        missing = [c for c in cols if slots[c] < 0]
        if missing:
            self.load(missing)

    def square(self, x: int, y: int, cond: Sequence[int]) -> np.ndarray:
        """The ``[x, y, *cond]`` submatrix for distinct columns whose ``cond`` rows are computed,
        read from those rows and from the row of ``x`` or ``y``."""
        slots = self._slots
        if slots[x] < 0 and slots[y] < 0:
            self.entry(x, y)
        pivot = 0 if slots[x] >= 0 else 1
        var = [x, y, *cond]
        sub = self._rows[np.ix_([slots[var[pivot]], *(slots[c] for c in cond)], var)]
        block = np.empty((len(var), len(var)))
        block[pivot] = sub[0]
        block[2:] = sub[1:]
        other = 1 - pivot
        block[other] = block[:, other]
        block[other, other] = self.unit[var[other]]
        return block

    def rows(self, cols: npt.ArrayLike) -> np.ndarray:
        """The rows of the columns ``cols``, computed where they are not yet, as a new array."""
        self.load(cols)
        return self._rows[self._slot[np.asarray(cols, dtype=np.intp)]]

    def entry(self, a: int, b: int) -> float:
        """The entry ``[a, b]`` for two distinct columns."""
        s = self._slots[a]
        if s >= 0:
            return float(self._rows[s, b])
        s = self._slots[b]
        if s >= 0:
            return float(self._rows[s, a])
        self._asked[a] += 1
        self._asked[b] += 1
        first = a if self._asked[a] >= self._asked[b] else b
        self.load((first,))
        return float(self._rows[self._slots[first], b if first == a else a])

    def cover(self, a: np.ndarray, b: np.ndarray) -> None:
        """Compute rows until each pair ``(a[k], b[k])`` of distinct columns has one: the column
        in the most pairs still uncovered first, ties by reads that found neither, then position."""
        while True:
            need = (self._slot[a] < 0) & (self._slot[b] < 0) & (a != b)
            if not need.any():
                return
            ends, counts = np.unique(np.concatenate((a[need], b[need])), return_counts=True)
            self._asked[ends] += counts
            best = np.lexsort((ends, -self._asked[ends], -counts))[0]
            self.load((ends[best],))

    def pairs(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The entries ``[a[k], b[k]]`` of distinct columns, each pair with one of its rows
        computed (:meth:`cover`): read from the row of ``a[k]`` when it is computed, else from the
        row of ``b[k]``."""
        sa = self._slot[a]
        own = sa >= 0
        return self._rows[np.where(own, sa, self._slot[b]), np.where(own, b, a)]

    def read(self, a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        """The entries ``[a, b]`` of the broadcast integer arrays ``a``, whose rows are computed,
        and ``b``, read from the rows of ``a``."""
        return self._rows[self._slot[a], b]

    def squares(self, var: np.ndarray) -> np.ndarray:
        """:meth:`square` of each row ``[x, y, *cond]`` of the ``(m, k)`` ``var``: the
        ``(m, k, k)`` stack of submatrices, for rows whose ``cond`` rows are computed and whose
        ``x`` or ``y`` row is."""
        m, k = var.shape
        q = np.arange(m)
        slots = self._slot[var]
        pivot = (slots[:, 0] < 0).astype(np.intp)
        other = 1 - pivot
        pick = slots[:, 1:].copy()
        pick[:, 0] = slots[q, pivot]
        sub = self._rows[pick[:, :, None], var[:, None, :]]
        block = np.empty((m, k, k))
        block[:, 2:] = sub[:, 1:]
        block[q, pivot] = sub[:, 0]
        block[q, other] = block[q, :, other]
        block[q, other, other] = self.unit[var[q, other]]
        return block
