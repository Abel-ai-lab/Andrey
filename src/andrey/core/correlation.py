"""The sample correlation matrix of a data set's columns, computed tile by tile, whole or lazily.

Every entry comes from one fixed computation, so the whole matrix and a row computed alone agree
bit for bit. :func:`standardized` centers each column and scales it to unit norm, on a contiguous
copy whose rows are the columns, so the result does not depend on the data's memory layout. The
columns fall into tiles of ``TILE`` consecutive positions, and the entries between tiles ``i <= j``
are one product of their standardized columns, ``Z_i @ Z_j.T``, on one BLAS thread, clipped to
``[-1, 1]``; the entries of tiles ``j < i`` are the transpose of tile ``(j, i)``, and a tile with
itself keeps its upper triangle, mirrored, so the matrix is exactly symmetric. The diagonal is
exactly ``1``, or ``NaN`` for a column whose values are all equal. :func:`correlation_matrix` gives
the whole matrix; :class:`LazyCorrelation` computes the rows of one tile when a row of it is first
read and keeps them, the same products in the same order, so its rows equal those of
:func:`correlation_matrix` bit for bit.

A search that reads few rows, as a local causal discovery method does, then costs time and memory
in proportion to ``n * d`` per tile of rows it reads instead of ``n * d**2`` time and ``d**2``
memory up front.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import backend
from .score import ZERO_RESIDUAL, copies_error

__all__ = ["TILE", "LazyCorrelation", "correlation_matrix", "standardized"]

#: The number of consecutive columns in one tile. A lazily read row computes the rows of its tile:
#: on one AMD EPYC 7713 core, 0.11 s at 2,000 rows and 16,000 columns (0.18 s with 64), while the
#: whole matrix of 4,000 columns takes 1.6 times the 1.1 s of ``np.corrcoef`` (1.25 times with 64).
TILE = 32


def standardized(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The columns of the ``(n, d)`` ``data`` centered and scaled to unit norm, and the diagonal.

    Returns the ``(d, n)`` C-contiguous array whose row ``j`` is column ``j`` less its mean and
    divided by its Euclidean norm, and the ``(d,)`` diagonal of the correlation matrix: ``1``, or
    ``NaN`` for a column whose centered values are all zero. Each row's sums run along that
    contiguous row, so the values depend only on the column's own values.
    """
    z = np.array(np.asarray(data, dtype=np.float64).T, order="C")
    d, n = z.shape
    z -= (z.sum(axis=1) / n)[:, None]
    norm = np.empty(d)
    for start in range(0, d, TILE):
        block = z[start : start + TILE]
        norm[start : start + TILE] = np.sqrt((block * block).sum(axis=1))
    z /= norm[:, None]
    return z, np.where(norm > 0.0, 1.0, np.nan)


def _tile(z: np.ndarray, unit: np.ndarray, i: int, j: int) -> np.ndarray:
    """The entries between tiles ``i <= j`` of the standardized columns ``z``, clipped to
    ``[-1, 1]``; tile ``i`` with itself is its upper triangle mirrored, diagonal ``unit``."""
    rows = z[i * TILE : (i + 1) * TILE]
    if i == j:
        product = rows @ rows.T
        product = np.triu(product, 1)
        product += product.T
        np.fill_diagonal(product, unit[i * TILE : (i + 1) * TILE])
    else:
        product = rows @ z[j * TILE : (j + 1) * TILE].T
    return np.clip(product, -1.0, 1.0, out=product)


def _matrix(z: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """The whole correlation matrix of the standardized columns ``z``, tile by tile."""
    d = z.shape[0]
    tiles = -(-d // TILE)
    out = np.empty((d, d))
    with backend.one_blas_thread():
        for i in range(tiles):
            for j in range(i, tiles):
                product = _tile(z, unit, i, j)
                out[i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE] = product
                if j > i:
                    out[j * TILE : (j + 1) * TILE, i * TILE : (i + 1) * TILE] = product.T
    return out


def correlation_matrix(data: np.ndarray) -> np.ndarray:
    """The ``(d, d)`` sample correlation matrix of the columns of the ``(n, d)`` ``data``.

    Computed tile by tile from :func:`standardized` columns as the module describes: exactly
    symmetric, its diagonal ``1`` (``NaN`` for a constant column), its entries clipped to
    ``[-1, 1]``, and each row equal bit for bit to the row :class:`LazyCorrelation` computes.
    """
    return _matrix(*standardized(data))


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
    """The correlation matrix of the columns of ``data``, a tile of rows computed when one of its
    rows is first read and kept for the life of the object.

    The rows equal :func:`correlation_matrix`'s bit for bit, and the matrix is exactly symmetric, so
    an entry ``[a, b]`` is read from whichever of the rows ``a`` and ``b`` is computed.
    :meth:`entry` and :meth:`entries` compute one when neither is: the column asked about more
    often, so the row of a search's target serves its candidates. The standardized columns, as
    large as the data, are made at the first read.

    Each computed tile is checked for columns that repeat another up to scale and shift, as
    :func:`~andrey.core.score.copy_groups` defines them: a copy shown by a computed row raises
    :class:`~andrey.core.score.CopiedColumnsError` naming its group, on that read and every later
    read of the tile. Copies among columns whose rows are never computed do not raise.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data: np.ndarray | None = data
        self.n, self.d = data.shape
        self._z = np.empty((0, 0))
        self.unit = np.empty(0)
        self._slot = np.full(self.d, -1, dtype=np.intp)  # each column's row in _rows, -1 if none
        self._slots = [-1] * self.d  # the same, for scalar reads
        self._rows = np.empty((0, self.d))
        self._count = 0
        self._asked = np.zeros(self.d, dtype=np.intp)  # reads that found neither row computed

    @property
    def computed(self) -> int:
        """The number of rows computed so far."""
        return self._count

    def _standardize(self) -> None:
        if self._data is not None:
            self._z, self.unit = standardized(self._data)
            self._data = None

    def matrix(self) -> np.ndarray:
        """The whole matrix, as :func:`correlation_matrix` computes it, checked for copies."""
        self._standardize()
        full = _matrix(self._z, self.unit)
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
        """Compute the rows of the tiles holding the columns ``cols``, those not computed yet."""
        cols = np.asarray(cols, dtype=np.intp).reshape(-1)
        missing = cols[self._slot[cols] < 0]
        if not len(missing):
            return
        self._standardize()
        tiles = -(-self.d // TILE)
        with backend.one_blas_thread():
            for i in np.unique(missing // TILE).tolist():
                ids = np.arange(i * TILE, min(self.d, (i + 1) * TILE))
                block = np.empty((len(ids), self.d))
                for j in range(tiles):
                    tile = _tile(self._z, self.unit, *((i, j) if j >= i else (j, i)))
                    block[:, j * TILE : (j + 1) * TILE] = tile if j >= i else tile.T
                self._check(block, ids)
                self._store(block, ids)

    def _store(self, block: np.ndarray, ids: np.ndarray) -> None:
        if self._count + len(ids) > len(self._rows):
            grown = np.empty((max(2 * len(self._rows), self._count + len(ids)), self.d))
            grown[: self._count] = self._rows[: self._count]
            self._rows = grown
        slots = np.arange(self._count, self._count + len(ids))
        self._rows[slots] = block
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

    def entries(self, a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
        """The entries ``[a, b]`` of the broadcast integer arrays ``a`` and ``b``, computing rows
        (:meth:`cover`) where neither of a pair's rows is computed yet."""
        a, b = np.broadcast_arrays(np.asarray(a, dtype=np.intp), np.asarray(b, dtype=np.intp))
        same = a == b
        if same.all():
            self._standardize()
            return self.unit[a]
        self.cover(a.reshape(-1), b.reshape(-1))
        sa = self._slot[a]
        own = sa >= 0
        out = self._rows[np.maximum(np.where(own, sa, self._slot[b]), 0), np.where(own, b, a)]
        return np.where(same, self.unit[a], out) if same.any() else out
