"""Shortcuts for the symmetric matrices and the median behind the Gaussian kernel matrices.

Each function returns the bits of the plain NumPy expression it replaces: :func:`exp_symmetric`
those of ``np.exp``, :func:`strict_upper` those of ``a[np.triu_indices(n, 1)]``, and
:func:`median_exact` those of ``np.median``. A kernel matrix is symmetric, so :func:`exp_symmetric`
exponentiates one triangle and copies it; :func:`gram_is_symmetric` says which inputs make the
squared distances behind it exactly symmetric. The median of the pairwise distances reads two order
statistics, which :func:`median_exact` brackets from a sorted sample instead of partitioning every
pair.
"""

from __future__ import annotations

import numpy as np

#: Rows of the blocks :func:`exp_symmetric` exponentiates at a time.
_BLOCK_ROWS = 64


def gram_is_symmetric(data: np.ndarray) -> bool:
    """Whether ``data @ data.T`` of an ``(n, d)`` matrix equals its transpose bit for bit.

    That holds for one column, whose entries are single products, and for a C- or F-contiguous
    matrix, whose product with its own transpose NumPy forms as one symmetric update. The general
    product of a strided view, such as a reversed or every-other-column view, can differ between
    entries ``(i, j)`` and ``(j, i)`` in the last bit.
    """
    return data.shape[1] == 1 or data.flags.c_contiguous or data.flags.f_contiguous


def exp_symmetric(a: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """``np.exp(a)`` for a square ``a`` that equals its transpose bit for bit.

    Blocks of rows exponentiate the entries on and above the diagonal, and each block's transpose
    fills the entries below it, so the exponential runs on half the entries. ``out`` may be ``a``
    itself; the result is ``out``, or a new array.
    """
    n = a.shape[0]
    if out is None:
        out = np.empty_like(a)
    for top in range(0, n, _BLOCK_ROWS):
        bottom = min(n, top + _BLOCK_ROWS)
        np.exp(a[top:bottom, top:], out=out[top:bottom, top:])
        out[bottom:, top:bottom] = out[top:bottom, bottom:].T
    return out


def strict_upper(a: np.ndarray) -> np.ndarray:
    """The entries above the diagonal of a square matrix, row by row, with no ``n x n`` mask."""
    n = a.shape[0]
    out = np.empty(n * (n - 1) // 2, dtype=a.dtype)
    at = 0
    for i in range(n - 1):
        width = n - 1 - i
        out[at : at + width] = a[i, i + 1 :]
        at += width
    return out


#: Values of the sorted sample that :func:`median_exact` brackets the middle order statistics from.
_SAMPLE = 4096
#: Sample positions the bracket extends past the middle's expected position on each side.
_MARGIN = 64
#: Fewer values than this go straight to ``np.median``.
_MIN_BRACKETED = 65536


def _order_statistics(values: np.ndarray, ks: list[int]) -> list[float] | None:
    """The ``ks``-th smallest values of ``values``, or ``None`` when ``values`` holds a NaN.

    The bracket is two entries of a sorted sample of ``values``. When it holds the ``ks``-th values,
    only the entries inside it are partitioned; otherwise all of ``values`` is, which reorders it.
    """
    m = values.size
    sample = np.sort(values[:: max(1, m // _SAMPLE)])
    s = sample.size
    low = sample[max(0, ks[0] * s // m - _MARGIN)]
    high = sample[min(s - 1, ks[-1] * s // m + _MARGIN)]
    below = int(np.count_nonzero(values < low))
    inside = values >= low
    inside &= values <= high
    window = values[inside]
    above = int(np.count_nonzero(values > high))
    if below + window.size + above != m:  # a NaN is none of the three
        return None
    if below <= ks[0] and ks[-1] < below + window.size:
        local = [k - below for k in ks]
        window.partition(local)
        return [window[k] for k in local]
    values.partition(ks)
    return [values[k] for k in ks]


def median_exact(values: np.ndarray) -> float:
    """``float(np.median(values))`` of a 1-D float array, which this may reorder.

    With a NaN, or fewer than 65,536 values, it is ``np.median`` itself. Otherwise it is the mean
    of the same one or two middle order statistics, found without partitioning every value. The
    sign of a zero median is not defined when ``values`` holds both ``-0.0`` and ``0.0``.
    """
    m = values.size
    if m < _MIN_BRACKETED:
        return float(np.median(values))
    middle = m // 2
    ks = [middle] if m % 2 else [middle - 1, middle]
    picked = _order_statistics(values, ks)
    if picked is None:
        return float(np.median(values))
    return float(picked[0]) if m % 2 else float(np.mean(np.array(picked)))
