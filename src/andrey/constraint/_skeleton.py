"""Batched conditional-independence scanning for the PC and FCI skeleton searches.

:func:`scan_conditioning_sets` tests one adjacent pair against every size-``k`` subset of a
candidate neighborhood. It uses the test's vectorized path when there is one (one stacked
``np.linalg.inv`` per chunk for the partial-correlation tests; one count of every chunk's
contingency tables for the chi-square tests) in place of thousands of scalar
calls, and the scalar loop otherwise, so a test that implements only
:class:`~andrey.core.ci.CITest` still works. Every shipped vectorized path returns the scalar
p-values exactly, so the skeleton is the same either way. Subsets are consumed in fixed-size
chunks, so the scan holds one chunk in memory however many subsets it visits. :func:`scan_pairs`
scans every pair of one conditioning size; for FisherZ, StudentT, RCoT, and the degenerate
Gaussian test it stacks the chunks of consecutive pairs into one evaluation, up to
``_STACK_BYTES`` of submatrices; RCoT evaluates a stack one conditioning set at a time, and the
degenerate Gaussian test regresses on each conditioning set once. :func:`vote_triples` decides the
unshielded triples of a final skeleton by the conservative or majority collider rule.
"""

from __future__ import annotations

import operator
import sys
from collections.abc import (
    Callable,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import nullcontext
from functools import cache, partial
from itertools import chain, combinations, groupby, islice
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np

from andrey.core import NULL, backend
from andrey.core.ci import CITest, DGTest, FisherZ, NormalScores, StudentT, pins_blas
from andrey.core.ci_missing import GaussianDeletion, TestwiseDeletion, not_tested
from andrey.core.orient import separating_set

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints

#: One removed pair: ``(x, y, separating set, the largest p-value among its separating sets)``; the
#: p-value is ``None`` when the scan does not record it.
Removal = tuple[int, int, tuple[int, ...], float | None]
#: The separating set of each removed pair, keyed ``(x, y)``: a dict, or a :class:`MaskedMapping`
#: that reads as one.
SeparatingSets = MutableMapping[tuple[int, int], tuple[int, ...]]
#: The p-value of each removed pair, keyed alike.
PValues = MutableMapping[tuple[int, int], float]
# A pair's subsets that one call evaluates: ``(the pair's position, x, y, subsets)``, ``x < y``,
# the subsets the rows of an ``(N, size)`` int array.
_Chunk = tuple[int, int, int, np.ndarray]

# One batched_call evaluates a chunk of subsets together; cap how many subsets a chunk holds so a
# huge C(|nbr|, k) neighborhood is consumed in bounded chunks, not one array. Each subset's p-value
# is computed independently, so the chunk boundary never changes a result.
CHUNK = 4096

# Conditioning sets per test call in a collider vote (:func:`vote_triples`). A vote keeps only the
# separating subsets, so the tests of a pair hold one chunk however large the endpoint degrees are.
_VOTE_CHUNK = 64

#: How the PC and FCI families and PCMCI+ decide an unshielded triple (:func:`vote_triples`).
COLLIDER_RULES: tuple[str, ...] = ("sepsets", "conservative", "majority")

# The most candidates whose membership in a skip set :func:`subset_rows` reads one by one in Python;
# past it, ``np.isin`` on the whole array is faster.
_SET_LOOKUP_LABELS = 300

# The most entries of the ``(pairs, d)`` candidate mask that one block of the size-1 pass
# (:func:`first_order_removals`) builds.
_FIRST_ORDER_MASK_ENTRIES = 1 << 18

# A block of the size-1 pass lists its candidates from the endpoints' neighbor lists, not from the
# mask, when this many times the listed neighbors are fewer than the mask's entries.
_LISTED_FACTOR = 8

# The most bytes of correlation submatrices one stacked evaluation of scan_pairs holds: a stack of
# B queries of conditioning size k takes 8 * B * (k + 2) ** 2 bytes, and holds at least one chunk.
# Its inverse and index arrays take about twice as much again, in each process that scans.
_STACK_BYTES = 1 << 22


def scan_conditioning_sets(
    test: CITest,
    a: int,
    b: int,
    candidates: list[int],
    size: int,
    alpha: float,
    separating: set[int],
    skip_within: set[int] | None = None,
    trace: list[dict] | None = None,
    largest: list[float] | None = None,
    visit: Callable[[np.ndarray, Sequence[float]], None] | None = None,
) -> bool:
    """Test every size-``size`` subset of ``candidates`` as a conditioning set for ``(a, b)``.

    Each subset whose p-value exceeds ``alpha`` (``a`` conditionally independent of ``b``
    given the subset) is unioned into ``separating`` in place. Returns whether any subset separated
    the pair. With ``largest`` given, the largest p-value among the separating subsets is appended
    to it when one separated the pair; without it, no p-value is compared. There is no early exit:
    the whole subset space is scanned so ``separating`` accumulates the union over every separating
    set.

    With ``skip_within`` given, subsets lying entirely inside it are not tested: the caller passes
    the candidates the pair's other endpoint already scanned at this size, so each shared subset is
    evaluated once per pair. Skipping a repeat changes neither the union nor the return value.

    ``candidates`` are ascending and exclude ``a`` and ``b``, so every subset is a sorted, distinct
    conditioning set. A test that exposes ``batched_call(a, b, condition_sets)`` evaluates the
    subsets in size-bounded chunks (:func:`subset_rows`) through it, a FisherZ, StudentT, or
    NormalScores test each chunk as one ``(N, size)`` int array and any other test as a list of
    tuples; otherwise each subset is tested with a scalar
    ``test(a, b, subset)`` call, or ``test._bounded_pvalue(a, b, subset, alpha, exact_above)``
    where the test has one, with ``exact_above`` true when ``largest`` is given. Both paths visit
    the same subsets in the same order and take the same ``p > alpha`` decisions, so the
    accumulated ``separating``, the largest p-value, and the return value are identical.

    With ``trace`` given, one record per test is appended (see :func:`record_tests`); the check
    runs once per chunk. With ``visit`` given, it is called with each chunk of subsets, as the rows
    of an ``(N, size)`` int array, and their p-values, so a caller can count the tests or keep a few
    of them without storing a record per test. Without ``trace``, a test that deletes rows per test
    asks no subset holding a candidate that no row observes together with ``a`` or ``b``: such a
    query keeps no rows, so it is counted as not tested without being built (:func:`kept_apart`),
    which changes neither the counts nor the return value.
    """
    if trace is None:
        kept, _ = kept_apart(test, (a, b), [(candidates, skip_within)], size)
        if kept is None:
            return False
        [(candidates, skip_within)] = kept
    whole = canonical_capable(test)
    # CMIknn's _bounded_pvalue stops its shuffles once the side of alpha is certain; a p-value
    # above alpha that `largest` records comes from every shuffle.
    bounded = getattr(test, "_bounded_pvalue", None)
    scalar = (
        test if bounded is None else partial(bounded, alpha=alpha, exact_above=largest is not None)
    )
    batched = getattr(test, "batched_call", None) or (
        lambda a, b, chunk: [scalar(a, b, subset) for subset in chunk]
    )
    found = False
    top = 0.0
    for rows in subset_rows(candidates, size, skip_within):
        pvalues = batched(a, b, rows if whole else [tuple(s) for s in rows.tolist()])
        if trace is not None:
            record_tests(trace, size, a, b, rows.tolist(), pvalues, alpha)
        if visit is not None:
            visit(rows, pvalues)
        values = np.asarray(pvalues, dtype=np.float64)
        passing = np.flatnonzero(values > alpha)
        if passing.size:
            found = True
            separating.update(rows[passing].ravel().tolist())
            top = max(top, float(values[passing].max()))
    if found and largest is not None:
        largest.append(top)
    return found


def subset_rows(
    candidates: Sequence[int], size: int, skip_within: set[int] | None
) -> Iterator[np.ndarray]:
    """The size-``size`` subsets of ``candidates`` in ``combinations`` order, less those lying
    entirely inside ``skip_within`` when it is given and holds ``size`` members or more, as the
    rows of ``(N, size)`` int arrays of ``CHUNK`` rows each, the last one fewer and none empty.

    The rows come from the cached position tables, so no subset passes through a tuple: all of a
    size of two or more at once when they fit in one chunk, else a block at a time
    (:func:`blocks`, :func:`block_positions`). Each yielded array is a copy; the tables are never
    written.
    """
    labels = np.asarray(candidates, dtype=np.intp)
    n = len(labels)
    inside = None
    if skip_within is not None and len(skip_within) >= size:
        inside = _members(labels, skip_within)
    if 1 < size <= n and comb(n, size) <= CHUNK:
        rows = _combination_table(n, size)
        if inside is not None:
            rows = rows[~inside[rows].all(axis=1)]
        if len(rows):
            yield labels[rows]
        return
    held: list[np.ndarray] = []
    count = 0
    for block in blocks(n, size, CHUNK):
        rows = block_positions(n, block)
        if inside is not None:
            rows = rows[~inside[rows].all(axis=1)]
            if not len(rows):
                continue
        held.append(rows)
        count += len(rows)
        while count >= CHUNK:
            rows = np.concatenate(held)
            yield labels[rows[:CHUNK]]
            held, count = [rows[CHUNK:]], count - CHUNK
    if count:
        yield labels[np.concatenate(held)]


def _members(labels: np.ndarray, members: set[int]) -> np.ndarray:
    """Whether each of ``labels`` is in ``members``: a set lookup for up to
    ``_SET_LOOKUP_LABELS`` labels, ``np.isin`` for more."""
    if len(labels) > _SET_LOOKUP_LABELS:
        return np.isin(labels, np.fromiter(members, np.intp, len(members)))
    return np.fromiter((label in members for label in labels.tolist()), bool, len(labels))


def kept_apart(
    test: CITest,
    fixed: Sequence[int],
    sides: list[tuple[list[int], set[int] | None]],
    size: int,
) -> tuple[list[tuple[list[int], set[int] | None]] | None, int]:
    """The ``sides`` of a scan less the candidates no row observes together with a column of
    ``fixed``, the columns every query of the scan holds, its pair first, and the number of sets
    left out.

    Each side is ``(candidates, skip_within)``, its sets the size-``size`` subsets of the
    candidates less those lying inside ``skip_within`` (:func:`subset_rows`). A set holding a
    dropped candidate keeps no rows under a test that deletes rows per test, so each such set is
    counted as a query of the pair not tested, without being built, as asking it would count it
    (:meth:`~andrey.core.ci_missing.TestwiseDeletion.count_not_tested`), and kept by the dropped
    candidate with the other columns of ``fixed``, which it holds. When no row observes the
    columns of ``fixed`` together, every set is counted and ``None`` is returned in place of the
    sides. Another test gets ``sides`` unchanged and none left out.
    """
    if not isinstance(test, TestwiseDeletion):
        return sides, 0
    together = test.together()
    columns = list(fixed)
    met: np.ndarray = np.all(together[columns], axis=0)
    whole = not met[columns].all()
    kept = [([c for c in side if met[c]], skip) for side, skip in sides]
    left = sum(_subset_count(side, size, skip) for side, skip in sides)
    held = columns[2:]
    if whole:
        keeping = [held]
    else:
        left -= sum(_subset_count(side, size, skip) for side, skip in kept)
        dropped = {c for side, _ in sides for c in side if not met[c]}
        keeping = [[c, *held] for c in sorted(dropped)]
    test.count_not_tested(columns[0], columns[1], left, keeping)
    return None if whole else kept, left


def _subset_count(candidates: list[int], size: int, skip_within: set[int] | None) -> int:
    """The number of sets :func:`subset_rows` yields."""
    count = comb(len(candidates), size)
    if skip_within is not None and len(skip_within) >= size:
        count -= comb(len(skip_within.intersection(candidates)), size)
    return count


def canonical_capable(test: CITest) -> bool:
    """Whether ``test`` takes canonical query arrays and runs on the worker pool: a FisherZ,
    StudentT, or NormalScores test, exactly.

    A subclass or a wrapper of either keeps the ``batched_call`` and ``first_order_pvalues`` it
    may override, and a worker cannot rebuild it from the correlation matrix, so it scans in the
    caller's process; the test-wise deletion of either takes the canonical query arrays there
    (:func:`_canonical_route`), and any other one call at a time.
    """
    return type(test) in (FisherZ, StudentT, NormalScores)


def _canonical_route(test: CITest) -> Callable[..., np.ndarray] | None:
    """The ``_canonical_pvalues``, which the canonical query arrays take, of a
    :func:`canonical_capable` test, of the test-wise deletion of either
    (:class:`~andrey.core.ci_missing.GaussianDeletion`), or of a degenerate Gaussian test, exactly,
    else ``None``.

    A subclass of the degenerate Gaussian test may change what a query returns, so it keeps its
    scalar calls.
    """
    if canonical_capable(test) or isinstance(test, GaussianDeletion) or type(test) is DGTest:
        return test._canonical_pvalues  # ty: ignore[unresolved-attribute]  # gated by the type
    return None


def _stacked_route(test: CITest) -> Callable[..., np.ndarray] | None:
    """The ``_canonical_pvalues`` that :func:`scan_pairs` stacks the chunks of many pairs into:
    :func:`_canonical_route`'s, or an RCoT test's, exactly, which evaluates a stack one
    conditioning set at a time; else ``None``."""
    canonical = _canonical_route(test)
    if canonical is not None:
        return canonical
    if is_rcot(test):
        return test._canonical_pvalues  # ty: ignore[unresolved-attribute]  # gated by the type
    return None


def is_rcot(test: CITest) -> bool:
    """Whether ``test`` is an RCoT test, exactly: a subclass may change what a query returns."""
    # An RCoT test has loaded its module; a fit with another test leaves it unloaded.
    kernel = sys.modules.get("andrey.core.ci_kernel")
    return kernel is not None and type(test) is kernel.RCoT


def _query_arrays(
    size: int, queries: Sequence[tuple[int, int, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The canonical query arrays ``(x, y, conditions)`` of ``queries``, in order.

    Each query is ``(x, y, subsets)`` with ``x < y``; the pair is tested against each row of the
    ``(N, size)`` int array ``subsets``, a sorted, distinct subset.
    """
    counts = [len(subsets) for _, _, subsets in queries]
    xs = np.repeat(np.array([x for x, _, _ in queries], dtype=np.intp), counts)
    ys = np.repeat(np.array([y for _, y, _ in queries], dtype=np.intp), counts)
    zs = np.concatenate([subsets for _, _, subsets in queries]).reshape(sum(counts), size)
    return xs, ys, zs


def _canonical_chunk(
    pvalues: Callable[..., np.ndarray], size: int, a: int, b: int, chunk: np.ndarray
) -> np.ndarray:
    """``pvalues`` of the pair ``(a, b)`` given each row of ``chunk``: ``size`` members, sorted
    and distinct, as the subsets of ascending candidates are."""
    return pvalues(*_query_arrays(size, [(min(a, b), max(a, b), chunk)]))


def marginal_separated(
    test: CITest, xs: np.ndarray, ys: np.ndarray, alpha: float
) -> np.ndarray | None:
    """Whether the marginal test separates each pair ``(xs[i], ys[i])``, ``xs[i] < ys[i]``.

    A FisherZ, StudentT, or degenerate Gaussian test answers every pair in one canonical query
    array, with the p-values of the scalar calls. Another test returns ``None``: its scalar calls
    keep their order. So does a call with no pair, whose query array a one-column test cannot
    index.
    """
    canonical = _canonical_route(test)
    if canonical is None or not len(xs):
        return None
    return canonical(xs, ys, np.empty((len(xs), 0), dtype=np.intp)) > alpha


def separating_subsets(
    test: CITest,
    x: int,
    y: int,
    size: int,
    subsets: Iterator[tuple[int, ...]],
    alpha: float,
) -> list[tuple[int, ...]]:
    """The ``subsets`` (sorted, distinct, of ``size`` members) whose test separates ``x < y``.

    A FisherZ, StudentT, or degenerate Gaussian test evaluates the subsets in chunks of ``CHUNK``
    canonical queries, in order. A chunk whose evaluation raises is tested again subset by subset
    with scalar ``test`` calls, so the error is the one the scalar loop raises, after the same
    queries. Another test makes one scalar call per subset.
    """
    canonical = _canonical_route(test)
    if canonical is None:
        return [cond for cond in subsets if test(x, y, cond) > alpha]
    found: list[tuple[int, ...]] = []
    while chunk := list(islice(subsets, CHUNK)):
        rows = np.fromiter(chain.from_iterable(chunk), np.intp, len(chunk) * size)
        try:
            pvalues = _canonical_chunk(canonical, size, x, y, rows.reshape(len(chunk), size))
        except Exception:
            pvalues = [test(x, y, cond) for cond in chunk]
        found.extend(cond for cond, p in zip(chunk, pvalues) if p > alpha)
    return found


def record_tests(
    trace: list[dict],
    depth: int,
    a: int,
    b: int,
    subsets: Iterable[tuple[int, ...]],
    pvalues: Iterable[float],
    alpha: float,
) -> None:
    """Append one record per test of the pair ``(a, b)``, lower index first as ``(x, y)``.

    A record holds ``step="test"``, ``depth``, ``x``, ``y``, the conditioning set ``S``, the p-value
    ``p``, and ``removed``: whether this test separated the pair (``p > alpha``).
    """
    x, y = min(a, b), max(a, b)
    trace.extend(
        dict(step="test", depth=depth, x=x, y=y, S=tuple(s), p=float(p), removed=bool(p > alpha))
        for s, p in zip(subsets, pvalues)
    )


# ---- vectorized size-0/1 prescreen ---------------------------------------------------------------
#
# A CI test that exposes vectorized marginal (|S| = 0) and first-order (|S| = 1) p-values
# (``marginal_pvalues`` and ``first_order_pvalues``: the partial-correlation closed forms over the
# correlation matrix, the chi-square tables counted together, the conditional Gaussian sets fit
# together) lets the size-0 pass run as a single all-pairs matrix threshold and the size-1 pass as
# one vectorized read per surviving pair, or per stack of pairs for FisherZ and StudentT, in place
# of the dense per-pair scalar loop. The decisions (p > alpha) and separating sets are identical to
# the scalar scan (the p-values match exactly), so the recovered skeleton is unchanged.


def prescreen_capable(test: CITest) -> bool:
    """Whether ``test`` exposes the vectorized size-0/1 prescreen primitives.

    The partial-correlation, chi-square, and conditional Gaussian tests do.
    """
    return callable(getattr(test, "marginal_pvalues", None)) and callable(
        getattr(test, "first_order_pvalues", None)
    )


def marginal_removals(
    test: CITest, adjacent_upper: np.ndarray, alpha: float, trace: list[dict] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(size 0) The still-adjacent pairs the marginal test separates, and the marginal p-values.

    ``adjacent_upper`` is the strictly-upper-triangular boolean adjacency mask. One vectorized
    all-pairs evaluation gives every pair's marginal p-value; the first return marks, in a boolean
    matrix of the same shape, the adjacent pairs with ``p > alpha``, each separated by the empty
    set. The second is the ``(d, d)`` p-value matrix, whose upper-triangle entry ``[x, y]`` is the
    test's p-value for the pair. With ``trace`` given, every adjacent pair's test is recorded, in
    row-major order.
    """
    pmat = test.marginal_pvalues()  # ty: ignore[unresolved-attribute]  # gated by prescreen_capable
    if trace is not None:
        for x, y in zip(*np.nonzero(adjacent_upper)):
            record_tests(trace, 0, int(x), int(y), [()], [pmat[x, y]], alpha)
    return adjacent_upper & (pmat > alpha), pmat


class MaskedMapping(MutableMapping):
    """A dict whose entries at the set bits of a boolean ``mask`` are held as the mask, with every
    other entry in a plain dict on top of it.

    ``mask`` is a strictly upper-triangular ``(d, d)`` matrix, whose bit ``[x, y]`` marks the pair
    key ``(x, y)``, or a ``(d,)`` vector, whose bit ``[x]`` marks the integer key ``x``. A marked
    key maps to the empty tuple, or, with ``values``, to ``float(values[key])``. Most pairs of a
    sparse graph are separated by the empty set, in the size-0 pass of the PC family and in MMPC's
    forward phase: the mask holds each such entry in one byte, where a dict holds a key object and
    a value object for it.

    It behaves as the dict that holds the marked keys first, in row-major order, and then each key
    set in it, in the order set: a key set over a marked one keeps its place and maps to the new
    value, and a deleted key is gone. Iteration, ``len``, ``==`` with a dict, ``[]``, ``get``, and
    ``in`` agree with that dict. A reversed or negative pair, a key outside the mask, and a key that
    is neither an integer nor a pair of integers (read with :func:`operator.index`) are marked by
    no bit. :meth:`copy` returns that dict, and :meth:`compact` returns it in place of the mapping
    where a dict entry for each marked key takes fewer bytes than the mask and ``values``. Two
    operations differ from the dict's: :func:`copy.copy` returns a mapping that shares the mask
    and the dict on top with the original, and ``popitem`` removes the first entry, not the last.
    A read-only ``mask`` is copied before a deletion or :meth:`mark` writes it, so two mappings may
    share one.
    """

    def __init__(self, mask: np.ndarray, values: np.ndarray | None = None) -> None:
        self._mask = mask
        self._values = values
        self._marked = int(np.count_nonzero(mask))
        self._overlay: dict = {}

    def _bit(self, key: Any) -> tuple[int, ...] | None:
        """The mask index of ``key`` when its bit is set, ``None`` otherwise."""
        mask = self._mask
        try:
            at = (operator.index(key),) if mask.ndim == 1 else tuple(operator.index(k) for k in key)
        except TypeError:
            return None
        if mask.ndim == 1:
            valid = len(at) == 1 and 0 <= at[0] < mask.shape[0]
        else:
            valid = len(at) == 2 and 0 <= at[0] < at[1] < mask.shape[0]
        return at if valid and mask[at] else None

    def _value(self, at: tuple[int, ...]) -> Any:
        return () if self._values is None else float(self._values[at])

    def __getitem__(self, key: Any) -> Any:
        if key in self._overlay:
            return self._overlay[key]
        at = self._bit(key)
        if at is None:
            raise KeyError(key)
        return self._value(at)

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._overlay:
            return self._overlay[key]
        at = self._bit(key)
        return default if at is None else self._value(at)

    def __contains__(self, key: object) -> bool:
        return key in self._overlay or self._bit(key) is not None

    def __setitem__(self, key: Any, value: Any) -> None:
        self._overlay[key] = value

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._overlay.update(*args, **kwargs)

    def __delitem__(self, key: Any) -> None:
        at = self._bit(key)
        if self._overlay.pop(key, _ABSENT) is _ABSENT and at is None:
            raise KeyError(key)
        if at is not None:
            self._writable()[at] = False
            self._marked -= 1

    def mark(self, index: Any) -> None:
        """Set the bits ``index`` of the mask, a NumPy index, of an empty mapping, so the keys
        marked come first. Raises ``ValueError`` when the mapping holds an entry."""
        if self._marked or self._overlay:
            raise ValueError("only an empty MaskedMapping takes new marks")
        mask = self._writable()
        mask[index] = True
        self._marked = int(np.count_nonzero(mask))

    def _writable(self) -> np.ndarray:
        """The mask, copied first when it is read-only."""
        if not self._mask.flags.writeable:
            self._mask = self._mask.copy()
        return self._mask

    def _marked_keys(self) -> Iterator:
        if self._mask.ndim == 1:
            yield from np.flatnonzero(self._mask).tolist()
            return
        for x in np.flatnonzero(self._mask.any(axis=1)).tolist():
            for y in np.flatnonzero(self._mask[x]).tolist():
                yield (x, y)

    def __iter__(self) -> Iterator:
        yield from self._marked_keys()
        yield from (key for key in self._overlay if self._bit(key) is None)

    def __len__(self) -> int:
        return self._marked + sum(1 for key in self._overlay if self._bit(key) is None)

    def copy(self) -> dict[Any, Any]:
        """The plain dict with the same entries in the same order, built without a lookup per
        marked key."""
        found = np.nonzero(self._mask)
        keys = (
            found[0].tolist() if self._mask.ndim == 1 else zip(*(axis.tolist() for axis in found))
        )
        if self._values is None:
            out = dict.fromkeys(keys, ())
        else:
            out = dict(zip(keys, self._values[found].tolist(), strict=True))
        out.update(self._overlay)
        return out

    def compact(self) -> MaskedMapping | dict[Any, Any]:
        """This mapping, or, when a dict entry for each marked key takes fewer bytes than the mask
        and ``values`` together, the dict it reads as (:meth:`copy`)."""
        held = self._mask.nbytes + (0 if self._values is None else self._values.nbytes)
        return self if self._marked * _ENTRY_BYTES[self._mask.ndim] >= held else self.copy()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.copy()!r})"


_ABSENT = object()
#: A floor on the bytes one dict entry takes, its key included, by the number of axes of the mask
#: that marks it: an integer key, whose integer the search already holds, takes at least about 26
#: bytes, and a pair key at least about 87 once the dict holds a few thousand entries (CPython 3.11
#: and 3.14).
_ENTRY_BYTES = {1: 24, 2: 80}


def first_order_removals(
    test: CITest,
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    alpha: float,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[Removal]:
    """(size 1) The ``pairs`` a single conditioner separates, as ``(x, y, separating, p)``.

    Each pair ``x < y`` is tested against its start-of-size snapshot conditioners: either
    endpoint's ``neighbors`` minus the pair, in PC and FCI alike. ``separating`` holds the
    conditioners whose test separates the pair, ascending (the separating-set key order), and ``p``
    is the largest p-value among the pair's tests, or ``None`` when ``record_pvalues`` is false.

    Without a ``trace``, a test that takes canonical query arrays (:func:`_canonical_route`) and
    deletes no rows per test has its queries built from arrays (:func:`_array_first_order`): the
    pass holds a ``(d, d)`` boolean adjacency matrix, a candidate mask of at most
    ``_FIRST_ORDER_MASK_ENTRIES`` entries, and one stack, however many pairs and conditioners it
    tests. Any other pass goes pair by pair, building the queries as it evaluates them, so it holds
    one stack however many pairs and conditioners it tests: a FisherZ or StudentT test evaluates
    the queries of consecutive pairs in one stack (:func:`_stacked_first_order`), another test
    each pair's candidates in one ``first_order_pvalues`` call. With ``trace`` given, every test
    is recorded, pair by pair.
    """
    canonical = _canonical_route(test)
    if trace is None and canonical is not None and not isinstance(test, TestwiseDeletion):
        found = _array_first_order(canonical, pairs, neighbors, alpha, record_pvalues)
        if found is not None:
            return found
    queries = _first_order_queries(pairs, neighbors, test if trace is None else None)
    tested = (
        _stacked_first_order(canonical, queries)
        if canonical is not None
        else (
            (x, y, c, test.first_order_pvalues(x, y, c))  # ty: ignore[unresolved-attribute]
            for x, y, c in queries
        )
    )
    removals: list[Removal] = []
    for x, y, candidates, pvals in tested:
        if trace is not None:
            record_tests(trace, 1, x, y, [(z,) for z in candidates], pvals, alpha)
        passing = np.flatnonzero(pvals > alpha)
        if passing.size:
            # The largest separating p-value: a query not tested (NaN) never separates.
            p = max(pvals[passing].tolist()) if record_pvalues else None
            removals.append((x, y, tuple(candidates[k] for k in passing.tolist()), p))
    return removals


def _array_first_order(
    pvalues: Callable[..., np.ndarray],
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    alpha: float,
    record_pvalues: bool,
) -> list[Removal] | None:
    """:func:`first_order_removals` from arrays, for ``pvalues`` that take canonical query arrays.

    A block of pairs takes its queries from the candidates of both endpoints less the pair, pair
    by pair and each pair's candidates ascending (:func:`_mask_candidates`, or
    :func:`_listed_candidates` when the neighbor lists are short against the block's mask),
    evaluates them in stacks of ``_STACK_BYTES`` of submatrices, and cuts each pair's separating
    candidates and largest p-value from the results. Each p-value is the one
    :func:`_stacked_first_order` gets, as the first-order closed form is elementwise, so the
    removals are the same. ``None`` when an evaluation raises, so the caller's pair-by-pair pass
    raises the error it raises.
    """
    d = len(neighbors)
    adjacent = np.zeros((d, d), dtype=bool)
    for i, nb in enumerate(neighbors):
        adjacent[i, nb] = True
    both = np.array(pairs, dtype=np.intp).reshape(-1, 2)
    sizes = np.fromiter((nb.size for nb in neighbors), np.intp, d)
    flat = np.concatenate(neighbors) if sizes.sum() else np.empty(0, dtype=np.intp)
    offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    step = max(1, _FIRST_ORDER_MASK_ENTRIES // max(d, 1))
    limit = max(1, _STACK_BYTES // (8 * 3**2))
    removals: list[Removal] = []
    for start in range(0, len(both), step):
        xs, ys = both[start : start + step, 0], both[start : start + step, 1]
        listed = int(sizes[xs].sum() + sizes[ys].sum())
        if _LISTED_FACTOR * listed < len(xs) * d:
            which, zs = _listed_candidates(flat, offsets, sizes, xs, ys, d)
        else:
            which, zs = _mask_candidates(adjacent, xs, ys)
        if not which.size:
            continue
        qx, qy = xs[which], ys[which]
        values = np.empty(which.size, dtype=np.float64)
        try:
            for at in range(0, which.size, limit):
                cut = slice(at, at + limit)
                values[cut] = pvalues(qx[cut], qy[cut], zs[cut].reshape(-1, 1))
        except Exception:
            return None
        passing = values > alpha
        if not passing.any():
            continue
        which, zs, values = which[passing], zs[passing], values[passing]
        starts = np.flatnonzero(np.diff(which, prepend=-1))
        stops = np.append(starts[1:], which.size).tolist()
        tops = np.maximum.reduceat(values, starts).tolist() if record_pvalues else None
        members = zs.tolist()
        for k, (i, a, b) in enumerate(zip(which[starts].tolist(), starts.tolist(), stops)):
            x, y = pairs[start + i]
            removals.append((x, y, tuple(members[a:b]), None if tops is None else tops[k]))
    return removals


def _mask_candidates(
    adjacent: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(pair index, candidate)`` of each query of the pairs ``(xs[i], ys[i])``, in pair order and
    candidates ascending: the rows of ``adjacent[xs] | adjacent[ys]`` less ``xs[i]`` and
    ``ys[i]``."""
    mask = adjacent[xs] | adjacent[ys]
    rows = np.arange(len(xs))
    mask[rows, xs] = False
    mask[rows, ys] = False
    which, zs = np.nonzero(mask)
    return which, zs


def _listed_candidates(
    flat: np.ndarray, offsets: np.ndarray, sizes: np.ndarray, xs: np.ndarray, ys: np.ndarray, d: int
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_mask_candidates` from the neighbor lists, concatenated in ``flat`` with the start of
    column ``i``'s at ``offsets[i]`` and its length ``sizes[i]``: both endpoints' lists, less the
    pair, sorted by ``pair * d + candidate`` and deduplicated, so a block of pairs in a wide, sparse
    graph costs its listed neighbors, not ``len(xs) * d`` mask entries."""
    ends = np.concatenate((xs, ys))
    owner = np.concatenate((np.arange(len(xs)), np.arange(len(ys))))
    lens = sizes[ends]
    total = int(lens.sum())
    if not total:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    first = np.repeat(offsets[ends] - (np.cumsum(lens) - lens), lens) + np.arange(total)
    zs = flat[first]
    pair = np.repeat(owner, lens)
    keep = (zs != xs[pair]) & (zs != ys[pair])
    keys = np.unique(pair[keep] * d + zs[keep])
    return keys // d, keys % d


def _first_order_queries(
    pairs: list[tuple[int, int]], neighbors: list[np.ndarray], test: CITest | None = None
) -> Iterator[tuple[int, int, list[int]]]:
    """Yield ``(x, y, candidates)`` for each of ``pairs`` with a candidate, in order: the
    ascending conditioners of :func:`first_order_removals`. With ``test``, a candidate no row
    observes together with ``x`` or ``y`` is left out and counted as not tested
    (:func:`kept_apart`)."""
    for x, y in pairs:
        candidates = sorted((set(neighbors[x].tolist()) | set(neighbors[y].tolist())) - {x, y})
        if test is not None and candidates:
            kept, _ = kept_apart(test, (x, y), [(candidates, None)], 1)
            candidates = [] if kept is None else kept[0][0]
        if candidates:
            yield x, y, candidates


def _stacked_first_order(
    pvalues: Callable[..., np.ndarray],
    queries: Iterable[tuple[int, int, list[int]]],
) -> Iterator[tuple[int, int, list[int], np.ndarray]]:
    """Each query of :func:`_first_order_queries` as ``(x, y, candidates, p-values)``, with p-values
    in candidate order.

    ``pvalues`` takes canonical query arrays. Consecutive queries are stacked, whole, as size-1
    chunks (:func:`_stacks`); the first-order closed form is elementwise, so each p-value is the one
    a query evaluated alone gets.
    """
    chunks = (
        (i, x, y, np.array(c, dtype=np.intp).reshape(-1, 1)) for i, (x, y, c) in enumerate(queries)
    )
    for stack, values in _stacks(pvalues, 1, chunks):
        stop = 0
        for _, x, y, rows in stack:
            start, stop = stop, stop + len(rows)
            yield x, y, rows[:, 0].tolist(), values[start:stop]


def scan_pairs(
    test: CITest,
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    size: int,
    alpha: float,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[Removal]:
    """Scan adjacent ``pairs`` at conditioning ``size`` against a start-of-size neighbor snapshot.

    Returns ``(x, y, separating, p)`` for every pair some size-``size`` conditioning set separates,
    with ``p`` the largest p-value among the separating tests, or ``None`` when ``record_pvalues``
    is false. Each pair is tested from either endpoint and the separating sets are unioned, so the
    removals do not depend on column order; a subset both endpoints offer is tested once, from the
    first.
    This is the general ``|S| >= 2`` scan the vectorized prescreen does not cover; the serial loop
    and the parallel workers share it, so they stay identical. A FisherZ, StudentT, RCoT, or
    degenerate Gaussian test evaluates the chunks of consecutive pairs in one stack
    (:func:`stacked_scan`); another test scans each pair with :func:`scan_conditioning_sets`. Both
    test the same subsets in the same order. With ``trace`` given, every test is recorded in the
    order it runs.
    """
    stacked = _stacked_route(test)
    if stacked is not None:
        apart = test if trace is None else None
        return stacked_scan(stacked, pairs, neighbors, size, alpha, trace, record_pvalues, apart)
    removals: list[Removal] = []
    for x, y in pairs:
        separating: set[int] = set()
        found = False
        largest: list[float] | None = [] if record_pvalues else None
        for a, b, candidates, seen in _sides(neighbors, x, y, size):
            if scan_conditioning_sets(
                test, a, b, candidates, size, alpha, separating, seen, trace, largest
            ):
                found = True
        if found:
            removals.append((x, y, tuple(sorted(separating)), max(largest) if largest else None))
    return removals


def _sides(
    neighbors: list[np.ndarray], x: int, y: int, size: int
) -> Iterator[tuple[int, int, list[int], set[int] | None]]:
    """The sides :func:`scan_pairs` tests the pair ``(x, y)`` from, as ``(a, b, candidates,
    skip_within)`` for :func:`scan_conditioning_sets`: ``x`` first, then ``y``, each side's
    candidates its other neighbors, and a side with fewer than ``size`` of them left out."""
    from_x = neighbors[x][neighbors[x] != y].tolist()
    from_y = neighbors[y][neighbors[y] != x].tolist()
    # Once the x side has scanned, the y side skips the subsets both sides offer.
    shared = set(from_x) & set(from_y) if len(from_x) >= size else None
    for a, b, candidates, seen in ((x, y, from_x, None), (y, x, from_y, shared)):
        if len(candidates) >= size:
            yield a, b, candidates, seen


def stacked_scan(
    pvalues: Callable[..., np.ndarray],
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    size: int,
    alpha: float,
    trace: list[dict] | None,
    record_pvalues: bool,
    test: CITest | None = None,
) -> list[Removal]:
    """:func:`scan_pairs` for a test whose ``pvalues`` take canonical query arrays.

    Each chunk is the one :func:`scan_conditioning_sets` evaluates in one call, in the per-pair
    order (:func:`_pair_chunks`); :func:`_stacks` evaluates consecutive chunks together. Every
    p-value is the one its chunk gets alone, and the chunks are read back in order, so the
    separating sets, the largest p-values, and the trace are those of the per-pair scan. With
    ``test``, the one ``pvalues`` belongs to, a set holding a candidate no row observes together
    with the pair is counted as not tested without being asked (:func:`kept_apart`), as
    :func:`scan_conditioning_sets` counts it.
    """
    separated: dict[int, tuple[set[int], list[float]]] = {}
    for stack, values in _stacks(pvalues, size, _pair_chunks(pairs, neighbors, size, test)):
        bounds = np.cumsum([0, *(len(chunk) for _, _, _, chunk in stack)]).tolist()
        passing = values > alpha
        if trace is not None:
            for (_, x, y, chunk), start, stop in zip(stack, bounds, bounds[1:]):
                record_tests(trace, size, x, y, chunk.tolist(), values[start:stop], alpha)
        for k in np.flatnonzero(np.logical_or.reduceat(passing, bounds[:-1])).tolist():
            i, _, _, chunk = stack[k]
            rows = np.flatnonzero(passing[bounds[k] : bounds[k + 1]])
            members, tops = separated.setdefault(i, (set(), []))
            members.update(chunk[rows].ravel().tolist())
            if record_pvalues:
                tops.append(float(values[bounds[k] + rows].max()))
    removals: list[Removal] = []
    for i, (members, tops) in separated.items():
        x, y = pairs[i]
        removals.append((x, y, tuple(sorted(members)), max(tops) if record_pvalues else None))
    return removals


def _pair_chunks(
    pairs: list[tuple[int, int]],
    neighbors: list[np.ndarray],
    size: int,
    test: CITest | None = None,
) -> Iterator[_Chunk]:
    """Yield ``(i, x, y, chunk)``, ``x < y``, for each chunk of subsets the per-pair scan of
    :func:`scan_pairs` evaluates in one call, in its order; ``i`` is the pair's position. With
    ``test``, each side leaves out the candidates :func:`kept_apart` drops. Size 0 under test-wise
    deletion reads them from arrays (:func:`_marginal_chunks`)."""
    if size == 0 and isinstance(test, TestwiseDeletion):
        yield from _marginal_chunks(pairs, test)
        return
    for i, (a, b) in enumerate(pairs):
        x, y = min(a, b), max(a, b)
        for side_a, side_b, candidates, seen in _sides(neighbors, a, b, size):
            if test is not None:
                kept, _ = kept_apart(test, (side_a, side_b), [(candidates, seen)], size)
                if kept is None:
                    continue
                [(candidates, seen)] = kept
            for chunk in subset_rows(candidates, size, seen):
                yield i, x, y, chunk


# The one set of size 0, as the rows of an ``(N, 0)`` int array; never written.
_NO_SET = np.empty((1, 0), dtype=np.intp)
_NO_SET.flags.writeable = False


def _marginal_chunks(pairs: list[tuple[int, int]], test: TestwiseDeletion) -> Iterator[_Chunk]:
    """:func:`_pair_chunks` at size 0 under test-wise deletion, from the matrix ``test.together()``
    that is read once.

    A pair some row observes together yields the empty set as its one chunk: the first endpoint
    tests it, and the second skips what the first tested. Any other pair is counted as one query
    not tested, as :func:`kept_apart` counts it, when the scan reaches it, so the counts and the
    chunks interleave as the per-pair scan's do.
    """
    if not pairs:
        return
    together = test.together()
    flat = np.fromiter(chain.from_iterable(pairs), np.intp, 2 * len(pairs))
    a, b = flat[0::2], flat[1::2]
    observed = (together[a, a] & together[b, a] & together[a, b] & together[b, b]).tolist()
    for i, ((x, y), seen) in enumerate(zip(pairs, observed)):
        if seen:
            yield (i, x, y, _NO_SET) if x < y else (i, y, x, _NO_SET)
        else:
            test.count_not_tested(x, y, 1, [[]])


def _stacks(
    pvalues: Callable[..., np.ndarray],
    size: int,
    chunks: Iterable[_Chunk],
) -> Iterator[tuple[list[_Chunk], np.ndarray]]:
    """Yield ``(stack, values)``: consecutive ``chunks``, whole, and the p-values of their queries.

    A stack holds as many chunks as fit in ``_STACK_BYTES`` of correlation submatrices, a query
    taking ``(size + 2) ** 2`` entries of 8 bytes, at least one (:func:`_evaluated`).
    """
    limit = max(1, _STACK_BYTES // (8 * (size + 2) ** 2))
    stack: list[_Chunk] = []
    rows = 0
    for item in chunks:
        if stack and rows + len(item[3]) > limit:
            yield from _evaluated(pvalues, size, stack)
            stack, rows = [], 0
        stack.append(item)
        rows += len(item[3])
    if stack:
        yield from _evaluated(pvalues, size, stack)


def _evaluated(
    pvalues: Callable[..., np.ndarray], size: int, stack: list[_Chunk]
) -> Iterator[tuple[list[_Chunk], np.ndarray]]:
    """Yield ``(stack, values)`` for the queries of ``stack`` evaluated in one call.

    When that call raises, each chunk is evaluated in a call of its own and yielded alone, so the
    error comes from the chunk the per-pair scan raises on, after the same calls.
    """
    arrays = _query_arrays(size, [item[1:] for item in stack])
    try:
        values = pvalues(*arrays)
    except Exception:
        values = None
    if values is not None:
        yield stack, values
        return
    for item in stack:
        yield [item], _canonical_chunk(pvalues, size, *item[1:])


def testable_queries(test: CITest, neighbors: list[np.ndarray], x: int, y: int, size: int) -> int:
    """:func:`pair_queries` excluding sets with a member that keeps no rows with the pair under
    test-wise deletion (:func:`kept_apart`): the queries the scan builds."""
    if not isinstance(test, TestwiseDeletion):
        return pair_queries(neighbors, x, y, size)
    together = test.together()
    if not together[x, y]:
        return 0
    met = together[x] & together[y]
    from_x = {c for c in neighbors[x].tolist() if c != y and met[c]}
    from_y = {c for c in neighbors[y].tolist() if c != x and met[c]}
    return comb(len(from_x), size) + comb(len(from_y), size) - comb(len(from_x & from_y), size)


def pair_queries(neighbors: list[np.ndarray], x: int, y: int, size: int) -> int:
    """The tests :func:`scan_pairs` runs on the adjacent pair ``(x, y)`` at conditioning ``size``.

    It counts the size-``size`` subsets of either endpoint's other neighbors, a subset both offer
    once.
    """
    shared = len(set(neighbors[x].tolist()) & set(neighbors[y].tolist()))
    return (
        comb(neighbors[x].size - 1, size) + comb(neighbors[y].size - 1, size) - comb(shared, size)
    )


# ---- possible-d-sep subsets ---------------------------------------------------------------------
#
# FCI's possible-d-sep step searches the subsets of a pair's candidate set by increasing size, each
# size in ``combinations`` order, for the first that separates the pair. The subsets are cut into
# blocks of positions in the candidate list, enumerated as arrays, and asked in batched calls.

#: A block ``(prefix, rest)`` of a size holds every subset of the candidates that starts with the
#: candidates at the positions ``prefix`` and takes its ``rest`` other members from the positions
#: after the last of them, in ``combinations`` order.
Block = tuple[tuple[int, ...], int]

#: The most candidates a search enumerates as arrays: a subset's positions form a non-negative
#: int64 bit mask.
MASK_BITS = 63


def block_subsets(n: int, block: Block) -> int:
    """The subsets ``block`` holds of ``n`` candidates."""
    prefix, rest = block
    return comb(n - (prefix[-1] + 1 if prefix else 0), rest)


def blocks(n: int, size: int, most: int) -> Iterator[Block]:
    """``combinations(range(n), size)`` cut into contiguous blocks of at most ``most`` subsets, in
    order: a block too large is split by the position its next member takes."""

    def split(prefix: tuple[int, ...], start: int, rest: int) -> Iterator[Block]:
        if comb(n - start, rest) <= most:
            yield prefix, rest
            return
        for i in range(start, n - rest + 1):
            yield from split((*prefix, i), i + 1, rest - 1)

    return split((), 0, size)


@cache
def _combination_table(m: int, r: int) -> np.ndarray:
    """``combinations(range(m), r)`` as a ``(comb(m, r), r)`` array, for ``r`` other than 1. A block
    holds at most a few thousand subsets, so the tables stay small."""
    count = comb(m, r)
    flat = np.fromiter(chain.from_iterable(combinations(range(m), r)), np.intp, count * r)
    return flat.reshape(count, r)


def block_positions(n: int, block: Block) -> np.ndarray:
    """The subsets of ``block`` of ``n`` candidates as rows of positions, in order."""
    prefix, rest = block
    start = prefix[-1] + 1 if prefix else 0
    tail = (
        np.arange(n - start, dtype=np.intp)[:, None]
        if rest == 1
        else _combination_table(n - start, rest)
    )
    rows = np.empty((len(tail), len(prefix) + rest), dtype=np.intp)
    rows[:, : len(prefix)] = prefix
    rows[:, len(prefix) :] = tail + start
    return rows


def _runs(groups: Iterable[Iterable[np.ndarray]], first: int, most: int) -> Iterator[np.ndarray]:
    """The rows of each group's arrays, in order, in runs of ``first``, twice that, and so on up to
    ``most`` rows. A run holds the rows of one group, and a group's last run the rest of them."""
    run = first
    for arrays in groups:
        held: list[np.ndarray] = []
        count = 0
        for array in arrays:
            held.append(array)
            count += len(array)
            while count >= run:
                rows = np.concatenate(held)
                yield rows[:run]
                held, count = [rows[run:]], count - run
                run = min(2 * run, most)
        if count:
            yield np.concatenate(held)
            run = min(2 * run, most)


def first_separating_subset(
    test: CITest,
    x: int,
    y: int,
    candidates: Sequence[int],
    within: Sequence[int],
    subsets: Iterable[Block],
    alpha: float,
    first: int = 1,
) -> tuple[int, ...] | None:
    """The first subset of ``candidates`` in ``subsets`` that separates ``x`` and ``y``
    (``p > alpha``), or ``None``.

    ``subsets`` are blocks of positions in ``candidates`` (:data:`Block`), in the order they are
    tested; a subset whose positions all lie in one of the bit masks ``within`` is left out. There
    are at most :data:`MASK_BITS` candidates. Each size's subsets are enumerated as arrays and
    asked in runs of ``first``, twice that, and so on up to :data:`CHUNK` subsets, each run one
    ``separating_prefix`` call over the sorted members, which answers it up to its first separating
    subset with the p-values of scalar calls, as :func:`~andrey.core.ci._separating_prefix` does. A
    test raises only from a subset that comes before every separating one. The subset returned
    lists its members in ``candidates`` order.
    """
    labels = np.asarray(candidates, dtype=np.intp)
    n = len(labels)
    bits = np.left_shift(np.int64(1), np.arange(n, dtype=np.int64))
    outside = [np.int64(~mask) for mask in within if mask]

    def kept(block: Block) -> np.ndarray:
        rows = block_positions(n, block)
        if outside:
            masks = bits[rows].sum(axis=1)
            keep = np.logical_and.reduce([(masks & mask) != 0 for mask in outside])
            rows = rows[keep]
        return rows

    groups = (map(kept, same) for _, same in groupby(subsets, key=lambda b: len(b[0]) + b[1]))
    for rows in _runs(groups, first, CHUNK):
        pvalues = test.separating_prefix(  # ty: ignore[unresolved-attribute]  # gated by the type
            x, y, np.sort(labels[rows], axis=1), alpha
        )
        if pvalues and pvalues[-1] > alpha:
            return tuple(labels[rows[len(pvalues) - 1]].tolist())
    return None


def separate_apart(
    test: CITest,
    adjacency: np.ndarray,
    apart: np.ndarray,
    alpha: float,
    *,
    max_depth: int | None = None,
    trace: list[dict] | None = None,
    record_pvalues: bool = True,
) -> list[Removal]:
    """Separating sets, over the final skeleton, for the pairs background knowledge makes
    non-adjacent.

    ``adjacency`` is the boolean skeleton the search ends with and ``apart`` the boolean pairs the
    knowledge makes non-adjacent, which the search never tested. Every such pair that shares a
    neighbor, and so forms an unshielded triple, is tested as the search tests a pair it removes:
    against subsets of either endpoint's neighbors by increasing size, up to ``max_depth`` when one
    is given, with :func:`scan_pairs`. Returns ``(x, y, separating, p)``, ``x < y``, for every pair
    some size separates: the union of the separating subsets at the first size that separates it
    and their largest p-value (``None`` when ``record_pvalues`` is false). A pair no tested subset
    separates is left out: the subsets come from the final neighborhoods and stop at
    ``max_depth``, so a separating set elsewhere is not found. With ``trace`` given, every test is
    recorded as :func:`scan_pairs` records it.
    """
    common = adjacency.astype(np.float32) @ adjacency.astype(np.float32) > 0
    pairs = adjacent_pairs(np.triu(apart & common, 1))
    neighbors = [np.flatnonzero(row) for row in adjacency]
    found: list[Removal] = []
    size = 0
    with backend.one_blas_thread() if pins_blas(test) else nullcontext():
        while pairs and (max_depth is None or size <= max_depth):
            removals = scan_pairs(test, pairs, neighbors, size, alpha, trace, record_pvalues)
            found.extend(removals)
            separated = {(x, y) for x, y, _, _ in removals}
            pairs = [
                (x, y)
                for x, y in pairs
                if (x, y) not in separated and max(neighbors[x].size, neighbors[y].size) > size
            ]
            size += 1
    return found


def unseparated_triples(
    adjacency: np.ndarray,
    sepsets: Mapping[tuple[int, int], object],
    constraints: Constraints | None,
    undecided: Collection[tuple[int, int]] = (),
) -> frozenset[tuple[int, int, int]]:
    """The unshielded triples ``(a, b, c)``, ``a < c``, over a non-adjacent pair with no
    separating set.

    ``adjacency`` is the boolean skeleton and ``sepsets`` the separating sets, keyed in either
    order. A pair ``constraints`` make non-adjacent that ``sepsets`` leaves out has no separating
    set the search found, and an ``undecided`` pair, keyed ``(a, c)`` with ``a < c``, which the
    search left out of the skeleton because no test with rows could decide it
    (:func:`~andrey.constraint.pc.adjacency_search`), may be adjacent or not; so whether ``b`` is a
    collider is unknown: no rule reads the triple as a collider or a non-collider, and no rule
    reads its ends as non-adjacent. Without constraints and undecided pairs the search tested
    every pair it left non-adjacent, and there are none.
    """
    if constraints is None and not undecided:
        return frozenset()
    d = len(adjacency)
    apart = np.zeros((d, d), dtype=bool) if constraints is None else ~constraints.support()
    for a, c in undecided:
        apart[a, c] = apart[c, a] = True
    triples = set()
    for b, row in enumerate(adjacency):
        for a, c in combinations(np.flatnonzero(row).tolist(), 2):
            if apart[a, c] and (a, c) not in sepsets and (c, a) not in sepsets:
                triples.add((a, b, c))
    return frozenset(triples)


def flagged_triples(
    adjacency: np.ndarray, flagged: Collection[tuple[int, int]]
) -> frozenset[tuple[int, int, int]]:
    """The unshielded triples ``(a, b, c)``, ``a < c``, of the boolean ``adjacency`` with a side in
    ``flagged``, pairs in either order.

    A flagged side might have been separated by an untested conditioning set
    (:func:`~andrey.core.ci_missing.running_flags`), so the edge and triple may be absent: no rule
    classifies ``b`` as a collider or non-collider. With no flagged pair, as on complete data, there
    are none.
    """
    if not flagged:
        return frozenset()
    triples = set()
    for u, v in flagged:
        for end, middle in ((u, v), (v, u)):
            for c in np.flatnonzero(adjacency[middle]).tolist():
                if c != end and not adjacency[end, c]:
                    triples.add((min(end, c), middle, max(end, c)))
    return frozenset(triples)


def check_collider_rule(collider_rule: str) -> None:
    """Raise ``ValueError`` unless ``collider_rule`` is one of :data:`COLLIDER_RULES`."""
    if collider_rule not in COLLIDER_RULES:
        raise ValueError(
            f"collider_rule must be 'sepsets', 'conservative', or 'majority', got {collider_rule!r}"
        )


def collider_verdict(
    separating: Sequence[Collection[Hashable]],
    b: Hashable,
    rule: str,
    skipped: Sequence[Collection[Hashable]] = (),
) -> str:
    """``"collider"``, ``"non-collider"``, or ``"ambiguous"``: what ``rule`` makes of middle node
    ``b`` from the ``separating`` subsets of its endpoints.

    ``"conservative"`` makes ``b`` a collider when no subset contains it and a non-collider when
    every subset does. ``"majority"`` makes it a collider when fewer than half contain it and a
    non-collider when more than half do. Any other share, an exact half under ``"majority"``
    included, is ambiguous, and so is an empty ``separating``. The ``skipped`` subsets were not
    tested and might have separated: the verdict stands only when it holds with none of them
    separating, with those holding ``b``, and with those without it, the least and the most share
    of subsets holding ``b`` they allow; otherwise the triple is ambiguous.
    """
    verdict = _verdict(separating, b, rule)
    if not skipped or verdict == "ambiguous":
        return verdict
    holding = [subset for subset in skipped if b in subset]
    lacking = [subset for subset in skipped if b not in subset]
    for extra in (holding, lacking):
        if _verdict([*separating, *extra], b, rule) != verdict:
            return "ambiguous"
    return verdict


def _verdict(separating: Sequence[Collection[Hashable]], b: Hashable, rule: str) -> str:
    """:func:`collider_verdict` from the ``separating`` subsets alone."""
    if not separating:
        return "ambiguous"
    share = sum(b in subset for subset in separating) / len(separating)
    if rule == "conservative":
        return "collider" if share == 0 else "non-collider" if share == 1 else "ambiguous"
    return "collider" if share < 0.5 else "non-collider" if share > 0.5 else "ambiguous"


def vote_triples(
    marks: np.ndarray,
    test: Callable[..., float],
    alpha: float,
    sepsets: SeparatingSets,
    rule: str,
    *,
    max_depth: int | None = None,
    skip: Collection[tuple[int, int, int]] = frozenset(),
) -> frozenset[tuple[int, int, int]]:
    """Decide every unshielded triple of ``marks`` by ``rule``, ``"conservative"`` or
    ``"majority"``; rewrite ``sepsets`` to match and return the ambiguous triples.

    ``marks`` is the final skeleton, a pair adjacent where its entry is not ``NULL``. For each
    unshielded triple ``(a, b, c)``, ``a < c``, the pair ``(a, c)`` is tested given every subset of
    ``a``'s neighbors, then every subset of ``c``'s, the empty set included, up to ``max_depth``
    members when one is given: a subset both endpoints offer is tested and counted twice. A subset
    separates the pair when its p-value is at least ``alpha``. :func:`collider_verdict` decides the
    triple from the separating subsets and those not tested for want of rows, and a pair no subset
    separates leaves its triples ambiguous:
    the conservative and majority rules call a triple unambiguous only when at least one subset of
    its ends' neighbors separates them [1]_ (Section 4.2), in PC and, after the
    possible-d-separation step, in FCI (Section 4.4). The separating sets already recorded do not
    vote. A collider removes ``b`` from the pair's separating sets and a non-collider adds it, under
    both key orders, a missing order taking the other's set first; an ambiguous triple leaves them
    as they are. A triple in ``skip``, keyed ``(a, b, c)`` with ``a < c``, is neither voted nor
    returned.

    Each pair is tested once, however many middle nodes it has, in chunks of ``_VOTE_CHUNK``
    subsets, through the test's ``batched_call`` when it has one. The chunks do not depend on the
    middle node, so every middle node counts the p-values one test of the pair gives. The cost
    grows as ``2**degree`` per pair.

    References
    ----------
    .. [1] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    """
    neighbors = [np.flatnonzero(row).tolist() for row in marks != NULL]
    votes: dict[tuple[int, int], tuple[list[tuple[int, ...]], list[tuple[int, ...]]]] = {}
    ambiguous = set()
    for b, adjacent in enumerate(neighbors):
        for a, c in combinations(adjacent, 2):
            if marks[a, c] != NULL or (a, b, c) in skip:
                continue
            if (a, c) not in votes:
                votes[a, c] = _separating_votes(test, a, c, neighbors, alpha, max_depth)
            found, skipped = votes[a, c]
            verdict = collider_verdict(found, b, rule, skipped)
            if verdict == "ambiguous":
                ambiguous.add((a, b, c))
                continue
            for pair in ((a, c), (c, a)):
                recorded = separating_set(sepsets, *pair) or frozenset()
                updated = recorded | {b} if verdict == "non-collider" else recorded - {b}
                sepsets[pair] = tuple(sorted(updated))
    return frozenset(ambiguous)


def _separating_votes(
    test: Callable[..., float],
    a: int,
    c: int,
    neighbors: list[list[int]],
    alpha: float,
    max_depth: int | None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The subsets of ``a``'s neighbors, then of ``c``'s, that separate ``a`` and ``c``
    (``p >= alpha``), each endpoint's by increasing size, as :func:`vote_triples` counts them, and
    those not tested for want of rows (:data:`~andrey.core.ci_missing.NOT_TESTED`)."""
    subsets = (
        subset
        for endpoint in (a, c)
        for size in range(
            len(neighbors[endpoint]) + 1
            if max_depth is None
            else min(len(neighbors[endpoint]), max_depth) + 1
        )
        for subset in combinations(neighbors[endpoint], size)
    )
    batched = getattr(test, "batched_call", None)
    found: list[tuple[int, ...]] = []
    skipped: list[tuple[int, ...]] = []
    while chunk := list(islice(subsets, _VOTE_CHUNK)):
        pvalues = batched(a, c, chunk) if callable(batched) else [test(a, c, s) for s in chunk]
        found.extend(subset for subset, p in zip(chunk, pvalues) if p >= alpha)
        skipped.extend(subset for subset, p in zip(chunk, pvalues) if not_tested(p))
    return found, skipped


def adjacent_pairs(adjacency: np.ndarray) -> list[tuple[int, int]]:
    """The still-adjacent pairs ``(x, y)``, ``x < y``, row-major (ascending ``x`` then ``y``)."""
    xs, ys = np.nonzero(np.triu(adjacency != 0, 1))
    return list(zip(xs.tolist(), ys.tolist()))
