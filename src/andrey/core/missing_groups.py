"""Pattern-grouped sums for the partial-correlation tests under test-wise deletion.

A partial-correlation query on data with missing values reads the rows observed in every column of
its block ``K``. Summing ``K`` over all ``n`` rows, as the per-row path does, costs ``n k^2`` per
block. Merged data sets, such as cohorts that each measure their own variables, have few
structural missingness patterns with scattered cells missing on top of them. :class:`GroupedSums`
splits the rows into groups ``g``, each with an observed set ``O_g`` holding every member's
observed columns, and keeps per group, once:

- the count, sums, and cross products over all its rows of the values less each column's observed
  mean, zero where unobserved;
- for each column ``t`` with scattered misses in the group (in ``O_g`` but unobserved), the same
  sums over the group's rows missing ``t``;
- the rows missing two or more columns of ``O_g``, with their values.

A block ``K`` inside ``O_g`` takes the group's sums, less the sums over its rows missing ``t`` for
each ``t`` in ``K``, plus ``m - 1`` times the own sums of each row missing ``m >= 2`` columns of
``K``. Each kept row then counts once and every other row zero times, so the sums are those over
the block's rows exactly, in exact arithmetic; a block inside no ``O_g`` has no rows. This is the
per-pattern bookkeeping of EM for normal data [1]_, applied to the rows each test keeps.

:func:`pattern_groups` chooses the groups by cost. The distinct patterns are taken in a fixed
order, the most rows first, then the most observed columns, then by their columns' tie ranks
(:func:`~andrey.core.ties.tie_ranks`), so the groups do not depend on the row or the column order.
Each pattern joins the group where it adds the least work, or opens its own group when every group
would charge it more than :data:`_OPEN_COST`, the work of one more group. The work a row adds is
the pairs of columns it misses in its group, which the block step reads once per block holding
two of them, and one scanned row when it misses two or more. A pattern joining a group that lacks
some of its columns grows the group, and every member then misses those columns. So the columns a
cohort never measures, missed by all its rows at once, stay outside its group, and only the
scattered cells count as missing.

References
----------
.. [1] Schafer, J. L. (1997). Analysis of Incomplete Multivariate Data. Chapman and Hall/CRC,
   chapter 5. https://doi.org/10.1201/9780367803025
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from . import backend, env

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: The default of ``ANDREY_TESTWISE_GROUP_BYTES``: the grouped sums are used only when their tables
#: fit in this many bytes; ``0`` keeps every fit on the per-row path.
GROUP_BYTES = 1 << 28

#: The default of ``ANDREY_TESTWISE_GROUP_MAX_SCATTER``: the grouped sums are used only when the
#: scattered cells, the cells of ``O_g`` a member misses, are at most this share of the cells of
#: ``O_g`` over all rows.
GROUP_MAX_SCATTER = 0.15

#: The default of ``ANDREY_TESTWISE_GROUP_MAX_GROUPS``: the most groups :func:`pattern_groups`
#: forms before it stops, so data with as many patterns as rows, where the grouped sums cannot
#: help, cost little to rule out.
MAX_GROUPS = 64

#: The default of ``ANDREY_TESTWISE_GROUP_MIN_COLUMNS``: the fewest columns a test's data need for
#: the grouped sums. A test over fewer columns, such as one built for a single query, asks few
#: distinct blocks, each summed once, so the groups cannot repay their setup.
MIN_COLUMNS = 8

#: The default of ``ANDREY_TESTWISE_GROUP_MIN_GAIN``: a block takes the grouped sums when its
#: estimated per-row cost is at least this many times its grouped cost
#: (:meth:`GroupedSums.cheaper`).
MIN_GAIN = 2.0

#: A pattern opens its own group when joining any group would add more work than this: the rows
#: missing two or more columns of the group, which the block step scans, plus the pairs of columns
#: each such row misses. A group costs the block step about as much as scanning this many rows.
#: Fixed, not a setting: it is the grouping's unit of cost, which decides how the rows split into
#: groups, while the settings above decide whether a test or a block uses them.
_OPEN_COST = 128

#: A tested block whose rows are fewer than its groups' rows over this, or fewer than twice its
#: columns, is summed per row (:meth:`GroupedSums.exact`). Its grouped sums would subtract nearly
#: all of the groups' sums, so rounding would grow with the groups' rows over the block's rows.
#: Fixed, not a setting: it bounds the rounding, which no speed setting should loosen.
FEW_ROWS = 16

# Each step of the block computation holds at most about this many bytes of temporaries.
_STEP_BYTES = 1 << 23


@dataclass(frozen=True)
class Grouping:
    """The groups :func:`pattern_groups` chose: ``group[i]`` is row ``i``'s group and
    ``observed[g]`` the observed set ``O_g``; ``missed`` marks each row's scattered cells, the
    columns of its group it misses."""

    group: np.ndarray
    observed: np.ndarray
    missed: np.ndarray

    @property
    def scattered_share(self) -> float:
        """The scattered cells over the cells of every row's ``O_g``."""
        cells = int(self.observed[self.group].sum())
        return float(self.missed.sum()) / cells if cells else 0.0

    def some_block_cheaper(self, gain: float) -> bool:
        """Whether the blocks that can take the grouped sums hold enough rows to repay the tables.

        A block reads the masks of the rows missing two or more columns of each of its groups, so
        :meth:`GroupedSums.cheaper` refuses every block in a group with more than ``n / gain``
        such rows. The other groups must hold at least ``n / gain`` rows: otherwise the blocks the
        grouped sums can take keep few rows, as when one group holds nearly every row and nearly
        every row misses two of its columns.
        """
        n = len(self.group)
        twice = np.bincount(self.group[self.missed.sum(axis=1) >= 2], minlength=len(self.observed))
        rows = np.bincount(self.group, minlength=len(self.observed))
        return bool(gain * rows[gain * twice <= n].sum() >= n)

    def table_bytes(self) -> int:
        """The bytes :class:`GroupedSums` allocates for these groups, counted before it does: per
        group, the count, sums, and cross products over all its rows and over its rows missing
        each scattered column, the pair counts of the rows missing two or more columns, and those
        rows' values and masks, and the lookups of each group's columns."""
        d = self.observed.shape[1]
        total = self.observed.nbytes
        for g, columns in enumerate(self.observed):
            o = int(columns.sum())
            members = self.missed[self.group == g][:, columns]
            spread = int(members.any(axis=0).sum())
            twice = int((members.sum(axis=1) >= 2).sum())
            total += 8 * (1 + o + o * o) * (2 + spread) + 8 * o * o + 9 * o * twice
            total += 8 * (d + o)  # each group's lookups: its local index and spread index
        return total


def pattern_groups(
    observed: np.ndarray, ranks: np.ndarray, max_groups: int = MAX_GROUPS
) -> Grouping | None:
    """Group the rows by their missingness patterns, by cost (see the module).

    ``observed`` is the ``(n, d)`` observed mask and ``ranks`` each column's tie rank. Returns
    ``None`` when the patterns need more than ``max_groups`` groups.
    """
    n, d = observed.shape
    packed = np.packbits(observed, axis=1)
    words = np.zeros((n, -(-packed.shape[1] // 8) * 8), dtype=np.uint8)
    words[:, : packed.shape[1]] = packed
    words = words.view(np.uint64)  # each row's pattern as whole words, sorted far faster
    by_word = np.lexsort(words.T[::-1])
    ranked = words[by_word]
    new = np.ones(n, dtype=bool)
    new[1:] = (ranked[1:] != ranked[:-1]).any(axis=1)
    starts = np.flatnonzero(new)
    inverse = np.empty(n, dtype=np.intp)
    inverse[by_word] = np.cumsum(new) - 1
    counts = np.diff(np.append(starts, n))
    patterns = packed[by_word[starts]]
    masks = observed[by_word[starts]]
    sizes = masks.sum(axis=1)
    by_rank = np.packbits(masks[:, np.argsort(ranks, kind="stable")], axis=1)
    keys = [by_rank[:, i] for i in range(by_rank.shape[1] - 1, -1, -1)]
    order = np.lexsort((*keys, -sizes, -counts))
    bits = [int.from_bytes(row.tobytes(), "big") for row in patterns]
    sets: list[int] = []
    rows: list[int] = []
    misses: list[int] = []  # the scattered cells of each group
    lean: list[int] = []  # the members missing at most one column of the group
    single: list[int] = []  # the members missing exactly one column
    assign = np.empty(len(patterns), dtype=np.int64)
    for p in order.tolist():
        pattern, count = bits[p], int(counts[p])
        best, best_cost = -1, _OPEN_COST
        for g, held in enumerate(sets):
            a = (held & ~pattern).bit_count()  # the group's columns the pattern misses
            j = (pattern & ~held).bit_count()  # the columns the group grows by
            cost = count * (a * (a - 1) // 2 + (a >= 2))
            if j:
                cost += j * misses[g] + rows[g] * j * (j - 1) // 2
                cost += lean[g] if j >= 2 else single[g]
            if cost < best_cost or (best < 0 and cost == best_cost):
                best, best_cost = g, cost
        if best < 0:
            if len(sets) == max_groups:
                return None
            sets.append(pattern)
            rows.append(count)
            misses.append(0)
            lean.append(count)
            single.append(0)
            assign[p] = len(sets) - 1
            continue
        held = sets[best]
        a = (held & ~pattern).bit_count()
        j = (pattern & ~held).bit_count()
        if j:  # every member now misses the new columns
            misses[best] += j * rows[best]
            single[best] = lean[best] - single[best] if j == 1 else 0
            lean[best] = single[best]
        sets[best] = held | pattern
        rows[best] += count
        misses[best] += count * a
        lean[best] += count if a <= 1 else 0
        single[best] += count if a == 1 else 0
        assign[p] = best
    group = assign[inverse]
    width = packed.shape[1]
    held = np.unpackbits(
        np.frombuffer(b"".join(s.to_bytes(width, "big") for s in sets), dtype=np.uint8).reshape(
            len(sets), width
        ),
        axis=1,
        count=d,
    ).astype(bool)
    return Grouping(group, held, held[group] & ~observed)


def grouping_for(observed: np.ndarray, ranks: Callable[[], np.ndarray]) -> Grouping | None:
    """The groups the grouped sums use on ``observed``, or ``None`` for the per-row path.

    ``ranks`` returns each column's tie rank; it is called only once the cheap checks pass. The
    grouped sums are used when the data have at least ``ANDREY_TESTWISE_GROUP_MIN_COLUMNS``
    columns, the patterns form at most ``ANDREY_TESTWISE_GROUP_MAX_GROUPS`` groups, their
    scattered share is at most ``ANDREY_TESTWISE_GROUP_MAX_SCATTER``, their tables fit in
    ``ANDREY_TESTWISE_GROUP_BYTES``, and some block can pass the block gate
    (:meth:`Grouping.some_block_cheaper`). The choice depends on the data alone.
    """
    budget = _read(env.TESTWISE_GROUP_BYTES, GROUP_BYTES)
    scatter = _read(env.TESTWISE_GROUP_MAX_SCATTER, GROUP_MAX_SCATTER)
    most = _read(env.TESTWISE_GROUP_MAX_GROUPS, MAX_GROUPS)
    fewest = _read(env.TESTWISE_GROUP_MIN_COLUMNS, MIN_COLUMNS)
    if budget <= 0 or most <= 0 or observed.shape[1] < fewest or observed.all():
        return None
    grouping = pattern_groups(observed, ranks(), most)
    if grouping is None:
        logger.debug("test-wise deletion: per-row sums, more than %d groups", most)
        return None
    share = grouping.scattered_share
    size = grouping.table_bytes()
    gain = _read(env.TESTWISE_GROUP_MIN_GAIN, MIN_GAIN)
    path = share <= scatter and size <= budget and grouping.some_block_cheaper(gain)
    logger.debug(
        "test-wise deletion: %s sums, %d groups, scattered share %.3f, %d table bytes",
        "grouped" if path else "per-row",
        len(grouping.observed),
        share,
        size,
    )
    return grouping if path else None


def _read(setting: env.Setting[T], default: T) -> T:
    """``setting``'s value, or ``default`` when it is unset."""
    value = setting.read()
    return default if value is None else value


def few_rows(count: np.ndarray, bound: np.ndarray, k: int) -> np.ndarray:
    """Which blocks of ``k`` columns, with ``count`` rows out of their groups' ``bound``, the
    per-row sums compute instead of the groups: fewer than ``bound / FEW_ROWS`` rows or fewer
    than ``2 k`` (see :data:`FEW_ROWS`)."""
    return (count * FEW_ROWS < bound) | (count < 2 * k)


class GroupedSums:
    """The sums over each block's rows from the groups of :func:`pattern_groups`.

    ``values`` holds each column's values less its observed mean, ``0`` where unobserved, one row
    per column, as :class:`~andrey.core.ci_missing.GaussianDeletion` keeps them. Every sum a block
    reads runs in a fixed order: the groups in their order, the columns of ``K`` in order, and the
    rows missing two or more columns of ``K`` in row order, one after another. So a block's sums
    have the same bits alone or with any other blocks.
    """

    def __init__(self, values: np.ndarray, grouping: Grouping) -> None:
        d = values.shape[0]
        self.observed = grouping.observed
        self.local: list[np.ndarray] = []
        self.rows: list[int] = []
        self.sums: list[np.ndarray] = []
        self.cross: list[np.ndarray] = []
        self.spread: list[np.ndarray] = []  # each column's index into the tables below, or last
        self.lost_n: list[np.ndarray] = []
        self.lost_s: list[np.ndarray] = []
        self.lost_c: list[np.ndarray] = []
        self.twice_values: list[np.ndarray] = []
        self.pairs: list[np.ndarray] = []  # rows missing both of two columns, per pair
        self.twice_missed: list[np.ndarray] = []
        with np.errstate(all="ignore"), backend.one_blas_thread():
            for g, held in enumerate(grouping.observed):
                columns = np.flatnonzero(held)
                local = np.full(d, -1, dtype=np.intp)
                local[columns] = np.arange(len(columns))
                members = np.flatnonzero(grouping.group == g)
                x = np.ascontiguousarray(values[np.ix_(columns, members)])
                missed = np.ascontiguousarray(grouping.missed[np.ix_(members, columns)].T)
                spread = np.flatnonzero(missed.any(axis=1))
                index = np.full(len(columns), len(spread), dtype=np.intp)
                index[spread] = np.arange(len(spread))
                o = len(columns)
                lost_n = np.zeros(len(spread) + 1, dtype=np.int64)
                lost_s = np.zeros((len(spread) + 1, o))
                lost_c = np.zeros((len(spread) + 1, o, o))
                for i, t in enumerate(spread.tolist()):
                    part = np.ascontiguousarray(x[:, missed[t]])
                    lost_n[i] = part.shape[1]
                    lost_s[i] = part.sum(axis=1)
                    lost_c[i] = part @ part.T
                twice = missed.sum(axis=0) >= 2
                self.local.append(local)
                self.rows.append(len(members))
                self.sums.append(x.sum(axis=1))
                self.cross.append(x @ x.T)
                self.spread.append(index)
                self.lost_n.append(lost_n)
                self.lost_s.append(lost_s)
                self.lost_c.append(lost_c)
                self.twice_values.append(np.ascontiguousarray(x[:, twice]))
                self.twice_missed.append(np.ascontiguousarray(missed[:, twice]).view(np.uint8))
                both = missed[:, twice].astype(np.float64)
                self.pairs.append(np.triu(np.rint(both @ both.T), 1))
        self.n = int(len(grouping.group))
        self.gain = _read(env.TESTWISE_GROUP_MIN_GAIN, MIN_GAIN)

    def cheaper(self, columns: np.ndarray) -> np.ndarray:
        """Which blocks, rows of ``columns``, the groups compute for less than the per-row sums.

        The per-row sums read the observed masks of all ``n`` rows for each block, then sum the
        rows it keeps. In each of its groups, the grouped sums read the masks of the rows missing
        two or more columns of the group, and correct each row missing two or more of the block's
        own columns, at most the rows missing both of each pair of them (the columns are sorted,
        so each pair's count sits above the diagonal). A block takes the grouped sums when those
        read rows plus that bound, summed over its groups, times ``ANDREY_TESTWISE_GROUP_MIN_GAIN``
        are at most ``n``. The choice depends on the block and the data alone.
        """
        k = columns.shape[1]
        work = np.zeros(len(columns))
        inside = np.asarray(self.observed[:, columns].all(axis=2))
        i, j = np.triu_indices(k, 1)
        for g, pairs in enumerate(self.pairs):
            at = np.flatnonzero(inside[g])
            if len(at):
                local = self.local[g][columns[at]]
                work[at] += pairs[local[:, i], local[:, j]].sum(axis=1)
                work[at] += self.twice_missed[g].shape[1]
        return self.gain * work <= self.n

    def counts(self, columns: np.ndarray) -> np.ndarray:
        """The most rows each block, a row of ``columns``, can hold: those of its groups."""
        inside = self.observed[:, columns].all(axis=2)
        return np.asarray(self.rows, dtype=np.int64) @ inside

    def blocks(self, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The count, sums, and cross products over each block's rows.

        ``columns`` is the ``(B, k)`` array of the blocks' sorted columns; returns arrays of shape
        ``(B,)``, ``(B, k)``, and ``(B, k, k)``.
        """
        b, k = columns.shape
        count = np.zeros(b, dtype=np.int64)
        sums = np.zeros((b, k))
        cross = np.zeros((b, k, k))
        inside = np.asarray(self.observed[:, columns].all(axis=2))
        for g in range(len(self.rows)):
            at = np.flatnonzero(inside[g])
            if not len(at):
                continue
            local = self.local[g][columns[at]]
            row, col = local[:, :, None], local[:, None, :]
            n_g = np.full(len(at), self.rows[g], dtype=np.int64)
            s_g = self.sums[g][local]
            c_g = self.cross[g][row, col]
            for i in range(k):
                t = self.spread[g][local[:, i]]
                n_g -= self.lost_n[g][t]
                s_g -= self.lost_s[g][t[:, None], local]
                c_g -= self.lost_c[g][t[:, None, None], row, col]
            self._twice(g, local, n_g, s_g, c_g)
            count[at] += n_g
            sums[at] += s_g
            cross[at] += c_g
        return count, sums, cross

    def _twice(
        self, g: int, local: np.ndarray, n_g: np.ndarray, s_g: np.ndarray, c_g: np.ndarray
    ) -> None:
        """Add back ``m - 1`` times the own sums of each of group ``g``'s rows missing ``m >= 2``
        of each block's columns ``local``, in place, in row order within each block."""
        missed = self.twice_missed[g]
        values = self.twice_values[g]
        width = missed.shape[1]
        if not width:
            return
        b, k = local.shape
        step = max(1, _STEP_BYTES // (2 * width))
        for start in range(0, b, step):
            part = local[start : start + step]
            # A count of up to k columns: one byte holds it while k < 256.
            m = missed[part[:, 0]].astype(np.uint8 if k < 256 else np.int32)
            for i in range(1, k):
                m += missed[part[:, i]]
            block, row = np.nonzero(m >= 2)
            if not len(block):
                continue
            weight = (m[block, row] - 1).astype(np.int64)
            bounds = np.append(np.flatnonzero(np.diff(block, prepend=-1)), len(block))
            limit = max(1, _STEP_BYTES // (24 * k * k))
            first = 0
            while first < len(bounds) - 1:  # whole blocks, at most ``limit`` rows unless one alone
                last = max(
                    first + 1, int(np.searchsorted(bounds, bounds[first] + limit, "right")) - 1
                )
                cut = bounds[first:last] - bounds[first]
                at = slice(bounds[first], bounds[last])
                owner = start + block[bounds[first:last]]
                n_g[owner] += np.add.reduceat(weight[at], cut)
                x = values[part[block[at]], row[at, None]]  # (E, k): each row on its block
                wx = weight[at, None] * x
                s_g[owner] += np.add.reduceat(wx, cut, axis=0)
                c_g[owner] += np.add.reduceat(wx[:, :, None] * x[:, None, :], cut, axis=0)
                first = last
