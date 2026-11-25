"""The answers to a test without enough rows that the benchmark compares, and the data arms.

Andrey skips such a test: it answers ``NaN`` (not tested), which never separates a pair and never
confirms a link. ``patch_fill(value)`` makes the search answer ``value`` to such a test instead:
``1.0`` is the answer p = 1 (the test separates its pair) and ``0.0`` the answer p = 0 (the test
keeps its pair adjacent). The patch wraps, in place and for the rest of the process, the two
places where the not-tested answer is produced, so no route or type check changes:

- ``TestwiseDeletion._query``: the scalar, statistic, and residual routes. A query that was not
  tested returns the very ``untested`` object it was given, which is replaced by the filled value
  of the same shape;
- ``GaussianDeletion._settle``: the canonical arrays, whose ``NaN`` entries are filled in place.

A filled answer is a tested result, so the search treats it as one:

- ``_skeleton.kept_apart`` returns the sides unchanged, so the conditioning sets that hold a column
  never observed with the pair are asked and answered, not skipped;
- ``TestwiseDeletion.conditionable`` returns every column asked, and ``count_not_tested`` counts
  nothing, so the time-series searches (PCMCI and PCMCI+) condition on the parents they would with
  every row observed, not on those that keep the test's block testable;
- ``RowCounts.untested`` lists no pair while the search runs, so the search keeps a pair whose
  filled answer left it adjacent (p = 0). The record's ``undecided`` list still holds the pairs no
  real test decided: ``_not_tested_metadata`` reads the real list.

These are Andrey's internals at the build under test. A change to them makes the arms stop
filling, and ``tests/test_overlapping_variables.py`` fails on that.
"""

from __future__ import annotations

import numpy as np

STATS = {"fills": 0, "nan_out": 0}


def patch_fill(value: float) -> None:
    """Answer ``value`` to every query that was not tested, in this process."""
    from andrey.constraint import _skeleton
    from andrey.core import ci_missing as cm
    from andrey.core.ci import ONE_MINUS_EPS

    stat = 0.0 if value == 1.0 else ONE_MINUS_EPS  # a p = 0 query reads as a perfect dependence

    def filled(untested):
        if isinstance(untested, tuple):
            if isinstance(untested[0], np.ndarray):
                return (np.array([stat]), np.array([value]))
            return (stat, value)
        return value

    query = cm.TestwiseDeletion._query

    def _query(self, ask, x, y, cond, untested):
        out = query(self, ask, x, y, cond, untested)
        if out is untested:
            STATS["fills"] += 1
            out = filled(untested)
        last = out[-1] if isinstance(out, tuple) else out
        if np.isnan(np.asarray(last, dtype=float)).any():
            STATS["nan_out"] += 1
        return out

    settle = cm.GaussianDeletion._settle

    def _settle(self, query_fn, pairs, conds, r, p, rows, flagged):
        settle(self, query_fn, pairs, conds, r, p, rows, flagged)
        hole = np.isnan(p)
        if hole.any():
            STATS["fills"] += int(hole.sum())
            p[hole] = value
            r[hole] = stat

    cm.TestwiseDeletion._query = _query
    cm.GaussianDeletion._settle = _settle
    _skeleton.kept_apart = lambda test, fixed, sides, size: (sides, 0)
    cm.TestwiseDeletion.conditionable = lambda self, x, y, ordered: list(ordered)
    cm.TestwiseDeletion.count_not_tested = lambda self, x, y, queries, held=None: None

    untested = cm.RowCounts.untested
    metadata = cm._not_tested_metadata

    def tested(self):
        return []

    def _not_tested_metadata(counter, skipped, name, drawn):
        cm.RowCounts.untested = untested  # the record lists the pairs no real test decided
        try:
            return metadata(counter, skipped, name, drawn)
        finally:
            cm.RowCounts.untested = tested

    cm.RowCounts.untested = tested  # the search keeps a pair whose answer was filled
    cm._not_tested_metadata = _not_tested_metadata


def apart_pairs(observed: np.ndarray) -> list[tuple[int, int]]:
    """Pairs of columns observed together on no row of the data."""
    o = observed.astype(np.int64)
    co = o.T @ o
    i, j = np.nonzero(np.triu(co == 0, 1))
    return list(zip(i.tolist(), j.tolist(), strict=True))


def drop_cover(observed: np.ndarray) -> list[int]:
    """The columns kept after dropping the fewest columns until every pair shares rows: the
    column in the most pairs never observed together first, then the one with the most missing
    cells, then the lowest index."""
    pairs = set(apart_pairs(observed))
    missing = (~observed).sum(axis=0)
    dropped: set[int] = set()
    while pairs:
        deg: dict[int, int] = {}
        for a, b in pairs:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        col = min(deg, key=lambda c: (-deg[c], -int(missing[c]), c))
        dropped.add(col)
        pairs = {p for p in pairs if col not in p}
    return [c for c in range(observed.shape[1]) if c not in dropped]
