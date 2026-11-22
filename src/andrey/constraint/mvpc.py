"""Missing-value PC (MVPC): the skeleton correction behind ``andrey.pc(missing="mvpc")``.

The algorithm, its settings, and its reference are described on :func:`andrey.pc`. Test-wise
deletion keeps every true edge of the skeleton but can keep extra ones: an extra edge ``x - y``
needs the missingness indicator of a variable in a test to be a common effect of ``x`` and ``y``,
or a descendant of one (Tu et al. 2019, Proposition 2). MVPC finds the parents of each indicator,
marks the edges such an indicator could have kept, and tests those again on virtual data whose
distribution is that of the complete data on the rows where those parents are observed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

import numpy as np

from andrey.core.ci import CITest, FisherZ, StudentT, pins_blas
from andrey.core.ci_missing import NOT_TESTED, RowCounts, count_answer

__all__ = ["IndicatorTest", "PermutationCorrection", "candidate_edges"]

#: The tests whose virtual data PermC defines: the partial-correlation tests.
PERMC_TESTS = {"fisherz": FisherZ, "student_t": StudentT}


class IndicatorTest:
    """A test of a missingness indicator, column ``indicator`` of the data ``test`` was built on.

    A query whose rows hold one indicator value, all observed or all missing, has no information on
    it: it returns :data:`~andrey.core.ci_missing.NOT_TESTED`, counts as not tested
    (:func:`~andrey.core.ci_missing.count_answer`), and never separates the pair. Other queries use
    ``test``, which returns :data:`~andrey.core.ci_missing.NOT_TESTED` and counts queries with too
    few rows.
    """

    def __init__(self, test: CITest, data: np.ndarray, indicator: int) -> None:
        self._test = test
        self._observed = ~np.isnan(data)
        self._values = data[:, indicator]
        self._indicator = indicator
        self.pin_blas = pins_blas(test)

    def constant_on(self, column: int) -> bool:
        """Whether the rows that observe ``column`` hold one value of the indicator: then so do
        the rows of every query of ``column``, and none is tested."""
        values = self._values[self._observed[:, column]]
        return bool(values.size) and values.min() == values.max()

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        given = [int(v) for v in condition_set]
        others = [int(v) for v in (x, y, *given) if int(v) != self._indicator]
        values = self._values[self._observed[:, others].all(axis=1)]
        if values.size and values.min() == values.max():
            count_answer(values.size, NOT_TESTED, int(x), int(y), given)
            return NOT_TESTED
        return self._test(x, y, given)


class PermutationCorrection:
    """The test of MVPC's correction step: PermC on a candidate pair, the deletion test elsewhere.

    A query of ``x`` and ``y`` given ``S`` on a pair in ``candidates`` reads ``W``, the parents of
    the indicators of ``{x, y} | S`` and, in turn, of the indicators of ``W``'s own columns. With
    an empty ``W``, or off the candidates, it is the test-wise deletion test ``deletion``. Else, on
    the rows where ``{x, y} | S`` and ``W`` are observed, each of ``x``, ``y``, and ``S`` is
    regressed on ``W`` with an intercept; as many rows as the block has are drawn without
    replacement from the rows where ``W`` is observed, in random order, and each variable's virtual
    value is its regression's prediction at the drawn ``W`` plus its own residual. The query's
    partial-correlation test runs on those virtual rows (Tu et al. 2019, Algorithm 2).

    The regressions and test need ``max(t, |W| + 2)`` rows in the block, with ``t`` the test's
    ``least_rows``: ``|S| + 4`` for ``FisherZ`` and ``|S| + 3`` for ``StudentT``, leaving a degree
    of freedom. Queries with fewer rows or empty ``W`` use ``deletion`` and add their pair to
    ``uncorrected``: the paper leaves this case open, and the correction can only remove an edge
    deletion kept. ``rows`` counts the rows each PermC query reads, no more than deletion reads for
    that query, and the queries answered ``p = 1``.

    The draw comes from a generator seeded by ``seed`` and by the fingerprints of the queried
    columns, a digest of where each one's values are missing: a p-value then depends on neither the
    order of the queries, nor the number of workers, nor the order of the columns, and, as the
    regressions have an intercept, not on a column's units or origin. The regressions run with BLAS
    pinned to one thread (``pin_blas``), so a p-value has the same bits whatever the host's thread
    count.
    """

    pin_blas = True

    def __init__(
        self,
        data: np.ndarray,
        indep_test: str,
        parents: Mapping[int, list[int]],
        candidates: np.ndarray,
        deletion: CITest,
        seed: int,
    ) -> None:
        self._data = data
        self._observed = ~np.isnan(data)
        self._test = PERMC_TESTS[indep_test]
        self._parents = parents
        self._candidates = candidates
        self._deletion = deletion
        self._seed = seed
        #: The pairs ``(x, y)``, ``x < y``, with a query too short of rows to correct.
        self.uncorrected: set[tuple[int, int]] = set()
        #: The rows the corrected queries read, a row count to the number of queries with it, and
        #: the queries answered ``p = 1``.
        self.rows = RowCounts()
        gaps = np.packbits(~self._observed, axis=0).T
        self._fingerprints = [
            int.from_bytes(hashlib.blake2b(g.tobytes(), digest_size=8).digest(), "big")
            for g in gaps
        ]

    def closure(self, columns: Iterable[int]) -> list[int]:
        """The parents of the indicators of ``columns``, closed under the parents of their own."""
        found: set[int] = set()
        frontier = [int(c) for c in columns]
        while frontier:
            new = {p for c in frontier for p in self._parents.get(c, ())} - found
            found |= new
            frontier = sorted(new)
        return sorted(found)

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        x, y = sorted((int(x), int(y)))
        given = sorted({int(s) for s in condition_set})
        if not self._candidates[x, y]:
            return self._deletion(x, y, given)
        parents = self.closure((x, y, *given))
        if not parents:
            return self._deletion(x, y, given)
        return self._permuted(x, y, given, parents)

    def _permuted(self, x: int, y: int, given: list[int], parents: list[int]) -> float:
        tested = [x, y, *given]
        rows = self._observed[:, tested + parents].all(axis=1)
        m = int(rows.sum())
        least = max(self._test.least_rows(len(given)), len(parents) + 2)
        if m < least:
            self.uncorrected.add((x, y))
            return self._deletion(x, y, given)
        block = self._data[rows]
        design = np.column_stack([np.ones(m), block[:, parents]])
        values = block[:, tested]
        coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
        residuals = values - design @ coefficients
        pool = np.flatnonzero(self._observed[:, parents].all(axis=1))
        drawn = self._generator(x, y, given).permutation(pool)[:m]
        shuffled = np.column_stack([np.ones(m), self._data[drawn][:, parents]])
        virtual = shuffled @ coefficients + residuals
        p = float(self._test(virtual)(0, 1, range(2, len(tested))))
        self.rows.add([m], [p], [x], [y])
        return p

    def _generator(self, x: int, y: int, given: list[int]) -> np.random.Generator:
        pair = sorted((self._fingerprints[x], self._fingerprints[y]))
        rest = sorted(self._fingerprints[s] for s in given)
        key = (*pair, len(rest), *rest)
        return np.random.default_rng(np.random.SeedSequence(self._seed, spawn_key=key))


def candidate_edges(adjacent: np.ndarray, parents: Mapping[int, list[int]]) -> np.ndarray:
    """The adjacent pairs that share an adjacent variable or a missingness indicator (Step 3)."""
    a = adjacent.astype(np.int64)
    d = a.shape[0]
    p = np.zeros((d, d), dtype=np.int64)
    for column, pa in parents.items():
        p[pa, column] = 1  # each parent of the indicator of column
    shared = ((a @ a) > 0) | ((p @ p.T) > 0)
    np.fill_diagonal(shared, False)
    return adjacent & shared
