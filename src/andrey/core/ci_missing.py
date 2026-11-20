"""Test-wise deletion: conditional-independence tests on data with missing values.

:func:`~andrey.core.ci.make_indep_test` returns a :class:`TestwiseDeletion` when the data hold
``NaN``; data without ``NaN`` get the registered test itself. A query of ``x`` and ``y`` given
``S`` reads only the rows where ``x``, ``y``, and every member of ``S`` are observed. For
:class:`~andrey.core.ci.FisherZ` and :class:`~andrey.core.ci.StudentT`, a
:class:`GaussianDeletion` computes the query's correlations directly from those rows, many queries
at once, so the p-value is the one the test gives on that block up to rounding. Every other test is
built over those rows of those columns, kept in data order, and asked the same query, so the
p-value is the one that test gives on that block exactly. Each test thus keeps every row its own
variables allow, where deleting every row with a missing value would drop rows for all of them.

Deleting rows per test is valid when the values are missing completely at random (MCAR): the rows
a test keeps are then a random sample of all rows, so each test keeps its level; with perfect
tests FCI then returns the graph it returns on complete data [1]_, and [2]_ argue the same for PC.
When whether a value is missing depends on the variables, the rows a test keeps are a selected
sample, and each test selects its own. The tests can then find dependence where the full population
has none, which keeps extra edges [2]_ and can turn orientations wrong; FCI with one test per query
is then not sound in general [1]_. The procedure that [1]_ proves sound beyond MCAR confirms each
independence on the rows complete in every column, and is not this one. :data:`TESTWISE_CLAIM`
is the sentence every method that deletes rows per test states.

A query whose block holds fewer rows than its test needs is not tested: it answers
:data:`NOT_TESTED`, which never separates a pair and never confirms a link, and is counted apart
(:func:`rows_metadata`). A pair whose columns are never observed together gets no test at all.

References
----------
.. [1] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
   non-random missingness by test-wise deletion. International Journal of Data Science and
   Analytics 6(1), 47-62. https://doi.org/10.1007/s41060-017-0094-6
.. [2] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
   discovery in the presence of missing data. Proceedings of Machine Learning Research 89
   (AISTATS), 1762-1770. https://proceedings.mlr.press/v89/tu19a.html
"""

from __future__ import annotations

import contextlib
import functools
from collections import Counter, OrderedDict
from collections.abc import Callable, Generator, Iterable, Mapping
from contextvars import ContextVar
from functools import partial
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from . import backend
from .ci import (
    BOUND_MARGIN,
    ONE_MINUS_EPS,
    SINGULAR_RTOL,
    CGTest,
    ChiSquare,
    CITest,
    DGTest,
    FisherZ,
    PartialCorrelationTest,
    SingularCorrelationError,
    StudentT,
    _canonical,
    p_value_statistics,
    per_query,
)
from .score import (
    ZERO_RESIDUAL,
    CopiedColumnsError,
    copies_error,
    copies_message,
    copy_groups,
)

__all__ = ["TestwiseDeletion"]

#: The values of ``missing=``, the ways a method can handle missing values: ``"testwise"``
#: deletes rows per test (:class:`TestwiseDeletion`), the default of every method that takes it.
TESTWISE = "testwise"
MISSING_HANDLING = (TESTWISE,)
#: When test-wise deletion is valid, in the one sentence that the docstring of every method that
#: runs it, the ``missing`` concept, and the notice of a default handling share.
TESTWISE_CLAIM = (
    "Test-wise deletion is valid when values are missing completely at random, which means that "
    "whether a value is missing depends on no value in the data; the rows a test reads are then a "
    "random sample of all rows, and otherwise they are a selected sample that more rows do not "
    "correct."
)
#: The p-value of a query that is not tested: its block holds fewer rows than its test needs
#: (:meth:`TestwiseDeletion.least_rows`). It is ``NaN``, so neither ``p > alpha`` nor ``p <= alpha``
#: holds: a search reads it as no evidence, separating no pair and confirming no link
#: (:func:`not_tested`). Its statistic is ``NaN`` too.
NOT_TESTED = float("nan")
#: The sentence on a query without the rows its test needs, which the docstring of every method
#: that deletes rows per test, the ``missing`` concept, and the notice share.
NOT_TESTED_CLAIM = (
    "A test whose variables are observed together on fewer rows than it needs or that has no "
    "degrees of freedom left, a pair never observed together included, is not tested: it never "
    "separates a pair and never confirms a link. A pair that no test with rows could decide, such "
    'as a pair never observed together, is listed in metadata["undecided_pairs"].'
)
#: What the error of a test without the rows it needs advises: a test over fewer columns keeps more
#: rows, and a dropped column is no longer there to condition on.
MORE_ROWS = (
    "cap the conditioning sets with max_depth where the method has it, or drop a column with many "
    "missing values, which can remove a common cause of the columns that stay"
)
#: Structural EM over the expected covariance (:mod:`andrey.search.em`), which the searches whose
#: fit reads only a covariance and the row count take.
EM = "em"
#: The methods that take ``missing="em"``: the searches over the linear-Gaussian BIC and the
#: continuous searches whose loss and screen read only a covariance and the row count.
EM_METHODS = (
    "ges",
    "xges",
    "hc",
    "tabu",
    "exact_search",
    "boss",
    "grasp",
    "notears",
    "dagma",
    "calm",
)
#: The methods in which ``missing="em"`` is supported: in the published comparison
#: (benchmarks/comparisons/missing-data.md) none was less accurate than its reference
#: implementation in a setting, on the first datasets and again on fresh ones. In the other
#: :data:`EM_METHODS` it is experimental: no reference implementation exists for them.
EM_SUPPORTED = ("ges", "hc", "tabu", "boss", "grasp", "notears")
#: Missing-value PC (Tu et al. 2019), which only ``pc`` takes: test-wise deletion, then a corrected
#: test of the edges a missingness indicator could have kept (:mod:`andrey.constraint.mvpc`).
MVPC = "mvpc"

# The most bytes of row blocks the tests one TestwiseDeletion keeps for reuse hold together.
_CACHE_BYTES = 1 << 27

# The most bytes of row values GaussianDeletion reads at once to compute new blocks' matrices.
_STEP_BYTES = 1 << 23
# The bytes of the Python objects that keep one block's matrix in GaussianDeletion's cache.
_ENTRY_BYTES = 512
# How far inside the exact path's thresholds a query stays to keep GaussianDeletion's own answer:
# a query whose block nears a copy, a singular correlation submatrix, or a constant column is
# answered by the exact path, which then decides it.
_MARGIN = 100.0
# The degrees of freedom of a partial-correlation test given |S| conditioners on n rows are
# n - |S| - offset.
_DF_OFFSET: dict[type, int] = {FisherZ: 3, StudentT: 2}

#: The tests run inside :func:`rows_tested`, or ``None`` outside it.
_ROWS: ContextVar[RowCounts | None] = ContextVar("andrey_rows_tested", default=None)


class RowCounts(Counter[int]):
    """The tests of a fit by the rows each read, the tests that answered ``p = 1``, and the
    queries that were not tested.

    Every query a :class:`TestwiseDeletion` answers is counted here, whichever test it runs and
    whichever way it is asked, by :meth:`add`. A tested query counts its row count, and ``p_one``
    when its p-value is exactly ``1``, as a test whose statistic is exactly zero answers. A query
    answered :data:`NOT_TESTED`, which a query with zero degrees of freedom is, counts in
    ``not_tested``, by its pair of the test's columns, ``(x, y)`` with ``x < y``. ``largest`` keeps
    each tested pair's largest p-value, which tells whether a tested query separated it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.p_one = 0
        self.not_tested: Counter[tuple[int, int]] = Counter()
        self.largest: dict[tuple[int, int], float] = {}
        #: The queries not tested whose pairs are not columns of the data, as a search over
        #: another matrix asks them.
        self.not_tested_elsewhere = 0

    def absorb(self, other: RowCounts) -> None:
        """Add the counts of ``other``, whose tests ran over another matrix: its row counts, its
        ``p = 1`` answers, and its queries not tested, without their pairs."""
        self.update(other)
        self.p_one += other.p_one
        self.not_tested_elsewhere += other.total_not_tested()

    def total_not_tested(self) -> int:
        """The number of queries not tested, over the data's pairs or elsewhere."""
        return sum(self.not_tested.values()) + self.not_tested_elsewhere

    def add(
        self, rows: npt.ArrayLike, p: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike
    ) -> None:
        """Count the queries of the pairs ``x < y`` answered with the row counts ``rows`` and the
        p-values ``p``, a :data:`NOT_TESTED` one apart."""
        values = np.asarray(p, dtype=np.float64).reshape(-1)
        if not values.size:
            return
        lo = np.asarray(x, dtype=np.int64).reshape(-1)
        hi = np.asarray(y, dtype=np.int64).reshape(-1)
        skipped = not_tested(values)
        tested = ~skipped
        self.update(np.asarray(rows).reshape(-1)[tested].tolist())
        self.p_one += int(np.count_nonzero(values == 1.0))
        if skipped.any():
            self.not_tested.update(zip(lo[skipped].tolist(), hi[skipped].tolist()))
        if tested.any():
            pairs, inverse = np.unique(
                np.stack([lo[tested], hi[tested]], axis=1), axis=0, return_inverse=True
            )
            top = np.full(len(pairs), -np.inf)
            np.maximum.at(top, inverse.reshape(-1), values[tested])
            for (a, b), value in zip(pairs.tolist(), top.tolist()):
                pair = (a, b)
                self.largest[pair] = max(self.largest.get(pair, -np.inf), value)

    def untested(self) -> list[tuple[int, int]]:
        """The pairs with a query that was not tested and no query that was: no test with rows
        could decide them, as a pair never observed together."""
        return [pair for pair in self.not_tested if pair not in self.largest]


def not_tested(p: npt.ArrayLike) -> np.ndarray:
    """Mark the p-values that are :data:`NOT_TESTED`: the queries whose block held fewer rows than
    their test needs."""
    return np.isnan(np.asarray(p, dtype=np.float64))


@contextlib.contextmanager
def rows_tested() -> Generator[RowCounts]:
    """Count the rows each test-wise deletion test reads inside the block.

    The counter maps a row count to the number of tests that read that many rows; it stays empty
    when no test met a missing value. Its ``p_one`` counts the tests of the block answered
    ``p = 1``, and its ``not_tested`` the queries without the rows their test needs, by pair.
    :func:`rows_metadata` summarizes them.
    """
    counter = RowCounts()
    token = _ROWS.set(counter)
    try:
        yield counter
    finally:
        _ROWS.reset(token)


def count_answer(rows: int, p: float, x: int, y: int) -> None:
    """Count, in the running :func:`rows_tested` counter if any, a query of ``x`` and ``y``
    answered ``p`` on ``rows`` rows by a test that is no :class:`TestwiseDeletion`, as MVPC's
    indicator search answers :data:`NOT_TESTED` one whose indicator is constant on its rows."""
    if (counter := _ROWS.get()) is not None:
        counter.add([rows], [p], [min(x, y)], [max(x, y)])


@contextlib.contextmanager
def rows_tested_elsewhere() -> Generator[RowCounts]:
    """:func:`rows_tested` for tests over another matrix than the data's, as MVPC's indicator
    search runs: on exit, the running counter outside, if any, absorbs the counts
    (:meth:`RowCounts.absorb`), without the pairs, which are not the data's."""
    outer = _ROWS.get()
    with rows_tested() as counter:
        yield counter
    if outer is not None:
        outer.absorb(counter)


def rows_metadata(
    counter: Mapping[int, int], name: Callable[[int], Any] | None = None
) -> dict[str, Any]:
    """Return the metadata of the tests ``counter`` holds; ``{}`` when it holds none.

    ``"rows_tested"`` is ``{"min": ..., "median": ...}``: ``min`` is the fewest rows a test read and
    ``median`` the median over the tests, the mean of the two middle counts when the number of
    tests is even; ``{"min": 0, "median": 0}`` when no query was tested. ``"p_one_tests"``,
    present when it is not zero, is the number of tests answered ``p = 1`` that a
    :class:`RowCounts` holds, whichever test it is, as a test whose statistic is exactly zero
    answers; a query with zero degrees of freedom is not tested, not counted there.
    ``"not_tested"``, present when it is not zero,
    is the number of queries not tested (:data:`NOT_TESTED`). With ``name``, ``"undecided_pairs"``
    lists the pairs no tested query could decide, those of :meth:`RowCounts.untested`, as
    ``[a, b]`` lists of ``name(column)`` by :func:`named_pairs`.
    """
    skipped = counter.total_not_tested() if isinstance(counter, RowCounts) else 0
    if not counter and not skipped:
        return {}
    if not counter:
        metadata: dict[str, Any] = {"rows_tested": {"min": 0, "median": 0}}
        return metadata | _not_tested_metadata(counter, skipped, name)
    counts = sorted(counter.items())
    total = sum(n for _, n in counts)
    middle = [(total - 1) // 2, total // 2]
    values, seen = [], 0
    for rows, n in counts:
        while middle and middle[0] < seen + n:
            values.append(rows)
            middle.pop(0)
        seen += n
    metadata = {"rows_tested": {"min": counts[0][0], "median": (values[0] + values[1]) / 2}}
    p_one = getattr(counter, "p_one", 0)
    metadata |= {"p_one_tests": p_one} if p_one else {}
    return metadata | _not_tested_metadata(counter, skipped, name)


def _not_tested_metadata(
    counter: Mapping[int, int], skipped: int, name: Callable[[int], Any] | None
) -> dict[str, Any]:
    """``"not_tested"`` and ``"undecided_pairs"`` of :func:`rows_metadata`."""
    if not skipped:
        return {}
    metadata: dict[str, Any] = {"not_tested": skipped}
    if name is not None and isinstance(counter, RowCounts):
        metadata["undecided_pairs"] = named_pairs(counter.untested(), name)
    return metadata


def named_pairs(pairs: Iterable[tuple[int, int]], name: Callable[[int], Any]) -> list[list]:
    """``pairs`` of columns as ``[name(a), name(b)]`` lists; each pair and the list in the order of
    the names' text, so the list does not depend on the column order when ``name`` gives
    labels."""
    named = [sorted((name(a), name(b)), key=str) for a, b in pairs]
    return sorted(named, key=lambda pair: (str(pair[0]), str(pair[1])))


class TestwiseDeletion:
    """A conditional-independence test that deletes rows per test on data with missing values.

    Built by :func:`~andrey.core.ci.make_indep_test` over data holding ``NaN``. A query of ``x`` and
    ``y`` given ``S`` keeps the rows observed in every one of the columns ``{x, y} | S``, builds the
    registered test over that block, its columns in data order, and returns that test's p-value
    for the query. Queries are canonicalized as :class:`~andrey.core.ci.FisherZ`'s are. The test of
    a block answers every later query over the same columns; the blocks the reused tests hold stay
    within ``_CACHE_BYTES``, the least recently used dropped first. A nonlinear or a registered
    test, whose state grows with its queries, is built anew for each query. A partial-correlation
    test gets the :class:`GaussianDeletion` subclass, which builds no test per block.

    A query whose block has fewer rows than its test needs (:meth:`least_rows`) is not tested: its
    rows are counted from the observed masks, no test is built, and it answers
    :data:`NOT_TESTED`, counted in the running :func:`rows_tested` counter with its pair. So is a
    query with zero degrees of freedom: a test with ``zero_dof_pvalue`` (the chi-square, G-square,
    degenerate and conditional Gaussian tests) answers :data:`NOT_TESTED` there. A query
    raises ``ValueError`` when a continuous column is constant on the block, as a constant column
    of the data is rejected before a fit, and when the test fails on the block or gives a p-value
    that is not a number, as a mixed-data test can on a block with fewer rows than columns. A
    categorical column with one category on the block is left to the test.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Values with ``NaN`` marking the missing ones; no infinities.
    build : callable
        ``build(block, columns)`` builds the registered test over ``block``, the kept rows of
        ``columns``.
    name : str
        The ``indep_test`` name, for the messages.
    test_class : type or None
        The class ``build`` returns, ``None`` for a registered test. It decides which of the
        partial-correlation methods the wrapper offers, what it reuses, and its BLAS pinning.
    continuous : ndarray of bool, shape (n_features,), or None
        Marks the continuous columns, which a block may not hold constant; ``None`` marks none.
    check_copies : bool
        Raise :class:`~andrey.core.score.CopiedColumnsError` when a partial-correlation block holds
        columns that repeat others up to scale and shift, as
        :func:`~andrey.core.ci.make_indep_test` does on complete data.
    least : callable or None
        ``least(size)``, the fewest rows the test answers a query given ``size`` conditioning
        variables on, with degrees of freedom left; ``None`` for ``2``, the fewest any test reads.
    """

    def __init__(
        self,
        data: np.ndarray,
        build: Callable[[np.ndarray, list[int]], CITest],
        *,
        name: str,
        test_class: type | None,
        continuous: np.ndarray | None,
        check_copies: bool,
        least: Callable[[int], int] | None = None,
    ) -> None:
        self._data = data
        self._least = least
        self._observed = ~np.isnan(data)
        self._build = build
        self._name = name
        self._partial = test_class is not None and issubclass(test_class, PartialCorrelationTest)
        self._continuous = np.zeros(data.shape[1], dtype=bool) if continuous is None else continuous
        self._most = np.full(data.shape[1], -1, dtype=np.int64)
        self._check_copies = check_copies and self._partial
        self._reuse = test_class in (ChiSquare, DGTest, CGTest) or self._partial
        self._cache: OrderedDict[tuple[int, ...], tuple[CITest, int, int]] = OrderedDict()
        self._held = 0
        self._source: np.ndarray | None = None
        self._environments = np.zeros(data.shape[1], dtype=bool)
        self.pin_blas = _pins_blas(test_class, data.shape[0])

    def mark_environments(self, columns: Iterable[int]) -> None:
        """Read ``columns`` as environment indicators, as CD-NOD and FCI-JCI add: a query whose
        block holds one of them constant is not tested, since the block's rows come from one
        environment and no test there can tell environments apart."""
        self._environments[list(columns)] = True
        self._cache.clear()
        self._held = 0

    @property
    def environments(self) -> np.ndarray:
        """The columns :meth:`mark_environments` marked."""
        return np.flatnonzero(self._environments)

    def from_columns(self, source: npt.ArrayLike) -> None:
        """Record the pairs of the queries in the counter as columns ``source[x]`` and
        ``source[y]``: the test runs over the columns of another matrix in another order, as a
        search in rank order runs, and ``source`` gives each one's position in that matrix."""
        self._source = np.asarray(source, dtype=np.int64)

    def _count(
        self, counter: RowCounts, rows: npt.ArrayLike, p: npt.ArrayLike, pairs: np.ndarray
    ) -> None:
        """Add the queries of ``pairs``, an ``(N, 2)`` array of this test's columns, to
        ``counter`` as pairs of the source columns (:meth:`from_columns`)."""
        if self._source is not None:
            pairs = np.sort(self._source[pairs], axis=1)
        counter.add(rows, p, pairs[:, 0], pairs[:, 1])

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        The p-value is the registered test's on the rows observed in all of the query's columns,
        or :data:`NOT_TESTED` when they are too few. A larger p-value means more independence.
        """
        x_int, y_int, cond = _canonical(x, y, condition_set)
        return float(
            self._query(lambda test, a, b, s: test(a, b, s), x_int, y_int, cond, NOT_TESTED)
        )

    def least_rows(self, size: int) -> int:
        """The fewest rows a query given ``size`` conditioning variables is tested on: the test's
        own ``least_rows(size, **options)`` where it has one, as a partial-correlation test has,
        ``size + 4`` for a Fisher z (``"fisherz"``, ``"normal_scores"``) and ``size + 3`` for a
        Student t, the rows that leave one degree of freedom, and a CMIknn with an int ``k_cmi``,
        ``k_cmi + 1``; ``2`` for every other test, the fewest rows any test reads."""
        return 2 if self._least is None else self._least(size)

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Statistics and p-values of many queries, each from the test over its own rows.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` is the block test's ``statistic_pvalues`` of query ``i``. A
        block test without that method reports only p-values: its statistic is
        :func:`~andrey.core.ci.p_value_statistic` of the p-value.
        """

        def ask(test: Any, a: int, b: int, s: list[int]) -> tuple[float, float]:
            cond = np.array([s], dtype=np.intp).reshape(1, len(s))
            batch = getattr(test, "statistic_pvalues", None) or partial(p_value_statistics, test)
            statistics, p_values = batch(np.array([a]), np.array([b]), cond)
            return float(statistics[0]), float(p_values[0])

        return per_query(
            lambda a, b, s: self._query(ask, *_canonical(a, b, s), (NOT_TESTED, NOT_TESTED)),
            x,
            y,
            conditions,
        )

    def _query(
        self, ask: Callable[..., Any], x: int, y: int, cond: list[int], untested: Any
    ) -> Any:
        """Return ``ask(test, a, b, s)`` for the canonical query ``x``, ``y``, ``cond``, or
        ``untested`` when its block has fewer rows than :meth:`least_rows`.

        ``test`` is the test over the query's block, and ``a``, ``b``, and ``s`` the query's
        columns as indices of that block; the answer's last entry is the p-value. The query's row
        count, and whether its p-value is ``1``, or that it was not tested, enter the running
        :func:`rows_tested` counter once it is answered, the same for every test. A copies error
        names the data's columns, and another error of the test on its block, or a p-value that is
        not a number, raises ``ValueError`` saying how many rows the block holds.
        """
        test, columns, rows = self._block_test(x, y, cond)
        counter = _ROWS.get()
        if test is None:
            if counter is not None:
                self._count(counter, [rows], [NOT_TESTED], np.array([[x, y]]))
            return untested
        at = {c: i for i, c in enumerate(columns)}
        try:
            out = ask(test, at[x], at[y], [at[c] for c in cond])
        except SingularCorrelationError as error:
            raise _in_data_columns(error, columns) from error
        except (ValueError, ArithmeticError) as error:
            raise self._failed(rows, f": {error}") from error
        p = np.asarray(out[-1] if isinstance(out, tuple) else out, dtype=np.float64)
        if not_tested(p).all() and hasattr(test, "zero_dof_pvalue"):  # zero degrees of freedom
            if counter is not None:
                self._count(counter, [rows], [NOT_TESTED], np.array([[x, y]]))
            return untested
        if not np.isfinite(p).all():
            raise self._failed(rows, " gives no p-value")
        if counter is not None:
            self._count(counter, [rows], p.reshape(-1)[-1:], np.array([[x, y]]))
        return out

    def _failed(self, rows: int, what: str) -> ValueError:
        """The error of a test that fails on a block of ``rows`` rows, ``what`` saying how."""
        return ValueError(
            f"indep_test={self._name!r} on the {rows} rows where all the columns of one of its "
            f"tests are observed{what}; {MORE_ROWS}"
        )

    def _block_test(self, x: int, y: int, cond: list[int]) -> tuple[CITest | None, list[int], int]:
        """The test over the block of the canonical query, the block's columns, and its rows; no
        test (``None``) when the rows, counted before the block is gathered, are too few."""
        columns = tuple(sorted((x, y, *cond)))
        entry = self._cache.get(columns)
        if entry is not None:
            self._cache.move_to_end(columns)
            return entry[0], list(columns), entry[1]
        rows = self._rows(columns)
        if len(rows) < self.least_rows(len(cond)):
            return None, list(columns), len(rows)
        block = self._data[rows[:, None], np.asarray(columns, dtype=np.intp)]
        held = self._environments[list(columns)]
        if held.any() and (np.ptp(block[:, held], axis=0) == 0).any():
            return None, list(columns), len(rows)  # one environment only: not tested
        self._check_block(block, columns)
        test = self._new_test(block, list(columns))
        if self._reuse and block.nbytes <= _CACHE_BYTES:
            self._cache[columns] = (test, block.shape[0], block.nbytes)
            self._held += block.nbytes
            while self._held > _CACHE_BYTES:
                self._held -= self._cache.popitem(last=False)[1][2]
        return test, list(columns), block.shape[0]

    def _rows(self, columns: Iterable[int]) -> np.ndarray:
        """The rows of ``data`` observed in every one of ``columns``."""
        index = np.asarray(list(columns), dtype=np.intp)
        return np.flatnonzero(self._observed[:, index].all(axis=1))

    def _block(self, columns: Iterable[int]) -> np.ndarray:
        """The rows of ``data`` observed in every one of ``columns``, those columns only."""
        index = np.asarray(list(columns), dtype=np.intp)
        return self._data[self._rows(index)[:, None], index]

    def _check_block(self, block: np.ndarray, columns: tuple[int, ...]) -> None:
        """Raise ``ValueError`` when ``block`` has a constant continuous column."""
        # A column can be constant on the block only when one of its values fills every row.
        index = list(columns)
        if not (block.shape[0] <= self._repeats(index)).any():
            return
        flat = np.ptp(block, axis=0) == 0
        if (flat & self._continuous[index]).any():
            raise ValueError(
                f"indep_test={self._name!r} meets a column that is constant on the "
                f"{block.shape[0]} rows where all the columns of a test are observed, so the test "
                f"measures no dependence; {MORE_ROWS}"
            )

    def _repeats(self, columns: list[int]) -> np.ndarray:
        """The most observed rows that share one value, in each of ``columns``.

        Counted when a column is first read; a column that is not continuous counts ``0``, as
        nothing checks it.
        """
        for j in columns:
            if self._most[j] < 0 and not self._continuous[j]:
                self._most[j] = 0
            elif self._most[j] < 0:
                values = self._data[self._observed[:, j], j]
                self._most[j] = int(np.unique(values, return_counts=True)[1].max())
        return self._most[columns]

    def _new_test(self, block: np.ndarray, columns: list[int]) -> CITest:
        """Build the registered test over ``block``; a copies error names the data's columns, and
        another error of the build raises as :meth:`_query` says."""
        try:
            test = self._build(block, columns)
            if hasattr(test, "zero_dof_pvalue"):
                test.zero_dof_pvalue = NOT_TESTED  # ty: ignore[invalid-assignment]
            if self._check_copies:
                groups = test.copies()  # ty: ignore[unresolved-attribute]  # gated by _partial
                if groups:
                    raise copies_error(groups)
        except (CopiedColumnsError, SingularCorrelationError) as error:
            raise _in_data_columns(error, columns) from error
        except (ValueError, ArithmeticError) as error:
            raise self._failed(block.shape[0], f": {error}") from error
        return test


class PartialCorrelationDeletion(TestwiseDeletion):
    """:class:`TestwiseDeletion` of a partial-correlation test, with its statistics.

    The searches that rank candidates by the partial correlation, and the time-series searches,
    read :meth:`statistic_pvalue`, :meth:`statistic_pvalues`, :meth:`residual_statistic_pvalues`,
    and :meth:`copies`; each reads, for every query, the rows observed in its columns.
    """

    def statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int] = ()
    ) -> tuple[float, float]:
        """Return the partial correlation and p-value of ``x`` and ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        r, p = self._query(
            lambda test, a, b, s: test.statistic_pvalue(a, b, s),
            x_int,
            y_int,
            cond,
            (NOT_TESTED, NOT_TESTED),
        )
        return float(r), float(p)

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partial correlations and p-values of many queries, each on its own rows.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets; entry ``i`` equals ``statistic_pvalue(x[i], y[i], conditions[i])``.
        """
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.asarray(conditions, dtype=np.intp).reshape(len(x), -1)
        r = np.empty(len(x), dtype=np.float64)
        p = np.empty(len(x), dtype=np.float64)
        for k in range(len(x)):
            r[k], p[k] = self.statistic_pvalue(int(x[k]), int(y[k]), conds[k].tolist())
        return r, p

    def residual_statistic_pvalues(
        self, data: np.ndarray, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`~andrey.core.ci.PartialCorrelationTest.residual_statistic_pvalues` per query.

        ``data``, the matrix the test was built over, is not read again: each query's residuals
        come from the test's own rows observed in the query's columns, gathered only when they are
        enough to test.
        """
        del data
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.asarray(conditions, dtype=np.intp).reshape(len(x), -1)
        r = np.empty(len(x), dtype=np.float64)
        p = np.empty(len(x), dtype=np.float64)
        for k in range(len(x)):
            x_int, y_int, cond = _canonical(int(x[k]), int(y[k]), conds[k].tolist())
            columns = sorted((x_int, y_int, *cond))

            def residual(test: Any, a: int, b: int, s: list[int]) -> tuple[np.ndarray, ...]:
                return test.residual_statistic_pvalues(
                    self._block(columns), np.array([a]), np.array([b]), np.array([s], dtype=np.intp)
                )

            untested = (np.array([NOT_TESTED]), np.array([NOT_TESTED]))
            rk, pk = self._query(residual, x_int, y_int, cond, untested)
            r[k], p[k] = rk[0], pk[0]
        return r, p

    def copies(self) -> tuple[tuple[int, ...], ...]:
        """The groups of columns that repeat one another up to scale and shift.

        Each pair of columns is compared on the rows where both are observed, by
        :func:`~andrey.core.score.copy_groups`; a pair observed together on at most two rows, or
        constant on them, is no copy.
        """
        if not hasattr(self, "_copies"):
            d = self._data.shape[1]
            corr = np.eye(d)
            for i in range(d):
                for j in range(i + 1, d):
                    rows = self._observed[:, i] & self._observed[:, j]
                    if np.count_nonzero(rows) > 2:
                        pair = self._data[rows][:, [i, j]]
                        if (np.ptp(pair, axis=0) > 0).all():
                            corr[i, j] = corr[j, i] = np.corrcoef(pair, rowvar=False)[0, 1]
            self._copies = copy_groups(corr, self._data.shape[0])
        return self._copies


class GaussianDeletion(PartialCorrelationDeletion):
    """:class:`PartialCorrelationDeletion` of :class:`~andrey.core.ci.FisherZ` or
    :class:`~andrey.core.ci.StudentT` that answers many queries at once.

    A query's correlation submatrix is the correlation of the rows observed in all of its columns,
    computed directly, with no test built for the block: from sums over those rows of each column's
    values less its observed mean, for every pair of columns once (:meth:`_pairwise`) and for every
    larger block once while it is kept (:meth:`_submatrices`). Its partial correlation, clamp,
    degrees of freedom, and p-value are those of the block's test, so the p-value equals that
    test's up to rounding, as the sums run in another order. A query's answer does not depend on
    the queries evaluated with it, so every query path (a single call, :meth:`batched_call`,
    :meth:`statistic_pvalues`, and the canonical query arrays the skeleton searches pass) gives it
    the same bits.

    A query whose block has fewer rows than the test needs is not tested: its rows are counted from
    the observed masks and its block is never summed. A query with a continuous column that may be
    constant on the block, columns within ``_MARGIN`` of a copy, a correlation submatrix within
    ``_MARGIN`` of the singular bound, or a partial correlation or p-value that is not a number is
    answered by :class:`PartialCorrelationDeletion` instead, which raises where it raises. So every
    error is the exact path's, at the same query.
    """

    def __init__(
        self,
        data: np.ndarray,
        build: Callable[[np.ndarray, list[int]], CITest],
        *,
        name: str,
        test_class: type | None,
        continuous: np.ndarray | None,
        check_copies: bool,
        least: Callable[[int], int] | None = None,
    ) -> None:
        super().__init__(
            data,
            build,
            name=name,
            test_class=test_class,
            continuous=continuous,
            check_copies=check_copies,
            least=least,
        )
        self._kind = cast("type[PartialCorrelationTest]", test_class)  # FisherZ or StudentT
        self._offset = _DF_OFFSET[self._kind]
        # Each column less its observed mean, 0 where unobserved, with the observed rows: one row
        # of n values per column, so a block's columns are contiguous.
        values = np.where(self._observed, data, 0.0)
        values -= values.sum(axis=0) / np.maximum(self._observed.sum(axis=0), 1)
        self._values = np.ascontiguousarray(np.where(self._observed, values, 0.0).T)
        self._seen = np.ascontiguousarray(self._observed.T)
        self._pairs: tuple[np.ndarray, np.ndarray] | None = None
        self._blocks: OrderedDict[tuple[int, ...], tuple[np.ndarray, int]] = OrderedDict()
        self._kept = 0
        self._repeats(list(range(data.shape[1])))

    def __call__(self, x: int, y: int, condition_set: Iterable[int] = ()) -> float:
        """Return the p-value for ``x`` independent of ``y`` given ``condition_set``.

        The p-value is the registered test's on the rows observed in all of the query's columns, up
        to rounding, or :data:`NOT_TESTED` when they are too few. A larger p-value means more
        independence.
        """
        return self.statistic_pvalue(x, y, condition_set)[1]

    def statistic_pvalue(
        self, x: int, y: int, condition_set: Iterable[int] = ()
    ) -> tuple[float, float]:
        """Return the partial correlation and p-value of ``x`` and ``y`` given ``condition_set``."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        r, p = self._answer(
            np.array([x_int]), np.array([y_int]), np.array([cond], dtype=np.intp).reshape(1, -1)
        )
        return float(r[0]), float(p[0])

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Partial correlations and p-values of many queries sharing one conditioning-set size.

        ``x`` and ``y`` are ``(N,)`` int arrays and ``conditions`` the ``(N, m)`` int array of
        conditioning sets, each distinct and excluding ``x[i]`` and ``y[i]``; entry ``i`` equals
        ``statistic_pvalue(x[i], y[i], conditions[i])``.
        """
        x = np.asarray(x, dtype=np.intp)
        y = np.asarray(y, dtype=np.intp)
        conds = np.sort(np.asarray(conditions, dtype=np.intp).reshape(len(x), -1), axis=1)
        return self._answer(np.minimum(x, y), np.maximum(x, y), conds)

    def _canonical_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """P-values of many canonical queries sharing one conditioning-set size.

        ``x[i] < y[i]``, and row ``i`` of ``conditions`` is sorted and distinct and excludes both,
        as the skeleton searches build their queries; entry ``i`` equals
        ``self(x[i], y[i], conditions[i])``.
        """
        return self._answer(
            np.asarray(x, dtype=np.intp),
            np.asarray(y, dtype=np.intp),
            np.asarray(conditions, dtype=np.intp).reshape(len(x), -1),
        )[1]

    def batched_call(self, x: int, y: int, condition_sets: Iterable[Iterable[int]]) -> list[float]:
        """Return the p-values of one pair over many conditioning sets, in their order.

        Entry ``i`` equals ``self(x, y, condition_sets[i])``; the sets may differ in size.
        """
        queries = [_canonical(x, y, cond) for cond in condition_sets]
        sizes: dict[int, list[int]] = {}
        for i, (_, _, cond) in enumerate(queries):
            sizes.setdefault(len(cond), []).append(i)
        r = np.empty(len(queries), dtype=np.float64)
        p = np.empty(len(queries), dtype=np.float64)
        rows = np.empty(len(queries), dtype=np.int64)
        flagged = np.empty(len(queries), dtype=bool)
        for size, at in sizes.items():
            arrays = _arrays([queries[i] for i in at], size)
            r[at], p[at], rows[at], flagged[at] = self._fast(*arrays)
        pairs = np.array([query[:2] for query in queries], dtype=np.int64).reshape(-1, 2)
        self._settle(queries.__getitem__, pairs, r, p, rows, flagged)
        return p.tolist()

    def _answer(
        self, lo: np.ndarray, hi: np.ndarray, conds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """The partial correlations and p-values of canonical queries of one size, in order."""
        r, p, rows, flagged = self._fast(lo, hi, conds)

        def query(k: int) -> tuple[int, int, list[int]]:
            return int(lo[k]), int(hi[k]), conds[k].tolist()

        self._settle(query, np.stack([lo, hi], axis=1), r, p, rows, flagged)
        return r, p

    def _settle(
        self,
        query: Callable[[int], tuple[int, int, list[int]]],
        pairs: np.ndarray,
        r: np.ndarray,
        p: np.ndarray,
        rows: np.ndarray,
        flagged: np.ndarray,
    ) -> None:
        """Answer the ``flagged`` queries by the exact path, in order, and count every query.

        ``query(k)`` is query ``k`` as ``(x, y, conditioning set)`` and ``pairs[k]`` its ``(x,
        y)``. ``r`` and ``p`` hold the fast answers, :data:`NOT_TESTED` for a query without the
        rows it needs, and take the exact ones in place; ``rows`` holds each query's row count. The
        queries are counted in order, so a query that raises does so after the queries before it,
        as the exact path's loop does.
        """
        counter = _ROWS.get()
        start = 0
        for k in np.flatnonzero(flagged).tolist():
            if counter is not None:
                self._count(counter, rows[start:k], p[start:k], pairs[start:k])
            x, y, cond = query(k)
            r[k], p[k] = self._query(
                lambda test, a, b, s: test.statistic_pvalue(a, b, s),
                x,
                y,
                cond,
                (NOT_TESTED, NOT_TESTED),
            )
            start = k + 1
        if counter is not None:
            self._count(counter, rows[start:], p[start:], pairs[start:])

    def _fast(
        self, lo: np.ndarray, hi: np.ndarray, conds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(r, p, rows, flagged)`` of canonical queries of one size, without the exact path.

        ``rows`` is each query's row count and ``flagged`` marks the queries the exact path
        answers instead (see the class); their ``r`` and ``p`` are meaningless. A query with fewer
        rows than :meth:`least_rows` is not flagged and answers :data:`NOT_TESTED`.
        """
        size = conds.shape[1]
        columns = np.empty((len(lo), size + 2), dtype=np.intp)
        columns[:, 0], columns[:, 1], columns[:, 2:] = lo, hi, conds
        with np.errstate(all="ignore"):
            if size == 0:
                count, corr = self._pairwise()
                rows = count[lo, hi]
                stack = np.empty((len(lo), 2, 2), dtype=np.float64)
                stack[:, 0, 0] = stack[:, 1, 1] = 1.0
                stack[:, 0, 1] = stack[:, 1, 0] = corr[lo, hi]
            else:
                stack, rows = self._submatrices(columns)
            short = rows < self.least_rows(size)
            flagged = np.asarray((rows[:, None] <= self._most[columns]).any(axis=1))
            flagged |= ~np.isfinite(stack).all(axis=(1, 2))
            if self._check_copies:
                i, j = _apart(size + 2)
                flagged |= (1.0 - stack[:, i, j] ** 2 <= _MARGIN * ZERO_RESIDUAL).any(axis=1)
            if size == 0:
                r = stack[:, 0, 1]
            elif size == 1:
                rxz, ryz = stack[:, 0, 2], stack[:, 1, 2]
                r = (stack[:, 0, 1] - rxz * ryz) / np.sqrt((1.0 - rxz * rxz) * (1.0 - ryz * ryz))
            else:
                inverse, singular = _inverse(stack, flagged)
                flagged |= singular
                r = -inverse[:, 0, 1] / np.sqrt(np.abs(inverse[:, 0, 0] * inverse[:, 1, 1]))
            r = np.clip(r, -ONE_MINUS_EPS, ONE_MINUS_EPS)
            p = np.asarray(self._kind._pvalue(r, rows - size - self._offset), dtype=np.float64)
        flagged |= ~(np.isfinite(r) & np.isfinite(p))
        flagged &= ~short
        r[short] = p[short] = NOT_TESTED
        return r, p, rows, flagged

    def _pairwise(self) -> tuple[np.ndarray, np.ndarray]:
        """The rows each pair of columns shares and the pair's correlation on them, ``(d, d)``.

        Computed once, from sums over the rows both columns observe of the values less each
        column's observed mean: ``[i, j]`` is canonical for ``i < j``.
        """
        if self._pairs is None:
            seen = self._observed.astype(np.float64)
            values = self._values.T
            # One BLAS thread: the products' bits then do not depend on the thread count.
            with np.errstate(all="ignore"), backend.one_blas_thread():
                count = seen.T @ seen
                sums = values.T @ seen  # [i, j]: column i's sum over the rows both observe
                spread = (values * values).T @ seen - sums * sums / count
                cov = values.T @ values - sums * sums.T / count
                corr = np.clip(cov / np.sqrt(spread * spread.T), -1.0, 1.0)
            self._pairs = (np.rint(count).astype(np.int64), corr)
        return self._pairs

    def _submatrices(self, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The correlation submatrices of the queries whose columns ``[x, y, *S]`` are the rows of
        ``columns``, on the rows each observes, and those row counts.

        A block's matrix is computed once, over its columns in data order, from the sums over its
        rows of the values less each column's observed mean, and kept for later queries over the
        same columns within ``_CACHE_BYTES``, each taking ``_ENTRY_BYTES`` besides its matrix; a
        query reads it in its own order. The blocks not kept are computed in steps of at most
        ``_STEP_BYTES`` of row values. A block's rows are counted from the observed masks first,
        and a block with fewer than :meth:`least_rows` is never summed: its matrix is ``NaN``.
        """
        k = columns.shape[1]
        keys = [tuple(row) for row in np.sort(columns, axis=1).tolist()]
        new = list(dict.fromkeys(key for key in keys if key not in self._blocks))
        step = max(1, _STEP_BYTES // (8 * k * self._data.shape[0]))
        least = self.least_rows(k - 2)
        for start in range(0, len(new), step):
            chunk = np.array(new[start : start + step], dtype=np.intp)
            kept = self._seen[chunk[:, 0]].copy()
            for c in range(1, k):
                kept &= self._seen[chunk[:, c]]
            counts = np.count_nonzero(kept, axis=1)
            blocks = np.full((len(chunk), k, k), np.nan)
            enough = counts >= least
            if enough.any():
                blocks[enough] = _block_correlations(
                    self._values, kept[enough], chunk[enough], counts[enough]
                )
            for key, block, count in zip(map(tuple, chunk.tolist()), blocks, counts.tolist()):
                self._blocks[key] = (block, count)
                self._kept += block.nbytes + _ENTRY_BYTES
        found = [self._blocks[key] for key in keys]
        for key in keys:
            self._blocks.move_to_end(key)
        while self._kept > _CACHE_BYTES:
            self._kept -= self._blocks.popitem(last=False)[1][0].nbytes + _ENTRY_BYTES
        ordered = np.stack([block for block, _ in found])
        rank = np.argsort(np.argsort(columns, axis=1), axis=1)
        at = np.arange(len(columns))[:, None, None]
        stack = ordered[at, rank[:, :, None], rank[:, None, :]]
        return stack, np.array([count for _, count in found], dtype=np.int64)

    def copies(self) -> tuple[tuple[int, ...], ...]:
        """The groups of columns that repeat one another up to scale and shift.

        Each pair of columns is compared on the rows where both are observed, as
        :meth:`PartialCorrelationDeletion.copies` compares them, with the same groups: a pair whose
        correlation from :meth:`_pairwise` is within ``_MARGIN`` of a copy, or is not a number, is
        compared on its rows as the exact path compares it, and every other pair is no copy.
        """
        if not hasattr(self, "_copies"):
            count, corr = self._pairwise()
            with np.errstate(invalid="ignore"):
                near = (count > 2) & ~(1.0 - corr * corr > _MARGIN * ZERO_RESIDUAL)
            exact = np.eye(self._data.shape[1])
            for a, b in zip(*np.nonzero(np.triu(near, 1))):
                rows = self._observed[:, a] & self._observed[:, b]
                pair = self._data[rows][:, [a, b]]
                if (np.ptp(pair, axis=0) > 0).all():
                    exact[a, b] = exact[b, a] = np.corrcoef(pair, rowvar=False)[0, 1]
            self._copies = copy_groups(exact, self._data.shape[0])
        return self._copies


class BoundedDeletion(TestwiseDeletion):
    """:class:`TestwiseDeletion` of a test that can stop once the side of ``alpha`` is certain."""

    def _bounded_pvalue(
        self,
        x: int,
        y: int,
        condition_set: Iterable[int],
        alpha: float | None,
        exact_above: bool = False,
    ) -> float:
        """The block test's ``_bounded_pvalue`` of the query, on the rows its columns share."""
        x_int, y_int, cond = _canonical(x, y, condition_set)
        return float(
            self._query(
                lambda test, a, b, s: test._bounded_pvalue(a, b, s, alpha, exact_above),
                x_int,
                y_int,
                cond,
                NOT_TESTED,
            )
        )


def testwise_deletion(
    name: str,
    factory: Callable[..., CITest],
    data: np.ndarray,
    categorical: npt.ArrayLike | None,
    options: Mapping[str, Any],
    *,
    mixed: bool,
    check_copies: bool = True,
) -> TestwiseDeletion:
    """Return the :class:`TestwiseDeletion` of the registered test ``name`` over ``data``.

    ``factory`` builds the test as :func:`~andrey.core.ci.make_indep_test` would, with ``options``,
    and with each block's slice of the ``categorical`` mask when the test is ``mixed``.
    """
    arr = np.asarray(data, dtype=np.float64)
    mask = None if categorical is None else np.asarray(categorical, dtype=bool)
    if mixed:

        def build(block: np.ndarray, columns: list[int]) -> CITest:
            return factory(block, None if mask is None else mask[columns], **options)

    else:

        def build(block: np.ndarray, columns: list[int]) -> CITest:
            return factory(block, **options)

    test_class = _shipped_class(factory)
    if test_class is None:
        continuous = None
    elif test_class is ChiSquare:
        continuous = np.zeros(arr.shape[1], dtype=bool)
    elif mixed:
        continuous = np.ones(arr.shape[1], dtype=bool) if mask is None else ~mask
    else:
        continuous = np.ones(arr.shape[1], dtype=bool)
    if test_class in _DF_OFFSET:
        kind: type[TestwiseDeletion] = GaussianDeletion
    elif test_class is not None and issubclass(test_class, PartialCorrelationTest):
        kind = PartialCorrelationDeletion
    elif test_class is not None and hasattr(test_class, "_bounded_pvalue"):
        kind = BoundedDeletion
    else:
        kind = TestwiseDeletion
    floor, floor_options = _row_floor(factory, test_class, options)
    return kind(
        arr,
        build,
        name=name,
        test_class=test_class,
        continuous=continuous,
        check_copies=check_copies,
        least=None if floor is None else lambda size: int(floor(size, **floor_options)),
    )


def _row_floor(
    factory: Callable[..., CITest], test_class: type | None, options: Mapping[str, Any]
) -> tuple[Callable[..., int] | None, Mapping[str, Any]]:
    """The ``least_rows`` of the test ``factory`` builds and the options it reads.

    A variant registered as ``partial(make_indep_test, name, **settings)`` has the floor of the
    test ``name`` with ``settings`` and ``options``; ``(None, options)`` when no test has one.
    """
    from . import ci

    if test_class is not None:
        return getattr(test_class, "least_rows", None), options
    if (
        isinstance(factory, partial)
        and factory.func is ci.make_indep_test
        and factory.args
        and factory.args[0] in ci._CI_REGISTRY
    ):
        inner = ci._CI_REGISTRY[factory.args[0]]
        return _row_floor(inner, _shipped_class(inner), {**factory.keywords, **options})
    return getattr(factory, "least_rows", None), options


def mark_environment(test: CITest, column: int) -> CITest:
    """``test``, with ``column`` marked as an environment indicator when it deletes rows per test
    (:meth:`TestwiseDeletion.mark_environments`)."""
    if isinstance(test, TestwiseDeletion):
        test.mark_environments([column])
    return test


def _block_correlations(
    values: np.ndarray, kept: np.ndarray, columns: np.ndarray, count: np.ndarray
) -> np.ndarray:
    """The correlation matrices of the blocks whose columns are the rows of ``columns``.

    ``values`` holds each column's values less its observed mean, ``0`` where unobserved, one row
    per column. Block ``b`` keeps the rows ``kept[b]`` marks, the ``count[b]`` rows observed in
    all of ``columns[b]``; its covariances come from the sums over every row with the others
    zeroed, the sums of its own values only. Each sum runs along one contiguous row of ``n``
    values, in NumPy's pairwise order for that length, the same whatever the BLAS, so a block's
    matrix has the same bits alone or with any others, in any order. Clipped to ``[-1, 1]``, as
    :func:`numpy.corrcoef` clips.
    """
    block = values[columns] * kept[:, None, :]
    sums = block.sum(axis=2)
    k = columns.shape[1]
    products = np.empty((len(columns), k, k), dtype=np.float64)
    pair = np.empty((len(columns), block.shape[2]), dtype=np.float64)
    for i in range(k):
        for j in range(i, k):
            np.multiply(block[:, i], block[:, j], out=pair)
            products[:, i, j] = products[:, j, i] = pair.sum(axis=1)
    cov = products - sums[:, :, None] * sums[:, None, :] / count[:, None, None]
    spread = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))
    return np.clip(cov / spread[:, :, None] / spread[:, None, :], -1.0, 1.0)


@functools.cache
def _apart(k: int) -> tuple[np.ndarray, np.ndarray]:
    """The row and column indices of the entries above the diagonal of a ``(k, k)`` matrix."""
    return np.triu_indices(k, 1)


def _arrays(
    queries: list[tuple[int, int, list[int]]], size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The canonical query arrays ``(x, y, conditions)`` of ``queries``, each of ``size``."""
    lo = np.array([x for x, _, _ in queries], dtype=np.intp)
    hi = np.array([y for _, y, _ in queries], dtype=np.intp)
    conds = np.array([cond for _, _, cond in queries], dtype=np.intp).reshape(len(queries), size)
    return lo, hi, conds


def _inverse(stack: np.ndarray, skip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverses of a ``(N, k, k)`` stack of correlation submatrices, and which are near
    singular.

    A matrix is near singular when an entry of its inverse reaches ``1 / _MARGIN`` of the bound
    :func:`~andrey.core.ci._invert_nonsingular` settles a matrix by, or, when the stack has an
    exactly singular matrix, when its smallest eigenvalue is at most ``_MARGIN`` times
    ``SINGULAR_RTOL`` of its largest. The rows ``skip`` marks are inverted as the identity. Each
    inverse is that of its matrix alone.
    """
    k = stack.shape[-1]
    stack = np.where(skip[:, None, None], np.eye(k), stack)
    singular = np.zeros(len(stack), dtype=bool)
    try:
        inverse = np.linalg.inv(stack)
    except np.linalg.LinAlgError:
        w = np.linalg.eigvalsh(stack)
        singular = ~(w[:, 0] > w[:, -1] * _MARGIN * SINGULAR_RTOL)
        stack[singular] = np.eye(k)
        inverse = np.linalg.inv(stack)
    limit = 1.0 / (_MARGIN * BOUND_MARGIN * SINGULAR_RTOL * k * k)
    return inverse, singular | ~(np.abs(inverse).max(axis=(1, 2)) < limit)


def _shipped_class(factory: Callable[..., CITest]) -> type | None:
    """The class a shipped test's factory builds; ``None`` for a registered one."""
    from . import ci

    if isinstance(factory, type):
        return factory
    if isinstance(factory, partial) and factory.func is ChiSquare:
        return ChiSquare
    if isinstance(factory, partial) and factory.func is ci._nonlinear:
        module, cls = ci.NONLINEAR_CLASSES[factory.args[0]]
        import importlib

        return getattr(importlib.import_module(f"{__package__}.{module}"), cls)
    return None


def _pins_blas(test_class: type | None, n: int) -> bool:
    """Whether the skeleton scans pin BLAS to one thread, as :func:`~andrey.core.ci.pins_blas`
    decides for the test over all ``n`` rows; a registered test, which may wrap a pinned one, is
    pinned, which changes no result."""
    if test_class is None:
        return True
    pin = getattr(test_class, "pin_blas", None)
    if isinstance(pin, property):  # KCI: unpinned from a row count on
        from .ci_kernel import KCI_BLAS_THREADS_FROM

        return n < KCI_BLAS_THREADS_FROM
    if pin is not None:
        return bool(pin)
    return issubclass(test_class, (PartialCorrelationTest, DGTest, CGTest))


def _in_data_columns(
    error: CopiedColumnsError | SingularCorrelationError, columns: list[int]
) -> CopiedColumnsError | SingularCorrelationError:
    """``error`` with its groups of copies, positions in a block, as the data's columns."""
    if not error.groups:
        return error
    groups = tuple(tuple(columns[i] for i in group) for group in error.groups)
    renamed = type(error)(str(error).replace(copies_message(error.groups), copies_message(groups)))
    renamed.groups = groups
    return renamed
