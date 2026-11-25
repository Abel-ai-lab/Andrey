"""Answers for many targets over one :class:`~andrey.constraint.blanket.LocalSearch`, in this
process or on worker processes.

One search answers every target, so a run or a spouse step that two targets read is made once.
Each run depends only on the data, the settings, and its variable, so an answer is the one a search
of its own would give, and its count of tests is the one such a search computes
(:func:`~andrey.constraint.blanket.read_tests`): its own tests and those of every step it read,
whichever answer made the step. A test that keeps answers, as the chi-square tests keep their
marginal answers and count a kept one as no test, counts each pair the answer or its steps asked
of them once, as a new search computes it the first time it is asked. A test-wise deletion test,
which counts the rows each test reads, the queries it does not test, and the pairs no test decided
(:func:`~andrey.core.ci_missing.rows_tested`), answers each target with a search of its own in
this process, so each answer counts what a call for that target alone counts.

With more than one worker, at least two targets, and a FisherZ, StudentT, or NormalScores test
(:func:`~andrey.constraint._skeleton.canonical_capable`), the targets are cut into contiguous
chunks that the workers answer in turn. Each worker holds a copy of the caller's search, made before
any step ran: the test's exact correlation bytes and the tie ranks, or the data to compute them
from, and pins BLAS to one thread, as the caller's search does inside
:meth:`~LocalSearch.pinned`. The answers come back in the order of the targets. These tests keep no
answers, so each kept step computes the same tests in any process, and each answer's count of tests
(:func:`~andrey.constraint.blanket.read_tests`) is the one a search of its own computes,
whatever the number of workers, and the caller's count of tests
(:func:`~andrey.constraint.blanket.first_reads`) is the one a single search computes. Any other
test answers in this process with the one search.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from andrey.constraint._skeleton import canonical_capable
from andrey.constraint.blanket import (
    LazyTieRanks,
    LocalSearch,
    Reading,
    Step,
    first_reads,
    read_tests,
)
from andrey.core.ci import count_tests, counting_tests
from andrey.core.ci_missing import TestwiseDeletion, rows_tested

T = TypeVar("T")

#: Chunks per worker: smaller chunks balance targets of unequal cost, and a worker keeps its search
#: from one chunk to the next, so only steps two workers both need are made twice.
CHUNKS_PER_WORKER = 4

_WORKER_SEARCH: LocalSearch | None = None
_WORKER_BLAS: object = None  # a held threadpoolctl controller capping this worker's BLAS


def _init_worker(search: LocalSearch) -> None:
    """Keep the caller's search, with BLAS on one thread and the NumPy backend, for the life of
    the worker."""
    global _WORKER_SEARCH, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend

    _WORKER_BLAS = threadpool_limits(limits=1)
    backend.config.backend = "numpy"
    if isinstance(search.ranks, LazyTieRanks):  # a forked worker inherits the caller's limits
        search.ranks.use_active_limits()
    _WORKER_SEARCH = search


def _answer_chunk(
    answer: Callable[[LocalSearch, int], T], targets: list[int]
) -> tuple[list[tuple[T, Reading]], list[tuple[Step, int]]]:
    """Worker side: each target's answer and reading, and the steps this chunk ran first with their
    tests."""
    search = _WORKER_SEARCH
    assert search is not None
    ran = len(search.step_tests)
    out = []
    with counting_tests():
        for t in targets:
            with search.reading() as reading:
                found = answer(search, t)
            out.append((found, reading))
    return out, list(search.step_tests.items())[ran:]


def answers(
    search_of: Callable[[], LocalSearch],
    answer: Callable[[LocalSearch, int], T],
    targets: Sequence[int],
    workers: int = 1,
) -> list[tuple[T, int, Counter[int]]]:
    """``answer(search, t)`` for each target ``t``, with the tests and the rows of every test it
    read (:func:`~andrey.core.ci_missing.rows_tested`), in the order of ``targets``.

    ``search_of`` makes a new search. An answer's tests and rows are those a search of its own would
    count, the steps an earlier answer made included and each pair asked of a chi-square test's
    kept answers once, so each answer counts what a call for that target alone counts, while the
    caller's :func:`~andrey.core.ci.counting_tests` count is the tests the call computed, each
    once. A test-wise deletion test answers each target with a search of its own, whose counts are
    a call for that target alone gives, the queries not tested and the pairs no test decided among
    them. ``answer`` is a module-level function when ``workers`` is above 1, so that it reaches the
    workers. The tests the workers compute are added to the caller's
    :func:`~andrey.core.ci.counting_tests` count. The rows are counted only in this process: a
    test-wise deletion test, the one test that counts them, answers here.
    """
    search = search_of()
    if isinstance(search.test, TestwiseDeletion):
        out = []
        for i, t in enumerate(targets):
            own = search if i == 0 else search_of()
            with own.pinned(), counting_tests() as tests, rows_tested() as rows:
                found = answer(own, t)
            out.append((found, tests.n, rows))
        return out
    if workers > 1 and len(targets) > 1 and canonical_capable(search.test):
        return _pooled(search, answer, targets, workers)
    out = []
    with search.pinned():
        for t in targets:
            with search.reading() as reading:
                found = answer(search, t)
            tests = read_tests(reading, search.step_tests, search.step_kept)
            out.append((found, tests, reading.rows))
    return out


def _pooled(
    search: LocalSearch,
    answer: Callable[[LocalSearch, int], T],
    targets: Sequence[int],
    workers: int,
) -> list[tuple[T, int, Counter[int]]]:
    """:func:`answers` on ``workers`` processes."""
    from andrey.search._parallel import collect, even_chunks, pick_context

    chunks = even_chunks(len(targets), workers * CHUNKS_PER_WORKER)
    with ProcessPoolExecutor(
        max_workers=min(workers, len(chunks)),
        mp_context=pick_context(),
        initializer=_init_worker,
        initargs=(search,),
    ) as pool:
        futures = [
            pool.submit(_answer_chunk, answer, [int(targets[i]) for i in chunk]) for chunk in chunks
        ]
        results = collect(futures, ordered=True)
    tests: dict[Step, int] = {}
    found: list[tuple[T, Reading]] = []
    for chunk_answers, steps in results:
        found += chunk_answers
        for step, n in steps:
            tests.setdefault(step, n)
    count_tests(sum(first_reads([reading for _, reading in found], tests)))
    return [(value, read_tests(reading, tests), reading.rows) for value, reading in found]
