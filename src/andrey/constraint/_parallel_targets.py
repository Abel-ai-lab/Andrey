"""Answers for many targets over one :class:`~andrey.constraint.blanket.LocalSearch`, in this
process or on worker processes.

One search answers every target, so a run or a spouse step that two targets read is made once.
Each run depends only on the data, the settings, and its variable, and the tests that keep answers
return the bits they would compute, so an answer is the one a search of its own would give.

With more than one worker, at least two targets, and a FisherZ, StudentT, or NormalScores test
(:func:`~andrey.constraint._skeleton.canonical_capable`), the targets are cut into contiguous
chunks that the workers answer in turn. Each worker holds a copy of the caller's search, made before
any step ran: the test's exact correlation bytes and the tie ranks, or the data to compute them
from, and pins BLAS to one thread, as the caller's search does inside
:meth:`~LocalSearch.pinned`. The answers come back in the order of the targets. These tests keep no
answers, so each kept step computes the same tests in any process, and each answer's count of tests
(:func:`~andrey.constraint.blanket.first_reads`) equals the one a single search gives the targets in
order, whatever the number of workers. Any other test answers in this process.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

from andrey.constraint._skeleton import canonical_capable
from andrey.constraint.blanket import LocalSearch, Reading, Step, first_reads
from andrey.core.ci import count_tests, counting_tests

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
    search: LocalSearch,
    answer: Callable[[LocalSearch, int], T],
    targets: Sequence[int],
    workers: int = 1,
) -> list[tuple[T, int, Counter[int]]]:
    """``answer(search, t)`` for each target ``t``, with the tests it computed and the rows read
    by every test it read (:func:`~andrey.core.ci_missing.rows_tested`), in the order of
    ``targets``.

    A test counts toward the first answer that reads it, so the counts add up to the tests the
    call computed, while the rows of an answer are those a search of its own would count, the
    steps an earlier answer made included. ``answer`` is a module-level function when ``workers`` is
    above 1, so that it reaches the workers. The tests the workers compute are added to the
    caller's :func:`~andrey.core.ci.counting_tests` count. The rows are counted only in this
    process: a test-wise deletion test, the one test that counts them, answers here.
    """
    if workers > 1 and len(targets) > 1 and canonical_capable(search.test):
        return _pooled(search, answer, targets, workers)
    out = []
    with search.pinned():
        for t in targets:
            with counting_tests() as tests, search.reading() as reading:
                found = answer(search, t)
            out.append((found, tests.n, reading.rows))
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
    counts = first_reads([reading for _, reading in found], tests)
    count_tests(sum(counts))
    return [(value, n, reading.rows) for (value, reading), n in zip(found, counts, strict=True)]
