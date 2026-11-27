"""Answers for many targets over one :class:`~andrey.constraint.blanket.LocalSearch`, in this
process or on worker processes.

One search answers every target, reusing runs shared by targets. Each run depends only on the data,
settings, and its variable, so each answer matches a separate search.

With more than one worker, at least two targets, and a FisherZ, StudentT, or NormalScores test
(:func:`~andrey.constraint._skeleton.canonical_capable`), the targets are cut into contiguous
chunks that the workers answer in turn. Each worker rebuilds the caller's search, as made before
any run, from its settings and its test's correlation matrix (one shared copy where
:class:`~andrey.search._parallel.PoolMatrix` makes one), and pins BLAS to one thread, as the
caller's search does inside :meth:`~LocalSearch.pinned`. The answers come back in the order of the
targets. A test-wise deletion test, which counts the rows each test reads
(:func:`~andrey.core.ci_missing.rows_tested`), answers each target with a search of its own in this
process, so each answer counts the rows a call for that target alone counts. Any other test
answers in this process with the one search.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, TypeVar, cast

from andrey.constraint._skeleton import canonical_capable
from andrey.constraint.blanket import LocalSearch
from andrey.core.ci_missing import TestwiseDeletion, rows_tested

if TYPE_CHECKING:
    import numpy as np

    from andrey.core.ci import PartialCorrelationTest
    from andrey.core.knowledge import Constraints
    from andrey.search._parallel import SharedMatrix

T = TypeVar("T")

#: Chunks per worker: smaller chunks balance unequal target costs; workers keep searches across
#: chunks, so only runs two workers both need are made twice.
CHUNKS_PER_WORKER = 4

_WORKER_SEARCH: LocalSearch | None = None
_WORKER_BLAS: object = None  # a held threadpoolctl controller capping this worker's BLAS


def _seed(search: LocalSearch, corr: np.ndarray | SharedMatrix) -> tuple:
    """What a worker rebuilds ``search`` from (:func:`_rebuild`), with ``corr`` standing for its
    test's correlation matrix: the test's class, sample count, and groups of copies when found, and
    the search's settings."""
    test = cast("PartialCorrelationTest", search.test)
    return (
        type(test),
        corr,
        test.to_seed()[1],
        getattr(test, "_copies", None),
        search.ranks,
        search.alpha,
        search.max_depth,
        search.constraints,
        search.levels,
        search.n,
    )


def _rebuild(
    test_class: type[PartialCorrelationTest],
    corr: np.ndarray | SharedMatrix,
    n_samples: int,
    copies: tuple[tuple[int, ...], ...] | None,
    ranks: Sequence[int],
    alpha: float,
    max_depth: int | None,
    constraints: Constraints | None,
    levels: np.ndarray | None,
    n: int,
) -> LocalSearch:
    """The search :func:`_seed` describes, over the matrix ``corr`` stands for
    (:func:`~andrey.search._parallel.attach`)."""
    from andrey.search._parallel import attach

    test = test_class.from_corr(attach(corr), n_samples)
    if copies is not None:
        test._copies = copies
    return LocalSearch(
        test, ranks, alpha, max_depth=max_depth, constraints=constraints, levels=levels, n=n
    )


def _init_worker(*seed) -> None:
    """Rebuild the caller's search from its :func:`_seed` and keep it, with BLAS on one thread and
    the NumPy backend, for the life of the worker."""
    global _WORKER_SEARCH, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    from andrey.core import backend

    _WORKER_BLAS = threadpool_limits(limits=1)
    backend.config.backend = "numpy"
    _WORKER_SEARCH = _rebuild(*seed)


def _answer_chunk(answer: Callable[[LocalSearch, int], T], targets: list[int]) -> list[T]:
    """Worker side: each target's answer, in order."""
    search = _WORKER_SEARCH
    assert search is not None
    return [answer(search, t) for t in targets]


def answers(
    search_of: Callable[[], LocalSearch],
    answer: Callable[[LocalSearch, int], T],
    targets: Sequence[int],
    workers: int = 1,
) -> list[tuple[T, Counter[int]]]:
    """``answer(search, t)`` for each target ``t``, in ``targets`` order, with the rows its tests
    read (:func:`~andrey.core.ci_missing.rows_tested`).

    ``search_of`` makes a new search. With ``workers`` above 1, ``answer`` is a module-level
    function so it reaches the workers.
    """
    search = search_of()
    if isinstance(search.test, TestwiseDeletion):
        out = []
        for i, t in enumerate(targets):
            own = search if i == 0 else search_of()
            with own.pinned(), rows_tested() as rows:
                out.append((answer(own, t), rows))
        return out
    if workers > 1 and len(targets) > 1 and canonical_capable(search.test):
        return [(found, Counter()) for found in _pooled(search, answer, targets, workers)]
    with search.pinned():
        return [(answer(search, t), Counter()) for t in targets]


def _pooled(
    search: LocalSearch,
    answer: Callable[[LocalSearch, int], T],
    targets: Sequence[int],
    workers: int,
) -> list[T]:
    """:func:`answers` on ``workers`` processes; the shared matrix's name is removed once the pool
    has closed."""
    from andrey.search._parallel import PoolMatrix, collect, even_chunks, pick_context

    chunks = even_chunks(len(targets), workers * CHUNKS_PER_WORKER)
    context = pick_context()
    corr, _ = cast("PartialCorrelationTest", search.test).to_seed()
    with (
        PoolMatrix(corr, context) as shared,
        ProcessPoolExecutor(
            max_workers=min(workers, len(chunks)),
            mp_context=context,
            initializer=_init_worker,
            initargs=_seed(search, shared.payload),
        ) as pool,
    ):
        futures = [
            pool.submit(_answer_chunk, answer, [int(targets[i]) for i in chunk]) for chunk in chunks
        ]
        results = collect(futures, ordered=True)
    return [found for chunk in results for found in chunk]
