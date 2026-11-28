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
(:func:`~andrey.core.ci_missing.rows_tested`), answers every target in this process with the one
search, which keeps each run's rows with the run (:meth:`~LocalSearch.counted`), so each answer
counts the rows of the runs it read, as a call for that target alone counts them. Any other test
answers in this process with the one search.

A full-graph fit of IAMB, Inter-IAMB, or MMPC (:meth:`~LocalSearch.structure`) runs the search of
every variable: with more than one worker, a test of that kind, and at least
:data:`MIN_POOL_BLANKET_WORK` target and candidate pairs, or the
``ANDREY_BLANKET_PARALLEL_MIN_WORK`` that overrides it, the runs are made on the workers the same
way and come back in the order of the variables (:func:`pooled_runs`). The pruning of spouses and
the collider tests of that fit stay in this process.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, TypeVar, cast

from andrey.constraint._skeleton import canonical_capable
from andrey.constraint.blanket import LocalSearch
from andrey.core import backend, env
from andrey.core.ci_missing import TestwiseDeletion

if TYPE_CHECKING:
    import numpy as np

    from andrey.constraint.blanket import RunSepsets
    from andrey.core.ci import PartialCorrelationTest
    from andrey.core.knowledge import Constraints
    from andrey.search._parallel import SharedMatrix

T = TypeVar("T")

#: The default for the fewest target and candidate pairs, ``d * (d - 1)`` over the ``d`` variables
#: of the data, a full-graph fit of IAMB, Inter-IAMB, or MMPC needs to make its runs on the worker
#: pool; a smaller fit makes them in the caller's process. ``ANDREY_BLANKET_PARALLEL_MIN_WORK``
#: overrides it (0 sends every fit to the pool), and a fit with workers warns once on the default
#: when it nears it; ``andrey calibrate --method blanket`` measures the value for a machine. On an
#: Intel Xeon Gold 6542Y server with 2, 4, and 8 workers, a fit on the first pool of its process,
#: the start included, beat the serial fit of each learner from about 120,000 pairs (Inter-IAMB) to
#: about 280,000 (MMPC on 2 workers), and at 359,400 pairs, 600 variables, every case took 0.26 to
#: 0.84 of the serial time. The first pool of a process starts in about 1.4 s, a later one in about
#: 0.05 s. The default stays above the crossovers.
MIN_POOL_BLANKET_WORK: int = 350_000

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
    corr: np.ndarray,
    n_samples: int,
    copies: tuple[tuple[int, ...], ...] | None,
    ranks: Sequence[int],
    alpha: float,
    max_depth: int | None,
    constraints: Constraints | None,
    levels: np.ndarray | None,
    n: int,
) -> LocalSearch:
    """The search :func:`_seed` describes, over ``corr``, a read-only map of the shared copy in a
    worker where there is one."""
    test = test_class.from_corr(corr, n_samples)
    if copies is not None:
        test._copies = copies
    return LocalSearch(
        test, ranks, alpha, max_depth=max_depth, constraints=constraints, levels=levels, n=n
    )


def _init_worker(shared_scan_min_work: int | None, *seed) -> None:
    """Rebuild the caller's search from its :func:`_seed` and keep it, with BLAS on one thread and
    the NumPy backend, for the life of the worker.

    ``shared_scan_min_work`` is the caller's ``ANDREY_SHARED_SCAN_MIN_WORK`` (``None``: unset). It
    chooses between two paths whose p-values differ in the last bits, so the worker reads the value
    the caller reads, not the one the forkserver started with.
    """
    global _WORKER_SEARCH, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    if shared_scan_min_work is None:
        os.environ.pop(env.SHARED_SCAN_MIN_WORK.name, None)
    else:
        os.environ[env.SHARED_SCAN_MIN_WORK.name] = str(shared_scan_min_work)
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
        with search.pinned():
            return [search.counted(partial(answer, search, t)) for t in targets]
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
    """:func:`answers` on ``workers`` processes; the shared matrix is closed once the pool has."""
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
            initargs=(env.SHARED_SCAN_MIN_WORK.read(), *_seed(search, shared.payload)),
        ) as pool,
    ):
        futures = [
            pool.submit(_answer_chunk, answer, [int(targets[i]) for i in chunk]) for chunk in chunks
        ]
        results = collect(futures, ordered=True)
    return [found for chunk in results for found in chunk]


def _run(method: str, search: LocalSearch, target: int) -> tuple[list[int], RunSepsets]:
    """Worker side of :func:`pooled_runs`: ``target``'s run by the learner ``method``."""
    return search.run(method, target)


def pooled_runs(
    search: LocalSearch, method: str, targets: Sequence[int]
) -> list[tuple[list[int], RunSepsets]] | None:
    """The runs of the learner ``method`` (:meth:`~LocalSearch.run`) for ``targets``, in order, made
    on the workers, or ``None`` when the fit makes them in this process.

    A fit makes them on the workers when it has more than one worker, at least two ``targets``, a
    :func:`~andrey.constraint._skeleton.canonical_capable` test, and ``len(targets) * (d - 1)``
    target and candidate pairs reaching :data:`MIN_POOL_BLANKET_WORK`, or the
    ``ANDREY_BLANKET_PARALLEL_MIN_WORK`` that overrides it. With more than one worker and the
    default cutoff, the first fit that reaches a tenth of it warns once. Each run is the one a
    search that makes it alone returns, so a fit gives the same sets and bits with any number of
    workers.
    """
    from andrey.search import calibrate
    from andrey.search._parallel import min_work

    workers = backend.worker_count()
    if workers < 2 or len(targets) < 2 or not canonical_capable(search.test):
        return None
    pairs = len(targets) * (search.d - 1)
    cutoff = min_work(env.BLANKET_PARALLEL_MIN_WORK, MIN_POOL_BLANKET_WORK)
    if pairs >= calibrate.WARN_FRACTION * cutoff and calibrate.cutoff_is_default("blanket"):
        calibrate.warn_default_cutoff("blanket", cutoff)
    if pairs < cutoff:
        return None
    return _pooled(search, partial(_run, method), targets, workers)
