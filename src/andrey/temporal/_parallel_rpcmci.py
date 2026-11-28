"""The starts of Regime-PCMCI on worker processes.

The starts of one fit (:func:`~andrey.temporal.rpcmci.rpcmci`) are independent: start ``a`` draws
its first weights from ``start_generator(seed, a)`` and reads only the series and the settings. With
more than one worker (``andrey.config.n_jobs``), at least two starts, and ``n_starts * n_regimes *
n_steps`` reaching :data:`MIN_POOL_RPCMCI_WORK`, or the ``ANDREY_RPCMCI_PARALLEL_MIN_WORK`` that
overrides it, :func:`pooled_starts` runs each start as one task and returns the results in start
order. A fit reduces them in that order (:func:`~andrey.temporal.rpcmci.best_start`), so the chosen
start, its objective, and the number of failed starts do not depend on the order the workers finish
in, and are those of a fit in the calling process. It reads each start once, so the calling process
holds the best start and the starts that finished ahead of the one it reads, not every start.

A fit whose statistics can run on a device runs its starts in the calling process: a pinned ``cpu``,
``cuda``, or ``mps`` backend runs on PyTorch, whose runtime each worker would start for itself, and
``auto`` with a CUDA runtime live in the calling process can take the device for a start that a
fresh worker runs on NumPy, which could give another result.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from andrey.core import backend, env

if TYPE_CHECKING:
    from collections.abc import Iterator

    from andrey.temporal.rpcmci import RPCMCIResult, Starts

#: The default for the fewest ``n_starts * n_regimes * n_steps`` a fit needs to run its starts on
#: the worker pool; a smaller fit runs them in the caller's process.
#: ``ANDREY_RPCMCI_PARALLEL_MIN_WORK`` overrides it (0 sends every fit on NumPy to the pool), and a
#: fit with workers warns once on the default when it nears it; ``andrey calibrate --method
#: rpcmci`` measures the value for a machine. The default is measured on an AMD EPYC 9534 server
#: with 2, 4, and 8 workers: the first pool of a process takes about 1.4 s and each later one about
#: 0.05 s, and a fit in a fresh process ran faster on the pool than in the calling process from
#: about 27,000 for a series of 2,400 steps to about 108,000 for one of 150 steps with 2 workers,
#: from less with more. The default is above them, so no fit runs slower than in the calling
#: process.
MIN_POOL_RPCMCI_WORK: int = 150_000

#: The environment variables a start reads besides the ones ``andrey.config`` holds: the
#: covariance's GPU cutoff and its calibration. A worker the forkserver started earlier does not see
#: a value the caller set or removed since, so each fit hands over the caller's.
ENVIRONMENT = (env.COV_GPU_THRESHOLD, env.GPU_CALIBRATE)

_WORKER_STARTS: Starts | None = None
_WORKER_BLAS: object = None  # a held threadpoolctl controller capping this worker's BLAS


def _init_worker(
    starts: Starts,
    tie_break: str,
    numeric_backend: str,
    environment: tuple[tuple[str, str | None], ...],
) -> None:
    """Keep the caller's ``starts`` for the life of the worker, on one BLAS thread, with the
    caller's ``tie_break`` rule, numeric backend, and ``environment`` (:data:`ENVIRONMENT`, as
    ``(name, value or None)``), and no workers of its own."""
    global _WORKER_STARTS, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    _WORKER_BLAS = threadpool_limits(limits=1)
    backend.config.n_jobs = 1
    backend.config.backend = numeric_backend
    backend.config.tie_break = tie_break
    for name, value in environment:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    _WORKER_STARTS = starts


def _run(a: int) -> RPCMCIResult | None:
    """Worker side of :func:`pooled_starts`: start ``a`` of the fit the worker holds."""
    from andrey.temporal.rpcmci import run_start

    assert _WORKER_STARTS is not None  # _init_worker runs before any task
    return run_start(_WORKER_STARTS, a)


def _starts_run_on_numpy() -> bool:
    """Whether a start's statistics run on NumPy in this process, as they do on a worker.

    That is so for a ``numpy`` backend, and for ``auto`` while no CUDA runtime is live in this
    process, since ``auto`` takes a device only once its runtime is running. A pinned ``cpu``,
    ``cuda``, or ``mps`` backend runs on PyTorch.
    """
    preference = backend.config.backend
    if preference == "numpy":
        return True
    return preference == "auto" and not any(
        backend._is_live(device) for device in backend.TORCH_AUTO if device != "numpy"
    )


def pooled_starts(starts: Starts, n_starts: int) -> Iterator[RPCMCIResult | None] | None:
    """Starts ``0`` to ``n_starts - 1`` of a fit, run on the workers and read in order, or ``None``
    when the fit runs them in this process.

    A fit runs them on the workers when it has more than one worker, at least two starts, its
    statistics run on NumPy (:func:`_starts_run_on_numpy`), and ``n_starts * n_regimes *
    n_steps`` reaches :data:`MIN_POOL_RPCMCI_WORK`, or the ``ANDREY_RPCMCI_PARALLEL_MIN_WORK`` that
    overrides it. With more than one worker and the default cutoff, the first fit that reaches a
    tenth of it warns once. The settings a worker holds are the ones in effect here; the pool starts
    when the first start is read, and shuts down after the last, when a start raises, or when the
    caller closes the iterator. A start that fails is ``None``, as in
    :func:`~andrey.temporal.rpcmci.run_start`; an error in a start raises when it is read, the one
    of the first start in order that raises it.
    """
    from andrey.search import calibrate
    from andrey.search._parallel import min_work

    workers = backend.worker_count()
    if workers < 2 or n_starts < 2 or not _starts_run_on_numpy():
        return None
    work = n_starts * starts.n_regimes * starts.data.shape[0]
    cutoff = min_work(env.RPCMCI_PARALLEL_MIN_WORK, MIN_POOL_RPCMCI_WORK)
    if work >= calibrate.WARN_FRACTION * cutoff and calibrate.cutoff_is_default("rpcmci"):
        calibrate.warn_default_cutoff("rpcmci", cutoff)
    if work < cutoff:
        return None
    initargs = (
        starts,
        backend.config.tie_break,
        backend.config.backend,
        tuple((setting.name, os.environ.get(setting.name)) for setting in ENVIRONMENT),
    )
    return _in_order(min(workers, n_starts), initargs, n_starts)


def _in_order(max_workers: int, initargs: tuple, n_starts: int) -> Iterator[RPCMCIResult | None]:
    """The results of the ``n_starts`` starts in start order, each dropped as it is read, so the
    starts already read are not kept. The pool starts and takes every start when the first result
    is read, with the settings ``initargs`` holds, and shuts down when the iterator ends, a start
    raises, or the iterator is closed."""
    from andrey.search._parallel import collect, pick_context

    pool = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=pick_context(),
        initializer=_init_worker,
        initargs=initargs,
    )
    try:
        futures = deque(pool.submit(_run, a) for a in range(n_starts))
        while futures:
            # a dead worker raises here; no name holds the start the caller now reads
            yield collect([futures.popleft()], ordered=True)[0]
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
