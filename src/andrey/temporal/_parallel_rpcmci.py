"""The starts of Regime-PCMCI on worker processes.

The starts of one fit (:func:`~andrey.temporal.rpcmci.rpcmci`) are independent: start ``a`` draws
its first weights from ``start_generator(seed, a)`` and reads only the series and the settings. With
more than one worker (``andrey.config.n_jobs``), at least two starts, and ``n_starts * n_regimes *
n_steps`` reaching :data:`MIN_POOL_RPCMCI_WORK`, or the ``ANDREY_RPCMCI_PARALLEL_MIN_WORK`` that
overrides it, :func:`pooled_starts` runs each start as one task and returns the results in start
order. A fit reduces them in that order (:func:`~andrey.temporal.rpcmci.best_start`), so the chosen
start, its objective, and the number of failed starts do not depend on the order the workers finish
in, and are those of a fit in the calling process.

A worker holds the caller's series, ranks, and settings, its ``tie_break`` rule, and its numeric
backend, and runs on one BLAS thread with one worker of its own, as a fit in the calling process
does inside :func:`~andrey.core.backend.one_blas_thread`. A fit reads no other setting.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from andrey.core import backend, env

if TYPE_CHECKING:
    from andrey.temporal.rpcmci import RPCMCIResult, Starts

#: The default for the fewest ``n_starts * n_regimes * n_steps`` a fit needs to run its starts on
#: the worker pool; a smaller fit runs them in the caller's process.
#: ``ANDREY_RPCMCI_PARALLEL_MIN_WORK`` overrides it (0 sends every fit to the pool), and a fit with
#: workers warns once on the default when it nears it; ``andrey calibrate --method rpcmci``
#: measures the value for a machine. @RPCMCI_DEFAULT_COMMENT@
MIN_POOL_RPCMCI_WORK: int = 20_000

_WORKER_STARTS: Starts | None = None
_WORKER_BLAS: object = None  # a held threadpoolctl controller capping this worker's BLAS


def _init_worker(starts: Starts, tie_break: str, numeric_backend: str) -> None:
    """Keep the caller's ``starts`` for the life of the worker, on one BLAS thread, with the
    caller's ``tie_break`` rule and numeric backend and no workers of its own.

    A worker the forkserver started earlier does not read the configuration the caller reads now.
    """
    global _WORKER_STARTS, _WORKER_BLAS
    from threadpoolctl import threadpool_limits

    _WORKER_BLAS = threadpool_limits(limits=1)
    backend.config.n_jobs = 1
    backend.config.backend = numeric_backend
    backend.config.tie_break = tie_break
    _WORKER_STARTS = starts


def _run(a: int) -> RPCMCIResult | None:
    """Worker side of :func:`pooled_starts`: start ``a`` of the fit the worker holds."""
    from andrey.temporal.rpcmci import run_start

    assert _WORKER_STARTS is not None  # _init_worker runs before any task
    return run_start(_WORKER_STARTS, a)


def pooled_starts(starts: Starts, n_starts: int) -> list[RPCMCIResult | None] | None:
    """Starts ``0`` to ``n_starts - 1`` of a fit, in order, run on the workers, or ``None`` when
    the fit runs them in this process.

    A fit runs them on the workers when it has more than one worker, at least two starts, and
    ``n_starts * n_regimes * n_steps`` reaching :data:`MIN_POOL_RPCMCI_WORK`, or the
    ``ANDREY_RPCMCI_PARALLEL_MIN_WORK`` that overrides it. With more than one worker and the default
    cutoff, the first fit that reaches a tenth of it warns once. A start that fails is ``None``, as
    in :func:`~andrey.temporal.rpcmci.run_start`; an error in a start raises here, the one of the
    first start in order that raises it.
    """
    from andrey.search import calibrate
    from andrey.search._parallel import collect, min_work, pick_context

    workers = backend.worker_count()
    if workers < 2 or n_starts < 2:
        return None
    work = n_starts * starts.n_regimes * starts.data.shape[0]
    cutoff = min_work(env.RPCMCI_PARALLEL_MIN_WORK, MIN_POOL_RPCMCI_WORK)
    if work >= calibrate.WARN_FRACTION * cutoff and calibrate.cutoff_is_default("rpcmci"):
        calibrate.warn_default_cutoff("rpcmci", cutoff)
    if work < cutoff:
        return None
    pool = ProcessPoolExecutor(
        max_workers=min(workers, n_starts),
        mp_context=pick_context(),
        initializer=_init_worker,
        initargs=(starts, backend.config.tie_break, backend.config.backend),
    )
    try:
        return collect([pool.submit(_run, a) for a in range(n_starts)], ordered=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
