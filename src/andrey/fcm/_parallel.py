"""Worker processes for the regressions of RESIT and of the exact search of CPCM.

Each regression of these fits depends only on the data, its variable, and its set of parents, so
the regressions are independent. With more than one worker (``andrey.config.n_jobs``) and enough
work, they run on worker processes that rebuild what a regression reads from the data and pin BLAS
to one thread, as the facade does in the calling process, so each result is the one the calling
process computes. Every choice stays in the calling process, which reads the same numbers in the
same order: RESIT's sink and its ties, and the exact search of CPCM and its scores.

- :func:`resit_pool` serves a RESIT fit from one pool: each candidate sink's regression and test of
  a step of the causal order, and each variable's edge removal.
- :func:`prefit` makes the residual of every column under a gaussian, gamma, or gumbel family on
  every set of parents of the exact search of CPCM (:func:`~andrey.fcm.cpcm.search`) before the
  search reads them. A fit that fails in a worker is left out, so the search makes it in the
  calling process and raises what it raises without workers.

A fit uses the workers only when its work reaches a cutoff: for RESIT the regressions of the causal
order times the rows, ``ANDREY_RESIT_PARALLEL_MIN_WORK`` or :data:`MIN_POOL_RESIT_WORK`; for CPCM
the fits of the exact search under those families times the rows,
``ANDREY_CPCM_PARALLEL_MIN_WORK`` or :data:`MIN_POOL_CPCM_WORK`. A fit with more than one worker
warns once when it nears a default cutoff, and ``andrey calibrate`` measures each cutoff for a
machine.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Self

import numpy as np

from andrey.core import backend, env

if TYPE_CHECKING:
    from andrey.fcm.cpcm import Residuals
    from andrey.fcm.gam import BasisCache

#: The default for the fewest regressions of RESIT's causal order times rows, ``(d * (d + 1) / 2 -
#: 1) * n`` for ``d`` variables and ``n`` rows, a fit needs to use the worker pool; a smaller fit
#: runs in the calling process. ``ANDREY_RESIT_PARALLEL_MIN_WORK`` overrides it (0 sends every fit
#: to the pool), and a fit with workers warns once on the default when it nears it; ``andrey
#: calibrate --method resit`` measures the value for a machine. On an AMD EPYC 9534 server with 2,
#: 4, and 8 workers, a fit on the first pool of its process, the start included, was no faster than
#: the serial fit up to 40,000 (6 variables, 2,000 rows, on 2 workers: 0.97 of the serial time),
#: and from 41,800 to 140,000 every case took 0.27 to 0.91 of it. The first pool of a process
#: starts in about 1.1 to 1.5 s, a later one in about 0.02 s. The default stays above the
#: crossovers.
MIN_POOL_RESIT_WORK: int = 50_000

#: The default for the fewest fits of the exact search of CPCM under a gaussian, gamma, or gumbel
#: family times rows, ``d * (2 ** (d - 1) - 1) * n`` for ``d`` variables of such families and ``n``
#: rows, a fit needs to use the worker pool; a smaller fit runs in the calling process.
#: ``ANDREY_CPCM_PARALLEL_MIN_WORK`` overrides it (0 sends every fit to the pool), and a fit with
#: workers warns once on the default when it nears it; ``andrey calibrate --method cpcm`` measures
#: the value for a machine. On an AMD EPYC 9534 server with 2, 4, and 8 workers, a fit of 3 or 4
#: variables under the gaussian family on the first pool of its process, the start included, was no
#: faster than the serial fit up to 1,800 (3 variables, 200 rows, on 2 workers: 1.10 of the serial
#: time), and from 7,200 to 44,800 every case took 0.31 to 0.87 of it. The gamma and gumbel
#: families cost more per fit; every other family's fits cost less than starting the pool, and with
#: 4 variables, 1,600 rows, and 2, 4, and 8 workers the pool gave the fixed-scale Gaussian family
#: 1.01, 0.96, and 0.92 of the serial time, so those fits stay in the calling process. The default
#: stays above the crossovers.
MIN_POOL_CPCM_WORK: int = 8_000

_BLAS: object = None  # a held threadpoolctl controller capping this worker's BLAS
_RESIT: tuple[BasisCache, np.ndarray, np.ndarray, np.ndarray] | None = None
_CPCM: Residuals | None = None


def _pin() -> None:
    """Worker side: BLAS on one thread and the NumPy backend, for the life of the worker."""
    global _BLAS
    from threadpoolctl import threadpool_limits

    _BLAS = threadpool_limits(limits=1)
    backend.config.backend = "numpy"


def _keeps_layout(*arrays: np.ndarray) -> bool:
    """Whether each array reaches a worker with its memory layout, which pickling keeps for a C or
    Fortran contiguous array and replaces by C order for any other."""
    return all(a.flags.c_contiguous or a.flags.f_contiguous for a in arrays)


def _uses_workers(method: str, setting: env.Setting[int], default: int, work: int) -> bool:
    """Whether a fit of ``work`` uses the workers: it has more than one and ``work`` reaches the
    cutoff ``setting``, or ``default``. With more than one worker and the default cutoff, the first
    fit that reaches a tenth of it warns once."""
    from andrey.search import calibrate
    from andrey.search._parallel import min_work

    if backend.worker_count() < 2:
        return False
    cutoff = min_work(setting, default)
    if work >= calibrate.WARN_FRACTION * cutoff and calibrate.cutoff_is_default(method):
        calibrate.warn_default_cutoff(method, cutoff)
    return work >= cutoff


def _pool(workers: int, initializer, initargs: tuple) -> ProcessPoolExecutor:
    from andrey.search._parallel import pick_context

    return ProcessPoolExecutor(
        max_workers=workers, mp_context=pick_context(), initializer=initializer, initargs=initargs
    )


def _init_resit(X: np.ndarray, unit: np.ndarray, allowed: np.ndarray, required: np.ndarray) -> None:
    """Worker side of :class:`ResitPool`: the fit's bases, kernel columns, and edge matrices."""
    from andrey.fcm.gam import BasisCache

    global _RESIT
    _pin()
    _RESIT = (BasisCache(X), unit, allowed, required)


def _sink_log_p(remaining: list[int], k: int) -> float:
    from andrey.fcm import resit

    state = _RESIT
    assert state is not None
    bases, unit, allowed, _ = state
    return resit._sink_log_p(bases, unit, allowed, remaining, k)


def _kept_parents(
    order: np.ndarray, position: int, alpha: float
) -> tuple[list[int], list[tuple[int, float]]]:
    from andrey.fcm import resit

    state = _RESIT
    assert state is not None
    bases, unit, allowed, required = state
    return resit._kept_parents(bases, unit, order, position, alpha, allowed, required)


class ResitPool:
    """Worker processes that answer the regressions of one RESIT fit.

    Each worker holds the fit's bases, kernel columns, and edge matrices, and answers
    :func:`~andrey.fcm.resit.causal_order`'s candidate tests and
    :func:`~andrey.fcm.resit.remove_edges`'s per-variable removals with the functions the calling
    process uses, in the order of the calling process's loops: a failure raises the first one that
    loop meets.
    """

    def __init__(
        self,
        X: np.ndarray,
        unit: np.ndarray,
        allowed: np.ndarray,
        required: np.ndarray,
        workers: int,
    ) -> None:
        self._pool = _pool(min(workers, X.shape[1]), _init_resit, (X, unit, allowed, required))

    def sink_log_ps(self, remaining: list[int], candidates: list[int]) -> list[float]:
        """The log p-value of each of ``candidates``' sink test, in order."""
        from andrey.search._parallel import collect

        futures = [self._pool.submit(_sink_log_p, remaining, k) for k in candidates]
        return collect(futures, ordered=True)

    def kept_parents(
        self, order: np.ndarray, alpha: float
    ) -> list[tuple[list[int], list[tuple[int, float]]]]:
        """:func:`~andrey.fcm.resit._kept_parents` of each position of ``order`` from 1, in order.

        The positions are submitted from the last, which has the most parents to test, so the
        workers' loads even out; the results are read in order.
        """
        from andrey.search._parallel import collect

        positions = range(1, len(order))
        futures = {
            p: self._pool.submit(_kept_parents, order, p, alpha) for p in reversed(positions)
        }
        return collect([futures[p] for p in positions], ordered=True)

    def close(self) -> None:
        self._pool.shutdown()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def resit_pool(
    X: np.ndarray, unit: np.ndarray, allowed: np.ndarray, required: np.ndarray
) -> AbstractContextManager[ResitPool | None]:
    """The pool a RESIT fit of the ``(n, d)`` data ``X`` and kernel columns ``unit`` runs on, as a
    context manager that gives ``None`` when it runs in this process.

    The fit uses the workers when it has more than one, its work, ``(d * (d + 1) / 2 - 1) * n``
    regressions of the causal order times rows, reaches ``ANDREY_RESIT_PARALLEL_MIN_WORK`` or
    :data:`MIN_POOL_RESIT_WORK`, and ``X`` and ``unit`` keep their layout on the way
    (:func:`_keeps_layout`). The decision is made here, not in a generator, so that a warning about
    the default cutoff points at the caller's line.
    """
    n, d = X.shape
    work = (d * (d + 1) // 2 - 1) * n
    if (
        d > 2
        and _keeps_layout(X, unit)
        and _uses_workers("resit", env.RESIT_PARALLEL_MIN_WORK, MIN_POOL_RESIT_WORK, work)
    ):
        return ResitPool(X, unit, allowed, required, backend.worker_count())
    return nullcontext()


def _init_cpcm(X: np.ndarray, names: list[str], seed: int) -> None:
    """Worker side of :func:`prefit`: the table's residuals under the families named."""
    from andrey.fcm import cpcm

    global _CPCM
    _pin()
    _CPCM = cpcm.Residuals(X, [cpcm.FAMILIES[name] for name in names], seed)


def _residual(j: int, parents: tuple[int, ...]) -> np.ndarray | None:
    """The residual of column ``j`` on ``parents``, or ``None`` when its fit fails, which the
    calling process then fits for itself and raises."""
    res = _CPCM
    assert res is not None
    try:
        return res.residual(j, parents)
    except Exception:  # noqa: BLE001 -- any failure is the calling process's to raise
        return None


def prefit(res: Residuals) -> None:
    """Fit the residual of each column under a gaussian, gamma, or gumbel family on every nonempty
    set of the others on worker processes and keep them in ``res``, so the exact search reads them.

    Only the columns whose family has a scale or shape that changes with the parents
    (``"gaussian"``, ``"gamma"``, and ``"gumbel"``) are fitted here: a fit under any other family
    costs less than starting the workers, and the scoring that follows in the calling process
    grows faster with the rows than those fits do, so they stay in the calling process.

    Does nothing below the cutoff: the fits of those columns times the rows must reach
    ``ANDREY_CPCM_PARALLEL_MIN_WORK`` or :data:`MIN_POOL_CPCM_WORK`, with more than one worker, at
    least 3 variables (two variables have two quick fits, which never repay the workers' start), a
    family that is one of :data:`~andrey.fcm.cpcm.FAMILIES` in every column, and a table that keeps
    its layout on the way. A fit that fails is not kept, so the search fits it in the calling
    process and raises its error.
    """
    from andrey.fcm import cpcm
    from andrey.search._parallel import collect

    n, d = res.X.shape
    smooth = [j for j, family in enumerate(res.families) if family.second == "smooth"]
    keys = [
        (j, parents)
        for size in range(d - 1, 0, -1)
        for j in smooth
        for parents in itertools.combinations([i for i in range(d) if i != j], size)
    ]
    if (
        d < 3
        or not keys
        or not _keeps_layout(res.X)
        or any(cpcm.FAMILIES.get(f.name) is not f for f in res.families)
        or not _uses_workers("cpcm", env.CPCM_PARALLEL_MIN_WORK, MIN_POOL_CPCM_WORK, len(keys) * n)
    ):
        return
    names = [family.name for family in res.families]
    # The fits with the most parents, the slowest, go first, so the workers' loads even out.
    with _pool(
        min(backend.worker_count(), len(keys)), _init_cpcm, (res.X, names, res.seed)
    ) as pool:
        futures = [pool.submit(_residual, j, parents) for j, parents in keys]
        for (j, parents), residual in zip(keys, collect(futures, ordered=True), strict=True):
            if residual is not None:
                res.store(j, parents, residual)
