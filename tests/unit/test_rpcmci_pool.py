"""The starts of Regime-PCMCI on the worker pool.

A pooled fit gives the result a fit in one process gives, bit for bit, whichever worker finishes
first, uses the pool only from its cutoff, and warns once when it runs on the default cutoff near
it. The pool is spied on in every test that claims it ran.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import warnings
import weakref
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
import pytest
import threadpoolctl

import andrey
from andrey.cli import main
from andrey.core import backend, env
from andrey.core.ties import tie_ranks
from andrey.search import _parallel as par
from andrey.search import calibrate as cal
from andrey.temporal import _parallel_rpcmci as pr
from andrey.temporal import rpcmci as rp

VARIABLE = cal.METHODS["rpcmci"].variable
GPU_VARIABLES = tuple(setting.name for setting in pr.ENVIRONMENT)


@pytest.fixture(autouse=True)
def unset_settings(monkeypatch):
    for name in (VARIABLE, *GPU_VARIABLES):
        monkeypatch.delenv(name, raising=False)


def series(kind: str, n_steps: int, n_vars: int = 2, seed: int = 0) -> tuple[np.ndarray, int]:
    """A series that switches between two regimes in runs of 70 to 100 steps, and the number of
    switches. ``"sign"``: a link between two variables that changes sign; ``"wide"``: a chain of
    ``n_vars`` variables whose variable ranks differ from their positions."""
    rng = np.random.default_rng(seed)
    path: list[int] = []
    while len(path) < n_steps:
        path += [len(path) % 2 if not path else 1 - path[-1]] * int(rng.integers(70, 101))
    regimes = np.array(path[:n_steps])
    if kind == "wide":
        x = rng.standard_normal((n_steps, n_vars))
        for t in range(1, n_steps):
            sign = 1.0 if regimes[t] == 0 else -1.0
            x[t] += 0.3 * x[t - 1]
            for j in range(1, n_vars):
                x[t, j] += sign * 0.7 * x[t - 1, j - 1]
    else:
        x = rng.standard_normal((n_steps, 2))
        for t in range(n_steps):
            sign = 0.8 if regimes[t] == 0 else -0.8
            if t >= 1:
                x[t, 1] += sign * x[t - 1, 0] + 0.2 * x[t - 1, 1]
                x[t, 0] += 0.2 * x[t - 1, 0]
    return x, int((np.diff(regimes) != 0).sum())


# The cases: a series, its settings, and what it shows. Starts 2, 5, and 7 of "sign" tie on the
# best objective, so the earliest of them is the answer, and 5 of its 12 starts fail; "wide" has
# ties and failed starts too, and its variable ranks differ from their positions.
CASES = {
    "ties and failed starts": ("sign", 300, 2, dict(n_starts=12)),
    "ranks that differ from positions": ("wide", 300, 3, dict(n_starts=12)),
    "one iteration": ("sign", 300, 2, dict(n_starts=6, max_iter=1)),
    "fisherz and fdr": (
        "sign",
        300,
        2,
        dict(n_starts=6, indep_test="fisherz", fdr_method="fdr_bh"),
    ),
    "three regimes": ("sign", 600, 2, dict(n_starts=8, n_regimes=3)),
    "two lags": ("sign", 400, 2, dict(n_starts=6, tau_max=2)),
    "every start fails": ("sign", 60, 2, dict(n_starts=3, n_regimes=10)),
}


def _outcome(case: str, workers: int, **settings):
    kind, n_steps, n_vars, options = CASES[case]
    x, switches = series(kind, n_steps, n_vars)
    options = {"max_transitions": max(1, switches), "tau_max": 1, "seed": 0, **options, **settings}
    try:
        with andrey.config(n_jobs=workers):
            return rp.rpcmci(x, **options)
    except Exception as error:  # noqa: BLE001 - the error is the outcome to compare
        return error


def _same(a, b, where: str = "result") -> None:
    """Two results hold the same types, shapes, dtypes, flags, and bytes, field by field."""
    assert type(a) is type(b), where
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, where
        assert a.tobytes() == b.tobytes(), where
        assert a.flags.writeable == b.flags.writeable, where
        assert a.flags.c_contiguous == b.flags.c_contiguous, where
    elif dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            _same(getattr(a, field.name), getattr(b, field.name), f"{where}.{field.name}")
    elif isinstance(a, tuple | list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same(x, y, f"{where}[{i}]")
    elif isinstance(a, float):
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), where
    elif isinstance(a, BaseException):
        assert str(a) == str(b), where
    else:
        assert a == b, where


@pytest.fixture
def pools(monkeypatch):
    """The pools the fit starts (their keyword arguments), made by the real executor."""
    started: list[dict] = []
    real = pr.ProcessPoolExecutor
    monkeypatch.setattr(
        pr, "ProcessPoolExecutor", lambda **kwargs: started.append(kwargs) or real(**kwargs)
    )
    return started


@pytest.fixture
def worker_state(monkeypatch):
    """Restore what a worker's initializer changes when a test runs it in this process."""
    config = backend.config
    saved = (config._backend, config._n_jobs, config._tie_break)
    environ = {name: os.environ.get(name) for name in GPU_VARIABLES}
    blas = [(lib, lib.num_threads) for lib in threadpoolctl.ThreadpoolController().lib_controllers]
    monkeypatch.setattr(pr, "_WORKER_STARTS", None)
    yield
    if pr._WORKER_BLAS is not None:
        pr._WORKER_BLAS.restore_original_limits()
        monkeypatch.setattr(pr, "_WORKER_BLAS", None)
    for lib, n_threads in blas:  # a fit that starts the pool inside its one-BLAS-thread block
        lib.set_num_threads(n_threads)  # leaves the worker's own limit at one thread
    config._backend, config._n_jobs, config._tie_break = saved
    for name, value in environ.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class InProcessPool:
    """An executor that runs the initializer and the tasks in this process, a task when its result
    is first read, the last start first when ``reverse`` (workers that finish in reverse order)."""

    reverse = False
    broken = False
    created: list[InProcessPool]

    def __init__(self, *, max_workers, mp_context, initializer, initargs):
        self.max_workers, self.initargs = max_workers, initargs
        self.ran: list[int] = []
        self._calls: list[tuple[Future, object, tuple]] = []
        initializer(*initargs)
        self.created.append(self)

    def submit(self, fn, *args):
        future = _Lazy(self)
        self._calls.append((future, fn, args))
        return future

    def run(self) -> None:
        for future, fn, args in reversed(self._calls) if self.reverse else self._calls:
            if future.done():
                continue
            if self.broken:
                future.set_exception(BrokenProcessPool("a process was terminated abruptly"))
                continue
            self.ran.append(args[0])
            try:
                future.set_result(fn(*args))
            except BaseException as error:  # noqa: BLE001 - a task's error goes to its future
                future.set_exception(error)

    def shutdown(self, wait=True, cancel_futures=False):
        pass


class _Lazy(Future):
    def __init__(self, pool: InProcessPool):
        super().__init__()
        self._pool = pool

    def result(self, timeout=None):
        self._pool.run()
        return super().result(timeout)


@pytest.fixture
def in_process(monkeypatch, worker_state):
    """The pool replaced by an :class:`InProcessPool`; the test sets ``reverse`` or ``broken``."""
    InProcessPool.created = []
    InProcessPool.reverse = InProcessPool.broken = False
    monkeypatch.setattr(pr, "ProcessPoolExecutor", InProcessPool)
    return InProcessPool


# --- the starts of one process --------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
def test_pooled_starts_are_the_starts_of_one_process(case, pools, monkeypatch):
    """The result, with its failed starts, ties, and stopping rule, or the error of a fit whose
    starts all fail, is the one a fit in this process gives, field by field and byte by byte."""
    serial = _outcome(case, 1)
    assert not pools
    monkeypatch.setenv(VARIABLE, "0")
    pooled = _outcome(case, 3)
    assert [p["max_workers"] for p in pools] == [3]
    _same(pooled, serial)
    assert isinstance(serial, rp.RegimeFitError) is (case == "every start fails")


def test_the_cases_hold_what_they_claim():
    """The cases have failed starts, ties for the best objective, and ranks off the positions."""
    for case in ("ties and failed starts", "ranks that differ from positions"):
        kind, n_steps, n_vars, options = CASES[case]
        x, switches = series(kind, n_steps, n_vars)
        starts = rp.Starts(
            data=x,
            ranks=tie_ranks(x),
            seed=0,
            n_regimes=2,
            max_transitions=max(1, switches),
            switch_threshold=0.5,
            max_iter=20,
            tau_max=1,
            pcmci_args=dict(
                pc_alpha=0.2, alpha_level=0.01, fdr_method="none", indep_test="student_t"
            ),
        )
        runs = [rp.run_start(starts, a) for a in range(options["n_starts"])]
        keys = [round(r.objective / n_steps, 9) for r in runs if r is not None]
        assert any(r is None for r in runs)
        assert len(set(keys)) < len(keys)  # two starts tie for an objective
        assert bool((tie_ranks(x) != np.arange(n_vars)).any()) == (case != "ties and failed starts")


def test_a_tie_goes_to_the_earliest_start_whichever_worker_finishes_first(in_process, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    serial = _outcome("ties and failed starts", 1)
    assert serial.start == 2  # starts 2, 5, and 7 tie for the best objective
    in_process.reverse = True
    pooled = _outcome("ties and failed starts", 3)
    [pool] = in_process.created
    assert pool.ran == list(reversed(range(12)))  # the last start finished first
    _same(pooled, serial)


def test_failed_starts_count_the_same_whichever_worker_finishes_first(in_process, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    serial = _outcome("ranks that differ from positions", 1)
    in_process.reverse = True
    pooled = _outcome("ranks that differ from positions", 2)
    assert serial.n_failed > 0
    _same(pooled, serial)


def test_the_result_arrays_come_back_as_a_fit_in_this_process_makes_them(pools, monkeypatch):
    """Pooled results travel through pickle: their dtype, layout, and flags are those of a serial
    fit's, which :func:`_same` compares for every array of the result."""
    serial = _outcome("two lags", 1)
    monkeypatch.setenv(VARIABLE, "0")
    pooled = _outcome("two lags", 2)
    assert pools
    _same(pooled, serial)


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_the_layout_of_the_series_does_not_change_a_pooled_fit(layout, pools, monkeypatch):
    x, switches = series("sign", 300)
    data = {
        "C": np.ascontiguousarray(x),
        "F": np.asfortranarray(x),
        "strided": np.repeat(x, 2, axis=0)[::2],
    }[layout]
    options = dict(n_starts=6, max_transitions=max(1, switches), seed=0)
    with andrey.config(n_jobs=1):
        serial = rp.rpcmci(data, **options)
    monkeypatch.setenv(VARIABLE, "0")
    with andrey.config(n_jobs=2):
        pooled = rp.rpcmci(data, **options)
    assert pools
    _same(pooled, serial)


def test_a_facade_fit_gives_the_same_output_with_workers(pools, monkeypatch):
    """Labels, column order, and the metadata of the output are the same field by field."""
    x, switches = series("wide", 300, 3)
    frame = pd.DataFrame(x[:, [2, 0, 1]], columns=["c", "a", "b"])
    options = dict(n_starts=8, max_transitions=max(1, switches), seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with andrey.config(n_jobs=1):
            serial = andrey.rpcmci(frame, **options)
        monkeypatch.setenv(VARIABLE, "0")
        with andrey.config(n_jobs=3):
            pooled = andrey.rpcmci(frame, **options)
    assert pools
    _same(pooled, serial)
    assert dict(pooled.metadata) == dict(serial.metadata)


# --- errors ---------------------------------------------------------------------------------------


def test_an_error_in_a_start_is_the_error_of_one_process(pools, monkeypatch):
    """A test that does not exist fails inside every start, in a worker as in this process."""
    x, _ = series("sign", 300)

    def error(workers: int) -> Exception:
        with andrey.config(n_jobs=workers), pytest.raises(NotImplementedError) as caught:
            rp.rpcmci(x, n_starts=4, max_transitions=3, indep_test="not_a_test")
        return caught.value

    serial = error(1)
    assert not pools
    monkeypatch.setenv(VARIABLE, "0")
    pooled = error(2)
    assert pools
    # The message ends with the tests each process has registered, which tests may add to.
    assert type(pooled) is type(serial) is NotImplementedError
    assert (
        str(pooled).split(";")[0] == str(serial).split(";")[0] == "unknown indep_test 'not_a_test'"
    )


def test_the_error_of_the_first_start_in_order_is_raised(in_process, monkeypatch):
    """Starts 3 and 5 raise; the last finishes first, and the one of start 3 is raised, as in one
    process, where the search stops at start 3."""
    monkeypatch.setenv(VARIABLE, "0")

    def run_start(starts, a):
        if a in (3, 5):
            raise RuntimeError(f"start {a}")
        return None

    monkeypatch.setattr(rp, "run_start", run_start)
    x, _ = series("sign", 300)
    in_process.reverse = True
    with andrey.config(n_jobs=2), pytest.raises(RuntimeError, match="start 3$"):
        rp.rpcmci(x, n_starts=8, max_transitions=3)
    with andrey.config(n_jobs=1), pytest.raises(RuntimeError, match="start 3$"):
        rp.rpcmci(x, n_starts=8, max_transitions=3)


def test_a_dead_worker_raises_an_error_that_names_the_setting(in_process, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    in_process.broken = True
    x, _ = series("sign", 300)
    with andrey.config(n_jobs=2), pytest.raises(RuntimeError, match="ANDREY_N_JOBS=1"):
        rp.rpcmci(x, n_starts=4, max_transitions=3)


# --- what a worker holds --------------------------------------------------------------------------


def test_an_initialized_worker_holds_the_callers_settings(worker_state):
    starts = object()
    pr._init_worker(
        starts, "position", "numpy", ((GPU_VARIABLES[0], "123"), (GPU_VARIABLES[1], None))
    )
    assert pr._WORKER_STARTS is starts
    assert backend.config.n_jobs == 1
    assert backend.config.backend == "numpy"
    assert backend.config.tie_break == "position"
    assert os.environ[GPU_VARIABLES[0]] == "123" and GPU_VARIABLES[1] not in os.environ
    assert {pool["num_threads"] for pool in threadpoolctl.threadpool_info()} == {1}


@pytest.mark.parametrize("scope", ["attribute", "context"])
def test_the_pool_is_given_the_callers_tie_break_backend_and_environment(
    scope, in_process, monkeypatch
):
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setenv(GPU_VARIABLES[0], "77")
    x, _ = series("wide", 300, 3)
    options = dict(n_starts=3, max_transitions=3, n_regimes=2)
    if scope == "attribute":
        monkeypatch.setattr(backend.config, "_tie_break", "position")
        monkeypatch.setattr(backend.config, "_backend", "numpy")
        context = andrey.config(n_jobs=2)
    else:
        context = andrey.config(n_jobs=2, tie_break="position", backend="numpy")
    with context, contextlib.suppress(rp.RegimeFitError):
        rp.rpcmci(x, **options)
    [pool] = in_process.created
    _, tie_break, numeric_backend, environment = pool.initargs
    assert (tie_break, numeric_backend) == ("position", "numpy")
    assert dict(environment) == {GPU_VARIABLES[0]: "77", GPU_VARIABLES[1]: None}


@pytest.mark.parametrize("tie_break", ["position", "rank"])
def test_a_pooled_fit_follows_the_callers_tie_break(tie_break, pools, monkeypatch):
    kind, n_steps, n_vars, options = CASES["ranks that differ from positions"]
    x, switches = series(kind, n_steps, n_vars)
    options = dict(n_starts=8, max_transitions=max(1, switches), seed=0)
    with andrey.config(n_jobs=1, tie_break=tie_break):
        serial = rp.rpcmci(x, **options)
    monkeypatch.setenv(VARIABLE, "0")
    with andrey.config(n_jobs=2, tie_break=tie_break):
        pooled = rp.rpcmci(x, **options)
    assert pools
    _same(pooled, serial)


def test_a_start_reads_only_the_settings_a_worker_is_given(monkeypatch):
    """Every setting a fit reads is held by ``andrey.config`` or is an environment variable of the
    pool, so a worker the forkserver started earlier cannot read a stale one."""
    read: set[str] = set()
    real = env.Setting.read
    monkeypatch.setattr(env.Setting, "read", lambda self: read.add(self.name) or real(self))
    x, switches = series("sign", 300)
    for indep_test in ("student_t", "fisherz"):
        rp.rpcmci(x, n_starts=4, max_transitions=max(1, switches), indep_test=indep_test)
    held = {env.TIE_BREAK.name, env.BACKEND.name, env.DEVICE.name, env.N_JOBS.name}
    assert read - held <= set(GPU_VARIABLES)


# --- the gate -------------------------------------------------------------------------------------

N_STARTS, N_REGIMES, N_STEPS = 4, 2, 200
WORK = N_STARTS * N_REGIMES * N_STEPS


class Started(Exception):
    """Raised in place of starting a pool."""


@pytest.fixture
def pool_started(monkeypatch):
    """The pool replaced by one that records that it was asked for and stops the fit."""
    asked: list[int] = []

    def refuse(**kwargs):
        asked.append(kwargs["max_workers"])
        raise Started

    monkeypatch.setattr(pr, "ProcessPoolExecutor", refuse)
    return asked


def _gate(workers: int = 4, n_starts: int = N_STARTS) -> bool:
    """Whether a fit of :data:`WORK` on ``workers`` asks for a pool."""
    x, _ = series("sign", N_STEPS)
    try:
        with andrey.config(n_jobs=workers), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rp.rpcmci(x, n_starts=n_starts, max_transitions=3, seed=0)
    except Started:
        return True
    return False


@pytest.mark.parametrize(
    "cutoff, pooled", [(0, True), (WORK, True), (WORK + 1, False), (10**9, False)]
)
def test_a_fit_uses_the_pool_from_the_cutoff(cutoff, pooled, pool_started, monkeypatch):
    """The work is ``n_starts * n_regimes * n_steps``: a fit of exactly the cutoff's work uses the
    pool, and one step fewer does not."""
    monkeypatch.setenv(VARIABLE, str(cutoff))
    assert _gate() is pooled


def test_the_pool_has_a_worker_for_each_start_up_to_the_workers(pool_started, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    assert _gate(workers=8, n_starts=3)
    assert _gate(workers=2)
    assert pool_started == [3, 2]


def test_the_default_cutoff_keeps_a_small_fit_in_this_process(pool_started):
    assert 2 * 2 * 60 < pr.MIN_POOL_RPCMCI_WORK
    x, _ = series("sign", 60)
    with andrey.config(n_jobs=4), contextlib.suppress(rp.RegimeFitError):
        rp.rpcmci(x, n_starts=2, max_transitions=1)
    assert not pool_started


def _starts() -> rp.Starts:
    """The starts of a fit of :data:`WORK`: :data:`N_REGIMES` regimes over :data:`N_STEPS` steps."""
    x, _ = series("sign", N_STEPS)
    return rp.Starts(
        data=x,
        ranks=tie_ranks(x),
        seed=0,
        n_regimes=N_REGIMES,
        max_transitions=3,
        switch_threshold=0.5,
        max_iter=20,
        tau_max=1,
        pcmci_args=dict(pc_alpha=0.2, alpha_level=0.01, fdr_method="none", indep_test="student_t"),
    )


@pytest.mark.parametrize(
    "preference, live, pooled",
    [
        ("numpy", None, True),
        ("numpy", "cuda", True),
        ("auto", None, True),
        ("auto", "mps", True),
        ("auto", "cuda", False),
        ("cpu", None, True),
        ("cpu", "cuda", True),
        ("cuda", None, False),
        ("mps", None, False),
    ],
)
def test_a_fit_that_can_use_a_gpu_runs_its_starts_in_this_process(
    preference, live, pooled, monkeypatch
):
    """``auto`` takes only CUDA, and only once its runtime is live, and ``cuda`` and ``mps`` are
    pinned devices; a ``numpy`` or ``cpu`` fit takes the pool whatever runtime is live."""
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setattr(backend, "_is_live", lambda device: device == live)
    with andrey.config(n_jobs=4, backend=preference):
        runs = pr.pooled_starts(_starts(), N_STARTS)
    assert (runs is not None) is pooled


def test_a_fit_with_a_live_cuda_runtime_gives_the_result_of_a_fit_in_this_process(
    pools, monkeypatch
):
    monkeypatch.setenv(VARIABLE, "0")
    with andrey.config(backend="auto"):
        serial = _outcome("two lags", 1)
        monkeypatch.setattr(backend, "_is_live", lambda device: device == "cuda")
        with_runtime = _outcome("two lags", 3)
    assert not pools
    _same(with_runtime, serial)


def test_a_single_worker_or_a_single_start_never_uses_the_pool(pool_started, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    assert not _gate(workers=1)
    assert not _gate(workers=0)
    assert not _gate(workers=4, n_starts=1)
    assert not pool_started


# --- the reduction -------------------------------------------------------------------------------


class _Start:
    """A stand-in for a start's result that a weak reference can follow."""


class FinishedPool:
    """An executor whose tasks are done at once, each result held by its future alone."""

    made: list[FinishedPool]
    fail_submit = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.refs: list[weakref.ref] = []
        self.shut_down: list[dict] = []
        self.made.append(self)

    def submit(self, fn, a):
        if self.fail_submit:
            raise RuntimeError("cannot submit")
        result = _Start()
        self.refs.append(weakref.ref(result))
        future = Future()
        future.set_result(result)
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shut_down.append({"wait": wait, "cancel_futures": cancel_futures})


@pytest.fixture
def finished(monkeypatch):
    FinishedPool.made, FinishedPool.fail_submit = [], False
    monkeypatch.setenv(VARIABLE, "0")
    monkeypatch.setattr(pr, "ProcessPoolExecutor", FinishedPool)
    return FinishedPool


def _pooled(n_starts: int, **settings):
    """The iterator over ``n_starts`` pooled starts, made under ``settings`` and read after them."""
    with andrey.config(n_jobs=2, **settings):
        runs = pr.pooled_starts(_starts(), n_starts)
    assert runs is not None
    return runs


def test_the_pool_starts_when_the_first_start_is_read_with_the_settings_of_the_call(finished):
    runs = _pooled(3, tie_break="position", backend="numpy")
    assert not finished.made  # a fit that never reads its starts starts no pool
    next(runs)
    [pool] = finished.made
    assert pool.kwargs["max_workers"] == 2
    assert pool.kwargs["initargs"][1:3] == ("position", "numpy")
    assert backend.config.tie_break != "position" or backend.config.backend != "numpy"


def test_a_start_the_fit_has_read_is_not_kept(finished):
    """The pool's results are read one at a time: a start read and let go is freed, and the ones
    not yet read wait in their futures."""
    runs = _pooled(4)
    assert not isinstance(runs, list | tuple) and iter(runs) is runs
    first = next(runs)
    [pool] = finished.made
    assert pool.refs[0]() is first
    del first
    assert pool.refs[0]() is None
    assert all(ref() is not None for ref in pool.refs[1:])
    next(runs)
    assert pool.refs[1]() is None and all(ref() is not None for ref in pool.refs[2:])


def test_the_pool_shuts_down_after_the_last_start_is_read(finished):
    runs = _pooled(3)
    assert len(list(runs)) == 3
    [pool] = finished.made
    assert len(pool.refs) == 3
    assert pool.shut_down == [{"wait": True, "cancel_futures": True}]


def test_the_pool_shuts_down_when_a_fit_stops_reading(finished):
    runs = _pooled(3)
    next(runs)
    [pool] = finished.made
    assert not pool.shut_down
    runs.close()
    assert pool.shut_down == [{"wait": True, "cancel_futures": True}]


def test_the_pool_shuts_down_when_a_task_cannot_be_submitted(finished):
    finished.fail_submit = True
    runs = _pooled(3)
    with pytest.raises(RuntimeError, match="cannot submit"):
        next(runs)
    [pool] = finished.made
    assert pool.shut_down == [{"wait": True, "cancel_futures": True}]


# --- the first-use warning ------------------------------------------------------------------------


def _fit_warnings(workers: int = 2) -> list[warnings.WarningMessage]:
    """The performance warnings of two fits of :data:`WORK` below the cutoff."""
    x, _ = series("sign", N_STEPS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with andrey.config(n_jobs=workers):
            for _ in range(2):
                rp.rpcmci(x, n_starts=N_STARTS, max_transitions=3, seed=0)
    return [w for w in caught if w.category is andrey.PerformanceWarning]


def test_the_default_cutoff_warns_once_naming_the_setting_and_command(monkeypatch):
    monkeypatch.setattr(pr, "MIN_POOL_RPCMCI_WORK", 10 * WORK)  # the fit reaches a tenth of it
    [warning] = _fit_warnings()  # two fits, one warning
    assert f"{VARIABLE}={10 * WORK}" in str(warning.message)
    assert "`andrey calibrate`" in str(warning.message)
    assert "Regime-PCMCI" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize("factor, warns", [(10, True), (20, False)])
def test_the_default_cutoff_warns_once_a_fit_reaches_a_tenth_of_it(factor, warns, monkeypatch):
    monkeypatch.setattr(pr, "MIN_POOL_RPCMCI_WORK", factor * WORK)
    assert bool(_fit_warnings()) is warns


@pytest.mark.parametrize("value", [WORK + 1, 10 * WORK])
def test_an_explicit_cutoff_never_warns(value, monkeypatch):
    """Any set value counts, including one equal to the built-in default (the last)."""
    monkeypatch.setattr(pr, "MIN_POOL_RPCMCI_WORK", 10 * WORK)
    monkeypatch.setenv(VARIABLE, str(value))
    assert not _fit_warnings()


def test_a_serial_fit_never_warns(monkeypatch):
    monkeypatch.setattr(pr, "MIN_POOL_RPCMCI_WORK", 10 * WORK)
    assert not _fit_warnings(workers=1)


def test_a_bad_cutoff_is_rejected_when_a_fit_reads_it(monkeypatch):
    monkeypatch.setenv(VARIABLE, "many")
    x, _ = series("sign", N_STEPS)
    with andrey.config(n_jobs=2), pytest.raises(ValueError, match=VARIABLE):
        rp.rpcmci(x, n_starts=N_STARTS, max_transitions=3)


# --- the forkserver and the calibrator ------------------------------------------------------------


def test_the_forkserver_preloads_the_modules_a_worker_runs():
    assert {"andrey.temporal._parallel_rpcmci", "andrey.temporal.rpcmci"} <= set(
        par._FORKSERVER_PRELOAD
    )


def test_the_rpcmci_calibrator_reads_the_built_in_cutoff(monkeypatch):
    method = cal.METHODS["rpcmci"]
    assert method.variable == "ANDREY_RPCMCI_PARALLEL_MIN_WORK" and method.at_cutoff
    assert method.default() == pr.MIN_POOL_RPCMCI_WORK
    monkeypatch.setattr(pr, "MIN_POOL_RPCMCI_WORK", 123)
    assert method.default() == 123


def _once(fn, repeats):
    """One run of ``fn``, standing in for the calibrator's median of several."""
    import time

    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_the_rpcmci_timers_fit_serially_and_use_the_pool_only_in_the_pooled_arm(pools, monkeypatch):
    monkeypatch.setattr(cal, "_time", _once)
    serial, pooled, work = cal.rpcmci_timers(workers=2)
    assert serial(2) > 0 and not pools
    assert pooled(2) > 0 and len(pools) == 1
    assert work(4) == 4 * cal.RPCMCI_REGIMES * cal.RPCMCI_ROWS


def test_the_rpcmci_timers_run_on_numpy_whatever_the_process_prefers(pools, monkeypatch):
    """A process pinned to a device would run both arms in this process, and time nothing."""
    monkeypatch.setattr(cal, "_time", _once)
    _, pooled, _ = cal.rpcmci_timers(workers=2)
    with andrey.config(backend="cuda"):
        assert pooled(2) > 0
    assert len(pools) == 1


def test_the_rpcmci_calibrator_measures_in_starts_and_counts_work_as_the_gate_does(
    pools, monkeypatch
):
    monkeypatch.setattr(cal, "_time", _once)
    tiny = dataclasses.replace(cal.METHODS["rpcmci"], sizes=((2, 3), 2, 4))
    monkeypatch.setitem(cal.METHODS, "rpcmci", tiny)
    c = cal.calibrate(2, ["rpcmci"])["rpcmci"]
    assert c.method == "rpcmci" and c.cutoff > 0 and c.start_s is not None
    assert all(c.work(d) == d * cal.RPCMCI_REGIMES * cal.RPCMCI_ROWS for d, _, _ in c.timings)
    assert 2 <= min(d for d, _, _ in c.timings) and max(d for d, _, _ in c.timings) <= 4


def test_the_rpcmci_probe_series_gives_a_fit_that_succeeds_at_every_probe_size():
    """The smallest probe fit raises when every start fails, so the probe must not."""
    x = cal.regime_sample()
    assert x.shape == (cal.RPCMCI_ROWS, 2)
    for d in (2, 3, 4):
        out = andrey.rpcmci(
            x, n_starts=d, max_transitions=cal.RPCMCI_ROWS // cal.RPCMCI_REGIME_STEPS + 1, seed=0
        )
        assert out.n_regimes == 2


def test_calibrate_command_measures_the_rpcmci_cutoff_in_starts(capsys, monkeypatch):
    import json

    def fake(workers, methods):
        timings, works = ((8, 0.1, 0.2), (16, 0.5, 0.2)), ((8, 8_000), (16, 16_000))
        return {m: cal.Crossover(8, 16, timings, works, m) for m in methods}

    monkeypatch.setattr(cal, "calibrate", fake)
    assert main(["calibrate", "--workers", "4", "--method", "rpcmci", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["settings"] == {"ANDREY_N_JOBS": "4", VARIABLE: "11000"}
    assert report["methods"]["rpcmci"]["pool_faster_at"] == {"d": 16, "work": 16_000}
