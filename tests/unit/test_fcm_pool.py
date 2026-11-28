"""The regressions of RESIT and of the exact search of CPCM on the worker pool.

A pooled fit gives the numbers and the output one process gives, bit for bit, uses the pool only
from its cutoff, and warns once when it runs on a default cutoff near it. The pool is spied on in
every test that claims it ran.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info

import andrey
from andrey.cli import main
from andrey.core.backend import one_blas_thread
from andrey.core.knowledge import Knowledge
from andrey.fcm import _parallel as par
from andrey.fcm import cpcm
from andrey.fcm import resit as resit_module
from andrey.fcm.cpcm import FAMILIES, Residuals, search
from andrey.fcm.gam import BasisCache
from andrey.fcm.resit import causal_order, remove_edges, resit
from andrey.search import calibrate as cal

RESIT = cal.METHODS["resit"].variable
CPCM = cal.METHODS["cpcm"].variable
FAMILY = "gaussian"


@pytest.fixture(autouse=True)
def unset_cutoffs(monkeypatch):
    monkeypatch.delenv(RESIT, raising=False)
    monkeypatch.delenv(CPCM, raising=False)


@pytest.fixture
def pools(monkeypatch):
    """The pools the fcm workers start."""
    started: list[int] = []
    real = par.ProcessPoolExecutor
    monkeypatch.setattr(
        par,
        "ProcessPoolExecutor",
        lambda *a, **k: started.append(k["max_workers"]) or real(*a, **k),
    )
    return started


def _nonlinear(d: int, n: int, seed: int = 0) -> np.ndarray:
    """A chain with a collider and nonlinear effects of uniform noise; ``n`` rows, ``d`` columns."""
    rng = np.random.default_rng(seed)
    x = [rng.uniform(-2, 2, n)]
    for j in range(1, d):
        parents = x[-1] + (x[j - 2] if j > 1 and j % 2 == 0 else 0.0)
        x.append(np.sin(1.5 * parents) + rng.uniform(-0.6, 0.6, n))
    return np.column_stack(x)


def _location_scale(d: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = [rng.normal(0.0, 1.4, n)]
    for j in range(1, d):
        p = x[-1]
        x.append(np.sin(2 * p) + p + (0.4 + 0.3 * np.tanh(p)) * rng.normal(size=n))
    return np.column_stack(x)


def _same_output(a: andrey.StructureOutput, b: andrey.StructureOutput) -> None:
    np.testing.assert_array_equal(a.structure.to_numpy(), b.structure.to_numpy())
    assert a.ordering == b.ordering
    assert dict(a.metadata) == dict(b.metadata)  # floats compare equal bit for bit


def _fit(method, x, workers: int, **kwargs):
    with warnings.catch_warnings(), andrey.config(n_jobs=workers):
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return method(x, **kwargs)


# --- RESIT ---------------------------------------------------------------------------------------


def _resit_parts(X: np.ndarray):
    """What RESIT's regressions read: bases, kernel columns, and the edge matrices."""
    d = X.shape[1]
    spread = X.std(axis=0)
    unit = (X - X.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    return BasisCache(X), unit, ~np.eye(d, dtype=bool), np.zeros((d, d), dtype=bool)


@pytest.mark.parametrize("d", [3, 5])
def test_pooled_sink_tests_and_edge_removals_are_the_numbers_of_one_process(d, pools, monkeypatch):
    """Every log p-value of the first step, and of each removal test, equals the one the loops of
    this process compute, bit for bit, so no rounding hides behind an unchanged graph."""
    X = _nonlinear(d, 150)
    bases, unit, allowed, required = _resit_parts(X)
    remaining = list(range(d))
    order = np.arange(d)
    recorded: list[float] = []
    real = resit_module.hsic_log_p

    def record(*args, **kwargs):
        recorded.append(real(*args, **kwargs))
        return recorded[-1]

    monkeypatch.setattr(resit_module, "hsic_log_p", record)
    with one_blas_thread():
        causal_order(bases, unit, allowed)
        serial_p = recorded[:d]  # the first step tests every candidate against the others
        recorded.clear()
        remove_edges(bases, unit, order, allowed=allowed, required=required)
        serial_tests = list(recorded)
        with par.ResitPool(X, unit, allowed, required, workers=3) as pool:
            pooled_p = pool.sink_log_ps(remaining, remaining)
            pooled_kept = pool.kept_parents(order, resit_module.ALPHA)
    assert pools == [3]  # the test's own pool
    assert pooled_p == serial_p
    assert [log_p for _, tests in pooled_kept for _, log_p in tests] == serial_tests


def test_a_pooled_fit_gives_the_numbers_of_the_serial_fit(pools, monkeypatch):
    """The core fit: DAG and order of a pooled fit equal the serial ones, with a pool started."""
    monkeypatch.setenv(RESIT, "0")
    X = _nonlinear(5, 150)
    with one_blas_thread(), andrey.config(n_jobs=1):
        serial = resit(X)
    with one_blas_thread(), andrey.config(n_jobs=3):
        pooled = resit(X)
    assert pools == [3]
    np.testing.assert_array_equal(pooled.dag, serial.dag)
    np.testing.assert_array_equal(pooled.order, serial.order)


@pytest.mark.parametrize("prep", ["plain", "labels", "knowledge"])
def test_a_facade_resit_gives_the_same_output_with_workers(prep, pools, monkeypatch):
    monkeypatch.setenv(RESIT, "0")
    X = _nonlinear(4, 120)
    kwargs = {}
    data: object = X
    if prep != "plain":
        names = ["a", "b", "c", "d"]
        data = pd.DataFrame(X, columns=names)
    if prep == "knowledge":
        kwargs["knowledge"] = Knowledge(
            required=[("a", "b")], forbidden=[("c", "a")], tiers=[["d"], ["a", "b", "c"]]
        )
    serial = _fit(andrey.resit, data, 1, **kwargs)
    assert pools == []
    pooled = _fit(andrey.resit, data, 3, **kwargs)
    assert pools == [3]
    _same_output(pooled, serial)


def test_a_pooled_fit_logs_what_a_serial_fit_logs(pools, monkeypatch, caplog):
    monkeypatch.setenv(RESIT, "0")
    X = _nonlinear(4, 120)

    def messages(workers: int) -> list[str]:
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="andrey.fcm.resit"):
            _fit(andrey.resit, X, workers)
        return [r.getMessage() for r in caplog.records if r.name == "andrey.fcm.resit"]

    serial = messages(1)
    assert serial and pools == []
    assert messages(3) == serial and pools == [3]


def test_a_failing_regression_raises_the_first_error_in_the_serial_order(pools):
    """Column 1 has two values and column 3 one, so a regression that reads either as a parent
    fails with a message of its own. Candidate 0 meets column 1 first and candidate 2 meets column
    3; the pool raises candidate 0's error, as the loop does, however the workers finish."""
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.normal(size=100), rng.integers(0, 2, 100), rng.normal(size=100), np.ones(100)]
    )
    X = np.asfortranarray(X)
    bases, unit, allowed, required = _resit_parts(X)
    allowed[3, 0] = False  # candidate 0: parent column 1 first
    allowed[1, 2] = False  # candidate 2: parent column 3
    remaining, candidates = [0, 1, 2, 3], [0, 1, 2, 3]
    with one_blas_thread(), pytest.raises(ValueError) as serial:
        causal_order(bases, unit, allowed)
    with one_blas_thread(), par.ResitPool(X, unit, allowed, required, workers=3) as pool:
        with pytest.raises(ValueError) as pooled:
            pool.sink_log_ps(remaining, candidates)
    assert str(pooled.value) == str(serial.value) and "column 1" in str(pooled.value)


def test_the_cutoff_is_the_regressions_of_the_causal_order_times_rows(pools, monkeypatch):
    """``d = 4`` has 4 + 3 + 2 = 9 regressions in its causal order: the work of 100 rows is 900, and
    a fit of exactly the cutoff's work uses the pool while one unit fewer does not."""
    X = _nonlinear(4, 100)
    serial = _fit(andrey.resit, X, 1)
    monkeypatch.setenv(RESIT, "900")
    pooled = _fit(andrey.resit, X, 3)
    assert pools == [3]
    monkeypatch.setenv(RESIT, "901")
    below = _fit(andrey.resit, X, 3)
    assert pools == [3]
    _same_output(pooled, serial)
    _same_output(below, serial)


def test_two_variables_never_use_the_pool(pools, monkeypatch):
    monkeypatch.setenv(RESIT, "0")
    _fit(andrey.resit, _nonlinear(2, 100), 3)
    assert pools == []


# --- CPCM ----------------------------------------------------------------------------------------


def _residuals(X: np.ndarray, family: str = FAMILY, seed: int = 0) -> Residuals:
    return Residuals(X, [FAMILIES[family]] * X.shape[1], seed, method="cpcm_graph")


@pytest.mark.parametrize("d", [3, 4])
def test_the_prefit_residuals_are_the_residuals_of_one_process(d, pools, monkeypatch):
    """Each stored residual, a fit of one variable on one set of parents, equals the one this
    process computes, bit for bit, for every variable and every nonempty set."""
    monkeypatch.setenv(CPCM, "0")
    X = _location_scale(d, 60)
    pooled = _residuals(X)
    with one_blas_thread(), andrey.config(n_jobs=3):
        par.prefit(pooled)
    assert pools == [3]
    assert len(pooled._residuals) == d * (2 ** (d - 1) - 1)
    serial = _residuals(X)
    with one_blas_thread():
        for (j, parents), residual in pooled._residuals.items():
            assert np.array_equal(residual, serial.residual(j, parents))
            assert residual.tobytes() == serial.residual(j, parents).tobytes()


@pytest.mark.parametrize("d, edge_penalty", [(3, 2.0), (4, 2.0), (3, 0.0)])
def test_the_exact_search_gives_the_same_numbers_with_workers(d, edge_penalty, pools, monkeypatch):
    """The DAG, the log p-value, and the score of the search are the one process's, bit for bit."""
    monkeypatch.setenv(CPCM, "0")
    X = _location_scale(d, 60)
    with one_blas_thread(), andrey.config(n_jobs=1):
        base = _residuals(X)
        serial = search(base, edge_penalty)
        serial_score = cpcm.score(base, serial[0], edge_penalty)
    assert pools == []
    with one_blas_thread(), andrey.config(n_jobs=3):
        res = _residuals(X)
        pooled = search(res, edge_penalty)
        pooled_score = cpcm.score(res, pooled[0], edge_penalty)
    assert pools == [3]
    assert pooled == serial
    assert pooled_score == serial_score


@pytest.mark.parametrize("prep", ["plain", "labels"])
def test_a_facade_cpcm_graph_gives_the_same_output_with_workers(prep, pools, monkeypatch):
    monkeypatch.setenv(CPCM, "0")
    X = _location_scale(3, 60)
    data = X if prep == "plain" else pd.DataFrame(X, columns=["u", "v", "w"])
    serial = _fit(andrey.cpcm_graph, data, 1, family=FAMILY, seed=0)
    assert pools == []
    pooled = _fit(andrey.cpcm_graph, data, 3, family=FAMILY, seed=0)
    assert pools == [3]
    _same_output(pooled, serial)


def test_only_the_columns_whose_scale_changes_with_the_parents_are_fitted_on_workers(
    pools, monkeypatch
):
    """The fits of a count column, whose randomized residual reads uniforms drawn from the seed
    and the column, stay in this process; the other columns' go to the pool, and the output is the
    serial one."""
    monkeypatch.setenv(CPCM, "0")
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    X = np.column_stack(
        [x, rng.poisson(np.exp(0.5 * np.tanh(x))).astype(float), rng.normal(size=80)]
    )
    family = ["gaussian", "poisson", "gaussian"]
    serial = _fit(andrey.cpcm_graph, X, 1, family=family, seed=3)
    pooled = _fit(andrey.cpcm_graph, X, 3, family=family, seed=3)
    assert pools == [3]
    _same_output(pooled, serial)
    res = Residuals(X, [FAMILIES[name] for name in family], 3)
    with andrey.config(n_jobs=3):
        par.prefit(res)
    assert {j for j, _ in res._residuals} == {0, 2}
    assert len(res._residuals) == 2 * 3


@pytest.mark.parametrize(
    "family", ["gaussian_fixed_scale", "poisson", "exponential", "negative_binomial"]
)
def test_a_family_with_a_fixed_scale_or_one_parameter_stays_in_this_process(
    family, pools, monkeypatch
):
    monkeypatch.setenv(CPCM, "0")
    rng = np.random.default_rng(2)
    X = rng.poisson(3.0, size=(60, 3)).astype(float) + 1.0
    res = Residuals(X, [FAMILIES[family]] * 3, 0)
    with andrey.config(n_jobs=3):
        par.prefit(res)
    assert pools == [] and not res._residuals


def test_a_failing_fit_raises_its_serial_error_from_the_search(pools, monkeypatch):
    """A column that is an exact function of another has no fit. The worker leaves it out, and
    the search, in this process, raises the error and at the DAG it raises without workers."""
    monkeypatch.setenv(CPCM, "0")
    X = _location_scale(3, 60)
    X[:, 2] = 2.0 * X[:, 0] + 1.0
    with pytest.raises(ValueError) as serial:
        _fit(andrey.cpcm_graph, X, 1, family=FAMILY, seed=0)
    assert pools == []
    with pytest.raises(ValueError) as pooled:
        _fit(andrey.cpcm_graph, X, 3, family=FAMILY, seed=0)
    assert pools == [3]
    assert str(pooled.value) == str(serial.value)


def test_the_cutoff_is_the_fits_of_the_exact_search_times_rows(pools, monkeypatch):
    """Three variables have 3 * (2 ** 2 - 1) = 9 fits: the work of 60 rows is 540, and a fit of
    exactly the cutoff's work uses the pool while one unit fewer does not."""
    X = _location_scale(3, 60)
    serial = _fit(andrey.cpcm_graph, X, 1, family=FAMILY, seed=0)
    monkeypatch.setenv(CPCM, "540")
    pooled = _fit(andrey.cpcm_graph, X, 3, family=FAMILY, seed=0)
    assert pools == [3]
    monkeypatch.setenv(CPCM, "541")
    below = _fit(andrey.cpcm_graph, X, 3, family=FAMILY, seed=0)
    assert pools == [3]
    _same_output(pooled, serial)
    _same_output(below, serial)


def test_two_variables_never_prefit(pools, monkeypatch):
    monkeypatch.setenv(CPCM, "0")
    _fit(andrey.cpcm_graph, _location_scale(2, 60), 3, family=FAMILY, seed=0)
    assert pools == []


def test_the_search_above_four_variables_never_prefits(pools, monkeypatch):
    monkeypatch.setenv(CPCM, "0")
    _fit(andrey.cpcm_graph, _location_scale(5, 60), 3, family=FAMILY, seed=0)
    assert pools == []


def test_a_family_a_worker_cannot_name_stays_in_this_process(pools, monkeypatch):
    """A copy of a family is not the one the worker finds by name, so the fit stays here."""
    monkeypatch.setenv(CPCM, "0")
    X = _location_scale(3, 60)
    res = Residuals(X, [dataclasses.replace(FAMILIES[FAMILY])] * 3, 0)
    with andrey.config(n_jobs=3):
        par.prefit(res)
    assert pools == [] and not res._residuals


@pytest.mark.parametrize("order", ["C", "F", "view"])
def test_a_table_keeps_its_layout_on_the_way_or_stays_in_this_process(order, pools, monkeypatch):
    """Pickling keeps a C or Fortran contiguous array's layout; any other array would arrive in C
    order, so its fit stays here."""
    monkeypatch.setenv(CPCM, "0")
    wide = np.asarray(_location_scale(3, 120))
    X = {
        "C": np.ascontiguousarray(wide[:60]),
        "F": np.asfortranarray(wide[:60]),
        "view": wide[::2],
    }[order]
    assert X.flags.c_contiguous or X.flags.f_contiguous or order == "view"
    res = _residuals(X)
    with one_blas_thread(), andrey.config(n_jobs=3):
        par.prefit(res)
    assert bool(pools) is (order != "view")


# --- both ----------------------------------------------------------------------------------------


def test_a_single_worker_never_uses_the_pool(pools, monkeypatch):
    monkeypatch.setenv(RESIT, "0")
    monkeypatch.setenv(CPCM, "0")
    _fit(andrey.resit, _nonlinear(3, 100), 1)
    _fit(andrey.cpcm_graph, _location_scale(3, 60), 1, family=FAMILY, seed=0)
    assert pools == []


def test_the_default_cutoffs_keep_small_fits_in_this_process(pools):
    _fit(andrey.resit, _nonlinear(4, 100), 3)
    _fit(andrey.cpcm_graph, _location_scale(3, 60), 3, family=FAMILY, seed=0)
    assert pools == [] and par.MIN_POOL_RESIT_WORK > 900 and par.MIN_POOL_CPCM_WORK > 540


def test_a_worker_holds_one_blas_thread():
    """The pool's initializer limits every BLAS and OpenMP pool of the worker to one thread, as the
    facade does in the calling process, so a result is the one this process computes."""
    with par._pool(1, par._pin, ()) as pool:
        info = pool.submit(threadpool_info).result()
    assert info and {library["num_threads"] for library in info} == {1}


def test_a_pooled_fit_runs_on_workers_started_by_spawn(pools, monkeypatch):
    """Windows starts workers by spawn: the tasks and initializers pickle by name."""
    monkeypatch.setenv(RESIT, "0")
    monkeypatch.setenv(CPCM, "0")
    monkeypatch.setattr("andrey.search._parallel.pick_context", lambda: mp.get_context("spawn"))
    X = _nonlinear(3, 80)
    serial = _fit(andrey.resit, X, 1)
    _same_output(_fit(andrey.resit, X, 2), serial)
    Y = _location_scale(3, 50)
    serial = _fit(andrey.cpcm_graph, Y, 1, family=FAMILY, seed=0)
    _same_output(_fit(andrey.cpcm_graph, Y, 2, family=FAMILY, seed=0), serial)
    assert pools == [2, 2]


# --- the cutoffs' warnings and the calibrator -----------------------------------------------------

D, N = 4, 100
RESIT_WORK = 9 * N
CPCM_ROWS = 60
CPCM_WORK = 9 * CPCM_ROWS


def _warnings(method: str, workers: int = 2) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            if method == "resit":
                _fit(andrey.resit, _nonlinear(D, N), workers)
            else:
                _fit(
                    andrey.cpcm_graph, _location_scale(3, CPCM_ROWS), workers, family=FAMILY, seed=0
                )
    return [w for w in caught if w.category is andrey.PerformanceWarning]


@pytest.fixture
def default_cutoffs(monkeypatch):
    """The built-in cutoffs, set so that the fits above reach a tenth of them."""

    def set_(resit_factor: int, cpcm_factor: int) -> None:
        monkeypatch.setattr(par, "MIN_POOL_RESIT_WORK", resit_factor * RESIT_WORK)
        monkeypatch.setattr(par, "MIN_POOL_CPCM_WORK", cpcm_factor * CPCM_WORK)

    return set_


@pytest.mark.parametrize(
    "method, variable, work", [("resit", RESIT, RESIT_WORK), ("cpcm", CPCM, CPCM_WORK)]
)
def test_the_default_cutoff_warns_once_naming_the_setting_and_command(
    method, variable, work, default_cutoffs
):
    default_cutoffs(10, 10)  # the fit reaches a tenth of it
    [warning] = _warnings(method)  # two fits, one warning
    assert f"{variable}={10 * work}" in str(warning.message)
    assert "`andrey calibrate`" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize("method", ["resit", "cpcm"])
@pytest.mark.parametrize("factor, warns", [(10, True), (20, False)])
def test_the_default_cutoff_warns_once_a_fit_reaches_a_tenth_of_it(
    method, factor, warns, default_cutoffs
):
    default_cutoffs(factor, factor)
    assert bool(_warnings(method)) is warns


@pytest.mark.parametrize(
    "method, variable, work", [("resit", RESIT, RESIT_WORK), ("cpcm", CPCM, CPCM_WORK)]
)
@pytest.mark.parametrize("factor", [0, 1, 10])
def test_an_explicit_cutoff_never_warns(
    method, variable, work, factor, default_cutoffs, monkeypatch
):
    """Any set value counts, including one equal to the built-in default (the last)."""
    default_cutoffs(10, 10)
    monkeypatch.setenv(variable, str(factor * work))
    assert not _warnings(method)


@pytest.mark.parametrize("method", ["resit", "cpcm"])
def test_a_serial_fit_never_warns(method, default_cutoffs):
    default_cutoffs(10, 10)
    assert not _warnings(method, workers=1)


@pytest.mark.parametrize("variable, method", [(RESIT, "resit"), (CPCM, "cpcm")])
def test_a_bad_cutoff_is_rejected_when_a_fit_reads_it(variable, method, monkeypatch):
    monkeypatch.setenv(variable, "many")
    with pytest.raises(ValueError, match=variable):
        if method == "resit":
            _fit(andrey.resit, _nonlinear(3, 60), 2)
        else:
            _fit(andrey.cpcm_graph, _location_scale(3, 60), 2, family=FAMILY, seed=0)


def _once(fn, repeats):
    """One timed run of ``fn``, standing in for the calibrator's median of several."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def test_the_calibrator_counts_the_regressions_the_gate_counts():
    assert [cal.resit_regressions(d) for d in (3, 4, 10)] == [5, 9, 54]
    assert [cal.cpcm_fits(d) for d in (2, 3, 4)] == [2, 9, 28]


@pytest.mark.parametrize(
    "name, variable, default",
    [("resit", RESIT, "MIN_POOL_RESIT_WORK"), ("cpcm", CPCM, "MIN_POOL_CPCM_WORK")],
)
def test_the_calibrator_reads_the_built_in_cutoff(name, variable, default, monkeypatch):
    method = cal.METHODS[name]
    assert method.variable == variable and method.at_cutoff
    assert method.default() == getattr(par, default)
    monkeypatch.setattr(par, default, 123)
    assert method.default() == 123


@pytest.mark.parametrize(
    "name, fit, variable", [("resit", "resit", RESIT), ("cpcm", "cpcm_graph", CPCM)]
)
def test_the_timers_fit_serially_and_use_the_pool_only_in_the_pooled_arm(
    name, fit, variable, pools, monkeypatch
):
    monkeypatch.setattr(cal, "_time", _once)
    fits = cal.resit_regressions if name == "resit" else cal.cpcm_fits
    serial, pooled, work = cal.regression_timers(fit, 2, 60, fits)
    assert serial(3) > 0 and pools == []
    assert pooled(3) > 0 and pools == [2]
    assert work(3) == fits(3) * 60


@pytest.mark.parametrize("name, sizes", [("resit", ((4, 5), 3, 6)), ("cpcm", ((3,), 3, 3))])
def test_the_calibrator_measures_a_cutoff_in_regressions_times_rows(name, sizes, monkeypatch):
    """Run on small fits: one timed run each."""
    monkeypatch.setattr(cal, "_time", _once)
    monkeypatch.setattr(cal, "RESIT_ROWS", 60)
    monkeypatch.setattr(cal, "CPCM_ROWS", 40)
    monkeypatch.setitem(cal.METHODS, name, dataclasses.replace(cal.METHODS[name], sizes=sizes))
    c = cal.calibrate(2, [name])[name]
    rows = 60 if name == "resit" else 40
    fits = cal.resit_regressions if name == "resit" else cal.cpcm_fits
    assert c.method == name and c.cutoff > 0 and c.start_s is not None
    assert all(c.work(d) == fits(d) * rows for d, _, _ in c.timings)
    assert sizes[1] <= min(d for d, _, _ in c.timings)
    assert max(d for d, _, _ in c.timings) <= sizes[2]


@pytest.mark.parametrize("name, variable", [("resit", RESIT), ("cpcm", CPCM)])
def test_calibrate_command_measures_the_cutoff_in_regressions_times_rows(
    name, variable, capsys, monkeypatch
):
    def fake(workers, methods):
        timings, works = ((3, 0.1, 0.2), (4, 0.5, 0.2)), ((3, 1_000), (4, 4_000))
        return {m: cal.Crossover(3, 4, timings, works, m) for m in methods}

    monkeypatch.setattr(cal, "calibrate", fake)
    assert main(["calibrate", "--workers", "4", "--method", name, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["settings"] == {"ANDREY_N_JOBS": "4", variable: "2000"}
    assert report["methods"][name]["pool_faster_at"] == {"d": 4, "work": 4_000}
