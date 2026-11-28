"""The runs of a full-graph fit of IAMB, Inter-IAMB, and MMPC on the worker pool.

A pooled fit gives the sets and bits one process gives, uses the pool only from its cutoff, and
warns once when it runs on the default cutoff near it. The pool is spied on in every test that
claims it ran.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.cli import main
from andrey.constraint import _parallel_targets as tg
from andrey.constraint.blanket import local_search
from andrey.core.ci import SingularCorrelationError
from andrey.search import calibrate as cal

VARIABLE = cal.METHODS["blanket"].variable
SHARED_SCAN = "ANDREY_SHARED_SCAN_MIN_WORK"
D = 12
PAIRS = D * (D - 1)
LEARNERS = {
    "iamb": ("moral", "skeleton", "cpdag"),
    "inter_iamb": ("moral", "skeleton", "cpdag"),
    "mmpc": ("skeleton", "cpdag"),
}
CASES = [(method, structure) for method, structures in LEARNERS.items() for structure in structures]
TESTS = ["fisherz", "student_t", "normal_scores"]


@pytest.fixture(autouse=True)
def unset_cutoff(monkeypatch):
    monkeypatch.delenv(VARIABLE, raising=False)
    monkeypatch.delenv(SHARED_SCAN, raising=False)


@pytest.fixture
def pools(monkeypatch):
    """The pools the target workers start, and the targets each one is asked for."""
    started: list[int] = []
    real = tg.ProcessPoolExecutor
    monkeypatch.setattr(
        tg, "ProcessPoolExecutor", lambda *a, **k: started.append(1) or real(*a, **k)
    )
    asked: list[int] = []
    pooled = tg._pooled
    monkeypatch.setattr(
        tg,
        "_pooled",
        lambda s, a, targets, w: asked.append(len(targets)) or pooled(s, a, targets, w),
    )
    return started, asked


def _once(fn, repeats):
    """One timed run of ``fn``, standing in for the calibrator's median of several."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _data(d: int = D, n: int = 500, seed: int = 0) -> np.ndarray:
    return cal.sample(d, seed=seed, rows=n)


def _frame(d: int = D) -> pd.DataFrame:
    return pd.DataFrame(_data(d), columns=[f"v{i}" for i in range(d)])


def _structure(x, method: str, structure: str, workers: int, **settings):
    """``(marks, search)`` of a full-graph fit on ``workers``, as ``local_structure`` runs it."""
    search = local_search(x, alpha=0.1, **settings)
    with andrey.config(n_jobs=workers), search.pinned():
        return search.structure(method, structure), search


def _same_runs(a, b) -> None:
    """Every run of two searches holds the same members, in order, and the same separating sets."""
    assert a._runs.keys() == b._runs.keys()
    for key, (members, sepsets) in a._runs.items():
        other_members, other_sepsets = b._runs[key]
        assert members == other_members
        assert type(sepsets) is type(other_sepsets)
        assert list(sepsets.items()) == list(other_sepsets.items())


@pytest.mark.parametrize("indep_test", TESTS)
@pytest.mark.parametrize("method, structure", CASES)
def test_pooled_runs_are_the_runs_of_one_process(method, structure, indep_test, pools, monkeypatch):
    x = _data()
    serial, base = _structure(x, method, structure, 1, indep_test=indep_test)
    assert not pools[0]
    monkeypatch.setenv(VARIABLE, "0")
    pooled, search = _structure(x, method, structure, 4, indep_test=indep_test)
    assert len(pools[0]) == 1 and pools[1] == [D]  # one pool, asked for every variable's run
    np.testing.assert_array_equal(pooled, serial)
    _same_runs(search, base)


@pytest.mark.parametrize("max_depth", [None, 1])
def test_pooled_runs_hold_with_a_depth_cap_and_knowledge(max_depth, pools, monkeypatch):
    from andrey.core.knowledge import Knowledge, resolve

    x = _data(seed=3)
    constraints = resolve(Knowledge(forbidden=[(0, 1)], required=[(2, 3)]), D, None, method="mmpc")
    settings = {"indep_test": "fisherz", "max_depth": max_depth, "constraints": constraints}
    serial, base = _structure(x, "mmpc", "cpdag", 1, **settings)
    monkeypatch.setenv(VARIABLE, "0")
    pooled, search = _structure(x, "mmpc", "cpdag", 4, **settings)
    assert pools[0]
    np.testing.assert_array_equal(pooled, serial)
    _same_runs(search, base)


@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc", "mmhc", "rsmax2"])
def test_a_facade_fit_gives_the_same_output_with_workers(method, pools, monkeypatch):
    """A DataFrame's labels travel too: the graph and the metadata are those of one process."""
    frame = _frame()
    fit = getattr(andrey, method)

    def run(workers: int):
        with andrey.config(n_jobs=workers):
            return fit(frame, alpha=0.1)

    serial = run(1)
    assert not pools[0]
    monkeypatch.setenv(VARIABLE, "0")
    pooled = run(4)
    assert pools[0]
    assert pooled.structure == serial.structure
    assert dict(pooled.metadata) == dict(serial.metadata)


@pytest.mark.parametrize("restrict", ["iamb", "inter_iamb"])
def test_a_restrict_phase_that_reads_blanket_sets_runs_on_the_pool(restrict, pools, monkeypatch):
    frame = _frame()
    with andrey.config(n_jobs=1):
        serial = andrey.rsmax2(frame, restrict=restrict, alpha=0.1)
    monkeypatch.setenv(VARIABLE, "0")
    with andrey.config(n_jobs=4):
        pooled = andrey.rsmax2(frame, restrict=restrict, alpha=0.1)
    assert pools[0]
    assert pooled.structure == serial.structure


# --- the gate -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cutoff, pooled", [(0, True), (PAIRS, True), (PAIRS + 1, False), (10**9, False)]
)
def test_a_fit_uses_the_pool_from_the_cutoff(cutoff, pooled, pools, monkeypatch):
    """The work is the target and candidate pairs, ``d * (d - 1)``: a fit of exactly the cutoff's
    work uses the pool, and one pair fewer does not. Either way the graph is the one-process one."""
    x = _data()
    serial, _ = _structure(x, "iamb", "moral", 1, indep_test="fisherz")
    monkeypatch.setenv(VARIABLE, str(cutoff))
    result, _ = _structure(x, "iamb", "moral", 4, indep_test="fisherz")
    assert bool(pools[0]) is pooled
    np.testing.assert_array_equal(result, serial)


def test_the_default_cutoff_keeps_a_small_fit_in_this_process(pools):
    _structure(_data(), "iamb", "moral", 4, indep_test="fisherz")
    assert not pools[0] and PAIRS < tg.MIN_POOL_BLANKET_WORK


def test_the_pool_takes_only_the_runs_the_search_has_not_made(pools, monkeypatch):
    x = _data()
    serial, base = _structure(x, "iamb", "moral", 1, indep_test="fisherz")
    monkeypatch.setenv(VARIABLE, "0")
    search = local_search(x, alpha=0.1, indep_test="fisherz")
    search.run("iamb", 4)
    with andrey.config(n_jobs=4), search.pinned():
        pooled = search.structure("iamb", "moral")
    assert pools[1] == [D - 1]
    np.testing.assert_array_equal(pooled, serial)
    _same_runs(search, base)


def test_a_single_worker_never_uses_the_pool(pools, monkeypatch):
    monkeypatch.setenv(VARIABLE, "0")
    _structure(_data(), "iamb", "moral", 1, indep_test="fisherz")
    assert not pools[0]


def test_a_test_that_a_worker_cannot_rebuild_stays_in_this_process(pools, monkeypatch):
    """A categorical test, and a test-wise deletion test with missing values, run in the caller."""
    monkeypatch.setenv(VARIABLE, "0")
    rng = np.random.default_rng(0)
    discrete = rng.integers(0, 3, (400, 8)).astype(float)
    discrete[:, 1] = (discrete[:, 0] + rng.integers(0, 2, 400)) % 3
    with andrey.config(n_jobs=4):
        andrey.iamb(discrete, indep_test="chisq", alpha=0.1)
        gappy = _data(8)
        gappy[rng.random(gappy.shape) < 0.05] = np.nan
        andrey.iamb(gappy, missing="testwise", alpha=0.1)
    assert not pools[0]


def test_a_worker_error_is_the_error_of_one_process(pools, monkeypatch):
    """A column that is the sum of two others makes a conditioning set singular inside a run."""
    frame = _frame()
    frame["total"] = frame["v3"] + frame["v5"]

    def error(workers: int) -> Exception:
        with andrey.config(n_jobs=workers), pytest.raises(ValueError) as caught:
            andrey.iamb(frame, structure="moral")
        return caught.value

    serial = error(1)
    assert not pools[0]
    monkeypatch.setenv(VARIABLE, "0")
    pooled = error(4)
    assert pools[0]
    assert type(pooled) is type(serial) is SingularCorrelationError
    assert str(pooled) == str(serial)


# --- the workers read the caller's setting --------------------------------------------------------


def _shared_scan_setting(search, target):
    return os.environ.get(SHARED_SCAN)


def test_a_worker_reads_the_setting_the_caller_has_not_the_forkserver(monkeypatch):
    """The first pool starts the forkserver with the variable unset; later pools get its value as
    the caller has it at that moment, or none."""
    search = local_search(_data(), alpha=0.1, indep_test="fisherz")
    assert set(tg._pooled(search, _shared_scan_setting, [0, 1, 2, 3], 2)) == {None}
    monkeypatch.setenv(SHARED_SCAN, "0")
    assert set(tg._pooled(search, _shared_scan_setting, [0, 1, 2, 3], 2)) == {"0"}
    monkeypatch.setenv(SHARED_SCAN, "7")
    assert set(tg._pooled(search, _shared_scan_setting, [0, 1], 2)) == {"7"}
    monkeypatch.delenv(SHARED_SCAN)
    assert set(tg._pooled(search, _shared_scan_setting, [0, 1], 2)) == {None}


@pytest.mark.parametrize("value", [None, 5])
def test_an_initialized_worker_holds_the_callers_setting_not_a_stale_one(value, monkeypatch):
    """A worker that inherited another value drops it: set to the caller's, or unset."""
    search = local_search(_data(), alpha=0.1, indep_test="fisherz")
    corr, _ = search.test.to_seed()
    monkeypatch.setattr(tg, "_WORKER_SEARCH", None)
    monkeypatch.setattr(tg, "_WORKER_BLAS", None)
    monkeypatch.setenv(SHARED_SCAN, "stale")
    try:
        tg._init_worker(value, *tg._seed(search, corr))
        assert os.environ.get(SHARED_SCAN) == (None if value is None else str(value))
    finally:
        if tg._WORKER_BLAS is not None:
            tg._WORKER_BLAS.restore_original_limits()


# --- the warning ----------------------------------------------------------------------------------


def _fit_warnings(workers: int = 2):
    with warnings.catch_warnings(record=True) as caught, andrey.config(n_jobs=workers):
        warnings.simplefilter("always")
        andrey.iamb(_data(), alpha=0.1)
        andrey.iamb(_data(), alpha=0.1)
    return [w for w in caught if w.category is andrey.PerformanceWarning]


def test_the_default_cutoff_warns_once_naming_the_setting_and_command(monkeypatch):
    monkeypatch.setattr(tg, "MIN_POOL_BLANKET_WORK", 10 * PAIRS)  # the fit reaches a tenth of it
    [warning] = _fit_warnings()  # two fits, one warning
    assert f"{VARIABLE}={10 * PAIRS}" in str(warning.message)
    assert "`andrey calibrate`" in str(warning.message)
    assert warning.filename == __file__


@pytest.mark.parametrize("factor, warns", [(10, True), (20, False)])
def test_the_default_cutoff_warns_once_a_fit_reaches_a_tenth_of_it(factor, warns, monkeypatch):
    monkeypatch.setattr(tg, "MIN_POOL_BLANKET_WORK", factor * PAIRS)
    assert bool(_fit_warnings()) is warns


@pytest.mark.parametrize("value", [0, PAIRS, 10 * PAIRS])
def test_an_explicit_cutoff_never_warns(value, monkeypatch):
    """Any set value counts, including one equal to the built-in default (the last)."""
    monkeypatch.setattr(tg, "MIN_POOL_BLANKET_WORK", 10 * PAIRS)
    monkeypatch.setenv(VARIABLE, str(value))
    assert not _fit_warnings()


def test_a_serial_fit_never_warns(monkeypatch):
    monkeypatch.setattr(tg, "MIN_POOL_BLANKET_WORK", 10 * PAIRS)
    assert not _fit_warnings(workers=1)


def test_a_bad_cutoff_is_rejected_when_a_fit_reads_it(monkeypatch):
    monkeypatch.setenv(VARIABLE, "many")
    with andrey.config(n_jobs=2), pytest.raises(ValueError, match=VARIABLE):
        andrey.iamb(_data(), alpha=0.1)


# --- the calibrator -------------------------------------------------------------------------------


def test_the_blanket_calibrator_reads_the_built_in_cutoff(monkeypatch):
    method = cal.METHODS["blanket"]
    assert method.variable == "ANDREY_BLANKET_PARALLEL_MIN_WORK" and method.at_cutoff
    assert method.default() == tg.MIN_POOL_BLANKET_WORK
    monkeypatch.setattr(tg, "MIN_POOL_BLANKET_WORK", 123)
    assert method.default() == 123


def test_the_blanket_timers_fit_serially_and_use_the_pool_only_in_the_pooled_arm(
    pools, monkeypatch
):
    monkeypatch.setattr(cal, "_time", _once)
    serial, pooled, work = cal.blanket_timers("iamb", "moral", workers=2)
    assert serial(10) > 0 and not pools[0]
    assert pooled(10) > 0 and pools[0]
    assert work(10) == 90


def test_the_blanket_calibrator_times_each_learner_and_takes_the_widest(pools, monkeypatch):
    """Run on small fits: one timed run each; every learner of the probe is timed."""
    monkeypatch.setattr(cal, "_time", _once)
    timed = []
    real = cal.blanket_timers
    monkeypatch.setattr(
        cal, "blanket_timers", lambda m, s, w: timed.append((m, s)) or real(m, s, w)
    )
    tiny = dataclasses.replace(cal.METHODS["blanket"], sizes=((8, 10), 6, 12))
    monkeypatch.setitem(cal.METHODS, "blanket", tiny)
    c = cal.calibrate(2, ["blanket"])["blanket"]
    assert timed == list(cal.BLANKET_FITS)
    assert c.method == "blanket" and c.cutoff > 0 and c.start_s is not None
    assert all(c.work(d) == d * (d - 1) for d, _, _ in c.timings)
    assert 6 <= min(d for d, _, _ in c.timings) and max(d for d, _, _ in c.timings) <= 12


def test_calibrate_command_measures_the_blanket_cutoff_in_pairs(capsys, monkeypatch):
    def fake(workers, methods):
        timings, works = ((100, 0.1, 0.2), (200, 0.5, 0.2)), ((100, 9_900), (200, 39_800))
        return {m: cal.Crossover(100, 200, timings, works, m) for m in methods}

    monkeypatch.setattr(cal, "calibrate", fake)
    assert main(["calibrate", "--workers", "4", "--method", "blanket", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["settings"] == {"ANDREY_N_JOBS": "4", VARIABLE: "20000"}
    assert report["methods"]["blanket"]["pool_faster_at"] == {"d": 200, "work": 39_800}
