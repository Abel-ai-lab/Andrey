"""One shared read-only matrix per worker pool: the bytes workers read, the segment's life, and the
pickled route.

On Linux with the forkserver, the GES, skeleton, and target-list pools give their workers one
shared-memory copy of the covariance or correlation matrix (:class:`~andrey.search._parallel.
PoolMatrix`); elsewhere, or when the segment cannot be made, each worker's start pickles the
matrix. A worker must read the parent's exact bytes either way, hold no private copy on the shared
route, and leave no segment behind once its pool closes, fails to start, or fails a task.
"""

from __future__ import annotations

import os
import pickle
import tracemalloc
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import shared_memory

import numpy as np
import pytest

import andrey
from andrey.constraint import _parallel_skeleton as sk
from andrey.constraint import _parallel_targets as tg
from andrey.constraint.blanket import local_search
from andrey.core.ci import make_indep_test
from andrey.core.knowledge import Knowledge, resolve
from andrey.core.score import BICScore
from andrey.search import _parallel as par
from andrey.search import _parallel_ges as pg


def _data(n: int = 300, d: int = 12, seed: int = 0) -> np.ndarray:
    """A linear-Gaussian chain with a few cross links over ``d`` columns."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    for j in range(1, d):
        x[:, j] += 0.6 * x[:, j - 1]
        if j >= 3:
            x[:, j] -= 0.4 * x[:, j - 3]
    return x


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


@pytest.fixture
def shared_route() -> None:
    """Skip a test of the shared route where pools take the pickled one."""
    if not par._shares(par.pick_context()):
        pytest.skip("the shared route needs Linux and the forkserver")


needs_shared = pytest.mark.usefixtures("shared_route")


@pytest.fixture
def segments(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The name of each shared segment a pool makes during the test."""
    names: list[str] = []
    real = par.PoolMatrix

    class Spy(real):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if isinstance(self.payload, par.SharedMatrix):
                names.append(self.payload.name)

    monkeypatch.setattr(par, "PoolMatrix", Spy)
    return names


@pytest.fixture
def no_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every segment fails to be made, as on a full ``/dev/shm``: pools take the pickled route."""

    def full(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(par.shared_memory, "SharedMemory", full)


def _exists(name: str) -> bool:
    return os.path.exists(f"/dev/shm/{name}")


def _mapped_file(address: int) -> str | None:
    """The file this process maps at ``address`` (``/proc/self/maps``), or ``None``."""
    with open("/proc/self/maps") as fh:
        for line in fh:
            span, *rest = line.split()
            low, high = (int(v, 16) for v in span.split("-"))
            if low <= address < high:
                return rest[4] if len(rest) > 4 else None
    return None


def _probe(matrix: np.ndarray) -> tuple[str, bool, str | None]:
    """Worker side: the matrix's digest, whether it is writeable, and the file it is mapped from."""
    address = matrix.__array_interface__["data"][0]
    return par.digest(matrix), bool(matrix.flags.writeable), _mapped_file(address)


def _ges_probe() -> tuple[str, bool, str | None]:
    assert pg._WORKER_DELTA is not None
    return _probe(pg._WORKER_DELTA.score_obj.cov)


def _skeleton_probe() -> tuple[str, bool, str | None]:
    assert sk._WORKER_TEST is not None
    return _probe(sk._WORKER_TEST._corr)


def _target_probe(search, t: int) -> tuple[str, bool, str | None]:
    return _probe(search.test._corr)


def _target_fails(search, t: int) -> None:
    raise ValueError(f"target {t} fails")


def _ges_pool(cov: np.ndarray, workers: int = 2, penalty: float = 1.0) -> par.StableWorkerSet:
    d = cov.shape[0]
    return pg._create_pool(cov, 300, d, penalty, 0.0, "tok", workers, list(range(d)))


# --- the helper -----------------------------------------------------------------------------------


@needs_shared
def test_a_pool_matrix_is_one_read_only_copy_until_closed():
    matrix = np.cov(_data(), rowvar=False)
    shared = par.PoolMatrix(matrix, par.pick_context())
    assert isinstance(shared.payload, par.SharedMatrix)
    view = par.attach(shared.payload)
    assert par.digest(view) == par.digest(matrix)
    assert view.dtype == matrix.dtype and view.flags.c_contiguous
    with pytest.raises(ValueError):
        view.flags.writeable = True  # read-only for good, not only by flag
    shared.close()
    shared.close()  # closing twice is a no-op
    assert not _exists(shared.payload.name)
    assert np.array_equal(view, matrix)  # a mapping outlives the name


def test_the_pickled_route_carries_the_matrix_itself(no_segment):
    matrix = np.cov(_data(), rowvar=False)
    for context in (par.pick_context(), par.mp.get_context("spawn")):
        with par.PoolMatrix(matrix, context) as shared:
            assert shared.payload is matrix
            assert par.attach(shared.payload) is matrix


@needs_shared
def test_a_matrix_that_is_not_c_contiguous_is_pickled():
    matrix = np.asfortranarray(np.cov(_data(), rowvar=False))
    with par.PoolMatrix(matrix, par.pick_context()) as shared:
        assert shared.payload is matrix


@needs_shared
def test_a_full_dev_shm_leaves_no_segment(monkeypatch):
    made: list[str] = []
    real = shared_memory.SharedMemory

    def recording(*args, **kwargs):
        segment = real(*args, **kwargs)
        made.append(segment.name)
        return segment

    def full(fd, offset, length):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(par.shared_memory, "SharedMemory", recording)
    monkeypatch.setattr(par.os, "posix_fallocate", full)
    matrix = np.cov(_data(), rowvar=False)
    with par.PoolMatrix(matrix, par.pick_context()) as shared:
        assert shared.payload is matrix
    assert made and not any(_exists(name) for name in made)


def test_a_digest_reads_shape_type_and_bytes():
    a = np.arange(16, dtype=np.float64)
    assert par.digest(a) == par.digest(a.copy())
    assert par.digest(a) != par.digest(a.reshape(4, 4))
    assert par.digest(a) != par.digest(a.astype(np.float32))
    b = a.copy()
    b[3] = np.nextafter(b[3], np.inf)
    assert par.digest(a) != par.digest(b)


# --- what each pool's workers read ----------------------------------------------------------------


@needs_shared
def test_ges_workers_read_the_parents_bytes_from_the_segment(segments):
    cov = BICScore(_data(d=40)).cov
    wset = _ges_pool(cov, workers=3)
    try:
        probes = [pool.submit(_ges_probe).result() for pool in wset._pools]
    finally:
        wset.close()
    assert len(segments) == 1 and not _exists(segments[0])  # named only while the workers boot
    for found, writeable, mapped in probes:
        assert found == par.digest(cov)
        assert not writeable
        assert mapped == f"/dev/shm/{segments[0]}"


@needs_shared
def test_skeleton_workers_read_the_parents_bytes_from_the_segment(segments):
    test = make_indep_test("fisherz", _data(d=40))
    pool = sk.SkeletonWorkerPool(test, 2)
    try:
        executor, _ = pool._ensure()
        probes = [executor.submit(_skeleton_probe).result() for _ in range(4)]
        assert _exists(segments[0])  # a worker may still start
    finally:
        pool.close()
    assert len(segments) == 1 and not _exists(segments[0])
    corr, _ = test.to_seed()
    for found, writeable, mapped in probes:
        assert found == par.digest(corr)
        assert not writeable
        assert mapped == f"/dev/shm/{segments[0]}"


@needs_shared
def test_target_workers_read_the_parents_bytes_from_the_segment(segments):
    search = local_search(_data(d=40), alpha=0.05, indep_test="fisherz")
    probes = tg._pooled(search, _target_probe, list(range(8)), 2)
    assert len(segments) == 1 and not _exists(segments[0])
    corr, _ = search.test.to_seed()
    for found, writeable, mapped in probes:
        assert found == par.digest(corr)
        assert not writeable
        assert mapped == f"/dev/shm/{segments[0]}"


def test_workers_read_the_parents_bytes_on_the_pickled_route(no_segment):
    x = _data(d=30)
    cov = BICScore(x).cov
    wset = _ges_pool(cov)
    try:
        ges = [pool.submit(_ges_probe).result() for pool in wset._pools]
    finally:
        wset.close()
    search = local_search(x, alpha=0.05, indep_test="fisherz")
    targets = tg._pooled(search, _target_probe, list(range(4)), 2)
    corr, _ = search.test.to_seed()
    assert {found for found, _, _ in ges} == {par.digest(cov)}
    assert {found for found, _, _ in targets} == {par.digest(corr)}


@pytest.mark.parametrize("name", ["fisherz", "student_t", "normal_scores"])
def test_a_target_worker_rebuilds_the_callers_search(name):
    """The rebuilt search holds what the caller's does, arrays bit for bit, test included."""
    x = _data(d=10)
    knowledge = Knowledge(forbidden=[(0, 1)], required=[(2, 3)])
    search = local_search(
        x,
        alpha=0.01,
        indep_test=name,
        max_depth=2,
        constraints=resolve(knowledge, 10, None, method="mmpc"),
    )
    corr, _ = search.test.to_seed()
    rebuilt = tg._rebuild(*pickle.loads(pickle.dumps(tg._seed(search, corr))))  # as a worker does

    def same(a, b) -> None:
        assert type(a) is type(b)
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        elif hasattr(a, "__dict__") and not isinstance(a, type):
            assert vars(a).keys() == vars(b).keys()
            for key in vars(a):
                same(vars(a)[key], vars(b)[key])
        else:
            assert a == b

    assert search.test._copies is not None  # the parent's groups of copies travel along
    same(rebuilt, search)


# --- no segment outlives its pool -----------------------------------------------------------------


@needs_shared
def test_no_segment_outlives_a_failed_start(segments):
    cov = BICScore(_data(d=20)).cov
    with pytest.raises(RuntimeError, match="failed to start"):
        _ges_pool(cov, penalty=-1.0)  # each worker's score rejects the penalty
    test = make_indep_test("fisherz", _data(d=20))
    pool = sk.SkeletonWorkerPool(test, 2)
    pool._n = "bad"  # each worker's test rejects the sample count
    try:
        executor, _ = pool._ensure()
        with pytest.raises(BrokenProcessPool):
            executor.submit(int).result()
    finally:
        pool.close()
    search = local_search(_data(d=20), alpha=0.05, indep_test="fisherz")
    search.test._n = "bad"
    with pytest.raises(RuntimeError, match="worker died"):
        tg._pooled(search, _target_probe, list(range(4)), 2)
    assert len(segments) == 3 and not any(_exists(name) for name in segments)


@needs_shared
def test_no_segment_outlives_a_task_error(segments):
    test = make_indep_test("fisherz", _data(d=20))
    pool = sk.SkeletonWorkerPool(test, 2)
    try:
        executor, _ = pool._ensure()
        payload = ([(0, 1)], np.ones((20, 20), dtype=bool), 2, 0.05, True, "wrong token")
        with pytest.raises(RuntimeError, match="reused across runs"):
            executor.submit(sk._scan_task, payload).result()
    finally:
        pool.close()
    search = local_search(_data(d=20), alpha=0.05, indep_test="fisherz")
    with pytest.raises(ValueError, match="fails"):
        tg._pooled(search, _target_fails, list(range(4)), 2)
    assert len(segments) == 2 and not any(_exists(name) for name in segments)


# --- the pool start holds no extra copy in the caller ---------------------------------------------


@needs_shared
def test_a_ges_pool_start_pickles_no_matrix_in_the_caller():
    """The caller's Python allocations while the workers start stay below one matrix; each start
    pickled one copy, and the run token another."""
    cov = BICScore(_data(n=500, d=400)).cov
    lazy = pg._LazyWorkerSet(BICScore.from_cov(cov, 500), 400, 3, list(range(400)))
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        lazy.get()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        lazy.close()
    assert peak - base < cov.nbytes / 2, (peak - base, cov.nbytes)


@needs_shared
def test_a_skeleton_pool_start_pickles_no_matrix_in_the_caller():
    test = make_indep_test("fisherz", _data(n=500, d=400))
    corr, _ = test.to_seed()
    pool = sk.SkeletonWorkerPool(test, 3)
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        executor, _ = pool._ensure()
        [future.result() for future in [executor.submit(int) for _ in range(6)]]
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        pool.close()
    assert peak - base < corr.nbytes / 2, (peak - base, corr.nbytes)


# --- the answers do not depend on the route -------------------------------------------------------


def _bits(value: object) -> object:
    """``value`` with every float as its bytes, so equal results mean equal bits."""
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, dict):
        return tuple((key, _bits(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_bits(item) for item in value)
    return value


def _fits(x: np.ndarray, n_jobs: int = 2) -> dict[str, object]:
    with andrey.config(n_jobs=n_jobs):
        ges = andrey.ges(x)
        pc = andrey.pc(x, indep_test="fisherz")
        targets = andrey.mmpc(x, target=list(range(x.shape[1])))
    return {
        name: _bits([(o.structure.to_numpy(), dict(o.metadata)) for o in outs])
        for name, outs in (("ges", [ges]), ("pc", [pc]), ("mmpc", targets))
    }


@pytest.fixture
def pooled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every GES and skeleton pass runs on the pool."""
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", "0")


@needs_shared
def test_the_shared_route_answers_as_one_process(pooled, segments):
    x = _data(n=400, d=14, seed=3)
    serial = _fits(x, n_jobs=1)
    assert not segments
    assert _fits(x) == serial
    assert len(segments) >= 3  # GES, the skeleton, and the targets each made one


@needs_shared
def test_the_pickled_route_answers_as_the_shared_one(pooled, request):
    x = _data(n=400, d=14, seed=4)
    shared = _fits(x)
    request.getfixturevalue("no_segment")
    assert _fits(x) == shared
