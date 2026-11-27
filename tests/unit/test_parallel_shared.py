"""One shared read-only matrix per worker pool: the bytes workers read, the shared file's life,
and the pickled route.

On Linux with the forkserver, the GES, skeleton, and target-list pools give their workers one copy
of the covariance or correlation matrix in an unnamed ``/dev/shm`` file (:class:`~andrey.search.
_parallel.PoolMatrix`); elsewhere, or when the file cannot be made, each worker's start pickles the
matrix. A worker must read the parent's exact bytes either way and hold no private copy on the
shared route. The file never has a name, and no process of the pool holds it once its pool closes,
fails to start, or fails a task.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import tracemalloc
import types
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

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


#: A file by its device and inode.
File = tuple[int, int]


@pytest.fixture
def files(monkeypatch: pytest.MonkeyPatch) -> list[tuple[File, par.SharedMatrix]]:
    """Each shared file a pool makes during the test, with its handle; each has no name."""
    made: list[tuple[File, par.SharedMatrix]] = []
    real = par.PoolMatrix

    class Spy(real):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if isinstance(self.payload, par.SharedMatrix):
                info = os.fstat(self.payload.fileno())
                assert info.st_nlink == 0  # no name in /dev/shm, from the start
                made.append(((info.st_dev, info.st_ino), self.payload))

    monkeypatch.setattr(par, "PoolMatrix", Spy)
    return made


@pytest.fixture
def no_room(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every shared file fails to be made, as on a full ``/dev/shm``: pools take the pickled
    route."""

    def full(matrix: np.ndarray) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(par, "_matrix_file", full)


def _descendants() -> list[int]:
    """This process and every process below it."""
    parents: dict[int, int] = {}
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as fh:
                    parents[int(name)] = int(fh.read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                pass
    found = [os.getpid()]
    for pid in found:
        found.extend(child for child, parent in parents.items() if parent == pid)
    return found


def _holds(pid: int, file: File) -> bool:
    """Whether process ``pid`` has ``file`` open or mapped."""
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
        with open(f"/proc/{pid}/maps") as fh:
            maps = fh.readlines()
    except OSError:
        return False
    for fd in fds:
        try:
            info = os.stat(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if (info.st_dev, info.st_ino) == file:
            return True
    return any(_mapping_file(line) == file for line in maps)


def _holders(file: File) -> list[int]:
    """The processes of this test's process tree that have ``file`` open or mapped."""
    return [pid for pid in _descendants() if _holds(pid, file)]


def _mapping_file(line: str) -> File | None:
    """The file a ``/proc/<pid>/maps`` line maps, or ``None`` for an anonymous mapping."""
    _, _, _, device, inode, *_ = line.split() + [""]
    if not inode or inode == "0":
        return None
    major, minor = (int(part, 16) for part in device.split(":"))
    return os.makedev(major, minor), int(inode)


def _mapped_file(address: int) -> File | None:
    """The file this process maps at ``address`` (``/proc/self/maps``), or ``None``, also where
    there is no ``/proc``."""
    try:
        with open("/proc/self/maps") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    for line in lines:
        low, high = (int(v, 16) for v in line.split()[0].split("-"))
        if low <= address < high:
            return _mapping_file(line)
    return None


def _probe(matrix: np.ndarray) -> tuple[str, bool, File | None]:
    """Worker side: the matrix's digest, whether it is read-only for good, and the file it is
    mapped from."""
    address = matrix.__array_interface__["data"][0]
    try:
        matrix.flags.writeable = True
        locked = False
    except ValueError:
        locked = True
    return par.digest(matrix), locked, _mapped_file(address)


def _ges_probe() -> tuple[str, bool, File | None]:
    assert pg._WORKER_DELTA is not None
    return _probe(pg._WORKER_DELTA.score_obj.cov)


def _skeleton_probe() -> tuple[str, bool, File | None]:
    assert sk._WORKER_TEST is not None
    return _probe(sk._WORKER_TEST._corr)


def _target_probe(search, t: int) -> tuple[str, bool, File | None]:
    return _probe(search.test._corr)


_KEPT: np.ndarray | None = None


def _keep(matrix: np.ndarray) -> None:
    global _KEPT
    _KEPT = matrix


def _kept_probe() -> tuple[str, bool, File | None]:
    assert _KEPT is not None
    return _probe(_KEPT)


def _target_fails(search, t: int) -> None:
    raise ValueError(f"target {t} fails")


def _ges_pool(cov: np.ndarray, workers: int = 2, penalty: float = 1.0) -> par.StableWorkerSet:
    d = cov.shape[0]
    return pg._create_pool(cov, 300, d, penalty, 0.0, "tok", workers, list(range(d)))


# --- the helper -----------------------------------------------------------------------------------


@needs_shared
def test_a_pool_matrix_is_one_unnamed_read_only_file_only_a_starting_worker_gets():
    matrix = np.cov(_data(), rowvar=False)
    context = par.pick_context()
    shared = par.PoolMatrix(matrix, context)
    payload = shared.payload
    assert isinstance(payload, par.SharedMatrix)
    info = os.fstat(payload.fileno())
    assert info.st_nlink == 0 and info.st_size == matrix.nbytes
    with pytest.raises(RuntimeError, match="inheritance"):
        pickle.dumps(payload)  # only a worker's start carries it
    with ProcessPoolExecutor(2, mp_context=context, initializer=_keep, initargs=(payload,)) as pool:
        probes = [pool.submit(_kept_probe).result() for _ in range(4)]
    assert probes and all(
        probe == (par.digest(matrix), True, (info.st_dev, info.st_ino)) for probe in probes
    )
    shared.close()
    shared.close()  # closing twice is a no-op
    assert payload.closed and not _holders((info.st_dev, info.st_ino))
    pool = ProcessPoolExecutor(1, mp_context=context, initializer=_keep, initargs=(payload,))
    try:
        with pytest.raises(ValueError, match="closed"):
            pool.submit(int)  # a closed handle starts no worker
    finally:
        pool.shutdown()


def test_the_pickled_route_carries_the_matrix_itself(no_room):
    matrix = np.cov(_data(), rowvar=False)
    for context in (par.pick_context(), par.mp.get_context("spawn")):
        with par.PoolMatrix(matrix, context) as shared:
            assert shared.payload is matrix


@needs_shared
def test_a_matrix_that_is_not_c_contiguous_is_pickled():
    matrix = np.asfortranarray(np.cov(_data(), rowvar=False))
    with par.PoolMatrix(matrix, par.pick_context()) as shared:
        assert shared.payload is matrix


class _FailingFile:
    """A real unnamed ``/dev/shm`` file whose writes raise ``error``."""

    def __init__(self, error: BaseException) -> None:
        self.file = tempfile.TemporaryFile(dir="/dev/shm")
        self.error = error

    def write(self, data: object) -> int:
        raise self.error

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


@needs_shared
@pytest.mark.parametrize(
    "error",
    [OSError(28, "No space left on device"), KeyboardInterrupt()],
    ids=["full", "interrupt"],
)
def test_a_failed_copy_closes_the_file(monkeypatch, error):
    """A full ``/dev/shm`` takes the pickled route; an interrupt propagates. Either way the file
    is closed."""
    context = par.pick_context()
    matrix = np.cov(_data(), rowvar=False)
    made: list[_FailingFile] = []

    def failing(**kwargs) -> _FailingFile:
        made.append(_FailingFile(error))
        return made[-1]

    monkeypatch.setattr(par, "tempfile", types.SimpleNamespace(TemporaryFile=failing))
    before = _open_descriptors()
    if isinstance(error, OSError):
        with par.PoolMatrix(matrix, context) as shared:
            assert shared.payload is matrix
    else:
        with pytest.raises(KeyboardInterrupt):
            par.PoolMatrix(matrix, context)
    assert len(made) == 1 and made[0].file.closed
    assert _open_descriptors() == before


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
def test_ges_workers_read_the_parents_bytes_from_the_shared_file(files):
    cov = BICScore(_data(d=40)).cov
    wset = _ges_pool(cov, workers=3)
    try:
        assert len(files) == 1 and files[0][1].closed  # every worker has mapped it
        probes = [pool.submit(_ges_probe).result() for pool in wset._pools]
    finally:
        wset.close()
    [(file, _)] = files
    assert probes and all(probe == (par.digest(cov), True, file) for probe in probes)
    assert not _holders(file)


@needs_shared
def test_skeleton_workers_read_the_parents_bytes_from_the_shared_file(files):
    test = make_indep_test("fisherz", _data(d=40))
    pool = sk.SkeletonWorkerPool(test, 2)
    try:
        executor, _ = pool._ensure()
        probes = [executor.submit(_skeleton_probe).result() for _ in range(4)]
        assert not files[0][1].closed  # a worker may still start
    finally:
        pool.close()
    [(file, handle)] = files
    corr, _ = test.to_seed()
    assert probes and all(probe == (par.digest(corr), True, file) for probe in probes)
    assert handle.closed and not _holders(file)


@needs_shared
def test_target_workers_read_the_parents_bytes_from_the_shared_file(files):
    search = local_search(_data(d=40), alpha=0.05, indep_test="fisherz")
    probes = tg._pooled(search, _target_probe, list(range(8)), 2)
    [(file, handle)] = files
    corr, _ = search.test.to_seed()
    assert probes and all(probe == (par.digest(corr), True, file) for probe in probes)
    assert handle.closed and not _holders(file)


def test_workers_read_the_parents_bytes_on_the_pickled_route(no_room):
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


# --- no shared file outlives its pool -------------------------------------------------------------


@needs_shared
def test_no_shared_file_outlives_a_failed_start(files):
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
    assert len(files) == 3
    assert all(handle.closed and not _holders(file) for file, handle in files)


@needs_shared
def test_no_shared_file_outlives_a_task_error(files):
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
    assert len(files) == 2
    assert all(handle.closed and not _holders(file) for file, handle in files)


# --- the pool start holds no extra copy in the caller ---------------------------------------------


@needs_shared
def test_a_ges_pool_start_pickles_no_matrix_in_the_caller():
    """The caller's Python allocations while the workers start stay below half a matrix; each start
    pickled one copy, and the run token another. At 1,000 columns the bound is several times the
    start's own allocations, which do not grow with the matrix."""
    cov = BICScore(_data(n=500, d=1000)).cov
    lazy = pg._LazyWorkerSet(BICScore.from_cov(cov, 500), 1000, 3, list(range(1000)))
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
    test = make_indep_test("fisherz", _data(n=500, d=1000))
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
def test_the_shared_route_answers_as_one_process(pooled, files):
    x = _data(n=400, d=14, seed=3)
    serial = _fits(x, n_jobs=1)
    assert not files
    assert _fits(x) == serial
    assert len(files) >= 3  # GES, the skeleton, and the targets each made one
    assert all(handle.closed and not _holders(file) for file, handle in files)


@needs_shared
def test_the_pickled_route_answers_as_the_shared_one(pooled, request):
    x = _data(n=400, d=14, seed=4)
    shared = _fits(x)
    request.getfixturevalue("no_room")
    assert _fits(x) == shared
