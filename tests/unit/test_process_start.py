"""What a fresh process imports: a fit pays only for the libraries it uses.

Each test runs its code in a new interpreter, because ``sys.modules`` of this one holds whatever an
earlier test imported.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import andrey
from andrey.api._method import _missing_cells
from andrey.core import backend


def _fresh_process(code: str) -> str:
    """The output of ``code`` run in a new interpreter that imports this tree's ``andrey``."""
    src_dir = str(pathlib.Path(andrey.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for name in ("ANDREY_BACKEND", "ANDREY_DEVICE", "ANDREY_GPU_CALIBRATE", "ANDREY_N_JOBS"):
        env.pop(name, None)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _cpu_torch_installed() -> bool:
    """Whether torch is installed as a build for neither CUDA nor ROCm, read unimported."""
    return importlib.util.find_spec("torch") is not None and not backend._torch_gpu_build()


def test_an_array_fit_imports_neither_scipy_stats_nor_pandas():
    code = (
        "import sys, numpy as np, andrey\n"
        "andrey.pc(np.random.default_rng(0).standard_normal((300, 5)))\n"
        "print(sorted({'scipy.stats', 'pandas', 'andrey.fcm.gam'} & set(sys.modules)))"
    )
    assert _fresh_process(code) == "[]"


def test_a_data_frame_is_read_once_pandas_is_imported():
    pd = pytest.importorskip("pandas")
    values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    for table in (pd.DataFrame(values, columns=["a", "b", "c"]), values, values.tolist()):
        assert np.array_equal(_missing_cells(table), np.isnan(values))
    assert _missing_cells(values[0]) is None  # one row of numbers is not a table
    nullable = pd.DataFrame({"a": pd.array([1, pd.NA, 3], dtype="Int64")})
    assert _missing_cells(nullable).tolist() == [[False], [True], [False]]


def test_a_data_frame_is_read_when_pandas_is_imported_after_andrey():
    # The module is looked up when a table is read, so a frame built after `import andrey` is found.
    pytest.importorskip("pandas")
    code = (
        "import sys, andrey\n"
        "from andrey.api._method import _missing_cells\n"
        "assert 'pandas' not in sys.modules\n"
        "import pandas as pd\n"
        "frame = pd.DataFrame({'a': pd.array([1, pd.NA, 3], dtype='Int64')})\n"
        "print(_missing_cells(frame).tolist())"
    )
    assert _fresh_process(code) == "[[False], [True], [False]]"


@pytest.mark.skipif("forkserver" not in mp.get_all_start_methods(), reason="no forkserver")
def test_a_forkserver_worker_has_the_registry_loaded_before_its_first_task():
    code = (
        "import sys\n"
        "from andrey.search import _parallel as par\n"
        "if __name__ == '__main__':\n"
        "    ctx = par.pick_context()\n"
        "    assert ctx.get_start_method() == 'forkserver', ctx\n"
        "    probe = \"'andrey.spec.registry' in __import__('sys').modules\"\n"
        "    with ctx.Pool(1) as pool:\n"
        "        print(pool.apply(eval, (probe,)))"
    )
    assert _fresh_process(code) == "True"


@pytest.mark.skipif(not _cpu_torch_installed(), reason="needs a CPU build of torch")
@pytest.mark.parametrize(
    "fit",
    [
        "andrey.direct_lingam(rng.standard_normal((1000, 20)))",
        # a chain, so the tests have conditioning sets
        "x = rng.standard_normal((200, 4))\n"
        "for j in range(1, 4): x[:, j] += x[:, j - 1]\n"
        "andrey.pc(x, indep_test='kci')",
    ],
    ids=["direct_lingam", "kci"],
)
def test_a_cpu_build_of_torch_is_not_imported_to_rule_out_cuda(fit):
    # Both fits clear their op's cutoff, so `auto` asks whether CUDA is there; a CPU build answers
    # no from its version file, and the fit stays on numpy without importing torch.
    code = (
        "import sys, numpy as np, andrey\n"
        "from andrey.core import backend\n"
        "rng = np.random.default_rng(0)\n"
        f"{fit}\n"
        "assert 'torch' not in sys.modules, 'the fit imported torch'\n"
        "assert {d for _, d in backend.dispatch_stats()} == {'numpy'}, backend.dispatch_stats()\n"
        "print('ok')"
    )
    assert _fresh_process(code) == "ok"


@pytest.mark.skipif(not _cpu_torch_installed(), reason="needs a CPU build of torch")
def test_a_cuda_pin_still_asks_torch():
    # The pin reaches torch for its answer and warns that CUDA is unavailable, as it did.
    code = (
        "import sys, warnings, andrey\n"
        "from andrey.core import backend\n"
        "with warnings.catch_warnings(record=True) as caught, andrey.config(backend='cuda'):\n"
        "    warnings.simplefilter('always')\n"
        "    dev = backend.resolve(10**9, {'cuda': 1, 'mps': None}, amortized=True)\n"
        "assert dev == 'numpy', dev\n"
        "assert 'torch' in sys.modules, 'a cuda pin did not ask torch'\n"
        "assert any(w.category is andrey.BackendFallbackWarning for w in caught), caught\n"
        "print('ok')"
    )
    assert _fresh_process(code) == "ok"
