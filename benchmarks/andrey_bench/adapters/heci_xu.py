"""HECI adapter for the authors' code (Xu, Mian, and Vreeken, ICML 2022), run unmodified.

The code is ``HECI.py`` from ``heci-v20220729.zip``, the supplementary upload on the authors'
project page, which carries no license statement for it. It is a measurement reference only:
``benchmarks/start_heci.sh`` downloads the archive, checks its SHA-256, extracts ``HECI.py``, and
builds the environment; the benchmark loads the file from the directory ``$ANDREY_BENCH_HECI``
names and copies none of it. The authors' README names NumPy as the one dependency, and the fit
runs in an environment of its own that pins NumPy 1.26.4 (``python_env``).

A fit calls the authors' ``HECI(X, Y, n_bins=20, standardize=False)``, the arguments their
Tuebingen experiment (``Benchmark.py``, ``testDataset``) uses and the function's own defaults, on
the two columns as they are: the function normalizes each to ``[0, 1]`` itself. The function returns
a flag and the two description lengths, ``scoreXtoY`` and ``scoreYtoX``, the cost of the best
binned polynomial fit in each direction; the experiment predicts ``X -> Y`` when
``scoreXtoY < scoreYtoX`` and ``Y -> X`` otherwise, forced. The adapter reads the same comparison
of the two scores, and returns no decision when they are equal or either is not a number, where the
code's own flag reads ``Y -> X``. Nothing is tuned, and the fit draws no random numbers.

The authors' scripts skip Tuebingen pair 47 from a fixed list of pairs they treat as not bivariate.
The function raises ``IndexError`` inside ``binning`` on a column with too few distinct values for
its ``min_support = 10`` bins, such as pair 47's two-valued one, and the adapter lets the error
through.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

#: Names the directory holding the extracted ``HECI.py``.
CODE_ENV = "ANDREY_BENCH_HECI"

#: The files a fit loads, relative to the code directory.
REQUIRED_FILES = ("HECI.py",)

#: Threads of numeric libraries a fit may start: one core per fit.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

_MODULE: ModuleType | None = None


def code_dir() -> Path:
    """The directory of the authors' files, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' HECI code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_heci.sh and export what it prints."
        )
    root = Path(raw)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks {', '.join(missing)}")
    return root


def source_sha256() -> str:
    """The SHA-256 of the files :data:`REQUIRED_FILES` names, as read now, so a record names the
    code that ran even if the folder changed after setup; ``unknown`` when they are missing."""
    root = Path(os.environ.get(CODE_ENV, "").strip())
    digest = hashlib.sha256()
    for name in REQUIRED_FILES:
        path = root / name
        if not path.is_file():
            return "unknown"
        digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def archive_sha256() -> str:
    """The SHA-256 of ``heci-v20220729.zip`` that ``start_heci.sh`` verified, or ``unknown``."""
    raw = os.environ.get(CODE_ENV, "").strip()
    marker = Path(raw) / "ARCHIVE_SHA256" if raw else None
    return marker.read_text().strip() if marker is not None and marker.is_file() else "unknown"


def _heci() -> Any:
    """The authors' ``HECI`` function, loaded by path from :func:`code_dir`.

    Loading the one file by path keeps the upload's other modules (``Benchmark``, ``baselines``)
    off ``sys.path``.
    """
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location("heci_authors", code_dir() / "HECI.py")
        if spec is None or spec.loader is None:
            raise SystemExit(f"cannot load HECI.py from ${CODE_ENV}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE.HECI


def edge_adjacency(forward: float, backward: float) -> np.ndarray:
    """The two-node DAG for the two scores: the lower one names the direction.

    A lower ``forward`` (the ``X -> Y`` cost) gives ``[0, 1] = 1``, a lower ``backward`` gives
    ``[1, 0] = 1``; equal scores or a score that is not a number give no edge.
    """
    adjacency = np.zeros((2, 2), dtype=np.int8)
    if forward < backward:
        adjacency[0, 1] = 1
    elif backward < forward:
        adjacency[1, 0] = 1
    return adjacency


class XuHECI:
    """The authors' HECI, forced to a direction by the lower description length.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "xu.heci"
    package: str = "HECI"
    package_version: str = "heci-v20220729"
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "heci"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_HECI_PYTHON"

    def setup(self) -> None:
        """Load the code and fit a tiny pair before timing, so first-call costs stay out."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        for var in _THREAD_VARS:
            os.environ[var] = "1"
        pair = tiny_table()[:, :2]
        self.fit(pair, self.params())

    def params(self) -> dict[str, Any]:
        """The two arguments the adapter passes, the code's defaults, and the archive checksum."""
        return {
            "archive_sha256": archive_sha256(),
            "source_sha256": source_sha256(),
            "n_bins": 20,
            "standardize": False,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' HECI; return the 0/1 DAG adjacency. The only timed region."""
        _, forward, backward = _heci()(
            data[:, 0], data[:, 1], n_bins=params["n_bins"], standardize=params["standardize"]
        )
        return edge_adjacency(float(forward), float(backward))

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
