"""QCCD adapter for the authors' R code (Tagasovska, Chavez-Demoulin, and Vatter, ICML 2020).

The code is the ``tagas/bQCD`` repository at commit :data:`COMMIT`, which carries no license file.
It is a measurement reference only: ``benchmarks/start_qccd.sh`` downloads it, checks the commit and
the SHA-256 of ``R/bqcd.R``, and installs the R packages the file loads; the benchmark ``source()``s
``R/bqcd.R`` from the directory ``$ANDREY_BENCH_QCCD`` names and copies none of it.

A fit calls the authors' real-data wrapper ``QCD_wrap(X, Y, type = "QCCD", m = 3)`` on the two
columns as they are. ``m = 3`` is the paper's setting ("we recommend the value m = 3 ... we report
results for bQCD with m = 3"), with the single quantile level below 200 rows that the paper uses on
real data, which the code applies itself. ``QCCD`` is the paper's copula estimator: it ranks each
column (``ties.method = "random"`` after ``set.seed(0)``, which the function sets itself), fits a
nonparametric ``rvinecopulib::bicop`` (family ``tll``, ``nonpar_method = "constant"``), and compares
the quantile-score code lengths of ``X -> Y`` and ``Y -> X`` at the ``m`` Gauss-Legendre quantile
levels. The wrapper returns ``cd``, ``"->"`` when the ``Y -> X`` code length exceeds the ``X -> Y``
one and ``"<-"`` otherwise, and ``eps``, their difference; the adapter reads ``cd`` as the
direction, forced. It returns no decision when ``cd`` is ``"--"`` (the code length is not a number)
or ``eps`` is 0, where the code's own comparison reads ``"<-"``. Nothing is tuned.
:class:`TagasovskaQCCDM7` calls the wrapper with its default ``m = 7``, a sensitivity row beside the
paper's setting.

The repository's ``R/benchmark-Tuebingen.R`` passes the type ``"bQCD_qnn"`` to the wrapper, a name
its ``switch`` does not list, so that script fails as shipped; the wrapper's default type
``"QCCD"`` is the estimator of the repository's ``results/QCCD_Tueb.tab``. The file's other two
estimators, a quantile forest and a quantile neural network, are not run.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

#: Names the repository root (the directory holding ``R/bqcd.R``).
CODE_ENV = "ANDREY_BENCH_QCCD"

#: The pinned commit of ``tagas/bQCD``; ``start_qccd.sh`` fetches it.
COMMIT = "0e6eefefc46f1abcd876fe34751e18fbd9d23ef8"

#: Files ``source()`` reads, relative to the code directory.
REQUIRED_FILES = ("R/bqcd.R",)

#: Threads of numeric libraries the embedded R may start: one core per fit.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

_R_SOURCE = """
source(file.path(Sys.getenv("ANDREY_BENCH_QCCD"), "R", "bqcd.R"))
andrey_tagasovska_qccd <- function(X) {
    res <- QCD_wrap(X[, 1], X[, 2], type = "QCCD", m = __M__)
    # `C[i, j] = 1` for i -> j, already the orientation the benchmark scores.
    C <- matrix(0, 2, 2, dimnames = list(colnames(X), colnames(X)))
    if (!is.na(res$eps) && res$eps != 0) {
        if (identical(res$cd, "->")) C[1, 2] <- 1
        if (identical(res$cd, "<-")) C[2, 1] <- 1
    }
    C
}
"""


def code_dir() -> Path:
    """The authors' repository root, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' bQCD code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_qccd.sh and export what it prints."
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


class TagasovskaQCCD(RAdapter):
    """The authors' R implementation of QCCD through the embedded R session.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "tagasovska.qccd"
    package: str = "bQCD"
    package_version: str = COMMIT[:8]
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "qccd"
    output_type: str = "dag"

    r_package: str = "rvinecopulib"
    levels: int = 3
    r_source: str = _R_SOURCE.replace("__M__", "3")
    r_function: str = "andrey_tagasovska_qccd"

    def setup(self) -> None:
        """Start R, source the code, and fit a tiny pair before timing.

        The warm-up pair has 250 rows, enough for the code's seven quantile levels.
        """
        from andrey_bench.adapters.fcm_warm import tiny_table

        code_dir()
        for var in _THREAD_VARS:
            os.environ[var] = "1"
        super().setup()
        self.call(tiny_table(n=250)[:, :2])

    def params(self) -> dict[str, Any]:
        """The wrapper's arguments and the versions fits ran with."""
        return {
            "commit": COMMIT,
            "source_sha256": source_sha256(),
            "type": "QCCD",
            "m": self.levels,
            "rvinecopulib": rsession.package_version("rvinecopulib"),
            "statmod": rsession.package_version("statmod"),
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' QCCD and return a 0/1 DAG adjacency. The only timed region."""
        return self.call(data).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class TagasovskaQCCDM7(TagasovskaQCCD):
    """The same wrapper at its default ``m = 7`` quantile levels."""

    name: str = "tagasovska.qccd_m7"
    levels: int = 7
    r_source: str = _R_SOURCE.replace("__M__", "7")
