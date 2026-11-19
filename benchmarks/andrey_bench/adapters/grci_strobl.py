"""GRCI adapter for the authors' R code (Strobl), run unmodified from a local download.

The code is the ``ericstrobl/GRCI`` repository at commit :data:`COMMIT`. Its ``DESCRIPTION``
declares the license CC BY-NC 4.0: non-commercial use with attribution. This adapter therefore reads
the code from a local download only. ``benchmarks/start_grci.sh`` fetches it, checks the commit and
the SHA-256 of every file used, and installs the R packages; the benchmark ``source()``s five files
from the directory ``$ANDREY_BENCH_GRCI`` names, and nothing of the repository enters Andrey's
repository, its packages, or the records the benchmark publishes.

A fit calls the authors' ``causal_direc(X, Y)`` with each column as an ``n x 1`` matrix, as their
Tuebingen experiment (``Experiments/Exp_Tuben_pairs.R``) does for the pair data, with no other
argument: the function has none. It standardizes each variable, partials out the other with the
heteroscedastic spline regression (``PartialOut``: a conditional mean and a conditional mean
absolute deviation, by leave-one-out cross-validated splines on random projections), and compares
the two directions by a difference of k-nearest-neighbor entropy estimates (``k = 10``,
``RANN::nn2``). It returns ``1`` for ``X -> Y`` and ``-1`` for ``Y -> X``, forced; the experiment
counts ``out >= 0`` as ``X -> Y``. The adapter reads the sign. The function maps a score of exactly
0 to ``-1`` and cannot return a tie, so a tie reads as ``Y -> X``. A number that is not a number
reaching one of the comparisons (``score > 0`` in the function, ``err_t < err`` in the regression)
raises "missing value where TRUE/FALSE needed"; the adapter maps exactly that error to no decision
and lets every other error through.

The regression draws random projections (``DirichletReg::rdirichlet``) and the authors' experiment
sets no seed, so the wrapper calls ``set.seed(0)`` before each fit and a fit is reproducible. The
knots and the cross-validation are the code's own. ``nn2`` is called with ``k = n`` and a
per-point radius, which allocates ``n x n`` matrices: memory and time grow with the square of the
row count.

The repository's ``RANN-master.zip``, the nearest-neighbor package its README asks the reader to
install first, is what ``start_grci.sh`` installs in place of CRAN's ``RANN``. It differs where the
code needs it to: its radius search reads one radius per query point, which the code passes as a
vector, where CRAN's reads one radius for all. The version numbers do not tell the two apart (the
bundled one reports 2.6.1, CRAN's has been 2.6.x), so ``setup`` checks the per-point behavior
itself and stops when the library path loads another ``RANN``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter

#: Names the repository root (the directory holding ``R/causal_direc.R``).
CODE_ENV = "ANDREY_BENCH_GRCI"

#: The pinned commit of ``ericstrobl/GRCI``; ``start_grci.sh`` fetches it.
COMMIT = "9f9b78d91184a2a5849182bec2b10426becb4d2b"

#: Files ``source()`` reads, relative to the code directory, in the order it reads them.
REQUIRED_FILES = (
    "R/normalize01.R",
    "R/normalizeData.R",
    "R/spline_regressionR.R",
    "R/PartialOut.R",
    "R/causal_direc.R",
)

#: Threads of numeric libraries the embedded R may start: one core per fit.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# The experiment script attaches MASS, RANN, Rfast, hetGP, and gptk; the five files call only
# `nn2` (RANN), `spdinv` (Rfast), and `rdirichlet` (DirichletReg).
_R_SOURCE = """
suppressMessages({library(RANN); library(Rfast); library(DirichletReg)})
# The bundled RANN reads one radius per point; a RANN that does not would score differently.
local({
    d <- matrix(c(0, 1, 10, 11), ncol = 1)
    found <- rowSums(nn2(d, k = 4, searchtype = "radius", radius = c(0.5, 2, 0.5, 2))$nn.idx > 0)
    if (!identical(as.integer(found), c(1L, 2L, 1L, 2L))) {
        stop("the loaded RANN ignores per-point radii: GRCI needs the RANN bundled with it")
    }
})
for (f in c("normalize01", "normalizeData", "spline_regressionR", "PartialOut", "causal_direc")) {
    source(file.path(Sys.getenv("ANDREY_BENCH_GRCI"), "R", paste0(f, ".R")))
}
andrey_strobl_grci <- function(X) {
    set.seed(0)
    out <- tryCatch(
        causal_direc(X[, 1, drop = FALSE], X[, 2, drop = FALSE]),
        error = function(e) {
            if (!grepl("missing value where TRUE/FALSE needed", conditionMessage(e))) stop(e)
            NA_real_
        }
    )
    # `C[i, j] = 1` for i -> j, already the orientation the benchmark scores.
    C <- matrix(0, 2, 2, dimnames = list(colnames(X), colnames(X)))
    if (!is.na(out) && out > 0) C[1, 2] <- 1
    if (!is.na(out) && out < 0) C[2, 1] <- 1
    C
}
"""


def code_dir() -> Path:
    """The authors' repository root, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' GRCI code is a measurement reference that the "
            "benchmark reads from a local download. Run benchmarks/start_grci.sh and export what "
            "it prints."
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


class StroblGRCI(RAdapter):
    """The authors' R implementation of GRCI's pair direction through the embedded R session.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "strobl.grci"
    package: str = "GRCI"
    package_version: str = COMMIT[:8]
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "grci"
    output_type: str = "dag"

    r_package: str = "RANN"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_strobl_grci"

    def setup(self) -> None:
        """Start R, source the code, and fit a tiny pair before timing."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        code_dir()
        for var in _THREAD_VARS:
            os.environ[var] = "1"
        super().setup()
        self.call(tiny_table()[:, :2])

    def params(self) -> dict[str, Any]:
        """The seed the wrapper sets and the versions fits ran with; the code has no arguments."""
        return {
            "commit": COMMIT,
            "source_sha256": source_sha256(),
            "seed": 0,
            "RANN": rsession.package_version("RANN"),
            "Rfast": rsession.package_version("Rfast"),
            "DirichletReg": rsession.package_version("DirichletReg"),
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' GRCI direction; return a 0/1 DAG adjacency. The only timed region."""
        return self.call(data).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
