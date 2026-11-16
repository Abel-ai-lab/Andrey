"""Local ISA-LiNG adapter for the authors' code (Dai et al., 2024), run unmodified in its own
environment.

The code is ``MarkDana/Local-LiNG-Discovery`` at commit :data:`COMMIT` (MIT license), a measurement
reference: ``benchmarks/start_local_ling.sh`` clones it at that commit and builds the environment
it runs in; the benchmark imports it from the directory ``$ANDREY_BENCH_LOCAL_LING`` names.

A fit does what the repository's two examples do, on the data as given: ``set_random_seed(1)``,
``estimate_markov_blanket(X, target)`` (a Lasso of the target on the other columns at ``alpha =
0.001``, keeping the coefficients above 0.01), then ``local_isa_ling`` on the target's column
followed by the blanket's (``list`` of the returned set, as the examples take it), with the code's
default thresholds. ``params["blanket"]``, when given, replaces the estimated blanket (the oracle
arm). ``params["acyclic"]`` picks the post-processing: ``"acyclic_sink"`` (the paper's Appendix C)
when true, else ``"block_diagonal"`` (Algorithm 1), whose answer is the code's most stable model.

The code's Algorithm 1 tries every one of the ``m!`` row permutations of the local demixing matrix,
so the adapter runs it only up to :data:`MAX_LOCAL` local variables and otherwise returns a
``NOT_RUN`` answer, which the lane records as not run, never as an empty graph.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.local_lane import TARGET_PARAM
from andrey_bench.local_ling_models import NOT_RUN, pack_answer, unpack_answer

#: Names the directory holding the cloned code.
CODE_ENV = "ANDREY_BENCH_LOCAL_LING"

#: The pinned commit of ``MarkDana/Local-LiNG-Discovery``; ``start_local_ling.sh`` checks it out.
COMMIT = "d7351d5"

#: The most local variables (the target and its blanket) the code's Algorithm 1 is run on:
#: ``10! = 3,628,800`` permutations, about 10 to 20 minutes on one core; ``11!`` is past an hour.
MAX_LOCAL = 10

_REQUIRED = ("local_isa_ling/local_isa_ling.py", "utils/markov_blanket.py", "utils/utils.py")


def _code_dir() -> Path:
    raw = os.environ.get(CODE_ENV, "").strip()
    directory = Path(raw)
    if not raw or not all((directory / name).is_file() for name in _REQUIRED):
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} does not hold Local-LiNG-Discovery; run "
            "benchmarks/start_local_ling.sh"
        )
    return directory


def _modules() -> tuple[Any, Any, Any]:
    """The code's ``local_isa_ling``, ``estimate_markov_blanket``, and ``set_random_seed``."""
    directory = str(_code_dir())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from local_isa_ling.local_isa_ling import local_isa_ling
    from utils.markov_blanket import estimate_markov_blanket
    from utils.utils import set_random_seed

    return local_isa_ling, estimate_markov_blanket, set_random_seed


class DaiLocalISALiNG:
    """The authors' local ISA-LiNG at one target, with its Lasso blanket or a given one.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.local_lane.LocalAdapter`.
    """

    package: str = "local-ling-discovery"
    package_version: str = COMMIT
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "local_isa_ling"
    output_type: str = "digraph"
    local: bool = True
    python_env: str = "ANDREY_BENCH_LOCAL_LING_PYTHON"

    def __init__(self, oracle: bool = False) -> None:
        self.oracle = oracle
        self.name = "local_isa_ling.authors" + (".oracle" if oracle else "")

    def setup(self) -> None:
        """Import the code (and compile its Numba kernels' imports) outside fit timing."""
        _modules()

    def params(self) -> dict[str, Any]:
        """The code's defaults; the lane adds ``acyclic`` per setting."""
        return {}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Answer at ``params["target"]``; the timed region."""
        local_isa_ling, estimate_markov_blanket, set_random_seed = _modules()
        X = np.asarray(data, dtype=np.float64)
        t = int(params[TARGET_PARAM])
        set_random_seed(1)
        if params.get("blanket") is None:
            blanket = list(estimate_markov_blanket(X, t))
        else:
            blanket = [int(j) for j in params["blanket"]]
        local = [t, *(int(j) for j in blanket)]
        weights = np.zeros((X.shape[1], X.shape[1]))
        if params.get("acyclic", False):
            set_random_seed(1)
            dag = local_isa_ling(X[:, local], target=0, postprocess_type="acyclic_sink")["dag"]
            models = 1
        elif len(local) > MAX_LOCAL:
            return pack_answer(weights, blanket, code=NOT_RUN, m=len(local))
        else:
            set_random_seed(1)
            res = local_isa_ling(X[:, local], target=0, postprocess_type="block_diagonal")
            dag, models = res["dcg"], len(res["dcgs"])
        for c, p in zip(*np.nonzero(dag)):
            weights[local[p], local[c]] = dag[c, p]
        return pack_answer(weights, blanket, m=len(local), models=models)

    def to_structure(self, native: np.ndarray) -> Any:
        """The answer's edges as a ``digraph`` ``GraphStructure`` (2-cycles allowed); untimed."""
        from andrey import GraphStructure
        from andrey.api._marks import directed_marks

        weights = unpack_answer(native)["weights"]
        return GraphStructure.from_numpy(directed_marks(weights.T), kind="digraph")
