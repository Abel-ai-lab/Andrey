"""DAS adapter for the authors' code in dodiscover (Montagna et al., 2023), in its own environment.

The paper's code is ``dodiscover.toporder.DAS`` (MIT), which its footnote names; the adapter runs it
at the commit the NoGAM adapter pins (:data:`~andrey_bench.adapters.dodiscover_nogam.COMMIT`), in
the environment ``benchmarks/start_stein.sh`` builds. A fit is the class's ``learn_graph`` with the
paper's values where the class's defaults differ: the edge screen's Welch tests at ``0.01`` (the
paper's Sect. 3.3 and Appendix B; the class's ``das_cutoff`` defaults to its ``alpha``) and CAM
pruning at ``0.001``, the cutoff of the SCORE experiments the paper follows (the class's default
is ``0.05``). The screen keeps at most ``max_parents = 20`` candidates per variable, the paper's
``K``, and the class's ``min_parents = 5`` largest entries without a test. The Stein estimators run
at ``eta_G = eta_H = 0.001``. The data go in as a data frame of ``float64`` columns.

The code estimates the score's Jacobian once, on all the variables, and ranks the absolute median
of each entry; the paper's Algorithm 1 reads each leaf's row at the step that removes it and ranks
the mean absolute entry. The adapter leaves that as the code has it.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.dodiscover_nogam import COMMIT

_DEFAULTS: dict[str, Any] = {
    "eta_G": 0.001,
    "eta_H": 0.001,
    "alpha": 0.001,
    "das_cutoff": 0.01,
    "n_splines": 10,
    "splines_degree": 3,
    "min_parents": 5,
    "max_parents": 20,
}


class DodiscoverDAS:
    """The authors' DAS: SCORE's order, the Hessian edge screen, then CAM pruning.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "dodiscover.das"
    package: str = "dodiscover"
    package_version: str = COMMIT[:8]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "das"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_STEIN_PYTHON"

    def setup(self) -> None:
        """Fit a tiny table before timing, so imports and first-call costs stay out of the fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), self.params())

    def params(self) -> dict[str, Any]:
        """The paper's screen level and pruning cutoff, the class's other defaults, the commit."""
        return {**_DEFAULTS, "commit": COMMIT}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' DAS and return a 0/1 DAG adjacency. The only timed region."""
        import networkx as nx
        import pandas as pd
        from dodiscover import make_context  # ty: ignore[unresolved-import]
        from dodiscover.toporder import DAS  # ty: ignore[unresolved-import]

        frame = pd.DataFrame(np.asarray(data, dtype=np.float64))
        model = DAS(**{k: v for k, v in params.items() if k != "commit"})
        model.learn_graph(frame, make_context().variables(data=frame).build())
        nodes = list(frame.columns)
        return nx.to_numpy_array(model.graph_, nodelist=nodes).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
