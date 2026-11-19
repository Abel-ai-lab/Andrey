"""Andrey DAS solution adapter.

Wraps ``andrey.das`` at its defaults: SCORE's causal order, each leaf's candidate parents screened
by Welch tests on its row of the Stein estimate of the score's Jacobian (level ``0.01``, at most 20
per variable), then smooth-term tests at ``alpha = 0.001`` (Montagna et al., 2023). The adapter
keeps the DAG.

The fits run in the interpreter named by ``$ANDREY_BENCH_CAM_PYTHON``, an environment holding the
build under test, like the other Andrey solutions of the Stein-score lane.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_cam import _andrey_version

_DEFAULTS: dict[str, Any] = {"alpha": 0.001, "max_parents": 20}


class AndreyDAS:
    """Solution adapter for Andrey's DAS.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.das.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "das"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_CAM_PYTHON"

    def setup(self) -> None:
        """Fit a tiny table before timing, so lazy imports stay out of the first timed fit."""
        from andrey_bench.adapters.fcm_warm import tiny_table

        self.fit(tiny_table(), {})

    def params(self) -> dict[str, Any]:
        """Andrey's defaults: the pruning level and the screen's parent cap."""
        return dict(_DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey DAS; return the DAG's endpoint marks. The only timed region."""
        import andrey

        out = andrey.das(np.asarray(data, dtype=np.float64), **{**_DEFAULTS, **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
