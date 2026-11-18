"""Andrey SkewD solution adapters for cause-effect pairs.

Wrap ``andrey.skewd``, which decides the direction of a two-column table under the skew-normal
location-scale noise model and returns a two-node DAG: ``0 -> 1``, ``1 -> 0``, or no edge.
:class:`AndreySkewD` runs the independence criterion (SkewD-IT in Klippert and Marx 2026) with
``alpha=None``, the direction with the larger p-value; :class:`AndreySkewDLikelihood` runs the
likelihood criterion (SkewD-LL), the direction with the larger likelihood. Each fit standardizes
both columns first, as the paper does. The timed fit runs the method's defaults, which are the
budgets of the paper's runs.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_loci import _andrey_version

#: The budgets of the untimed warm-up fit in ``setup``; at the defaults it costs as much as a timed
#: fit.
WARMUP = {"tuning_iter": 1, "max_iter": 5}


class AndreySkewD:
    """Solution adapter for Andrey's SkewD under the independence criterion.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter` and
    :class:`andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "andrey.skewd.independence.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "skewd"
    output_type: str = "dag"
    criterion: str = "independence"

    def setup(self) -> None:
        """Fit a tiny pair before timing, so lazy imports and the warnings stay out of the fit.

        The warm-up fit runs :data:`WARMUP`'s budgets, which reach every code path of a fit.
        """
        import warnings

        import andrey
        from andrey_bench.adapters.fcm_warm import tiny_table

        with warnings.catch_warnings():
            # The warm-up budgets end every search at its limit.
            warnings.simplefilter("ignore", andrey.SearchLimitWarning)
            self.fit(tiny_table()[:, :2], WARMUP)

    def params(self) -> dict[str, Any]:
        """The criterion, and ``alpha=None``: the larger p-value, or no level at all."""
        return {"alpha": None, "criterion": self.criterion}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey SkewD; return the two-node DAG's endpoint marks. The only timed region."""
        import warnings

        import andrey

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.skewd(np.asarray(data, dtype=np.float64), **{**self.params(), **params})
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreySkewDLikelihood(AndreySkewD):
    """Solution adapter for Andrey's SkewD under the likelihood criterion."""

    name: str = "andrey.skewd.likelihood.numpy"
    criterion: str = "likelihood"
