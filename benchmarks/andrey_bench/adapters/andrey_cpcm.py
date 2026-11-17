"""Andrey CPCM solution adapters for cause-effect pairs and graphs.

Wrap ``andrey.cpcm``, which decides the direction of a two-column table under a conditionally
parametric causal model, and ``andrey.cpcm_graph``, which searches the DAG of more columns with a
per-edge penalty. :class:`AndreyCPCM` runs ``cpcm(data, alpha=None, family=..., seed=...)``: with
no level, the direction with the larger p-value, as the authors' forced estimate decides.
:class:`AndreyCPCMGraph` runs ``cpcm_graph(data, family=..., edge_penalty=..., seed=...)``. Each
returns the structure's endpoint marks.

``family`` is one family name, or a list with one name per column. The names are Andrey's:
``gaussian``, ``gaussian_fixed_scale``, ``exponential``, ``pareto``, ``poisson``, ``gamma``,
``gamma_fixed_shape``, ``gumbel``, ``gumbel_fixed_scale``, and ``negative_binomial``; those of the
authors' R code that match them are in :data:`~andrey_bench.adapters.bodik_cpcm.R_FAMILY`. A list
whose entries depend on the table (for example, which column is the effect) is not a solution's
setting: call :func:`fit_pair` or :func:`fit_graph` with the table's own list.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

# The family of the warm-up fits: it takes any real column and is the cheapest to fit.
WARM_FAMILY = "gaussian_fixed_scale"


def _andrey_version() -> str:
    """The installed Andrey version (distribution metadata, else the module attribute)."""
    from importlib import metadata

    try:
        return metadata.version("andrey")
    except metadata.PackageNotFoundError:  # editable/dev checkout with no dist metadata
        import andrey

        return getattr(andrey, "__version__", "unknown")


def _call(function: str, data: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Call ``andrey.<function>`` and return its structure's endpoint marks."""
    import warnings

    import andrey

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        out = getattr(andrey, function)(np.asarray(data, dtype=np.float64), **kwargs)
    return np.asarray(out.structure.to_numpy())


def fit_pair(data: np.ndarray, family: str | Sequence[str], seed: int = 0) -> np.ndarray:
    """Andrey's forced CPCM direction of a pair, as the two-node DAG's endpoint marks."""
    return _call("cpcm", data, alpha=None, family=family, seed=seed)


def fit_graph(
    data: np.ndarray, family: str | Sequence[str], edge_penalty: float = 2.0, seed: int = 0
) -> np.ndarray:
    """Andrey's CPCM graph search of a table, as the DAG's endpoint marks."""
    return _call("cpcm_graph", data, family=family, edge_penalty=edge_penalty, seed=seed)


class AndreyCPCM:
    """Solution adapter for Andrey's CPCM on pairs, with the forced decision (``alpha=None``).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "cpcm"
    output_type: str = "dag"

    def __init__(self, family: str | Sequence[str] = "gaussian", seed: int = 0) -> None:
        self.family = family if isinstance(family, str) else list(family)
        self.seed = seed
        label = self.family if isinstance(self.family, str) else "_".join(self.family)
        self.name = f"andrey.cpcm.{label}.numpy"

    def setup(self) -> None:
        """Fit a tiny pair before timing, so lazy imports and the warning stay out of the fit.

        The pair is fitted in the Gaussian family, which takes any real column, so the warm-up
        holds for every family of the solution; all families share the same modules.
        """
        from andrey_bench.adapters.fcm_warm import tiny_table

        fit_pair(tiny_table()[:, :2], WARM_FAMILY, self.seed)

    def params(self) -> dict[str, Any]:
        """The family, the seed, and ``alpha=None``: the larger p-value, or no level at all."""
        return {"alpha": None, "family": self.family, "seed": self.seed}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey CPCM; return the two-node DAG's endpoint marks. The only timed region."""
        merged = {**self.params(), **params}
        return _call("cpcm", data, **merged)

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class AndreyCPCMGraph(AndreyCPCM):
    """Solution adapter for Andrey's CPCM graph search, with the per-edge penalty."""

    def __init__(
        self, family: str | Sequence[str] = "gaussian", edge_penalty: float = 2.0, seed: int = 0
    ) -> None:
        super().__init__(family, seed)
        self.edge_penalty = edge_penalty
        self.name = self.name.replace("andrey.cpcm.", "andrey.cpcm_graph.", 1)

    def setup(self) -> None:
        """Fit a tiny table before timing, so lazy imports and the warning stay out of the fit.

        The table holds negative values, so the fit uses the Gaussian family whatever the solution's.
        """
        from andrey_bench.adapters.fcm_warm import tiny_table

        fit_graph(tiny_table(), WARM_FAMILY, self.edge_penalty, self.seed)

    def params(self) -> dict[str, Any]:
        """The family, the per-edge penalty, and the seed."""
        return {"family": self.family, "edge_penalty": self.edge_penalty, "seed": self.seed}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey's CPCM graph search; return the DAG's endpoint marks. The timed region."""
        merged = {**self.params(), **params}
        return _call("cpcm_graph", data, **merged)
