"""Andrey's local ISA-LiNG, and global LiNGAM read at a target, for the local LiNG lane.

``AndreyLocalISALiNG`` runs ``andrey.local_isa_ling`` at the lane's target (``params["target"]``)
with ``acyclic`` as the lane passes it. With ``params["blanket"]`` (the oracle arm) it runs the
method's engine, ``andrey.lingam.local.local_isa_ling``, on that blanket instead of the one the
method learns, and reports the first model's weights in the data's units as the facade does.

``AndreyGlobalLiNGAM`` runs ``andrey.direct_lingam`` or ``andrey.ica_lingam`` on every column; the
lane reads the answer's local structure at the target, so one fit answers for every target and its
cost is the whole fit's.

Both return :func:`~andrey_bench.local_ling_models.pack_answer` of ``weights[i, j]`` for the edge
``i -> j``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version
from andrey_bench.local_lane import TARGET_PARAM
from andrey_bench.local_ling_models import pack_answer, unpack_answer


class AndreyLocalISALiNG:
    """``andrey.local_isa_ling`` with its own blanket, or the engine on a given one.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.local_lane.LocalAdapter`.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "local_isa_ling"
    output_type: str = "digraph"
    local: bool = True

    def __init__(self, oracle: bool = False, acyclic: bool | None = None) -> None:
        self.oracle = oracle
        self.acyclic = acyclic
        suffix = ".oracle" if oracle else ""
        if acyclic is False:
            suffix += ".alg1"
        self.name = "andrey.local_isa_ling" + suffix

    def setup(self) -> None:
        """Import Andrey and the engine before timing."""
        import andrey.lingam.local  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The method's defaults; the lane adds ``acyclic`` per setting unless fixed here."""
        return {} if self.acyclic is None else {"acyclic": self.acyclic}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Answer at ``params["target"]``; the timed region."""
        import andrey
        from andrey.lingam.local import local_isa_ling

        X = np.asarray(data, dtype=np.float64)
        t = int(params[TARGET_PARAM])
        acyclic = bool(params.get("acyclic", False))
        if params.get("blanket") is None:
            out = andrey.local_isa_ling(X, target=t, acyclic=acyclic)
            return pack_answer(
                out.weighted_adjacency,
                out.metadata["markov_blanket"],
                m=1 + len(out.metadata["markov_blanket"]),
                models=len(out.metadata["equivalent_models"]),
                fallback=acyclic and "subspaces" in out.metadata,
            )
        blanket = [int(j) for j in params["blanket"]]
        fit = local_isa_ling(X, t, acyclic=acyclic, blanket=blanket)
        std = X.std(axis=0, ddof=1)
        weights = np.zeros((X.shape[1], X.shape[1]))
        first = fit.models[0] if fit.models else None
        for (k, j), w in ({} if first is None else first.edges).items():
            weights[k, j] = w * std[j] / std[k]
        return pack_answer(
            weights,
            blanket,
            m=1 + len(blanket),
            models=len(fit.models),
            fallback=acyclic and first is not None and bool(first.subspaces),
        )

    def to_structure(self, native: np.ndarray) -> Any:
        """The answer's edges as a ``digraph`` ``GraphStructure`` (2-cycles allowed); untimed."""
        from andrey import GraphStructure
        from andrey.api._marks import directed_marks

        weights = unpack_answer(native)["weights"]
        return GraphStructure.from_numpy(directed_marks(weights.T), kind="digraph")


class AndreyGlobalLiNGAM:
    """``andrey.direct_lingam`` (``measure="pwling"``) or ``andrey.ica_lingam`` (seed 0) on every
    column.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`; ``local = False``.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    output_type: str = "dag"
    local: bool = False

    def __init__(self, algorithm: str = "direct_lingam") -> None:
        if algorithm not in ("direct_lingam", "ica_lingam"):
            raise ValueError(f"unknown algorithm {algorithm!r}")
        self.algorithm = algorithm
        self.name = f"andrey.{algorithm}"

    def setup(self) -> None:
        """Import Andrey before timing."""
        import andrey  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The facade's defaults, with ``ica_lingam``'s seed fixed."""
        return {"measure": "pwling"} if self.algorithm == "direct_lingam" else {"seed": 0}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Fit every column; the timed region."""
        import andrey

        X = np.asarray(data, dtype=np.float64)
        out = getattr(andrey, self.algorithm)(X, **params)
        return pack_answer(np.asarray(out.weighted_adjacency, dtype=np.float64), None)

    def to_structure(self, native: np.ndarray) -> Any:
        """The DAG of the answer's edges; untimed."""
        from andrey import GraphStructure

        weights = unpack_answer(native)["weights"]
        return GraphStructure.from_numpy((weights != 0).astype(np.int8), kind="dag")
