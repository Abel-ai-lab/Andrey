"""Fast Greedy Equivalence Search, the engine behind :func:`andrey.fges`.

FGES (Ramsey et al. 2017) is GES with one-edge faithfulness: its first forward phase adds an edge
between two variables only when the graph of that single edge scores better than the empty graph,
and a second pass then covers the unshielded noncolliders (steps fGES.9-11 of the pseudocode in
Supplement 1). The search is :func:`andrey.search.ges.ges`, with its cached candidate scans and its
worker processes; the first phase reads the screen that
:func:`~andrey.search._parallel_ges.one_edge_screen` builds, so each target scans the sources that
pass it, and the second the sources of :func:`~andrey.search._parallel_ges.cover_sources`. The
algorithm, its settings, and its reference are described on :func:`andrey.fges`.
"""

from __future__ import annotations

import numpy as np

from andrey.core import GraphStructure
from andrey.core.knowledge import Constraints
from andrey.search.ges import ges


def fges(
    data: np.ndarray,
    *,
    score_func: str = "bic",
    penalty_discount: float = 1.0,
    ebic_gamma: float = 1.0,
    equivalent_sample_size: float = 1.0,
    max_parents: float | None = None,
    categorical: np.ndarray | None = None,
    constraints: Constraints | None = None,
    workers: int | None = None,
    one_edge_faithfulness: bool = True,
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Greedy Equivalence Search with one-edge faithfulness.

    The parameters are those of :func:`andrey.search.ges.ges`; the first forward phase scans only
    the pairs whose one-edge graph improves the score, and a second forward and backward pass
    covers the unshielded noncolliders. With ``one_edge_faithfulness=False``, a third forward and
    backward pass scans every pair on the graph the first two reached. Returns the CPDAG
    (``kind="cpdag"``) and its total score.
    """
    return ges(
        data,
        score_func=score_func,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        max_parents=max_parents,
        categorical=categorical,
        constraints=constraints,
        workers=workers,
        one_edge=True,
        all_pairs=not one_edge_faithfulness,
    )
