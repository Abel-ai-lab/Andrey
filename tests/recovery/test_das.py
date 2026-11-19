"""DAS recovers nonlinear additive-noise graphs with Gaussian noise: fixed cases, seeded ground
truth, and a recorded output.

The fixed cases have one certain answer: independent columns give no edge, a cause with a
nonlinear effect points to it in either column order, and a chain of two nonlinear effects is found
with its order. The ground truth is the simulation of the method's paper at test size, links drawn
from a Gaussian process on a random graph, with enough variables that the edge screen tests some
candidates: the errors stay within half the empty graph's, the edges follow the causal order, and
the order is SCORE's. The recorded output pins the graph and the causal order on one fixed case.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.metrics import shd
from tests.recovery.helpers import datagen, gp
from tests.recovery.helpers.harness import assert_recorded_output

N = 400


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, kind in out.structure.oriented_edges() if kind == "directed"}


def _pair(seed: int) -> np.ndarray:
    """``x -> y`` with ``y = sin(2 x)`` plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    return np.column_stack([x, np.sin(2 * x) + 0.3 * rng.standard_normal(N)])


def _chain(seed: int) -> np.ndarray:
    """``x0 -> x1 -> x2`` with ``x1 = sin(2 x0)`` and ``x2 = x1**2``, each plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(N)
    x1 = np.sin(2 * x0) + 0.5 * rng.standard_normal(N)
    return np.column_stack([x0, x1, x1**2 + 0.5 * rng.standard_normal(N)])


def test_independent_columns_give_no_edges():
    out = andrey.das(np.random.default_rng(0).standard_normal((N, 4)))
    assert out.structure.oriented_edges() == []
    assert sorted(out.ordering) == [0, 1, 2, 3]


@pytest.mark.parametrize("reverse", [False, True], ids=["cause first", "effect first"])
def test_a_nonlinear_cause_points_to_its_effect(reverse):
    X = _pair(1)
    out = andrey.das(X[:, ::-1] if reverse else X)
    cause, effect = (1, 0) if reverse else (0, 1)
    assert _edges(out) == {(cause, effect)}
    assert out.ordering == (cause, effect)


def test_a_chain_of_two_nonlinear_effects_is_found():
    out = andrey.das(_chain(2))
    assert _edges(out) == {(0, 1), (1, 2)}
    assert out.ordering == (0, 1, 2)


def test_a_graph_of_the_papers_simulation_is_recovered():
    """Gaussian-process links on a random graph of 10 variables and 10 edges, 1,000 rows (the
    paper's): DAS's errors stay within half the empty graph's, its edges follow its causal order,
    and the order is SCORE's."""
    data, truth = gp.sample(10, 1000, 10, seed=0)
    out = andrey.das(data)
    assert shd(out.structure, truth) <= len(truth.oriented_edges()) / 2
    order = list(out.ordering)
    assert all(order.index(i) < order.index(j) for i, j in _edges(out))
    assert out.ordering == andrey.score_matching(data).ordering


def test_the_recorded_output_repeats():
    out = andrey.das(datagen.generate("rational_chain_4v"))
    assert_recorded_output(out, "DAS", case="rational_chain_4v")
