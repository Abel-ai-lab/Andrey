"""The majority vote tests its conditioning subsets in bounded chunks, with unchanged decisions.

A test with no canonical route votes pair by pair, ``_VOTE_CHUNK`` subsets to a call; a FisherZ,
StudentT, NormalScores, or degenerate Gaussian test votes on stacks of canonical queries, whatever
the cap on a stack and on a chunk.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint import _skeleton
from andrey.core import CIRCLE
from andrey.core.ci import make_indep_test


def _star_triple(k: int) -> np.ndarray:
    """A circle graph with the triple 0 - 1 - 2, each endpoint adjacent to ``k - 1`` other nodes."""
    n = 3 + 2 * (k - 1)
    marks = np.zeros((n, n), dtype=np.int8)
    edges = [(0, 1), (2, 1)]
    edges += [(0, 3 + i) for i in range(k - 1)]
    edges += [(2, 3 + k - 1 + i) for i in range(k - 1)]
    for u, v in edges:
        marks[u, v] = marks[v, u] = CIRCLE
    return marks


class _Recording:
    """A Fisher-Z test whose batched calls record how many conditioning sets each one receives."""

    def __init__(self, x: np.ndarray):
        self._test = make_indep_test("fisherz", x)
        self.sizes: list[int] = []

    def __call__(self, a, c, subset=()):
        return self._test(a, c, subset)

    def batched_call(self, a, c, subsets):
        subsets = list(subsets)
        self.sizes.append(len(subsets))
        return self._test.batched_call(a, c, subsets)


def _vote(monkeypatch, chunk: int, x: np.ndarray, marks: np.ndarray):
    monkeypatch.setattr(_skeleton, "_VOTE_CHUNK", chunk)
    test = _Recording(x)
    sepsets: dict = {}
    ambiguous = _skeleton.vote_triples(marks.copy(), test, 0.05, sepsets, "majority")
    return test.sizes, ambiguous, sepsets


@pytest.mark.parametrize("chunk", [1, 4, 7])
def test_no_call_receives_more_than_the_chunk(chunk, monkeypatch):
    marks = _star_triple(4)  # each endpoint has 4 neighbors: 16 subsets per endpoint
    x = np.random.default_rng(0).standard_normal((300, marks.shape[0]))
    sizes, _, _ = _vote(monkeypatch, chunk, x, marks)
    assert sizes and max(sizes) <= chunk
    assert max(sizes) == chunk  # the heavy triple fills whole chunks


@pytest.mark.parametrize("seed", range(4))
def test_chunked_decisions_equal_one_call(seed, monkeypatch):
    marks = _star_triple(5)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((200, marks.shape[0]))
    x[:, 1] += x[:, 0] + x[:, 2]  # a collider at node 1, so the vote has separating sets to count
    _, ambiguous, sepsets = _vote(monkeypatch, 4, x, marks)
    _, whole_ambiguous, whole_sepsets = _vote(monkeypatch, 10**9, x, marks)
    assert (ambiguous, sepsets) == (whole_ambiguous, whole_sepsets)


@pytest.mark.parametrize("engine", ["fci", "gfci"])
def test_public_majority_results_do_not_depend_on_the_stack_or_the_chunk(engine, monkeypatch):
    """The stacked vote gives the same graph at every cap on a stack and on a chunk, and with the
    pair-by-pair vote."""
    dataset = data.sample_scm(d=12, n=600, density=3.0, seed=3, scale="raw")
    fit = getattr(andrey, engine)
    results = []
    for stack, chunk, route in (
        (1, 2, True),
        (4096, 5, True),
        (None, None, True),
        (None, None, False),
    ):
        if stack is not None:
            monkeypatch.setattr(_skeleton, "_STACK_BYTES", stack)
            monkeypatch.setattr(_skeleton, "CHUNK", chunk)
        if not route:
            monkeypatch.setattr(_skeleton, "_canonical_route", lambda test: None)
        results.append(fit(dataset.data, collider_rule="majority").structure.to_numpy())
        monkeypatch.undo()
    for result in results[1:]:
        np.testing.assert_array_equal(results[0], result)
