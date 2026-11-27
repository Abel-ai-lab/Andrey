from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, TAIL
from andrey.core.score import make_score
from andrey.core.score_delta import DeltaBICScore
from andrey.search import _parallel_ges
from andrey.search._parallel_ges import cover_sources, one_edge_screen
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _adjacent(out: andrey.StructureOutput) -> np.ndarray:
    marks = out.structure.to_numpy()
    return (marks != 0) | (marks.T != 0)


def _chain_data(n: int = 2000, d: int = 12, seed: int = 0) -> np.ndarray:
    """A chain with a collider at every third node, so faithful data of one connected graph."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    for j in range(1, d):
        x[:, j] += 0.8 * x[:, j - 1] + (0.6 * x[:, j - 2] if j % 3 == 0 else 0.0)
    return x


def _canceling_data(n: int = 3000, seed: int = 2) -> np.ndarray:
    """x0 -> x1 -> x2 -> x3 and x0 -> x3, the direct effect canceling the path through x1 and x2.

    The columns x0 and x3 are uncorrelated although the graph joins them.
    """
    rng = np.random.default_rng(seed)
    a, b, c = -0.85, -0.75, 0.65
    x0 = rng.standard_normal(n)
    x1 = a * x0 + rng.standard_normal(n)
    x2 = b * x1 + rng.standard_normal(n)
    x3 = c * x2 - a * b * c * x0 + rng.standard_normal(n)
    return np.column_stack([x0, x1, x2, x3])


def _screen_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the cover phase no candidate, so ``fges`` ends after its screened forward pass."""
    monkeypatch.setattr(_parallel_ges, "cover_sources", lambda adj: [0] * adj.shape[0])


def test_fges_recovery():
    assert_recorded_output(
        andrey.fges(datagen.generate("gauss_5v"), penalty_discount=2.0), "FGES", case="gauss_5v"
    )


def test_fges_equals_ges_on_faithful_data():
    x = _chain_data()
    assert np.array_equal(andrey.fges(x).structure.to_numpy(), andrey.ges(x).structure.to_numpy())


def test_fges_screened_pass_adds_edges_only_between_pairs_whose_one_edge_graph_improves(
    monkeypatch: pytest.MonkeyPatch,
):
    _screen_only(monkeypatch)
    x = _chain_data(seed=1)
    d = x.shape[1]
    score = make_score(
        "bic", x, penalty_discount=1.0, ebic_gamma=1.0, equivalent_sample_size=1.0, categorical=None
    )
    rows = one_edge_screen(DeltaBICScore(score), d)
    admitted = np.array([[(rows[i] >> j) & 1 for j in range(d)] for i in range(d)], dtype=bool)
    assert np.array_equal(admitted, admitted.T)
    assert not (_adjacent(andrey.fges(x)) & ~admitted).any()


def test_fges_covers_a_pair_the_screen_rejects(monkeypatch: pytest.MonkeyPatch):
    x = _canceling_data()
    rows = one_edge_screen(
        DeltaBICScore(
            make_score(
                "bic",
                x,
                penalty_discount=1.0,
                ebic_gamma=1.0,
                equivalent_sample_size=1.0,
                categorical=None,
            )
        ),
        x.shape[1],
    )
    assert not (rows[0] >> 3) & 1  # the screen rejects x0 - x3
    full = andrey.fges(x)
    assert _adjacent(full)[0, 3]
    assert np.array_equal(full.structure.to_numpy(), andrey.ges(x).structure.to_numpy())
    _screen_only(monkeypatch)
    assert not _adjacent(andrey.fges(x))[0, 3]


def test_fges_cover_pass_does_not_worsen_the_score(monkeypatch: pytest.MonkeyPatch):
    for x in (_canceling_data(), _chain_data(seed=4)):
        full = andrey.fges(x).metadata["score"]
        with monkeypatch.context() as screened:
            _screen_only(screened)
            screen_only = andrey.fges(x).metadata["score"]
        assert full <= screen_only + 1e-9


def test_cover_sources_are_the_two_step_noncolliders():
    # 0 - 1 - 2 -> 3 <- 4: the collider at 3 does not make 2 and 4 candidates for each other.
    adj = np.zeros((5, 5), dtype=np.int8)

    def undirected(i: int, j: int) -> None:
        adj[i, j] = adj[j, i] = TAIL

    def directed(i: int, j: int) -> None:
        adj[i, j], adj[j, i] = TAIL, ARROW

    undirected(0, 1)
    undirected(1, 2)
    directed(2, 3)
    directed(4, 3)
    masks = cover_sources(adj)
    sources = [[m for m in range(5) if (mask >> m) & 1] for mask in masks]
    assert sources == [[2], [3], [0], [1], []]


@pytest.mark.parametrize("ebic_gamma", [0.0, 1.0])
def test_fges_screen_is_the_plain_bic_bump_of_each_pair(ebic_gamma: float):
    x = _chain_data(seed=2)
    n, d = x.shape
    penalty_discount = 2.0
    score = make_score(
        "bic",
        x,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=1.0,
        categorical=None,
    )
    rows = one_edge_screen(DeltaBICScore(score), d)
    admitted = np.array([[(rows[i] >> j) & 1 for j in range(d)] for i in range(d)], dtype=bool)
    r2 = np.corrcoef(x, rowvar=False) ** 2
    np.fill_diagonal(r2, 0.0)  # a column with itself is no pair
    bump = -n * np.log1p(-r2) - penalty_discount * np.log(n)
    assert np.array_equal(admitted, (bump > 0) & ~np.eye(d, dtype=bool))


@pytest.mark.parametrize(
    "x", [_chain_data(n=500, d=10), _canceling_data()], ids=["chain", "cancel"]
)
def test_fges_is_the_same_graph_in_worker_processes(monkeypatch: pytest.MonkeyPatch, x: np.ndarray):
    serial = andrey.fges(x).structure.to_numpy()
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")
    with andrey.config(n_jobs=2):
        pooled = andrey.fges(x).structure.to_numpy()
    assert np.array_equal(serial, pooled)


def test_fges_empty_for_independent_columns():
    x = np.random.default_rng(3).standard_normal((1000, 6))
    assert not _adjacent(andrey.fges(x)).any()
