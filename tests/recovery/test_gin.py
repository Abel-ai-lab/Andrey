"""Test GIN on fixed cases, on a planted measurement model, and against its recorded output."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import andrey
from andrey.core import LATENT, OBSERVED, GraphStructure, StructureOutput
from andrey.core.ties import tie_ranks
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

_CASE = "measurement_6v"


@pytest.fixture(scope="module")
def fit() -> tuple[np.ndarray, StructureOutput]:
    data = datagen.generate(_CASE)
    return data, andrey.gin(data)


def test_gin_recovers_the_planted_clusters_and_their_order(fit) -> None:
    _, out = fit
    planted = datagen.clusters(_CASE)  # earliest latent first

    assert out.metadata["algorithm"] == "gin"
    assert len(out.metadata["clusters"]) == 2
    assert {frozenset(c) for c in out.metadata["clusters"]} == {frozenset(c) for c in planted}
    assert [sorted(c) for c in out.metadata["cluster_order"]] == planted

    s = out.structure
    assert isinstance(s, GraphStructure)
    assert s.kind == "dag"
    assert s.n_nodes == 8  # six observed indicators + two discovered latents
    assert list(s.node_types) == [OBSERVED] * 6 + [LATENT] * 2
    assert s.labels == ("X1", "X2", "X3", "X4", "X5", "X6", "L1", "L2")
    latent_edges = [(6, child, "directed") for child in planted[0]]
    latent_edges += [(6, 7, "directed")]
    latent_edges += [(7, child, "directed") for child in planted[1]]
    assert sorted(s.oriented_edges(index=True)) == sorted(latent_edges)


def test_gin_orders_a_chain_of_three_latents() -> None:
    case = "measurement_chain_9v"
    planted = datagen.clusters(case)  # earliest latent first
    out = andrey.gin(datagen.generate(case))
    assert {frozenset(c) for c in out.metadata["clusters"]} == {frozenset(c) for c in planted}
    assert [sorted(c) for c in out.metadata["cluster_order"]] == planted


def test_gin_finds_the_same_latents_for_reversed_columns() -> None:
    """Reversing the columns relabels the clusters and the latent order and changes nothing else."""
    data = datagen.generate("measurement_chain_9v")
    last = data.shape[1] - 1
    out = andrey.gin(data).metadata
    reversed_out = andrey.gin(data[:, ::-1]).metadata

    def back(groups: list[list[int]]) -> list[list[int]]:
        return [sorted(last - i for i in group) for group in groups]

    assert sorted(back(reversed_out["clusters"])) == sorted(out["clusters"])
    assert back(reversed_out["cluster_order"]) == out["cluster_order"]
    assert sorted(back(reversed_out["unordered"])) == sorted(out["unordered"])


def _latent_model(seed: int, hidden_weight: float, n_samples: int) -> np.ndarray:
    """Three latents, each the hidden cause times a weight near ``hidden_weight`` plus uniform
    noise, measured by three indicators each, the columns in an order drawn from ``seed``. A
    strong hidden cause leaves no latent a root, so the latents stay unordered."""
    rng = np.random.default_rng(seed)
    hidden = rng.uniform(-1.0, 1.0, n_samples)
    weights = hidden_weight + rng.uniform(-0.5, 0.5, 3)
    latents = weights * hidden[:, None] + rng.uniform(-1.0, 1.0, (n_samples, 3))
    loadings = rng.uniform(0.7, 1.3, 9) * rng.choice([-1.0, 1.0], 9)
    x = np.repeat(latents, 3, axis=1) * loadings + 0.3 * rng.uniform(-1.0, 1.0, (n_samples, 9))
    return x[:, rng.permutation(9)]


def _marks_moved_back(data: np.ndarray, order: np.ndarray) -> np.ndarray:
    """GIN's marks on the columns of ``data`` in ``order``, each observed node moved back to its
    column and the latents left in place."""
    marks = andrey.gin(data[:, order]).structure.to_numpy()
    back = np.concatenate([np.argsort(order), np.arange(len(order), len(marks))])
    return marks[np.ix_(back, back)]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(datagen.generate("measurement_hidden_cause_6v"), id="hidden_cause_6v"),
        pytest.param(_latent_model(1, 1.0, 500), id="ordered_9v"),
        pytest.param(_latent_model(1, 2.0, 800), id="unordered_9v"),
    ],
)
def test_gin_graph_relabels_under_every_column_order(data: np.ndarray) -> None:
    """Reordering the columns moves the observed nodes and changes nothing else: the latents keep
    their numbering, the ordered ones by the order and the unordered ones by their lowest-ranked
    column."""
    d = data.shape[1]
    orders = [np.arange(d)[::-1], *(np.random.default_rng([d, k]).permutation(d) for k in range(3))]
    want = _marks_moved_back(data, np.arange(d))
    for order in orders:
        np.testing.assert_array_equal(_marks_moved_back(data, order), want)


def test_gin_numbers_unordered_latents_by_position_under_the_position_tie_break() -> None:
    """Under ``tie_break="position"`` the unordered latents follow their smallest column index, so
    putting the other cluster's columns first renumbers them."""
    data = datagen.generate("measurement_hidden_cause_6v")
    out = andrey.gin(data).metadata
    assert out["cluster_order"] == [] and len(out["unordered"]) == 2
    first, second = out["unordered"]
    with andrey.config(tie_break="position"):
        marks = [
            _marks_moved_back(data, np.array(order)) for order in (first + second, second + first)
        ]
    assert (marks[0] != marks[1]).any()


def test_gin_matches_its_recorded_output(fit) -> None:
    _, out = fit
    assert_recorded_output(out, "GIN", case=_CASE)


def test_gin_finds_one_latent_behind_three_indicators() -> None:
    rng = np.random.default_rng(0)
    latent = rng.uniform(-1.0, 1.0, 500)
    data = np.column_stack(
        [w * latent + 0.1 * rng.uniform(-1.0, 1.0, 500) for w in (0.8, 1.2, -0.9)]
    )
    out = andrey.gin(data)
    assert out.metadata["clusters"] == [[0, 1, 2]]
    assert sorted(out.structure.oriented_edges(index=True)) == [
        (3, i, "directed") for i in range(3)
    ]


def test_gin_needs_a_variable_outside_a_cluster() -> None:
    """Two columns leave no variable to test a pair against, so no latent is found."""
    rng = np.random.default_rng(0)
    latent = rng.uniform(-1.0, 1.0, 500)
    data = np.column_stack([latent + 0.1 * rng.uniform(-1.0, 1.0, 500) for _ in range(2)])
    out = andrey.gin(data)
    assert out.metadata["clusters"] == []
    assert out.structure.n_nodes == 2
    assert out.structure.oriented_edges(index=True) == []


def test_gin_leaves_latents_unordered_when_no_root_passes() -> None:
    """Two latents share a hidden cause, so neither is a root and the order stays empty."""
    case = "measurement_hidden_cause_6v"
    planted = datagen.clusters(case)
    data = datagen.generate(case)
    out = andrey.gin(data)
    assert {frozenset(c) for c in out.metadata["clusters"]} == {frozenset(c) for c in planted}
    assert out.metadata["cluster_order"] == []
    ranks = tie_ranks(data)
    unordered = sorted(planted, key=lambda c: ranks[c].min())  # by lowest-ranked column
    assert out.metadata["unordered"] == unordered
    latent_edges = [(6, child, "directed") for child in unordered[0]]
    latent_edges += [(7, child, "directed") for child in unordered[1]]
    assert sorted(out.structure.oriented_edges(index=True)) == sorted(latent_edges)


def test_gin_root_is_the_passing_candidate_with_the_highest_p_value(monkeypatch) -> None:
    """A root passes the test against every other cluster; ties go to the highest lowest p-value.

    The stub gives each root test a p-value by the candidate and the other cluster. Cluster
    ``[4, 5]`` passes with lowest p-value 0.3 and ``[2, 3]`` with 0.6, so ``[2, 3]`` is the root;
    then ``[0, 1]`` and ``[4, 5]`` both fail, so the ordering stops.
    """
    engine = importlib.import_module("andrey.latent.gin")

    first_step = {(0, 2): 0.9, (0, 4): 0.01, (2, 0): 0.6, (2, 4): 0.8, (4, 0): 0.3, (4, 2): 0.7}
    later_step = {(0, 4): 0.04, (4, 0): 0.001}

    def pooled(data, cov, grams, candidate, other, ordered):
        table = later_step if ordered else first_step
        return table[(candidate[0], other[0])]

    monkeypatch.setattr(engine, "_root_pvalue", pooled)
    clusters = [[0, 1], [2, 3], [4, 5]]
    ordered, unordered = engine._order_clusters(np.zeros((1, 6)), np.eye(6), [], clusters, 0.05)
    assert ordered == [[2, 3]]
    assert unordered == [[0, 1], [4, 5]]
