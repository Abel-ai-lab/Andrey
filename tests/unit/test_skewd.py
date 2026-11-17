"""SkewD and SkewD-MV: the class listing and its cap, the density, units, column order, reruns."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skewnorm

from andrey import skewd, skewd_mv
from andrey.core.orient import ARROW, NULL, TAIL, dag2cpdag
from andrey.fcm import skewd as native

SHORT_PAIR = {"tuning_iter": 1, "max_iter": 5}
SHORT_GRAPH = {**SHORT_PAIR, "indep_test": "fisherz"}


def _marks(d: int, lines=(), arrows=()) -> np.ndarray:
    marks = np.full((d, d), NULL, dtype=np.int8)
    for i, j in lines:
        marks[i, j] = marks[j, i] = TAIL
    for i, j in arrows:
        marks[i, j], marks[j, i] = TAIL, ARROW
    return marks


def _acyclic(adjacency: np.ndarray) -> bool:
    reach = adjacency.astype(bool)
    for k in range(len(reach)):
        reach |= reach[:, k, None] & reach[None, k, :]
    return not np.diagonal(reach).any()


def _equivalent_dags(cpdag: np.ndarray) -> set[bytes]:
    """The DAGs whose CPDAG is ``cpdag``, found by trying every orientation of its lines."""
    lines = np.argwhere(np.triu((cpdag == TAIL) & (cpdag.T == TAIL))).tolist()
    found = set()
    for flips in itertools.product((False, True), repeat=len(lines)):
        dag = cpdag.copy()
        for (i, j), flip in zip(lines, flips, strict=True):
            tail, head = (j, i) if flip else (i, j)
            dag[tail, head], dag[head, tail] = TAIL, ARROW
        adjacency = ((dag == TAIL) & (dag.T == ARROW)).astype(np.int8)
        if _acyclic(adjacency) and np.array_equal(dag2cpdag(dag), cpdag):
            found.add(adjacency.tobytes())
    return found


def _random_dag(rng: np.random.Generator, d: int) -> np.ndarray:
    adjacency = np.triu(rng.random((d, d)) < 0.5, k=1)
    order = rng.permutation(d)
    adjacency = adjacency[np.ix_(order, order)]
    marks = np.zeros((d, d), dtype=np.int8)
    marks[adjacency.T] = ARROW
    marks[adjacency] = TAIL
    return marks


def test_class_members_are_the_dags_of_the_equivalence_class():
    rng = np.random.default_rng(0)
    for _ in range(40):
        cpdag = dag2cpdag(_random_dag(rng, 5))
        listed = [m.tobytes() for m in native.class_members(cpdag)]
        assert len(listed) == len(set(listed))
        assert set(listed) == _equivalent_dags(cpdag)


@pytest.mark.parametrize(
    ("marks", "size"),
    [
        (_marks(3, lines=[(0, 1), (1, 2)]), 3),  # a chain: one DAG per root
        (_marks(3, arrows=[(0, 1), (2, 1)]), 1),  # a collider fixes every edge
        (_marks(4, lines=itertools.combinations(range(4), 2)), 24),  # a clique: every order
        (_marks(5, lines=[(0, i) for i in range(1, 5)]), 5),  # a star: one DAG per root
    ],
)
def test_class_members_count_known_classes(marks, size):
    assert len(list(native.class_members(marks))) == size


def test_a_graph_with_conflicting_colliders_falls_back_to_every_acyclic_orientation():
    cycle = _marks(4, lines=[(0, 1), (1, 2), (2, 3), (3, 0)])
    assert list(native.class_members(cycle)) == []
    components, _, fits = native.class_candidates(cycle, None)
    assert [len(members) for _, members in components] == [14]  # the acyclic orientations
    assert fits


def test_the_listing_stops_at_the_cap_instead_of_listing_the_class():
    clique = _marks(12, lines=itertools.combinations(range(12), 2))  # 12! DAGs
    with pytest.raises(ValueError, match=r"more than 100 conditional models.*max_fits"):
        native.class_candidates(clique, 100)
    components, base, fits = native.class_candidates(_marks(4, lines=[(0, 1), (1, 2)]), 100)
    assert [(nodes, len(members)) for nodes, members in components] == [([0, 1, 2], 3)]
    assert (base.sum(), len(fits)) == (0, 4)


def test_separate_groups_of_edges_are_listed_apart_not_as_their_product():
    # 22 disjoint edges: 2**22 DAGs, 44 conditional models. Each edge is listed on its own.
    edges = [(2 * k, 2 * k + 1) for k in range(22)]
    components, _, fits = native.class_candidates(_marks(44, lines=edges), 100)
    assert [len(members) for _, members in components] == [2] * 22
    assert len(fits) == 44
    with pytest.raises(ValueError, match="max_fits"):  # 120 models
        native.class_candidates(_marks(120, lines=[(2 * k, 2 * k + 1) for k in range(60)]), 100)


def test_the_directed_edges_stay_in_every_component_dag():
    marks = _marks(4, lines=[(0, 1)], arrows=[(2, 0), (2, 3)])
    components, base, _ = native.class_candidates(marks, None)
    assert base[2, 0] == base[2, 3] == 1
    assert all(m[2, 0] == m[2, 3] == 1 for _, members in components for m in members)


def _single_factor(d: int, n: int) -> np.ndarray:
    """Columns that are all dependent given any others, so PC keeps every edge undirected."""
    rng = np.random.default_rng(0)
    return 0.9 * rng.standard_normal((n, 1)) + 0.3 * rng.standard_normal((n, d))


def test_skewd_mv_refuses_a_class_past_the_cap_before_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a conditional model was fitted")

    monkeypatch.setattr(native, "fit", forbidden)
    with pytest.raises(ValueError, match="max_fits"):
        skewd_mv(_single_factor(10, 1500), indep_test="fisherz")
    with pytest.raises(ValueError, match="more than 2 conditional models"):
        skewd_mv(_single_factor(4, 1500), indep_test="fisherz", max_fits=2)


def test_skewd_mv_rejects_a_cap_below_one():
    independent = np.random.default_rng(0).standard_normal((300, 3))  # no model is needed
    with pytest.raises(ValueError, match="max_fits"):
        skewd_mv(independent, indep_test="fisherz", max_fits=0)


def test_the_likelihood_is_the_skew_normal_density():
    rng = np.random.default_rng(1)
    location, scale = rng.standard_normal((50, 4)), rng.standard_normal((50, 3))
    effect = rng.standard_normal(50)
    theta = rng.standard_normal((6, 8))  # 4 location, 3 log-scale, 1 shape coefficients
    theta[:, -1] *= 5
    rows = native._row_log_likelihoods(theta, location, scale, effect)
    for k in range(6):
        mu = location @ theta[k, :4]
        sigma = np.exp(0.5 * scale @ theta[k, 4:7])
        expected = skewnorm.logpdf(effect, theta[k, -1], loc=mu, scale=sigma)
        np.testing.assert_allclose(rows[:, k], expected, rtol=1e-10, atol=1e-10)


def _pair(n: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.tanh(2 * x) + (0.5 + 0.3 * np.tanh(x)) * skewnorm.rvs(10, size=n, random_state=seed)
    return np.column_stack([x, y])


def _chain(n: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.tanh(2 * x) + (0.5 + 0.3 * np.tanh(x)) * skewnorm.rvs(10, size=n, random_state=seed)
    return np.column_stack([x, y, y + 0.5 * rng.standard_normal(n)])


@pytest.mark.parametrize("criterion", ["independence", "likelihood"])
def test_skewd_reads_no_units_and_repeats(criterion):
    # The columns are standardized first, so a change of units (exact products) changes nothing.
    data = pd.DataFrame(_pair(), columns=["x", "y"])
    first = skewd(data, criterion=criterion, **SHORT_PAIR)
    assert skewd(data, criterion=criterion, **SHORT_PAIR).metadata == first.metadata
    scaled = skewd(data * np.array([2.0**-30, 2.0**20]), criterion=criterion, **SHORT_PAIR)
    assert scaled.structure.oriented_edges() == first.structure.oriented_edges()
    assert scaled.metadata == first.metadata


def test_skewd_swapped_columns_swap_the_directions():
    data = pd.DataFrame(_pair(), columns=["x", "y"])
    first = skewd(data, criterion="likelihood", **SHORT_PAIR)
    swapped = skewd(data[["y", "x"]], criterion="likelihood", **SHORT_PAIR)
    assert swapped.structure.oriented_edges() == first.structure.oriented_edges()
    assert swapped.metadata["log_likelihoods"]["forward"] == pytest.approx(
        first.metadata["log_likelihoods"]["backward"]
    )
    assert swapped.metadata["log_likelihoods"]["backward"] == pytest.approx(
        first.metadata["log_likelihoods"]["forward"]
    )


def test_skewd_mv_reads_no_units_ignores_column_order_and_repeats():
    data = pd.DataFrame(_chain(), columns=["x", "y", "z"])
    first = skewd_mv(data, **SHORT_GRAPH)
    assert {frozenset(edge[:2]) for edge in first.structure.oriented_edges()} == {
        frozenset("xy"),
        frozenset("yz"),
    }
    assert first.metadata["candidates"] == 3  # the chain's class: one DAG per root
    assert skewd_mv(data, **SHORT_GRAPH).metadata == first.metadata
    scaled = skewd_mv(data * np.array([2.0**-30, 2.0**20, 2.0**-10]), **SHORT_GRAPH)
    assert scaled.structure.oriented_edges() == first.structure.oriented_edges()
    reordered = skewd_mv(data[["z", "x", "y"]], **SHORT_GRAPH)
    assert sorted(reordered.structure.oriented_edges()) == sorted(first.structure.oriented_edges())
