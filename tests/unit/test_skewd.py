"""SkewD and SkewD-MV: the class listing and its cap, the density, units, column order, reruns."""

from __future__ import annotations

import itertools
import warnings

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


def test_skewd_needs_a_complete_row_for_each_cross_validation_fold():
    with pytest.raises(ValueError, match="at least 8 rows"):
        skewd(_pair(7), **SHORT_PAIR)
    data = _pair(20)
    data[:13, 0] = np.nan  # 7 rows where both columns are observed
    with pytest.raises(ValueError, match="at least 8 rows"):
        skewd(data, **SHORT_PAIR)


def test_skewd_mv_reads_columns_whose_moments_overflow():
    # Every value is near 1e308 in magnitude, so a plain mean or variance overflows.
    data = pd.DataFrame(_chain(), columns=["x", "y", "z"])
    huge = skewd_mv((data + 4.0) * 1e307, **SHORT_GRAPH)
    assert (
        huge.structure.oriented_edges() == skewd_mv(data, **SHORT_GRAPH).structure.oriented_edges()
    )


def test_skewd_mv_breaks_ties_without_overflowing_on_two_huge_columns():
    # The tie ranks standardize the raw columns they compare: values near 1e308 overflow the sums.
    data = pd.DataFrame(_chain(), columns=["x", "y", "z"])[["x", "y"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        huge = skewd_mv((data + 4.0) * 1e307, **SHORT_GRAPH)
    assert (
        huge.structure.oriented_edges() == skewd_mv(data, **SHORT_GRAPH).structure.oriented_edges()
    )


@pytest.mark.parametrize("missing_cell", [False, True])
def test_a_column_that_spans_the_float_range_is_read_without_overflow(missing_cell):
    # Column 0 holds both signs up to 1.7e308, so its largest value less its smallest overflows.
    data = _pair()
    huge = data.copy()
    huge[:, 0] *= 1.7e308 / np.abs(data[:, 0]).max()
    if missing_cell:
        data[3, 1] = huge[3, 1] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pair = skewd(huge, **SHORT_PAIR)
        graph = skewd_mv(huge[~np.isnan(huge).any(axis=1)], **SHORT_GRAPH)
    assert pair.structure.oriented_edges() == skewd(data, **SHORT_PAIR).structure.oriented_edges()
    assert len(graph.structure.oriented_edges()) == 1


def _orient(base: np.ndarray, *edges: tuple[int, int]) -> np.ndarray:
    dag = base.copy()
    for i, j in edges:
        dag[i, j] = 1
    return dag


def test_components_whose_best_orientations_close_a_cycle_take_the_best_acyclic_combination():
    # A--B and C--D with B -> C and D -> A: A -> B and C -> D are each the best orientation of their
    # component and together close A -> B -> C -> D -> A.
    base = _orient(np.zeros((4, 4), dtype=np.int8), (1, 2), (3, 0))
    components = [
        ([0, 1], [_orient(base, (0, 1)), _orient(base, (1, 0))]),
        ([2, 3], [_orient(base, (2, 3)), _orient(base, (3, 2))]),
    ]
    dag = native.merge_components(base, components, [[3.0, 0.0], [3.0, 1.0]])
    assert _acyclic(dag)
    assert dag[0, 1] == dag[3, 2] == dag[1, 2] == dag[3, 0] == 1  # the best acyclic total, 4


def test_a_graph_with_no_acyclic_orientation_raises_instead_of_returning_a_cycle():
    marks = _marks(4, lines=[(0, 1), (2, 3)], arrows=[(1, 2), (3, 0)])
    components, base, _ = native.class_candidates(marks, None)
    with pytest.raises(ValueError, match="no DAG that keeps its directed edges"):
        native.merge_components(
            base, components, [[0.0] * len(members) for _, members in components]
        )


def test_a_cycle_among_many_independent_components_still_finds_the_acyclic_combination():
    # The cycle case above next to twelve independent edges: 2^14 combinations, more than any cap on
    # combinations admits, and exactly one cycle to avoid.
    base = _orient(np.zeros((28, 28), dtype=np.int8), (1, 2), (3, 0))
    free = [(4 + 2 * i, 5 + 2 * i) for i in range(12)]
    components = [
        ([0, 1], [_orient(base, (0, 1)), _orient(base, (1, 0))]),
        ([2, 3], [_orient(base, (2, 3)), _orient(base, (3, 2))]),
        *(([i, j], [_orient(base, (i, j)), _orient(base, (j, i))]) for i, j in free),
    ]
    scores = [[3.0, 0.0], [3.0, 1.0], *([[1.0, 0.0]] * len(free))]
    dag = native.merge_components(base, components, scores)
    assert _acyclic(dag)
    assert dag[0, 1] == dag[3, 2] == 1  # the best acyclic total reverses C--D
    assert all(dag[i, j] == 1 for i, j in free)  # each independent edge keeps its best orientation


def _random_components(rng: np.random.Generator, d: int = 8):
    """Directed edges along a random order and up to four disjoint pairs or paths of free edges,
    each component's members the orientations that keep the directed edges acyclic."""
    order = rng.permutation(d)
    base = np.zeros((d, d), dtype=np.int8)
    for i, j in itertools.combinations(range(d), 2):
        if rng.random() < 0.3:
            base[order[i], order[j]] = 1
    nodes, components = rng.permutation(d).tolist(), []
    for size in rng.choice([2, 3], size=4):
        if len(nodes) < size:
            break
        group, nodes = nodes[:size], nodes[size:]
        edges = list(itertools.pairwise(group))
        members = []
        for flips in itertools.product((False, True), repeat=len(edges)):
            member = _orient(base, *(e[::-1] if flip else e for e, flip in zip(edges, flips)))
            if not _acyclic(member):
                continue
            members.append(member)
        if members:
            components.append((sorted(group), members))
    return base, components


def _combined(base, components, choice):
    return np.maximum.reduce([base, *(c[1][i] for c, i in zip(components, choice, strict=True))])


def test_the_joint_search_matches_exhaustive_search_on_random_conflicting_graphs():
    rng = np.random.default_rng(1)
    conflicts = 0
    for _ in range(1000):
        base, components = _random_components(rng)
        scores = [list(rng.normal(size=len(members))) for _, members in components]
        best = tuple(int(np.argmax(row)) for row in scores)
        conflicts += not _acyclic(_combined(base, components, best))
        totals = {
            _combined(base, components, c).tobytes(): sum(
                row[i] for row, i in zip(scores, c, strict=True)
            )
            for c in itertools.product(*(range(len(row)) for row in scores))
            if _acyclic(_combined(base, components, c))
        }
        if not totals:
            with pytest.raises(ValueError, match="no DAG that keeps its directed edges"):
                native.merge_components(base, components, scores)
            continue
        merged = native.merge_components(base, components, scores)
        assert _acyclic(merged)
        assert totals[merged.tobytes()] == pytest.approx(max(totals.values()))
    assert conflicts > 20  # the draw reaches the cycle case often


def test_a_constant_cross_validation_fold_starts_from_a_symmetric_shape():
    # Eight folds of eight rows: the fold that holds out the one row that differs trains on a
    # constant effect, whose skewness is 0 / 0.
    parents = np.random.default_rng(0).standard_normal((8, 1))
    effect = np.array([0.0] * 7 + [1.0])
    budget = native.Budget(knots=8, folds=8, tuning_iter=1, initial=1, max_iter=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a division by zero is a RuntimeWarning
        result = native.fit(parents, effect, budget, 0)
    assert np.isfinite(result.log_likelihood)
    design = native._design(parents, budget.knots)
    start = native._start(design.location[:7], design.scale[:7], effect[:7])
    assert np.isfinite(start).all()
    assert start[-1] == 0.0
    nearly = native._start(design.location[:7], design.scale[:7], np.full(7, 1 / 3))
    assert nearly[-1] == 0.0  # rounding in the mean leaves no skewness either
