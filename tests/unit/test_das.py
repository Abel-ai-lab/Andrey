"""DAS's contract: the edge screen, the shared order, knowledge, units, and the data it refuses."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.fcm.das import MIN_PARENTS, screen
from andrey.fcm.score_matching import leaf_order

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.das(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, _ in out.structure.oriented_edges()}


def _named(out: andrey.StructureOutput) -> set[tuple[str, str]]:
    return {(str(i), str(j)) for i, j, _ in out.structure.oriented_edges()}


def _chain(seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with ``x1 = sin(2 x0)`` and ``x2 = x1**2``, each plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(N)
    x1 = np.sin(2 * x0) + 0.5 * rng.standard_normal(N)
    return np.column_stack([x0, x1, x1**2 + 0.5 * rng.standard_normal(N)])


def _star(seed: int = 3, n: int = 500) -> np.ndarray:
    """Eight causes of a sink ``x8``, two of them parents: ``x8 = sin(2 x1) + x5^2 / 2 + e``."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 9))
    X[:, 8] = np.sin(2 * X[:, 1]) + X[:, 5] ** 2 / 2 + 0.4 * rng.standard_normal(n)
    return X


@pytest.fixture(scope="module")
def chain_fit():
    return _fit(_chain())


def test_the_chain_is_found_with_score_matchings_order(chain_fit):
    assert _edges(chain_fit) == {(0, 1), (1, 2)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        assert chain_fit.ordering == andrey.score_matching(_chain()).ordering == (0, 1, 2)


def test_two_variables_keep_their_edge():
    """The only predecessor is also the reference; the untested largest medians keep it."""
    out = _fit(_chain()[:, :2])
    assert _edges(out) == {(0, 1)}


def test_the_screen_drops_tested_non_parents_before_the_pruning():
    """Of the sink's eight predecessors, the five largest medians pass untested, so ``alpha`` near 1
    keeps them; the Welch tests drop the other three, which no ``alpha`` brings back."""
    X = _star()
    loose = _fit(X, alpha=0.999999)
    parents = {i for i, j in _edges(loose) if j == 8}
    assert {1, 5} <= parents and len(parents) <= MIN_PARENTS
    assert _edges(_fit(X)) >= {(1, 8), (5, 8)}
    assert {i for i, j in _edges(_fit(X, max_parents=None, alpha=0.999999)) if j == 8} == parents


def test_max_parents_caps_every_node():
    out = _fit(_star(), max_parents=1, alpha=0.999999)
    counts = np.bincount([j for _, j in _edges(out)], minlength=9)
    assert counts.max() <= 1
    assert _fit(_star(), max_parents=0).structure.oriented_edges() == []


def test_exact_rows_keep_every_parent_and_drop_every_tested_non_parent():
    """The exact row of an additive model is ``f'(x_j) / sigma^2`` at a parent and 0 elsewhere:
    the parents pass, the zero columns past the untested five fail against the zero reference."""
    rng = np.random.default_rng(0)
    entries = np.zeros((300, 12))
    for j in (2, 7, 9):
        entries[:, j] = np.abs(np.cos(rng.standard_normal(300))) + 0.1
    kept = screen(entries, 20)
    # The five largest medians: the three parents, then zeros 0 and 1 by column order.
    assert sorted(kept) == [0, 1, 2, 7, 9]
    assert kept[:3] == sorted((2, 7, 9), key=lambda j: -np.median(entries[:, j]))


def test_the_screen_ranks_by_the_absolute_median_of_the_entries():
    """Column 0 has rare large entries and column 1 a sign that alternates: both have a larger mean
    absolute entry than column 2's steady 0.3, and a zero median, so column 2 ranks first."""
    n = 200
    entries = np.zeros((n, 3))
    entries[::20, 0] = 10.0
    entries[:, 1] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    entries[:, 2] = 0.3
    assert np.abs(entries).mean(axis=0) == pytest.approx([0.5, 1.0, 0.3])
    assert screen(entries, 1) == [2]
    assert screen(entries, 2) == [2, 0]


def test_the_reference_beyond_the_slots_is_never_kept():
    rng = np.random.default_rng(1)
    entries = np.abs(rng.standard_normal((200, 4))) + np.arange(4, 0, -1)
    assert screen(entries, 3) == [0, 1, 2]
    assert screen(entries, None) == [0, 1, 2, 3]
    assert screen(entries, 0) == []
    assert screen(np.zeros((10, 0)), 5) == []


def test_a_tested_column_stays_when_its_mean_is_clearly_above_the_references():
    rng = np.random.default_rng(2)
    entries = np.abs(rng.standard_normal((400, 8))) * 0.01
    entries[:, :MIN_PARENTS] += 5.0
    entries[:, 6] += 1.0
    assert sorted(screen(entries, 20)) == [0, 1, 2, 3, 4, 6]


def test_a_forbidden_column_stands_for_the_zero_entry_when_it_ranks_below_the_reference():
    """Seven parents with alike entries: among themselves the weakest is the reference and is lost,
    with the zero columns ranked beside them it is a zero column that is, and all seven stay."""
    rng = np.random.default_rng(5)
    parents = np.abs(rng.standard_normal((400, 7))) + 1.0
    zeros = np.abs(rng.standard_normal((400, 3))) * 0.01
    entries = np.hstack([zeros[:, :1], parents, zeros[:, 1:]])
    allowed = np.array([False] + [True] * 7 + [False] * 2)
    assert len(screen(parents, 20)) == MIN_PARENTS
    assert sorted(screen(entries, 20, allowed)) == list(range(1, 8))
    # A forbidden column with a large entry is ranked but never kept, and is not the reference.
    big = np.hstack([entries, entries[:, [1]] * 4])
    kept = screen(big, 20, np.append(allowed, False))
    assert sorted(kept) == list(range(1, 8)) and 10 not in kept
    # With more parents than slots the zero column still cuts the list at the slots.
    assert len(screen(entries, 3, allowed)) == 3
    assert screen(entries, 20, np.zeros(10, dtype=bool)) == []


def _forbidden_zeros(seed: int, n: int = 500, d: int = 14, parents: int = 7) -> np.ndarray:
    """Independent standard normals and a last column that sums odd cubic terms of the first
    ``parents``: every other column is a known non-parent whose entries are zero."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    c = rng.uniform(0.8, 1.2, parents)
    terms = sum(c[k] * (X[:, k] + X[:, k] ** 3 / 6) for k in range(parents))
    X[:, -1] = 0.5 * terms + 0.5 * rng.standard_normal(n)
    return X


@pytest.mark.parametrize("seed", [0, 1])
def test_correct_forbidden_knowledge_never_removes_a_parent_found_without_it(seed):
    X = _forbidden_zeros(seed)
    tiers = [list(range(13)), [13]]
    forbidden = [(k, 13) for k in range(7, 13)]  # the true non-parents of the last column
    without = _fit(X, knowledge=andrey.Knowledge(tiers=tiers))
    with_it = _fit(X, knowledge=andrey.Knowledge(tiers=tiers, forbidden=forbidden))
    found = {i for i, j in _edges(without) if j == 13}
    assert found == set(range(7))
    assert found <= {i for i, j in _edges(with_it) if j == 13}


def test_on_leaf_sees_each_step_before_its_leaf_leaves():
    steps = []
    values = {3: [0.3, 0.1, 0.2], 2: [0.5, 0.4]}
    order = leaf_order(
        np.zeros((2, 3)),
        lambda remaining: np.array(values[remaining.shape[1]]),
        on_leaf=lambda remaining, leaf: steps.append((list(remaining), leaf)),
    )
    assert steps == [([0, 1, 2], 1), ([0, 2], 2)]
    assert order.tolist() == [0, 2, 1]


def test_the_result_follows_the_columns_in_any_order_and_units(chain_fit):
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_chain(), columns=list("abc"))
    moved = frame[["c", "a", "b"]] * [1e-3, 250.0, 7.0] + [5.0, -40.0, 1e3]
    out = _fit(moved)
    assert _named(out) == {("a", "b"), ("b", "c")}
    assert [moved.columns[j] for j in out.ordering] == ["a", "b", "c"]


def test_the_screen_follows_the_columns_in_any_order_and_units():
    """With eight predecessors of the sink the Welch tests run, and give the same graph over the
    variables under a new column order and new units."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_star(), columns=[f"v{j}" for j in range(9)])
    columns = ["v8", "v3", "v0", "v6", "v1", "v7", "v2", "v5", "v4"]
    scales = 2.0 ** np.arange(-4, 5)
    moved = frame[columns] * scales + np.arange(9)
    assert _named(_fit(moved)) == _named(_fit(frame))


def test_a_rerun_repeats():
    X = _star()
    first, second = _fit(X), _fit(X)
    assert _edges(first) == _edges(second) and first.ordering == second.ordering


def test_a_required_edge_stays_and_a_forbidden_one_never_enters(chain_fit):
    X = np.random.default_rng(1).standard_normal((N, 3))
    out = _fit(X, knowledge=andrey.Knowledge(required=[(2, 0)]))
    assert _edges(out) == {(2, 0)}
    out = _fit(_chain(), knowledge=andrey.Knowledge(forbidden=[(0, 1)]))
    assert (0, 1) not in _edges(out) and out.ordering == chain_fit.ordering


def test_more_required_parents_than_max_parents_are_refused():
    X = np.random.default_rng(1).standard_normal((N, 3))
    with pytest.raises(ValueError, match="max_parents=1"):
        _fit(X, max_parents=1, knowledge=andrey.Knowledge(required=[(0, 2), (1, 2)]))


def test_max_parents_sets_the_rows_needed():
    """The pruning fits at most ``max_parents`` parents, so ``2 max_parents + 2`` rows suffice."""
    X = np.random.default_rng(4).standard_normal((12, 8))
    assert _fit(X, max_parents=5).structure.n_nodes == 8
    with pytest.raises(ValueError, match="at least 14 rows"):
        _fit(X, max_parents=6)
    with pytest.raises(ValueError, match="at least 16 rows"):
        _fit(X, max_parents=None)


def test_rows_that_mostly_repeat_are_refused_by_name():
    X = np.random.default_rng(6).standard_normal((40, 2))
    X[:30] = 0.0
    with pytest.raises(ValueError, match="das: the median distance"):
        _fit(X)
