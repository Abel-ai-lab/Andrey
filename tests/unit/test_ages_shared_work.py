"""AGES shares work across the penalties of its solution path and gives the unshared path's output.

A backward phase of the solution path keeps three things that read no penalty: the inverses of the
covariance blocks, the remove ratios, and each target's Delete candidates. This file pins, bit for
bit against the unshared paths:

* ``remove_ratio`` minus the penalty is the one remove delta of ``delta_many_with_base``, and it is
  ``None`` exactly where that function falls back to two full scores;
* each Delete the backward phases take is the one ``backward_chunk_best`` selects on a fresh engine,
  the candidates they keep equal a fresh enumeration after every refresh, and the whole solution
  path equals the path of the plain backward phase, with the fallback reached through a lowered
  conditioning gate;
* the objective and the neighborhood state read their rows through ``row_lists``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from andrey.core import orient, score_delta
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore, _InvCache, remove_ratio
from andrey.data import sample_scm
from andrey.search import ages
from andrey.search import operators as ops
from andrey.search._parallel_ges import backward_chunk_best, neighborhood_state
from andrey.search.ges import recompute_objective

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")


def _cov_of(seed: int, d: int, n: int = 400) -> np.ndarray:
    return np.cov(sample_scm(d=d, n=n, seed=seed, density=3.0).data, rowvar=False)


def _fell_back(
    monkeypatch: pytest.MonkeyPatch, score: BICScore, i: int, base: tuple[int, ...], u: int, **kw
) -> tuple[float, bool]:
    """The remove delta of ``delta_many_with_base`` and whether it took the two-score fallback."""
    calls = []
    real = score_delta._delta_two_scores

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(score_delta, "_delta_two_scores", spy)
    delta = score_delta.delta_many_with_base(score, i, base, [("remove", u)], **kw)[0]
    monkeypatch.setattr(score_delta, "_delta_two_scores", real)
    return delta, bool(calls)


@pytest.mark.parametrize("penalty_discount", [0.5, 1.0, 2.3])
@pytest.mark.parametrize("cond_threshold", [score_delta._COND_THRESHOLD, 30.0, 3.0])
def test_remove_ratio_minus_penalty_is_the_remove_delta(
    monkeypatch: pytest.MonkeyPatch, penalty_discount: float, cond_threshold: float
) -> None:
    """The ratio less the penalty equals the remove delta bit for bit; it is None on a fallback.

    The lowered thresholds send ill-conditioned bases to the fallback; a near-copied column and a
    move of a node outside the base reach it too.
    """
    rng = np.random.default_rng(7)
    d = 9
    x = rng.standard_normal((300, d))
    x[:, 4] = x[:, 3] + 1e-7 * rng.standard_normal(300)  # near-singular blocks holding both
    score = BICScore.from_cov(np.cov(x, rowvar=False), 300, penalty_discount=penalty_discount)
    cost = score.log_n * score.penalty_discount
    inverses = _InvCache()
    seen = {"ratio": 0, "fallback": 0}
    for i in range(d):
        others = [v for v in range(d) if v != i]
        for size in range(1, 6):
            for base in [tuple(sorted(rng.choice(others, size, replace=False).tolist()))] * 2:
                for u in [*base, others[0] if others[0] not in base else others[-1]]:
                    delta, fell_back = _fell_back(
                        monkeypatch, score, i, base, u, cond_threshold=cond_threshold
                    )
                    ratio = remove_ratio(score, i, base, u, inverses, cond_threshold=cond_threshold)
                    assert (ratio is None) == fell_back, (i, base, u)
                    if ratio is None:
                        seen["fallback"] += 1
                    else:
                        seen["ratio"] += 1
                        assert float(ratio - cost) == delta, (i, base, u)
    assert seen["ratio"] > 0
    assert seen["fallback"] > 0


def test_delta_engine_takes_a_shared_inverse_cache() -> None:
    """An engine built with a cache uses it, and one built without owns a new one."""
    score = BICScore.from_cov(_cov_of(0, 6), 400)
    cache = _InvCache()
    engine = DeltaBICScore(score, inv_cov_cache=cache)
    assert engine.inv_cov_cache is cache
    assert engine.delta_many_with_base(0, (1, 2), [("remove", 1)])
    assert (1, 2) in cache
    other = DeltaBICScore(score)
    assert other.inv_cov_cache is not cache
    assert len(other.inv_cov_cache) == 0


def _plain_backward(adj, score, n, ranks, deletes):
    """The backward phase as it scans every target with ``backward_chunk_best``, one engine and one
    kept-delta dictionary per phase, from nothing the other phases computed."""
    engine = DeltaBICScore(score)
    kept: dict = {}
    total = recompute_objective(adj, score, n)
    while True:
        best = backward_chunk_best(range(n), adj, engine, n, neighborhood_state(adj), ranks, kept)
        if best is None:
            return
        _, x, y, H = best
        adj = ops.delete_cpdag(adj.copy(), x, y, H)
        after = recompute_objective(adj, score, n)
        yield adj, score.penalty_discount + (after - total) / score.log_n
        total = after


def _gated(monkeypatch: pytest.MonkeyPatch, limit: float) -> None:
    """Lower the conditioning gate of every inverse to ``limit``, so many removes fall back."""
    real = score_delta._get_inv_with_cond

    def gated(cache, S, cov, S_list, cond_threshold):
        return real(cache, S, cov, S_list, min(cond_threshold, limit))

    monkeypatch.setattr(score_delta, "_get_inv_with_cond", gated)


def _checked(monkeypatch: pytest.MonkeyPatch, steps: dict[str, int]) -> None:
    """Check each backward phase and refresh of the shared path against the plain computation."""
    real_backward = ages._backward
    real_refresh = ages._Deletes.refresh

    def refresh(self, adj):
        real_refresh(self, adj)
        fresh = ages._Deletes(len(self.of), self.inverses)
        real_refresh(fresh, adj)
        assert self.of == fresh.of, "kept candidates differ from a fresh enumeration"
        steps["refreshes"] += 1

    def backward(adj, score, n, ranks, deletes) -> Iterator[tuple[np.ndarray, float]]:
        current = adj
        for after, limit in real_backward(adj, score, n, ranks, deletes):
            plain = backward_chunk_best(
                range(n), current, DeltaBICScore(score), n, neighborhood_state(current), ranks
            )
            assert plain is not None, "a Delete was taken where the plain scan finds none"
            _, x, y, H = plain
            assert np.array_equal(ops.delete_cpdag(current.copy(), x, y, H), after)
            steps["deletes"] += 1
            current = after
            yield after, limit
        end = backward_chunk_best(
            range(n), current, DeltaBICScore(score), n, neighborhood_state(current), ranks
        )
        assert end is None, "a phase stopped where the plain scan finds a Delete"

    monkeypatch.setattr(ages._Deletes, "refresh", refresh)
    monkeypatch.setattr(ages, "_backward", backward)


def _path(X: np.ndarray, **kw) -> list[tuple[float, bytes]]:
    return [(penalty, cpdag.tobytes()) for penalty, cpdag in ages.solution_path(X, **kw)]


def _data(graph: str, d: int, seed: int, density: float = 4.0, n: int = 500) -> np.ndarray:
    return sample_scm(graph=graph, d=d, n=n, seed=seed, density=density).data


CASES = [
    pytest.param(_data("erdos_renyi", 16, 0), {}, id="erdos_renyi"),
    pytest.param(_data("scale_free", 16, 1, density=5.0), {}, id="scale_free"),
    pytest.param(_data("small_world", 16, 2, density=6.0), {}, id="small_world"),
    pytest.param(_data("erdos_renyi", 14, 3, density=3.0), {"penalty_discount": 0.5}, id="low"),
    pytest.param(_data("erdos_renyi", 14, 4, density=3.0), {"penalty_discount": 2.0}, id="high"),
    pytest.param(np.round(3 * _data("erdos_renyi", 14, 6, density=3.0)), {}, id="rounded"),
]


@pytest.mark.parametrize(("X", "kw"), CASES)
def test_solution_path_equals_the_plain_backward_phases(monkeypatch, X, kw) -> None:
    """The shared path is the path of plain backward phases: penalties and CPDAGs, bit for bit."""
    steps = {"refreshes": 0, "deletes": 0}
    with monkeypatch.context() as plain:
        plain.setattr(ages, "_backward", _plain_backward)
        want = _path(X, **kw)
    _checked(monkeypatch, steps)
    assert _path(X, **kw) == want
    assert steps["deletes"] > 0
    assert steps["refreshes"] > steps["deletes"]


@pytest.mark.parametrize("limit", [10.0, 30.0, 1e3])
def test_solution_path_through_the_two_score_fallback(monkeypatch, limit) -> None:
    """With a lowered conditioning gate the removes that fall back give the plain path too."""
    X = _data("erdos_renyi", 14, 6, density=5.0)
    _gated(monkeypatch, limit)
    with monkeypatch.context() as plain:
        plain.setattr(ages, "_backward", _plain_backward)
        want = _path(X)
    results = {"ratio": 0, "fallback": 0}
    real = ages.remove_ratio

    def count(*args, **kwargs):
        ratio = real(*args, **kwargs)
        results["fallback" if ratio is None else "ratio"] += 1
        return ratio

    monkeypatch.setattr(ages, "remove_ratio", count)
    _checked(monkeypatch, {"refreshes": 0, "deletes": 0})
    assert _path(X) == want
    assert results["fallback"] > 0
    assert results["ratio"] > 0


def test_the_path_inverts_fewer_matrices(monkeypatch) -> None:
    """The path inverts each covariance block once where the plain phases invert it again."""
    X = _data("erdos_renyi", 16, 7, density=4.0)
    inversions = {"shared": 0, "plain": 0}
    real = np.linalg.inv
    arm = "shared"

    def count(a, *args, **kwargs):
        inversions[arm] += 1
        return real(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "inv", count)
    shared = _path(X)
    arm = "plain"
    monkeypatch.setattr(ages, "_backward", _plain_backward)
    assert _path(X) == shared
    assert 0 < inversions["shared"] < inversions["plain"]


def test_recompute_objective_reads_the_parents_of_a_dag() -> None:
    """The objective sums each node's local score over its parents in a DAG of the class."""
    X = _data("erdos_renyi", 12, 8)
    score = BICScore(X)
    n = X.shape[1]
    for _, cpdag in ages.solution_path(X)[:4]:
        dag = orient.pdag2dag(cpdag)
        want = float(sum(score.score(v, ops.parents(dag, v).tolist()) for v in range(n)))
        assert recompute_objective(cpdag, score, n) == want


def test_neighborhood_state_holds_the_operators_nodes() -> None:
    """A node's entries hold the nodes of ``ops.adjacent``, ``ops.parents`` and the undirected
    neighbors, the parents as an ascending list of ints."""
    X = _data("scale_free", 14, 9)
    for _, cpdag in ages.solution_path(X)[:4]:
        adj_sets, par_of, nbr = neighborhood_state(cpdag)
        for v in range(cpdag.shape[0]):
            assert adj_sets[v] == set(ops.adjacent(cpdag, v).tolist())
            assert par_of[v] == ops.parents(cpdag, v).tolist()
            assert nbr[v] == set(ops.undirected_neighbors(cpdag, v).tolist())
            assert all(type(p) is int for p in par_of[v])
