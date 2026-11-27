"""The Insert scan of LGES and the turning scan it shares with GIES, against their plain forms.

``insert_moves`` and ``insert_best`` keep each SafeInsert verdict for the whole search, share one
``clique_extensions`` run between sources with the same ``NA``, read one semi-directed reach set
per clique, and score each base once per target. They must return the moves, the deltas, and the
best move of the plain enumerator below bit for bit, with or without a verdict cache and with one
cache reused across the states of a search. ``_turning_best`` builds the key of an improving turn
only, and its best turn and ``found`` list must follow from the keys of the improving turns alone.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np
import pytest

from andrey.core import ARROW, TAIL
from andrey.core.orient import dag2cpdag
from andrey.core.score import BICScore, LocalScore, make_score
from andrey.core.ties import EPS_IMPROVE
from andrey.search import operators as ops
from andrey.search._parallel_ges import move_key, neighborhood_state
from andrey.search.gies import _neighborhoods, _turning_best
from andrey.search.lges import extension, insert_best, insert_moves

Move = tuple[float, int, int, tuple[int, ...]]
Verdicts = dict[tuple[int, tuple[int, ...]], frozenset[int]]
Turn = tuple[int, int, set[int], bool]


def _plain_insert_moves(
    adj: np.ndarray, score: LocalScore, n: int, ranks: list[int], conservative: bool
) -> Iterator[Move]:
    """Every valid ``Insert(x, y, T)`` the less greedy rule keeps, one pair and one set at a time.

    Each pair is tested for SafeInsert by itself, each ``(x, T)`` runs its own semi-directed path
    search, and each Insert scores its base again.
    """
    dag = extension(adj, ranks)
    adj_sets, par_of, nbr = neighborhood_state(adj)
    succ = ops.tail_successor_lists(adj)
    for y in range(n):
        dag_pa = set(ops.parents(dag, y).tolist())
        dag_base = score.score(y, sorted(dag_pa))
        pa = {int(p) for p in par_of[y]}
        for x in range(n):
            if x == y or x in adj_sets[y]:
                continue
            if score.score(y, sorted(dag_pa | {x})) > dag_base:
                continue
            na = nbr[y] & adj_sets[x]
            t0 = sorted(nbr[y] - adj_sets[x])
            moves: list[Move] = []
            for _index, T in ops.clique_extensions(adj, na, t0):
                clique = na | set(T)
                if not ops.blocks_semi_directed_paths(succ, y, x, clique):
                    continue
                kept = clique | pa
                delta = score.score(y, sorted(kept | {x})) - score.score(y, sorted(kept))
                if conservative and delta > 0:
                    moves = []
                    break
                moves.append((delta, x, y, T))
            yield from moves


def _plain_insert_best(
    adj: np.ndarray, score: LocalScore, n: int, ranks: list[int], conservative: bool
) -> Move | None:
    """The improving move of :func:`_plain_insert_moves` with the least ``move_key``, if any."""
    improving = (
        m for m in _plain_insert_moves(adj, score, n, ranks, conservative) if m[0] < -EPS_IMPROVE
    )
    return min(improving, key=lambda m: move_key(m[0], n, m[1], m[2], m[3], ranks), default=None)


def _dag(rng: np.random.Generator, d: int, kind: str) -> np.ndarray:
    """A random DAG as marks: ``adj[j, i] == ARROW`` and ``adj[i, j] == TAIL`` for ``i -> j``.

    ``er`` draws each pair with one probability. ``chordal`` gives every node parents that are
    adjacent to each other, so the CPDAG has undirected components with large cliques and no
    v-structure; ``chordal_v`` adds a few forward edges to create some. Nodes are relabeled at
    random so the topological order is not the column order.
    """
    if kind == "er":
        p = min(0.9, rng.uniform(1.0, 4.0) / (d - 1))
        edges = [(i, j) for i, j in itertools.combinations(range(d), 2) if rng.random() < p]
    else:
        parents: dict[int, list[int]] = {0: []}
        degree = [0] * d
        for i in range(1, d):
            open_ = [j for j in range(i) if degree[j] < 6]
            if rng.random() < 0.15 or not open_:
                parents[i] = []
                continue
            j = int(rng.choice(open_))
            parents[i] = [j] + [p for p in parents[j] if degree[p] < 7 and rng.random() < 0.7][:3]
            for p in parents[i]:
                degree[p] += 1
            degree[i] += len(parents[i])
        found = {(p, i) for i in parents for p in parents[i]}
        if kind == "chordal_v":
            for i, j in rng.integers(0, d, (4, 2)):
                if i < j:
                    found.add((int(i), int(j)))
        edges = sorted(found)
    perm = rng.permutation(d)
    dag = np.zeros((d, d), dtype=np.int8)
    for i, j in edges:
        dag[perm[i], perm[j]], dag[perm[j], perm[i]] = TAIL, ARROW
    return dag


def _gaussian_data(rng: np.random.Generator, dag: np.ndarray, n: int) -> np.ndarray:
    """Draws of the linear Gaussian SEM whose edges are the arrows of ``dag``."""
    d = dag.shape[0]
    weights = (dag.T == ARROW) * rng.uniform(0.5, 1.0, (d, d)) * rng.choice([-1.0, 1.0], (d, d))
    return rng.standard_normal((n, d)) @ np.linalg.inv(np.eye(d) - weights)


def _same_moves(a: list[Move], b: list[Move]) -> bool:
    return len(a) == len(b) and all(
        float(x[0]).hex() == float(y[0]).hex() and x[1:] == y[1:] for x, y in zip(a, b, strict=True)
    )


def _same_best(a: Move | None, b: Move | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return _same_moves([a], [b])


CASES = [
    (d, kind, seed) for d in (9, 14) for kind in ("er", "chordal", "chordal_v") for seed in (0, 1)
]
CASES += [(70, "er", 0), (70, "chordal", 1), (70, "chordal_v", 2)]


@pytest.mark.parametrize(("d", "kind", "seed"), CASES)
def test_insert_scan_matches_the_plain_enumerator_on_random_classes(d, kind, seed):
    rng = np.random.default_rng(1000 * d + seed)
    x = _gaussian_data(rng, _dag(rng, d, "er"), 120)
    kept, plain = make_score("bic", x), make_score("bic", x)
    verdicts: Verdicts = {}
    ranks_of = [list(range(d)), rng.permutation(d).tolist()]
    for state, ranks in enumerate(ranks_of * 2):
        # One verdict cache serves every state: a target whose class has adjacent non-parents
        # reads verdicts that an earlier state computed for a source then not adjacent.
        adj = dag2cpdag(_dag(rng, d, kind if state < 2 else "er"))
        for conservative in (False, True):
            expect = list(_plain_insert_moves(adj, plain, d, ranks, conservative))
            assert _same_moves(expect, list(insert_moves(adj, kept, d, ranks, conservative)))
            assert _same_moves(
                expect, list(insert_moves(adj, kept, d, ranks, conservative, verdicts))
            )
            assert _same_best(
                _plain_insert_best(adj, plain, d, ranks, conservative),
                insert_best(adj, kept, d, ranks, conservative, verdicts),
            )


@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize(("d", "seed"), [(8, 0), (8, 1), (10, 2)])
def test_a_search_picks_the_same_inserts_with_one_cache_for_all_its_states(d, seed, conservative):
    rng = np.random.default_rng(seed)
    x = _gaussian_data(rng, _dag(rng, d, "er"), 300)
    ranks = rng.permutation(d).tolist()
    kept, plain = make_score("bic", x), make_score("bic", x)
    verdicts: Verdicts = {}
    adj_new = np.zeros((d, d), dtype=np.int8)
    adj_old = adj_new.copy()
    steps = 0
    while True:
        want = _plain_insert_best(adj_old, plain, d, ranks, conservative)
        got = insert_best(adj_new, kept, d, ranks, conservative, verdicts)
        assert _same_best(want, got)
        assert np.array_equal(adj_new, adj_old)
        if want is None or got is None:
            break
        adj_old = ops.insert_cpdag(adj_old, want[1], want[2], want[3])
        adj_new = ops.insert_cpdag(adj_new, got[1], got[2], got[3])
        steps += 1
    assert steps >= 2
    assert verdicts


def test_a_verdict_is_kept_for_a_target_and_its_parents_in_the_dag():
    rng = np.random.default_rng(3)
    d = 8
    x = _gaussian_data(rng, _dag(rng, d, "er"), 200)
    score = make_score("bic", x)
    ranks = list(range(d))
    adj = dag2cpdag(_dag(rng, d, "chordal"))
    verdicts: Verdicts = {}
    list(insert_moves(adj, score, d, ranks, False, verdicts))
    dag = extension(adj, ranks)
    assert set(verdicts) == {(y, tuple(np.flatnonzero(dag[y] == ARROW).tolist())) for y in range(d)}
    for (y, parents), safe in verdicts.items():
        base = score.score(y, list(parents))
        expect = {
            v
            for v in range(d)
            if v != y and v not in parents and not score.score(y, sorted({*parents, v})) > base
        }
        assert safe == expect


def _turn_delta(adj: np.ndarray, score: BICScore, move: Turn) -> float:
    """The score change of turning ``v -> u`` or ``v -- u`` into ``u -> v``, from its definition."""
    u, v, C, essential = move
    pa, ne, ad = _neighborhoods(adj)
    base = score.score(v, pa[v] | C)
    if essential:
        return (
            score.score(v, pa[v] | C | {u})
            + score.score(u, pa[u] - {v})
            - base
            - score.score(u, pa[u])
        )
    kept = pa[u] | (C & (ne[v] & ad[u]))
    return (
        score.score(v, pa[v] | C | {u}) + score.score(u, kept) - base - score.score(u, kept | {v})
    )


@pytest.mark.parametrize(("d", "kind", "seed"), [(8, "er", 0), (10, "chordal_v", 1), (12, "er", 2)])
def test_turning_scan_keys_follow_the_deltas_of_the_improving_turns(d, kind, seed):
    rng = np.random.default_rng(seed)
    x = _gaussian_data(rng, _dag(rng, d, "er"), 150)
    score = make_score("bic", x)
    assert isinstance(score, BICScore)
    ranks = rng.permutation(d).tolist()
    turns = 0
    for _ in range(4):
        adj = dag2cpdag(_dag(rng, d, kind))
        found: list[tuple[tuple, Turn]] = []
        best = _turning_best(adj, score, d, ranks, found=found)
        assert best == _turning_best(adj, score, d, ranks)
        for key, move in found:
            delta = _turn_delta(adj, score, move)
            assert delta < -EPS_IMPROVE
            assert key == move_key(delta, d, move[0], move[1], move[2], ranks)
        assert (best is None) == (not found)
        if not found:
            continue
        turns += len(found)
        assert best == min(found, key=lambda item: item[0])[1]
        # With every other improving turn excluded, the rest choose the best.
        excluded = {(m[0], m[1], frozenset(m[2])) for _, m in found[::2]}
        rest: list[tuple[tuple, Turn]] = []
        again = _turning_best(adj, score, d, ranks, excluded=excluded, found=rest)
        assert rest == [
            item
            for item in found
            if (item[1][0], item[1][1], frozenset(item[1][2])) not in excluded
        ]
        assert (again is None) == (not rest)
        if rest:
            assert again == min(rest, key=lambda item: item[0])[1]
    assert turns
