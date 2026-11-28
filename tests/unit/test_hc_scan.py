"""Hill climbing's and tabu search's move scan picks what a full rescan picks.

:class:`andrey.search.hc.MoveScanner` keeps its deltas between passes, recomputes only the rows of
targets whose parents changed, and checks only its lowest candidates for cycles. The reference below
rescans every move each pass, with each target's row deltas computed afresh. A delta depends only on
its target and parent set, so equal picks and equal keys show the scanner's tables, cycle checks,
and tie-break are right. Exact ties break by each column's hash, so the graph found does not depend
on the column order.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import andrey
from andrey.core.score import BICScore, make_score
from andrey.core.score_delta import (
    DeltaBICScore,
    FullScoreDelta,
    add_deltas,
    make_delta,
    remove_deltas,
)
from andrey.core.ties import EPS_IMPROVE, KEY_DP, tie_ranks
from andrey.search.hc import MoveScanner, _key_after, apply_move, hc
from andrey.search.tabu import tabu
from qa import identity


def _reference_scan(score, adj_int, parents_of, d, floor, tabu, ranks):
    """Every legal move with delta below ``floor``, rescanned u-major, v-minor."""
    add = [add_deltas(score, v, parents_of[v]) for v in range(d)]
    remove = [remove_deltas(score, v, parents_of[v]) for v in range(d)]

    def reaches(a, b):
        seen, todo = 0, adj_int[a]
        while todo:
            lsb = todo & -todo
            todo ^= lsb
            if not seen & lsb:
                seen |= lsb
                todo |= adj_int[lsb.bit_length() - 1]
        return bool(seen >> b & 1)

    inv_n = 1.0 / score.n
    best = None

    def offer(value, u, v, rank, op):
        nonlocal best
        if value < floor and _key_after(adj_int, (op, u, v)) not in tabu:
            cand = (round(value * inv_n, KEY_DP), u, v, rank, (op, u, v))
            key = (cand[0], ranks[u], ranks[v], rank)
            if best is None or key < (best[0], ranks[best[1]], ranks[best[2]], best[3]):
                best = cand

    for u in range(d):
        for v in range(d):
            if u == v:
                continue
            if adj_int[u] >> v & 1:
                offer(remove[v][u], u, v, 0, "remove")
                others = adj_int[u] & ~(1 << v)
                if not any(others >> c & 1 and reaches(c, v) for c in range(d)):
                    covered = parents_of[v] == sorted([*parents_of[u], u])
                    value = 0.0 if covered else add[u][v] + remove[v][u]
                    offer(value, u, v, 1, "reverse")
            elif not adj_int[v] >> u & 1 and not reaches(v, u):
                offer(add[v][u], u, v, 0, "add")
    return best


def _fresh_deltas(score, v, parents, d):
    """The add deltas and the remove deltas at target ``v`` with ``parents``, from ``score`` alone.

    An add delta is ``inf`` for ``v`` and its parents, a remove delta for every other column.
    """
    if isinstance(score, BICScore):
        return add_deltas(score, v, parents), remove_deltas(score, v, parents)
    base = score.score(v, parents)
    add, remove = np.full(d, np.inf), np.full(d, np.inf)
    for u in range(d):
        if u != v and u not in parents:
            add[u] = score.score(v, sorted([*parents, u])) - base
    for u in parents:
        remove[u] = score.score(v, [p for p in parents if p != u]) - base
    return add, remove


def _reachable(adj_int, d):
    """``reach[x, y]``: a directed path leads from ``x`` to ``y``."""
    reach = np.zeros((d, d), dtype=bool)
    for x in range(d):
        seen, todo = 0, adj_int[x]
        while todo:
            lsb = todo & -todo
            todo ^= lsb
            if not seen & lsb:
                seen |= lsb
                todo |= adj_int[lsb.bit_length() - 1]
        reach[x] = [seen >> y & 1 for y in range(d)]
    return reach


def _reference_scan_under_rules(
    score,
    adj_int,
    parents_of,
    d,
    floor,
    tabu,
    ranks,
    *,
    fresh=None,
    forbidden=None,
    required=None,
    max_parents=None,
):
    """Every legal move with delta below ``floor``, rescanned u-major, v-minor.

    ``forbidden[v, u]`` bars an edge ``u -> v`` no add or reverse makes, ``required[v, u]`` one no
    move removes or reverses, and ``max_parents`` caps the parents of a node that an add or a
    reverse would give one. ``fresh`` is the add and remove deltas of every target, from
    :func:`_fresh_deltas`.
    """
    fresh = fresh or [_fresh_deltas(score, v, parents_of[v], d) for v in range(d)]
    add = [f[0] for f in fresh]
    remove = [f[1] for f in fresh]
    reach = _reachable(adj_int, d)
    inv_n = 1.0 / score.n
    best = None

    def offer(value, u, v, rank, op):
        nonlocal best
        if value < floor and _key_after(adj_int, (op, u, v)) not in tabu:
            cand = (round(value * inv_n, KEY_DP), u, v, rank, (op, u, v))
            key = (cand[0], ranks[u], ranks[v], rank)
            if best is None or key < (best[0], ranks[best[1]], ranks[best[2]], best[3]):
                best = cand

    def barred(rules, v, u):
        return rules is not None and rules[v, u]

    def full(v):
        return max_parents is not None and len(parents_of[v]) >= max_parents

    for u in range(d):
        for v in range(d):
            if u == v:
                continue
            if adj_int[u] >> v & 1:
                if barred(required, v, u):
                    continue
                offer(remove[v][u], u, v, 0, "remove")
                others = adj_int[u] & ~(1 << v)
                blocked = any(others >> c & 1 and reach[c, v] for c in range(d))
                if not (blocked or barred(forbidden, u, v) or full(u)):
                    covered = parents_of[v] == sorted([*parents_of[u], u])
                    with np.errstate(invalid="ignore"):
                        value = add[u][v] + remove[v][u]
                    offer(0.0 if covered and np.isfinite(value) else value, u, v, 1, "reverse")
            elif not adj_int[v] >> u & 1 and not reach[v, u] and not barred(forbidden, v, u):
                if not full(v):
                    offer(add[v][u], u, v, 0, "add")
    return best


def _reference_table(fresh, adj_int, parents_of, d, forbidden, required, max_parents):
    """The move table a scanner holds, built from scratch, and the cells it leaves unchecked.

    Cell ``[v, u]`` is the add or remove delta of ``u -> v``, ``inf`` where ``v -> u`` exists, the
    delta is NaN, a forbidden edge would be added, a required edge would be removed, or a full
    parent set would take an add. A scanner drops an add that closes a cycle (``v`` reaches ``u``)
    only when it is that row's lowest, so those cells are left out.
    """
    parent = np.zeros((d, d), dtype=bool)
    for v in range(d):
        parent[v, parents_of[v]] = True
    reach = _reachable(adj_int, d)
    table = np.empty((d, d))
    for v in range(d):
        row = np.where(parent[v], fresh[v][1], fresh[v][0])
        row[parent[:, v] | reach[v] | np.isnan(row)] = np.inf
        if forbidden is not None:
            row[forbidden[v] & ~parent[v]] = np.inf
        if required is not None:
            row[required[v]] = np.inf
        if max_parents is not None and len(parents_of[v]) >= max_parents:
            row[~parent[v]] = np.inf
        table[v] = row
    return table, ~reach


def _data(seed: int, n: int, d: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    b = np.tril(rng.uniform(0.4, 1.2, (d, d)) * (rng.random((d, d)) < 0.3), -1)
    x = np.zeros((n, d))
    for j in range(d):
        x[:, j] = rng.standard_normal(n) + x @ b[j]
    return x


def _categorical(seed: int, n: int, d: int) -> np.ndarray:
    """Codes in ``0..2``: each column copies the sum of earlier ones mod 3, or else is noise."""
    rng = np.random.default_rng(seed)
    parents = np.tril(rng.random((d, d)) < 0.3, -1)
    x = np.zeros((n, d), dtype=np.int64)
    for j in range(d):
        copy = x[:, parents[j]].sum(axis=1) % 3
        x[:, j] = np.where(rng.random(n) < 0.6, copy, rng.integers(0, 3, n))
    return x.astype(np.float64)


@pytest.mark.parametrize("seed", range(3))
def test_scanner_picks_as_full_rescan(seed: int) -> None:
    """Along a walk that adds, reverses, and drops edges, picks and keys match a full rescan.

    The walk takes the best move outside a short tabu list, worsening or not, and every third step
    drops a random edge, so targets return to earlier parent sets and adds that closed a cycle no
    longer do.
    """
    d = 12
    x = _data(seed, 400, d)
    ranks = tie_ranks(x)
    scanner = MoveScanner(DeltaBICScore(BICScore(x)), d, ranks=ranks)
    reference = BICScore(x)
    rng = np.random.default_rng(seed)
    adj_int = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    recent: list[tuple[int, ...]] = []
    for step in range(90):
        recent = [*recent[-5:], tuple(adj_int)]
        tabu = set(recent)
        for floor in (-EPS_IMPROVE, math.inf):
            want = _reference_scan(reference, adj_int, parents_of, d, floor, tabu, ranks)
            assert scanner.best(adj_int, parents_of, floor=floor, tabu=tabu) == want
        move = want
        edges = [(u, v) for v in range(d) for u in parents_of[v]]
        if step % 3 == 2 and edges:
            u, v = edges[rng.integers(len(edges))]
            apply_move(adj_int, parents_of, ("remove", u, v))
        elif move is not None:
            apply_move(adj_int, parents_of, move[4])
    assert sum(len(p) for p in parents_of.values()) > 0


def _rules(d):
    """Edges the knowledge requires (``0 -> 1``, ``2 -> 3``, ``1 -> 3``) and forbids (``5 -> 0``,
    ``6 -> 0``, ``7 -> 4``), as ``[v, u]`` matrices."""
    required = np.zeros((d, d), dtype=bool)
    for u, v in ((0, 1), (2, 3), (1, 3)):
        required[v, u] = True
    forbidden = np.zeros((d, d), dtype=bool)
    for u, v in ((5, 0), (6, 0), (7, 4)):
        forbidden[v, u] = True
    return forbidden, required


def _with_nan_scores(score):
    """``score`` with a NaN for about one parent set in eleven, by target and set."""
    real = score.score

    def one(v, parents):
        return (
            math.nan if (v * 31 + sum(parents) * 7 + len(parents)) % 11 == 0 else real(v, parents)
        )

    score.score = one
    score.score_many = lambda v, sets: [one(v, parents) for parents in sets]
    return score


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    ("score_func", "rules", "max_parents", "nan"),
    [
        ("bic", False, None, False),
        ("bic", True, 2, False),
        ("bdeu", False, None, False),
        ("discrete_bic", True, None, False),
        ("bdeu", True, 3, True),
    ],
    ids=["bic", "bic rules", "bdeu", "discrete bic rules", "bdeu nan rules"],
)
def test_scanner_keeps_every_delta_exactly(seed, score_func, rules, max_parents, nan) -> None:
    """Along a walk, each call's table holds the deltas a scan from scratch computes, to the bit.

    The scanner keeps no table of remove deltas of its own: the move table holds each one at its
    edge, so a row rebuilt for new reach or a changed neighbor must find it there, a held tabu move
    must leave it unchanged, and the reverse scores must read it. Besides the picks, which match a
    full rescan under the same rules, every cell of the table that no cycle check has left behind
    equals the cell built from scratch. The scores include NaN deltas, a required edge, a full
    parent set, and a forbidden edge.
    """
    d = 9
    x = _data(seed, 400, d) if score_func == "bic" else _categorical(seed, 800, d)
    scores = [make_score(score_func, x) for _ in range(2)]
    if nan:
        scores = [_with_nan_scores(score) for score in scores]
    scored, reference = scores
    forbidden, required = _rules(d) if rules else (None, None)
    ranks = tie_ranks(x)
    scanner = MoveScanner(
        make_delta(scored),
        d,
        ranks=ranks,
        forbidden=forbidden,
        required=required,
        max_parents=max_parents,
    )
    adj_int: list[int] = [0] * d
    parents_of: dict[int, list[int]] = {i: [] for i in range(d)}
    for v, u in zip(*np.nonzero(required)) if rules else ():
        apply_move(adj_int, parents_of, ("add", int(u), int(v)))
    rng = np.random.default_rng(seed)
    kwargs = {"forbidden": forbidden, "required": required, "max_parents": max_parents}
    recent: list[tuple[int, ...]] = []
    held = {"remove": 0, "add": 0, "reverse": 0}
    for step in range(90):
        recent = [*recent[-5:], tuple(adj_int)]
        tabu = set(recent)
        fresh = [_fresh_deltas(reference, v, parents_of[v], d) for v in range(d)]
        table, checked = _reference_table(
            fresh, adj_int, parents_of, d, forbidden, required, max_parents
        )
        want = None
        for floor in (-EPS_IMPROVE, math.inf):
            args = (reference, adj_int, parents_of, d, floor)
            want = _reference_scan_under_rules(*args, tabu, ranks, fresh=fresh, **kwargs)
            free = _reference_scan_under_rules(*args, set(), ranks, fresh=fresh, **kwargs)
            assert scanner.best(adj_int, parents_of, floor=floor, tabu=tabu) == want
            if free != want and free is not None:
                held[free[4][0]] += 1
            got = scanner._single
            np.testing.assert_array_equal(
                got[checked].view(np.uint64), table[checked].view(np.uint64)
            )
        edges = [(u, v) for v in range(d) for u in parents_of[v] if not rules or not required[v, u]]
        reach = _reachable(adj_int, d)
        adds = [
            (u, v)
            for u, v in itertools.permutations(range(d), 2)
            if not (adj_int[u] >> v & 1 or adj_int[v] >> u & 1 or reach[v, u])
            and not (rules and forbidden[v, u])
            and (max_parents is None or len(parents_of[v]) < max_parents)
            and np.isfinite(fresh[v][0][u])
        ]
        if step >= 20 and step % 4 == 2 and edges:
            u, v = edges[rng.integers(len(edges))]
            apply_move(adj_int, parents_of, ("remove", u, v))
        elif step >= 20 and step % 4 == 0 and adds:  # the worst add: the best move then undoes it
            u, v = max(adds, key=lambda edge: fresh[edge[1]][0][edge[0]])
            apply_move(adj_int, parents_of, ("add", u, v))
        elif want is not None:
            apply_move(adj_int, parents_of, want[4])
    assert held["remove"] > 0, held


@pytest.mark.parametrize("seed", range(8))
def test_reversing_a_covered_edge_ties_at_zero(seed: int) -> None:
    """Reversing a covered edge leads to an equivalent DAG: its delta is zero and never improves.

    On the complete DAG ``0 -> 1 -> 2, 0 -> 2`` with nearly collinear ``0`` and ``1``, every removal
    loses a strong effect and both legal reverses are covered. Their Schur deltas carry rounding
    noise far above ``EPS_IMPROVE``, so only an exact zero leaves no improving move.
    """
    rng = np.random.default_rng(seed)
    n = 100_000
    w = rng.standard_normal(n)
    u = w + 1e-3 * rng.standard_normal(n)
    x = np.column_stack([w, u, 1000 * (u - w) + rng.standard_normal(n)])
    scanner = MoveScanner(DeltaBICScore(BICScore(x)), 3, ranks=np.arange(3))
    adj_int, parents_of = [0b110, 0b100, 0], {0: [], 1: [0], 2: [0, 1]}
    assert scanner.best(adj_int, parents_of) is None
    assert scanner.best(adj_int, parents_of, floor=math.inf) == (0.0, 0, 1, 1, ("reverse", 0, 1))


@pytest.mark.parametrize("search", [hc, tabu])
def test_no_columns_give_an_empty_graph(search) -> None:
    graph, _ = search(np.zeros((10, 0)))
    assert graph.to_numpy().shape == (0, 0)


_TIED_DATA = {
    "bic": lambda seed: _data(seed, 500, 10),
    "bdeu": lambda seed: _categorical(seed, 1000, 10),
    "discrete_bic": lambda seed: _categorical(seed, 1000, 10),
}


def _graph(search, x, score_func, order):
    """The CPDAG ``search`` finds on the columns of ``x`` in ``order``, mapped back."""
    back = np.argsort(order)
    return search(x[:, order], score_func=score_func)[0].to_numpy()[np.ix_(back, back)]


@pytest.mark.parametrize("search", [hc, tabu])
@pytest.mark.parametrize("score_func", list(_TIED_DATA))
def test_exact_ties_break_independently_of_column_order(search, score_func) -> None:
    """Every column order gives the graph of the original order, mapped back.

    Under these score-equivalent scores, the first move adds an edge whose opposite direction has
    the same key, so ties decide the walk. Ranking the columns by position instead changes the graph
    under at least one of the orders, so the orders exercise the tie-break.
    """
    x = _TIED_DATA[score_func](1)
    d = x.shape[1]
    score = make_score(score_func, x)
    delta = DeltaBICScore(score) if isinstance(score, BICScore) else FullScoreDelta(score)
    adj_int, parents_of = [0] * d, {i: [] for i in range(d)}
    first = MoveScanner(delta, d, ranks=tie_ranks(x)).best(adj_int, parents_of)
    assert first is not None and first[4][0] == "add"
    _, u, v = first[4]
    assert round((score.score(u, [v]) - score.score(u, [])) / score.n, KEY_DP) == first[0]

    rng = np.random.default_rng(0)
    orders = [np.arange(d)[::-1], *(rng.permutation(d) for _ in range(3))]
    want = _graph(search, x, score_func, np.arange(d))
    for order in orders:
        np.testing.assert_array_equal(_graph(search, x, score_func, order), want)

    with andrey.config(tie_break="position"):
        by_position = _graph(search, x, score_func, np.arange(d))
        assert any((_graph(search, x, score_func, o) != by_position).any() for o in orders)


def test_tie_ranks_move_with_the_columns() -> None:
    """Reordering columns reorders their ranks, also for columns that only their pattern or their
    values in row order tell apart; identical columns rank by position."""
    x = _data(0, 50, 6)
    order = np.array([3, 0, 5, 1, 4, 2])
    np.testing.assert_array_equal(tie_ranks(x[:, order]), tie_ranks(x)[order])
    np.testing.assert_array_equal(tie_ranks(x[:, [2, 2, 2]]), [0, 1, 2])
    copies = x[:, [2, 2]] * np.array([1.0, -3.7]) + np.array([0.0, 10.0])
    swapped = np.vstack([x[:, :3], x[:, [1, 0, 2]]])  # columns 0 and 1 trade places row by row
    near = x[:, [2, 2]] + np.column_stack([np.zeros(50), 1e-11 * np.sin(np.arange(50))])
    for data, order in ((copies, [1, 0]), (swapped, [1, 0, 2]), (near, [1, 0])):
        np.testing.assert_array_equal(tie_ranks(data[:, order]), tie_ranks(data)[order])


def test_tie_ranks_stay_importable_from_hc() -> None:
    """Code written against earlier builds imports ``tie_ranks`` from ``andrey.search.hc``."""
    from andrey.search.hc import tie_ranks as from_hc

    assert from_hc is tie_ranks


_COPIED = np.r_[0, np.random.default_rng(10).integers(-10, 11, 127), 0].astype(np.float64)


@pytest.mark.parametrize("scale", [0.5, 2, 3.7])
@pytest.mark.parametrize("offset", [1, -2.5, 1000])
@pytest.mark.parametrize("unrelated", [False, True])
def test_copies_in_other_units_rank_by_position(scale, offset, unrelated) -> None:
    """Copies ``scale * x + offset`` of one column, which nothing in the data tells apart, rank in
    the order of their columns, beside an unrelated column or alone."""
    x, copy = _COPIED, scale * _COPIED + offset
    other = [np.random.default_rng(11).standard_normal(len(x))] if unrelated else []
    for first, second in ((x, copy), (copy, x)):
        ranks = tie_ranks(np.column_stack([first, *other, second]))
        assert ranks[0] < ranks[-1]


@pytest.mark.parametrize("unrelated", [False, True])
def test_relabeled_categories_rank_by_position(unrelated) -> None:
    """A categorical column and its relabelings share one partition of the rows, so they rank in
    the order of their columns."""
    rng = np.random.default_rng(4)
    codes = rng.integers(0, 3, 200)
    labels = [
        np.array(labeling, dtype=np.float64)[codes]
        for labeling in ([0, 1, 2], [2, 0, 1], [5, 3, 9])
    ]
    other = [rng.integers(0, 2, 200).astype(np.float64)] if unrelated else []
    for first, second in itertools.permutations(labels, 2):
        ranks = tie_ranks(np.column_stack([first, *other, second]), np.ones(2 + len(other), bool))
        assert ranks[0] < ranks[-1]


_ROWS = np.random.default_rng(1).permutation(300)


def _with_missing(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[np.random.default_rng(2).random(x.shape) < 0.05] = np.nan
    return x


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x[_ROWS],
        lambda x: 3.7 * x + 10,
        lambda x: x * np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0]),
    ],
    ids=["row shuffle", "units", "sign flip"],
)
@pytest.mark.parametrize("missing", [False, True])
def test_tie_ranks_ignore_row_order_units_and_sign(change, missing) -> None:
    """Column 5 rises with column 0, so the two share their relations and their marginals decide."""
    x = _data(0, 300, 6)
    x[:, 5] = np.exp(x[:, 0])
    if missing:
        x = _with_missing(x)
    np.testing.assert_array_equal(tie_ranks(change(x)), tie_ranks(x))


def test_one_or_two_columns_rank_by_their_marginals() -> None:
    """Two columns have one relation, which cannot tell them apart; their marginals do."""
    x = _data(0, 300, 6)[:, [1, 4]]
    want = tie_ranks(x)
    np.testing.assert_array_equal(tie_ranks(x[:, ::-1]), want[::-1])
    np.testing.assert_array_equal(tie_ranks(-3.7 * x[_ROWS] + 10), want)
    np.testing.assert_array_equal(tie_ranks(x[:, :1]), [0])


def test_recoding_categories_keeps_the_tie_ranks() -> None:
    """A categorical column ranks by which rows share a category, not by the codes or the rows."""
    x = _categorical(0, 300, 6)
    recoded = np.array([2.0, 0.0, 1.0])[x[_ROWS].astype(int)]
    every = np.ones(6, dtype=bool)
    assert not np.array_equal(tie_ranks(recoded), tie_ranks(x))
    np.testing.assert_array_equal(tie_ranks(recoded, every), tie_ranks(x, every))


def test_mixed_tie_ranks_ignore_rows_units_sign_and_labels() -> None:
    """Categorical columns beside continuous ones: the changes of the continuous tests and a
    relabeling of the categories keep the ranks, and a column order relabels them."""
    x = _data(3, 300, 6)
    x[:, 1] = np.digitize(x[:, 0], [-0.5, 0.5])
    x[:, 4] = np.random.default_rng(3).integers(0, 2, 300)
    mask = np.isin(np.arange(6), [1, 4])
    want = tie_ranks(x, mask)
    changed = x[_ROWS] * np.array([3.7, 1.0, -1.0, 1.0, 1.0, 0.2]) + 10
    changed[:, 1] = np.array([5.0, -2.0, 9.0])[x[_ROWS, 1].astype(int)]
    changed[:, 4] = 1 - x[_ROWS, 4]
    np.testing.assert_array_equal(tie_ranks(changed, mask), want)
    missing = _with_missing(x)
    np.testing.assert_array_equal(tie_ranks(missing[_ROWS], mask), tie_ranks(missing, mask))
    order = np.array([4, 2, 0, 5, 1, 3])
    np.testing.assert_array_equal(tie_ranks(x[:, order], mask[order]), want[order])


def test_rescaling_the_identity_corpus_moves_no_rank() -> None:
    """On each draw the identity check fits, units that are not a power of two, which move the
    last bits of the marginals, leave every rank in place."""
    draws = {(dataset, seed): data for _, dataset, seed, data in identity.corpus(identity.FITS, 2)}
    assert len(draws) > 50
    for data in draws.values():
        x = data.reshape(-1, data.shape[-1])
        np.testing.assert_array_equal(tie_ranks(3.7 * x), tie_ranks(x))


@pytest.mark.parametrize("search", [hc, tabu])
def test_rounding_in_the_last_bits_keeps_the_graph(search) -> None:
    """Small perturbations of these data preserve column ranks and the final CPDAG.

    The perturbations model last-bit differences in BLAS output across machines.
    """
    x = _TIED_DATA["bic"](1)
    rng = np.random.default_rng(0)
    unit = np.finfo(np.float64).eps * np.abs(x).max(axis=0)
    nudged = x + rng.integers(-4, 5, size=x.shape) * unit
    assert (nudged != x).any(axis=0).all()
    np.testing.assert_array_equal(tie_ranks(nudged), tie_ranks(x))
    np.testing.assert_array_equal(search(nudged)[0].to_numpy(), search(x)[0].to_numpy())
