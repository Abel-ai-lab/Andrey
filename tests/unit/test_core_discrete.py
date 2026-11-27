"""The categorical primitives: the chi-square and G-square tests, the BDeu and multinomial BIC
scores, and the codes.

Each statistic is checked against an independent computation: SciPy's contingency-table test for
one table, a per-configuration loop for a conditional test, the BDeu formula written out with
``math.lgamma``, and the multinomial BIC written out per cell.
"""

from __future__ import annotations

import math
import tracemalloc
from itertools import combinations, permutations

import numpy as np
import pytest
from scipy.special import chdtrc
from scipy.stats import chi2, chi2_contingency, hypergeom

from andrey.core import categorical, ci
from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore, CategoricalBICScore, make_score


def _codes(seed: int = 0, n: int = 400, cards=(2, 3, 4, 3)) -> np.ndarray:
    """Categorical columns where column 1 copies column 0 in most rows, so some pairs depend."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.integers(0, k, n) for k in cards])
    x[:, 1] = np.where(rng.random(n) < 0.6, x[:, 0], x[:, 1])
    return x.astype(np.float64)


def _table(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """The observed table of columns ``a`` and ``b``, without empty rows or columns."""
    va, ia = np.unique(x[:, a], return_inverse=True)
    vb, ib = np.unique(x[:, b], return_inverse=True)
    table = np.zeros((va.size, vb.size))
    np.add.at(table, (ia, ib), 1)
    return table


def _null_mean(table: np.ndarray, g_square: bool) -> float:
    """The mean of the statistic over the arrangements of ``y`` that keep a table's margins.

    Every cell's count is hypergeometric, so the mean of G-square is a sum over each cell's
    support, and Pearson's is Haldane's ``n * (r - 1) * (c - 1) / (n - 1)``.
    """
    n, r, c = table.sum(), table.sum(axis=1), table.sum(axis=0)
    if min(table.shape) < 2:
        return 0.0
    if not g_square:
        return float(n * (len(r) - 1) * (len(c) - 1) / (n - 1))
    total = 0.0
    for ri in r:
        for cj in c:
            t = np.arange(max(1, ri + cj - n), min(ri, cj) + 1)
            total += float(np.sum(hypergeom.pmf(t, n, ri, cj) * 2 * t * np.log(t * n / (ri * cj))))
    return total


@pytest.mark.parametrize(("g_square", "lambda_"), [(False, None), (True, "log-likelihood")])
def test_a_marginal_test_refers_the_statistic_to_its_null_mean(g_square, lambda_):
    """The statistic is SciPy's; the p-value is the chi-square survival function at the mean the
    statistic has when ``y`` is shuffled (Pearson's exactly, G-square's to the expansion error)."""
    x = _codes()
    test = ChiSquare(x, g_square=g_square)
    for a, b in combinations(range(x.shape[1]), 2):
        table = _table(x, a, b)
        stat = chi2_contingency(table, correction=False, lambda_=lambda_)[0]
        expected = chi2.sf(stat, _null_mean(table, g_square))
        assert test(a, b) == pytest.approx(expected, rel=0.05 if g_square else 1e-9, abs=1e-300)


@pytest.mark.parametrize(("g_square", "lambda_"), [(False, None), (True, "log-likelihood")])
def test_a_conditional_test_sums_one_table_per_configuration(g_square, lambda_):
    """The statistic and the null mean add up over the conditioning configurations, each table
    counting only the values present in it."""
    x = _codes(seed=1, n=600)
    statistic = mean = 0.0
    for config in np.unique(x[:, [2, 3]], axis=0):
        rows = x[(x[:, [2, 3]] == config).all(axis=1)]
        table = _table(rows, 0, 1)
        if min(table.shape) > 1:
            statistic += chi2_contingency(table, correction=False, lambda_=lambda_)[0]
            mean += _null_mean(table, g_square)
    assert ChiSquare(x, g_square=g_square)(0, 1, (2, 3)) == pytest.approx(
        chi2.sf(statistic, mean), rel=0.05 if g_square else 1e-9
    )


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize("shape", [(2, 2), (2, 3), (3, 3)])
def test_the_null_mean_is_the_mean_over_every_arrangement(g_square, shape):
    """Brute force: average the statistic over every arrangement of ``y`` in a small table."""
    kx, ky = shape
    x = np.arange(7) % kx
    y = np.array([0, 1, 2, 0, 1, 2, 1])[:7] % ky
    n_sx = np.bincount(x, minlength=kx)[None, :]
    n_sy = np.bincount(y, minlength=ky)[None, :]
    stats = []
    for arrangement in set(permutations(y)):  # each distinct one occurs equally often
        table = np.zeros(shape)
        np.add.at(table, (x, np.array(arrangement)), 1)
        expected = np.outer(table.sum(1), table.sum(0)) / 7
        seen = table > 0
        stats.append(
            2 * np.sum(table[seen] * np.log(table[seen] / expected[seen]))
            if g_square
            else np.sum((table - expected) ** 2 / expected)
        )
    got = ci._null_means(n_sx, n_sy, np.array([7]), g_square)
    assert got[0] == pytest.approx(np.mean(stats), rel=1e-9)


def test_a_pair_with_no_degrees_of_freedom_has_p_value_one():
    """When ``y`` takes one value in every configuration of the conditioning set, no table has a
    degree of freedom, and the test cannot reject independence."""
    x = _codes()
    x[:, 3] = x[:, 2]  # y is a copy of the conditioning column
    for g_square in (False, True):
        assert ChiSquare(x, g_square=g_square)(0, 3, (2,)) == 1.0


def _sparse_strata() -> np.ndarray:
    """``x`` and ``y`` with three values each, given two binary columns, in three of the four
    configurations: every cell occupied in ``(0, 0)``, ``x`` without its third value in ``(0, 1)``,
    ``y`` with one value in ``(1, 0)``, and no row in ``(1, 1)``."""
    rows = [(a, b, 0, 0) for a in range(3) for b in range(3)] * 2 + [(0, 0, 0, 0), (2, 2, 0, 0)]
    rows += [(a, b, 0, 1) for a in range(2) for b in range(3)] + [(0, 0, 0, 1)]
    rows += [(a, 1, 1, 0) for a in range(3)]
    return np.array(rows, dtype=np.float64)


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
def test_nonempty_degrees_of_freedom_strike_the_empty_rows_and_columns(g_square):
    """Each configuration counts ``(r_x - 1) * (r_y - 1)`` over the values present: 4 for the full
    table, 2 without a value of ``x``, 0 with one value of ``y`` or with no row, where the nominal
    ``(3 - 1) * (3 - 1) * 4 = 16`` counts every cell. The statistic is the same under both rules."""
    x = _sparse_strata()
    nonempty = ChiSquare(x, g_square=g_square, degrees_of_freedom="nonempty")
    mean = ChiSquare(x, g_square=g_square)
    known = {(): 4, (2,): 4, (3,): 6, (2, 3): 6}
    for cond, df in known.items():
        statistic, p_value = nonempty._statistic_pvalue(0, 1, cond)
        assert statistic == mean._statistic_pvalue(0, 1, cond)[0]
        assert p_value == chdtrc(df, statistic) == nonempty(1, 0, cond)
    assert nonempty(0, 1, (2, 3)) != mean(0, 1, (2, 3))


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
def test_nonempty_degrees_of_freedom_count_the_values_present_in_each_configuration(g_square):
    """Against a direct count over the rows of each configuration, on sparse random tables."""
    x = _codes(seed=5, n=40, cards=(3, 4, 3, 3))
    test = ChiSquare(x, g_square=g_square, degrees_of_freedom="nonempty")
    for a, b in combinations(range(4), 2):
        others = [c for c in range(4) if c not in (a, b)]
        for cond in (s for size in range(3) for s in combinations(others, size)):
            df = 0
            for key in {tuple(row) for row in x[:, list(cond)]}:
                rows = np.all(x[:, list(cond)] == key, axis=1)
                df += (np.unique(x[rows, a]).size - 1) * (np.unique(x[rows, b]).size - 1)
            statistic = test._statistic_pvalue(a, b, cond)[0]
            assert test(a, b, cond) == (1.0 if df == 0 else chdtrc(df, statistic))


def test_the_degrees_of_freedom_reach_the_categorical_tests_alone():
    """``make_indep_test`` passes ``degrees_of_freedom`` to ``chisq`` and ``gsq``, also on data with
    missing values, and the other tests ignore it; an unknown rule raises."""
    x = _sparse_strata()
    want = ChiSquare(x, g_square=True, degrees_of_freedom="nonempty")(0, 1, (2, 3))
    assert ci.make_indep_test("gsq", x, degrees_of_freedom="nonempty")(0, 1, (2, 3)) == want
    gaps = np.vstack([x, [[np.nan, 0, 0, 0]]])
    assert ci.make_indep_test("gsq", gaps, degrees_of_freedom="nonempty")(0, 1, (2, 3)) == want
    fisherz = ci.make_indep_test("fisherz", x)(0, 1, (2,))
    assert ci.make_indep_test("fisherz", x, degrees_of_freedom="nonempty")(0, 1, (2,)) == fisherz
    with pytest.raises(ValueError, match="degrees_of_freedom must be one of"):
        ChiSquare(x, degrees_of_freedom="cells")


def test_dependence_and_independence_are_told_apart():
    x = _codes(n=2000)
    test = ChiSquare(x)
    assert test(0, 1) < 1e-10  # column 1 copies column 0
    assert test(2, 3) > 0.01  # independent draws


def test_p_values_do_not_depend_on_codes_or_argument_order():
    x = _codes()
    recoded = x * np.array([3.0, -2.0, 1.0, 7.0]) + 5.0
    for g in (False, True):
        a, b = ChiSquare(x, g_square=g), ChiSquare(recoded, g_square=g)
        assert a(1, 0, (3, 2, 3)) == a(0, 1, (2, 3))
        assert b(0, 1, (2, 3)) == pytest.approx(a(0, 1, (2, 3)), rel=1e-12)


def test_a_sparse_table_counts_only_the_occupied_cells(monkeypatch):
    """Above the dense-table limit the cells that occur are counted directly; same p-value."""
    from andrey.core import ci

    x = _codes(n=300, cards=(4, 5, 6, 6))
    dense = ChiSquare(x)(0, 1, (2, 3))
    monkeypatch.setattr(ci, "_DENSE_CELLS", 0)
    assert ChiSquare(x)(0, 1, (2, 3)) == pytest.approx(dense, rel=1e-12)


def _fast_path_data(case: str) -> np.ndarray:
    """Categorical columns of one of three kinds of table.

    ``"dependent"``: tables with every cell filled. ``"sparse"``: few rows and many categories,
    so most cells are empty and many configurations hold one row. ``"one value"``: a column with
    one category, and a pair whose first column takes one value in some configurations, so a test
    has zero degrees of freedom.
    """
    if case == "dependent":
        return _codes(n=400, cards=(2, 3, 4, 3, 5))
    if case == "sparse":
        return _codes(seed=2, n=15, cards=(6, 5, 7, 4, 6))
    x = _codes(seed=3, n=60, cards=(2, 3, 3, 2, 4))
    x[:, 3] = 1.0
    x[:, 0] = np.where(x[:, 2] == 0, 0.0, x[:, 0])
    return x


@pytest.mark.parametrize("rule", ci.DEGREES_OF_FREEDOM)
@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize("case", ["dependent", "sparse", "one value"])
@pytest.mark.parametrize(
    ("batch", "rows", "table"),
    [(1 << 22, 4, 1 << 16), (64, 4, 1 << 16), (1 << 22, 0, 40)],
    ids=["one count", "a count per set", "large tables on the scalar path"],
)
def test_every_fast_path_returns_the_scalar_p_values(
    monkeypatch, rule, g_square, case, batch, rows, table
):
    """The batched, first-order, and all-pairs marginal paths equal the scalar call bit for bit,
    and so do the statistics of one pair's queries over many sets: whatever the argument order,
    with every table in one count or one count per set and the marginal tables in one Gram block
    or several, and with tables over the size limit on the scalar path."""
    from andrey.core import ci

    monkeypatch.setattr(ci, "BATCH_CELLS", batch)
    monkeypatch.setattr(ci, "DENSE_ROWS", rows)
    monkeypatch.setattr(ci, "DENSE_MIN", table)
    x = _fast_path_data(case)
    d = x.shape[1]
    test = ChiSquare(x, g_square=g_square, degrees_of_freedom=rule)
    marginal = test.marginal_pvalues()
    for a, b in combinations(range(d), 2):
        assert marginal[a, b] == marginal[b, a] == test(a, b)
        others = [c for c in range(d) if c not in (a, b)]
        sets = [s[::-1] for size in range(4) for s in combinations(others, size)][::-1]
        assert test.batched_call(b, a, sets) == [test(a, b, s) for s in sets]
        assert test.first_order_pvalues(b, a, others).tolist() == [test(a, b, (c,)) for c in others]
        for size in range(4):
            same = np.array([s for s in sets if len(s) == size], dtype=np.intp)
            same = same.reshape(len(same), size)
            got = test.statistic_pvalues(np.full(len(same), b), np.full(len(same), a), same)
            want = np.array([test._statistic_pvalue(a, b, s) for s in same.tolist()]).T
            assert np.array_equal(np.array(got).view(np.int64), want.view(np.int64))


def _dense_tables(data: np.ndarray, x: int, y: int, conds: list[tuple[int, ...]], g_square: bool):
    """Statistic, degrees of freedom, and p-value of each set, from one table per possible
    configuration of the set, counted together in one ``bincount``, the empty ones dropped.

    Every set has the same size. The arithmetic after the counting is the batched path's: each
    set's statistic is one ``np.sum`` over its occupied cells, in configuration, ``x``, ``y`` order.
    """
    codes, cards = categorical.encode(data)
    n = codes.shape[0]
    kx, ky = int(cards[x]), int(cards[y])
    cells = kx * ky
    sets = np.array(conds, dtype=np.intp).reshape(len(conds), -1)
    radix = cards[sets]
    q = radix.prod(axis=1)
    offset = np.cumsum(q) - q
    total = int(q.sum())
    index = np.repeat(offset[:, None], n, axis=1)
    for j in range(sets.shape[1]):
        stride = np.prod(radix[:, j + 1 :], axis=1)
        index += codes[:, sets[:, j]].T * stride[:, None]
    index = index * cells + codes[:, x] * ky + codes[:, y]
    tables = np.bincount(index.ravel(), minlength=total * cells).reshape(total, kx, ky)
    occurs = tables.reshape(total, cells).any(axis=1)
    bounds = np.concatenate(([0], np.cumsum(np.add.reduceat(occurs.astype(np.int64), offset))))
    tables = tables[occurs]
    n_sx = tables.sum(axis=2)
    n_sy = tables.sum(axis=1)
    n_s = n_sx.sum(axis=1)
    flat = tables.reshape(-1)
    occupied = np.flatnonzero(flat)
    observed = flat[occupied]
    s, rest = np.divmod(occupied, cells)
    sx, sy = np.divmod(rest, ky)
    expected = n_sx[s, sx] * n_sy[s, sy] / n_s[s]
    if g_square:
        values = observed * np.log(observed / expected)
    else:
        values = observed * (observed / expected)
    cuts = np.searchsorted(occupied, bounds * cells).tolist()
    sums = np.array([np.sum(values[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    statistic = 2.0 * sums if g_square else sums - float(n)
    per_table = ci._null_means(n_sx, n_sy, n_s, g_square)
    cuts = bounds.tolist()
    dof = np.array([np.sum(per_table[a:b]) for a, b in zip(cuts[:-1], cuts[1:])])
    pvalue = np.ones(len(dof))
    live = dof != 0
    pvalue[live] = chdtrc(dof[live], np.maximum(statistic[live], 0.0))
    return statistic, dof, pvalue


def _occupied_tables_data(case: str) -> np.ndarray:
    """Categorical columns whose conditioning sets leave many configurations empty.

    ``"empty configurations"``: more configurations than rows. ``"one value per configuration"``:
    column 0 takes one value wherever column 2 is 0, so those tables have an empty row and fewer
    degrees of freedom. ``"one category"``: column 1 has one category. ``"many categories"``:
    twelve categories per conditioning column, so a set of four has 20,736 configurations against
    150 rows. ``"two rows"``: every table has one cell per row, so no degree of freedom.
    """
    if case == "empty configurations":
        return _codes(seed=4, n=200, cards=(2, 3, 4, 3, 5, 2))
    if case == "one value per configuration":
        x = _codes(seed=5, n=300, cards=(3, 3, 3, 4, 2, 3))
        x[:, 0] = np.where(x[:, 2] == 0, 0.0, x[:, 0])
        return x
    if case == "one category":
        x = _codes(seed=6, n=120, cards=(3, 2, 4, 3, 2))
        x[:, 1] = 1.0
        return x
    if case == "many categories":
        return _codes(seed=7, n=150, cards=(3, 3, 12, 12, 12, 12))
    return _codes(seed=8, n=2, cards=(2, 2, 3, 3))


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize(
    "case",
    [
        "empty configurations",
        "one value per configuration",
        "one category",
        "many categories",
        "two rows",
    ],
)
@pytest.mark.parametrize("batch", [1 << 22, 64], ids=["one count", "a count per set"])
def test_batched_counts_equal_the_tables_of_every_configuration(monkeypatch, g_square, case, batch):
    """Counting only the configurations that occur gives every set the statistic, degrees of
    freedom, and p-value, bit for bit, of counting a table for every possible configuration and
    dropping the empty ones, whatever the set's size."""
    from andrey.core import ci

    monkeypatch.setattr(ci, "BATCH_CELLS", batch)
    monkeypatch.setattr(ci, "DENSE_MIN", 1 << 24)  # every set on the batched path
    recorded: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_chdtrc(dof, statistic):
        recorded.append((np.asarray(dof).copy(), np.asarray(statistic).copy()))
        return chdtrc(dof, statistic)

    monkeypatch.setattr(ci, "chdtrc", recording_chdtrc)
    x = _occupied_tables_data(case)
    d = x.shape[1]
    test = ChiSquare(x, g_square=g_square)
    for a, b in [(0, 1), (1, d - 1), (0, d - 1)]:
        others = [c for c in range(d) if c not in (a, b)]
        for size in range(min(len(others), 4) + 1):
            sets = list(combinations(others, size))
            if len(sets) == 1:
                sets = sets * 2  # a batch holds at least two sets
            statistic, dof, pvalue = _dense_tables(x, a, b, sets, g_square)
            recorded.clear()
            got = np.array(test.batched_call(b, a, sets))
            assert np.array_equal(got.view(np.int64), pvalue.view(np.int64))
            live = dof != 0
            got_dof = np.concatenate([r[0] for r in recorded]) if recorded else np.array([])
            got_statistic = np.concatenate([r[1] for r in recorded]) if recorded else np.array([])
            assert sorted(zip(got_dof.tolist(), got_statistic.view(np.int64).tolist())) == sorted(
                zip(dof[live].tolist(), np.maximum(statistic[live], 0.0).view(np.int64).tolist())
            )
    if case == "two rows":
        assert (test.batched_call(0, 1, [(2,), (3,), (2, 3)]) == np.ones(3)).all()


def _bdeu(counts: np.ndarray, q: int, a: float = 1.0) -> float:
    """Log BDeu marginal likelihood of ``counts`` (one row per parent configuration)."""
    r = counts.shape[1]
    total = 0.0
    for row in counts:
        total += math.lgamma(a / q) - math.lgamma(row.sum() + a / q)
        for c in row:
            total += math.lgamma(c + a / (r * q)) - math.lgamma(a / (r * q))
    return total


def test_bdeu_matches_the_formula():
    """Three variables, so the structure prior gives each other variable probability 1/2."""
    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]], float)
    score = BDeuScore(x)
    half = math.log(0.5)
    # Node 0 with no parent: counts (4, 2).
    assert score.score(0, []) == pytest.approx(-2 * (_bdeu(np.array([[4, 2]]), 1) + 2 * half))
    # Node 0 given node 1: (2, 0) where node 1 is 0, (2, 2) where it is 1.
    assert score.score(0, [1]) == pytest.approx(
        -2 * (_bdeu(np.array([[2, 0], [2, 2]]), 2) + 2 * half)
    )


def test_bdeu_structure_prior_penalizes_each_parent():
    """From five variables up, p = 1 / (d - 1) makes each parent cost 2 log((1 - p) / p) more."""
    x = _codes(n=300, cards=(2, 3, 2, 3, 2))
    with_prior = BDeuScore(x)
    p = 1 / 4
    penalty = 2 * math.log((1 - p) / p)
    likelihood_gain = BDeuScore(x[:, :3]).score(0, [1]) - BDeuScore(x[:, :3]).score(0, [])
    # At d = 3 the prior is flat (p = 1/2), so the difference there is the likelihood's alone.
    assert with_prior.score(0, [1]) - with_prior.score(0, []) == pytest.approx(
        likelihood_gain + penalty
    )


def test_bdeu_is_score_equivalent():
    """``x -> y`` and ``y -> x`` describe one equivalence class, so they score the same."""
    x = _codes(n=500)
    s = BDeuScore(x)
    for a, b in combinations(range(x.shape[1]), 2):
        assert s.score(a, [b]) + s.score(b, []) == pytest.approx(s.score(b, [a]) + s.score(a, []))


@pytest.mark.parametrize("d", [1, 2])
def test_bdeu_leaves_out_an_undefined_structure_prior(d):
    """With two or fewer variables, structure_prior / (d - 1) is not below 1: no prior term."""
    x = _codes()[:, :d]
    score = BDeuScore(x)
    counts = np.bincount(x[:, 0].astype(int))[None, :]
    assert score.score(0, []) == pytest.approx(-2 * _bdeu(counts, 1))
    if d == 2:
        assert np.isfinite(score.score(0, [1]))


def test_bdeu_does_not_depend_on_the_codes():
    x = _codes()
    a, b = BDeuScore(x), BDeuScore(x * -3.0 + 2.0)
    assert b.score(1, [2, 0]) == pytest.approx(a.score(1, [0, 2]), rel=1e-12)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("equivalent_sample_size", 0.0),
        ("equivalent_sample_size", float("inf")),
        ("structure_prior", -1.0),
    ],
)
def test_bdeu_priors_must_be_positive(name, value):
    with pytest.raises(ValueError, match=name):
        BDeuScore(_codes(), **{name: value})


def test_bdeu_equivalent_sample_size_matches_a_hand_computation():
    """Two variables, so the structure prior is left out and only the Dirichlet term remains.

    Node 0 has counts (4, 2) and no parent. With equivalent sample size ``a`` the marginal
    likelihood is ``G(a) G(4 + a / 2) G(2 + a / 2) / (G(6 + a) G(a / 2) ** 2)``, with ``G`` the
    gamma function: 7 / 1024 for a = 1 and 1 / 105 for a = 2. Given the binary node 1, with counts
    (2, 0) where it is 0 and (2, 2) where it is 1, a = 2 gives 3 / 8 times 3 / 128.
    """
    x = np.array([[0, 0], [0, 0], [0, 1], [0, 1], [1, 1], [1, 1]], float)
    default = BDeuScore(x)
    assert default.equivalent_sample_size == 1.0
    assert BDeuScore(x).score(0, []) == BDeuScore(x, equivalent_sample_size=1.0).score(0, [])
    assert default.score(0, []) == pytest.approx(2 * math.log(1024 / 7))
    two = BDeuScore(x, equivalent_sample_size=2.0)
    assert two.score(0, []) == pytest.approx(2 * math.log(105))
    assert two.score(0, [1]) == pytest.approx(2 * math.log(1024 / 9))
    ten = BDeuScore(x, equivalent_sample_size=10.0)
    assert ten.score(0, []) == pytest.approx(-2 * _bdeu(np.array([[4, 2]]), 1, a=10.0))
    assert ten.score(0, [1]) == pytest.approx(-2 * _bdeu(np.array([[2, 0], [2, 2]]), 2, a=10.0))


def test_make_score_builds_bdeu_with_the_equivalent_sample_size():
    x = _codes()
    score = make_score("bdeu", x, equivalent_sample_size=5.0)
    assert isinstance(score, BDeuScore) and score.equivalent_sample_size == 5.0
    assert make_score("bdeu", x).score(1, [0]) == BDeuScore(x).score(1, [0])
    assert score.on_rows(np.arange(len(x)) % 2 == 0).equivalent_sample_size == 5.0


def test_configurations_number_the_observed_rows_in_code_order():
    """Many high-cardinality columns overflow a plain mixed-radix code; the configurations that
    occur are still numbered 0..m-1 in lexicographic order, at most one per row, whether the codes
    are read row-major or from a column-contiguous int32 copy."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 50, (1000, 12)).astype(np.float64)
    x[:500, :6] = 0.0  # repeated rows, so some configurations occur more than once
    codes, cards = categorical.encode(x)
    columns = list(range(12))
    _, expected = np.unique(codes[:, columns], axis=0, return_inverse=True)
    for view in (codes, codes.T.astype(np.int32, order="C").T):
        index, m = categorical.configurations(view, cards, columns)
        assert m == expected.max() + 1 <= len(x)
        assert index.dtype == np.int64 and np.array_equal(index, expected.reshape(-1))


def test_no_columns_is_one_configuration():
    codes, cards = categorical.encode(_codes())
    index, m = categorical.configurations(codes, cards, [])
    assert m == 1 and not index.any()


@pytest.mark.parametrize("categories", [(2.9, 2.1), (2, 3.5), (True, 3), (3, 2), (1, 4), (2,)])
def test_the_sampler_rejects_category_bounds_that_are_not_ordered_whole_numbers(categories):
    from andrey.data import sample_discrete

    with pytest.raises(ValueError, match="categories must be"):
        sample_discrete(d=3, n=20, seed=0, categories=categories)


def _multinomial_loglik(counts: np.ndarray) -> float:
    """Maximized multinomial log-likelihood of ``counts`` (one row per parent configuration)."""
    total = 0.0
    for row in counts:
        for c in row:
            if c:
                total += c * math.log(c / row.sum())
    return total


def test_categorical_bic_matches_the_formula():
    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 1, 1]], float)
    score = CategoricalBICScore(x)
    log_n = math.log(6)
    # Node 0 with no parent: counts (4, 2), one free parameter.
    assert score.score(0, []) == pytest.approx(-2 * _multinomial_loglik(np.array([[4, 2]])) + log_n)
    # Node 0 given nodes 1 and 2: four configurations, each with one free parameter.
    counts = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    assert score.score(0, [2, 1]) == pytest.approx(-2 * _multinomial_loglik(counts) + 4 * log_n)


def test_categorical_bic_counts_the_parameters_of_unseen_configurations():
    """Two parents of 2 and 3 categories give q = 6 configurations; the 3-category node has 2
    free parameters in each, whether or not a row falls in it."""
    x = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 2], [0, 1, 0], [1, 0, 2], [2, 1, 1]], float)
    x = np.vstack([x, [[0, 0, 1], [0, 0, 2]]])  # parent configuration (0, 0) seen; (1, 2) never
    score = CategoricalBICScore(x, penalty_discount=0.0)
    assert score.score(0, [1, 2]) - CategoricalBICScore(x).score(0, [1, 2]) == pytest.approx(
        -2 * 3 * 2 * math.log(len(x))
    )


def test_categorical_bic_without_a_penalty_is_minus_twice_the_log_likelihood():
    x = _codes()
    score = CategoricalBICScore(x, penalty_discount=0.0)
    counts = np.zeros((2, 3))
    np.add.at(counts, (x[:, 0].astype(int), x[:, 1].astype(int)), 1)
    assert score.score(1, [0]) == pytest.approx(-2 * _multinomial_loglik(counts), rel=1e-12)


def test_categorical_bic_penalty_scales_with_penalty_discount():
    x = _codes()
    plain, double = CategoricalBICScore(x), CategoricalBICScore(x, penalty_discount=2.0)
    free = CategoricalBICScore(x, penalty_discount=0.0)
    for parents in ([], [0], [0, 2]):
        penalty = plain.score(1, parents) - free.score(1, parents)
        assert double.score(1, parents) - free.score(1, parents) == pytest.approx(2 * penalty)


def test_categorical_bic_is_score_equivalent():
    """``x -> y`` and ``y -> x`` describe one equivalence class, so they score the same."""
    x = _codes(n=500)
    s = CategoricalBICScore(x)
    for a, b in combinations(range(x.shape[1]), 2):
        assert s.score(a, [b]) + s.score(b, []) == pytest.approx(s.score(b, [a]) + s.score(a, []))


def test_categorical_bic_does_not_depend_on_the_codes():
    x = _codes()
    a, b = CategoricalBICScore(x), CategoricalBICScore(x * -3.0 + 2.0)
    assert b.score(1, [2, 0]) == pytest.approx(a.score(1, [0, 2]), rel=1e-12)


def test_categorical_bic_and_bdeu_agree_on_which_parent_explains_a_copy():
    """Column 1 copies column 0 in most rows: both scores improve most by that parent."""
    x = _codes(n=2000)
    for score in (BDeuScore(x), CategoricalBICScore(x)):
        gains = {p: score.score(1, [p]) - score.score(1, []) for p in (0, 2, 3)}
        assert min(gains, key=gains.get) == 0 and gains[0] < 0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_categorical_bic_penalty_discount_must_be_finite_and_non_negative(value):
    with pytest.raises(ValueError, match="penalty_discount"):
        CategoricalBICScore(_codes(), penalty_discount=value)


def test_make_score_builds_the_categorical_bic_with_its_weight():
    x = _codes()
    built = make_score("discrete_bic", x, penalty_discount=2.0)
    assert isinstance(built, CategoricalBICScore)
    assert built.score(1, [0]) == CategoricalBICScore(x, penalty_discount=2.0).score(1, [0])


def _counts_by_configurations(
    codes: np.ndarray, cards: np.ndarray, node: int, parents: tuple[int, ...]
) -> np.ndarray:
    """The node's table over the occurring configurations that :func:`configurations` numbers."""
    config, m = categorical.configurations(codes, cards, parents)
    r = int(cards[node])
    return np.bincount(config * r + codes[:, node], minlength=m * r).reshape(m, r)


def _configuration_counts(x: np.ndarray, node: int, parents: tuple[int, ...]) -> np.ndarray:
    codes, cards = categorical.encode(x)
    return _counts_by_configurations(codes, cards, node, parents)


def test_batched_table_counts_equal_one_numbering_per_set(monkeypatch):
    """A batch counts each set's table as :func:`configurations` numbers it, small or large, in one
    bincount or several."""
    x = _codes(n=300, cards=(2, 3, 4, 3, 5, 4))
    codes, cards = categorical.encode(x)
    sets = [(), (1,), (2, 1), (3, 4, 5), (1, 2, 3, 4, 5), (4,)]
    expected = [_configuration_counts(x, 0, s) for s in sets]
    for batch in (1 << 22, 1):  # every set in one bincount, then each in its own
        monkeypatch.setattr(categorical, "BATCH_CELLS", batch)
        for dense in (1 << 16, 0):  # counted over every configuration, then numbered
            monkeypatch.setattr(categorical, "DENSE_MIN", dense)
            monkeypatch.setattr(categorical, "DENSE_ROWS", 4 if dense else 0)
            got = categorical.table_counts(codes, cards, 0, sets)
            for a, b in zip(got, expected, strict=True):
                np.testing.assert_array_equal(a, b)


def _set_orders(d: int, node: int, rng: np.random.Generator) -> dict[str, list[tuple[int, ...]]]:
    """Orders of one node's parent sets, most of them with neighbors that share their first
    parents, as a search that lists sets in lexicographic order does."""
    ascending = _parent_sets(d, node, 4)
    depth_first = sorted(ascending)
    longest = depth_first[-1]
    return {
        "by size": ascending,
        "depth first": depth_first,
        "reversed": depth_first[::-1],
        "digits shuffled": [tuple(rng.permutation(s).tolist()) for s in depth_first],
        "random": [ascending[i] for i in rng.permutation(len(ascending))],
        "repeats and drops": [longest, longest, depth_first[1], longest[:1], (), longest, (), ()],
        "empty and one": [()] + [longest],
    }


@pytest.mark.parametrize("dense_min", [1 << 16, 0], ids=["dense", "mixed"])
@pytest.mark.parametrize("batch", [1 << 22, 1000, 250, 7])
def test_table_counts_reuse_the_code_of_shared_first_parents(monkeypatch, batch, dense_min):
    """Each table equals the one :func:`configurations` numbers, in value, shape, and dtype, for
    sets that repeat, follow a longer set, or start with a set's first parents, across layouts and
    batch sizes, and with sets too large for the dense count among those that are not."""
    monkeypatch.setattr(categorical, "BATCH_CELLS", batch)
    monkeypatch.setattr(categorical, "DENSE_MIN", dense_min)
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.integers(0, k, 100) for k in (2, 3, 50, 4, 5, 2, 3)])
    codes, cards = categorical.encode(x.astype(np.float64))
    for node in (0, 2, 5):
        for order in _set_orders(x.shape[1], node, rng).values():
            expected = [_counts_by_configurations(codes, cards, node, s) for s in order]
            for layout in (np.ascontiguousarray, np.asfortranarray):
                got = categorical.table_counts(layout(codes), cards, node, order)
                assert len(got) == len(expected)
                for table, reference in zip(got, expected, strict=True):
                    assert table.dtype == reference.dtype
                    np.testing.assert_array_equal(table, reference)


def test_table_counts_hold_the_codes_of_few_prefixes_of_a_long_set(monkeypatch):
    """A set of many parents that do not grow its table, here constant columns, is counted with the
    codes of at most ``BATCH_CELLS`` entries held for shared prefixes, and its table is the one
    :func:`configurations` numbers."""
    n = 2000
    monkeypatch.setattr(categorical, "BATCH_CELLS", 3 * n)
    rng = np.random.default_rng(4)
    x = np.column_stack([rng.integers(0, 3, n), rng.integers(0, 2, n), np.zeros((n, 300))])
    codes, cards = categorical.encode(x)
    sets = [tuple(range(1, 302)), tuple(range(1, 301)), (1, *range(2, 302, 2)), (1, 2)]
    expected = [_counts_by_configurations(codes, cards, 0, s) for s in sets]
    tracemalloc.start()
    try:
        got = categorical.table_counts(codes, cards, 0, sets)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    for table, reference in zip(got, expected, strict=True):
        assert table.dtype == reference.dtype
        np.testing.assert_array_equal(table, reference)
    assert peak < 40 * n * 8  # a prefix per parent would hold 300 arrays of n integers


def _mixed_tables(seed: int = 0, n: int = 400) -> np.ndarray:
    """Columns of 1, 2, 5, 3, 12, and 2 categories, one with negative non-integer values, so the
    tables of a node's parent sets have many shapes."""
    rng = np.random.default_rng(seed)
    x = np.column_stack([rng.integers(0, k, n) for k in (1, 2, 5, 3, 12, 2)]).astype(np.float64)
    x[:, 2] = np.where(rng.random(n) < 0.5, x[:, 1], x[:, 2])
    x[:, 3] = x[:, 3] * 1.7 - 2.5
    return x


def _parent_sets(d: int, node: int, size: int = 3) -> list[tuple[int, ...]]:
    others = [v for v in range(d) if v != node]
    return [c for k in range(size + 1) for c in combinations(others, k)]


#: The scores whose batched and one-table paths must give the same bits, with the settings the
#: BDeu prior and the multinomial penalty read.
SCORES = [
    pytest.param(lambda x: BDeuScore(x), id="bdeu"),
    pytest.param(
        lambda x: BDeuScore(x, equivalent_sample_size=0.5, structure_prior=0.3), id="bdeu-0.5"
    ),
    pytest.param(lambda x: BDeuScore(x, equivalent_sample_size=3.7), id="bdeu-3.7"),
    pytest.param(
        lambda x: BDeuScore(x, equivalent_sample_size=10.0, structure_prior=0.3), id="bdeu-10"
    ),
    pytest.param(lambda x: CategoricalBICScore(x), id="discrete_bic"),
    pytest.param(lambda x: CategoricalBICScore(x, penalty_discount=2.5), id="discrete_bic-2.5"),
]


def _exact(values) -> list[str]:
    """Each value's exact bits, as the hex of the float."""
    return [float(v).hex() for v in values]


@pytest.mark.parametrize("make", SCORES)
def test_a_score_is_the_same_alone_or_in_a_batch(make):
    """Every way to ask for a set's score gives the same bits: alone, in one batch, in batches of
    several sizes with duplicates, sets in any order, and some already memoized."""
    x = _mixed_tables()
    for node in range(x.shape[1]):
        sets = _parent_sets(x.shape[1], node)
        alone = _exact(make(x).score(node, s) for s in sets)  # each set on a fresh score
        assert _exact(make(x).score_many(node, sets)) == alone
        rng = np.random.default_rng(node)
        for size in (1, 2, 3, 7):
            scorer = make(x)
            for s in sets[::4]:
                scorer.score(node, s)
            got = {}
            order = rng.permutation(len(sets))
            for start in range(0, len(order), size):
                batch = [sets[i] for i in order[start : start + size]]
                batch = [tuple(reversed(s)) for s in (*batch, batch[0])]
                for s, value in zip(batch, scorer.score_many(node, batch), strict=True):
                    got[tuple(sorted(s))] = value
            assert _exact(got[s] for s in sets) == alone
        # One set not yet memoized: a batch of one table.
        scorer = make(x)
        scorer.score_many(node, sets[:-1])
        assert _exact(scorer.score_many(node, sets)) == alone


@pytest.mark.parametrize("dense_min", [1 << 16, 0], ids=["dense", "numbered"])
@pytest.mark.parametrize("make", SCORES)
def test_a_batch_scores_the_tables_of_the_configurations_numbering(monkeypatch, make, dense_min):
    """A batch scores each set as :meth:`_local` scores the table :func:`configurations` numbers."""
    monkeypatch.setattr(categorical, "DENSE_MIN", dense_min)
    monkeypatch.setattr(categorical, "DENSE_ROWS", 4 if dense_min else 0)
    x = _mixed_tables()
    codes, cards = categorical.encode(x)
    scorer = make(x)
    for node in range(x.shape[1]):
        sets = _parent_sets(x.shape[1], node)
        expected = [
            scorer._local(node, s, _counts_by_configurations(codes, cards, node, s)) for s in sets
        ]
        assert _exact(make(x).score_many(node, sets)) == _exact(expected)


@pytest.mark.parametrize("make", SCORES)
def test_on_rows_scores_the_rows_with_the_categories_of_all_rows(make):
    """``on_rows`` by a mask or by indices scores the selected rows over the categories of all
    rows, and selecting every row gives the score's own values."""
    x = _mixed_tables()
    codes, cards = categorical.encode(x)
    rng = np.random.default_rng(1)
    mask = rng.random(len(x)) < 0.7
    mask[:2] = True
    whole = make(x)
    for rows in (mask, np.flatnonzero(mask)[::-1].copy(), np.ones(len(x), dtype=bool)):
        selected = whole.on_rows(rows)
        for node in range(x.shape[1]):
            sets = _parent_sets(x.shape[1], node)
            expected = [
                whole._local(node, s, _counts_by_configurations(codes[rows], cards, node, s))
                for s in sets
            ]
            assert _exact(selected.score_many(node, sets)) == _exact(expected)
            assert _exact(selected.score(node, s) for s in sets) == _exact(expected)
    everything = whole.on_rows(np.ones(len(x), dtype=bool))
    for node in range(x.shape[1]):
        sets = _parent_sets(x.shape[1], node)
        assert _exact(everything.score_many(node, sets)) == _exact(whole.score_many(node, sets))


def test_a_cell_mean_matches_the_hypergeometric_sum_on_both_sides_of_the_expansion():
    """Sparse cells are summed over the support and dense ones expanded; both are within 2e-3."""
    rng = np.random.default_rng(0)
    n = rng.integers(20, 5000, 400)
    r = np.minimum(rng.integers(2, 1 + n // 2), n - 2)
    c = np.maximum(r, np.minimum(rng.integers(2, n - 1), n - 2))
    mu = r * c / n
    assert (mu < ci._SERIES_MEAN).any() and (mu >= ci._SERIES_MEAN).any()
    for ni, ri, ci_, got in zip(n, r, c, ci._cell_g_means(n, r, c), strict=True):
        t = np.arange(max(1, ri + ci_ - ni), ri + 1)
        exact = np.sum(hypergeom.pmf(t, ni, ri, ci_) * 2 * t * np.log(t * ni / (ri * ci_)))
        assert got == pytest.approx(exact, abs=2e-3)


def test_the_cell_mean_table_returns_the_evaluated_means(monkeypatch):
    """Each mean the table returns is the one evaluated for its triple, bit for bit: for every
    cell of the smallest ``n``, and for random triples on both sides of the table's limit,
    looked up in any order and in batches that fill the table a few rows at a time."""
    monkeypatch.setattr(ci, "_MEANS_TABLE", np.empty_like(ci._MEANS_TABLE))
    monkeypatch.setattr(ci, "_MEANS_FILLED", np.zeros_like(ci._MEANS_FILLED))
    limit = ci._MEANS_N
    every = np.array(
        [(n, r, c) for n in range(2, 13) for c in range(1, n) for r in range(1, c + 1)]
    )
    rng = np.random.default_rng(0)
    n = rng.integers(2, 2 * limit, 3000)
    r = rng.integers(1, n // 2 + 1)
    c = rng.integers(r, n)
    assert (n <= limit).any() and (n > limit).any()
    triples = np.concatenate([every, np.column_stack([n, r, c])])
    triples = triples[rng.permutation(len(triples))]
    for batch in np.array_split(triples, 40):
        got = ci._g_means(*batch.T)
        want = ci._cell_g_means(*batch.T)
        assert np.array_equal(got.view(np.int64), want.view(np.int64))
    assert ci._MEANS_FILLED.any() and not ci._MEANS_FILLED.all()


def test_the_cell_mean_table_is_bounded():
    """The table holds one float per ``(n, r, c)`` with ``r <= c <= n <= _MEANS_N``, a few
    megabytes, and no lookup writes outside it."""
    limit = ci._MEANS_N
    last = ci._table_slot(np.array([limit]), np.array([limit]), np.array([limit]))[0]
    assert last == ci._MEANS_TABLE.size - 1
    assert ci._MEANS_TABLE.nbytes < 4 * 2**20


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize("categories", [3, 4, 5])
def test_a_true_independence_is_rejected_at_the_level_it_is_tested_at(g_square, categories):
    """Given two conditioning variables, ``x`` and ``y`` are drawn independently from their own
    random tables, in tables sparse enough that the chi-square law of the statistic is poor
    (``n`` over ``categories ** 4`` cells is 8 to 1.6). The rate at which ``alpha = 0.05``
    rejects stays inside the binomial band of 0.05; a test that over-rejects fails the upper
    bound, and one that never rejects fails the lower."""
    k, n, draws, alpha = categories, 1000, 300, 0.05
    rng = np.random.default_rng(100 * categories + g_square)
    rejected = 0
    for _ in range(draws):
        z = rng.integers(0, k * k, n)
        px = rng.dirichlet(np.ones(k), size=k * k)
        py = rng.dirichlet(np.ones(k), size=k * k)
        u, v = rng.random(n), rng.random(n)
        xs = (u[:, None] > np.cumsum(px[z], axis=1)).sum(axis=1)
        ys = (v[:, None] > np.cumsum(py[z], axis=1)).sum(axis=1)
        data = np.column_stack([xs, ys, z // k, z % k]).astype(np.float64)
        rejected += ChiSquare(data, g_square=g_square)(0, 1, (2, 3)) < alpha
    sd = math.sqrt(alpha * (1 - alpha) / draws)
    assert alpha - 3.5 * sd < rejected / draws < alpha + 3.5 * sd


@pytest.mark.parametrize("g_square", [False, True], ids=["chisq", "gsq"])
@pytest.mark.parametrize(
    ("batch", "table"),
    [(1 << 22, 1 << 20), (64, 1 << 20), (1 << 22, 40)],
    ids=["one count", "a count per block", "tables over the size limit on the scalar path"],
)
def test_a_scan_of_candidates_equals_a_test_per_candidate(monkeypatch, g_square, batch, table):
    """``pvalues_given`` and the queries of ``statistic_pvalues`` that share a conditioning set
    return the statistic and p-value of each candidate's own call, bit for bit, on random discrete
    data with few rows, one-category columns, and any order of candidates and conditioners, and
    so do the same queries asked again of the same test. A candidate's own call evaluates every
    cell mean; the scan reads the small ones from the table and evaluates the blocks of a scan
    together or, with a small batch, one at a time."""
    from andrey.core import ci

    monkeypatch.setattr(ci, "BATCH_CELLS", batch)
    monkeypatch.setattr(ci, "_DENSE_CELLS", table)
    rng = np.random.default_rng(0)
    for _ in range(20):
        n, d = int(rng.integers(3, 500)), 7
        x = np.column_stack([rng.integers(0, k, n) for k in rng.integers(1, 6, d)]).astype(float)
        x[:, 1] = np.where(rng.random(n) < 0.5, x[:, 0], x[:, 1])
        test = ChiSquare(x, g_square=g_square)
        target = int(rng.integers(d))
        rest = rng.permutation([v for v in range(d) if v != target])
        cond = rest[: int(rng.integers(0, 4))].tolist()
        xs = rest[len(cond) :].tolist()
        with monkeypatch.context() as direct:
            direct.setattr(ci, "_MEANS_N", 0)  # no cell mean comes from the table
            alone_test = ChiSquare(x, g_square=g_square)
            alone = [alone_test._statistic_pvalue(target, v, cond) for v in xs]
        p = test.pvalues_given(target, xs, cond)
        assert p.view(np.int64).tolist() == np.array([a[1] for a in alone]).view(np.int64).tolist()
        shuffled = [int(i) for i in rng.permutation(len(xs))]
        queries = (
            np.array([target] * len(xs))[shuffled],
            np.array(xs)[shuffled],
            np.array([cond] * len(xs)).reshape(len(xs), -1)[shuffled],
        )
        want = np.array(alone)[shuffled].T
        for asked in (ChiSquare(x, g_square=g_square), test):
            got = asked.statistic_pvalues(*queries)
            assert np.array_equal(np.array(got).view(np.int64), want.view(np.int64))
