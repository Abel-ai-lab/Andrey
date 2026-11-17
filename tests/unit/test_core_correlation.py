"""The blocked correlation matrix and lazy correlation.

The whole matrix and a lazily computed row come from the same block products, so they agree bit for
bit whatever the data's memory layout and the BLAS thread count; a lazy partial-correlation test
answers every query path as the test over the whole matrix does; a local search reads few rows; the
copied-column check runs on the rows computed; and a backend that selects a device takes the whole
matrix.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

import andrey
from andrey.constraint.blanket import local_search
from andrey.core import backend, ci, stats
from andrey.core.ci import FisherZ, NormalScores, StudentT
from andrey.core.correlation import BLOCK, LazyCorrelation, correlation_matrix
from andrey.core.score import CopiedColumnsError


@pytest.fixture(autouse=True)
def force_numpy(monkeypatch):
    monkeypatch.setenv("ANDREY_BACKEND", "numpy")


def _data(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((d, d)) * (rng.random((d, d)) < 0.1)
    return (rng.standard_normal((n, d)) @ (np.eye(d) + mix)) * rng.uniform(0.01, 100.0, d)


@pytest.mark.parametrize(("n", "d"), [(40, 3), (300, 2 * BLOCK + 5), (30, 3 * BLOCK)])
def test_rows_equal_the_whole_matrix_bit_for_bit_in_any_layout(n, d):
    X = _data(n, d)
    full = correlation_matrix(X)
    assert np.array_equal(full, full.T)
    assert np.array_equal(np.diag(full), np.ones(d))
    np.testing.assert_allclose(full, np.corrcoef(X, rowvar=False), rtol=0, atol=1e-13)
    rng = np.random.default_rng(1)
    for layout in (X, np.asfortranarray(X), np.repeat(X, 2, axis=1)[:, ::2]):
        assert np.array_equal(correlation_matrix(layout), full)
        lazy = LazyCorrelation(layout)
        for v in rng.permutation(d)[:5]:
            assert np.array_equal(lazy.rows([v])[0], full[v])
        a, b = rng.integers(0, d, 400), rng.integers(0, d, 400)
        a, b = a[a != b], b[a != b]
        lazy.cover(a, b)
        assert np.array_equal(lazy.pairs(a, b), full[a, b])
        assert np.array_equal(lazy.matrix(), full)
        lazy.load(np.arange(d))
        assert np.array_equal(lazy.matrix(), full)


def test_a_constant_column_has_a_nan_row():
    X = _data(50, 4)
    X[:, 2] = 1.0
    with np.errstate(invalid="ignore"):
        full = correlation_matrix(X)
        lazy = LazyCorrelation(X).rows([2])[0]
    assert np.isnan(full[2]).all() and np.isnan(full[:, 2]).all()
    assert np.array_equal(lazy, full[2], equal_nan=True)


@pytest.mark.parametrize("test_class", [FisherZ, StudentT, NormalScores])
def test_a_lazy_test_answers_every_path_as_the_whole_matrix(test_class):
    X = _data(850, 3 * BLOCK + 7, seed=2)
    d = X.shape[1]
    whole, lazy = test_class(X), test_class(X, lazy=True)
    target = d // 2
    for cond in ([], [3], [3, d - 1], [3, 40, d - 1], list(range(8, 14))):
        ys = [y for y in range(d) if y != target and y not in cond]
        for path in ("statistic_pvalues_given",):
            r0, p0 = getattr(whole, path)(target, ys, cond)
            r1, p1 = getattr(lazy, path)(target, ys, cond)
            assert np.array_equal(r0, r1) and np.array_equal(p0, p1)
        xs = np.full(len(ys), target)
        sets = np.tile(np.asarray(cond, dtype=np.intp), (len(ys), 1))
        r0, p0 = whole.statistic_pvalues(ys, xs, sets)
        r1, p1 = lazy.statistic_pvalues(ys, xs, sets)
        assert np.array_equal(r0, r1) and np.array_equal(p0, p1)
        for y in ys[:: max(1, len(ys) // 12)]:
            assert whole.statistic_pvalue(y, target, cond) == lazy.statistic_pvalue(y, target, cond)
        assert whole.batched_call(0, 1, [[5], [6], [7]]) == lazy.batched_call(0, 1, [[5], [6], [7]])
    assert lazy.correlation_rows() < d
    assert np.array_equal(whole.marginal_pvalues(), lazy.marginal_pvalues())
    assert lazy.correlation_rows() == d
    assert np.array_equal(
        whole.statistic_pvalues_given_rest(0, [1, 2])[1],
        lazy.statistic_pvalues_given_rest(0, [1, 2])[1],
    )


def test_a_local_search_reads_few_rows_and_answers_as_the_whole_matrix(monkeypatch):
    rng = np.random.default_rng(3)
    n, d = 2200, 8 * BLOCK
    X = rng.standard_normal((n, d))
    for j in range(1, d):
        X[:, j] += 0.8 * X[:, j - 1]
    target = d // 2
    lazy = local_search(X, alpha=1e-5, indep_test="fisherz")
    with lazy.pinned():
        found = lazy.local("iamb", target)
    assert lazy.test.correlation_rows() < d // 2
    monkeypatch.setenv("ANDREY_LAZY_CORRELATION", "0")
    whole = local_search(X, alpha=1e-5, indep_test="fisherz")
    with whole.pinned():
        assert whole.local("iamb", target) == found
    assert whole.test.correlation_rows() == d


def test_a_copy_raises_when_a_row_that_shows_it_is_computed(monkeypatch):
    X = _data(200, 4 * BLOCK, seed=4)
    copy, kept = 3 * BLOCK + 1, 2 * BLOCK + 3
    X[:, copy] = 2.0 * X[:, kept] - 1.0
    test = ci.make_indep_test("fisherz", X, lazy=True)
    assert 0.0 <= test(0, 1, [2]) <= 1.0  # rows of the first block show no copy
    with pytest.raises(CopiedColumnsError, match=f"{copy} repeats {kept}"):
        test(0, 1, [kept])
    with pytest.raises(CopiedColumnsError, match=f"{copy} repeats {kept}"):
        test(0, 1, [kept])  # and again: the block is not kept
    monkeypatch.setenv("ANDREY_LAZY_CORRELATION", "0")
    with pytest.raises(CopiedColumnsError, match=f"{copy} repeats {kept}"):
        ci.make_indep_test("fisherz", X, lazy=True)


def test_a_local_method_names_a_copy_it_reads():
    pd = pytest.importorskip("pandas")
    X = _data(300, 6, seed=5)
    X[:, 4] = 3.0 * X[:, 1]
    frame = pd.DataFrame(X, columns=[f"x{j}" for j in range(6)])
    with pytest.raises(CopiedColumnsError, match="'x4' repeats 'x1'"):
        andrey.iamb(frame, target="x1")


def _need_blas_threads(count: int = 4) -> None:
    with threadpool_limits(limits=count, user_api="blas"):
        if not any(p["num_threads"] >= 2 for p in threadpool_info() if p["user_api"] == "blas"):
            pytest.skip("threadpoolctl cannot give the BLAS two or more threads")


def test_a_lazy_read_computes_the_block_of_the_row():
    X = _data(60, 2 * BLOCK + 77, seed=7)
    d = X.shape[1]
    lazy = LazyCorrelation(X)
    assert lazy.computed == 0
    lazy.rows([BLOCK + 5])
    assert lazy.computed == BLOCK
    lazy.rows([BLOCK, 2 * BLOCK - 1])
    assert lazy.computed == BLOCK
    lazy.rows([2 * BLOCK + 76])
    assert lazy.computed == BLOCK + 77
    lazy.rows([0])
    assert lazy.computed == d


@pytest.mark.parametrize("n", [50, 300, 2000])
def test_the_entries_do_not_depend_on_the_blas_threads(n):
    _need_blas_threads()
    # Three blocks, the last one ragged. The BLAS itself, split among threads, differs from one
    # thread in the last bits at 50 and 300 rows.
    X = _data(n, 2 * BLOCK + 77, seed=6)
    rows = [0, BLOCK + 3, 2 * BLOCK + 76]
    with threadpool_limits(limits=1, user_api="blas"):
        one = correlation_matrix(X)
        lazy_one = LazyCorrelation(X)
    with threadpool_limits(limits=4, user_api="blas"):
        several = correlation_matrix(X)
        lazy_several = LazyCorrelation(X)
        pinned = LazyCorrelation(X)
    assert np.array_equal(one, several)
    # Rows computed on one thread equal the matrix computed on several, and the reverse; a first
    # read inside a search's one-thread block takes the threads of the object's making.
    with threadpool_limits(limits=4, user_api="blas"):
        assert np.array_equal(lazy_one.rows(rows), several[rows])
    with threadpool_limits(limits=1, user_api="blas"):
        assert np.array_equal(lazy_several.rows(rows), one[rows])
    with backend.one_blas_thread():
        assert np.array_equal(pinned.rows(rows), several[rows])
    for lazy in (lazy_one, lazy_several, pinned):
        lazy.load(np.arange(X.shape[1]))
        assert np.array_equal(lazy.matrix(), one)


def test_a_lazy_test_follows_a_backend_that_selects_a_device(monkeypatch):
    X = _data(100, 40, seed=8)
    d = X.shape[1]
    picks, asked = [], []
    monkeypatch.setattr(backend, "resolve", lambda *a, **kw: picks.append(kw["op"]) or "cuda")

    def whole(data, rowvar=True):
        asked.append((data.shape, rowvar))
        return correlation_matrix(data)

    monkeypatch.setattr(stats, "corrcoef", whole)
    for test_class in (FisherZ, StudentT, NormalScores):
        test = test_class(X, lazy=True)
        assert test.correlation_rows() == d
        assert test(0, 1, [2]) == test_class(X)(0, 1, [2])
    assert asked == [(X.shape, False)] * 3 * 2
    assert picks == [None] * 3  # asked without counting a corrcoef dispatch; whole() counts it
    assert ci.make_indep_test("fisherz", X, lazy=True).correlation_rows() == d
    copied = X.copy()
    copied[:, 7] = 2.0 * copied[:, 3] - 1.0
    with pytest.raises(CopiedColumnsError, match="7 repeats 3"):
        ci.make_indep_test("fisherz", copied, lazy=True)
    monkeypatch.setattr(backend, "resolve", lambda *a, **kw: "numpy")
    assert ci.make_indep_test("fisherz", X, lazy=True).correlation_rows() == 0


def test_rows_computed_in_any_order_keep_their_values_as_the_store_grows():
    d = 3 * BLOCK + 40
    x = np.random.default_rng(5).standard_normal((60, d))
    whole = correlation_matrix(x)
    lazy = LazyCorrelation(x)
    for block in (2, 0, 3, 1):
        column = block * BLOCK + 3
        assert np.array_equal(lazy.rows([column]), whole[[column]])
        assert lazy.entry(column, 7) == whole[column, 7]
    assert lazy.computed == d
    assert np.array_equal(lazy.rows(np.arange(d)), whole)


@pytest.mark.parametrize("protocol", [4, 5])
def test_a_pickled_copy_grows_and_keeps_its_rows(protocol):
    d = 2 * BLOCK + 10
    x = np.random.default_rng(6).standard_normal((60, d))
    whole = correlation_matrix(x)
    copy = pickle.loads(pickle.dumps(LazyCorrelation(x), protocol=protocol))
    for block in (1, 0, 2):
        column = block * BLOCK + 1
        assert np.array_equal(copy.rows([column]), whole[[column]])
    assert copy.computed == d
    assert np.array_equal(copy.rows(np.arange(d)), whole)
