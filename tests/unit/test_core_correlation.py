"""The tiled correlation matrix and lazy correlation.

The whole matrix and a lazily computed row come from the same tile products, so they agree bit for
bit whatever the data's memory layout; a lazy partial-correlation test answers every query path as
the test over the whole matrix does; a local search reads few rows; and the copied-column check
runs on the rows computed.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.constraint.blanket import local_search
from andrey.core import ci
from andrey.core.ci import FisherZ, NormalScores, StudentT
from andrey.core.correlation import TILE, LazyCorrelation, correlation_matrix
from andrey.core.score import CopiedColumnsError


@pytest.fixture(autouse=True)
def force_numpy(monkeypatch):
    monkeypatch.setenv("ANDREY_BACKEND", "numpy")


def _data(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((d, d)) * (rng.random((d, d)) < 0.1)
    return (rng.standard_normal((n, d)) @ (np.eye(d) + mix)) * rng.uniform(0.01, 100.0, d)


@pytest.mark.parametrize(("n", "d"), [(40, 3), (300, 2 * TILE + 5), (30, 3 * TILE)])
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
    X = _data(400, 3 * TILE + 7, seed=2)
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
    n, d = 600, 12 * TILE
    X = rng.standard_normal((n, d))
    for j in range(1, d):
        X[:, j] += 0.8 * X[:, j - 1]
    target = d // 2
    lazy = local_search(X, alpha=0.01, indep_test="fisherz")
    with lazy.pinned():
        found = lazy.local("iamb", target)
    assert lazy.test.correlation_rows() < d // 2
    monkeypatch.setenv("ANDREY_LAZY_CORRELATION", "0")
    whole = local_search(X, alpha=0.01, indep_test="fisherz")
    with whole.pinned():
        assert whole.local("iamb", target) == found
    assert whole.test.correlation_rows() == d


def test_a_copy_raises_when_a_row_that_shows_it_is_computed(monkeypatch):
    X = _data(200, 4 * TILE, seed=4)
    copy, kept = 3 * TILE + 1, 2 * TILE + 3
    X[:, copy] = 2.0 * X[:, kept] - 1.0
    test = ci.make_indep_test("fisherz", X, lazy=True)
    assert 0.0 <= test(0, 1, [2]) <= 1.0  # rows of the first tile show no copy
    with pytest.raises(CopiedColumnsError, match=f"{copy} repeats {kept}"):
        test(0, 1, [kept])
    with pytest.raises(CopiedColumnsError, match=f"{copy} repeats {kept}"):
        test(0, 1, [kept])  # and again: the tile is not kept
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
