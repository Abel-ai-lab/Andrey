"""The pattern-grouped sums of the partial-correlation tests under test-wise deletion."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core import env, missing_groups
from andrey.core.ci import make_indep_test
from andrey.core.ties import tie_ranks

GAUSSIAN = ("fisherz", "student_t")
RTOL = 1e-8


def _cohorts(n: int = 6_000, bridge: float = 0.03, scatter: float = 0.08, seed: int = 0):
    """Three cohorts over 14 columns: 6 core columns, a module of 3 per pair of cohorts, and one
    column per cohort; ``bridge`` of the rows observe every column; ``scatter`` of the measured
    cells are missing at random; lab-like units."""
    rng = np.random.default_rng(seed)
    d = 14
    weights = np.triu(rng.uniform(0.3, 0.9, (d, d)) * (rng.random((d, d)) < 0.3), 1)
    x = rng.standard_normal((n, d))
    for j in range(d):
        x[:, j] += x @ weights[:, j]
    x = x * rng.uniform(0.5, 50.0, d) + rng.uniform(100.0, 1_000.0, d)
    sets = np.zeros((3, d), dtype=bool)
    sets[:, :6] = True
    sets[[0, 1], 6:9] = True
    sets[[1, 2], 9:11] = True
    sets[[0, 2], 11] = True
    sets[0, 12] = sets[1, 13] = True
    seen = sets[rng.integers(0, 3, n)]
    seen[rng.random(n) < bridge] = True
    seen &= rng.random((n, d)) >= scatter
    return np.where(seen, x, np.nan)


@pytest.fixture
def grouped(monkeypatch):
    """Every gapped Gaussian test takes the grouped path when its patterns allow it."""
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_MAX_SCATTER", "1")


def _per_row(monkeypatch) -> None:
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_BYTES", "0")


def test_bridging_rows_form_their_own_group_and_leave_the_cohorts_intact():
    x = _cohorts()
    grouping = missing_groups.pattern_groups(~np.isnan(x), tie_ranks(x))
    assert grouping is not None
    assert len(grouping.observed) == 4
    assert grouping.observed.sum(axis=1).max() == x.shape[1]  # the bridging rows' group
    assert grouping.scattered_share < 0.1  # the scattered cells only, about 8%


def test_the_groups_do_not_depend_on_the_row_or_the_column_order():
    x = _cohorts(seed=3)
    rng = np.random.default_rng(1)
    rows, columns = rng.permutation(x.shape[0]), rng.permutation(x.shape[1])

    def summary(data, row_back, column_back):
        """The groups as sets of original rows with their observed original columns."""
        grouping = missing_groups.pattern_groups(~np.isnan(data), tie_ranks(data))
        assert grouping is not None
        held = grouping.observed[:, column_back]
        groups = {
            (frozenset(row_back[grouping.group == g].tolist()), tuple(held[g].tolist()))
            for g in range(len(held))
        }
        return groups, grouping.scattered_share, grouping.table_bytes()

    n, d = x.shape
    base = summary(x, np.arange(n), np.arange(d))
    assert summary(x[rows], rows, np.arange(d)) == base
    assert summary(x[:, columns], np.arange(n), np.argsort(columns)) == base


def test_patterns_with_no_structure_take_the_per_row_path():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4_000, 30))
    x[rng.random(x.shape) < 0.3] = np.nan
    assert missing_groups.grouping_for(~np.isnan(x), tie_ranks(x)) is None


def test_a_zero_byte_budget_keeps_the_per_row_path(monkeypatch):
    x = _cohorts()
    assert missing_groups.grouping_for(~np.isnan(x), tie_ranks(x)) is not None
    _per_row(monkeypatch)
    assert missing_groups.grouping_for(~np.isnan(x), tie_ranks(x)) is None


@pytest.mark.parametrize("value", ["-0.1", "1.5", "a lot"])
def test_a_scatter_cutoff_outside_zero_to_one_is_refused(monkeypatch, value):
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_MAX_SCATTER", value)
    with pytest.raises(ValueError, match="ANDREY_TESTWISE_GROUP_MAX_SCATTER"):
        env.TESTWISE_GROUP_MAX_SCATTER.read()


def _queries(d: int, seed: int = 4):
    rng = np.random.default_rng(seed)
    out = {}
    for size in (1, 2, 3, 4):
        picks = {tuple(rng.permutation(d)[: size + 2].tolist()) for _ in range(150)}
        out[size] = sorted(picks)
    return out


@pytest.mark.parametrize("name", GAUSSIAN)
def test_grouped_sums_answer_as_the_per_row_sums_up_to_rounding(monkeypatch, grouped, name):
    x = _cohorts()
    test = make_indep_test(name, x)
    paths = {}
    for path in ("grouped", "per-row"):
        if path == "per-row":
            _per_row(monkeypatch)
        test = make_indep_test(name, x)
        found = {}
        for queries in _queries(x.shape[1]).values():
            q = np.array(queries)
            found[len(q[0])] = test.statistic_pvalues(q[:, 0], q[:, 1], q[:, 2:])
        assert (test._grouped_sums() is not None) == (path == "grouped")
        paths[path] = found
    for size, (r, p) in paths["grouped"].items():
        r_ref, p_ref = paths["per-row"][size]
        assert np.array_equal(np.isnan(p), np.isnan(p_ref))  # the same queries not tested
        assert np.allclose(r, r_ref, rtol=RTOL, atol=1e-12, equal_nan=True)
        assert np.allclose(p, p_ref, rtol=RTOL, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("name", GAUSSIAN)
def test_a_grouped_query_gives_the_same_bits_alone_and_in_any_batch(grouped, name):
    x = _cohorts(n=12_000, seed=5)
    rng = np.random.default_rng(6)
    for queries in _queries(x.shape[1], seed=7).values():
        alone = [make_indep_test(name, x).statistic_pvalue(q[0], q[1], q[2:]) for q in queries]
        order = rng.permutation(len(queries))
        for batch in (order[:1], order[1:7], order[7:], order[::-1]):
            q = np.array([queries[i] for i in batch])
            test = make_indep_test(name, x)
            assert test._grouped_sums() is not None
            r, p = test.statistic_pvalues(q[:, 1], q[:, 0], q[:, 2:])
            assert list(zip(r.tolist(), p.tolist())) == [alone[i] for i in batch]


@pytest.mark.parametrize("method", ["pc", "fci", "iamb"])
def test_a_fit_on_merged_cohorts_finds_the_same_graph_on_either_path(monkeypatch, method):
    x = _cohorts(n=4_000, seed=8)[:, :10]
    fit = getattr(andrey, method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = fit(x, indep_test="fisherz", missing="testwise")
        _per_row(monkeypatch)
        reference = fit(x, indep_test="fisherz", missing="testwise")
    assert np.array_equal(found.structure.to_numpy(), reference.structure.to_numpy())
    for key in ("not_tested", "undecided_pairs", "rows_min", "rows_median"):
        assert found.metadata.get(key) == reference.metadata.get(key)
