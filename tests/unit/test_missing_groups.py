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
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_MAX_SHARED", "1")


def _spy(monkeypatch, cls, name: str) -> list[int]:
    """Count the blocks each call of ``cls.name`` computes."""
    calls: list[int] = []
    original = getattr(cls, name)

    def counted(self, columns, *args):
        calls.append(len(columns))
        return original(self, columns, *args)

    monkeypatch.setattr(cls, name, counted)
    return calls


def _grouping_for(x: np.ndarray):
    """:func:`~andrey.core.missing_groups.grouping_for` on ``x``'s observed mask."""
    seen = ~np.isnan(x)
    return missing_groups.grouping_for(
        seen, lambda: tie_ranks(x), lambda: seen.T.astype(np.int64) @ seen
    )


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
    assert _grouping_for(x) is None


def test_one_group_whose_rows_mostly_miss_two_columns_takes_the_per_row_path(monkeypatch):
    """Cells missing completely at random at 10% over 50 columns form one group in which nearly
    every row misses two or more columns, so no block can take the grouped sums."""
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_MAX_SHARED", "1")  # past the pair gate
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4_000, 50))
    x[rng.random(x.shape) < 0.1] = np.nan
    grouping = missing_groups.pattern_groups(~np.isnan(x), tie_ranks(x))
    assert grouping is not None and len(grouping.observed) == 1
    assert not grouping.some_block_cheaper(missing_groups.MIN_GAIN)
    assert _grouping_for(x) is None


def test_small_groups_beside_one_scattered_group_take_the_per_row_path():
    """Cells missing at random over 50 columns, plus a few small cohorts: the small groups can take
    the grouped sums, but they hold too few rows to repay the tables."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((6_000, 50))
    x[rng.random(x.shape) < 0.1] = np.nan
    x[:60, :25] = np.nan  # 60 rows of a cohort that measures half the columns
    grouping = missing_groups.pattern_groups(~np.isnan(x), tie_ranks(x))
    assert grouping is not None and len(grouping.observed) > 1
    assert not grouping.some_block_cheaper(missing_groups.MIN_GAIN)


def test_outages_that_leave_every_pair_half_the_rows_take_the_per_row_path(monkeypatch):
    """Each column of a series is out for three stretches of a twelfth of the rows, and 5% of the
    cells are missing besides: every pair of columns is observed together in about half the rows,
    so the patterns form many overlapping groups that cost more than the blocks save."""
    rng = np.random.default_rng(6)
    n, d = 6_000, 12
    seen = rng.random((n, d)) >= 0.05
    for j in range(d):
        for start in rng.integers(0, n, 3).tolist():
            seen[start : start + n // 12, j] = False
    x = np.where(seen, rng.standard_normal((n, d)), np.nan)
    assert _grouping_for(x) is None
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_MAX_SHARED", "1")
    assert _grouping_for(x) is not None
    assert _grouping_for(_cohorts()) is not None  # a quarter of the cohorts' pairs share few rows


def test_a_zero_byte_budget_keeps_the_per_row_path(monkeypatch):
    x = _cohorts()
    assert _grouping_for(x) is not None
    _per_row(monkeypatch)
    assert _grouping_for(x) is None


@pytest.mark.parametrize("value", ["-0.1", "1.5", "a lot"])
@pytest.mark.parametrize("name", ["MAX_SCATTER", "MAX_SHARED"])
def test_a_share_cutoff_outside_zero_to_one_is_refused(monkeypatch, name, value):
    monkeypatch.setenv(f"ANDREY_TESTWISE_GROUP_{name}", value)
    with pytest.raises(ValueError, match=f"ANDREY_TESTWISE_GROUP_{name}"):
        getattr(env, f"TESTWISE_GROUP_{name}").read()


def _queries(d: int, seed: int = 4):
    rng = np.random.default_rng(seed)
    out = {}
    for size in (1, 2, 3, 4):
        picks = {tuple(rng.permutation(d)[: size + 2].tolist()) for _ in range(150)}
        out[size] = sorted(picks)
    return out


# Cohorts with no row observing every column: some pairs are never observed together, so some
# queries are not tested.
APART = {"n": 1_500, "bridge": 0.0, "seed": 3}


@pytest.mark.parametrize("cohorts", [{}, APART])
@pytest.mark.parametrize("name", GAUSSIAN)
def test_grouped_sums_answer_as_the_per_row_sums_up_to_rounding(
    monkeypatch, grouped, name, cohorts
):
    x = _cohorts(**cohorts)
    summed = _spy(monkeypatch, missing_groups.GroupedSums, "blocks")
    test = make_indep_test(name, x)
    paths = {}
    for path in ("grouped", "per-row"):
        if path == "per-row":
            _per_row(monkeypatch)
        test = make_indep_test(name, x)
        found = {}
        for queries in _queries(x.shape[1]).values():
            q = np.sort(np.array(queries), axis=1)  # canonical: x < y, the set sorted
            lo, hi, conds = q[:, 0], q[:, 1], q[:, 2:]
            found[len(q[0])] = (
                *test.statistic_pvalues(lo, hi, conds),
                test._fast(lo, hi, conds)[2],
            )
        assert (test._grouped_sums() is not None) == (path == "grouped")
        if path == "grouped":
            assert sum(summed) > 0  # some blocks took the grouped sums
        paths[path] = found
    for size, (r, p, rows) in paths["grouped"].items():
        r_ref, p_ref, rows_ref = paths["per-row"][size]
        assert np.array_equal(np.isnan(p), np.isnan(p_ref))  # the same queries not tested
        tested = ~np.isnan(p)
        assert np.array_equal(rows[tested], rows_ref[tested])
        assert np.allclose(r, r_ref, rtol=RTOL, atol=1e-12, equal_nan=True)
        assert np.allclose(p, p_ref, rtol=RTOL, atol=1e-12, equal_nan=True)
    if cohorts:
        assert any(np.isnan(p).any() for _, p, _ in paths["grouped"].values())


@pytest.mark.parametrize("name", GAUSSIAN)
def test_a_grouped_query_gives_the_same_bits_alone_and_in_any_batch(monkeypatch, grouped, name):
    x = _cohorts(n=12_000, seed=5)
    summed = _spy(monkeypatch, missing_groups.GroupedSums, "blocks")
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
    assert sum(summed) > 0


@pytest.mark.parametrize("apart", [False, True])
@pytest.mark.parametrize("method", ["pc", "fci", "iamb"])
def test_a_fit_on_merged_cohorts_finds_the_same_graph_on_either_path(
    monkeypatch, grouped, method, apart
):
    x = _cohorts(**APART) if apart else _cohorts(n=4_000, seed=8)[:, :10]
    fit = getattr(andrey, method)
    summed = _spy(monkeypatch, missing_groups.GroupedSums, "blocks")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = fit(x, indep_test="fisherz", missing="testwise")
        assert sum(summed) > 0  # some blocks took the grouped sums
        _per_row(monkeypatch)
        reference = fit(x, indep_test="fisherz", missing="testwise")
    assert np.array_equal(found.structure.to_numpy(), reference.structure.to_numpy())
    for key in ("not_tested", "undecided_pairs", "rows_tested", "never_observed_together"):
        assert found.metadata.get(key) == reference.metadata.get(key)
    assert found.metadata["rows_tested"]["min"] > 0
    if apart:
        assert found.metadata["not_tested"] > 0
        assert found.metadata["never_observed_together"]


@pytest.mark.parametrize("method", ["pc", "fci", "iamb"])
def test_the_queries_not_tested_do_not_depend_on_the_column_order(method):
    x = _cohorts(**APART)
    fit = getattr(andrey, method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        found = fit(x, indep_test="fisherz", missing="testwise")
        assert found.metadata["not_tested"] > 0
        for seed in range(3):
            order = np.random.default_rng(seed).permutation(x.shape[1])
            moved = fit(x[:, order], indep_test="fisherz", missing="testwise")
            back = np.argsort(order)
            graph = np.asarray(moved.structure.to_numpy())[np.ix_(back, back)]
            assert np.array_equal(graph, found.structure.to_numpy())
            assert moved.metadata["not_tested"] == found.metadata["not_tested"]
            assert moved.metadata["rows_tested"] == found.metadata["rows_tested"]
            undecided = {
                frozenset(int(order[i]) for i in pair) for pair in moved.metadata["undecided_pairs"]
            }
            assert undecided == {frozenset(pair) for pair in found.metadata["undecided_pairs"]}


def test_a_block_with_few_of_its_groups_rows_takes_the_per_row_sums(monkeypatch, grouped):
    """A column seen in 40 of 20,000 rows, every other cell observed: one group holds every row,
    and the grouped sums of the column's blocks would subtract nearly all of the group's sums.
    They give the per-row bits."""
    rng = np.random.default_rng(9)
    n, d = 20_000, 10
    z = rng.standard_normal((n, d))
    z[:, 1:] += 0.4 * z[:, :-1]
    x = z * rng.uniform(0.5, 2.0, d) + rng.uniform(1e3, 1e4, d)
    rare = np.zeros(n, dtype=bool)
    rare[rng.choice(n, 40, replace=False)] = True
    x[~rare, d - 1] = np.nan
    q = np.array(
        [[0, d - 1, *rng.choice(np.arange(1, d - 1), 2, replace=False)] for _ in range(60)]
    )
    lo, hi, conds = q[:, 0], q[:, 1], np.sort(q[:, 2:], axis=1)
    test = make_indep_test("fisherz", x)
    grouped_sums = test._grouped_sums()
    assert grouped_sums is not None and len(grouped_sums.rows) == 1
    r, p = test.statistic_pvalues(lo, hi, conds)
    _per_row(monkeypatch)
    r_ref, p_ref = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    assert np.isfinite(p_ref).all()
    assert r.tolist() == r_ref.tolist() and p.tolist() == p_ref.tolist()


def test_one_chunk_mixes_grouped_and_per_row_blocks(monkeypatch, grouped):
    x = _cohorts(seed=10)
    monkeypatch.setenv("ANDREY_TESTWISE_GROUP_MIN_GAIN", "40")  # sends some blocks per row
    summed = _spy(monkeypatch, missing_groups.GroupedSums, "blocks")
    per_row = _spy(monkeypatch, type(make_indep_test("fisherz", x)), "_summed")
    queries = np.sort(np.array(_queries(x.shape[1], seed=11)[2]), axis=1)
    lo, hi, conds = queries[:, 0], queries[:, 1], queries[:, 2:]
    r, p = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    assert summed and per_row  # both paths answered blocks of the same chunk
    _per_row(monkeypatch)
    r_ref, p_ref = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    assert np.array_equal(np.isnan(p), np.isnan(p_ref))
    assert np.allclose(r, r_ref, rtol=RTOL, atol=1e-12, equal_nan=True)
    assert np.allclose(p, p_ref, rtol=RTOL, atol=1e-12, equal_nan=True)


def test_the_correction_step_gives_the_same_bits_in_smaller_steps(monkeypatch, grouped):
    """With the step size lowered, the rows missing several columns of a block are added in many
    pieces, a block's rows split across them; the sums keep their bits."""
    x = _cohorts(seed=12)
    queries = np.sort(np.array(_queries(x.shape[1], seed=13)[4]), axis=1)
    lo, hi, conds = queries[:, 0], queries[:, 1], queries[:, 2:]
    r, p = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    monkeypatch.setattr(missing_groups, "_STEP_BYTES", 1 << 12)
    corrected: list[int] = []
    original = missing_groups.GroupedSums._twice

    def counted(self, g, local, *sums):
        corrected.append(len(local) if self.twice_missed[g].shape[1] else 0)
        return original(self, g, local, *sums)

    monkeypatch.setattr(missing_groups.GroupedSums, "_twice", counted)
    r_small, p_small = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    assert sum(corrected) > 0  # rows missing several columns were added, in many small steps
    assert r.tolist() == r_small.tolist()
    assert np.array_equal(p, p_small, equal_nan=True)


def test_a_row_missing_more_than_255_columns_of_a_block_is_corrected_once():
    rng = np.random.default_rng(14)
    n, d = 60, 300
    observed = rng.random((n, d)) >= 0.01
    observed[:3, :280] = False  # three rows miss 280 columns of the block
    values = np.where(observed, rng.standard_normal((n, d)), 0.0).T.copy()
    grouping = missing_groups.Grouping(
        np.zeros(n, dtype=np.int64), np.ones((1, d), dtype=bool), ~observed
    )
    block = np.arange(d)[None, :]
    count, sums, _ = missing_groups.GroupedSums(values, grouping).blocks(block)
    kept = observed.all(axis=1)
    assert count[0] == kept.sum()
    assert np.allclose(sums[0], values[:, kept].sum(axis=1), atol=1e-9)


@pytest.mark.parametrize("method", ["iamb", "inter_iamb"])
def test_sets_left_out_for_keeping_no_rows_count_as_asking_them_would(monkeypatch, method):
    from andrey.constraint import _skeleton, blanket

    x = _cohorts(**APART)
    fit = getattr(andrey, method)
    left: list[int] = []
    kept_apart = _skeleton.kept_apart

    def counted(*args):
        kept = kept_apart(*args)
        left.append(kept[1])
        return kept

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for module in (_skeleton, blanket):
            monkeypatch.setattr(module, "kept_apart", counted)
        found = fit(x, indep_test="fisherz", missing="testwise")
        assert sum(left) > 0  # some sets were left out
        for module in (_skeleton, blanket):
            monkeypatch.setattr(module, "kept_apart", lambda test, fixed, sides, size: (sides, 0))
        asked = fit(x, indep_test="fisherz", missing="testwise")
    assert np.array_equal(found.structure.to_numpy(), asked.structure.to_numpy())
    for key in ("not_tested", "undecided_pairs", "rows_tested", "never_observed_together"):
        assert found.metadata.get(key) == asked.metadata.get(key), key
    assert found.metadata["not_tested"] > 0


def test_a_block_with_more_correction_rows_than_a_step_holds_is_added_in_parts(
    monkeypatch, grouped
):
    x = _cohorts()
    q = np.sort(np.array(_queries(x.shape[1])[3]), axis=1)
    lo, hi, conds = q[:, 0], q[:, 1], q[:, 2:]
    reference = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    parts: list[int] = []
    original = missing_groups.GroupedSums._parts

    def counted(*args):
        parts.append(1)
        return original(*args)

    monkeypatch.setattr(missing_groups.GroupedSums, "_parts", staticmethod(counted))
    monkeypatch.setattr(missing_groups, "_PART_BYTES", 1)  # one correction row per part
    test = make_indep_test("fisherz", x)
    assert test._grouped_sums() is not None
    r, p = test.statistic_pvalues(lo, hi, conds)
    assert parts
    _per_row(monkeypatch)
    r_ref, p_ref = make_indep_test("fisherz", x).statistic_pvalues(lo, hi, conds)
    for got, want in ((r, r_ref), (p, p_ref), (reference[0], r_ref)):
        assert np.allclose(got, want, rtol=RTOL, atol=1e-12, equal_nan=True)
