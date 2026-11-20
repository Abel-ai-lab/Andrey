"""Missing values: test-wise deletion in the CI tests, and the methods that accept or refuse NaN.

A test on data with ``NaN`` reads the rows observed in all of its variables, and gives the p-value
the same test gives on that block: bit for bit, or for the partial-correlation tests, which compute
each block's correlations directly, within ``RTOL``. The methods whose spec sets ``data.missing``
accept ``NaN``; every other method raises a ``ValueError`` that names it.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.cli import main
from andrey.core import ci, warning_policy
from andrey.core.ci import (
    FisherZ,
    PartialCorrelationTest,
    StudentT,
    make_indep_test,
    register_indep_test,
)
from andrey.core.ci_missing import (
    _STEP_BYTES,
    EM_METHODS,
    MORE_ROWS,
    TESTWISE_CLAIM,
    rows_metadata,
    rows_tested,
)
from andrey.core.score import CopiedColumnsError
from andrey.spec import REGISTRY
from andrey.temporal.pcmci import design_test, lagged_design
from tests.recovery.helpers.ci_tests import PValueOnly
from tests.unit.methods import SHORT

SUPPORTED = [name for name, spec in REGISTRY.items() if spec.data.missing]
REFUSED = [name for name, spec in REGISTRY.items() if not spec.data.missing]
SERIES = [name for name in SUPPORTED if REGISTRY[name].family == "temporal"]
# The partial-correlation tests compute a block's correlations from sums over every row with the
# unobserved ones zeroed; their p-values and partial correlations match the test on the block within
# these tolerances.
GAUSSIAN = ("fisherz", "student_t")
RTOL = 1e-8
ATOL_R = 1e-10


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        yield


def _gaps(x: np.ndarray, rate: float, seed: int = 1) -> np.ndarray:
    """``x`` with each cell missing completely at random with probability ``rate``."""
    out = np.array(x, dtype=np.float64)
    out[np.random.default_rng(seed).random(out.shape) < rate] = np.nan
    return out


def _continuous(n: int = 300, seed: int = 0) -> np.ndarray:
    """A chain 0 -> 2 -> 3 with independent columns 1 and 4."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 5))
    x[:, 2] += x[:, 0]
    x[:, 3] += x[:, 2]
    return x


def _categorical(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, (n, 4)).astype(np.float64)
    x[:, 1] = np.where(rng.random(n) < 0.7, x[:, 0], x[:, 1])
    x[:, 2] = np.where(rng.random(n) < 0.7, x[:, 1], x[:, 2])
    return x


def _mixed(n: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x = _continuous(n, seed)
    x[:, 0] = np.random.default_rng(seed + 1).integers(0, 3, n)
    x[:, 2] += x[:, 0]
    return x, np.array([True, False, False, False, False])


# Each shipped test, with the data it reads and its categorical mask.
TESTS = {
    "fisherz": (_continuous, None),
    "student_t": (_continuous, None),
    "normal_scores": (_continuous, None),
    "kci": (lambda: _continuous(120), None),
    "rcot": (_continuous, None),
    "cmiknn": (lambda: _continuous(120), None),
    "gpdc": (lambda: _continuous(60), None),
    "chisq": (_categorical, None),
    "gsq": (_categorical, None),
    "degen_gauss": (lambda: _mixed()[0], _mixed()[1]),
    "cond_gauss": (lambda: _mixed()[0], _mixed()[1]),
}
QUERIES = [(0, 3, ()), (3, 0, (2,)), (0, 3, (2, 1)), (1, 2, (3,))]
#: The shipped partial-correlation tests, whose degrees of freedom set the rows a query needs.
PARTIAL = [
    name
    for name, test in ci._CI_REGISTRY.items()
    if isinstance(test, type) and issubclass(test, PartialCorrelationTest)
]


def _block(x: np.ndarray, columns: list[int]) -> np.ndarray:
    rows = ~np.isnan(x[:, columns]).any(axis=1)
    return x[np.ix_(rows, columns)]


@pytest.mark.parametrize("name", list(TESTS))
def test_a_test_on_missing_values_is_the_test_on_the_rows_its_variables_share(name):
    make, mask = TESTS[name]
    x = _gaps(make(), 0.1)
    test = make_indep_test(name, x, categorical=mask)
    for a, b, cond in QUERIES:
        columns = sorted({a, b, *cond})
        at = {c: i for i, c in enumerate(columns)}
        sub_mask = None if mask is None else mask[columns]
        alone = make_indep_test(name, _block(x, columns), categorical=sub_mask)
        want = alone(at[a], at[b], [at[c] for c in cond])
        if name in GAUSSIAN:
            assert test(a, b, cond) == pytest.approx(want, rel=RTOL, abs=0)
        else:
            assert test(a, b, cond) == want


@pytest.mark.parametrize("name", ["fisherz", "student_t"])
def test_a_partial_correlation_on_missing_values_reports_its_block_statistic(name):
    x = _gaps(_continuous(), 0.15)
    test = make_indep_test(name, x)
    statistic, p = test.statistic_pvalue(3, 0, [2])
    want_r, want_p = make_indep_test(name, _block(x, [0, 2, 3])).statistic_pvalue(0, 2, [1])
    assert statistic == pytest.approx(want_r, rel=0, abs=ATOL_R)
    assert p == pytest.approx(want_p, rel=RTOL, abs=0)
    r, ps = test.statistic_pvalues(np.array([0, 3]), np.array([3, 1]), np.array([[2], [2]]))
    assert (r[0], ps[0]) == (statistic, p)
    assert (r[1], ps[1]) == test.statistic_pvalue(1, 3, [2])


@pytest.mark.parametrize("name", GAUSSIAN)
def test_a_partial_correlation_matches_the_test_on_its_block_on_random_gaps(name):
    rng = np.random.default_rng(11)
    for _ in range(12):
        d, n = int(rng.integers(4, 9)), int(rng.choice([40, 300]))
        x = rng.standard_normal((n, d)) @ rng.standard_normal((d, d)) + rng.uniform(-50, 50, d)
        x[rng.random(x.shape) < rng.choice([0.05, 0.2, 0.4])] = np.nan
        test = make_indep_test(name, x)
        for size in range(min(4, d - 2) + 1):
            for _ in range(5):
                a, b, *cond = rng.permutation(d)[: size + 2].tolist()
                columns = sorted({a, b, *cond})
                at = {c: i for i, c in enumerate(columns)}
                block = _block(x, columns)
                if len(block) < size + 3:
                    assert np.isnan(test.statistic_pvalue(a, b, cond)).all()  # not tested
                    continue
                alone = make_indep_test(name, block)
                want_r, want_p = alone.statistic_pvalue(at[a], at[b], [at[c] for c in cond])
                r, p = test.statistic_pvalue(a, b, cond)
                assert r == pytest.approx(want_r, rel=0, abs=ATOL_R)
                assert p == pytest.approx(want_p, rel=RTOL, abs=0)
                for alpha in (0.01, 0.05):
                    assert (p > alpha) == (want_p > alpha)


def test_every_query_path_of_a_partial_correlation_gives_the_same_bits():
    x = _gaps(_continuous(), 0.2)
    queries = [(0, 3, (2, 1)), (1, 2, (4, 3)), (0, 4, (1, 3)), (2, 4, (0, 1))]
    alone = [make_indep_test("fisherz", x)(a, b, cond) for a, b, cond in queries]
    lo, hi = np.array([q[0] for q in queries]), np.array([q[1] for q in queries])
    conds = np.array([q[2] for q in queries])
    together = make_indep_test("fisherz", x)
    assert together._canonical_pvalues(lo, hi, np.sort(conds, axis=1)).tolist() == alone
    assert make_indep_test("fisherz", x).statistic_pvalues(hi, lo, conds)[1].tolist() == alone
    mixed = make_indep_test("fisherz", x).batched_call(0, 3, [(2, 1), (), (1,), (2,)])
    assert mixed[0] == alone[0]
    assert mixed[1:] == [make_indep_test("fisherz", x)(0, 3, cond) for cond in [(), (1,), (2,)]]


@pytest.mark.parametrize("name", GAUSSIAN)
def test_a_query_gives_the_same_bits_alone_and_in_any_batch_on_many_rows(name):
    # On many rows a sum can run in pieces whose bounds follow the batch; the bits stay the block's.
    rng = np.random.default_rng(5)
    n, d = 9_000, 10
    x = rng.standard_normal((n, d)) @ rng.standard_normal((d, d)) + rng.uniform(-50, 50, d)
    x[rng.random(x.shape) < 0.15] = np.nan
    by_size = {}
    for size in range(1, 5):
        picks = {
            tuple(rng.permutation(d)[: size + 2].tolist()) for _ in range(60 if size == 1 else 16)
        }
        by_size[size] = sorted(picks)
    # More new blocks than one step of row values holds, so a batch is computed in several steps.
    assert len({tuple(sorted(q)) for q in by_size[1]}) > _STEP_BYTES // (8 * 3 * n)
    alone = {
        q: make_indep_test(name, x).statistic_pvalue(q[0], q[1], q[2:])
        for qs in by_size.values()
        for q in qs
    }
    for size, queries in by_size.items():
        order = [queries[i] for i in rng.permutation(len(queries))]
        batches = [order[:1], order[1:3], order[3:6], order[6:11], order[11:], order[::-1]]
        for batch in batches:
            r, p = make_indep_test(name, x).statistic_pvalues(
                np.array([q[1] for q in batch]),
                np.array([q[0] for q in batch]),
                np.array([q[2:] for q in batch]),
            )
            assert list(zip(r.tolist(), p.tolist())) == [alone[q] for q in batch]
        canonical = make_indep_test(name, x)._canonical_pvalues(
            np.array([min(q[:2]) for q in order]),
            np.array([max(q[:2]) for q in order]),
            np.sort(np.array([q[2:] for q in order]), axis=1),
        )
        assert canonical.tolist() == [alone[q][1] for q in order]
    a, b = by_size[2][0][:2]
    sets = [q[2:] for qs in by_size.values() for q in qs if a not in q[2:] and b not in q[2:]][::-1]
    mixed = make_indep_test(name, x).batched_call(a, b, sets)
    assert mixed == [make_indep_test(name, x).statistic_pvalue(a, b, s)[1] for s in sets]


def test_a_registered_test_is_built_on_each_query_block():
    seen = []

    def factory(block):
        seen.append(block)
        return FisherZ(block)

    register_indep_test("seen", factory)
    try:
        x = _gaps(_continuous(), 0.1)
        make_indep_test("seen", x)(4, 1, [3])
    finally:
        ci._CI_REGISTRY.pop("seen")
    assert len(seen) == 1 and np.array_equal(seen[0], _block(x, [1, 3, 4]))


@pytest.mark.parametrize("name", ["fisherz", "chisq", "cond_gauss"])
def test_a_query_gives_one_p_value_whatever_its_argument_order_and_history(name):
    make, mask = TESTS[name]
    x = _gaps(make(), 0.1)
    first = make_indep_test(name, x, categorical=mask)
    later = make_indep_test(name, x, categorical=mask)
    for a, b, cond in QUERIES[::-1]:  # a different history of queries
        later(a, b, cond)
    for a, b, cond in QUERIES:
        p = first(a, b, cond)
        assert p == first(b, a, tuple(reversed(cond))) == later(a, b, cond)


def test_complete_data_get_the_test_itself():
    x = _continuous()
    assert type(make_indep_test("fisherz", x)) is FisherZ
    assert type(make_indep_test("student_t", x)) is StudentT


def test_rows_tested_reports_the_fewest_and_the_median_rows():
    assert rows_metadata({}) == {}
    assert rows_metadata({7: 1}) == {"rows_tested": {"min": 7, "median": 7.0}}
    assert rows_metadata({3: 1, 9: 1}) == {"rows_tested": {"min": 3, "median": 6.0}}
    assert rows_metadata({3: 1, 5: 2, 9: 2}) == {"rows_tested": {"min": 3, "median": 5.0}}
    x = _continuous(100)
    x[:10, 0] = np.nan  # column 0 is observed on 90 rows, the others on all 100
    test = make_indep_test("fisherz", x)
    with rows_tested() as rows:
        test(0, 1)
        test(1, 2)
        test(0, 2, [3])
    assert dict(rows) == {90: 2, 100: 1}


# --- edge cases -----------------------------------------------------------------------------------


@pytest.mark.parametrize("page", ["api.md", "guides/getting-started.md"])
def test_a_page_that_names_the_tests_needing_more_rows_names_each_partial_correlation_test(page):
    """A partial-correlation test needs ``|S| + 3`` rows; the other tests need 2."""
    text = " ".join((Path(__file__).parents[2] / "docs" / "docs" / page).read_text().split())
    spans = re.findall(r"\|S\| \+ 3` for (.*?) and 2 for the other tests", text)
    assert spans
    for span in spans:
        named = set(re.findall(r'`"?(\w+)"?`', span)) & set(ci.SHIPPED_INDEP_TESTS)
        assert named in (set(), set(PARTIAL))


def _advises_more_rows(error) -> None:
    """The advice caps the conditioning sets first, and says what dropping a column can remove."""
    advice = str(error.value).partition("; ")[2]
    assert advice.startswith("cap the conditioning sets with max_depth")
    assert "or drop a column with many missing values, which can remove a common cause" in advice


@pytest.mark.parametrize(
    ("name", "size"), [*((name, size) for name in PARTIAL for size in (1, 2)), ("chisq", 1)]
)
def test_a_test_with_too_few_rows_is_not_tested_and_counted(name, size):
    """A partial-correlation test needs the rows that leave it one degree of freedom."""
    make, _ = TESTS[name]
    least = {"student_t": size + 3, "chisq": 2}.get(name, size + 4)
    x = make()[:40]
    x[least - 1 :, 0] = np.nan  # column 0 observed on least - 1 rows
    test = make_indep_test(name, x)
    with rows_tested() as rows:
        assert np.isnan(test(0, 1, list(range(2, 2 + size))))
    assert test.least_rows(size) == least
    assert rows.not_tested == {(0, 1): 1} and not rows  # counted apart, reading no rows
    assert rows_metadata(rows, int) == {
        "rows_tested": {"min": 0, "median": 0},
        "not_tested": 1,
        "undecided_pairs": [[0, 1]],
    }
    if name in PARTIAL:  # one row more: the test runs, with one degree of freedom
        x[least - 1, 0] = 1.0
        assert 0.0 <= make_indep_test(name, x)(0, 1, list(range(2, 2 + size))) <= 1.0


@pytest.mark.parametrize("name", list(TESTS))
def test_a_small_block_gives_a_p_value_or_raises(name):
    make, mask = TESTS[name]
    for rows in (2, 3, 4, 5):
        x = make()[:40]
        x[rows:, 1] = np.nan
        test = make_indep_test(name, x, categorical=mask)
        for size in (0, 1, 2):
            try:
                p = test(0, 1, list(range(2, 2 + size)))
            except ValueError as error:
                assert f"the {rows} rows where all" in str(error)
                assert str(error).endswith(MORE_ROWS)
            else:  # too few rows or no degrees of freedom: not tested
                zero_dof = hasattr(ci._CI_REGISTRY[name], "zero_dof_pvalue") or name == "gsq"
                assert 0.0 <= p <= 1.0 or (
                    np.isnan(p) and (rows < test.least_rows(size) or zero_dof)
                )


def test_a_registered_test_with_p_values_only_reports_statistics_under_deletion(monkeypatch):
    """Without a many-query method, each query is asked of the test over its own rows and counted
    in ``rows_tested``; the statistic orders the queries by p-value, smallest first."""
    monkeypatch.setitem(ci._CI_REGISTRY, "p_value_only", PValueOnly)
    test = make_indep_test("p_value_only", _gaps(_continuous(), 0.1))
    x, y = np.array([0, 0, 1, 4]), np.array([3, 2, 4, 2])
    conditions = np.array([[2], [4], [3], [1]])
    with rows_tested() as rows:
        statistics, p_values = test.statistic_pvalues(x, y, conditions)
    assert sum(rows.values()) == len(x)
    assert p_values.tolist() == [test(a, b, c) for a, b, c in zip(x, y, conditions.tolist())]
    assert np.array_equal(np.argsort(-statistics), np.argsort(p_values))


def test_a_registered_test_runs_on_one_blas_thread():
    register_indep_test("pinned", lambda block: FisherZ(block))
    try:
        assert make_indep_test("pinned", _gaps(_continuous(), 0.1)).pin_blas
    finally:
        ci._CI_REGISTRY.pop("pinned")


def test_a_column_constant_on_a_test_rows_raises_for_a_continuous_test():
    x = _continuous(60)
    x[:30, 1] = 2.0
    x[30:, 0] = np.nan  # the rows where column 0 is observed hold column 1 constant
    with pytest.raises(ValueError, match="constant on the 30 rows") as error:
        make_indep_test("fisherz", x)(0, 1)
    _advises_more_rows(error)
    assert 0.0 <= make_indep_test("fisherz", x)(1, 2) <= 1.0


def test_a_category_alone_on_a_test_rows_is_left_to_the_test():
    x = _categorical(60)
    x[:30, 1] = 2.0
    x[30:, 0] = np.nan
    assert make_indep_test("chisq", x)(0, 1) == 1.0  # no degree of freedom


def test_copies_raise_naming_the_data_columns():
    x = _continuous()
    x[:, 4] = 3.0 * x[:, 1] - 1.0
    x = _gaps(x, 0.1)
    with pytest.raises(CopiedColumnsError, match="4 repeats 1") as error:
        make_indep_test("fisherz", x)(1, 4)
    assert error.value.groups == ((1, 4),)
    frame = pd.DataFrame(x, columns=list("abcde"))
    with pytest.raises(CopiedColumnsError, match="'e' repeats 'b'"):
        andrey.pc(frame)


@pytest.mark.parametrize("name", SUPPORTED)
def test_a_column_with_no_observed_value_raises(name):
    x = _gaps(_continuous(), 0.1)
    x[:, 1] = np.nan
    frame = pd.DataFrame(x, columns=list("abcde"))
    with pytest.raises(ValueError, match="column 'b' holds no observed value"):
        _fit(name, frame)


def test_a_column_constant_on_its_observed_values_raises():
    x = _gaps(_continuous(), 0.1)
    x[~np.isnan(x[:, 2]), 2] = 5.0
    with pytest.raises(ValueError, match="column 2 is constant"):
        andrey.pc(x)


@pytest.mark.parametrize("name", SUPPORTED)
def test_inf_still_raises(name):
    x = _gaps(_continuous(), 0.1)
    x[4, 0] = np.inf
    with pytest.raises(ValueError, match=r"holds inf in 1 cell at row 4, column 0"):
        _fit(name, x)


@pytest.mark.parametrize("indep_test", ["chisq", "gsq", "cond_gauss"])
def test_a_missing_category_is_a_missing_value(indep_test):
    pd_frame = pd.DataFrame(_categorical(), columns=list("abcd"))
    pd_frame["b"] = pd_frame["b"].map({0.0: "low", 1.0: "mid", 2.0: "high"})
    pd_frame.loc[[3, 17, 40], "b"] = None
    pd_frame.loc[[5, 9], "a"] = np.nan
    out = andrey.pc(pd_frame, indep_test=indep_test)
    assert out.metadata["rows_tested"]["min"] < len(pd_frame)
    rows = pd_frame.notna().all(axis=1).to_numpy()
    assert out.metadata["rows_tested"]["min"] >= rows.sum()


def test_pandas_na_is_a_missing_value():
    x = _gaps(_continuous(), 0.1)
    frame = pd.DataFrame(x, columns=list("abcde")).astype("Float64")
    assert pd.isna(frame.iloc[:, 0]).any() and frame.iloc[:, 0].dtype == "Float64"
    assert andrey.pc(frame) == andrey.pc(pd.DataFrame(x, columns=list("abcde")))
    with pytest.raises(
        ValueError, match="^ges: pass missing=.*; ges does not accept missing values"
    ):
        andrey.ges(frame)


def test_the_category_code_warning_reads_the_observed_values():
    """Column a holds three codes and gaps: the warning names it, and declaring it reads it as
    categorical with its gaps."""
    x, _ = _mixed()
    frame = pd.DataFrame(_gaps(x, 0.1), columns=list("abcde"))
    warning_policy._warned.clear()
    with pytest.warns(andrey.DataTypeWarning, match="columns 'a' hold at most 10"):
        out = andrey.pc(frame, indep_test="cond_gauss")
    assert out.metadata["categorical"] == []
    out = andrey.pc(frame, indep_test="cond_gauss", categorical=["a"])
    assert out.metadata["categorical"] == ["a"]


def test_cdnod_reads_a_complete_domain_index_and_refuses_the_independent_change_step():
    x = _gaps(_continuous(), 0.1)
    domains = np.repeat([0.0, 1.0], 150)
    assert "rows_tested" in andrey.cdnod(x, domains).metadata
    gappy = domains.copy()
    gappy[3] = np.nan
    with pytest.raises(ValueError, match="env contains NaN"):
        andrey.cdnod(x, gappy)
    with pytest.raises(ValueError, match="dependence_threshold step .* does not accept missing"):
        andrey.cdnod(x, domains, dependence_threshold=0.1)


def test_a_lagged_design_reads_the_steps_its_lagged_variables_share():
    x = _gaps(_continuous(400), 0.05)
    design = lagged_design(x, 4)
    test = design_test("student_t", x, 2)
    columns = [0, 6, 13]  # variable 0 now, variable 1 one step back, variable 3 two steps back
    rows = ~np.isnan(design[:, columns]).any(axis=1)
    alone = StudentT(design[np.ix_(rows, columns)])
    statistic, p = test.statistic_pvalues(np.array([0]), np.array([13]), np.array([[6]]))
    want_r, want_p = alone.statistic_pvalue(0, 2, [1])
    assert statistic[0] == pytest.approx(want_r, rel=0, abs=ATOL_R)
    assert p[0] == pytest.approx(want_p, rel=RTOL, abs=0)


# --- the methods ----------------------------------------------------------------------------------


def _fit(name: str, x, **kwargs):
    fit = getattr(andrey, name)
    kwargs = {**SHORT.get(name, {}), **kwargs}
    if name in ("cdnod", "fci_jci"):
        return fit(x, np.repeat([0.0, 1.0], len(x) // 2 + 1)[: len(x)], **kwargs)
    if REGISTRY[name].family == "pairwise":  # a pair test reads two columns
        return fit(x.iloc[:, :2] if isinstance(x, pd.DataFrame) else x[:, :2], **kwargs)
    return fit(x, **kwargs)


@pytest.mark.parametrize("name", SUPPORTED)
def test_a_supported_method_runs_on_missing_values_and_repeats(name):
    x = _gaps(_continuous(400), 0.1)
    first, again = _fit(name, x), _fit(name, x)
    assert first.structure == again.structure
    assert first.metadata == again.metadata
    rows = first.metadata["rows_tested"]
    assert 0 < rows["min"] <= rows["median"] <= len(x)
    assert "rows_tested" not in _fit(name, _continuous(400)).metadata


@pytest.mark.parametrize("name", SUPPORTED)
def test_testwise_is_the_default_handling_and_the_fit_records_it(name):
    x = _gaps(_continuous(400), 0.1)
    default, testwise = _fit(name, x), _fit(name, x, missing="testwise")
    assert default.structure == testwise.structure
    assert default.metadata == testwise.metadata
    assert default.metadata["missing"] == "testwise"
    complete = _fit(name, _continuous(400), missing="testwise")
    assert "missing" not in complete.metadata
    assert complete.structure == _fit(name, _continuous(400)).structure


@pytest.mark.parametrize("name", SUPPORTED)
def test_a_handling_the_method_does_not_list_raises_naming_the_argument(name):
    with pytest.raises(ValueError, match=rf"^{name}: missing must be one of 'testwise'.*'em'"):
        _fit(name, _gaps(_continuous(400), 0.1), missing="em")


def test_the_methods_that_take_missing_accept_missing_values_or_fit_them_by_em():
    takes = {m for m, spec in REGISTRY.items() if any(p.name == "missing" for p in spec.params)}
    assert takes == set(SUPPORTED) | set(EM_METHODS)
    assert not set(SUPPORTED) & set(EM_METHODS)


@pytest.mark.parametrize("name", ["pc", "fci", "cdnod"])
def test_more_workers_give_the_same_result_on_missing_values(name, monkeypatch):
    # Every pass with a conditioning set goes to the pool where a pool runs.
    monkeypatch.setenv("ANDREY_SKELETON_PARALLEL_MIN_WORK", "1")
    x = _gaps(_continuous(400), 0.1)
    serial = _fit(name, x)
    with andrey.config(n_jobs=2):
        assert _fit(name, x) == serial


def _labeled_marks(out) -> dict:
    m, labels = out.structure.to_numpy(), out.structure.labels
    return {(a, b): int(m[i, j]) for i, a in enumerate(labels) for j, b in enumerate(labels)}


@pytest.mark.parametrize("name", ["pc", "fci", "mmpc"])
def test_a_column_order_changes_no_graph_on_missing_values(name):
    from andrey import data

    sample = data.sample_scm(d=8, n=500, seed=2, density=3.0, scale="standardize")
    frame = pd.DataFrame(_gaps(np.asarray(sample.data), 0.1), columns=list("abcdefgh"))
    want = _labeled_marks(getattr(andrey, name)(frame))
    for seed in (0, 1):
        order = list(np.random.default_rng(seed).permutation(list(frame.columns)))
        assert _labeled_marks(getattr(andrey, name)(frame[order])) == want


def _refused_inputs(name: str) -> list:
    spec = REGISTRY[name]
    x = _gaps(_continuous(200), 0.05)
    args: list = []
    for inp in spec.data.inputs:
        if not inp.required:
            continue
        if inp.kind == "panel":
            args.append([x, _continuous(200, 1)])
        elif inp.kind == "context":
            args.append(np.repeat([0.0, 1.0], 100))
        elif spec.family == "pairwise":
            args.append(x[:, :2])
        else:
            args.append(x)
    return args


@pytest.mark.parametrize("name", REFUSED)
def test_every_other_method_refuses_missing_values_naming_itself(name):
    if name == "calm":
        pytest.importorskip("torch")  # without the extra, calm raises ImportError first
    # The message leads with what the caller can do, then says what was refused and where.
    pattern = (
        rf"^{name}: (drop|use|pass) .*; {name} does not accept missing values, "
        r"and data\S* holds NaN .*row \d+, column \d+"
    )
    with pytest.raises(ValueError, match=pattern) as error:
        getattr(andrey, name)(*_refused_inputs(name))
    if name == "longitudinal_lingam":  # a panel: a unit's rows go together
        assert (
            f"{name}: drop the units that hold a missing value from every dataset of data"
            in str(error.value)
        )
    elif name == "jpcmciplus":  # datasets read with their contexts: no other method stands in
        assert f"{name}: drop the datasets of data that hold a missing value" in str(error.value)
        assert "use pcmci, pcmciplus, or lpcmci" not in str(error.value)
    elif REGISTRY[name].family == "temporal":  # one series: a step cannot be dropped
        assert f"{name}: use pcmci, pcmciplus, or lpcmci" in str(error.value)
        assert "keep every time step" in str(error.value)
        assert "fill in" not in str(error.value)  # a filled step is a single imputation
    elif name in EM_METHODS:
        assert f'{name}: pass missing="em"' in str(error.value)
        assert "drop the rows of data that hold a missing value yourself" in str(error.value)
    else:
        assert f"{name}: drop the rows of data that hold a missing value yourself" in str(
            error.value
        )
        assert "the same rows of every argument that has one entry per row" in str(error.value)
        assert "pc, fci, cdnod, fci_jci, iamb, inter_iamb, or mmpc" in str(error.value)
        assert 'missing="em"' not in str(error.value)


def test_a_refused_text_category_names_the_method():
    frame = pd.DataFrame(_categorical(), columns=list("abcd"))
    frame["b"] = frame["b"].map({0.0: "low", 1.0: "mid", 2.0: "high"})
    frame.loc[3, "b"] = None
    with pytest.raises(
        ValueError, match=r"^hc: drop .*; hc does not accept .* column 'b' has a missing value"
    ):
        andrey.hc(frame, score_func="bdeu")


def test_the_refusal_lists_every_supported_method_but_the_pair_tests():
    from andrey.api._adapt import TESTWISE_METHODS, TESTWISE_SERIES_METHODS

    tables = [name for name in SUPPORTED if REGISTRY[name].family != "pairwise"]
    assert [*TESTWISE_METHODS, *TESTWISE_SERIES_METHODS] == tables


@pytest.mark.parametrize("name", ["anm", "pnl", "loci"])
def test_a_pair_test_reads_the_rows_where_both_columns_are_observed(name):
    x = np.random.default_rng(0).uniform(-1, 1, (300, 2))
    x[:, 1] = x[:, 0] ** 3 + 0.3 * x[:, 1]
    gappy = _gaps(x, 0.1)
    out = getattr(andrey, name)(gappy)
    rows = ~np.isnan(gappy).any(axis=1)
    alone = getattr(andrey, name)(gappy[rows])
    assert out.structure == alone.structure
    assert out.metadata == {
        **alone.metadata,
        "rows_tested": {"min": rows.sum(), "median": rows.sum()},
        "missing": "testwise",
    }


@pytest.mark.parametrize("name", ["anm", "pnl", "loci"])
def test_a_pair_test_with_too_few_rows_raises(name):
    x = _continuous(40)[:, :2]
    x[2:, 0] = np.nan
    x[1, 1] = np.nan  # both columns are observed on row 0 only
    with pytest.raises(ValueError, match="needs at least 2 rows where both columns are observed"):
        getattr(andrey, name)(x)
    x = _continuous(40)[:, :2]
    x[:20, 1] = 1.0
    x[20:, 0] = np.nan
    frame = pd.DataFrame(x, columns=["u", "v"])
    with pytest.raises(ValueError, match="column 'v' is constant on the 20 rows where both"):
        getattr(andrey, name)(frame)


@pytest.mark.parametrize("name", ["pc", "fci"])
def test_bootstrap_resamples_rows_with_missing_values(name):
    x = _gaps(_continuous(300), 0.1)
    boot = andrey.bootstrap(name, x, n_resamples=4, seed=0)
    assert boot.n_resamples == 4


# --- the command line -----------------------------------------------------------------------------


def _csv(tmp_path) -> str:
    x = _gaps(_continuous(300), 0.1)
    path = tmp_path / "gaps.csv"
    rows = [",".join("" if np.isnan(v) else repr(float(v)) for v in row) for row in x]
    path.write_text("a,b,c,d,e\n" + "\n".join(rows) + "\n")
    return str(path)


def test_the_cli_reads_an_empty_cell_as_a_missing_value(tmp_path, capsys):
    code = main(["run", "pc", "--data", _csv(tmp_path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["metadata"]["rows_tested"]["min"] < 300
    assert out["structure"]["labels"] == list("abcde")


@pytest.mark.parametrize("token", ["NA", "N/A", "nan", ""])
def test_the_cli_reads_na_and_na_with_a_slash_as_missing_values(tmp_path, capsys, token):
    x = _continuous(300)
    gappy = _gaps(x, 0.1)
    path = tmp_path / "tokens.csv"
    rows = [",".join(token if np.isnan(v) else repr(float(v)) for v in row) for row in gappy]
    path.write_text("\n".join(rows) + "\n")  # no header: the first row may hold a missing value
    code = main(["run", "pc", "--data", str(path), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["metadata"] == andrey.pc(gappy).metadata


def test_the_cli_reports_a_refusal_as_a_data_error(tmp_path, capsys):
    code = main(["run", "ges", "--data", _csv(tmp_path), "--json"])
    err = json.loads(capsys.readouterr().err)
    assert code == 3 and err["error"]["code"] == "data_error"
    assert "ges does not accept missing values" in err["error"]["message"]


def test_the_cli_reads_an_empty_cell_in_the_first_data_row(tmp_path, capsys):
    path = tmp_path / "first.csv"
    rows = _continuous(200)
    text = "\n".join(",".join(repr(float(v)) for v in row) for row in rows)
    path.write_text("," + text[text.index(",") + 1 :] + "\n")  # the first cell is empty
    code = main(["run", "pc", "--data", str(path), "--json"])
    assert code == 0 and json.loads(capsys.readouterr().out)["structure"]["labels"] is None


@pytest.mark.parametrize("name", SERIES)
def test_a_series_method_refits_a_series_with_missing_steps(name):
    x = _gaps(_continuous(400), 0.05)
    out = getattr(andrey, name)(x, tau_max=1)
    assert out.metadata["rows_tested"]["min"] < 400 - 2


@pytest.mark.parametrize("name", ["varma_lingam", "granger"])
def test_the_other_series_methods_refuse_missing_steps(name):
    with pytest.raises(
        ValueError, match=rf"^{name}: use pcmci, .*; {name} does not accept missing values"
    ):
        getattr(andrey, name)(_gaps(_continuous(400), 0.05))


def _notices(caplog) -> list[str]:
    """The notices of a default handling logged so far."""
    return [r.getMessage() for r in caplog.records if r.name == "andrey.api._method"]


@pytest.mark.parametrize("name", SUPPORTED)
def test_the_default_handling_logs_one_notice_naming_it_and_its_assumption(name, caplog):
    x = _gaps(_continuous(400), 0.1)
    with caplog.at_level(logging.WARNING, logger="andrey"):
        _fit(name, _continuous(400))  # no missing value
        _fit(name, x, missing="testwise")  # the handling chosen by name
        assert not _notices(caplog)
        _fit(name, x)
        _fit(name, x)
    (notice,) = _notices(caplog)
    assert notice.startswith(f"andrey.{name}: ")
    assert "test-wise deletion" in notice and TESTWISE_CLAIM in notice
    assert 'missing="testwise"' in notice


def test_the_notice_is_one_per_process_whatever_the_method(caplog):
    x = _gaps(_continuous(400), 0.1)
    with caplog.at_level(logging.WARNING, logger="andrey"):
        andrey.pc(x)
        andrey.fci(x)
        andrey.anm(x[:, :2])
    (notice,) = _notices(caplog)
    assert notice.startswith("andrey.pc: ")


def test_a_fit_by_em_logs_no_notice_of_a_default(caplog):
    with caplog.at_level(logging.WARNING, logger="andrey"):
        out = andrey.ges(_gaps(_continuous(400), 0.1), missing="em")
    assert out.metadata["missing"] == "em" and not _notices(caplog)


_BOOTSTRAP_PROBE = """
import numpy as np

import andrey

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((300, 4))
    x[rng.random(x.shape) < 0.1] = np.nan
    with andrey.config(n_jobs=2):
        for _ in range(2):
            andrey.bootstrap("pc", x, n_resamples=4, seed=0, missing=MISSING)
"""


@pytest.mark.parametrize(("missing", "notices"), [(None, 1), ("testwise", 0)])
def test_bootstrap_on_worker_processes_logs_the_notice_once_in_all(missing, notices, tmp_path):
    """The workers of every bootstrap call stay silent; the calling process logs the one notice."""
    probe = tmp_path / "probe.py"
    probe.write_text(_BOOTSTRAP_PROBE.replace("MISSING", repr(missing)))
    run = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, check=True)
    assert run.stderr.count("the default handling, test-wise deletion") == notices
    if notices:
        assert "andrey.pc: " in run.stderr


def _one_category_where_observed() -> np.ndarray:
    """Categorical data whose column 1 holds one category on the rows that observe column 0."""
    x = _categorical(60)
    x[:30, 1] = 2.0
    x[30:, 0] = np.nan
    return x


def _starved_queries(name: str) -> tuple[np.ndarray, np.ndarray | None, list[tuple]]:
    """Data, categorical mask, and queries for the test ``name`` where some block is starved.

    Categorical and mixed tests get a column with one category on the rows that observe another
    (zero degrees of freedom); the others get a column observed on 4 rows, so a test given one
    conditioner reads ``|S| + 3`` rows, where the partial-correlation tests have no degrees of
    freedom left.
    """
    if name in ci.DISCRETE_INDEP_TESTS or name in ci.MIXED_INDEP_TESTS:
        x = _one_category_where_observed()
        mask = None
        if name in ci.MIXED_INDEP_TESTS:
            mask = np.ones(x.shape[1], dtype=bool)
    else:
        x = np.random.default_rng(5).standard_normal((40, 4))
        x[4:, 2] = np.nan
        mask = None
    d = x.shape[1]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    queries = [(i, j, []) for i, j in pairs]
    queries += [(i, j, [k]) for i, j in pairs for k in range(d) if k not in (i, j)]
    return x, mask, queries


def _ask(test, query: tuple, path: str) -> float | None:
    """The p-value of ``query`` through the wrapper's method ``path``; ``None`` when it raises."""
    try:
        return _ask_path(test, query, path)
    except ValueError:
        return None


def _ask_path(test, query: tuple, path: str) -> float:
    x, y, cond = query
    if path == "call":
        return float(test(x, y, cond))
    if path == "batched_call":
        return float(test.batched_call(x, y, [cond])[0])
    xs, ys, conds = np.array([x]), np.array([y]), np.array([cond], dtype=np.intp).reshape(1, -1)
    return float(test.statistic_pvalues(xs, ys, conds)[1][0])


@pytest.mark.parametrize("name", ci.available_indep_tests())
def test_every_registered_test_counts_the_queries_it_answers_with_p_one(name):
    x, mask, queries = _starved_queries(name)
    test = make_indep_test(name, x, categorical=mask)
    paths = [
        "call",
        "statistic_pvalues",
        *(["batched_call"] if hasattr(test, "batched_call") else []),
    ]
    for path in paths:
        with rows_tested() as rows:
            answers = [p for p in (_ask(test, q, path) for q in queries) if p is not None]
        ones = sum(p == 1.0 for p in answers)
        skipped = sum(np.isnan(p) for p in answers)
        assert answers and sum(rows.values()) == len(answers) - skipped, (name, path)
        assert rows.total_not_tested() == skipped, (name, path)
        assert rows.p_one == ones, (name, path)
        assert rows_metadata(rows).get("p_one_tests", 0) == ones, (name, path)


@pytest.mark.parametrize(
    "name", ["fisherz", "normal_scores", "chisq", "gsq", "degen_gauss", "cond_gauss"]
)
def test_a_test_with_no_degrees_of_freedom_left_is_not_tested(name):
    x, mask, queries = _starved_queries(name)
    test = make_indep_test(name, x, categorical=mask)
    with rows_tested() as rows:
        answers = np.array([test(*q) for q in queries])
    assert np.isnan(answers).any() and rows.p_one == 0
    assert rows.total_not_tested() == np.isnan(answers).sum() < len(queries)


def test_a_registered_test_is_counted_as_the_shipped_ones_are(monkeypatch):
    answer = {"p": 1.0}
    monkeypatch.setitem(ci._CI_REGISTRY, "constant", lambda data: lambda x, y, s=(): answer["p"])
    data = _gaps(_continuous(60), 0.1)
    test = make_indep_test("constant", data)
    with rows_tested() as rows:
        for pair in [(0, 1), (0, 2), (1, 2)]:
            test(*pair, [3])
        answer["p"] = 0.5
        test(0, 4)
    assert rows.p_one == 3 and sum(rows.values()) == 4


def test_a_fit_records_tests_without_degrees_of_freedom_as_not_tested():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(60)
    data = np.column_stack(
        [x, x + 0.3 * rng.standard_normal(60), x + 0.3 * rng.standard_normal(60)]
    )
    data[4:, 2] = np.nan  # the tests that read column 2 and a conditioner read 4 rows: no df left
    fisher = andrey.pc(data, indep_test="fisherz").metadata
    assert fisher["not_tested"] >= 1 and "p_one_tests" not in fisher
    student = andrey.pc(data, indep_test="student_t").metadata
    assert "not_tested" not in student and "p_one_tests" not in student
    out = andrey.pc(_one_category_where_observed(), indep_test="chisq")
    assert out.metadata["not_tested"] >= 1 and "p_one_tests" not in out.metadata
    assert "p_one_tests" not in andrey.pc(_gaps(_categorical(), 0.05), indep_test="chisq").metadata
    assert "p_one_tests" not in andrey.pc(_gaps(_continuous(400), 0.1)).metadata
    assert "p_one_tests" not in andrey.pc(_categorical(), indep_test="chisq").metadata
