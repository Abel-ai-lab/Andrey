"""Missing values: test-wise deletion in the CI tests, and the methods that accept or refuse NaN.

A test on data with ``NaN`` reads the rows observed in all of its variables, and gives the p-value
the same test gives on that block: bit for bit, or for the partial-correlation tests, which compute
each block's correlations directly, within ``RTOL``. The methods whose spec sets ``data.missing``
accept ``NaN``; every other method raises a ``ValueError`` that names it.
"""

from __future__ import annotations

import inspect
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
from andrey.constraint import blanket
from andrey.core import categorical, ci, ci_missing, warning_policy
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
    MISSING_GUIDE,
    MORE_ROWS,
    TESTWISE_CLAIM,
    reading_flags,
    rows_metadata,
    rows_tested,
)
from andrey.core.score import CopiedColumnsError
from andrey.spec import REGISTRY
from andrey.spec.generate import render_skill
from andrey.temporal.pcmci import design_test, lagged_design
from qa import identity
from tests.recovery.helpers.ci_tests import PValueOnly
from tests.release import release_cases
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
    "gcm": (lambda: _continuous(60), None),
    "wgcm": (lambda: _continuous(60), None),
    "pcm": (lambda: _continuous(60), None),
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
#: Settings that keep the regression tests fast; the wrapping does not depend on the regression.
OPTIONS = {
    "gcm": {"regression": "linear"},
    "wgcm": {"regression": "linear"},
    "pcm": {"regression": "linear", "n_splits": 1},
}


def _block(x: np.ndarray, columns: list[int]) -> np.ndarray:
    rows = ~np.isnan(x[:, columns]).any(axis=1)
    return x[np.ix_(rows, columns)]


@pytest.mark.parametrize("name", list(TESTS))
def test_a_test_on_missing_values_is_the_test_on_the_rows_its_variables_share(name):
    make, mask = TESTS[name]
    x = _gaps(make(), 0.1)
    test = make_indep_test(name, x, categorical=mask, **OPTIONS.get(name, {}))
    for a, b, cond in QUERIES:
        columns = sorted({a, b, *cond})
        at = {c: i for i, c in enumerate(columns)}
        sub_mask = None if mask is None else mask[columns]
        alone = make_indep_test(
            name, _block(x, columns), categorical=sub_mask, **OPTIONS.get(name, {})
        )
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
                if len(block) < ci._CI_REGISTRY[name].least_rows(size):
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


#: The gaps of :func:`_scan_data`, each sending the queries of a scan down one path of
#: :class:`~andrey.core.ci_missing.GaussianDeletion`.
SCAN_KINDS = ["cells", "blocks", "few", "tied", "near", "copy", "apart"]


def _scan_data(kind: str, seed: int, d: int = 24, n: int = 1200) -> np.ndarray:
    """Data of ``d`` columns with gaps of one kind.

    ``cells`` are missing completely at random. ``blocks`` come from four sources with a few
    patterns, which the grouped sums read. ``few`` has few rows and many gaps, so queries are not
    tested. ``tied`` has columns of three levels, one of them most of the rows, so a block may be
    constant. In ``near``, column 7 is a hair from a copy of column 3, within the copy margin, so
    the exact path answers; in ``copy`` it is a copy up to rounding, which the exact path raises on.
    In ``apart``, columns 2 and 9 are never observed together.
    """
    rng = np.random.default_rng(seed)
    x = np.array(andrey.data.sample_scm(d=d, n=n, seed=seed, density=2.0).data, dtype=np.float64)
    rate = 0.05
    if kind == "blocks":
        own = np.array_split(np.arange(5, d), 4)
        for source, rows in enumerate(np.array_split(rng.permutation(n), 4)):
            hidden = np.concatenate([own[other] for other in range(4) if other != source])
            x[np.ix_(rows, hidden)] = np.nan
        rate = 0.01
    elif kind == "few":
        rate = 0.45
    elif kind == "tied":
        x[:, ::3] = np.round(x[:, ::3] / (3 * x[:, ::3].std(axis=0)))
        rate = 0.08
    elif kind in ("near", "copy"):
        x[:, 7] = (
            2.0 * x[:, 3] + 1.0 + (1.1e-5 if kind == "near" else 1e-9) * rng.standard_normal(n)
        )
    elif kind == "apart":
        x[: n // 2, 2] = np.nan
        x[n // 2 :, 9] = np.nan
    x[rng.random(x.shape) < rate] = np.nan
    return x


def _scans(d: int, rng: np.random.Generator, always: tuple[int, ...] = (3, 7, 2)):
    """Scans ``(target, candidates, conditioning set)`` of every size from 0 to 6: for the targets
    ``always`` and four more, the set unsorted as a blanket is and the candidates in any order."""
    targets = [*always, *rng.choice([c for c in range(d) if c not in always], 4, replace=False)]
    for target in targets:
        for size in (0, 1, 2, 3, 4, 6):
            others = [c for c in range(d) if c != target]
            cond = rng.choice(others, size=size, replace=False).tolist()
            ys = [c for c in others if c not in cond]
            rng.shuffle(ys)
            yield int(target), ys, cond


def _counted(counter: ci_missing.RowCounts) -> tuple:
    """Everything a :class:`~andrey.core.ci_missing.RowCounts` holds, its row counts and its
    untested queries in the order they were added."""
    return (
        list(counter.items()),
        counter.p_one,
        list(counter.not_tested.items()),
        counter.largest,
        counter.untested_sets,
        counter.not_tested_elsewhere,
    )


@pytest.mark.parametrize("flags", [False, True], ids=["no_flags", "flags"])
@pytest.mark.parametrize("name", GAUSSIAN)
@pytest.mark.parametrize("kind", SCAN_KINDS)
def test_a_scan_on_missing_values_is_each_candidate_asked_alone_bit_for_bit(
    kind, name, flags, monkeypatch
):
    """``statistic_pvalues_given`` gives each candidate the partial correlation and p-value of the
    same query asked alone, bit for bit and NaN for NaN, counts the rows and the untested queries as
    the separate calls do, and raises where they raise, the same error with the same queries counted
    before it, whichever path the query takes: per-row or grouped sums, the exact path, or not
    tested."""
    x = _scan_data(kind, seed=1, d=14 if kind == "few" else 24, n=60 if kind == "few" else 1200)
    scanned, alone = make_indep_test(name, x), make_indep_test(name, x)
    assert type(scanned) is ci_missing.GaussianDeletion
    exact = [0]
    fast = scanned._fast

    def counting_fast(*args):
        out = fast(*args)
        exact[0] += int(out[3].sum())  # the queries the exact path answers
        return out

    monkeypatch.setattr(scanned, "_fast", counting_fast)
    raised = untested = 0
    for target, ys, cond in _scans(x.shape[1], np.random.default_rng(1)):
        got_error = want_error = None
        with reading_flags(flags), rows_tested() as got_counter:
            try:
                got = scanned.statistic_pvalues_given(target, ys, cond)
            except ValueError as error:
                got_error = error
        with reading_flags(flags), rows_tested() as want_counter:
            try:
                want = [alone.statistic_pvalue(target, y, cond) for y in ys]
            except ValueError as error:
                want_error = error
        assert _counted(got_counter) == _counted(want_counter)
        if got_error is not None or want_error is not None:
            assert type(got_error) is type(want_error)
            assert str(got_error) == str(want_error)
            raised += 1
            continue
        r, p = got
        np.testing.assert_array_equal(r, [a for a, _ in want])
        np.testing.assert_array_equal(p, [b for _, b in want])
        untested += int(np.isnan(p).sum())
    # Each kind reaches the path it is named for.
    assert raised or kind != "copy"
    assert exact[0] or kind not in ("near", "copy", "tied")
    assert untested or kind not in ("blocks", "few", "apart")
    assert scanned._grouped is not None or kind != "blocks"


@pytest.mark.parametrize("kind", ["cells", "blocks", "few", "copy"])
def test_a_scan_in_chunks_is_the_scan_at_once_bit_for_bit(kind, monkeypatch):
    """A scan whose candidates go in chunks, as a large conditioning set makes it, gives the
    answers, the rows counted, and the error of the scan at once, whatever the chunk."""
    x = _scan_data(kind, seed=4, d=14 if kind == "few" else 24, n=60 if kind == "few" else 1200)
    scans = list(_scans(x.shape[1], np.random.default_rng(4)))

    def run(entries: int):
        monkeypatch.setattr(ci_missing, "FALLBACK_STACK_ENTRIES", entries)
        test, out = make_indep_test("fisherz", x), []
        for target, ys, cond in scans:
            with rows_tested() as counter:
                try:
                    out.append(test.statistic_pvalues_given(target, ys, cond))
                except ValueError as error:
                    out.append((type(error), str(error)))
            out.append(_counted(counter))
        return out

    once = run(1 << 20)
    for entries in (1, 50, 300):  # chunks of one, a few, and about a dozen candidates
        chunked = run(entries)
        for want, got in zip(once, chunked, strict=True):
            if isinstance(want, tuple) and isinstance(want[0], np.ndarray):
                for a, b in zip(want, got, strict=True):
                    np.testing.assert_array_equal(a, b)
            else:
                assert got == want
    assert any(isinstance(out, tuple) and isinstance(out[0], type) for out in once) == (
        kind == "copy"
    )


def test_a_scan_takes_its_conditioning_set_as_one_query_does():
    """The set may be unsorted, repeat a column, or come from a generator; a candidate or ``x`` in
    it raises the assertion a query raises; no candidate gives no answer."""
    x = _gaps(_continuous(), 0.1)
    test = make_indep_test("fisherz", x)
    want = test.statistic_pvalues_given(0, [1, 4], [2, 3])
    for cond in ([3, 2], [2, 3, 3, 2], (c for c in (3, 2)), np.array([3, 2])):
        r, p = test.statistic_pvalues_given(0, np.array([1, 4]), cond)
        assert (r.tolist(), p.tolist()) == (want[0].tolist(), want[1].tolist())
    assert [test.statistic_pvalue(0, y, [2, 3]) for y in (1, 4)] == list(zip(*map(list, want)))
    for target, ys, cond in [(0, [1, 2], [2]), (2, [1, 4], [2])]:
        with pytest.raises(AssertionError) as scan:
            test.statistic_pvalues_given(target, ys, cond)
        with pytest.raises(AssertionError) as alone:
            test.statistic_pvalue(target, 2 if 2 in ys else ys[0], cond)
        assert str(scan.value) == str(alone.value)
    for cond in ([], [2, 3]):
        r, p = test.statistic_pvalues_given(0, [], cond)
        assert r.shape == p.shape == (0,)


#: The gaps of :func:`_discrete_scan_data`, each sending the queries of a chi-square scan down one
#: path of :class:`~andrey.core.ci_missing.ChiSquareDeletion`.
DISCRETE_SCAN_KINDS = ["cells", "rare", "few", "apart", "pair", "flat", "levels"]


def _discrete_scan_data(kind: str, seed: int, d: int = 10, n: int | None = None) -> np.ndarray:
    """Categorical data of ``d`` columns, each close to the one before it, with gaps of one kind.

    ``cells`` are missing completely at random. In ``rare``, columns 1 and 4 hold a level that three
    rows have, so a block can lose it, column 0 holds non-integer values, and a fifth of the cells
    are missing. ``few`` has few rows and many gaps, so queries are not tested. In ``apart``,
    columns 2 and 6 are never observed together, and column 4 is never observed; in ``pair``,
    columns 2 and 6 are observed together on two rows only, the fewest a query is tested on. In
    ``flat``, column 5 holds one category, so a query over it has no degrees of freedom. In
    ``levels``, columns 3 and 6 hold a hundred levels on few rows with many gaps, so a block has
    more of them than four for each of its rows.
    """
    n = n or {"few": 40, "levels": 150}.get(kind, 600)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 3, (n, d)).astype(np.float64)
    for column in range(1, d):
        x[:, column] = np.where(rng.random(n) < 0.5, x[:, column - 1], x[:, column])
    rate = 0.06
    if kind == "rare":
        for column in (1, 4):
            x[rng.choice(n, 3, replace=False), column] = 7.0
        x[:, 0] = x[:, 0] * 0.5 - 1.25
        rate = 0.2
    elif kind == "few":
        rate = 0.45
    elif kind in ("apart", "pair"):
        x[: n // 2, 2] = np.nan
        x[n // 2 :, 6] = np.nan
        if kind == "apart":
            x[:, 4] = np.nan
    elif kind == "flat":
        x[:, 5] = 1.0
    elif kind == "levels":
        x[:, [3, 6]] = rng.integers(0, 100, (n, 2))
        rate = 0.35
    x[rng.random(x.shape) < rate] = np.nan
    if kind == "pair":  # two rows observe every column, and differ in columns 2 and 6
        x[[0, -1]] = rng.integers(0, 3, (2, d))
        x[0, [2, 6]], x[-1, [2, 6]] = 0.0, 1.0
    return x


def _bits(p) -> list[int]:
    """The bits of each p-value, so a ``NaN`` equals a ``NaN``."""
    return np.asarray(p, dtype=np.float64).view(np.uint64).tolist()


@pytest.mark.parametrize("flags", [False, True], ids=["no_flags", "flags"])
@pytest.mark.parametrize("degrees_of_freedom", ["mean", "nonempty"])
@pytest.mark.parametrize("name", ["chisq", "gsq"])
@pytest.mark.parametrize("kind", DISCRETE_SCAN_KINDS)
def test_a_chi_square_scan_on_missing_values_is_each_candidate_asked_alone_bit_for_bit(
    kind, name, degrees_of_freedom, flags, monkeypatch
):
    """``pvalues_given`` gives each candidate the p-value of the same query asked alone on a fresh
    test, bit for bit and ``NaN`` for ``NaN``, and the running counter holds what the separate calls
    leave in it: the rows, the queries not tested, their pairs and sets, and the largest p-values,
    whichever way the query goes: a block that lost a category, too few rows, a pair never observed
    together, or no degrees of freedom."""
    x = _discrete_scan_data(kind, seed=1)
    built, sparse = [0], [0]
    real = ci_missing.ChiSquareDeletion._block_pvalue
    renumber = ci_missing._compact

    def counting(self, *args):
        built[0] += 1
        return real(self, *args)

    def renumbering(index, size, n):
        sparse[0] += size > 4 * n  # more categories than four for each row: renumbered by sorting
        return renumber(index, size, n)

    monkeypatch.setattr(ci_missing.ChiSquareDeletion, "_block_pvalue", counting)
    monkeypatch.setattr(ci_missing, "_compact", renumbering)
    untested = lost = fewest = 0
    for target, xs, cond in _scans(x.shape[1], np.random.default_rng(1)):
        scanned = make_indep_test(name, x, degrees_of_freedom=degrees_of_freedom)
        alone = make_indep_test(name, x, degrees_of_freedom=degrees_of_freedom)
        assert type(scanned) is ci_missing.ChiSquareDeletion
        with reading_flags(flags), rows_tested() as got_counter:
            got = scanned.pvalues_given(target, xs, cond)
        with reading_flags(flags), rows_tested() as want_counter:
            want = [alone(target, y, cond) for y in xs]
        assert _bits(got) == _bits(want)
        assert _counted(got_counter) == _counted(want_counter)
        untested += int(np.isnan(got).sum())
        fewest += 2 in got_counter  # a query tested on two rows
        for y in xs:  # a block with fewer categories in a column than the column has
            kept = ~np.isnan(x[:, [target, y, *cond]]).any(axis=1)
            lost += any(
                len(np.unique(x[kept, c])) < len(np.unique(x[~np.isnan(x[:, c]), c]))
                for c in (target, y, *cond)
                if kept.any()
            )
    assert built[0]
    assert untested or kind in ("cells", "rare")
    assert lost or kind != "rare"
    assert fewest or kind != "pair"
    assert sparse[0] or kind != "levels"


@pytest.mark.parametrize("kind", ["rare", "levels"])
def test_a_chi_square_scan_numbers_each_block_as_the_test_numbers_a_block_it_is_built_over(
    kind, monkeypatch
):
    """The codes a scan gives a candidate's block, its categories, and the numbering of the
    conditioning set's configurations are those of the test built over the block's values, also
    where the block lost a category or a configuration the other candidates have."""
    x = _discrete_scan_data(kind, seed=8)
    seen = []
    real = ci.ChiSquare._from_codes.__func__  # ty: ignore[unresolved-attribute]

    def capture(cls, codes, cardinalities, **options):
        seen.append((codes.copy(), cardinalities.copy()))
        return real(cls, codes, cardinalities, **options)

    monkeypatch.setattr(ci.ChiSquare, "_from_codes", classmethod(capture))
    lost = checked = 0
    for target, xs, cond in _scans(x.shape[1], np.random.default_rng(8)):
        seen.clear()
        make_indep_test("chisq", x).pvalues_given(target, xs, cond)
        blocks = iter(seen)
        for y in xs:
            a, b = sorted((target, y))
            columns = sorted((a, b, *cond))
            rows = np.flatnonzero(~np.isnan(x[:, columns]).any(axis=1))
            if len(rows) < 2:
                continue
            codes, cardinalities = next(blocks)
            block = ci.ChiSquare(x[np.ix_(rows, columns)])
            at = {column: i for i, column in enumerate(columns)}
            for i, column in enumerate((a, b)):
                np.testing.assert_array_equal(codes[:, i], block._codes[:, at[column]])
                assert cardinalities[i] == block._cardinalities[at[column]]
            if cond:
                layer, held = categorical.configurations(
                    block._rows.T, block._cardinalities, [at[c] for c in sorted(cond)]
                )
                np.testing.assert_array_equal(codes[:, 2], layer)
                assert cardinalities[2] == held
            assert len(cardinalities) == 2 + bool(cond)
            categories = [len(np.unique(x[~np.isnan(x[:, c]), c])) for c in (a, b)]
            base = ~np.isnan(x[:, [target, *cond]]).any(axis=1)
            configurations_of_base = len(np.unique(x[base][:, sorted(cond)], axis=0))
            lost += bool(
                (cardinalities[:2] < categories).any() or (cond and held < configurations_of_base)
            )
            checked += 1
    assert checked > 200
    assert lost


def test_a_chi_square_scan_answers_a_held_block_an_environment_and_a_source_as_separate_calls():
    """A candidate whose block the test holds, a scan over an environment column, and a test whose
    columns are another matrix's (``from_columns``) give the answers and the counter of the
    separate calls."""
    x = _discrete_scan_data("cells", seed=2, d=8)
    x[:, 7] = np.random.default_rng(2).integers(0, 2, len(x))  # an environment indicator
    x[np.random.default_rng(3).random(x.shape) < 0.1] = np.nan
    source = np.random.default_rng(4).permutation(11)[:8]
    for setup in ("held", "environment", "source"):
        scanned, alone = make_indep_test("chisq", x), make_indep_test("chisq", x)
        for test in (scanned, alone):
            if setup == "environment":
                test.mark_environments([7])
            if setup == "source":
                test.from_columns(source)
        for target, xs, cond in _scans(8, np.random.default_rng(5), always=(1, 3)):
            if setup == "held":  # both hold the blocks of every other candidate
                for y in xs[::2]:
                    scanned(target, y, cond)
                    alone(target, y, cond)
            with rows_tested() as got_counter:
                got = scanned.pvalues_given(target, xs, cond)
            with rows_tested() as want_counter:
                want = [alone(target, y, cond) for y in xs]
            assert _bits(got) == _bits(want)
            assert _counted(got_counter) == _counted(want_counter)


def test_a_chi_square_scan_that_raises_counts_the_queries_before_it(monkeypatch):
    """A scan whose fourth query fails, as one asked alone does, raises its error and leaves the
    first three in the counter; so does a candidate that is in the conditioning set."""
    x = _discrete_scan_data("cells", seed=6)
    real = ci.ChiSquare._counted_statistic_pvalue
    xs, cond = [0, 2, 3, 4, 7, 8], [1, 5]

    def run(ask, fails):
        calls = [0]

        def failing(self, *args):
            calls[0] += 1
            if fails and calls[0] == 4:
                raise ValueError("a failed test")
            return real(self, *args)

        with monkeypatch.context() as patch:
            patch.setattr(ci.ChiSquare, "_counted_statistic_pvalue", failing)
            with rows_tested() as counter, pytest.raises((ValueError, AssertionError)) as error:
                ask()
        return type(error.value), str(error.value), _counted(counter)

    scanned, alone = make_indep_test("gsq", x), make_indep_test("gsq", x)
    got = run(lambda: scanned.pvalues_given(6, xs, cond), fails=True)
    assert got == run(lambda: [alone(6, y, cond) for y in xs], fails=True)
    assert got[0] is ValueError and "a failed test" in got[1]
    assert sum(got[2][0][i][1] for i in range(len(got[2][0]))) == 3  # three queries counted
    scanned, alone = make_indep_test("gsq", x), make_indep_test("gsq", x)
    got = run(lambda: scanned.pvalues_given(6, [0, 2, 5, 3], cond), fails=False)
    assert got == run(lambda: [alone(6, y, cond) for y in [0, 2, 5, 3]], fails=False)
    assert got[0] is AssertionError


def test_a_chi_square_scan_over_a_column_with_an_infinity_raises_as_separate_calls_do():
    """The test reads a block with an infinity as ``ChiSquare`` reads one, and raises, so a scan
    answers a candidate, a target, or a conditioning column over such a column as the separate
    query does: a block that leaves the infinity's row out answers, and any other raises, after the
    queries before it are counted. A scan over the other columns is the scan of any other data."""
    x = _discrete_scan_data("cells", seed=9, d=8)
    x[5, 2], x[40, 6] = np.inf, -np.inf
    x[40, 3] = np.nan  # the row of the second infinity is left out of every block over column 3
    raised = answered = 0
    for target, xs, cond in _scans(8, np.random.default_rng(9), always=(2, 3, 4)):
        apart = [c for c in range(8) if c not in (2, 6)]
        kept = (target, [y for y in xs if y in apart], [c for c in cond if c in apart])
        for scan_target, scan_xs, scan_cond in ((target, xs, cond), kept):
            if scan_target not in apart and scan_xs is not xs:
                continue
            outcomes = []
            for ask in ("scan", "alone"):
                test = make_indep_test("chisq", x)
                with rows_tested() as counter:
                    try:
                        if ask == "scan":
                            out = _bits(test.pvalues_given(scan_target, scan_xs, scan_cond))
                        else:
                            out = _bits([test(scan_target, y, scan_cond) for y in scan_xs])
                    except ValueError as error:
                        out = (type(error), str(error))
                outcomes.append((out, _counted(counter)))
            assert outcomes[0] == outcomes[1]
            raised += isinstance(outcomes[0][0], tuple)
            answered += isinstance(outcomes[0][0], list)
    assert raised
    assert answered


def test_a_chi_square_test_over_missing_values_checks_its_degrees_of_freedom_when_built():
    """A scan builds no test over values, which would raise for a rule that does not exist, so the
    wrapper raises the error of the test when it is built."""
    x = _discrete_scan_data("cells", seed=7)
    for name in ("chisq", "gsq"):
        with pytest.raises(ValueError, match="degrees_of_freedom must be one of"):
            make_indep_test(name, x, degrees_of_freedom="exact")
        assert type(make_indep_test(name, x, degrees_of_freedom="nonempty")) is (
            ci_missing.ChiSquareDeletion
        )


def _blanket_search_digest(fit, x, options):
    """The digest of ``fit(x, **options)``'s output, or the type and message of its error."""
    try:
        return identity.digest(fit(x, **options))
    except ValueError as error:
        return type(error), str(error)


@pytest.mark.parametrize("options", [{}, {"target": [1, 6]}], ids=["graph", "targets"])
@pytest.mark.parametrize("fit", [andrey.iamb, andrey.inter_iamb], ids=["iamb", "inter_iamb"])
@pytest.mark.parametrize("kind", SCAN_KINDS)
def test_a_blanket_search_on_missing_values_is_the_search_asking_one_query_at_a_time(
    kind, fit, options, monkeypatch
):
    """``iamb`` and ``inter_iamb`` on data with gaps return the blanket, the separating sets, the
    rows counted, and every metadata value of the search that asks each candidate alone, or raise
    its error."""
    x = _scan_data(kind, seed=2, d=10 if kind == "few" else 18, n=60 if kind == "few" else 500)
    got = _blanket_search_digest(fit, x, options)
    assert blanket._dependence_scan(make_indep_test("fisherz", x)) is not None
    with monkeypatch.context() as asking_alone:
        asking_alone.setattr(ci_missing.GaussianDeletion, "statistic_pvalues_given", None)
        assert blanket._dependence_scan(make_indep_test("fisherz", x)) is None
        assert _blanket_search_digest(fit, x, options) == got


@pytest.mark.parametrize(
    "options",
    [
        {"indep_test": "student_t"},
        {"max_depth": 2},
        {"possibly_separable": "ambiguous"},
        {"knowledge": andrey.Knowledge(required=[(0, 5)], forbidden=[(2, 3), (3, 2)])},
    ],
    ids=["student_t", "max_depth", "ambiguous", "knowledge"],
)
@pytest.mark.parametrize("fit", [andrey.iamb, andrey.inter_iamb], ids=["iamb", "inter_iamb"])
@pytest.mark.parametrize("kind", ["cells", "apart"])
def test_a_blanket_search_with_an_option_on_missing_values_is_the_search_asking_one_query_at_a_time(
    kind, fit, options, monkeypatch
):
    """The options the grow step reads, each with untested queries among the scans or without."""
    x = _scan_data(kind, seed=3, d=14, n=400)
    got = _blanket_search_digest(fit, x, options)
    with monkeypatch.context() as asking_alone:
        asking_alone.setattr(ci_missing.GaussianDeletion, "statistic_pvalues_given", None)
        assert _blanket_search_digest(fit, x, options) == got


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"target": [1, 6]},
        {"degrees_of_freedom": "nonempty"},
        {"max_depth": 2},
        {"possibly_separable": "ambiguous"},
        {"knowledge": andrey.Knowledge(required=[(0, 5)], forbidden=[(2, 3), (3, 2)])},
    ],
    ids=["graph", "targets", "nonempty", "max_depth", "ambiguous", "knowledge"],
)
@pytest.mark.parametrize("indep_test", ["chisq", "gsq"])
@pytest.mark.parametrize("fit", [andrey.iamb, andrey.inter_iamb], ids=["iamb", "inter_iamb"])
@pytest.mark.parametrize("kind", ["cells", "rare", "few", "apart"])
def test_a_chi_square_blanket_search_on_missing_values_is_the_search_asking_one_query_at_a_time(
    kind, fit, indep_test, options, monkeypatch
):
    """``iamb`` and ``inter_iamb`` with ``chisq`` or ``gsq`` on data with gaps return the blanket,
    the separating sets, the rows counted, and every metadata value of the search that asks each
    candidate alone, or raise its error."""
    x = _discrete_scan_data(kind, seed=3, d=9, n=40 if kind == "few" else 400)
    options = {"indep_test": indep_test, **options}
    got = _blanket_search_digest(fit, x, options)
    assert blanket._dependence_scan(make_indep_test(indep_test, x)) is not None
    with monkeypatch.context() as asking_alone:
        asking_alone.setattr(ci_missing.ChiSquareDeletion, "pvalues_given", None)
        assert blanket._dependence_scan(make_indep_test(indep_test, x)) is None
        assert _blanket_search_digest(fit, x, options) == got


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


#: A row floor ``|S| + k`` and the tests it is given for, in backticks, as Notes and pages write it.
FLOOR = re.compile(
    r'`{1,2}\|S\| \+ (\d)`{1,2}(?: rows)? for ((?:`{1,2}"?\w+"?`{1,2}(?:,? and |, )?)+)'
)


def named_floors(text: str) -> list[tuple[int, list[str]]]:
    """Each ``|S| + k`` of ``text`` given for tests by name: ``k`` and the names."""
    return [(int(k), re.findall(r'`{1,2}"?(\w+)"?`{1,2}', run)) for k, run in FLOOR.findall(text)]


def assert_floors_match_the_tests(text: str) -> None:
    """Every test named beside ``|S| + k`` in ``text`` needs ``|S| + k`` rows (``least_rows``), and
    every partial-correlation test is named with its floor."""
    floors = named_floors(text)
    assert len(floors) == text.count("|S| + ")
    for offset, names in floors:
        for name in names:
            least = ci._CI_REGISTRY[name].least_rows
            assert [least(size) - size for size in range(4)] == [offset] * 4, name
    assert set(PARTIAL) <= {name for _, names in floors for name in names}


@pytest.mark.parametrize("page", ["api.md", "guides/missing-values.md"])
def test_a_page_that_names_the_tests_needing_more_rows_gives_each_its_floor(page):
    """A page gives each partial-correlation test, and each other test it names, the rows its
    ``least_rows`` asks for."""
    text = " ".join((Path(__file__).parents[2] / "docs" / "docs" / page).read_text().split())
    assert named_floors(text)
    assert_floors_match_the_tests(text)


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


@pytest.mark.parametrize(
    ("name", "options", "least"),
    [("gcm", {}, 4), ("wgcm", {}, 17), ("wgcm", {"variant": "fix"}, 4), ("pcm", {}, 12)],
)
def test_a_regression_test_with_too_few_rows_is_not_tested_and_counted(name, options, least):
    """A regression test needs the rows its fits split, whatever the conditioning set's size."""
    settings = OPTIONS[name] | options
    x = _continuous(60)[:40]
    x[least - 1 :, 0] = np.nan  # column 0 observed on least - 1 rows
    test = make_indep_test(name, x, seed=0, **settings)
    for size in (0, 1, 2, 3):
        assert test.least_rows(size) == least
        with rows_tested() as rows:
            assert np.isnan(test(0, 1, list(range(2, 2 + size))))
        assert rows.not_tested == {(0, 1): 1} and not rows  # counted apart, reading no rows
    x[least - 1, 0] = 1.0  # one row more: the test runs
    test = make_indep_test(name, x, seed=0, **settings)
    for size in (0, 1, 2, 3):
        assert 0.0 <= test(0, 1, list(range(2, 2 + size))) <= 1.0


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


def test_a_category_alone_on_a_test_rows_is_not_tested():
    x = _categorical(60)
    x[:30, 1] = 2.0
    x[30:, 0] = np.nan
    with rows_tested() as rows:
        assert np.isnan(make_indep_test("chisq", x)(0, 1))  # no degree of freedom
    assert rows.not_tested == {(0, 1): 1} and rows.p_one == 0


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


@pytest.mark.parametrize("name", release_cases(SUPPORTED, {"skewd"}))
def test_a_supported_method_runs_on_missing_values_and_repeats(name):
    x = _gaps(_continuous(400), 0.1)
    first, again = _fit(name, x), _fit(name, x)
    assert first.structure == again.structure
    assert first.metadata == again.metadata
    rows = first.metadata["rows_tested"]
    assert 0 < rows["min"] <= rows["median"] <= len(x)
    assert "rows_tested" not in _fit(name, _continuous(400)).metadata


@pytest.mark.parametrize("name", release_cases(SUPPORTED, {"skewd"}))
def test_testwise_is_the_default_handling_and_the_fit_records_it(name):
    x = _gaps(_continuous(400), 0.1)
    default, testwise = _fit(name, x), _fit(name, x, missing="testwise")
    assert default.structure == testwise.structure
    assert default.metadata == testwise.metadata
    assert default.metadata["missing"] == "testwise"
    complete = _fit(name, _continuous(400), missing="testwise")
    assert "missing" not in complete.metadata
    assert complete.structure == _fit(name, _continuous(400)).structure


def test_skewd_repeats_on_missing_values_and_records_its_handling():
    """SkewD on gapped data, on fewer rows than the registry-wide cases: the fit repeats and
    records the default handling and the rows each test reads."""
    x = _gaps(_continuous(60), 0.1)
    first, again = _fit("skewd", x), _fit("skewd", x)
    assert first.structure == again.structure
    assert first.metadata == again.metadata
    assert first.metadata["missing"] == "testwise"
    rows = first.metadata["rows_tested"]
    assert 0 < rows["min"] <= rows["median"] <= len(x)


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
    assert MISSING_GUIDE in str(error.value)
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


def test_the_guide_the_refusal_names_is_a_page_of_the_docs():
    site, page = MISSING_GUIDE.split("/docs/", 1)
    source = Path(__file__).parents[2] / "docs" / "docs" / page.replace(".html", ".md")
    assert site == "https://andrey.abel.ai" and source.is_file()


def test_no_text_says_that_fci_stays_sound_when_values_are_missing_for_a_reason():
    """FCI with one test per query is valid for values missing completely at random only."""
    texts = {name: inspect.getdoc(getattr(andrey, name)) or "" for name in SUPPORTED}
    texts |= {"ci_missing": ci_missing.__doc__ or "", "agent text": render_skill()}
    texts = {name: " ".join(text.split()) for name, text in texts.items()}
    assert not [name for name, text in texts.items() if "missingness mechanism" in text]
    assert TESTWISE_CLAIM in texts["fci"]
    assert TESTWISE_CLAIM in texts["agent text"]


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


@pytest.mark.parametrize("name", release_cases(SUPPORTED, {"skewd"}))
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


@pytest.mark.parametrize("name", ["pc", "fci", "iamb", "mmpc"])
def test_a_fit_that_runs_no_test_on_data_with_gaps_logs_the_notice_and_records_it(name, caplog):
    fit = getattr(andrey, name)
    x = _gaps(_continuous(400), 0.1)[:, :1]  # one column: no pair to test
    with caplog.at_level(logging.WARNING, logger="andrey"):
        assert "missing" not in fit(_continuous(400)[:, :1]).metadata
        out = fit(x)
    assert out.metadata["missing"] == "testwise"
    assert out.metadata["rows_tested"] == {"min": 0, "median": 0}
    (notice,) = _notices(caplog)
    assert notice.startswith(f"andrey.{name}: ")
    if name == "pc":
        assert andrey.pc(x, missing="mvpc").metadata["missing"] == "mvpc"


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
    conditioner reads ``|S| + 3`` rows, fewer than ``"fisherz"`` and ``"normal_scores"`` need.
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


@pytest.mark.parametrize("name", release_cases(ci.available_indep_tests(), {"wgcm"}))
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
