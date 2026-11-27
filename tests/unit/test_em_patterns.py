"""The E-step on a pattern plan equals the per-pattern loop it replaces, bit for bit.

``reference_moments`` is that loop: one solve per missingness pattern, each pattern's share added
to the expected covariance and its term subtracted from the log-likelihood in the order of the
patterns. ``expected_moments`` solves the patterns of one row that miss the same number of columns
as stacks, so each case compares the mean, the covariance, and the log-likelihood byte for byte,
and the log-likelihood's type. A total can hide a pattern term that moved by one rounding, so the
random cases hold two or three rows, where a moved term moves the total, with one, two, and three
observed columns: the stacked einsum of two observed columns sums in another order than a single
call, which only such a case shows. A pattern that observes one column or misses one multiplies in
a dot product, which Apple's Accelerate sums by the memory alignment of its second operand: a stack
that held it moved a bit for an odd slice of 5 or more columns, on macOS arm64 only, so such a
pattern stays alone, and the cases cover every count of observed columns with several rows each.
The sizes sit at the edges of the code: a stack of one pattern, the stack limit and the patterns on
each side of it, and the patterns that stay alone (a pattern of several rows, a complete row, a row
that observes nothing, a row that observes one column or misses one).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import andrey.search.em as em_module
from andrey.search.em import Patterns, expected_moments

nan = np.nan


def same_bits(a, b) -> bool:
    """Equal shape, dtype, and bytes, so NaN and the sign of zero count."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def reference_moments(X, mean, precision):
    """The E-step with one solve and one share per missingness pattern."""
    n, d = X.shape
    observed = ~np.isnan(X)
    patterns, group = np.unique(observed, axis=0, return_inverse=True)
    filled = np.where(observed, X, 0.0)
    extra = np.zeros((d, d))
    log_det = np.linalg.slogdet(precision)[1]
    loglik = 0.0
    for k, pattern in enumerate(patterns):
        rows = np.flatnonzero(group.ravel() == k)
        o, m = np.flatnonzero(pattern), np.flatnonzero(~pattern)
        resid = X[np.ix_(rows, o)] - mean[o]
        if m.size == 0:
            inner = precision
            log_det_o = log_det
        else:
            k_mm = precision[np.ix_(m, m)]
            k_mo = precision[np.ix_(m, o)]
            factor = np.linalg.cholesky(k_mm)
            solved = np.linalg.solve(factor.T, np.linalg.solve(factor, k_mo))
            filled[np.ix_(rows, m)] = mean[m] - resid @ solved.T
            inverse = np.linalg.inv(factor)
            extra[np.ix_(m, m)] += rows.size * (inverse.T @ inverse)
            inner = precision[np.ix_(o, o)] - k_mo.T @ solved
            log_det_o = log_det - 2.0 * float(np.log(np.diagonal(factor)).sum())
        if o.size:
            quadratic = float(np.einsum("ij,jk,ik->", resid, inner, resid))
            loglik -= 0.5 * (quadratic + rows.size * (o.size * np.log(2 * np.pi) - log_det_o))
    expected_mean = filled.mean(axis=0)
    centered = filled - expected_mean
    return expected_mean, (centered.T @ centered + extra) / n, loglik


def model(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A mean and the precision matrix of a sparse linear-Gaussian model of ``d`` columns."""
    coefficients = np.triu(rng.normal(size=(d, d)) * (rng.random((d, d)) < 0.4), 1)
    lower = np.eye(d) - coefficients
    return rng.normal(size=d), (lower / rng.uniform(0.2, 3.0, d)) @ lower.T


def rows_observing(
    d: int, observed: int, count: int, rng: np.random.Generator, distinct: bool = True
) -> np.ndarray:
    """``count`` rows of ``d`` columns, each observing ``observed`` random columns."""
    masks: list[tuple[int, ...]] = []
    while len(masks) < count:
        mask = tuple(sorted(rng.choice(d, observed, replace=False)))
        if not distinct or mask not in masks:
            masks.append(mask)
    X = np.full((count, d), nan)
    for row, mask in enumerate(masks):
        X[row, list(mask)] = 5.0 * rng.standard_normal(observed)
    return X


def check(X, mean, precision) -> None:
    """The plan's E-step, built inside the call or beforehand, matches the reference loop."""
    want = reference_moments(X, mean, precision)
    for got in (
        expected_moments(X, mean, precision),
        expected_moments(Patterns(X), mean, precision),
    ):
        assert same_bits(got[0], want[0])
        assert same_bits(got[1], want[1])
        assert type(got[2]) is type(want[2])
        assert same_bits(np.float64(got[2]), np.float64(want[2]))


@pytest.mark.parametrize("observed", [1, 2, 3])
@pytest.mark.parametrize("d", [4, 5, 6, 8, 12])
@pytest.mark.parametrize("rows", [2, 3])
def test_two_or_three_one_row_patterns_match_the_loop_term_by_term(observed, d, rows):
    # Each draw has two or three patterns of one row with the same number of observed columns, so
    # they form one stack when they observe and miss two or more columns, and a total of two or
    # three terms moves when a term moves.
    for seed in range(40):
        rng = np.random.default_rng(1000 * observed + 10 * d + seed)
        mean, precision = model(d, rng)
        check(rows_observing(d, observed, rows, rng), mean, precision)


@pytest.mark.parametrize("d", [4, 5, 6, 8, 12, 20])
@pytest.mark.parametrize("rows", [2, 3, 4])
def test_patterns_missing_one_column_match_the_loop(d, rows):
    # Each row observes all columns but one, so its filled value is a dot product of d - 1 terms.
    for seed in range(40):
        rng = np.random.default_rng(100 * d + 10 * rows + seed)
        mean, precision = model(d, rng)
        check(rows_observing(d, d - 1, rows, rng), mean, precision)


@pytest.mark.parametrize("d", [7, 12, 20])
def test_every_count_of_observed_columns_matches_the_loop_over_odd_and_even_slices(d):
    # Nine or fewer one-row patterns for each count of observed columns, so a stack has slices at
    # odd and even offsets of memory, with an odd and an even number of columns in each part.
    for observed in range(1, d):
        for seed in range(6):
            rng = np.random.default_rng(10_000 * d + 100 * observed + seed)
            mean, precision = model(d, rng)
            check(rows_observing(d, observed, min(9, math.comb(d, observed)), rng), mean, precision)


def test_no_stack_holds_a_pattern_that_observes_one_column_or_misses_one():
    d = 9
    rng = np.random.default_rng(8)
    mean, precision = model(d, rng)
    X = np.vstack([rows_observing(d, observed, 4, rng) for observed in range(1, d)])
    plan = Patterns(X)
    assert plan.stacks
    assert all(min(os_.shape[1], ms.shape[1]) >= 2 for _, _, os_, ms, _ in plan.stacks)
    assert {o.size for _, _, o, *_ in plan.alone} >= {1, d - 1}  # one observed, one missing
    assert {m.size for _, _, _, m, *_ in plan.alone} >= {1, d - 1}
    check(X, mean, precision)


@pytest.fixture
def mixed():
    """Data of seven columns with every kind of pattern, and a model for it."""
    rng = np.random.default_rng(3)
    d = 7
    mean, precision = model(d, rng)
    parts = [rng.standard_normal((30, d))]  # complete rows
    for observed in (1, 2, 3, 4, 5, 6):
        parts.append(rows_observing(d, observed, 14, rng, distinct=False))  # some patterns repeat
    X = np.vstack([*parts, np.full((3, d), nan)])  # rows that observe nothing
    return rng.permutation(X, axis=0), mean, precision


def test_every_kind_of_pattern_matches_the_loop(mixed):
    X, mean, precision = mixed
    patterns, counts = np.unique(~np.isnan(X), axis=0, return_counts=True)
    assert (counts == 1).any() and (counts > 1).any()  # one-row and several-row patterns
    assert patterns.all(axis=1).any()  # a complete pattern
    assert (~patterns.any(axis=1)).any()  # a pattern that observes nothing
    check(X, mean, precision)


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 8, 256])
def test_the_stack_limit_moves_no_bit(mixed, monkeypatch, limit):
    X, mean, precision = mixed
    monkeypatch.setattr(em_module, "STACK_LIMIT", limit)
    plan = Patterns(X)
    sizes = [len(ks) for ks, *_ in plan.stacks]
    assert max(sizes) <= limit
    assert limit > 1 or set(sizes) == {1}
    check(X, mean, precision)


@pytest.mark.parametrize("patterns", [255, 256, 257, 513])
def test_patterns_on_each_side_of_the_stack_limit_match_the_loop(patterns):
    # More patterns of one size than the limit, which split into stacks of the limit and one of the
    # rest, among them a stack of one pattern.
    rng = np.random.default_rng(patterns)
    d, observed = 14, 5
    mean, precision = model(d, rng)
    X = rows_observing(d, observed, patterns, rng)
    plan = Patterns(X)
    assert sorted(len(ks) for ks, *_ in plan.stacks) == sorted(
        [em_module.STACK_LIMIT] * (patterns // em_module.STACK_LIMIT)
        + [patterns % em_module.STACK_LIMIT] * bool(patterns % em_module.STACK_LIMIT)
    )
    check(X, mean, precision)


def test_a_plan_over_its_cell_budget_solves_each_pattern_alone(mixed, monkeypatch):
    X, mean, precision = mixed
    patterns = np.unique(~np.isnan(X), axis=0)
    shares = int(((X.shape[1] - patterns.sum(axis=1)) ** 2).sum())
    for budget, direct in [(shares, False), (shares - 1, True), (1, True)]:
        monkeypatch.setattr(em_module, "PLAN_CELLS", budget)
        plan = Patterns(X)
        assert plan.direct_adds is direct
        assert bool(plan.stacks) is not direct
        check(X, mean, precision)


def test_a_stack_holds_no_more_cells_than_the_budget(monkeypatch):
    # 100 one-row patterns of nine missing columns among 14, which add up to 8,100 shares: the
    # budget lets a stack hold 41 patterns, of 14 * 14 entries each in its widest array.
    rng = np.random.default_rng(2)
    mean, precision = model(14, rng)
    X = rows_observing(14, 5, 100, rng)
    monkeypatch.setattr(em_module, "PLAN_CELLS", 8100)
    plan = Patterns(X)
    assert not plan.direct_adds
    assert sorted(len(ks) for ks, *_ in plan.stacks) == [18, 41, 41]
    check(X, mean, precision)


def test_a_plan_serves_every_model_and_is_not_changed_by_a_call(mixed):
    X, mean, precision = mixed
    before = X.copy()
    plan = Patterns(X)
    rng = np.random.default_rng(5)
    for _ in range(3):  # each round of an EM fit has its own model
        mean, precision = model(X.shape[1], rng)
        got = expected_moments(plan, mean, precision)
        want = reference_moments(X, mean, precision)
        assert same_bits(got[0], want[0]) and same_bits(got[1], want[1])
        assert same_bits(np.float64(got[2]), np.float64(want[2]))
    assert same_bits(X, before)


def test_data_without_a_missing_value_or_without_an_observed_value_matches_the_loop():
    rng = np.random.default_rng(11)
    mean, precision = model(5, rng)
    check(rng.standard_normal((20, 5)), mean, precision)  # one complete pattern, nothing stacks
    nothing = np.full((4, 5), nan)
    got = expected_moments(nothing, mean, precision)
    want = reference_moments(nothing, mean, precision)
    assert type(got[2]) is type(want[2]) is float
    assert got[2] == want[2] == 0.0
    assert same_bits(got[0], want[0]) and same_bits(got[1], want[1])
