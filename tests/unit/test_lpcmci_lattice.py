"""The subset lattice of LPCMCI's separating-set search: its values, its screen, and its fallbacks.

The lattice reads the partial correlation of a ``p``-subset from its parent's factor, so its values
agree with the test's inversion to rounding, never bit for bit. What the search records of a batch,
the first separating subset, the smallest absolute statistic, and the largest p-value, carries the
test's own bits (the screen), so the search's result equals the batched path's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations

import numpy as np
import pytest

import andrey
from andrey.core import backend
from andrey.core.ci import FisherZ, StudentT
from andrey.data import sample_scm
from andrey.temporal import lpcmci as engine
from andrey.temporal import lpcmci_lattice as lattice
from andrey.temporal.lpcmci_lattice import SubsetLattice
from andrey.temporal.pcmci import DesignTest, design_test

ALPHA = 0.05
NO_FIXED = np.array([], dtype=np.intp)


def mixed(n: int, d: int, seed: int) -> np.ndarray:
    """``n`` rows of ``d`` columns, each a random mix of independent normals."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)) @ rng.uniform(-1, 1, (d, d)) + rng.standard_normal((n, d))


def design_of(data: np.ndarray, test_class: type = FisherZ) -> DesignTest:
    return DesignTest(test_class(data), data)


def reference(design: DesignTest, x: int, y: int, rows: np.ndarray):
    """The test's own statistics and p-values of ``x`` against ``y`` given each row."""
    n = len(rows)
    return design.statistic_pvalues(np.full(n, x), np.full(n, y), rows)


def level_rows(state: SubsetLattice, p: int) -> np.ndarray:
    """The queries of every ``p``-subset of the state's candidates, in search order."""
    return state.rows(np.arange(math.comb(state.m, p)))


def counting_builds(monkeypatch) -> list:
    """Wrap :meth:`SubsetLattice.build` to record each call, and return the list."""
    builds: list = []
    original = SubsetLattice.build

    def build(*args):
        builds.append(args)
        return original(*args)

    monkeypatch.setattr(SubsetLattice, "build", build)
    return builds


def same(a, b) -> bool:
    """Whether two metadata values hold the same bits."""
    if isinstance(a, Mapping):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(same(u, v) for u, v in zip(a, b, strict=True))
    return repr(a) == repr(b)


def unrank_list(m: int, q: int, ranks: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(row) for row in lattice.unrank(m, q, ranks).tolist()]


def test_unrank_gives_combinations_in_order():
    for m, q in [(5, 1), (7, 3), (8, 4), (6, 6), (9, 2)]:
        ranks = np.arange(math.comb(m, q))
        assert unrank_list(m, q, ranks) == list(combinations(range(m), q))
    picked = np.array([4, 0, 11])
    assert unrank_list(8, 3, picked) == [list(combinations(range(8), 3))[k] for k in picked]


@pytest.mark.parametrize("test_class", [FisherZ, StudentT])
@pytest.mark.parametrize("fixed_count", [0, 3])
def test_levels_agree_with_the_inversion_to_rounding(test_class, fixed_count):
    data = mixed(400, 12, seed=3)
    design = design_of(data, test_class)
    columns = np.array([5, 2, 9, 3, 7, 4, 6], dtype=np.intp)  # search order, not sorted
    fixed = np.array([10, 8, 11][:fixed_count], dtype=np.intp)
    state = SubsetLattice.build(design, 0, 1, columns, fixed)
    assert state is not None
    for p in range(1, len(columns) + 1):
        state.grow(p - 1)
        total = math.comb(len(columns), p)
        got_r, got_p = state.prefix(0, total, alpha=2.0)  # no p-value exceeds 2: nothing separates
        want_r, want_p = reference(design, 0, 1, level_rows(state, p))
        assert len(got_r) == total
        assert np.abs(got_r - want_r).max() < 1e-12
        assert (np.abs(got_p - want_p) <= 1e-9 * want_p + 1e-300).all()
        if p < len(columns):
            state.advance()


def test_rows_are_the_sorted_columns_of_the_subsets():
    data = mixed(200, 8, seed=1)
    columns, fixed = np.array([6, 2, 4, 3], dtype=np.intp), np.array([7, 5], dtype=np.intp)
    state = SubsetLattice.build(design_of(data), 0, 1, columns, fixed)
    assert state is not None
    state.grow(1)
    expected = [sorted([[6, 2, 4, 3][i] for i in c] + [5, 7]) for c in combinations(range(4), 2)]
    assert level_rows(state, 2).tolist() == expected
    assert [state.subset(k) for k in range(6)] == list(combinations(range(4), 2))


def test_the_screen_gives_the_tests_own_values_where_the_search_reads_them():
    """The first subset above alpha, the smallest absolute statistic, and the largest p-value are
    the inversion's, in bits, whatever the lattice's last digits."""
    unequal = 0
    for seed in range(12):
        data = mixed(300, 10, seed=seed)
        design = design_of(data)
        state = SubsetLattice.build(design, 0, 1, np.arange(2, 10, dtype=np.intp), NO_FIXED)
        assert state is not None
        state.grow(3)
        total = math.comb(8, 4)
        want_r, want_p = reference(design, 0, 1, level_rows(state, 4))
        raw_r = state._extend(0, total, store=False)
        raw_p = np.asarray(state.test._pvalue(raw_r, state.test._degrees_of_freedom(4)))
        unequal += int((raw_p != want_p).any())
        # alpha at the largest p-value: no subset is above it, and the largest sits on the edge.
        r, p = state.prefix(0, total, alpha=float(want_p.max()))
        assert len(p) == total
        assert p.max() == want_p.max() and int(np.argmax(p)) == int(np.argmax(want_p))
        assert np.abs(r).min() == np.abs(want_r).min()
        assert r[np.argmax(p)] == want_r[np.argmax(want_p)]
        # alpha at the p-value of one subset: the first above it is the next larger one.
        alpha = float(np.sort(want_p)[total // 2])
        r, p = state.prefix(0, total, alpha=alpha)
        first = int(np.flatnonzero(want_p > alpha)[0])
        assert len(p) == first + 1 and p[-1] == want_p[first] and r[-1] == want_r[first]
    assert unequal  # the lattice alone differs from the inversion in bits, so the screen matters


def test_a_search_on_the_lattice_equals_the_batched_search(monkeypatch):
    """Four mediators of a link separate it only together: the set found, the smallest statistic,
    the largest p-value, and the tested links equal the batched path's, in bits."""
    rng = np.random.default_rng(0)
    n = 3000
    cause = rng.standard_normal(n)
    mediators = cause[:, None] + rng.standard_normal((n, 4))
    effect = mediators.sum(axis=1) + rng.standard_normal(n)
    others = rng.standard_normal((n, 4)) + 0.3 * effect[:, None]
    data = np.column_stack([cause, effect, mediators, others])
    results = []
    builds = counting_builds(monkeypatch)
    for minimum in (1, 10**9):  # the lattice, then the batched path
        monkeypatch.setattr(lattice, "MIN_SUBSETS", minimum)
        search = engine.LPCMCISearch(design_of(data), 10, 0, ALPHA, list(range(10)))
        order = [(v, 0) for v in (6, 2, 7, 3, 8, 4, 9, 5)]
        found = [search.separating_subsets((0, 0), (1, 0), order, set(), p) for p in (4, 5)]
        results.append((found, search.imin, search.best, search.tested))
    assert len(builds) == 1  # only the first arm ran on the lattice
    assert results[0] == results[1]
    assert results[0][0][0] == frozenset({(2, 0), (3, 0), (4, 0), (5, 0)})


def test_a_state_resumes_between_rounds_and_a_new_order_builds_another(monkeypatch):
    monkeypatch.setattr(lattice, "MIN_SUBSETS", 1)
    data = mixed(500, 10, seed=5)
    search = engine.LPCMCISearch(design_of(data), 10, 0, 2.0, list(range(10)))  # nothing separates
    order = [(v, 0) for v in range(2, 10)]
    for p in (4, 5):
        assert search.separating_subsets((0, 0), (1, 0), order, set(), p) is None
    (state,) = search.lattices.values()
    assert state.level == 5
    search.separating_subsets((0, 0), (1, 0), order, set(), 6)
    assert list(search.lattices.values()) == [state] and state.level == 6  # resumed, not rebuilt
    # A new order is a state of its own, built where the levels below are cheap to rebuild.
    search.separating_subsets((0, 0), (1, 0), order[::-1], set(), 4)
    assert len(search.lattices) == 2
    assert search.separating_subsets((0, 0), (1, 0), order[::-1], set(), 6) is None  # batched
    assert len(search.lattices) == 1


@pytest.mark.parametrize("fallback", ["collinear", "bytes", "small", "deletion"])
def test_a_search_the_lattice_does_not_take_runs_the_batched_path(monkeypatch, fallback):
    """A state that fails the certificate, would hold too many bytes, is too small, or reads
    missing values answers as the batched path does, and no lattice is kept."""
    data = mixed(400, 12, seed=2)
    design = design_of(data)
    minimum = 1
    if fallback == "collinear":
        data = data.copy()
        data[:, 11] = data[:, 2] + data[:, 3]  # a column in the candidates and the fixed set
        design = design_of(data)
    elif fallback == "bytes":
        monkeypatch.setattr(lattice, "MAX_BYTES", 1000)
    elif fallback == "small":
        minimum = 10**9
    else:
        data = data.copy()
        data[5, 4] = np.nan
        design = design_test("fisherz", data, 0)  # type: ignore[assignment]
    outcomes = []
    for floor in (minimum, 10**9):
        monkeypatch.setattr(lattice, "MIN_SUBSETS", floor)
        search = engine.LPCMCISearch(design, 12, 0, ALPHA, list(range(12)))
        order = [(v, 0) for v in range(2, 11)]
        found = search.separating_subsets((0, 0), (1, 0), order, {(11, 0)}, 4)
        assert not search.lattices
        outcomes.append((found, search.imin, search.best))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(("n", "seed"), [(800, 0), (600, 3)])
def test_a_fit_on_the_lattice_equals_the_batched_fit(monkeypatch, n, seed):
    """A seeded series whose searches run on the lattice (small searches included) returns the
    batched fit's result: the same links, p-values, and statistics, in bits."""
    data = sample_scm(functional="var", d=8, n=n, seed=seed, density=2).data
    builds = counting_builds(monkeypatch)
    fits = []
    for floor in (1, 10**9):
        monkeypatch.setattr(lattice, "MIN_SUBSETS", floor)
        fits.append(andrey.lpcmci(data, tau_max=2))
    assert builds  # the first fit read some level from the lattice
    first, second = fits
    assert same(first.metadata, second.metadata)
    assert same(first.structure, second.structure)


def test_a_worker_of_a_pool_holds_its_share_of_the_bytes(monkeypatch):
    """The cap is divided by the workers of the pool: a search that fits the cap alone does not
    fit a quarter of it, and then answers as the batched path does."""
    monkeypatch.setattr(lattice, "MIN_SUBSETS", 1)
    data = mixed(500, 10, seed=5)
    order = [(v, 0) for v in range(2, 10)]
    need = lattice.live_bytes(len(order), 0, 4)
    monkeypatch.setattr(lattice, "MAX_BYTES", 2 * need)
    outcomes = []
    for workers in (1, 2, 4):
        monkeypatch.setattr(backend, "_pool_workers", workers)
        search = engine.LPCMCISearch(design_of(data), 10, 0, 2.0, list(range(10)))
        found = search.separating_subsets((0, 0), (1, 0), order, set(), 4)
        outcomes.append((bool(search.lattices), found, search.imin, search.best, search.tested))
    assert [kept for kept, *_ in outcomes] == [True, True, False]
    assert outcomes[0][1:] == outcomes[1][1:] == outcomes[2][1:]


@pytest.mark.parametrize("cap", [1000, 50_000, lattice.MAX_BYTES])
def test_a_fit_gives_the_same_bits_whatever_the_cap(monkeypatch, cap):
    """The cap only chooses between the lattice and the inversion for each search, so a fit under a
    tiny, a middle, and the full cap returns the batched fit's result."""
    data = sample_scm(functional="var", d=8, n=800, seed=0, density=2).data
    monkeypatch.setattr(lattice, "MIN_SUBSETS", 1)
    monkeypatch.setattr(lattice, "MAX_BYTES", cap)
    capped = andrey.lpcmci(data, tau_max=2)
    monkeypatch.setattr(lattice, "MIN_SUBSETS", 10**9)
    batched = andrey.lpcmci(data, tau_max=2)
    assert same(capped.metadata, batched.metadata)
    assert same(capped.structure, batched.structure)
