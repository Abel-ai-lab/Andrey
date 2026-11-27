"""CAM-UV's contract: knowledge, units, column order, ties, cycles, and the data it refuses."""

from __future__ import annotations

import logging
from collections import Counter
from typing import NamedTuple

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, CIRCLE
from andrey.core.independence import hsic_log_p
from andrey.core.ties import tie_ranks
from andrey.data import sample_scm
from andrey.fcm import camuv as camuv_engine
from andrey.fcm.camuv import drop_cycles

N = 600


def _edges(out: andrey.StructureOutput, kind: str = "directed") -> set[tuple]:
    return {(i, j) for i, j, k in out.structure.oriented_edges() if k == kind}


def _fork(seed: int = 0) -> np.ndarray:
    """``x0 <- h -> x1``, ``x1 -> x2``, ``x0 -> x3`` with ``h`` hidden, nonlinear effects, and
    uniform noise."""
    rng = np.random.default_rng(seed)
    h, *noise = rng.uniform(-1, 1, (5, N))
    x0 = np.sin(3 * h) + 0.3 * noise[0]
    x1 = np.cos(3 * h) + 0.3 * noise[1]
    return np.column_stack(
        [x0, x1, np.tanh(2 * x1) + 0.3 * noise[2], np.tanh(2 * x0) + 0.3 * noise[3]]
    )


@pytest.fixture(scope="module")
def fork_fit():
    return andrey.camuv(_fork())


def test_the_fork_is_found(fork_fit):
    assert _edges(fork_fit) == {(0, 3), (1, 2)}
    assert _edges(fork_fit, "circle") == {(0, 1)}


def test_a_larger_alpha_marks_more_pairs(fork_fit):
    """A pair is marked when the test finds its residuals dependent at a p-value below ``alpha``."""
    assert _edges(andrey.camuv(_fork(), alpha=0.3), "circle") > _edges(fork_fit, "circle")


def _additive(seed: int = 0, *, d: int = 6, n: int = 400, latents: int = 0) -> np.ndarray:
    case = sample_scm(
        functional="additive_noise",
        d=d,
        n=n,
        seed=seed,
        density=2,
        scale="rescale",
        latents=latents,
    )
    return np.asarray(case.data, dtype=float)


def test_a_new_row_order_changes_nothing():
    X = _additive()
    rows = np.random.default_rng(100).permutation(X.shape[0])
    assert andrey.camuv(X[rows]).structure == andrey.camuv(X).structure


def test_flipping_a_column_sign_changes_nothing():
    X = _additive()
    assert (
        andrey.camuv(X * np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])).structure
        == andrey.camuv(X).structure
    )


def test_a_power_of_two_unit_changes_nothing():
    X = np.arange(60.0)[:, None] + np.random.default_rng(0).uniform(-0.4, 0.4, (60, 4))
    assert andrey.camuv(X * np.array([4.0, 1.0, 1.0, 1.0])).structure == andrey.camuv(X).structure


def test_a_new_column_order_relabels_the_result():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_fork(), columns=list("abcd"))
    fits = [andrey.camuv(frame[list(columns)]) for columns in ("abcd", "dbca", "cadb")]
    named = [(_edges(out), _edges(out, "circle")) for out in fits]
    assert named[0] == ({("a", "d"), ("b", "c")}, {("a", "b")})
    for edges, circles in named[1:]:
        assert edges == named[0][0]
        assert {frozenset(pair) for pair in circles} == {frozenset(("a", "b"))}


def _mirrored(seed: int = 0) -> np.ndarray:
    """Linear Gaussian ``x`` and ``y`` with each row's swap also a row.

    Swapping ``x`` and ``y`` only reorders the rows, so either one as the sink of the pair has the
    same p-value up to rounding.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(N // 2)
    b = 0.8 * a + 0.6 * rng.standard_normal(N // 2)
    return np.column_stack([np.concatenate([a, b]), np.concatenate([b, a])])


def test_a_tied_sink_is_the_lower_ranked_column_in_any_column_order():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_mirrored(), columns=list("xy"))
    ranks = dict(zip("xy", tie_ranks(frame.to_numpy()).tolist(), strict=True))
    sink, cause = sorted("xy", key=ranks.get)
    for columns in (["x", "y"], ["y", "x"]):
        assert _edges(andrey.camuv(frame[columns])) == {(cause, sink)}
    with andrey.config(tie_break="position"):
        assert _edges(andrey.camuv(frame[["y", "x"]])) == {("x", "y")}


def test_every_edge_on_a_directed_cycle_goes_but_a_required_one():
    required = np.zeros((4, 4), dtype=bool)
    cycle = [{2}, {0}, {1}, {0}]  # 0 -> 1 -> 2 -> 0, and 0 -> 3 off the cycle
    parents = [set(p) for p in cycle]
    dropped = drop_cycles(parents, required)
    assert parents == [set(), set(), set(), {0}]
    assert {(int(i), int(j)) for i, j in zip(*np.nonzero(dropped))} == {(0, 1), (1, 2), (2, 0)}
    required[0, 1] = True
    parents = [set(p) for p in cycle]
    drop_cycles(parents, required)
    assert parents == [set(), {0}, set(), {0}]


def _has_directed_cycle(edges: set[tuple[int, int]], d: int) -> bool:
    reach = np.zeros((d, d), dtype=bool)
    for i, j in edges:
        reach[i, j] = True
    for k in range(d):
        reach |= np.outer(reach[:, k], reach[k, :])
    return bool(reach.diagonal().any())


def test_a_fit_whose_parents_close_cycles_drops_and_records_their_edges(caplog):
    case = sample_scm(functional="additive_noise", d=12, n=500, seed=14, density=2)
    with caplog.at_level(logging.INFO, logger="andrey.api.fcm"):
        out = andrey.camuv(np.asarray(case.data, dtype=float))
    dropped = {tuple(e) for e in out.metadata["cycle_edges"]}
    assert dropped
    assert not _has_directed_cycle(_edges(out), 12)
    assert not dropped & _edges(out)
    assert "directed cycles" in caplog.text


def test_a_pair_forbidden_both_ways_is_never_marked():
    out = andrey.camuv(_fork(), knowledge=andrey.Knowledge(forbidden=[(0, 1), (1, 0)]))
    assert _edges(out, "circle") == set()
    assert _edges(out) == {(0, 3), (1, 2)}


def _tiered(seed: int = 0) -> np.ndarray:
    """Ten variables: the fork of ``_fork`` with ``h`` hidden, and the chains x4 -> x5 -> x6 and
    x7 -> x8 -> x9, with nonlinear effects and uniform noise."""
    rng = np.random.default_rng(seed)
    X = _fork(seed)
    noise = 0.3 * rng.uniform(-1, 1, (N, 6))
    x4, x7 = rng.uniform(-1, 1, (2, N))
    x5 = np.sin(3 * x4) + noise[:, 0]
    x8 = np.tanh(3 * x7) + noise[:, 1]
    rest = [x4, x5, x5**2 + noise[:, 2], x7, x8, np.abs(x8) + noise[:, 3]]
    return np.column_stack([X, *rest])


def test_tiers_and_edges_hold_among_ten_variables():
    early, late = [0, 4, 7], [1, 2, 3, 5, 6, 8, 9]
    knowledge = andrey.Knowledge(tiers=[early, late], required=[(7, 9)], forbidden=[(5, 6)])
    out = andrey.camuv(_tiered(), knowledge=knowledge)
    edges = _edges(out)
    assert (7, 9) in edges and (5, 6) not in edges
    assert not any(i in late and j in early for i, j in edges)
    across = [(i, j) for i, j in out.metadata["latent_pairs"] if (i in early) != (j in early)]
    assert across  # x0 o-o x1, from the hidden common cause, crosses the tiers
    for i, j in across:
        first, later = (i, j) if i in early else (j, i)
        assert out.structure.endpoints(later, first)[0] == ARROW
        assert out.structure.endpoints(first, later)[0] == CIRCLE


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_binary_column_is_named():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_fork(), columns=list("abcd"))
    frame["b"] = np.arange(N) % 2
    with pytest.raises(ValueError, match="'b'"):
        andrey.camuv(frame)


def test_a_column_that_totals_others_is_named():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_fork(), columns=list("abcd"))
    frame["t"] = frame["a"] + frame["c"]
    with pytest.raises(ValueError, match="'t'"):
        andrey.camuv(frame)


def test_fewer_rows_than_twice_the_columns_are_refused():
    X = np.random.default_rng(4).standard_normal((15, 8))
    with pytest.raises(ValueError, match="16"):
        andrey.camuv(X)
    assert andrey.camuv(X[:, :7]).structure.n_nodes == 7


def test_a_balanced_grid_of_independent_columns_fits():
    """Each value of one column meets each of the other once, so the HSIC statistic is 0 in exact
    arithmetic and slightly negative after rounding."""
    X = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    out = andrey.camuv(X)
    assert out.structure.n_nodes == 2
    assert out.structure.oriented_edges() == []


class _Refitting(camuv_engine._Tests):
    """The reference: a sink test's fresh residual is fitted for that test and not kept."""

    def log_p_fresh(self, child, regressors, y):
        x = ((child, tuple(sorted(regressors))),)
        key = (min(x, y), max(x, y))
        found = self._log_p.get(key)
        if found is None:
            residual = self._residuals.get(x[0])
            fresh = self._gram([self._fit(x[0]) if residual is None else residual])
            grams = {x: fresh, y: self._side_gram(y)}
            found = self._log_p[key] = hsic_log_p(*(grams[side] for side in key), "camuv")
        return found


class _Fit(NamedTuple):
    result: camuv_engine.CAMUVResult
    #: Every test's key and log p-value as float hex, in the order the tests ran.
    log_p: list
    #: The number of times each (variable, regressors) pair was fitted.
    fitted: Counter


def _engine_fit(X: np.ndarray, tests, *, gram_bytes: int | None = None, **options) -> _Fit:
    """Fit the engine with ``tests`` as its test class, and record what its tests computed."""
    made = []
    fitted: Counter = Counter()

    class Recording(tests):
        def __init__(self, data):
            super().__init__(data)
            made.append(self)

        def _fit(self, term):
            fitted[term] += 1
            return super()._fit(term)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(camuv_engine, "_Tests", Recording)
        if gram_bytes is not None:
            patch.setattr(camuv_engine, "_GRAM_BYTES", gram_bytes)
        result = camuv_engine.camuv(X, **options)
    (engine,) = made
    return _Fit(result, [(k, float(v).hex()) for k, v in engine._log_p.items()], fitted)


def _marks(d: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((d, d), dtype=bool)
    for i, j in pairs:
        out[i, j] = True
    return out


def _same_fit(kept: _Fit, reference: _Fit) -> None:
    assert kept.log_p == reference.log_p
    for name in ("edges", "pairs", "cycle_edges"):
        assert np.array_equal(getattr(kept.result, name), getattr(reference.result, name)), name


@pytest.mark.parametrize(
    ("seed", "latents", "options"),
    [
        (0, 0, {}),
        (1, 0, {}),
        (2, 0, {"alpha": 0.05}),
        (0, 0, {"max_depth": 1}),
        (0, 2, {}),
        (1, 0, {"required": _marks(8, [(0, 1)]), "forbidden": _marks(8, [(3, 2), (5, 4)])}),
    ],
)
def test_a_kept_fresh_residual_changes_no_bit(seed, latents, options):
    """The tests' log p-values, in the order they ran, and the fit's marks equal the reference's,
    which fits a fresh residual again for each test."""
    X = _additive(seed, d=8, n=300, latents=latents)
    kept = _engine_fit(X, camuv_engine._Tests, **options)
    _same_fit(kept, _engine_fit(X, _Refitting, **options))


@pytest.mark.parametrize("grams", [0, 1, 2])
def test_each_residual_is_fitted_once(grams):
    """A residual is fitted once however few kernel matrices the fit keeps, and the same pairs are
    fitted as in the reference, which repeats some."""
    X = _additive(0, d=8, n=300)
    gram_bytes = grams * 8 * X.shape[0] ** 2
    kept = _engine_fit(X, camuv_engine._Tests, gram_bytes=gram_bytes)
    reference = _engine_fit(X, _Refitting, gram_bytes=gram_bytes)
    assert max(reference.fitted.values()) > 1
    assert max(kept.fitted.values()) == 1
    assert set(kept.fitted) == set(reference.fitted)
    _same_fit(kept, reference)
