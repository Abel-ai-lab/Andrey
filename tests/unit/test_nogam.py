"""NoGAM's contract: its regressions, knowledge, units, and the data it refuses."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

import andrey
from andrey.core import _symmetric
from andrey.core.ties import tie_ranks
from andrey.fcm import nogam as core
from andrey.fcm.stein import score, squared_distances

N = 400


def _fit(X, **kwargs) -> andrey.StructureOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.nogam(X, **kwargs)


def _edges(out: andrey.StructureOutput) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j, _ in out.structure.oriented_edges()}


def _named(out: andrey.StructureOutput) -> set[tuple[str, str]]:
    return {(str(i), str(j)) for i, j, _ in out.structure.oriented_edges()}


def _chain(seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with ``x1 = sin(2 x0)`` and ``x2 = x1**2``, each plus uniform noise."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-2, 2, N)
    x1 = np.sin(2 * x0) + rng.uniform(-0.5, 0.5, N)
    return np.column_stack([x0, x1, x1**2 + rng.uniform(-0.5, 0.5, N)])


@pytest.fixture(scope="module")
def chain_fit():
    return _fit(_chain())


def test_the_chain_is_found(chain_fit):
    assert _edges(chain_fit) == {(0, 1), (1, 2)}
    assert chain_fit.ordering == (0, 1, 2)
    with andrey.config(tie_break="position"):
        assert _edges(_fit(_chain())) == {(0, 1), (1, 2)}


def test_a_constant_kernel_predicts_the_shrunk_training_sum():
    """With the kernel 1 between every pair of rows, the ridge solution has a closed form: each
    row is predicted by the sum of the other folds' responses over their count plus the ridge
    term."""
    y = np.arange(12.0)
    predicted = core.cross_predict(np.ones((12, 12)), y)
    folds = np.array_split(np.arange(12), core.FOLDS)
    assert [len(fold) for fold in folds] == [3, 3, 2, 2, 2]
    for fold in folds:
        expected = (y.sum() - y[fold].sum()) / (12 - len(fold) + core.RIDGE)
        np.testing.assert_allclose(predicted[fold], expected, rtol=1e-10)


def test_each_row_is_predicted_without_its_own_fold():
    """A response that is noise has nothing to predict: the predictions made without a row's fold
    do not follow it, where a fit on every row would."""
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((2, 200))
    kernel = np.exp(-core.GAMMA * (x[:, None] - x[None, :]) ** 2)
    assert abs(np.corrcoef(core.cross_predict(kernel, y), y)[0, 1]) < 0.2


def test_the_residual_of_an_effect_is_its_noise_and_a_cause_keeps_its_spread():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N)
    noise = 0.5 * rng.standard_normal(N)
    unit = np.column_stack([x, np.sin(2 * x) + noise])
    unit = (unit - unit.mean(axis=0)) / unit.std(axis=0)
    residual = core.residuals(unit)
    assert np.corrcoef(residual[:, 1], noise)[0, 1] > 0.9
    errors = core.prediction_errors(unit)
    assert errors[1] < errors[0]


def test_alpha_sets_which_parents_stay_and_not_the_order(chain_fit):
    loose = _fit(_chain(), alpha=0.999999)
    assert _edges(chain_fit) < _edges(loose) == {(0, 1), (0, 2), (1, 2)}
    assert loose.ordering == chain_fit.ordering


def test_the_result_follows_the_columns_in_any_order_and_units(chain_fit):
    """Reordered, rescaled, and shifted columns give the same graph and order over the
    variables."""
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_chain(), columns=list("abc"))
    moved = frame[["c", "a", "b"]] * [1e-3, 250.0, 7.0] + [5.0, -40.0, 1e3]
    out = _fit(moved)
    assert _named(out) == {("a", "b"), ("b", "c")}
    assert [moved.columns[j] for j in out.ordering] == ["a", "b", "c"]


def test_a_required_edge_stays_on_independent_columns():
    X = np.random.default_rng(1).standard_normal((N, 3))
    out = _fit(X, knowledge=andrey.Knowledge(required=[(2, 0)]))
    assert _edges(out) == {(2, 0)}
    assert out.ordering.index(2) < out.ordering.index(0)


def test_a_forbidden_edge_is_never_returned(chain_fit):
    out = _fit(_chain(), knowledge=andrey.Knowledge(forbidden=[(0, 1)]))
    assert (0, 1) not in _edges(out)
    assert out.ordering == chain_fit.ordering


def test_tiers_order_the_variables_and_forbid_edges_into_an_earlier_tier():
    # The highest-ranked column goes first and the lowest-ranked last, against the rank's own pull.
    X = np.random.default_rng(5).standard_normal((200, 5))
    ranks = tie_ranks(X)
    first, last = int(np.argmax(ranks)), int(np.argmin(ranks))
    middle = [j for j in range(5) if j not in (first, last)]
    out = _fit(X, knowledge=andrey.Knowledge(tiers=[[first], middle, [last]]))
    assert out.ordering[0] == first
    assert out.ordering[-1] == last
    assert all(j != first and i != last for i, j in _edges(out))


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_a_binary_column_is_refused_by_name():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(_chain(), columns=["a", "flag", "c"])
    frame["flag"] = np.arange(N) % 2
    with pytest.raises(ValueError) as error:
        _fit(frame)
    assert "flag" in str(error.value)
    assert "nogam" in str(error.value)


def test_twice_as_many_rows_as_columns_fit_and_one_fewer_is_refused():
    X = np.random.default_rng(4).standard_normal((16, 8))
    assert _fit(X).structure.n_nodes == 8
    with pytest.raises(ValueError):
        _fit(X[:15])


def test_fewer_rows_than_folds_are_refused():
    X = np.random.default_rng(4).standard_normal((5, 2))
    assert _fit(X).structure.n_nodes == 2
    with pytest.raises(ValueError, match="at least 5 rows"):
        _fit(X[:4])


def test_rows_that_mostly_repeat_are_refused_by_name():
    X = np.random.default_rng(6).standard_normal((40, 2))
    X[:30] = 0.0
    with pytest.raises(ValueError, match="nogam: the median distance"):
        _fit(X)


# ---- the regressions equal their plain NumPy forms, bit for bit -------------------------------


def same_bits(a, b) -> bool:
    """Equal shape, dtype, and bytes, so NaN and the sign of zero count."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def plain_cross_predict(kernel: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``cross_predict`` with each fold's system gathered by ``np.ix_`` and every input checked."""
    n = y.shape[0]
    predicted = np.empty(n)
    held_out = np.zeros(n, dtype=bool)
    for fold in np.array_split(np.arange(n), core.FOLDS):
        held_out[:] = False
        held_out[fold] = True
        train = np.flatnonzero(~held_out)
        system = kernel[np.ix_(train, train)]
        system[np.diag_indices(train.size)] += core.RIDGE
        weights = cho_solve(cho_factor(system, lower=True, overwrite_a=True), y[train])
        predicted[fold] = kernel[np.ix_(fold, train)] @ weights
    return predicted


def plain_residuals(unit: np.ndarray) -> np.ndarray:
    """``residuals`` with each kernel built by whole-matrix ``np.exp``."""
    total = squared_distances(unit)
    out = np.empty_like(unit)
    for j in range(unit.shape[1]):
        column = unit[:, j]
        others = np.maximum(total - (column[:, None] - column[None, :]) ** 2, 0.0)
        out[:, j] = column - plain_cross_predict(np.exp(-core.GAMMA * others), column)
    return out


def plain_prediction_errors(unit: np.ndarray) -> np.ndarray:
    """``prediction_errors`` over the plain regressions."""
    estimate = score(unit, method="nogam")
    residual = plain_residuals(unit)
    errors = np.empty(unit.shape[1])
    for j in range(unit.shape[1]):
        r = residual[:, j]
        kernel = np.exp(-core.GAMMA * (r[:, None] - r[None, :]) ** 2)
        errors[j] = np.mean((estimate[:, j] - plain_cross_predict(kernel, estimate[:, j])) ** 2)
    return errors


def unit_columns(n: int, d: int, seed: int = 0) -> np.ndarray:
    """``(n, d)`` columns centered and divided by their standard deviations, nonlinearly related."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X[:, 1:] = np.sin(X[:, :-1]) + 0.5 * X[:, 1:]
    return (X - X.mean(axis=0)) / X.std(axis=0)


def layouts(n: int, d: int, seed: int = 0) -> dict[str, np.ndarray]:
    """The same ``(n, d)`` unit columns in several memory layouts."""
    wide = unit_columns(n, 2 * d, seed)
    return {
        "C": np.ascontiguousarray(wide[:, :d]),
        "F": np.asfortranarray(wide[:, :d]),
        "every other column": wide[:, ::2],
        "reversed columns": wide[:, d - 1 :: -1],
        "reversed rows": wide[::-1, :d],
    }


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("y_dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 64, 101, 1001])
def test_cross_predict_equals_the_gathered_folds(n, y_dtype, dtype):
    """Each fold's system and held-out rows are slices of the kernel, at fold sizes that differ by
    one, with no row in some folds, and with a kernel or a response of single precision."""
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    kernel = np.exp(-core.GAMMA * (x[:, None] - x[None, :]) ** 2).astype(dtype)
    y = np.sin(x + rng.standard_normal(n)).astype(y_dtype)
    before = kernel.copy()
    assert same_bits(core.cross_predict(kernel, y), plain_cross_predict(kernel.copy(), y))
    assert same_bits(kernel, before)


@pytest.mark.parametrize("order", ["C", "F"])
def test_cross_predict_reads_a_kernel_in_any_layout(order):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(97)
    kernel = np.asarray(np.exp(-core.GAMMA * (x[:, None] - x[None, :]) ** 2), order=order)
    wide = np.zeros((97, 194))
    wide[:, ::2] = kernel
    for k in (kernel, wide[:, ::2][:, :97]):
        assert same_bits(core.cross_predict(k, x), plain_cross_predict(np.array(k), x))


@pytest.mark.parametrize("n", [63, 64, 65, 129, 257])
def test_residuals_and_prediction_errors_equal_the_plain_exponentials(n):
    """At sizes around the 64-row block of the symmetric exponential, in every layout."""
    for name, unit in layouts(n, 4, seed=n).items():
        assert same_bits(core.residuals(unit), plain_residuals(unit)), name
        assert same_bits(core.prediction_errors(unit), plain_prediction_errors(unit)), name


@pytest.mark.parametrize("n", [65, 130])
def test_a_single_precision_matrix_equals_the_plain_build(n):
    unit = unit_columns(n, 3).astype(np.float32)
    plain = plain_residuals(unit)
    assert plain.dtype == np.float32
    assert same_bits(core.residuals(unit), plain)
    assert same_bits(core.prediction_errors(unit), plain_prediction_errors(unit))


@pytest.mark.parametrize("n", [255, 257, 513])
@pytest.mark.parametrize("d", [8, 40])
def test_the_exponential_of_a_strided_view_keeps_its_distances(n, d, monkeypatch):
    """The squared distances of a strided view can differ in the last bit between ``(i, j)`` and
    ``(j, i)``, so the residual kernels of such a matrix are exponentiated whole."""
    views = layouts(n, d, seed=n + d)
    for name in ("every other column", "reversed columns", "reversed rows"):
        assert same_bits(core.residuals(views[name]), plain_residuals(views[name])), name
    calls: list[int] = []
    real = _symmetric.exp_symmetric
    monkeypatch.setattr(core, "exp_symmetric", lambda a, out=None: calls.append(1) or real(a, out))
    core.residuals(views["every other column"])
    assert calls == []
    core.residuals(views["C"])
    assert len(calls) == d


def test_a_fit_equals_the_fit_over_the_plain_regressions(monkeypatch):
    """The order, the graph, and its metadata, with knowledge, column labels, and ties."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(2)
    X = np.column_stack([_chain(), np.round(rng.standard_normal(N), 1)])
    frame = pd.DataFrame(X[::-1], columns=list("wxyz"))
    knowledge = andrey.Knowledge(forbidden=[(0, 1)], required=[(3, 0)])
    fits = {}
    for label, patch in (("change", False), ("plain", True)):
        with monkeypatch.context() as patched:
            if patch:
                patched.setattr(core, "cross_predict", plain_cross_predict)
                patched.setattr(core, "residuals", plain_residuals)
                patched.setattr(core, "prediction_errors", plain_prediction_errors)
            fits[label] = (_fit(X), _fit(frame, knowledge=knowledge))
    for change, plain in zip(*fits.values(), strict=True):
        assert change.ordering == plain.ordering
        assert same_bits(change.structure.adjacency, plain.structure.adjacency)
        assert change.structure.oriented_edges() == plain.structure.oriented_edges()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_data_is_refused_before_the_regressions(bad):
    X = np.random.default_rng(0).standard_normal((60, 3))
    X[3, 1] = bad
    with pytest.raises(ValueError):
        _fit(X)
    with pytest.raises(ValueError):
        core.prediction_errors(X)
