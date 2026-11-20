"""Regime-PCMCI: the regime linear program, the per-regime PCMCI rows, the result type, and the
facade's contract."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import RegimeOutput, StructureOutput
from andrey.core.warning_policy import SearchLimitWarning
from andrey.temporal.pcmci import pcmci
from andrey.temporal.rpcmci import FEWEST_STEPS, regime_costs, regime_lp
from tests.recovery.helpers import datagen

CASE = "regime_var_2v"
FAST = {"max_transitions": 7, "n_starts": 4}


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    return datagen.generate(CASE, shuffle=False)


@pytest.fixture(scope="module")
def fitted(series) -> RegimeOutput:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return andrey.rpcmci(series, seed=0, **FAST)


# --- the regime linear program ---------------------------------------------------------------


def test_lp_assigns_each_step_to_its_cheaper_regime_within_the_budget():
    """Regime 0 is cheaper on the first 6 steps and regime 1 on the last 4: one switch each."""
    costs = np.array([[0.0] * 6 + [2.0] * 4, [1.0] * 6 + [0.0] * 4])
    weights, objective = regime_lp(costs, max_transitions=1)
    np.testing.assert_array_equal(weights, [[1] * 6 + [0] * 4, [0] * 6 + [1] * 4])
    assert objective == pytest.approx(0.0)


def test_lp_budget_splits_a_short_stretch_between_the_regimes():
    """A 2-step stretch inside a run of regime 0 is cheaper in regime 1. Two switches move it
    there whole; with one, the optimum of the linear program weighs it half in each regime, which
    spends the budget on two half switches."""
    costs = np.array([[0.0] * 4 + [1.0] * 2 + [0.0] * 4, [0.5] * 10])
    weights, objective = regime_lp(costs, max_transitions=2)
    np.testing.assert_array_equal(weights[1], [0] * 4 + [1] * 2 + [0] * 4)
    assert objective == pytest.approx(1.0)
    weights, objective = regime_lp(costs, max_transitions=1)
    np.testing.assert_array_equal(weights[1], [0] * 4 + [0.5] * 2 + [0] * 4)
    assert objective == pytest.approx(1.5)


def test_lp_weights_sum_to_one_at_every_step(series):
    rng = np.random.default_rng(0)
    weights, _ = regime_lp(rng.uniform(size=(3, 50)), max_transitions=3)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    assert (np.abs(np.diff(weights, axis=1)).sum(axis=1) <= 3 + 1e-9).all()


def test_costs_are_standardized_residuals_with_zero_for_the_first_steps():
    """No parents: each variable's cost is its squared z-score over the regime's steps."""
    rng = np.random.default_rng(1)
    data = rng.standard_normal((40, 2))
    selected = np.zeros(40, dtype=bool)
    selected[10:30] = True
    cost = regime_costs(data, ((), ()), selected, 2, np.array([0, 1]))
    fit = data[2:][selected[2:]]
    z = (data[2:] - fit.mean(axis=0)) / fit.std(axis=0)
    np.testing.assert_allclose(cost[2:], (z**2).sum(axis=1))
    np.testing.assert_array_equal(cost[:2], 0.0)


# --- the per-regime PCMCI rows ---------------------------------------------------------------


def test_pcmci_on_every_row_equals_pcmci(series):
    full = pcmci(series, tau_max=2, alpha_level=0.01)
    every = pcmci(series, tau_max=2, alpha_level=0.01, rows=np.ones(len(series) - 4, bool))
    np.testing.assert_array_equal(full.p_values, every.p_values)
    assert full.parents == every.parents


def test_pcmci_rows_read_the_lagged_values_of_unselected_steps(series):
    """Keeping every other step tests the same links over half the rows: the lagged values come
    from the full series, so the effect X0 -> X1 of the selected steps is still found."""
    rows = np.zeros(len(series) - 2, dtype=bool)
    rows[::2] = True
    regime = datagen.regime_path(CASE)[2:]
    result = pcmci(series, tau_max=1, alpha_level=0.01, rows=rows & (regime == 0))
    assert result.graph[0, 0, 1]
    assert result.statistics[0, 0, 1] > 0.5


# --- the result type -------------------------------------------------------------------------


def test_regime_output_rejects_weights_that_do_not_fit(fitted):
    with pytest.raises(ValueError, match="rows"):
        RegimeOutput.new(fitted.outputs, np.ones((3, 5)) / 3)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RegimeOutput.new(fitted.outputs[:1], np.full((1, 5), 2.0))


def test_regimes_take_the_first_of_equal_weights(fitted):
    out = RegimeOutput.new(fitted.outputs, np.array([[0.5, 0.2, 0.5 + 1e-12], [0.5, 0.8, 0.5]]))
    np.testing.assert_array_equal(out.regimes, [0, 1, 0])


def test_result_is_read_only(fitted):
    weights = fitted.weights
    weights[:] = 0
    assert fitted.weights.sum() > 0


# --- the facade ------------------------------------------------------------------------------


def test_output_holds_a_temporal_result_per_regime(fitted, series):
    assert fitted.n_regimes == 2
    assert fitted.regimes.shape == (len(series),)
    assert set(np.unique(fitted.regimes)) <= {0, 1}
    for out in fitted.outputs:
        assert isinstance(out, StructureOutput)
        assert out.structure.type == "temporal"
        assert out.structure.lags == (1,)
        assert out.metadata["algorithm"] == "rpcmci"
    md = fitted.metadata
    assert md["algorithm"] == "rpcmci" and md["seed"] == 0 and md["indep_test"] == "student_t"
    assert {"objective", "start", "n_iterations", "converged", "failed_starts"} <= set(md)
    assert "rpcmci  |  2 regimes over 600 time steps" in repr(fitted)


def test_fit_repeats(fitted, series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = andrey.rpcmci(series, seed=0, **FAST)
    np.testing.assert_array_equal(again.weights, fitted.weights)
    for a, b in zip(again.outputs, fitted.outputs, strict=True):
        assert a == b


def test_column_order_relabels_the_result(fitted, series):
    """Reversed columns give the same regime weights to the bit and the graphs relabeled."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swapped = andrey.rpcmci(series[:, ::-1], seed=0, **FAST)
    np.testing.assert_array_equal(swapped.weights, fitted.weights)
    for a, b in zip(swapped.outputs, fitted.outputs, strict=True):
        np.testing.assert_array_equal(
            a.structure.lag(1).to_numpy(), b.structure.lag(1).to_numpy()[::-1, ::-1]
        )


def test_column_units_change_nothing(fitted, series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaled = andrey.rpcmci(series * np.array([1e3, -2e-3]), seed=0, **FAST)
    np.testing.assert_array_equal(scaled.regimes, fitted.regimes)
    for a, b in zip(scaled.outputs, fitted.outputs, strict=True):
        assert a.structure == b.structure


def test_dataframe_labels_name_every_regime(series):
    frame = pd.DataFrame(series, columns=["enso", "rain"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = andrey.rpcmci(frame, seed=0, **FAST)
    assert all(o.structure.labels == ("enso", "rain") for o in out.outputs)


def test_stopping_on_the_cap_warns(series):
    with pytest.warns(SearchLimitWarning, match="max_iter=1"):
        out = andrey.rpcmci(series, seed=0, max_iter=1, **FAST)
    assert out.metadata["converged"] is False


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda x: np.vstack([x[:5]]), "at least 9 rows"),
        (lambda x: np.where(np.arange(len(x))[:, None] == 3, np.nan, x), "rpcmci"),
        (lambda x: np.column_stack([x, 2 * x[:, 0] + 1]), "repeat other columns"),
    ],
    ids=["short", "missing", "copies"],
)
def test_unusable_data_raises(series, change, match):
    with pytest.raises(ValueError, match=match):
        andrey.rpcmci(change(series), **FAST)


def test_every_start_failing_raises():
    """Ten regimes of a 30-step series leave a regime with fewer than six steps in every start."""
    rng = np.random.default_rng(0)
    assert FEWEST_STEPS == 6
    with pytest.raises(ValueError, match="every one of the 2 starts"):
        andrey.rpcmci(rng.standard_normal((30, 2)), n_regimes=10, n_starts=2)


def test_bootstrap_refuses_regimes(series):
    with pytest.raises(NotImplementedError, match="rpcmci"):
        andrey.bootstrap("rpcmci", series, n_resamples=2)
