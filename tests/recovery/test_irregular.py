from __future__ import annotations

import functools

import numpy as np
import pytest

import andrey
from andrey.irregular.anm import anm
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

# The columns of ``anm_pair_cubic``'s cause and effect.
_CAUSE, _EFFECT = (int(column) for column in datagen.column_of("anm_pair_cubic"))
_PLANTED = [(_CAUSE, _EFFECT, "directed")]


def test_pnl_matches_the_recorded_output():
    out = andrey.pnl(datagen.generate("anm_pair_cubic"))
    assert_recorded_output(out, "PNL", case="anm_pair_cubic")


@pytest.mark.parametrize("method", ["anm", "pnl"])
def test_equal_p_values_are_broken_by_the_statistic(method, monkeypatch):
    # Both coordinates of a noiseless circle are functions of its angle, so neither direction fits.
    # Both tests' p-values are set to one value below the smallest double here, so they read zero
    # and tie, and the smaller HSIC statistic over its null mean decides.
    import importlib

    from andrey.api import irregular

    module = importlib.import_module(f"andrey.irregular.{method}")
    name = "hsic_statistic_log_p" if method == "anm" else "_gamma_test"
    test = getattr(module, name)
    monkeypatch.setattr(module, name, lambda *args: (test(*args)[0], -1000.0))
    angle = np.pi * np.random.default_rng(0).uniform(-1, 1, 200)
    out = getattr(irregular, method)(np.column_stack([np.cos(angle), np.sin(angle)]))
    p, stat = out.metadata["p_values"], out.metadata["statistics"]
    assert p["forward"] == p["backward"] == 0.0
    assert stat["forward"] != stat["backward"]
    edge = (0, 1) if stat["forward"] < stat["backward"] else (1, 0)
    assert out.structure.oriented_edges() == [(*edge, "directed")]


def test_the_larger_p_value_decides_far_below_1e_16():
    # Neither direction fits a noiseless circle, and both p-values lie far below 1e-16, where they
    # keep their digits: the larger one orients the edge, against the order of the statistics.
    from andrey.api.irregular import pnl as pnl_facade

    angle = np.pi * np.random.default_rng(7).uniform(-1, 1, 200)
    out = pnl_facade(np.column_stack([np.cos(angle), np.sin(angle)]))
    p, stat = out.metadata["p_values"], out.metadata["statistics"]
    assert 0.0 < min(p.values()) < max(p.values()) < 1e-16
    forward = p["forward"] > p["backward"]
    assert forward == (stat["forward"] > stat["backward"])
    assert out.structure.oriented_edges() == [((0, 1) if forward else (1, 0)) + ("directed",)]


def test_pnl_draws_no_edge_when_the_statistics_tie_too():
    # Two equal columns fit the same model to the same data in both directions.
    from andrey.api.irregular import pnl as pnl_facade

    x = datagen.generate("anm_pair_cubic")[:, 0]
    out = pnl_facade(np.column_stack([x, x]))
    p, stat = out.metadata["p_values"], out.metadata["statistics"]
    assert p["forward"] == p["backward"]
    assert stat["forward"] == stat["backward"]
    assert out.structure.oriented_edges() == []


def _pairs() -> dict[str, np.ndarray]:
    """Return three sample pairs: directed, unrelated, and linear Gaussian."""
    planted = datagen.generate("anm_pair_cubic")
    rng = np.random.default_rng(0)
    n = planted.shape[0]
    unrelated = np.column_stack([rng.laplace(size=n), rng.laplace(size=n)])
    g = rng.standard_normal(n)
    gaussian = np.column_stack([g, 0.8 * g + 0.6 * rng.standard_normal(n)])
    return {"planted": planted, "unrelated": unrelated, "gaussian": gaussian}


# The measured p-values, the planted pair's from cause to effect first, are (0.226, 0.020), (0.594,
# 0.298), and (0.881, 0.922): the comparison rule always draws an edge; the threshold rule draws one
# only for the planted pair.
@pytest.mark.parametrize(
    ("pair", "alpha", "edges"),
    [
        ("planted", None, _PLANTED),
        ("planted", 0.05, _PLANTED),
        ("unrelated", None, [(0, 1, "directed")]),
        ("unrelated", 0.05, []),
        ("gaussian", None, [(1, 0, "directed")]),
        ("gaussian", 0.05, []),
    ],
)
def test_pnl_decision_rules_on_measured_pairs(pair, alpha, edges):
    from andrey.api.irregular import pnl as pnl_facade

    assert pnl_facade(_pairs()[pair], alpha=alpha).structure.oriented_edges() == edges


def test_pnl_names_the_direction_whatever_the_column_order():
    import pandas as pd

    from andrey.api.irregular import pnl as pnl_facade

    names = [("cause", "effect")[v] for v in datagen.column_order("anm_pair_cubic")]
    df = pd.DataFrame(datagen.generate("anm_pair_cubic"), columns=names)
    for columns in (["cause", "effect"], ["effect", "cause"]):
        edges = pnl_facade(df[columns]).structure.oriented_edges()
        assert edges == [("cause", "effect", "directed")]


def test_anm_matches_the_recorded_output():
    out = andrey.anm(datagen.generate("anm_pair_cubic"))
    assert_recorded_output(out, "ANM", case="anm_pair_cubic")


def test_anm_hyperparameter_objective_is_scikit_learns():
    """The search's objective is scikit-learn's log marginal likelihood, and so is its gradient."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    from andrey.irregular.anm import _kernel, _negative_log_marginal_likelihood

    x = datagen.generate("anm_pair_cubic")
    cause, effect = np.ascontiguousarray(x[:, [_CAUSE]]), np.ascontiguousarray(x[:, [_EFFECT]])
    gp = GaussianProcessRegressor(kernel=_kernel(), optimizer=None).fit(cause, effect)
    # The start, an interior point, and the signal variance at its floor (y unrelated to x).
    for theta in (gp.kernel_.theta, np.log([2.0, 0.3, 0.05]), np.log([1e-3, 1.0, 1.0])):
        value, gradient = gp.log_marginal_likelihood(theta, eval_gradient=True)
        negative, negative_gradient = _negative_log_marginal_likelihood(theta, cause, effect)
        assert -negative == value
        np.testing.assert_allclose(-negative_gradient, gradient, rtol=1e-9, atol=1e-9)


def test_anm_matches_scikit_learns_own_search():
    """The p-values of scikit-learn's default L-BFGS-B fit, up to rounding."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor

    from andrey.core.independence import hsic_test
    from andrey.irregular.anm import _kernel

    x = datagen.generate("anm_pair_cubic")
    expected = []
    for cause, effect in ((x[:, :1], x[:, 1:]), (x[:, 1:], x[:, :1])):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel=_kernel()).fit(cause, effect)
        expected.append(hsic_test(cause, effect - gp.predict(cause).reshape(-1, 1)))
    assert np.allclose(
        np.exp(anm((x[:, 0], x[:, 1]), (x[:, 1], x[:, 0]))[:2]), expected, rtol=0, atol=1e-8
    )


def test_anm_names_the_planted_direction_in_any_units_and_column_order():
    import pandas as pd

    from andrey.api.irregular import anm as anm_facade

    planted = datagen.generate("anm_pair_cubic")
    first = anm_facade(planted)
    assert first.structure.oriented_edges() == _PLANTED
    assert set(first.metadata) == {"algorithm", "p_values", "statistics"}
    pvalues = tuple(first.metadata["p_values"].values())
    for scale in (100.0, 0.01, 1e-200):  # 1e-200: squared deviations would underflow
        out = anm_facade(planted * np.array([scale, 1 / scale]))
        assert np.allclose(tuple(out.metadata["p_values"].values()), pvalues)
    names = [("cause", "effect")[v] for v in datagen.column_order("anm_pair_cubic")]
    df = pd.DataFrame(planted, columns=names)
    for columns in (["cause", "effect"], ["effect", "cause"]):
        edges = anm_facade(df[columns]).structure.oriented_edges()
        assert edges == [("cause", "effect", "directed")]


# The measured p-values, the planted pair's from cause to effect first, are (0.057, 0.0), (0.818,
# 0.478), and (0.795, 0.785): the comparison rule draws an edge between unrelated variables; the
# threshold rule draws one only for the planted pair.
@pytest.mark.parametrize(
    ("pair", "alpha", "edges"),
    [
        ("planted", 0.02, _PLANTED),
        ("unrelated", None, [(0, 1, "directed")]),
        ("unrelated", 0.05, []),
        ("gaussian", 0.05, []),
    ],
)
def test_anm_decision_rules_on_measured_pairs(pair, alpha, edges):
    from andrey.api.irregular import anm as anm_facade

    assert anm_facade(_pairs()[pair], alpha=alpha).structure.oriented_edges() == edges


# Every pairwise direction test follows one rule: each direction's fit reads its rows in one
# canonical order, so the row order changes nothing, and a table whose rows stay the same when its
# columns swap gives both directions the same numbers, which tie.
_PAIRWISE = {
    "anm": andrey.anm,
    "pnl": andrey.pnl,
    "loci": andrey.loci,
    "loci-likelihood": functools.partial(andrey.loci, criterion="likelihood"),
}


def _mirrored(table: str) -> np.ndarray:
    """A two-valued table with as many ``(0, 1)`` rows as ``(1, 0)`` rows, or a continuous pair
    stacked on its column swap."""
    if table == "binary":
        return np.repeat([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [20, 13, 13, 54], axis=0)
    pair = datagen.generate("anm_pair_cubic")[:100]
    return np.vstack([pair, pair[:, ::-1]])


@pytest.mark.parametrize("table", ["binary", "continuous"])
@pytest.mark.parametrize(
    ("method", "alpha"),
    [(m, None) for m in _PAIRWISE] + [(m, 0.05) for m in ("anm", "pnl", "loci")],
)
def test_a_mirrored_table_gives_no_edge_in_any_row_order(method, alpha, table):
    # For anm and pnl the last permutation also puts the second column in other units, which
    # leaves its standardized values different in their last bits; the tie rule absorbs that.
    # loci's solver carries such a difference up to its stopping tolerance, past the tie rule.
    X = _mirrored(table)
    fit = functools.partial(_PAIRWISE[method], alpha=alpha) if alpha else _PAIRWISE[method]
    rng = np.random.default_rng(0)
    for scale in (1.0, 1.0, 1.0, 1.0 if method.startswith("loci") else 1e3):
        out = fit(X[rng.permutation(len(X))] * np.array([1.0, scale]))
        assert out.structure.oriented_edges() == []
        if scale == 1.0:  # the same arrays in both directions: the same numbers, bit for bit
            for key in ("p_values", "statistics", "log_likelihoods"):
                if key in out.metadata:
                    assert out.metadata[key]["forward"] == out.metadata[key]["backward"]


@pytest.mark.parametrize("method", list(_PAIRWISE))
def test_a_row_permutation_gives_the_same_output(method):
    X = datagen.generate("anm_pair_cubic")
    first = _PAIRWISE[method](X)
    for seed in range(2):
        out = _PAIRWISE[method](X[np.random.default_rng(seed).permutation(len(X))])
        assert out.structure.oriented_edges() == first.structure.oriented_edges()
        assert out.metadata == first.metadata


@pytest.mark.parametrize("method", list(_PAIRWISE))
def test_the_direction_does_not_depend_on_the_units(method):
    X = datagen.generate("anm_pair_cubic")
    first = _PAIRWISE[method](X)
    for scale in ([1e-3, 7.0], [2.0**-30, 2.0**20]):
        out = _PAIRWISE[method](X * np.array(scale))
        assert out.structure.oriented_edges() == first.structure.oriented_edges()
