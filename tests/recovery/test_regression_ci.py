"""PC with the regression-based tests GCM, wGCM, and PCM: ground truth on planted nonlinear graphs,
the case of independent columns, and graphs that depend on neither the column order nor the run.

The ground truth runs each test at its default regression, the setting its paper claims: GCM and
wGCM on an additive-noise graph, PCM on a nonlinear chain. The other checks reach the same code
paths with least-squares regressions, which keeps them fast.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.core import ci
from andrey.core.ci_regression import GCM, PCM, WGCM
from andrey.metrics import shd, to_cpdag

#: The tests with least-squares regressions, under the names the fixture registers.
LINEAR = {
    "gcm_linear": partial(GCM, regression="linear"),
    "wgcm_linear": partial(WGCM, regression="linear"),
    "wgcm_fix_linear": partial(WGCM, regression="linear", variant="fix"),
    "pcm_linear": partial(PCM, regression="linear"),
}


@pytest.fixture
def linear_test(request):
    """The name PC selects a least-squares variant by, registered for the test's duration."""
    snapshot = dict(ci._CI_REGISTRY)
    ci.register_indep_test(request.param, LINEAR[request.param])
    try:
        yield request.param
    finally:
        ci._CI_REGISTRY.clear()
        ci._CI_REGISTRY.update(snapshot)


def _additive_noise_case(n: int, seed: int) -> tuple[np.ndarray, andrey.GraphStructure]:
    """Additive nonlinear noise on a 4-node Erdos-Renyi DAG, and its true CPDAG."""
    ds = data.sample_scm(
        graph="erdos_renyi",
        functional="additive_noise",
        d=4,
        n=n,
        seed=seed,
        density=2.0,
        scale="standardize",
    )
    return ds.data, to_cpdag(ds.graph)


# ---- ground truth ------------------------------------------------------------------------------


# wGCM's fitted weight misses this draw at 200 rows (SHD 3) and recovers it at 400.
@pytest.mark.parametrize(("indep_test", "n"), [("gcm", 200), ("wgcm", 400)])
def test_pc_recovers_the_additive_noise_graph(indep_test, n):
    x, truth = _additive_noise_case(n, 0)
    assert shd(andrey.pc(x).structure, truth) >= 2  # partial correlation errs here
    assert shd(andrey.pc(x, indep_test=indep_test).structure, truth) <= 1


def test_pc_with_pcm_recovers_a_nonlinear_chain():
    """``x0 -> x1 -> x2`` with smooth nonlinear links: the conditional means change along the chain
    in both directions, so the test finds each link whichever variable it takes as the response."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(300)
    x1 = x0 + np.sin(2 * x0) + 0.5 * rng.standard_normal(300)
    x2 = np.tanh(x1) + 0.3 * rng.standard_normal(300)
    out = andrey.pc(np.column_stack([x0, x1, x2]), indep_test="pcm")
    assert out.structure.oriented_edges() == [(0, 1, "undirected"), (1, 2, "undirected")]


# ---- fixed cases -------------------------------------------------------------------------------


@pytest.mark.parametrize("linear_test", LINEAR, indirect=True)
def test_pc_finds_no_edges_between_independent_columns(linear_test):
    x = np.random.default_rng(0).standard_normal((300, 4))
    assert andrey.pc(x, indep_test=linear_test).structure.oriented_edges() == []


# ---- what the tests promise ----------------------------------------------------------------------


@pytest.mark.parametrize("linear_test", LINEAR, indirect=True)
def test_pc_graph_depends_on_neither_the_column_nor_the_row_order(linear_test):
    x, _ = _additive_noise_case(200, 0)
    cols = [2, 0, 3, 1]
    rows = np.random.default_rng(1).permutation(len(x))
    base = andrey.pc(x, indep_test=linear_test).structure.to_numpy()
    shuffled = andrey.pc(x[rows][:, cols], indep_test=linear_test).structure.to_numpy()
    np.testing.assert_array_equal(shuffled, base[np.ix_(cols, cols)])
    again = andrey.pc(x, indep_test=linear_test).structure.to_numpy()
    np.testing.assert_array_equal(again, base)
