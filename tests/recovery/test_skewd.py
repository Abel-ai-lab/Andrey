"""SkewD and SkewD-MV recover the direction of data from their own model, skew-normal
location-scale models, better than the floors of ``andrey.metrics``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest, skewnorm

from andrey import skewd, skewd_mv
from andrey.fcm.skewd import OPTIMIZERS
from andrey.metrics import decision_rate, direction_accuracy, shd

PAIRS = 12
PAIR_ROWS = 400
FIT = {"tuning_iter": 2, "max_iter": 200}


def _noise(seed: int, n: int) -> np.ndarray:
    """Strongly skewed skew-normal noise (shape 20, as the paper's skewed pair sets)."""
    return skewnorm.rvs(20, size=n, random_state=seed)


def _pair(seed: int) -> np.ndarray:
    """``y = f(x) + g(x) n`` with a Gaussian cause, a nonlinear mean, a scale that changes with
    the cause, and skew-normal noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(PAIR_ROWS)
    y = np.tanh(2 * x) + (0.5 + 0.3 * np.tanh(x)) * _noise(seed, PAIR_ROWS)
    return np.column_stack([x, y])


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_skewd_names_the_direction_of_skew_normal_location_scale_pairs(optimizer):
    # Both criteria read the same two fits of a pair, and the likelihood criterion's edge points
    # from the larger of the reported log-likelihoods (tests/unit/test_skewd.py), so each pair is
    # fitted once.
    by_test, by_likelihood = [], []
    for seed in range(PAIRS):
        out = skewd(_pair(seed), criterion="independence", seed=seed, optimizer=optimizer, **FIT)
        edges = out.structure.oriented_edges()
        by_test.append(1 if edges == [(0, 1, "directed")] else -1 if edges else 0)
        fits = out.metadata["log_likelihoods"]
        by_likelihood.append(1 if fits["forward"] > fits["backward"] else -1)
    for predicted in (by_test, by_likelihood):
        right = sum(p == 1 for p in predicted)
        interval = binomtest(right, PAIRS).proportion_ci(method="wilson")
        assert direction_accuracy(predicted, np.ones(PAIRS)) == right / PAIRS
        assert decision_rate(predicted) == 1.0  # with no alpha, every pair gets a direction
        assert interval.low > 0.5, f"{right} of {PAIRS} pairs: {interval}"


def _tree(seed: int, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """``x0 -> x1``, ``x1 -> x2``, ``x1 -> x3``: no collider, so the class is the undirected
    tree and only the likelihood can orient it."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = np.tanh(2 * x0) + (0.5 + 0.3 * np.tanh(x0)) * _noise(seed, n)
    x2 = np.sin(2 * x1) + (0.5 + 0.3 * np.tanh(x1)) * _noise(seed + 1, n)
    x3 = np.tanh(2 * x1) + (0.5 - 0.3 * np.tanh(x1)) * _noise(seed + 2, n)
    truth = np.zeros((4, 4), dtype=np.int8)
    truth[0, 1] = truth[1, 2] = truth[1, 3] = 1
    return np.column_stack([x0, x1, x2, x3]), truth


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_skewd_mv_beats_the_empty_graph_on_a_skew_normal_tree(optimizer):
    data, truth = _tree(0)
    out = skewd_mv(data, seed=0, optimizer=optimizer, **FIT)
    assert out.metadata["indep_test"] == "kci"  # PC's test is the default's, not the picked one
    dag = out.structure
    assert shd(dag, truth) < shd(np.zeros((4, 4), dtype=np.int8), truth)
    assert shd(dag, truth) == 0  # the likelihood picks the true DAG among the tree's four
