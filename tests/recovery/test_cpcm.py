"""CPCM recovers the direction of pairs and small graphs drawn from each family's own model.

Each pair has a Gaussian cause and an effect drawn from one family whose parameters are smooth
functions of the cause; the direction is drawn per seed, so a fixed answer scores one in two. The
cause's column takes a Gaussian family with as many parameters as the effect's, as the paper
advises. Twelve pairs per family with at least ten right put the 95% Wilson interval of the
accuracy above one half.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api.fcm import cpcm_graph
from andrey.api.irregular import cpcm
from andrey.core import GraphStructure
from andrey.metrics import shd
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output


def _effect(family: str, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw of the effect given the cause ``x`` under ``family``."""
    wave = np.sin(2 * x) + 0.5 * x
    if family == "gaussian":
        return wave + (0.3 + 0.25 * np.tanh(x)) * rng.standard_normal(x.size)
    if family == "gaussian_fixed_scale":
        return wave + 0.3 * rng.standard_normal(x.size)
    if family in ("gumbel", "gumbel_fixed_scale"):
        scale = 0.3 + 0.25 * np.tanh(x) if family == "gumbel" else 0.3
        return rng.gumbel(wave, scale)
    mean = np.exp(0.5 + np.sin(1.5 * x))
    if family == "exponential":
        return rng.exponential(mean)
    if family == "pareto":  # shape theta(x) = 1 + mean, scale 1
        return (1.0 - rng.random(x.size)) ** (-1.0 / (1.0 + mean))
    if family in ("gamma", "gamma_fixed_shape"):
        shape = np.exp(1.0 + 0.8 * np.tanh(x)) if family == "gamma" else 3.0
        return rng.gamma(shape, mean / shape)
    if family == "poisson":
        return rng.poisson(2 * mean).astype(float)
    assert family == "negative_binomial"
    return rng.negative_binomial(3, 3 / (3 + 2 * mean)).astype(float)


TWO_PARAMETERS = ("gaussian", "gamma", "gumbel", "negative_binomial")
FAMILIES = (
    "gaussian",
    "gaussian_fixed_scale",
    "exponential",
    "pareto",
    "poisson",
    "gamma",
    "gamma_fixed_shape",
    "gumbel",
    "gumbel_fixed_scale",
    "negative_binomial",
)


@pytest.mark.parametrize("family", FAMILIES)
def test_cpcm_names_the_direction_of_pairs_from_each_family(family):
    cause_family = "gaussian" if family in TWO_PARAMETERS else "gaussian_fixed_scale"
    right = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.4, 400)
        table, families = np.column_stack([x, _effect(family, x, rng)]), [cause_family, family]
        truth = [(0, 1, "directed")]
        if seed % 2:
            table, families, truth = table[:, ::-1], families[::-1], [(1, 0, "directed")]
        right += cpcm(table, family=families).structure.oriented_edges() == truth
    assert right >= 10


def test_the_negative_binomial_family_reads_poisson_counts_as_its_limit():
    # Poisson counts are the negative binomial's limit of infinite size, where its likelihood is
    # flat: the direction holds and the forward residual stays independent of the cause.
    right = rejected = 0
    for seed in range(100, 112):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=300)
        table = np.column_stack([x, rng.poisson(np.exp(1 + np.sin(2 * x))).astype(float)])
        out = cpcm(table, family=["gaussian_fixed_scale", "negative_binomial"], seed=0)
        right += out.structure.oriented_edges() == [(0, 1, "directed")]
        rejected += out.metadata["p_values"]["forward"] < 0.01
    assert right >= 11
    assert rejected <= 1


def test_cpcm_matches_the_recorded_output():
    assert_recorded_output(cpcm(datagen.generate("lsnm_pair")), "CPCM", case="lsnm_pair")


def test_cpcm_graph_matches_the_recorded_output():
    out = cpcm_graph(datagen.generate("nonmonotone_4v"))
    assert_recorded_output(out, "CPCMGraph", case="nonmonotone_4v")


def _exponential_graph(seed: int) -> tuple[np.ndarray, GraphStructure, list[str]]:
    """Section 6.4's exponential model on 5 variables: Gaussian roots, each child exponential
    with rate ``max(0.1, 1 / sum |parents|)``, on a random DAG of about 5 edges."""
    rng = np.random.default_rng(seed)
    d, n = 5, 600
    order = rng.permutation(d)
    adjacency = np.zeros((d, d), dtype=np.int8)
    for a in range(d):
        for b in range(a + 1, d):
            adjacency[order[a], order[b]] = rng.random() < 2 / (d - 1)
    X = np.zeros((n, d))
    for v in order:
        parents = np.flatnonzero(adjacency[:, v])
        if parents.size:
            rate = np.maximum(0.1, 1.0 / np.abs(X[:, parents]).sum(axis=1))
            X[:, v] = rng.exponential(1.0 / rate)
        else:
            X[:, v] = rng.standard_normal(n)
    families = [
        "exponential" if adjacency[:, v].any() else "gaussian_fixed_scale" for v in range(d)
    ]
    marks = np.zeros((d, d), dtype=np.int8)
    for a, b in zip(*np.nonzero(adjacency), strict=True):
        marks[a, b], marks[b, a] = 1, 2
    return X, GraphStructure.from_numpy(marks, kind="dag"), families


def test_cpcm_graph_recovers_exponential_graphs_better_than_the_empty_graph():
    errors = edges = 0
    for seed in range(3):
        X, truth, families = _exponential_graph(seed)
        errors += shd(cpcm_graph(X, family=families).structure, truth)
        edges += len(truth.oriented_edges())
    assert errors <= edges / 2
