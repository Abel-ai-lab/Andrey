"""Regime-PCMCI recovers the regimes and each regime's lagged graph.

The ground-truth cases are toy models of Saggioro et al. (2020, Table 1): two variables, each
autocorrelated at lag 1, whose lagged link changes sign, direction, or lag between two regimes of
70 to 100 steps in turn, here over 1,000 steps (the paper's 3,000) with 10 starts (its 50).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import andrey
from andrey.api._marks import directed_marks
from andrey.core import GraphStructure, TemporalStructure
from andrey.metrics import temporal_scores

N_STEPS = 1000
TAU_MAX = 2

#: (effect, cause, lag) -> coefficient, per regime: Table 1's "sign X1X2", "arrow direction", and
#: "lag" examples.
MODELS = {
    "sign": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(1, 0, 1): -0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
    "direction": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(0, 1, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
    "lag": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(1, 0, 2): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
}


def simulate(model: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The series of ``MODELS[model]`` and the regime of each step."""
    rng = np.random.default_rng(seed)
    path: list[int] = []
    while len(path) < N_STEPS:
        path += [len(path) % 2 if not path else 1 - path[-1]] * int(rng.integers(70, 101))
    regimes = np.array(path[:N_STEPS])
    x = rng.standard_normal((N_STEPS, 2))
    for t in range(N_STEPS):
        for (j, i, lag), c in MODELS[model][regimes[t]].items():
            if t >= lag:
                x[t, j] += c * x[t - lag, i]
    return x, regimes


def truth(links: dict) -> TemporalStructure:
    """The lag graphs ``1 .. TAU_MAX`` of one regime's links."""
    graphs = []
    for lag in range(1, TAU_MAX + 1):
        effects = np.zeros((2, 2))  # [effect, cause]
        for j, i, k in links:
            effects[j, i] = k == lag
        marks = directed_marks(effects, allow_self_loops=True)
        graphs.append(GraphStructure.from_numpy(marks, kind="digraph", allow_self_loops=True))
    return TemporalStructure.from_lag_graphs(graphs, lags=range(1, TAU_MAX + 1))


def fit(x: np.ndarray, regimes: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return andrey.rpcmci(
            x,
            max_transitions=int((np.diff(regimes) != 0).sum()),
            tau_max=TAU_MAX,
            n_starts=10,
            seed=0,
        )


@pytest.mark.parametrize("model", list(MODELS))
def test_recovers_the_regimes_and_their_graphs(model):
    """The regime of at least 9 steps in 10 and, for each regime, a lag-resolved SHD of at most 1
    against the true graph (the empty graph's is 3)."""
    x, regimes = simulate(model, 1000)
    out = fit(x, regimes)
    matched = [np.array(p)[out.regimes] for p in itertools.permutations(range(2))]
    best = max(range(2), key=lambda m: (matched[m] == regimes).mean())
    accuracy = (matched[best] == regimes)[TAU_MAX:].mean()
    assert accuracy >= 0.9, accuracy
    for k, true_k in enumerate(list(itertools.permutations(range(2)))[best]):
        scores = temporal_scores(out.outputs[k], truth(MODELS[model][true_k]))
        assert scores["lagged"]["shd"] <= 1, (model, k, scores["lagged"])


def test_independent_series_have_no_link_in_any_regime():
    """Independent noise, whatever the regimes the fit draws: no regime has a lagged link."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((400, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = andrey.rpcmci(x, max_transitions=20, n_starts=10, seed=0)
    for o in out.outputs:
        assert o.structure.lag(1).oriented_edges() == []


def test_more_starts_never_raise_the_objective():
    """The returned start has the lowest objective, so ten starts include the first one's."""
    x, regimes = simulate("sign", 1001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = andrey.rpcmci(x, max_transitions=12, n_starts=1, seed=0)
    assert fit(x, regimes).metadata["objective"] <= one.metadata["objective"]
