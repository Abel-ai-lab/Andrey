from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.core import ARROW, TAIL, stats
from andrey.core.ties import tie_ranks
from andrey.lingam import pwling
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output


def _random_linear_sem(seed: int, n: int, d: int, noise: str) -> np.ndarray:
    """A random strictly-lower-triangular linear SEM (uniform, or near-tie Gaussian, errors)."""
    rng = np.random.default_rng(seed)
    B = np.tril(rng.uniform(0.3, 1.2, (d, d)) * rng.choice([-1.0, 1.0], (d, d)), -1)
    e = rng.uniform(-1.0, 1.0, (n, d)) if noise == "uniform" else rng.standard_normal((n, d))
    X = np.zeros((n, d))
    for j in range(d):
        X[:, j] = e[:, j] + X @ B[j]
    return X


def _score_candidates_serial(X: np.ndarray, U: list[int]) -> np.ndarray:
    """The readable per-pair reference of :func:`pwling.score_candidates`."""

    def entropy(u: np.ndarray) -> float:
        return float(stats.entropy(u)[0])

    cols = list(U)
    x_std = {k: (X[:, k] - X[:, k].mean()) / X[:, k].std() for k in cols}
    ent = {k: entropy(x_std[k]) for k in cols}
    scores = np.empty(len(cols), dtype=np.float64)
    for a, i in enumerate(cols):
        xi, e_xi = x_std[i], ent[i]
        acc = 0.0
        for j in cols:
            if j == i:
                continue
            xj, e_xj = x_std[j], ent[j]
            r_ij = pwling.compute_residual(xi, xj)
            r_ji = pwling.compute_residual(xj, xi)
            diff = (e_xj + entropy(r_ij / np.std(r_ij))) - (e_xi + entropy(r_ji / np.std(r_ji)))
            acc += min(0.0, diff) ** 2
        scores[a] = -acc
    return scores


def _order_by_serial_scores(X: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The causal order ``find_causal_order`` picks when it scores with the per-pair reference."""
    monkeypatch.setattr(
        pwling, "pruned_scores", lambda groups, U, **_: _score_candidates_serial(groups[0], U)
    )
    return pwling.find_causal_order(X)


def _full_scores(groups: list[np.ndarray], U: list[int], weights: list[float] | None) -> np.ndarray:
    """Every candidate's :func:`pwling.score_candidates` score, weighted and summed over groups."""
    if weights is None:
        return pwling.score_candidates(groups[0], U)
    total = np.zeros(len(U))
    for X, w in zip(groups, weights, strict=True):
        total += w * pwling.score_candidates(X, U)
    return total


def _assert_pick_unchanged(pruned, groups, U, *, weights=None, allowed=None, hint=None) -> int:
    """Check ``pruned`` scores against the full scores under the given and two other visit orders.

    Every order picks the full scores' candidate, the pick carries the full score's bits, and every
    other candidate has its full score's bits or a value at or above its score whose
    ``sqrt(-value)`` lies more than ``_TIE_TOL`` above the best, out of the tie. Returns the number
    of candidates that stopped early under the given order.
    """
    full = _full_scores(groups, U, weights)
    mask = np.ones(len(U), dtype=bool) if allowed is None else allowed
    chosen = [c for c, ok in zip(U, mask, strict=True) if ok]
    ranks = tie_ranks(np.vstack(groups))
    expected = pwling.pick(full[mask], chosen, ranks)
    rng = np.random.default_rng(len(U))
    hints = [hint, {c: float(c) for c in U}, dict(zip(U, rng.standard_normal(len(U)).tolist()))]
    stopped = []
    for h in hints:
        got = pruned(groups, U, weights=weights, allowed=allowed, hint=h)
        assert pwling.pick(got[mask], chosen, ranks) == expected
        k = U.index(expected)
        assert got[k].tobytes() == full[k].tobytes()
        if np.isnan(full[mask]).any():
            assert got.tobytes() == full.tobytes()  # such a step is scored in full
            stopped.append(0)
            continue
        cut = np.sqrt(-full[mask]).min() + pwling._TIE_TOL
        exact = np.array([got[a].tobytes() == full[a].tobytes() for a in range(len(U))])
        assert np.all(exact[mask] | ((got >= full) & (np.sqrt(-got) > cut))[mask])
        stopped.append(int((mask & ~exact).sum()))
    return stopped[0]


def _check_every_step(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make every ``pruned_scores`` call of an order search prune and check itself.

    Returns a list that receives, per call, the number of candidates that stopped early.
    """
    calls: list[int] = []
    monkeypatch.setattr(pwling, "_MIN_PRUNED_VALUES", 0)
    monkeypatch.setattr(pwling, "pruned_scores", _counted(pwling.pruned_scores, calls))
    return calls


def _counted(original, calls: list[int]):
    def checked(groups, U, *, weights=None, allowed=None, hint=None):
        got = original(groups, U, weights=weights, allowed=allowed, hint=hint)
        calls.append(
            _assert_pick_unchanged(original, groups, U, weights=weights, allowed=allowed, hint=hint)
        )
        return got

    return checked


def _exchangeable_columns(seed: int, n: int) -> np.ndarray:
    """Columns 0 and 1 swap into a row permutation of the sample: their scores tie but for rounding.

    Both are exogenous causes of column 2, and columns 3 and 4 are independent.
    """
    rng = np.random.default_rng(seed)
    u, v, w = rng.uniform(-1.0, 1.0, (3, n // 2))
    half = np.column_stack([u, v, u + v + 0.5 * w])
    X = np.vstack([half, half[:, [1, 0, 2]]])
    return np.column_stack([X, rng.laplace(size=(len(X), 2))])


def test_direct_lingam_recovery():
    out = andrey.direct_lingam(datagen.generate("lingam_5v_uniform"))
    assert_recorded_output(out, "DirectLiNGAM", case="lingam_5v_uniform")


def test_lingam_weighted_orientation_matches_structure():
    out = andrey.direct_lingam(datagen.generate("lingam_5v_uniform"))
    marks = out.structure.to_numpy()
    rows, cols = np.nonzero(out.weighted_adjacency)
    weighted_edges = set(zip(rows.tolist(), cols.tolist()))
    graph_edges = {
        (i, j)
        for i in range(marks.shape[0])
        for j in range(marks.shape[0])
        if marks[i][j] == TAIL and marks[j][i] == ARROW  # tail at source i, arrow at target j
    }
    assert weighted_edges
    assert weighted_edges == graph_edges


@pytest.mark.parametrize("d", [8, 20])
@pytest.mark.parametrize("noise", ["uniform", "gaussian"])
def test_pwling_serial_vectorized_property(d: int, noise: str, monkeypatch):
    # Batched and serial pwling must pick the same causal order on random SEMs up to d=20,
    # including near-tie Gaussian inputs (where pwling scores cluster): float associativity in the
    # batched mean stays below any order-flipping gap. `_random_linear_sem` is seeded, so this is
    # reproducible.
    X = _random_linear_sem(seed=0, n=600, d=d, noise=noise)
    vectorized = pwling.find_causal_order(X)
    reference = _order_by_serial_scores(X, monkeypatch)
    assert vectorized == reference


def test_pwling_residual_is_least_squares():
    # The order search regresses one variable on another by least squares (Shimizu et al. 2011,
    # Lemma 1): the residual equals the lstsq residual and is uncorrelated with the regressor.
    # A small n makes a slope off by n / (n - 1) visible.
    rng = np.random.default_rng(0)
    b = rng.uniform(-1.0, 1.0, 40)
    a = 0.7 * b + rng.uniform(-1.0, 1.0, 40)
    r = pwling.compute_residual(a, b)
    slope = np.linalg.lstsq(np.column_stack([np.ones_like(b), b]), a, rcond=None)[0][1]
    expected = a - slope * b
    np.testing.assert_allclose(r, expected, atol=1e-12)
    assert abs(np.mean((r - r.mean()) * (b - b.mean()))) < 1e-12


def test_pwling_batched_scores_match_serial():
    # The batched scores use the same least-squares slopes as the per-pair reference.
    X = _random_linear_sem(seed=2, n=40, d=6, noise="uniform")
    U = list(range(6))
    np.testing.assert_allclose(
        pwling.score_candidates(X, U), _score_candidates_serial(X, U), rtol=1e-9, atol=1e-12
    )


def test_score_candidates_single_candidate():
    # The foundation must survive a lone candidate (MultiGroup calls it directly in its peel loop);
    # DirectLiNGAM is shielded by causal_order's guard, but the primitive must not crash.
    scores = pwling.score_candidates(_random_linear_sem(seed=1, n=200, d=3, noise="uniform"), [1])
    assert scores.shape == (1,)


@pytest.mark.parametrize(
    "case", ["uniform", "gaussian", "exchangeable", "copy_with_precedence", "forbidden_paths"]
)
def test_pruned_scores_keep_every_pick(case: str, monkeypatch):
    """The pruned search picks as the full scores do at every step of an order search.

    Gaussian errors give near-tied scores, the exchangeable pair a tie within ``_TIE_TOL``, a copy
    with precedence NaN scores (scored in full), and forbidden paths restrict the pick.
    """
    precedence = forbidden = None
    if case in ("uniform", "gaussian"):
        X = _random_linear_sem(seed=4, n=500, d=10, noise=case)
    elif case == "exchangeable":
        X = _exchangeable_columns(seed=5, n=600)
    else:
        X = _random_linear_sem(seed=6, n=400, d=7, noise="uniform")
        precedence = np.zeros((7, 7), dtype=bool)
        precedence[5, 1] = True
        if case == "copy_with_precedence":
            X = np.column_stack([X, 2.0 * X[:, 2]])
            precedence = np.pad(precedence, (0, 1))
        else:
            forbidden = np.zeros((7, 7), dtype=bool)
            forbidden[[0, 1, 2, 4], 3] = forbidden[[0, 1, 2, 4], 6] = True
    expected = pwling.find_causal_order(X, precedence=precedence, forbidden_paths=forbidden)
    calls = _check_every_step(monkeypatch)
    order = pwling.find_causal_order(X, precedence=precedence, forbidden_paths=forbidden)
    assert order == expected
    assert len(calls) >= 4 and sum(calls) > 0


@pytest.mark.parametrize("with_precedence", [False, True])
def test_pruned_scores_keep_every_multi_group_pick(with_precedence: bool, monkeypatch):
    """The size-weighted sum over groups keeps every pick of the shared order search."""
    groups = [
        _random_linear_sem(seed=7, n=500, d=9, noise="uniform"),
        _random_linear_sem(seed=7, n=300, d=9, noise="gaussian"),
    ]
    knowledge = andrey.Knowledge(required_paths=[(0, 4)]) if with_precedence else None
    calls = _check_every_step(monkeypatch)
    andrey.multi_group_direct_lingam(groups, knowledge=knowledge)
    assert len(calls) >= 4 and sum(calls) > 0
