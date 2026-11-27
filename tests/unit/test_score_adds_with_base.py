"""A batch of adds in ``score_many_with_base`` returns what the general path returns, bit for bit.

``score_many_with_base`` scores a batch of only adds on a lean path and sends every other batch,
and every add batch that fails a guard, to ``_score_many_with_base_general``. The tests compare the
two on fresh score objects, over the scores and over the bounds, as float hex.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from andrey.core import score_delta as sd
from andrey.core.score import BICScore
from andrey.core.score_delta import DeltaBICScore

Batch = tuple[list[float], list[float]]
#: The general path as it stands, kept apart from the spy that counts the public function's calls.
_GENERAL = sd._score_many_with_base_general


def _hex(values: Sequence[float]) -> list[str]:
    return [float(x).hex() for x in values]


def _general(
    score: BICScore, v: int, base: Sequence[int], moves: Sequence[tuple[str, int]]
) -> Batch:
    bounds: list[float] = []
    return _GENERAL(score, v, base, moves, bounds=bounds), bounds


def _public(
    score: BICScore, v: int, base: Sequence[int], moves: Sequence[tuple[str, int]]
) -> Batch:
    bounds: list[float] = []
    return sd.score_many_with_base(score, v, base, moves, bounds=bounds), bounds


@pytest.fixture
def fallbacks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The batch sizes the public function sent to the general path."""
    sent: list[int] = []

    def spy(score_obj: BICScore, i: int, base: Sequence[int], deltas: Sequence, **options):
        sent.append(len(deltas))
        return _GENERAL(score_obj, i, base, deltas, **options)

    monkeypatch.setattr(sd, "_score_many_with_base_general", spy)
    return sent


def _scores(cov: np.ndarray, n: int, **options: float) -> tuple[BICScore, BICScore]:
    """Two scores over one covariance, each with its own caches."""
    return BICScore.from_cov(cov, n, **options), BICScore.from_cov(cov, n, **options)


def _adds(d: int, v: int, base: Sequence[int]) -> list[tuple[str, int]]:
    return [("add", u) for u in range(d) if u != v and u not in base]


@pytest.mark.parametrize("ebic_gamma", [0.0, 1.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_adds_match_the_general_path_at_every_base_size(
    ebic_gamma: float, seed: int, fallbacks: list[int]
) -> None:
    """Base sizes 0 to 10 with five or more candidates each, for several vertices; a repeated call
    reads the constants the first one kept."""
    d, n = 16, 400
    rng = np.random.default_rng(seed)
    mix = rng.standard_normal((d, d)) / np.sqrt(d) + np.eye(d)
    cov = np.cov((rng.standard_normal((n, d)) @ mix).T)
    lean, reference = _scores(cov, n, penalty_discount=2.0, ebic_gamma=ebic_gamma)
    for v in (0, 7, 15):
        order = [u for u in rng.permutation(d) if u != v]
        for size in range(11):
            base = sorted(int(u) for u in order[:size])
            moves = _adds(d, v, base)
            want = _general(reference, v, base, moves)
            for _ in range(2):
                got = _public(lean, v, base, moves)
                assert _hex(got[0]) == _hex(want[0])
                assert _hex(got[1]) == _hex(want[1])
    assert fallbacks == []


def test_batch_of_one_and_unsorted_base(fallbacks: list[int]) -> None:
    """A single add, and a base given out of order, with numpy integers as moves."""
    d, n = 10, 200
    cov = np.cov(np.random.default_rng(3).standard_normal((n, d)).T)
    lean, reference = _scores(cov, n)
    for base, moves in [((4, 1, 8), [("add", np.int64(5))]), ((), [("add", 2)])]:
        got = _public(lean, 0, base, moves)
        want = _general(reference, 0, base, moves)
        assert _hex(got[0]) == _hex(want[0])
        assert _hex(got[1]) == _hex(want[1])
    assert fallbacks == []


def _collinear_cov(noise: float, n: int = 500) -> np.ndarray:
    """Columns 4 and 5 are close to combinations of columns 0 to 3, and column 6 to a combination
    of the base ``(0, 1, 2, 4, 5)``; ``noise`` sets how close."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, 4))
    x4 = z[:, 0] + z[:, 1] + noise * rng.standard_normal(n)
    x5 = z[:, 2] - z[:, 0] + noise * rng.standard_normal(n)
    x6 = z[:, 0] + 2 * x4 - x5 + noise * rng.standard_normal(n)
    x7 = x6 + z[:, 3] + noise * rng.standard_normal(n)
    return np.cov(np.column_stack([z, x4, x5, x6, x7]).T)


def test_guarded_adds_match_the_general_path(fallbacks: list[int]) -> None:
    """Near-collinear columns push batches off the lean path, to the general one, whose answer is
    the one returned: some batch falls back, and every one matches."""
    n = 500
    for noise in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        cov = _collinear_cov(noise, n)
        lean, reference = _scores(cov, n)
        for v, base in [(6, (0, 1, 2, 4, 5)), (7, (0, 1, 2, 3, 4, 5)), (3, (4, 5)), (6, ())]:
            moves = _adds(cov.shape[0], v, base)
            got = _public(lean, v, base, moves)
            want = _general(reference, v, base, moves)
            assert _hex(got[0]) == _hex(want[0]), (noise, v, base)
            assert _hex(got[1]) == _hex(want[1]), (noise, v, base)
    assert fallbacks


def test_a_base_too_large_for_the_rows_takes_the_general_path(fallbacks: list[int]) -> None:
    """With ``s + 1 > n - 2`` no update is kept, and the batch is scored in full."""
    d, n = 10, 8
    cov = np.cov(np.random.default_rng(5).standard_normal((n, d)).T)
    lean, reference = _scores(cov, n)
    base = (1, 2, 3, 4, 5, 6)
    moves = _adds(d, 0, base)
    got = _public(lean, 0, base, moves)
    want = _general(reference, 0, base, moves)
    assert _hex(got[0]) == _hex(want[0])
    assert _hex(got[1]) == _hex(want[1])
    assert fallbacks == [len(moves)]


def test_an_ill_base_takes_the_general_path(fallbacks: list[int]) -> None:
    """Two equal columns make the base's covariance singular or past the conditioning gate."""
    d, n = 8, 300
    x = np.random.default_rng(6).standard_normal((n, d))
    x[:, 3] = x[:, 2]
    cov = np.cov(x.T)
    lean, reference = _scores(cov, n)
    base = (1, 2, 3)
    moves = _adds(d, 0, base)
    got = _public(lean, 0, base, moves)
    want = _general(reference, 0, base, moves)
    assert _hex(got[0]) == _hex(want[0])
    assert _hex(got[1]) == _hex(want[1])
    assert fallbacks == [len(moves)]


def _outcome(call: Callable[[], Batch]) -> tuple[str, ...] | Batch:
    try:
        scores, bounds = call()
    except (ValueError, ArithmeticError) as err:
        return ("raised", type(err).__name__, str(err))
    return _hex(scores), _hex(bounds)  # ty: ignore[invalid-return-type]


@pytest.mark.parametrize("base", [(), (1, 2)])
def test_a_constant_vertex_gives_the_general_paths_outcome(base: tuple[int, ...]) -> None:
    """A vertex without variance has no residual to update; the batch is scored as the general
    path scores it, whatever that is."""
    d, n = 6, 120
    x = np.random.default_rng(7).standard_normal((n, d))
    x[:, 0] = 1.0
    cov = np.cov(x.T)
    lean, reference = _scores(cov, n)
    moves = _adds(d, 0, base)
    with np.errstate(all="ignore"):
        got = _outcome(lambda: _public(lean, 0, base, moves))
        want = _outcome(lambda: _general(reference, 0, base, moves))
    assert got == want


@pytest.mark.parametrize(
    "first, second",
    [
        (dict(penalty_discount=1.0, ebic_gamma=0.0), dict(penalty_discount=3.0, ebic_gamma=1.0)),
        (dict(penalty_discount=2.0, ebic_gamma=1.0), dict(penalty_discount=0.5, ebic_gamma=0.5)),
    ],
)
def test_constants_stay_per_score_object(first: dict, second: dict, fallbacks: list[int]) -> None:
    """Scores with different penalties that share one inverse cache each get their own penalty and
    rounding; a call on one never serves the other's constants."""
    d, n = 12, 300
    cov = np.cov(np.random.default_rng(8).standard_normal((n, d)).T)
    shared = sd._InvCache()
    a = BICScore.from_cov(cov, n, **first)
    b = BICScore.from_cov(cov, n, **second)
    DeltaBICScore(a, inv_cov_cache=shared)
    DeltaBICScore(b, inv_cov_cache=shared)
    ref_a = BICScore.from_cov(cov, n, **first)
    ref_b = BICScore.from_cov(cov, n, **second)
    for base in [(), (3,), (3, 5, 9)]:
        moves = _adds(d, 1, base)
        for lean, reference in [(a, ref_a), (b, ref_b), (a, ref_a)]:
            got = _public(lean, 1, base, moves)
            want = _general(reference, 1, base, moves)
            assert _hex(got[0]) == _hex(want[0])
            assert _hex(got[1]) == _hex(want[1])
    assert fallbacks == []


def test_constants_are_bounded(monkeypatch: pytest.MonkeyPatch, fallbacks: list[int]) -> None:
    """A score's store of constants starts over at its cap, and the batches are scored as before."""
    monkeypatch.setattr(sd, "_ADD_CONSTANTS_CAP", 5)
    d, n = 16, 400
    cov = np.cov(np.random.default_rng(9).standard_normal((n, d)).T)
    lean, reference = _scores(cov, n)
    rng = np.random.default_rng(10)
    for v in range(d):
        others = [u for u in range(d) if u != v]
        for size in range(8):
            base = sorted(int(u) for u in rng.permutation(others)[:size])
            moves = _adds(d, v, base)
            got = _public(lean, v, base, moves)
            want = _general(reference, v, base, moves)
            assert _hex(got[0]) == _hex(want[0])
            assert _hex(got[1]) == _hex(want[1])
            assert len(sd._ADD_CONSTANTS[lean]) <= 5
    assert fallbacks == []
