"""``conditionable`` with a ``limit``: the first ``limit`` of the full selection, found by a scan
that stops there.

PCMCI and PCMCI+ on data with missing values choose each candidate's conditions with
``conditionable`` and keep the first ``p`` of them. These tests pin that ``limit=p`` gives what the
full scan's first ``p`` give, through every wrapper, on missingness patterns where the scan skips
columns and falls short of ``p``, that a scan with a limit leaves the rest of its columns unread,
and that fits equal the fits of a scan that ignores the limit, bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

import numpy as np
import pytest

import andrey
from andrey.core import ci_missing
from andrey.data import sample_scm
from andrey.temporal.pcmci import (
    _conditionable_others,
    conditionable,
    design_test,
    ranked_design_test,
)
from andrey.temporal.pcmciplus import Recorder

TAU_MAX = 2
D = 6


@pytest.fixture(autouse=True)
def _force_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """numpy is the exact float64 oracle for the correlation matrix and every p-value."""
    monkeypatch.setenv("ANDREY_DEVICE", "numpy")


def _gapped(seed: int, n: int = 80, rate: float = 0.45) -> np.ndarray:
    """Noise with ``rate`` of its entries missing, where columns 0 and 1 are never observed
    together: 0 is missing in the first half of the rows and 1 in the second."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, D))
    x[rng.random(x.shape) < rate] = np.nan
    x[: n // 2, 0] = np.nan
    x[n // 2 :, 1] = np.nan
    return x


def _tests(seed: int):
    """The three wrappers of the test-wise deletion test that PCMCI and PCMCI+ ask: the design
    test, the test over the columns in rank order, and PCMCI+'s recorder."""
    x = _gapped(seed)
    ranks = np.random.default_rng(seed).permutation(D).tolist()
    return (
        design_test("student_t", x, TAU_MAX),
        ranked_design_test("student_t", x, TAU_MAX, ranks),
        Recorder(design_test("student_t", x, TAU_MAX)),
    )


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("which", [0, 1, 2], ids=["design", "ranked", "recorder"])
def test_a_limit_gives_the_first_of_the_full_selection_for_every_limit(which, seed):
    test = _tests(seed)[which]
    rng = np.random.default_rng(seed)
    columns = D * (2 * TAU_MAX + 1)  # the design's columns: 2 * tau_max + 1 lags
    skipping = short = checked = 0
    for x in range(columns):
        ordered = [c for c in rng.permutation(columns).tolist() if c not in (x, 0)]
        full = conditionable(test, x, 0, ordered)
        skipping += full != ordered[: len(full)]
        short += len(full) < len(ordered)
        for p in range(len(ordered) + 1):
            assert conditionable(test, x, 0, ordered, p) == full[:p], (x, p)
            checked += 1
    assert checked > 500
    assert skipping > 0 and short > 0  # the scan skips columns and falls short of p


def _consumed(test, x: int, y: int, ordered: list[int], limit: int | None) -> int:
    """How many of ``ordered`` the scan reads."""
    read = 0

    def counting() -> Iterator[int]:
        nonlocal read
        for column in ordered:
            read += 1
            yield column

    conditionable(test, x, y, counting(), limit)
    return read


def test_a_scan_with_a_limit_stops_after_the_column_that_fills_it():
    test = design_test("student_t", _gapped(0, rate=0.15), TAU_MAX)
    ordered = [c for c in range(D * (2 * TAU_MAX + 1)) if c not in (3, 2)]
    full = conditionable(test, 3, 2, ordered)
    assert len(full) >= 4
    assert _consumed(test, 3, 2, ordered, None) == len(ordered)
    for p in (1, 2, 3):
        # The scan reads up to the p-th kept column, and one more entry of ``ordered`` to see that
        # the limit is reached.
        assert _consumed(test, 3, 2, ordered, p) == ordered.index(full[p - 1]) + 2
    assert _consumed(test, 3, 2, ordered, 0) == 1


class _ThreeArguments:
    """An outside test whose ``conditionable`` takes no ``limit``."""

    def conditionable(self, x: int, y: int, ordered: Iterable[int]) -> list[int]:
        return [c for c in ordered if c % 2 == 0]


class _IgnoresLimit:
    """A test whose ``conditionable`` takes a ``limit`` and answers every column."""

    def conditionable(
        self, x: int, y: int, ordered: Iterable[int], limit: int | None = None
    ) -> list[int]:
        return list(ordered)


def test_a_wrapper_without_a_limit_is_asked_with_three_arguments():
    outside = _ThreeArguments()
    assert conditionable(outside, 0, 1, [2, 3, 4, 6]) == [2, 4, 6]
    assert Recorder(outside).conditionable(0, 1, [2, 3, 4, 6]) == [2, 4, 6]
    with pytest.raises(TypeError):  # a limit is passed to the wrapper only when one is given
        conditionable(outside, 0, 1, [2, 3, 4, 6], 2)


def test_a_test_without_conditionable_keeps_the_first_of_the_columns_it_is_given():
    assert conditionable(object(), 0, 1, [5, 4, 3], 2) == [5, 4]
    assert conditionable(object(), 0, 1, [5, 4, 3], 0) == []
    assert conditionable(object(), 0, 1, (5, 4, 3)) == [5, 4, 3]


def test_the_others_of_each_candidate_are_the_first_p_when_a_test_ignores_the_limit():
    columns = np.array([7, 8, 9, 10, 11])
    others, has = _conditionable_others(_IgnoresLimit(), columns, 2, 3)
    np.testing.assert_array_equal(others, [[8, 9], [7, 9], [7, 8], [7, 8], [7, 8]])
    assert has.all()


def _bits(value) -> object:
    """``value`` with each float as its hex and each array as its bytes, so equal values are equal
    bit for bit."""
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, float | np.floating):
        return float(value).hex()
    if isinstance(value, Mapping):
        return [(_bits(k), _bits(v)) for k, v in value.items()]
    if isinstance(value, list | tuple):
        return [_bits(v) for v in value]
    return value


def _fit(method: str, x: np.ndarray, **options) -> object:
    out = getattr(andrey, method)(x, tau_max=TAU_MAX, **options)
    structure = out.structure
    return _bits([structure.lags, structure.lag_weights, structure.labels, out.metadata])


def _cases() -> list[tuple[str, np.ndarray, dict]]:
    series = sample_scm(functional="var", d=D, n=250, seed=3, density=2).data.astype(float)
    mcar = series.copy()
    mcar[np.random.default_rng(1).random(mcar.shape) < 0.08] = np.nan
    blocks = mcar.copy()
    blocks[:125, 0] = np.nan
    blocks[125:, 1] = np.nan
    permuted = blocks[:, np.random.default_rng(2).permutation(D)]
    return [
        ("mcar", mcar, {}),
        ("blocks", blocks, {}),
        ("permuted", permuted, {}),
        ("fisherz", blocks, {"indep_test": "fisherz", "max_depth": 1}),
    ]


@pytest.mark.parametrize("method", ["pcmci", "pcmciplus"])
@pytest.mark.parametrize(("x", "options"), [pytest.param(x, o, id=name) for name, x, o in _cases()])
def test_a_fit_on_missing_values_equals_the_fit_of_a_scan_that_ignores_the_limit(
    monkeypatch, method, x, options
):
    asked: list[tuple[int, int]] = []
    scan = ci_missing.TestwiseDeletion.conditionable

    def limited(self, x, y, ordered, limit=None):
        ordered = list(ordered)
        kept = scan(self, x, y, ordered, limit)
        asked.append((len(kept), len(scan(self, x, y, ordered))))
        return kept

    def unlimited(self, x, y, ordered, limit=None):
        return scan(self, x, y, ordered)

    monkeypatch.setattr(ci_missing.TestwiseDeletion, "conditionable", limited)
    got = _fit(method, x, **options)
    monkeypatch.setattr(ci_missing.TestwiseDeletion, "conditionable", unlimited)
    assert got == _fit(method, x, **options)
    assert any(kept < full for kept, full in asked)  # a limit cuts some selection short
