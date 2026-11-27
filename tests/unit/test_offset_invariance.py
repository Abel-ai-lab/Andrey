"""A column's offset changes no method's graph.

A method's result depends on where a column's zero lies only where its model does. A step that
reads the offset through its arithmetic, such as squared distances from ``x^2 + y^2 - 2xy`` or a
sum of terms far from 0, is a numerical artifact. The data here lie on a grid that an offset of
``1e9`` keeps exactly, so every fit sees the same values at both offsets and any difference comes
from the method.

``varma_lingam`` also has a case of its own on a larger series: its likelihood fit takes a constant
trend, so a column's mean is not part of the model. The LiNGAM order search has cases of its own on
columns whose third moment is zero or within rounding of it, where a shift that reaches the order
through the arithmetic decides the sign of the skew measure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import andrey
from andrey.core import TemporalStructure
from andrey.core.independence import gamma_gram, hsic_gamma_from_grams
from andrey.spec import list_specs
from tests.recovery.helpers import datagen
from tests.release import release_cases
from tests.unit.methods import domains, environments, fit_settings, regime_paths, results

N = 400
#: The grid spacing, ``2**-23``: the spacing of the floats in ``[2**29, 2**30)``, so a column on it
#: plus any offset below ``2**29`` stays exactly the same values.
GRID = 2.0**-23
#: Shifts of the columns: a large offset, and a scale with an offset.
SHIFTS = {"offset": (1.0, 1e9), "scale_and_offset": (3.7, 1e6)}


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        warnings.simplefilter("ignore", andrey.DataTypeWarning)
        warnings.simplefilter("ignore", andrey.SampleSizeWarning)
        warnings.simplefilter("ignore", andrey.SearchLimitWarning)
        yield


def on_grid(x: np.ndarray) -> np.ndarray:
    """``x`` rounded to the multiples of :data:`GRID`."""
    return np.round(x / GRID) * GRID


def chain(seed: int = 0) -> np.ndarray:
    """A non-Gaussian chain 0 -> 1 -> 2, so every method family finds structure."""
    u = np.random.default_rng(seed).uniform(-1, 1, size=(N, 3)) ** 3
    x = u.copy()
    x[:, 1] += 1.2 * u[:, 0]
    x[:, 2] += 0.7 * x[:, 1]
    return on_grid(x)


def shifted_chain(seed: int = 1) -> np.ndarray:
    """The chain whose variable 1 has its noise shifted by 1 in the second half of the rows, the
    second domain of :func:`domains`: ICP finds the parent 1 of variable 2 through the shift."""
    u = np.random.default_rng(seed).uniform(-1, 1, size=(N, 3)) ** 3
    u[N // 2 :, 1] += 1.0
    x = u.copy()
    x[:, 1] += 1.2 * u[:, 0]
    x[:, 2] += 0.7 * x[:, 1]
    return on_grid(x)


def graphs(structure) -> list[np.ndarray]:
    """The endpoint marks of every graph a structure holds: itself, or each lag and each cell."""
    if isinstance(structure, TemporalStructure):
        cells = [structure.at(t, k) for t in structure.times for k in structure.lags]
        return [structure.lag(k).to_numpy() for k in structure.lags] + [c.to_numpy() for c in cells]
    return [structure.to_numpy()]


def marks(out) -> list[np.ndarray]:
    return [
        *(m for result in results(out) for m in graphs(result.structure)),
        *regime_paths(out),
    ]


def inputs(spec, data: np.ndarray, shift: tuple[float, float]) -> list:
    """The method's required inputs, gnies's env, and the environments of a method in ``CONTEXT``,
    built from ``data`` shifted to ``data * scale + offset``."""
    scale, offset = shift

    def moved(x: np.ndarray) -> np.ndarray:
        return x * scale + offset

    args = []
    for inp in spec.data.inputs:
        if not inp.required and (spec.name, inp.name) != ("gnies", "env"):
            if spec.name in CONTEXT and inp.kind == "context":
                args.append(environments(N))
            continue
        if inp.kind == "matrix":
            args.append(moved(data[:, :2] if spec.family == "pairwise" else data))
        elif inp.kind == "panel":
            args.append([moved(chain(seed)) for seed in range(3)])
        elif inp.kind == "context":
            args.append(domains(N))
    return args


#: The targets of a method that reads three environments: observational, one with the known
#: target 2, and one whose targets are unknown (ut_igsp) or 0 (igsp).
CONTEXT = {
    "igsp": {"targets": {0: [], 1: [2], 2: [0]}},
    "ut_igsp": {"targets": {0: [], 1: [2], 2: None}},
}

TABLE_SPECS = [
    spec for spec in list_specs() if {i.kind for i in spec.data.inputs} & {"matrix", "panel"}
]


@pytest.mark.parametrize(
    "spec", release_cases(TABLE_SPECS, {"skewd", "skewd_mv"}, id=lambda spec: spec.name)
)
def test_shifting_the_columns_keeps_every_methods_graphs(spec):
    if spec.name == "calm":
        pytest.importorskip("torch")
    fit = getattr(andrey, spec.name)
    data = shifted_chain() if spec.name == "icp" else chain()
    settings = {**fit_settings(spec.name), **CONTEXT.get(spec.name, {})}
    want = marks(fit(*inputs(spec, data, (1.0, 0.0)), **settings))
    if spec.name == "icp":
        assert any(m.any() for m in want), "ICP finds no parent on the unshifted data"
    for shift in SHIFTS.values():
        got = marks(fit(*inputs(spec, data, shift), **settings))
        assert len(got) == len(want)
        for have, expected in zip(got, want, strict=True):
            np.testing.assert_array_equal(have, expected, err_msg=f"{shift}")


@pytest.mark.parametrize("shift", SHIFTS.values(), ids=SHIFTS.keys())
@pytest.mark.parametrize(
    ("method", "case"),
    [
        ("rcd", "confounded_fork_4v"),
        ("rcd", "fork_5v_observed"),
        ("gin", "measurement_6v"),
        ("gin", "measurement_chain_9v"),
    ],
)
@pytest.mark.parametrize("replicate", range(3))
def test_rcd_and_gin_keep_their_graph_on_shifted_columns(method, case, replicate, shift):
    """RCD and GIN keep their graph and metadata on rescaled and shifted columns. Each case runs
    three replicates, each a new sample in a new column order: the claim covers every draw, and an
    invariance test shows it only for the draws it runs."""
    scale, offset = shift
    data = on_grid(datagen.generate(case, replicate=replicate))
    fit = getattr(andrey, method)
    want, got = fit(data), fit(data * scale + offset)
    for have, expected in zip(marks(got), marks(want), strict=True):
        np.testing.assert_array_equal(have, expected)
    assert got.metadata == want.metadata


@pytest.mark.parametrize("offset", [1e3, 1e6, 1e9])
def test_gamma_hsic_does_not_read_the_columns_offset(offset):
    rng = np.random.default_rng(0)
    x = on_grid(rng.uniform(-1, 1, N) ** 3)
    y = on_grid(x**2 + 0.3 * rng.standard_normal(N))
    want = hsic_gamma_from_grams(gamma_gram(x), gamma_gram(y), N)
    got = hsic_gamma_from_grams(gamma_gram(x + offset), gamma_gram(y - offset), N)
    assert got == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("shift", SHIFTS.values(), ids=SHIFTS.keys())
def test_varma_lingam_keeps_its_graph_on_shifted_columns(shift):
    """A series 400 rows by 9 columns, fit with one optimizer iteration, keeps its graphs."""
    scale, offset = shift
    data = on_grid(datagen.generate("measurement_chain_9v", replicate=4))
    want = andrey.varma_lingam(data, max_iter=1)
    got = andrey.varma_lingam(data * scale + offset, max_iter=1)
    for have, expected in zip(marks(got), marks(want), strict=True):
        np.testing.assert_array_equal(have, expected)
    assert list(got.ordering) == list(want.ordering)


def exchangeable(seed: int, d: int = 4) -> np.ndarray:
    """Columns that hold one grid of values, the later ones a few swaps from the first, as the order
    search sees exchangeable variables: every column has the same third moment, about 1e-13, the
    rounding of the values onto the grid, so each step's sign of the skew measure is decided by how
    exactly the search computes it."""
    rng = np.random.default_rng(seed)
    first = rng.permutation(N) + 1.0
    columns = [first, *(rng.permutation(N) + 1.0 for _ in range(d - 2))]
    swapped = first.copy()
    for k in rng.choice(np.arange(1, N, 2), size=20, replace=False):
        i, j = np.flatnonzero(first == k)[0], np.flatnonzero(first == k + 1)[0]
        swapped[i], swapped[j] = k + 1, k
    return on_grid(np.column_stack([first, swapped, *columns[1:]]) / N)


def binary(seed: int, d: int = 4) -> np.ndarray:
    """Correlated binary columns split at their medians: exactly as many ones as zeros, a third
    moment of exactly zero."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(np.zeros(d), 0.5 * np.eye(d) + 0.5, size=N)
    return (z > np.median(z, axis=0)).astype(np.float64)


def mirrored(seed: int, d: int = 4) -> np.ndarray:
    """A chain of Laplace columns, each row followed by its negative in a shuffled order: a third
    moment of exactly zero."""
    rng = np.random.default_rng(seed)
    half = on_grid(rng.laplace(size=(N // 2, d)))
    half[:, 1] = on_grid(half[:, 1] + 0.8 * half[:, 0])
    return np.vstack([half, -half])[rng.permutation(N)]


def skewed_chain(seed: int, d: int = 5) -> np.ndarray:
    """A chain of exponential columns: a third moment far from zero."""
    rng = np.random.default_rng(seed)
    x = rng.exponential(size=(N, d)) - 1.0
    for j in range(1, d):
        x[:, j] += 0.8 * x[:, j - 1]
    return on_grid(x)


ORDER_KINDS = {
    "exchangeable": exchangeable,
    "binary": binary,
    "mirrored": mirrored,
    "skewed_chain": skewed_chain,
}
#: Offsets from 1e3 to 1e9, and a power-of-two scale with an offset, each exact on the grid.
ORDER_SHIFTS = {
    "offset 1e3": (1.0, 1e3),
    "offset 1e6": (1.0, 1e6),
    "offset 1e9": (1.0, 1e9),
    "scale 8 and offset 1e6": (8.0, 1e6),
}


@pytest.mark.parametrize("measure", ["pwling", "skew", "tanh"])
@pytest.mark.parametrize("kind", ORDER_KINDS)
def test_the_lingam_order_search_keeps_its_order_on_shifted_columns(kind, measure):
    """The order of DirectLiNGAM on shifted columns is the order of the unshifted ones, on every
    seed of four kinds of columns, three measures, and four shifts. Each shifted copy holds the
    unshifted values exactly, so any difference comes from the search. The grid is a tripwire
    for the search reading the offset: it does not show that no other column does."""
    for seed in range(4 if measure == "pwling" else 10):
        data = ORDER_KINDS[kind](seed)
        want = andrey.direct_lingam(data, measure=measure).ordering
        for scale, offset in ORDER_SHIFTS.values():
            got = andrey.direct_lingam(data * scale + offset, measure=measure).ordering
            assert got == want, (seed, scale, offset)


@pytest.mark.parametrize("measure", ["pwling", "skew", "tanh"])
@pytest.mark.parametrize("kind", ORDER_KINDS)
def test_the_multi_group_order_search_keeps_its_order_on_shifted_columns(kind, measure):
    """The shared order of two groups, all columns shifted alike, is the unshifted order."""
    for seed in range(3 if measure == "pwling" else 8):
        groups = [ORDER_KINDS[kind](seed), ORDER_KINDS[kind](seed + 100)[: N * 3 // 5]]
        want = andrey.multi_group_direct_lingam(groups, measure=measure)[0].ordering
        for scale, offset in ORDER_SHIFTS.values():
            moved = [group * scale + offset for group in groups]
            got = andrey.multi_group_direct_lingam(moved, measure=measure)[0].ordering
            assert got == want, (seed, scale, offset)
