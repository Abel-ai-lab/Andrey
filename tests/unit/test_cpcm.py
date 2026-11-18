from __future__ import annotations

import math

import numpy as np
import pytest

from andrey.api.fcm import cpcm_graph
from andrey.api.irregular import cpcm
from andrey.core.independence import dhsic_gram, dhsic_log_p
from andrey.core.ties import tie_ranks
from andrey.fcm.cpcm import FAMILIES, LOG_SCALE_FLOOR, _log_size_bound, _term, all_dags, fit

SUPPORT = {
    "gaussian": "real",
    "gaussian_fixed_scale": "real",
    "gumbel": "real",
    "gumbel_fixed_scale": "real",
    "exponential": "positive",
    "gamma": "positive",
    "gamma_fixed_shape": "positive",
    "pareto": "above_one",
    "poisson": "counts",
    "negative_binomial": "counts",
}


def _values(kind: str, rng: np.random.Generator, n: int) -> np.ndarray:
    return {
        "real": rng.normal(size=n),
        "positive": rng.exponential(size=n) + 0.1,
        "above_one": 1.0 + rng.exponential(size=n),
        "counts": rng.poisson(3.0, size=n).astype(float),
    }[kind]


def _pair(seed: int = 0, n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.4, n)
    return np.column_stack([x, np.sin(2 * x) + x + (0.4 + 0.3 * np.tanh(x)) * rng.normal(size=n)])


def test_every_family_is_offered_and_tested():
    assert set(FAMILIES) == set(SUPPORT)


@pytest.mark.parametrize("name", sorted(SUPPORT))
def test_each_family_has_the_derivatives_of_its_likelihood(name):
    family, rng, n, h = FAMILIES[name], np.random.default_rng(1), 7, 1e-6
    y = _values(SUPPORT[name], rng, n)
    etas = [rng.normal(0.0, 0.3, n) + 0.5 * (j == 0) for j in range(family.predictors)]
    _, grad, hess = family.derivs(y, etas)

    def per_row(e: list[np.ndarray]) -> np.ndarray:
        return np.array(
            [family.derivs(y[i : i + 1], [v[i : i + 1] for v in e])[0] for i in range(n)]
        )

    for j in range(family.predictors):
        up = [e + h * (k == j) for k, e in enumerate(etas)]
        down = [e - h * (k == j) for k, e in enumerate(etas)]
        np.testing.assert_allclose((per_row(up) - per_row(down)) / (2 * h), grad[j], atol=1e-6)
        for k in range(family.predictors):
            fd = (family.derivs(y, up)[1][k] - family.derivs(y, down)[1][k]) / (2 * h)
            np.testing.assert_allclose(fd, hess[k][j], atol=1e-6)


@pytest.mark.parametrize(
    ("family", "value", "words"),
    [
        ("poisson", 2.5, "counts"),
        ("negative_binomial", -1.0, "counts"),
        ("pareto", 0.5, "at least 1"),
        ("exponential", -0.1, "non-negative"),
        ("gamma", 0.0, "positive"),
        ("gamma_fixed_shape", -2.0, "positive"),
    ],
)
def test_a_column_outside_its_family_support_fails_fast(family, value, words):
    rng = np.random.default_rng(0)
    table = np.column_stack([rng.normal(size=60), rng.poisson(3.0, 60) + 1.0])
    table[7, 1] = value
    with pytest.raises(ValueError, match=f"family {family!r} models .*{words}.*row 7"):
        cpcm(table, family=["gaussian", family])
    with pytest.raises(ValueError, match=f"family {family!r}"):
        cpcm_graph(np.column_stack([table, rng.normal(size=60)]), family=family)


def test_an_unknown_family_or_a_list_of_the_wrong_length_fails_fast():
    with pytest.raises(ValueError, match="must be one of"):
        cpcm(_pair(), family="weibull")
    with pytest.raises(ValueError, match="one name per column, got 3 names for 2 columns"):
        cpcm(_pair(), family=["gaussian"] * 3)
    with pytest.raises(ValueError, match="two variables"):
        cpcm(np.column_stack([_pair(), _pair(1)]))


def test_the_graph_method_checks_the_table_shape_and_its_row_minimum():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2-D"):
        cpcm_graph(rng.normal(size=60))
    with pytest.raises(ValueError, match="2-D"):
        cpcm_graph(3.0)
    with pytest.raises(ValueError, match="at least 24 rows .*got 23"):
        cpcm_graph(rng.normal(size=(23, 3)), family="gaussian_fixed_scale")


def test_the_seed_is_recorded_only_for_a_count_family(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    gaussian = np.column_stack([x, x**2 + rng.normal(size=80)])
    counts = np.column_stack([x, rng.poisson(np.exp(0.5 * x)).astype(float)])
    assert "seed" not in cpcm(gaussian, seed=3).metadata
    assert cpcm(counts, family=["gaussian", "poisson"], seed=3).metadata["seed"] == 3
    monkeypatch.setenv("ANDREY_SEED", "11")
    assert cpcm(counts, family=["gaussian", "poisson"]).metadata["seed"] == 11
    three = np.column_stack([counts, rng.poisson(3.0, 80).astype(float)])
    assert (
        cpcm_graph(three, family=["gaussian_fixed_scale", "poisson", "poisson"], seed=3).metadata[
            "seed"
        ]
        == 3
    )
    assert (
        "seed"
        not in cpcm_graph(
            np.column_stack([gaussian, x + rng.normal(size=80)]), family="gaussian_fixed_scale"
        ).metadata
    )


def test_a_fit_with_as_many_unpenalized_coefficients_as_rows_fails_fast():
    rng = np.random.default_rng(0)
    table = rng.integers(0, 9, size=(32, 5)).astype(float) + rng.uniform(0, 0.01, (32, 1))
    with pytest.raises(ValueError, match="unpenalized coefficients"):
        cpcm_graph(np.round(table), family="gaussian_fixed_scale")


def test_swapping_the_columns_mirrors_the_answer():
    first, second = cpcm(_pair()), cpcm(_pair()[:, ::-1])
    assert first.structure.oriented_edges() == [(0, 1, "directed")]
    assert second.structure.oriented_edges() == [(1, 0, "directed")]
    p, q = first.metadata["p_values"], second.metadata["p_values"]
    assert (p["forward"], p["backward"]) == (q["backward"], q["forward"])


def _family_pair(name: str, seed: int = 5, n: int = 150) -> np.ndarray:
    """A cause and an effect that depends on it, on the support of family ``name``."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.4, n)
    wave = np.sin(2 * x)
    effect = {
        "real": wave + x + (0.4 + 0.3 * np.tanh(x)) * rng.normal(size=n),
        "positive": rng.exponential(np.exp(wave)) + 0.01,
        "above_one": 1.0 + rng.exponential(np.exp(wave)),
        "counts": rng.poisson(np.exp(1.0 + wave)).astype(float),
    }[SUPPORT[name]]
    return np.column_stack([x, effect])


def _numbers(out) -> list[float]:
    return [*out.metadata["statistics"].values(), *out.metadata["p_values"].values()]


def _assert_same_in_units(first, moved, factor: float) -> None:
    """Equal numbers under a power of two, which rescales exactly; equal to rounding otherwise."""
    assert moved.structure.oriented_edges() == first.structure.oriented_edges()
    if math.frexp(factor)[0] == 0.5:
        assert _numbers(moved) == _numbers(first)
    else:
        np.testing.assert_allclose(_numbers(moved), _numbers(first), rtol=1e-9)


#: Powers of two up to the edge of the floats, and factors that round.
UNIT_FACTORS = (2.0**-10, 2.0**7, 2.0**1000, 2.0**-1000, 3.7, 0.37, 1e3, 1e-300, 1e300)


def test_the_families_that_fix_a_scale_are_the_ones_that_read_units():
    reading = {name for name, family in FAMILIES.items() if family.reads_units}
    assert reading == {"pareto", "poisson", "negative_binomial"}


@pytest.mark.parametrize("name", [n for n, f in FAMILIES.items() if not f.reads_units])
def test_a_family_that_fixes_no_scale_gives_the_same_answer_in_any_units(name):
    table, family = _family_pair(name), ["gaussian_fixed_scale", name]
    first = cpcm(table, family=family)
    for factor in UNIT_FACTORS:
        for factors in ([1.0, factor], [factor, 1.0]):
            _assert_same_in_units(first, cpcm(table * np.array(factors), family=family), factor)


@pytest.mark.parametrize(
    ("name", "outside"), [("pareto", 0.5), ("poisson", 3.7), ("negative_binomial", 3.7)]
)
def test_a_family_that_fixes_a_scale_reads_the_units(name, outside):
    table, family = _family_pair(name, n=300), ["gaussian_fixed_scale", name]
    first = cpcm(table, family=family)
    doubled = cpcm(table * np.array([1.0, 2.0]), family=family)
    assert not np.allclose(_numbers(doubled), _numbers(first), rtol=1e-3)
    for factor in UNIT_FACTORS:  # the cause's units are not part of the effect's family
        moved = cpcm(table * np.array([factor, 1.0]), family=family)
        _assert_same_in_units(first, moved, factor)
    with pytest.raises(ValueError, match=f"family '{name}' models"):
        cpcm(table * np.array([1.0, outside]), family=family)


def test_a_mostly_tied_cause_gives_the_same_answer_in_any_units():
    # More than half of the cause's pairs of rows are tied, so its kernel width is the one for a
    # zero median distance.
    rng = np.random.default_rng(0)
    x = rng.choice([0.0, 1.0, 2.5], size=200, p=[0.8, 0.1, 0.1])
    table = np.column_stack([x, x + (0.5 + 0.5 * x) * rng.normal(size=200)])
    first = cpcm(table)
    for factor in UNIT_FACTORS:
        _assert_same_in_units(first, cpcm(table * np.array([factor, 1.0])), factor)
    _assert_same_in_units(first, cpcm(table + np.array([5.0, 0.0])), 3.7)


def test_a_count_column_in_units_too_large_to_fit_fails_fast():
    table = _family_pair("negative_binomial", n=300) * np.array([1.0, 2.0**1000])
    with pytest.raises(ValueError, match="stopped away from a minimum"):
        cpcm(table, family=["gaussian_fixed_scale", "negative_binomial"])


def test_location_scale_families_read_no_origin():
    table = _pair()
    first = cpcm(table)
    moved = cpcm(table * np.array([1e-3, 250.0]) + np.array([40.0, -7.0]))
    assert moved.structure.oriented_edges() == first.structure.oriented_edges()
    np.testing.assert_allclose(_numbers(moved), _numbers(first), rtol=1e-9)


@pytest.mark.parametrize(("seed", "n"), [(23, 200), (9, 50)])
def test_a_nearly_singular_fit_gives_an_answer_that_no_rounding_moves(seed, n):
    # On these tables the reverse fit's Hessian is close to singular at low smoothing.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    table = np.column_stack([x, x**2 + (1 + 0.5 * np.abs(x)) * rng.normal(size=n)])
    first = cpcm(table)
    assert first.structure.oriented_edges() == [(0, 1, "directed")]
    assert all(0.0 < p <= 1.0 for p in first.metadata["p_values"].values())
    for factor in (3.7, 0.37, 2.0):
        for factors in ([1.0, factor], [factor, 1.0]):
            _assert_same_in_units(first, cpcm(table * np.array(factors)), factor)


def test_a_parent_value_seen_once_gives_an_answer_under_a_fitted_scale():
    rng = np.random.default_rng(3)
    x = np.sort(rng.integers(0, 3, size=200)).astype(float)
    x[-1] = 3.0  # one row holds the fourth value, so the effect is constant on its rows
    table = np.column_stack([x, x + (0.5 + 0.2 * x) * rng.normal(size=200)])
    out = cpcm(table)
    assert all(0.0 < p <= 1.0 for p in out.metadata["p_values"].values())
    assert np.isfinite(list(out.metadata["statistics"].values())).all()


def test_a_row_the_location_fits_alone_keeps_its_scale_above_the_floor():
    rng = np.random.default_rng(4)
    x = rng.normal(size=200)
    x[0] = 40.0  # far from every other row, so the spline can pass through its value
    y = np.sin(x) + 0.3 * rng.normal(size=200)
    y[0] += 3.0
    family = FAMILIES["gaussian"]
    _, log_scale = fit(family.rescale(y), [_term(x)], family)
    scale = np.exp(np.logaddexp(log_scale, LOG_SCALE_FLOOR["gaussian"]))
    assert 0.01 <= scale[0] < 0.05 < scale[1:].min()
    out = cpcm(np.column_stack([x, y]))
    assert all(0.0 < p <= 1.0 for p in out.metadata["p_values"].values())


@pytest.mark.parametrize("name", ["poisson", "negative_binomial"])
@pytest.mark.parametrize("zero", [0.0, 1.0])
def test_a_parent_value_whose_counts_are_all_zero_has_mean_zero(name, zero):
    # Value 0 is the indicators' reference, so its rows' mean sits in the intercept.
    rng = np.random.default_rng(7)
    x = rng.integers(0, 3, size=150).astype(float)
    y = rng.poisson(np.where(x == 2.0, 6.0, 2.0)).astype(float)
    y[x == zero] = 0.0
    family = FAMILIES[name]
    log_mean = fit(y, [_term(x)], family)[0]
    assert np.all(log_mean[x == zero] == -np.inf)
    for value in {0.0, 1.0, 2.0} - {zero}:
        np.testing.assert_allclose(np.exp(log_mean[x == value]), y[x == value].mean(), rtol=1e-6)
    out = cpcm(np.column_stack([x, y]), family=["gaussian_fixed_scale", name], seed=0)
    assert all(0.0 < p <= 1.0 for p in out.metadata["p_values"].values())


def _single_value_table(name: str, seed: int, *, reference: bool = False) -> np.ndarray:
    """A count cause with a value seen once and an effect of family ``name``; with ``reference``
    that value is the lowest, the indicators' reference."""
    rng = np.random.default_rng(seed)
    x = rng.poisson(2.0, size=100).astype(float) + 1.0
    if reference:
        x[0] = 0.0
    if name == "gamma":
        return np.column_stack([x, np.exp(0.3 * x) * rng.gamma(3.0, 1.0 / 3.0, size=100)])
    return np.column_stack([x, np.sin(x) + (0.5 + 0.1 * x) * rng.normal(size=100)])


@pytest.mark.parametrize(
    ("name", "seed", "reference"),
    [("gumbel", 23, False), ("gamma", 5, False), ("gaussian", 1, True), ("gumbel", 1, True)],
)
def test_a_scale_that_falls_to_its_floor_is_held_there_in_any_units(name, seed, reference):
    # The location fits the rows of a value seen once, so their scale falls toward the floor
    # without reaching it; the fit holds them at the floor, wherever its steps stop.
    table = _single_value_table(name, seed, reference=reference)
    family = FAMILIES[name]
    log_scale = fit(family.rescale(table[:, 1]), [_term(table[:, 0])], family)[1]
    values, counts = np.unique(table[:, 0], return_counts=True)
    once = np.isin(table[:, 0], values[counts == 1])
    assert np.all(log_scale[once] == -np.inf)  # the floored scale reads this as the floor
    assert np.all(log_scale[~once] > family.floor)
    families = ["gaussian", name]
    first = cpcm(table, family=families)
    for factor in (3.7, 1e-7, 2.0**40, 2.0**-1000):
        moved = cpcm(table * np.array([1.0, factor]), family=families)
        _assert_same_in_units(first, moved, factor)


def _rare_value_table(seed: int) -> np.ndarray:
    """A count parent whose largest value is seen once, a smooth parent, and their effect."""
    rng = np.random.default_rng(seed)
    count = np.minimum(rng.poisson(1.0, size=200), 3).astype(float)
    count[int(np.argmax(count))] = 4.0
    x = rng.normal(size=200)
    noise = rng.normal(size=200)
    return np.column_stack([count, x, np.sin(x) + 0.5 * count + 0.5 * np.exp(0.3 * x) * noise])


def test_a_parent_value_seen_once_beside_a_smooth_parent_takes_the_reference_scale():
    table = _rare_value_table(0)
    assert np.sum(table[:, 0] == 4.0) == 1
    effect = FAMILIES["gaussian"].rescale(table[:, 2])
    _, log_scale = fit(effect, [_term(table[:, 0]), _term(table[:, 1])], FAMILIES["gaussian"])
    assert log_scale[table[:, 0] == 4.0][0] > LOG_SCALE_FLOOR["gaussian"] + 1.0
    out = cpcm_graph(table, family=["poisson", "gaussian", "gaussian"], seed=0)
    assert out.structure.oriented_edges() == [(0, 2, "directed"), (1, 2, "directed")]


def test_a_column_that_is_an_exact_function_of_the_other_fails_fast():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 6, size=100).astype(float)
    table = np.column_stack([x, (x >= 3).astype(float)])
    for scaled in (table, table * np.array([1.0, 3.7]), table * np.array([3.7, 1.0])):
        with pytest.raises(ValueError, match="cpcm: column 1 on 0"):
            cpcm(scaled)
        with pytest.raises(ValueError, match="cpcm: column 1 on 0"):
            cpcm(scaled, alpha=0.05)
        with pytest.raises(ValueError, match="cpcm_graph: column"):
            cpcm_graph(scaled, seed=0)
    with pytest.raises(ValueError, match="the residual is the same on every row"):
        cpcm(table + np.array([1.0, 1.0]), family="exponential")


def test_two_directions_that_mirror_each_other_give_no_edge_in_any_units():
    # A symmetric two-by-two table: both directions have the same p-value and statistic, up to
    # the last digits, which differ with the units.
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, size=100).astype(float)
    y = (b + rng.normal(size=100) > 0.5).astype(float)
    assert np.sum((b == 0) & (y == 1)) == np.sum((b == 1) & (y == 0))
    for factor in (1.0, 3.7, 0.37, 1e3):
        out = cpcm(np.column_stack([b, y * factor]))
        assert out.structure.oriented_edges() == []


#: Factors that round, on the cause, the effect, and both.
ROUNDING_FACTORS = ((3.0, 1.0), (1.0, 3.0), (1e6, 1.0), (1.0, 1e-6), (3.7, 3.7), (3.7, 1e-6))


@pytest.mark.parametrize("name", [n for n, f in FAMILIES.items() if not f.reads_units])
@pytest.mark.parametrize("seed", [9, 18, 63, 64])
def test_a_heavy_tailed_table_gives_the_same_answer_in_any_units(name, seed):
    # Positive columns with a lognormal ratio: the fits are ill-conditioned, with coefficients in
    # the thousands, and the two smooths' penalties differ by many orders of magnitude. On seed 18
    # the Gaussian fit's Hessian is indefinite along the Newton path.
    rng = np.random.default_rng(seed)
    x = rng.normal(size=40)
    table = np.column_stack([np.exp(x), rng.lognormal(0.0, 1.5, size=40) * np.exp(x)])
    first = cpcm(table, family=name)
    for factors in ROUNDING_FACTORS:
        _assert_same_in_units(first, cpcm(table * np.array(factors), family=name), 3.7)


def test_rows_that_are_equal_in_every_column_are_at_distance_zero():
    rng = np.random.default_rng(8)
    X = rng.poisson(0.1, size=(200, 3)).astype(float)
    same = (X[:, None, :] == X[None, :, :]).all(axis=2)
    assert same.mean() > 0.5
    gram = dhsic_gram(X, standardize=True)
    assert np.all(gram[same] == 1.0)
    assert np.all(gram[~same] < 1e-6)  # the width for a zero median distance
    moved = dhsic_gram(X * np.array([3.7e5, 1e-9, 12.0]), standardize=True)
    np.testing.assert_allclose(moved, gram, rtol=1e-9, atol=1e-12)


def test_the_negative_binomial_size_bound_follows_the_counts():
    small, large = np.array([0.0, 3.0, 40.0]), np.array([0.0, 3.0, 2e6])
    assert _log_size_bound(small) == 12.0
    assert _log_size_bound(large) == pytest.approx(math.log(2e6) + 5.0)
    assert _log_size_bound(np.array([1e12])) == 20.0


def test_the_graph_method_gives_the_same_answer_in_extreme_units():
    rng = np.random.default_rng(1)
    a = rng.normal(size=150)
    b = np.sin(2 * a) + (0.3 + 0.2 * np.abs(a)) * rng.normal(size=150)
    table = np.column_stack([a, b, np.cos(b) + 0.3 * rng.normal(size=150)])
    first = cpcm_graph(table)
    moved = cpcm_graph(table * np.array([2.0**1000, 2.0**-1000, 2.0**40]))
    assert moved.structure.oriented_edges() == first.structure.oriented_edges()
    assert moved.metadata["score"] == first.metadata["score"]
    assert moved.metadata["p_value"] == first.metadata["p_value"]
    rounded = cpcm_graph(table * np.array([1e300, 1e-300, 3.7]))
    assert rounded.structure.oriented_edges() == first.structure.oriented_edges()
    np.testing.assert_allclose(rounded.metadata["score"], first.metadata["score"], rtol=1e-9)


def test_the_column_ranks_are_the_same_in_units_near_the_ends_of_the_float_range():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        table = np.column_stack([rng.normal(size=60), rng.exponential(size=60)])
        first = tie_ranks(table)  # two columns: their marginals decide
        for factor in (2.0**-1000, 2.0**990, 1e300, 1e-300, 1e307):
            for j in range(2):
                factors = np.ones(2)
                factors[j] = factor
                np.testing.assert_array_equal(tie_ranks(table * factors), first)


def test_a_table_that_mirrors_itself_keeps_its_tied_graph_in_any_units():
    # Rows (a, b) and (b, a): the two one-edge DAGs score the same, and the first in the
    # columns' rank order wins, which must not move when a column's units do.
    for seed in range(3):
        rng = np.random.default_rng(seed)
        a = rng.normal(size=30)
        b = a + 0.3 * rng.normal(size=30)
        table = np.vstack([np.column_stack([a, b]), np.column_stack([b, a])])
        first = cpcm_graph(table)
        for factor in (2.0**-1000, 2.0**990, 1e300, 1e-300, 3.7):
            moved = cpcm_graph(table * np.array([1.0, factor]))
            assert moved.structure.oriented_edges() == first.structure.oriented_edges()
            np.testing.assert_allclose(moved.metadata["score"], first.metadata["score"], rtol=1e-9)


def test_a_gumbel_graph_with_a_count_parent_keeps_its_answer_in_other_units():
    # A value of the count parent seen twice: the location fits its rows, so their scale falls to
    # the floor, where the fit holds it; the score moved by 0.04 under y * 1e-7 before.
    rng = np.random.default_rng(5)
    c = rng.poisson(1.0, 200).astype(float)
    x = rng.normal(size=200)
    y = np.sin(x) + 0.5 * c + np.exp(0.3 * x) * 0.5 * rng.normal(size=200)
    table = np.column_stack([c, x, y])
    families = ["poisson", "gaussian", "gumbel"]
    first = cpcm_graph(table, family=families)
    for factors in ([1.0, 1.0, 1e-7], [1.0, 3.7, 1e9]):
        moved = cpcm_graph(table * np.array(factors), family=families)
        assert moved.structure.oriented_edges() == first.structure.oriented_edges()
        np.testing.assert_allclose(moved.metadata["score"], first.metadata["score"], rtol=1e-9)


def test_a_count_family_repeats_under_its_seed_and_follows_it():
    rng = np.random.default_rng(3)
    x = rng.normal(size=300)
    table = np.column_stack([x, rng.poisson(np.exp(1 + np.sin(2 * x)))])
    family = ["gaussian_fixed_scale", "poisson"]
    first = cpcm(table, family=family, seed=5)
    assert cpcm(table, family=family, seed=5).metadata == first.metadata
    assert cpcm(table, family=family, seed=6).metadata["p_values"] != first.metadata["p_values"]
    assert cpcm(_pair(), seed=5).metadata["p_values"] == cpcm(_pair(), seed=6).metadata["p_values"]


def test_independent_columns_give_no_edge_at_a_level():
    rng = np.random.default_rng(4)
    out = cpcm(rng.normal(size=(300, 2)), alpha=0.05)
    assert out.structure.oriented_edges() == []
    assert out.metadata["independence_p_value"] > 0.05


def test_the_graph_search_relabels_under_any_column_order():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    y = rng.exponential(1 + x**2)
    z = rng.exponential(1 + np.sqrt(y))
    table = np.column_stack([x, y, z])
    family = ["gaussian_fixed_scale", "exponential", "exponential"]
    first = cpcm_graph(table, family=family)
    assert first.structure.oriented_edges() == [(0, 1, "directed"), (1, 2, "directed")]
    perm = [2, 0, 1]
    out = cpcm_graph(table[:, perm], family=[family[j] for j in perm])
    assert sorted(out.structure.oriented_edges()) == [(1, 2, "directed"), (2, 0, "directed")]
    assert out.metadata["p_value"] == pytest.approx(first.metadata["p_value"], rel=1e-3)


@pytest.mark.parametrize(("d", "count"), [(2, 3), (3, 25), (4, 543)])
def test_the_exact_search_lists_every_dag_once(d, count):
    dags = all_dags(d)
    assert len(dags) == count
    assert len({tuple(dag) for dag in dags}) == count
    assert [sum(map(len, dag)) for dag in dags] == sorted(sum(map(len, dag)) for dag in dags)


def test_dhsic_of_two_variables_is_the_hsic_v_statistic():
    rng = np.random.default_rng(6)
    x = rng.normal(size=200)
    gx, gy = dhsic_gram(x), dhsic_gram(x**2 + rng.normal(size=200))
    n = x.size
    H = np.eye(n) - 1.0 / n
    statistic, log_p = dhsic_log_p([gx, gy])
    mean = (1 - gx.mean() - gy.mean() + gx.mean() * gy.mean()) / n
    assert statistic * mean == pytest.approx(np.trace(gx @ H @ gy @ H) / n**2, rel=1e-10)
    assert log_p < np.log(1e-3)
    assert dhsic_log_p([gx, dhsic_gram(rng.normal(size=200))])[1] > np.log(1e-3)
