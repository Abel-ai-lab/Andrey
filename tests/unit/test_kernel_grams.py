"""The Gaussian kernel matrices equal the plain NumPy expressions they replace, bit for bit.

``exp_symmetric`` is ``np.exp``, ``median_exact`` is ``np.median``, and the grams of the HSIC tests,
the Stein kernel, and ANM's objective are what a plain ``np.exp`` and ``np.median`` give. Each size
sits at an edge of the code: the 64-row block of ``exp_symmetric`` (63, 64, 65, 129), the 100-row
index cache of ``_upper`` (100, 101), and the 65,536 values from which ``median_exact`` brackets
the middle (``n = 363``). The layout guard is exercised on views whose ``a @ a.T`` BLAS may not make
symmetric.
"""

from __future__ import annotations

import numpy as np
import pytest

from andrey.core import _symmetric, independence
from andrey.core._symmetric import exp_symmetric, gram_is_symmetric, median_exact, strict_upper
from andrey.fcm import stein
from andrey.irregular import anm

SIZES = (2, 7, 63, 64, 65, 100, 101, 129, 255, 256, 257, 363, 513)


def same_bits(a, b) -> bool:
    """Equal shape, dtype, and bytes, so NaN and the sign of zero count."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def layouts(n: int, q: int, seed: int = 0) -> dict[str, np.ndarray]:
    """One ``(n, q)`` matrix of standard normals in several memory layouts."""
    rng = np.random.default_rng(seed)
    wide = rng.standard_normal((n, 2 * q))
    return {
        "C": np.ascontiguousarray(wide[:, :q]),
        "F": np.asfortranarray(wide[:, :q]),
        "every other column": wide[:, ::2],
        "reversed columns": wide[:, q - 1 :: -1],
        "reversed rows": wide[::-1, :q],
    }


# ---- plain references -------------------------------------------------------------------------


def plain_sq_dists(data: np.ndarray) -> np.ndarray:
    sq_norms = np.sum(data * data, axis=1)
    return sq_norms[None, :] + sq_norms[:, None] - 2.0 * (data @ data.T)


def plain_median_width(sq_dists: np.ndarray) -> float:
    n = sq_dists.shape[0]
    upper = sq_dists[np.triu(np.ones((n, n), dtype=bool), k=1)]
    return float(np.sqrt(0.5 * np.median(upper[upper > 0])))


def plain_gamma_gram(column: np.ndarray, median_rows: int | None) -> tuple[np.ndarray, float]:
    """``gamma_gram`` from whole-matrix expressions: ``np.exp`` of every entry, ``np.median``."""
    col = np.asarray(column, dtype=np.float64)
    if col.ndim == 1:
        col = col.reshape(-1, 1)
    n = col.shape[0]
    if median_rows is not None:
        col = col - col.mean(axis=0)
    rows = col[:median_rows] if median_rows is not None and n > median_rows else col
    width = plain_median_width(plain_sq_dists(rows))
    exponent = -plain_sq_dists(col) / 2.0 / width**2
    raw = np.exp(exponent)
    col_sums, row_sums = raw.sum(axis=0), raw.sum(axis=1)
    centered = raw - (col_sums[None, :] + row_sums[:, None]) / n + (row_sums.sum() / n**2)
    raw[np.diag_indices(n)] = 0.0
    return centered, 1.0 / n / (n - 1) * raw.sum()


# ---- the helpers ------------------------------------------------------------------------------


@pytest.mark.parametrize("n", (1, 2, 63, 64, 65, 129, 255, 256, 257, 513, 1000))
def test_exp_symmetric_is_np_exp(n):
    """On an exactly symmetric matrix, in a new array and in place, in C and F order."""
    x = np.random.default_rng(n).standard_normal(n) * 30
    a = -0.5 * np.subtract.outer(x, x) ** 2  # to -1e3: underflow to 0, denormals, and 1
    assert same_bits(a, a.T)
    expected = np.exp(a)
    assert same_bits(exp_symmetric(a), expected)
    assert same_bits(a, -0.5 * np.subtract.outer(x, x) ** 2)  # the input is left alone
    in_place = a.copy()
    assert exp_symmetric(in_place, out=in_place) is in_place
    assert same_bits(in_place, expected)
    assert same_bits(exp_symmetric(np.asfortranarray(a)), np.exp(np.asfortranarray(a)))
    target = np.empty_like(a)
    assert exp_symmetric(a, out=target) is target
    assert same_bits(target, expected)


@pytest.mark.parametrize("n", (1, 2, 5, 100, 257))
def test_strict_upper_is_the_masked_upper_triangle(n):
    a = np.random.default_rng(n).standard_normal((n, n))
    assert same_bits(strict_upper(a), a[np.triu(np.ones((n, n), dtype=bool), k=1)])
    assert same_bits(strict_upper(a), a[np.triu_indices(n, 1)])


def large_values(m: int, kind: str) -> np.ndarray:
    rng = np.random.default_rng(m)
    return {
        "normal": rng.standard_normal(m),
        "positive": rng.exponential(size=m),
        "ties": rng.integers(0, 5, size=m).astype(float),
        "mostly tied": np.where(rng.random(m) < 0.9, 1.0, rng.exponential(size=m)),
        "sorted": np.sort(rng.standard_normal(m)),
        "descending": np.sort(rng.standard_normal(m))[::-1].copy(),
        "heavy tail": rng.standard_cauchy(m),
        "with inf": np.where(rng.random(m) < 0.01, np.inf, rng.standard_normal(m)),
    }[kind]


@pytest.mark.parametrize("m", (1, 2, 5, 65535, 65536, 65537, 200000, 200001))
@pytest.mark.parametrize(
    "kind",
    ("normal", "positive", "ties", "mostly tied", "sorted", "descending", "heavy tail", "with inf"),
)
def test_median_exact_is_np_median(m, kind):
    """Odd and even lengths, below and above the bracketing size, ties, order, and infinities."""
    values = large_values(m, kind)
    expected = np.median(values.copy())
    assert same_bits(np.float64(median_exact(values.copy())), np.float64(expected))


def test_median_exact_brackets_from_a_sample_and_partitions_all_on_a_miss():
    """A hit reads only the bracket, leaving the values as they were; a miss partitions them all."""
    m = 200001
    values = np.arange(m, dtype=float)  # an evenly spread sample brackets the middle
    original = values.copy()
    assert same_bits(np.float64(median_exact(values)), np.float64(np.median(original)))
    assert same_bits(values, original)

    # Every value the sample reads is tiny, so the sample's bracket sits far below the median.
    rng = np.random.default_rng(1)
    step = m // _symmetric._SAMPLE
    skewed = rng.uniform(1.0, 2.0, size=m)
    skewed[::step] = rng.uniform(0.0, 0.5, size=skewed[::step].size)
    original = skewed.copy()
    assert same_bits(np.float64(median_exact(skewed)), np.float64(np.median(original)))
    assert not same_bits(skewed, original)  # the fallback partitioned every value in place


@pytest.mark.parametrize("m", (65537, 200000))
def test_median_exact_with_a_nan_is_np_median(m):
    """A NaN gives ``np.median``'s NaN, wherever it lies, with no error."""
    for position in (0, m // 2, m - 1):
        values = large_values(m, "normal")
        values[position] = np.nan
        assert np.isnan(np.median(values))
        assert np.isnan(median_exact(values))


def test_gram_is_symmetric_names_the_layouts_whose_products_are():
    for n, q in ((10, 1), (300, 9)):
        views = layouts(n, q)
        assert gram_is_symmetric(views["C"]) and gram_is_symmetric(views["F"])
        if q > 1:
            assert not gram_is_symmetric(views["every other column"])
            assert not gram_is_symmetric(views["reversed columns"])
    assert gram_is_symmetric(layouts(40, 3)["C"][:, :1])  # one column, whatever its strides
    assert gram_is_symmetric(layouts(40, 6)["every other column"][:, :1])


# ---- the squared distances and the grams ------------------------------------------------------


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("q", (1, 3, 9))
def test_sq_dists_are_the_plain_expression_and_symmetric_for_c_and_f(n, q):
    for name, data in layouts(n, q).items():
        got = independence._sq_dists(data)
        assert same_bits(got, plain_sq_dists(data)), name
        if gram_is_symmetric(data):
            assert same_bits(got, got.T), name


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("median_rows", (None, 100))
@pytest.mark.parametrize("q", (1, 3, 9))
def test_gamma_gram_is_the_plain_gram(n, q, median_rows):
    """Every layout, at ``n`` below and above the index cache and the bracketed median."""
    for name, data in layouts(n, q).items():
        centered, mu = independence.gamma_gram(data, median_rows=median_rows)
        expected, expected_mu = plain_gamma_gram(data, median_rows)
        assert same_bits(centered, expected), name
        assert same_bits(np.float64(mu), np.float64(expected_mu)), name


@pytest.mark.parametrize("n", (255, 257, 513))
@pytest.mark.parametrize("q", (8, 16, 40))
def test_gamma_gram_of_wide_strided_views_is_the_plain_gram(n, q):
    """The layout guard: views whose ``a @ a.T`` BLAS can leave asymmetric by one bit."""
    for seed in range(3):
        for name, data in layouts(n, q, seed).items():
            if gram_is_symmetric(data):
                continue
            centered, mu = independence.gamma_gram(data, median_rows=None)
            expected, expected_mu = plain_gamma_gram(data, None)
            assert same_bits(centered, expected), (name, seed)
            assert same_bits(np.float64(mu), np.float64(expected_mu)), (name, seed)


@pytest.mark.parametrize("n", (7, 100, 363, 513))
def test_hsic_gram_is_the_plain_gram(n):
    data = layouts(n, 3)["F"]
    centered, mu, sum_squares = independence.hsic_gram(data)
    expected, expected_mu = plain_gamma_gram(data, None)
    assert same_bits(centered, expected) and same_bits(np.float64(mu), np.float64(expected_mu))
    assert same_bits(np.float64(sum_squares), np.float64(np.sum(expected * expected)))


def plain_dhsic_gram(column: np.ndarray) -> np.ndarray:
    """``dhsic_gram`` of one column: ``np.partition`` of every pair's squared distance."""
    col = independence.within_one(column.reshape(-1, 1))[0]
    col = col - col.mean(axis=0)
    sq = np.maximum(plain_sq_dists(col), 0.0)
    n = sq.shape[0]
    upper = sq[np.triu(np.ones((n, n), dtype=bool), k=1)]
    middle = upper.size // 2
    width = np.sqrt(0.5 * float(np.partition(upper, middle)[middle]))
    sq *= -1.0 / (2.0 * width * width)
    return np.exp(sq, out=sq)


@pytest.mark.parametrize("n", (7, 100, 101, 363, 513))
def test_dhsic_gram_is_the_plain_gram(n):
    for column in (layouts(n, 3)["C"][:, 0], layouts(n, 3)["every other column"][:, 1]):
        assert same_bits(independence.dhsic_gram(column), plain_dhsic_gram(column))


def plain_kernel_into(z: np.ndarray, width: float, rows: int) -> np.ndarray:
    """The centered kernel of a standardized column, every entry computed on its own."""
    n = z.shape[0]
    out = np.empty((n, n))
    for start in range(0, n, rows):
        part = out[start : start + rows]
        np.subtract(z[start : start + rows, None], z[None, :], out=part)
        np.square(part, out=part)
        np.multiply(part, -0.5, out=part)
        np.multiply(part, width, out=part)
        np.exp(part, out=part)
    col_sums = out.sum(axis=0)
    shift = col_sums.sum() / n**2
    for start in range(0, n, rows):
        part = out[start : start + rows]
        offsets = (col_sums[None, :] + col_sums[start : start + rows, None]) / n
        np.subtract(part, offsets, out=part)
        np.add(part, shift, out=part)
    return out


@pytest.mark.parametrize("n", (2, 5, 63, 64, 65, 129, 257, 400))
@pytest.mark.parametrize("rows", (1, 7, 32, 100, 1000))
def test_kernel_into_is_the_whole_row_kernel(n, rows):
    z = np.random.default_rng(n).standard_normal(n)
    block = np.empty((min(rows, n), n))
    got = independence._kernel_into(z, 4.0, np.empty((n, n)), block)
    assert same_bits(got, plain_kernel_into(z, 4.0, block.shape[0]))
    assert same_bits(got, got.T)


# ---- the Stein kernel and ANM's objective -----------------------------------------------------


def plain_stein_kernel(X: np.ndarray):
    dist = stein.squared_distances(X)
    n = X.shape[0]
    bandwidth = float(np.sqrt(np.median(dist[np.triu_indices(n, 1)])))
    return np.exp(dist / (-2.0 * bandwidth**2)), bandwidth


@pytest.mark.parametrize("n", (3, 64, 65, 100, 363, 513))
@pytest.mark.parametrize("q", (1, 4, 9, 16))
def test_stein_kernel_is_the_plain_kernel(n, q):
    """Every layout, including the views whose ``X @ X.T`` is not symmetric to the last bit."""
    for name, X in layouts(n, q).items():
        kernel = stein._kernel(X, stein.ETA, "stein")
        matrix, bandwidth = plain_stein_kernel(X)
        assert same_bits(kernel.matrix, matrix), name
        assert same_bits(np.float64(kernel.bandwidth), np.float64(bandwidth)), name


@pytest.mark.parametrize("n", (50, 400))
def test_stein_kernel_with_a_nan_distance_raises_as_np_median_does(n):
    """A column that overflows the distances leaves NaN in them, and no bandwidth."""
    X = layouts(n, 3)["C"].copy()
    X[:, 0] *= 1e200
    assert np.isnan(np.median(stein.squared_distances(X)[np.triu_indices(n, 1)]))
    with pytest.raises(ValueError, match="median distance between the rows of data is 0"):
        stein._kernel(X, stein.ETA, "stein")


@pytest.mark.parametrize("n", (5, 65, 200))
@pytest.mark.parametrize("q", (1, 3))
def test_anm_objective_is_the_plain_exp(n, q, monkeypatch):
    """ANM's value and gradient are the same with ``np.exp`` of the whole matrix."""
    rng = np.random.default_rng(n)
    cause, effect = rng.standard_normal((n, q)), rng.standard_normal((n, 1))
    theta = np.array([0.1, -0.2, -1.5])
    got = anm._negative_log_marginal_likelihood(theta, cause, effect)
    monkeypatch.setattr(anm, "exp_symmetric", lambda a, out: np.exp(a, out=out))
    expected = anm._negative_log_marginal_likelihood(theta, cause, effect)
    assert same_bits(np.float64(got[0]), np.float64(expected[0]))
    assert same_bits(got[1], expected[1])
