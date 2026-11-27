"""Shared adapter tests for adjacency, metadata coercion, and result envelopes."""

from __future__ import annotations

import re

import numpy as np
import pytest

from andrey.api import _adapt
from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput

# --- json_safe -----------------------------------------------------------------


def test_json_safe_coerces_numpy_scalars_and_arrays():
    assert isinstance(_adapt.json_safe(np.int64(3)), int)
    assert _adapt.json_safe(np.int64(3)) == 3
    assert isinstance(_adapt.json_safe(np.float64(1.5)), float)
    assert _adapt.json_safe(np.bool_(True)) is True
    assert _adapt.json_safe(np.array([1, 2])) == [1, 2]
    assert _adapt.json_safe({"a": np.float32(2.0), "b": [np.int8(1)]}) == {"a": 2.0, "b": [1]}


def test_json_safe_rejects_unserializable():
    with pytest.raises(TypeError):
        _adapt.json_safe(object())
    with pytest.raises(TypeError):
        _adapt.json_safe({"s": {1, 2}})  # a set has no JSON-safe form


# --- structure_output ----------------------------------------------------------


def test_structure_output_graph_only():
    out = _adapt.structure_output(
        GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="cpdag")
    )
    assert isinstance(out, StructureOutput)
    assert out.ordering is None and out.weighted_adjacency is None


def test_structure_output_lingam_shape():
    # DAG + causal order + weighted B (j->i), the LiNGAM adapter's envelope.
    g = GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="dag")  # 0 -> 1
    B = np.array([[0.0, 0.0], [0.7, 0.0]])  # x1 = 0.7 x0  -> weight of edge 0->1 at B[1, 0]
    out = _adapt.structure_output(
        g, ordering=np.array([0, 1]), weighted_adjacency=B, metadata={"score": np.float64(1.5)}
    )
    assert out.ordering == (0, 1)
    assert np.allclose(out.weighted_adjacency, B)
    assert isinstance(out.metadata["score"], float)
    assert out.metadata["score"] == 1.5


def test_structure_output_metadata_coerced_json_safe():
    g = GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="dag")
    out = _adapt.structure_output(g, metadata={"score": np.float64(2.0), "p": np.array([0.1, 0.2])})
    # numpy leaking into metadata would raise on save; coercion makes the envelope round-trip.
    assert out.metadata == {"score": 2.0, "p": [0.1, 0.2]}


def test_structure_output_rejects_inconsistent_kind():
    # Circle marks tagged dag: envelope validation (StructureOutput.new) rejects it.
    S = np.array([[0, CIRCLE], [ARROW, 0]])  # 0 o-> 1 (circle at 0), requires pag
    with pytest.raises(ValueError):
        _adapt.structure_output(GraphStructure.from_numpy(S, kind="dag"))


def test_canonical_weights_transposes_coefficients():
    """``_canonical_weights`` maps coefficient ``B[i,j]=j->i`` to Andrey ``W[i,j]=i->j`` (W = B.T).

    A standalone pin (W = B.T) so the LiNGAM round-trips cannot be fooled by a matched-pair
    convention flip -- an adapter and the fixture decode changed together -- since this fixes the
    orientation independently of both.
    """
    from andrey.api.lingam import _canonical_weights

    B = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, -3.0, 0.0]])  # B[1,0]: 0->1; B[2,1]: 1->2
    W = _canonical_weights(B)
    assert np.array_equal(W, B.T)
    assert W[0, 1] == 2.0 and W[1, 2] == -3.0  # Andrey W[i, j] = edge i -> j


# --- data_matrix: finiteness and constant columns from the column extremes -----------------------


def _reference_outcome(X, missing):
    """Reference outcome of the checks: a full ``isfinite`` pass, then the matrix extremes."""
    finite = np.isfinite(X)
    try:
        if finite.all():
            kept, flat = X, X.max(axis=0) == X.min(axis=0)
        else:
            kept = _adapt._missing_values(
                X, X, finite, name="data", method=None, missing=missing, fix=_adapt.DROP_ROWS
            )
            flat = np.nanmax(kept, axis=0) == np.nanmin(kept, axis=0)
    except ValueError as error:
        return "error", str(error)
    return ("constant", np.flatnonzero(flat).tolist()) if flat.any() else ("ok", kept)


def _outcome(X, missing):
    try:
        return "ok", _adapt.data_matrix(X, missing=missing)
    except ValueError as error:
        message = str(error)
        if "constant" in message:
            return "constant", [int(c) for c in re.findall(r"\d+", message.split(" constant")[0])]
        return "error", message


def _matrices():
    """Matrices over the block boundary, layouts, and a non-finite or constant cell anywhere."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 100, 511, 512, 513, 1000, 1543):
        for d in (1, 3, 10):
            base = rng.standard_normal((n, d))
            cells = {"none": []}
            for name, row in (("head", 0), ("mid", n // 2), ("tail", n - 1)):
                for value in (np.nan, np.inf, -np.inf):
                    cells[f"{value}-{name}"] = [(row, d - 1, value)]
            cells["nan-and-inf"] = [(0, 0, np.nan), (n - 1, d - 1, np.inf)]
            cells["const"] = [(i, 0, 2.5) for i in range(n)]
            cells["const-but-head"] = [(i, 0, 2.5) for i in range(1, n)]
            cells["const-but-tail"] = [(i, 0, 2.5) for i in range(n - 1)]
            cells["const-nan"] = [(i, 0, 2.5) for i in range(1, n)] + [(0, 0, np.nan)]
            cells["const-zeros"] = [(i, 0, 0.0 if i % 2 else -0.0) for i in range(n)]
            for name, edits in cells.items():
                for layout in ("C", "F"):
                    X = base.copy(order=layout)
                    for i, j, v in edits:
                        X[i, j] = v
                    yield f"{n}x{d}-{name}-{layout}", X


@pytest.mark.parametrize("missing", [False, True])
def test_data_matrix_matches_the_full_isfinite_reference(missing):
    for label, X in _matrices():
        want, got = _reference_outcome(X, missing), _outcome(X, missing)
        assert want[0] == got[0], label
        if want[0] == "ok":
            assert got[1].shape == want[1].shape and got[1].dtype == want[1].dtype, label
            assert np.array_equal(got[1], want[1], equal_nan=True), label
            assert got[1].flags.c_contiguous == want[1].flags.c_contiguous, label
        else:
            assert got[1] == want[1], label


def test_column_extremes_equal_numpys_in_every_layout():
    rng = np.random.default_rng(1)
    for n in (1, 2, 255, 511, 512, 513, 1000, 1543):
        for d in (1, 3, 10):
            wide = rng.standard_normal((n * 2, d))
            for X in (wide[:n], np.asfortranarray(wide[:n]), wide[::2]):
                high, low = _adapt._column_extremes(X)
                assert np.array_equal(high, X.max(axis=0)) and np.array_equal(low, X.min(axis=0))
                assert high.dtype == low.dtype == X.dtype and high.shape == low.shape == (d,)


def test_column_extremes_carry_a_nan_or_an_infinity_from_any_row():
    rng = np.random.default_rng(2)
    for n in (512, 513, 1000, 1543):
        for row in (0, 255, 256, n // 2, n - 257, n - 1):
            for value in (np.nan, np.inf, -np.inf):
                X = rng.standard_normal((n, 4))
                X[row, 2] = value
                high, low = _adapt._column_extremes(X)
                assert np.array_equal(high, X.max(axis=0), equal_nan=True)
                assert np.array_equal(low, X.min(axis=0), equal_nan=True)
                assert not (np.isfinite(high).all() and np.isfinite(low).all())


def test_column_extremes_of_a_matrix_without_columns_or_with_one_row_block():
    for shape in ((600, 0), (5, 0), (1, 3)):
        X = np.ones(shape)
        high, low = _adapt._column_extremes(X)
        assert high.shape == low.shape == (shape[1],)


# --- the category screen reads the first rows, then every row of a column that passes -----------


def _reference_category_counts(X, max_categories):
    """Reference for ``_category_counts``: every row of every column is read."""
    counts = np.zeros(X.shape[1], dtype=np.int64)
    observed = ~np.isnan(X)
    for j in np.flatnonzero(np.all((X == np.round(X)) | ~observed, axis=0)).tolist():
        k = np.unique(X[observed[:, j], j]).size
        if k <= max_categories:
            counts[j] = k
    return counts


def test_category_counts_equal_the_full_scan_around_the_screen_boundary():
    rng = np.random.default_rng(3)
    screen = _adapt._SCREEN_ROWS
    for n in (1, 2, 1023, screen, screen + 1, 3000):
        for k in (0, 1, 3, 4, 10, 11, 50):
            X = np.column_stack(
                [rng.integers(0, max(k, 1), n).astype(float) for _ in range(6)]
                + [rng.standard_normal(n)]
            )
            for row in (0, screen - 1, screen, n - 1, None):
                for value in (0.5, np.nan, np.inf):
                    Y = X.copy()
                    if row is not None and row < n:
                        Y[row, 1] = value
                    Y[:, 2] = np.where(np.arange(n) < screen, np.nan, Y[:, 2])  # an all-NaN head
                    Y[:, 3] = np.nan  # a column with no observed value
                    Y[0, 4] += 0.25  # a fraction in the first row only
                    assert np.array_equal(
                        _adapt._category_counts(Y, 10), _reference_category_counts(Y, 10)
                    ), (n, k, row, value)


def test_category_counts_of_a_matrix_without_columns_or_rows():
    for shape in ((0, 3), (5, 0)):
        X = np.zeros(shape)
        assert np.array_equal(_adapt._category_counts(X, 10), _reference_category_counts(X, 10))
