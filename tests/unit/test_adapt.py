"""Shared adapter tests for adjacency, metadata coercion, and result envelopes."""

from __future__ import annotations

import numpy as np
import pytest

from andrey.api import _adapt
from andrey.core import ARROW, CIRCLE, TAIL, GraphStructure, StructureOutput
from andrey.core.orient import to_structure

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


# --- edge_structure ------------------------------------------------------------


def _same_store(a: GraphStructure, b: GraphStructure) -> bool:
    return (
        a == b
        and a._layout == b._layout
        and np.array_equal(a._indptr, b._indptr)
        and np.array_equal(a._indices, b._indices)
        and np.array_equal(a._marks, b._marks)
    )


@pytest.mark.parametrize("kind", ["cpdag", "pag", "dag", "undirected"])
def test_edge_structure_equals_the_structure_of_the_matrix_it_fills(kind):
    rng = np.random.default_rng(3)
    marks = {
        "cpdag": [(TAIL, ARROW), (ARROW, TAIL), (TAIL, TAIL)],
        "pag": [(TAIL, ARROW), (ARROW, ARROW), (CIRCLE, ARROW), (CIRCLE, CIRCLE), (TAIL, TAIL)],
        "dag": [(TAIL, ARROW), (ARROW, TAIL)],
        "undirected": [(TAIL, TAIL)],
    }[kind]
    for d, k in [(1, 0), (2, 1), (9, 0), (9, 5), (40, 12), (40, 40)]:
        pairs = rng.permutation([(i, j) for i in range(d) for j in range(i + 1, d)])[:k]
        edges = []
        for i, j in pairs.tolist():
            at_i, at_j = marks[rng.integers(len(marks))]
            edges.append((i, j, at_i, at_j) if rng.random() < 0.5 else (j, i, at_j, at_i))
        dense = np.zeros((d, d), dtype=np.int8)
        for i, j, at_i, at_j in edges:
            dense[i, j], dense[j, i] = at_i, at_j
        built = _adapt.edge_structure(d, edges, kind=kind)
        assert _same_store(built, GraphStructure.from_numpy(dense, kind=kind))
        if kind != "pag":
            assert _same_store(built, to_structure(dense, kind=kind))


def test_edge_structure_builds_no_matrix_over_all_nodes():
    import tracemalloc

    d = 50_000  # a (d, d) int8 matrix is 2.5 GB
    tracemalloc.start()
    try:
        built = _adapt.edge_structure(
            d, [(7, v, TAIL, ARROW) for v in (3, 9, 49_999)], kind="cpdag"
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert built.n_nodes == d and len(built.to_edges()) == 3
    assert peak < 50_000_000


@pytest.mark.parametrize(
    ("edges", "kind", "error"),
    [
        ([(1, 1, TAIL, TAIL)], "cpdag", ValueError),
        ([(0, 1, TAIL, ARROW), (1, 0, ARROW, TAIL)], "cpdag", ValueError),
        ([(0, 1, CIRCLE, ARROW)], "cpdag", ValueError),
        ([(0, 1, 0, ARROW)], "pag", ValueError),
        ([(0, 4, TAIL, ARROW)], "cpdag", IndexError),
        ([(-1, 2, TAIL, ARROW)], "cpdag", IndexError),
    ],
)
def test_edge_structure_refuses_what_the_matrix_refuses(edges, kind, error):
    with pytest.raises(error):
        _adapt.edge_structure(4, edges, kind=kind)


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
