"""Reference-equivalence and determinism tests for the graph types."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from andrey.core import EDGE_DTYPE, GraphStructure, Structure, StructureOutput, TemporalStructure
from andrey.core.structure import (
    ARROW,
    CIRCLE,
    NULL,
    TAIL,
    directed_support,
    endpoint_marks,
    has_cycle,
    topological_order,
)


def directed(pairs, n, kind="dag", labels=None):
    """A ``GraphStructure`` from directed edges ``i -> j`` (mark TAIL at i, ARROW at j)."""
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j] = TAIL
        M[j, i] = ARROW
    return GraphStructure.from_numpy(M, kind=kind, labels=labels)


def test_structure_base_is_abstract():
    with pytest.raises(TypeError):
        Structure(_n_nodes=3)


def test_from_numpy_to_numpy_roundtrip_is_exact():
    M = np.zeros((4, 4), dtype=np.int8)
    M[0, 1], M[1, 0] = TAIL, ARROW  # 0 -> 1
    M[2, 3], M[3, 2] = ARROW, ARROW  # 2 <-> 3 (bidirected -> pag)
    M[0, 2], M[2, 0] = CIRCLE, ARROW  # 0 o-> 2
    g = GraphStructure.from_numpy(M)
    assert g.kind == "pag"
    assert np.array_equal(g.to_numpy(), M)
    assert np.array_equal(g.adjacency, M)
    assert g.to_numpy().dtype == np.int8
    assert np.array_equal(g.to_numpy(dtype=np.int64), M.astype(np.int64))


@pytest.mark.parametrize(
    "build,expected_kind",
    [
        (lambda: directed([(0, 1), (1, 2)], 3), "dag"),
        (lambda: GraphStructure.from_numpy(_undirected([(0, 1)], 3)), "cpdag"),
    ],
)
def test_kind_inference(build, expected_kind):
    assert build().kind == expected_kind


def _undirected(pairs, n):
    M = np.zeros((n, n), dtype=np.int8)
    for i, j in pairs:
        M[i, j] = TAIL
        M[j, i] = TAIL
    return M


def test_kind_override_beats_inference():
    # An all-directed matrix inferred as dag can be tagged cpdag by the producing algorithm.
    g = directed([(0, 1)], 2, kind="cpdag")
    assert g.kind == "cpdag"


def test_isolated_nodes_preserved():
    g = directed([(0, 1)], 5)  # nodes 2,3,4 isolated
    assert g.n_nodes == 5
    assert np.array_equal(g.to_numpy(), directed([(0, 1)], 5).to_numpy())


def test_edges_roundtrip():
    g = directed([(0, 1), (1, 2), (0, 3)], 4)
    edges = g.to_edges()
    assert edges.dtype.names == ("i", "j", "mark_i", "mark_j")
    assert GraphStructure.from_edges(edges, n_nodes=4) == g


@pytest.mark.parametrize("pair", [(0, -1), (-1, 0), (0, 3), (3, 0)])
def test_from_edges_rejects_a_node_outside_the_graph(pair):
    """A negative index would wrap to the last node and build a different graph."""
    edges = np.array([(*pair, TAIL, ARROW)], dtype=EDGE_DTYPE)
    with pytest.raises(IndexError, match="outside range"):
        GraphStructure.from_edges(edges, n_nodes=3)


def test_networkx_roundtrip_dag_and_pag():
    g = directed([(0, 1), (2, 1)], 3)
    G = g.to_networkx()
    assert type(G).__name__ == "DiGraph"
    assert GraphStructure.from_networkx(G) == g

    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # bidirected
    M[1, 2], M[2, 1] = CIRCLE, CIRCLE  # o-o
    pag = GraphStructure.from_numpy(M)
    P = pag.to_networkx()
    assert type(P).__name__ == "MultiDiGraph"
    assert GraphStructure.from_networkx(P) == pag


def test_networkx_digraph_two_cycle_and_self_loop_roundtrip():
    import networkx as nx

    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # 2-cycle 0 <=> 1 (two directed edges, not bidirected)
    M[1, 2], M[2, 1] = TAIL, ARROW  # directed 1 -> 2
    M[0, 0] = ARROW  # autoregressive self-loop 0 -> 0
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    G = g.to_networkx()
    assert type(G).__name__ == "DiGraph"  # a digraph stays a plain DiGraph, not a MultiDiGraph
    assert G.has_edge(0, 1) and G.has_edge(1, 0)  # both directions of the 2-cycle are real edges
    assert G.has_edge(0, 0)  # self-loop present
    assert not nx.is_directed_acyclic_graph(G)
    assert GraphStructure.from_networkx(G, kind="digraph") == g


def test_networkx_digraph_three_cycle_is_topology_faithful():
    import networkx as nx

    # a 3-cycle 0->1->2->0 (includes the backward edge 2->0, arrowhead at the lower index): each
    # edge must orient by its marks so nx sees the cycle. Index-order emission renders it acyclic.
    g = directed([(0, 1), (1, 2), (2, 0)], 3, kind="digraph")
    G = g.to_networkx()
    assert type(G).__name__ == "DiGraph"
    assert set(G.edges()) == {(0, 1), (1, 2), (2, 0)}  # oriented by marks, not by index order
    assert not nx.is_directed_acyclic_graph(G)
    assert GraphStructure.from_networkx(G, kind="digraph") == g


def test_from_networkx_bare_reciprocal_pair_is_a_two_cycle():
    import networkx as nx

    # a bare DiGraph (no endpoints attrs) carrying both 0->1 and 1->0 imports as a 2-cycle, not a
    # single edge collapsed by last-writer-wins.
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 0)])
    g = GraphStructure.from_networkx(G, kind="digraph")
    assert g.endpoints(0, 1) == (ARROW, ARROW)
    g.validate()


def test_digraph_edges_roundtrip():
    # the documented to_edges / from_edges inverse for a digraph 2-cycle + self-loop; kind is passed
    # explicitly (a bare edge-list infers pag for a double arrowhead, exactly like from_networkx).
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # 2-cycle 0 <=> 1
    M[1, 2], M[2, 1] = TAIL, ARROW  # directed 1 -> 2
    M[0, 0] = ARROW  # autoregressive self-loop 0 -> 0
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    assert GraphStructure.from_edges(g.to_edges(), n_nodes=3, kind="digraph") == g


# ---- kind-aware compact mark storage (internal) -------------------------------------------------


def _sym_twin(g):
    """A byte-for-byte symmetric-layout twin of ``g`` (same graph, forced ``_layout='sym'``)."""
    from andrey.core.structure import _dense_to_csr, _frozen

    indptr, indices, marks = _dense_to_csr(g.to_numpy())
    return GraphStructure(
        _n_nodes=g.n_nodes,
        _labels=g.labels,
        _indptr=_frozen(indptr, np.int64),
        _indices=_frozen(indices, np.int32),
        _marks=_frozen(marks, np.int8),
        _kind=g.kind,
        _layout="sym",
        _node_types=g._node_types,
    )


def test_compact_layout_is_view_identical_to_symmetric():
    sl = np.zeros((3, 3), np.int8)
    sl[0, 0] = ARROW  # self-loop 0 -> 0
    sl[2, 1], sl[1, 2] = ARROW, TAIL  # directed 1 -> 2
    fixtures = [
        directed([(0, 1), (1, 2), (2, 0)], 3),  # dag, directed cycle-topology (backward edges)
        directed([(2, 0), (2, 1)], 3),  # dag fork
        GraphStructure.from_numpy(sl, kind="dag", allow_self_loops=True),  # dag + self-loop
        _undirected_graph([(0, 1), (1, 2)], 3, kind="cpdag"),  # cpdag, all undirected
        GraphStructure.from_numpy(  # cpdag mixed directed + undirected
            np.array([[0, 1, 0], [2, 0, 1], [0, 1, 0]], np.int8), kind="cpdag"
        ),
        directed([], 4),  # empty (isolated nodes)
    ]
    for g in fixtures:
        twin = _sym_twin(g)
        assert g._layout == "compact", g.kind  # dag/cpdag halve the store
        assert g._indices.size <= twin._indices.size  # never larger than symmetric
        # every public view is byte-identical to the symmetric twin.
        assert np.array_equal(g.to_numpy(), twin.to_numpy())
        assert np.array_equal(g.to_edges(), twin.to_edges())
        assert np.array_equal(g.to_scipy_sparse().toarray(), twin.to_scipy_sparse().toarray())
        for i in range(g.n_nodes):
            assert np.array_equal(g.neighbors(i), twin.neighbors(i))
            assert g.neighbors(i).dtype == np.int32
            for j in range(g.n_nodes):
                assert g.endpoints(i, j) == twin.endpoints(i, j)
        g_nx, t_nx = g.to_networkx(), twin.to_networkx()
        assert type(g_nx).__name__ == type(t_nx).__name__
        assert set(g_nx.edges()) == set(t_nx.edges())
        g.validate()
        assert g == twin and twin == g  # layout-normalizing equality


def test_pag_and_digraph_stay_symmetric():
    pag = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="pag")
    di = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="digraph")
    assert pag._layout == "sym" and di._layout == "sym"  # per-endpoint marks needed, no reduction


def test_compact_neighbors_are_read_only():
    g = directed([(0, 1), (1, 2)], 3)  # dag -> compact
    assert g._layout == "compact"
    with pytest.raises(ValueError):
        g.neighbors(1)[0] = 9  # the reconstructed neighbors array is a fresh read-only copy


def test_to_scipy_sparse_read_only_across_layouts():
    # to_scipy_sparse must not leak the layout: both a compact (dag) and a symmetric (digraph) store
    # hand back read-only buffers, so mutating them raises regardless of how the marks are stored.
    compact = directed([(0, 1), (1, 2)], 3)  # dag -> compact
    sym = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="digraph")  # sym
    assert compact._layout == "compact" and sym._layout == "sym"
    for g in (compact, sym):
        with pytest.raises(ValueError):
            g.to_scipy_sparse().data[:] = 9


def test_degenerate_store_stays_symmetric():
    # a store validate() would reject must NOT compact -- compaction would silently canonicalize it.
    half = GraphStructure._from_csr(2, indptr=[0, 1, 1], indices=[1], marks=[TAIL], kind="dag")
    assert half._layout == "sym"  # an asymmetric half-edge is left symmetric for validate() to flag
    with pytest.raises(ValueError, match="asymmetric"):
        half.validate()


def test_compact_store_serializes_symmetric_at_v0(tmp_path):
    import json

    from andrey.core import StructureOutput

    g = directed([(0, 1), (1, 2)], 3)  # dag -> compact (nnz 2), symmetric on disk (4 marks)
    out = StructureOutput.new(g)
    for fmt, name in (("json", "g.json"), ("npz", "g.npz")):
        out.save(tmp_path / name, fmt=fmt)
        assert StructureOutput.load(tmp_path / name) == out  # re-compacts on load
    doc = json.loads((tmp_path / "g.json").read_text())
    assert doc["format_version"] == 0  # the compact store never touches the on-disk format
    assert len(doc["marks"]) == 4  # symmetric: 2 edges x 2 endpoints, not the halved 2


def test_networkx_cpdag_multigraph_and_asymmetric_marks():
    # cpdag undirected -> DiGraph carrying (TAIL, TAIL); multigraph=True forces MultiDiGraph.
    cpdag = _undirected_graph([(0, 1)], 2, kind="cpdag")
    assert type(cpdag.to_networkx()).__name__ == "DiGraph"
    assert type(cpdag.to_networkx(multigraph=True)).__name__ == "MultiDiGraph"
    assert GraphStructure.from_networkx(cpdag.to_networkx(multigraph=True)) == cpdag

    # Asymmetric compound mark o-> (CIRCLE, ARROW) must round-trip through networkx.
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    pag = GraphStructure.from_numpy(M, kind="pag")
    assert GraphStructure.from_networkx(pag.to_networkx()) == pag


def test_networkx_labels_roundtrip():
    g = directed([(0, 1)], 2, labels=("A", "B"))
    G = g.to_networkx()
    assert list(G.nodes()) == ["A", "B"]
    back = GraphStructure.from_networkx(G)
    assert back == g
    assert back.labels == ("A", "B")


# Every mark pair on edge 0-1, and the (source, target, type) row it reads as.
ORIENTED = [
    (TAIL, ARROW, (0, 1, "directed")),
    (ARROW, TAIL, (1, 0, "directed")),
    (CIRCLE, ARROW, (0, 1, "partially_directed")),
    (ARROW, CIRCLE, (1, 0, "partially_directed")),
    (TAIL, CIRCLE, (0, 1, "partially_undirected")),
    (CIRCLE, TAIL, (1, 0, "partially_undirected")),
    (TAIL, TAIL, (0, 1, "undirected")),
    (ARROW, ARROW, (0, 1, "bidirected")),
    (CIRCLE, CIRCLE, (0, 1, "circle")),
]


@pytest.mark.parametrize(("mark_0", "mark_1", "row"), ORIENTED)
def test_oriented_edges_follow_the_marks(mark_0, mark_1, row):
    M = np.array([[0, mark_0], [mark_1, 0]])
    assert GraphStructure.from_numpy(M, kind="pag").oriented_edges() == [row]
    named = GraphStructure.from_numpy(M, kind="pag", labels=("a", "b"))
    source, target, edge_type = row
    assert named.oriented_edges() == [("ab"[source], "ab"[target], edge_type)]
    assert named.oriented_edges(index=True) == [row]


def test_oriented_edges_ignore_node_order():
    g = directed([(2, 0), (2, 1)], 3, labels=("wet", "slippery", "rain"))
    assert g.oriented_edges() == [("rain", "wet", "directed"), ("rain", "slippery", "directed")]
    # to_networkx keeps index order, so the arc for rain -> wet runs (wet, rain).
    assert list(g.to_networkx().edges) == [("wet", "rain"), ("slippery", "rain")]


def test_oriented_edges_self_loop_is_directed():
    M = np.array([[ARROW, TAIL], [ARROW, 0]])  # X(t-1) -> X(t), and 0 -> 1
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    assert g.oriented_edges() == [(0, 0, "directed"), (0, 1, "directed")]


def test_digraph_two_cycles_are_directed_edges():
    M = np.array([[ARROW, ARROW, 0], [ARROW, 0, ARROW], [0, TAIL, 0]], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="digraph", labels=("a", "b", "c"), allow_self_loops=True)
    assert g.oriented_edges() == [
        ("a", "a", "directed"),
        ("a", "b", "directed"),
        ("b", "a", "directed"),
        ("c", "b", "directed"),
    ]
    assert g.oriented_edges(index=True) == [
        (0, 0, "directed"),
        (0, 1, "directed"),
        (1, 0, "directed"),
        (2, 1, "directed"),
    ]
    assert repr(g).splitlines() == [
        "digraph  |  3 nodes  |  4 edges (4 directed)",
        "",
        "  a -> a",
        "  a -> b",
        "  b -> a",
        "  c -> b",
    ]
    assert {(s, t) for s, t, _ in g.oriented_edges()} == set(g.to_networkx().edges)


@pytest.mark.parametrize("limit", [0, 1, 2, 30, None])
def test_digraph_two_cycles_count_omitted_directions(limit):
    M = np.zeros((17, 17), dtype=np.int8)
    M[0, 1:] = M[1:, 0] = ARROW
    g = GraphStructure.from_numpy(M, kind="digraph")
    edges = [edge for j in range(1, 17) for edge in (f"0 -> {j}", f"{j} -> 0")]
    shown = edges[:limit]
    expected = ["digraph  |  17 nodes  |  32 edges (32 directed)", ""]
    expected += [f"  {edge}" for edge in shown]
    if len(shown) < 32:
        expected += [f"  ... and {32 - len(shown)} more"]
    assert g._summary(limit=limit) == expected
    if limit == 30:
        assert repr(g).splitlines() == expected


def test_temporal_two_cycles_and_summary_graph():
    M = np.array([[0, ARROW], [ARROW, 0]], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="digraph", labels=("x", "y"))
    temporal = TemporalStructure.from_lag_graphs([g], lags=[1], labels=("x", "y"))
    assert repr(temporal).splitlines() == [
        "temporal  |  2 nodes  |  1 lags",
        "",
        "  lag 1  |  2 edges (2 directed)",
        "    x -> y",
        "    y -> x",
    ]
    assert repr(temporal.summary_graph()) == repr(g)


def test_repr_lists_the_edges_by_name():
    g = directed([(0, 2), (1, 2), (2, 3)], 4, labels=("rain", "sprinkler", "wet", "slippery"))
    assert repr(g) == (
        "dag  |  4 nodes  |  3 edges (3 directed)\n"
        "\n"
        "  rain -> wet\n"
        "  sprinkler -> wet\n"
        "  wet -> slippery"
    )
    assert str(g) == repr(g)  # print() shows the same text


def test_repr_uses_each_mark_pair_symbol_and_counts_types_in_order():
    M = np.zeros((6, 6), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW  # 0 o-> 1
    M[2, 3], M[3, 2] = ARROW, ARROW  # 2 <-> 3
    M[4, 5], M[5, 4] = CIRCLE, CIRCLE  # 4 o-o 5
    M[0, 5], M[5, 0] = TAIL, CIRCLE  # 0 -o 5
    lines = repr(GraphStructure.from_numpy(M, kind="pag")).splitlines()
    assert lines[0] == (
        "pag  |  6 nodes  |  4 edges "
        "(1 partially_directed, 1 partially_undirected, 1 bidirected, 1 circle)"
    )
    assert lines[2:] == ["  0 o-> 1", "  0 -o 5", "  2 <-> 3", "  4 o-o 5"]


def test_repr_lists_at_most_thirty_edges():
    g = directed([(0, j) for j in range(1, 36)], 36)
    lines = repr(g).splitlines()
    assert lines[0] == "dag  |  36 nodes  |  35 edges (35 directed)"
    assert lines[2:32] == [f"  0 -> {j}" for j in range(1, 31)]
    assert lines[32:] == ["  ... and 5 more"]


def test_repr_of_an_empty_graph_and_latent_nodes():
    assert repr(directed([], 2)) == "dag  |  2 nodes  |  0 edges (no edges)"
    M = np.zeros((3, 3), dtype=np.int8)
    M[2, 0], M[0, 2] = TAIL, ARROW  # the latent L1 -> a
    types = np.array([0, 0, 1], dtype=np.int8)
    g = GraphStructure.from_numpy(M, kind="dag", labels=("a", "b", "L1"), node_types=types)
    assert repr(g).splitlines()[-1] == "  latent: L1"


def test_repr_of_a_temporal_stack_and_its_summary_graph():
    lag0 = directed([(0, 1)], 2, labels=("x", "y"))
    loop = np.array([[ARROW, TAIL], [ARROW, 0]], dtype=np.int8)  # x(t-1) -> x(t), x(t-1) -> y(t)
    lag1 = GraphStructure.from_numpy(loop, kind="digraph", labels=("x", "y"), allow_self_loops=True)
    t = TemporalStructure.from_lag_graphs([lag0, lag1], labels=("x", "y"))
    assert repr(t) == (
        "temporal  |  2 nodes  |  2 lags\n"
        "\n"
        "  lag 0  |  1 edges (1 directed)\n"
        "    x -> y\n"
        "\n"
        "  lag 1  |  2 edges (2 directed)\n"
        "    x -> x\n"
        "    x -> y"
    )
    assert repr(t.summary_graph()).startswith("digraph  |  2 nodes  |  2 edges (2 directed)")


def test_repr_needs_a_summary_from_each_structure():
    class Bare(Structure):
        type = "graph"

    with pytest.raises(NotImplementedError, match="Bare"):
        repr(Bare(_n_nodes=1))


def test_type_counts_match_the_oriented_edges():
    from collections import Counter

    rng = np.random.default_rng(0)
    for _ in range(25):
        M = np.zeros((10, 10), dtype=np.int8)
        for i in range(10):
            for j in range(i + 1, 10):
                if rng.random() < 0.3:
                    M[i, j], M[j, i] = rng.integers(1, 4, size=2)
        g = GraphStructure.from_numpy(M, kind="pag")
        assert g._type_counts() == dict(Counter(t for *_, t in g.oriented_edges()))
        assert list(zip(*(a.tolist() for a in g._mark_pairs()))) == list(g._iter_edges())


def test_bare_digraph_defaults_to_arrows():
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge(0, 1)  # no endpoints attribute
    g = GraphStructure.from_networkx(G)
    assert g.endpoints(0, 1) == (TAIL, ARROW)


def test_neighbors_and_endpoints():
    g = directed([(0, 1), (0, 2)], 3)
    assert set(g.neighbors(0).tolist()) == {1, 2}
    assert g.endpoints(0, 1) == (TAIL, ARROW)
    assert g.endpoints(1, 0) == (ARROW, TAIL)
    assert g.endpoints(1, 2) == (NULL, NULL)


def test_to_scipy_sparse_matches_dense():
    g = directed([(0, 1), (1, 2)], 3)
    assert np.array_equal(g.to_scipy_sparse().toarray(), g.to_numpy())


def test_validate_accepts_valid_and_symmetric():
    directed([(0, 1), (1, 2)], 3).validate()
    _undirected_graph([(0, 1)], 2, kind="cpdag").validate()


def _undirected_graph(pairs, n, kind):
    return GraphStructure.from_numpy(_undirected(pairs, n), kind=kind)


def test_validate_rejects_circle_outside_pag():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    with pytest.raises(ValueError, match="CIRCLE"):
        GraphStructure.from_numpy(M, kind="cpdag").validate()


def test_validate_rejects_bidirected_outside_pag():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW
    with pytest.raises(ValueError, match="bidirected"):
        GraphStructure.from_numpy(M, kind="cpdag").validate()


def test_validate_rejects_undirected_in_dag():
    with pytest.raises(ValueError, match="undirected"):
        _undirected_graph([(0, 1)], 2, kind="dag").validate()


def test_validate_accepts_digraph_two_cycle_and_self_loop():
    # kind="digraph" reads ARROW/ARROW as two directed edges (a 2-cycle), not a bidirected PAG edge,
    # and carries autoregressive self-loops -- both are legal where dag/cpdag would reject them.
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW  # 2-cycle 0 <=> 1
    M[2, 2] = ARROW  # self-loop 2 -> 2
    g = GraphStructure.from_numpy(M, kind="digraph", allow_self_loops=True)
    g.validate()
    assert g.kind == "digraph"


def test_validate_rejects_two_cycle_outside_pag_or_digraph():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = ARROW, ARROW
    with pytest.raises(ValueError, match="digraph"):  # a dag double arrowhead names both escapes
        GraphStructure.from_numpy(M, kind="dag").validate()


def test_validate_rejects_undirected_in_digraph():
    with pytest.raises(ValueError, match="undirected"):
        _undirected_graph([(0, 1)], 2, kind="digraph").validate()


def test_an_undirected_graph_holds_undirected_edges_only():
    graph = _undirected_graph([(0, 1), (1, 2)], 3, kind="undirected")
    graph.validate()
    assert graph.oriented_edges() == [(0, 1, "undirected"), (1, 2, "undirected")]
    assert graph == GraphStructure.from_numpy(graph.to_numpy(), kind="undirected")
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = TAIL, ARROW
    with pytest.raises(ValueError, match="arrowhead"):
        GraphStructure.from_numpy(M, kind="undirected").validate()


def test_validate_rejects_circle_in_digraph():
    M = np.zeros((2, 2), dtype=np.int8)
    M[0, 1], M[1, 0] = CIRCLE, ARROW
    with pytest.raises(ValueError, match="CIRCLE"):
        GraphStructure.from_numpy(M, kind="digraph").validate()


def test_from_numpy_rejects_bad_input():
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.zeros((2, 3), dtype=np.int8))  # non-square
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.array([[TAIL, 0], [0, 0]], dtype=np.int8))  # self-loop
    with pytest.raises(ValueError):
        GraphStructure.from_numpy(np.array([[0, 9], [0, 0]], dtype=np.int8))  # out-of-range code
    with pytest.raises(ValueError):
        # negative endpoint codes are invalid
        GraphStructure.from_numpy(np.array([[0, -1], [ARROW, 0]], dtype=np.int8))


def test_from_numpy_rejects_asymmetric_support():
    # (0,1) has a mark but (1,0) does not - a malformed one-sided edge from a buggy remap.
    with pytest.raises(ValueError, match="symmetric"):
        GraphStructure.from_numpy(np.array([[0, TAIL], [0, 0]], dtype=np.int8))


def test_stored_arrays_are_read_only():
    g = GraphStructure.from_numpy(np.array([[0, 2], [2, 0]], np.int8), kind="digraph")
    assert g._layout == "sym"
    with pytest.raises(ValueError):
        g.neighbors(0)[0] = 5  # neighbors() is a view into the frozen store
    with pytest.raises(ValueError):
        g._marks[0] = 9


def test_labels_length_checked():
    with pytest.raises(ValueError):
        directed([(0, 1)], 2, labels=("only-one",))


@pytest.mark.parametrize(
    ("names", "labels"),
    [((1, 0, 2), ("1", "0", "2")), ((None, "y", "z, ÿ→é"), ("None", "y", "z, ÿ→é"))],
)
def test_labels_are_text_through_every_reader_and_writer(tmp_path, names, labels):
    g = directed([(0, 1), (2, 1)], 3, labels=names)
    assert g.labels == labels
    out = StructureOutput.new(g)
    out.save(tmp_path / "g.json")
    out.save(tmp_path / "g.npz", fmt="npz")
    backs = [
        GraphStructure.from_networkx(g.to_networkx()),
        GraphStructure.from_edges(g.to_edges(), n_nodes=3, labels=names),
        TemporalStructure.from_lag_graphs([g], labels=names).lag(0),
        StructureOutput.load(tmp_path / "g.json").structure,
        StructureOutput.load(tmp_path / "g.npz").structure,
    ]
    assert all(back == g for back in backs)


def test_a_saved_file_with_numeric_labels_loads_them_as_text(tmp_path):
    path = tmp_path / "g.json"
    StructureOutput.new(directed([(0, 1)], 2, labels=("a", "b"))).save(path)
    doc = json.loads(path.read_text())
    path.write_text(json.dumps({**doc, "labels": [1, 0]}))
    assert StructureOutput.load(path).structure.labels == ("1", "0")


def _networkx_nodes(*nodes):
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    return G


@pytest.mark.parametrize(
    "build",
    [
        lambda: directed([(0, 1)], 3, labels=(1, "1", "z")),
        lambda: GraphStructure.from_networkx(_networkx_nodes(1, "1", "z")),
        lambda: TemporalStructure.from_lag_graphs([directed([(0, 1)], 3)], labels=("1", "1", "z")),
        lambda: dataclasses.replace(
            directed([(0, 1)], 3, labels=("a", "b", "c")), _labels=("1", 1, "z")
        ),
    ],
    ids=["from_numpy", "from_networkx", "from_lag_graphs", "replace"],
)
def test_labels_equal_as_text_raise(build):
    with pytest.raises(ValueError, match=r"'1' names nodes \[0, 1\]"):
        build()


def test_equality_and_unhashable():
    a = directed([(0, 1)], 3)
    b = directed([(0, 1)], 3)
    c = directed([(1, 0)], 3)
    assert a == b
    assert a != c
    assert a.__hash__ is None
    with pytest.raises(TypeError):
        {a}  # unhashable


def test_graph_not_equal_temporal():
    g = directed([(0, 1)], 2)
    t = TemporalStructure(_n_nodes=2, _lags=(g,))
    assert (g == t) is False
    assert (t == g) is False


def test_temporal_structure():
    g0 = directed([(0, 1)], 2)
    g1 = directed([], 2)
    t = TemporalStructure(_n_nodes=2, _lags=(g0, g1))
    assert t.n_lags == 2
    assert t.lag(0) == g0
    assert t.type == "temporal"
    assert t == TemporalStructure(_n_nodes=2, _lags=(g0, g1))
    with pytest.raises(ValueError):  # a lag with a mismatched node count
        TemporalStructure(_n_nodes=3, _lags=(g0,))


def test_to_numpy_is_deterministic():
    g = directed([(0, 1), (1, 2), (2, 0)], 3, kind="cpdag")
    first = g.to_numpy()
    for _ in range(3):
        assert np.array_equal(g.to_numpy(), first)


def test_node_types_latent_flagging(tmp_path):
    """Optional per-node types flag latents; they persist through equality and json."""
    from andrey.core import LATENT, OBSERVED, StructureOutput

    # a 3-node dag where L2 (latent) causes the two observed X0, X1.
    g = directed([(2, 0), (2, 1)], 3, kind="dag", labels=("X0", "X1", "L2"))
    node_types = np.array([OBSERVED, OBSERVED, LATENT], dtype=np.int8)
    flagged = GraphStructure.from_numpy(
        g.to_numpy(), kind="dag", labels=("X0", "X1", "L2"), node_types=node_types
    )
    assert np.array_equal(flagged.node_types, [OBSERVED, OBSERVED, LATENT])
    # an all-observed graph reports None and is not equal to a latent-flagged one.
    assert g.node_types is None
    assert flagged != g
    # a wrong-length node_types is rejected.
    with pytest.raises(ValueError, match="node_types"):
        GraphStructure.from_numpy(g.to_numpy(), kind="dag", node_types=np.array([0, 1], np.int8))
    # the flags round-trip through both json and npz (a graph structure supports both).
    out = StructureOutput.new(flagged)
    for fmt, fname in (("json", "gin.json"), ("npz", "gin.npz")):
        p = tmp_path / fname
        out.save(p, fmt=fmt)
        assert StructureOutput.load(p) == out


def test_self_loop_round_trips_through_inverses():
    """A self-loop graph survives ``from_edges`` / ``from_networkx`` (its documented inverses)."""
    M = np.zeros((3, 3), dtype=np.int8)
    M[0, 0] = ARROW  # autoregressive self-loop 0 -> 0
    M[2, 1], M[1, 2] = ARROW, TAIL  # directed edge 1 -> 2 (arrow at 2, tail at 1)
    g = GraphStructure.from_numpy(M, kind="dag", allow_self_loops=True)
    assert np.array_equal(GraphStructure.from_edges(g.to_edges(), n_nodes=3).to_numpy(), M)
    assert np.array_equal(GraphStructure.from_networkx(g.to_networkx()).to_numpy(), M)


def test_from_numpy_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        GraphStructure.from_numpy(np.array([[0, TAIL], [ARROW, 0]]), kind="foo")


@pytest.mark.parametrize("mark", [TAIL, CIRCLE])
def test_a_self_loop_is_marked_with_an_arrowhead(mark):
    with pytest.raises(ValueError, match="self-loop is marked 2"):
        GraphStructure.from_numpy(np.diag([mark, 0]), kind="digraph", allow_self_loops=True)


def test_a_self_loop_reads_the_same_in_the_lag_graph_and_the_summary():
    empty = GraphStructure.from_numpy(np.zeros((2, 2), int))
    loop = GraphStructure.from_numpy(np.diag([ARROW, 0]), kind="digraph", allow_self_loops=True)
    summary = TemporalStructure.from_lag_graphs([empty, loop]).summary_graph()
    assert (
        loop.oriented_edges(index=True)
        == summary.oriented_edges(index=True)
        == [(0, 0, "directed")]
    )


# --- pcalg's amat -----------------------------------------------------------------------------
# pcalg's amat.pag puts the mark of edge a-b at the column node, in amat[a, b], coded 0 none,
# 1 circle, 2 arrowhead, 3 tail; its amat.cpdag writes a -> b at the child's row, amat[b, a] = 1
# (pcalg's amatType help page).

PCALG_PAG_CODE = {TAIL: 3, ARROW: 2, CIRCLE: 1}


@pytest.mark.parametrize(("mark_0", "mark_1", "row"), ORIENTED)
def test_pcalg_pag_puts_each_mark_at_its_column_node(mark_0, mark_1, row):
    M = np.array([[0, mark_0], [mark_1, 0]], dtype=np.int8)
    pag = GraphStructure.from_numpy(M, kind="pag")
    amat = np.array([[0, PCALG_PAG_CODE[mark_1]], [PCALG_PAG_CODE[mark_0], 0]])
    assert np.array_equal(pag.to_pcalg(), amat)
    assert GraphStructure.from_pcalg(amat, "pag") == pag
    assert GraphStructure.from_pcalg(amat, "pag").oriented_edges() == [row]


@pytest.mark.parametrize("kind", ["dag", "cpdag"])
def test_pcalg_cpdag_writes_a_directed_edge_at_the_childs_row(kind):
    # 0 -> 2 <- 1, and 2 -- 3 when the kind allows an undirected edge.
    pairs = [(0, 2), (1, 2)]
    g = directed(pairs, 4, kind=kind)
    amat = np.zeros((4, 4), dtype=np.int8)
    amat[2, 0] = amat[2, 1] = 1
    if kind == "cpdag":
        g = GraphStructure.from_numpy(g.to_numpy() + _undirected([(2, 3)], 4), kind=kind)
        amat[2, 3] = amat[3, 2] = 1
    assert np.array_equal(g.to_pcalg(), amat) and g.to_pcalg().dtype == np.int8
    assert GraphStructure.from_pcalg(amat, kind) == g


def _random_graph(kind, seed, n=6):
    """Edges between random pairs, each of a type the kind allows, with labels."""
    rng = np.random.default_rng(seed)
    allowed = {
        "dag": [(TAIL, ARROW)],
        "cpdag": [(TAIL, ARROW), (TAIL, TAIL)],
        "pag": [(a, b) for a in (TAIL, ARROW, CIRCLE) for b in (TAIL, ARROW, CIRCLE)],
    }[kind]
    M = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                M[i, j], M[j, i] = allowed[rng.integers(len(allowed))]
    return GraphStructure.from_numpy(M, kind=kind, labels=tuple("abcdef"[:n]))


@pytest.mark.parametrize("kind", ["dag", "cpdag", "pag"])
@pytest.mark.parametrize("seed", range(3))
def test_pcalg_amat_round_trips_with_labels(kind, seed):
    g = _random_graph(kind, seed)
    amat = g.to_pcalg()
    assert amat.shape == (6, 6) and amat.dtype == np.int8
    back = GraphStructure.from_pcalg(amat, kind, labels=g.labels)
    assert back == g and back.labels == g.labels
    assert np.array_equal(back.to_pcalg(), amat)


def test_pcalg_amat_from_r_may_be_float_or_bool():
    # rpy2 hands R's numeric matrix over as float64, and a logical one as bool.
    amat = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    want = GraphStructure.from_pcalg(amat, "cpdag")
    assert GraphStructure.from_pcalg(amat.astype(float), "cpdag") == want
    assert GraphStructure.from_pcalg(amat.astype(bool), "cpdag") == want


@pytest.mark.parametrize(
    ("amat", "kind", "message"),
    [
        (np.zeros((2, 3)), "pag", "square"),
        (np.array([[0, 2], [0, 0]]), "cpdag", "codes"),  # a PAG code in a CPDAG's matrix
        (np.array([[0, 4], [4, 0]]), "pag", "codes"),
        (np.array([[0, 0.5], [0, 0]]), "cpdag", "codes"),
        (np.array([[0, 2], [0, 0]]), "pag", "symmetric"),  # a mark at one end only
        (np.array([[1, 0], [0, 0]]), "cpdag", "diagonal"),
        (np.array([[0, 1], [1, 0]]), "dag", "undirected"),  # an undirected edge in a DAG
        (np.array([[0, 1], [0, 0]]), "digraph", "kind"),
    ],
)
def test_from_pcalg_rejects_what_pcalg_cannot_have_written(amat, kind, message):
    with pytest.raises(ValueError, match=message):
        GraphStructure.from_pcalg(amat, kind)


@pytest.mark.parametrize(
    "graph",
    [
        directed([(0, 1), (1, 0)], 2, kind="digraph"),  # a 2-cycle
        GraphStructure.from_numpy(np.array([[0, CIRCLE], [ARROW, 0]]), kind="cpdag"),
        GraphStructure.from_numpy(np.array([[0, ARROW], [ARROW, 0]]), kind="dag"),
        GraphStructure.from_numpy(
            np.array([[ARROW, 0], [0, 0]]), kind="pag", allow_self_loops=True
        ),
    ],
    ids=["digraph", "circle in a cpdag", "double arrowhead in a dag", "self-loop"],
)
def test_to_pcalg_rejects_a_graph_pcalg_cannot_write(graph):
    with pytest.raises(ValueError):
        graph.to_pcalg()


def test_endpoint_marks_end_a_pair_joined_both_ways_and_a_self_loop_in_arrowheads():
    support = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=bool)  # 0 <-> 1, 0 -> 2, 0 -> 0
    marks = endpoint_marks(support)
    assert marks.dtype == np.int8
    assert (marks[0, 1], marks[1, 0]) == (ARROW, ARROW)
    assert (marks[0, 2], marks[2, 0]) == (TAIL, ARROW)
    assert marks[0, 0] == ARROW
    assert np.array_equal(directed_support(marks), support)


def test_has_cycle_finds_a_directed_cycle_and_a_self_loop():
    dag = np.zeros((3, 3), dtype=bool)
    dag[0, 1] = dag[1, 2] = dag[0, 2] = True
    assert not has_cycle(dag)
    loop, cycle = dag.copy(), dag.copy()
    loop[1, 1] = cycle[2, 0] = True
    assert has_cycle(loop) and has_cycle(cycle)


def test_topological_order_takes_the_lowest_key_among_the_ready_nodes():
    adj = np.zeros((4, 4), dtype=bool)
    adj[3, 1] = adj[2, 0] = True  # 3 -> 1, 2 -> 0
    assert topological_order(adj) == [2, 0, 3, 1]
    assert topological_order(adj, key=("d", "c", "b", "a")) == [3, 2, 1, 0]
    adj[1, 3] = True
    with pytest.raises(ValueError, match="the network has a directed cycle"):
        topological_order(adj, what="the network")


# ---- NumPy CSR results against SciPy and edge-by-edge references --------------------------------

_KINDS = ("dag", "cpdag", "pag", "mag", "digraph", "undirected")


def _random_marks(rng, n, kind, density, loops=False, valid=True):
    """A random ``(n, n)`` endpoint matrix with marks allowed for ``kind``, or any marks."""
    pairs = {
        "dag": [(TAIL, ARROW)],
        "cpdag": [(TAIL, ARROW), (TAIL, TAIL)],
        "undirected": [(TAIL, TAIL)],
        "digraph": [(TAIL, ARROW), (ARROW, ARROW)],
        "mag": [(TAIL, ARROW), (ARROW, ARROW), (TAIL, TAIL)],
    }.get(kind)
    M = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                if valid and pairs is not None:
                    a, b = pairs[int(rng.integers(len(pairs)))]
                    a, b = (a, b) if rng.random() < 0.5 else (b, a)
                else:
                    a, b = rng.integers(1, 4, size=2)
                M[i, j], M[j, i] = a, b
    if loops:
        M[np.diag_indices(n)] = np.where(rng.random(n) < 0.3, ARROW, NULL)
    return M


def _csr(M):
    """Reference CSR of ``M`` built by SciPy, with sorted columns and no stored zeros."""
    import scipy.sparse

    sp = scipy.sparse.csr_array(M)
    sp.sum_duplicates()
    sp.sort_indices()
    return sp.indptr.astype(np.int64), sp.indices.astype(np.int32), sp.data.astype(np.int8)


def _damage(rng, store, how):
    """A copy of the CSR ``store`` that is no longer canonical, or leaves ``_far_marks`` unsure."""
    indptr, indices, marks = (a.copy() for a in store)
    n, nnz = indptr.size - 1, indices.size
    if not nnz:
        return indptr, indices, marks
    k = int(rng.integers(nnz))
    row = int(np.searchsorted(indptr, k, side="right") - 1)
    if how == "half":  # a stored cell whose partner is missing
        indices, marks = np.delete(indices, k), np.delete(marks, k)
        indptr[row + 1 :] -= 1
    elif how == "null":  # an explicitly stored NULL
        marks[k] = NULL
    elif how == "duplicate":
        indices, marks = np.insert(indices, k, indices[k]), np.insert(marks, k, marks[k])
        indptr[row + 1 :] += 1
    elif how == "unsorted":
        lo, hi = indptr[row], indptr[row + 1]
        if hi - lo > 1:
            indices[lo:hi], marks[lo:hi] = indices[lo:hi][::-1].copy(), marks[lo:hi][::-1].copy()
    elif how == "out_of_range":
        indices[k] = n + int(rng.integers(3))
    elif how == "negative":
        indices[k] = -1
    return indptr, indices, marks


def _reference_try_compact(n, indptr, indices, marks, kind):
    """Reference for ``_try_compact``: the partner marks come from SciPy's transpose."""
    import scipy.sparse

    if kind not in ("dag", "cpdag", "undirected"):
        return None
    if (marks == NULL).any() or (marks == CIRCLE).any():
        return None
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    cols = indices.astype(np.int64)
    if bool(np.any((rows[1:] == rows[:-1]) & (np.diff(cols) <= 0))):
        return None
    sp = scipy.sparse.csr_array((marks, indices, indptr), shape=(n, n))
    spT = sp.T.tocsr()
    spT.sort_indices()
    if not (np.array_equal(sp.indptr, spT.indptr) and np.array_equal(sp.indices, spT.indices)):
        return None
    partner = spT.data.astype(np.int8)
    upper = (rows != cols) & (rows < cols)
    mi, mj = marks[upper], partner[upper]
    fwd, bwd = (mi == TAIL) & (mj == ARROW), (mi == ARROW) & (mj == TAIL)
    und = (mi == TAIL) & (mj == TAIL)
    ok = und if kind == "undirected" else fwd | bwd | (und if kind == "cpdag" else False)
    if not bool(np.all(ok)):
        return None
    off = rows != cols
    r_up, c_up = rows[upper], cols[upper]
    er = np.concatenate([np.where(bwd, c_up, r_up), rows[~off]])
    ec = np.concatenate([np.where(bwd, r_up, c_up), cols[~off]])
    em = np.concatenate([np.where(und, TAIL, ARROW).astype(np.int8), marks[~off]])
    order = np.lexsort((ec, er))
    new_indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(new_indptr, er[order] + 1, 1)
    np.cumsum(new_indptr, out=new_indptr)
    return new_indptr, ec[order].astype(np.int32), em[order].astype(np.int8)


def _reference_mark_pairs(g):
    """Reference for ``_mark_pairs``: SciPy looks up every partner mark."""
    import scipy.sparse

    indptr, indices, marks = g._sym_csr()
    rows = np.repeat(np.arange(g.n_nodes, dtype=np.intp), np.diff(indptr))
    cols = indices.astype(np.intp)
    keep = rows <= cols
    rows, cols, near = rows[keep], cols[keep], marks[keep].astype(np.int64)
    if not rows.size:
        return rows, cols, near, near.copy()
    sym = scipy.sparse.csr_array((marks, indices, indptr), shape=(g.n_nodes, g.n_nodes))
    return rows, cols, near, np.asarray(sym[cols, rows], dtype=np.int64).ravel()


def _reference_validate(g):
    """Reference for ``validate``: the message of the first edge-by-edge failure, else ``None``."""
    from andrey.core.structure import _lookup

    indptr, indices, marks = g._sym_csr()
    if g.kind != "pag" and (marks == CIRCLE).any():
        return f"CIRCLE marks require kind='pag', not '{g.kind}'"
    for i, j, mi, mj in g._iter_edges():
        if i == j:
            continue
        if g.kind in ("dag", "cpdag") and mi == ARROW and mj == ARROW:
            return (
                f"double arrowhead {i}<->{j} requires kind='pag' or 'mag' (bidirected) or "
                f"kind='digraph' (2-cycle)"
            )
        if g.kind in ("dag", "digraph") and mi == TAIL and mj == TAIL:
            return f"undirected edge {i}--{j} requires kind='cpdag', 'pag', 'mag', or 'undirected'"
        if g.kind == "undirected" and (mi, mj) != (TAIL, TAIL):
            return f"edge {i}-{j} has an arrowhead, which kind='undirected' forbids"
    rows = np.repeat(np.arange(g.n_nodes, dtype=np.intp), np.diff(indptr))
    for i, j in zip(rows.tolist(), indices.tolist()):
        if _lookup(indptr, indices, marks, j, i) == NULL:
            return f"asymmetric half-edge: ({i}, {j}) stored but ({j}, {i}) missing"
    return None


def _validation_message(g):
    try:
        g.validate()
    except ValueError as error:
        return str(error)
    return None


def test_dense_to_csr_equals_scipys_canonical_csr():
    from andrey.core.structure import _dense_to_csr

    rng = np.random.default_rng(0)
    for n in (0, 1, 2, 7, 40):
        for kind in _KINDS:
            for loops in (False, True):
                M = _random_marks(rng, n, kind, 0.3, loops, valid=False)
                got, want = _dense_to_csr(M), _csr(M)
                for a, b in zip(got, want):
                    assert a.dtype == b.dtype and np.array_equal(a, b)


def test_far_marks_find_each_partner_mark_in_a_canonical_store():
    from andrey.core.structure import _far_marks

    rng = np.random.default_rng(1)
    for n in (0, 1, 2, 9, 30):
        M = _random_marks(rng, n, "pag", 0.3, loops=True, valid=False)
        M[np.triu(rng.random((n, n)) < 0.2, 1)] = NULL  # an upper cell dropped leaves a half-edge
        indptr, indices, marks = _csr(M)
        rows, cols, far = _far_marks(n, indptr, indices, marks)
        assert np.array_equal(rows, np.repeat(np.arange(n), np.diff(indptr)))
        assert np.array_equal(cols, indices)
        assert np.array_equal(far, M[cols, rows]) and far.dtype == np.int8


@pytest.mark.parametrize("how", ["duplicate", "unsorted", "out_of_range", "negative"])
def test_far_marks_decline_a_store_that_is_not_canonical(how):
    from andrey.core.structure import _far_marks

    rng = np.random.default_rng(2)
    declined = 0
    for _ in range(30):
        M = _random_marks(rng, 8, "pag", 0.5, valid=False)
        indptr, indices, marks = _damage(rng, _csr(M), how)
        found = _far_marks(8, indptr, indices, marks)
        declined += found is None
        assert how == "unsorted" or found is None  # an unsorted row of one cell stays canonical
    assert declined


def test_far_marks_decline_an_indptr_that_does_not_fit_the_store():
    from andrey.core.structure import _far_marks

    indptr, indices, marks = _csr(_random_marks(np.random.default_rng(3), 5, "pag", 0.6))
    assert _far_marks(5, indptr, indices, marks) is not None
    assert _far_marks(6, indptr, indices, marks) is None  # one row too few
    assert _far_marks(5, indptr + 1, indices, marks) is None  # does not start at 0
    assert _far_marks(5, indptr, indices[:-1], marks[:-1]) is None  # ends past the entries
    assert _far_marks(5, indptr, indices, marks[:-1]) is None  # marks and indices differ
    steps_down = indptr.copy()
    steps_down[2] = steps_down[3] + 1  # starts at 0 and ends at the entry count, but steps down
    assert (np.diff(steps_down) < 0).any()
    assert _far_marks(5, steps_down, indices, marks) is None


def test_validate_raises_the_first_error_of_the_edge_by_edge_reference():
    rng = np.random.default_rng(4)
    seen = set()
    for seed in range(40):
        for n in (2, 3, 6, 12):
            for kind in _KINDS:
                for valid in (True, False):
                    M = _random_marks(rng, n, kind, 0.4, loops=kind == "digraph", valid=valid)
                    store = _csr(M)
                    if seed % 4 == 3:
                        store = _damage(
                            rng, store, ("half", "null", "duplicate", "unsorted")[seed % 3]
                        )
                    g = GraphStructure._from_csr(n, *store, kind=kind)
                    message = _validation_message(g)
                    assert message == _reference_validate(g)
                    seen.add((kind, None if message is None else message.split()[0]))
    # every kind meets a valid graph, and each message of the reference is raised for some kind
    assert {m for _, m in seen} >= {None, "double", "undirected", "edge", "asymmetric", "CIRCLE"}
    assert {kind for kind, m in seen if m is None} == set(_KINDS)


def test_validate_of_a_half_edge_matches_the_edge_by_edge_reference():
    # 0 -> 1 stored with (1, 0) absent: an undirected graph reads the missing mark as an arrowhead,
    # every other kind names the half-edge
    indptr, indices, marks = np.array([0, 1, 1]), np.array([1]), np.array([TAIL])
    for kind in _KINDS:
        message = _validation_message(
            GraphStructure._from_csr(2, indptr, indices, marks, kind=kind)
        )
        assert message is not None
        assert message == _reference_validate(
            GraphStructure._from_csr(2, indptr, indices, marks, kind=kind)
        )


def test_compact_store_equals_the_scipy_construction():
    from andrey.core.structure import _try_compact

    rng = np.random.default_rng(5)
    compacted = 0
    for n in (0, 1, 2, 5, 12, 40):
        for kind in _KINDS:
            for valid in (True, False):
                for how in (None, "half", "null", "duplicate", "unsorted"):
                    M = _random_marks(rng, n, kind, 0.3, loops=kind == "digraph", valid=valid)
                    store = _csr(M) if how is None else _damage(rng, _csr(M), how)
                    got = _try_compact(n, *store, kind)
                    want = _reference_try_compact(n, *store, kind)
                    assert (got is None) == (want is None)
                    if got is not None:
                        compacted += 1
                        for a, b in zip(got, want):
                            assert a.dtype == b.dtype and np.array_equal(a, b)
    assert compacted


@pytest.mark.parametrize("edges", [0, 1, 5, 40, 511, 512, 513, 700])
def test_mark_pairs_equal_the_scipy_lookup_on_both_sides_of_the_entry_cutoff(edges):
    rng = np.random.default_rng(6)
    nodes = 100
    pairs = np.array(np.triu_indices(nodes, 1)).T
    for kind in _KINDS:
        for how in (None, "half", "null", "duplicate", "unsorted"):
            M = np.zeros((nodes, nodes), dtype=np.int8)
            for i, j in pairs[rng.choice(len(pairs), edges, replace=False)]:
                M[i, j], M[j, i] = TAIL, (TAIL if kind == "undirected" else ARROW)
            store = _csr(M) if how is None else _damage(rng, _csr(M), how)
            g = GraphStructure._from_csr(nodes, *store, kind=kind)
            for got, want in zip(g._mark_pairs(), _reference_mark_pairs(g)):
                assert got.dtype == want.dtype and np.array_equal(got, want)


def test_mark_pairs_use_the_array_lookup_up_to_the_entry_cutoff_only(monkeypatch):
    from andrey.core import structure

    calls = []
    real = structure._far_marks
    monkeypatch.setattr(structure, "_far_marks", lambda *a: calls.append(a[0]) or real(*a))
    for edges in (structure._NUMPY_PARTNER_ENTRIES // 2, structure._NUMPY_PARTNER_ENTRIES // 2 + 1):
        M = np.zeros((edges + 1, edges + 1), np.int8)
        M[np.arange(edges), np.arange(1, edges + 1)] = TAIL  # a path: two stored entries per edge
        M[np.arange(1, edges + 1), np.arange(edges)] = ARROW
        g = GraphStructure.from_numpy(M, kind="dag")
        calls.clear()
        g._mark_pairs()
        assert calls == ([edges + 1] if 2 * edges <= structure._NUMPY_PARTNER_ENTRIES else [])


# ---- a column index outside range(n) --------------------------------------------------------


_RANGE_MESSAGE = r"column indices must lie in range\({n}\), got values from {low} to {high}"


def _kind_store(kind, n=6):
    """A random valid ``(n, n)`` matrix of ``kind`` with at least one edge, and its CSR store."""
    M = _random_marks(np.random.default_rng(7), n, kind, 0.7, loops=kind == "digraph")
    assert M.any()
    return M, _csr(M)


@pytest.mark.parametrize("kind", _KINDS)
def test_a_column_index_outside_the_range_raises_a_value_error(kind):
    n = 6
    _, (indptr, indices, marks) = _kind_store(kind, n)
    for position in range(indices.size):
        for bad in (n, n + 1, n + 2, -1, -n, np.iinfo(np.int32).max, np.iinfo(np.int32).min):
            damaged = indices.copy()
            damaged[position] = bad
            with pytest.raises(ValueError, match="column indices must lie in range"):
                GraphStructure._from_csr(n, indptr, damaged, marks, kind=kind)


def test_the_error_gives_the_range_and_the_values_found():
    expected = _RANGE_MESSAGE.format(n=3, low=5, high=5)
    with pytest.raises(ValueError, match=expected):
        GraphStructure._from_csr(3, [0, 1, 1, 1], [5], [TAIL], kind="dag")
    expected = _RANGE_MESSAGE.format(n=3, low=-1, high=9)
    with pytest.raises(ValueError, match=expected):
        GraphStructure._from_csr(3, [0, 2, 2, 2], [-1, 9], [TAIL, TAIL], kind="pag")
    expected = _RANGE_MESSAGE.format(n=0, low=0, high=0)  # no node can be named
    with pytest.raises(ValueError, match=expected):
        GraphStructure._from_csr(0, [0], [0], [TAIL], kind="dag")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("as_list", [False, True], ids=["int64 array", "list"])
def test_an_index_that_the_int32_cast_would_wrap_into_range_raises(kind, as_list):
    # 2**32 + 1 is 1 in int32, a valid column of a two-node graph
    for wide in (2**32 + 1, 1 - 2**32, 2**70):
        indices = [wide] if as_list or wide == 2**70 else np.array([wide], dtype=np.int64)
        with pytest.raises(ValueError, match="column indices must lie in range"):
            GraphStructure._from_csr(2, [0, 1, 1], indices, [TAIL], kind=kind)


@pytest.mark.parametrize("kind", _KINDS)
def test_a_store_with_no_entries_and_an_empty_list_builds(kind):
    for n in (0, 1, 4):
        g = GraphStructure._from_csr(n, [0] * (n + 1), [], [], kind=kind)
        assert g.n_nodes == n and len(g.to_edges()) == 0


@pytest.mark.parametrize("kind", _KINDS)
def test_a_store_with_every_index_in_range_builds_as_it_always_did(kind):
    from andrey.core.structure import _try_compact

    rng = np.random.default_rng(8)
    dtypes = (np.int64, np.int32, np.int8)
    built = 0
    for n in (0, 1, 2, 5, 12):
        for valid in (True, False):
            for how in (None, "half", "null", "duplicate", "unsorted", "bad_mark"):
                M = _random_marks(rng, n, kind, 0.4, loops=kind == "digraph", valid=valid)
                store = _csr(M)
                if how == "bad_mark":
                    if store[2].size:
                        store[2][int(rng.integers(store[2].size))] = 5
                elif how is not None:
                    store = _damage(rng, store, how)
                g = GraphStructure._from_csr(n, *store, kind=kind)
                compact = _try_compact(n, *store, kind)
                want = compact or tuple(np.asarray(a, dt) for a, dt in zip(store, dtypes))
                assert g._layout == ("sym" if compact is None else "compact")
                for got, expected in zip((g._indptr, g._indices, g._marks), want):
                    assert got.dtype == expected.dtype and np.array_equal(got, expected)
                # the lists of a JSON file build the same store
                assert GraphStructure._from_csr(n, *(a.tolist() for a in store), kind=kind) == g
                built += 1
    assert built


def test_the_first_and_the_last_node_are_in_range():
    g = GraphStructure._from_csr(4, [0, 1, 1, 1, 1], [3], [TAIL], kind="pag")
    assert g._indices.tolist() == [3]
    g = GraphStructure._from_csr(4, [0, 1, 1, 1, 2], [0, 3], [TAIL, TAIL], kind="pag")
    assert g._indices.tolist() == [0, 3]


def _saved_graph(tmp_path, kind, fmt):
    M, _ = _kind_store(kind)
    path = tmp_path / f"{kind}.{fmt}"
    g = GraphStructure.from_numpy(M, kind=kind, allow_self_loops=True)
    StructureOutput.new(g).save(path, fmt=fmt)
    return path


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("bad", [6, 8, -1])
def test_a_saved_json_with_an_index_out_of_range_raises_on_load(tmp_path, kind, bad):
    path = _saved_graph(tmp_path, kind, "json")
    assert StructureOutput.load(path).structure.n_nodes == 6
    doc = json.loads(path.read_text())
    doc["indices"][len(doc["indices"]) // 2] = bad
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="column indices must lie in range"):
        StructureOutput.load(path)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("bad", [6, 8, -1])
def test_a_saved_npz_with_an_index_out_of_range_raises_on_load(tmp_path, kind, bad):
    path = _saved_graph(tmp_path, kind, "npz")
    assert StructureOutput.load(path).structure.n_nodes == 6
    with np.load(path, allow_pickle=False) as npz:
        arrays = dict(npz)
    arrays["indices"] = arrays["indices"].copy()
    arrays["indices"][arrays["indices"].size // 2] = bad
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="column indices must lie in range"):
        StructureOutput.load(path)


@pytest.mark.parametrize("bad", [3, -1])
def test_a_saved_lag_graph_with_an_index_out_of_range_raises_on_load(tmp_path, bad):
    path = tmp_path / "temporal.json"
    lags = [directed([(0, 1)], 3), directed([(1, 2)], 3)]
    StructureOutput.new(TemporalStructure.from_lag_graphs(lags)).save(path)
    doc = json.loads(path.read_text())
    doc["lags"][1]["indices"][0] = bad
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="column indices must lie in range"):
        StructureOutput.load(path)
