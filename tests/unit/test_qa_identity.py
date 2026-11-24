"""The identity harness hashes every bit of a result, maps columns back, and counts queries."""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import andrey
from andrey import GraphStructure, StructureOutput
from andrey.core import ci, score_delta
from andrey.core.ci import make_indep_test
from andrey.core.score import BICScore
from qa import identity
from qa import quality_gate as qg


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    """A chain ``0 -> 1 -> 2`` with an independent fourth column."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal(500)
    b = a + rng.standard_normal(500)
    c = b + rng.standard_normal(500)
    return np.column_stack([a, b, c, rng.standard_normal(500)])


def test_a_result_hashes_the_same_on_every_fit(x: np.ndarray) -> None:
    first, second = andrey.ges(x), andrey.ges(x)
    assert identity.digest(first) == identity.digest(second)
    assert identity.graph_digest(first) == identity.graph_digest(second)


def test_one_float_bit_changes_the_hash_and_not_the_graph_hash(x: np.ndarray) -> None:
    out = andrey.ges(x)
    score = float(out.metadata["score"])
    nudged = StructureOutput.new(
        out.structure, metadata={**out.metadata, "score": float(np.nextafter(score, np.inf))}
    )
    assert identity.digest(nudged) != identity.digest(out)
    assert identity.graph_digest(nudged) == identity.graph_digest(out)


def test_a_changed_edge_changes_the_graph_hash() -> None:
    marks = np.zeros((3, 3), dtype=np.int8)
    marks[0, 1], marks[1, 0] = 1, 2  # 0 -> 1
    out = StructureOutput.new(GraphStructure.from_numpy(marks, kind="dag"))
    marks[1, 2], marks[2, 1] = 1, 2  # and 1 -> 2
    changed = StructureOutput.new(GraphStructure.from_numpy(marks, kind="dag"))
    assert identity.graph_digest(changed) != identity.graph_digest(out)


def test_floats_enter_as_bits_with_one_nan() -> None:
    other_nan = np.frombuffer(struct.pack("<Q", 0x7FF8000000000001), dtype=np.float64)
    assert identity.digest(other_nan) == identity.digest(np.array([np.nan]))
    assert identity.digest(-0.0) != identity.digest(0.0)
    assert identity.digest(np.float32(1.0)) != identity.digest(1.0)
    assert identity.digest({"b": 1, "a": 2}) == identity.digest({"a": 2, "b": 1})
    with pytest.raises(TypeError):
        identity.digest({1, 2})


def test_a_column_order_maps_back_to_the_original_nodes() -> None:
    marks = np.zeros((4, 4), dtype=np.int8)
    marks[0, 1], marks[1, 0] = 1, 2  # 0 -> 1
    marks[1, 2], marks[2, 1] = 1, 2  # 1 -> 2
    weights = np.zeros((4, 4))
    weights[0, 1], weights[1, 2] = 0.5, -1.5
    out = StructureOutput.new(
        GraphStructure.from_numpy(marks, kind="dag"),
        ordering=[3, 0, 1, 2],
        weighted_adjacency=weights,
        metadata={"algorithm": "test", "p_values": [np.arange(16.0).reshape(4, 4).tolist()]},
    )
    order = np.array([2, 0, 3, 1])  # permuted column k is original column order[k]
    back = np.argsort(order)
    fitted = StructureOutput.new(  # the same answer, as a fit on x[:, order] gives it
        GraphStructure.from_numpy(marks[np.ix_(order, order)], kind="dag"),
        ordering=[int(back[i]) for i in out.ordering],
        weighted_adjacency=weights[np.ix_(order, order)],
        metadata={
            "algorithm": "test",
            "p_values": [np.arange(16.0).reshape(4, 4)[np.ix_(order, order)].tolist()],
        },
    )
    assert identity.digest(identity.unpermute(fitted, order)) == identity.digest(
        identity.unpermute(out, np.arange(4))
    )
    assert identity.graph_digest(identity.unpermute(fitted, order)) == identity.graph_digest(out)


def test_a_swapped_pair_trades_its_directions() -> None:
    marks = np.array([[0, 1], [2, 0]], dtype=np.int8)  # 0 -> 1
    fitted = StructureOutput.new(  # 1 -> 0 on the swapped columns
        GraphStructure.from_numpy(marks.T, kind="dag"),
        metadata={"p_values": {"forward": 0.5, "backward": 0.01}},
    )
    back = identity.unpermute(fitted, np.array([1, 0]))
    assert back.structure.to_numpy().tolist() == marks.tolist()
    assert dict(back.metadata) == {"p_values": {"forward": 0.01, "backward": 0.5}}


def test_permuted_moves_each_block_of_variables_alike() -> None:
    x = np.arange(12.0).reshape(1, 12)  # three groups of four columns
    order = np.array([3, 1, 0, 2])
    expected = [3, 1, 0, 2, 7, 5, 4, 6, 11, 9, 8, 10]
    assert identity.permuted(x, order, 3).ravel().tolist() == expected


def spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[tuple[tuple, dict]]:
    """Record each call's arguments to ``andrey.<name>``, which still fits."""
    calls: list[tuple[tuple, dict]] = []
    fit = getattr(andrey, name)

    def recorded(*args, **kwargs):
        calls.append((args, kwargs))
        return fit(*args, **kwargs)

    monkeypatch.setattr(andrey, name, recorded)
    return calls


def test_icp_moves_its_target_with_the_columns_and_keeps_its_environments(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    x = np.column_stack([np.arange(120) * 3 // 120, rng.standard_normal((120, 4))])
    x[:, 4] += x[:, 2]  # the target, the last variable, follows variable 1
    calls = spy(monkeypatch, "icp")
    refit = identity.FITS["ICP"].refit
    assert refit is not None
    refit(x, np.random.default_rng(5))
    order = np.random.default_rng(5).permutation(4)
    (samples, env), kwargs = calls[0]
    assert np.array_equal(samples, x[:, 1:][:, order])
    assert np.array_equal(env, x[:, 0])
    assert np.array_equal(samples[:, kwargs["target"]], x[:, -1])


def test_igsp_moves_its_targets_with_the_columns_and_keeps_its_environments(monkeypatch) -> None:
    rng = np.random.default_rng(1)
    samples = rng.standard_normal((150, 4))
    env = np.arange(150) * 3 // 150
    x = qg.known_target_stack(samples, env, [[1], [2, 3]])
    calls = spy(monkeypatch, "igsp")
    refit = identity.FITS["IGSP"].refit
    assert refit is not None
    refit(x, np.random.default_rng(5))
    order = np.random.default_rng(5).permutation(4)
    (moved, moved_env), kwargs = calls[0]
    assert np.array_equal(moved, samples[:, order])
    assert np.array_equal(moved_env, env)
    # the columns each setting names are the original targets
    assert {k: sorted(int(order[j]) for j in v) for k, v in kwargs["targets"].items()} == {
        0: [],
        1: [1],
        2: [2, 3],
    }


def test_fci_jci_keeps_its_environment_column_last(monkeypatch) -> None:
    rng = np.random.default_rng(2)
    x = np.column_stack([rng.standard_normal((150, 4)), np.arange(150) * 2 // 150])
    calls = spy(monkeypatch, "fci_jci")
    refit = identity.FITS["FCI-JCI"].refit
    assert refit is not None
    refit(x, np.random.default_rng(5))
    order = np.random.default_rng(5).permutation(4)
    (moved, env), _ = calls[0]
    assert np.array_equal(moved, x[:, order])
    assert np.array_equal(env, x[:, -1])


@pytest.mark.filterwarnings("ignore::andrey.DataTypeWarning")
def test_jpcmciplus_names_the_contexts_new_columns(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 60, qg.CONTEXT_SYSTEM + 2))  # one time and one space context
    x[..., 5] = rng.standard_normal(60)  # the time context is the same in every dataset
    x[..., 6] = np.arange(2)[:, None]  # the space context is constant within a dataset
    contexts = qg.context_columns(x)
    assert contexts == {"time_contexts": [5], "space_contexts": [6]}
    calls = spy(monkeypatch, "jpcmciplus")
    refit = identity.FITS["J-PCMCI+"].refit
    assert refit is not None
    refit(x, np.random.default_rng(5))
    order = np.random.default_rng(5).permutation(qg.CONTEXT_SYSTEM + 2)
    (moved,), kwargs = calls[0]
    for dataset, original in zip(moved, x, strict=True):
        assert np.array_equal(dataset, original[:, order])
        assert np.array_equal(dataset[:, kwargs["time_contexts"]], original[:, [5]])
        assert np.array_equal(dataset[:, kwargs["space_contexts"]], original[:, [6]])


def test_icps_target_and_per_variable_metadata_map_back() -> None:
    marks = np.zeros((3, 3), dtype=np.int8)
    marks[1, 2], marks[2, 1] = 1, 2  # 1 -> 2, the target
    meta = {
        "algorithm": "icp",
        "target": 2,
        "parents": [1],
        "p_values": [0.2, 0.01, None],
        "confidence_intervals": [[-1.0, 1.0], [0.5, 0.9], None],
        "accepted_sets": [[1], [0, 1]],
    }
    order = np.array([2, 0, 1])  # permuted column k is original column order[k]
    back = np.argsort(order)
    fitted = StructureOutput.new(  # the same answer, as a fit on x[:, order] gives it
        GraphStructure.from_numpy(marks[np.ix_(order, order)], kind="dag"),
        metadata={
            **meta,
            "target": int(back[2]),
            "p_values": [meta["p_values"][i] for i in order],
            "confidence_intervals": [meta["confidence_intervals"][i] for i in order],
            "accepted_sets": [[0], [0, 2]],  # laid out in the order of the columns
        },
    )
    original = StructureOutput.new(GraphStructure.from_numpy(marks, kind="dag"), metadata=meta)
    assert identity.digest(identity.unpermute(fitted, order)) == identity.digest(
        identity.unpermute(original, np.arange(3))
    )


def test_a_rejected_icp_fit_keeps_its_missing_intervals() -> None:
    """ICP sets ``confidence_intervals`` to ``None`` when it rejects every candidate set."""
    meta = {"algorithm": "icp", "target": 1, "p_values": [0.0, None], "confidence_intervals": None}
    out = StructureOutput.new(
        GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="dag"), metadata=meta
    )
    back = identity.unpermute(out, np.array([1, 0]))
    assert back.metadata["confidence_intervals"] is None
    assert back.metadata["p_values"] == [None, 0.0]
    assert back.metadata["target"] == 0


def test_ut_igsps_targets_map_back_to_the_original_columns() -> None:
    original = {0: [], 1: [2], 2: [0, 3]}
    order = np.array([2, 3, 1, 0])  # permuted column k is original column order[k]
    back = np.argsort(order)
    fitted = {label: sorted(int(back[t]) for t in found) for label, found in original.items()}
    assert fitted == {0: [], 1: [0], 2: [1, 3]}

    def output(targets: dict[int, list[int]]) -> StructureOutput:
        return StructureOutput.new(
            GraphStructure.from_numpy(np.zeros((4, 4), dtype=np.int8), kind="cpdag"),
            metadata={"algorithm": "ut_igsp", "targets": targets},
        )

    mapped = identity.unpermute(output(fitted), order)
    assert mapped.metadata["targets"] == original
    assert identity.digest(mapped) == identity.digest(
        identity.unpermute(output(original), np.arange(4))
    )


def _output(p_value: float, *, ordering: list[int] | None = None, edge: bool = True):
    marks = np.zeros((3, 3), dtype=np.int8)
    if edge:
        marks[0, 1], marks[1, 0] = 1, 2  # 0 -> 1
    return StructureOutput.new(
        GraphStructure.from_numpy(marks, kind="dag"),
        ordering=ordering,
        metadata={"algorithm": "test", "p_values": [p_value]},
    )


def test_the_last_bits_of_a_value_differ_under_a_column_order_without_failing() -> None:
    first, refit = _output(0.25), _output(float(np.nextafter(0.25, 1.0)))
    assert identity.classify(first, refit, column_order=False) == ("bits", ["metadata.p_values"])
    assert identity.classify(first, refit, column_order=True) == ("digits", ["metadata.p_values"])
    assert identity.classify(first, first, column_order=True) is None
    assert "digits" not in identity.FAILING and "bits" in identity.FAILING


def test_a_value_that_changes_by_more_than_rounding_fails_under_a_column_order() -> None:
    kind, changed = identity.classify(_output(0.25), _output(0.9), column_order=True)
    assert (kind, changed) == ("values", ["metadata.p_values"])
    assert "values" in identity.FAILING


def test_an_integer_that_becomes_a_float_is_not_rounding() -> None:
    first = StructureOutput.new(
        GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="dag"),
        metadata={"score": 1, "counts": [1, 2]},
    )
    refit = StructureOutput.new(
        GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="dag"),
        metadata={"score": 1.0, "counts": [1.0, 2.0]},
    )
    assert identity.classify(first, refit, column_order=True) == (
        "values",
        ["metadata.counts", "metadata.score"],
    )


def test_a_boolean_that_becomes_a_number_is_not_rounding() -> None:
    def output(flag: object) -> StructureOutput:
        graph = GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="dag")
        return StructureOutput.new(graph, metadata={"flag": flag})

    for first, refit in ((True, 1), (1, True), (False, 0), (True, 1.0)):
        kind, changed = identity.classify(output(first), output(refit), column_order=True)
        assert (kind, changed) == ("values", ["metadata.flag"])
    assert identity.classify(output(True), output(True), column_order=True) is None


def test_a_time_series_weights_differ_by_rounding_or_more() -> None:
    def output(lag, time=None, lags=(1,)):
        structure = SimpleNamespace(
            lags=lags, times=(0,), lag_weights=lag, lag_weights_ma=None, time_weights=time
        )
        return SimpleNamespace(structure=structure, weighted_adjacency=None, metadata={})

    base = output(np.array([[[0.5, 0.0], [0.25, 0.0]]]), np.full((1, 1, 2, 2), np.nan))
    rounded = output(base.structure.lag_weights * (1 + 1e-13), base.structure.time_weights)
    assert identity._rounded(base, rounded)
    assert not identity._rounded(base, output(base.structure.lag_weights * 1.1))
    assert not identity._rounded(base, output(base.structure.lag_weights, np.zeros((1, 1, 2, 2))))
    assert not identity._rounded(base, output(base.structure.lag_weights, lags=(2,)))


def test_a_different_graph_or_causal_order_fails_under_a_column_order() -> None:
    first = _output(0.25, ordering=[0, 1, 2])
    kind, changed = identity.classify(_output(0.25, edge=False), first, column_order=True)
    assert kind == "graph" and "graph" in changed
    kind, changed = identity.classify(_output(0.25, ordering=[1, 0, 2]), first, column_order=True)
    assert (kind, changed) == ("structure", ["ordering"])
    assert {"graph", "structure"} <= set(identity.FAILING)


def test_a_documented_exception_is_expected_only_in_the_parts_it_names() -> None:
    # FastICA's solution decides the graph, the causal order, and the weights, not the stamps
    assert identity.documented("ICA-LiNGAM", ["graph", "ordering", "weights"])
    assert identity.documented("ICA-LiNGAM", ["ordering"])
    assert identity.documented("ICA-LiNGAM", []) is None
    assert identity.documented("PC", ["graph", "ordering"]) is None
    assert identity.documented("ICA-LiNGAM", ["metadata.seed"]) is None
    assert identity.documented("ICA-LiNGAM", ["ordering", "metadata.tie_break"]) is None
    assert "documented" not in identity.FAILING
    assert set(identity.DOCUMENTED) <= set(identity.FITS)


def test_a_documented_exception_is_not_a_failure_under_a_column_order(monkeypatch) -> None:
    graph = GraphStructure.from_numpy(np.zeros((2, 2), dtype=np.int8), kind="dag")
    outputs = {
        False: StructureOutput.new(graph, metadata={"p_values": [[[1.0, 0.5], [0.5, 1.0]]]}),
        True: StructureOutput.new(graph, metadata={"p_values": [[[1.0, 0.9], [0.9, 1.0]]]}),
    }
    calls = []

    def call(x):
        calls.append(x)
        return outputs[len(calls) > 1]  # the first fit, then the column-order refit

    monkeypatch.setitem(identity.FITS, "PCMCIplus", identity.Fit(call))
    monkeypatch.setitem(
        identity.DOCUMENTED, "PCMCIplus", (frozenset({"metadata.p_values"}), "a documented reason")
    )
    draw = iter([("PCMCIplus", "d", 0, np.zeros((3, 2)))])
    monkeypatch.setattr(identity, "corpus", lambda methods, seeds: draw)
    _, findings = identity.run(["PCMCIplus"], 1, shuffle=1)
    assert [kind for kind, _ in findings] == ["documented"]
    assert "a documented reason" in findings[0][1]
    assert not any(kind in identity.FAILING for kind, _ in findings)


@pytest.mark.parametrize(
    ("kind", "code"),
    [("digits", 0), ("documented", 0), ("bits", 1), ("values", 1), ("structure", 1), ("graph", 1)],
)
def test_the_exit_status_fails_on_a_structural_difference_and_reports_the_rest(
    kind, code, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        identity, "run", lambda *args: ([], [(kind, "PC d seed 0 permutation 1: x")])
    )
    assert identity.main(["PC", "--shuffle", "1"]) == code
    out = capsys.readouterr().out
    assert identity.LABELS[kind] + ": PC d seed 0 permutation 1: x" in out
    assert (f"{identity.LABELS[kind]}" in out) and "does not fail the check" in out


def test_counts_separation_and_distinct_conditioning_sets() -> None:
    counts = identity.Counts(alpha=0.05)
    counts.ci([(0, 2, (), 0.001), (2, 0, (1,), 0.4), (0, 2, (3,), 0.01), (2, 0, (1, 3), 0.9)])
    counts.ci([(1, 3, (1,), 0.5)])
    counts.score([(0, 1), (1, 0), (), (2,)])
    assert counts.summary() == {
        "ci_queries": 5,
        "ci_after_separation": 2,  # the two queries of (0, 2) after p = 0.4
        "ci_sets": {"0": 1, "1": 2, "2": 1},
        "score_queries": 4,
        "score_sets": {"0": 1, "1": 1, "2": 1},
    }


def test_counting_records_each_outermost_query_and_restores_the_methods(x: np.ndarray) -> None:
    test = make_indep_test("fisherz", x)
    original = ci.PartialCorrelationTest.__dict__["batched_call"]
    with identity.counting(alpha=0.05) as counts:
        test(0, 2, (1,))  # one query, though it calls statistic_pvalue
        test.statistic_pvalue(0, 3, (1,))
        test.batched_call(0, 2, iter([(), (1, 3)]))
        test.first_order_pvalues(0, 1, [2, 3])
        test._canonical_pvalues(np.array([0, 2]), np.array([3, 3]), np.array([[1, 2], [0, 1]]))
    assert ci.PartialCorrelationTest.__dict__["batched_call"] is original
    assert counts.summary()["ci_queries"] == 8
    assert counts.summary()["ci_sets"] == {"0": 1, "1": 3, "2": 3}


@pytest.mark.parametrize(
    ("name", "ask", "conditions"),
    [
        ("fisherz", lambda test, data, *q: test.statistic_pvalues(*q), [[1], [0], [2]]),
        ("fisherz", lambda test, data, *q: test.statistic_pvalues(*q), np.empty((3, 0), int)),
        (
            "fisherz",
            lambda test, data, *q: test.residual_statistic_pvalues(data, *q),
            [[1], [0], [2]],
        ),
        ("chisq", lambda test, data, *q: test.statistic_pvalues(*q), [[1], [0], [2]]),
    ],
    ids=["statistic_pvalues", "empty_sets", "residual_statistic_pvalues", "chisq"],
)
def test_counting_records_one_query_per_pvalue_of_a_batched_call(
    x: np.ndarray, name: str, ask: Callable[..., tuple[np.ndarray, np.ndarray]], conditions: object
) -> None:
    data = np.digitize(x, [-1.0, 1.0]) if name == "chisq" else x
    test = make_indep_test(name, data)
    with identity.counting(alpha=0.0) as counts:  # every p > 0 separates its pair
        _, p_values = ask(test, data, np.array([0, 1, 0]), np.array([2, 3, 3]), conditions)
    assert counts.summary()["ci_queries"] == len(p_values) == 3
    assert counts.separated == {(0, 2), (1, 3), (0, 3)}
    assert counts.summary()["ci_sets"] == ({"1": 3} if np.size(conditions) else {"0": 1})


def test_counting_wraps_every_public_pvalue_method_of_the_ci_tests() -> None:
    public = {
        name
        for cls in identity.CI_CLASSES
        for owner in cls.__mro__
        for name, value in vars(owner).items()
        if callable(value) and "pvalue" in name and not name.startswith("_")
    }
    assert public <= set(identity.CI_QUERIES)


def test_counting_wraps_every_name_of_the_score_delta_functions() -> None:
    for module in pkgutil.walk_packages(andrey.__path__, "andrey."):
        with contextlib.suppress(ImportError):  # an optional extra
            importlib.import_module(module.name)
    functions = (score_delta.add_deltas, score_delta.remove_deltas)
    naming = {
        name
        for name, module in sys.modules.items()
        if name.startswith("andrey.")
        and any(v is f for v in vars(module).values() for f in functions)
    }
    assert naming == {module.__name__ for module in identity.DELTA_MODULES}


def test_counting_leaves_the_result_unchanged(x: np.ndarray) -> None:
    for fit in (andrey.pc, andrey.hc):
        with identity.counting(alpha=0.05) as counts:
            counted = fit(x)
        assert counts.summary()
        assert identity.digest(counted) == identity.digest(fit(x))


def _manifest(**changes: object) -> dict:
    row = {
        "method": "PC",
        "dataset": "d",
        "seed": 0,
        "permutation": 0,
        "workers": 1,
        "hash": "h",
        "graph_hash": "g",
        "ci_queries": 10,
    }
    return {"header": {"commit": "a"}, "rows": [{**row, **changes}]}


@pytest.mark.parametrize(
    ("changes", "status", "message"),
    [
        ({}, 0, "0 with a different graph, 0 with only different bits"),
        ({"hash": "h2"}, 1, "0 with a different graph, 1 with only different bits"),
        ({"hash": "h2", "graph_hash": "g2"}, 1, "1 with a different graph"),
        ({"ci_queries": 11}, 1, "1 counts rose"),
        ({"ci_queries": 9}, 1, "1 fell"),
        ({"seed": 1}, 1, "2 in one only"),
    ],
)
def test_compare_counts_the_rows_that_differ(changes, status, message, capsys) -> None:
    assert identity.compare(_manifest(), _manifest(**changes)) == status
    assert message in capsys.readouterr().out


def test_compare_names_the_metadata_keys_that_alone_differ(capsys) -> None:
    parts = {"graph": "g", "structure": "s", "metadata.algorithm": "a"}
    before = _manifest(parts=parts)
    renamed = _manifest(hash="h2", parts={**parts, "metadata.algorithm": "b", "metadata.seed": "c"})
    assert identity.compare(before, renamed) == 1
    assert "1 with only different metadata, 0 with only a different kind (algorithm, seed)" in (
        capsys.readouterr().out
    )
    moved = _manifest(hash="h2", parts={**parts, "structure": "t", "metadata.algorithm": "b"})
    assert identity.compare(before, moved) == 1
    assert "1 with only different bits, 0 with only different metadata" in capsys.readouterr().out


def test_compare_tells_a_new_kind_from_new_edges(capsys) -> None:
    parts = {"graph": "g", "kind": "k", "edges": "e", "structure": "s", "metadata.dag": "d"}
    before = _manifest(parts=parts)
    kind = {**parts, "graph": "g2", "kind": "k2", "structure": "s2", "metadata.dag": "d2"}
    assert identity.compare(before, _manifest(hash="h2", graph_hash="g2", parts=kind)) == 1
    assert "1 with only a different kind (dag)" in capsys.readouterr().out
    edges = {**kind, "edges": "e2"}
    assert identity.compare(before, _manifest(hash="h2", graph_hash="g2", parts=edges)) == 1
    out = capsys.readouterr().out
    assert "1 with a different graph" in out and "0 with only a different kind" in out


def test_a_fall_in_a_count_passes_only_when_allowed(capsys) -> None:
    fewer = _manifest(ci_queries=9)
    assert identity.compare(_manifest(), fewer) == 1
    assert identity.compare(_manifest(), fewer, allow_fewer=True) == 0
    assert identity.compare(_manifest(), _manifest(ci_queries=11), allow_fewer=True) == 1
    assert "count fell: PC d 0 0 1 ci_queries: 10 -> 9" in capsys.readouterr().out


def test_counting_records_the_exact_search_scores(x: np.ndarray) -> None:
    bic = BICScore(x)
    with identity.counting(None) as counts:
        bic.score_many(0, [(1,), (2,), (3,)])
        bic.score(1, (0, 2))
    assert counts.summary() == {
        "score_queries": 4,
        "score_sets": {"1": 3, "2": 1},
    }


def test_every_supported_method_is_fitted_on_its_datasets() -> None:
    assert set(identity.FITS) - set(identity.TARGET_LISTS) == set(qg.SUPPORTED)
    assert set(identity.TARGET_LISTS.values()) <= set(qg.SUPPORTED)
    assert set(identity.TARGET_LISTS) <= set(identity.WORKER_METHODS)
    fitted = [identity.TARGET_LISTS.get(m, m) for m in identity.FITS]
    assert all(m in identity.LARGER or m in identity.UNKNOWN_TARGETS for m in fitted)


def test_pnl_is_refitted_on_workers_and_gives_the_same_bits() -> None:
    assert "PNL" in identity.WORKER_METHODS
    rows, findings = identity.run(["PNL"], seeds=1, workers=2)
    assert findings == []
    assert {r["workers"] for r in rows} == {1, 2}


def test_the_header_names_the_imported_package_and_its_tree(tmp_path, monkeypatch) -> None:
    code = identity.commit()
    assert code["andrey_path"] == str(Path(andrey.__file__).resolve().parent)
    assert code["andrey_commit"] is None or len(code["andrey_commit"]) == 40
    outside = tmp_path / "andrey"
    outside.mkdir()
    monkeypatch.setattr(andrey, "__file__", str(outside / "__init__.py"))
    assert identity.commit()["andrey_commit"] is None


def test_compare_ignores_the_package_header(capsys) -> None:
    before, after = _manifest(), _manifest()
    before["header"].update(andrey_path="/a/src/andrey", andrey_commit="a")
    after["header"].update(andrey_path="/b/src/andrey", andrey_commit=None)
    assert identity.compare(before, after) == 0
    assert "0 with a different graph, 0 with only different bits" in capsys.readouterr().out
