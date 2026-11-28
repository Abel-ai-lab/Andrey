"""The identity harness hashes every bit of a result, maps columns back, and counts queries."""

from __future__ import annotations

import contextlib
import importlib
import json
import pkgutil
import re
import struct
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import andrey
from andrey import GraphStructure, StructureOutput
from andrey.constraint import blanket
from andrey.core import ci, ci_missing, score_delta
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


@pytest.mark.parametrize("name", ["fisherz", "student_t", "normal_scores"])
def test_counting_records_the_stacked_queries_up_to_the_first_that_separates(
    x: np.ndarray, name: str
) -> None:
    """A stacked call that stops at its first separating query counts the p-values it returns, not
    the queries it computed ahead and dropped: the queries of a scalar loop that stops there."""
    test = make_indep_test(name, x)
    rows = (np.array([1, 1, 1]), np.array([3, 0, 2]), np.array([[0, 2], [3, 2], [3, 0]]))
    with identity.counting(alpha=0.05) as stacked:
        found = test.pvalues_until_separated(*rows, 0.05)
    with identity.counting(alpha=0.05) as alone:
        for target, member, cond in zip(*rows):
            if test(target, member, cond) > 0.05:
                break
    assert len(found) == 1  # column 3 is independent of column 1, so the first query separates
    assert (
        stacked.summary()
        == alone.summary()
        == {
            "ci_queries": 1,
            "ci_after_separation": 0,
            "ci_sets": {"2": 1},
        }
    )
    assert stacked.separated == alone.separated == {(1, 3)}


def test_counting_records_the_queries_of_a_stack_that_falls_back_to_scalar_calls() -> None:
    """A stack that raises on a singular row answers by scalar calls up to the first separating
    one, and those are the queries counted, once each."""
    X = np.random.default_rng(2).standard_normal((200, 5))
    X[:, 4] = X[:, 1]
    test = ci.FisherZ(X)  # make_indep_test refuses copies
    rows = (np.array([0, 0, 0]), np.array([2, 2, 3]), np.array([[1, 3], [1, 4], [2, 1]]))
    with identity.counting(alpha=0.05) as counts:
        found = test.pvalues_until_separated(*rows, 0.05)
    assert len(found) == 1 and counts.summary()["ci_queries"] == 1
    with pytest.raises(ValueError, match="singular"), identity.counting(alpha=1.0) as raised:
        test.pvalues_until_separated(*rows, 1.0)
    assert raised.ci_queries == 0  # a call that raises records nothing


def test_counting_a_stacked_shrink_gives_the_counts_of_the_loop_that_asks_one_member_at_a_time(
    x: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shrink that asks every member in one call, drops one with members after it, and asks the
    rest again counts the queries, the pairs separated, and the sets of a loop that asks one at a
    time, though the stacks computed more queries than that."""

    class Alone:
        """A test that answers only one query at a time, so a shrink asks it one at a time."""

        def __init__(self, test: ci.FisherZ) -> None:
            self.test = test

        def __call__(self, a: int, b: int, cond: object = ()) -> float:
            return self.test(a, b, cond)  # type: ignore[arg-type]

    asked: list[int] = []
    stack = ci.PartialCorrelationTest.statistic_pvalues

    def spied(test: Any, a: Any, b: Any, conditions: Any) -> Any:
        asked.append(len(a))
        return stack(test, a, b, conditions)

    monkeypatch.setattr(ci.PartialCorrelationTest, "statistic_pvalues", spied)
    members_stacked, members_alone = [3, 0, 2], [3, 0, 2]  # column 3 is independent of column 1
    with identity.counting(alpha=0.05) as stacked:
        blanket._shrink(ci.FisherZ(x), 1, 0.05, members_stacked)
    stacked_rows = list(asked)
    with identity.counting(alpha=0.05) as alone:
        blanket._shrink(Alone(ci.FisherZ(x)), 1, 0.05, members_alone)  # type: ignore[arg-type]
    assert members_stacked == members_alone == [0, 2]
    assert stacked_rows == [3, 2]  # five queries computed
    assert stacked.summary() == alone.summary()  # three of them are the loop's
    assert stacked.ci_queries == 3
    assert stacked.separated == alone.separated == {(1, 3)}


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("fisherz", ci_missing.GaussianDeletion),
        ("chisq", ci_missing.ChiSquareDeletion),
        ("cond_gauss", ci_missing.TestwiseDeletion),
        ("cmiknn", ci_missing.BoundedDeletion),
    ],
)
def test_counting_records_the_queries_of_a_test_that_deletes_rows(
    x: np.ndarray, name: str, kind: type
) -> None:
    """On data with missing values, each public p-value method of the test-wise deletion tests
    records one query for each p-value, once, though the block test it builds answers it too."""
    gapped = x[:200].copy()
    gapped[::5, 3] = np.nan  # column 3 is missing on every fifth row
    if name == "chisq":
        gapped = np.where(np.isnan(gapped), np.nan, np.digitize(gapped, [-1.0, 1.0]))
    test = make_indep_test(name, gapped)
    assert type(test) is kind
    with identity.counting(alpha=0.0) as counts:  # every p > 0 separates its pair
        test(0, 3, (1,))
        test(0, 3, (1, 2))
        test.statistic_pvalues(np.array([0, 1, 0]), np.array([2, 3, 3]), [[1], [0], [2]])
    assert counts.summary() == {
        "ci_queries": 5,
        "ci_after_separation": 2,  # (0, 3) is asked again twice after its first query
        "ci_sets": {"1": 3, "2": 1},
    }


def test_counting_records_one_query_per_candidate_of_a_scan_on_missing_values(
    x: np.ndarray,
) -> None:
    """A scan of one conditioning set on data with missing values counts as the separate queries
    it answers, in the data's columns and in the order of its candidates."""
    gapped = x[:200].copy()
    gapped[::5, 3] = np.nan
    scanned, alone = make_indep_test("fisherz", gapped), make_indep_test("fisherz", gapped)
    with identity.counting(alpha=0.0) as together:  # every p > 0 separates its pair
        scanned.statistic_pvalues_given(0, [3, 2], [1])
    with identity.counting(alpha=0.0) as apart:
        alone.statistic_pvalue(0, 3, [1])
        alone.statistic_pvalue(0, 2, [1])
    assert (
        together.summary()
        == apart.summary()
        == {
            "ci_queries": 2,
            "ci_after_separation": 0,
            "ci_sets": {"1": 1},
        }
    )
    assert together.separated == apart.separated == {(0, 3), (0, 2)}


def test_counting_records_one_query_per_candidate_of_a_chi_square_scan_on_missing_values(
    x: np.ndarray,
) -> None:
    """A scan of one conditioning set on discrete data with missing values counts as the separate
    queries it answers, in the data's columns and in the order of its candidates, though the
    scan builds no test for most of them."""
    gapped = np.digitize(x[:200], [-1.0, 1.0]).astype(np.float64)
    gapped[::5, 3] = np.nan
    scanned, alone = make_indep_test("chisq", gapped), make_indep_test("chisq", gapped)
    with identity.counting(alpha=0.0) as together:  # every p > 0 separates its pair
        scanned.pvalues_given(0, [3, 2], [1])
    with identity.counting(alpha=0.0) as apart:
        alone(0, 3, [1])
        alone(0, 2, [1])
    assert (
        together.summary()
        == apart.summary()
        == {
            "ci_queries": 2,
            "ci_after_separation": 0,
            "ci_sets": {"1": 1},
        }
    )
    assert together.separated == apart.separated == {(0, 3), (0, 2)}


def test_a_query_that_is_not_tested_counts_and_separates_nothing(x: np.ndarray) -> None:
    """A pair never observed together is asked, so the search's path is counted, and answers NaN."""
    gapped = x.copy()
    gapped[:250, 0] = np.nan
    gapped[250:, 1] = np.nan
    test = make_indep_test("fisherz", gapped)
    with identity.counting(alpha=0.0) as counts:
        assert np.isnan([test(0, 1), test(0, 1, (2,))]).all()
    assert counts.separated == set()
    assert counts.summary() == {
        "ci_queries": 2,
        "ci_after_separation": 0,
        "ci_sets": {"0": 1, "1": 1},
    }


def test_a_deletion_test_over_reordered_columns_counts_in_the_columns_of_its_source(
    x: np.ndarray,
) -> None:
    """FCI's possible-d-sep search asks a test over the columns in rank order, which
    ``from_columns`` maps to the data's, so its queries count as pairs of the data's columns."""
    gapped = x.copy()
    gapped[::5, 3] = np.nan
    test = make_indep_test("fisherz", gapped)
    test.from_columns(np.array([3, 2, 1, 0]))  # column k of the test is column 3 - k of the data
    with identity.counting(alpha=0.0) as counts:
        test(0, 1, (2,))
        test.statistic_pvalues(np.array([0]), np.array([3]), [[1]])
    assert counts.separated == {(2, 3), (0, 3)}
    assert counts.summary()["ci_sets"] == {"1": 2}
    assert counts.ci_sets[1] == {frozenset({1}), frozenset({2})}


def test_a_fit_on_data_with_missing_values_records_its_query_counts() -> None:
    rows, findings = identity.run(["PC-ambiguous"], seeds=1)
    assert findings == [] and rows[0]["ci_queries"] > 0
    assert rows[0]["ci_sets"]


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
    extra = set(identity.TARGET_LISTS) | set(identity.OPTIONS) | set(identity.EM_FITS)
    assert set(identity.FITS) - extra == set(qg.SUPPORTED)
    assert set(identity.TARGET_LISTS.values()) <= set(qg.SUPPORTED)
    assert set(identity.TARGET_LISTS) <= set(identity.WORKER_METHODS)
    own = set(identity.OPTIONS) | set(identity.EM_FITS)
    fitted = [identity.TARGET_LISTS.get(m, m) for m in identity.FITS if m not in own]
    assert all(m in identity.LARGER or m in identity.UNKNOWN_TARGETS for m in fitted)


def test_each_option_fit_names_a_supported_method_and_its_own_data() -> None:
    """Both options of 0.19.0 are fitted: ``"ambiguous"`` on data with missing values, and
    ``"nonempty"`` on discrete data, each on a dataset of its own."""
    options = {name: next(iter(o.options)) for name, o in identity.OPTIONS.items()}
    assert set(options.values()) == {"possibly_separable", "degrees_of_freedom"}
    for name, option in identity.OPTIONS.items():
        assert option.method in qg.SUPPORTED
        assert option.dataset in identity.OPTION_DRAWS
        assert identity.datasets(name) == (option.dataset,)
        x = identity.draws(option.dataset)[0].data
        if options[name] == "possibly_separable":
            assert np.isnan(x).any()
        else:
            assert not np.isnan(x).any() and np.array_equal(x, np.round(x))
    nonempty = [n for n, key in options.items() if key == "degrees_of_freedom"]
    assert {identity.OPTIONS[n].method for n in nonempty} >= {"PC", "FCI"}
    assert {n.split("-")[1] for n in nonempty} == {"chisq", "gsq"}


def test_no_row_of_the_cohort_draws_measures_both_ends_of_the_cohorts() -> None:
    """No row observes a column of 4-6 with one of 10-13: those pairs stay undecided."""
    for draw in identity.draws("cohorts_d14_n2000"):
        observed = ~np.isnan(draw.data)
        assert observed[:, :4].all()
        assert not (observed[:, 4:7].any(axis=1) & observed[:, 10:14].any(axis=1)).any()
        assert observed.any(axis=1).all() and (~observed).any(axis=1).all()


@pytest.mark.parametrize("name", list(identity.OPTIONS))
def test_each_option_changes_the_answer_on_a_draw_of_its_data(name: str) -> None:
    """The option is fitted where it matters: the same call at the option's default returns a
    different answer on at least one of the first two draws."""
    option = identity.OPTIONS[name]
    with andrey.config(backend="numpy", n_jobs=1):
        differs = []
        for draw in identity.draws(option.dataset)[: identity.SEEDS]:
            x = np.ascontiguousarray(draw.data)
            differs.append(
                identity.digest(option.call(x)) != identity.digest(option.call(x, **option.options))
            )
    assert any(differs)


@pytest.mark.parametrize("name", ["PC-ambiguous", "CDNOD-ambiguous", "PC-gsq-nonempty"])
def test_an_option_fit_gives_the_same_graph_in_any_column_order_on_any_workers(name: str) -> None:
    """The lists of pairs that name columns, ``possibly_separable`` and ``undecided_pairs``, are
    left out of the column-order comparison, as the other lists of nodes are."""
    assert {"possibly_separable", "undecided_pairs", "never_observed_together"} <= set(
        identity.NODE_LISTS
    )
    rows, findings = identity.run([name], seeds=1, shuffle=2, workers=2)
    assert findings == []
    assert {r["method"] for r in rows} == {name}
    assert {(r["permutation"], r["workers"]) for r in rows} == {(0, 1), (1, 1), (2, 1), (0, 2)}


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


# ---- checkpoints -----------------------------------------------------------------------------

pc_module = importlib.import_module("andrey.constraint.pc")

#: Each family's fit on a dataset where it makes every stage listed, as the harness fits it.
FAMILY_STAGES: dict[str, tuple[str, set[str]]] = {
    "PC": (
        "asia_gauss",
        {"skeleton.size", "skeleton.end", "pc.colliders", "pc.meek", "pc.conflicts"},
    ),
    "FCI": (
        "asia_gauss",
        {"skeleton.size", "skeleton.end", "fci.colliders", "fci.possible_dsep", "fci.rules"},
    ),
    "CDNOD": ("domain_shift_er_d20", {"skeleton.size", "skeleton.end", "pc.colliders"}),
    "GFCI": ("asia_gauss", {"ges.forward", "skeleton.end", "gfci.sepsets", "fci.rules"}),
    "GES": ("linear_gauss_er_d20", {"ges.forward", "ges.backward"}),
    "AGES": ("ages_paper_qs0.3_qw0.3_n10000", {"ages.forward", "ages.backward"}),
    "XGES": ("asia_gauss", {"xges.move", "xges.branch", "xges.accept", "xges.total"}),
    "GIES": ("interventional_gauss_er_d20", {"gies.step", "gies.total"}),
    "HC": ("asia_gauss", {"hc.climb"}),
    "Tabu": ("asia_gauss", {"tabu.move"}),
    "IAMB": ("asia_gauss", {"blanket.grow", "blanket.shrink", "blanket.structure"}),
    "InterIAMB": ("asia_gauss", {"blanket.grow", "blanket.shrink", "blanket.structure"}),
    "MMPC": ("asia_gauss", {"blanket.grow", "blanket.shrink", "blanket.structure"}),
    "NOTEARS": ("notears_paper_raw_d10", {"notears.solve", "notears.dual", "continuous.result"}),
    "DAGMA": ("notears_paper_raw_d10", {"dagma.stage", "continuous.result"}),
    "DYNOTEARS": ("dynotears_paper_d10_n500", {"notears.solve", "dynotears.result"}),
    "GES-em": ("linear_gauss_er_d10", {"em.round", "em.loglik", "ges.forward"}),
}


def _fit_on(name: str, dataset: str, *, checkpoints: bool) -> tuple[Any, dict, list]:
    x = np.ascontiguousarray(identity.draws(dataset)[0].data)
    fit = identity.FITS[name]
    with identity.counting(fit.alpha) as counts, identity.checkpointing(checkpoints) as recorder:
        out = fit.call(x)
    return out, counts.summary(), [] if recorder is None else recorder.records


def _same_arrays(a: list, b: list) -> bool:
    pieces = [p for records in (a, b) for r in records for p in r.pieces]
    half = len(pieces) // 2
    return all(
        x.digest == y.digest and np.array_equal(x.array, y.array)
        for x, y in zip(pieces[:half], pieces[half:], strict=True)
    )


def test_the_checkpoint_block_sets_the_sink_and_resets_it_on_exit_and_on_an_exception() -> None:
    sink = importlib.import_module("andrey.core._checkpoint")
    assert sink.TAKE.get() is None
    with identity.checkpointing(True) as recorder:
        assert sink.TAKE.get() is recorder
    assert sink.TAKE.get() is None
    with pytest.raises(RuntimeError, match="inside"), identity.checkpointing(True):
        raise RuntimeError("inside")
    assert sink.TAKE.get() is None
    with identity.checkpointing(False) as nothing:
        assert nothing is None and sink.TAKE.get() is None


@pytest.mark.parametrize("name", list(FAMILY_STAGES))
def test_a_checkpointed_fit_gives_the_answer_and_counts_of_the_plain_fit_and_the_same_records(
    name: str,
) -> None:
    dataset, stages = FAMILY_STAGES[name]
    plain, plain_counts, none = _fit_on(name, dataset, checkpoints=False)
    first, first_counts, records = _fit_on(name, dataset, checkpoints=True)
    _, _, again = _fit_on(name, dataset, checkpoints=True)
    assert none == []
    assert identity.digest(first) == identity.digest(plain)
    assert first_counts == plain_counts
    assert stages <= {r.stage for r in records}
    assert identity.labels(records) == identity.labels(again)
    assert _same_arrays(records, again)


@pytest.mark.parametrize("name", ["PC", "FCI", "CDNOD", "GFCI", "BOSS-FCI", "FCI-JCI"])
def test_a_skeleton_search_records_every_size_from_zero_in_order_and_then_its_end(
    name: str,
) -> None:
    dataset = min(identity.datasets(name), key=lambda d: identity.draws(d)[0].data.shape[1])
    _, _, records = _fit_on(name, dataset, checkpoints=True)
    skeleton = [(r.stage, r.level) for r in records if r.stage.startswith("skeleton.")]
    sizes = [level for stage, level in skeleton if stage == "skeleton.size"]
    assert sizes == list(range(len(sizes))) and sizes
    assert skeleton[-1] == ("skeleton.end", None)
    assert [stage for stage, _ in skeleton].count("skeleton.end") == 1


@pytest.mark.parametrize("tie_break", ["rank", "position"])
def test_fci_records_its_rule_sweeps_under_either_tie_break(x: np.ndarray, tie_break: str) -> None:
    with andrey.config(tie_break=tie_break):
        plain = andrey.fci(x, seed=0)
        with identity.checkpointing(True) as recorder:
            checkpointed = andrey.fci(x, seed=0)
    assert identity.digest(checkpointed) == identity.digest(plain)
    rules = [r for r in recorder.records if r.stage == "fci.rules"]
    assert rules and all(r.pieces[0].name == "marks" for r in rules)


@pytest.mark.parametrize("stage", ["ages.forward", "ages.backward"])
def test_ages_records_each_moves_limit_with_the_move(stage: str) -> None:
    dataset = FAMILY_STAGES["AGES"][0]
    _, _, records = _fit_on("AGES", dataset, checkpoints=True)
    moves = [r for r in records if r.stage == stage]
    assert moves and all([p.name for p in r.pieces] == ["move", "cpdag", "limit"] for r in moves)


def test_hc_records_its_perturbations_and_climbs(x: np.ndarray) -> None:
    with identity.checkpointing(True) as recorder:
        andrey.hc(x, n_starts=2, n_perturbations=2, seed=0)
    assert {"hc.climb", "hc.perturb"} <= {r.stage for r in recorder.records}


def test_hc_records_the_total_of_each_start(x: np.ndarray) -> None:
    with identity.checkpointing(True) as recorder:
        out = andrey.hc(x, n_starts=3, n_perturbations=2, seed=0)
    totals = [r for r in recorder.records if r.stage == "hc.total"]
    assert len(totals) == 3 and all([p.name for p in r.pieces] == ["total"] for r in totals)
    best = min(float(r.pieces[0].array[0]) for r in totals)
    assert best == pytest.approx(out.metadata["score"], abs=1e-9)


def test_tabu_records_the_total_after_each_move(x: np.ndarray) -> None:
    with identity.checkpointing(True) as recorder:
        out = andrey.tabu(x, n_starts=2, seed=0)
    moves = [r for r in recorder.records if r.stage == "tabu.move"]
    assert moves and all([p.name for p in r.pieces] == ["move", "parents", "total"] for r in moves)
    best = min(float(r.pieces[2].array[0]) for r in moves)
    assert best == pytest.approx(out.metadata["score"], abs=1e-9)


def test_ges_records_only_the_moves_the_knowledge_admits(monkeypatch) -> None:
    """A move whose class holds no DAG that satisfies the knowledge is skipped for the next best
    one, and changes nothing, so no checkpoint is recorded for it."""
    ges_module = importlib.import_module("andrey.search._parallel_ges")
    real, verdicts = ges_module.admits, []

    def rejecting_every_other_move(cpdag: np.ndarray, constraints: Any) -> bool:
        verdicts.append(len(verdicts) % 2 == 1 and real(cpdag, constraints))
        return verdicts[-1]

    monkeypatch.setattr(ges_module, "admits", rejecting_every_other_move)
    data = np.ascontiguousarray(identity.draws("linear_gauss_er_d20")[0].data)
    with identity.checkpointing(True) as recorder:
        andrey.ges(data, knowledge=andrey.Knowledge(forbidden=[(0, 1)]))
    stages = [r.stage for r in recorder.records if r.stage != "ges.total"]
    assert not all(verdicts) and any(verdicts)
    assert len(stages) == sum(verdicts)
    assert {"ges.forward", "ges.backward"} <= set(stages)


def test_ges_records_its_recomputed_total_once_after_the_search(x: np.ndarray) -> None:
    with identity.checkpointing(True) as recorder:
        out = andrey.ges(x)
    stages = [r.stage for r in recorder.records]
    assert stages.count("ges.total") == 1 and stages[-1] == "ges.total"
    assert float(recorder.records[-1].pieces[0].array[0]) == out.metadata["score"]


@pytest.mark.parametrize("name", ["XGES", "GIES"])
def test_xges_and_gies_end_with_the_recomputed_total_they_return(name: str) -> None:
    dataset, _ = FAMILY_STAGES[name]
    out, _, records = _fit_on(name, dataset, checkpoints=True)
    assert [r.stage for r in records].count(f"{name.lower()}.total") == 1
    assert records[-1].stage == f"{name.lower()}.total"
    assert float(records[-1].pieces[0].array[0]) == out.metadata["score"]


def test_structural_em_ends_with_every_likelihood_it_reports() -> None:
    dataset, _ = FAMILY_STAGES["GES-em"]
    out, _, records = _fit_on("GES-em", dataset, checkpoints=True)
    assert [r.stage for r in records].count("em.loglik") == 1 and records[-1].stage == "em.loglik"
    assert records[-1].pieces[0].array.tolist() == out.metadata["em"]["log_likelihood"]


def test_pc_records_the_graph_after_conflict_cleanup_and_not_the_one_before_it(
    monkeypatch, x: np.ndarray
) -> None:
    monkeypatch.setattr(
        pc_module, "undirect_conflicts", lambda cpdag, **kwargs: np.zeros_like(cpdag)
    )
    with identity.checkpointing(True) as recorder:
        out = andrey.pc(x)
    stages = [r.stage for r in recorder.records]
    assert stages[-2:] == ["pc.meek", "pc.conflicts"]
    before, after = recorder.records[-2].pieces[0], recorder.records[-1].pieces[0]
    assert before.array.any() and not after.array.any()
    assert not np.asarray(out.structure.to_numpy()).any()


@pytest.mark.parametrize("name", ["NOTEARS", "DAGMA", "DYNOTEARS"])
def test_a_continuous_search_ends_with_the_result_it_returns(name: str) -> None:
    dataset, _ = FAMILY_STAGES[name]
    out, _, records = _fit_on(name, dataset, checkpoints=True)
    last = records[-1]
    stage = "dynotears.result" if name == "DYNOTEARS" else "continuous.result"
    assert last.stage == stage and [r.stage for r in records].count(stage) == 1
    pieces = {p.name: p.array for p in last.pieces}
    if out.weighted_adjacency is not None:
        assert np.array_equal(pieces["weights"], np.asarray(out.weighted_adjacency).ravel())
    assert float(pieces["h_final"][0]) == out.metadata["h_final"]
    assert int(pieces["edges_removed"][0]) == out.metadata["edges_removed"]


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
def test_calm_records_its_returned_support_and_objective_after_the_rounds(x: np.ndarray) -> None:
    pytest.importorskip("torch")
    with identity.checkpointing(True) as recorder:
        out = andrey.calm(x, seed=0, max_iter=2, subproblem_iter=60)
    last = recorder.records[-1]
    support, weighted, objective = last.pieces
    assert last.stage == "calm.result"
    assert (support.name, weighted.name, objective.name) == ("support", "weighted", "objective")
    assert float(objective.array[0]) == out.metadata["objective"]
    assert np.array_equal(weighted.array, np.asarray(out.weighted_adjacency).ravel())
    assert np.array_equal(support.array != 0, weighted.array != 0)


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
def test_calm_records_each_outer_round_without_changing_the_answer(x: np.ndarray) -> None:
    pytest.importorskip("torch")

    def fit() -> Any:
        return andrey.calm(x, seed=0, max_iter=2, subproblem_iter=60)

    plain = fit()
    with identity.checkpointing(True) as recorder:
        checkpointed = fit()
    assert identity.digest(checkpointed) == identity.digest(plain)
    assert [r.stage for r in recorder.records].count("calm.outer") == 2


def test_the_refits_run_with_the_sink_unset(monkeypatch) -> None:
    sink = importlib.import_module("andrey.core._checkpoint")
    seen: list[bool] = []

    def call(x: np.ndarray) -> StructureOutput:
        seen.append(sink.TAKE.get() is not None)
        return andrey.ges(x)

    monkeypatch.setitem(identity.FITS, "GES", identity.Fit(call))
    monkeypatch.setattr(identity, "datasets", lambda method: ("asia_gauss",))
    identity.run(["GES"], seeds=1, shuffle=2, checkpoints={})
    assert seen == [True, False, False]


def test_a_piece_is_serialized_at_once_so_a_later_change_does_not_reach_its_record() -> None:
    marks, sets = np.zeros((3, 3), dtype=np.int8), {(0, 1): (2,)}
    recorder = identity.Checkpoints()
    recorder("stage", 1, marks=marks, sepsets=sets, parents=[0b110, 0, 0])
    before = [p.digest for p in recorder.records[0].pieces]
    marks[1, 2] = 2
    sets[(0, 1)] = ()
    assert [p.digest for p in recorder.records[0].pieces] == before


def test_a_checkpoint_site_reads_the_sink_once_and_builds_nothing_when_it_is_unset() -> None:
    """Each site is ``if (take := _checkpoint.TAKE.get()) is not None:``, so no argument is built
    unless the harness is recording."""
    guard = "if (take := _checkpoint.TAKE.get()) is not None:"
    for path in sorted(Path(andrey.__file__).parent.rglob("*.py")):
        text = path.read_text()
        if path.name == "_checkpoint.py":
            continue
        assert text.count("_checkpoint.TAKE.get()") == text.count(guard), path.name
        assert len(re.findall(re.escape(guard) + r"\n\s+take\(", text)) == text.count(guard), (
            path.name
        )


def test_no_module_of_the_package_imports_the_quality_tools() -> None:
    import ast

    for path in sorted(Path(andrey.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            assert not any(n == "qa" or n.startswith("qa.") for n in names), path.name


# ---- forms and the element an index reads ----------------------------------------------------


def _piece(name: str, value: Any) -> tuple[Any, identity.Label]:
    recorder = identity.Checkpoints()
    recorder("stage", None, **{name: value})
    (piece,) = recorder.records[0].pieces
    label = identity.Label(0, "stage", None, name, piece.form, piece.width, piece.digest, 0)
    return piece, label


def test_a_symmetric_adjacency_is_its_upper_triangle_and_an_index_reads_as_a_pair() -> None:
    support = np.zeros((4, 4), dtype=bool)
    support[1, 3] = support[3, 1] = True
    piece, label = _piece("adjacency", support)
    assert piece.form == "pairs" and piece.width == 4 and piece.array.tolist() == [0, 0, 0, 0, 1, 0]
    assert identity._element(label, piece.array, 5) == "pair (2, 3)"
    assert identity._element(label, piece.array, 4) == "pair (1, 3)"


def test_a_matrix_is_row_major_and_an_index_reads_as_an_entry() -> None:
    piece, label = _piece("cpdag", np.arange(6, dtype=np.int8).reshape(2, 3))
    assert piece.form == "matrix" and piece.array.tolist() == [0, 1, 2, 3, 4, 5]
    assert identity._element(label, piece.array, 4) == "entry (1, 1)"


def test_separating_sets_are_rows_in_key_order_and_an_index_reads_as_a_pair() -> None:
    piece, label = _piece("sepsets", {(2, 4): frozenset(), (0, 1): frozenset({3, 2})})
    assert piece.form == "sets" and piece.array.tolist() == [0, 1, 2, 2, 3, 2, 4, 0]
    assert identity._element(label, piece.array, 3) == "pair (0, 1)"
    assert identity._element(label, piece.array, 6) == "pair (2, 4)"


def test_values_are_rows_of_a_key_and_a_float_and_an_index_reads_as_a_pair() -> None:
    piece, label = _piece("pvalues", {(2, 3): 0.5, (0, 1): 0.25})
    assert piece.form == "values" and piece.width == 3
    assert piece.array.tolist() == [0.0, 1.0, 0.25, 2.0, 3.0, 0.5]
    assert identity._element(label, piece.array, 4) == "pair (2, 3)"


def test_a_sequence_a_scalar_and_a_string_read_by_position_value_and_character() -> None:
    sequence, listed = _piece("members", [3, (1, 2), frozenset({9, 4})])
    assert sequence.form == "list" and sequence.array.tolist() == [3, 1, 2, 4, 9]
    assert identity._element(listed, sequence.array, 3) == "position 3"
    scalar, valued = _piece("total", 1.5)
    assert scalar.form == "scalar" and identity._element(valued, scalar.array, 0) == "value"
    text, written = _piece("phase", "add")
    assert text.form == "text" and identity._element(written, text.array, 1) == "character 1"


def test_a_wide_integer_is_its_limbs_and_a_bitset_row_is_a_matrix_of_children() -> None:
    wide, _ = _piece("graph_key", (1 << 70) + 5)
    assert wide.array.tolist() == [5, 64]
    parents, _ = _piece("parents", [0b110, 0b100, 0])
    assert parents.form == "matrix" and parents.array.tolist() == [0, 1, 1, 0, 0, 1, 0, 0, 0]


def test_a_mapping_that_is_empty_or_a_piece_that_is_none_records_no_elements() -> None:
    recorder = identity.Checkpoints()
    recorder("stage", None, sepsets={}, undecided=None)
    (piece,) = recorder.records[0].pieces
    assert piece.name == "sepsets" and piece.array.size == 0


def test_a_skeleton_size_record_holds_only_the_pairs_that_size_removed() -> None:
    recorder = identity.Checkpoints()
    support = np.ones((4, 4), dtype=bool)
    sepsets: dict[tuple[int, int], tuple[int, ...]] = {(0, 1): ()}
    recorder("skeleton.size", 0, adjacency=support, sepsets=sepsets, pvalues={(0, 1): 0.5})
    sepsets[(2, 3)] = (0,)
    recorder("skeleton.size", 1, adjacency=support, sepsets=sepsets, pvalues={(0, 1): 0.5})
    recorder("skeleton.end", adjacency=support, sepsets=sepsets, pvalues={}, undecided=None)
    recorder("skeleton.size", 0, adjacency=support, sepsets={(1, 2): ()}, pvalues={})
    by_name = [{p.name: p.array.tolist() for p in r.pieces} for r in recorder.records]
    assert by_name[0]["sepsets"] == [0, 1, 0] and by_name[1]["sepsets"] == [2, 3, 1, 0]
    assert by_name[1]["pvalues"] == []
    assert by_name[2]["sepsets"] == [0, 1, 0, 2, 3, 1, 0]
    assert by_name[3]["sepsets"] == [1, 2, 0]


# ---- the first stage where two builds part ---------------------------------------------------


def _fit_record(stage_pieces: list[tuple[str, int | None, dict[str, Any]]]) -> list:
    recorder = identity.Checkpoints()
    for stage, level, pieces in stage_pieces:
        recorder(stage, level, **pieces)
    return recorder.records


def _row(method: str, dataset: str, records: list, output: str = "h") -> dict:
    return {
        "method": method,
        "dataset": dataset,
        "seed": 0,
        "permutation": 0,
        "workers": 1,
        "hash": output,
        "graph_hash": output,
        **identity.checkpoint_fields(records),
    }


def _compared(before: list, after: list, capsys, *, outputs=("h", "h2")) -> tuple[int, str]:
    rows = [[_row("GES", "d", fit, out)] for fit, out in zip((before, after), outputs, strict=True)]
    stages = tuple(
        {tuple(r[k] for k in identity.KEY): identity.stages_of(fit)}
        for r, fit in zip((rows[0][0], rows[1][0]), (before, after), strict=True)
    )
    rc = identity.compare(
        {"header": {}, "rows": rows[0]}, {"header": {}, "rows": rows[1]}, stages=stages
    )
    return rc, capsys.readouterr().out


def test_a_perturbed_stage_is_the_first_difference_and_later_ones_are_not_reported(capsys) -> None:
    move = [("ges.forward", None, {"move": (0, 1, ()), "total": 1.0})]
    base = _fit_record([*move, ("ges.forward", None, {"move": (1, 2, ()), "total": 2.0})])
    head = _fit_record(
        [("ges.forward", None, {"move": (0, 1, ()), "total": 1.5}), *move[:0]]
        + [("ges.forward", None, {"move": (9, 9, ()), "total": 7.0})]
    )
    rc, out = _compared(base, head, capsys)
    assert rc == 1
    assert "first checkpoint difference: GES d 0 0 1: record 1 of 2, ges.forward, total:" in out
    assert "element 0 = value, 1 -> 1.5" in out and out.count("first checkpoint difference") == 1
    assert "fingerprint 1 rows: GES ges.forward total value 0 1 1" in out


def test_bits_are_told_from_values_sequences_and_missing_records(capsys) -> None:
    one = _fit_record([("a", None, {"total": 1.0})])
    nudged = _fit_record([("a", None, {"total": float(np.nextafter(1.0, 2.0))})])
    other = _fit_record([("b", None, {"total": 1.0})])
    longer = _fit_record([("a", None, {"total": 1.0}), ("a", None, {"total": 2.0})])
    kinds = {}
    for name, after in (("bits", nudged), ("sequence", other), ("missing", longer)):
        found = identity.first_difference(identity.stages_of(one), identity.stages_of(after))
        assert found is not None
        kinds[name] = found.kind
    assert kinds == {"bits": "bits", "sequence": "sequence", "missing": "missing"}
    assert identity.first_difference(identity.stages_of(one), identity.stages_of(one)) is None
    rc, out = _compared(one, nudged, capsys)
    assert "(bits)" in out and rc == 1


def test_the_same_bytes_under_another_form_width_or_record_are_a_difference() -> None:
    row = _fit_record([("a", None, {"total": [1.0, 2.0]})])
    matrix = _fit_record([("a", None, {"total": np.array([[1.0, 2.0]])})])
    pair = _fit_record([("a", None, {"x": [1.0], "y": [2.0]})])
    split = _fit_record([("a", None, {"x": [1.0]}), ("a", None, {"y": [2.0]})])
    assert identity.first_difference(identity.stages_of(row), identity.stages_of(row)) is None
    layout = identity.first_difference(identity.stages_of(row), identity.stages_of(matrix))
    assert layout is not None and layout.kind == "sequence" and layout.piece == "total"
    assert (
        layout.where == "a total (record 0, list, width 0) -> a total (record 0, matrix, width 2)"
    )
    regrouped = identity.first_difference(identity.stages_of(pair), identity.stages_of(split))
    assert regrouped is not None and regrouped.kind == "sequence" and regrouped.piece == "y"
    assert "(record 0," in regrouped.where and "(record 1," in regrouped.where


def test_two_fits_that_part_alike_group_under_one_fingerprint_and_one_stage(capsys) -> None:
    def fit(total: float) -> list:
        return _fit_record([("ges.forward", None, {"move": (0, 1, ()), "total": total})])

    rows, stages = [[], []], [{}, {}]
    for side, total in enumerate((1.0, 2.0)):
        for dataset in ("a", "b"):
            records = fit(total)
            rows[side].append(_row("GES", dataset, records, f"h{side}"))
            stages[side][tuple(rows[side][-1][k] for k in identity.KEY)] = identity.stages_of(
                records
            )
    identity.compare(
        {"header": {}, "rows": rows[0]}, {"header": {}, "rows": rows[1]}, stages=tuple(stages)
    )
    out = capsys.readouterr().out
    assert out.count("fingerprint ") == 1
    assert "fingerprint 2 rows: GES ges.forward total value 0 1 1" in out
    assert "by stage 2 rows: ges.forward total value -- GES (a, b)" in out


def test_a_row_with_equal_output_and_different_checkpoints_is_listed_as_information(
    capsys,
) -> None:
    base = _fit_record([("ges.forward", None, {"total": 1.0})])
    head = _fit_record([("ges.forward", None, {"total": 2.0})])
    rc, out = _compared(base, head, capsys, outputs=("h", "h"))
    assert rc == 0
    assert (
        "checkpoints differ, output equal: GES d 0 0 1: record 1, ges.forward total (value)" in out
    )
    assert "first checkpoint difference" not in out and "fingerprint" not in out


def test_the_exit_status_ignores_the_checkpoint_report(capsys) -> None:
    base = _fit_record([("a", None, {"total": 1.0})])
    head = _fit_record([("a", None, {"total": 2.0})])
    for outputs, status in ((("h", "h"), 0), (("h", "other"), 1)):
        rc, _ = _compared(base, head, capsys, outputs=outputs)
        assert rc == status


def test_compare_says_so_when_only_one_manifest_has_checkpoints(capsys) -> None:
    with_records = _row("GES", "d", _fit_record([("a", None, {"total": 1.0})]))
    without = {k: v for k, v in with_records.items() if not k.startswith("checkpoint")}
    rc = identity.compare({"header": {}, "rows": [with_records]}, {"header": {}, "rows": [without]})
    assert rc == 0
    assert "checkpoints: only one manifest has them" in capsys.readouterr().out


def test_an_array_of_more_than_two_dimensions_keeps_its_shape_as_a_piece() -> None:
    block = np.arange(24.0)
    base = _fit_record([("a", None, {"weights": block.reshape(2, 3, 4)})])
    same = _fit_record([("a", None, {"weights": block.reshape(2, 3, 4)})])
    reshaped = _fit_record([("a", None, {"weights": block.reshape(3, 2, 4)})])
    assert [p.name for p in base[0].pieces] == ["weights", "weights.shape"]
    assert base[0].pieces[1].array.tolist() == [2, 3, 4]
    assert identity.first_difference(identity.stages_of(base), identity.stages_of(same)) is None
    found = identity.first_difference(identity.stages_of(base), identity.stages_of(reshaped))
    assert found is not None and found.piece == "weights.shape" and found.kind == "value"
    assert identity.labels(base)[0].digest == identity.labels(reshaped)[0].digest
    for shape in ((24,), (4, 6)):
        flat = _fit_record([("a", None, {"weights": block.reshape(shape)})])
        assert [p.name for p in flat[0].pieces] == ["weights"]


def test_a_sequence_difference_names_the_level_of_each_build() -> None:
    skipping = _fit_record([("a", 1, {"total": 1.0})])
    full = _fit_record([("a", 0, {"total": 1.0}), ("a", 1, {"total": 1.0})])
    found = identity.first_difference(identity.stages_of(full), identity.stages_of(skipping))
    assert found is not None and found.kind == "sequence"
    assert found.where == "a level 0 total -> a level 1 total"
    other = _fit_record([("b", None, {"total": 1.0})])
    found = identity.first_difference(identity.stages_of(skipping), identity.stages_of(other))
    assert found is not None and found.where == "a level 1 total -> b total"


def test_the_report_gives_the_record_counts_of_both_fits_when_they_differ(capsys) -> None:
    one = _fit_record([("a", None, {"total": 1.0})])
    two = _fit_record([("a", None, {"total": 1.0}), ("a", None, {"total": 2.0})])
    _, out = _compared(one, two, capsys)
    assert "GES d 0 0 1: record 2 (1 before, 2 after), a, total: in one fit only" in out
    _, out = _compared(two, one, capsys)
    assert "GES d 0 0 1: record 2 (2 before, 1 after), a, total: in one fit only" in out
    _, out = _compared([], one, capsys)
    assert "GES d 0 0 1: record 1 (0 before, 1 after), a, total: in one fit only" in out
    assert " of 1," not in out and " of 0," not in out


def _compared_rows(pairs: list[tuple[dict, dict]], capsys) -> tuple[int, str]:
    """Compare the manifests whose rows are the first and the second of each pair, each row with
    the stages it carries under ``_records``."""
    sides: tuple[list, list] = ([], [])
    stages: tuple[dict, dict] = ({}, {})
    for pair in pairs:
        for side, row in enumerate(pair):
            records = row.pop("_records")
            sides[side].append(row)
            stages[side][tuple(row[k] for k in identity.KEY)] = identity.stages_of(records)
    rc = identity.compare(
        {"header": {}, "rows": sides[0]}, {"header": {}, "rows": sides[1]}, stages=stages
    )
    return rc, capsys.readouterr().out


def _with_parts(method: str, records: list, output: str, **parts: str) -> dict:
    return {**_row(method, "a", records, output), "parts": parts, "_records": records}


def test_a_fit_with_equal_stages_and_a_different_output_parts_after_the_last_stage(capsys) -> None:
    records = _fit_record(
        [("hc.climb", None, {"parents": [0b10, 0]}), ("hc.total", None, {"total": 1.0})]
    )
    pairs = [
        (
            _with_parts(method, records, "h1", graph="g", edges="e", weights="w"),
            _with_parts(method, records, "h2", graph="g2", edges="e2", weights="w"),
        )
        for method in ("HC", "Tabu")
    ]
    rc, out = _compared_rows(pairs, capsys)
    assert rc == 1
    assert "first checkpoint difference: HC a 0 0 1: after all 2 records, the output differs" in out
    assert "in edges, graph (after)" in out and "Tabu a 0 0 1: after all 2 records" in out
    assert "fingerprint 1 rows: HC output edges, graph after - 0 0" in out
    assert "by stage 2 rows: output edges, graph after -- HC (a), Tabu (a)" in out


def test_a_fit_with_equal_stages_and_equal_output_or_no_stages_is_not_reported(capsys) -> None:
    records = _fit_record([("hc.total", None, {"total": 1.0})])
    equal = (_with_parts("HC", records, "h", graph="g"), _with_parts("HC", records, "h", graph="g"))
    rc, out = _compared_rows([equal], capsys)
    assert rc == 0 and "checkpoint difference" not in out and "by stage" not in out
    plain = [
        {**_row("MMPC", "a", [], output), "parts": {"graph": output}, "_records": []}
        for output in ("h1", "h2")
    ]
    rc, out = _compared_rows([(plain[0], plain[1])], capsys)
    assert rc == 1 and "checkpoint difference" not in out and "by stage" not in out
    bare = (_with_parts("HC", records, "h1"), _with_parts("HC", records, "h2"))
    _, out = _compared_rows([bare], capsys)
    assert "after all 1 records, the output differs in hash (after)" in out


# ---- a stage a planted change reaches --------------------------------------------------------


def _recorded(method, datasets, monkeypatch, *patches, seeds=1) -> tuple[dict, dict]:
    """The manifest and the stages of ``method`` on ``datasets`` with ``patches`` applied."""
    with monkeypatch.context() as patched:
        patched.setattr(identity, "datasets", lambda _method: datasets)
        for patch in patches:
            patch(patched)
        records: dict[tuple, list] = {}
        rows, _ = identity.run([method], seeds, checkpoints=records)
    return {"header": {}, "rows": rows}, {k: identity.stages_of(v) for k, v in records.items()}


def _keep_the_last_pair_removed_at_size_two(patched) -> None:
    """Drop the last removal of conditioning size two and every later removal of that pair, so the
    pair stays in the skeleton."""
    scan, kept = pc_module.scan_pairs, set()

    def keeping(test, pairs, neighbors, size, *args, **kwargs):
        removals = scan(test, pairs, neighbors, size, *args, **kwargs)
        if size == 2:
            kept.add(removals[-1][:2])
            return removals[:-1]
        return [r for r in removals if r[:2] not in kept]

    patched.setattr(pc_module, "scan_pairs", keeping)


def _end_the_first_directed_edge_after_meek(patched) -> None:
    meek = pc_module.meek

    def ending(*args, **kwargs):
        cpdag = np.array(meek(*args, **kwargs))
        i, j = np.argwhere(cpdag == andrey.core.ARROW)[0]
        cpdag[i, j] = andrey.core.TAIL
        return cpdag

    patched.setattr(pc_module, "meek", ending)


D50 = ("linear_gauss_er_d50",)


def test_a_dropped_removal_is_the_first_stage_that_differs_with_the_pair_that_stayed(
    monkeypatch, capsys
) -> None:
    base, base_stages = _recorded("PC", D50, monkeypatch)
    sizes = [lab for lab in next(iter(base_stages.values())).labels if lab.stage == "skeleton.size"]
    removed = [lab for lab in sizes if lab.level == 2 and lab.piece == "sepsets" and lab.length]
    assert removed, "the unperturbed fit removes a pair at size 2, so the patch has an effect"
    head, head_stages = _recorded("PC", D50, monkeypatch, _keep_the_last_pair_removed_at_size_two)
    rc = identity.compare(base, head, stages=(base_stages, head_stages))
    out = capsys.readouterr().out
    assert rc == 1
    assert "skeleton.size level 2, adjacency: element " in out and "0 -> 1" in out
    assert out.count("first checkpoint difference") == 1
    assert "pc.colliders" not in out and "pc.meek" not in out
    assert "fingerprint 1 rows: PC skeleton.size level 2 adjacency value" in out


def test_the_front_most_difference_hides_the_one_behind_it_until_it_is_fixed(
    monkeypatch, capsys
) -> None:
    base, base_stages = _recorded("PC", D50, monkeypatch)
    both, both_stages = _recorded(
        "PC",
        D50,
        monkeypatch,
        _keep_the_last_pair_removed_at_size_two,
        _end_the_first_directed_edge_after_meek,
    )
    identity.compare(base, both, stages=(base_stages, both_stages))
    first = capsys.readouterr().out
    assert "skeleton.size level 2" in first and "pc.meek" not in first
    behind, behind_stages = _recorded(
        "PC", D50, monkeypatch, _end_the_first_directed_edge_after_meek
    )
    identity.compare(base, behind, stages=(base_stages, behind_stages))
    second = capsys.readouterr().out
    assert "skeleton.size" not in second
    assert "fingerprint 1 rows: PC pc.meek cpdag value" in second


def test_one_planted_change_on_two_datasets_gives_two_fingerprints_and_one_stage(
    monkeypatch, capsys
) -> None:
    sets = ("linear_gauss_er_d20", "asia_gauss")
    base, base_stages = _recorded("PC", sets, monkeypatch)
    head, head_stages = _recorded("PC", sets, monkeypatch, _end_the_first_directed_edge_after_meek)
    identity.compare(base, head, stages=(base_stages, head_stages))
    out = capsys.readouterr().out
    assert out.count("\nfingerprint 1 rows: PC pc.meek cpdag value") == 2
    assert out.count("\nby stage ") == 1
    assert "by stage 2 rows: pc.meek cpdag value -- PC (asia_gauss, linear_gauss_er_d20)" in out


def test_one_planted_change_on_two_fits_of_one_draw_gives_one_fingerprint(
    monkeypatch, capsys
) -> None:
    monkeypatch.setitem(identity.LARGER_DATASETS, "again", lambda: identity.draws("asia_gauss"))
    pair = ("asia_gauss", "again")
    base, base_stages = _recorded("PC", pair, monkeypatch)
    head, head_stages = _recorded("PC", pair, monkeypatch, _end_the_first_directed_edge_after_meek)
    identity.compare(base, head, stages=(base_stages, head_stages))
    out = capsys.readouterr().out
    assert out.count("\nfingerprint ") == 1
    assert "\nfingerprint 2 rows: PC pc.meek cpdag value" in out


# ---- the manifest and the sidecar ------------------------------------------------------------


def test_with_the_mode_off_a_row_gains_nothing_and_with_it_on_only_the_two_fields(
    monkeypatch,
) -> None:
    monkeypatch.setattr(identity, "datasets", lambda method: ("asia_gauss",))
    for method in ("PC", "GES", "IAMB"):
        plain, _ = identity.run([method], seeds=1)
        records: dict[tuple, list] = {}
        recorded, _ = identity.run([method], seeds=1, checkpoints=records)
        assert all(not k.startswith("checkpoint") for r in plain for k in r)
        fields = ("checkpoints", "checkpoint_records")
        assert [{k: v for k, v in r.items() if k not in fields} for r in recorded] == plain
        assert all(r["checkpoint_records"] > 0 for r in recorded)
        assert identity.header({}, 1, 0, 1).keys() == identity.header({}, 1, 0, 1, None).keys()
    assert "checkpoints" in identity.header({}, 1, 0, 1, "m.json.checkpoints.npz")


def test_a_manifest_and_its_sidecar_compare_end_to_end(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(identity, "datasets", lambda method: ("asia_gauss",))
    base, head = tmp_path / "base.json", tmp_path / "head.json"
    assert identity.main(["PC", "-o", str(base), "--checkpoints", "--seeds", "1"]) == 0
    with monkeypatch.context() as patched:
        _end_the_first_directed_edge_after_meek(patched)
        assert identity.main(["PC", "-o", str(head), "--checkpoints", "--seeds", "1"]) == 0
    sidecar = tmp_path / "base.json.checkpoints.npz"
    assert (
        sidecar.is_file() and json.loads(base.read_text())["header"]["checkpoints"] == sidecar.name
    )
    capsys.readouterr()
    assert identity.main(["--compare", str(base), str(base)]) == 0
    assert "first checkpoint difference" not in capsys.readouterr().out
    assert identity.main(["--compare", str(base), str(head)]) == 1
    out = capsys.readouterr().out
    assert "first checkpoint difference: PC asia_gauss 0 0 1: record " in out and "pc.meek" in out


def _checkpointed_pc_manifests(tmp_path, monkeypatch) -> tuple[Path, Path, Path]:
    """The manifests of two plain fits and of a third that parts at its second stage, each with
    its sidecar."""
    monkeypatch.setattr(identity, "datasets", lambda method: ("asia_gauss",))
    paths = tuple(tmp_path / f"{name}.json" for name in ("base", "head", "other"))
    for path in paths[:2]:
        assert identity.main(["PC", "-o", str(path), "--checkpoints", "--seeds", "1"]) == 0
    with monkeypatch.context() as patched:
        _end_the_first_directed_edge_after_meek(patched)
        assert identity.main(["PC", "-o", str(paths[2]), "--checkpoints", "--seeds", "1"]) == 0
    return paths


def test_a_sidecar_that_is_not_the_manifests_is_refused(tmp_path, monkeypatch) -> None:
    base, _, other = _checkpointed_pc_manifests(tmp_path, monkeypatch)
    rows = json.loads(base.read_text())["rows"]
    identity.read_stages(tmp_path / identity.sidecar_name(base), rows)
    with pytest.raises(ValueError, match="not the manifest's"):
        identity.read_stages(tmp_path / identity.sidecar_name(other), rows)
    moved = [{**r, "seed": 99} for r in rows]
    with pytest.raises(ValueError, match="has no checkpoints"):
        identity.read_stages(tmp_path / identity.sidecar_name(base), moved)


def test_compare_says_when_a_sidecar_is_not_its_manifests_and_still_compares_the_outputs(
    tmp_path, monkeypatch, capsys
) -> None:
    base, head, other = _checkpointed_pc_manifests(tmp_path, monkeypatch)
    (tmp_path / identity.sidecar_name(head)).write_bytes(
        (tmp_path / identity.sidecar_name(other)).read_bytes()
    )
    capsys.readouterr()
    assert identity.main(["--compare", str(base), str(head)]) == 0
    out = capsys.readouterr().out
    assert "checkpoints not compared: the sidecar's checkpoints of the fit" in out
    assert "first checkpoint difference" not in out
    assert "0 with a different graph, 0 with only different bits" in out


@pytest.mark.parametrize("damage", ["garbage", "truncated"])
def test_compare_goes_on_without_checkpoints_when_a_sidecar_cannot_be_read(
    tmp_path, monkeypatch, capsys, damage: str
) -> None:
    base, head, _ = _checkpointed_pc_manifests(tmp_path, monkeypatch)
    sidecar = tmp_path / identity.sidecar_name(head)
    sidecar.write_bytes(
        b"not an archive"
        if damage == "garbage"
        else sidecar.read_bytes()[: sidecar.stat().st_size // 2]
    )
    capsys.readouterr()
    assert identity.main(["--compare", str(base), str(head)]) == 0
    out = capsys.readouterr().out
    assert "checkpoints not compared:" in out and "first checkpoint difference" not in out
    assert "0 with a different graph, 0 with only different bits" in out


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_compare_goes_on_without_checkpoints_when_a_piece_of_a_sidecar_cannot_be_read(
    tmp_path, monkeypatch, capsys, damage: str
) -> None:
    base, _, other = _checkpointed_pc_manifests(tmp_path, monkeypatch)
    sidecar = tmp_path / identity.sidecar_name(other)
    with zipfile.ZipFile(sidecar) as archive:
        members = {i.filename: archive.read(i.filename) for i in archive.infolist()}
        spans = {i.filename: (i.header_offset, i.compress_size) for i in archive.infolist()}
    pieces = [name for name in members if not name.endswith("labels.npy")]
    assert pieces
    if damage == "missing":
        with zipfile.ZipFile(sidecar, "w") as archive:
            for name in members.keys() - set(pieces):
                archive.writestr(name, members[name])
    else:
        raw = bytearray(sidecar.read_bytes())
        for name in pieces:
            offset, size = spans[name]
            name_length, extra_length = struct.unpack_from("<HH", raw, offset + 26)
            start = offset + 30 + name_length + extra_length
            for i in range(start, start + size):
                raw[i] ^= 0xFF
        sidecar.write_bytes(bytes(raw))
    capsys.readouterr()
    assert identity.main(["--compare", str(base), str(other)]) == 1
    out = capsys.readouterr().out
    assert "checkpoints not compared: a sidecar piece cannot be read" in out
    assert "first checkpoint difference" not in out
    assert "1 with a different graph" in out


def test_a_piece_that_changes_dtype_differs_at_its_first_element() -> None:
    def fit(value: np.ndarray) -> list:
        return _fit_record([("a", None, {"total": value})])

    before, after = fit(np.array([1], dtype=np.int64)), fit(np.array([1.0]))
    found = identity.first_difference(identity.stages_of(before), identity.stages_of(after))
    assert found is not None and (found.stage, found.piece, found.index) == ("a", "total", 0)


def test_the_structural_em_fit_runs_again_on_workers() -> None:
    assert "GES-em" in identity.WORKER_METHODS


def test_the_checkpoint_mode_needs_a_manifest_and_a_package_that_has_the_sink(
    tmp_path, monkeypatch
) -> None:
    with pytest.raises(SystemExit):
        identity.main(["PC", "--checkpoints"])
    monkeypatch.setitem(sys.modules, "andrey.core._checkpoint", None)
    with pytest.raises(SystemExit):
        identity.main(["PC", "--checkpoints", "-o", str(tmp_path / "m.json")])


def test_a_plain_manifest_compares_from_the_command_line_with_no_checkpoint_report(
    tmp_path, capsys
) -> None:
    manifest = tmp_path / "plain.json"
    manifest.write_text(json.dumps(_manifest()))
    assert identity.main(["--compare", str(manifest), str(manifest)]) == 0
    assert "checkpoint" not in capsys.readouterr().out


@pytest.mark.parametrize("name", ["IAMB", "InterIAMB", "MMPC", "MMHC"])
def test_the_full_graph_searches_are_refitted_on_workers_through_the_pool(
    name, monkeypatch
) -> None:
    """The refit sends every fit to the pool, so the check compares pooled runs, not serial ones."""
    from andrey.constraint import _parallel_targets as tg

    assert name in identity.WORKER_METHODS
    started: list[int] = []
    real = tg.ProcessPoolExecutor
    monkeypatch.setattr(
        tg, "ProcessPoolExecutor", lambda *a, **k: started.append(1) or real(*a, **k)
    )
    rows, findings = identity.run([name], seeds=1, workers=2)
    assert findings == []
    assert {r["workers"] for r in rows} == {1, 2}
    assert started
