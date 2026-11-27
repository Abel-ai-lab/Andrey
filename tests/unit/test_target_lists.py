"""A list of targets: one result per target, each the answer a call for that target alone gives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.constraint._parallel_targets import answers
from andrey.constraint.blanket import local_search
from andrey.core import ci_missing
from andrey.core.ci_missing import RowCounts, rows_tested

SET_LEARNERS = ("iamb", "inter_iamb", "mmpc")
TARGETS = [5, 0, 3, 7]


def _gaussian(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A linear-Gaussian DAG over eight variables with a collider, a chain and a hub."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 8))
    noise = rng.standard_normal((n, 8))
    x[:, 0] = noise[:, 0]
    x[:, 1] = noise[:, 1]
    x[:, 2] = 0.8 * x[:, 0] + 0.7 * x[:, 1] + noise[:, 2]
    x[:, 3] = 0.9 * x[:, 2] + noise[:, 3]
    x[:, 4] = noise[:, 4]
    x[:, 5] = 0.7 * x[:, 3] - 0.8 * x[:, 4] + noise[:, 5]
    x[:, 6] = 0.8 * x[:, 5] + noise[:, 6]
    x[:, 7] = 0.6 * x[:, 2] + 0.7 * x[:, 6] + noise[:, 7]
    return pd.DataFrame(x, columns=[f"x{j}" for j in range(8)])


def _discrete(n: int = 1500, seed: int = 1) -> np.ndarray:
    """Binary variables along a chain and a collider, read with the chi-square test."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = (a ^ (rng.random(n) < 0.15)).astype(int)
    c = rng.integers(0, 2, n)
    e = ((b | c) ^ (rng.random(n) < 0.1)).astype(int)
    f = (e ^ (rng.random(n) < 0.2)).astype(int)
    return np.column_stack([a, b, c, e, f]).astype(float)


def _fit(method: str, data, target, **kwargs):
    if method == "icp":
        return andrey.icp(data, env="env", target=target, **kwargs)
    return getattr(andrey, method)(data, target=target, **kwargs)


def _same(a, b) -> None:
    """Assert two results hold the same graph and metadata."""
    assert a.structure == b.structure
    assert dict(a.metadata) == dict(b.metadata)


@pytest.mark.parametrize("method", SET_LEARNERS)
def test_each_result_is_the_answer_of_its_own_call(method):
    data = _gaussian()
    alone = [_fit(method, data, t) for t in TARGETS]
    outs = _fit(method, data, TARGETS)
    assert isinstance(outs, list)
    assert len(outs) == len(TARGETS)
    for out, one in zip(outs, alone, strict=True):
        _same(out, one)


def test_icp_answers_each_target():
    rng = np.random.default_rng(3)
    env = np.repeat([0, 1, 2], 300)
    a = rng.standard_normal(900) + env
    b = 0.8 * a + rng.standard_normal(900)
    c = 0.7 * b + rng.standard_normal(900)
    data = pd.DataFrame({"a": a, "b": b, "c": c, "env": env})
    outs = _fit("icp", data, ["c", "b"])
    for out, t in zip(outs, ["c", "b"], strict=True):
        _same(out, _fit("icp", data, t))


@pytest.mark.parametrize("method", ["mmpc", "iamb"])
def test_kept_chi_square_answers_change_no_result(method):
    data = _discrete()
    outs = _fit(method, data, [3, 1, 4, 0], indep_test="chisq")
    for out, t in zip(outs, [3, 1, 4, 0], strict=True):
        _same(out, _fit(method, data, t, indep_test="chisq"))


@pytest.mark.parametrize("method", ["iamb", "mmpc"])
def test_a_list_of_one_is_the_one_result(method):
    data = _gaussian()
    (out,) = _fit(method, data, ["x3"])
    one = _fit(method, data, "x3")
    assert not isinstance(one, list)
    _same(out, one)


@pytest.mark.parametrize("method", SET_LEARNERS)
def test_results_do_not_depend_on_the_workers(method):
    data = _gaussian()
    targets = list(range(8))
    serial = _fit(method, data, targets)
    with andrey.config(n_jobs=4):
        pooled = _fit(method, data, targets)
    for a, b in zip(serial, pooled, strict=True):
        _same(a, b)


@pytest.mark.parametrize(
    ("target", "message"),
    [([], "empty list"), (["x1", 1], "names a column twice"), (["x1", "nope"], "nope")],
)
def test_a_bad_list_raises(target, message):
    with pytest.raises(ValueError, match=message):
        andrey.mmpc(_gaussian(), target=target)


def test_bootstrap_takes_one_target():
    with pytest.raises(ValueError, match="one target at a time"):
        andrey.bootstrap("mmpc", _gaussian(), target=["x1", "x2"], n_resamples=2)


@pytest.mark.parametrize("n_jobs", [1, 4])
@pytest.mark.parametrize("method", SET_LEARNERS)
def test_rows_tested_is_the_answer_of_its_own_call_with_missing_values(method, n_jobs):
    """A run an earlier target made still counts its rows for the later target that reads it."""
    data = _gaussian()
    data.iloc[::7, 2] = np.nan
    data.iloc[3::11, 5] = np.nan
    with andrey.config(n_jobs=n_jobs):
        outs = _fit(method, data, TARGETS)
    assert "rows_tested" in outs[0].metadata
    for out, t in zip(outs, TARGETS, strict=True):
        _same(out, _fit(method, data, t))


def test_a_tuple_of_columns_is_a_list():
    data = _gaussian()
    outs = andrey.mmpc(data, target=(0, 1, 2))
    assert [out.metadata["target"] for out in outs] == ["x0", "x1", "x2"]
    (out,) = andrey.mmpc(data, target=("x3",))
    _same(out, andrey.mmpc(data, target="x3"))


@pytest.mark.parametrize("method", ["mmpc", "icp"])
@pytest.mark.parametrize("target", [np.arange(2), {"x1", "x2"}, pd.Index(["x1", "x2"]), range(2)])
def test_an_unordered_or_other_collection_raises(method, target):
    data = _gaussian().assign(env=np.repeat([0, 1], 300))
    with pytest.raises(TypeError, match="list or tuple"):
        _fit(method, data, target)


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_a_randomized_test_with_a_seed_answers_as_separate_calls(n_jobs):
    """GCM draws from the seed, so each listed answer is the one a call with that seed gives."""
    rng = np.random.default_rng(4)
    a = rng.standard_normal(100)
    b = 0.9 * a + rng.standard_normal(100)
    c = 0.8 * b + rng.standard_normal(100)
    data = pd.DataFrame({"a": a, "b": b, "c": c, "d": rng.standard_normal(100)})
    with andrey.config(n_jobs=n_jobs):
        outs = andrey.mmpc(data, target=["c", "b"], indep_test="gcm", seed=7)
    for out, t in zip(outs, ["c", "b"], strict=True):
        _same(out, andrey.mmpc(data, target=t, indep_test="gcm", seed=7))


def _bits(out) -> tuple:
    """A result as bytes: its graph, kind, and labels, and its metadata as printed, float bits
    included."""
    graph = out.structure
    adjacency = np.asarray(graph.to_numpy())
    return adjacency.tobytes(), adjacency.shape, graph.kind, graph.labels, repr(dict(out.metadata))


def _apart(data: np.ndarray, seed: int = 0) -> np.ndarray:
    """``data`` with some values missing, columns 0 and 1 never observed together, and column 2
    observed with column 0 on eight rows only."""
    x = np.array(data, dtype=float)
    rng = np.random.default_rng(seed)
    x[rng.random(x.shape) < 0.05] = np.nan
    half = rng.permutation(len(x))
    x[half[: len(x) // 2], 0] = np.nan
    x[half[len(x) // 2 :], 1] = np.nan
    x[half[len(x) // 2 + 8 :], 2] = np.nan
    return x


def _counts(rows: RowCounts) -> tuple:
    """Every count of a counter: its rows, ``p = 1`` answers, queries not tested by pair, largest
    p-values, and untested sets."""
    return (
        dict(rows),
        rows.p_one,
        dict(rows.not_tested),
        rows.not_tested_elsewhere,
        rows.largest,
        rows.untested_sets,
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("method", SET_LEARNERS)
@pytest.mark.parametrize("kind", ["gaussian", "chisq"])
def test_a_list_over_columns_never_observed_together_is_each_call_alone_bit_for_bit(
    kind, method, seed
):
    """The list's one search answers each target with the rows, the queries not tested, and the
    undecided pairs of the call for that target alone, in any order of the targets."""
    if kind == "gaussian":
        x, kwargs = _apart(_gaussian(seed=seed).to_numpy(), seed), {}
    else:
        x, kwargs = _apart(_discrete(seed=seed), seed), {"indep_test": "chisq"}
    targets = [3, 0, 4, 1, 2][: x.shape[1]]
    outs = _fit(method, x, targets, missing="testwise", **kwargs)
    assert len(outs) == len(targets)
    for out, t in zip(outs, targets, strict=True):
        assert _bits(out) == _bits(_fit(method, x, t, missing="testwise", **kwargs))
    assert any("undecided_pairs" in out.metadata for out in outs)
    assert any("not_tested" in out.metadata for out in outs)


@pytest.mark.parametrize("method", ["mmpc", "iamb"])
def test_a_list_runs_each_learner_run_once_and_counts_it_for_each_target_that_reads_it(method):
    """MMPC's targets read each other's runs in the symmetry check, so a target's counter holds the
    rows of runs another target's answer made."""
    x = _apart(_gaussian().to_numpy())
    settings = {"alpha": 0.05, "indep_test": "fisherz"}
    targets = [3, 0, 4, 1, 2, 5]
    searches: list = []

    def search_of():
        searches.append(local_search(x, **settings))
        return searches[-1]

    def members(search, t):
        pc, mb, _ = search.local(method, t)
        return pc if mb is None else mb

    found = answers(search_of, members, targets)
    assert len(searches) == 1
    assert isinstance(searches[0].test, ci_missing.TestwiseDeletion)
    read = 0
    for (members_of, rows), t in zip(found, targets, strict=True):
        alone = local_search(x, **settings)
        with alone.pinned(), rows_tested() as expected:
            expected_members = members(alone, t)
        assert members_of == expected_members
        assert _counts(rows) == _counts(expected)
        read += len(alone._runs)
    if method == "mmpc":
        assert len(searches[0]._runs) < read


def test_the_tests_an_answer_makes_outside_any_run_are_counted_for_it():
    """A test an answer asks itself, as a learner's spouse step does, counts after the runs it
    read, and a run another answer made counts again for the answer that reads it."""
    x = _apart(_gaussian().to_numpy())
    settings = {"alpha": 0.05, "indep_test": "fisherz"}

    def answer(search, target):
        search.run("mmpc", target)
        search.test(3, 5, [4])
        return target

    shared = local_search(x, **settings)
    with shared.pinned():
        got = [shared.counted(lambda t=t: answer(shared, t)) for t in (4, 5)]
    for (found, rows), t in zip(got, (4, 5), strict=True):
        alone = local_search(x, **settings)
        with alone.pinned(), rows_tested() as expected:
            answer(alone, t)
        assert found == t
        assert _counts(rows) == _counts(expected)


@pytest.mark.parametrize("method", SET_LEARNERS)
def test_a_list_raises_the_error_of_its_first_failing_target(method):
    """A column constant where another is observed raises in the targets that test the pair; the
    list raises what the first of them raises alone."""
    x = _apart(_gaussian().to_numpy())
    seen = ~np.isnan(x[:, 3])
    x[seen, 2] = 1.5
    x[np.random.default_rng(0).random(len(x)) < 0.5, 3] = np.nan
    x[~np.isnan(x[:, 3]), 2] = 1.5
    targets = [5, 0, 2, 3]

    def raised(target):
        try:
            _fit(method, x, target, missing="testwise")
        except ValueError as error:
            return str(error)
        return None

    messages = [raised(t) for t in targets]
    first = next(message for message in messages if message is not None)
    assert raised(targets) == first
