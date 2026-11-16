"""A list of targets: one result per target, each the answer a call for that target alone gives."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.constraint.blanket import LazyTieRanks
from andrey.core.ci import counting_tests

SET_LEARNERS = (
    *("iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc", "hiton_mb", "mmmb"),
    "pcmb",
)
SEQUENTIAL = ("mb_by_mb", "cmb", "mmb_by_mmb")
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


def _same(a, b, *, tests: bool = True) -> None:
    """Assert two results hold the same graph and metadata, ``n_tests`` included when asked."""
    assert a.structure == b.structure
    left, right = dict(a.metadata), dict(b.metadata)
    if not tests:
        left.pop("n_tests", None)
        right.pop("n_tests", None)
    assert left == right


@pytest.mark.parametrize("method", [*SET_LEARNERS, *SEQUENTIAL, "local_isa_ling"])
def test_each_result_is_the_answer_of_its_own_call(method):
    data = _gaussian()
    if method == "local_isa_ling":
        data = data.apply(lambda c: np.sign(c) * c**2)  # non-Gaussian noise
    alone = [_fit(method, data, t) for t in TARGETS]
    with counting_tests() as total:
        outs = _fit(method, data, TARGETS)
    assert isinstance(outs, list)
    assert len(outs) == len(TARGETS)
    for out, one in zip(outs, alone, strict=True):
        _same(out, one, tests=False)
    _same(outs[0], alone[0])
    counts = [out.metadata.get("n_tests") for out in outs]
    if counts[0] is not None:
        assert sum(counts) == total.n
        assert all(c <= one.metadata["n_tests"] for c, one in zip(counts, alone, strict=True))


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


@pytest.mark.parametrize("method", ["mmpc", "hiton_mb", "iamb"])
def test_kept_chi_square_answers_change_no_result(method):
    data = _discrete()
    outs = _fit(method, data, [3, 1, 4, 0], indep_test="chisq")
    for out, t in zip(outs, [3, 1, 4, 0], strict=True):
        _same(out, _fit(method, data, t, indep_test="chisq"), tests=False)


@pytest.mark.parametrize("method", ["iamb", "hiton_mb", "cmb"])
def test_a_list_of_one_is_the_one_result(method):
    data = _gaussian()
    (out,) = _fit(method, data, ["x3"])
    one = _fit(method, data, "x3")
    assert not isinstance(one, list)
    _same(out, one)


@pytest.mark.parametrize("method", ["mmpc", "hiton_mb", "pcmb", "mb_by_mb", "mmb_by_mmb"])
def test_results_do_not_depend_on_the_workers(method):
    data = _gaussian()
    targets = list(range(8))
    serial = _fit(method, data, targets)
    with andrey.config(n_jobs=4), counting_tests() as total:
        pooled = _fit(method, data, targets)
    for a, b in zip(serial, pooled, strict=True):
        _same(a, b)
    assert total.n == sum(out.metadata["n_tests"] for out in serial)


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


@pytest.mark.parametrize("method", ["mmpc", "hiton_mb", "pcmb", "mb_by_mb"])
def test_rows_tested_is_the_answer_of_its_own_call_with_missing_values(method):
    """A step an earlier target made still counts its rows for the later target that reads it."""
    data = _gaussian()
    data.iloc[::7, 2] = np.nan
    data.iloc[3::11, 5] = np.nan
    outs = _fit(method, data, TARGETS)
    assert "rows_tested" in outs[0].metadata
    for out, t in zip(outs, TARGETS, strict=True):
        _same(out, _fit(method, data, t), tests=False)


def test_any_sequence_of_columns_is_a_list():
    data = _gaussian()
    outs = andrey.mmpc(data, target=range(3))
    assert [out.metadata["target"] for out in outs] == ["x0", "x1", "x2"]
    (out,) = andrey.mmpc(data, target=("x3",))
    _same(out, andrey.mmpc(data, target="x3"))


def test_unpickled_ranks_compute_under_the_active_thread_limit():
    """A worker pins its BLAS threads after it unpickles the search, and the ranks it computes
    later run under that pin, not under the limits of the moment it unpickled them."""
    from threadpoolctl import threadpool_info, threadpool_limits

    ranks = LazyTieRanks(np.round(_gaussian().to_numpy()), np.zeros(8, dtype=bool))
    with threadpool_limits(limits=2):
        copy = pickle.loads(pickle.dumps(ranks))
    with threadpool_limits(limits=1), copy._threads():
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
    assert list(copy) == list(ranks)
    with threadpool_limits(limits=2):  # a forked worker inherits the object instead
        inherited = LazyTieRanks(np.round(_gaussian().to_numpy()), np.zeros(8, dtype=bool))
    inherited.use_active_limits()
    with threadpool_limits(limits=1), inherited._threads():
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
