"""A list of targets: one result per target, each the answer a call for that target alone gives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import andrey

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


def test_any_sequence_of_columns_is_a_list():
    data = _gaussian()
    outs = andrey.mmpc(data, target=range(3))
    assert [out.metadata["target"] for out in outs] == ["x0", "x1", "x2"]
    (out,) = andrey.mmpc(data, target=("x3",))
    _same(out, andrey.mmpc(data, target="x3"))
