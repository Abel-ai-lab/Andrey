"""A test-wise query without the rows its test needs is not tested, in every search family.

The fixed case is a merged-cohort table: one cohort observes ``{S1, S2, S3, A}``, another
``{S1, S2, S3, B}``, and ``k`` rows observe every column. The truth is ``S2 -> S1``, ``S1 -> A``,
``S1 -> B``, ``S1 -> S3``, so ``A`` and ``B`` are separated by ``{S1}`` only on the ``k`` rows.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW, ci
from andrey.temporal.pcmci import pc1

COHORT = 300


def _cohorts(k: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 2 * COHORT + k
    s2 = rng.normal(size=n)
    s1 = 0.8 * s2 + rng.normal(size=n)
    df = pd.DataFrame(
        {
            "S1": s1,
            "S2": s2,
            "S3": 0.7 * s1 + rng.normal(size=n),
            "A": 0.9 * s1 + rng.normal(size=n),
            "B": -0.9 * s1 + rng.normal(size=n),
        }
    )
    df.loc[: COHORT - 1, "B"] = np.nan  # cohort 1 never measures B
    df.loc[COHORT : 2 * COHORT - 1, "A"] = np.nan  # cohort 2 never measures A
    return df


def _series(k: int, seed: int = 0, steps: int = 400) -> pd.DataFrame:
    """A VAR(1) series: A observed in the first half, B in the second, A also at k steps there."""
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, 4))
    for t in range(1, steps):
        e = rng.normal(size=4)
        s1 = 0.5 * x[t - 1, 0] + e[0]
        x[t] = (
            s1,
            0.5 * x[t - 1, 1] + 0.4 * s1 + e[1],
            0.6 * x[t - 1, 0] + 0.3 * x[t - 1, 2] + e[2],
            0.6 * x[t - 1, 0] + 0.3 * x[t - 1, 3] + e[3],
        )
    df = pd.DataFrame(x, columns=["S1", "S2", "A", "B"])
    half = steps // 2
    df.loc[half:, "A"] = np.nan
    df.loc[: half + 1, "B"] = np.nan
    for j in range(k):
        df.loc[half + 20 + 10 * j, "A"] = x[half + 20 + 10 * j, 2]
    return df


def _env(df: pd.DataFrame) -> np.ndarray:
    """An environment that varies inside every cohort, so no block holds it constant."""
    return np.arange(len(df)) % 2


def _cohort_env(df: pd.DataFrame) -> np.ndarray:
    """The environment is the cohort, so it is constant on the rows that observe ``A`` or ``B``
    when ``k = 0``; the rows that observe every column are a third environment."""
    env = np.full(len(df), 2)
    env[:COHORT], env[COHORT : 2 * COHORT] = 0, 1
    return env


TABLE_METHODS = {
    "pc": lambda df: andrey.pc(df),
    "pc_student_t": lambda df: andrey.pc(df, indep_test="student_t"),
    "pc_normal_scores": lambda df: andrey.pc(df, indep_test="normal_scores"),
    "pc_mvpc": lambda df: andrey.pc(df, missing="mvpc"),
    "fci": lambda df: andrey.fci(df),
    "fci_fast": lambda df: andrey.fci(df, possible_dsep="fast"),
    "cdnod": lambda df: andrey.cdnod(df, _env(df)),
    "fci_jci": lambda df: andrey.fci_jci(df, _env(df)),
    "iamb": lambda df: andrey.iamb(df),
    "inter_iamb": lambda df: andrey.inter_iamb(df),
    "mmpc": lambda df: andrey.mmpc(df),
}
SKELETON = ("pc", "pc_student_t", "pc_normal_scores", "pc_mvpc", "fci", "fci_fast", "cdnod")
SERIES_METHODS = {
    "pcmci": lambda df: andrey.pcmci(df, tau_max=1),
    "pcmciplus": lambda df: andrey.pcmciplus(df, tau_max=1),
    "lpcmci": lambda df: andrey.lpcmci(df, tau_max=1),
}


def _adjacent(out, a: str, b: str) -> bool:
    labels = list(out.structure.labels)
    return bool(out.structure.adjacency[labels.index(a), labels.index(b)])


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("name", list(TABLE_METHODS))
def test_every_table_method_runs_when_a_pair_shares_few_or_no_rows(name, k):
    out = TABLE_METHODS[name](_cohorts(k))
    assert out.metadata["missing"] in ("testwise", "mvpc")
    if k == 0:
        assert out.metadata["never_observed_together"] == [["A", "B"]]
        assert ["A", "B"] in out.metadata["undecided_pairs"]
        assert out.metadata["not_tested"] > 0
    else:
        assert "never_observed_together" not in out.metadata


@pytest.mark.parametrize("name", [*SKELETON, "iamb", "inter_iamb", "mmpc"])
def test_a_pair_never_observed_together_stays_adjacent_and_keeps_the_true_edges(name):
    out = TABLE_METHODS[name](_cohorts(0))
    for a, b in (("A", "B"), ("A", "S1"), ("B", "S1")):
        assert _adjacent(out, a, b), (a, b)


@pytest.mark.parametrize("name", list(TABLE_METHODS))
def test_no_collider_rests_on_a_pair_never_observed_together(name):
    out = TABLE_METHODS[name](_cohorts(0))
    if not _adjacent(out, "A", "B") and _adjacent(out, "A", "S1") and _adjacent(out, "B", "S1"):
        into = [out.structure.endpoints(v, "S1")[1] == ARROW for v in ("A", "B")]
        assert not all(into)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("name", list(SERIES_METHODS))
def test_a_link_between_columns_never_observed_together_is_undecided_not_found(name, k):
    out = SERIES_METHODS[name](_series(k))
    assert out.metadata["missing"] == "testwise"
    if k == 0:
        assert out.metadata["never_observed_together"] == [["A", "B"]]
        undecided = out.metadata["undecided_pairs"]
        assert ["A at lag 1", "B"] in undecided or ["A", "B at lag 1"] in undecided
        if name != "lpcmci":  # PC1 and phase 2 condition on no link without a tested result
            assert all({a.split()[0], b.split()[0]} == {"A", "B"} for a, b in undecided)
        p = np.asarray(out.metadata["p_values"], dtype=np.float64)
        labels = list(out.structure.labels)
        a, b = labels.index("A"), labels.index("B")
        assert not (p[..., a, b] <= 0.05).any() and not (p[..., b, a] <= 0.05).any()


@pytest.mark.parametrize("name", ["pc", "fci", "iamb", "mmpc"])
def test_row_and_column_order_change_neither_the_graph_nor_the_metadata(name):
    df = _cohorts(0)
    out = TABLE_METHODS[name](df)
    rows = TABLE_METHODS[name](df.sample(frac=1.0, random_state=1).reset_index(drop=True))
    columns = TABLE_METHODS[name](df[["B", "S3", "A", "S2", "S1"]])
    for other in (rows, columns):
        order = [list(other.structure.labels).index(v) for v in out.structure.labels]
        moved = np.asarray(other.structure.adjacency)[np.ix_(order, order)]
        assert np.array_equal(moved, out.structure.adjacency)
        for key in ("undecided_pairs", "never_observed_together", "not_tested"):
            assert other.metadata[key] == out.metadata[key], key


def test_markov_check_leaves_facts_not_tested_out_of_the_uniformity_test():
    df = _cohorts(1)
    dag = andrey.pc(df.fillna(0.0))
    out = andrey.markov_check(dag, df, seed=0)
    assert out.metadata["not_tested"] > 0
    assert np.isfinite(out.p_values).all() and out.n_tests == len(out.p_values)


def test_pairs_never_observed_together_are_found_in_a_table_of_text():
    from andrey.api._method import never_observed_together

    frame = pd.DataFrame(
        {
            "a": pd.Categorical(["x", "y", None, None]),
            "b": ["u", "v", "u", None],
            "c": [None, None, 1.0, 2.0],
        }
    )
    assert never_observed_together(frame) == [["a", "c"]]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("method", ["cdnod", "fci_jci"])
def test_an_environment_constant_on_a_block_is_not_tested(method, k):
    df = _cohorts(k)
    env = _cohort_env(df)
    out = getattr(andrey, method)(df, env)
    assert out.metadata["missing"] == "testwise"
    if k == 0:  # every block that holds A or B holds one cohort, one environment
        assert out.metadata["not_tested"] > 0
    order = np.random.default_rng(3).permutation(len(df))
    moved = getattr(andrey, method)(df.iloc[order].reset_index(drop=True), env[order])
    assert np.array_equal(moved.structure.adjacency, out.structure.adjacency)
    assert moved.metadata.get("not_tested") == out.metadata.get("not_tested")


@pytest.mark.parametrize("name", ["iamb", "inter_iamb", "mmpc"])
def test_a_target_set_holds_a_variable_never_observed_with_the_target(name):
    out = getattr(andrey, name)(_cohorts(0), target="B")
    key = "parents_and_children" if name == "mmpc" else "markov_blanket"
    assert {"A", "S1"} <= set(out.metadata[key])
    assert ["A", "B"] in out.metadata["undecided_pairs"]


class _InfiniteStatistic:
    """A link test answering every query of ``given`` columns: candidate 2 has an infinite statistic
    and a p-value of 0, candidate 3 is not tested."""

    def statistic_pvalues(self, x, y, conditions):
        statistics = np.where(x == 2, np.inf, np.nan)
        return statistics, np.where(x == 2, 0.0, np.nan)


def test_a_tested_candidate_with_an_infinite_statistic_stays_a_parent():
    undecided: list = []
    parents = pc1(_InfiniteStatistic(), 2, 0, 1, 0.05, [0, 1], max_depth=0, undecided=undecided)
    assert parents == ((0, 1),) and undecided == [(1, 1)]


def test_markov_check_rejects_a_nan_no_deletion_counted(monkeypatch):
    df = _cohorts(1).fillna(0.0)
    dag = andrey.pc(df)
    failing = lambda data, **options: lambda x, y, s=(): float("nan")  # noqa: E731
    monkeypatch.setitem(ci._CI_REGISTRY, "fisherz", failing)
    with pytest.raises(ValueError, match="outside"):
        andrey.markov_check(dag, df, indep_test="fisherz", seed=0)


def test_a_registered_variant_keeps_the_row_floor_of_its_test(monkeypatch):
    variant = partial(ci.make_indep_test, "cmiknn", k_cmi=5)
    monkeypatch.setitem(ci._CI_REGISTRY, "cmiknn_k5", variant)
    x = np.random.default_rng(0).standard_normal((60, 3))
    x[5:, 1] = np.nan  # column 1 observed on 5 rows, fewer than the 6 that k_cmi=5 needs
    test = ci.make_indep_test("cmiknn_k5", x)
    assert test.least_rows(0) == 6
    assert np.isnan(test(0, 1))
