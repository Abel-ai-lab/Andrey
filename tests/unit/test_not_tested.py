"""A test-wise query without the rows its test needs is not tested, in every search family.

The fixed case is a merged-cohort table: one cohort observes ``{S1, S2, S3, A}``, another
``{S1, S2, S3, B}``, and ``k`` rows observe every column. The truth is ``S2 -> S1``, ``S1 -> A``,
``S1 -> B``, ``S1 -> S3``, so ``A`` and ``B`` are separated by ``{S1}`` only on the ``k`` rows.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.api._method import _not_tested_notice
from andrey.constraint.blanket import LocalSearch, local_search
from andrey.core import ARROW, ci
from andrey.core.ci_missing import NOT_TESTED, rows_tested
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
def test_a_pair_never_observed_together_is_left_out_and_the_true_edges_stay(name):
    out = TABLE_METHODS[name](_cohorts(0))
    assert not _adjacent(out, "A", "B")
    assert ["A", "B"] in out.metadata["undecided_pairs"]
    for a, b in (("A", "S1"), ("B", "S1")):
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
        # No test conditions on a link without a tested result, so only the links of A and B are
        # left undecided.
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
        for key in (
            "undecided_pairs",
            "possibly_separable",
            "never_observed_together",
            "not_tested",
        ):
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
def test_a_target_set_leaves_out_a_variable_never_observed_with_the_target(name):
    out = getattr(andrey, name)(_cohorts(0), target="B")
    key = "parents_and_children" if name == "mmpc" else "markov_blanket"
    assert "S1" in out.metadata[key] and "A" not in out.metadata[key]
    assert ["A", "B"] in out.metadata["undecided_pairs"]


def _neighbors_seen_together_on_three_rows() -> pd.DataFrame:
    """``x1 -> x0 <- x2`` with ``x1`` and ``x2`` observed together on 3 rows, so a test of ``x0``
    and one of them given the other has too few rows."""
    rng = np.random.default_rng(0)
    x1, x2, noise = rng.normal(size=(3, 600))
    df = pd.DataFrame({"x0": x1 + x2 + noise, "x1": x1, "x2": x2})
    df.loc[300:596, "x1"] = np.nan
    df.loc[:299, "x2"] = np.nan
    return df


@pytest.mark.parametrize(
    "name",
    ["iamb", "inter_iamb", "gs", "fast_iamb", "hiton_pc", "hiton_mb", "mmpc", "mmmb", "pcmb"],
)
def test_a_test_with_too_few_rows_drops_no_neighbor_of_the_target(name):
    out = getattr(andrey, name)(_neighbors_seen_together_on_three_rows(), target="x0")
    sets = [
        out.metadata[k] for k in ("markov_blanket", "parents_and_children") if k in out.metadata
    ]
    assert sets and all(sorted(s) == ["x1", "x2"] for s in sets)


def test_total_conditioning_leaves_a_variable_not_tested_given_the_rest_undecided():
    """Given the rest, x1 tests dependent, x2 independent, and x3 is not tested: x3 has no test that
    had rows, so it is left out with no separating set."""
    answers = {1: 0.0, 2: 0.5, 3: NOT_TESTED}

    def test(x, y, condition_set=()):
        return answers[max(x, y)]

    search = LocalSearch(test, [0, 1, 2, 3], 0.05)
    assert search.run("total_conditioning", 0) == ([1], {2: (1, 3)})
    assert search.undecided == {(0, 3)}


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


def _never_together() -> pd.DataFrame:
    """``X <- Z -> Y``: the first half of the rows observes only ``X`` and ``Z``, the second only
    ``Y`` and ``Z``, so ``X`` and ``Y`` are never observed together (a case from a user report)."""
    rng = np.random.default_rng(0)
    n = 1000
    z = rng.normal(size=n)
    x = 0.8 * z + rng.normal(scale=0.6, size=n)
    y = 0.8 * z + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})
    df.loc[: n // 2 - 1, "Y"] = np.nan
    df.loc[n // 2 :, "X"] = np.nan
    return df


@pytest.mark.parametrize(
    "fit",
    [
        partial(andrey.pc, missing="testwise"),
        partial(andrey.pc, missing="mvpc"),
        partial(andrey.fci, missing="testwise"),
    ],
)
def test_a_pair_never_observed_together_is_undecided_and_makes_no_collider(fit):
    out = fit(_never_together())
    assert _adjacent(out, "X", "Z") and _adjacent(out, "Y", "Z")
    assert not _adjacent(out, "X", "Y")
    assert out.metadata["undecided_pairs"] == [["X", "Y"]]
    assert out.metadata["never_observed_together"] == [["X", "Y"]]
    assert out.metadata["not_tested"] > 0
    # X - Z - Y with X and Y undecided is no collider, and Y could have separated X and Z.
    assert not any(out.structure.endpoints(v, "Z")[1] == ARROW for v in ("X", "Y"))
    assert out.metadata["possibly_separable"] == [["X", "Z"], ["Y", "Z"]]


def test_the_notice_names_a_pair_never_observed_together(caplog):
    with caplog.at_level(logging.WARNING, logger="andrey"):
        andrey.pc(_never_together())
        andrey.fci(_never_together())
    (notice,) = [r.getMessage() for r in caplog.records if r.name == "andrey.api._method"]
    assert "1 pair of columns (X and Y) is never observed together" in notice
    assert "1 pair (X and Y) is undecided." in notice


@pytest.mark.parametrize(
    ("fit", "data", "count"),
    [
        (andrey.pc, _never_together, 4),
        (andrey.fci, _never_together, 4),
        # Each blanket beside an undecided candidate counts one set per member (X's and Y's).
        (andrey.iamb, _never_together, 10),
        (andrey.pc, partial(_cohorts, 0), 20),
        (andrey.iamb, partial(_cohorts, 0), 14),
    ],
)
def test_the_conditioning_sets_not_tested_match_their_record(fit, data, count):
    assert fit(data(), missing="testwise").metadata["not_tested"] == count


def test_the_notice_names_few_undecided_pairs_and_gives_no_count_of_sets():
    notice = _not_tested_notice(
        {"not_tested": 2_621_200_269, "undecided_pairs": [["A", "B"], ["A", "C"]]}
    )
    assert notice.endswith("2 pairs (A and B, A and C) are undecided.")
    assert "2621200269" not in notice and "2,621,200,269" not in notice


def test_the_notice_counts_many_pairs_without_naming_them():
    pairs = [[f"A{i}", f"B{i}"] for i in range(1_200)]
    notice = _not_tested_notice(
        {"never_observed_together": pairs, "not_tested": 7, "undecided_pairs": pairs}
    )
    assert "1,200 pairs of columns are never observed together, listed" in notice
    assert notice.endswith("1,200 pairs are undecided.")
    assert "A0" not in notice


def test_the_notice_names_at_most_five_pairs():
    pairs = [[f"A{i}", f"B{i}"] for i in range(6)]
    named = _not_tested_notice({"undecided_pairs": pairs[:5]})
    assert named.endswith(
        "5 pairs (A0 and B0, A1 and B1, A2 and B2, A3 and B3, A4 and B4) are undecided."
    )
    counted = _not_tested_notice({"undecided_pairs": pairs})
    assert counted.endswith("6 pairs are undecided.")
    assert not any(a in counted for a, _ in pairs)


def test_the_notice_names_undecided_pairs_without_a_count_of_sets():
    assert _not_tested_notice({"undecided_pairs": [["A", "B"]]}).endswith(
        "1 pair (A and B) is undecided."
    )


def test_the_notice_says_no_pair_is_undecided_when_every_pair_was_decided():
    assert _not_tested_notice({"not_tested": 3}).endswith("No pair is undecided.")
    assert _not_tested_notice({"not_tested": 3, "undecided_pairs": []}).endswith(
        "No pair is undecided."
    )
    assert _not_tested_notice({}) == ""


_BOOTSTRAP_PROBE = """
import andrey
from test_not_tested import _never_together

if __name__ == "__main__":
    with andrey.config(n_jobs=2):
        andrey.bootstrap("pc", _never_together(), n_resamples=4, seed=0)
"""


def test_the_notice_of_a_bootstrap_on_worker_processes_names_a_pair_never_observed_together(
    tmp_path,
):
    probe = tmp_path / "probe.py"
    probe.write_text(_BOOTSTRAP_PROBE)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent), *sys.path])}
    run = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, check=True, env=env
    )
    assert run.stderr.count("the default handling, test-wise deletion") == 1
    assert "(X and Y) is never observed together" in run.stderr
    assert "1 pair (X and Y) is undecided." in run.stderr


def test_a_bootstrap_in_one_process_logs_one_notice_from_every_fit(caplog, monkeypatch):
    """In one process, as on workers, the fits log no notice of their own: bootstrap logs one, from
    the pairs of every fit."""
    from andrey import resample

    over = resample._not_tested_over
    later = {"undecided_pairs": [["P", "Q"]]}  # as if a later fit had left P and Q undecided
    monkeypatch.setattr(resample, "_not_tested_over", lambda fits: over(fits) | later)
    with andrey.config(n_jobs=1), caplog.at_level(logging.WARNING, logger="andrey"):
        andrey.bootstrap("pc", _never_together(), n_resamples=2, seed=0)
    notices = [r.getMessage() for r in caplog.records if "default handling" in r.getMessage()]
    assert len(notices) == 1
    assert "1 pair (P and Q) is undecided." in notices[0]


def test_em_refuses_a_pair_never_observed_together_by_name():
    with pytest.raises(ValueError, match=r"^drop column 'X' or column 'Y', .*share none"):
        andrey.ges(_never_together(), missing="em")


def test_lpcmci_conditions_no_test_on_a_link_never_observed_together(monkeypatch):
    from andrey.temporal.lpcmci import LPCMCISearch

    searches = []
    initialize = LPCMCISearch.initialize

    def kept(self, *args, **kwargs):
        searches.append(self)
        return initialize(self, *args, **kwargs)

    monkeypatch.setattr(LPCMCISearch, "initialize", kept)
    df = _series(0)
    andrey.lpcmci(df, tau_max=1)
    a, b = list(df.columns).index("A"), list(df.columns).index("B")
    (search,) = set(searches)
    # Only the tests of A and B lack rows: no test of another link conditions on A given B.
    assert search.skipped and all({i, j} == {a, b} for i, j, _ in search.skipped)


def _oracle(answers: dict) -> object:
    """A test of target 0 answering ``answers[(candidate, sorted set)]``, any other query not
    tested."""

    def test(x, y, condition_set=()):
        return answers.get((max(x, y), tuple(sorted(condition_set))), NOT_TESTED)

    return test


def test_fast_iamb_keeps_a_dependent_candidate_whose_last_test_was_not_tested():
    """1, 2, and 3 test dependent on 0 alone; the sparse-table rule admits 1 and 2, the shrink drops
    1 given 2, and 3 is not tested given 2. No test with rows separated 3, so it stays, as in IAMB,
    and the shrink keeps both."""
    answers = {(1, ()): 0.0, (2, ()): 0.001, (3, ()): 0.01, (1, (2,)): 0.5, (2, (3,)): 0.001}
    search = LocalSearch(_oracle(answers), [0, 1, 2, 3], 0.05, levels=np.full(4, 2), n=50)
    assert search.run("fast_iamb", 0) == ([2, 3], {1: (2,)})
    assert not search.undecided


def _sparse_chain(seed: int = 2, n: int = 55) -> pd.DataFrame:
    """Binary ``T -> v2 -> v1`` and ``T -> v3``, with ``v4``, three levels, a parent of ``v2``;
    ``v2`` is observed on the first 35 rows and ``v3`` on the others, so the two are never observed
    together."""
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 2, n)
    v4 = rng.choice(3, n, p=[0.45, 0.4, 0.15])
    v2 = t ^ (v4 == 2)
    v1 = v2 ^ (rng.random(n) < 0.1)
    v3 = t ^ (rng.random(n) < 0.15)
    df = pd.DataFrame({"T": t, "v1": v1, "v2": v2, "v3": v3, "v4": v4}).astype(float)
    df.loc[35:, "v2"] = np.nan
    df.loc[:34, "v3"] = np.nan
    return df


def test_fast_iamb_keeps_a_candidate_not_tested_when_the_sparse_table_rule_ends_the_search():
    """v3 tests dependent on T alone and is not tested given v2; v4, dependent given v2, fails the
    sparse-table rule, which ends the search. No test with rows separated v3, so it stays, as in
    IAMB; v4 is taken as independent, as the rule says."""
    df = _sparse_chain()
    fast = andrey.fast_iamb(df, target="T", indep_test="chisq").metadata
    assert andrey.iamb(df, target="T", indep_test="chisq").metadata["markov_blanket"] == [
        "v2",
        "v4",
        "v3",
    ]
    assert fast["markov_blanket"] == ["v2", "v3"]
    assert fast["undecided_pairs"] == []


def test_fast_iamb_keeps_a_candidate_not_tested_when_its_blanket_repeats():
    """Binary columns, 50 rows: the sparse-table rule admits two members. Round one adds 1 and 2
    and keeps 3 out; the shrinks then move the blanket from 2 to 4 to 5 and back to 2, which ends
    the search. 3, dependent on 0 alone, is not tested given any of them, and no test separated
    it: it stays."""
    answers = {
        **{(1, ()): 0.001, (2, ()): 0.002, (3, ()): 0.003, (4, ()): 0.01, (5, ()): 0.02},
        **{(1, (2,)): 0.5, (4, (2,)): 0.001, (5, (2,)): 0.5},
        **{(2, (4,)): 0.5, (5, (4,)): 0.001, (2, (5,)): 0.001, (4, (5,)): 0.5},
    }
    search = LocalSearch(_oracle(answers), range(6), 0.05, levels=np.full(6, 2), n=50)
    assert search.run("fast_iamb", 0) == ([2, 3], {1: (2,), 4: (5,), 5: (2,)})
    assert not search.undecided


def test_inter_iamb_keeps_a_candidate_not_tested_when_its_blanket_repeats():
    """2 joins; given 2, 3 joins and 1 is not tested; the shrink drops both, and 2 joins again,
    which ends the search. 1 tests dependent on 0 alone and no test separated it: it stays, as in
    IAMB."""
    answers = {(1, ()): 0.0067, (2, ()): 0.0002, (3, ()): 0.0575, (3, (2,)): 0.0458}
    answers[2, (3,)] = 0.2262
    search = LocalSearch(_oracle(answers), range(4), 0.05)
    assert search.run("inter_iamb", 0) == ([2, 1], {3: ()})
    assert search.run("iamb", 0)[0] == [2, 3, 1]


def test_inter_iamb_keeps_out_a_candidate_a_shrink_separated():
    """1, 2, and 3 join; the shrink drops 1 given 2 and 3, then 2 given 3, and 1 is not tested
    given 3 after: the shrink's test separated it, so it stays out, as in IAMB."""
    answers = {
        **{(1, ()): 0.001, (2, ()): 0.002, (3, ()): 0.003, (2, (1,)): 0.001, (1, (2,)): 0.001},
        **{(3, (1, 2)): 0.001, (1, (2, 3)): 0.5, (2, (3,)): 0.5},
    }
    for name in ("inter_iamb", "iamb"):
        search = LocalSearch(_oracle(answers), range(4), 0.05)
        assert search.run(name, 0) == ([3], {1: (2, 3), 2: (3,)})


def test_fast_iamb_leaves_out_a_candidate_no_test_with_rows_decided():
    """3 is never tested: it is undecided, and the member 1 is flagged beside it."""
    search = LocalSearch(_oracle({(1, ()): 0.0, (2, ()): 0.5}), [0, 1, 2, 3], 0.05)
    assert search.run("fast_iamb", 0) == ([1], {2: ()})
    assert search.undecided == {(0, 3)}


@pytest.mark.parametrize(
    "name", ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc", "get_pcd"]
)
def test_a_run_flags_each_member_beside_an_undecided_candidate(name):
    """B's run never tests A, which it never observes: A is undecided, and a set holding A, never
    tested, might separate B from its member S1, so the pair counts a set not tested."""
    x = _cohorts(0).to_numpy()  # columns S1, S2, S3, A, B
    search = local_search(x, alpha=0.05, indep_test="fisherz")
    with rows_tested() as rows:
        members, _ = search.run(name, 4)
    assert 0 in members and 3 not in members
    assert (0, 4) in rows.possibly_separable(np.ones((5, 5), dtype=bool))
    assert search.undecided == {(4, 3)}


@pytest.mark.parametrize(
    "name",
    ["iamb", "inter_iamb", "gs", "fast_iamb", "mmpc", "hiton_pc", "hiton_mb", "mmmb", "pcmb"],
)
def test_a_target_set_lists_undecided_pairs_and_flags_its_members_beside_them(name):
    out = getattr(andrey, name)(_cohorts(0), target="B")
    assert ["A", "B"] in out.metadata["undecided_pairs"]
    assert ["B", "S1"] in out.metadata["possibly_separable"]


#: The local facades, each on ``_pair_never_together()`` around ``x1``.
LOCAL_FACADES = {
    "mb_by_mb": lambda df: andrey.mb_by_mb(df, target="x1", blanket="iamb"),
    "mb_by_mb_gs": lambda df: andrey.mb_by_mb(df, target="x1", blanket="gs"),
    "cmb": lambda df: andrey.cmb(df, target="x1"),
    "mmb_by_mmb": lambda df: andrey.mmb_by_mmb(df, target="x1", blanket="iamb"),
    "ldecc": lambda df: andrey.ldecc(df, treatment="x1", outcome="x0"),
    "local_optimal_adjustment": lambda df: andrey.local_optimal_adjustment(
        df, variables=["x1", "x0"]
    ),
    "ldp": lambda df: andrey.ldp(df, treatment="x1", outcome="x0"),
    "ld3": lambda df: andrey.ld3(df, treatment="x1", outcome="x0"),
    "locpc_cde": lambda df: andrey.locpc_cde(df, treatment="x1", outcome="x0"),
    "y_structures": lambda df: andrey.y_structures(df),
}


def _pair_never_together() -> pd.DataFrame:
    """``x1 -> x0 <- x2`` and an independent ``x3``, with ``x1`` and ``x2`` never observed
    together."""
    rng = np.random.default_rng(0)
    n = 600
    x1, x2, noise = rng.normal(size=(3, n))
    df = pd.DataFrame({"x0": x1 + x2 + noise, "x1": x1, "x2": x2, "x3": rng.normal(size=n)})
    df.loc[: n // 2 - 1, "x1"] = np.nan
    df.loc[n // 2 :, "x2"] = np.nan
    return df


@pytest.mark.parametrize("name", list(LOCAL_FACADES))
def test_a_local_facade_lists_the_pair_no_test_decided(name):
    """Its notice names the undecided pair rather than saying no pair is undecided."""
    out = LOCAL_FACADES[name](_pair_never_together())
    assert out.metadata["not_tested"] > 0
    assert ["x1", "x2"] in out.metadata["undecided_pairs"]
    notice = _not_tested_notice(out.metadata)
    assert "x1 and x2" in notice and "No pair is undecided" not in notice


@pytest.mark.parametrize("k", [0, 2])
def test_load_and_ldecc_leave_the_effect_undecided_when_no_test_decided_a_pair(k):
    """A and B share fewer rows than a test needs, so B's blanket leaves A undecided and no test
    reads their relation: neither method reads the effect, though each did before, LOAD as no
    effect, identified by the empty set."""
    out = andrey.local_optimal_adjustment(_cohorts(k), variables=["B", "A"])
    assert out.metadata["undecided_pairs"] == [["A", "B"]]
    for record in out.metadata["effects"]:
        assert (record["nonzero"], record["identifiable"], record["adjustment_sets"]) == (
            None,
            False,
            [],
        )
    out = andrey.ldecc(_cohorts(k), treatment="A", outcome="B")
    assert out.metadata["undecided_pairs"] == [["A", "B"]]
    (record,) = out.metadata["effects"]
    assert (record["nonzero"], record["identifiable"], record["adjustment_sets"]) == (
        None,
        None,
        [],
    )


def test_load_and_ldecc_leave_the_effect_undecided_when_total_conditioning_left_one_undecided():
    """``a -> b -> c`` and ``v``, categorical, ``v`` one category on the rows that observe ``c``:
    every chi-square test of ``v`` given all the rest has no degrees of freedom, so total
    conditioning leaves ``v`` undecided, and neither method reads the effect of ``a`` on ``b``."""
    rng = np.random.default_rng(0)
    n = 600
    a = rng.integers(0, 2, n)
    b = a ^ (rng.random(n) < 0.1)
    c = b ^ (rng.random(n) < 0.1)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "v": rng.integers(0, 2, n)}).astype(float)
    df.loc[: n // 2 - 1, "c"] = np.nan
    df.loc[n // 2 :, "v"] = 0.0
    fit = partial(andrey.ldecc, df, treatment="a", outcome="b", indep_test="chisq")
    out = fit(blanket="total_conditioning")
    assert ["a", "v"] in out.metadata["undecided_pairs"]
    (record,) = out.metadata["effects"]
    assert (record["nonzero"], record["identifiable"], record["adjustment_sets"]) == (
        None,
        None,
        [],
    )
    out = andrey.local_optimal_adjustment(
        df, variables=["a", "b"], blanket="total_conditioning", indep_test="chisq"
    )
    assert ["a", "v"] in out.metadata["undecided_pairs"]
    for record in out.metadata["effects"]:
        assert (record["nonzero"], record["identifiable"], record["adjustment_sets"]) == (
            None,
            False,
            [],
        )


def _five(n: int = 1200) -> pd.DataFrame:
    """``x0 -> x1 -> x3 <- x2 <- x0`` and an independent ``x4``."""
    rng = np.random.default_rng(0)
    x0, e1, e2, e3, x4 = rng.normal(size=(5, n))
    x1, x2 = x0 + e1, x0 + e2
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x1 + x2 + e3, "x4": x4})


def _own_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Each column observed on its own block of rows: no pair is ever observed together."""
    x = df.to_numpy().copy()
    block = np.arange(len(x)) // (len(x) // x.shape[1])
    x[block[:, None] != np.arange(x.shape[1])] = np.nan
    return pd.DataFrame(x, columns=df.columns)


def _alone(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """``column`` observed only on the first quarter of the rows, where every other column is
    missing."""
    out = df.copy()
    quarter = len(df) // 4
    out.loc[quarter:, column] = np.nan
    out.loc[: quarter - 1, [c for c in df.columns if c != column]] = np.nan
    return out


#: Data on which a pair the direct effect of x1 on x3 rests on has no test with rows.
NEVER_TESTED = {
    "no pair observed together": _own_rows,
    "treatment alone": partial(_alone, column="x1"),
    "outcome alone": partial(_alone, column="x3"),
}


@pytest.mark.parametrize("case", list(NEVER_TESTED))
@pytest.mark.parametrize("name", ["ld3", "locpc_cde", "ldp"])
def test_an_effect_resting_on_a_pair_no_test_decided_is_undecided(name, case):
    out = getattr(andrey, name)(NEVER_TESTED[case](_five()), treatment="x1", outcome="x3")
    (record,) = out.metadata["effects"]
    assert (record["nonzero"], record["identifiable"], record["adjustment_sets"]) == (
        None,
        None,
        [],
    )
    assert out.metadata["undecided_pairs"]


def test_ld3_draws_no_edge_from_a_pair_no_test_decided():
    """x3 is never observed with another column: no test confirms a parent."""
    out = andrey.ld3(_alone(_five(), "x3"), treatment="x1", outcome="x3")
    assert not any(_adjacent(out, c, "x3") for c in ("x0", "x1", "x2", "x4"))


def test_locpc_cde_leaves_a_pair_no_test_decided_out_of_its_graph():
    """x1 is never observed with another column: its edge to x3 is undecided, not adjacent."""
    out = andrey.locpc_cde(_alone(_five(), "x1"), treatment="x1", outcome="x3")
    assert ["x1", "x3"] in out.metadata["undecided_pairs"]
    assert not _adjacent(out, "x1", "x3")


def test_ldp_and_ld3_read_no_dependence_from_a_query_not_tested():
    """The treatment 0 and the outcome 1 test dependent and every query of 2 is not tested: 2 is
    no parent and takes no label, and both answers are undecided."""
    from andrey.constraint import partition

    def test(x, y, condition_set=()):
        return 0.0 if {x, y} == {0, 1} else NOT_TESTED

    ld3 = partition.ld3(LocalSearch(test, [0, 1, 2], 0.05), 0, 1, [2])
    assert (ld3.parents, ld3.partition["unlabeled"], ld3.direct, ld3.undecided) == (
        [],
        [2],
        True,
        True,
    )
    ldp = partition.ldp(LocalSearch(test, [0, 1, 2], 0.05), 0, 1, [2])
    assert (ldp.partition["unlabeled"], ldp.valid, ldp.undecided) == ([2], False, True)


def test_ldecc_reads_no_dependence_from_a_query_not_tested():
    """UCChildren marks the neighbor 1 a child when the spouse 2, separated from 0 by no set, is
    dependent on 1; a query of 1 and 2 not tested marks nothing."""
    from andrey.constraint.ldecc import Tests, uc_children

    def test(x, y, condition_set=()):
        return 0.5 if {x, y} == {0, 2} and not condition_set else NOT_TESTED

    search = LocalSearch(test, [0, 1, 2], 0.05)
    assert uc_children(Tests(search), 0, [1, 2], [1]) == set()


def test_load_reads_no_ancestry_from_a_query_not_tested():
    """Whether 0 is an explicit ancestor of 1 is a test of the two given 0's parents; a query not
    tested decides neither."""
    from andrey.constraint.local_optimal_adjustment import NotTested, explicit_ancestor
    from andrey.constraint.sequential import Network

    search = LocalSearch(_oracle({}), [0, 1], 0.05)
    with pytest.raises(NotTested):
        explicit_ancestor(search, 0, 1, Network(2))
