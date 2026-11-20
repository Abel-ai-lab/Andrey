"""Recovery and behavior checks for LGES.

Fixed cases have a CPDAG that can be checked by hand, and the paper's Example 3 fixes which edge
additions SafeInsert and ConservativeInsert keep on one class. Ground-truth cases are
linear-Gaussian SCMs on sparse Erdos-Renyi graphs from ``andrey.data``, where the BIC score is
consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
import andrey.data as data
from andrey.api._adapt import dag_from_adjacency
from andrey.core import TAIL
from andrey.core.score import BICScore
from andrey.metrics import shd, to_cpdag
from andrey.search.lges import extension, insert_moves
from tests.recovery.helpers import datagen
from tests.recovery.helpers.harness import assert_recorded_output

pytestmark = pytest.mark.filterwarnings("ignore::andrey.ExperimentalWarning")

INSERTIONS = ("safe", "conservative")
SCHEDULES = ("lges", "lges0_released")


def _er_sample(d: int, n: int, seed: int) -> data.CausalDataset:
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(d=d, avg_degree=2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    return scm.sample(n=n, seed=seed, scale="raw")


# ---- fixed cases ---------------------------------------------------------------------------------


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("insertion", INSERTIONS)
def test_independent_columns_give_no_edges(insertion, schedule):
    x = np.random.default_rng(0).standard_normal((1000, 4))
    out = andrey.lges(x, insertion=insertion, schedule=schedule)
    assert out.structure.oriented_edges() == []


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("insertion", INSERTIONS)
@pytest.mark.parametrize("replicate", [0, 1])
def test_recovers_the_cpdag_of_gauss_7v(replicate, insertion, schedule):
    """The collider 0 -> 2 <- 1 orients 2 -> 3 -> 4; nothing orients 5 -- 6."""
    weights = np.asarray(datagen.CASES["gauss_7v"]["weights"])
    order = datagen.column_order("gauss_7v", replicate=replicate)
    truth = to_cpdag(dag_from_adjacency((weights.T != 0)[np.ix_(order, order)]))
    x = datagen.generate("gauss_7v", replicate=replicate)
    assert shd(andrey.lges(x, insertion=insertion, schedule=schedule).structure, truth) == 0


def _example_3() -> tuple[np.ndarray, BICScore]:
    """The paper's Example 2: X1 -> Z <- X2, Z -> Y; columns (X1, X2, Y, Z); the class Z -- Y."""
    rng = np.random.default_rng(0)
    x1, x2 = rng.standard_normal(2000), rng.standard_normal(2000)
    z = x1 + x2 + rng.standard_normal(2000)
    y = z + rng.standard_normal(2000)
    adj = np.zeros((4, 4), dtype=np.int8)
    adj[2, 3] = adj[3, 2] = TAIL
    return adj, BICScore(np.column_stack([x1, x2, y, z]), ebic_gamma=0.0)


@pytest.mark.parametrize(
    ("ranks", "parents_of_y", "safe_keeps"),
    [((1, 2, 0, 3), [3], False), ((1, 2, 3, 0), [], True)],
    ids=["G1 Z->Y", "G2 Y->Z"],
)
def test_example_3_keeps_x1_to_y_only_where_the_paper_does(ranks, parents_of_y, safe_keeps):
    """SafeInsert drops X1 -> Y on G1, where X1 is independent of Y given Z, and keeps it on G2;
    ConservativeInsert drops it on both, as Insert(X1, Y, {Z}) lowers the score."""
    adj, score = _example_3()
    dag = extension(adj, ranks)
    assert np.flatnonzero(dag[2] == andrey.core.ARROW).tolist() == parents_of_y
    safe = {(x, y) for _, x, y, _ in insert_moves(adj, score, 4, ranks, conservative=False)}
    conservative = {(x, y) for _, x, y, _ in insert_moves(adj, score, 4, ranks, conservative=True)}
    assert ((0, 2) in safe) is safe_keeps
    assert (0, 2) not in conservative
    assert {(0, 3), (1, 3)} <= safe & conservative  # X1 -> Z and X2 -> Z stay


def _random_case(p: int, rows: int, seed: int) -> np.ndarray:
    """A linear-Gaussian SCM on ``p`` variables in column order, each forward edge kept with
    probability 1/2 with a weight of size 0.5 to 1.5 and a random sign, and ``rows`` rows."""
    rng = np.random.default_rng(seed)
    w = np.triu(rng.uniform(0.5, 1.5, (p, p)) * rng.choice([-1, 1], (p, p)), 1)
    w *= rng.random((p, p)) < 0.5
    x = np.zeros((rows, p))
    for j in range(p):
        x[:, j] = x @ w[:, j] + rng.standard_normal(rows)
    return x


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("tie_break", ["position", "rank"])
@pytest.mark.parametrize("insertion", INSERTIONS)
def test_turning_phase_reaches_the_authors_cpdag(insertion, tie_break, schedule, monkeypatch):
    """Four variables, edges 0 -> 2 <- 1 and 2 -> 3, 200 rows. The authors' code applies one
    reversal and returns 0 -> 1 <- 2, 1 <- 3, 0 -- 2, 2 -- 3, and so do its step functions run in
    Algorithm 1's order; without the turning phase the search stops at another class."""
    x = _random_case(4, 200, 15)
    authors = [(0, 1, "directed"), (0, 2, "undirected"), (2, 1, "directed"), (3, 1, "directed")]
    authors.append((2, 3, "undirected"))
    with andrey.config(tie_break=tie_break):
        full = andrey.lges(x, insertion=insertion, schedule=schedule, ebic_gamma=0.0).structure
        monkeypatch.setattr("andrey.search.lges._turning_best", lambda *args: None)
        unturned = andrey.lges(x, insertion=insertion, schedule=schedule, ebic_gamma=0.0).structure
    assert sorted(full.oriented_edges()) == sorted(authors)
    assert shd(full, unturned) > 0


_SCHEDULE_CASE = [
    (0, 1, "undirected"),
    (0, 5, "directed"),
    (1, 2, "undirected"),
    (1, 5, "directed"),
    (2, 4, "directed"),
    (2, 5, "directed"),
    (3, 4, "directed"),
    (4, 5, "directed"),
]


@pytest.mark.parametrize("tie_break", ["position", "rank"])
@pytest.mark.parametrize("insertion", INSERTIONS)
@pytest.mark.parametrize(
    ("schedule", "extra"), [("lges", [(1, 4, "directed")]), ("lges0_released", [])]
)
def test_each_schedule_returns_its_references_cpdag(schedule, extra, insertion, tie_break):
    """Six variables, 300 rows: the authors' code returns the eight edges of ``_SCHEDULE_CASE``
    after four deletions; its step functions run in Algorithm 1's order also keep ``1 -> 4``."""
    x = _random_case(6, 300, 45)
    with andrey.config(tie_break=tie_break):
        out = andrey.lges(x, insertion=insertion, schedule=schedule, ebic_gamma=0.0)
    assert sorted(out.structure.oriented_edges()) == sorted(_SCHEDULE_CASE + extra)


def test_matches_the_recorded_output_on_gauss_5v():
    assert_recorded_output(andrey.lges(datagen.generate("gauss_5v")), "LGES", case="gauss_5v")


def test_rejects_an_unknown_insertion():
    with pytest.raises(ValueError, match="insertion"):
        andrey.lges(np.random.default_rng(0).standard_normal((100, 3)), insertion="greedy")


def test_rejects_an_unknown_schedule():
    with pytest.raises(ValueError, match="schedule"):
        andrey.lges(np.random.default_rng(0).standard_normal((100, 3)), schedule="lges+")


# ---- ground truth --------------------------------------------------------------------------------


@pytest.mark.parametrize(("schedule", "bound"), [("lges", 10), ("lges0_released", 35)])
@pytest.mark.parametrize("insertion", INSERTIONS)
def test_recovers_sparse_er_graphs(insertion, schedule, bound):
    """20 variables, mean degree 2, 5,000 rows, the standard BIC: the first seed is recovered
    exactly, and the errors over all four stay far below the empty graph's (SHD per seed 0, 1, 3,
    1 for LGES; 0, 23, 7, 1 and 0, 23, 3, 1 for LGES-0 under the two insertions)."""
    errors, empty = [], 0
    for seed in range(4):
        sample = _er_sample(20, 5000, seed)
        truth = to_cpdag(sample.graph)
        out = andrey.lges(
            np.asarray(sample.data), insertion=insertion, schedule=schedule, ebic_gamma=0.0
        )
        errors.append(shd(out.structure, truth))
        empty += int(np.count_nonzero(truth.to_numpy())) // 2
    assert errors[0] == 0
    assert sum(errors) <= bound < empty


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("insertion", INSERTIONS)
def test_refits_give_the_same_cpdag_and_score(insertion, schedule):
    x = np.asarray(_er_sample(10, 1000, 1).data)
    first = andrey.lges(x, insertion=insertion, schedule=schedule)
    second = andrey.lges(x, insertion=insertion, schedule=schedule)
    np.testing.assert_array_equal(first.structure.to_numpy(), second.structure.to_numpy())
    assert first.metadata["score"] == second.metadata["score"]
