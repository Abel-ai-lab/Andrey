"""The local LiNG lane: the paper's samplers, its local-graph scorer, and the driver's rules.

The driver tests replace ``run_task`` with an in-process fit, so they need no harness environment;
the test of the authors' code skips unless its environment is built (``start_local_ling.sh``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import local_lingam_cpu  # noqa: E402

from andrey_bench import local_ling_models as M  # noqa: E402
from andrey_bench.adapters import dai_local_isa_ling as dai  # noqa: E402


def _edges(d: int, *edges: tuple[int, int]) -> np.ndarray:
    w = np.zeros((d, d))
    for i, j in edges:
        w[i, j] = 1.0
    return w


@pytest.mark.parametrize(
    ("estimated", "expected"),
    [
        ([], 2),  # the empty answer misses both true adjacencies
        ([(0, 1), (1, 2)], 0),  # half of the 2-cycle still draws its adjacency
        ([(1, 0), (2, 1)], 1),  # 2 -> 1 is reversed
        ([(2, 1)], 2),  # one reversed edge and one missing adjacency
        ([(0, 1), (1, 0), (1, 2), (3, 2)], 1),  # one extra parent of the child
        ([(0, 1), (1, 0), (1, 2), (3, 4)], 0),  # an edge outside the local structure is ignored
        ([(1, 3)], 3),  # a false child, and both true adjacencies missing
    ],
)
def test_the_shd_counts_as_the_authors_count_dcg_accuracy(estimated, expected):
    # Truth: 0 <-> 1 (a 2-cycle), 1 -> 2, 4 -> 3; target 1. The values are the authors'
    # count_dcg_accuracy on the same graphs.
    true = _edges(5, (0, 1), (1, 0), (1, 2), (4, 3))
    assert M.local_scores(_edges(5, *estimated), true, 1)["shd"] == expected


def test_the_weight_distance_is_the_norm_of_the_local_difference():
    true = np.zeros((4, 4))
    true[0, 1], true[1, 2], true[3, 2] = 0.5, -0.8, 0.6
    est = np.zeros((4, 4))
    est[0, 1], est[1, 2] = 0.4, -0.8
    s = M.local_scores(est, true, 1)
    assert s["weight_dist"] == pytest.approx(np.hypot(0.1, 0.6))
    assert M.local_scores(np.zeros((4, 4)), true, 1)["weight_dist"] == pytest.approx(
        np.linalg.norm([0.5, 0.8, 0.6])
    )


@pytest.mark.parametrize("seed", range(4))
def test_the_cyclic_sampler_draws_disjoint_stable_cycles_through_the_target(seed):
    draw = M.sample("cyclic", 300, seed)
    adj = draw.weights != 0
    assert (adj.sum(0) + adj.sum(1)).max() <= M.CYCLIC["max_degree"]
    assert np.max(np.abs(np.linalg.eigvals(draw.weights))) < 1
    cycles = M._cycles(adj)
    assert cycles and any(draw.target in c for c in cycles)
    magnitudes = np.abs(draw.weights[adj])
    assert magnitudes.min() >= 0.5 and magnitudes.max() <= 0.9
    assert draw.blanket == M.true_blanket(draw.weights, draw.target)


@pytest.mark.parametrize("degree", [3, 5])
def test_the_acyclic_sampler_keeps_the_model_across_sizes_and_picks_a_large_blanket(degree):
    small, large = M.sample("acyclic", 100, 0, degree), M.sample("acyclic", 400, 0, degree)
    assert np.array_equal(small.weights, large.weights) and small.target == large.target
    assert int((small.weights != 0).sum()) == degree * 50 // 2
    assert M._cycles(small.weights != 0) == []
    sizes = [len(M.true_blanket(small.weights, t)) for t in range(50)]
    goal = M.BLANKET_GOAL[degree]
    assert abs(len(small.blanket) - goal) == min(abs(s - goal) for s in sizes)


def test_the_data_follow_the_shuffled_weights():
    draw = M.sample("acyclic", 20000, 1, 3)
    x, w = draw.data, draw.weights
    child = int(np.flatnonzero((w != 0).any(axis=0))[0])
    parents = np.flatnonzero(w[:, child])
    coef = np.linalg.lstsq(x[:, parents], x[:, child], rcond=None)[0]
    assert coef == pytest.approx(w[parents, child], abs=0.05)


def test_an_answer_round_trips_and_a_declined_fit_is_marked():
    w = np.arange(25, dtype=float).reshape(5, 5)
    a = M.unpack_answer(M.pack_answer(w, [1, 3], m=3, models=2, fallback=True))
    assert np.array_equal(a["weights"], w) and a["blanket"] == [1, 3]
    assert (a["code"], a["m"], a["models"], a["fallback"]) == (M.ANSWERED, 3, 2, True)
    assert M.unpack_answer(M.pack_answer(w, [], code=M.NOT_RUN, m=12))["code"] == M.NOT_RUN


class TruthLocal:
    """A local solution that answers with the true local structure, or declines when told."""

    name = "truth.local"
    package = "test"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    algorithm = "truth"
    output_type = "digraph"
    local = True
    oracle = True

    def params(self):
        return {}

    def fit(self, data, params):  # replaced by the fake run_task
        raise AssertionError


class Global(TruthLocal):
    name = "truth.global"
    local = False
    oracle = False


@pytest.fixture
def lane(monkeypatch):
    truths = {}

    def fit_here(adapter, data, params, **_):
        key = data.tobytes()[:64]
        w = truths[key]
        if adapter.local and len(params["blanket"]) > 9:
            return [{"status": "ok", "adj": M.pack_answer(w * 0, [], code=M.NOT_RUN, m=12)}]
        blanket = params.get("blanket")
        return [{"status": "ok", "wall_s": 0.1, "adj": M.pack_answer(w, blanket, m=3)}]

    real = local_lingam_cpu.datasets

    def datasets(cell, rows, seeds):
        for item in real(cell, rows, seeds):
            truths[item[2].tobytes()[:64]] = item[3]
            yield item

    monkeypatch.setattr(local_lingam_cpu, "datasets", datasets)
    monkeypatch.setattr(local_lingam_cpu, "run_task", fit_here)
    monkeypatch.setattr(local_lingam_cpu, "check_interpreters", lambda field: {})
    monkeypatch.setattr(local_lingam_cpu, "SOLUTIONS", [TruthLocal(), Global()])
    return local_lingam_cpu


def test_the_driver_scores_every_arm_and_skips_global_fits_on_cyclic_data(lane, tmp_path):
    out = tmp_path / "run"
    argv = ["--out", str(out), "--cells", "cyclic", "acyclic-d3", "--rows", "200", "--seeds", "2"]
    assert lane.main(argv) == 0
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    by = {(r["cell"], r["solution"]) for r in records}
    assert ("cyclic", "truth.global") not in by and ("acyclic-d3", "truth.global") in by
    for r in records:
        if r["solution"] == "empty":
            assert r["scores"]["est_edges"] == 0 and r["scores"]["shd"] > 0
        elif r["status"] == "ok":
            assert r["scores"]["shd"] == 0 and r["scores"]["weight_dist"] == 0
        assert r["status"] in ("ok", "not_run")
        assert r["status"] == "ok" or r["scores"] is None
    acyclic = [r for r in records if r["cell"] == "acyclic-d3" and r["solution"] == "truth.local"]
    assert all(r["status"] == "not_run" and r["m"] == 12 for r in acyclic)  # 14-member blankets
    assert all(r["params"]["acyclic"] for r in acyclic)
    before = [r["scores"] for r in records]
    lane.main(["--out", str(out), "--rescore"])
    after = [
        json.loads(line)["scores"] for line in (out / "records.jsonl").read_text().splitlines()
    ]
    assert after == before


def test_the_adapter_names_its_environment_and_the_script_that_builds_it(monkeypatch):
    monkeypatch.delenv(dai.CODE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="start_local_ling.sh"):
        dai._code_dir()
    script = Path(__file__).resolve().parents[1] / "start_local_ling.sh"
    assert dai.COMMIT in script.read_text()


@pytest.mark.skipif(
    not (os.environ.get(dai.CODE_ENV) and os.environ.get(dai.DaiLocalISALiNG.python_env)),
    reason="the authors' code is not built (start_local_ling.sh)",
)
def test_the_driver_runs_the_authors_code_in_its_own_environment(tmp_path):
    out = tmp_path / "run"
    flags = ["--out", str(out), "--cells", "cyclic", "--rows", "1000", "--seeds", "1"]
    assert local_lingam_cpu.main([*flags, "--solutions", "local_isa_ling.authors.oracle"]) == 0
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    [fit] = [r for r in records if r["solution"] != "empty"]
    assert fit["status"] == "ok" and fit["package_version"] == dai.COMMIT
    assert (
        fit["scores"]["shd"]
        <= next(r for r in records if r["solution"] == "empty")["scores"]["shd"]
    )
