"""The local LiNG lane's judge: paired verdicts, coverage, the floor, and the accuracy-cost frontier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import local_lingam_summary as S  # noqa: E402


def _record(solution, seed, shd, dist, wall, status="ok"):
    scores = {"shd": shd, "weight_dist": dist} if status == "ok" else None
    return {
        "cell": "acyclic-d3",
        "n": 2000,
        "seed": seed,
        "solution": solution,
        "target": 0,
        "status": status,
        "wall_s": wall,
        "scores": scores,
        "m": 5,
        "fallback": False,
    }


def _run(tmp_path, rows):
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return S.load([str(run)])


def test_a_cell_is_judged_on_the_seeds_both_finished_and_floored_by_the_empty_answer(tmp_path):
    rows = []
    for seed in range(20):
        rows.append(_record("empty", seed, 10, 3.0, None))
        rows.append(_record("andrey.local_isa_ling", seed, 2 + seed % 2, 0.5, 30.0))
        status = "not_run" if seed == 0 else "ok"
        rows.append(_record("local_isa_ling.authors", seed, 4, 0.5, 20.0, status))
        rows.append(_record("andrey.direct_lingam", seed, 1, 0.2, 60.0))
        rows.append(_record("andrey.ica_lingam", seed, 6, 1.0, 5.0))
    runs = _run(tmp_path, rows)
    [row] = [
        r for r in S.summarize(runs, range(20), range(20, 40))[0] if r["blanket"] == "estimated"
    ]
    assert row["first"]["paired"] == 19
    assert row["coverage"]["reference_not_run"] == 1 and row["coverage"]["reference_short"] == 1
    assert row["shd_result"] == "better" and row["weight_dist_result"] == "tie"
    assert row["floor"]["shd"]["andrey_minus_empty"]["pass"]
    [summary] = S.summary_rows([row], "shd", "estimated", "20-39")
    assert summary["andrey_shd_mean"] < summary["comparator_shd_mean"] and not summary["above_zero"]
    assert summary["floor"]["applies"] and summary["floor"]["pass"]


def test_the_frontier_keeps_every_solution_no_other_beats_on_accuracy_and_cost(tmp_path):
    rows = []
    for seed in range(12):
        rows.append(_record("empty", seed, 10, 3.0, None))
        rows.append(_record("andrey.local_isa_ling", seed, 3, 0.5, 30.0))
        rows.append(_record("andrey.direct_lingam", seed, 1, 0.2, 60.0))
        rows.append(_record("andrey.ica_lingam", seed, 6, 1.0, 5.0))
    front = S.frontier(_run(tmp_path, rows), "acyclic-d3", 2000, range(12))
    on = {k for k, v in front["points"].items() if v["pareto_shd"]}
    # Each point trades accuracy against cost; none is beaten on both at once.
    assert on == {"andrey.local_isa_ling", "andrey.direct_lingam", "andrey.ica_lingam", "empty"}


def test_sachs_keeps_each_target_of_its_one_dataset(tmp_path):
    rows = [
        {**_record("andrey.local_isa_ling", 0, shd, None, 1.0), "cell": "sachs", "target": t}
        for t, shd in [(3, 2), (4, 2), (6, 1)]
    ]
    found = S.sachs(_run(tmp_path, rows))
    assert [(r["target"], r["shd"]) for r in found] == [(3, 2), (4, 2), (6, 1)]
