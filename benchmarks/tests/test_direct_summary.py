"""A target-centered method judged per cell: the paired comparisons with global PC, the replication
that a worse cell needs, the floor, and the paper's counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import direct_summary  # noqa: E402

CELL = ("alarm", 1000, 0.02)


def _counts(extra: int = 0, missing: int = 0, under: int = 0) -> dict:
    return {
        "extra": extra,
        "missing": missing,
        "reversed": under,
        "wrongdir": 0,
        "over": 0,
        "under": under,
        "total": extra + missing + under,
    }


def _record(solution: str, seed: int, target: int, counts: dict | None, **extra) -> dict:
    alpha = None if solution == direct_summary.EMPTY else CELL[2]
    return {
        "cell": "net:alarm:1000",
        "solution": solution,
        "seed": seed,
        "target": target,
        "scope": "target" if solution == "andrey.cmb" else "graph",
        "params": {} if alpha is None else {"alpha": alpha},
        "status": "ok" if counts is not None else "timeout",
        "scores": None if counts is None else {"cpdag": counts},
        "wall_s": 1.0,
        "n_tests": 10,
    } | extra


def _run(tmp_path: Path, missing_of: dict[str, list[int]], seeds: range = range(40)) -> Path:
    """Two targets per dataset; ``missing_of[solution][seed // 20]`` edges missed at each."""
    lines = []
    for seed in seeds:
        for target in (0, 1):
            for solution, missing in missing_of.items():
                lines.append(_record(solution, seed, target, _counts(missing=missing[seed // 20])))
            lines.append(_record(direct_summary.EMPTY, seed, target, _counts(missing=3)))
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return run


def _row(run: Path) -> dict:
    per = direct_summary.dataset_means(direct_summary.load([str(run)]))
    return direct_summary.cell_row(per, "cmb", CELL, list(range(20)), list(range(20, 40)))


def test_fewer_errors_than_pc_on_every_dataset_is_better_and_passes_the_floor(tmp_path):
    run = _run(
        tmp_path, {"andrey.cmb": [1, 1], "andrey.pc.gsq": [2, 2], "bnlearn.pc-stable.graph": [1, 1]}
    )
    row = _row(run)
    assert row["against"]["andrey.pc.gsq"]["first"]["mean_D"] == pytest.approx(-1.0)
    assert row["against"]["andrey.pc.gsq"]["verdict"] == "better"
    assert row["against"]["bnlearn.pc-stable.graph"]["verdict"] == "tie"
    assert row["floor"]["mean_D"] == pytest.approx(-2.0) and row["floor"]["pass"]
    assert row["solutions"]["andrey.cmb"]["total"] == pytest.approx(1.0)
    assert row["paper"]["total"] == 1.81


def test_a_worse_cell_counts_only_when_the_fresh_seeds_repeat_it(tmp_path):
    once = _row(
        _run(
            tmp_path / "a",
            {"andrey.cmb": [2, 1], "andrey.pc.gsq": [1, 1], "bnlearn.pc-stable.graph": [1, 1]},
        )
    )
    assert once["against"]["andrey.pc.gsq"]["first"]["verdict"] == "worse"
    assert once["against"]["andrey.pc.gsq"]["fresh"]["verdict"] == "tie"
    assert once["against"]["andrey.pc.gsq"]["verdict"] == "tie"
    twice = _row(
        _run(
            tmp_path / "b",
            {"andrey.cmb": [2, 2], "andrey.pc.gsq": [1, 1], "bnlearn.pc-stable.graph": [1, 1]},
        )
    )
    assert twice["against"]["andrey.pc.gsq"]["verdict"] == "worse"
    unrun = _row(
        _run(
            tmp_path / "c",
            {"andrey.cmb": [2, 2], "andrey.pc.gsq": [1, 1], "bnlearn.pc-stable.graph": [1, 1]},
            seeds=range(20),
        )
    )
    assert unrun["against"]["andrey.pc.gsq"]["verdict"] == "worse, not replicated"


def test_a_dataset_with_a_timeout_is_left_out_of_its_pairs_never_scored_empty(tmp_path):
    run = _run(
        tmp_path, {"andrey.cmb": [1, 1], "andrey.pc.gsq": [2, 2], "bnlearn.pc-stable.graph": [1, 1]}
    )
    records = (run / "records.jsonl").read_text()
    timeout = _record("andrey.cmb", 0, 2, None)
    (run / "records.jsonl").write_text(records + json.dumps(timeout) + "\n")
    row = _row(run)
    assert row["solutions"]["andrey.cmb"]["datasets"] == 19
    assert row["against"]["andrey.pc.gsq"]["first"]["paired"] == 19


def test_the_lenient_reading_leaves_out_the_edges_left_undirected(tmp_path):
    lines = []
    for seed in range(20):
        lines.append(_record("andrey.cmb", seed, 0, _counts(under=1)))
        lines.append(_record("andrey.pc.gsq", seed, 0, _counts(missing=1)))
        lines.append(_record(direct_summary.EMPTY, seed, 0, _counts(missing=2)))
    run = tmp_path / "run"
    run.mkdir()
    (run / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    row = _row(run)
    pc = row["against"]["andrey.pc.gsq"]
    assert pc["first"]["verdict"] == "tie"
    assert pc["lenient"]["mean_D"] == pytest.approx(-1.0)
    assert row["against"]["bnlearn.pc-stable.graph"]["first"]["verdict"] == "too few pairs"


MB_CELL = ("alarm", 500, 0.01)
TARGET_CENTERED = ("andrey.mb_by_mb", "andrey.cmb")


def _dag(miss: int = 0) -> dict:
    return {"undirected": 0, "miss": miss, "extra": 0, "reverse": 0, "shd": miss}


def _mb_run(
    tmp_path: Path,
    missing_of: dict[str, list[int]],
    tests_of: dict[str, tuple[int, int]],
    seeds: range = range(40),
    network: str = "alarm",
) -> Path:
    """Two targets per dataset, scored against the DAG; ``missing_of[solution][seed // 20]`` edges
    missed at each, ``tests_of[solution]`` the tests of the walk at each target, or of the global
    fit, and a tenth of them the seconds. The empty answer misses 3 edges at each target."""
    missing_of = missing_of | {direct_summary.EMPTY: [3, 3]}
    tests_of = tests_of | {direct_summary.EMPTY: (0, 0)}
    lines = []
    for seed in seeds:
        for target in (0, 1):
            for solution, missing in missing_of.items():
                tests = tests_of[solution][target]
                alpha = {} if solution == direct_summary.EMPTY else {"alpha": MB_CELL[2]}
                scope = "target" if solution in TARGET_CENTERED else "graph"
                lines.append(
                    {
                        "cell": f"net:{network}:500",
                        "solution": solution,
                        "seed": seed,
                        "target": target,
                        "scope": scope,
                        "params": alpha,
                        "status": "ok",
                        "scores": {"dag": _dag(miss=missing[seed // 20])},
                        "wall_s": tests / 10,
                        "n_tests": tests,
                    }
                )
    run = tmp_path / "run"
    run.mkdir(parents=True)
    (run / "records.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return run


def _mb_row(run: Path, first: range = range(20), fresh: range = range(20, 40)) -> dict:
    per = direct_summary.dataset_means(direct_summary.load([str(run)]), "dag")
    return direct_summary.cell_row(per, "mb_by_mb", MB_CELL, list(first), list(fresh))


def test_mb_by_mb_is_judged_on_the_dag_counts_beside_its_paper(tmp_path):
    run = _mb_run(
        tmp_path,
        {"andrey.mb_by_mb": [1, 1], "andrey.pc.gsq": [2, 2]},
        {"andrey.mb_by_mb": (10, 10), "andrey.pc.gsq": (50, 50)},
    )
    row = _mb_row(run)
    mine = row["solutions"]["andrey.mb_by_mb"]
    assert (mine["miss"], mine["shd"], mine["undirected"]) == (1.0, 1.0, 0.0)
    assert row["against"]["andrey.pc.gsq"]["first"]["mean_D"] == pytest.approx(-1.0)
    assert row["against"]["andrey.pc.gsq"]["verdict"] == "better"
    assert "lenient" not in row["against"]["andrey.pc.gsq"]
    assert row["floor"]["mean_D"] == pytest.approx(-2.0) and row["floor"]["pass"]
    assert row["paper"]["shd"] == 1.92 and row["paper"]["undirected"] == 0.59
    # bnlearn and CMB did not run in this cell, so the row leaves them out.
    assert set(row["against"]) == {"andrey.pc.gsq"}


def test_the_cost_per_answer_averages_a_walk_and_over_every_target_sums_it(tmp_path):
    run = _mb_run(
        tmp_path,
        {"andrey.mb_by_mb": [1, 1], "andrey.pc.gsq": [1, 1], "andrey.cmb": [1, 1]},
        {"andrey.mb_by_mb": (10, 30), "andrey.pc.gsq": (15, 15), "andrey.cmb": (40, 40)},
    )
    sides = _mb_row(run)["solutions"]
    assert sides["andrey.mb_by_mb"]["tests_per_answer"] == 20
    assert sides["andrey.mb_by_mb"]["tests_all_targets"] == 40
    assert sides["andrey.mb_by_mb"]["seconds_per_answer"] == pytest.approx(2.0)
    # A global fit answers every target at once: its one fit is the cost either way.
    assert sides["andrey.pc.gsq"]["tests_per_answer"] == 15
    assert sides["andrey.pc.gsq"]["tests_all_targets"] == 15
    assert sides["andrey.pc.gsq"]["seconds_all_targets"] == pytest.approx(1.5)
    assert sides["andrey.cmb"]["tests_all_targets"] == 80


def test_a_more_accurate_comparison_with_fewer_tests_per_answer_dominates(tmp_path):
    missing = {"andrey.mb_by_mb": [2, 2], "andrey.pc.gsq": [1, 1]}
    tests = {"andrey.mb_by_mb": (10, 10), "andrey.pc.gsq": (5, 5)}
    row = _mb_row(_mb_run(tmp_path / "a", missing, tests))
    assert row["against"]["andrey.pc.gsq"]["verdict"] == "worse"
    assert row["frontier"]["dominated"] and row["frontier"]["dominated_by"] == ["andrey.pc.gsq"]
    assert row["frontier"]["dominated_all_targets"]
    # Without fresh seeds the worse cell is not replicated, and nothing dominates.
    unrun = _mb_row(_mb_run(tmp_path / "b", missing, tests, seeds=range(20)))
    assert unrun["frontier"]["against"]["andrey.pc.gsq"]["verdict"] == "worse, not replicated"
    assert not unrun["frontier"]["dominated"]


def test_a_more_accurate_global_fit_costing_more_per_answer_dominates_only_over_every_target(
    tmp_path,
):
    row = _mb_row(
        _mb_run(
            tmp_path,
            {"andrey.mb_by_mb": [2, 2], "andrey.pc.gsq": [1, 1]},
            {"andrey.mb_by_mb": (10, 10), "andrey.pc.gsq": (15, 15)},
        )
    )
    pc = row["frontier"]["against"]["andrey.pc.gsq"]
    assert pc["verdict"] == "worse"
    assert not pc["fewer_tests_per_answer"] and pc["fewer_tests_all_targets"]
    assert not row["frontier"]["dominated"] and row["frontier"]["dominated_by"] == []
    assert row["frontier"]["dominated_all_targets_by"] == ["andrey.pc.gsq"]


def test_the_chessboard_is_judged_on_its_own_seed_windows(tmp_path):
    run = _mb_run(
        tmp_path,
        {"andrey.mb_by_mb": [1, 1], "andrey.pc.gsq": [2, 2]},
        {"andrey.mb_by_mb": (10, 10), "andrey.pc.gsq": (50, 50)},
        seeds=range(20),
        network="chess45",
    )
    out = tmp_path / "rows.json"
    direct_summary.main([str(run), "--method", "mb_by_mb", "--json", str(out)])
    (row,) = [r for r in json.loads(out.read_text()) if r["alpha"] == MB_CELL[2]]
    assert row["network"] == "chess45"
    assert row["solutions"]["andrey.mb_by_mb"]["datasets"] == 10
    assert row["fresh_solutions"]["andrey.mb_by_mb"]["datasets"] == 10
