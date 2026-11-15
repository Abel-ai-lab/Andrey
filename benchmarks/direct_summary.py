#!/usr/bin/env python
"""Judge a target-centered method of ``direct_cpu.py`` per cell: against global PC read at the
target, against the empty answer, and beside its paper's published counts.

Each method is scored in its paper's reading (:data:`PAPERS`). CMB's are the errors at the target
against the truth's CPDAG (:func:`~andrey_bench.direct_lane.cpdag_counts`): the extra, missing, and
reversed edges, whose sum is ``total``; ``lenient`` leaves out the ``under`` reversals, edges the
CPDAG directs that the answer leaves undirected. MB-by-MB's are the errors at the target against
the true DAG (:func:`~andrey_bench.direct_lane.dag_counts`): the ``undirected``, ``miss``,
``extra``, and ``reverse`` edges, whose sum is ``shd``. The paired quantity is a dataset's mean
over its targets of the method's primary measure (``total`` or ``shd``). For each comparison,
:func:`~andrey_bench.pairing.compare` gives the mean over paired datasets of the method's measure
minus the other's, with its percentile-bootstrap 95% interval; the verdict is ``better`` or
``worse`` when the interval lies wholly on that side of 0, over at least
:data:`~andrey_bench.pairing.MIN_PAIRS` datasets. A cell is judged on the ``--first`` seeds, or
the paper's own window for its network; a cell worse there is judged again on the ``--fresh``
seeds, and counts as worse only when it is worse again. The floor is the method's measure minus
the empty answer's, which passes when its interval lies wholly below 0.

A method with a ``frontier`` also gets each solution's cost: per answer, what answering one
target costs, a target-centered method's tests and seconds averaged over the targets and a global
method's whole fit; over every target, a target-centered method's sum over the targets and a
global method's one fit; each the mean over the datasets. A comparison dominates the method when
it is more accurate (the method's verdict against it is ``worse``) and takes fewer tests per
answer; ``all_targets`` reads the cost over every target instead.

    $ANDREY_BENCH_PYTHON benchmarks/direct_summary.py RUN [RUN ...] --method cmb --json rows.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from andrey_bench.pairing import MIN_PAIRS, compare

#: The measures of each reading: the CPDAG's of the CMB paper and the DAG's of the MB-by-MB paper.
READINGS = {
    "cpdag": ("extra", "missing", "reversed", "under", "total", "lenient"),
    "dag": ("undirected", "miss", "extra", "reverse", "shd"),
}

#: Wang et al. (2014), Tables 1-3, the MB-by-MB rows: per cell (network, rows, level), the means
#: over its datasets of the undirected, missing, extra, and reversed edges at a target and their sum.
MB_BY_MB_PAPER = {
    ("alarm", 500, 0.01): (0.59, 1.16, 0.08, 0.09, 1.92),
    ("alarm", 500, 0.05): (0.39, 1.02, 0.25, 0.23, 1.89),
    ("alarm", 1000, 0.01): (0.45, 0.79, 0.05, 0.11, 1.40),
    ("alarm", 1000, 0.05): (0.30, 0.70, 0.21, 0.24, 1.45),
    ("alarm", 2000, 0.01): (0.30, 0.53, 0.04, 0.12, 0.99),
    ("alarm", 2000, 0.05): (0.24, 0.49, 0.18, 0.21, 1.11),
    ("alarm", 5000, 0.01): (0.22, 0.30, 0.03, 0.08, 0.63),
    ("alarm", 5000, 0.05): (0.18, 0.27, 0.16, 0.17, 0.78),
    ("hailfinder", 500, 0.01): (0.42, 1.71, 0.12, 0.27, 2.51),
    ("hailfinder", 500, 0.05): (0.39, 1.64, 0.16, 0.33, 2.51),
    ("hailfinder", 1000, 0.01): (0.48, 1.46, 0.14, 0.23, 2.31),
    ("hailfinder", 1000, 0.05): (0.34, 1.40, 0.19, 0.36, 2.30),
    ("hailfinder", 2000, 0.01): (0.40, 1.11, 0.14, 0.30, 1.95),
    ("hailfinder", 2000, 0.05): (0.25, 1.10, 0.20, 0.47, 2.02),
    ("hailfinder", 5000, 0.01): (0.19, 0.91, 0.12, 0.22, 1.44),
    ("hailfinder", 5000, 0.05): (0.24, 0.91, 0.16, 0.20, 1.51),
    ("chess45", 500, 0.01): (0.05, 0.96, 0.38, 0.66, 2.05),
    ("chess45", 500, 0.05): (0.01, 0.93, 0.69, 0.62, 2.25),
    ("chess45", 1000, 0.01): (0.03, 0.71, 0.40, 0.58, 1.72),
    ("chess45", 1000, 0.05): (0.01, 0.70, 0.69, 0.80, 2.20),
    ("chess45", 2000, 0.01): (0.01, 0.34, 0.39, 0.53, 1.27),
    ("chess45", 2000, 0.05): (0.01, 0.33, 0.58, 0.59, 1.51),
    ("chess45", 5000, 0.01): (0.02, 0.06, 0.31, 0.43, 0.82),
    ("chess45", 5000, 0.05): (0.00, 0.06, 0.72, 0.69, 1.47),
}

#: Each method's paper: its reading and primary measure, its cells (network, rows, level) with the
#: published means per target, the solutions it is judged against (``against``, every cell, and
#: ``against_where_run``, the cells where they ran), a network's seed windows (first, fresh) where
#: they differ from the command line's, and whether its rows weigh accuracy against cost.
PAPERS = {
    "cmb": {
        "solution": "andrey.cmb",
        "source": "Gao and Ji (2015), Table 1",
        "reading": "cpdag",
        "primary": "total",
        "cells": {
            ("alarm", 1000, 0.02): {"extra": 0.69, "missing": 0.61, "reversed": 0.51, "total": 1.81}
        },
        "against": ("andrey.pc.gsq", "bnlearn.pc-stable.graph"),
    },
    "mb_by_mb": {
        "solution": "andrey.mb_by_mb",
        "source": "Wang et al. (2014), Tables 1-3",
        "reading": "dag",
        "primary": "shd",
        "cells": {
            cell: dict(zip(READINGS["dag"], counts)) for cell, counts in MB_BY_MB_PAPER.items()
        },
        "against": ("andrey.pc.gsq",),
        "against_where_run": ("bnlearn.pc-stable.graph", "andrey.cmb"),
        "windows": {"chess45": ("0-9", "10-19")},
        "frontier": True,
    },
}

#: The empty answer of the floor.
EMPTY = "baseline.empty"
#: Each solution's cost on a side of a row: the field written and the dataset's field it averages.
COSTS = {
    "tests_per_answer": "tests_per_answer",
    "tests_all_targets": "tests",
    "seconds_per_answer": "wall_s_per_answer",
    "seconds_all_targets": "wall_s",
}


def load(runs: list[str]) -> list[dict]:
    """Every record of the run folders, each ``records.jsonl`` of ``direct_cpu.py``."""
    return [
        json.loads(line)
        for run in runs
        for line in (Path(run) / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]


def dataset_means(
    records: list[dict], reading: str = "cpdag"
) -> dict[tuple[str, str, float, int], dict]:
    """Per ``(cell, solution, alpha, seed)``: the mean of each measure of ``reading`` over the
    targets, the number of targets, whether every target was answered (no timeout or failure), and
    the cost over every target (``wall_s``, ``tests``) and per answer."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["cell"], r["solution"], r["params"].get("alpha"), r["seed"])].append(r)
    out = {}
    for key, rows in groups.items():
        scored = [r["scores"][reading] for r in rows if r["status"] == "ok" and r["scores"]]
        complete = len(scored) == len(rows)
        means = {}
        if complete:
            for m in READINGS[reading]:
                values = [s["total"] - s["under"] if m == "lenient" else s[m] for s in scored]
                means[m] = float(np.mean(values))
        walls = {r["wall_s"] for r in rows if r["wall_s"] is not None}
        tests = [r["n_tests"] for r in rows if r["n_tests"] is not None]
        graph = rows[0]["scope"] == "graph"
        wall_s = sum(walls) if graph else sum(r["wall_s"] or 0 for r in rows)
        n_tests = (tests[0] if graph else sum(tests)) if tests else None
        out[key] = {
            "complete": complete,
            "targets": len(rows),
            "means": means,
            "wall_s": wall_s,
            "tests": n_tests,
            "wall_s_per_answer": wall_s if graph else wall_s / len(rows),
            "tests_per_answer": n_tests if graph or n_tests is None else n_tests / len(rows),
        }
    return out


def _side(per: dict, seeds: list[int], measures: tuple[str, ...], cost: bool = False) -> dict:
    fits = [per[s] for s in seeds if s in per and per[s]["complete"]]
    side = {"datasets": len(fits)}
    for m in measures:
        side[m] = round(float(np.mean([f["means"][m] for f in fits])), 4) if fits else None
    side["wall_s_median"] = (
        round(float(np.median([f["wall_s"] for f in fits])), 3) if fits else None
    )
    if cost:
        for name, field in COSTS.items():
            values = [f[field] for f in fits if f[field] is not None]
            side[name] = round(float(np.mean(values)), 4) if values else None
    return side


def _paired(a: dict, b: dict, seeds: list[int], measure: str) -> dict:
    both = [s for s in seeds if a.get(s, {}).get("complete") and b.get(s, {}).get("complete")]
    if not both:
        return {"paired": 0, "mean_D": None, "bootstrap_95": None, "verdict": "too few pairs"}
    result = compare([a[s]["means"][measure] for s in both], [b[s]["means"][measure] for s in both])
    if result["paired"] < MIN_PAIRS:
        result["verdict"] = "too few pairs"
    return result


def replicated(first: dict, fresh: dict | None) -> str:
    """The cell's verdict: a cell worse on the first seeds is ``worse`` only when it is worse again
    on the fresh ones, ``tie`` when it is not, and ``worse, not replicated`` when the fresh seeds
    give too few pairs to judge."""
    if first["verdict"] != "worse":
        return first["verdict"]
    if fresh is None or fresh["verdict"] == "too few pairs":
        return "worse, not replicated"
    return "worse" if fresh["verdict"] == "worse" else "tie"


def frontier(row: dict, solution: str) -> dict:
    """The comparisons that dominate ``solution`` in ``row``: more accurate, its verdict against
    them ``worse``, and fewer tests per answer; the ``all_targets`` reading compares the tests over
    every target instead."""

    def fewer(name: str, cost: str) -> bool:
        theirs, mine = row["solutions"][name][cost], row["solutions"][solution][cost]
        return theirs is not None and mine is not None and theirs < mine

    out = {"dominated": False, "dominated_by": [], "dominated_all_targets": False}
    out["dominated_all_targets_by"], out["against"] = [], {}
    for name, entry in row["against"].items():
        more_accurate = entry["verdict"] == "worse"
        per_answer, all_targets = fewer(name, "tests_per_answer"), fewer(name, "tests_all_targets")
        out["against"][name] = {
            "verdict": entry["verdict"],
            "fewer_tests_per_answer": per_answer,
            "fewer_tests_all_targets": all_targets,
            "dominates": more_accurate and per_answer,
            "dominates_all_targets": more_accurate and all_targets,
        }
        if more_accurate and per_answer:
            out["dominated_by"].append(name)
        if more_accurate and all_targets:
            out["dominated_all_targets_by"].append(name)
    out["dominated"] = bool(out["dominated_by"])
    out["dominated_all_targets"] = bool(out["dominated_all_targets_by"])
    return out


def cell_row(per: dict, method: str, cell: tuple, first: list[int], fresh: list[int]) -> dict:
    """The row of one cell: each solution's means, the comparisons, the floor, the paper, and for
    a method with a ``frontier``, each solution's cost and the comparisons that dominate it."""
    network, n, alpha = cell
    label = f"net:{network}:{n}"
    paper = PAPERS[method]
    solution, primary = paper["solution"], paper["primary"]
    measures, cost = READINGS[paper["reading"]], paper.get("frontier", False)

    def of(name: str) -> dict:
        # The empty answer takes no level.
        level = None if name == EMPTY else alpha
        return {k[3]: v for k, v in per.items() if k[:3] == (label, name, level)}

    mine, empty = of(solution), of(EMPTY)
    against = [*paper["against"], *(a for a in paper.get("against_where_run", ()) if of(a))]
    names = (solution, *against, EMPTY)
    row = {
        "cell": label,
        "network": network,
        "n": n,
        "alpha": alpha,
        "solutions": {name: _side(of(name), first, measures, cost) for name in names},
        "paper": paper["cells"][cell],
        "against": {},
        "floor": _paired(mine, empty, first, primary),
    }
    for name in against:
        first_d = _paired(mine, of(name), first, primary)
        entry = {"first": first_d}
        if "lenient" in measures:
            entry["lenient"] = _paired(mine, of(name), first, "lenient")
        if fresh:
            entry["fresh"] = _paired(mine, of(name), fresh, primary)
        entry["verdict"] = replicated(first_d, entry.get("fresh"))
        row["against"][name] = entry
    if fresh:
        row["fresh_solutions"] = {name: _side(of(name), fresh, measures, cost) for name in names}
    interval = row["floor"]["bootstrap_95"]
    row["floor"]["pass"] = row["floor"]["paired"] >= MIN_PAIRS and interval[1] < 0
    if cost:
        row["frontier"] = frontier(row, solution)
    return row


def _seeds(text: str) -> list[int]:
    low, high = (int(x) for x in text.split("-"))
    return list(range(low, high + 1))


def _said(result: dict, verdict: str) -> str:
    mean = result["mean_D"]
    return verdict if mean is None else f"{verdict} ({mean:+.3f})"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="run folders of direct_cpu.py holding records.jsonl")
    p.add_argument("--method", choices=sorted(PAPERS), required=True)
    p.add_argument("--first", default="0-19", help="seeds judged first (default: 0-19)")
    p.add_argument("--fresh", default="20-39", help="seeds of the reruns (default: 20-39)")
    p.add_argument("--json", type=Path, default=None, help="write the rows here")
    args = p.parse_args(argv)
    paper = PAPERS[args.method]
    per = dataset_means(load(args.runs), paper["reading"])
    rows = []
    for cell in paper["cells"]:
        network, n, alpha = cell
        if (f"net:{network}:{n}", paper["solution"], alpha) not in {k[:3] for k in per}:
            print(f"{cell}: no records, left out")
            continue
        first, fresh = paper.get("windows", {}).get(network, (args.first, args.fresh))
        rows.append(cell_row(per, args.method, cell, _seeds(first), _seeds(fresh)))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    primary = paper["primary"]
    for row in rows:
        mine = row["solutions"][paper["solution"]]
        verdicts = ", ".join(
            f"{k}: {_said(v['first'], v['verdict'])}" for k, v in row["against"].items()
        )
        floor = _said(row["floor"], "pass" if row["floor"]["pass"] else "FAIL")
        print(
            f"{row['cell']} alpha {row['alpha']}: {primary} {mine[primary]} (paper "
            f"{row['paper'][primary]}); {verdicts}; floor {floor}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
