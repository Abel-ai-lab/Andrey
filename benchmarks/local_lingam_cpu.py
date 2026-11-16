#!/usr/bin/env python
"""Benchmark the weighted edges into a target and into its children: local ISA-LiNG against the
authors' code, global LiNGAM read at the target, and the empty answer.

Runs the simulations of the local ISA-LiNG paper (Dai et al., 2024, Sect. 5;
:mod:`andrey_bench.local_ling_models`) and its real-data check:

- ``cyclic``: 50 variables, at most 4 neighbors each, disjoint stable cycles, a target on a cycle;
- ``acyclic-d3`` and ``acyclic-d5``: Erdos-Renyi DAGs of 50 variables with expected degree 3 and
  5, a target whose true blanket has about 14 and 20 members;
- ``sachs``: the 853 observational cells of Sachs et al. (2005) at the targets PIP2, PIP3, and Akt,
  scored on the 17-arc consensus network (structure only: it has no weights).

``--rows`` sets the sample sizes of the simulated cells (the paper's 2,000; its ladder is 100,
300, 1,000, 3,000, and 10,000 at degree 3). Every dataset has one target. A local solution is
fitted once per target with ``acyclic`` true in the acyclic cells and on Sachs (the paper's
Appendix C there) and false in the cyclic cell (Algorithm 1); an oracle arm receives the true
blanket in ``params["blanket"]``. A global solution (``local = False``) is fitted on every column
of an acyclic dataset and read at the target. Each answer is scored on its own local structure
against the truth's (:func:`~andrey_bench.local_ling_models.local_scores`): the paper's SHD and
the Euclidean distance between the weights.

One JSON line per fit goes to ``<out>/records.jsonl`` with the answer's local weights, its blanket,
and the scores, so ``--rescore`` recomputes the scores without fitting; ``<out>/provenance.json``
records the build and the machine. A fit past the per-fit wall cap is a ``timeout``, and an
answer the authors' code declined to compute (Algorithm 1 past its local size) is ``not_run``;
neither has scores. Set ``$ANDREY_BENCH_LOCAL_LING`` and ``$ANDREY_BENCH_LOCAL_LING_PYTHON`` as
``start_local_ling.sh`` prints before running the references.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/local_lingam_cpu.py --out outputs/lil
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from andrey_bench import local_ling_models as models
from andrey_bench import rundir
from andrey_bench.adapters.andrey_local_isa_ling import AndreyGlobalLiNGAM, AndreyLocalISALiNG
from andrey_bench.adapters.dai_local_isa_ling import DaiLocalISALiNG
from andrey_bench.local_lane import TARGET_PARAM
from andrey_bench.runner import available_cores, check_interpreters, run_task, split_cores

#: Every solution, in the order records are written; ``empty`` is scored without a fit.
SOLUTIONS = [
    AndreyLocalISALiNG(),
    AndreyLocalISALiNG(oracle=True),
    DaiLocalISALiNG(),
    DaiLocalISALiNG(oracle=True),
    AndreyGlobalLiNGAM("direct_lingam"),
    AndreyGlobalLiNGAM("ica_lingam"),
    AndreyLocalISALiNG(acyclic=False),
]
EMPTY = "empty"

#: The cells: setting and expected degree.
CELLS = {
    "cyclic": ("cyclic", 4),
    "acyclic-d3": ("acyclic", 3),
    "acyclic-d5": ("acyclic", 5),
    "sachs": ("sachs", 0),
}

#: The paper's sample size and Sachs targets.
ROWS = (2000,)
SACHS_TARGETS = ("pip2", "pip3", "akt")

#: Smallest run that touches every solution: two datasets of the acyclic cell at 500 rows.
SMOKE = {"cells": ["acyclic-d3"], "rows": [500], "seeds": 2, "cap_wall_s": 600.0}

RECORDS = "records.jsonl"


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(out: Path) -> list[dict]:
    path = out / RECORDS
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def sachs() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """The observational Sachs data, its consensus DAG as 0/1 ``truth[i, j]`` for ``i -> j``, and
    the column names."""
    from andrey.data import load_dataset

    ds = load_dataset("sachs", as_frame=False, shuffle=False)
    names = list(ds.feature_names)
    truth = np.zeros((len(names), len(names)))
    for j, child in enumerate(names):
        for parent in ds.graph.parents(child):
            truth[names.index(parent), j] = 1.0
    return np.asarray(ds.data, dtype=np.float64), truth, tuple(ds.feature_names)


def datasets(cell: str, rows: list[int], seeds: range):
    """Yield ``(n, seed, data, truth, targets)`` of ``cell``, ``targets`` as ``{target: blanket}``."""
    setting, degree = CELLS[cell]
    if setting == "sachs":
        data, truth, names = sachs()
        targets = {
            names.index(t): models.true_blanket(truth, names.index(t)) for t in SACHS_TARGETS
        }
        yield len(data), 0, data, truth, targets
        return
    for seed in seeds:
        for n in rows:
            draw = models.sample(setting, n, seed, degree)
            yield n, seed, draw.data, draw.weights, {draw.target: draw.blanket}


def _params(adapter, cell: str, target: int | None, blanket: tuple[int, ...]) -> dict:
    params = dict(adapter.params())
    if getattr(adapter, "local", False):
        params[TARGET_PARAM] = target
        params.setdefault("acyclic", CELLS[cell][0] != "cyclic")
        if getattr(adapter, "oracle", False):
            params["blanket"] = list(blanket)
    return params


def runs_in(adapter, cell: str) -> bool:
    """Global LiNGAM assumes a DAG, and the default-path arm is context on acyclic data, so both
    run on the acyclic cells and Sachs only."""
    acyclic_only = not getattr(adapter, "local", False) or adapter.params().get("acyclic") is False
    return not (acyclic_only and CELLS[cell][0] == "cyclic")


def score(weights: np.ndarray, truth: np.ndarray, target: int, weighted: bool) -> dict:
    s = models.local_scores(weights, truth, target)
    if not weighted:
        s["weight_dist"] = None
    return s


def blanket_scores(found: list[int], true: tuple[int, ...]) -> dict:
    tp = len(set(found) & set(true))
    return {
        "blanket_size": len(found),
        "blanket_true": len(true),
        "blanket_precision": tp / len(found) if found else None,
        "blanket_recall": tp / len(true) if true else None,
    }


def local_entries(weights: np.ndarray, target: int) -> list[list]:
    loc = models.local_structure(weights, target)
    return [[int(i), int(j), float(loc[i, j])] for i, j in zip(*np.nonzero(loc))]


def rescore(out: Path) -> list[dict]:
    """Recompute every record's scores from its stored local weights and rewrite the records."""
    records, truths = _read(out), {}
    for r in records:
        if r.get("weights") is None:
            continue
        key = (r["cell"], r["n"], r["seed"])
        if key not in truths:
            truths[key] = next(datasets(r["cell"], [r["n"]], range(r["seed"], r["seed"] + 1)))[3]
        w = np.zeros((r["d"], r["d"]))
        for i, j, v in r["weights"]:
            w[i, j] = v
        r["scores"] = score(w, truths[key], r["target"], r["cell"] != "sachs")
    (out / RECORDS).write_text("".join(json.dumps(r) + "\n" for r in records))
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local ISA-LiNG against the authors' code, CPU")
    p.add_argument("--out", required=True, help="output directory, created if absent")
    p.add_argument(
        "--cells", nargs="+", default=["cyclic", "acyclic-d3", "acyclic-d5"], choices=list(CELLS)
    )
    p.add_argument("--rows", type=int, nargs="+", default=list(ROWS), help="sample sizes (2000)")
    p.add_argument("--seeds", type=int, default=20, help="datasets per cell and size (default: 20)")
    p.add_argument("--first-seed", type=int, default=0, help="the first seed (default: 0)")
    p.add_argument("--part", type=int, default=0, help="run every --parts-th dataset from this one")
    p.add_argument("--parts", type=int, default=1, help="parts the datasets are split into")
    p.add_argument("--cap-wall-s", type=float, default=1800.0, help="wall cap per fit, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="memory cap per fit, MB")
    p.add_argument("--cores", type=int, default=available_cores(), help="core budget per fit")
    p.add_argument(
        "--solutions", default=None, help="run only these solutions, comma-separated (and empty)"
    )
    p.add_argument("--machine-id", default=socket.gethostname(), help="machine id on every record")
    p.add_argument("--smoke", action="store_true", help="acyclic-d3, 500 rows, two seeds, 600 s")
    p.add_argument("--resume", action="store_true", help="skip the fits already in --out")
    p.add_argument("--rescore", action="store_true", help="fit nothing; rescore --out's records")
    args = p.parse_args(argv)
    if args.smoke:
        vars(args).update(SMOKE)

    field = list(SOLUTIONS)
    if args.solutions is not None:
        chosen = set(args.solutions.split(",")) - {EMPTY}
        known = {a.name for a in field}
        if chosen - known:
            p.error(
                f"unknown solutions {sorted(chosen - known)}; the solutions are {sorted(known)}"
            )
        field = [a for a in field if a.name in chosen]

    out = Path(args.out)
    if args.rescore:
        records = rescore(out)
        _log(f"rescored    : {len(records)} records -> {out / RECORDS}")
        _summarize(records)
        return 0
    if (out / RECORDS).exists() and not args.resume:
        raise SystemExit(f"{out} already holds a run; pass --resume to continue it")
    out.mkdir(parents=True, exist_ok=True)
    check_interpreters(field)
    run_meta = {
        "driver": Path(__file__).name,
        "solution list": [a.name for a in field],
        "params": {a.name: a.params() for a in field},
        "cells": args.cells,
        "rows": args.rows,
        "seeds": args.seeds,
        "first_seed": args.first_seed,
        "part": args.part,
        "parts": args.parts,
        "cap_wall_s": args.cap_wall_s,
        "cap_mem_mb": args.cap_mem_mb,
        "cores": args.cores,
        "machine_id": args.machine_id,
    }
    meta_path = out / "run_meta.json"
    if args.resume and meta_path.exists() and json.loads(meta_path.read_text()) != run_meta:
        raise SystemExit(f"--resume: {meta_path} differs from these flags; pass the first run's")
    meta_path.write_text(json.dumps(run_meta, indent=2))
    ref = os.environ.get(DaiLocalISALiNG.python_env)
    venvs = {"local-ling": Path(ref).absolute().parent.parent} if ref else None
    prov = rundir.provenance_once(
        out, timestamp=datetime.now(timezone.utc).isoformat(), extra_venvs=venvs
    )
    _log(f"=== local ISA-LiNG lane -> {out} (git {prov['git_sha']}) ===")
    for name, value in run_meta.items():
        _log(f"{name:12s}: {value}")

    done = {(r["cell"], r["n"], r["seed"], r["solution"], r["target"]) for r in _read(out)}
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    units = [
        (cell, n, seed) for cell in args.cells for n, seed in _unit_keys(cell, args.rows, seeds)
    ]
    units = units[args.part :: args.parts]
    start = time.perf_counter()
    with (out / RECORDS).open("a") as sink:
        for cell, n, seed in units:
            _, _, data, truth, targets = next(datasets(cell, [n], range(seed, seed + 1)))
            weighted = cell != "sachs"
            base = {
                "cell": cell,
                "n": n,
                "seed": seed,
                "d": data.shape[1],
                "machine_id": args.machine_id,
            }
            walls: dict[str, float] = {}
            for target, blanket in targets.items():
                key = (cell, n, seed, EMPTY, target)
                if key not in done:
                    empty = np.zeros_like(truth)
                    sink.write(
                        json.dumps(
                            {
                                **base,
                                "solution": EMPTY,
                                "target": target,
                                "status": "ok",
                                "weights": [],
                                "scores": score(empty, truth, target, weighted),
                            }
                        )
                        + "\n"
                    )
            for adapter in field:
                if not runs_in(adapter, cell):
                    continue
                local = getattr(adapter, "local", False)
                todo = [t for t in targets if (cell, n, seed, adapter.name, t) not in done]
                if not todo:
                    continue
                answers = {}
                for target in todo if local else [None]:
                    params = _params(adapter, cell, target, targets.get(target, ()))
                    threads, n_jobs = split_cores(args.cores, adapter.mode)
                    (m,) = run_task(
                        adapter,
                        data,
                        params,
                        cap_wall_s=args.cap_wall_s,
                        cap_mem_mb=args.cap_mem_mb,
                        repeats=1,
                        warmup=0,
                        threads=threads,
                        n_jobs=n_jobs,
                    )
                    answers[target] = (params, m)
                for target in todo:
                    params, m = answers[target if local else None]
                    record = {
                        **base,
                        "solution": adapter.name,
                        "package": adapter.package,
                        "package_version": adapter.package_version,
                        "target": target,
                        "global": not local,
                        "status": m.get("status"),
                        "wall_s": m.get("wall_s"),
                        "cpu_s": m.get("cpu_s"),
                        "peak_rss_mb": m.get("peak_rss_mb"),
                        "error_text": m.get("error_text") or m.get("stderr"),
                        "params": {k: v for k, v in params.items() if k != "blanket"},
                        "weights": None,
                        "scores": None,
                    }
                    if m.get("status") == "ok" and m.get("adj") is not None:
                        a = models.unpack_answer(np.asarray(m["adj"]))
                        if a["code"] == models.NOT_RUN:
                            record["status"] = "not_run"
                            record["m"] = a["m"]
                        else:
                            record.update(m=a["m"], models=a["models"], fallback=a["fallback"])
                            record["weights"] = local_entries(a["weights"], target)
                            record["scores"] = score(a["weights"], truth, target, weighted)
                            if local:
                                record.update(blanket_scores(a["blanket"], targets[target]))
                        walls[adapter.name] = m.get("wall_s") or 0.0
                    sink.write(json.dumps(record, default=str) + "\n")
                    sink.flush()
            cost = "  ".join(f"{a}={w:.3g}s" for a, w in walls.items())
            _log(f"[{cell} n={n} seed={seed}] {cost} | elapsed {time.perf_counter() - start:.0f}s")
    _summarize(_read(out))
    return 0


def _unit_keys(cell: str, rows: list[int], seeds: range) -> list[tuple[int, int]]:
    if CELLS[cell][0] == "sachs":
        return [(853, 0)]
    return [(n, seed) for seed in seeds for n in rows]


def _summarize(records: list[dict]) -> None:
    """Print, per cell, size, and solution: finished fits, median time, mean SHD and distance."""
    if not records:
        return
    _log("")
    _log(
        f"{'solution':34s} {'cell':11s} {'n':>6s} {'ok/all':>7s} {'wall_s':>8s} {'shd':>6s} "
        f"{'w_dist':>7s}"
    )
    for cell, n in sorted({(r["cell"], r["n"]) for r in records}):
        for name in sorted({r["solution"] for r in records}):
            mine = [r for r in records if (r["cell"], r["n"], r["solution"]) == (cell, n, name)]
            fit = [r for r in mine if r.get("scores")]
            if not fit:
                continue
            walls = [r["wall_s"] for r in fit if r.get("wall_s") is not None]
            dists = [
                r["scores"]["weight_dist"] for r in fit if r["scores"]["weight_dist"] is not None
            ]
            _log(
                f"{name:34s} {cell:11s} {n:6d} {len(fit):3d}/{len(mine):<3d} "
                f"{statistics.median(walls) if walls else float('nan'):8.3g} "
                f"{statistics.mean(r['scores']['shd'] for r in fit):6.3g} "
                f"{statistics.mean(dists) if dists else float('nan'):7.3g}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
