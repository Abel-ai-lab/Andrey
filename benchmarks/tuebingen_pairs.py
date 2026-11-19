#!/usr/bin/env python
"""Run the pairwise methods and the authors' code of QCCD, HECI, and GRCI on the Tuebingen pairs.

``andrey.load_dataset("tuebingen")`` gives the cause-effect pairs of Mooij et al. (2016). The lane
runs every solution (:data:`SOLUTIONS`; ``--solutions`` picks some) on each pair that has one column
per side, in a forced decision: ``x -> y``, ``y -> x``, or no decision. Andrey's ``anm``, ``pnl``,
``loci`` (both criteria), and ``resit`` are the Andrey arms; ``qccd`` (the paper's ``m = 3``),
``qccd_m7`` (the wrapper's default ``m = 7``), ``heci``, and ``grci`` are the authors' code, each
unmodified and read from the directory an environment variable names (each
``benchmarks/start_<name>.sh`` downloads it and prints the export). Nothing of the authors' code is
copied into this repository, and GRCI's license (CC BY-NC 4.0) limits it to benchmark runs from a
local download.

Each fit runs in its own process with one numeric thread under a per-fit wall cap (30 minutes by
default); a fit past the cap is recorded as ``timeout`` and has no decision. ``--workers`` fits run
at once, so a job packs many pairs: give it that many cores. One JSON line per fit goes to
``<out>/records.jsonl``, with the machine, so the summary can state the hardware; a rerun skips the
(pair, solution) units already there. ``tuebingen_summary.py`` turns the records of one or more runs
into the table.

    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/tuebingen_pairs.py --prepare-only
    ANDREY_BENCH_PYTHON=... $ANDREY_BENCH_PYTHON benchmarks/tuebingen_pairs.py --out outputs/tue-a \\
        --pairs 1-20 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from andrey_bench.adapters.andrey_anm import AndreyANM
from andrey_bench.adapters.andrey_loci import AndreyLOCI, AndreyLOCILikelihood
from andrey_bench.adapters.andrey_pnl import AndreyPNL
from andrey_bench.adapters.andrey_resit import AndreyRESIT
from andrey_bench.runner import check_interpreters, run_task

#: The authors' code, as ``(module, class)`` of its adapter; imported only when it is asked for.
REFERENCES = {
    "qccd": ("andrey_bench.adapters.qccd_bqcd", "TagasovskaQCCD"),
    "qccd_m7": ("andrey_bench.adapters.qccd_bqcd", "TagasovskaQCCDM7"),
    "heci": ("andrey_bench.adapters.heci_xu", "XuHECI"),
    "grci": ("andrey_bench.adapters.grci_strobl", "StroblGRCI"),
}

#: Andrey's pairwise methods and RESIT on a pair, by the short name the summary uses.
ANDREY = {
    "anm": AndreyANM,
    "pnl": AndreyPNL,
    "loci": AndreyLOCI,
    "loci_likelihood": AndreyLOCILikelihood,
    "resit": AndreyRESIT,
}

SOLUTIONS = (*ANDREY, *REFERENCES)

#: Per-fit wall cap in seconds, import and setup included: the campaign's 30 minutes.
CAP_WALL_S = 1800.0


def make_solution(name: str) -> Any:
    """Return the adapter of solution ``name``."""
    if name in ANDREY:
        return ANDREY[name]()
    from importlib import import_module

    module, cls = REFERENCES[name]
    return getattr(import_module(module), cls)()


def preflight(adapters: dict[str, Any]) -> dict[str, str]:
    """Check every fit interpreter and every reference's code folder before any fit runs.

    Returns ``{solution: version}``: the version of the package each solution's fits run, read from
    the interpreter that runs them when it is not the driver's.
    """
    from importlib import import_module

    interpreters = check_interpreters(list(adapters.values()))
    for name in adapters:
        if name in REFERENCES:
            import_module(REFERENCES[name][0]).code_dir()
    versions = {}
    for name, adapter in adapters.items():
        versions[name] = adapter.package_version
        if adapter.package == "andrey" and getattr(adapter, "python_env", None):
            command = [
                str(interpreters[adapter.name]),
                "-c",
                "import andrey; print(andrey.__version__)",
            ]
            run = subprocess.run(command, capture_output=True, text=True, check=True)
            versions[name] = run.stdout.strip()
    return versions


def parse_pairs(text: str) -> set[int]:
    """Read pair numbers such as ``1-20,25,40-45`` into a set."""
    numbers: set[int] = set()
    for part in text.split(","):
        first, _, last = part.partition("-")
        numbers.update(range(int(first), int(last or first) + 1))
    return numbers


def machine() -> dict[str, str]:
    """The node and CPU the fit ran on, for the summary's hardware line."""
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor()
    return {
        "node": socket.gethostname(),
        "cpu": cpu,
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    }


def done_units(path: Path) -> set[tuple[str, str]]:
    """The ``(pair, solution)`` units already in ``path``."""
    if not path.is_file():
        return set()
    units = set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        units.add((record["pair"], record["solution"]))
    return units


def decision(adapter: Any, adj: Any) -> str:
    """Name the fit's edge on the pair's two columns: ``x->y``, ``y->x``, or ``none``."""
    estimate = adapter.to_structure(np.asarray(adj))
    edges = [(int(i), int(j)) for i, j, _ in estimate.oriented_edges()]
    return {(): "none", ((0, 1),): "x->y", ((1, 0),): "y->x"}.get(tuple(edges), "other")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tuebingen pairs: Andrey's pairwise methods and refs")
    p.add_argument("--out", type=Path, help="output directory, created if absent")
    p.add_argument("--pairs", default="1-108", help="pair numbers to run, such as 1-20,25")
    p.add_argument("--solutions", nargs="+", choices=SOLUTIONS, default=list(SOLUTIONS))
    p.add_argument("--workers", type=int, default=1, help="fits at once, one core each")
    p.add_argument("--cap-wall-s", type=float, default=CAP_WALL_S, help="per-fit wall cap, seconds")
    p.add_argument("--cap-mem-mb", type=float, default=None, help="per-fit memory cap, MB")
    p.add_argument("--data-home", type=Path, default=None, help="load_dataset's cache directory")
    p.add_argument("--prepare-only", action="store_true", help="download the pairs and stop")
    args = p.parse_args(argv)

    from andrey.data import load_dataset

    collection = load_dataset("tuebingen", as_frame=False, data_home=args.data_home)
    if args.prepare_only:
        print(f"{len(collection)} pairs cached", flush=True)
        return 0
    if args.out is None:
        p.error("--out is required unless --prepare-only")
    wanted = parse_pairs(args.pairs)
    # One column per side: the six multivariate pairs (52 to 55, 71, 105) are left out.
    pairs = [q for q in collection if not q.multivariate and int(q.name[4:]) in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    records = args.out / "records.jsonl"
    done = done_units(records)
    adapters = {name: make_solution(name) for name in args.solutions}
    versions = preflight(adapters)
    # The selected pairs, so the summary counts a pair no fit reached as unfinished.
    manifest = args.out / "pairs.json"
    expected = json.loads(manifest.read_text()) if manifest.is_file() else {}
    expected.update({q.name: q.weight for q in pairs})
    manifest.write_text(json.dumps(dict(sorted(expected.items())), indent=1) + "\n")
    # Largest pairs first, so a worker is not left with the slow fits at the end.
    tasks = [
        (q, name)
        for q in sorted(pairs, key=lambda q: -len(q.data))
        for name in adapters
        if (q.name, adapters[name].name) not in done
    ]
    lock = threading.Lock()
    host = machine()
    print(f"{len(pairs)} pairs, {len(tasks)} fits to run, {args.workers} at once", flush=True)

    def run(task: tuple[Any, str]) -> None:
        pair, name = task
        adapter = adapters[name]
        x = np.asarray(pair.data, dtype=np.float64)
        [measure] = run_task(
            adapter,
            x,
            adapter.params(),
            cap_wall_s=args.cap_wall_s,
            cap_mem_mb=args.cap_mem_mb,
            repeats=1,
            warmup=0,
            threads=1,
        )
        truth = "x->y" if pair.cause == ("x",) else "y->x"
        record: dict[str, Any] = {
            "pair": pair.name,
            "n": int(x.shape[0]),
            "weight": pair.weight,
            "solution": adapter.name,
            "method": name,
            "package": adapter.package,
            "package_version": versions[name],
            "params": dict(adapter.params()),
            "status": measure.get("status"),
            "wall_s": measure.get("wall_s"),
            "cpu_s": measure.get("cpu_s"),
            "peak_rss_mb": measure.get("peak_rss_mb"),
            "cap_wall_s": measure.get("cap_wall_s"),
            "cap_mem_mb": measure.get("cap_mem_mb"),
            "mem_route": measure.get("mem_route"),
            "truth": truth,
            "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **host,
        }
        if measure.get("status") != "ok":
            # What the fit said, so a failure is explained and not only counted.
            text = measure.get("error_text") or measure.get("stderr") or ""
            record["error"] = str(text).strip().splitlines()[-1][:300] if text else ""
        if measure.get("status") == "ok" and measure.get("adj") is not None:
            found = decision(adapter, measure["adj"])
            record["decision"] = found
            record["correct"] = found == truth
        with lock:
            with records.open("a") as sink:
                sink.write(json.dumps(record, default=str) + "\n")
            print(pair.name, name, record["status"], record.get("decision"), flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(run, tasks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
