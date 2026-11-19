#!/usr/bin/env python
"""Run the pairwise methods and the authors' code of QCCD, HECI, and GRCI on the Tuebingen pairs.

``andrey.data.load_dataset("tuebingen")`` gives the cause-effect pairs of Mooij et al. (2016). The
lane runs every solution (:data:`SOLUTIONS`; ``--solutions`` picks some) on each pair that has one
column per side, in a forced decision: ``x -> y``, ``y -> x``, or no decision. Andrey's ``anm``,
``pnl``, ``loci`` (both criteria), and ``resit`` are the Andrey arms; ``qccd`` (the paper's
``m = 3``), ``qccd_m7`` (the wrapper's default ``m = 7``), ``heci``, and ``grci`` are the authors'
code, each unmodified and read from the directory an environment variable names (each
``benchmarks/start_<name>.sh`` downloads it and prints the export). Nothing of the authors' code is
copied into this repository, and GRCI's license (CC BY-NC 4.0) limits it to benchmark runs from a
local download.

Each fit runs in its own process with one numeric thread under a per-fit wall cap (30 minutes by
default); a fit past the cap is recorded as ``timeout`` and has no decision. ``--workers`` fits run
at once, so a job packs many pairs: give it that many cores. One JSON line per fit goes to
``<out>/records.jsonl``, with the machine and the fit interpreter's runtime, so the summary can
state the hardware. A rerun skips the (pair, solution) units already there, and stops before any fit
when the records there were made with another build, parameters, caps, or runtime.
``tuebingen_summary.py`` turns the records of one or more runs into the table.

``resit`` runs its fits in the interpreter ``$ANDREY_BENCH_RESIT_PYTHON`` names, the build under
test; the example sets it to the bench environment's, as the other Andrey arms use.

    export ANDREY_BENCH_PYTHON=... ANDREY_BENCH_RESIT_PYTHON=$ANDREY_BENCH_PYTHON
    $ANDREY_BENCH_PYTHON benchmarks/tuebingen_pairs.py --prepare-only
    $ANDREY_BENCH_PYTHON benchmarks/tuebingen_pairs.py --out outputs/tue-a --pairs 1-20 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tuebingen_summary import identity

from andrey_bench.adapters.andrey_anm import AndreyANM
from andrey_bench.adapters.andrey_loci import AndreyLOCI, AndreyLOCILikelihood
from andrey_bench.adapters.andrey_pnl import AndreyPNL
from andrey_bench.adapters.andrey_resit import AndreyRESIT
from andrey_bench.adapters.r_adapter import RAdapter
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


#: Prints, as JSON, the Python of the interpreter that runs a solution's fits and the version of
#: each numeric package the fits may import (``null`` where it is not installed).
_PYTHON_RUNTIME = """
import json, platform
from importlib import metadata

def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

names = ("numpy", "scipy", "scikit-learn")
print(json.dumps({"python": platform.python_version(), **{n: version(n) for n in names}}))
"""

#: Prints, as JSON, the R and rpy2 that an R solution's fit interpreter embeds.
_R_RUNTIME = (
    "import json, importlib.metadata as m, rpy2.robjects as ro; "
    "print(json.dumps({'R': str(ro.r('R.version.string')[0]), 'rpy2': m.version('rpy2')}))"
)

#: Prints, as JSON, the Andrey an Andrey arm's fit interpreter imports: its version and the SHA-256
#: of its source files, which tells apart two builds that carry one version string.
_ANDREY_BUILD = """
import hashlib, json
from pathlib import Path
import andrey

root = Path(andrey.__file__).parent
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    digest.update(str(path.relative_to(root)).encode() + b"\\0" + path.read_bytes())
print(json.dumps({"version": andrey.__version__, "andrey_source_sha256": digest.hexdigest()}))
"""


def preflight(
    adapters: dict[str, Any], cap_wall_s: float, cap_mem_mb: float | None
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Check every fit interpreter, every reference's code folder, and one small fit per solution
    before the campaign, so a setup problem stops the run once instead of failing every unit.

    The small fits run under the campaign's caps. Returns ``{solution: version}``, the version of
    the package each solution's fits run, Andrey's read from the interpreter that runs them, and
    ``{solution: runtime}``, the Python, NumPy, SciPy, and scikit-learn of that interpreter and, for
    an Andrey arm, the digest of the Andrey source it imports, or, for an R solution, the R and rpy2
    it embeds.
    """
    from importlib import import_module

    from andrey_bench.adapters.fcm_warm import tiny_table

    interpreters = check_interpreters(list(adapters.values()))
    for name in adapters:
        if name in REFERENCES:
            import_module(REFERENCES[name][0]).code_dir()

    def ask(name: str, code: str) -> str:
        """The last line ``code`` prints in solution ``name``'s fit interpreter."""
        command = [str(interpreters[adapters[name].name]), "-c", code]
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode != 0:
            lines = run.stderr.strip().splitlines()
            raise SystemExit(
                f"{name}: {command[0]} fails: {lines[-1] if lines else run.returncode}"
            )
        return run.stdout.strip().splitlines()[-1]

    versions = {}
    runtimes = {}
    for name, adapter in adapters.items():
        runtimes[name] = json.loads(ask(name, _PYTHON_RUNTIME))
        versions[name] = adapter.package_version
        if adapter.package == "andrey":
            build = json.loads(ask(name, _ANDREY_BUILD))
            versions[name] = build.pop("version")
            runtimes[name] |= build
        if isinstance(adapter, RAdapter):
            runtimes[name] |= json.loads(ask(name, _R_RUNTIME))
    # 250 rows: enough for QCCD's seven quantile levels at the wrapper's default.
    pair = tiny_table(n=250)[:, :2]
    for name, adapter in adapters.items():
        [measure] = run_task(
            adapter,
            pair,
            adapter.params(),
            cap_wall_s=cap_wall_s,
            cap_mem_mb=cap_mem_mb,
            repeats=1,
            warmup=0,
            threads=1,
        )
        if measure.get("status") != "ok":
            text = str(measure.get("error_text") or measure.get("stderr") or "").strip()
            last = text.splitlines()[-1] if text else measure.get("status")
            raise SystemExit(f"{name} cannot fit a small pair here: {last}")
    return versions, runtimes


def check_resumption(path: Path, current: dict[str, dict]) -> None:
    """Refuse to resume ``path`` when a solution's records there differ from this run's identity.

    ``current`` holds, per solution, the fields of a record that :func:`tuebingen_summary.identity`
    reads. A mismatch stops the run before any fit, instead of leaving a mix the summary refuses.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        record = json.loads(line)
        expected = current.get(record["method"])
        if expected is not None and identity(record) != identity(expected):
            raise SystemExit(
                f"{path} holds {record['method']} records with another build, parameters, caps, or "
                "runtime: write this run to a new directory"
            )


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


def update_manifest(path: Path, pairs: dict[str, float], solutions: list[str]) -> None:
    """Add the run's pairs (``{pair: weight}``) and its ``(pair, solution)`` units to ``path``.

    The summary counts a requested unit that no fit reached as unfinished. The units are kept as
    asked, never as all pairs times all solutions, so a rerun into the same directory with other
    pairs or solutions requests only what it asked for.
    """
    manifest = json.loads(path.read_text()) if path.is_file() else {"pairs": {}, "units": []}
    units = {tuple(u) for u in manifest["units"]} | {(p, m) for p in pairs for m in solutions}
    manifest = {
        "pairs": dict(sorted({**manifest["pairs"], **pairs}.items())),
        "units": sorted(units),
    }
    path.write_text(json.dumps(manifest, indent=1) + "\n")


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
    if not pairs:
        p.error(f"--pairs {args.pairs} names no pair with one column per side")
    adapters = {name: make_solution(name) for name in args.solutions}
    versions, runtimes = preflight(adapters, args.cap_wall_s, args.cap_mem_mb)
    params = {name: json.loads(json.dumps(a.params(), default=str)) for name, a in adapters.items()}
    records = args.out / "records.jsonl"
    check_resumption(
        records,
        {
            name: {
                "package_version": versions[name],
                "params": params[name],
                "cap_wall_s": args.cap_wall_s,
                "cap_mem_mb": args.cap_mem_mb,
                "runtime": runtimes[name],
            }
            for name in adapters
        },
    )
    args.out.mkdir(parents=True, exist_ok=True)
    done = done_units(records)
    update_manifest(args.out / "manifest.json", {q.name: q.weight for q in pairs}, args.solutions)
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
            "runtime": runtimes[name],
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

    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    try:
        # As each fit completes, so the first failure stops the queue at once.
        for future in as_completed([pool.submit(run, task) for task in tasks]):
            future.result()
    except BaseException as exc:
        # An interrupt or a failed unit drops the queued fits instead of running them all first.
        pool.shutdown(wait=False, cancel_futures=True)
        # A running fit holds its worker thread until it ends or reaches the cap, and the
        # interpreter joins those threads on exit. Leaving at once closes each fit's death pipe
        # instead, so its supervisor kills the fit's process group; holding the lock lets a record
        # being written finish first.
        lock.acquire()
        if not isinstance(exc, KeyboardInterrupt):
            traceback.print_exception(exc)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130 if isinstance(exc, KeyboardInterrupt) else 1)
    pool.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
