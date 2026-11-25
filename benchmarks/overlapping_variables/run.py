"""Run the benchmark: list the fits of a size, run each in its own process, record each fit.

    python -m overlapping_variables.run records.jsonl --size small
    python -m overlapping_variables.run records.jsonl --size card --seeds 0-19 --procs 8

One JSON line per finished fit goes to the records file, keyed by :func:`fit.key`. A restart skips
the keys already in the file. A fit over its cap is killed and recorded as a timeout, never as an
empty graph. The run uses one thread per fit, so ``--procs`` fits run side by side.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ANDREY_N_JOBS"):
    os.environ[_name] = "1"

from . import fit, generate  # noqa: E402

HAND_CASES = ("H1", "H1c", "H4", "H5k3")
HAND_SIZES = (1000, 10000)
UNCAPPED = ("S3_b0s0", "S3_b0s10", "M8_b0s10", "L6_b0s10")
SIZES = {
    "small": {"configs": ("S3",), "hand": ("H1", "H4"), "sizes": (1000,), "seeds": (0, 1)},
    "card": {
        "configs": tuple(generate.CONFIGS),
        "hand": HAND_CASES,
        "sizes": HAND_SIZES,
        "seeds": tuple(range(20)),
    },
}


def _spec(design, cell, arm, method, seed, depth=3) -> dict:
    return {
        "design": design,
        "cell": cell,
        "arm": arm,
        "method": method,
        "seed": seed,
        "depth": depth,
    }


def parse_seeds(text: str) -> tuple[int, ...]:
    """Seeds from ``0-19``, ``1000-1019``, or ``0,3,7``."""
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += range(int(lo), int(hi or lo) + 1)
    return tuple(seeds)


def plan(
    size: str = "small",
    seeds: tuple[int, ...] | None = None,
    configs: tuple[str, ...] | None = None,
    designs: tuple[str, ...] = ("hand", "cohort", "lagged"),
) -> list[dict]:
    """The fits of a size, as specs. ``small`` is the 20-variable configuration, two hand cases,
    and the lagged cells on 2 seeds; ``card`` is every configuration, hand case, and lagged cell of
    the 0.19.0 release card, on seeds 0-19. ``seeds``, ``configs``, and ``designs`` narrow or move
    the size."""
    s = SIZES[size]
    seeds = tuple(seeds) if seeds is not None else s["seeds"]
    configs = tuple(configs) if configs is not None else s["configs"]
    out: list[dict] = []
    for case in s["hand"] if "hand" in designs else ():
        for n in s["sizes"]:
            for seed in seeds:
                for method in ("pc", "fci"):
                    cell = f"{case}_n{n}"
                    out += [
                        _spec("hand", cell, arm, method, seed, d) for arm in "ab" for d in (3, 0)
                    ]
                    out.append(_spec("hand", cell, "c", method, seed))
    for method in ("pc", "fci") if "hand" in designs else ():  # the fixed fork is one dataset
        out += [_spec("hand", "H1_fixed", arm, method, 0, d) for arm in "ab" for d in (3, 0)]
        out.append(_spec("hand", "H1_fixed", "c", method, 0))
    for config in configs if "cohort" in designs else ():
        for seed in seeds:
            for method in ("pc", "fci"):
                for variant in generate.VARIANTS:
                    cell = f"{config}_{variant}"
                    out += [
                        _spec("cohort", cell, arm, method, seed) for arm in ("a", "amb", "b", "c")
                    ]
                    if variant.startswith("b0"):
                        out.append(_spec("cohort", cell, "dcover", method, seed))
                out.append(_spec("cohort", f"{config}_complete", "f", method, seed))
                out.append(_spec("cohort", f"{config}_complete", "amb", method, seed))
            for cell in UNCAPPED:  # the search without a depth limit runs on the first seeds only
                if cell.startswith(config) and seed < 1000:
                    out.append(_spec("cohort", cell, "a", "pc", seed, None))
    for seed in seeds if "lagged" in designs else ():
        for method in ("pcmci", "pcmciplus"):
            for variant in ("nolap0", "nolap5"):
                out += [_spec("lagged", variant, arm, method, seed, None) for arm in "ab"]
            out.append(_spec("lagged", "complete", "f", method, seed, None))
    return out


def _child(spec: dict, conn) -> None:
    try:
        rec = fit.fit(spec)
    except Exception as error:  # an error is a result: record it
        rec = {
            "key": fit.key(spec),
            **spec,
            "status": "error",
            "error": f"{type(error).__name__}: {error}"[:600],
            "trace": traceback.format_exc()[-1500:],
        }
    conn.send(rec)
    conn.close()


def done_keys(path: str) -> set[str]:
    """The keys of the fits already recorded in ``path``."""
    keys: set[str] = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return keys


def run(specs: list[dict], records: str, procs: int = 1, cap: float = 1800.0) -> int:
    """Run the fits not yet in ``records``, ``procs`` at a time; return how many ran."""
    if procs < 1:
        raise ValueError(f"procs must be at least 1, not {procs}")
    done = done_keys(records)
    todo = [s for s in specs if fit.key(s) not in done]
    ctx = mp.get_context("fork")
    running: list = []
    ran = 0
    with open(records, "a") as out:
        while todo or running:
            while todo and len(running) < procs:
                spec = todo.pop(0)
                parent, kid = ctx.Pipe(duplex=False)
                p = ctx.Process(target=_child, args=(spec, kid))
                p.start()
                kid.close()
                running.append((p, parent, spec, time.time()))
            still = []
            for p, conn, spec, t0 in running:
                rec = None
                if conn.poll():
                    try:
                        rec = conn.recv()
                    except EOFError:
                        rec = {"key": fit.key(spec), **spec, "status": "error", "error": "died"}
                    p.join()
                elif not p.is_alive():
                    p.join()
                    err = f"exit {p.exitcode}"
                    rec = {"key": fit.key(spec), **spec, "status": "error", "error": err}
                elif time.time() - t0 > cap:
                    p.kill()
                    p.join()
                    rec = {"key": fit.key(spec), **spec, "status": "timeout", "cap": cap}
                if rec is None:
                    still.append((p, conn, spec, t0))
                else:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    ran += 1
            running = still
            time.sleep(0.02)
    return ran


def main() -> None:
    """Command line: ``records`` and the options of the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", help="the JSON lines file the fits are appended to")
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument(
        "--seeds", help="seeds, such as 0-19, 1000-1019, or 0,3,7 (default: the size's)"
    )
    ap.add_argument("--configs", help="configurations, such as S3,M8 (default: the size's)")
    ap.add_argument("--designs", default="hand,cohort,lagged", help="designs to run")
    ap.add_argument("--procs", type=int, default=1, help="fits run side by side")
    ap.add_argument("--cap", type=float, default=1800.0, help="seconds per fit before a timeout")
    args = ap.parse_args()
    specs = plan(
        args.size,
        parse_seeds(args.seeds) if args.seeds else None,
        tuple(args.configs.split(",")) if args.configs else None,
        tuple(args.designs.split(",")),
    )
    n = run(specs, args.records, args.procs, args.cap)
    print(f"{n} of {len(specs)} fits ran; the records are in {args.records}")


if __name__ == "__main__":
    main()
