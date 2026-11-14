#!/usr/bin/env python
"""Judge ``local_isa_ling`` against the authors' code per cell, from the records of
``local_lingam_cpu.py``.

A cell is one setting (``cyclic``, ``acyclic-d3``, ``acyclic-d5``) at one sample size, with one
blanket: the one each method learns (``estimated``: Andrey's facade against the authors' code with
its own Lasso) or the true one (``true``: both given the true blanket). Each dataset has one
target, and its measures are the record's scores (:func:`~andrey_bench.local_ling_models.local_scores`):

- ``shd``: the paper's SHD of the local graph (the edges into the target and into its children);
- ``weight_dist``: the Euclidean distance between the estimated and the true local weights.

A dataset counts for a solution only when its fit finished; the authors' Algorithm 1 declining a
blanket past its size (``not_run``) and a timeout both leave the dataset out, and ``coverage``
counts them. :func:`judge` gives the mean of the per-dataset difference, Andrey minus the other
solution, with its percentile bootstrap 95% interval (:func:`~andrey_bench.pairing.compare`). A
cell is worse on a measure when the interval lies wholly above 0 on the first seeds and again on
the fresh seeds; a cell worse on the first seeds only is a tie. The floor is each measure minus
the empty answer's, which a solution passes when the interval lies wholly below 0.

Global LiNGAM read at the target is a different method: :func:`frontier` sets each solution's
mean SHD and weight distance against its median seconds per target, and marks the solutions no
other one beats on both accuracy and cost at once. Sachs has one dataset and no weights, so its
three targets are listed with their SHD (:func:`sachs`).

    $ANDREY_BENCH_PYTHON benchmarks/local_lingam_summary.py RUN [RUN ...] --json cells.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from andrey_bench.pairing import MIN_PAIRS, compare

#: The judged pairs, by blanket: Andrey's solution and the authors' code's.
PAIRS = {
    "estimated": ("andrey.local_isa_ling", "local_isa_ling.authors"),
    "true": ("andrey.local_isa_ling.oracle", "local_isa_ling.authors.oracle"),
}
EMPTY = "empty"

#: Solutions set on the accuracy-cost frontier of the estimated-blanket cells.
FRONTIER = (
    "andrey.local_isa_ling",
    "local_isa_ling.authors",
    "andrey.direct_lingam",
    "andrey.ica_lingam",
    EMPTY,
)

#: The measures per dataset; both are errors (fewer is better).
MEASURES = ("shd", "weight_dist")

#: The release summary's metric of each measure.
SUMMARY_METRICS: dict[str, str] = {"shd": "shd", "weight_dist": "weight_distance"}

#: The settings in the order rows are listed, with their expected degree.
CELLS = {"cyclic": None, "acyclic-d3": 3, "acyclic-d5": 5}


def load(runs: list[str]) -> dict[str, dict[tuple[str, int, int], dict]]:
    """``{solution: {(cell, n, seed): record}}`` over the run folders (one target per dataset)."""
    out: dict = defaultdict(dict)
    for run in runs:
        for line in (Path(run) / "records.jsonl").read_text().splitlines():
            r = json.loads(line)
            out[r["solution"]][(r["cell"], r["n"], r["seed"])] = r
    return out


def dataset(record: dict | None, empty: dict | None) -> dict | None:
    """One solution's measures on one dataset beside the empty answer's; ``None`` unless finished."""
    if record is None or record.get("status") != "ok" or not record.get("scores"):
        return None
    s, e = record["scores"], empty["scores"]
    return {
        **{m: s[m] for m in MEASURES},
        **{f"empty_{m}": e[m] for m in MEASURES},
        "wall_s": record.get("wall_s"),
        "m": record.get("m"),
        "fallback": record.get("fallback"),
        "blanket_precision": record.get("blanket_precision"),
        "blanket_recall": record.get("blanket_recall"),
    }


def judge(a: dict[int, dict], b: dict[int, dict], seeds: range) -> dict:
    """The paired verdicts of ``a`` against ``b`` on the datasets of ``seeds`` both finished."""
    paired = sorted(s for s in seeds if a.get(s) and b.get(s))
    out: dict = {"paired": len(paired)}
    for name in MEASURES:
        va, vb = [a[s][name] for s in paired], [b[s][name] for s in paired]
        c = compare(va, vb) if paired else {"verdict": "tie"}
        if len(paired) < MIN_PAIRS:
            c["verdict"] = "too few pairs"
        c["andrey_mean"] = round(float(np.mean(va)), 4) if paired else None
        c["other_mean"] = round(float(np.mean(vb)), 4) if paired else None
        c["D_by_seed"] = {str(s): round(x - y, 4) for s, x, y in zip(paired, va, vb, strict=True)}
        out[name] = c
    return out


def floor(fits: list[dict], measure: str) -> dict:
    """``measure`` of each dataset minus the empty answer's, with its bootstrap interval."""
    c = compare([f[measure] for f in fits], [f[f"empty_{measure}"] for f in fits])
    return {
        "paired": c["paired"],
        "mean": c["mean_D"],
        "bootstrap_95": c["bootstrap_95"],
        "pass": c["paired"] >= MIN_PAIRS and c["bootstrap_95"][1] < 0,
    }


def _median(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


def by_seed(runs: dict, name: str, cell: str, n: int) -> dict[int, dict]:
    """``{seed: dataset(...)}`` of one solution in one cell (finished fits only)."""
    empty = runs.get(EMPTY, {})
    return {
        s: dataset(r, empty.get((c, m, s)))
        for (c, m, s), r in runs.get(name, {}).items()
        if (c, m) == (cell, n)
    }


def cell(runs: dict, name: str, n: int, blanket: str, first: range, fresh: range) -> dict:
    """The judged row of one cell and blanket: Andrey against the authors' code."""
    andrey, authors = PAIRS[blanket]
    mine, theirs = by_seed(runs, andrey, name, n), by_seed(runs, authors, name, n)
    row: dict = {"cell": name, "mean_degree": CELLS[name], "d": 50, "n": n, "blanket": blanket}
    tried_a = {s for s in first if (name, n, s) in runs.get(andrey, {})}
    tried_b = {s for s in first if (name, n, s) in runs.get(authors, {})}
    ok_a = {s for s in tried_a if mine.get(s)}
    ok_b = {s for s in tried_b if theirs.get(s)}
    status = [runs[authors][(name, n, s)]["status"] for s in tried_b]
    row["coverage"] = {
        "datasets": len(tried_a | tried_b),
        "andrey_short": len(ok_b - ok_a),
        "reference_short": len(ok_a - ok_b),
        "reference_not_run": status.count("not_run"),
        "reference_timeout": status.count("timeout"),
    }
    row["first"] = judge(mine, theirs, first)
    fresh_judged = judge(mine, theirs, fresh)
    row["fresh"] = fresh_judged if fresh_judged["paired"] else None
    both = sorted(ok_a & ok_b)
    row["floor"] = {
        m: {
            "andrey_minus_empty": floor([mine[s] for s in sorted(ok_a)], m),
            "other_minus_empty": floor([theirs[s] for s in both], m),
        }
        for m in MEASURES
    }
    for m in MEASURES:
        verdict = row["first"][m]["verdict"]
        if verdict == "worse":
            again = row["fresh"][m]["verdict"] if row["fresh"] else None
            verdict = {"worse": "worse", None: "worse, not rerun"}.get(again, "tie")
        row[f"{m}_result"] = verdict
    results = [row[f"{m}_result"] for m in MEASURES]
    row["result"] = next(
        (r for r in ("worse", "worse, not rerun", "better", "too few pairs") if r in results), "tie"
    )
    for side, fits in (("andrey", [mine[s] for s in both]), ("other", [theirs[s] for s in both])):
        row[f"{side}_s"] = _median([f["wall_s"] for f in fits])
        row[f"{side}_m"] = _mean([f["m"] for f in fits])
    row["andrey_fallback"] = sum(bool(mine[s]["fallback"]) for s in ok_a)
    row["andrey_blanket"] = {
        k: _mean([mine[s][k] for s in ok_a]) for k in ("blanket_precision", "blanket_recall")
    }
    row["empty"] = {m: _mean([mine[s][f"empty_{m}"] for s in ok_a]) for m in MEASURES}
    return row


def frontier(runs: dict, name: str, n: int, seeds: range) -> dict:
    """Mean accuracy against median seconds per target of :data:`FRONTIER`, on the datasets every
    one of them finished; ``pareto`` marks the solutions no other one beats on both at once."""
    fits = {s: by_seed(runs, s, name, n) for s in FRONTIER if runs.get(s)}
    common = [x for x in seeds if all(f.get(x) for f in fits.values())]
    points = {}
    for sol, f in fits.items():
        wall = 0.0 if sol == EMPTY else _median([f[x]["wall_s"] for x in common])
        points[sol] = {
            "shd": _mean([f[x]["shd"] for x in common]),
            "weight_dist": _mean([f[x]["weight_dist"] for x in common]),
            "seconds": wall,
        }
    for m in MEASURES:
        for sol, p in points.items():
            p[f"pareto_{m}"] = not any(
                q[m] <= p[m]
                and q["seconds"] <= p["seconds"]
                and (q[m], q["seconds"]) != (p[m], p["seconds"])
                for o, q in points.items()
                if o != sol and None not in (q[m], p[m], q["seconds"], p["seconds"])
            )
    return {"cell": name, "n": n, "datasets": len(common), "points": points}


def sachs(runs: dict) -> list[dict]:
    """Per Sachs target and solution, the SHD of the local graph (Table 1 of the paper: 1 each)."""
    return [
        {
            "solution": name,
            "target": r["target"],
            "status": r["status"],
            "shd": (r.get("scores") or {}).get("shd"),
        }
        for name, recs in sorted(runs.items())
        for (c, _, _), r in recs.items()
        if c == "sachs"
    ]


def summarize(runs: dict, first: range, fresh: range) -> tuple[list[dict], list[dict]]:
    """Every judged row and every frontier, in the cells' order."""
    sizes = sorted(
        {(c, n) for recs in runs.values() for (c, n, _) in recs if c in CELLS},
        key=lambda k: (list(CELLS).index(k[0]), k[1]),
    )
    rows = [
        cell(runs, c, n, b, first, fresh) for c, n in sizes for b in PAIRS if runs.get(PAIRS[b][0])
    ]
    fronts = [frontier(runs, c, n, first) for c, n in sizes if c != "cyclic"]
    return rows, fronts


def _summary_side(judged: dict, measure: str) -> dict:
    """One measure of :func:`judge` as the release summary's error fields."""
    metric = SUMMARY_METRICS[measure]
    m = judged[measure]
    return {
        "paired": judged["paired"],
        f"andrey_{metric}_mean": m["andrey_mean"],
        f"comparator_{metric}_mean": m["other_mean"],
        "mean_D": m["mean_D"],
        "bootstrap_95": m["bootstrap_95"],
        "above_zero": m["bootstrap_95"][0] > 0,
    }


def summary_rows(rows: list[dict], measure: str, blanket: str, fresh: str) -> list[dict]:
    """The rows of one blanket as rows of the release summary for one measure."""
    out = []
    for r in rows:
        if r["blanket"] != blanket:
            continue
        fl = r["floor"][measure]
        applies = fl["other_minus_empty"]["bootstrap_95"][1] < 0
        row = {
            "graph": r["cell"],
            **({"mean_degree": r["mean_degree"]} if r["mean_degree"] is not None else {}),
            "d": r["d"],
            "n": r["n"],
            "seeds": r["coverage"]["datasets"],
            **_summary_side(r["first"], measure),
            "verdict": r["first"][measure]["verdict"],
            "coverage": r["coverage"],
            "floor": {
                "applies": applies,
                "pass": fl["andrey_minus_empty"]["bootstrap_95"][1] < 0 if applies else None,
                "andrey_minus_empty": {
                    k: fl["andrey_minus_empty"][k] for k in ("paired", "mean", "bootstrap_95")
                },
                "reference_minus_empty": {
                    k: fl["other_minus_empty"][k] for k in ("paired", "mean", "bootstrap_95")
                },
            },
            "andrey_s": r["andrey_s"],
            "comparator_s": r["other_s"],
            "result": r[f"{measure}_result"],
        }
        if r["fresh"] and r["first"][measure]["verdict"] == "worse":
            row["replication"] = {"seeds": fresh, **_summary_side(r["fresh"], measure)}
        out.append(row)
    return out


def _seeds(text: str) -> range:
    lo, hi = (int(x) for x in text.split("-"))
    return range(lo, hi + 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Local LiNG lane: local_isa_ling judged per cell")
    p.add_argument("runs", nargs="+", help="run folders of local_lingam_cpu.py")
    p.add_argument("--first", default="0-19", help="seeds judged first (default: 0-19)")
    p.add_argument("--fresh", default="20-39", help="seeds of the reruns (default: 20-39)")
    p.add_argument("--json", default=None, help="also write the rows, frontiers, and Sachs here")
    args = p.parse_args(argv)
    runs = load(args.runs)
    rows, fronts = summarize(runs, _seeds(args.first), _seeds(args.fresh))
    print(
        f"{'cell':11s} {'n':>6s} {'blanket':9s} {'pairs':>5s} {'SHD A/B':>13s} "
        f"{'dist A/B':>15s} {'result':>8s} {'floor':>11s} {'cover':>6s} {'s A/B':>13s}"
    )
    for r in rows:
        s, d = r["first"]["shd"], r["first"]["weight_dist"]
        fl = "/".join(str(r["floor"][m]["andrey_minus_empty"]["pass"])[0] for m in MEASURES)
        print(
            f"{r['cell']:11s} {r['n']:6d} {r['blanket']:9s} {r['first']['paired']:5d} "
            f"{s['andrey_mean']!s:>6s}/{s['other_mean']!s:<6s} "
            f"{d['andrey_mean']!s:>7s}/{d['other_mean']!s:<7s} {r['result']:>8s} {fl:>11s} "
            f"{r['coverage']['datasets'] - r['coverage']['reference_short']:>3d}/{r['coverage']['datasets']:<2d} "
            f"{r['andrey_s']!s:>6s}/{r['other_s']!s:<6s}"
        )
    for f in fronts:
        pts = "  ".join(
            f"{k.split('.')[-1]}={v['shd']}/{v['weight_dist']}@{v['seconds']}s"
            f"{'*' if v['pareto_shd'] else ''}"
            for k, v in f["points"].items()
        )
        print(f"frontier {f['cell']} n={f['n']} ({f['datasets']}): {pts}")
    for r in sachs(runs):
        print(f"sachs {r}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"rows": rows, "frontiers": fronts, "sachs": sachs(runs)}, indent=1)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
