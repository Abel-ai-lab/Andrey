#!/usr/bin/env python
"""Summarize ``tuebingen_pairs.py`` runs: weighted accuracy with an interval, coverage, and time.

Reads ``records.jsonl`` and, where the driver wrote it, ``manifest.json`` (the pairs and the
(pair, solution) units the run was asked for) from each run directory, and writes ``summary.md``
and ``summary.json`` to ``--out``. For each solution it reports:

- **coverage**: the pairs whose fit finished (``ok``), as a count and as the share of the pairs'
  total weight. A timeout, a memory kill, an error, or a requested pair with no record is not an
  answer: it is not scored as a guess and not counted in the accuracy, and the column says how much
  of the set the accuracy covers.
- **accuracy**: the weighted share of finished pairs whose decision names the true direction, with
  the collection's weights; a finished fit with no edge counts as wrong, and ``undecided`` counts
  them. The interval is a percentile bootstrap over pairs: the same resamples of the pair set for
  every solution, each pair keeping its weight, each solution scored on the resampled pairs it
  finished.
- **common set**: the same accuracy over the pairs every solution finished, which is the table to
  compare solutions on, with a paired interval for each solution's difference from the first.
- **time**: the median and the total wall seconds of the timed fit over the pairs it finished.

    python benchmarks/tuebingen_summary.py --out outputs/tue-summary outputs/tue-a outputs/tue-b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

#: The rows of the table, Andrey's methods first; the first is the base of the paired differences.
METHOD_ORDER = ("anm", "pnl", "loci", "loci_likelihood", "resit", "qccd", "qccd_m7", "heci", "grci")
RESAMPLES = 10_000
LEVEL = 0.95


def read_manifest(runs: list[Path]) -> tuple[dict[str, float], set[tuple[str, str]]]:
    """The pairs (``{pair: weight}``) and the ``(pair, solution)`` units the runs were asked for,
    from each run's ``manifest.json``, each unit as the driver recorded it."""
    pairs: dict[str, float] = {}
    units: set[tuple[str, str]] = set()
    for run in runs:
        path = run / "manifest.json"
        if path.is_file():
            manifest = json.loads(path.read_text())
            pairs.update(manifest["pairs"])
            units.update((p, m) for p, m in manifest["units"])
    return pairs, units


def read_records(runs: list[Path]) -> list[dict]:
    """Every fit record of the run directories, refusing runs that cannot be pooled.

    A solution's records must agree on its build, parameters, and caps, and a (pair, solution)
    unit may appear once, so a row never mixes builds or budgets.
    """
    units: dict[tuple[str, str], dict] = {}
    identity: dict[str, tuple] = {}
    for run in runs:
        path = run / "records.jsonl"
        if not path.is_file() and not (run / "manifest.json").is_file():
            raise SystemExit(f"{run} holds neither records.jsonl nor manifest.json")
        # A run stopped before its first fit finished has a manifest and no records yet.
        for line in path.read_text().splitlines() if path.is_file() else []:
            record = json.loads(line)
            unit = (record["pair"], record["method"])
            if unit in units:
                raise SystemExit(f"{unit[1]} on {unit[0]} appears twice; summarize one record each")
            units[unit] = record
            key = (
                record.get("package_version"),
                json.dumps(record.get("params"), sort_keys=True),
                record.get("cap_wall_s"),
                record.get("cap_mem_mb"),
            )
            if identity.setdefault(unit[1], key) != key:
                raise SystemExit(f"{unit[1]}: records differ in build, parameters, or caps")
    return list(units.values())


def weighted(correct: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted share of correct decisions; NaN when the weights sum to 0."""
    total = weights.sum()
    return float((weights * correct).sum() / total) if total > 0 else float("nan")


def interval(draws: np.ndarray) -> tuple[float, float]:
    """The percentile interval of the bootstrap draws, ignoring NaN draws."""
    lo, hi = np.nanpercentile(draws, [50 * (1 - LEVEL), 50 * (1 + LEVEL)])
    return float(lo), float(hi)


def summarize(
    records: list[dict],
    expected: dict[str, float] | None = None,
    units: set[tuple[str, str]] | None = None,
    seed: int = 0,
) -> dict:
    """The tables of the module docstring as a dict, from the fit records.

    ``expected`` (``{pair: weight}``) and ``units`` (``(pair, solution)``) add what the runs were
    asked for: a requested unit with no record counts as ``missing``, unfinished.
    """
    weights = dict(expected or {})
    weights.update({r["pair"]: r["weight"] for r in records})
    pairs = sorted(weights)
    weight = np.array([weights[p] for p in pairs])
    units = units or set()
    seen = {r["method"] for r in records} | {m for _, m in units}
    methods = [m for m in METHOD_ORDER if m in seen] + sorted(seen - set(METHOD_ORDER))
    index = {p: i for i, p in enumerate(pairs)}
    ok = {m: np.zeros(len(pairs), dtype=bool) for m in methods}
    right = {m: np.zeros(len(pairs)) for m in methods}
    wall = {m: np.full(len(pairs), np.nan) for m in methods}
    statuses: dict[str, dict[str, int]] = {m: {} for m in methods}
    undecided = {m: 0 for m in methods}
    for r in records:
        m, i = r["method"], index[r["pair"]]
        statuses[m][r["status"]] = statuses[m].get(r["status"], 0) + 1
        if r["status"] == "ok":
            ok[m][i] = True
            right[m][i] = float(bool(r.get("correct")))
            wall[m][i] = r["wall_s"]
            undecided[m] += r.get("decision") in ("none", "other")
    recorded = {(r["pair"], r["method"]) for r in records}
    for _, m in units - recorded:
        statuses[m]["missing"] = statuses[m].get("missing", 0) + 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pairs), size=(RESAMPLES, len(pairs)))
    common = np.logical_and.reduce([ok[m] for m in methods])

    def boot(m: str, mask: np.ndarray) -> np.ndarray:
        w = weight * mask
        num = (w[draws] * right[m][draws]).sum(axis=1)
        den = w[draws].sum(axis=1)
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

    out: dict = {
        "pairs": len(pairs),
        "total_weight": float(weight.sum()),
        "common_pairs": int(common.sum()),
        "common_weight": float(weight[common].sum()),
        "solutions": {},
    }
    base = methods[0]
    for m in methods:
        done = ok[m]
        acc = weighted(right[m][done], weight[done])
        lo, hi = interval(boot(m, done))
        c_acc = weighted(right[m][common], weight[common])
        c_lo, c_hi = interval(boot(m, common))
        diff = boot(m, common) - boot(base, common)
        d_lo, d_hi = interval(diff)
        out["solutions"][m] = {
            "finished": int(done.sum()),
            "finished_weight": float(weight[done].sum()),
            "statuses": statuses[m],
            "undecided": undecided[m],
            "accuracy": acc,
            "accuracy_ci": [lo, hi],
            "common_accuracy": c_acc,
            "common_accuracy_ci": [c_lo, c_hi],
            "common_difference_from_first": [
                weighted(right[m][common], weight[common])
                - weighted(right[base][common], weight[common]),
                d_lo,
                d_hi,
            ],
            "median_wall_s": float(np.nanmedian(wall[m])) if done.any() else None,
            "total_wall_s": float(np.nansum(wall[m])),
        }
    return out


def markdown(summary: dict, records: list[dict]) -> str:
    """Render the summary as the tables a pull request or a card shows."""
    machines = sorted({f"{r.get('cpu', '?')}" for r in records})
    first = next(iter(summary["solutions"]))
    lines = [
        f"Pairs: {summary['pairs']} two-column pairs, total weight {summary['total_weight']:.3f}; "
        f"{summary['common_pairs']} finished by every solution (weight {summary['common_weight']:.3f}).",
        f"Interval: {int(LEVEL * 100)}% percentile bootstrap over pairs, {RESAMPLES} resamples, "
        "weights carried, the same resamples for every solution.",
        "CPU: " + "; ".join(machines),
        "",
        "| solution | finished | weight finished | undecided | weighted accuracy [95% CI] | "
        "common set accuracy [95% CI] | common set minus "
        + first
        + " [95% CI] | median s | total s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, s in summary["solutions"].items():
        d = s["common_difference_from_first"]
        median = "-" if s["median_wall_s"] is None else f"{s['median_wall_s']:.2f}"
        lines.append(
            f"| {m} | {s['finished']}/{summary['pairs']} | {s['finished_weight']:.3f} | "
            f"{s['undecided']} | {s['accuracy']:.3f} [{s['accuracy_ci'][0]:.3f}, {s['accuracy_ci'][1]:.3f}] | "
            f"{s['common_accuracy']:.3f} [{s['common_accuracy_ci'][0]:.3f}, {s['common_accuracy_ci'][1]:.3f}] | "
            f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | {median} | {s['total_wall_s']:.0f} |"
        )
    other = {
        m: s["statuses"] for m, s in summary["solutions"].items() if set(s["statuses"]) - {"ok"}
    }
    if other:
        lines += [
            "",
            "Fits that did not finish: " + "; ".join(f"{m} {st}" for m, st in other.items()),
        ]
    return "\n".join(lines) + "\n"


def without_nan(value: Any) -> Any:
    """``value`` with every NaN replaced by ``None``, so ``summary.json`` is strict JSON: a value no
    pair supports (no finished pair, or no pair every solution finished) is ``null``."""
    if isinstance(value, dict):
        return {k: without_nan(v) for k, v in value.items()}
    if isinstance(value, list):
        return [without_nan(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize Tuebingen pair runs")
    p.add_argument("runs", nargs="+", type=Path, help="run directories holding records.jsonl")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    records = read_records(args.runs)
    summary = summarize(records, *read_manifest(args.runs))
    args.out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(without_nan(summary), indent=2, allow_nan=False)
    (args.out / "summary.json").write_text(text + "\n")
    table = markdown(summary, records)
    (args.out / "summary.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
