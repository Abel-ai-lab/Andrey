"""Summarize the records into the figures of the 0.19.0 release card, and print its tables.

    python -m overlapping_variables.summary records.jsonl [more.jsonl ...] [--json figures.json]

A mean is over the seeds below 1000 (the fresh seeds 1000-1019 count in the check that no listed
pair is drawn), with a 95% percentile bootstrap interval of 10,000 resamples from a generator
seeded 0. The resamples index the seeds in ascending order, so the same records give the same
figures whatever order their fits finished in. The one exception is the card's first table: its
intervals for the medium and large configurations were drawn with the seeds in the order the fits
of that run finished, which ``CARD_ORDER`` records, so that the two cells print the card's ends.
Arms compare on the seeds where both finished, a difference per seed. A figure whose cells are not
in the records is left out, so a small run gives the figures of the cells it ran.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json

import numpy as np

from . import score as scoring

B = 10_000
#: What every fit of the benchmark shares.
SETTINGS = {
    "indep_test": "fisherz",
    "alpha": 0.05,
    "alpha_pcmciplus": 0.01,
    "max_depth": 3,
    "tau_max": 2,
    "seeds": "0-19, each seed a new graph, new weights, and new data",
    "interval": "95% percentile bootstrap of the mean over seeds, 10,000 resamples, generator seed 0",
    "interval_order": "the resamples index the seeds in ascending order, except the two cells of CARD_ORDER",
}
CONFIGS = {"S3": "small", "M8": "medium", "L6": "large"}
#: The seeds in the order the fits of the medium and large cells finished in the run behind the
#: card's first table, which the intervals of its "Andrey 0.19.0" column were drawn with. The key
#: is ``(design, cell, method, depth, arm)``, as in :class:`Scored`.
CARD_ORDER = {
    ("cohort", "M8_b0s10", "pc", 3, "a"): (0, 2, 4, 6, 10, 12, 8, 14, 16, 18)
    + (3, 1, 5, 7, 9, 11, 13, 15, 17, 19),
    ("cohort", "L6_b0s10", "pc", 3, "a"): (2, 0, 4, 10, 8, 12, 6, 14, 16, 18)
    + (3, 11, 13, 9, 7, 17, 15, 5, 1, 19),
}
ARMS = ("a", "amb", "b", "c", "dcover", "f")


def boot(values) -> tuple[float, float, float] | None:
    """The mean of ``values`` and its 95% percentile bootstrap interval; ``None`` if empty."""
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None
    if v.size == 1 or np.all(v == v[0]):
        return float(v.mean()), float(v.mean()), float(v.mean())
    idx = np.random.default_rng(0).integers(0, v.size, (B, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def verdict(diffs, margin: float, higher_better: bool) -> str:
    """``better``, ``not worse``, ``worse``, or ``not shown`` for per-seed differences ``a - z``
    against a margin: the interval of their mean lies beyond the margin on one side, or within
    it (``not worse``), or neither."""
    b = boot(diffs)
    if b is None:
        return "no data"
    mean, lo, hi = b if higher_better else (-b[0], -b[2], -b[1])
    if lo > 0 and mean > margin:
        return "better"
    if lo >= -margin:
        return "not worse"
    return "worse" if hi < 0 and mean < -margin else "not shown"


def load(paths) -> list[dict]:
    """The records of the files, a later record of a key replacing an earlier one."""
    recs: dict[str, dict] = {}
    for path in paths:
        for name in sorted(glob.glob(path)):
            with open(name) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    recs[r["key"]] = r
    return list(recs.values())


class Scored:
    """The scored fits, looked up by ``(design, cell, method, depth, arm)``."""

    def __init__(self, recs: list[dict]) -> None:
        self.recs = recs
        self.rows: dict[tuple, dict[int, dict]] = collections.defaultdict(dict)
        for r in recs:
            row = {"status": r["status"], "rec": r, **scoring.score(r)}
            self.rows[(r["design"], r["cell"], r["method"], r["depth"], r["arm"])][r["seed"]] = row

    def get(self, design, cell, method, depth, arm, fresh=False) -> list[dict]:
        """The finished fits of a cell and arm on the seeds below 1000 (or from 1000)."""
        rows = self.rows.get((design, cell, method, depth, arm), {})
        return [r for s, r in sorted(rows.items()) if (s >= 1000) == fresh and r["status"] == "ok"]

    def stat(self, key: tuple, metric: str):
        """The mean and interval of a metric over the seeds of a cell and arm, the seeds in
        ascending order or in ``CARD_ORDER``."""
        rows = {r["rec"]["seed"]: r for r in self.get(*key)}
        order = CARD_ORDER.get(key) or sorted(rows)
        return boot([rows[s].get(metric) for s in order if s in rows])

    def paired(self, key_a: tuple, key_z: tuple, metric_a: str, metric_z: str | None = None):
        """Per-seed ``a - z`` of two cells' metrics, on the seeds where both finished."""
        a = {s: r for s, r in self.rows.get(key_a, {}).items() if s < 1000 and r["status"] == "ok"}
        z = {s: r for s, r in self.rows.get(key_z, {}).items() if s < 1000 and r["status"] == "ok"}
        out = []
        for s in sorted(set(a) & set(z)):
            x, y = a[s].get(metric_a), z[s].get(metric_z or metric_a)
            if x is not None and y is not None:
                out.append(x - y)
        return out


def _rng(values) -> list[float] | None:
    v = [x for x in values if x is not None]
    return [min(v), max(v)] if v else None


def figures(recs: list[dict]) -> dict:
    """Every figure of the card's section that the records hold, as a dict."""
    sc = Scored(recs)
    out: dict = {"fits": len(recs), "status": dict(collections.Counter(r["status"] for r in recs))}
    cohort = lambda c, v, m, arm, depth=3: ("cohort", f"{c}_{v}", m, depth, arm)  # noqa: E731
    present = [c for c in CONFIGS if sc.get(*cohort(c, "b0s10", "pc", "a"))]

    # Table 1: PC's SHD with no complete rows and 10% of the cells missing at random.
    shd, rec_tab = {}, {}
    for c in present:
        row = {"empty": sc.stat(cohort(c, "b0s10", "pc", "a"), "shd_empty")}
        for arm in ("a", "b", "c", "dcover"):
            row[arm] = sc.stat(cohort(c, "b0s10", "pc", arm), "shd")
        row["f"] = sc.stat(cohort(c, "complete", "pc", "f"), "shd")
        shd[CONFIGS[c]] = row
        rec_tab[CONFIGS[c]] = {
            "recall": {
                arm: sc.stat(cohort(c, "b0s10", "pc", arm), "i_rec") for arm in ("a", "b", "dcover")
            }
            | {"f": sc.stat(cohort(c, "complete", "pc", "f"), "i_rec")},
            "false_colliders": {
                arm: sc.stat(cohort(c, "b0s10", "pc", arm), "false_colliders")
                for arm in ("a", "b", "c")
            }
            | {"f": sc.stat(cohort(c, "complete", "pc", "f"), "false_colliders")},
        }
    out["shd_table"], out["recall_collider_table"] = shd, rec_tab

    # Fewer errors than the empty graph, in each of the unbridged settings.
    better = {}
    for c in present:
        for v in ("b0s0", "b0s10"):
            for m in ("pc", "fci"):
                k = cohort(c, v, m, "a")
                margin = max(1.0, 0.02 * (sc.stat(k, "true_edges") or [0])[0])
                better[f"{CONFIGS[c]}|{v}|{m}"] = verdict(
                    sc.paired(k, k, "shd", "shd_empty"), margin, False
                )
    out["fewer_than_empty"] = {
        "verdicts": better,
        "better": sum(v == "better" for v in better.values()),
    }
    for arm, name in (("b", "p1_over_empty"), ("c", "p0_over_empty")):
        ratios = []
        for c in present:
            for v in ("b0s0", "b0s10"):
                mean = sc.stat(cohort(c, v, "pc", arm), "shd")
                empty = sc.stat(cohort(c, v, "pc", "a"), "true_edges")
                if mean and empty:
                    ratios.append(mean[0] / empty[0])
        out[name] = _rng(ratios)

    def span(metric, arm, variants=("b0s10",), methods=("pc",), complete=False):
        vals = []
        for c in present:
            for v in variants:
                for m in methods:
                    stat = sc.stat(cohort(c, "complete" if complete else v, m, arm), metric)
                    vals.append(stat[0] if stat else None)
        return _rng(vals)

    both, unbridged = ("pc", "fci"), ("b0s0", "b0s10")
    out["class_i_recall"] = {arm: span("i_rec", arm) for arm in ("a", "b")} | {
        "f": span("i_rec", "f", complete=True)
    }
    out["next_to_apart_recall"] = {arm: span("eadj_rec", arm, unbridged) for arm in ("a", "b")}
    out["precision"] = {
        "a": span("all_prec", "a", unbridged, both),
        "f": span("all_prec", "f", methods=both, complete=True),
    }
    out["undecided_true_edges_per_fit"] = {
        CONFIGS[c]: sc.stat(cohort(c, "b0s10", "pc", "a"), "ul_true") for c in present
    }
    out["flagged_share_of_drawn_edges"] = span("ps_share", "a", unbridged, both)

    # No pair decided without data: every fit of arm a on data with pairs never observed together.
    fits = exceptions = 0
    for (design, cell, method, depth, arm), rows in sc.rows.items():
        apart = (design == "cohort" and "_b0" in cell) or (
            design == "hand" and cell.split("_")[0] in ("H1", "H1c", "H4", "H5k3")
        )
        if arm != "a" or not apart or method not in ("pc", "fci"):
            continue
        for r in rows.values():
            if r["status"] != "ok":
                continue
            fits += 1
            exceptions += int(r["ii_drawn"] > 0 or r["und_drawn_n"] > 0 or r["ii_unlisted"] > 0)
    out["undecided_pairs_listed_not_drawn"] = {"fits": fits, "exceptions": exceptions}

    # possibly_separable="ambiguous"
    amb: dict = {"false_colliders_max": {}, "head_recall": {}, "shd": {}, "fci_shd_rise": []}
    for m in ("pc", "fci"):
        vals = [
            (sc.stat(cohort(c, v, m, "amb"), "false_colliders") or [None])[0]
            for c in present
            for v in ("b0s0", "b0s10")
        ]
        amb["false_colliders_max"][m] = max([x for x in vals if x is not None], default=None)
    for c in present:
        for v in ("b0s0", "b0s10"):
            amb["head_recall"][f"{CONFIGS[c]}|{v}"] = sc.stat(cohort(c, v, "pc", "amb"), "head_rec")
            amb["shd"][f"{CONFIGS[c]}|{v}"] = {
                m: sc.stat(cohort(c, v, m, "amb"), "shd") for m in ("pc", "fci")
            }
            diffs = sc.paired(cohort(c, v, "fci", "amb"), cohort(c, v, "fci", "a"), "shd")
            if diffs:
                amb["fci_shd_rise"].append(float(np.mean(diffs)))
        margin = max(1.0, 0.02 * (sc.stat(cohort(c, "b0s10", "pc", "a"), "true_edges") or [0])[0])
        for v in unbridged:
            k = cohort(c, v, "pc", "amb")
            amb["shd"][f"{CONFIGS[c]}|{v}|pc vs empty"] = verdict(
                sc.paired(k, k, "shd", "shd_empty"), margin, False
            )
    amb["fci_shd_rise"] = _rng(amb["fci_shd_rise"])
    same = total = 0
    for (design, cell, method, depth, arm), rows in sc.rows.items():
        if design != "cohort" or arm != "amb" or ("_b0" in cell and not cell.endswith("complete")):
            continue
        for s, r in rows.items():
            twin = sc.rows.get(
                (design, cell, method, depth, "a" if "complete" not in cell else "f"), {}
            ).get(s)
            if r["status"] == "ok" and twin and twin["status"] == "ok":
                total += 1
                same += int(r["rec"]["edges"] == twin["rec"]["edges"])
    amb["same_graph_as_default_with_complete_rows_or_data"] = {"same": same, "of": total}
    out["ambiguous"] = amb

    # Complete rows: bridging rows.
    rows_out = {}
    for c in present:
        for v in ("b2s0", "b2s10", "b5s0", "b5s10"):
            rows_out[f"{CONFIGS[c]}|{v}"] = {
                "shd": sc.stat(cohort(c, v, "pc", "a"), "shd"),
                "class_i_recall": sc.stat(cohort(c, v, "pc", "a"), "i_rec"),
            }
    same = total = 0
    for (design, cell, method, depth, arm), rows in sc.rows.items():
        if design == "cohort" and arm == "a" and cell.split("_")[1].startswith(("b2", "b5")):
            for s, r in rows.items():
                for other in ("b", "c"):
                    z = sc.rows.get((design, cell, method, depth, other), {}).get(s)
                    if r["status"] == "ok" and z and z["status"] == "ok":
                        total += 1
                        same += int(r["rec"]["edges"] == z["rec"]["edges"])
    out["complete_rows"] = {"cells": rows_out, "a_equals_p1_or_p0": {"same": same, "of": total}}

    # The default call, with no limit on the conditioning sets.
    un = {}
    for c in ("M8", "L6", "S3"):
        k = ("cohort", f"{c}_b0s10", "pc", None, "a")
        if sc.rows.get(k):
            capped = sc.stat(cohort(c, "b0s10", "pc", "a"), "shd")
            un[CONFIGS[c]] = {
                "finished": len(sc.get(*k)),
                "of": len(sc.rows[k]),
                "shd": sc.stat(k, "shd"),
                "shd_depth_3": capped,
            }
    out["no_depth_limit"] = un

    # Lagged cohorts.
    lag: dict = {}
    for m in ("pcmci", "pcmciplus"):
        for v in ("nolap0", "nolap5"):
            for arm in ("a", "b"):
                k = ("lagged", v, m, None, arm)
                if sc.get(*k):
                    lag[f"{m}|{v}|{arm}"] = {
                        "class_i_recall": sc.stat(k, "i_rec"),
                        "precision": sc.stat(k, "all_prec"),
                        "undecided_drawn": sc.stat(k, "und_drawn"),
                    }
        k = ("lagged", "complete", m, None, "f")
        if sc.get(*k):
            lag[f"{m}|complete|f"] = {
                "class_i_recall": sc.stat(k, "i_rec"),
                "precision": sc.stat(k, "all_prec"),
            }
    out["lagged"] = lag

    # Hand cases: the graph each arm returns on the fork X <- Z -> Y of 1,000 rows.
    hand = {}
    for case in ("H1_n1000", "H1_fixed"):
        for arm, depth in (("a", 3), ("b", 3), ("b", 0), ("c", 3)):
            rows = [r["rec"] for r in sc.get("hand", case, "pc", depth, arm)]
            if rows:
                pattern = collections.Counter(
                    "|".join(map(str, sorted(map(tuple, r["edges"])))) or "empty" for r in rows
                )
                hand[f"{case}|{arm}|depth {depth}"] = {"fits": len(rows), **dict(pattern)}
    out["hand"] = hand
    return out


def fmt(t, nd: int = 1, interval: bool = False) -> str:
    """``mean`` or ``mean [lo, hi]`` of a bootstrap triple."""
    if t is None:
        return "-"
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if interval else f"{m:,.{nd}f}"


def tables(fig: dict) -> str:
    """The two tables of the card, as Markdown."""
    lines = [
        "| Configuration | Empty graph | Andrey 0.19.0 | p = 1 | p = 0 | Fewest columns dropped"
        " | Complete data |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in fig["shd_table"].items():
        lines.append(
            f"| {name} | {fmt(r['empty'])} | {fmt(r['a'], interval=True)} | {fmt(r['b'])} | "
            f"{fmt(r['c'])} | {fmt(r['dcover'])} | {fmt(r['f'])} |"
        )
    lines += [
        "",
        "| Configuration | Recall: Andrey | p = 1 | Fewest columns dropped | Complete data | False"
        " colliders: Andrey | p = 1 | p = 0 | Complete data |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in fig["recall_collider_table"].items():
        rc, fc = r["recall"], r["false_colliders"]
        lines.append(
            f"| {name} | {fmt(rc['a'], 2)} | {fmt(rc['b'], 2)} | {fmt(rc['dcover'], 2)} | "
            f"{fmt(rc['f'], 2)} | {fmt(fc['a'])} | {fmt(fc['b'])} | {fmt(fc['c'])} | "
            f"{fmt(fc['f'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Command line: print the tables and the other figures, or write them with ``--json``."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+", help="JSON lines files from run.py (globs allowed)")
    ap.add_argument("--json", help="write every figure to this file")
    args = ap.parse_args()
    fig = figures(load(args.records))
    print(tables(fig))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"settings": SETTINGS, "figures": fig}, fh, indent=1, default=float)
            fh.write("\n")
    else:
        rest = {k: v for k, v in fig.items() if k not in ("shd_table", "recall_collider_table")}
        print(json.dumps(rest, indent=1, default=float))


if __name__ == "__main__":
    main()
