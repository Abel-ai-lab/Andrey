"""The 0.18.0 release card's accuracy and cost of MB-by-MB and LocPC-CDE, saved as HTML beside this
script.

``mb-by-mb-frontier.html``: one panel per network of the MB-by-MB paper, Alarm, Hailfinder, and the
45 x 45 chessboard, each cell (rows, level) a point per method: the mean SHD per target against
the mean CI tests that answering one target costs, on a log scale. A target-centered method,
MB-by-MB or CMB, pays its walk at the target; a global PC, Andrey's or bnlearn's pc.stable, pays
its whole fit. CMB and bnlearn ran on Alarm and Hailfinder only.

The points come from the MB-by-MB rows of the card's direct lane (``LANE``), the rows whose first
solution is ``andrey.mb_by_mb``.

``locpc-cde-frontier.html``: one panel per data family of the LocPC-CDE paper, linear-Gaussian and
binary, each graph size a point per method: the verdict error over the identifiable and the
non-identifiable pair of each seed against the median CI tests of one fit, on a log scale. The
points come from the ``pooled_verdict_floor`` rows of the LocPC-CDE lane (``PAIR_LANE``).

The script runs with ``uv run python docs/docs/benchmarks/0.18.0/frontier.py [direct.json]``.
"""

from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
LANE = HERE.parents[3] / "benchmarks" / "published" / "0.18.0" / "direct.json"
PAIR_LANE = LANE.with_name("locpc-cde.json")
# The LocPC-CDE panels, by the lane's family, and its series, as ``SERIES`` gives them.
FAMILIES = {
    "gaussian": ("Linear-Gaussian", "Fisher's z test, 5,000 rows"),
    "binary": ("Binary", "G-square test, 5,000 rows"),
}
PAIR_SERIES = (
    ("andrey.locpc_cde", "LocPC-CDE", 1, "circle"),
    ("locpc_cde.authors", "Authors' code", 2, "diamond"),
    ("andrey.pc.adjustment", "Andrey's PC", 3, "square"),
)
# The panels, in the paper's order: the network, its title, and what a dataset holds.
NETWORKS = {
    "alarm": ("Alarm", "37 variables, every node a target"),
    "hailfinder": ("Hailfinder", "56 variables, every node a target"),
    "chess45": ("45 x 45 chessboard", "2,025 variables, 10 targets"),
}
# Each series: its solution, its name, its color (a categorical token), and its marker.
SERIES = (
    ("andrey.mb_by_mb", "MB-by-MB", 1, "circle"),
    ("andrey.cmb", "CMB", 2, "diamond"),
    ("andrey.pc.gsq", "Andrey's PC", 3, "square"),
    ("bnlearn.pc-stable.graph", "bnlearn pc.stable", 4, "triangle"),
)
# The corners of each marker drawn as a polygon, about as wide as the circle's 8 units.
CORNERS = {
    "square": ((-3.6, -3.6), (3.6, -3.6), (3.6, 3.6), (-3.6, 3.6)),
    "diamond": ((0, -4.9), (4.9, 0), (0, 4.9), (-4.9, 0)),
    "triangle": ((0, -4.7), (4.7, 3.6), (-4.7, 3.6)),
}
# The panel's drawing box, in viewBox units, and the plot area inside it.
WIDTH, HEIGHT = 250, 202
LEFT, RIGHT, TOP, BOTTOM = 36, 242, 8, 164
STYLE = (
    "<style>"
    ".mbf { margin: 2rem 0; padding: clamp(1rem, 2.5vw, 1.5rem); border: 1px solid "
    "var(--andrey-line); border-radius: 12px; background: var(--andrey-surface); "
    "container-type: inline-size; }"
    ".mbf-legend { display: flex; flex-wrap: wrap; gap: 0.25rem 1rem; margin: 0 0 1rem; "
    "color: var(--andrey-muted); font-size: 0.82rem; }"
    ".mbf-key { display: inline-flex; align-items: center; gap: 0.4rem; }"
    ".mbf-key svg { width: 0.8rem; height: 0.8rem; overflow: visible; }"
    ".mbf-panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); "
    "gap: 1.25rem; }"
    ".mbf-panel { margin: 0; }"
    ".mbf-panel strong { display: block; color: var(--andrey-ink); font-size: 1rem; "
    "font-weight: 600; }"
    ".mbf-panel span { display: block; color: var(--andrey-muted); font-size: 0.8rem; }"
    ".mbf-panel svg { display: block; width: 100%; height: auto; margin-top: 0.4rem; "
    "overflow: visible; }"
    ".mbf-panel svg text { fill: var(--andrey-muted); "
    "font: 500 11px var(--font-sans, sans-serif); }"
    ".mbf-panel svg text.mbf-title { fill: var(--andrey-ink); }"
    ".mbf-panel svg line { stroke: var(--andrey-line); stroke-width: 1; }"
    ".mbf-mark { stroke: var(--andrey-surface); stroke-width: 1; }"
    + "".join(f".mbf-s{c} {{ fill: var(--andrey-categorical-{c}); }}" for _, _, c, _ in SERIES)
    + ".mbf-note { margin: 1rem 0 0; color: var(--andrey-muted); font-size: 0.8rem; "
    "line-height: 1.6; text-align: start; }"
    "@container (max-width: 44rem) { .mbf-panels { "
    "grid-template-columns: repeat(2, minmax(0, 1fr)); } }"
    "@container (max-width: 30rem) { .mbf-panels { grid-template-columns: minmax(0, 1fr); } }"
    "</style>"
)


def marker(shape: str, x: float, y: float, color: int, title: str = "") -> str:
    """One point's mark, about 8 units across, its shape and color naming its series."""
    tip = f"<title>{escape(title)}</title>" if title else ""
    attrs = f'class="mbf-mark mbf-s{color}"'
    if shape == "circle":
        return f'<circle {attrs} cx="{x:.1f}" cy="{y:.1f}" r="4">{tip}</circle>'
    points = " ".join(f"{x + dx:.1f},{y + dy:.1f}" for dx, dy in CORNERS[shape])
    return f'<polygon {attrs} points="{points}">{tip}</polygon>'


def count(value: float) -> str:
    """A power of ten as a tick reads it: 100, 1k, 10k, 1M."""
    for size, suffix in ((1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:g}{suffix}"
    return f"{value:g}"


def panel(network: str, rows: list[dict]) -> str:
    """One network's cells: SHD per target against CI tests per answer, per method."""
    title, about = NETWORKS[network]
    points = [
        (solution, row, row["solutions"][solution])
        for row in rows
        for solution, *_ in SERIES
        if row["solutions"].get(solution, {}).get("datasets")
    ]
    tests = [side["tests_per_answer"] for _, _, side in points]
    low, high = math.floor(math.log10(min(tests))), math.ceil(math.log10(max(tests)))
    top = math.ceil(max(side["shd"] for _, _, side in points) * 2) / 2

    def x(value: float) -> float:
        return LEFT + (math.log10(value) - low) / (high - low) * (RIGHT - LEFT)

    def y(value: float) -> float:
        return BOTTOM - value / top * (BOTTOM - TOP)

    axis = []
    for k in range(int(top * 2) + 1):
        level = k / 2
        axis.append(
            f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y(level):.1f}" y2="{y(level):.1f}"/>'
            f'<text x="{LEFT - 5}" y="{y(level) + 3.5:.1f}" text-anchor="end">{level:g}</text>'
        )
    for power in range(low, high + 1):
        axis.append(
            f'<line x1="{x(10**power):.1f}" x2="{x(10**power):.1f}" y1="{TOP}" y2="{BOTTOM}"/>'
            f'<text x="{x(10**power):.1f}" y="{BOTTOM + 15}" text-anchor="middle">'
            f"{count(10**power)}</text>"
        )
    marks = []
    for solution, name, color, shape in SERIES:
        for _, row, side in (p for p in points if p[0] == solution):
            tip = (
                f"{name}, {row['n']:,} rows, alpha {row['alpha']}: SHD {side['shd']:.2f}, "
                f"{side['tests_per_answer']:,.0f} tests per answer"
            )
            marks.append(marker(shape, x(side["tests_per_answer"]), y(side["shd"]), color, tip))
    names = ", ".join(
        name for solution, name, *_ in SERIES if any(p[0] == solution for p in points)
    )
    label = (
        f"{title}: SHD per target against CI tests per answer on a log scale, one point per cell "
        f"for {names}."
    )
    return (
        f'<figure class="mbf-panel"><figcaption><strong>{escape(title)}</strong>'
        f"<span>{escape(about)}</span></figcaption>"
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(label)}">'
        f"<g>{''.join(axis)}</g>"
        f'<text class="mbf-title" x="{(LEFT + RIGHT) / 2:.0f}" y="{HEIGHT - 4}" '
        'text-anchor="middle">CI tests per answer (log scale)</text>'
        f'<text class="mbf-title" transform="rotate(-90)" x="{-(TOP + BOTTOM) / 2:.0f}" y="11" '
        'text-anchor="middle">SHD per target</text>'
        f"{''.join(marks)}</svg></figure>"
    )


def legend() -> str:
    """Every series' name beside its mark."""
    keys = "".join(
        f'<span class="mbf-key"><svg viewBox="-5 -5 10 10" aria-hidden="true">'
        f"{marker(shape, 0, 0, color)}</svg>{escape(name)}</span>"
        for _, name, color, shape in SERIES
    )
    return f'<p class="mbf-legend">{keys}</p>'


def frontier(lane: dict) -> str:
    """The figure: the legend, a panel per network, and the caption."""
    rows = [r for r in lane["rows"] if next(iter(r["solutions"])) == "andrey.mb_by_mb"]
    panels = "".join(
        panel(network, [r for r in rows if r["network"] == network])
        for network in NETWORKS
        if any(r["network"] == network for r in rows)
    )
    note = (
        "One point per cell, 500 to 5,000 rows at alpha 0.01 and 0.05: lower is more accurate and "
        "left is cheaper; a target-centered method pays its walk at the target, a global PC its "
        "whole fit."
    )
    return (
        f'{STYLE}<figure class="mbf">{legend()}<div class="mbf-panels">{panels}</div>'
        f'<p class="mbf-note">{note}</p></figure>\n'
    )


def pair_panel(family: str, rows: list[dict]) -> str:
    """One family's sizes: the verdict error against the median CI tests of a fit, per method."""
    title, about = FAMILIES[family]
    points = [(solution, row, row[solution]) for row in rows for solution, *_ in PAIR_SERIES]
    tests = [side["tests_median"] for _, _, side in points]
    low, high = math.floor(math.log10(min(tests))), math.ceil(math.log10(max(tests)))
    top = math.ceil(max(side["verdict_error_mean"] for _, _, side in points) * 4) / 4

    def x(value: float) -> float:
        return LEFT + (math.log10(value) - low) / (high - low) * (RIGHT - LEFT)

    def y(value: float) -> float:
        return BOTTOM - value / top * (BOTTOM - TOP)

    axis = []
    for k in range(int(top * 4) + 1):
        level = k / 4
        axis.append(
            f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y(level):.1f}" y2="{y(level):.1f}"/>'
            f'<text x="{LEFT - 5}" y="{y(level) + 3.5:.1f}" text-anchor="end">{level:g}</text>'
        )
    for power in range(low, high + 1):
        axis.append(
            f'<line x1="{x(10**power):.1f}" x2="{x(10**power):.1f}" y1="{TOP}" y2="{BOTTOM}"/>'
            f'<text x="{x(10**power):.1f}" y="{BOTTOM + 15}" text-anchor="middle">'
            f"{count(10**power)}</text>"
        )
    marks = []
    for solution, name, color, shape in PAIR_SERIES:
        for _, row, side in (p for p in points if p[0] == solution):
            tip = (
                f"{name}, {row['d']} variables: verdict error {side['verdict_error_mean']:.3f}, "
                f"{side['tests_median']:,.0f} tests per fit"
            )
            marks.append(
                marker(shape, x(side["tests_median"]), y(side["verdict_error_mean"]), color, tip)
            )
    label = (
        f"{title}: verdict error against CI tests per fit on a log scale, one point per graph "
        "size for LocPC-CDE, the authors' code, and Andrey's PC."
    )
    return (
        f'<figure class="mbf-panel"><figcaption><strong>{escape(title)}</strong>'
        f"<span>{escape(about)}</span></figcaption>"
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(label)}">'
        f"<g>{''.join(axis)}</g>"
        f'<text class="mbf-title" x="{(LEFT + RIGHT) / 2:.0f}" y="{HEIGHT - 4}" '
        'text-anchor="middle">CI tests per fit (log scale)</text>'
        f'<text class="mbf-title" transform="rotate(-90)" x="{-(TOP + BOTTOM) / 2:.0f}" y="11" '
        'text-anchor="middle">Verdict error</text>'
        f"{''.join(marks)}</svg></figure>"
    )


def pair_frontier(lane: dict) -> str:
    """The LocPC-CDE figure: the legend, a panel per family, and the caption."""
    rows = lane["pooled_verdict_floor"]
    panels = "".join(pair_panel(f, [r for r in rows if r["family"] == f]) for f in FAMILIES)
    keys = "".join(
        f'<span class="mbf-key"><svg viewBox="-5 -5 10 10" aria-hidden="true">'
        f"{marker(shape, 0, 0, color)}</svg>{escape(name)}</span>"
        for _, name, color, shape in PAIR_SERIES
    )
    note = (
        "One point per graph size, 10 to 200 variables, over 40 datasets: lower is more accurate "
        'and left is cheaper; the constant answer "not identifiable" scores 0.5.'
    )
    # Two panels share the width; on a narrow page they stack, as the three of MB-by-MB do.
    two = (
        "<style>.mbf-panels.mbf-two { grid-template-columns: repeat(2, minmax(0, 1fr)); }"
        "@container (max-width: 30rem) { .mbf-panels.mbf-two { "
        "grid-template-columns: minmax(0, 1fr); } }</style>"
    )
    return (
        f'{STYLE}{two}<figure class="mbf"><p class="mbf-legend">{keys}</p>'
        f'<div class="mbf-panels mbf-two">{panels}</div><p class="mbf-note">{note}</p></figure>\n'
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE
    lane = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "mb-by-mb-frontier.html").write_text(frontier(lane), encoding="utf-8")
    pair = json.loads(PAIR_LANE.read_text(encoding="utf-8"))
    (HERE / "locpc-cde-frontier.html").write_text(pair_frontier(pair), encoding="utf-8")


if __name__ == "__main__":
    main()
