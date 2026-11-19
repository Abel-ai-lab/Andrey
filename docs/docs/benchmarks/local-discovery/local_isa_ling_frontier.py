"""The local discovery release card's accuracy and cost of local ISA-LiNG, saved as HTML beside this
script.

``local-isa-ling-frontier.html``: one panel per graph of the local ISA-LiNG paper's simulations,
the DAGs of mean degree 3 (100 to 10,000 rows) and 5 (2,000 rows) and the graphs with cycles
(2,000 rows), each cell a point per method: the mean SHD of the local graph against the median
seconds that answering one target costs, on a log scale. Local ISA-LiNG fits the target and its
blanket; global DirectLiNGAM and ICA-LiNGAM fit all 50 columns and are read at the target.

The points come from the lane's frontiers (``LANE``); the graphs with cycles come from its
``cyclic_context``, whose global fits were timed in one process rather than one fresh process per
fit. The script runs with ``uv run python
docs/docs/benchmarks/local-discovery/local_isa_ling_frontier.py [local-isa-ling.json]``.
"""

from __future__ import annotations

import json
import math
import sys
from html import escape
from pathlib import Path

from frontier import BOTTOM, HEIGHT, LEFT, RIGHT, STYLE, TOP, WIDTH, marker

HERE = Path(__file__).parent
LANE = HERE.parents[3] / "benchmarks" / "published" / "local-discovery" / "local-isa-ling.json"
# The panels: the cell, its title, and what a dataset holds.
CELLS = {
    "acyclic-d3": ("DAG, mean degree 3", "50 variables, 100 to 10,000 rows"),
    "acyclic-d5": ("DAG, mean degree 5", "50 variables, 2,000 rows"),
    "cyclic": ("Graph with cycles", "50 variables, 2,000 rows"),
}
# Each series: its solution, its name, its color (a categorical token), and its marker.
SERIES = (
    ("andrey.local_isa_ling", "Local ISA-LiNG", 1, "circle"),
    ("andrey.direct_lingam", "DirectLiNGAM, global", 3, "square"),
    ("andrey.ica_lingam", "ICA-LiNGAM, global", 4, "triangle"),
)


def panel(cell: str, fronts: list[dict]) -> str:
    """One graph's cells: mean SHD against median seconds per target, per method."""
    title, about = CELLS[cell]
    points = [
        (solution, front, front["points"][solution])
        for front in fronts
        for solution, *_ in SERIES
        if front["points"].get(solution, {}).get("seconds")
    ]
    seconds = [side["seconds"] for _, _, side in points]
    low, high = math.floor(math.log10(min(seconds))), math.ceil(math.log10(max(seconds)))
    step = 10 if max(side["shd"] for _, _, side in points) > 20 else 5
    top = math.ceil(max(side["shd"] for _, _, side in points) / step) * step

    def x(value: float) -> float:
        return LEFT + (math.log10(value) - low) / (high - low) * (RIGHT - LEFT)

    def y(value: float) -> float:
        return BOTTOM - value / top * (BOTTOM - TOP)

    axis = []
    for level in range(0, top + 1, step):
        axis.append(
            f'<line x1="{LEFT}" x2="{RIGHT}" y1="{y(level):.1f}" y2="{y(level):.1f}"/>'
            f'<text x="{LEFT - 5}" y="{y(level) + 3.5:.1f}" text-anchor="end">{level}</text>'
        )
    for power in range(low, high + 1):
        axis.append(
            f'<line x1="{x(10**power):.1f}" x2="{x(10**power):.1f}" y1="{TOP}" y2="{BOTTOM}"/>'
            f'<text x="{x(10**power):.1f}" y="{BOTTOM + 15}" text-anchor="middle">'
            f"{10**power:g} s</text>"
        )
    marks = []
    for solution, name, color, shape in SERIES:
        for _, front, side in (p for p in points if p[0] == solution):
            tip = (
                f"{name}, {front['n']:,} rows: SHD {side['shd']:.2f}, "
                f"{side['seconds']:.1f} s per target"
            )
            marks.append(marker(shape, x(side["seconds"]), y(side["shd"]), color, tip))
    label = (
        f"{title}: mean SHD of the local graph against median seconds per target on a log scale, "
        "one point per cell for local ISA-LiNG and global DirectLiNGAM and ICA-LiNGAM."
    )
    return (
        f'<figure class="mbf-panel"><figcaption><strong>{escape(title)}</strong>'
        f"<span>{escape(about)}</span></figcaption>"
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{escape(label)}">'
        f"<g>{''.join(axis)}</g>"
        f'<text class="mbf-title" x="{(LEFT + RIGHT) / 2:.0f}" y="{HEIGHT - 4}" '
        'text-anchor="middle">Seconds per target (log scale)</text>'
        f'<text class="mbf-title" transform="rotate(-90)" x="{-(TOP + BOTTOM) / 2:.0f}" y="11" '
        'text-anchor="middle">SHD of the local graph</text>'
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
    """The figure: the legend, a panel per graph, and the caption."""
    fronts = [*lane["frontiers"], lane["cyclic_context"]]
    panels = "".join(
        panel(cell, [f for f in fronts if f["cell"] == cell])
        for cell in CELLS
        if any(f["cell"] == cell for f in fronts)
    )
    note = (
        "One point per cell, each method with its own blanket or none: lower is more accurate and "
        "left is cheaper. The global fits on the graphs with cycles were timed in one process."
    )
    return (
        f'{STYLE}<figure class="mbf">{legend()}<div class="mbf-panels">{panels}</div>'
        f'<p class="mbf-note">{note}</p></figure>\n'
    )


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else LANE
    lane = json.loads(path.read_text(encoding="utf-8"))
    (HERE / "local-isa-ling-frontier.html").write_text(frontier(lane), encoding="utf-8")


if __name__ == "__main__":
    main()
