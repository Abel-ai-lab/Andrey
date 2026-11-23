"""The overlapping-variables post's mock-up: which variables each dataset measures.

It prints the post's three tables in Markdown, the first rows of the two studies of ``merged.py``
and the table that stacks them, and writes ``pattern.html``: the missingness pattern of the
benchmark's small configuration, three cohorts over 20 variables, as inline SVG whose colors are
``--andrey-*`` tokens, so it follows the page's light or dark theme. Run with
``uv run python docs/blog/0.19.0/pattern.py``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SHOWN = 3  # rows shown from each study

# The small configuration: the rows of each cohort, the core columns that every cohort measures,
# each module's size and the cohorts that measure it, and each cohort's own columns.
SIZES = (2000, 1700, 1300)
CORE = 8
MODULES = ((4, (0, 1)), (4, (2,)))
OWN = (1, 1, 2)
GROUPS = ["Core", "Module A", "Module B", "Own"]

CELL = 20  # column width, in SVG units
GAP = 3  # between columns
GROUP_GAP = 10  # added between column groups
ROW_SCALE = 0.045  # SVG units per data row
BAND_GAP = 8
LEFT = 78  # room for the cohort labels
TOP = 34  # room for the group labels


def studies() -> pd.DataFrame:
    """The data of ``merged.py``: rows 0-999 are the first study, rows 1000-1999 the second."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    w = rng.standard_normal(n)
    z = 0.8 * x + 0.8 * w + 0.6 * rng.standard_normal(n)
    y = 0.8 * z + 0.6 * rng.standard_normal(n)
    data = pd.DataFrame({"X": x, "W": w, "Z": z, "Y": y})
    data.loc[1000:, "X"] = np.nan
    data.loc[:999, "Y"] = np.nan
    return data


def markdown(frame: pd.DataFrame) -> str:
    """A frame as a Markdown table, numbers to two decimals and a missing value as ``NaN``."""

    def cell(v: object) -> str:
        if isinstance(v, float):
            return "NaN" if np.isnan(v) else f"{v:.2f}"
        return str(v)

    lines: list[str] = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "|".join("---:" for _ in frame.columns) + "|",
    ]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(False)]
    return "\n".join(lines)


def tables() -> str:
    """The post's three tables: each study as it was measured, then the two stacked."""
    data = studies()
    first, second = data.iloc[:SHOWN], data.iloc[1000 : 1000 + SHOWN]
    stacked = pd.concat([first, second])
    stacked.insert(0, "Study", ["1"] * SHOWN + ["2"] * SHOWN)
    return "\n\n".join(
        [
            markdown(first[["X", "W", "Z"]]),
            markdown(second[["W", "Z", "Y"]]),
            markdown(stacked),
        ]
    )


def patterns() -> tuple[np.ndarray, list[int]]:
    """``(K, d)`` booleans, the columns each cohort measures, and each column's group index."""
    d = CORE + sum(size for size, _ in MODULES) + sum(OWN)
    group = [0] * CORE + [1] * MODULES[0][0] + [2] * MODULES[1][0] + [3] * sum(OWN)
    out = np.zeros((len(SIZES), d), dtype=bool)
    for k in range(len(SIZES)):
        out[k, :CORE] = True
        start = CORE
        for size, cohorts in MODULES:
            if k in cohorts:
                out[k, start : start + size] = True
            start += size
        start += sum(OWN[:k])
        out[k, start : start + OWN[k]] = True
    return out, group


def never_together(measured: np.ndarray) -> int:
    """The number of pairs of columns that no cohort measures together."""
    d = measured.shape[1]
    return sum(not (measured[:, i] & measured[:, j]).any() for i, j in combinations(range(d), 2))


def figure() -> str:
    """The small configuration's missingness pattern as HTML."""
    measured, group = patterns()
    d = measured.shape[1]
    xs = [LEFT + j * (CELL + GAP) + group[j] * GROUP_GAP for j in range(d)]
    heights = [size * ROW_SCALE for size in SIZES]
    ys = [TOP + sum(heights[:k]) + k * BAND_GAP for k in range(len(SIZES))]
    bottom = ys[-1] + heights[-1]
    parts = []
    for g, name in enumerate(GROUPS):
        cols = [xs[j] for j in range(d) if group[j] == g]
        x0, x1 = cols[0], cols[-1] + CELL
        parts.append(f'<text x="{(x0 + x1) / 2}" y="{TOP - 16}" class="ov-group">{name}</text>')
        parts.append(f'<line x1="{x0}" x2="{x1}" y1="{TOP - 9}" y2="{TOP - 9}" class="ov-rule"/>')
    for k, (y0, h) in enumerate(zip(ys, heights)):
        parts.append(
            f'<text x="{LEFT - 10}" y="{y0 + h / 2:.1f}" class="ov-cohort">Cohort {k + 1}</text>'
        )
        for j in range(d):
            kind = f"ov-c{k + 1}" if measured[k, j] else "ov-miss"
            parts.append(
                f'<rect x="{xs[j]}" y="{y0:.1f}" width="{CELL}" height="{h:.1f}" rx="2" '
                f'class="{kind}"/>'
            )
    key_y = bottom + 22
    key_x = LEFT
    parts += [
        f'<rect x="{key_x}" y="{key_y - 7}" width="14" height="14" rx="2" class="ov-c1"/>',
        f'<text x="{key_x + 20}" y="{key_y}" class="ov-key">Measured</text>',
        f'<rect x="{key_x + 110}" y="{key_y - 7}" width="14" height="14" rx="2" class="ov-miss"/>',
        f'<text x="{key_x + 130}" y="{key_y}" class="ov-key">Never measured: missing</text>',
    ]
    width = xs[-1] + CELL + 8
    height = key_y + 16
    style = (
        "<style>"
        ".ov-figure svg{width:100%;height:auto;max-width:600px;display:block;margin:0 auto}"
        ".ov-figure text{font-family:var(--andrey-font-sans);fill:var(--andrey-ink)}"
        ".ov-group{font-size:12px;font-weight:600;text-anchor:middle}"
        ".ov-rule{stroke:var(--andrey-muted);stroke-width:1}"
        ".ov-cohort{font-size:12px;text-anchor:end;dominant-baseline:middle}"
        ".ov-key{font-size:12px;dominant-baseline:middle}"
        ".ov-c1{fill:var(--andrey-categorical-1)}"
        ".ov-c2{fill:var(--andrey-categorical-2)}"
        ".ov-c3{fill:var(--andrey-categorical-3)}"
        ".ov-miss{fill:none;stroke:var(--andrey-muted);stroke-width:1;stroke-dasharray:3 2}"
        "</style>"
    )
    svg = (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Which variables each '
        "of three cohorts measures: every cohort measures the core variables, cohorts 1 and 2 "
        "measure module A, cohort 3 measures module B, and each cohort measures a few variables of "
        'its own. Every other cell is a block of missing values.">'
        f"{''.join(parts)}</svg>"
    )
    caption = (
        "<figcaption>The small configuration of the benchmark: three cohorts over 20 variables, "
        "each drawn as a band as tall as its rows. Every cohort measures the core, cohorts 1 and 2 "
        "also module A, cohort 3 module B, and each cohort a few variables of its own. The hollow "
        f"blocks are never measured, which leaves {never_together(measured)} pairs of variables "
        "never observed together.</figcaption>"
    )
    return f'{style}<figure class="ov-figure">{svg}{caption}</figure>\n'


if __name__ == "__main__":
    print(tables())
    (HERE / "pattern.html").write_text(figure())
