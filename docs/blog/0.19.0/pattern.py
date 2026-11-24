"""The overlapping-variables post's mock-up: which variables each dataset measures.

It prints the post's three tables in Markdown, the first rows of the two studies of
``overlapping.py`` and the table that stacks them, and writes ``pattern.html``: the missingness
pattern of the benchmark's small configuration, three cohorts over 20 variables, as inline SVG whose
colors are ``--andrey-*`` tokens, so it follows the page's light or dark theme. Each row of the
pattern is a cohort and each column a variable; a filled cell is measured, a hollow one is not.
Run with ``uv run python docs/blog/0.19.0/pattern.py``.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SHOWN = 3  # rows shown from each study

# The small configuration: the rows of each cohort, the number of columns that every cohort
# measures, the columns that only some cohorts measure (their number and those cohorts), and the
# number of columns that only one cohort measures, for each cohort.
SIZES = (2000, 1700, 1300)
EVERY = 8
SOME = ((4, (0, 1)), (4, (2,)))
ONE = (1, 1, 2)

CELL = 22  # column width, in SVG units
GAP = 3  # between columns
ROW_SCALE = 0.045  # SVG units per data row
BAND_GAP = 8
LEFT = 80  # room for the cohort labels
TOP = 26  # room for the column labels


def studies() -> pd.DataFrame:
    """The data of ``overlapping.py``: rows 0-999 are the first study, rows 1000-1999 the second."""
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


def patterns() -> np.ndarray:
    """``(K, d)`` booleans: the columns each cohort measures."""
    d = EVERY + sum(size for size, _ in SOME) + sum(ONE)
    out = np.zeros((len(SIZES), d), dtype=bool)
    for k in range(len(SIZES)):
        out[k, :EVERY] = True
        start = EVERY
        for size, cohorts in SOME:
            if k in cohorts:
                out[k, start : start + size] = True
            start += size
        start += sum(ONE[:k])
        out[k, start : start + ONE[k]] = True
    return out


def never_together(measured: np.ndarray) -> int:
    """The number of pairs of columns that no cohort measures together."""
    d = measured.shape[1]
    return sum(not (measured[:, i] & measured[:, j]).any() for i, j in combinations(range(d), 2))


def figure() -> str:
    """The small configuration's missingness pattern as HTML."""
    measured = patterns()
    d = measured.shape[1]
    xs = [LEFT + j * (CELL + GAP) for j in range(d)]
    heights = [size * ROW_SCALE for size in SIZES]
    ys = [TOP + sum(heights[:k]) + k * BAND_GAP for k in range(len(SIZES))]
    bottom = ys[-1] + heights[-1]
    parts = [
        f'<text x="{x + CELL / 2}" y="{TOP - 9}" class="ov-tick">X{j + 1}</text>'
        for j, x in enumerate(xs)
    ]
    for k, (y0, h) in enumerate(zip(ys, heights)):
        parts.append(
            f'<text x="{LEFT - 10}" y="{y0 + h / 2:.1f}" class="ov-cohort">Cohort {k + 1}</text>'
        )
        for j in range(d):
            kind = "ov-yes" if measured[k, j] else "ov-no"
            parts.append(
                f'<rect x="{xs[j]}" y="{y0:.1f}" width="{CELL}" height="{h:.1f}" rx="2" '
                f'class="{kind}"/>'
            )
    key_y = bottom + 22
    parts += [
        f'<rect x="{LEFT}" y="{key_y - 7}" width="14" height="14" rx="2" class="ov-yes"/>',
        f'<text x="{LEFT + 20}" y="{key_y}" class="ov-key">measured</text>',
        f'<rect x="{LEFT + 110}" y="{key_y - 7}" width="14" height="14" rx="2" class="ov-no"/>',
        f'<text x="{LEFT + 130}" y="{key_y}" class="ov-key">not measured</text>',
    ]
    width = xs[-1] + CELL + 8
    height = key_y + 16
    style = (
        "<style>"
        ".ov-figure svg{width:100%;height:auto;max-width:620px;display:block;margin:0 auto}"
        ".ov-figure text{font-family:var(--andrey-font-sans);fill:var(--andrey-ink)}"
        ".ov-tick{font-size:11px;text-anchor:middle;fill:var(--andrey-muted)!important}"
        ".ov-cohort{font-size:13px;text-anchor:end;dominant-baseline:middle}"
        ".ov-key{font-size:13px;dominant-baseline:middle}"
        ".ov-yes{fill:var(--andrey-categorical-1)}"
        ".ov-no{fill:none;stroke:var(--andrey-muted);stroke-width:1;stroke-dasharray:3 2}"
        "</style>"
    )
    svg = (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Missingness pattern '
        "of three cohorts over 20 variables, X1 to X20: a filled cell means the cohort measured "
        'the variable, a hollow cell that it did not.">'
        f"{''.join(parts)}</svg>"
    )
    caption = (
        "<figcaption>The missingness pattern of the benchmark's smallest configuration. Each row "
        "is a cohort, drawn as tall as its number of rows, and each column is one of the 20 "
        "variables. A filled cell means that the cohort measured the variable; a hollow cell means "
        "that it did not, so the stacked table is missing that whole block. The pattern leaves "
        f"{never_together(measured)} pairs of variables never measured together.</figcaption>"
    )
    return f'{style}<figure class="ov-figure">{svg}{caption}</figure>\n'


if __name__ == "__main__":
    print(tables())
    (HERE / "pattern.html").write_text(figure())
