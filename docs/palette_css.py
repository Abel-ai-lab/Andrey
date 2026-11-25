"""Render an ``andrey.viz`` palette as CSS custom properties, for the docs and the homepage.

Each palette field becomes ``--andrey-<field>``; tuple fields are numbered from 1
(``--andrey-ramp-1`` ... ``--andrey-ramp-5``). The stylesheets name these properties, never a hex.

The code colors come from the same palette (``code_style``): a Pygments style whose every color is
a palette token, or that token mixed with ``ink`` where it would read under 4.5:1 on the code
block's ``surface_2``.
"""

from __future__ import annotations

import dataclasses

from pygments.formatters import HtmlFormatter
from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Literal,
    Name,
    Number,
    Operator,
    String,
    Token,
)

import andrey.viz

# The WCAG AA minimum for normal-size text, which every code color meets on the code block.
CODE_CONTRAST = 4.5

# Each kind of code token and the palette token it takes, named as in the CSS
# (``categorical-5`` is ``Palette.categorical[4]``). Text, names, operators, and punctuation stay
# in the ink; comments and a console's prompt and output are muted. Keywords take the purple of
# inline code and the API's names; strings take the green; numbers, constants, and a string's
# escapes take the magenta; functions, classes, and built-ins take the sky blue; errors and
# deletions take the red. The accent is left to links and amber to its one mark, so neither colors
# code. A kind not listed takes its parent's color.
CODE_TOKENS = {
    Token: "ink",
    Comment: "muted",
    Keyword: "categorical-5",
    Operator.Word: "categorical-5",
    Name.Tag: "categorical-5",
    String: "categorical-3",
    String.Escape: "categorical-7",
    String.Interpol: "categorical-7",
    String.Regex: "categorical-7",
    Number: "categorical-7",
    Literal: "categorical-7",
    Name.Constant: "categorical-7",
    Name.Entity: "categorical-7",
    Name.Label: "categorical-7",
    Name.Variable: "categorical-7",
    Name.Builtin: "categorical-6",
    Name.Function: "categorical-6",
    Name.Class: "categorical-6",
    Name.Decorator: "categorical-6",
    Name.Attribute: "categorical-6",
    Name.Exception: "categorical-4",
    Error: "categorical-4",
    Generic.Error: "categorical-4",
    Generic.Traceback: "categorical-4",
    Generic.Deleted: "categorical-4",
    Generic.Inserted: "categorical-3",
    Generic.Prompt: "muted",
    Generic.Output: "muted",
    Generic.Subheading: "muted",
}
# Type alone: emphasis, strong text, and a diff's headings.
CODE_TYPE = {
    Generic.Emph: "italic",
    Generic.Strong: "bold",
    Generic.EmphStrong: "bold italic",
    Generic.Heading: "bold",
    Generic.Subheading: "bold",
}


def palette_block(selector: str, palette: andrey.viz.Palette) -> str:
    """Render one theme's palette as a CSS rule of ``--andrey-*`` custom properties."""
    lines = [f"{selector} {{"]
    for field in dataclasses.fields(palette):
        value = getattr(palette, field.name)
        name = field.name.replace("_", "-")
        if isinstance(value, tuple):
            lines += [f"  --andrey-{name}-{i}: {v};" for i, v in enumerate(value, 1)]
        else:
            lines.append(f"  --andrey-{name}: {value};")
    lines.append("}")
    return "\n".join(lines)


def token_color(palette: andrey.viz.Palette, name: str) -> str:
    """The palette's color for a CSS token name such as ``muted`` or ``categorical-5``."""
    field, _, index = name.partition("-")
    if index.isdigit():
        return getattr(palette, field)[int(index) - 1]
    return getattr(palette, name.replace("-", "_"))


def _channels(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def contrast(color: str, ground: str) -> float:
    """The WCAG 2 contrast ratio of two ``#rrggbb`` colors, from 1 to 21."""

    def luminance(hex_color: str) -> float:
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in (v / 255 for v in _channels(hex_color))
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(color), luminance(ground)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def mix(color: str, other: str, percent: int) -> str:
    """``color`` with ``percent`` of ``other``, as CSS ``color-mix(in srgb, ...)`` mixes them."""
    return "#" + "".join(
        f"{round(a + (b - a) * percent / 100):02x}"
        for a, b in zip(_channels(color), _channels(other), strict=True)
    )


def code_color(palette: andrey.viz.Palette, name: str) -> tuple[str, int]:
    """A code token's color and the share of ``ink`` mixed into it, in percent.

    The share is 0 when the palette token reads at ``CODE_CONTRAST`` or more on ``surface_2``,
    else the smallest whole percent that does, which keeps the token's hue.
    """
    color = token_color(palette, name)
    for percent in range(101):
        shade = mix(color, palette.ink, percent)
        if contrast(shade, palette.surface_2) >= CODE_CONTRAST:
            return shade, percent
    raise ValueError(f"{name} cannot reach {CODE_CONTRAST}:1 on surface_2, even as ink")


def code_style(palette: andrey.viz.Palette) -> type[Style]:
    """A Pygments style of the palette's code colors (``CODE_TOKENS``) on its ``surface_2``."""
    styles = {token: code_color(palette, name)[0] for token, name in CODE_TOKENS.items()}
    for token, type_ in CODE_TYPE.items():
        styles[token] = f"{type_} {styles.get(token, '')}".strip()
    return type(
        "AndreyCode",
        (Style,),
        {
            "background_color": palette.surface_2,
            "highlight_color": mix(palette.surface_2, palette.accent, 14),
            "line_number_color": palette.muted,
            "line_number_special_color": palette.ink,
            "line_number_special_background_color": mix(palette.surface_2, palette.accent, 14),
            "styles": styles,
        },
    )


def code_block(prefix: str, palette: andrey.viz.Palette) -> str:
    """The palette's code colors as CSS rules under ``prefix``, such as ``.highlight``, the line
    numbers' rules included."""
    formatter = HtmlFormatter(style=code_style(palette))
    lines = [f"{prefix} {line}" for line in formatter.get_linenos_style_defs()]
    lines += formatter.get_background_style_defs(prefix)
    lines += formatter.get_token_style_defs(prefix)
    return "\n".join(lines)
