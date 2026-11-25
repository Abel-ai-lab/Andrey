"""Run one fit: ``fit(spec) -> record``.

A spec is a dict with ``design`` (``hand``, ``cohort``, or ``lagged``), ``cell``, ``arm``,
``method`` (``pc``, ``fci``, ``pcmci``, or ``pcmciplus``), ``seed``, and ``depth`` (an int, or
``None`` for no limit). The arms are:

- ``a``: Andrey as it is;
- ``amb``: Andrey with ``possibly_separable="ambiguous"`` (PC and FCI);
- ``b``: a test without enough rows answers p = 1;
- ``c``: a test without enough rows answers p = 0;
- ``dcover``: columns dropped until every pair shares rows (:func:`arms.drop_cover`), then Andrey as
  it is;
- ``f``: the complete data, the values before masking.

The record holds the returned graph as edges ``[i, j, mark_i, mark_j]`` over the full column
indices (``i < j``), or for the lagged design the links ``[lag, i, j]`` (``X_i(t - lag) ->
X_j(t)``) and the lag-0 adjacencies; the pairs ``undecided_pairs`` lists, as column indices; the
edges ``possibly_separable`` lists; and the wall time of the fit alone.

An arm patches Andrey in the process it runs in, so one process runs one fit of arms ``b`` and
``c`` (``run.py`` forks one process per fit).
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np

from . import arms, generate

KEY_FIELDS = ("design", "cell", "arm", "method", "seed", "depth")
ARMS = ("a", "amb", "b", "c", "dcover", "f")


def key(spec: dict) -> str:
    """The identity of a fit, the same text in every run of it."""
    return "|".join(str(spec[k]) for k in KEY_FIELDS)


def _lagged_name(name: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)(?: at lag (\d+))?", name)
    if m is None:
        raise ValueError(f"not a lagged column name: {name!r}")
    return int(m.group(1)), int(m.group(2) or 0)


def fit(spec: dict) -> dict:
    """Run the fit ``spec`` names and return its record."""
    logging.disable(logging.WARNING)
    import andrey

    arm, method, depth = spec["arm"], spec["method"], spec["depth"]
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if arm == "b":
        arms.patch_fill(1.0)
    elif arm == "c":
        arms.patch_fill(0.0)
    ds = generate.make(spec["design"], spec["cell"], int(spec["seed"]))
    data = ds["complete"] if arm == "f" else ds["data"]
    d = data.shape[1]
    columns = list(range(d))
    rec: dict = {"key": key(spec), **spec, "d": d}
    if arm == "dcover":
        columns = arms.drop_cover(~np.isnan(data))
        data = data[:, columns]
        rec["kept"] = columns
    if len(columns) < 2:
        return {**rec, "status": "ok", "edges": [], "undecided": [], "possibly": [], "wall": 0.0}
    rec["n"] = int(data.shape[0])
    extra = {"possibly_separable": "ambiguous"} if arm == "amb" else {}
    t0 = time.perf_counter()
    if method in ("pc", "fci"):
        f = andrey.pc if method == "pc" else andrey.fci
        out = f(data, alpha=0.05, indep_test="fisherz", max_depth=depth, **extra)
    elif method == "pcmci":
        out = andrey.pcmci(data, tau_max=generate.TAU_MAX, alpha=0.05, indep_test="fisherz")
    elif method == "pcmciplus":  # alpha: its own default, 0.01
        out = andrey.pcmciplus(data, tau_max=generate.TAU_MAX, indep_test="fisherz", **extra)
    else:
        raise ValueError(method)
    rec["wall"] = time.perf_counter() - t0
    meta = out.metadata
    rec["not_tested_meta"] = int(meta.get("not_tested", 0) or 0)
    s = out.structure
    if method in ("pc", "fci"):
        m = s.to_numpy()
        ii, jj = np.nonzero(np.triu(m != 0, 1))
        rec["edges"] = [
            [columns[i], columns[j], int(m[i, j]), int(m[j, i])]
            for i, j in zip(ii, jj, strict=True)
        ]
        rec["undecided"] = [
            sorted((columns[int(a)], columns[int(b)]))
            for a, b in meta.get("undecided_pairs", []) or []
        ]
        rec["possibly"] = [
            sorted((columns[int(a)], columns[int(b)]))
            for a, b in meta.get("possibly_separable", []) or []
        ]
    else:
        links, lag0 = [], []
        for k in s.lags:
            m = s.lag(k).to_numpy()
            if k == 0:
                ii, jj = np.nonzero(np.triu(m != 0, 1))
                lag0 = [
                    [int(i), int(j), int(m[i, j]), int(m[j, i])]
                    for i, j in zip(ii, jj, strict=True)
                ]
                continue
            for i in range(d):
                for j in range(d):
                    if (i == j and m[i, i] == 2) or (i != j and m[j, i] == 2):
                        links.append([int(k), i, j])
        rec["links"], rec["lag0"] = links, lag0
        und = []
        for a, b in meta.get("undecided_pairs", []) or []:
            (va, la), (vb, lb) = _lagged_name(str(a)), _lagged_name(str(b))
            if la and not lb:  # X_a(t - la) and X_b(t)
                und.append([la, va, vb])
            elif lb and not la:
                und.append([lb, vb, va])
            else:  # lag 0
                und.append([0, min(va, vb), max(va, vb)])
        rec["undecided"] = und
        rec["possibly"] = [
            list(map(str, pair)) for pair in meta.get("possibly_separable", []) or []
        ]
    rec["status"] = "ok"
    return rec
