"""Agreement of Andrey's Regime-PCMCI with the authors' code, start by start, on the paper's models.

The data are the toy models of Saggioro et al. (2020): Table 1's "arrow direction", "causal
effect", "lag", "sign X1", and "sign X1X2" (two regimes of 70 to 100 steps in turn) and Table 4's
three regimes (60, 70, or 80 steps each). Both packages fit the same series one start at a time,
start ``a`` from seed ``a`` (tigramite's ``RPCMCI(seed=a)`` with ``max_anneal=1``, Andrey's
``n_starts=1, seed=a``), which gives both the same first regime weights. The settings are the
paper's (Table 2): partial correlation with the Student-t p-value, ``pc_alpha=0.2``,
``alpha_level=0.01``, no FDR correction, ``max_iter=20``, and ``max_transitions`` set to the true
number of switches.

    ANDREY_BENCH_TIGRAMITE_PYTHON=<interpreter of benchmarks/start_tigramite.sh> \\
        uv run python benchmarks/rpcmci_agreement.py <empty folder> [--steps 1000] [--seeds 3] \\
        [--starts 3] [--threshold 0.5]

tigramite is GPL-licensed: its fits run in its own environment (``--reference``, below), which
imports this module for nothing but numpy and the models. Both join a step to a regime when its
weight is above ``threshold``. The script prints one row per model and one for all of them: the
starts with the same regime path and graphs, those with also the same per-iteration changes and
weights within ``1e-9``, those that differ or fail, and each package's mean regime accuracy and mean
lagged SHD (summed over the regimes) against the truth over its fits that did not fail. A step's
regime is its largest weight rounded to 9 decimals, as ``andrey.RegimeOutput`` reads it, so weights
that differ only in their last bits give the same path.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

#: (effect, cause, lag) -> coefficient, per regime; variable 0 is the paper's X1, 1 is X2.
MODELS = {
    "arrow direction": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(0, 1, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
    "causal effect": [{(0, 0, 1): 0.8, (1, 1, 1): 0.4}, {(0, 0, 1): 0.1, (1, 1, 1): 0.4}],
    "lag": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(1, 0, 2): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
    "sign X1": [{(0, 0, 1): 0.8, (1, 1, 1): 0.2}, {(0, 0, 1): -0.8, (1, 1, 1): 0.2}],
    "sign X1X2": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(1, 0, 1): -0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
    "three regimes": [
        {(1, 0, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(1, 0, 1): -0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
        {(0, 1, 1): 0.8, (0, 0, 1): 0.2, (1, 1, 1): 0.2},
    ],
}
TAU_MAX = 3
SETTINGS = {"pc_alpha": 0.2, "alpha_level": 0.01, "max_iter": 20}


def simulate(model: str, n_steps: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``(n_steps, 2)`` series of ``MODELS[model]`` and the regime of each step."""
    rng = np.random.default_rng(seed)
    path: list[int] = []
    regime = 0
    while len(path) < n_steps:
        if model == "three regimes":
            length = int(rng.choice([60, 70, 80]))
        else:
            length = int(rng.integers(70, 101))
        path += [regime] * length
        regime = (regime + 1) % len(MODELS[model])
    regimes = np.array(path[:n_steps])
    x = np.zeros((n_steps, 2))
    e = rng.standard_normal((n_steps, 2))
    for t in range(n_steps):
        x[t] = e[t]
        for (j, i, lag), c in MODELS[model][regimes[t]].items():
            if t >= lag:
                x[t, j] += c * x[t - lag, i]
    return x, regimes


def truth(model: str, k: int) -> np.ndarray:
    """Regime ``k``'s ``(TAU_MAX, 2, 2)`` lagged graph, ``[lag - 1, cause, effect]``."""
    graph = np.zeros((TAU_MAX, 2, 2), dtype=bool)
    for j, i, lag in MODELS[model][k]:
        graph[lag - 1, i, j] = True
    return graph


def reference(folder: Path, starts: int, threshold: float) -> None:
    """Fit every case in ``folder`` with tigramite's RPCMCI; runs in tigramite's environment."""
    from tigramite import data_processing
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.rpcmci import RPCMCI

    for case in sorted(folder.glob("*.npz")):
        d = np.load(case)
        rows = []
        for a in range(starts):
            fit = RPCMCI(
                data_processing.DataFrame(d["x"]), cond_ind_test=ParCorr(), seed=a, verbosity=-1
            )
            res = fit.run_rpcmci(
                num_regimes=int(d["n_regimes"]),
                max_transitions=int(d["switches"]),
                switch_thres=threshold,
                num_iterations=SETTINGS["max_iter"],
                max_anneal=1,
                tau_min=1,
                tau_max=TAU_MAX,
                pc_alpha=SETTINGS["pc_alpha"],
                alpha_level=SETTINGS["alpha_level"],
                n_jobs=1,
            )
            if res is None:
                rows.append(None)
                continue
            graphs = [
                (res["causal_results"][k]["graph"] == "-->").transpose(2, 0, 1)[1:].tolist()
                for k in range(int(d["n_regimes"]))
            ]
            changes = [float(c) for c in res["diff_g_f"][1]]
            rows.append({"weights": res["regimes"].tolist(), "graphs": graphs, "changes": changes})
        case.with_suffix(".tigramite.json").write_text(json.dumps(rows))


def andrey_fits(x: np.ndarray, n_regimes: int, switches: int, starts: int, threshold: float):
    """Andrey's fits of one series, one start per seed; ``None`` for a failed start."""
    from andrey.temporal.rpcmci import RegimeFitError, rpcmci

    rows = []
    for a in range(starts):
        try:
            r = rpcmci(
                x,
                n_regimes=n_regimes,
                max_transitions=switches,
                switch_threshold=threshold,
                n_starts=1,
                tau_max=TAU_MAX,
                fdr_method="none",
                indep_test="student_t",
                seed=a,
                **SETTINGS,
            )
        except RegimeFitError:
            rows.append(None)
            continue
        graphs = [res.graph.astype(bool).tolist() for res in r.results]
        rows.append({"weights": r.weights.tolist(), "graphs": graphs, "changes": list(r.changes)})
    return rows


def regime_path(weights) -> np.ndarray:
    """The regime of each step: its largest weight, rounded as ``andrey.RegimeOutput`` rounds it."""
    from andrey.core.ties import KEY_DP

    return np.round(np.asarray(weights), KEY_DP).argmax(axis=0)


def against_truth(row: dict, model: str, regimes: np.ndarray) -> tuple[float, int]:
    """Regime accuracy after the first ``TAU_MAX`` steps under the best relabeling, and the lagged
    SHD summed over the regimes so matched."""
    path = regime_path(row["weights"])
    perms = list(itertools.permutations(range(len(MODELS[model]))))
    accuracy = [(np.array(p)[path] == regimes)[TAU_MAX:].mean() for p in perms]
    best = perms[int(np.argmax(accuracy))]
    shd = sum(int((np.array(g) != truth(model, k)).sum()) for g, k in zip(row["graphs"], best))
    return float(max(accuracy)), shd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference:
        reference(args.folder, args.starts, args.threshold)
        return
    args.folder.mkdir(parents=True, exist_ok=True)
    cases = {}
    for m, model in enumerate(MODELS):
        for s in range(args.seeds):
            x, regimes = simulate(model, args.steps, 1000 + s)
            name = f"m{m}_s{s}"
            np.savez(
                args.folder / f"{name}.npz",
                x=x,
                n_regimes=len(MODELS[model]),
                switches=int((np.diff(regimes) != 0).sum()),
            )
            cases[name] = (model, x, regimes)
    python = os.environ["ANDREY_BENCH_TIGRAMITE_PYTHON"]
    command = [python, __file__, str(args.folder), "--reference"]
    command += ["--starts", str(args.starts), "--threshold", str(args.threshold)]
    subprocess.run(command, check=True)
    tally: dict[str, dict] = {}
    for name, (model, x, regimes) in cases.items():
        t_rows = json.loads((args.folder / f"{name}.tigramite.json").read_text())
        a_rows = andrey_fits(
            x, len(MODELS[model]), int((np.diff(regimes) != 0).sum()), args.starts, args.threshold
        )
        row = tally.setdefault(
            model,
            {
                "same": 0,
                "identical": 0,
                "differ": 0,
                "both fail": 0,
                "one fails": 0,
                "andrey": [],
                "tigramite": [],
            },
        )
        for a, t in zip(a_rows, t_rows, strict=True):
            for side, r in (("andrey", a), ("tigramite", t)):
                row[side].append(None if r is None else against_truth(r, model, regimes))
            if a is None or t is None:
                row["both fail" if a is None and t is None else "one fails"] += 1
                continue
            wa, wt = np.asarray(a["weights"])[:, TAU_MAX:], np.asarray(t["weights"])[:, TAU_MAX:]
            same = (regime_path(wa) == regime_path(wt)).all() and a["graphs"] == t["graphs"]
            changes = len(a["changes"]) == len(t["changes"]) and np.allclose(
                a["changes"], t["changes"], rtol=0, atol=1e-6
            )
            row["same" if same else "differ"] += 1
            row["identical"] += bool(same and changes and np.abs(wa - wt).max() <= 1e-9)

    def summary(fits: list) -> str:
        ok = [f for f in fits if f is not None]
        if not ok:
            return f"-, -, {len(fits)} of {len(fits)}"
        return (
            f"{np.mean([f[0] for f in ok]):.2f}, {np.mean([f[1] for f in ok]):.2f}, "
            f"{len(fits) - len(ok)} of {len(fits)}"
        )

    header = [
        "model",
        "same final result",
        "of them identical",
        "differ",
        "both fail",
        "one fails",
        "Andrey: accuracy, SHD, failed",
        "tigramite: accuracy, SHD, failed",
    ]
    print("| " + " | ".join(header) + " |\n|" + "---|" * len(header))
    counts = ["same", "identical", "differ", "both fail", "one fails"]
    tally["all"] = {c: sum(r[c] for r in tally.values()) for c in counts}
    for side in ("andrey", "tigramite"):
        tally["all"][side] = [f for model in MODELS for f in tally[model][side]]
    for model, r in tally.items():
        cells = [model, r["same"], r["identical"], r["differ"], r["both fail"], r["one fails"]]
        cells += [summary(r["andrey"]), summary(r["tigramite"])]
        print("| " + " | ".join(map(str, cells)) + " |")


if __name__ == "__main__":
    sys.exit(main())
