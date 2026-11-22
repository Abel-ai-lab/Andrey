"""Agreement of Andrey's GCM, wGCM, and PCM tests with the authors' code on the papers' settings.

Each case is drawn here, put into the layout the Andrey test reads for the query of x and y given
the rest (its standardized, rounded, and signed columns, the rows in that query's order), and
written to a folder that `benchmarks/regression_ci_reference.R` reads,
so both sides test the same numbers. PCM is compared in its paper's direction, the mean of y given
(x, z). Two layers are compared:

- the statistic: Andrey's statistic and p-value computed from the residuals, weights, and sample
  split the reference produced, which isolates the test from the regressions;
- end to end: each side with its own regressions (boosted trees, random forests, or the PCM's
  additive models), the sample splits shared, so a difference comes from the regressions.

    ANDREY_BENCH_RSCRIPT=Rscript ANDREY_BENCH_PCM_CODE=<clone of the PCM authors' code> \\
        uv run python benchmarks/regression_ci_agreement.py <empty folder>

`benchmarks/start_regression_ci_r.sh` builds the two R libraries the reference needs, read from
`ANDREY_BENCH_R_LIBS` (GCM and PCM) and `ANDREY_BENCH_R_LIBS_WGCM` (wGCM) when they are set, and
clones the PCM authors' code, printing its `ANDREY_BENCH_PCM_CODE`. The R lane's start scripts use
`ANDREY_BENCH_R_LIBS` for their own library, which the `env.sh` of `benchmarks/run_tests.sh`
records, so both variables are set here to the libraries the start script built. The script prints
one Markdown table per test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

from andrey.core import ci_regression as cr

SCRIPT = Path(__file__).with_name("regression_ci_reference.R")
SEEDS = (1, 2, 3)


def _h(t: np.ndarray, b1: float, b2: float) -> np.ndarray:
    """The function family of the wGCM paper's Section 3.1, symmetry ``b1`` and wiggliness ``b2``."""
    h1 = (b1 * np.cos(3 * b2 * t) + (1 - b1) * np.sin(3 * b2 * t)) * np.exp(-(t**2) / 2)
    h2 = 0.3 * (b1 * np.abs(t) + (1 - b1) * t)
    return (1 - b2) * h2 - b2 * h1


def wgcm_case(setting: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Columns x, y, z... of the wGCM paper's settings: 1D (null) and its example (7) (power)."""
    if setting.startswith("1D"):
        b1, b2 = (float(v) for v in setting[3:].split(","))
        z = rng.normal(size=n)
        x = _h(z, b1, b2) + 0.3 * rng.normal(size=n)
        y = _h(z, b1, b2) + 0.3 * rng.normal(size=n)
    else:  # "eq7 lambda"
        lam = float(setting.split()[1])
        z = rng.normal(size=n)
        x = z + 0.3 * rng.normal(size=n)
        y = z + 0.3 * rng.normal(size=n) + 0.3 * (lam * x + 0.5 * (1 - lam) * x**2)
    return np.column_stack([x, y, z])


def pcm_case(setting: int, n: int, rng: np.random.Generator, *, forest: bool) -> np.ndarray:
    """Columns x, y, z1..z7 of the PCM authors' simulations: GAM (Section 6.1) or forest (6.2)."""
    z = rng.normal(size=(n, 7))
    if not forest:
        f = np.sin(2 * np.pi * z[:, 0])
        x = f + (0.1 if setting == 0 else 1.0) * rng.normal(size=n)
        extra = {0: 0.0, 1: 0.2 * x**2, 3: 0.4 * x**2 * z[:, 1]}[setting]
        y = f + extra + rng.normal(size=n)
    else:
        f = np.sin(np.pi * z[:, 0]) * (1 + z[:, 2])
        x = f + rng.normal(size=n)
        eps = np.sqrt(0.5 + (x > 0)) * rng.normal(size=n)
        extra = {0: 0.0, 1: 0.04 * x**2, 3: 0.04 * x**2 * z[:, 1]}[setting]
        y = f + extra + eps
    return np.column_stack([x, y, z])


def write_case(folder: Path, name: str, test: cr._RegressionTest, raw: np.ndarray) -> None:
    """Write the test's values as the reference reads them: the columns x, y, z1, ... in the sign of
    ``raw``, the data the test was built on, and the rows in the order of the query of x and y given
    the rest, so a row's position is its position there."""
    rows, _ = test._layout(range(test._data.shape[1]))
    sign = np.where(np.sum((raw - raw.mean(axis=0)) * test._data, axis=0) < 0, -1.0, 1.0)
    data = test._data[rows] * sign
    header = ",".join(["x", "y", *(f"z{j}" for j in range(1, data.shape[1] - 1))])
    np.savetxt(folder / f"{name}.csv", data, delimiter=",", header=header, comments="", fmt="%.17g")


def run_reference(folder: Path, mode: str, names: list[str], seeds: list[int]) -> list[dict]:
    """Run the reference in ``mode`` over the cases; its rows, one per case."""
    with open(folder / "cases.csv", "w") as f:
        f.write("case,seed\n" + "".join(f"{c},{s}\n" for c, s in zip(names, seeds)))
    rscript = os.environ.get("ANDREY_BENCH_RSCRIPT", "Rscript")
    lib = os.environ.get(
        "ANDREY_BENCH_R_LIBS_WGCM" if mode.startswith("wgcm") else "ANDREY_BENCH_R_LIBS"
    )
    env = {**os.environ, "R_LIBS": lib} if lib else None
    if not (folder / f"reference_{mode}.csv").exists():  # a rerun reuses the reference's output
        subprocess.run([rscript, "--vanilla", str(SCRIPT), str(folder), mode], check=True, env=env)
    rows = np.genfromtxt(
        folder / f"reference_{mode}.csv",
        delimiter=",",
        names=True,
        dtype=None,
        missing_values="NA",  # a reference call that stopped with an error
        filling_values=np.nan,
    )
    return [dict(zip(rows.dtype.names, r)) for r in np.atleast_1d(rows)]


def read(folder: Path, name: str) -> np.ndarray:
    """A structured array of a CSV the reference wrote."""
    return np.genfromtxt(folder / f"{name}.csv", delimiter=",", names=True)


def table(header: list[str], rows: list[list]) -> str:
    """A Markdown table."""
    fmt = [" | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) for r in rows]
    return "\n".join(
        ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + [f"| {r} |" for r in fmt]
    )


def gcm_and_wgcm(folder: Path) -> None:
    """GCM, wGCM.est, and wGCM.fix on the wGCM paper's settings, n = 200 and 400."""
    settings = ("1D 0,0", "1D 1,1", "eq7 0", "eq7 1")
    names, seeds, tests = [], [], {}
    for setting in settings:
        for n in (200, 400):
            for seed in SEEDS:
                name = f"w{len(names)}"
                data = wgcm_case(setting, n, np.random.default_rng([seed, n]))
                tests[name] = (setting, n, seed, data)
                write_case(folder, name, cr.GCM(data, seed=seed), data)
                names.append(name)
                seeds.append(seed)
    gcm_rows = run_reference(folder, "gcm", names, seeds)
    est_rows = run_reference(folder, "wgcm_est", names, seeds)
    fix_rows = run_reference(folder, "wgcm_fix", names, seeds)
    out_gcm, out_est, out_fix = [], [], []
    for name, g, e, f in zip(names, gcm_rows, est_rows, fix_rows):
        setting, n, seed, data = tests[name]
        res = read(folder, f"{name}_gcm")
        exact = cr._gcm_answer(res["rx"], res["ry"])
        own = cr.GCM(data, seed=seed)._statistic_pvalue(0, 1, [2])
        out_gcm.append(
            [
                setting,
                n,
                seed,
                g["p"],
                exact[1],
                abs(exact[1] - g["p"]),
                own[1],
                (own[1] < 0.05) == (g["p"] < 0.05),
            ]
        )
        w = read(folder, f"{name}_wgcm_est")
        exact = cr._estimated_answer(w["eps"] * w["xi"], w["w"])
        split = read(folder, f"{name}_split")["row"].astype(bool)
        test = cr.WGCM(data, seed=seed)
        test._parts = (np.flatnonzero(split), np.flatnonzero(~split))
        own = test._statistic_pvalue(0, 1, [2])
        out_est.append(
            [
                setting,
                n,
                seed,
                e["p"],
                exact[1],
                abs(exact[1] - e["p"]),
                own[1],
                (own[1] < 0.05) == (e["p"] < 0.05),
            ]
        )
        w = read(folder, f"{name}_wgcm_fix")
        z = read(folder, name)["z1"][:, None]
        multipliers = np.random.default_rng([seed, n, 3]).standard_normal((n, 499))
        exact = cr._fixed_answer(w["eps"] * w["xi"], z, multipliers)
        own = cr.WGCM(data, variant="fix", seed=seed)._statistic_pvalue(0, 1, [2])
        mc = np.sqrt(max(f["p"] * (1 - f["p"]), 1 / 500) / 500)
        out_fix.append(
            [
                setting,
                n,
                seed,
                f["p"],
                exact[1],
                abs(exact[1] - f["p"]) / mc,
                own[1],
                (own[1] < 0.05) == (f["p"] < 0.05),
            ]
        )
    head = [
        "setting",
        "n",
        "seed",
        "reference p",
        "Andrey p (same residuals)",
        "|difference|",
        "Andrey p (own regressions)",
        "same decision at 0.05",
    ]
    print("\n### GCM\n\n" + table(head, out_gcm))
    print("\n### wGCM.est\n\n" + table(head, out_est))
    head[5] = "|difference| / Monte Carlo sd"
    print("\n### wGCM.fix\n\n" + table(head, out_fix))


# The PCM comparisons: the reference mode, Andrey's regression, the sizes, the splits, the seeds,
# and the table's label. Least squares is the exact layer; forests (Section 6.2's designs) and
# additive models (Section 6.1's) are end to end.
PCM_MODES = (
    ("pcm_lm", "linear", (250,), 1, SEEDS, "least squares, 1 split"),
    ("pcm_rf", "forest", (250, 500), 6, SEEDS[:2], "random forests, 6 splits"),
    ("pcm_gam", "additive", (250,), 6, SEEDS[:2], "additive models, 6 splits"),
)


def pcm(folder: Path) -> None:
    """PCM: least squares, random forests, and additive models on the authors' settings."""
    for mode, regression, ns, n_splits, case_seeds, label in PCM_MODES:
        forest = regression == "forest"
        names, seeds, tests = [], [], {}
        for setting in (0, 1, 3):
            for n in ns:
                for seed in case_seeds:
                    name = f"{mode}_{len(names)}"
                    data = pcm_case(setting, n, np.random.default_rng([seed, n]), forest=forest)
                    test = cr.PCM(data, regression=regression, n_splits=n_splits, seed=seed)
                    write_case(folder, name, test, data)  # column 1 is the paper's Y
                    tests[name] = (setting, n, seed, test)
                    names.append(name)
                    seeds.append(seed)
        rows = run_reference(folder, mode, names, seeds)
        out = []
        for name, r in zip(names, rows):
            setting, n, seed, test = tests[name]
            split = read(folder, f"{name}_split")
            test._splits = [
                (np.flatnonzero(split[c].astype(bool)), np.flatnonzero(~split[c].astype(bool)))
                for c in split.dtype.names
            ]
            stat_ref = float(np.mean([-ndtri(r[f"p{b + 1}"]) for b in range(len(test._splits))]))
            p_ref = float(ndtr(-stat_ref))
            # The paper's direction, y given (x, z), whichever variable the query would make the
            # response; the last column says whether the query does.
            rows, cols = test._layout(range(test._data.shape[1]))
            block, features = test._data[rows], cols[:]
            features.remove(0)
            features.remove(1)
            mean = float(
                np.mean([test._split_statistic(block, s, 1, 0, features) for s in test._splits])
            )
            out.append(
                [
                    setting,
                    n,
                    seed,
                    stat_ref,
                    mean,
                    mean * np.sqrt((n // 2 + n % 2 - 1) / (n // 2 + n % 2)),
                    p_ref,
                    float(ndtr(-mean)),
                    # A reference call that stopped with an error gives no decision.
                    "reference failed"
                    if np.isnan(p_ref)
                    else (ndtr(-mean) < 0.05) == (p_ref < 0.05),
                    cols.index(1) > cols.index(0),
                ]
            )
        print(
            f"\n### PCM, {label}\n\n"
            + table(
                [
                    "setting",
                    "n",
                    "seed",
                    "reference mean T",
                    "Andrey mean T",
                    "Andrey T x sqrt((m-1)/m)",
                    "reference p",
                    "Andrey p",
                    "same decision at 0.05",
                    "query takes y as the response",
                ],
                out,
            )
        )


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    folder.mkdir(parents=True, exist_ok=True)
    gcm_and_wgcm(folder)
    pcm(folder)
