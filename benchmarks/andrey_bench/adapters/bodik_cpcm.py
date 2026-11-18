"""CPCM adapter for the authors' R code (Bodik and Chavez-Demoulin, 2025), run unmodified.

The code is ``CPCM_function.R`` of ``jurobodik/Causal_CPCM`` at commit :data:`COMMIT`, a repository
with no license file. It is a measurement reference only: ``benchmarks/start_cpcm.sh`` clones it,
checks the commit, and installs the R packages it loads; the benchmark ``source()``s the file from
the directory ``$ANDREY_BENCH_CPCM_R`` names and copies none of it. Fits run in ``Rscript`` through
``cpcm_worker.R``, one launch per batch of tables.

A fit calls ``CPCM_graph_estimate(X, family_of_distributions, greedy_method, lambda)`` with every
argument passed, so no default of the code decides a measurement:

- ``family_of_distributions`` is one family name for every column (:data:`R_FAMILY` maps Andrey's
  names to the code's), the joint collection ``1`` (Gaussian with fixed sigma, Poisson,
  Exponential, Pareto) or ``2`` (Gaussian, Negative_binomial, Gamma, Pareto2), under which the code
  picks each column's family from its values, or ``"Sequential choice"``.
- ``greedy_method`` and ``lambda`` matter only for more than two columns: the greedy search of the
  graph estimate and its per-edge penalty. Two columns always give the forced bivariate decision.

Two columns return the code's forced estimate: the direction with the larger p-value of the
residual-independence test, ``1 --> 2`` on a tie. The p-values reach the caller rounded to six
digits, as the code rounds them, so two printed zeros cannot be compared. More columns return the
DAG the greedy search ends at, which carries no p-values or families. The code draws random numbers
only for discrete families (randomized residuals), so each fit seeds R's generator first.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

#: Names the directory holding ``CPCM_function.R``.
CODE_ENV = "ANDREY_BENCH_CPCM_R"

#: The pinned commit of ``jurobodik/Causal_CPCM``; ``start_cpcm.sh`` fetches it.
COMMIT = "2ada90e342d75500494d692fdc86be93044ed659"

#: The files a fit reads, relative to the code directory.
REQUIRED_FILES = ("CPCM_function.R",)

#: The R worker that sources the code and fits the rows of a manifest.
WORKER = Path(__file__).with_name("cpcm_worker.R")

#: Andrey's family names and the names ``CPCM_graph_estimate`` uses for them.
R_FAMILY = {
    "gaussian": "Gaussian",
    "gaussian_fixed_scale": "Gaussian with fixed sigma",
    "exponential": "Exponential",
    "pareto": "Pareto",
    "poisson": "Poisson",
    "gamma": "Gamma",
    "gamma_fixed_shape": "Gamma with fixed scale",
    "gumbel": "Gumbel",
    "gumbel_fixed_scale": "Gumbel with fixed scale",
    "negative_binomial": "Negative_binomial",
}

#: The joint collections and the sequential choice, which the code takes as they are.
_COLLECTIONS = ("1", "2", "Sequential choice")

#: The manifest's columns, in the order ``cpcm_worker.R`` documents them.
_MANIFEST = ("id", "table", "family", "greedy_method", "lambda", "seed", "warmup")


def r_family(family: str | int) -> str:
    """The code's ``family_of_distributions`` for an Andrey family name, a name of the code's own,
    or a joint collection (``1``, ``2``, ``"Sequential choice"``), as a string.

    Raises ``ValueError`` for anything else.
    """
    name = str(family)
    if name in R_FAMILY:
        return R_FAMILY[name]
    if name in R_FAMILY.values() or name in _COLLECTIONS or name == "Pareto2":
        return name
    raise ValueError(f"unknown CPCM family {family!r}")


def code_dir() -> Path:
    """The authors' code directory, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' CPCM code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_cpcm.sh and export what it prints."
        )
    root = Path(raw)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks {', '.join(missing)}")
    if shutil.which("Rscript") is None:
        raise SystemExit("Rscript is not on PATH: load the R module (benchmarks/start_r.sh --env)")
    return root


def _number(text: str) -> float:
    """A worker field as a float; empty and ``NA`` are not a number."""
    return float("nan") if text in ("", "NA") else float(text)


def _parse(row: Mapping[str, str]) -> dict[str, Any]:
    """One worker result row as a typed dict."""
    out: dict[str, Any] = {
        "id": row["id"],
        "status": row["status"],
        "seconds": _number(row["seconds"]),
        "error": None if row["error"] in ("", "NA") else row["error"],
    }
    if row["status"] != "ok":
        return out
    if row["adjacency"] not in ("", "NA"):
        cells = [int(float(v)) for v in row["adjacency"].split(";")]
        d = round(len(cells) ** 0.5)
        out["adjacency"] = np.array(cells, dtype=np.int8).reshape(d, d)
    else:
        out["forced"] = row["forced"]
        out["p_empty"] = _number(row["p_empty"])
        out["p_12"] = _number(row["p_12"])
        out["p_21"] = _number(row["p_21"])
        out["score_estimate"] = row["score"]
        out["testing_estimate"] = row["testing"]
        out["families"] = row["families"].split(";")
    return out


def run_fits(
    rows: Sequence[Mapping[str, Any]], out_dir: str | Path, cores: int = 1
) -> list[dict[str, Any]]:
    """Fit every manifest row in one ``Rscript`` launch and return the parsed results, in order.

    Each row names an ``id``, a CSV ``table`` with a header, a ``family`` (see :func:`r_family`),
    a ``greedy_method``, a ``lambda``, a ``seed``, and ``warmup`` untimed calls. One result file per
    fit lands in `out_dir`, and a fit whose file is already there is not run again.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST)
        writer.writeheader()
        writer.writerows({k: r[k] for k in _MANIFEST} for r in rows)
    env = {**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    command = ["Rscript", "--vanilla", str(WORKER), str(code_dir()), str(manifest), str(out)]
    done = subprocess.run([*command, str(cores)], capture_output=True, text=True, env=env)
    if done.returncode != 0:
        raise RuntimeError(f"cpcm_worker.R exited {done.returncode}: {done.stderr[-2000:]}")
    results = []
    for row in rows:
        path = out / f"{row['id']}.csv"
        if not path.is_file():
            raise RuntimeError(
                f"cpcm_worker.R wrote no result for {row['id']}: {done.stderr[-2000:]}"
            )
        with path.open(newline="") as handle:
            results.append(_parse(next(csv.DictReader(handle))))
    return results


def fit_table(
    data: np.ndarray,
    family: str | int,
    *,
    greedy_method: str = "RESIT_greedy",
    lambda_: float = 2.0,
    seed: int = 0,
    warmup: int = 0,
) -> dict[str, Any]:
    """Fit the authors' CPCM on one table and return the parsed result.

    The dict holds ``status``, ``seconds`` (the call alone), and ``error``. For two columns it also
    holds ``forced`` (``"1 --> 2"`` or ``"2 --> 1"``), ``p_empty``, ``p_12``, ``p_21``,
    ``score_estimate``, ``testing_estimate``, and ``families`` (the family of column 1, then of
    column 2); for more, ``adjacency``, the 0/1 DAG matrix with ``adjacency[i, j] = 1`` for
    ``i -> j``.
    """
    z = np.asarray(data, dtype=np.float64)
    header = ",".join(f"X{i + 1}" for i in range(z.shape[1]))
    with tempfile.TemporaryDirectory() as tmp:
        table = Path(tmp) / "table.csv"
        np.savetxt(table, z, delimiter=",", header=header, comments="", fmt="%.17g")
        row = {
            "id": "fit",
            "table": str(table),
            "family": r_family(family),
            "greedy_method": greedy_method,
            "lambda": lambda_,
            "seed": seed,
            "warmup": warmup,
        }
        return run_fits([row], Path(tmp) / "out")[0]


class BodikCPCM:
    """The authors' CPCM for one family setting, through ``Rscript``.

    ``fit`` returns the 0/1 DAG adjacency: for two columns the forced direction, for more the
    greedy search's DAG. Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.

    ``search_s`` and ``warmup`` follow the harness's convention for a search timed in another
    process (``contracts.SolutionAdapter``): ``fit`` sets ``search_s`` to the seconds R measured
    around the ``CPCM_graph_estimate`` call, and the worker sets ``warmup``, the calls the R process
    makes untimed before it.
    """

    package: str = "Causal_CPCM"
    package_version: str = COMMIT[:8]
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "cpcm"
    output_type: str = "dag"

    search_s: float | None = None
    warmup: int = 0

    def __init__(
        self,
        family: str | int = 1,
        *,
        greedy_method: str = "RESIT_greedy",
        lambda_: float = 2.0,
        seed: int = 0,
    ) -> None:
        self.family = r_family(family)
        self.greedy_method = greedy_method
        self.lambda_ = lambda_
        self.seed = seed
        # A run_id holds the name but not the params, so every setting off its default is named.
        self.name = "bodik.cpcm." + self.family.lower().replace(" ", "_")
        self.name += f".{greedy_method}" if greedy_method != "RESIT_greedy" else ""
        self.name += f".lambda{lambda_:g}" if lambda_ != 2.0 else ""
        self.name += f".seed{seed}" if seed != 0 else ""

    def setup(self) -> None:
        """Check the environment before the first fit: one error here, not one per unit."""
        code_dir()

    def params(self) -> dict[str, Any]:
        """Every argument of ``CPCM_graph_estimate`` the adapter sets, the seed, and the commit."""
        return {
            "commit": COMMIT,
            "family_of_distributions": self.family,
            "greedy_method": self.greedy_method,
            "lambda": self.lambda_,
            "seed": self.seed,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' CPCM; return a 0/1 DAG adjacency. The only timed region."""
        merged = {**self.params(), **params}
        result = fit_table(
            data,
            merged["family_of_distributions"],
            greedy_method=merged["greedy_method"],
            lambda_=merged["lambda"],
            seed=merged["seed"],
            warmup=self.warmup,
        )
        if result["status"] != "ok":
            raise RuntimeError(f"CPCM_graph_estimate failed: {result['error']}")
        self.search_s = float(result["seconds"])
        if "adjacency" in result:
            return result["adjacency"]
        adjacency = np.zeros((2, 2), dtype=np.int8)
        adjacency[0, 1] = result["forced"] == "1 --> 2"
        adjacency[1, 0] = not adjacency[0, 1]
        return adjacency

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
