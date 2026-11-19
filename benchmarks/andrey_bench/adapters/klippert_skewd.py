"""SkewD adapter for the authors' code (Klippert and Marx, ICML 2026), run unmodified.

The code is ``causality-lab/skewd`` at commit :data:`COMMIT` (MIT, arXiv 2511.14441): its ``src``
package holds ``skewd``. It is a measurement reference only: ``benchmarks/start_skewd.sh`` checks
the commit out and builds the environment it runs in; the benchmark imports ``src.skewd`` from the
directory ``$ANDREY_BENCH_SKEWD`` names and copies none of it.

A fit standardizes each column to mean 0 and variance 1, as the authors' experiments do, then calls
``skewd(x, y, **CAMPAIGN)``, the arguments of their skew-noise experiment script
(:data:`CAMPAIGN`): 8 cross-validation folds, 60 Bayesian-optimization calls of which 40 are
initial points, and up to 5,000 CMA-ES generations and 5,000 ECM iterations. Everything else is at
the function's defaults: ``rng_seed=192``, 12 knots for the location and 5 for the scale, cubic
splines, a CMA-ES population of 100, and Gaussian marginals for the likelihood score.

One call returns both criteria: the independence score and direction, then the likelihood score and
direction, then the two directions' fits. Each of the two classes calls ``skewd`` once and reads
its own criterion, so a campaign that runs both repeats the whole fit, and each arm's time is the
time of that fit. A direction is 1 for ``x -> y``, the first column causing the second, and 0 for
the reverse.

The code differs from the paper (Klippert and Marx 2026) in these places, and the benchmark runs
the code:

- SkewD-IT compares each direction's raw biased HSIC statistic, a Gaussian kernel at the median
  heuristic's bandwidth, of the scaled residual against the proposed cause. The score is the
  backward statistic minus the forward one; the paper decides by the p-values.
- A CMA-ES candidate becomes the best one only when its penalized negative log-likelihood is at
  least 0.
- CMA-ES stops when the best penalized value has changed by less than ``1e-6`` over 200 generations,
  or by the optimizer's own rule; ECM stops when the parameters have moved by less than ``1e-6``
  over 25 iterations, or at ``ecm_max_iter``.
- The function's budgets are 30 Bayesian-optimization calls of which 20 are initial points, and 5
  cross-validation folds; the 60 and 40 and the 8 folds are the experiment scripts'.

The edge follows the direction: ``0 -> 1`` for 1 and ``1 -> 0`` for 0. A score of exactly 0 or not a
number gives no edge, though the code reports direction 0 there because ``score > 0`` is false, so
that a fit with no usable score is credited to neither direction.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.immer_loci import standardized

#: Names the directory holding the checkout of the authors' code.
CODE_ENV = "ANDREY_BENCH_SKEWD"

#: The pinned commit of ``causality-lab/skewd``; ``start_skewd.sh`` checks it out.
COMMIT = "312632b848a22a1eca22320204b0657154bcbb2c"

#: The arguments of the authors' skew-noise experiment script, which ``skewd`` takes beyond the
#: data. A fit reads them from its ``params``.
CAMPAIGN = {
    "cma_iter": 5000,
    "n_cv_splits_boptim": 8,
    "ecm_max_iter": 5000,
    "ecm_max_iter_rho": 100,
    "n_calls_boptim": 60,
    "n_inits_boptim": 40,
}

#: The arguments of the untimed warm-up fit in ``setup``; at :data:`CAMPAIGN`'s a fit takes hours.
WARMUP = {
    "cma_iter": 20,
    "n_cv_splits_boptim": 2,
    "ecm_max_iter": 5,
    "ecm_max_iter_rho": 5,
    "n_calls_boptim": 4,
    "n_inits_boptim": 3,
}

#: The files a fit imports, relative to the code directory.
REQUIRED_FILES = (
    "src/__init__.py",
    "src/skewd.py",
    "src/sncma.py",
    "src/snecm.py",
    "src/snlik.py",
    "src/hsic.py",
    "src/utils.py",
)

#: The positions of each criterion's score and direction in what ``skewd`` returns.
_RESULT = {"independence": (0, 1), "likelihood": (2, 3)}

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW = 1, 2


def code_dir() -> Path:
    """The directory of the authors' code, or ``SystemExit`` naming what is missing."""
    raw = os.environ.get(CODE_ENV, "").strip()
    if not raw:
        raise SystemExit(
            f"${CODE_ENV} is unset: the authors' SkewD code is a measurement reference that the "
            "benchmark reads from disk. Run benchmarks/start_skewd.sh and export what it prints."
        )
    root = Path(raw)
    missing = [f for f in REQUIRED_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"${CODE_ENV}={raw!r} lacks {', '.join(missing)}")
    return root


def _skewd():
    """The authors' ``skewd`` function, imported from :func:`code_dir`."""
    root = str(code_dir())
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.skewd import skewd  # ty: ignore[unresolved-import]  # the checkout, on sys.path

    return skewd


def edge_marks(score: float, direction: int) -> np.ndarray:
    """The two-node DAG's endpoint marks: direction 1 is ``0 -> 1``, 0 is ``1 -> 0``.

    A `score` of 0 or not a number gives no edge.
    """
    marks = np.zeros((2, 2), dtype=np.int8)
    if np.isnan(score) or score == 0:
        return marks
    cause, effect = (0, 1) if direction == 1 else (1, 0)
    marks[cause, effect], marks[effect, cause] = _TAIL, _ARROW
    return marks


class KlippertSkewD:
    """The authors' SkewD under the independence criterion (SkewD-IT).

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    name: str = "klippert.skewd.independence"
    package: str = "skewd"
    package_version: str = COMMIT[:8]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "skewd"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_SKEWD_PYTHON"
    criterion: str = "independence"

    def setup(self) -> None:
        """Import the code and fit a tiny pair before timing, so first-call costs stay out.

        The warm-up fit runs :data:`WARMUP`'s arguments, which reach every code path of a fit; the
        timed fit runs :data:`CAMPAIGN`'s.
        """
        from andrey_bench.adapters.fcm_warm import tiny_table

        z = standardized(tiny_table()[:, :2])
        _skewd()(z[:, :1], z[:, 1:], **WARMUP)

    def params(self) -> dict[str, Any]:
        """The arguments of the call, the criterion read, the standardization, and the commit."""
        return {
            **CAMPAIGN,
            "commit": COMMIT,
            "criterion": self.criterion,
            "standardize": True,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the authors' SkewD; return the two-node DAG's marks. The only timed region."""
        skewd = _skewd()
        z = standardized(data)
        out = skewd(z[:, :1], z[:, 1:], **{name: params[name] for name in CAMPAIGN})
        score, direction = (out[i] for i in _RESULT[self.criterion])
        return edge_marks(float(score), int(direction))

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the two-node DAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class KlippertSkewDLikelihood(KlippertSkewD):
    """The authors' SkewD under the likelihood criterion (SkewD-LL)."""

    name: str = "klippert.skewd.likelihood"
    criterion: str = "likelihood"
