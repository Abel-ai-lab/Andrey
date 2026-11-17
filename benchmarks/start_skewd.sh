#!/usr/bin/env bash
# Fetches the authors' SkewD code and builds the environment it runs in, the reference of the
# SkewD lane.
#
#   benchmarks/start_skewd.sh    # then: skewd_pairs.py, with the exports this script prints
#
# SkewD is causality-lab/skewd at the commit below (MIT). The benchmark runs it unmodified as a
# measurement reference and copies none of it: the script checks out its `src` package and its
# skew-noise data sets under `$ANDREY_BENCH_SKEWD` (default `benchmarks/.skewd`), and the adapter
# imports them from there. requirements-skewd.txt pins every package of the environment,
# `.venv-skewd`: the code asks for NumPy 1, `cmaes`, and `scikit-optimize`, and its graph version
# runs causal-learn's PC. The environment imports none of Andrey; the driver scores the results in
# bench-env. Export what it prints before running skewd_pairs.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_SKEWD:-$REPO/benchmarks/.skewd}"
VENV="$REPO/benchmarks/.venv-skewd"
LOCK="$REPO/benchmarks/requirements-skewd.txt"
COMMIT="312632b848a22a1eca22320204b0657154bcbb2c"

echo ">> SkewD $COMMIT"
if [[ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null)" != "$COMMIT" ]]; then
    git clone --filter=blob:none --no-checkout https://github.com/causality-lab/skewd.git "$DEST"
    git -C "$DEST" sparse-checkout set src data/skew data/MVskew6 data/MVskew10
    git -C "$DEST" checkout "$COMMIT"
fi
git -C "$DEST" diff --quiet "$COMMIT" -- src || { echo "$DEST/src differs from $COMMIT" >&2; exit 1; }

echo ">> skewd-env"
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu -r "$LOCK"
sha256sum "$LOCK"

# One small fit of the pair method, both criteria, with tiny budgets.
ANDREY_BENCH_SKEWD="$DEST" "$VENV/bin/python" - <<'PY'
import importlib.metadata as m
import os
import sys

sys.path.insert(0, os.environ["ANDREY_BENCH_SKEWD"])
import numpy as np

from src.skewd import skewd

rng = np.random.default_rng(0)
x = rng.standard_normal(200)
y = np.tanh(x) + (0.5 + 0.2 * np.tanh(x)) * rng.standard_normal(200)
z = np.column_stack([x, y])
z = (z - z.mean(axis=0)) / z.std(axis=0)
out = skewd(z[:, :1], z[:, 1:], cma_iter=20, n_calls_boptim=4, n_inits_boptim=3,
            n_cv_splits_boptim=2, ecm_max_iter=5, ecm_max_iter_rho=5)
print(
    f"skewd-env OK numpy {np.__version__} cmaes {m.version('cmaes')} scikit-optimize "
    f"{m.version('scikit-optimize')} | it {int(out[1])} ll {int(out[3])}"
)
PY

printf 'export ANDREY_BENCH_SKEWD_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_SKEWD=%q\n' "$DEST"
echo ">> done"
