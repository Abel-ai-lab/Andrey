#!/usr/bin/env bash
# Fetches the authors' local ISA-LiNG code and builds the environment it runs in, the reference of
# the local LiNG lane.
#
#   benchmarks/start_local_ling.sh    # then: local_lingam_cpu.py, with the exports this script prints
#
# The code is MarkDana/Local-LiNG-Discovery at the commit below (MIT license); the benchmark runs it
# unmodified as a measurement reference from `$ANDREY_BENCH_LOCAL_LING_DIR` (default
# `benchmarks/.local-ling`). requirements-local-ling.txt pins every package of its environment,
# `.venv-local-ling`, which imports none of Andrey; the driver scores the results in bench-env.
# Export what the script prints before running local_lingam_cpu.py.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_LOCAL_LING_DIR:-$REPO/benchmarks/.local-ling}"
VENV="$REPO/benchmarks/.venv-local-ling"
LOCK="$REPO/benchmarks/requirements-local-ling.txt"
COMMIT="d7351d57c7f5204983e05afa557d378a0e47a53b"

echo ">> Local-LiNG-Discovery $COMMIT"
if [[ ! -d "$DEST/.git" ]]; then
    git clone --quiet https://github.com/MarkDana/Local-LiNG-Discovery.git "$DEST"
fi
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || { echo "not at $COMMIT" >&2; exit 1; }
git -C "$DEST" diff --quiet || { echo "$DEST has local changes" >&2; exit 1; }

echo ">> local-ling-env"
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" -r "$LOCK"
sha256sum "$LOCK"

# One fit on a chain x0 -> x1 -> x2 with the target x1, the paper's acyclic post-processing.
ANDREY_BENCH_LOCAL_LING="$DEST" "$VENV/bin/python" - "$REPO/benchmarks" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.dai_local_isa_ling import DaiLocalISALiNG
from andrey_bench.local_ling_models import unpack_answer

rng = np.random.default_rng(0)
e = rng.uniform(-1, 1, size=(2000, 4)) ** 5
x0 = e[:, 0]
x1 = 0.8 * x0 + e[:, 1]
x2 = -0.7 * x1 + e[:, 2]
answer = unpack_answer(DaiLocalISALiNG().fit(np.column_stack([x0, x1, x2, e[:, 3]]), {"target": 1, "acyclic": True}))
edges = {(int(i), int(j)): round(float(answer["weights"][i, j]), 2) for i, j in zip(*np.nonzero(answer["weights"]))}
print(f"local-ling-env OK numpy {np.__version__} | blanket {answer['blanket']} | edges {edges}")
PY

printf 'export ANDREY_BENCH_LOCAL_LING_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_LOCAL_LING=%q\n' "$DEST"
echo ">> done"
