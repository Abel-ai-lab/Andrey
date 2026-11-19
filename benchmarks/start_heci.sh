#!/usr/bin/env bash
# Fetches the authors' HECI code and builds the environment it runs in, the reference of the HECI
# solution (``xu.heci``).
#
#   benchmarks/start_heci.sh    # then: a pair lane, with the exports this script prints
#
# The code is `heci-v20220729.zip`, the supplementary upload of Xu, Mian, and Vreeken (ICML 2022)
# from the authors' project page. The archive and its page carry no license statement for HECI's
# own files. The benchmark runs `HECI.py` unmodified as a measurement reference and redistributes
# none of it. The script downloads the archive, checks its SHA-256, extracts the two files a fit
# needs (`HECI.py` and `README.txt`) under `$ANDREY_BENCH_HECI_DIR` (default `benchmarks/.heci`),
# checks `HECI.py`'s SHA-256, and builds `.venv-heci` beside it: the authors' README names NumPy as
# its one dependency, and the environment pins NumPy 1.26.4, the last release line before 2.0, as
# their 2021-2022 environment (NumPy 1.17) predates 2.0's changes to scalar conversion. The
# environment imports none of Andrey. Export what it prints before running a pair lane.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
DEST="${ANDREY_BENCH_HECI_DIR:-$REPO/benchmarks/.heci}"
VENV="${ANDREY_BENCH_HECI_VENV:-$DEST/venv}"
URL="https://vreeken.groups.cispa.de/prj/heci/heci-v20220729.zip"
SHA256="ef0f6b480c775749b803b9e4a938247294171afdfffae6614aac18295a74de5f"
HECI_SHA256="82f89d8cb3efa4d571ef4613ececa65a4e569fe67a33560a5d79ed7ab0beb056"
CODE="$DEST/HECI_supplementary_upload"

echo ">> HECI heci-v20220729.zip"
mkdir -p "$DEST"
if [[ ! -f "$DEST/heci-v20220729.zip" ]]; then
    curl -fsSL -o "$DEST/heci-v20220729.zip.part" "$URL"
    mv "$DEST/heci-v20220729.zip.part" "$DEST/heci-v20220729.zip"
fi
echo "$SHA256  $DEST/heci-v20220729.zip" | sha256sum --check --status \
    || { echo "heci-v20220729.zip does not match $SHA256" >&2; exit 1; }
unzip -q -o "$DEST/heci-v20220729.zip" \
    HECI_supplementary_upload/HECI.py HECI_supplementary_upload/README.txt -d "$DEST"
echo "$HECI_SHA256  $CODE/HECI.py" | sha256sum --check --status \
    || { echo "HECI.py does not match $HECI_SHA256" >&2; exit 1; }
echo "$SHA256" > "$CODE/ARCHIVE_SHA256"

echo ">> heci-env"
start=$SECONDS
uv venv "$VENV" --python 3.11 --clear
uv pip install --python "$VENV/bin/python" "numpy==1.26.4"
echo ">> environment built in $((SECONDS - start)) s"

# One fit of the adapter on a small pair.
ANDREY_BENCH_HECI="$CODE" "$VENV/bin/python" -I - "$REPO/benchmarks" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
import numpy as np

from andrey_bench.adapters.heci_xu import XuHECI

rng = np.random.default_rng(0)
x = rng.standard_normal(300)
pair = np.column_stack([x, x**3 + rng.standard_normal(300)])
adapter = XuHECI()
adapter.setup()
print("heci-env OK numpy", np.__version__, "| fit reads", adapter.fit(pair, adapter.params()).tolist())
PY

printf 'export ANDREY_BENCH_HECI_PYTHON=%q\n' "$VENV/bin/python"
printf 'export ANDREY_BENCH_HECI=%q\n' "$CODE"
echo ">> done"
