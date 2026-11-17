#!/usr/bin/env bash
# Builds every harness environment, then runs `benchmarks/tests` in them: the harness step of a
# release freeze. CI runs only two of the files, because the rest need these environments.
#
#   benchmarks/run_tests.sh [pytest arguments]
#
# The script runs `start.sh`, `start_r.sh`, and then every other `start_*.sh` in name order. A
# failed build is reported and the run goes on: its lane's tests skip or fail, and the exit status
# is 1. Each `export ANDREY_BENCH_*` line a start script prints is applied to the test run unless
# the variable is already set, so a lane can be pointed at a build of your own by setting its
# variable first. The run requires R (`ANDREY_BENCH_REQUIRE_R=1`; set it empty to let the R lane
# skip, and a failed R build, `start_r.sh` or `start_*_r.sh`, is then reported without failing the
# run) and takes the tools no start script fetches from the environment: `ANDREY_BENCH_TETRAD_JAR`
# and `ANDREY_BENCH_TETRAD_GIN_JAR` (the Tetrad jars the lane pages name) and `ANDREY_BENCH_JAVA`.
# A test whose variable is unset skips, and the summary lists it.
#
# Logs go to `$ANDREY_BENCH_TEST_LOGS` (default `benchmarks/.test-logs`): one log per step,
# `summary.txt` (steps, variables, failures, skips), and `env.sh`, the variables the run set.
# `source env.sh` and run `pytest` to rerun a test without rebuilding.

set -uo pipefail

REPO="$(git rev-parse --show-toplevel)"
LOGS="${ANDREY_BENCH_TEST_LOGS:-$REPO/benchmarks/.test-logs}"
mkdir -p "$LOGS"
LOGS="$(cd "$LOGS" && pwd)"  # a relative path names the caller's directory, not the repository
SUMMARY="$LOGS/summary.txt"
: > "$LOGS/env.sh"
: > "$SUMMARY"
cd "$REPO" || exit 1

: "${ANDREY_BENCH_REQUIRE_R=1}"
export ANDREY_BENCH_REQUIRE_R
FAILED=0

# Applies the export lines on stdin that name a harness variable the caller has not set.
apply_exports() {
    local line name
    while IFS= read -r line; do
        [[ $line =~ ^export\ (ANDREY_BENCH_[A-Z0-9_]+)= ]] || continue
        name="${BASH_REMATCH[1]}"
        [[ -n "${!name:-}" ]] && continue
        eval "$line"
        echo "$line" >> "$LOGS/env.sh"
    done
}

# Runs one start script; a second argument, when set, keeps its failure from failing the run.
step() {
    local name="$1" optional="${2:-}" start rc
    start=$(date +%s)
    bash "benchmarks/$name.sh" > "$LOGS/$name.log" 2>&1
    rc=$?
    echo "step $name rc=$rc secs=$(($(date +%s) - start)) :: $(tail -1 "$LOGS/$name.log" | cut -c1-160)" \
        | tee -a "$SUMMARY"
    apply_exports < "$LOGS/$name.log"
    if [[ $rc -ne 0 && -n $optional ]]; then
        echo "step $name: not counted, ANDREY_BENCH_REQUIRE_R is empty" | tee -a "$SUMMARY"
    elif [[ $rc -ne 0 ]]; then
        FAILED=1
    fi
}

R_OPTIONAL=""
[[ -n $ANDREY_BENCH_REQUIRE_R ]] || R_OPTIONAL=1
step start
step start_r "$R_OPTIONAL"
for script in benchmarks/start_*.sh; do
    name="$(basename "$script" .sh)"
    if [[ $name == *_r ]]; then
        [[ $name == start_r ]] || step "$name" "$R_OPTIONAL"
    else
        step "$name"
    fi
done

# The R modules and library every process that fits an R solution needs. lmod's init reads
# variables that may be unset, so `set -u` is off around it.
set +u
if ! command -v module > /dev/null && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
fi
R_ENV="$(bash benchmarks/start_r.sh --env)"
eval "$R_ENV" >> "$LOGS/start_r_env.log" 2>&1
if command -v Rscript > /dev/null; then  # env.sh repeats the R setup for a rerun
    {
        echo 'if ! command -v module > /dev/null && [[ -f /usr/share/lmod/lmod/init/bash ]]; then'
        echo '    source /usr/share/lmod/lmod/init/bash'
        echo 'fi'
        echo "$R_ENV"
    } >> "$LOGS/env.sh"
fi
set -u

# The environments whose start script prints no export are where the README puts them.
for lane in XGES:xges TIGRAMITE:tigramite LINGAM:lingam LINGAM_TS:lingam-ts CAUSALNEX:causalnex \
    UTIGSP:utigsp GML_DEF42:gml-def42 GNIES:gnies; do
    name="ANDREY_BENCH_${lane%%:*}_PYTHON"
    python="$REPO/benchmarks/.venv-${lane#*:}/bin/python"
    if [[ -z "${!name:-}" && -e $python ]]; then
        export "$name=$python"
        echo "export $name=$python" >> "$LOGS/env.sh"
    fi
done

# CAM and RESIT fit in the same environment as the build under test.
[[ -n "${ANDREY_BENCH_PYTHON:-}" ]] || export ANDREY_BENCH_PYTHON="$REPO/benchmarks/.venv-bench/bin/python"
export ANDREY_BENCH_CAM_PYTHON="${ANDREY_BENCH_CAM_PYTHON:-$ANDREY_BENCH_PYTHON}"
export ANDREY_BENCH_RESIT_PYTHON="${ANDREY_BENCH_RESIT_PYTHON:-$ANDREY_BENCH_PYTHON}"
export ANDREY_BENCH_PYTHON

{
    echo "-- variables (a path that does not exist is flagged MISSING)"
    for name in $(env | grep -oE '^ANDREY_BENCH_[A-Z0-9_]+' | sort); do
        value="${!name}"
        flag=""
        if [[ $name == *_PYTHON || $name == *_JAR || $name == *_JAVA ]]; then
            [[ -e $value ]] || flag=" MISSING"
        fi
        echo "$name=$value$flag"
    done
    echo "-- tests"
} >> "$SUMMARY"

# COLUMNS keeps pytest from truncating the FAILED lines to the terminal width.
COLUMNS=250 "$ANDREY_BENCH_PYTHON" -m pytest benchmarks/tests -q -rfEs --tb=short \
    -p no:cacheprovider "$@" > "$LOGS/pytest.log" 2>&1
RC=$?
{
    echo "pytest rc=$RC: $(grep -E ' in [0-9.]+s' "$LOGS/pytest.log" | tail -1)"
    echo "-- FAILED / ERROR"
    grep -E '^(FAILED|ERROR) ' "$LOGS/pytest.log" | cut -c1-300
    echo "-- SKIPPED (reasons)"
    grep -E '^SKIPPED ' "$LOGS/pytest.log" | cut -c1-300
} >> "$SUMMARY"

[[ $RC -eq 0 && $FAILED -eq 0 ]] || { echo "FAILED: see $SUMMARY" | tee -a "$SUMMARY"; exit 1; }
echo "ok: see $SUMMARY"
