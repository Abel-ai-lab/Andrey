#!/usr/bin/env bash
# Fetches the authors' R code for CPCM, the reference for Andrey's CPCM, and installs the R packages
# it loads.
#
#   benchmarks/start_r.sh       # first: the R lane
#   benchmarks/start_cpcm.sh    # then: the authors' code, read from disk by the benchmark
#
# The code is `CPCM_function.R` of https://github.com/jurobodik/Causal_CPCM, the supplementary code
# of Bodik and Chavez-Demoulin (2025), at the commit below. The repository has no license file, so
# the benchmark `source()`s the file from the directory this script prints and copies none of it.
# The script clones into `$ANDREY_BENCH_CPCM_R_DIR` (default `benchmarks/.cpcm-r`), checks the
# commit, and installs the CRAN packages the file loads into the R lane's library
# (`$ANDREY_BENCH_R_LIBS`, `$ANDREY_BENCH_R_MODULE`); export what it prints before running the CPCM
# lane.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-bench}"
DEST="${ANDREY_BENCH_CPCM_R_DIR:-$REPO/benchmarks/.cpcm-r}"
URL="https://github.com/jurobodik/Causal_CPCM"
COMMIT="2ada90e342d75500494d692fdc86be93044ed659"

if [[ ! -f "$DEST/CPCM_function.R" ]]; then
    [[ -e "$DEST" ]] && [[ -n "$(ls -A "$DEST")" ]] \
        && { echo "$DEST exists and is not a CPCM checkout" >&2; exit 1; }
    git clone --quiet "$URL" "$DEST"
fi
git -C "$DEST" checkout --quiet --detach "$COMMIT"
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] \
    || { echo "$DEST is not at commit $COMMIT" >&2; exit 1; }
[[ -z "$(git -C "$DEST" status --porcelain)" ]] \
    || { echo "$DEST has local changes; the file must be as the authors wrote it" >&2; exit 1; }

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
# `dHSIC` and `dplyr` compile C++ code and take most of the time. Only the missing packages are
# installed, so packages used in earlier measurements are not rebuilt.
Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS_USER")
    .libPaths(lib)
    options(repos = c(CRAN = "https://cloud.r-project.org"),
            Ncpus = max(1L, parallel::detectCores()))
    pkgs <- c("mgcv", "dHSIC", "bnlearn", "MASS", "gamlss", "stringr", "dplyr", "gtools")
    need <- Filter(function(p) length(find.package(p, quiet = TRUE)) == 0L, pkgs)
    if (length(need)) install.packages(need, lib = lib)
    for (p in pkgs) cat(p, as.character(packageVersion(p)), "\n")
'
echo "CPCM_function.R at commit $COMMIT"
printf 'export ANDREY_BENCH_CPCM_R=%q\n' "$DEST"
