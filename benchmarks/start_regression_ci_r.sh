#!/usr/bin/env bash
# Installs the authors' code of the GCM, wGCM, and PCM tests, the references of Andrey's "gcm",
# "wgcm", and "pcm", for `benchmarks/regression_ci_agreement.py`.
#
#   benchmarks/start_regression_ci_r.sh [folder for the PCM simulation code]
#
# GeneralisedCovarianceMeasure 0.2.0 left CRAN, so it comes from the CRAN archive with xgboost
# 1.7.8.1, a release whose interface it calls; weightedGCM 0.1.2 needs xgboost 3, so the two go into
# two libraries, each tarball with its SHA-256 checked. The PCM authors' simulation code, which
# loads vimp and the archived CondIndTests, is cloned at a fixed commit into the folder given, else
# `$ANDREY_BENCH_PCM_CODE` (default `benchmarks/.pcm-code`); it carries no license, so it is never
# committed. The script prints the `ANDREY_BENCH_PCM_CODE` export the agreement script reads, which
# `benchmarks/run_tests.sh` applies and records. The libraries are `$ANDREY_BENCH_R_LIBS`
# and, for wGCM, `$ANDREY_BENCH_R_LIBS_WGCM`; the agreement script reads both variables.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-regression-ci}"
R_LIBS_WGCM="${ANDREY_BENCH_R_LIBS_WGCM:-$REPO/benchmarks/.r-lib-regression-ci-wgcm}"
PCM_CODE="${1:-${ANDREY_BENCH_PCM_CODE:-$REPO/benchmarks/.pcm-code}}"
CONTRIB="https://cran.r-project.org/src/contrib"
ARCHIVE="$CONTRIB/Archive"
TARBALLS=(
    "GeneralisedCovarianceMeasure/GeneralisedCovarianceMeasure_0.2.0.tar.gz 8ea9f6e913be11b5d2725d9aad27c523190963e4c5b70ecbadb68e7e99abfd75"
    "xgboost/xgboost_1.7.8.1.tar.gz 394d6fd00b2fe97549c7a7e6598df86448cdfbd7c0af45b0c17f7b9e81bc1be9"
    "CondIndTests/CondIndTests_0.1.5.tar.gz 55f3589c4d317ac6e909889cb0bb6e9105be187362c25b6f78ba166939faa44e"
)
# The wGCM authors' current release and the xgboost release it was measured with, from CRAN or, once
# superseded, its archive.
WGCM_TARBALLS=(
    "weightedGCM/weightedGCM_0.1.2.tar.gz f9fdf5d0a9684532c1d7aac0d83ce08c6fafe3bdc14c258c4a1556e8f1581729"
    "xgboost/xgboost_3.2.1.1.tar.gz 54f3f59aa5689117977aab725b6bda88db71fff4b1b6d92289f81cb10bcee2a5"
)
PCM_REPO="https://github.com/ARLundborg/pcm_code.git"
PCM_COMMIT="3822659390510d65c36857f8801c893e94850774"

if [ -f /usr/share/lmod/lmod/init/bash ]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
    set -u
fi
if command -v module >/dev/null 2>&1; then
    # Unquoted on purpose: one word per module.
    # shellcheck disable=SC2086
    module load $R_MODULE
fi
mkdir -p "$R_LIBS" "$R_LIBS_WGCM"
export R_LIBS

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for entry in "${TARBALLS[@]}" "${WGCM_TARBALLS[@]}"; do
    read -r path sha <<< "$entry"
    file="$tmp/$(basename "$path")"
    curl -fsL -o "$file" "$CONTRIB/$(basename "$path")" || curl -fsSL -o "$file" "$ARCHIVE/$path"
    echo "$sha  $file" | sha256sum --check --status || { echo "$path does not match $sha" >&2; exit 1; }
done

Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS")
    tmp <- commandArgs(TRUE)[1]
    options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))
    need <- function(p) length(find.package(p, lib.loc = lib, quiet = TRUE)) == 0L
    deps <- Filter(need, c("data.table", "jsonlite", "CVST", "kernlab", "mgcv", "ranger", "MASS",
                           "vimp", "randomForest", "quantregForest", "lawstat", "RPtests",
                           "caTools", "pracma", "mize"))
    if (length(deps)) install.packages(deps, lib = lib)
    for (f in c("xgboost_1.7.8.1", "GeneralisedCovarianceMeasure_0.2.0", "CondIndTests_0.1.5"))
        install.packages(file.path(tmp, paste0(f, ".tar.gz")), lib = lib, repos = NULL, type = "source")
    for (p in c("GeneralisedCovarianceMeasure", "xgboost", "mgcv", "ranger"))
        cat(p, as.character(packageVersion(p, lib.loc = lib)), "\n")
' "$tmp"

R_LIBS="$R_LIBS_WGCM" Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS")
    tmp <- commandArgs(TRUE)[1]
    options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))
    need <- function(p) length(find.package(p, lib.loc = lib, quiet = TRUE)) == 0L
    deps <- Filter(need, c("Matrix", "data.table", "jsonlite", "mgcv"))
    if (length(deps)) install.packages(deps, lib = lib)
    # An xgboost 3.2.1.1 that the R installation already holds is used as it is.
    have <- tryCatch(packageVersion("xgboost") == "3.2.1.1", error = function(e) FALSE)
    for (f in c(if (!have) "xgboost_3.2.1.1", "weightedGCM_0.1.2"))
        install.packages(file.path(tmp, paste0(f, ".tar.gz")), lib = lib, repos = NULL, type = "source")
    stopifnot(packageVersion("xgboost") == "3.2.1.1")
    for (p in c("weightedGCM", "xgboost"))
        cat(p, as.character(packageVersion(p)), "\n")
' "$tmp"

if [ ! -d "$PCM_CODE/.git" ]; then
    git clone --quiet "$PCM_REPO" "$PCM_CODE"
fi
git -C "$PCM_CODE" checkout --quiet "$PCM_COMMIT"
echo "pcm_code $PCM_COMMIT"
printf 'export ANDREY_BENCH_PCM_CODE=%q\n' "$(cd "$PCM_CODE" && pwd)"
