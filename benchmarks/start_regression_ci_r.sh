#!/usr/bin/env bash
# Installs the authors' code of the GCM, wGCM, and PCM tests, the references of Andrey's "gcm",
# "wgcm", and "pcm", for `benchmarks/regression_ci_agreement.py`.
#
#   benchmarks/start_regression_ci_r.sh <folder for the PCM authors' code>
#
# GeneralisedCovarianceMeasure 0.2.0 left CRAN and weightedGCM 0.1.0 is the version its paper ran,
# so both come from the CRAN archive with their SHA-256 checked, with xgboost 1.7.8.1, the last
# release whose interface they call. The PCM authors' simulation code, which loads vimp and the
# archived CondIndTests, is cloned at a fixed commit into the folder given; it carries no license,
# so it stays outside the repository. The library is `$ANDREY_BENCH_R_LIBS`; put it first on
# `R_LIBS` when running the agreement script.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
R_LIBS="${ANDREY_BENCH_R_LIBS:-$REPO/benchmarks/.r-lib-regression-ci}"
PCM_CODE="${1:?usage: start_regression_ci_r.sh <folder for the PCM authors' code>}"
ARCHIVE="https://cran.r-project.org/src/contrib/Archive"
TARBALLS=(
    "GeneralisedCovarianceMeasure/GeneralisedCovarianceMeasure_0.2.0.tar.gz 8ea9f6e913be11b5d2725d9aad27c523190963e4c5b70ecbadb68e7e99abfd75"
    "weightedGCM/weightedGCM_0.1.0.tar.gz 231718693b39265bbcb02eb21f9849bef33a2341fc08635ebeaae17d7b258211"
    "xgboost/xgboost_1.7.8.1.tar.gz 394d6fd00b2fe97549c7a7e6598df86448cdfbd7c0af45b0c17f7b9e81bc1be9"
    "CondIndTests/CondIndTests_0.1.5.tar.gz 55f3589c4d317ac6e909889cb0bb6e9105be187362c25b6f78ba166939faa44e"
)
PCM_REPO="https://github.com/ARLundborg/pcm_code.git"
PCM_COMMIT="3822659390510d65c36857f8801c893e94850774"

if command -v module >/dev/null 2>&1 || [ -f /usr/share/lmod/lmod/init/bash ]; then
    set +u
    # shellcheck disable=SC1091
    source /usr/share/lmod/lmod/init/bash
    set -u
    # Unquoted on purpose: one word per module.
    # shellcheck disable=SC2086
    module load $R_MODULE
fi
mkdir -p "$R_LIBS"
export R_LIBS

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for entry in "${TARBALLS[@]}"; do
    read -r path sha <<< "$entry"
    file="$tmp/$(basename "$path")"
    curl -fsSL -o "$file" "$ARCHIVE/$path"
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
    for (f in c("xgboost_1.7.8.1", "GeneralisedCovarianceMeasure_0.2.0", "weightedGCM_0.1.0",
                "CondIndTests_0.1.5"))
        install.packages(file.path(tmp, paste0(f, ".tar.gz")), lib = lib, repos = NULL, type = "source")
    for (p in c("GeneralisedCovarianceMeasure", "weightedGCM", "xgboost", "mgcv", "ranger"))
        cat(p, as.character(packageVersion(p, lib.loc = lib)), "\n")
' "$tmp"

if [ ! -d "$PCM_CODE/.git" ]; then
    git clone --quiet "$PCM_REPO" "$PCM_CODE"
fi
git -C "$PCM_CODE" checkout --quiet "$PCM_COMMIT"
echo "pcm_code $PCM_COMMIT"
