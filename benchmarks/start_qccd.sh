#!/usr/bin/env bash
# Fetches the authors' bQCD code and installs the R packages it loads, the reference of the QCCD
# solution (``tagasovska.qccd``).
#
#   benchmarks/start_r.sh        # first: R modules, rpy2 in the bench venv
#   benchmarks/start_qccd.sh     # then: the authors' code and its R library
#
# The code is the supplementary repository of Tagasovska, Chavez-Demoulin, and Vatter (ICML 2020),
# tagas/bQCD at the commit below. The repository carries no license file. The benchmark runs it
# unmodified as a measurement reference, `source()`s `R/bqcd.R` from the directory this script
# prints, and copies none of it. The tarball's commit id (from its archive header) and the SHA-256
# of the file a fit sources are checked. The script writes the code and the R library under
# `$ANDREY_BENCH_QCCD_DIR` (default `benchmarks/.qccd`); export what it prints before running the
# solution. The library is its own (`$ANDREY_BENCH_QCCD_R_LIBS`, default `$DIR/rlib`) so it neither
# changes nor depends on the R lane's: join the libraries with a colon when one job loads several.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
DEST="${ANDREY_BENCH_QCCD_DIR:-$REPO/benchmarks/.qccd}"
R_LIBS="${ANDREY_BENCH_QCCD_R_LIBS:-$DEST/rlib}"
COMMIT="0e6eefefc46f1abcd876fe34751e18fbd9d23ef8"
URL="https://codeload.github.com/tagas/bQCD/tar.gz/$COMMIT"
BQCD_SHA256="13e0229a06daa9f33d82d6716376ab2093097d8f6508f1f3a6380632e2f69b85"
CODE="$DEST/bQCD-$COMMIT"

echo ">> bQCD $COMMIT"
mkdir -p "$DEST"
if [[ ! -f "$CODE/R/bqcd.R" ]]; then
    curl -fsSL -o "$DEST/bQCD-$COMMIT.tar.gz" "$URL"
    [[ "$(gzip -dc "$DEST/bQCD-$COMMIT.tar.gz" | git get-tar-commit-id)" == "$COMMIT" ]] \
        || { echo "bQCD tarball is not commit $COMMIT" >&2; exit 1; }
    tar -xzf "$DEST/bQCD-$COMMIT.tar.gz" -C "$DEST"
fi
echo "$BQCD_SHA256  $CODE/R/bqcd.R" | sha256sum --check --status \
    || { echo "R/bqcd.R does not match $BQCD_SHA256" >&2; exit 1; }
echo "$COMMIT" > "$CODE/COMMIT"

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
# `rvinecopulib` compiles a large C++ library: let make use the cores of the job.
export MAKEFLAGS="-j${ANDREY_BENCH_BUILD_JOBS:-8}"

echo ">> R library: $R_LIBS"
start=$SECONDS
# `R/bqcd.R` loads these four with `library()`. `quantregForest` and `qrnn` serve the other two
# estimators of the file; the one the benchmark calls, `QCCD`, needs `rvinecopulib` and `statmod`.
Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS_USER")
    .libPaths(lib)
    options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))
    need <- Filter(function(p) length(find.package(p, quiet = TRUE)) == 0L,
                   c("quantregForest", "rvinecopulib", "statmod", "qrnn"))
    if (length(need)) install.packages(need, lib = lib)
    for (p in c("quantregForest", "rvinecopulib", "statmod", "qrnn")) {
        cat(p, as.character(packageVersion(p)), "\n")
    }
'
echo ">> library built in $((SECONDS - start)) s"

# One fit through the authors' wrapper on a small pair.
Rscript --vanilla -e '
    .libPaths(Sys.getenv("R_LIBS_USER"))
    source(file.path(commandArgs(TRUE)[1], "R", "bqcd.R"))
    set.seed(1)
    x <- rnorm(300)
    res <- QCD_wrap(x, x^3 + rnorm(300), type = "QCCD", m = 7)
    cat("bqcd OK: x -> y fit reads", res$cd, "eps", res$eps, "\n")
' "$CODE"

printf 'export ANDREY_BENCH_QCCD=%q\n' "$CODE"
echo "module load $R_MODULE"
printf 'export R_LIBS_USER=%q${R_LIBS_USER:+:$R_LIBS_USER}\n' "$R_LIBS"
echo ">> done"
