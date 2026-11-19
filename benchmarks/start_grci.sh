#!/usr/bin/env bash
# Fetches the authors' GRCI code and installs the R packages `causal_direc` needs, the reference of
# the GRCI solution (``strobl.grci``).
#
#   benchmarks/start_r.sh        # first: R modules, rpy2 in the bench venv
#   benchmarks/start_grci.sh     # then: the authors' code and its R library
#
# The code is ericstrobl/GRCI at the commit below. Its DESCRIPTION declares CC BY-NC 4.0, so it is
# for non-commercial use with attribution: the benchmark runs it from this local download as a
# measurement reference, `source()`s five of its files from the directory this script prints, and
# redistributes none of it, in the repository or in any output. The tarball's commit id (from its
# archive header) and the SHA-256 of the files a fit reads are checked. The script writes the code
# and the R library under `$ANDREY_BENCH_GRCI_DIR` (default `benchmarks/.grci`); export what it
# prints before running the solution.
#
# The repository bundles `RANN-master.zip`, the nearest-neighbor package its README tells the reader
# to install before GRCI, so that is what the script installs, from the zip, in place of CRAN's
# RANN. The library is its own (`$ANDREY_BENCH_GRCI_R_LIBS`, default `$DIR/rlib`) so that package
# never reaches the R lane's. The printed export puts it in front of the `R_LIBS_USER` already set:
# evaluate this script's exports after the other setup scripts', so its RANN is the one R finds.

set -euo pipefail

REPO="$(git rev-parse --show-toplevel)"
R_MODULE="${ANDREY_BENCH_R_MODULE:-gcc-12.1.0-gcc-11.2.0 r-4.6.1-gcc-12.1.0}"
DEST="${ANDREY_BENCH_GRCI_DIR:-$REPO/benchmarks/.grci}"
R_LIBS="${ANDREY_BENCH_GRCI_R_LIBS:-$DEST/rlib}"
COMMIT="9f9b78d91184a2a5849182bec2b10426becb4d2b"
URL="https://codeload.github.com/ericstrobl/GRCI/tar.gz/$COMMIT"
CODE="$DEST/GRCI-$COMMIT"

echo ">> GRCI $COMMIT"
mkdir -p "$DEST"
if [[ ! -f "$CODE/R/causal_direc.R" ]]; then
    curl -fsSL -o "$DEST/GRCI-$COMMIT.tar.gz" "$URL"
    [[ "$(gzip -dc "$DEST/GRCI-$COMMIT.tar.gz" | git get-tar-commit-id)" == "$COMMIT" ]] \
        || { echo "GRCI tarball is not commit $COMMIT" >&2; exit 1; }
    tar -xzf "$DEST/GRCI-$COMMIT.tar.gz" -C "$DEST"
fi
while read -r sha file; do
    echo "$sha  $CODE/$file" | sha256sum --check --status \
        || { echo "$file does not match $sha" >&2; exit 1; }
done <<'FILES'
ad7bc701cc53c6d29521497d58ae5ca33cf961481186e1c7a62e106edd1cbf4c R/causal_direc.R
52ffa9b678a0e07d47d04e9c909f5e3614107f3b26e2db9d75268e7b3ddbd115 R/PartialOut.R
dd1d85bff023341917b06d6b4aa016a372ee3c89f6ce04077587486565aee0f0 R/normalizeData.R
23ed9be57576aeaf742b1c69b89e5c5d8e99692ae894336fb8f7bfe509cd25b9 R/spline_regressionR.R
f24c9c9d9896aa6144cd70f71a3e45ad6e13156f8426b3ba8bc88051c9bd959d R/normalize01.R
88d2233eefefd0ad743918efb70f35595b4109c827a4af38bd50484fef5be17f DESCRIPTION
1d53e03c2764d26c0f0096a72787a938646668d21e87a73ac5bdbde17bc9f04b RANN-master.zip
FILES
echo "$COMMIT" > "$CODE/COMMIT"
grep '^License:' "$CODE/DESCRIPTION"

# shellcheck disable=SC1091
source /usr/share/lmod/lmod/init/bash
# Unquoted on purpose: one word per module.
# shellcheck disable=SC2086
module load $R_MODULE
mkdir -p "$R_LIBS"
export R_LIBS_USER="$R_LIBS"
export MAKEFLAGS="-j${ANDREY_BENCH_BUILD_JOBS:-8}"

echo ">> R library: $R_LIBS"
start=$SECONDS
# `RANN` first, from the bundled zip. `Rfast` supplies `spdinv`, `DirichletReg` `rdirichlet`; the
# five files the benchmark sources call nothing else. The repository's package Depends on more
# (pcalg, hetGP, gptk, xgboost, treeshap, ...) for its other methods, which the benchmark does not
# run.
if [[ ! -d "$R_LIBS/RANN" ]]; then
    mkdir -p "$DEST/rann-src"
    unzip -q -o "$CODE/RANN-master.zip" -d "$DEST/rann-src"
    R CMD INSTALL --library="$R_LIBS" "$DEST/rann-src/RANN-master"
fi
Rscript --vanilla -e '
    lib <- Sys.getenv("R_LIBS_USER")
    .libPaths(lib)
    options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = max(1L, parallel::detectCores()))
    need <- Filter(function(p) length(find.package(p, quiet = TRUE)) == 0L,
                   c("Rfast", "DirichletReg"))
    if (length(need)) install.packages(need, lib = lib)
    for (p in c("RANN", "Rfast", "DirichletReg")) cat(p, as.character(packageVersion(p)), "\n")
'
echo ">> library built in $((SECONDS - start)) s"

# One fit through the authors' function on a small pair.
Rscript --vanilla -e '
    .libPaths(Sys.getenv("R_LIBS_USER"))
    suppressMessages({library(RANN); library(Rfast); library(DirichletReg)})
    for (f in c("normalize01", "normalizeData", "spline_regressionR", "PartialOut", "causal_direc")) {
        source(file.path(commandArgs(TRUE)[1], "R", paste0(f, ".R")))
    }
    set.seed(1)
    x <- rnorm(300)
    cat("grci OK: x -> y fit reads", causal_direc(as.matrix(x), as.matrix(x^3 + rnorm(300))), "\n")
' "$CODE"

printf 'export ANDREY_BENCH_GRCI=%q\n' "$CODE"
echo "module load $R_MODULE"
printf 'export R_LIBS_USER=%q${R_LIBS_USER:+:$R_LIBS_USER}\n' "$R_LIBS"
echo ">> done"
