# Runs the authors' code of the GCM, wGCM, and PCM tests on cases written by
# `benchmarks/regression_ci_agreement.py`, unmodified, and writes what each returns. The GCM
# authors' package (with xgboost 1.7.8.1) and the wGCM authors' current one (with xgboost 3.2.1.1)
# need different xgboost releases, so the gcm and the wgcm modes run on different libraries.
#
#   Rscript --vanilla benchmarks/regression_ci_reference.R <case folder> <mode>
#
# Each case is `<case>.csv` with columns x, y, z1, ... and a seed in `cases.csv`. The modes:
#   gcm       GeneralisedCovarianceMeasure::gcm.test with its default regression (boosted trees),
#             and the residuals it tests
#   wgcm_est  weightedGCM 0.1.2's wgcm.est with boosted trees, and the sample split, weights, and
#             residuals it uses: its steps in its order, which draw the same random numbers, checked
#             to give wgcm.est's p-value
#   wgcm_fix  weightedGCM 0.1.2's wgcm.fix with boosted trees and 7 weights per variable, and the
#             residuals it uses: its steps in its order, checked to give wgcm.fix's p-value
#   pcm_lm    pcm_test of the PCM authors' simulation code with their least-squares regressions
#   pcm_rf    6 calls of that pcm_test with their random forests, as their simulations run it
#   pcm_gam   6 calls of that pcm_test with their additive models (gam_reg_method, gam_gtilde, and
#             gam_vhat_reg_method), as their Section 6.1 simulations run it; a call that stops with
#             an error of the smoothing fit gives NA
# The PCM authors' code is read from `$ANDREY_BENCH_PCM_CODE` (a clone of their simulation repository).

args <- commandArgs(TRUE)
folder <- args[1]
mode <- args[2]
cases <- read.csv(file.path(folder, "cases.csv"))

if (mode == "gcm") {
  suppressPackageStartupMessages(library(GeneralisedCovarianceMeasure))
  suppressPackageStartupMessages(library(xgboost))
}
if (mode %in% c("wgcm_est", "wgcm_fix")) {
  suppressPackageStartupMessages(library(weightedGCM))
  suppressPackageStartupMessages(library(xgboost))
  resid <- function(v, Z) as.numeric(weightedGCM:::get_residuals(v, Z, "xgboost"))
}
if (mode %in% c("pcm_lm", "pcm_rf", "pcm_gam")) {
  source(file.path(Sys.getenv("ANDREY_BENCH_PCM_CODE"), "test_functions.R"))
  suppressPackageStartupMessages(library(ranger))
  suppressPackageStartupMessages(library(mgcv))
}

out <- list()
for (k in seq_len(nrow(cases))) {
  case <- cases$case[k]
  seed <- cases$seed[k]
  d <- read.csv(file.path(folder, paste0(case, ".csv")))
  x <- d$x
  y <- d$y
  Z <- as.matrix(d[, grepl("^z", names(d)), drop = FALSE])
  n <- length(x)
  row <- list(case = case)
  if (mode == "gcm") {
    rx <- GeneralisedCovarianceMeasure:::comp.resids(x, Z, list(), "xgboost")
    ry <- GeneralisedCovarianceMeasure:::comp.resids(y, Z, list(), "xgboost")
    r <- gcm.test(x, y, Z)
    row$statistic <- r$test.statistic
    row$p <- r$p.value
    write.csv(data.frame(rx = as.numeric(rx), ry = as.numeric(ry)),
              file.path(folder, paste0(case, "_gcm.csv")), row.names = FALSE)
  } else if (mode == "wgcm_est") {
    set.seed(seed)
    estimate <- sample(1:n, ceiling(0.3 * n))
    Ztrain <- Z[estimate, , drop = FALSE]
    Ztest <- Z[-estimate, , drop = FALSE]
    w <- weightedGCM:::predict_weight(x[estimate], y[estimate], Ztrain, Ztest, "xgboost", list())
    eps <- resid(x[-estimate], Ztest)
    xi <- resid(y[-estimate], Ztest)
    R <- eps * xi * w
    row$p <- 1 - pnorm(sqrt(length(R)) * mean(R) / sqrt(mean(R^2) - mean(R)^2))
    set.seed(seed)
    stopifnot(isTRUE(all.equal(row$p, wgcm.est(x, y, Z, regr.meth = "xgboost"), tolerance = 1e-12)))
    write.csv(data.frame(row = as.integer(seq_len(n) %in% estimate)),
              file.path(folder, paste0(case, "_split.csv")), row.names = FALSE)
    write.csv(data.frame(w = w, eps = eps, xi = xi),
              file.path(folder, paste0(case, "_wgcm_est.csv")), row.names = FALSE)
  } else if (mode == "wgcm_fix") {
    set.seed(seed)
    eps <- resid(x, Z)
    xi <- resid(y, Z)
    R <- t(eps * xi * weightedGCM:::weight_matrix(Z, 7, "sign"))
    R <- R / sqrt(rowMeans(R^2) - rowMeans(R)^2)
    T.stat <- sqrt(n) * max(abs(rowMeans(R)))
    T.sim <- apply(abs(R %*% matrix(rnorm(n * 499), n, 499)), 2, max) / sqrt(n)
    row$p <- (sum(T.sim >= T.stat) + 1) / 500
    set.seed(seed)
    stopifnot(isTRUE(all.equal(row$p, wgcm.fix(x, y, Z, regr.meth = "xgboost", weight.num = 7))))
    write.csv(data.frame(eps = eps, xi = xi), file.path(folder, paste0(case, "_wgcm_fix.csv")),
              row.names = FALSE)
  } else {
    reps <- if (mode == "pcm_lm") 1 else 6
    splits <- matrix(0L, n, reps)
    for (b in seq_len(reps)) {
      set.seed(seed * 10 + b)
      splits[sample(1:n, n / 2), b] <- 1L
      set.seed(seed * 10 + b)
      row[[paste0("p", b)]] <- switch(mode,
        pcm_lm = pcm_test(y, x, Z, lm_reg_method),
        pcm_rf = pcm_test(y, x, Z, ranger_reg_method),
        pcm_gam = tryCatch(
          pcm_test(y, x, Z, gam_reg_method, gam_gtilde, gam_vhat_reg_method),
          error = function(e) NA
        )
      )
    }
    write.csv(as.data.frame(splits), file.path(folder, paste0(case, "_split.csv")),
              row.names = FALSE)
  }
  out[[k]] <- as.data.frame(row)
}
write.csv(do.call(rbind, out), file.path(folder, paste0("reference_", mode, ".csv")),
          row.names = FALSE)
