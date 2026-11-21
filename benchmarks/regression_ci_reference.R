# Runs the authors' code of the GCM, wGCM, and PCM tests on cases written by
# `benchmarks/regression_ci_agreement.py`, unmodified (the wgcm_est mode calls its functions one
# by one, as below), and writes what each returns.
#
#   Rscript --vanilla benchmarks/regression_ci_reference.R <case folder> <mode>
#
# Each case is `<case>.csv` with columns x, y, z1, ... and a seed in `cases.csv`. The modes:
#   gcm       GeneralisedCovarianceMeasure::gcm.test with its default regression (boosted trees),
#             and the residuals it tests
#   wgcm_est  weightedGCM::wgcm.est with boosted trees and the weight of its releases after 0.1.0,
#             the fit itself: 0.1.0's steps in wgcm.est's order, without predict_weight's sign;
#             the sample split, weights, and residuals it uses, and its p-value
#   wgcm_fix  weightedGCM::wgcm.fix with boosted trees and 7 weights per variable, and its residuals
#   pcm_lm    pcm_test of the PCM authors' simulation code with their least-squares regressions
#   pcm_rf    6 calls of that pcm_test with their random forests, as their simulations run it
# The PCM authors' code is read from `$PCM_CODE` (a clone of their simulation repository).

args <- commandArgs(TRUE)
folder <- args[1]
mode <- args[2]
cases <- read.csv(file.path(folder, "cases.csv"))

if (mode %in% c("gcm", "wgcm_est", "wgcm_fix")) {
  suppressPackageStartupMessages(library(GeneralisedCovarianceMeasure))
  suppressPackageStartupMessages(library(weightedGCM))
  suppressPackageStartupMessages(library(xgboost))
}
if (mode %in% c("pcm_lm", "pcm_rf")) {
  source(file.path(Sys.getenv("PCM_CODE"), "test_functions.R"))
  suppressPackageStartupMessages(library(ranger))
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
    products <- as.numeric(comp.resids(x[estimate], Ztrain, list(), "xgboost")) *
      as.numeric(comp.resids(y[estimate], Ztrain, list(), "xgboost"))
    w <- weightedGCM:::predict_weight_xgboost(products, Ztrain, Ztest, list())
    row$p <- weightedGCM:::wgcm.1d.1sided(x[-estimate], y[-estimate], Ztest, w, "xgboost", list())
    eps <- as.numeric(comp.resids(x[-estimate], Ztest, list(), "xgboost"))
    xi <- as.numeric(comp.resids(y[-estimate], Ztest, list(), "xgboost"))
    write.csv(data.frame(row = as.integer(seq_len(n) %in% estimate)),
              file.path(folder, paste0(case, "_split.csv")), row.names = FALSE)
    write.csv(data.frame(w = w, eps = eps, xi = xi),
              file.path(folder, paste0(case, "_wgcm_est.csv")), row.names = FALSE)
  } else if (mode == "wgcm_fix") {
    eps <- as.numeric(comp.resids(x, Z, list(), "xgboost"))
    xi <- as.numeric(comp.resids(y, Z, list(), "xgboost"))
    set.seed(seed)
    row$p <- wgcm.fix(x, y, Z, regr.meth = "xgboost", weight.num = 7)
    write.csv(data.frame(eps = eps, xi = xi), file.path(folder, paste0(case, "_wgcm_fix.csv")),
              row.names = FALSE)
  } else {
    reps <- if (mode == "pcm_lm") 1 else 6
    reg <- if (mode == "pcm_lm") lm_reg_method else ranger_reg_method
    splits <- matrix(0L, n, reps)
    for (b in seq_len(reps)) {
      set.seed(seed * 10 + b)
      splits[sample(1:n, n / 2), b] <- 1L
      set.seed(seed * 10 + b)
      row[[paste0("p", b)]] <- pcm_test(y, x, Z, reg)
    }
    write.csv(as.data.frame(splits), file.path(folder, paste0(case, "_split.csv")),
              row.names = FALSE)
  }
  out[[k]] <- as.data.frame(row)
}
write.csv(do.call(rbind, out), file.path(folder, paste0("reference_", mode, ".csv")),
          row.names = FALSE)
