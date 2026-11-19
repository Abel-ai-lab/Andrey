# Fit the authors' CPCM_graph_estimate on every row of a manifest, with the file unmodified.
#
#   Rscript --vanilla cpcm_worker.R <code dir> <manifest.csv> <out dir> [cores]
#
# <code dir> holds CPCM_function.R (benchmarks/start_cpcm.sh fetches it). The manifest has one row
# per fit and the columns
#
#   id               names the fit, and its result file
#   table            path of a CSV with a header, one column per variable
#   family           family_of_distributions: a family name, 1, 2, or "Sequential choice"
#   greedy_method    "exact", "RESIT_greedy", "edge_greedy", or "RESIT" (read only when d > 2)
#   lambda           the per-edge penalty (read only when d > 2)
#   seed             set.seed() before every call: the Poisson and negative binomial residuals draw
#                    uniform numbers
#   warmup           untimed calls before the timed one, in this process
#
# <out dir>/<id>.csv gets one row per fit, written when the fit ends; a fit whose file exists is
# skipped. Its columns:
#
#   status           "ok", or "error" with the message in `error`
#   seconds          wall time of the CPCM_graph_estimate call alone
#   forced           d = 2: "1 --> 2" or "2 --> 1", the code's forced estimate, from the unrounded
#                    p-values (a tie goes to "1 --> 2")
#   p_empty, p_12, p_21   d = 2: the three p-values as the code returns them, rounded to 6 digits
#   score, testing   d = 2: the code's score-based and testing estimates
#   families         d = 2: "Families used", the family of column 1 then of column 2
#   adjacency        d > 2: the estimated DAG's 0/1 matrix, row i -> column j, row by row, joined
#                    by ";" and in the table's column order
#
# The p-values are the only numbers the code rounds, so two printed zeros cannot be compared.
args <- commandArgs(trailingOnly = TRUE)
code_dir <- args[1]
manifest <- read.csv(args[2], stringsAsFactors = FALSE, colClasses = "character")
out_dir <- args[3]
cores <- if (length(args) >= 4) as.integer(args[4]) else 1L
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages(source(file.path(code_dir, "CPCM_function.R")))

fit_row <- function(i) {
    row <- manifest[i, ]
    path <- file.path(out_dir, paste0(row$id, ".csv"))
    if (file.exists(path)) return(invisible(NULL))
    X <- read.csv(row$table)
    family <- if (row$family %in% c("1", "2")) as.numeric(row$family) else row$family
    once <- function() {
        set.seed(as.integer(row$seed))
        CPCM_graph_estimate(
            X, family_of_distributions = family, greedy_method = row$greedy_method,
            lambda = as.numeric(row$lambda), quiet = TRUE
        )
    }
    for (w in seq_len(as.integer(row$warmup))) try(once(), silent = TRUE)
    out <- list(
        id = row$id, status = "ok", seconds = NA_real_, forced = NA, p_empty = NA, p_12 = NA,
        p_21 = NA, score = NA, testing = NA, families = NA, adjacency = NA, error = NA
    )
    start <- proc.time()[["elapsed"]]
    res <- tryCatch(once(), error = function(e) e)
    out$seconds <- proc.time()[["elapsed"]] - start
    if (inherits(res, "error") || is.null(res)) {
        out$status <- "error"
        out$error <- if (is.null(res)) "no result" else gsub("[\r\n]+", " ", conditionMessage(res))
    } else if (inherits(res, "bn")) {
        nodes <- colnames(X)
        out$adjacency <- paste(as.vector(t(bnlearn::amat(res)[nodes, nodes])), collapse = ";")
    } else {
        v <- as.character(res[[1]])
        out[c("p_empty", "p_12", "p_21", "score", "testing", "forced", "families")] <- as.list(v)
    }
    write.csv(as.data.frame(out), path, row.names = FALSE)
    invisible(NULL)
}

invisible(parallel::mclapply(seq_len(nrow(manifest)), fit_row, mc.cores = cores,
                             mc.preschedule = FALSE))
