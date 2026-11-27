---
name: configuration
description: The environment variables Andrey reads, by group, with their values and defaults.
---

# Configuration

Andrey reads a few environment variables, all named `ANDREY_*`. Each has a default, so none needs
to be set. `andrey config` prints the value of each one in your shell and the backends your machine
can use; `andrey config --json` prints the same as JSON.

A blank value counts as unset. A switch takes `1`, `true`, `yes`, or `on`, and `0`, `false`, `no`,
or `off`. Any other value Andrey cannot read raises `ValueError` naming the variable, and
`andrey config` and `andrey run` exit with `invalid_configuration`.

The backend, the worker count, and the tie-break rule can also be set in code. The most specific
setting wins:

1. `with andrey.config(backend=..., n_jobs=..., tie_break=...):`, for the calls inside the
   block;
2. `andrey.config.backend = ...`, `andrey.config.n_jobs = ...`, or
   `andrey.config.tie_break = ...`, for every later call;
3. the environment variables below, read at each call;
4. the defaults.

## Backend

- `ANDREY_BACKEND`: your preferred implementation for numeric operations. `auto` (the default)
  considers `cuda` when it is available and the input meets the operation's size threshold, subject
  to the startup rule under [GPU dispatch](#gpu-dispatch); otherwise it uses `numpy`. It never picks
  `cpu` or `mps`. `numpy` uses NumPy on the CPU in float64, which we use as the reference. `cpu`
  uses PyTorch on the CPU, `cuda` uses PyTorch on a CUDA-compatible GPU, and `mps` uses PyTorch on
  an Apple GPU in float32; Apple GPUs are not supported in the alpha. `none` and `metal` are aliases
  for `cpu` and `mps`. An explicit choice bypasses the size threshold and startup rule. If an
  operation supports your choice but the machine cannot run it, Andrey warns once with
  `andrey.BackendFallbackWarning` and uses `auto` for that operation.
- `ANDREY_DEVICE`: another name for `ANDREY_BACKEND`; if you set both, `ANDREY_BACKEND` takes
  precedence, but both values must be valid.

A GPU speeds up covariance, correlation, entropy, the KCI and CMIknn tests, and the independence
tests of `pnl`. GES, hill climbing, and the other searches run on the CPU. An operation without the
requested backend uses `auto`, without a warning.

## Parallel search

- `ANDREY_N_JOBS`: the worker processes for GES and GFCI, and the worker threads for the two
  directions of `pnl` and for the searches of `skewd` and `skewd_mv`. PC, FCI, GFCI, BOSS-FCI,
  GRaSP-FCI, CD-NOD, and RSMAX2 with `restrict="pc"` also use the worker processes for their
  skeleton search with the `fisherz`, `student_t`, `normal_scores`, and `rcot` tests on complete
  data, and send a conditioning-set size to them only when it has enough tests to repay them; with
  `rcot` the p-values are the same with any number of workers. FCI and GFCI also send each large
  possible-d-separation search to them with the `fisherz`, `student_t`, and `normal_scores` tests on
  complete data, and FCI-JCI with `fisherz` and `student_t`; the pairs are still searched one at a
  time, in the serial order, and every graph is the serial one. IAMB, Inter-IAMB, and MMPC split a
  list of two or more targets over them with the `fisherz`, `student_t`, and `normal_scores` tests
  on complete data, with the same results for any number of workers. `andrey.bootstrap` fits one
  resample per worker, with the same result at any number. `1` (the default) and `0` run serially;
  `-1` uses every usable core. `pnl` with one worker leaves the BLAS threads unlimited and with
  several limits each thread to one, so on a machine with several cores its p-values and statistics
  can differ in the last digits between one worker and several. Each `pnl` thread holds three n x n
  float64 matrices; when the threads you ask for would hold more than 4 GiB, fewer run, with a
  `PerformanceWarning`. `skewd` runs its two directions and their searches on the threads,
  and `skewd_mv` its conditional models and their searches; every search draws from its own random
  stream, so the results are the same bit for bit with any number of workers. Unlike `pnl`, both run
  on one BLAS thread with any number of workers, so their results do not depend on the BLAS thread
  count either; the limit applies to the whole process while a fit runs.
- `ANDREY_GES_PARALLEL_MIN_WORK`: a GES pass uses the workers only when it expects to evaluate more
  candidates than this. The default is `10000`; `0` always uses them.
- `ANDREY_SKELETON_PARALLEL_MIN_WORK`: a conditioning-size pass of the skeleton search of PC, FCI,
  GFCI, BOSS-FCI, GRaSP-FCI, CD-NOD, and RSMAX2 with `restrict="pc"` uses the workers only when it
  holds at least this many conditional-independence tests. The default is `500000`, measured on an
  AMD EPYC 9534 server running Linux; `0` always uses them.
- `ANDREY_POSSIBLE_DSEP_PARALLEL_MIN_WORK`: a possible-d-separation search of FCI, FCI-JCI, and
  GFCI uses the workers only when its candidate set has at least this many subsets, counted over
  every size it tests. The default is `250000`, measured on an AMD EPYC server running Linux with 4
  and 8 workers; `0` always uses them.
- `ANDREY_RCOT_PARALLEL_MIN_WORK`: with `rcot`, a conditioning-size pass of the skeleton search uses
  the workers only when its tests times the data's rows reach this value, since an RCoT test costs
  far more than a Fisher's z test and its cost grows with the rows. The default is `2000000`,
  measured on an Intel Core i7-7820X running Linux with 4 workers; `0` always uses them.

The cutoffs matter only with more than one worker. `andrey calibrate` measures all four for your
machine and prints the values to set; `andrey calibrate --method skeleton` measures the skeleton's
alone, `--method rcot` RCoT's, and `--method possible_dsep` the possible-d-separation search's.
With more than one worker and a default cutoff, a fit warns once when a pass or a
possible-d-separation search reaches a tenth of it; setting the variable, to any value, silences
the warning.

The workers start from a forkserver, whose socket goes in your temporary directory (`TMPDIR`). If
that directory is on a file system that rejects the socket, as some cluster file systems do, we
start the forkserver with its socket in `/dev/shm`, or `/tmp`, and warn once; if neither works,
the workers start with `fork` or `spawn`. The results do not change. Set `TMPDIR` to a directory on
a local file system to silence the warning.

On Linux, workers started from the forkserver read one shared copy of the covariance or correlation
matrix in `/dev/shm`, where GES, the skeleton and possible-d-separation searches, or a list of
targets need one, rather than a copy each. The copy has no name there, so its memory goes back when
the last worker that reads it exits, even when the fit is killed. If `/dev/shm` cannot hold the
matrix, each worker gets its own copy. The results do not change.

## Scan dispatch

- `ANDREY_SHARED_SCAN_MIN_WORK`: with `fisherz`, `student_t`, or `normal_scores` on complete data, a
  grow scan of `iamb` or `inter_iamb` (also as the restrict phase of `rsmax2` with `restrict="iamb"`
  or `"inter_iamb"`) of two or more candidates against one conditioning set of two or more variables
  uses one Cholesky factor of the set's correlation block only when its candidates times the block
  size (the set plus two) reach this value; otherwise each candidate inverts its submatrix. The
  default is `128`, measured on an AMD EPYC 7713 server running Linux on one core, where the factor
  was slower below about 64 and faster from 128 for every set size sampled. The two paths agree to
  about 1e-10 relative in the p-value; the per-query path is the exact one, and `0` factors every
  such scan.
- `ANDREY_LINGAM_PRUNE_MIN_WORK`: a step of the `pwling` order search (DirectLiNGAM and the methods
  built on it) stops scoring a candidate that can no longer be picked only when its rows times
  ordered pairs of candidates reach this value (default `100000`, about the break-even on one core
  of an AMD EPYC 7713); the order is the same bit for bit either way, and `0` prunes every step.

## Test-wise deletion

With `fisherz` or `student_t` on data with missing values, a test reads the rows observed in all its
columns. When the rows fall into a few missingness patterns, as in data merged from cohorts that
each measure their own variables, the rows are grouped by pattern once, and each test sums the
groups' tables instead of every row. The groups follow the patterns, not the row or the column
order. The variables below select this path once per test from the data, only when the groups in
which a block can take the cheaper grouped sums hold enough rows to repay the tables. Within it,
each block of columns takes the cheaper of the two sums, and sums every row when it keeps so few of
its groups' rows that the groups' sums would lose digits. A block whose groups hold too few rows for
its test is not tested without being summed. The two paths count the same rows for every tested
query and agree up to the last digits of the p-values.

- `ANDREY_TESTWISE_GROUP_BYTES`: the grouped path is used only when its tables fit in this many
  bytes (default `268435456`, 256 MiB); `0` keeps every test on the per-row path.
- `ANDREY_TESTWISE_GROUP_MAX_SCATTER`: the grouped path is used only when the cells a row misses
  among its group's columns are at most this share of those cells (default `0.15`). Above it,
  correcting for the rows that miss several columns of a test costs more than summing every row.
- `ANDREY_TESTWISE_GROUP_MAX_GROUPS`: the grouped path is used only when the patterns form at most
  this many groups (default `64`).
- `ANDREY_TESTWISE_GROUP_MAX_SHARED`: the grouped path is used only when at least a quarter of the
  pairs of columns are observed together in at most this share of the rows (default `0.4`). When
  nearly every pair shares more, as with outages in a time series or cells missing at random, each
  block lies in many groups, and grouping costs more than it saves.
- `ANDREY_TESTWISE_GROUP_MIN_COLUMNS`: the grouped path is used only for a test over at least this
  many columns (default `8`). A test over fewer asks too few distinct blocks to repay the grouping.
- `ANDREY_TESTWISE_GROUP_MIN_GAIN`: on the grouped path, a block takes the groups' sums only when
  its estimated per-row cost is at least this many times its grouped cost (default `2`).

Three constants are fixed. The cost of opening a group sets how the rows split into groups, not
whether a test uses them. `ANDREY_TESTWISE_GROUP_MAX_SHARED` is read on a quarter of the column
pairs. A block that keeps fewer than one sixteenth of its groups' rows, or fewer rows than twice its
columns, is summed over every row; this bounds the rounding, which no speed setting loosens.

## GPU dispatch

- `ANDREY_COV_GPU_THRESHOLD`: under `auto`, covariance and correlation run on `cuda` from this many
  elements (rows times columns). The default is `90000`.
- `ANDREY_ENTROPY_GPU_THRESHOLD`: the same for the entropy terms of DirectLiNGAM and the methods
  built on it. The default is `6250`.
- `ANDREY_KCI_GPU_THRESHOLD`: under `auto`, a KCI query with a conditioning set runs on `cuda`, in
  float64, from this many kernel-matrix elements (rows squared). The default is `31250`, about 177
  rows.
- `ANDREY_CMIKNN_GPU_THRESHOLD`: the same for a CMIknn query, from this many distance-matrix
  elements (rows squared). The default is `1024`, 32 rows. Its p-values on `cuda` equal those on
  the CPU.
- `ANDREY_PNL_GPU_THRESHOLD`: the same for the independence tests of `pnl`, in float64, from this
  many kernel-matrix elements (rows squared). The default is `26278`, about 163 rows. A single pair
  uses the GPU only when it is already running, by the rule below.
- `ANDREY_GPU_CALIBRATE`: when on, the first `auto` use of covariance, correlation, or entropy
  measures where `cuda` starts to beat NumPy on this machine, and keeps the result in
  `$XDG_CACHE_HOME/andrey/gpu_calibration.json`, else `~/.cache/andrey/gpu_calibration.json`. Off
  by default. A threshold you set wins.

Under `auto`, an operation that runs once per fit uses the GPU only when PyTorch and the device are
already running in the process, so a CPU fit never pays to start a GPU.
`andrey.core.backend.dispatch_stats()` counts which backend each operation ran on.

## Reproducibility

- `ANDREY_SEED`: the seed of every method that draws random numbers when the call gives no `seed=`:
  BOSS, GRaSP, ICA-LiNGAM, CALM, IGSP and UT-IGSP with `n_starts` above 1 (the default), hill
  climbing and tabu search with `n_starts` above 1, and the searches that test independence, with
  the `rcot`, `cmiknn`, `gpdc`, `gcm`, `wgcm`, and `pcm` tests. The default is `0`; it takes any
  integer from `0` to `2**32 - 1`. `seed=None` uses this configured seed, so a call repeats. A call
  that gives `seed=` does not depend on it. Deterministic methods ignore it.
- `ANDREY_TIE_BREAK`: how the searches choose between tied candidates. `rank` (the default) orders
  them by a rank computed from the data, which a row shuffle, a change of units, or a sign flip
  leaves in place, with [a few exceptions](guides/column-order.md#how-andrey-breaks-ties), so
  reordering the columns relabels the result. `position` orders them by column position, which skips
  computing the ranks, fires the orientation rules of `pc`, `cdnod`, `fci`, `gfci`, `boss_fci`,
  `grasp_fci`, and `fci_jci` one at a time in column order, and visits the possible-d-separation
  step of `fci`, `gfci`, and `fci_jci` in column order; the result can then depend on the column
  order. A fit whose result the rule can change records it in
  `metadata["tie_break"]`.

Results repeat on the same machine; across platforms they can differ, CALM's most of all.
[Column order and machines](guides/column-order.md) lists what each method keeps under a new column
order and on another machine.
`andrey.seed_all(seed)` seeds Python, NumPy, and PyTorch from one number, for data you simulate.

## Datasets

- `ANDREY_DATA_DIR`: where `andrey.data.load_dataset` keeps the files it downloads. The default is
  `$XDG_CACHE_HOME/andrey`, else `~/.cache/andrey`. A `data_home=` argument wins.

## Testing

- `ANDREY_REQUIRE_GPU`: when on, Andrey's own GPU tests fail, instead of skipping, without a CUDA
  or Apple GPU. It does not change a fit.
