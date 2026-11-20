"""Define the machine-readable list of ``ANDREY_*`` environment variables.

Each variable has one :class:`EnvVarSpec`, grouped by backend selection, reproducibility,
determinism, or data. CLI help, configuration reports, and ``SKILL.md`` render these
definitions. Each variable's parser is in :mod:`andrey.core.env`. The human-readable list is
``docs/docs/configuration.md``.
"""

from __future__ import annotations

from andrey.constraint._parallel_skeleton import (
    MIN_POOL_DSEP_SUBSETS,
    MIN_POOL_QUERIES,
    MIN_POOL_RCOT_WORK,
)
from andrey.core.backend import GPU_THRESHOLDS
from andrey.core.ci import MIN_SHARED_FACTOR_WORK
from andrey.core.env import SEED_MAX
from andrey.core.missing_groups import (
    GROUP_BYTES,
    GROUP_MAX_SCATTER,
    MAX_GROUPS,
    MAX_SHARED,
    MIN_COLUMNS,
    MIN_GAIN,
)
from andrey.lingam.pwling import MIN_PRUNE_WORK
from andrey.search._parallel_ges import MIN_POOL_GES_WORK

from .models import EnvVarSpec


def _gpu_default(op: str) -> str:
    """``op``'s default GPU cutoffs as the list shows them, such as ``cuda 90000, mps never``."""
    return ", ".join(
        f"{device} {'never' if cutoff is None else cutoff}"
        for device, cutoff in GPU_THRESHOLDS[op].items()
    )


ENV_VARS: tuple[EnvVarSpec, ...] = (
    EnvVarSpec(
        name="ANDREY_BACKEND",
        values="auto, numpy, cpu/none, cuda, mps/metal",
        default="auto",
        effect=(
            "Compute backend preference. `auto` uses `cuda` above an operation's size threshold, "
            "else numpy; it never selects `cpu` or `mps`. `numpy` forces plain NumPy in float64 "
            "(the reference results); `cpu` runs torch on the CPU; `mps` runs torch on an Apple "
            "GPU in float32, and Apple GPUs are not supported in the alpha. An unavailable "
            "backend warns once and falls back. `ANDREY_DEVICE` is an alias."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_DEVICE",
        values="same as ANDREY_BACKEND",
        default="ANDREY_BACKEND",
        effect="Alias for `ANDREY_BACKEND`; set either one.",
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_N_JOBS",
        values="integer >= -1 (-1 = all usable cores)",
        default="1",
        effect=(
            "Worker processes for GES, GFCI (its GES phase), the fits of `andrey.bootstrap`, "
            "the skeleton searches of PC, FCI, GFCI, BOSS-FCI, GRaSP-FCI, CD-NOD, and RSMAX2 "
            '(`restrict="pc"`) with `fisherz`, `student_t`, `normal_scores`, or `rcot`, and the '
            "possible-d-separation searches of FCI and GFCI with `fisherz`, `student_t`, or "
            "`normal_scores` and of FCI-JCI with `fisherz` or `student_t`, and worker threads for "
            "the two directions of `pnl`; `0` or `1` runs serially. "
            "Separate from `ANDREY_BACKEND`, which picks the numeric code."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_GES_PARALLEL_MIN_WORK",
        values="integer (estimated candidate evaluations)",
        default=str(MIN_POOL_GES_WORK),
        effect=(
            "Per-pass crossover gate for parallel GES: a greedy pass fans out to the worker pool "
            "only when its estimated candidate-evaluation count (density-aware, ~2^degree) exceeds "
            "this, else in-process. `0` sends every pass to the pool. No effect "
            "unless the worker count (`n_jobs` / `ANDREY_N_JOBS`) is above 1. A fit with "
            "more than one worker warns once on the default when a pass reaches a tenth of it; "
            "`andrey calibrate` measures a value for a process that runs many fits."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_SKELETON_PARALLEL_MIN_WORK",
        values="integer (CI tests per conditioning-size pass)",
        default=str(MIN_POOL_QUERIES),
        effect=(
            "Per-pass crossover gate for the parallel skeleton search of PC, FCI, GFCI, BOSS-FCI, "
            'GRaSP-FCI, CD-NOD, and RSMAX2 (`restrict="pc"`) with `fisherz`, `student_t`, or '
            "`normal_scores`: a conditioning-size pass fans out to the worker pool only when it "
            "holds at least this many CI tests, else in-process. `0` sends every pass to the pool. "
            "The default was measured on an AMD EPYC 9534 server running Linux. No effect unless "
            "the worker count (`n_jobs` / `ANDREY_N_JOBS`) is above 1. A fit with more than one "
            "worker warns once on the default when a pass reaches a tenth of it; `andrey "
            "calibrate --method skeleton` measures a value for your machine."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_POSSIBLE_DSEP_PARALLEL_MIN_WORK",
        values="integer (subsets per possible-d-separation search)",
        default=str(MIN_POOL_DSEP_SUBSETS),
        effect=(
            "Per-search crossover gate for the parallel possible-d-separation step of FCI and "
            "GFCI with `fisherz`, `student_t`, or `normal_scores`, and of FCI-JCI with `fisherz` "
            "or `student_t`: a pair's search runs on the worker pool only when its candidate set "
            "has at least this many subsets, counted over every size it tests, else in-process. "
            "`0` sends every search to the pool. The default was measured on an AMD EPYC server "
            "running Linux with 4 and 8 workers. No effect unless the worker count (`n_jobs` / "
            "`ANDREY_N_JOBS`) is above 1. "
            "A fit with more than one worker warns once on the default when a search reaches a "
            "tenth of it; `andrey calibrate --method possible_dsep` measures a value for your "
            "machine."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_RCOT_PARALLEL_MIN_WORK",
        values="integer (RCoT tests times rows per conditioning-size pass)",
        default=str(MIN_POOL_RCOT_WORK),
        effect=(
            "Per-pass crossover gate for the parallel skeleton search of PC, FCI, GFCI, BOSS-FCI, "
            'GRaSP-FCI, CD-NOD, and RSMAX2 (`restrict="pc"`) with `rcot`: a conditioning-size '
            "pass fans out to the worker pool only when its RCoT tests times the data's rows reach "
            "this value, else in-process. `0` sends every pass to the pool. The default was "
            "measured on an Intel Core i7-7820X running Linux with 4 workers. No effect unless the "
            "worker count (`n_jobs` / `ANDREY_N_JOBS`) is above 1. A fit with more than one "
            "worker warns once on the default when a pass reaches a tenth of it; `andrey "
            "calibrate --method rcot` measures a value for your machine."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_LINGAM_PRUNE_MIN_WORK",
        values="integer (rows times ordered pairs of candidates per step)",
        default=str(MIN_PRUNE_WORK),
        effect=(
            "Per-step gate for the `pwling` causal-order search of DirectLiNGAM, "
            "MultiGroupDirectLiNGAM, VARMA-LiNGAM, and longitudinal LiNGAM: a step stops scoring "
            "a candidate once its partial score shows it cannot be picked only when the rows "
            "times the ordered pairs of candidates reach this value, else it scores every "
            "candidate in full. Both give the same order bit for bit. `0` prunes every step. The "
            "default was measured on an AMD EPYC 7713 server running Linux on one core, where "
            "pruning broke even at 60,000 to 80,000."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_TESTWISE_GROUP_BYTES",
        values="integer (bytes)",
        default=str(GROUP_BYTES),
        effect=(
            "Path gate for test-wise deletion with `fisherz` or `student_t` on data with missing "
            "values: the rows are grouped by their missingness patterns, and each block's sums "
            "come from the groups' tables, only when those tables fit in this many bytes and the "
            "other test-wise grouping variables below allow it; otherwise each block is summed "
            "over every row. The path is chosen once per test from the data. The two paths count "
            "the same rows for every tested query and agree up to rounding. `0` keeps every test "
            "on the per-row path."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_TESTWISE_GROUP_MAX_SCATTER",
        values="number from 0 to 1 (share of cells)",
        default=str(GROUP_MAX_SCATTER),
        effect=(
            "Path gate for test-wise deletion with `fisherz` or `student_t` (see "
            "`ANDREY_TESTWISE_GROUP_BYTES`): the grouped path is used only when the cells a row "
            "misses within its group's columns are at most this share of those columns' cells. "
            "Above it, the corrections for rows missing several columns of a block cost more "
            "than summing every row."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_TESTWISE_GROUP_MAX_GROUPS",
        values="integer",
        default=str(MAX_GROUPS),
        effect=(
            "Path gate for test-wise deletion with `fisherz` or `student_t` (see "
            "`ANDREY_TESTWISE_GROUP_BYTES`): the grouped path is used only when the missingness "
            "patterns form at most this many groups. Data with as many patterns as rows stop "
            "early, at little cost. `0` keeps every test on the per-row path."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_TESTWISE_GROUP_MAX_SHARED",
        values="number from 0 to 1 (share of rows)",
        default=str(MAX_SHARED),
        effect=(
            "Path gate for test-wise deletion with `fisherz` or `student_t` (see "
            "`ANDREY_TESTWISE_GROUP_BYTES`): the grouped path is used only when at least a "
            "quarter of the pairs of columns are observed together in at most this share of the "
            "rows. When nearly every pair shares more, as with outages or cells missing at "
            "random, each block lies in many groups and grouping costs more than it saves. `0` "
            "keeps every test on the per-row path unless a quarter of the pairs are never "
            "observed together."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_TESTWISE_GROUP_MIN_COLUMNS",
        values="integer",
        default=str(MIN_COLUMNS),
        effect=(
            "Path gate for test-wise deletion with `fisherz` or `student_t` (see "
            "`ANDREY_TESTWISE_GROUP_BYTES`): the grouped path is used only for a test over at "
            "least this many columns. A test over fewer, such as one built for a single query, "
            "asks too few distinct blocks to repay the grouping."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_TESTWISE_GROUP_MIN_GAIN",
        values="number of at least 1",
        default=str(MIN_GAIN),
        effect=(
            "Block gate for test-wise deletion with `fisherz` or `student_t` on the grouped path "
            "(see `ANDREY_TESTWISE_GROUP_BYTES`): a block takes the groups' sums only when its "
            "estimated per-row cost is at least this many times its grouped cost; any other "
            "block is summed over every row."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_SHARED_SCAN_MIN_WORK",
        values="integer (candidates times block size per scan)",
        default=str(MIN_SHARED_FACTOR_WORK),
        effect=(
            "Per-scan gate for the grow scans of IAMB and Inter-IAMB (also in RSMAX2's restrict "
            'phase with `restrict="iamb"` or `"inter_iamb"`) with `fisherz`, `student_t`, or '
            "`normal_scores` on complete data: a scan of two or more candidates against one "
            "conditioning set of two or more variables is answered from one Cholesky factor of the "
            "set's correlation block only when its candidates times the block size (the set plus "
            "two) reach this value, else each candidate inverts its own submatrix. The factor's "
            "p-values agree with the per-query ones to about 1e-10 relative, not bit for bit; the "
            "per-query path is the exact one. `0` factors every such scan. The default was "
            "measured on an AMD EPYC 7713 server running Linux on one core, where the factor was "
            "slower below a value of about 64 and faster from 128 for every set size sampled."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_COV_GPU_THRESHOLD",
        values="integer (elements: rows times columns)",
        default=_gpu_default("cov"),
        effect=(
            "Smallest data size (rows times columns) at which `auto` computes covariance and "
            "correlation matrices, used by the CI tests and BIC scores, on a CUDA GPU. A set value "
            "replaces the `cuda` default; it never enables `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_ENTROPY_GPU_THRESHOLD",
        values="integer (elements: rows times the columns scored at once)",
        default=_gpu_default("entropy"),
        effect=(
            "Smallest input size at which `auto` computes the entropy terms of DirectLiNGAM-based "
            "methods on a CUDA GPU. A set value replaces the `cuda` default; it never enables "
            "`mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_KCI_GPU_THRESHOLD",
        values="integer (elements: rows squared)",
        default=_gpu_default("kci"),
        effect=(
            "Smallest kernel matrix (rows squared) at which `auto` runs a KCI query with a "
            "conditioning set on a CUDA GPU, in float64. A set value replaces the `cuda` default; "
            "KCI never runs on `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_CMIKNN_GPU_THRESHOLD",
        values="integer (elements: rows squared)",
        default=_gpu_default("cmiknn"),
        effect=(
            "Smallest distance matrix (rows squared) at which `auto` runs a CMIknn query on a CUDA "
            "GPU. A set value replaces the `cuda` default; CMIknn never runs on `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_PNL_GPU_THRESHOLD",
        values="integer (elements: rows squared)",
        default=_gpu_default("pnl"),
        effect=(
            "Smallest kernel matrix (rows squared) at which `auto` runs the HSIC tests of PNL's "
            "pairs on a CUDA GPU, in float64. A set value replaces the `cuda` default; PNL never "
            "runs on `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_GPU_CALIBRATE",
        values="1/true/yes/on or 0/false/no/off",
        default="off",
        effect=(
            "Opt-in auto-tuning for `cuda`. The first `auto` use of covariance, correlation, or "
            "entropy measures the local numpy-to-`cuda` crossover, caches it by operation and "
            "machine fingerprint, and logs the cutoff at INFO. A manual `*_GPU_THRESHOLD` takes "
            "precedence. `auto` never selects `mps`, so calibration never measures it. KCI and "
            "CMIknn are not calibrated."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_SEED",
        values=f"integer from 0 to {SEED_MAX} (2**32 - 1)",
        default="0",
        effect=(
            "Global default seed for every method that draws random numbers: BOSS, GRaSP, "
            "ICA-LiNGAM, CALM, IGSP and UT-IGSP with n_starts above 1 (the default), HC and tabu "
            "search with n_starts above 1, and the searches that test independence, with the "
            "rcot, cmiknn, gpdc, gcm, wgcm, and pcm tests. An explicit `seed=` "
            "overrides it. Deterministic methods (GES, DirectLiNGAM, ...) and the other tests "
            "ignore it."
        ),
        group="reproducibility",
    ),
    EnvVarSpec(
        name="ANDREY_TIE_BREAK",
        values="rank, position",
        default="rank",
        effect=(
            "How the searches order tied candidates. `rank` orders them by a rank computed from "
            "the data, which a row shuffle, a change of units, or a sign flip leaves in place, so "
            "reordering the columns relabels the result. `position` "
            "orders them by column position, which skips computing the ranks, fires the "
            "orientation rules of `pc`, `cdnod`, `fci`, `gfci`, and `fci_jci` one at a time in "
            "column order, and visits the possible-d-separation step of `fci`, `gfci`, and "
            "`fci_jci` in column order; the result can then depend on the column order. Also "
            "`andrey.config(tie_break=...)`."
        ),
        group="reproducibility",
    ),
    EnvVarSpec(
        name="ANDREY_REQUIRE_GPU",
        values="1/true/yes/on or 0/false/no/off",
        default="off",
        effect=(
            "For the package's own test suite: its GPU tests fail, instead of skipping, when no "
            "CUDA or MPS device is present. No effect on a fit."
        ),
        group="determinism",
    ),
    EnvVarSpec(
        name="ANDREY_DATA_DIR",
        values="a directory path (`~` expands)",
        default="$XDG_CACHE_HOME/andrey, else ~/.cache/andrey",
        effect=(
            "Where `andrey.data.load_dataset` keeps downloaded datasets. A `data_home=` argument "
            "overrides it."
        ),
        group="data",
    ),
)

# Which setting wins where several apply, most specific first.
PRECEDENCE: tuple[str, ...] = (
    "Backend: `with andrey.config(backend=...)` > `andrey.config.backend = ...` > "
    "`ANDREY_BACKEND`/`ANDREY_DEVICE` > default (`auto`).",
    "Seed: explicit `seed=` argument > `ANDREY_SEED` > `0`.",
    "Tie break: `with andrey.config(tie_break=...)` > `andrey.config.tie_break = ...` > "
    "`ANDREY_TIE_BREAK` > default (`rank`).",
    "GPU threshold: manual `ANDREY_*_GPU_THRESHOLD` > `ANDREY_GPU_CALIBRATE` measurement > "
    "per-device default.",
)
