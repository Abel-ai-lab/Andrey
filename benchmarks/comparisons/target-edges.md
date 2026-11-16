# Edges and directions at a target

`direct_cpu.py` runs Andrey's methods that learn the edges at one variable and their directions
(`mb_by_mb`, `cmb`) as their papers did, beside global PC read at the target: Andrey's `pc` with the
same test and bnlearn's `pc.stable`. `direct_summary.py` judges a method per cell against both PCs
and the empty answer, and sets its means beside its paper's. Neither method has a reference
implementation, so global PC, which the papers compare with, is the comparison, and the papers'
numbers are a soft check. A different method is judged on accuracy against cost, the CI tests and
seconds an answer takes, and on the Pareto frontier of the two, not on parity.

## Data

- `alarm` (37 variables), forward samples of the network's own tables from `andrey.data`, the
  columns in an order drawn from the seed; every variable is a target.
- `hailfinder` (56), forward samples of the published network, its columns permuted by the seed;
  every variable is a target.
- `chess45`, the 45 x 45 chessboard of the MB-by-MB paper: 2,025 binary variables, each square
  pointing to the squares to its right and below, with 10 targets drawn once from a fixed seed, as
  the paper draws 10 (`andrey_bench/chessboard.py` gives the tables).

## Solutions

| Solution | What it runs |
|---|---|
| `andrey.mb_by_mb`, `andrey.cmb` | the method at one target per fit, with its own defaults but `alpha` |
| `andrey.pc.gsq` | Andrey's `pc` with `gsq` on the whole dataset, read at every target |
| `bnlearn.pc-stable.graph` | bnlearn 5.2.1 `pc.stable` with `mi-adf` on the columns that vary, read at every target |
| `baseline.empty` | the empty answer, the floor |

All run at the same `--alpha`; `--max-depth` caps the conditioning sets of the two PCs only.

## Measures

Each answer is scored on the edges at its target in both papers' readings
(`andrey_bench/direct_lane.py`). Against the DAG (`dag_counts`, the MB-by-MB paper's): undirected,
missing, extra, and reversed edges, whose sum is the SHD. Against the truth's CPDAG (`cpdag_counts`,
the CMB paper's): extra, missing, and reversed edges, an edge the CPDAG directs and the answer
leaves undirected (`under`) among the reversed, whose sum is `total`. `direct_summary.py` takes a
dataset's mean over its targets, pairs the datasets of two solutions, and gives the mean
difference with its 95% percentile-bootstrap interval: better or worse when the interval lies
wholly on that side of 0 over at least 10 datasets. A cell worse on seeds 0-19 counts only when it
is worse again on seeds 20-39.

A row of `mb_by_mb` also gives each solution's cost: `tests_per_answer` and `seconds_per_answer`,
what answering one target costs (the tests and seconds of MB-by-MB's or CMB's walk at that target,
and of a global PC's whole fit, which answers every target), and `tests_all_targets` and
`seconds_all_targets`, the cost over every target (the sum of the walks, or the one fit); each is
the mean over the datasets. Its `frontier` says, per comparison, whether the comparison is more
accurate (MB-by-MB's verdict against it is worse) and takes fewer tests per answer, which dominates
MB-by-MB in the cell (`dominated`, `dominated_by`), and the same over every target
(`dominated_all_targets`). The tests of a global PC are the ones it spent on the fit; bnlearn's
`pc.stable` counts the tests it ran.

## CMB against global PC

On the CMB paper's setting, Alarm at 1,000 rows with G-square tests at alpha 0.02 and no limit on
the conditioning set, CMB's mean total per target was 0.751 on seeds 0-19, against 0.997 for
Andrey's PC (-0.246 [-0.368, -0.118]), 1.800 for bnlearn's `pc.stable` (-1.049 [-1.137, -0.962]),
and 2.486 for the empty answer; on seeds 20-39 it was 0.689 against 0.954 and 1.822. Without the
undirected edges, 0.430 against 0.638 and 1.335. The paper reports 0.69 extra, 0.61 missing, and
0.51 reversed edges per target, 1.81 in all; CMB's extra edges here are 0.02. Without the
symmetry check of HITON-MB, CMB's default blanket learner, they were 0.21; counting the degrees of
freedom from the cells that occur, or capping the conditioning sets at 3, left them at 0.01 to 0.02.
With exact tests, by d-separation in the true graph, CMB's only errors were 0.09 edges per target
left undirected that the CPDAG directs. The [0.18.0 release card][card] has the table.

## MB-by-MB against global PC and CMB

On the MB-by-MB paper's settings, Alarm, Hailfinder, and the 45 x 45 chessboard at 500, 1,000,
2,000, and 5,000 rows with G-square tests at alpha 0.01 and 0.05, IAMB's grow phase for the
blankets, and at most 5 conditioning variables in the local structures and in both PCs, MB-by-MB
is judged on the SHD per target against the true DAG (the paper's) and on the cost of an answer
against Andrey's PC, bnlearn's `pc.stable`, and CMB where it ran (Alarm and Hailfinder). Alarm and
Hailfinder cells hold 20 datasets (seeds 0-19) and 20 fresh ones (seeds 20-39); chessboard cells
hold 10 and 10 (seeds 0-9 and 10-19), with fresh PC fits at 500 and 1,000 rows only, since a PC fit
there takes 4 to 39 million tests. The cap is 1,800 s per fit, 3,600 s on the chessboard.

No cell is dominated per answer: an answer took 279 to 930 tests on Alarm and Hailfinder against
3,954 to 14,070 for one fit of Andrey's PC, and 27,000 to 93,000 on the chessboard against 4.4 to
39 million. MB-by-MB was better than `pc.stable` in all 16 Alarm and Hailfinder cells and better
than Andrey's PC in 8, tied in 6, and worse in 2, Alarm at 5,000 rows and alpha 0.05 and
Hailfinder at 500 rows and alpha 0.05. On the chessboard it was worse in all 8 cells on the first
datasets, and again on the fresh ones at 500 and 1,000 rows. For every target at once, one PC fit
takes fewer tests than the 37 or 56 walks in every Alarm and Hailfinder cell, and is better on
both accuracy and tests in those two. The [0.18.0 release card][card-mb] has the tables and the plot.

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with bnlearn 5.2.1 in the library, for
the bnlearn arm; the hailfinder network comes from the published networks the harness downloads.

```shell
$ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/direct-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/cmb --paper cmb --seeds 40
$ANDREY_BENCH_PYTHON benchmarks/direct_summary.py outputs/cmb --method cmb --json cmb-rows.json
for alpha in 0.01 0.05; do
  $ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/mb-by-mb-$alpha --paper mb_by_mb \
    --alpha $alpha --seeds 40
  $ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/mb-by-mb-cmb-$alpha --paper mb_by_mb \
    --alpha $alpha --seeds 40 --solutions andrey.cmb
  $ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/mb-by-mb-chess-$alpha \
    --paper mb_by_mb --networks chess45 --alpha $alpha --seeds 20 --cap-wall-s 3600
done
$ANDREY_BENCH_PYTHON benchmarks/direct_summary.py outputs/mb-by-mb-* --method mb_by_mb \
  --json mb-by-mb-rows.json
```

`--paper` sets the networks, rows, level, conditioning-set cap, and solutions of a paper (`cmb`:
alarm at 1,000 rows, alpha 0.02, no cap; `mb_by_mb`: alarm and hailfinder at 500, 1,000, 2,000,
and 5,000 rows, alpha 0.05, a cap of 5, MB-by-MB and the two PCs); `--networks`, `--rows`,
`--alpha`, `--max-depth`, and `--solutions` override them (the paper's level 0.01 with
`--alpha 0.01`, the chessboard with `--networks chess45`, CMB's arm with `--solutions
andrey.cmb`), and `--part` with `--parts` splits the datasets over jobs. A fit
starts a process and has a wall cap (`--cap-wall-s`, 1,800 s); a fit past it is a timeout, never
an empty answer. The records go to `records.jsonl`, one line per fit and target with the marks at
the target, so `--rescore` recomputes the scores without fitting, and the build and machine to
`provenance.json`.

[card]: https://andrey.abel.ai/docs/benchmarks/0.18.0.html#cmb
[card-mb]: https://andrey.abel.ai/docs/benchmarks/0.18.0.html#mb-by-mb
