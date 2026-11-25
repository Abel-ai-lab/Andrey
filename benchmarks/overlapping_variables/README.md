# Datasets with overlapping variables

The benchmark behind the "Datasets with overlapping variables" section of the
[0.19.0 release card](../../docs/docs/benchmarks/0.19.0.md#datasets-with-overlapping-variables).
It measures what PC, FCI, PCMCI, and PCMCI+ return when cohorts, studies, or sites measure
overlapping sets of variables, so that some pairs of variables are never observed in the same row
and some tests have no rows. Each search runs once for every answer to a test without enough rows,
on the same data, and each graph is scored against the true one.

## Arms

| Arm | Answer to a test without enough rows |
|---|---|
| `a` | Andrey as it is: the test is skipped, and a pair no test decides is listed in `undecided_pairs` and left out of the graph |
| `amb` | as `a`, with `possibly_separable="ambiguous"` (PC and FCI) |
| `b` | p = 1: the test separates its pair |
| `c` | p = 0: the test keeps its pair adjacent |
| `dcover` | the fewest columns dropped until every pair shares rows, then `a` |
| `f` | the complete data, the values before masking |

Arms `b` and `c` emulate what the usual answers do with a test that has no rows, inside the same
search (`arms.py`): the filled answer counts as a tested result, a conditioning set is asked
whether or not a row observes its columns together, and a pair that p = 0 leaves adjacent stays in
the graph. They patch internals of Andrey, so they fit the build the test in `benchmarks/tests`
passes on, and the test fails when a build changes them.

## Data

Linear-Gaussian graphs of expected degree 3, a new graph, weights, and data for each seed, columns
in units with spreads from 0.1 to 100, and rows in random order.

| Design | Cells |
|---|---|
| Cohorts | `S3` (3 cohorts, 20 variables, 5,000 rows), `M8` (8, 40, 20,000), `L6` (6, 50, 30,000), each with 0, 2, or 5% of the rows as bridging rows that observe every column (`b0`, `b2`, `b5`) and then 0 or 10% of all cells, bridging rows included, missing completely at random (`s0`, `s10`), and the complete data; with `s10`, 14 (`b2`) and 29 (`b5`) of the 5,000 rows of `S3` seed 0 stay complete, and with `s0` all 100 and 250 do |
| Hand cases | the forks and colliders of three columns `H1`, `H1c`, `H4`, and `H5k3` with 1,000 and 10,000 rows, and `H1_fixed`, one fork of 1,000 rows drawn from `default_rng(0)` |
| Lagged cohorts | a VAR(1) of 10 variables and 6,000 time steps in three periods, whose two modules are never observed at the same time (`nolap0`, and `nolap5` with 5% of the cells missing at random), and the complete data |

`classes.py` sorts the pairs of a cohort dataset by its cohort patterns: class i, observed
together; class ii, never observed together; class iii, observed together but not separable by any
cohort that observes both. `score.py` scores skeleton precision and recall per class, the CPDAG's
structural Hamming distance (PC; the skeleton's for FCI), false colliders, arrowheads, and the pairs
Andrey lists against the pairs it draws.

## Run

Run the commands from this folder's parent, `benchmarks/`, in an environment with Andrey
(`andrey-core`) installed. A run uses one thread per fit and `--procs` fits side by side, and it
resumes: a fit already in the records file is skipped.

```shell
python -m overlapping_variables.run records.jsonl --size small --procs 4
python -m overlapping_variables.summary records.jsonl
```

The small size is the 20-variable configuration, two hand cases, and the lagged cells on 2 seeds
(186 fits, about 2 minutes on 4 cores). The card's size is `--size card`: 5,250 fits on seeds 0-19
over all three configurations, the hand cases, and the lagged cells. `--seeds 1000-1019` runs the
fresh seeds (`--designs cohort` restricts them to the cohorts, 3,360 fits), `--configs S3,M8` and
`--designs` narrow a run so that its parts can run apart, and `--cap` sets the seconds a fit may
take before it is recorded as a timeout (a timeout is never scored as an empty graph). The summary
reads several records files at once.

```shell
python -m overlapping_variables.run records.jsonl --size card --procs 16
python -m overlapping_variables.run records.jsonl --size card --seeds 1000-1019 --designs cohort --procs 16
python -m overlapping_variables.summary records.jsonl --json figures.json
```

The card's size takes about 2.4 CPU-hours of fit time for the 5,250 fits (18 minutes on 16 cores)
and 1.5 for the 3,360 fits of the fresh seeds (11 minutes on 16 cores); the longest fit takes about
a minute.

The summary prints the card's two tables, and with `--json` writes them with the other figures the
section's text quotes: the settings that finished with fewer errors than the empty graph, the
ranges of recall and precision, the pairs listed and never drawn (which should have no
exceptions), the `"ambiguous"` arm, the cells with complete rows, the search without a depth
limit, the lagged cells, and the graphs of the hand cases.

## What a rerun reproduces

A run at the card's size, seeds 0-19 and the fresh seeds 1000-1019, finishes its 8,610 fits with no
error and no timeout, and the summary prints the card's two tables as published and the figures its
text quotes: 1,204 fits of Andrey as it is on data with pairs never observed together (480 cohort
fits, 80 cohort fits without a depth limit, and 644 hand-case fits), none drawing or leaving
unlisted such a pair, and the 12 settings with fewer errors than the empty graph.

Of those fits, 7,148 have a graph recorded for the card, and each is the same graph, edge for edge,
with the same undecided pairs, flagged edges, and kept columns, in every arm: Andrey as it is (`a`,
2,244 fits), `ambiguous` (`amb`, 1,458), p = 1 (`b`, 1,684), p = 0 (`c`, 1,282), the fewest columns
dropped (`dcover`, 320), and the complete data (`f`, 160). None differs. The other 1,462 fits are
fresh seeds the card's run did not cover for these arms. A rerun reproduces the graphs, not the
wall times, which depend on the hardware.

The intervals index the seeds in ascending order. The card's first table gives the medium and large
configurations' Andrey intervals as [44.1, 54.8] and [49.3, 63.0], which its fits drew with the
seeds in the order they finished; `CARD_ORDER` in `summary.py` holds that order for those two cells,
and in ascending order their ends are 54.7 and 49.2. The means do not depend on the order.

## Layout

| File | What it holds |
|---|---|
| `generate.py` | the datasets: cohorts, hand cases, lagged cohorts |
| `classes.py` | the pair classes and the check of their rule against a search of every separating set |
| `arms.py` | the p = 1 and p = 0 answers and the dropped columns |
| `fit.py` | one fit of an arm, as a record |
| `run.py` | the list of fits of a size and the runner, one process per fit |
| `score.py` | the metrics of a record |
| `summary.py` | the intervals, the verdicts, and the card's tables and figures |
