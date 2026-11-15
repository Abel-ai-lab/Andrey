# Local ISA-LiNG

`local_lingam_cpu.py` measures the weighted edges a method learns into one target and into each of
its children. It runs Andrey's `local_isa_ling` and the authors' local ISA-LiNG code on the
simulations of the method's paper (Dai, Ng, Zheng, Gao, and Zhang, 2024, Section 5), with each
method's own Markov blanket and with the true one, and global DirectLiNGAM and ICA-LiNGAM read at
the target, and it scores each answer's local graph against the truth's.

## Data

The models are linear with non-Gaussian noise, `X = B X + E`, on 50 variables
(`andrey_bench/local_ling_models.py`): each nonzero weight is uniform in magnitude on `[0.5, 0.9]`
with a random sign, and each noise term is uniform on `[-c, c]` raised to the fifth power, with `c`
uniform on `[0.75, 1.25]`, the paper's draws.

- **Graphs with cycles** (`cyclic`). Every variable has at most 4 neighbors, every cycle is
  disjoint from the others, and the weights are drawn again until the model is stable (spectral
  radius of `B` below 1). The target lies on a cycle.
- **DAGs** (`acyclic-d3`, `acyclic-d5`). Erdos-Renyi DAGs of mean degree 3 and 5. The target is the
  variable whose true blanket size is closest to the paper's "roughly 14 and 20".
- **Rows and seeds.** 2,000 rows, the paper's, in every graph, and the paper's ladder of 100, 300,
  1,000, 3,000, and 10,000 rows at degree 3; 20 datasets (seeds 0-19) per cell. The columns come in
  an order drawn from the seed, so neither the graph nor the target follows it.
- **Sachs.** The 853 observational cells of Sachs et al. (2005) at the paper's targets PIP2, PIP3,
  and Akt, scored on the 17-arc consensus network, which has no weights.

## Solutions

- `andrey.local_isa_ling` runs `andrey.local_isa_ling` at its defaults, `w_threshold=0.05` and
  `alpha=0.05`, with `acyclic=True` on the DAGs and on Sachs (the paper's Appendix C) and `False`
  on the graphs with cycles (Algorithm 1). Its blanket is the support of a Lasso of the
  standardized target on the other standardized columns.
- `andrey.local_isa_ling.oracle` runs the same engine on the true blanket.
- `local_isa_ling.authors` and `local_isa_ling.authors.oracle` are the authors' code,
  `MarkDana/Local-LiNG-Discovery` at a pinned commit (MIT license), run unmodified in an environment
  of its own as its examples run it: its Lasso blanket, or the true one, then `acyclic_sink` on the
  DAGs and `block_diagonal` (its most stable model) on the graphs with cycles.
  `benchmarks/start_local_ling.sh` clones it at that commit. Its Algorithm 1 tries every row
  permutation of the local demixing matrix, so the adapter runs it only up to 10 local variables,
  and a larger blanket is recorded as not run, never as an empty answer.
- `andrey.direct_lingam` and `andrey.ica_lingam` fit Andrey's global DirectLiNGAM and ICA-LiNGAM
  on all 50 columns of a DAG and are read at the target. They are a different method, set beside
  local ISA-LiNG on accuracy against cost, not judged against it.
- `empty` answers no edge.

Each fit runs in a fresh process on one core with one BLAS thread, with a cap of 1,800 s (3,600 s
at degree 5 and at 10,000 rows); a fit past it is a timeout and never scored.

## Measures

A local graph holds the edges into and out of the target and the edges into each of its children,
each graph's own children (the paper's Appendix E.3).

- `shd`, the paper's SHD of the local graph, counted as the authors' code counts it: the
  adjacencies one local graph has and the other lacks, plus the edges whose reverse alone is true.
- `weight_dist`, the Euclidean norm of the difference of the two local weight matrices.

The empty answer's SHD is the number of true local edges, and its weight distance their norm.

## Verdicts

Per cell (graph, rows, blanket), the judged difference is Andrey's minus the authors' code per
dataset, on the datasets both finished, with its percentile bootstrap 95% interval over the
datasets (10,000 resamples, `default_rng(0)`). A cell is worse when the interval of either measure
lies wholly above 0 on seeds 0-19 and again on 20 fresh datasets (seeds 20-39), better when it lies
wholly below 0 on seeds 0-19, and a tie otherwise. The floor is each measure minus the empty
answer's, which a solution passes when its interval lies wholly below 0.

## Results

Release 0.18.0 judged `andrey.local_isa_ling` against the authors' code in every cell, 20 datasets
(seeds 0-19) each, and reran each cell worse on seeds 0-19 on seeds 20-39.
`published/0.18.0/local-isa-ling.json` holds every cell, and the release summary's Local ISA-LiNG
rows are built from it; the method's claim is graphs with cycles, so the DAG rows are reported, not
gated. The [release card][card] reads it.

| Graph | Rows | Blanket | Datasets | SHD, Andrey | SHD, authors | Distance, Andrey | Distance, authors | Result | SHD, empty answer |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| Cycles | 2,000 | own | 12 | 0.92 | 1.00 | 0.55 | 0.46 | tie | 5.12 |
| Cycles | 2,000 | true | 20 | 0.25 | 0.15 | 0.19 | 0.15 | tie | 4.80 |
| DAG, degree 3 | 100 | own | 20 | 10.55 | 9.65 | 2.35 | 2.21 | tie | 14.30 |
| DAG, degree 3 | 100 | true | 20 | 7.15 | 6.45 | 1.95 | 1.95 | worse, on both seed sets | 14.30 |
| DAG, degree 3 | 300 | own | 20 | 8.00 | 8.75 | 2.10 | 2.14 | tie | 14.30 |
| DAG, degree 3 | 300 | true | 20 | 6.05 | 4.65 | 1.84 | 1.76 | worse, on both seed sets | 14.30 |
| DAG, degree 3 | 1,000 | own | 20 | 6.15 | 8.05 | 1.80 | 2.16 | better | 14.30 |
| DAG, degree 3 | 1,000 | true | 20 | 3.95 | 2.85 | 1.60 | 1.43 | tie (worse on seeds 0-19 only) | 14.30 |
| DAG, degree 3 | 2,000 | own | 20 | 5.65 | 7.25 | 1.58 | 2.35 | better | 14.30 |
| DAG, degree 3 | 2,000 | true | 20 | 4.20 | 4.70 | 1.30 | 1.65 | better | 14.30 |
| DAG, degree 3 | 3,000 | own | 20 | 5.80 | 6.20 | 1.55 | 2.17 | better | 14.30 |
| DAG, degree 3 | 3,000 | true | 20 | 2.90 | 1.95 | 0.98 | 1.40 | tie (worse on seeds 0-19 only) | 14.30 |
| DAG, degree 3 | 10,000 | own | 20 | 4.70 | 5.20 | 1.48 | 1.59 | tie | 14.30 |
| DAG, degree 3 | 10,000 | true | 20 | 2.45 | 2.30 | 0.85 | 0.81 | tie | 14.30 |
| DAG, degree 5 | 2,000 | own | 20 | 15.00 | 17.00 | 2.69 | 3.08 | tie | 24.75 |
| DAG, degree 5 | 2,000 | true | 20 | 9.95 | 9.85 | 2.42 | 2.61 | tie | 24.75 |

The SHD and distance differences are Andrey's minus the authors' (negative is better), each with
its interval; the last column is Andrey's SHD minus the empty answer's.

| Graph | Rows | Blanket | SHD difference | Distance difference | Fresh SHD | Fresh distance | SHD minus empty |
|---|---:|---|---|---|---|---|---|
| Cycles | 2,000 | own | -0.08 [-0.83, +0.67] | +0.09 [-0.28, +0.53] | - | - | -4.20 [-4.75, -3.65] |
| Cycles | 2,000 | true | +0.10 [+0.00, +0.30] | +0.04 [-0.00, +0.10] | - | - | -4.55 [-5.30, -3.90] |
| DAG, degree 3 | 100 | own | +0.90 [-1.00, +2.75] | +0.14 [-0.12, +0.39] | - | - | -3.75 [-5.85, -1.75] |
| DAG, degree 3 | 100 | true | +0.70 [+0.10, +1.30] | -0.00 [-0.16, +0.14] | +0.90 [+0.20, +1.70] | +0.10 [+0.02, +0.22] | -7.15 [-9.05, -5.15] |
| DAG, degree 3 | 300 | own | -0.75 [-2.55, +0.90] | -0.04 [-0.29, +0.20] | - | - | -6.30 [-8.25, -4.35] |
| DAG, degree 3 | 300 | true | +1.40 [+0.70, +2.15] | +0.08 [-0.04, +0.19] | +2.05 [+0.95, +3.50] | +0.13 [-0.01, +0.26] | -8.25 [-10.50, -6.00] |
| DAG, degree 3 | 1,000 | own | -1.90 [-4.20, +0.15] | -0.36 [-0.78, -0.04] | - | - | -8.15 [-9.90, -6.35] |
| DAG, degree 3 | 1,000 | true | +1.10 [+0.45, +1.85] | +0.17 [+0.07, +0.29] | +0.15 [-0.85, +0.95] | -0.24 [-0.54, +0.03] | -10.35 [-12.10, -8.60] |
| DAG, degree 3 | 2,000 | own | -1.60 [-3.15, -0.15] | -0.76 [-1.20, -0.38] | - | - | -8.65 [-10.70, -6.50] |
| DAG, degree 3 | 2,000 | true | -0.50 [-1.30, +0.15] | -0.35 [-0.75, -0.04] | - | - | -10.10 [-12.25, -7.80] |
| DAG, degree 3 | 3,000 | own | -0.40 [-2.25, +1.40] | -0.62 [-1.49, -0.02] | - | - | -8.50 [-10.40, -6.70] |
| DAG, degree 3 | 3,000 | true | +0.95 [+0.10, +2.20] | -0.41 [-1.30, +0.15] | +1.15 [-0.20, +2.85] | +0.04 [-0.27, +0.34] | -11.40 [-13.15, -9.55] |
| DAG, degree 3 | 10,000 | own | -0.50 [-1.80, +0.85] | -0.11 [-0.34, +0.14] | - | - | -9.60 [-11.55, -7.55] |
| DAG, degree 3 | 10,000 | true | +0.15 [-0.90, +1.30] | +0.04 [-0.12, +0.25] | - | - | -11.85 [-13.75, -9.75] |
| DAG, degree 5 | 2,000 | own | -2.00 [-5.10, +1.15] | -0.40 [-1.01, +0.10] | - | - | -9.75 [-13.20, -6.00] |
| DAG, degree 5 | 2,000 | true | +0.10 [-1.15, +1.30] | -0.18 [-0.39, +0.00] | - | - | -14.80 [-17.60, -11.80] |

Global DirectLiNGAM and ICA-LiNGAM read at the target, with the mean SHD and the median seconds of
one answer (the global fits on the graphs with cycles were timed in one process):

| Graph | Rows | Local ISA-LiNG | DirectLiNGAM, global | ICA-LiNGAM, global | Empty |
|---|---:|---|---|---|---:|
| DAG, degree 3 | 100 | 10.55 / 5.6 s | 24.25 / 2.9 s (dominated) | 36.80 / 2.1 s (dominated) | 14.30 |
| DAG, degree 3 | 300 | 8.00 / 9.0 s (dominated) | 6.80 / 6.7 s | 42.10 / 2.3 s (dominated) | 14.30 |
| DAG, degree 3 | 1,000 | 6.15 / 16.6 s (dominated) | 1.85 / 6.7 s | 2.15 / 1.8 s | 14.30 |
| DAG, degree 3 | 2,000 | 5.65 / 27.4 s (dominated) | 1.25 / 7.7 s (dominated) | 1.10 / 1.7 s | 14.30 |
| DAG, degree 3 | 3,000 | 5.80 / 43.7 s (dominated) | 0.60 / 12.9 s | 0.65 / 2.6 s | 14.30 |
| DAG, degree 3 | 10,000 | 4.70 / 137.4 s (dominated) | 0.15 / 31.0 s (dominated) | 0.15 / 3.2 s | 14.30 |
| DAG, degree 5 | 2,000 | 15.00 / 37.0 s (dominated) | 1.30 / 10.6 s | 3.60 / 2.2 s | 24.75 |
| Cycles | 2,000 | 0.60 / 28.6 s | 4.40 / 4.6 s | 7.25 / 0.7 s | 4.80 |

On Sachs, at PIP2, PIP3, and Akt, Andrey and the authors' code each made 2, 2, and 1 errors with
either blanket, against the empty answer's 2, 3, and 2.

## Run

Run `start_local_ling.sh` and export what it prints, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/local_lingam_cpu.py --out outputs/lil-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/local_lingam_cpu.py --out outputs/lil --cores 1
$ANDREY_BENCH_PYTHON benchmarks/local_lingam_cpu.py --out outputs/lil-ladder --cells acyclic-d3 \
  --rows 100 300 1000 3000 10000 --cores 1 --cap-wall-s 3600
$ANDREY_BENCH_PYTHON benchmarks/local_lingam_cpu.py --out outputs/lil-sachs --cells sachs --cores 1
$ANDREY_BENCH_PYTHON benchmarks/local_lingam_summary.py outputs/lil outputs/lil-ladder \
  outputs/lil-sachs --json cells.json
```

The defaults are the three graphs at 2,000 rows, 20 seeds from `--first-seed 0`, and `--cap-wall-s`
of 1800. `--part` and `--parts` split the datasets between jobs, `--solutions` selects solutions,
`--resume` continues a run, and `--rescore` recomputes the scores from the stored answers without
fitting. One JSON line per fit goes to `records.jsonl`: the solution, the cell, the seed, the target,
the time, the answer's local weights and blanket, and the measures above. A later run folder's
record replaces an earlier one's in the summary, so a rerun of some solutions is listed after the
run it updates.

[card]: https://andrey.abel.ai/docs/benchmarks/0.18.0.html#local-isa-ling
