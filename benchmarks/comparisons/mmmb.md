# MMMB

This campaign compares Andrey's `mmmb` with the authors' MMMB in Causal Explorer 1.5 on the Markov
blanket each learns at every target, on the networks and sizes of its paper. `blanket_cpu.py` fits
the solutions, as [Markov blankets](blankets.md) describes, and `blanket_summary.py` judges each
cell. The rows are in [`published/0.18.0/mmmb.json`](../published/0.18.0/mmmb.json), and the
[0.18.0 release card](../../docs/docs/benchmarks/0.18.0.md#mmmb) summarizes them.

## Data

The paper (Tsamardinos, Aliferis, and Statnikov, 2003) ran MMMB on samples of alarm, hailfinder,
pigs, insurance, and win95pts at 300, 500, 700, 1,000, and 5,000 rows with a G-square test at 0.05.
We ran alarm (37 variables), hailfinder (56), insurance (27), and win95pts (76) at those five sizes
with every variable a target, and pigs (441 variables) at 500 rows with 50 targets drawn from each
dataset's seed. Every cell has 20 datasets, seeds 0 to 19, and a cell rerun on fresh datasets has
seeds 20 to 39. A column that holds one value is left out of every fit and answers the empty set at
its target. The paper runs categorical data only, so the campaign has no continuous cells.

## Solutions

| Solution | Setting |
|---|---|
| `andrey.mmmb` | the defaults: `max_depth=None`, `degrees_of_freedom="nonempty"`, alpha 0.05, `"gsq"` |
| Causal Explorer `MMMB` | no cap on the conditioning set, G-square at 0.05 run only with at least five rows per cell of its table (a fixed setting), at each target; the toolbox's form runs no symmetry check |
| context | `iamb`, `inter_iamb`, `gs`, and `pcmb` at their defaults (seeds 0 to 9); `mmmb` with `degrees_of_freedom="mean"` |

The toolbox stops at a target now and then with "Deseparating set S is not found". Such a target
is left out of both sides of its dataset's pair, and the tables count the targets left.

## Measures

Per dataset we average, over the targets with a nonempty true blanket, the distance of the set's
precision and recall from (1, 1), where an empty answer counts precision 1 and distance 1, and the
set's F1. D is Andrey's value minus the toolbox's per dataset. A cell is worse when the 95%
percentile-bootstrap interval of the mean D (10,000 resamples of the datasets) lies on Andrey's
worse side on seeds 0 to 19 and again on seeds 20 to 39, better when it lies on the better side on
seeds 0 to 19, and a tie otherwise. The floor counts the errors per target, missing plus extra
members, against the empty set's, which are the true set's size; `mmmb` passes when the paired
interval of the difference lies below 0.

## Results (0.18.0)

Against the toolbox, `mmmb` was better in 8 of the 21 settings, tied in 3, and worse in 10, where
the first datasets and the 20 fresh ones agree: hailfinder at 500, 700, 1,000, and 5,000 rows,
insurance at 5,000 rows on F1, and win95pts at every size (at 300 rows on F1). On the fresh datasets
the distance on hailfinder at 5,000 rows was 0.606 against 0.517, and on win95pts at 1,000 rows
0.662 against 0.612. Alarm at 1,000 rows was worse on F1 on the first datasets and tied on the fresh
ones.

A trace that logs the toolbox's tests and replays them with Andrey's G-square, which gives the same
p-values, finds three differences. The toolbox's MMMB runs MMPC without the symmetry check that its
authors added to it (Tsamardinos, Brown, and Aliferis, 2006); Andrey's keeps a variable only when
the target is also among the variable's parents and children. With the check left out, Andrey's sets
tie or beat the toolbox's on hailfinder at 500, 700, and 1,000 rows, insurance at 5,000 rows, and
win95pts at 500 rows; the check raises the precision, 0.84 against 0.65 on hailfinder at 500 rows,
at a larger cost in recall, 0.39 against 0.59. On win95pts and hailfinder some columns are copies of
each other, so a test given one of a pair has no degrees of freedom: the toolbox returns no p-value
and never separates there, and Andrey reads the test as independence, which holds in the data. For a
candidate spouse the toolbox searches afresh for a separating set, the first subset of the target's
parents and children with a p-value above alpha, where Andrey takes the set MMPC recorded; any
separating set meets the paper's definition. With all three set the toolbox's way, Andrey's sets tie
the toolbox's on win95pts at 300, 700, 1,000, and 5,000 rows, equal on 95 to 98% of the targets. On
hailfinder at 5,000 rows they come closer, 0.528 against 0.525 in distance and 0.604 against 0.577
in F1 (seeds 0 to 19), but stay worse in distance. There they equal the toolbox's on every target of
14 datasets only when a test also runs only with at least five rows per cell of its table, the
toolbox's fixed setting and its form of the rule of the paper's experiments. `mmmb` runs every test;
its errors there are 2.13 per target, against 6.92 with all four set the toolbox's way (4.12 with
the first three).

Andrey's errors are below the toolbox's in every setting, from 0.74 against 1.93 per target on pigs
to 3.23 against 3.31 on win95pts at 5,000 rows, and below the empty set's in all 21. As in the
paper, the distance was no worse than IAMB's and GS's on alarm and insurance at every size (seeds 0
to 9). `mmmb` keeps `degrees_of_freedom="nonempty"`, the degrees of freedom of Spirtes et al. that
its paper names; the context tables give `"mean"` as well. A median fit took 1.2 to 13 s on the four
networks and 39 s on pigs; the toolbox's took 23 to 552 s and 71 s.

### Against Causal Explorer

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 20 | 0.375 / 0.448 | -0.073 [-0.092, -0.052] | 0.705 / 0.672 | +0.033 [+0.015, +0.048] | better / better | - | 1.56 / 3.02 (3.52) |
| alarm, 500 | 20 | 0.304 / 0.347 | -0.043 [-0.063, -0.024] | 0.765 / 0.753 | +0.012 [-0.002, +0.025] | better / tie | - | 1.32 / 2.13 (3.51) |
| alarm, 700 | 20 | 0.249 / 0.305 | -0.056 [-0.071, -0.041] | 0.811 / 0.782 | +0.028 [+0.018, +0.039] | better / better | - | 1.13 / 1.94 (3.51) |
| alarm, 1,000 | 20 | 0.215 / 0.218 | -0.002 [-0.018, +0.013] | 0.836 / 0.851 | -0.016 [-0.029, -0.003] | tie / worse | better / tie | 1.00 / 1.12 (3.52) |
| alarm, 5,000 | 20 | 0.111 / 0.178 | -0.066 [-0.078, -0.056] | 0.920 / 0.882 | +0.038 [+0.030, +0.045] | better / better | - | 0.57 / 0.87 (3.51) |
| hailfinder, 300 | 20 | 0.651 / 0.768 | -0.117 [-0.135, -0.097] | 0.474 / 0.363 | +0.111 [+0.094, +0.125] | better / better | - | 2.25 / 13.77 (3.42) |
| hailfinder, 500 | 20 | 0.677 / 0.652 | +0.025 [+0.010, +0.040] | 0.450 / 0.470 | -0.020 [-0.029, -0.011] | worse / worse | worse / worse | 2.17 / 9.14 (3.27) |
| hailfinder, 700 | 20 | 0.674 / 0.624 | +0.051 [+0.040, +0.063] | 0.449 / 0.497 | -0.049 [-0.056, -0.041] | worse / worse | worse / worse | 2.41 / 8.24 (3.53) |
| hailfinder, 1,000 | 20 | 0.666 / 0.578 | +0.087 [+0.076, +0.100] | 0.459 / 0.536 | -0.077 [-0.086, -0.070] | worse / worse | worse / worse | 2.35 / 7.31 (3.54) |
| hailfinder, 5,000 | 20 | 0.619 / 0.525 | +0.094 [+0.086, +0.103] | 0.495 / 0.577 | -0.083 [-0.087, -0.079] | worse / worse | worse / worse | 2.13 / 6.94 (3.54) |
| insurance, 300 | 20 | 0.598 / 0.647 | -0.049 [-0.070, -0.029] | 0.528 / 0.518 | +0.009 [-0.008, +0.027] | better / tie | - | 3.13 / 5.83 (5.18) |
| insurance, 500 | 20 | 0.540 / 0.559 | -0.019 [-0.042, +0.002] | 0.588 / 0.595 | -0.007 [-0.025, +0.011] | tie / tie | - | 2.81 / 4.05 (5.15) |
| insurance, 700 | 20 | 0.502 / 0.534 | -0.031 [-0.051, -0.014] | 0.619 / 0.615 | +0.004 [-0.011, +0.021] | better / tie | - | 2.64 / 3.75 (5.18) |
| insurance, 1,000 | 20 | 0.445 / 0.458 | -0.013 [-0.029, +0.002] | 0.666 / 0.675 | -0.010 [-0.021, +0.002] | tie / tie | - | 2.42 / 2.95 (5.19) |
| insurance, 5,000 | 20 | 0.329 / 0.324 | +0.005 [-0.009, +0.020] | 0.765 / 0.783 | -0.019 [-0.030, -0.007] | tie / worse | tie / worse | 1.81 / 1.99 (5.19) |
| pigs, 500 | 20 | 0.129 / 0.366 | -0.237 [-0.249, -0.225] | 0.917 / 0.757 | +0.161 [+0.152, +0.169] | better / better | - | 0.74 / 1.93 (3.64) |
| win95pts, 300 | 20 | 0.776 / 0.769 | +0.007 [-0.010, +0.024] | 0.329 / 0.416 | -0.087 [-0.104, -0.071] | tie / worse | worse / worse | 4.98 / 5.47 (5.91) |
| win95pts, 500 | 20 | 0.740 / 0.708 | +0.032 [+0.018, +0.047] | 0.378 / 0.470 | -0.092 [-0.103, -0.083] | worse / worse | worse / worse | 4.79 / 5.27 (5.91) |
| win95pts, 700 | 20 | 0.694 / 0.648 | +0.046 [+0.036, +0.057] | 0.423 / 0.519 | -0.096 [-0.104, -0.089] | worse / worse | worse / worse | 4.60 / 4.91 (5.91) |
| win95pts, 1,000 | 20 | 0.652 / 0.599 | +0.053 [+0.039, +0.066] | 0.464 / 0.557 | -0.094 [-0.103, -0.084] | worse / worse | worse / worse | 4.34 / 4.63 (5.92) |
| win95pts, 5,000 | 20 | 0.454 / 0.416 | +0.038 [+0.028, +0.048] | 0.651 / 0.705 | -0.054 [-0.062, -0.046] | worse / worse | worse / worse | 3.23 / 3.31 (5.89) |

### Context

The context arms are reported and never judged; `iamb`, `inter_iamb`, `gs`, and `pcmb` ran on seeds
0 to 9, and `mmmb` with `degrees_of_freedom="mean"`, its former default, on an earlier build of the
release branch.

#### `mmmb` against iamb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 10 | 0.381 / 0.492 | -0.111 [-0.137, -0.084] | 0.697 / 0.649 | +0.048 [+0.025, +0.071] | better / better | - | 1.58 / 2.37 (3.51) |
| alarm, 500 | 10 | 0.312 / 0.428 | -0.116 [-0.150, -0.078] | 0.759 / 0.702 | +0.057 [+0.030, +0.081] | better / better | - | 1.35 / 2.17 (3.51) |
| alarm, 700 | 10 | 0.267 / 0.381 | -0.114 [-0.134, -0.096] | 0.795 / 0.737 | +0.058 [+0.042, +0.076] | better / better | - | 1.18 / 1.96 (3.51) |
| alarm, 1,000 | 10 | 0.221 / 0.375 | -0.154 [-0.179, -0.133] | 0.833 / 0.741 | +0.092 [+0.074, +0.114] | better / better | - | 1.01 / 1.94 (3.51) |
| alarm, 5,000 | 10 | 0.118 / 0.331 | -0.213 [-0.247, -0.181] | 0.916 / 0.774 | +0.142 [+0.121, +0.167] | better / better | - | 0.59 / 1.72 (3.51) |
| hailfinder, 300 | 10 | 0.698 / 0.667 | +0.031 [-0.054, +0.110] | 0.425 / 0.515 | -0.091 [-0.143, -0.036] | tie / worse | - | 2.50 / 2.99 (3.54) |
| hailfinder, 500 | 10 | 0.686 / 0.627 | +0.059 [-0.023, +0.139] | 0.440 / 0.544 | -0.104 [-0.161, -0.044] | tie / worse | - | 2.43 / 2.88 (3.54) |
| hailfinder, 700 | 10 | 0.666 / 0.628 | +0.038 [-0.036, +0.108] | 0.453 / 0.543 | -0.090 [-0.137, -0.042] | tie / worse | - | 2.38 / 2.93 (3.54) |
| hailfinder, 1,000 | 10 | 0.667 / 0.620 | +0.047 [-0.028, +0.119] | 0.457 / 0.549 | -0.092 [-0.142, -0.039] | tie / worse | - | 2.34 / 2.92 (3.54) |
| hailfinder, 5,000 | 10 | 0.618 / 0.614 | +0.004 [-0.057, +0.063] | 0.495 / 0.548 | -0.053 [-0.095, -0.009] | tie / worse | - | 2.12 / 3.16 (3.54) |
| insurance, 300 | 10 | 0.603 / 0.583 | +0.021 [-0.021, +0.066] | 0.523 / 0.578 | -0.055 [-0.093, -0.019] | tie / worse | - | 3.17 / 3.45 (5.19) |
| insurance, 500 | 10 | 0.548 / 0.535 | +0.013 [-0.015, +0.041] | 0.581 / 0.621 | -0.040 [-0.066, -0.015] | tie / worse | - | 2.86 / 3.14 (5.19) |
| insurance, 700 | 10 | 0.510 / 0.497 | +0.013 [-0.007, +0.035] | 0.614 / 0.649 | -0.035 [-0.053, -0.021] | tie / worse | - | 2.66 / 3.00 (5.19) |
| insurance, 1,000 | 10 | 0.457 / 0.452 | +0.005 [-0.013, +0.023] | 0.653 / 0.687 | -0.034 [-0.051, -0.016] | tie / worse | - | 2.44 / 2.76 (5.19) |
| insurance, 5,000 | 10 | 0.334 / 0.361 | -0.027 [-0.065, +0.003] | 0.758 / 0.753 | +0.005 [-0.021, +0.037] | tie / tie | - | 1.84 / 2.34 (5.19) |
| win95pts, 300 | 10 | 0.782 / 0.733 | +0.050 [+0.035, +0.067] | 0.325 / 0.432 | -0.107 [-0.119, -0.096] | worse / worse | - | 5.00 / 5.09 (5.92) |
| win95pts, 500 | 10 | 0.752 / 0.656 | +0.097 [+0.075, +0.113] | 0.367 / 0.501 | -0.134 [-0.145, -0.121] | worse / worse | - | 4.85 / 4.80 (5.92) |
| win95pts, 700 | 10 | 0.700 / 0.609 | +0.090 [+0.068, +0.111] | 0.415 / 0.537 | -0.121 [-0.141, -0.102] | worse / worse | - | 4.65 / 4.58 (5.92) |
| win95pts, 1,000 | 10 | 0.649 / 0.568 | +0.081 [+0.052, +0.104] | 0.460 / 0.572 | -0.112 [-0.129, -0.090] | worse / worse | - | 4.34 / 4.36 (5.92) |
| win95pts, 5,000 | 10 | 0.451 / 0.479 | -0.029 [-0.048, -0.011] | 0.652 / 0.650 | +0.002 [-0.010, +0.016] | better / tie | - | 3.24 / 3.91 (5.92) |

#### `mmmb` against inter_iamb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 10 | 0.381 / 0.485 | -0.104 [-0.130, -0.075] | 0.697 / 0.656 | +0.041 [+0.016, +0.064] | better / better | - | 1.58 / 2.35 (3.51) |
| alarm, 500 | 10 | 0.312 / 0.429 | -0.117 [-0.151, -0.080] | 0.759 / 0.701 | +0.058 [+0.031, +0.082] | better / better | - | 1.35 / 2.19 (3.51) |
| alarm, 700 | 10 | 0.267 / 0.384 | -0.117 [-0.135, -0.100] | 0.795 / 0.736 | +0.060 [+0.044, +0.077] | better / better | - | 1.18 / 1.96 (3.51) |
| alarm, 1,000 | 10 | 0.221 / 0.377 | -0.156 [-0.181, -0.133] | 0.833 / 0.739 | +0.094 [+0.074, +0.116] | better / better | - | 1.01 / 1.94 (3.51) |
| alarm, 5,000 | 10 | 0.118 / 0.344 | -0.226 [-0.252, -0.198] | 0.916 / 0.767 | +0.150 [+0.132, +0.169] | better / better | - | 0.59 / 1.77 (3.51) |
| hailfinder, 300 | 10 | 0.698 / 0.667 | +0.032 [-0.053, +0.111] | 0.425 / 0.516 | -0.091 [-0.143, -0.036] | tie / worse | - | 2.50 / 2.99 (3.54) |
| hailfinder, 500 | 10 | 0.686 / 0.626 | +0.060 [-0.022, +0.139] | 0.440 / 0.545 | -0.105 [-0.162, -0.045] | tie / worse | - | 2.43 / 2.89 (3.54) |
| hailfinder, 700 | 10 | 0.666 / 0.627 | +0.039 [-0.034, +0.109] | 0.453 / 0.544 | -0.091 [-0.138, -0.043] | tie / worse | - | 2.38 / 2.93 (3.54) |
| hailfinder, 1,000 | 10 | 0.667 / 0.620 | +0.047 [-0.026, +0.118] | 0.457 / 0.549 | -0.092 [-0.142, -0.040] | tie / worse | - | 2.34 / 2.93 (3.54) |
| hailfinder, 5,000 | 10 | 0.618 / 0.616 | +0.002 [-0.060, +0.060] | 0.495 / 0.547 | -0.051 [-0.092, -0.008] | tie / worse | - | 2.12 / 3.20 (3.54) |
| insurance, 300 | 10 | 0.603 / 0.587 | +0.017 [-0.028, +0.065] | 0.523 / 0.575 | -0.052 [-0.091, -0.016] | tie / worse | - | 3.17 / 3.48 (5.19) |
| insurance, 500 | 10 | 0.548 / 0.537 | +0.012 [-0.016, +0.040] | 0.581 / 0.620 | -0.039 [-0.065, -0.014] | tie / worse | - | 2.86 / 3.16 (5.19) |
| insurance, 700 | 10 | 0.510 / 0.501 | +0.009 [-0.011, +0.033] | 0.614 / 0.646 | -0.033 [-0.052, -0.018] | tie / worse | - | 2.66 / 3.02 (5.19) |
| insurance, 1,000 | 10 | 0.457 / 0.454 | +0.003 [-0.015, +0.019] | 0.653 / 0.685 | -0.032 [-0.048, -0.015] | tie / worse | - | 2.44 / 2.76 (5.19) |
| insurance, 5,000 | 10 | 0.334 / 0.358 | -0.024 [-0.060, +0.005] | 0.758 / 0.755 | +0.003 [-0.022, +0.033] | tie / tie | - | 1.84 / 2.29 (5.19) |
| win95pts, 300 | 10 | 0.782 / 0.733 | +0.050 [+0.035, +0.067] | 0.325 / 0.432 | -0.108 [-0.120, -0.096] | worse / worse | - | 5.00 / 5.09 (5.92) |
| win95pts, 500 | 10 | 0.752 / 0.657 | +0.095 [+0.074, +0.112] | 0.367 / 0.500 | -0.133 [-0.143, -0.119] | worse / worse | - | 4.85 / 4.80 (5.92) |
| win95pts, 700 | 10 | 0.700 / 0.610 | +0.090 [+0.068, +0.111] | 0.415 / 0.537 | -0.121 [-0.141, -0.102] | worse / worse | - | 4.65 / 4.58 (5.92) |
| win95pts, 1,000 | 10 | 0.649 / 0.570 | +0.079 [+0.050, +0.103] | 0.460 / 0.571 | -0.111 [-0.128, -0.089] | worse / worse | - | 4.34 / 4.36 (5.92) |
| win95pts, 5,000 | 10 | 0.451 / 0.479 | -0.028 [-0.047, -0.011] | 0.652 / 0.650 | +0.002 [-0.010, +0.015] | better / tie | - | 3.24 / 3.88 (5.92) |

#### `mmmb` against gs

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 10 | 0.381 / 0.505 | -0.124 [-0.149, -0.099] | 0.697 / 0.639 | +0.058 [+0.037, +0.079] | better / better | - | 1.58 / 2.46 (3.51) |
| alarm, 500 | 10 | 0.312 / 0.439 | -0.127 [-0.162, -0.093] | 0.759 / 0.693 | +0.066 [+0.042, +0.089] | better / better | - | 1.35 / 2.23 (3.51) |
| alarm, 700 | 10 | 0.267 / 0.398 | -0.131 [-0.152, -0.111] | 0.795 / 0.722 | +0.073 [+0.058, +0.091] | better / better | - | 1.18 / 2.09 (3.51) |
| alarm, 1,000 | 10 | 0.221 / 0.394 | -0.173 [-0.199, -0.150] | 0.833 / 0.724 | +0.109 [+0.092, +0.130] | better / better | - | 1.01 / 2.08 (3.51) |
| alarm, 5,000 | 10 | 0.118 / 0.344 | -0.226 [-0.256, -0.199] | 0.916 / 0.763 | +0.154 [+0.136, +0.175] | better / better | - | 0.59 / 1.82 (3.51) |
| hailfinder, 300 | 10 | 0.698 / 0.694 | +0.004 [-0.082, +0.086] | 0.425 / 0.494 | -0.070 [-0.125, -0.013] | tie / worse | - | 2.50 / 3.06 (3.54) |
| hailfinder, 500 | 10 | 0.686 / 0.651 | +0.035 [-0.052, +0.116] | 0.440 / 0.526 | -0.086 [-0.146, -0.023] | tie / worse | - | 2.43 / 2.96 (3.54) |
| hailfinder, 700 | 10 | 0.666 / 0.638 | +0.028 [-0.054, +0.105] | 0.453 / 0.535 | -0.083 [-0.135, -0.027] | tie / worse | - | 2.38 / 2.91 (3.54) |
| hailfinder, 1,000 | 10 | 0.667 / 0.620 | +0.047 [-0.035, +0.126] | 0.457 / 0.547 | -0.090 [-0.146, -0.031] | tie / worse | - | 2.34 / 2.89 (3.54) |
| hailfinder, 5,000 | 10 | 0.618 / 0.570 | +0.048 [-0.007, +0.102] | 0.495 / 0.580 | -0.084 [-0.122, -0.045] | tie / worse | - | 2.12 / 2.82 (3.54) |
| insurance, 300 | 10 | 0.603 / 0.573 | +0.030 [-0.006, +0.068] | 0.523 / 0.585 | -0.062 [-0.096, -0.030] | tie / worse | - | 3.17 / 3.38 (5.19) |
| insurance, 500 | 10 | 0.548 / 0.533 | +0.016 [-0.011, +0.041] | 0.581 / 0.622 | -0.041 [-0.061, -0.018] | tie / worse | - | 2.86 / 3.12 (5.19) |
| insurance, 700 | 10 | 0.510 / 0.495 | +0.015 [-0.001, +0.032] | 0.614 / 0.650 | -0.036 [-0.051, -0.023] | tie / worse | - | 2.66 / 2.96 (5.19) |
| insurance, 1,000 | 10 | 0.457 / 0.461 | -0.004 [-0.022, +0.015] | 0.653 / 0.680 | -0.026 [-0.043, -0.010] | tie / worse | - | 2.44 / 2.84 (5.19) |
| insurance, 5,000 | 10 | 0.334 / 0.362 | -0.029 [-0.066, +0.003] | 0.758 / 0.751 | +0.007 [-0.019, +0.039] | tie / tie | - | 1.84 / 2.27 (5.19) |
| win95pts, 300 | 10 | 0.782 / 0.749 | +0.033 [+0.015, +0.057] | 0.325 / 0.421 | -0.096 [-0.112, -0.084] | worse / worse | - | 5.00 / 5.21 (5.92) |
| win95pts, 500 | 10 | 0.752 / 0.672 | +0.081 [+0.062, +0.096] | 0.367 / 0.490 | -0.123 [-0.133, -0.112] | worse / worse | - | 4.85 / 4.88 (5.92) |
| win95pts, 700 | 10 | 0.700 / 0.632 | +0.068 [+0.049, +0.087] | 0.415 / 0.521 | -0.105 [-0.121, -0.089] | worse / worse | - | 4.65 / 4.70 (5.92) |
| win95pts, 1,000 | 10 | 0.649 / 0.590 | +0.060 [+0.040, +0.079] | 0.460 / 0.555 | -0.095 [-0.109, -0.080] | worse / worse | - | 4.34 / 4.53 (5.92) |
| win95pts, 5,000 | 10 | 0.451 / 0.496 | -0.046 [-0.067, -0.023] | 0.652 / 0.635 | +0.017 [+0.001, +0.033] | better / better | - | 3.24 / 4.09 (5.92) |

#### `mmmb` against pcmb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 10 | 0.381 / 0.380 | +0.001 [-0.008, +0.010] | 0.697 / 0.701 | -0.004 [-0.012, +0.005] | tie / tie | - | 1.58 / 1.68 (3.51) |
| alarm, 500 | 10 | 0.312 / 0.326 | -0.015 [-0.025, -0.003] | 0.759 / 0.746 | +0.013 [+0.006, +0.019] | better / better | - | 1.35 / 1.52 (3.51) |
| alarm, 700 | 10 | 0.267 / 0.253 | +0.014 [+0.005, +0.024] | 0.795 / 0.806 | -0.011 [-0.018, -0.004] | worse / worse | - | 1.18 / 1.10 (3.51) |
| alarm, 1,000 | 10 | 0.221 / 0.206 | +0.015 [+0.006, +0.023] | 0.833 / 0.844 | -0.011 [-0.017, -0.004] | worse / worse | - | 1.01 / 0.92 (3.51) |
| alarm, 5,000 | 10 | 0.118 / 0.116 | +0.002 [+0.000, +0.004] | 0.916 / 0.917 | -0.001 [-0.003, +0.000] | tie / tie | - | 0.59 / 0.59 (3.51) |
| hailfinder, 300 | 10 | 0.698 / 0.625 | +0.073 [+0.060, +0.086] | 0.425 / 0.497 | -0.072 [-0.077, -0.067] | worse / worse | - | 2.50 / 4.38 (3.54) |
| hailfinder, 500 | 10 | 0.686 / 0.590 | +0.096 [+0.082, +0.110] | 0.440 / 0.527 | -0.087 [-0.094, -0.081] | worse / worse | - | 2.43 / 4.03 (3.54) |
| hailfinder, 700 | 10 | 0.666 / 0.571 | +0.094 [+0.083, +0.106] | 0.453 / 0.543 | -0.090 [-0.097, -0.083] | worse / worse | - | 2.38 / 3.39 (3.54) |
| hailfinder, 1,000 | 10 | 0.667 / 0.555 | +0.112 [+0.103, +0.120] | 0.457 / 0.562 | -0.105 [-0.111, -0.099] | worse / worse | - | 2.34 / 2.97 (3.54) |
| hailfinder, 5,000 | 10 | 0.618 / 0.618 | -0.001 [-0.004, +0.003] | 0.495 / 0.497 | -0.002 [-0.004, +0.000] | tie / tie | - | 2.12 / 2.13 (3.54) |
| insurance, 300 | 10 | 0.603 / 0.596 | +0.007 [-0.007, +0.021] | 0.523 / 0.538 | -0.015 [-0.030, -0.001] | tie / worse | - | 3.17 / 3.28 (5.19) |
| insurance, 500 | 10 | 0.548 / 0.542 | +0.007 [-0.008, +0.021] | 0.581 / 0.587 | -0.006 [-0.016, +0.005] | tie / tie | - | 2.86 / 2.90 (5.19) |
| insurance, 700 | 10 | 0.510 / 0.501 | +0.009 [-0.003, +0.024] | 0.614 / 0.623 | -0.009 [-0.021, -0.000] | tie / worse | - | 2.66 / 2.64 (5.19) |
| insurance, 1,000 | 10 | 0.457 / 0.454 | +0.004 [-0.005, +0.014] | 0.653 / 0.657 | -0.004 [-0.014, +0.004] | tie / tie | - | 2.44 / 2.43 (5.19) |
| insurance, 5,000 | 10 | 0.334 / 0.343 | -0.009 [-0.021, +0.004] | 0.758 / 0.755 | +0.004 [-0.009, +0.013] | tie / tie | - | 1.84 / 1.87 (5.19) |
| win95pts, 300 | 10 | 0.782 / 0.782 | +0.000 [+0.000, +0.000] | 0.325 / 0.325 | +0.000 [+0.000, +0.000] | tie / tie | - | 5.00 / 5.00 (5.92) |
| win95pts, 500 | 10 | 0.752 / 0.752 | +0.000 [+0.000, +0.000] | 0.367 / 0.367 | +0.000 [+0.000, +0.000] | tie / tie | - | 4.85 / 4.85 (5.92) |
| win95pts, 700 | 10 | 0.700 / 0.700 | +0.000 [+0.000, +0.000] | 0.415 / 0.415 | -0.000 [-0.000, +0.000] | tie / tie | - | 4.65 / 4.65 (5.92) |
| win95pts, 1,000 | 10 | 0.649 / 0.650 | -0.000 [-0.003, +0.002] | 0.460 / 0.460 | +0.000 [-0.002, +0.003] | tie / tie | - | 4.34 / 4.34 (5.92) |
| win95pts, 5,000 | 10 | 0.451 / 0.449 | +0.002 [-0.001, +0.005] | 0.652 / 0.654 | -0.002 [-0.004, +0.000] | tie / tie | - | 3.24 / 3.21 (5.92) |

#### `mmmb, degrees_of_freedom mean` against Causal Explorer MMMB

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 20 | 0.412 / 0.448 | -0.037 [-0.060, -0.012] | 0.676 / 0.672 | +0.004 [-0.018, +0.024] | better / tie | - | 1.72 / 3.02 (3.52) |
| alarm, 500 | 20 | 0.322 / 0.347 | -0.025 [-0.042, -0.009] | 0.754 / 0.753 | +0.001 [-0.009, +0.012] | better / tie | - | 1.40 / 2.13 (3.51) |
| alarm, 700 | 20 | 0.263 / 0.305 | -0.042 [-0.059, -0.026] | 0.802 / 0.782 | +0.019 [+0.008, +0.031] | better / better | - | 1.19 / 1.94 (3.51) |
| alarm, 1,000 | 20 | 0.234 / 0.218 | +0.016 [-0.001, +0.032] | 0.824 / 0.851 | -0.027 [-0.040, -0.014] | tie / worse | - | 1.09 / 1.12 (3.52) |
| alarm, 5,000 | 20 | 0.121 / 0.178 | -0.056 [-0.067, -0.047] | 0.914 / 0.882 | +0.031 [+0.024, +0.039] | better / better | - | 0.62 / 0.87 (3.51) |
| hailfinder, 300 | 20 | 0.632 / 0.768 | -0.135 [-0.152, -0.117] | 0.460 / 0.363 | +0.096 [+0.078, +0.113] | better / better | - | 2.13 / 13.77 (3.42) |
| hailfinder, 500 | 20 | 0.642 / 0.652 | -0.010 [-0.021, +0.002] | 0.444 / 0.470 | -0.026 [-0.036, -0.016] | tie / worse | - | 2.01 / 9.14 (3.27) |
| hailfinder, 700 | 20 | 0.643 / 0.624 | +0.020 [+0.012, +0.029] | 0.447 / 0.497 | -0.050 [-0.057, -0.043] | worse / worse | - | 2.26 / 8.24 (3.53) |
| hailfinder, 1,000 | 20 | 0.638 / 0.578 | +0.060 [+0.048, +0.071] | 0.456 / 0.536 | -0.080 [-0.088, -0.072] | worse / worse | - | 2.24 / 7.31 (3.54) |
| hailfinder, 5,000 | 20 | 0.601 / 0.525 | +0.076 [+0.067, +0.088] | 0.493 / 0.577 | -0.084 [-0.089, -0.079] | worse / worse | - | 2.04 / 6.94 (3.54) |
| insurance, 300 | 20 | 0.639 / 0.647 | -0.009 [-0.032, +0.014] | 0.490 / 0.518 | -0.029 [-0.049, -0.008] | tie / worse | - | 3.28 / 5.83 (5.18) |
| insurance, 500 | 20 | 0.567 / 0.559 | +0.008 [-0.008, +0.025] | 0.560 / 0.595 | -0.035 [-0.050, -0.022] | tie / worse | - | 2.88 / 4.05 (5.15) |
| insurance, 700 | 20 | 0.518 / 0.534 | -0.016 [-0.036, +0.004] | 0.606 / 0.615 | -0.009 [-0.025, +0.008] | tie / tie | - | 2.67 / 3.75 (5.18) |
| insurance, 1,000 | 20 | 0.469 / 0.458 | +0.011 [-0.010, +0.030] | 0.647 / 0.675 | -0.029 [-0.044, -0.013] | tie / worse | - | 2.48 / 2.95 (5.19) |
| insurance, 5,000 | 20 | 0.343 / 0.324 | +0.020 [+0.003, +0.036] | 0.754 / 0.783 | -0.029 [-0.042, -0.016] | worse / worse | - | 1.88 / 1.99 (5.19) |
| pigs, 500 | 20 | 0.151 / 0.366 | -0.215 [-0.226, -0.205] | 0.901 / 0.757 | +0.144 [+0.137, +0.152] | better / better | - | 0.83 / 1.93 (3.64) |
| win95pts, 300 | 20 | 0.776 / 0.769 | +0.007 [-0.006, +0.021] | 0.331 / 0.416 | -0.086 [-0.100, -0.072] | tie / worse | - | 4.98 / 5.47 (5.91) |
| win95pts, 500 | 20 | 0.736 / 0.708 | +0.028 [+0.016, +0.041] | 0.380 / 0.470 | -0.091 [-0.098, -0.084] | worse / worse | - | 4.78 / 5.27 (5.91) |
| win95pts, 700 | 20 | 0.697 / 0.648 | +0.049 [+0.036, +0.061] | 0.418 / 0.519 | -0.101 [-0.109, -0.092] | worse / worse | - | 4.60 / 4.91 (5.91) |
| win95pts, 1,000 | 20 | 0.653 / 0.599 | +0.054 [+0.039, +0.068] | 0.464 / 0.557 | -0.094 [-0.105, -0.082] | worse / worse | - | 4.34 / 4.63 (5.92) |
| win95pts, 5,000 | 20 | 0.452 / 0.416 | +0.037 [+0.027, +0.046] | 0.652 / 0.705 | -0.053 [-0.060, -0.045] | worse / worse | - | 3.24 / 3.31 (5.89) |

#### `mmmb, degrees_of_freedom mean` against mmmb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 300 | 20 | 0.412 / 0.376 | +0.037 [+0.022, +0.051] | 0.675 / 0.704 | -0.029 [-0.042, -0.016] | worse / worse | - | 1.71 / 1.56 (3.51) |
| alarm, 500 | 20 | 0.322 / 0.304 | +0.018 [+0.008, +0.029] | 0.754 / 0.765 | -0.011 [-0.019, -0.003] | worse / worse | - | 1.40 / 1.32 (3.51) |
| alarm, 700 | 20 | 0.263 / 0.249 | +0.014 [+0.002, +0.026] | 0.802 / 0.811 | -0.009 [-0.018, -0.000] | worse / worse | - | 1.19 / 1.13 (3.51) |
| alarm, 1,000 | 20 | 0.233 / 0.215 | +0.018 [+0.009, +0.028] | 0.825 / 0.836 | -0.012 [-0.019, -0.005] | worse / worse | - | 1.09 / 1.00 (3.51) |
| alarm, 5,000 | 20 | 0.121 / 0.111 | +0.010 [+0.003, +0.018] | 0.914 / 0.920 | -0.006 [-0.011, -0.003] | worse / worse | - | 0.62 / 0.57 (3.51) |
| hailfinder, 300 | 20 | 0.675 / 0.703 | -0.027 [-0.038, -0.015] | 0.410 / 0.423 | -0.012 [-0.020, -0.006] | better / worse | - | 2.37 / 2.52 (3.54) |
| hailfinder, 500 | 20 | 0.653 / 0.686 | -0.034 [-0.047, -0.020] | 0.434 / 0.439 | -0.005 [-0.013, +0.001] | better / tie | - | 2.30 / 2.45 (3.54) |
| hailfinder, 700 | 20 | 0.644 / 0.675 | -0.031 [-0.039, -0.022] | 0.446 / 0.448 | -0.002 [-0.004, +0.001] | better / tie | - | 2.27 / 2.42 (3.54) |
| hailfinder, 1,000 | 20 | 0.639 / 0.667 | -0.028 [-0.036, -0.019] | 0.455 / 0.458 | -0.003 [-0.007, +0.001] | better / tie | - | 2.24 / 2.35 (3.54) |
| hailfinder, 5,000 | 20 | 0.601 / 0.619 | -0.017 [-0.023, -0.011] | 0.493 / 0.495 | -0.001 [-0.005, +0.002] | better / tie | - | 2.03 / 2.12 (3.54) |
| insurance, 300 | 20 | 0.638 / 0.597 | +0.041 [+0.028, +0.055] | 0.491 / 0.529 | -0.038 [-0.051, -0.026] | worse / worse | - | 3.29 / 3.12 (5.19) |
| insurance, 500 | 20 | 0.567 / 0.539 | +0.028 [+0.014, +0.043] | 0.561 / 0.589 | -0.029 [-0.042, -0.015] | worse / worse | - | 2.90 / 2.82 (5.19) |
| insurance, 700 | 20 | 0.517 / 0.500 | +0.017 [+0.004, +0.029] | 0.607 / 0.621 | -0.014 [-0.025, -0.003] | worse / worse | - | 2.68 / 2.63 (5.19) |
| insurance, 1,000 | 20 | 0.468 / 0.444 | +0.023 [+0.014, +0.033] | 0.648 / 0.667 | -0.019 [-0.028, -0.010] | worse / worse | - | 2.48 / 2.41 (5.19) |
| insurance, 5,000 | 20 | 0.343 / 0.329 | +0.014 [+0.005, +0.024] | 0.755 / 0.765 | -0.010 [-0.019, -0.002] | worse / worse | - | 1.87 / 1.80 (5.19) |
| pigs, 500 | 20 | 0.151 / 0.129 | +0.022 [+0.013, +0.031] | 0.901 / 0.917 | -0.016 [-0.023, -0.009] | worse / worse | - | 0.83 / 0.74 (3.63) |
| win95pts, 300 | 20 | 0.776 / 0.776 | +0.000 [-0.007, +0.008] | 0.331 / 0.329 | +0.001 [-0.004, +0.006] | tie / tie | - | 5.00 / 4.99 (5.92) |
| win95pts, 500 | 20 | 0.736 / 0.740 | -0.004 [-0.011, +0.003] | 0.380 / 0.379 | +0.001 [-0.005, +0.007] | tie / tie | - | 4.79 / 4.80 (5.92) |
| win95pts, 700 | 20 | 0.697 / 0.695 | +0.003 [-0.005, +0.010] | 0.417 / 0.422 | -0.004 [-0.009, +0.000] | tie / tie | - | 4.61 / 4.61 (5.92) |
| win95pts, 1,000 | 20 | 0.653 / 0.652 | +0.001 [-0.006, +0.008] | 0.463 / 0.463 | +0.000 [-0.006, +0.006] | tie / tie | - | 4.35 / 4.35 (5.92) |
| win95pts, 5,000 | 20 | 0.453 / 0.454 | -0.002 [-0.007, +0.004] | 0.652 / 0.650 | +0.002 [-0.003, +0.006] | tie / tie | - | 3.27 / 3.26 (5.92) |

Each table gives Andrey's value / the other's, D with its 95% interval, the verdicts on distance /
F1, and the errors per target, missing plus extra members, of Andrey / the other (the empty set's).

## Run

Set up the toolbox as [The harness](../README.md) describes for Causal Explorer, with `matlab` on
the `PATH`. Each arm is one run of the driver:

```shell
B="benchmarks/blanket_cpu.py --mode target --drop-constant --discrete-test plain --warmup 0 --cores 1"
B="$B --first-seed 0 --seeds 20 --max-depth none"
NETS=$(for x in alarm hailfinder insurance win95pts; do
  for n in 300 500 700 1000 5000; do echo -n "net:$x:$n "; done; done)
$ANDREY_BENCH_PYTHON $B --out out/mm --cells $NETS --solutions andrey.mmmb,causal-explorer.mmmb
$ANDREY_BENCH_PYTHON $B --out out/mm-pigs --cells net:pigs:500 --targets 50 \
  --solutions andrey.mmmb,causal-explorer.mmmb
$ANDREY_BENCH_PYTHON benchmarks/blanket_summary.py 'mm=out/mm*' --json cells.json
```

The driver's defaults supply the rest: `--indep-test gsq`, `--alpha 0.05`, and `--cap-wall-s` of
1800 (3,600 on pigs). A rerun is the same command with `--first-seed 20` on the cells to rerun, into
a folder the arm's pattern matches; the context arm is `'ctx=out/ctx*'` with the solutions
`andrey.iamb,andrey.inter_iamb,andrey.gs,andrey.pcmb`. Times will differ with the hardware; the sets
should match closely.
