---
name: missing-values
description: Which `missing` value to use for your data and the reason its values are missing,
  the methods that take it, and what each handling assumes.
meta:
  type: guide
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Missing values

Real tables have gaps. This guide says how to mark them, which handling fits your data and the
reason its values are missing, and what each handling assumes. Every output on this page is
computed when the site is built.

## Mark the missing values

Mark a missing value with `NaN` or pandas' `NA`, or, for `andrey run`, leave its field empty or
write `NA` or `N/A` in the data file. Infinity is not a missing value and raises. A column with no
observed value, or with one value throughout, raises before the fit.

The argument `missing` names the handling. `None`, the default, picks the method's default
handling: test-wise deletion in the methods whose tests delete rows, and a refusal in the searches
that take only `"em"`. A value a method does not list raises a `ValueError` that names the values
it takes.

## Which handling to use

The right handling depends on why the values are missing:

- **Completely at random.** Whether a value is missing depends on nothing in the data, as when a
  sensor drops readings by chance.
- **At random.** Whether a value is missing depends on variables that are always observed, as when
  older patients skip a test and every age is recorded.
- **Not at random.** Whether a value is missing depends on a variable that has missing values
  itself. When a value causes its own missingness, as when high incomes go unreported, the
  mechanism is called self-masking.

Find your data in the first column and the reason in the first row:

| Your data | Completely at random | At random | Not at random |
|---|---|---|---|
| Continuous, linear with Gaussian noise | `"testwise"`, or `"em"` in a search that takes it | `"em"` in a search that takes it, or `"mvpc"` in `pc` | `"mvpc"` in `pc` |
| Continuous, nonlinear or not Gaussian | `"testwise"` | No handling corrects it | No handling corrects it |
| Categorical or mixed | `"testwise"` | No handling corrects it | No handling corrects it |
| A time series | `"testwise"` in `pcmci`, `pcmciplus`, or `lpcmci` | No handling corrects it | No handling corrects it |

The methods that take each value:

| `missing` | What it does | Methods |
|---|---|---|
| `"testwise"` | [Test-wise deletion](#test-wise-deletion): each test reads the rows where its variables are observed | {{ missing_names }}, where it is the default |
| `"mvpc"` | [Missing-Value PC](#missing-value-pc): tests again the edges that the reason for the gaps could have kept | `pc`, with `indep_test="fisherz"` or `"student_t"` |
| `"em"` | [Structural EM](#structural-em): the search runs on the expected covariance of the data | {{ em_names }}, with `score_func="bic"` and no `env` where the search takes them |

Four limits apply to the first table:

- No handling corrects self-masking. `"mvpc"` assumes that no value causes its own missingness, and
  `"em"` has no guarantee when values are missing not at random.
- `"testwise"` runs in every method that takes it, whatever the reason, and is valid only when
  values are missing completely at random. Outside the cells that name it, a test can find
  dependence that the population does not have, and the fit can keep extra edges and get
  orientations wrong. This holds for `fci` too: the procedure that
  [Strobl, Visweswaran, and Spirtes (2018)][strobl] prove sound for values missing for a reason
  confirms each independence on the complete rows, and it is not the one `"testwise"` runs.
- Dropping every row that holds a missing value, then running the method, is your own step. For
  independent rows it keeps the population when values are missing completely at random and enough
  complete rows are left. Check how many rows are left first, and drop the same rows from `env`
  and every other argument that has one entry per row. Never drop a step of a time series.
- Every other method [refuses missing values](#methods-that-refuse-missing-values).

The data alone do not tell you the reason. What you know about how the data were collected is the
best guide. A fit with `missing="mvpc"` also reports what it found: `metadata["missingness_parents"]`
names the variables that each column's gaps depend on. It cannot detect self-masking.

## Test-wise deletion

{{ missing_methods }} accept missing values: each conditional-independence test reads only the rows
where all of its variables are observed. This keeps rows that would be dropped if every incomplete
row were deleted first. The pair tests of ANM, PNL, and LOCI read the rows where both columns are
observed, which for a pair is the same deletion. Here a tenth of the Sachs measurements go missing
completely at random:

```{code-cell} ipython3
import numpy as np

import andrey
from andrey.data import load_dataset

sachs = load_dataset("sachs")
gappy = sachs.data.mask(np.random.default_rng(0).random(sachs.data.shape) < 0.1)
out = andrey.pc(gappy)
out.metadata["rows_tested"]
```

`metadata["rows_tested"]` gives the minimum (`min`) and median number of rows read by a test.
`metadata["missing"]` names the handling the fit used; a fit on complete data has neither entry.
`missing="testwise"` names test-wise deletion, which is what the default picks in these methods:

```{code-cell} ipython3
andrey.pc(gappy, missing="testwise").structure == out.structure
```

Test-wise deletion is valid when values are missing completely at random, which means that whether
a value is missing depends on no value in the data; the rows a test reads are then a random sample
of all rows, and otherwise they are a selected sample that more rows do not correct. This is the
condition for every method that deletes rows per test. With a selected sample, the tests can find
dependence where the population has none, and a fit can keep extra edges
([Tu et al., 2019][tu]) and get orientations wrong.

A time series keeps every step: each row of the lagged design that reads a missing value holds it.
Each test of PCMCI, PCMCI+, or LPCMCI reads the design rows where all of its lagged variables are
observed.

A test needs enough rows where its variables are observed: `|S| + 3` for the partial-correlation
tests (`fisherz`, `student_t`, and `normal_scores`), with `|S|` the size of its conditioning set,
and 2 for the other tests. A test with fewer rows raises a `ValueError` that says so. The
`total_conditioning` blanket learner, the default of `mmb_by_mmb`, conditions every test on all the
other variables, so with a partial-correlation test it needs `d + 2` rows with every variable
observed and raises before the fit without them. Cap the conditioning sets with `max_depth`
(`max_p_global` in LPCMCI and the same-step phases of PCMCI+), or drop a column with many missing
values, which can remove a common cause of the columns that stay.
CD-NOD and FCI-JCI need a complete `env`.

With `fisherz` and `student_t`, each test computes its correlations directly from its rows, for many
tests at once. The other tests are built anew on each test's rows, which costs more.

We measured test-wise deletion with PC and FCI, with `fisherz` and `student_t`, on values missing
completely at random and one pattern missing at random: the
[0.11.0 release card](../benchmarks/0.11.0.md#missing-values) has the numbers. The other methods and
tests are not benchmarked with missing values yet.

## Missing-Value PC

`andrey.pc(data, missing="mvpc")` corrects PC for the extra edges with Missing-Value PC
([Tu et al., 2019][tu]). It finds the variables each missingness indicator depends on, then tests
again the edges such an indicator could have kept. These tests use virtual data rebuilt from the
rows where those variables are observed. `metadata["missingness_parents"]` names the parents it
found and `metadata["corrected_edges"]` the edges it tested again and removed. On complete data,
MVPC gives the test-wise result.

A corrected test reads the rows where its pair, its conditioning set, and those parents are all
observed. When too few rows are available, it uses the test-wise test instead. The pair then keeps
its test-wise result for that conditioning set, and `metadata["corrected_edges"]["uncorrected"]`
lists the pair.

The correction holds when no indicator causes a variable, every independence in the observed rows
also holds in the missing ones, no indicator is a deterministic function of others, and no value
causes its own missingness. The data cannot check these conditions. The permutation correction also
needs linear-Gaussian variables and the paper's two conditions: given `W`, the parents of the
indicators involved, those indicators are independent of the tested variables, and conditioning on
the indicators of `W` leaves the tested relation unchanged. When a variable used to rebuild the
data goes missing because of a common effect of the tested pair, the extra edge can stay, as in the
paper's Figure 1c. The paper's density-ratio correction for those cases is not implemented.

On simulated data in the paper's setting, MVPC made fewer errors than test-wise deletion in 10 of
16 settings where values go missing for a reason and tied in the rest. At 50 variables and 10,000
rows it made 5.1 errors against 14.3. Two limits showed on larger graphs:

- It does not help when every missingness indicator depends weakly on many variables, and its
  search for those variables can then take a long time; `max_depth` bounds it.
- With 50 or 100 variables and 20 or 30% of the values missing, it made 0.7 to 3.6 more errors than
  test-wise deletion in 12 settings. Use test-wise deletion there.

The [0.16.0 release card](../benchmarks/0.16.0.md#what-we-measured-on-missing-data) has the numbers.

## Structural EM

The searches whose fit reads the data only through their covariance, {{ em_methods }}, cannot delete
rows per test. They refuse missing values by default, with a message that names `missing="em"`.
With `missing="em"`, structural EM (Friedman, 1997; [Gao et al., 2022][missdag]) alternates two
steps. Under a linear-Gaussian model of the current graph, it replaces each row's missing values by
their expected values given the row's observed ones and adds their uncertainty. This gives the
expected covariance of the data. The search then runs on that covariance over every row, and the
model is fit again to the graph it returns. It runs 10 rounds. Before each search after the first,
5 more steps refit the model on the graph it has (Friedman's alternating scheme).
`metadata["em"]` records whether each round changed the graph (`graph_changed`) and the
observed-data log-likelihood after each round:

```{code-cell} ipython3
em = andrey.ges(gappy, missing="em")
em.metadata["em"]["log_likelihood"][-1]
```

EM suits data that fit its model, a linear model with Gaussian noise, when whether a value is
missing depends at most on variables that are always observed, and when the variables are observed
together often enough. This is measured, not proved: no result we know of covers the loop as it
runs. When whether a value is missing depends on missing values, it has no guarantee. Where a search
takes a `score_func` or an `env`, `missing="em"` takes only `"bic"` and no `env`. A row with no
observed value is dropped first.

`missing="em"` is supported in {{ em_supported_names }}, where no setting we measured was less
accurate than a reference implementation on the first datasets and again on fresh ones. In
{{ em_experimental_names }} it is experimental: no reference implementation exists to compare them
with. What to expect from a fit:

- EM costs time. A fit runs the search 10 times, and it took 2 to 72 times as long as the same
  search on complete data.
- EM does not always settle within its 10 rounds. The last round still changed the graph in 2 to 6%
  of the fits we measured, and in 20% of `grasp`'s. Such a fit logs a warning, and more rows help.
- With about 100 rows, `hc` and `tabu` are often no better than the empty graph, with EM and on
  complete data alike.

The [0.17.0 release card](../benchmarks/0.17.0.md#missing-data) has the numbers.

## Methods that refuse missing values

Every other method raises a `ValueError` that names it: it has no handling of missing values, and
each method's Notes say why. Drop the incomplete rows yourself, with `data.dropna()` for a
DataFrame, which suits values missing completely at random, or use one of the methods above.
A time series cannot lose a step without joining the steps around it, and a filled step is a guess
the fit reads as data, so for VARMA-LiNGAM, Granger, DYNOTEARS, and J-PCMCI+, use PCMCI, PCMCI+, or
LPCMCI instead, whose tests delete rows per test and keep every time step.

## Try it on data you control

`andrey.data.missingness` removes values from simulated data for each of the reasons above and
records which variables each gap depends on, so you can see where a handling helps before you trust
it on your own data. The [missing values notebook](../examples/missing_values) runs every handling
on simulated data: values missing at random and not at random, FCI, categorical data, and a time
series.

## References

- [Strobl, Visweswaran, and Spirtes (2018), Fast Causal Inference with Non-Random Missingness by
  Test-Wise Deletion][strobl]: FCI with test-wise deletion, and a procedure that confirms each
  independence on the complete rows.
- [Tu et al. (2019), Causal Discovery in the Presence of Missing Data][tu]: Missing-Value PC.
- Friedman (1997), Learning Belief Networks in the Presence of Missing Values and Hidden
  Variables, Proceedings of the 14th International Conference on Machine Learning: structural EM.
- [Gao et al. (2022), MissDAG: Causal Discovery in the Presence of Missing Data with Continuous
  Additive Noise Models][missdag]: structural EM with a continuous search.

[strobl]: https://doi.org/10.1007/s41060-017-0094-6
[tu]: https://proceedings.mlr.press/v89/tu19a.html
[missdag]: https://arxiv.org/abs/2205.13869
