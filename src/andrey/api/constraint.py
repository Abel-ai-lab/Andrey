"""Constraint-based facade adapters.

The underlying searches use :mod:`andrey.core` for the conditional-independence tests (Fisher-Z and
Student-t on continuous data, chi-square and G-square on categorical data, degenerate and
conditional Gaussian likelihood ratios on mixed data, and the nonlinear tests) and orientation.
``pc`` and ``cdnod`` produce CPDAGs; ``fci``, ``gfci``, and ``fci_jci`` produce PAGs. GFCI also
uses GES with a BIC, BDeu, or mixed-data score. ``icp`` tests the invariance of one target's
regressions across environments and returns its estimated parents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from andrey.api._adapt import (
    categorical_metadata,
    check_categories,
    check_rank_env,
    check_same_data,
    column_labels,
    column_namer,
    copies_named,
    data_kind,
    env_index,
    fit_inputs,
    index_column,
    index_namer,
    label_of,
    split_index,
    structure_output,
    target_columns,
)
from andrey.api._method import method_spec
from andrey.core import NULL
from andrey.core.ci_missing import rows_metadata, rows_tested
from andrey.core.knowledge import Knowledge

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy.typing as npt

    from andrey.core import GraphStructure, StructureOutput


@method_spec("pc")
def pc(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    collider_order: str | None = None,
    collider_rule: str = "majority",
    possibly_separable: str = "orient",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a CPDAG from observational data with the PC algorithm.

    PC [1]_ [2]_ uses conditional-independence tests to remove edges from a complete graph. These
    tests check whether variables are independent given other variables. It then orients colliders,
    where arrows meet at a variable, and applies Meek's rules [4]_. The result is a CPDAG,
    representing an equivalence class of graphs whose unresolved edges stay undirected.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` and ``"student_t"`` test partial correlation,
        for continuous data: ``"fisherz"`` applies Fisher's z-transform and the normal distribution;
        ``"student_t"`` applies the Student-t distribution, which is exact for Gaussian data at any
        sample size; it favors exact p-values when the rows barely exceed the conditioning set, and
        otherwise gives the graphs of ``"fisherz"``. ``"normal_scores"`` applies ``"fisherz"`` to
        the normal scores of each column, for data whose columns are monotone transformations of
        Gaussian variables: its p-values do not change under a strictly increasing transformation of
        a column. It favors columns that are monotone transformations of Gaussian variables and data
        with heavy-tailed noise; it gives up some adjacencies on Gaussian data with few rows.
        ``"chisq"`` (Pearson's chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test
        categorical data, coded by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and
        ``"cond_gauss"`` (conditional Gaussian) test mixed data, continuous and categorical columns,
        by a likelihood ratio. Seven more tests detect nonlinear dependence in continuous data, at a
        far higher cost per test: ``"kci"`` is the kernel test; ``"rcot"`` its random-feature
        approximation; ``"cmiknn"`` a nearest-neighbor conditional mutual information test;
        ``"gpdc"`` Gaussian-process regression with a distance correlation test; and ``"gcm"``,
        ``"wgcm"``, and ``"pcm"`` the generalized, weighted generalized, and projected covariance
        measures, which test the covariance of regression residuals. ``None`` picks from the
        declared column types: ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when
        some are, else ``"fisherz"``. A column is declared categorical by ``categorical`` or by a
        DataFrame's ``category``, ``bool``, or text type, never by its values. See **Categorical
        data.** and **Mixed data.** in Notes. On mixed data ``"cond_gauss"`` favors accuracy at
        about the time of ``"degen_gauss"``. ``metadata["indep_test"]`` names the test used,
        including the choice for ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one; with ``"gsq"`` it gives
        up time, a sum of hypergeometric terms per cell, and in the sparsest tables it rejects below
        that level. ``"nonempty"`` favors degrees of freedom read from the table alone, as Spirtes,
        Glymour and Scheines (1993) count them; it gives up the level, as it rejects true
        independences more often than the nominal rate. Only ``"chisq"`` and ``"gsq"`` read it:
        passing it with another test raises ``ValueError``.
    collider_order : {"p_value", "column"} or None, default=None
        The order in which colliders are oriented, which decides between two colliders that would
        orient one edge in opposite directions. ``"p_value"`` orients first the collider whose
        endpoints were separated with the largest p-value; it favors fewer orientation errors than
        orienting in column order, and colliders that do not depend on the column order.
        ``"column"`` orients them in column order and keeps the first of two that conflict; it
        favors matching position-ordered PC (``andrey.config.tie_break="position"``) and gives up
        orientation accuracy. ``None`` follows ``andrey.config.tie_break``: ``"p_value"`` under
        ``"rank"``, ``"column"`` under ``"position"``. See Notes.
    collider_rule : {"sepsets", "conservative", "majority"}, default="majority"
        How PC decides whether ``z`` is a collider ``x -> z <- y`` on a triple ``x - z - y`` whose
        ends are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the separating
        set that removed the edge ``x - y``. ``"conservative"`` [11]_ and ``"majority"`` [3]_ test
        ``x`` and ``y`` again given every subset of ``x``'s neighbors and every subset of ``y``'s
        neighbors, and count the subsets that separate them (``p >= alpha``). If at least one does,
        ``"conservative"`` makes ``z`` a collider when none of them contains it and a non-collider
        when all do; ``"majority"`` when fewer than half contain it and when more than half do. Any
        other triple is ambiguous, and no rule orients through it. Both run up to
        ``2**degree(x) + 2**degree(y)`` extra tests per pair. ``"sepsets"`` favors orienting more
        edges when effects are weak and rows are few compared with the variables; it gives up the
        most false arrowheads, from colliders that one test error creates. ``"conservative"`` favors
        the fewest false arrowheads in every setting measured; it gives up true arrowheads on sparse
        graphs with weak effects, which it leaves undirected. ``"majority"`` favors fewer errors
        overall than ``"sepsets"`` in most settings measured, with far fewer false arrowheads, and
        more true arrowheads than ``"conservative"`` on sparse graphs with weak effects; it gives up
        more false arrowheads than ``"conservative"`` everywhere, and trails ``"sepsets"`` with many
        variables and few rows. Against ``"conservative"`` its errors overall are lower in some
        settings and higher in others. See Notes.
    possibly_separable : {"orient", "ambiguous"}, default="orient"
        How the orientation rules read a triple ``x - z - y`` with a side listed in
        ``metadata["possibly_separable"]``, on data with missing values. ``"orient"`` reads it as
        any other triple. ``"ambiguous"`` makes it neither a collider nor a non-collider, and no
        rule orients through it, when a conditioning set not tested of that side lies within the
        other nodes adjacent to one of its ends or undecided with it: a set that separates a pair
        lies within the parents of one end, so the search might have missed that separating set.
        ``"orient"`` favors arrowheads and accuracy; it gives up protection from false colliders.
        ``"ambiguous"`` favors no false collider; it gives up the true arrowheads beside them, and
        accuracy.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    missing : {"testwise", "mvpc"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``"mvpc"`` is Missing-Value PC
        [13]_: after the test-wise skeleton, it finds the parents of each variable's missingness
        indicator and tests again, on virtual data that correct for them, the edges such an
        indicator could have kept; it takes ``"fisherz"`` or ``"student_t"``. ``"mvpc"`` assumes
        that no missingness indicator causes a variable or is a deterministic function of others,
        that no value causes its own missingness, and that the independences of the observed rows
        hold in the missing ones [13]_. ``None`` picks the method's default, ``"testwise"``. On data
        with missing values, ``metadata["missing"]`` records the handling used. ``"testwise"``
        favors every row each test can read and, on large graphs where an indicator has many weak
        parents or parents with missing values, accuracy; it gives up its guarantee outside values
        missing completely at random. ``"mvpc"`` favors values missing for a reason that other
        variables carry; it gives up every test but those two and, where an indicator has many weak
        parents or parents with missing values, accuracy on large graphs. See **Missing values.** in
        Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make, and of
        the permutations of ``missing="mvpc"``; the other tests ignore it. With those tests, or with
        ``"mvpc"`` on data with missing values, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It also caps the
        subsets the conservative and majority rules test.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. A pair the knowledge makes non-adjacent is tested for a separating set, so colliders
        over it are oriented as usual; when the search finds none within ``max_depth``, no rule
        reads its unshielded triples as colliders or non-colliders. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, giving the maximally
        oriented PDAG (MPDAG) :class:`~andrey.Knowledge` describes; knowledge that orients the edges
        the search kept into a directed cycle raises ``ValueError``. In a CPDAG, ``required_paths``
        and ``forbidden_paths`` keep the DAGs the graph stands for in which each listed path holds
        or does not, as :class:`~andrey.Knowledge` describes; a claim no such DAG meets is dropped
        with a :class:`~andrey.KnowledgeWarning` and listed in ``metadata["unmet_knowledge"]``; when
        the graph found has no consistent extension, and so stands for no DAG, a claim on a pair
        joined by an undirected edge directs that edge, Meek's rules follow,
        ``metadata["edge_claims"]`` lists the claims applied this way, and the result is not
        guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"pc"``. Directed marks (``->``) are oriented edges, undirected (``--``) the ones the
        equivalence class leaves open. Under a mixed-data setting, ``metadata["categorical"]`` lists
        the categorical columns it read, by name or 0-based position. With ``collider_rule``
        ``"conservative"`` or ``"majority"``, ``metadata["ambiguous_triples"]`` lists every
        unshielded triple no rule reads as a collider or a non-collider, as ``[x, z, y]``: 0-based
        column positions, the middle node second and ``x < y``, in ascending order.
        ``metadata["conflicts"]`` lists the edges two colliders ask to orient both ways, as
        ``[x, y]`` pairs of 0-based column positions with ``x < y``, in ascending order; Notes say
        which orientation each keeps. With ``missing="mvpc"`` on data with missing values,
        ``metadata["missingness_parents"]`` maps each column whose missingness indicator has
        parents to them, and ``metadata["corrected_edges"]`` lists the edges tested again
        (``"retested"``), those removed (``"removed"``), and those with a test that had too few
        rows to correct (``"uncorrected"``), as ``[x, y]`` pairs in ascending order, and, when a
        test was corrected, the fewest and the median rows a corrected test read
        (``"rows_tested"``, with ``"min"`` and ``"median"``) and, when it is not zero, the number
        of corrected tests that returned ``p = 1`` (``"p_one_tests"``); both name a DataFrame's
        columns and number an array's from 0, and ``metadata["seed"]`` is the seed of the
        permutations. With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` has too few samples
        for a test or a singular correlation matrix; with ``"chisq"`` or ``"gsq"``, if ``data``
        holds a value that is not a whole number; or, with ``"degen_gauss"`` or ``"cond_gauss"``, if
        ``categorical`` names a column ``data`` lacks, or if a categorical column holds a number
        that is not whole or a missing value. Also if a continuous test meets a column declared
        categorical, by ``categorical`` or by its type. With ``"fisherz"`` or ``"student_t"``, also
        if a column repeats another up to scale and shift, before any test runs; one message names
        every group of copies and says to keep one column per group. Also if ``knowledge`` names a
        column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists. Also if
        ``missing="mvpc"`` is given with a test other than ``"fisherz"`` or ``"student_t"``.

    Warns
    -----
    PerformanceWarning
        When a conditioning pass of size two or greater requires at least ten million CI tests;
        each message warns once per process. The warning does not limit the search.
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    Notes
    -----
    **When to use.** PC suits i.i.d. observational data with no unmeasured common causes, in which
    every independence follows from the graph (faithfulness) [1]_. Its default Fisher-Z test assumes
    continuous data, where each variable depends linearly on its causes with roughly Gaussian noise;
    for categorical data, use ``"chisq"`` or ``"gsq"``, for mixed data ``"degen_gauss"`` or
    ``"cond_gauss"``, and for nonlinear effects in continuous data ``"kci"``, ``"rcot"``,
    ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, or ``"pcm"``, at a far higher cost per test. It
    is designed for sparse graphs, where every variable has few neighbors, and then stays feasible
    with many variables [5]_; it returns a CPDAG. When unmeasured common causes are plausible, use
    :func:`~andrey.fci`; when the noise is non-Gaussian and every edge should be oriented, use
    :func:`~andrey.direct_lingam`. When whether a value is missing depends on the variables, through
    variables other than itself, ``missing="mvpc"`` removes the extra edges that test-wise deletion
    can keep [13]_.

    **Choosing alpha.** A test removes an edge when it cannot reject independence at ``alpha``. A
    larger ``alpha`` therefore keeps more edges, true and false. On the Sachs data in the
    getting-started guide, raising ``alpha`` from 0.05 to 0.2 found more known links, but a smaller
    share of the links found were known. No value is right in general: it trades missed edges
    against false ones, so a result is best reported with how the graph changes with ``alpha``.

    **How it works.** Edge removal is order-independent (PC-stable) [3]_. At each conditioning-set
    size, every test uses the adjacencies from the start of that size. Edges found independent are
    removed together. A removed pair's separating set combines all conditioning sets that separated
    it. Two colliders conflict when they would orient one edge in opposite directions, so the order
    in which they are oriented can change the result [3]_. A collider is skipped when it would give
    an edge an arrowhead at both ends, so of two conflicting colliders the one oriented first is
    kept. With ``collider_order="p_value"``, which the default ``None`` picks under the default
    ``andrey.config.tie_break="rank"``, each collider ``x -> y <- z`` ranks by the largest p-value
    among the conditioning sets that separated ``x`` and ``z``, and colliders are oriented from the
    highest rank to the lowest. When two conflicting colliders rank equal to nine decimals, the edge
    they dispute stays undirected, and Meek's rules do not orient it. The colliders therefore do not
    depend on the column order. This adapts the conflict rule of Ramsey [10]_, which ranks a
    collider by separating sets that additional tests choose; Andrey ranks it by the tests that
    removed the edge, so orienting runs no new test. With ``collider_order="column"``, colliders are
    oriented by middle node, then by endpoint pair, in column order, so the result can depend on the
    column order. After the colliders, Meek's rules run in rounds: each round collects every
    orientation they imply on the graph as it stands and applies them together, except that an edge
    two of them would direct both ways stays undirected, and so do orientations that would together
    close a directed cycle. The rounds repeat until one orients nothing, so Meek's rules do not
    depend on the column order. Such conflicts arise only when no DAG on the skeleton has exactly
    the colliders found, as test errors can cause; otherwise the rounds give the CPDAG that firing
    the rules one at a time gives. Under ``andrey.config(tie_break="position")``, the rules fire one
    at a time in column order instead, and of two firings that would orient one edge in opposite
    directions the first decides. The :doc:`column-order guide </docs/guides/column-order>` lists
    what reordering the columns does to each method.

    **Collider rules.** A test error can make the separating set that removed an edge disagree with
    other sets that also separate the pair, and ``collider_rule="sepsets"`` then decides a collider
    from that one set. ``"conservative"``, conservative PC [11]_, and ``"majority"``, majority-rule
    PC [3]_, decide each triple ``x - z - y`` on the final skeleton instead. They test ``x`` and
    ``y`` given each subset of ``x``'s neighbors, then each subset of ``y``'s, the empty set
    included, and keep the subsets that separate them; a subset both offer counts twice. Under
    ``"conservative"``, ``z`` is a collider when no kept subset contains it, a non-collider when
    every one does, and ambiguous otherwise. Under ``"majority"``, ``z`` is a collider when fewer
    than half contain it, a non-collider when more than half do, and ambiguous at exactly half. A
    triple whose ends no subset separates is ambiguous under both, as both rules call a triple
    unambiguous only when some subset separates its ends [3]_ (Section 4.2): the search separated
    them only given sets holding a node whose edge to that end it removed later. An ambiguous
    triple is never oriented as a collider, and Meek's rules do not read it as a non-collider, so
    edges around it can stay undirected. The colliders are then oriented in ``collider_order`` as
    under ``"sepsets"``, so with ``"p_value"`` the result still does not depend on the column order.
    Conservative PC's orientations are correct under adjacency-faithfulness, a weaker assumption
    than PC's [11]_, at the cost of more undirected edges; the majority rule orients more of them
    [3]_.

    **Categorical data.** ``"chisq"`` and ``"gsq"`` take each distinct value of a column as one
    category, so the codes need not start at 0 or be consecutive; a DataFrame's text, ``bool``, and
    ``category`` columns are read the same way. A test compares the counts of ``x`` and ``y`` with
    independence within each configuration of the conditioning set that occurs in the data [2]_.
    The p-value refers the statistic to a chi-square law whose degrees of freedom are its
    mean when ``y`` is shuffled within each configuration: ``(r_x - 1) * (r_y - 1)`` per
    configuration for large counts, with ``r_x`` and ``r_y`` the values present in it [6]_, and
    more for few rows, where the usual degrees of freedom over-reject a true independence; with
    none, the p-value is 1. A table with fewer rows than cells has most cells empty: the p-values
    are then a little too large, ``"gsq"``'s more than ``"chisq"``'s, which loses ``"gsq"`` some
    power there. Over 35,808 true independences of random discrete networks, with 3 to 5
    categories, conditioning sets of 0 to 3 variables, and 500 to 5,000 rows, a test at level 0.05
    rejected 4.9% (``"chisq"``) and 3.8% (``"gsq"``); the usual degrees of freedom rejected 11.1%
    and 21.7%. ``"gsq"``, the default, takes 2 to 5 times as long as ``"chisq"`` for its exact
    mean; on discrete networks and generated tables of 500 to 5,000 rows the graph it finds was
    no farther from the truth than ``"chisq"``'s in all 16 settings and nearer in 3.

    **Mixed data.** ``"degen_gauss"`` and ``"cond_gauss"`` test data with continuous and categorical
    columns. Each distinct value of a categorical column is one category, and a DataFrame column of
    text or categories needs no coding. ``"degen_gauss"``, the degenerate Gaussian test [7]_,
    replaces a categorical column of ``k`` categories by ``k - 1`` indicator columns and treats
    every column as Gaussian, so it suits effects that are linear in the continuous variables and
    add up across categories. ``"cond_gauss"``, the conditional Gaussian test [8]_, fits the
    continuous variables in each joint category of the categorical ones with a Gaussian of that
    category's own mean and covariance, so a category can change a variance as well as a mean. Each
    category's covariance is shrunk toward that of all rows by one pseudo-row [9]_, so a joint
    category with few rows takes mostly the covariance of all rows; to tell its own apart, a test
    needs several rows in each joint category of its categorical variables. Each compares the
    likelihoods of the models with and without the dependence, a chi-square statistic on the
    difference in their numbers of parameters. With no categorical column, both are the
    likelihood-ratio form of the partial-correlation test, which agrees with ``"fisherz"`` as the
    rows grow.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [12]_ [13]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A pair no tested query could
    decide, such as one never observed together, is undecided: it is left out of the graph and
    listed in ``metadata["undecided_pairs"]``, and a triple whose ends are undecided is neither a
    collider nor a non-collider to any rule. A drawn edge an untested conditioning set might have
    separated is listed in ``metadata["possibly_separable"]``: a flag of what the data could not
    check, not a finding that the edge is false; under ``possibly_separable="ambiguous"``, a triple
    with a side it reads (see ``possibly_separable``) is neither a collider nor a non-collider to
    any rule. On merged cohorts, pairs measured only in different cohorts are undecided, and an edge
    whose separating set is never observed together with its pair is drawn and flagged. A few rows
    that observe every column decide such pairs on those rows alone, with little power, so they can
    remove true edges between variables observed together. A continuous column constant on a test's
    rows raises ``ValueError``. With missing values, the skeleton search runs in one process.
    Test-wise deletion is valid when values are missing completely at random, which means that
    whether a value is missing depends on no value in the data; the rows a test reads are then a
    random sample of all rows, and otherwise they are a selected sample that more rows do not
    correct.

    **Missing-Value PC.** ``missing="mvpc"`` follows Algorithm 1 of Tu et al. [13]_. The test-wise
    skeleton keeps every true edge but can keep extra ones: an extra edge ``x - y`` needs the
    missingness indicator of a variable in the test to be a common effect of ``x`` and ``y``, or a
    descendant of one (their Proposition 2). Step 2 finds the parents of each indicator: the
    variable's column is replaced by its 0/1 indicator, and the skeleton search tests the indicator
    against each other variable given subsets of the indicator's neighbors, with ``indep_test``,
    ``alpha``, and ``max_depth``. A test whose rows hold only one indicator value, carrying no
    information on it, or fewer rows than the test needs is not tested; a variable on whose rows the
    indicator holds one value conditions no test, and one left adjacent with no test of its pair
    tested is no parent. Step 3 marks as candidates the adjacent pairs that share an adjacent
    variable or an indicator they both cause. Step 4 runs the skeleton search again over the
    candidates, every other edge kept, with the permutation-based correction (PermC, their Algorithm
    2): a test of ``x`` and ``y`` given ``S`` reads ``W``, the parents of the indicators of ``x``,
    ``y``, and ``S``, and of the indicators of ``W``'s own variables. On the rows where all of them
    are observed, each of ``x``, ``y``, and ``S`` is regressed on ``W``; ``W``'s values on as many
    rows, drawn in random order from the rows where ``W`` is observed, plus each variable's
    residuals give virtual data distributed as the complete values of ``x``, ``y``, and ``S`` are on
    the rows where ``W`` is observed (their Eq. (4)), and the test runs on them. A test with an
    empty ``W`` is the test-wise one. So is a test with fewer than ``max(t, |W| + 2)`` rows where
    ``x``, ``y``, ``S``, and ``W`` are all observed, ``t`` being the rows the test needs,
    ``|S| + 4`` for ``"fisherz"`` and ``|S| + 3`` for ``"student_t"``, too few for the regressions
    on ``W`` or for the test: the paper leaves that case open, and as the correction can only remove
    an edge the test-wise tests kept, the pair keeps its test-wise answer for that ``S`` and is
    listed in ``metadata["corrected_edges"]["uncorrected"]``. A corrected test reads no more rows
    than the test-wise test of the same ``x``, ``y``, and ``S``, and fewer when a row misses a value
    of ``W``: ``metadata["corrected_edges"]["rows_tested"]`` gives the fewest and the median rows
    the corrected tests read, where ``metadata["rows_tested"]`` counts the tests that delete rows,
    and is absent when no test was corrected. The colliders follow the corrected separating sets;
    ``"conservative"`` and ``"majority"`` vote with the same tests as the skeleton search, the
    corrected ones for a pair the correction retests, and their Algorithm 1 orients the corrected
    skeleton with PC's rules. Each regression has an intercept, where their Eq. (2) has none and so
    assumes that every variable has mean 0: with the intercept, the regressions fit data whose means
    are not 0, and a p-value does not change when a column is shifted. The permutations draw from
    ``seed`` and from where the queried columns' values are missing, so a result depends on neither
    the column order, nor a column's units or origin, nor the number of workers. On complete data,
    ``"mvpc"`` gives the result of ``"testwise"``.

    **Limitations.** On linear-Gaussian data it made fewer errors than the empty graph on sparse
    graphs of up to hundreds of variables. On larger sparse graphs its errors passed the empty
    graph's, on dense graphs they came close to it, and its run time grows combinatorially with the
    largest neighborhood. Test errors can produce colliders that fit no DAG: the ``cpdag`` then has
    no consistent DAG extension, and a graph query such as ``markov_blanket`` raises ``ValueError``.
    The `0.10.0 release card <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#the-boundary>`__
    has the numbers.

    References
    ----------
    .. [1] Spirtes, P., and Glymour, C. (1991). An algorithm for fast recovery of sparse causal
       graphs. Social Science Computer Review 9(1), 62-72.
    .. [2] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [3] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [4] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [5] Kalisch, M., and Buhlmann, P. (2007). Estimating high-dimensional directed acyclic graphs
       with the PC-algorithm. JMLR 8, 613-636.
    .. [6] Tsamardinos, I., Brown, L. E., and Aliferis, C. F. (2006). The max-min hill-climbing
       Bayesian network structure learning algorithm. Machine Learning 65(1), 31-78.
    .. [7] Andrews, B., Ramsey, J., and Cooper, G. F. (2019). Learning high-dimensional directed
       acyclic graphs with mixed data-types. Proceedings of Machine Learning Research 104, 4-21.
    .. [8] Andrews, B., Ramsey, J., and Cooper, G. F. (2018). Scoring Bayesian networks of mixed
       variables. International Journal of Data Science and Analytics 6(1), 3-18.
    .. [9] Friedman, J. H. (1989). Regularized discriminant analysis. Journal of the American
       Statistical Association 84(405), 165-175.
    .. [10] Ramsey, J. (2016). Improving accuracy and scalability of the PC algorithm by maximizing
       P-value. arXiv:1610.00378, Section 5.
    .. [11] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [12] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [13] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019).
       Causal discovery in the presence of missing data. Proceedings of Machine Learning Research
       89, 1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import pc
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> z = x + y + 0.5 * rng.standard_normal(500)
    >>> pc(pd.DataFrame({"x": x, "y": y, "z": z})).structure.oriented_edges()
    [('x', 'z', 'directed'), ('y', 'z', 'directed')]

    On categorical data, with a chi-square test:

    >>> a = rng.integers(0, 2, 2000)
    >>> b = rng.integers(0, 2, 2000)
    >>> c = np.where(rng.random(2000) < 0.8, a + b, rng.integers(0, 3, 2000))
    >>> data = pd.DataFrame({"a": a, "b": b, "c": c})
    >>> pc(data, indep_test="chisq").structure.oriented_edges()
    [('a', 'c', 'directed'), ('b', 'c', 'directed')]

    On mixed data, with a category column and the degenerate Gaussian test:

    >>> g = rng.integers(0, 3, 2000)
    >>> u = rng.standard_normal(2000)
    >>> v = np.array([-1.0, 0.0, 1.0])[g] + u + 0.5 * rng.standard_normal(2000)
    >>> data = pd.DataFrame({"g": pd.Categorical(g), "u": u, "v": v})
    >>> pc(data, indep_test="degen_gauss").structure.oriented_edges()
    [('g', 'v', 'directed'), ('u', 'v', 'directed')]
    """
    from andrey.constraint.mvpc import PERMC_TESTS
    from andrey.constraint.pc import pc as _pc
    from andrey.core.ties import resolve_collider_order, resolve_in_column_order

    assert indep_test is not None  # method_spec picks it from the column types
    assert seed is not None  # method_spec resolves it
    if missing == "mvpc" and indep_test not in PERMC_TESTS:
        raise ValueError(
            'pc: missing="mvpc" corrects the partial-correlation tests, so indep_test must be '
            f'"fisherz" or "student_t", got {indep_test!r}'
        )
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="pc", missing=True
    )
    ambiguous: set[tuple[int, int, int]] = set()
    conflicts: set[tuple[int, int]] = set()
    found: dict = {}
    with copies_named(labels), rows_tested() as rows:
        cpdag = _pc(
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            collider_order=resolve_collider_order(collider_order),
            collider_rule=collider_rule,
            categorical=mask,
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            missing=missing or "testwise",
            _ambiguous=ambiguous,
            _conflicts=conflicts,
            _mvpc=found,
        )
    metadata = {
        "conflicts": [list(edge) for edge in sorted(conflicts)],
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(rows, label_of(labels), _drawn(cpdag)),
        **_mvpc_metadata(found, labels, seed),
    }
    return structure_output(cpdag, labels=labels, metadata=metadata)


def _drawn(graph: GraphStructure, index_edges: Sequence[int] = ()) -> np.ndarray:
    """Return the boolean adjacency of ``graph``.

    With ``index_edges``, the adjacency has one more node, the index the search appends, joined to
    each of those columns.
    """
    marks = graph.to_numpy() != NULL
    if not index_edges:
        return marks
    d = len(marks)
    padded = np.zeros((d + 1, d + 1), dtype=bool)
    padded[:d, :d] = marks
    padded[list(index_edges), d] = padded[d, list(index_edges)] = True
    return padded


def _mvpc_metadata(found: dict, labels: Sequence[str] | None, seed: int) -> dict:
    """Return MVPC's indicator parents, corrected edges, and ``seed`` as metadata, or ``{}``.

    ``found`` is empty when the correction did not run, on data without missing values, and the fit
    then drew no random numbers. A column is its label when ``labels`` is given, else its 0-based
    position. When corrected tests ran, the corrected edges hold ``rows_tested``, their minimum and
    median row counts, and nonzero ``p_one_tests``, the count of tests returning ``p = 1``
    (:func:`~andrey.core.ci_missing.rows_metadata`).
    """
    if not found:
        return {}

    def name(j: int) -> object:
        return j if labels is None else labels[j]

    return {
        "missingness_parents": {
            name(c): [name(p) for p in pa] for c, pa in sorted(found["parents"].items())
        },
        "corrected_edges": {
            **{
                key: [[name(x), name(y)] for x, y in found[key]]
                for key in ("retested", "removed", "uncorrected")
            },
            **rows_metadata(found["rows"]),
        },
        "seed": seed,
    }


def _ambiguous_metadata(
    collider_rule: str, ambiguous: set[tuple[int, int, int]]
) -> dict[str, list[list[int]]]:
    """Return the ambiguous triples as metadata, or ``{}`` under ``collider_rule="sepsets"``.

    The triples become ``{"ambiguous_triples": [[a, b, c], ...]}``, in ascending order.
    """
    if collider_rule == "sepsets":
        return {}
    return {"ambiguous_triples": [list(triple) for triple in sorted(ambiguous)]}


@method_spec("fci")
def fci(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    collider_rule: str = "majority",
    possibly_separable: str = "orient",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    possible_dsep: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG from observational data with FCI, sound under latent confounders.

    FCI extends PC to handle unmeasured common causes (latent confounders) and selection bias
    [1]_ [2]_. It uses conditional-independence tests to remove edges, then orients their endpoints.
    The result is a PAG, whose circles (``o``) mark endpoints the data cannot determine. Use it when
    unmeasured confounding is plausible.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` and ``"student_t"`` test partial correlation,
        for continuous data: ``"fisherz"`` applies Fisher's z-transform and the normal distribution;
        ``"student_t"`` applies the Student-t distribution, which is exact for Gaussian data at any
        sample size; it favors exact p-values when the rows barely exceed the conditioning set, and
        otherwise gives the graphs of ``"fisherz"``. ``"normal_scores"`` applies ``"fisherz"`` to
        the normal scores of each column, for data whose columns are monotone transformations of
        Gaussian variables: its p-values do not change under a strictly increasing transformation of
        a column. It favors columns that are monotone transformations of Gaussian variables and data
        with heavy-tailed noise; it gives up some adjacencies on Gaussian data with few rows.
        ``"chisq"`` (Pearson's chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test
        categorical data, coded by whole numbers. ``"degen_gauss"`` (degenerate Gaussian) and
        ``"cond_gauss"`` (conditional Gaussian) test mixed data, continuous and categorical columns.
        Seven more tests detect nonlinear dependence in continuous data, at a far higher cost per
        test: ``"kci"`` is the kernel test; ``"rcot"`` its random-feature approximation;
        ``"cmiknn"`` a nearest-neighbor conditional mutual information test; ``"gpdc"``
        Gaussian-process regression with a distance correlation test; and ``"gcm"``, ``"wgcm"``, and
        ``"pcm"`` the generalized, weighted generalized, and projected covariance measures, which
        test the covariance of regression residuals. ``None`` picks from the declared column types:
        ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are, else
        ``"fisherz"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values. See Notes.
        ``metadata["indep_test"]`` names the test used, including the choice for ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one; with ``"gsq"`` it gives
        up time, a sum of hypergeometric terms per cell, and in the sparsest tables it rejects below
        that level. ``"nonempty"`` favors degrees of freedom read from the table alone, as Spirtes,
        Glymour and Scheines (1993) count them; it gives up the level, as it rejects true
        independences more often than the nominal rate. Only ``"chisq"`` and ``"gsq"`` read it:
        passing it with another test raises ``ValueError``.
    collider_rule : {"sepsets", "conservative", "majority"}, default="majority"
        How FCI decides whether ``z`` is a collider on a triple ``x - z - y`` where ``x`` and ``y``
        are not adjacent. ``"sepsets"`` makes it a collider when ``z`` is not in the separating set
        that removed the edge ``x - y``. ``"conservative"`` [4]_ [5]_ and ``"majority"`` [4]_ test
        ``x`` and ``y`` again given every subset of ``x``'s neighbors and every subset of ``y``'s
        neighbors, and count the subsets that separate them (``p >= alpha``). If at least one does,
        ``"conservative"`` makes ``z`` a collider when none of them contains it and a non-collider
        when all do; ``"majority"`` when fewer than half contain it and when more than half do. Any
        other triple is ambiguous, and no rule orients through it. Both run up to
        ``2**degree(x) + 2**degree(y)`` extra tests per pair. ``"sepsets"`` favors orienting more
        edges when effects are weak and rows are few compared with the variables; it gives up the
        most false arrowheads, from colliders that one test error creates. ``"conservative"`` favors
        the fewest false arrowheads in every setting measured; it gives up true arrowheads on sparse
        graphs with weak effects, which it leaves undirected. ``"majority"`` favors fewer errors
        overall than ``"sepsets"`` in most settings measured, with far fewer false arrowheads, and
        more true arrowheads than ``"conservative"`` on sparse graphs with weak effects; it gives up
        more false arrowheads than ``"conservative"`` everywhere, and trails ``"sepsets"`` with many
        variables and few rows. Against ``"conservative"`` its errors overall are lower in some
        settings and higher in others. See Notes.
    possibly_separable : {"orient", "ambiguous"}, default="orient"
        How the orientation rules read a triple ``x - z - y`` with a side listed in
        ``metadata["possibly_separable"]``, on data with missing values. ``"orient"`` reads it as
        any other triple. ``"ambiguous"`` makes every such triple neither a collider nor a
        non-collider, and no rule orients through it: with latent variables, a set that separates a
        pair lies within the possible-d-separating set of one end, beyond its adjacencies, so any
        untested conditioning set might be a missed separating set. The collider pass before the
        possible-d-separation step reads it too, so that step's sets, the skeleton, and
        ``metadata["not_tested"]`` can change. ``"orient"`` favors arrowheads and accuracy; it gives
        up protection from false colliders. ``"ambiguous"`` favors no false collider; it gives up
        the true arrowheads beside them, and accuracy.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. ``"testwise"`` favors every row each test can read; it gives up its guarantee
        outside values missing completely at random. See **Missing values.** in Notes.
    possible_dsep : {"order_free", "fast"} or None, default=None
        The order of FCI's possible-d-separation step [2]_ [3]_, which removes edges the first step
        kept, one at a time, so each removal changes what the step tests after it. ``"order_free"``
        visits the pairs and nodes by a rank computed from the data, so the result does not depend
        on the column order unless two columns are identical. ``"fast"`` visits them in column
        order. ``None`` follows ``andrey.config.tie_break``: ``"order_free"`` under ``"rank"``,
        ``"fast"`` under ``"position"``. ``"order_free"`` favors a result that does not depend on
        the column order; it gives up time on some large graphs, where its order needs more tests.
        ``"fast"`` favors time on some graphs; it gives up that independence. See Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It caps the tests of
        the adjacency search, the possible-d-separation step, the conservative and majority
        rules, and the discriminating-path rule.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. A pair the knowledge makes non-adjacent is tested for a separating set, so colliders
        over it are oriented as usual; when the search finds none within ``max_depth``, no rule
        reads its unshielded triples as colliders or non-colliders. In the PAG, a forbidden
        ``a -> b`` on an adjacent pair puts an arrowhead at ``a`` and can leave a circle at ``b``,
        and a required one is ``a -> b``; knowledge that puts an arrowhead at ``a`` on an edge
        ``a - b`` while its required edges lead from ``a`` to ``b`` raises ``ValueError``. In the
        PAG, ``required_paths`` makes an adjacent pair ``a -> b``, ``forbidden_paths`` puts an
        arrowhead at the cause, and no test removes a ``confounded`` pair, as
        :class:`~andrey.Knowledge` describes; these marks, and the check that drops a claim the PAG
        does not meet with a :class:`~andrey.KnowledgeWarning` and lists it in
        ``metadata["unmet_knowledge"]``, assume no selection bias.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"fci"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``collider_rule`` ``"conservative"`` or ``"majority"``,
        ``metadata["ambiguous_triples"]`` lists every unshielded triple no rule reads as a collider
        or a non-collider, as ``[x, z, y]``: 0-based column positions, the middle node second and
        ``x < y``, in ascending order. With a randomized ``indep_test``, ``metadata["seed"]`` is the
        seed used. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` has too few samples
        for a test or a singular correlation matrix, or if ``data`` or ``categorical`` does not suit
        the test, as for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column
        repeats another up to scale and shift, before any test runs; one message names every group
        of copies and says to keep one column per group. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** FCI suits i.i.d. observational data that may be affected by unmeasured common
    causes or sample selection, in which every independence follows from the graph (faithfulness)
    [1]_ [2]_. Its default Fisher-Z test assumes continuous data with linear effects and roughly
    Gaussian noise; for categorical data, use ``"chisq"`` or ``"gsq"``, and for mixed data
    ``"degen_gauss"`` or ``"cond_gauss"``, which read the data as :func:`~andrey.pc` describes. Its
    tests suit sparse graphs, where every variable has few neighbors, as PC's do, and it runs an
    extra round of tests that PC does not. It returns a PAG, which claims less than PC's CPDAG: an
    arrowhead at a node rules out that node causing the node at the edge's other end, and a circle
    leaves the mark open. When every common cause is measured and no selection bias is possible, use
    :func:`~andrey.pc`, which returns a CPDAG without that extra round of tests.

    **Choosing alpha.** A larger ``alpha`` keeps more edges, true and false.

    **How it works.** FCI runs in three steps. First, it removes edges by conditional-independence
    tests, as PC does, and orients colliders from the separating sets. Next, it tests each remaining
    pair given subsets of its possible-d-separation set: nodes that can separate the pair when some
    causes are unmeasured. It removes pairs found independent and orients colliders again. Last, it
    applies Zhang's orientation rules R1-R10 [3]_ until no mark changes. An endpoint no rule
    resolves stays a circle.

    The rules run in rounds: each round collects the marks every rule implies on the graph as it
    stands and sets them together, except that an endpoint two of them would mark one as an
    arrowhead and one as a tail stays a circle. The rounds repeat until no mark changes. Rule R4
    decides an endpoint by every shortest discriminating path, so paths whose tests disagree leave
    it a circle. The orientation rules therefore do not depend on the column order, and with
    ``possible_dsep="order_free"`` neither does the result, unless two columns are identical. With
    ``"fast"`` the result can depend on the column order through the possible-d-separation step. A
    fixed column order gives reproducible results. Under ``andrey.config(tie_break="position")``,
    the rules instead fire one at a time in column order, and of two firings that conflict the
    first decides. ``possible_dsep`` then defaults to ``"fast"``. The
    :doc:`column-order guide </docs/guides/column-order>` lists what reordering the columns does to
    each method.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only if the whole cycle,
    including the closing edge, is uncovered, that is, no two nodes two steps apart are adjacent.
    It considers every circle path, up to a fixed search bound, so a covered route through a node
    never hides an uncovered route through the same node. Rule R8 checks each pair of endpoints in
    both orders, so whether it fires does not depend on the node labels.

    **Possible-d-separation set.** FCI builds each possible-d-separation set [2]_ [3]_ with a search
    that walks out from one endpoint of the pair and keeps a node when a node it was reached from
    has a possibly directed path, one with no arrowhead pointing back along it, to the endpoint or
    to the node. FCI checks a node when the search reaches it, against the nodes it was reached
    from so far, so the set depends on the order the search visits nodes. The set can be smaller
    than the published definition of the set [2]_, so FCI can keep an edge that the published set
    would remove, even with exact tests. FCI keeps this search because the published set grows
    quickly with the graph, the search's set is the one FCI has been measured with, and on
    simulated data the published set did not give more accurate PAGs. FCI tests every subset of the
    set, except the empty set and the subsets of either endpoint's neighbors, which the first step
    already tested. It visits the pairs one at a time, removes a pair as soon as a subset separates
    it, and records that subset, so each removal changes the sets of the pairs after it. With
    ``possible_dsep="order_free"``, the pairs, the nodes of each search, and the subsets follow a
    tie rank computed from the data, as for :func:`~andrey.hc`, which moves with its
    column when the columns are reordered; with ``"fast"``, they follow the column order.

    **Collider rules.** ``collider_rule="conservative"``, conservative FCI [4]_ after the idea of
    conservative PC [5]_, and ``"majority"``, majority-rule FCI [4]_, decide each triple
    ``x - z - y`` after the possible-d-separation step, on the final skeleton. They test ``x`` and
    ``y`` given each subset of ``x``'s neighbors, then each subset of ``y``'s, the empty set
    included, and keep the subsets that separate them; a subset both offer counts twice. Under
    ``"conservative"``, ``z`` is a collider when no kept subset contains it, a non-collider when
    every one does, and ambiguous otherwise. Under ``"majority"``, ``z`` is a collider when fewer
    than half contain it, a non-collider when more than half do, and ambiguous at exactly half. A
    triple whose ends no subset separates is ambiguous under both, as both rules call a triple
    unambiguous only when some subset separates its ends [4]_ (Sections 4.2 and 4.4): the
    separating set that removed the edge, from a possible-d-separation set or an earlier
    neighborhood, does not vote. An ambiguous triple is never oriented as a collider, and rules
    R1-R10 do not read it as a non-collider, so endpoints around it can stay circles. A vote reads
    only the skeleton and the tests, so the rules keep the result independent of the column order
    wherever ``"sepsets"`` does.

    **Separating sets.** When several conditioning sets of the same size make a pair independent
    in the first step, FCI records their union as the pair's separating set. FCI keeps the union
    because it does not depend on the order of the tests, and on simulated data it gave more
    accurate PAGs than the first set found.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [6]_ [7]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A pair no tested query could
    decide, such as one never observed together, is undecided: it is left out of the graph and
    listed in ``metadata["undecided_pairs"]``, and a triple whose ends are undecided is neither a
    collider nor a non-collider to any rule. A drawn edge an untested conditioning set might have
    separated is listed in ``metadata["possibly_separable"]``: a flag of what the data could not
    check, not a finding that the edge is false; under ``possibly_separable="ambiguous"``, a triple
    with a flagged side is neither a collider nor a non-collider to any rule. On merged cohorts,
    pairs measured only in different cohorts are undecided, and an edge whose separating set is
    never observed together with its pair is drawn and flagged. A few rows that observe every column
    decide such pairs on those rows alone, with little power, so they can remove true edges between
    variables observed together. A continuous column constant on a test's rows raises
    ``ValueError``. With missing values, the skeleton search runs in one process. Test-wise deletion
    is valid when values are missing completely at random, which means that whether a value is
    missing depends on no value in the data; the rows a test reads are then a random sample of all
    rows, and otherwise they are a selected sample that more rows do not correct.

    **Limitations.** With hidden variables it made fewer errors than the empty graph on graphs of up
    to hundreds of variables, but on the largest sparse graphs its errors passed the empty graph's,
    and :func:`~andrey.gfci` made fewer at nearly every size. Test errors can give results that no
    MAG fits: the ``pag`` is then not the PAG of any MAG, and a graph query such as ``m_separated``
    can raise ``ValueError``. The
    `0.8.0 release card <https://andrey.abel.ai/docs/benchmarks/0.8.0.html#fci-s-feasible-sizes>`__
    has the numbers.

    References
    ----------
    .. [1] Spirtes, P., Meek, C., and Richardson, T. (1995). Causal inference in the presence of
       latent variables and selection bias. UAI, 499-506.
    .. [2] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [3] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [4] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [5] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [6] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [7] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import fci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> fci(np.column_stack([x, y])).structure.kind
    'pag'
    """
    from andrey.constraint.fci import fci as _fci
    from andrey.core.ties import resolve_in_column_order, resolve_possible_dsep

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="fci", missing=True
    )
    ambiguous: set[tuple[int, int, int]] = set()
    with copies_named(labels), rows_tested() as rows:
        pag = _fci(
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            collider_rule=collider_rule,
            categorical=mask,
            possible_dsep=resolve_possible_dsep(possible_dsep),
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
        )
    metadata = {
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(rows, label_of(labels), _drawn(pag)),
    }
    return structure_output(pag, labels=labels, metadata=metadata)


def fci_hybrid(
    name: str,
    data: npt.ArrayLike,
    env: npt.ArrayLike | None,
    *,
    score_func: str | None,
    indep_test: str | None,
    categorical: Sequence[str | int] | None,
    collider_rule: str,
    knowledge: Knowledge | None,
    **settings: object,
) -> StructureOutput:
    """Run ``name``, ``"gfci"``, ``"boss_fci"``, or ``"grasp_fci"``, a search and FCI's rules.

    The score and the test must read the data alike (:func:`~andrey.api._adapt.check_same_data`);
    ``settings`` reach the engine of the same name in :mod:`andrey.search.gfci`, ``possible_dsep``
    among them for ``"gfci"`` alone. With ``env``, the environment index joins the data as a last
    column, the engine's ``context``, and leaves the PAG again: its edges go in ``metadata``.
    """
    from andrey.core.ties import resolve_in_column_order
    from andrey.search import gfci as engines

    assert score_func is not None and indep_test is not None  # method_spec picks them
    # Both phases read the same data, so the score and the test must agree on its type.
    check_same_data(name, score_func, indep_test)
    check_rank_env(name, score_func, env, indep_test)
    data, index = env_index(data, env, method=name)
    X, mask, labels, constraints = fit_inputs(
        data,
        "indep_test",
        indep_test,
        categorical,
        knowledge,
        method=name,
        # One environment adds no index column (index_column), so it is not a categorical one.
        index_categorical=index is not None and np.ptp(index) > 0,
    )
    augmented, augmented_mask, augmented_constraints, column = index_column(
        X, mask, constraints, index, "indep_test", indep_test
    )
    ambiguous: set[tuple[int, int, int]] = set()
    namer = index_namer(labels, column) or column_namer(labels)
    with copies_named(labels, namer):
        pag = getattr(engines, name)(
            augmented,
            score_func=score_func,
            indep_test=indep_test,
            collider_rule=collider_rule,
            categorical=augmented_mask,
            constraints=augmented_constraints,
            in_column_order=resolve_in_column_order(),
            context=column,
            _ambiguous=ambiguous,
            **settings,
        )
    pag, index_edges = split_index(pag, column)
    metadata = {
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
    }
    if env is not None:
        variable = label_of(labels)
        metadata["index_edges"] = [[variable(j), kind] for j, kind in index_edges]
    return structure_output(pag, labels=labels, metadata=metadata)


@method_spec("gfci")
def gfci(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    score_func: str | None = None,
    penalty_discount: float = 1.0,
    ebic_gamma: float = 0.0,
    equivalent_sample_size: float = 1.0,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    alpha: float = 0.05,
    collider_rule: str = "sepsets",
    categorical: Sequence[str | int] | None = None,
    possible_dsep: str | None = None,
    edge_removal: str = "all",
    confounded_index: bool = True,
    seed: int | None = None,
    max_depth: int | None = None,
    max_parents: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG from observational data with GFCI, a score-then-FCI hybrid.

    GFCI [1]_ starts from the adjacencies of a score-based GES search, removes those that
    conditional-independence tests reject, and orients the endpoints with FCI's rules to account
    for possible latent confounding. Its authors report a more accurate skeleton than FCI's
    test-only one on simulated data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names become
        the structure's ``labels``. A column that ``env`` names is not a variable: it leaves the
        data before the fit, and ``categorical``, ``knowledge``, and the result's positions count
        the other columns. A list of DataFrames or 2-D arrays gives each environment's rows apart,
        labeled by position ``0, 1, ...``, and a mapping labels each by its key: the rows are
        stacked in that order, and ``env`` stays ``None``. A position is the environment's number,
        so with ``"bic"`` and ``"fisherz"`` and three or more environments, the order of the list
        can change the result (see Notes).
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. The environment index then joins
        the search as a variable that no data variable causes; see Notes. Numbers are coerced to
        ``float64`` and read as numbers, whole numbers under a categorical or mixed-data setting;
        string labels are coded ``0, 1, ...`` in sorted order, whatever the row order. When every
        row has the same value, the data come from one environment, and the graph equals the one
        without ``env``. ``None``, the default, fits ``data`` as one environment.
    score_func : str or None, default=None
        Local score driving the GES skeleton phase, as for :func:`~andrey.ges`: ``"bic"``, the
        linear-Gaussian BIC; ``"bdeu"`` or ``"discrete_bic"`` for categorical data, coded by whole
        numbers; or ``"degen_gauss"`` or ``"cond_gauss"``, the degenerate or conditional Gaussian
        BIC for mixed data. ``"normal_scores"`` is ``"bic"`` on each column's normal scores, for
        continuous data that are Gaussian only after an unknown increasing transformation of each
        column, as :func:`~andrey.ges` describes, with its limits. With
        ``indep_test="normal_scores"`` both phases read one nonparanormal model, and a strictly
        increasing transformation of any column leaves the graph unchanged; the consistency results
        for the rank-based test and score cover PC, GES, and ARGES, so with this search it is the
        analogous construction, not a proven result. ``None`` picks from the declared column types,
        as for :func:`~andrey.ges`. ``metadata["score_func"]`` names the score the fit used: the one
        picked from the declared column types when this is ``None``.
    penalty_discount : float, default=1.0
        Multiplier on the BIC penalty of ``log(n_samples)`` per parameter in the GES phase. Larger
        values give sparser graphs; ``1.0`` with ``ebic_gamma=0.0`` is the standard BIC. Must be
        finite and non-negative. BDeu ignores it.
    ebic_gamma : float, default=0.0
        Weight on the extended-BIC term of the linear-Gaussian BIC: each variable's score adds
        ``2 * ebic_gamma * log C(n_variables - 1, n_parents)``, which grows with the number of
        parent sets a variable can choose from. Larger values give sparser graphs; ``0.0`` is the
        standard BIC. Must be from 0 to 1. Only ``"bic"`` and ``"normal_scores"`` use it: passing it
        with another score raises ``ValueError``. See Notes.
    equivalent_sample_size : float, default=1.0
        The equivalent sample size of the BDeu score: how many rows the uniform prior over each
        variable's conditional distributions is worth. A larger value weighs the prior more
        against the data, and the graph can change with it. Must be finite and positive. Only
        ``"bdeu"`` uses it: passing it with another score raises ``ValueError``.
    indep_test : str or None, default=None
        Conditional-independence test for the FCI phase. ``"fisherz"`` (Fisher's z) and
        ``"student_t"`` (Student-t) test partial correlation, for continuous data, and
        ``"normal_scores"`` does so on each column's normal scores and takes no ``env``, as its
        ranks over every environment's rows would mix their distributions. ``"chisq"`` (Pearson's
        chi-square) and ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by
        whole numbers, and ``"degen_gauss"`` and ``"cond_gauss"`` mixed data. ``"kci"``, ``"rcot"``,
        ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` also detect nonlinear
        dependence in continuous data, at a far higher cost per test. Each as for
        :func:`~andrey.pc`. ``None`` picks from the declared column types: ``"gsq"`` when every
        column is categorical, ``"cond_gauss"`` when some are, else ``"fisherz"``. A column is
        declared categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text
        type, never by its values. ``metadata["indep_test"]`` names the test used, including the
        choice for ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one; with ``"gsq"`` it gives
        up time, a sum of hypergeometric terms per cell, and in the sparsest tables it rejects below
        that level. ``"nonempty"`` favors degrees of freedom read from the table alone, as Spirtes,
        Glymour and Scheines (1993) count them; it gives up the level, as it rejects true
        independences more often than the nominal rate. Only ``"chisq"`` and ``"gsq"`` read it:
        passing it with another test raises ``ValueError``.
    alpha : float, default=0.05
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        sparser graphs.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How GFCI decides whether ``z`` is a collider on a triple ``x - z - y`` where the tests
        removed the GES edge between ``x`` and ``y``; on the other triples whose ends are not
        adjacent, the GES CPDAG decides (see Notes). ``"sepsets"`` makes it a collider when ``z``
        is not in the recorded separating set of ``x`` and ``y``. ``"conservative"`` [3]_ [5]_ and
        ``"majority"`` [3]_ test ``x`` and ``y`` given every subset of ``x``'s neighbors and every
        subset of ``y``'s neighbors and count the subsets that separate them (``p >= alpha``), as
        for :func:`~andrey.fci`. If at least one does, ``"conservative"`` makes ``z`` a collider
        when none of them contains it and a non-collider when all do; ``"majority"`` when fewer
        than half contain it and when more than half do. Any other triple is ambiguous, and no
        rule orients through it. Both run ``2**degree(x) + 2**degree(y)`` tests per pair.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for the mixed-data score and test: column labels
        (strings) or 0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and
        text columns, never a column by its values. ``[]`` marks none. A categorical setting reads
        every column as categorical, and a continuous setting raises ``ValueError`` on a declared
        column, named here or by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    possible_dsep : {"order_free", "fast"} or None, default=None
        The order of GFCI's possible-d-separation step [2]_ [4]_, which removes edges the adjacency
        search kept, one at a time, so each removal changes what the step tests after it.
        ``"order_free"`` visits the pairs and nodes by a rank computed from the data, so the result
        does not depend on the column order unless two columns are identical. ``"fast"`` visits them
        in column order. ``None`` follows ``andrey.config.tie_break``: ``"order_free"`` under
        ``"rank"``, ``"fast"`` under ``"position"``. ``"order_free"`` favors a result that does not
        depend on the column order; it gives up time on some large graphs, where its order needs
        more tests. ``"fast"`` favors time on some graphs; it gives up that independence. See Notes.
    edge_removal : {"all", "triangles"}, default="all"
        Which edges of the GES CPDAG the tests may remove: ``"all"`` tests every edge, and
        ``"triangles"`` only the edges in a triangle of the CPDAG, whose ends share a neighbor in
        it, as Section 5 of [1]_ does, in the adjacency search and the possible-d-separation step.
        An edge in no triangle then stays in the PAG and still conditions the tests of the other
        pairs. The triangles come from the CPDAG, so the option adds no dependence on the column
        order. ``"all"`` favors removing every edge the tests reject; it gives up the true edges in
        no triangle that a test removes in error. ``"triangles"`` favors those true edges, most with
        a strict penalty and alpha on dense graphs; it gives up removing the false edges in no
        triangle that the GES phase adds, which costs accuracy at the defaults on sparse graphs. See
        Notes.
    confounded_index : bool, default=True
        With ``env``, whether a hidden variable may cause both the index and a data variable, as for
        :func:`~andrey.fci_jci`: FCI-JCI1. ``False`` assumes the index is set from outside, as in a
        randomized experiment, so every edge the index keeps points out of it. Without ``env``, it
        has no effect.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It caps the tests of
        the adjacency search, the separating sets for the pairs GES leaves non-adjacent, the
        possible-d-separation step, the conservative and majority rules, and the discriminating-path
        rule, but not the GES phase.
    max_parents : int or None, default=None
        A cap on every node's parents in the GES phase, at least ``0``: no DAG the GES CPDAG
        represents gives a node more than ``max_parents`` parents, as :func:`~andrey.ges` applies
        it. The test phase starts from that CPDAG's adjacencies and removes edges, so a node of the
        PAG keeps at most its CPDAG neighbors; the cap does not bound the arrowheads the orientation
        rules put at a node. ``None`` sets no cap; the GES phase then keeps its gate of
        ``n_variables / 2``. With ``env``, the index counts as a parent in the GES phase of each
        variable it points into.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The GES phase honors it as
        :func:`~andrey.ges` does, and the test phase starts from the adjacencies GES found and never
        removes a required pair. In the PAG, a forbidden ``a -> b`` on an adjacent pair puts an
        arrowhead at ``a`` and can leave a circle at ``b``, and a required one is ``a -> b``. With
        ``env``, the environment index is unconstrained. In the PAG, ``required_paths`` makes an
        adjacent pair ``a -> b``, ``forbidden_paths`` puts an arrowhead at the cause, and no test
        removes a ``confounded`` pair, as :class:`~andrey.Knowledge` describes; these marks, and the
        check that drops a claim the PAG does not meet with a :class:`~andrey.KnowledgeWarning` and
        lists it in ``metadata["unmet_knowledge"]``, assume no selection bias.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"gfci"``. Endpoints carry arrowheads, tails, or circle marks (``o``); a bidirected edge
        (``<->``) marks a probable latent common cause. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``collider_rule`` ``"conservative"`` or ``"majority"``,
        ``metadata["ambiguous_triples"]`` lists the triples the rule left ambiguous, as
        ``[x, z, y]``: 0-based column positions, the middle node second and ``x < y``, in ascending
        order; with ``env``, the index, in a triple through it, is numbered ``n_variables``. With
        ``env``, the PAG is over the data variables, and ``metadata["index_edges"]`` lists the
        index's edges as ``[variable, kind]`` pairs, by name or 0-based position, in column order:
        ``"directed"`` for ``index -> variable``, ``"partially_directed"`` for
        ``index o-> variable``, and ``"bidirected"`` for ``index <-> variable``; it is empty for a
        single environment. With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``, and ``metadata`` also records the penalties its score used,
        under their parameter names.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, or ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``score_func`` and
        ``indep_test`` read different data (continuous, categorical, or mixed), or if ``data`` or
        ``categorical`` does not suit them, as for :func:`~andrey.pc`. With the BIC score, also if a
        column repeats another up to scale and shift given more than two rows or, with more rows
        than columns, is a linear function of other columns. One message names every group of copies
        and says to keep one column per group. Also if a given ``max_parents`` is below the parents
        some graph of the required edges of ``knowledge`` gives a node, or if ``knowledge`` names a
        column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists. Also if
        ``env`` does not hold one label per row, misses a label or is infinite, or mixes strings and
        numbers, or, with a categorical or mixed-data setting, holds a number that is not whole.

    Warns
    -----
    DataTypeWarning
        When the mixed-data score or test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** GFCI suits i.i.d. observational data that may be affected by unmeasured common
    causes, in which every independence follows from the graph (faithfulness) [1]_. Its default BIC
    score and Fisher-Z test assume continuous data with linear effects and Gaussian noise; with
    categorical or mixed columns, the defaults follow the declared column types, as for
    :func:`~andrey.ges` and :func:`~andrey.pc`. Its tests start from the adjacencies of a full GES
    search rather than a complete graph, which shrinks the candidate edges, but its
    possible-d-separation round of tests is exponential in the worst case and can make it much
    slower than GES, while its authors report more accurate adjacencies and endpoints than FCI's on
    simulated data [1]_. With ``env``, it takes data from several environments, as
    :func:`~andrey.fci_jci` does. It returns a PAG, whose circle marks (``o``) are the endpoints the
    data cannot determine and whose bidirected edges (``<->``) mark probable latent common causes.
    When every common cause is measured, use :func:`~andrey.ges`, whose CPDAG the first phase of
    GFCI already computes.

    GFCI first runs :func:`~andrey.ges` and keeps its CPDAG. FCI's adjacency search then starts
    from the CPDAG's adjacencies instead of a complete graph: it removes an edge ``x - y`` at the
    smallest size at which a subset of ``x``'s or ``y``'s neighbors makes the two independent, and
    records the union of those subsets as their separating set. On a triple ``x - z - y`` whose
    ends are not adjacent, ``z`` is a collider when the CPDAG has ``x -> z <- y``; when the CPDAG
    had the edge ``x - y``, the separating set decides, as ``collider_rule`` describes. FCI's
    possible-d-separation step can remove more edges, after which the colliders are decided again.
    No step adds an edge, so the PAG's adjacencies are a subset of the GES CPDAG's. Zhang's rules
    R1-R10 [2]_ then orient the endpoints, in rounds as in :func:`~andrey.fci`; an endpoint no rule
    resolves stays a circle. A rule that needs the separating set of a pair GES left non-adjacent
    uses the union of the smallest subsets of the two endpoints' CPDAG neighbors that make the pair
    independent, which does not depend on the column order. Under
    ``andrey.config(tie_break="position")``, the rules fire one at a time in column order, such a
    pair's separating set is the first subset found in column order, and ``possible_dsep`` defaults
    to ``"fast"``.

    With ``max_parents``, the GES phase takes no move that gives a node more parents than the cap,
    as :func:`~andrey.ges` does; the published GFCI [1]_ has no such cap. The test phase only
    removes edges, but its orientation rules can put more arrowheads at a node than the cap.

    Rule R5 turns a cycle of circle-circle edges into undirected edges only when the whole cycle,
    closing edge included, is uncovered, that is, no two nodes two steps apart on it are adjacent.
    It considers every circle path, up to a fixed search bound, and rule R8 checks each pair of
    endpoints in both orders. GFCI shares FCI's possible-d-separation step [2]_ [4]_, in the order
    ``possible_dsep`` selects, and its separating sets; the Notes of :func:`~andrey.fci` describe
    both and the reasons for keeping them.

    **Testing only the edges in triangles.** With ``edge_removal="triangles"``, the adjacency
    search and the possible-d-separation step test only the CPDAG's edges that lie in a triangle
    of it, a pair whose ends share a neighbor in the CPDAG. Every other edge stays in the PAG, and
    its ends still condition the tests of the other pairs, as a required edge's do. Section 5 of
    [1]_ tests only these edges to save time, and its authors conjecture that this is always
    correct. On finite samples the two choices differ: ``"triangles"`` keeps both the true edges in
    no triangle that a test would remove and the false edges in no triangle that the GES phase
    adds. It helps with a strict penalty and alpha, most on dense graphs, where the GES phase adds
    few false edges and the tests remove more true ones, and hurts at the defaults on sparse graphs,
    where the GES phase adds more false edges. On simulated linear-Gaussian data with hidden common
    causes, 20 measured variables with 0 to 8 hidden ones, mean degree 2 to 6, and 200 to 5,000
    rows, and 100 measured variables with 10 hidden ones, mean degree 6, and 1,000 rows, it lowered
    the mean endpoint errors (SHD against the true PAG) with ``penalty_discount=2.0`` and
    ``alpha=0.01`` at every mean degree and size, from 649 to 604 at 100 variables (with
    ``max_depth=7``). At the defaults, it raised them with mean degree 2, from 21.4 to 24.7 at 20
    variables, where it kept 3.3 false edges per graph against 1.9, and with mean degree 4, from
    64.8 to 65.9. It did not shorten the fit in these runs: the median fit times differ by at most
    0.04 s. The default ``"all"`` tests every edge.

    **Choosing the penalty.** The GES phase scores a variable with the linear-Gaussian BIC, which
    charges ``penalty_discount * log(n_samples)`` per parent, plus the extended-BIC term ``2 *
    ebic_gamma * log C(n_variables - 1, n_parents)`` [6]_, where ``C(n_variables - 1, n_parents)``
    counts the parent sets of that size. The term grows with the number of variables, so a larger
    ``ebic_gamma`` gives a sparser GES graph, with fewer false adjacencies and more missed ones; the
    tests then start from fewer adjacencies and never add one. ``ebic_gamma=0.0`` is the standard
    BIC, GFCI's default; :func:`~andrey.ges` and most other score-based searches default to ``1.0``.
    On simulated linear-Gaussian data with 20 to 400 variables and 200 to 4,000 rows,
    ``ebic_gamma=1.0`` lowered GFCI's mean errors (SHD against the true graph) per true edge by
    0.064 (95% interval 0.055 to 0.074), against 0.78 for GES. In 11 of 65 settings of graph,
    variables, and rows it made more errors on average than ``0.0``, and in three of them the
    difference's 95% interval lies above zero: 300.6 against 291.0 with 200 variables, mean degree
    4, and 2,000 rows (400 true edges); 149.2 against 140.6 with 100 variables, mean degree 4,
    ``scale="unitless"`` data, and 1,000 rows (200 true edges); and 38.4 against 36.2 with 20
    variables, mean degree 4, ``scale="unitless"`` data, and 200 rows (40 true edges). Across the
    295 datasets with both fits, it made more errors on 56, fewer on 198, and the same on 41. Its
    mean errors stayed below the empty graph's in each of the 65 settings, where the standard BIC's
    exceeded it in 16. Pass ``ebic_gamma=1.0`` to use the extended BIC.

    **Several environments.** With ``env``, GFCI runs as :func:`~andrey.fci_jci` runs FCI [7]_: the
    environment index joins the data as a last variable that no move of the GES phase gives a
    parent, as for :func:`~andrey.ges` with ``env``, and before each collider pass every edge the
    index keeps gets an arrowhead at its data variable, because no data variable causes the index,
    and, with ``confounded_index=False``, a tail at the index. Joint Causal Inference assumes no
    selection bias, so the rules leave out R5-R7. The index removes the edges that pooling adds and
    can orient more, and it is dropped from the PAG; a variable keeps an edge to it when its
    distribution changes across the environments and no other variable explains the change.
    ``"bic"`` and ``"fisherz"`` read the index as a number, and the categorical and mixed-data
    settings each label as a category, as for :func:`~andrey.fci_jci`. The GES phase then runs in
    this process.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: its GES phase scores every
    variable over the same rows, and score terms over different rows do not add up to the score of
    one dataset. Drop the incomplete rows yourself, which suits values missing completely at random,
    or use :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** With hidden variables it made fewer errors than the empty graph wherever its
    fits finished, and fewer than :func:`~andrey.fci` at nearly every size, but some fits on large
    graphs did not finish in time. It runs a full :func:`~andrey.ges` search before any test, so it
    takes at least as long as GES, and with fewer rows than columns its BIC score is unreliable. The
    `0.8.0 release card <https://andrey.abel.ai/docs/benchmarks/0.8.0.html#fci-s-feasible-sizes>`__
    has the numbers.

    References
    ----------
    .. [1] Ogarrio, J. M., Spirtes, P., and Ramsey, J. (2016). A hybrid causal search algorithm
       for latent variable models. PGM, PMLR 52, 368-379.
    .. [2] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [3] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [4] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [5] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [6] Chen, J., and Chen, Z. (2008). Extended Bayesian information criteria for model
       selection with large model spaces. Biometrika 95(3), 759-771.
    .. [7] Mooij, J. M., Magliacane, S., and Claassen, T. (2020). Joint causal inference from
       multiple contexts. JMLR 21(99), 1-108.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import gfci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = x + 0.1 * rng.standard_normal(500)
    >>> gfci(np.column_stack([x, y])).structure.kind
    'pag'
    """
    from andrey.core.ties import resolve_possible_dsep

    return fci_hybrid(
        "gfci",
        data,
        env,
        score_func=score_func,
        indep_test=indep_test,
        degrees_of_freedom=degrees_of_freedom,
        categorical=categorical,
        collider_rule=collider_rule,
        possible_dsep=resolve_possible_dsep(possible_dsep),
        edge_removal=edge_removal,
        knowledge=knowledge,
        penalty_discount=penalty_discount,
        ebic_gamma=ebic_gamma,
        equivalent_sample_size=equivalent_sample_size,
        alpha=alpha,
        max_depth=max_depth,
        max_parents=max_parents,
        seed=seed,
        confounded_index=confounded_index,
    )


@method_spec("cdnod")
def cdnod(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    collider_order: str | None = None,
    collider_rule: str = "sepsets",
    possibly_separable: str = "orient",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    dependence_threshold: float | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a partially directed graph over the data variables from multi-domain data with CD-NOD.

    CD-NOD [1]_ adds the environment of each row, its domain, as an extra causal parent, so
    distribution shifts across environments are modeled, then runs PC-style discovery. It applies
    to samples from several domains, regimes, or time blocks. The environment index is appended as
    a trailing pseudo-node and removed from the result.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``. A column that ``env`` names is not a variable: it leaves
        the data before the fit, and ``categorical``, ``knowledge``, and the result's positions
        count the other columns. A list of DataFrames or 2-D arrays gives each domain's rows
        apart, labeled by position ``0, 1, ...``, and a mapping labels each by its key: the rows
        are stacked in that order, and ``env`` stays ``None``. A position is the domain's number,
        so with ``"fisherz"`` or ``"student_t"`` and three or more domains, the order of the list
        can change the result (see Notes).
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment (domain) of each row of ``data``: one label per row, such as ``0`` and
        ``1`` for two domains or the time step for data that drift over time, or the name of a
        column of the DataFrame ``data`` that holds them. Numbers are coerced to ``float64`` and
        read as numbers; whole numbers with ``"chisq"``, ``"gsq"``, ``"degen_gauss"``, or
        ``"cond_gauss"``. String labels are coded ``0, 1, ...`` in sorted order, whatever the row
        order; a label is a string or a finite number, not both. See Notes for more than two
        domains. When every row has the same value, the data come from one domain, and the result
        equals :func:`~andrey.pc`'s. ``None`` needs ``data`` as each domain's rows apart.
    alpha : float, default=0.05
        Significance level of the conditional-independence test, in ``(0, 1)``. Smaller values give
        sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` (Fisher's z) and ``"student_t"`` (Student-t)
        test partial correlation, for continuous data. ``"chisq"`` (Pearson's chi-square) and
        ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by whole numbers,
        and ``"degen_gauss"`` and ``"cond_gauss"`` mixed data. ``"kci"``, ``"rcot"``, ``"cmiknn"``,
        ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` also detect nonlinear dependence in
        continuous data, at a far higher cost per test. Each as for :func:`~andrey.pc`; see Notes
        for the index. ``"normal_scores"`` is not offered: the tests read every environment's rows,
        and ranked over them, a column's normal scores mix the environments' distributions. ``None``
        picks from the declared column types: ``"gsq"`` when every column is categorical,
        ``"cond_gauss"`` when some are, else ``"fisherz"``. A column is declared categorical by
        ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type, never by its
        values. ``metadata["indep_test"]`` names the test used, including the choice for ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one; with ``"gsq"`` it gives
        up time, a sum of hypergeometric terms per cell, and in the sparsest tables it rejects below
        that level. ``"nonempty"`` favors degrees of freedom read from the table alone, as Spirtes,
        Glymour and Scheines (1993) count them; it gives up the level, as it rejects true
        independences more often than the nominal rate. Only ``"chisq"`` and ``"gsq"`` read it:
        passing it with another test raises ``ValueError``.
    collider_order : {"p_value", "column"} or None, default=None
        The order in which colliders are oriented, as for :func:`~andrey.pc`: ``"p_value"`` by the
        largest p-value that separated each collider's endpoints, so the colliders do not depend on
        the column order, and ``"column"`` in column order. ``None`` follows
        ``andrey.config.tie_break``, which also sets whether Meek's rules run in order-free rounds
        or fire one at a time in column order, as for :func:`~andrey.pc`. Under either order, the
        edges out of the index are directed first, and no collider puts an arrowhead at the index.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How an unshielded triple ``x - z - y`` is decided, as for :func:`~andrey.pc`: by the
        separating set that removed the edge ``x - y`` (``"sepsets"``), or by a vote over the
        subsets of ``x``'s and ``y``'s neighbors that separate them, with the index among the
        variables, which, if at least one subset separates them, makes ``z`` a collider when none
        of them contains it and a non-collider when all do (``"conservative"`` [3]_), or when
        fewer than half and when more than half do (``"majority"`` [4]_). Any other triple is
        ambiguous, and no rule orients through it. A triple whose middle node is the index is not
        voted: the index is never a collider.
    possibly_separable : {"orient", "ambiguous"}, default="orient"
        How the orientation rules read a triple ``x - z - y`` with a side listed in
        ``metadata["possibly_separable"]``, on data with missing values. ``"orient"`` reads it as
        any other triple. ``"ambiguous"`` makes it neither a collider nor a non-collider, and no
        rule orients through it, when a conditioning set not tested of that side lies within the
        other nodes adjacent to one of its ends or undecided with it: a set that separates a pair
        lies within the parents of one end, so the search might have missed that separating set.
        ``"orient"`` favors arrowheads; it gives up protection from false colliders. ``"ambiguous"``
        favors no false collider; it gives up the true arrowheads beside them.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. The domain index is categorical too, so
        ``data`` may have no categorical column. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. ``"testwise"`` favors every row each test can read; it gives up its guarantee
        outside values missing completely at random. See **Missing values.** in Notes.
    dependence_threshold : float or None, default=None
        Runs the independent-change step at this threshold, in ``(0, 1)``, on its module
        dependence measure: two modules, the distributions of a candidate cause and of its effect
        given it, whose measure is at most the threshold count as changing independently. ``None``
        skips the step. See Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It also caps the
        subsets the conservative and majority rules test.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. A pair the knowledge makes non-adjacent is tested for a separating set, so colliders
        over it are oriented as usual; when the search finds none within ``max_depth``, no rule
        reads its unshielded triples as colliders or non-colliders. The result directs every edge
        the knowledge decides, then Meek's rules direct the edges that follow, as
        :class:`~andrey.Knowledge` describes; knowledge that orients the edges the search kept into
        a directed cycle raises ``ValueError``. The environment index is unconstrained. In a CPDAG,
        ``required_paths`` and ``forbidden_paths`` keep the DAGs the graph stands for in which each
        listed path holds or does not, as :class:`~andrey.Knowledge` describes; a claim no such DAG
        meets is dropped with a :class:`~andrey.KnowledgeWarning` and listed in
        ``metadata["unmet_knowledge"]``; when the graph found has no consistent extension, and so
        stands for no DAG, a claim on a pair joined by an undirected edge directs that edge, Meek's
        rules follow, ``metadata["edge_claims"]`` lists the claims applied this way, and the result
        is not guaranteed to stand for a DAG.

    Returns
    -------
    StructureOutput
        Wraps a :class:`~andrey.core.GraphStructure` over the ``d`` data variables;
        ``metadata["algorithm"]`` is ``"cdnod"``. Directed marks (``->``) are oriented edges,
        undirected (``--``) the ones left open. The graph is partially directed, not the CPDAG of an
        observational equivalence class: the index orients edges that the true DAG's CPDAG leaves
        undirected, and the graph stands for the DAGs that agree on every direction CD-NOD can
        identify [1]_. Andrey has no kind for such a graph, so it is tagged ``cpdag``, and
        :mod:`andrey.metrics` scores a ``cpdag`` against the truth's CPDAG; score the result against
        the graph CD-NOD can identify instead. ``metadata["index_edges"]`` lists the index's edges,
        each ``env -> variable``, as ``[variable, "directed"]`` pairs, and
        ``metadata["changing_variables"]`` the same variables, by name or 0-based position, in
        column order: the variables whose causal module, the distribution given their causes,
        changes across the environments, under CD-NOD's assumption that the index is only a cause
        and no hidden common cause varies apart from it [1]_. Their descendants change in
        distribution too, through them, and are not listed. Both are empty for a single environment.
        With ``"fisherz"`` or ``"student_t"``, only a change in a variable's mean that follows the
        index's values is found. Under a mixed-data setting, ``metadata["categorical"]`` lists the
        categorical columns it read, by name or 0-based position. With ``collider_rule``
        ``"conservative"`` or ``"majority"``, ``metadata["ambiguous_triples"]`` lists every
        unshielded triple no rule reads as a collider or a non-collider, as ``[x, z, y]``: 0-based
        column positions, the middle node second and ``x < y``, in ascending order; the index, in a
        triple through it, is numbered ``n_variables``. ``metadata["conflicts"]`` lists the edges
        between variables that two colliders ask to orient both ways, as ``[x, y]`` pairs of 0-based
        column positions with ``x < y``, in ascending order, decided as in :func:`~andrey.pc`. With
        a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`, or ``env`` is ``None`` with one table.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``env`` is given with each
        domain's rows apart; if those rows' tables differ in kind or columns, or one has no rows or
        holds ``inf``; if ``env`` does not hold one value per row of ``data``, holds ``NaN`` /
        ``inf`` or a missing label, mixes strings and numbers, or names a column ``data`` lacks; if
        ``data`` has too few rows for a test, if ``data`` or ``categorical`` does not suit the test,
        as for :func:`~andrey.pc`, or, with a categorical or mixed-data test, if a numeric ``env``
        holds a value that is not a whole number. With ``"fisherz"`` or ``"student_t"``, also if a
        column of ``data``, or ``env`` with more than one domain, repeats another up to scale and
        shift, before any test runs; one message names every group of copies and says to keep one
        column per group. Also if ``knowledge`` names a column ``data`` lacks or contradicts itself,
        as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** CD-NOD suits data from several domains or regimes, or data whose
    distribution drifts over time, in which the causal mechanisms may change with ``env``, any
    unmeasured common cause is a function of that index, and every
    independence follows from the graph with the index added (faithfulness) [1]_. Its default
    Fisher-Z test assumes continuous data with linear effects and Gaussian noise and detects only a
    change in a variable's mean that follows the index's values; ``"chisq"``, ``"gsq"``,
    ``"degen_gauss"``, and ``"cond_gauss"`` take each index value as a category. Its search is
    PC's with the index as one more variable, so it suits the sparse graphs PC suits. It returns a
    partially directed graph over the data variables, tagged ``cpdag``, in which the index can
    orient edges that the CPDAG of one domain leaves undirected. With data from one domain, use
    :func:`~andrey.pc`; when each row records which variables were intervened on, use
    :func:`~andrey.gies`.

    The index is added as a variable that can only be a cause. When a variable's distribution
    changes with the index, the edge from the index into it is directed, and Meek's rules can then
    orient edges that PC leaves undirected on the same data. CD-NOD needs only each row's
    environment, not which variables changed in it; when every environment's intervened variables
    are known, :func:`~andrey.gies` uses them.

    ``"fisherz"`` and ``"student_t"`` test the index as a number, by linear correlation, so they
    find a change in a variable's mean that follows the index's values. With two domains the coding
    does not matter. With three domains coded ``0, 1, 2``, a shift in domain ``1`` alone can go
    undetected, and recoding the domains can change the result. A change in a variable's noise
    variance alone, or in the size of an effect whose cause has mean zero, leaves the mean
    unchanged, so the index keeps no edge to that variable. ``"kci"``, the kernel
    conditional-independence test the paper uses [1]_ [2]_, detects such changes, at a far higher
    cost per test. ``"chisq"``, ``"gsq"``, ``"degen_gauss"``, and ``"cond_gauss"`` take each index
    value as a category, so they find a shift in any domain, whatever the coding. With
    ``"degen_gauss"`` or ``"cond_gauss"``, the index is categorical, so every data column may be
    continuous.

    With ``dependence_threshold`` set, the independent-change step [1]_ orients undirected edges
    between two variables that both keep an edge to the index. For a candidate direction
    ``x -> y``, it measures how strongly the change of ``P(x, Z)`` across the index values and the
    change of ``P(y | x, Z)`` depend on each other, where ``Z`` is a set of neighbors that blocks
    their common causes. The measure is the normalized HSIC of the two modules' kernel embeddings
    at each row's index value, with the Gaussian kernel widths of [2]_. ``x -> y`` is oriented when
    that measure is at most the threshold and the measure for ``y -> x`` is above it; when both are
    above it, the edge stays undirected. An orientation that would close a directed cycle is
    skipped, so that edge stays undirected unless Meek's rules orient it. Meek's rules run before
    and after the step. The measure compares the index values as separate domains, so the step needs
    many domains, each with many rows: with two domains, the two changes always depend on each
    other, and nothing is oriented; a per-row index, such as a time step, makes every row its own
    domain, and the step cannot measure a change. The paper leaves the threshold open. For two
    modules that change independently, the measure sits near ``1 / (K - 1)`` over ``K`` domains, so
    the threshold has to rise as the number of domains falls; the 0.06 below is the value the
    measurements used, not a default. The step is off by default, because with ``"fisherz"`` or
    ``"student_t"`` only a variable whose mean follows the index reaches it, and such a shared trend
    makes the changes depend on each other.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A pair no tested query could
    decide, such as one never observed together, is undecided: it is left out of the graph and
    listed in ``metadata["undecided_pairs"]``, and a triple whose ends are undecided is neither a
    collider nor a non-collider to any rule; a test whose ``env`` is constant on its rows, which
    come from one environment, is not tested. A drawn edge an untested conditioning set might have
    separated is listed in ``metadata["possibly_separable"]``: a flag of what the data could not
    check, not a finding that the edge is false; under ``possibly_separable="ambiguous"``, a triple
    with a side it reads (see ``possibly_separable``) is neither a collider nor a non-collider to
    any rule. On merged cohorts, pairs measured only in different cohorts are undecided. A
    continuous column constant on a test's rows raises ``ValueError``. ``env`` must be complete, and
    ``dependence_threshold``, whose kernel measures read every row, raises ``ValueError`` on data
    with missing values. With missing values, the skeleton search runs in one process. Test-wise
    deletion is valid when values are missing completely at random, which means that whether a value
    is missing depends on no value in the data; the rows a test reads are then a random sample of
    all rows, and otherwise they are a selected sample that more rows do not correct.

    **Limitations.** On linear-Gaussian data whose means shift across domains it made fewer errors
    than the empty graph on sparse graphs, until its fits became too slow. Its partial-correlation
    tests see only a change in a variable's mean, so a change of a variance alone, or of an effect's
    weight on a mean-zero cause, orients no edge. Where the graph found is wrong, a true path claim
    can direct an edge against the truth; a known edge belongs in ``required``. The
    `0.10.0 release card <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#the-boundary>`__ has
    the numbers.

    References
    ----------
    .. [1] Huang, B., Zhang, K., Zhang, J., Ramsey, J., Sanchez-Romero, R., Glymour, C., and
       Scholkopf, B. (2020). Causal discovery from heterogeneous/nonstationary data. JMLR 21(89),
       1-53.
    .. [2] Zhang, K., Peters, J., Janzing, D., and Scholkopf, B. (2011). Kernel-based conditional
       independence test and application in causal discovery. UAI, 804-813.
    .. [3] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [4] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [5] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [6] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import cdnod
    >>> rng = np.random.default_rng(0)
    >>> env = np.repeat([0.0, 1.0], 300)
    >>> x = 2.0 * env + rng.standard_normal(600)
    >>> y = x + 0.5 * rng.standard_normal(600)
    >>> cdnod(np.column_stack([x, y]), env).structure.oriented_edges()
    [(0, 1, 'directed')]

    The environments can be a column of a DataFrame, with labels of any kind:

    >>> import pandas as pd
    >>> data = pd.DataFrame({"x": x, "y": y, "site": np.repeat(["north", "south"], 300)})
    >>> cdnod(data, "site").structure.oriented_edges()
    [('x', 'y', 'directed')]
    """
    from andrey.constraint.cdnod import cdnod as _cdnod
    from andrey.core.ties import resolve_collider_order, resolve_in_column_order

    if env is None:
        raise TypeError(
            "cdnod needs env, the environment of each row, or data as each environment's rows apart"
        )
    data, c = env_index(data, env, method="cdnod")
    assert c is not None  # env is not None
    assert indep_test is not None  # method_spec picks it from the column types
    # A mixed-data test takes the index as categorical, so every data column may be continuous.
    X, mask, labels, constraints = fit_inputs(
        data,
        "indep_test",
        indep_test,
        categorical,
        knowledge,
        method="cdnod",
        missing=True,
        index_categorical=True,
    )
    if data_kind(indep_test) != "continuous":
        check_categories(c, None, f"indep_test={indep_test!r}", name="env")
    if dependence_threshold is not None and np.isnan(X).any():
        raise ValueError(
            "cdnod's dependence_threshold step compares kernel measures over every row, so it "
            "does not accept missing values; leave dependence_threshold=None, whose tests delete "
            "rows per test, or drop the rows that hold NaN"
        )
    # The search appends the index as column d, so a copy of it is named as the index.
    d, column = X.shape[1], column_namer(labels)
    ambiguous: set[tuple[int, int, int]] = set()
    conflicts: set[tuple[int, int]] = set()
    index_edges: list[int] = []
    with (
        copies_named(labels, lambda j: "env" if j == d else column(j)),
        rows_tested() as rows,
    ):
        cpdag = _cdnod(
            X,
            c,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            collider_order=resolve_collider_order(collider_order),
            collider_rule=collider_rule,
            categorical=mask,
            dependence_threshold=dependence_threshold,
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
            _conflicts=conflicts,
            _index_edges=index_edges,
        )
    variable = (lambda j: j) if labels is None else (lambda j: labels[j])
    metadata = {
        "conflicts": [list(edge) for edge in sorted(conflicts)],
        "index_edges": [[variable(j), "directed"] for j in index_edges],
        "changing_variables": [variable(j) for j in index_edges],
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(
            rows, lambda j: "env" if j == d else variable(j), _drawn(cpdag, index_edges)
        ),
    }
    return structure_output(cpdag, labels=labels, metadata=metadata)


@method_spec("fci_jci")
def fci_jci(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    collider_rule: str = "sepsets",
    possibly_separable: str = "orient",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    possible_dsep: str | None = None,
    confounded_index: bool = True,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG from multi-domain data with FCI and the environment index as a context variable.

    FCI-JCI [1]_ runs FCI [2]_ [3]_ on data from several domains, regimes, or experiments,
    with the environment index as one more variable that no data variable causes. It allows hidden
    common causes, as :func:`~andrey.fci` does, and uses the index, as :func:`~andrey.cdnod` does,
    to remove the edges that pooling adds and to orient more edges. The index is removed from the
    result, and its edges are listed in ``metadata``: they mark the variables whose distribution the
    domains change and no other variable explains.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample, from all the domains. Coerced to ``float64``. A
        DataFrame's column names become the structure's ``labels``. A column that ``env`` names is
        not a variable: it leaves the data before the fit, and ``categorical``, ``knowledge``, and
        the result's positions count the other columns. A list of DataFrames or 2-D arrays gives
        each domain's rows apart, labeled by position ``0, 1, ...``, and a mapping labels each by
        its key: the rows are stacked in that order, and ``env`` stays ``None``. A position is the
        domain's number, so with ``"fisherz"`` or ``"student_t"`` and three or more domains, the
        order of the list can change the result (see Notes).
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment (domain) of each row of ``data``: one label per row, such as ``0`` and
        ``1`` for two domains, or the name of a column of the DataFrame ``data`` that holds them.
        Numbers are coerced to ``float64`` and read as numbers; whole numbers with ``"chisq"``,
        ``"gsq"``, ``"degen_gauss"``, or ``"cond_gauss"``. String labels are coded ``0, 1, ...``
        in sorted order, whatever the row order; a label is a string or a finite number, not both.
        When every row has the same value, the data come from one domain, and the result equals
        :func:`~andrey.fci`'s. ``None`` needs ``data`` as each domain's rows apart.
    alpha : float, default=0.05
        Significance level of the conditional-independence test. Smaller values give sparser graphs.
    indep_test : str or None, default=None
        Conditional-independence test. ``"fisherz"`` (Fisher's z) and ``"student_t"`` (Student-t)
        test partial correlation, for continuous data. ``"chisq"`` (Pearson's chi-square) and
        ``"gsq"`` (G-square, the likelihood ratio) test categorical data, coded by whole numbers,
        and ``"degen_gauss"`` and ``"cond_gauss"`` mixed data. ``"kci"``, ``"rcot"``, ``"cmiknn"``,
        ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` also detect nonlinear dependence in
        continuous data, at a far higher cost per test. Each as for :func:`~andrey.fci`. The index
        is one more column of the data it tests: the continuous tests read it as a number, and the
        categorical and mixed-data tests as a category; see Notes. ``"normal_scores"`` is not
        offered: ranked over every domain's rows, a column's normal scores mix the domains'
        distributions. ``None`` picks from the declared column types: ``"gsq"`` when every column is
        categorical, ``"cond_gauss"`` when some are, else ``"fisherz"``. A column is declared
        categorical by ``categorical`` or by a DataFrame's ``category``, ``bool``, or text type,
        never by its values. ``metadata["indep_test"]`` names the test used, including the choice
        for ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one; with ``"gsq"`` it gives
        up time, a sum of hypergeometric terms per cell, and in the sparsest tables it rejects below
        that level. ``"nonempty"`` favors degrees of freedom read from the table alone, as Spirtes,
        Glymour and Scheines (1993) count them; it gives up the level, as it rejects true
        independences more often than the nominal rate. Only ``"chisq"`` and ``"gsq"`` read it:
        passing it with another test raises ``ValueError``.
    collider_rule : {"sepsets", "conservative", "majority"}, default="sepsets"
        How a collider is decided on a triple ``x - z - y`` where ``x`` and ``y`` are not adjacent,
        as for :func:`~andrey.fci`: by the separating set that removed the edge ``x - y``
        (``"sepsets"``), or by a vote over the subsets of ``x``'s and ``y``'s neighbors that
        separate them, with the index among the variables, which, if at least one subset separates
        them, makes ``z`` a collider when none of them contains it and a non-collider when all do
        (``"conservative"`` [4]_ [5]_), or when fewer than half and when more than half do
        (``"majority"`` [4]_). Any other triple is ambiguous, and no rule orients through it.
    possibly_separable : {"orient", "ambiguous"}, default="orient"
        How the orientation rules read a triple ``x - z - y`` with a side listed in
        ``metadata["possibly_separable"]``, on data with missing values. ``"orient"`` reads it as
        any other triple. ``"ambiguous"`` makes every such triple neither a collider nor a
        non-collider, and no rule orients through it: with latent variables, a set that separates a
        pair lies within the possible-d-separating set of one end, beyond its adjacencies, so any
        untested conditioning set might be a missed separating set. The collider pass before the
        possible-d-separation step reads it too, so that step's sets, the skeleton, and
        ``metadata["not_tested"]`` can change. ``"orient"`` favors arrowheads; it gives up
        protection from false colliders. ``"ambiguous"`` favors no false collider; it gives up the
        true arrowheads beside them.
    categorical : sequence of str or int, optional
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. The environment index is categorical too,
        so ``data`` may have no categorical column. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. See *Data types* in the
        :doc:`getting-started guide </docs/guides/getting-started>`.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. ``"testwise"`` favors every row each test can read; it gives up its guarantee
        outside values missing completely at random. See **Missing values.** in Notes.
    possible_dsep : {"order_free", "fast"} or None, default=None
        The order of the possible-d-separation step, as for :func:`~andrey.fci`: ``"order_free"``
        visits the pairs and nodes by a rank computed from the data, and ``"fast"`` in column order.
        ``None`` follows ``andrey.config.tie_break``: ``"order_free"`` under ``"rank"``, ``"fast"``
        under ``"position"``. ``"order_free"`` favors a result that does not depend on the column
        order; it gives up time on some graphs, where its order needs more tests. ``"fast"`` favors
        time on some graphs; it gives up that independence.
    confounded_index : bool, default=True
        Whether a hidden variable may cause both the index and a data variable, such as a patient's
        health deciding both the hospital and an outcome: FCI-JCI1 [1]_. ``False`` assumes the index
        is set from outside, as in a randomized experiment, so every edge the index keeps points out
        of it, which can orient more edges: FCI-JCI123, which with one context variable is
        FCI-JCI12. See Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, at least ``0``:
        ``0`` runs only unconditional tests, and ``None`` sets no limit. A smaller cap runs fewer
        tests and can keep edges that a larger conditioning set would remove. It caps the tests of
        the adjacency search, the possible-d-separation step, the conservative and majority
        rules, and the discriminating-path rule.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. The adjacency search starts from
        the allowed pairs and never removes a required pair; the tests can remove any other allowed
        pair. In the PAG, a forbidden ``a -> b`` on an adjacent pair puts an arrowhead at ``a`` and
        can leave a circle at ``b``, and a required one is ``a -> b``; knowledge that puts an
        arrowhead at ``a`` on an edge ``a - b`` while its required edges lead from ``a`` to ``b``
        raises ``ValueError``. The environment index is unconstrained. In the PAG,
        ``required_paths`` makes an adjacent pair ``a -> b``, ``forbidden_paths`` puts an arrowhead
        at the cause, and no test removes a ``confounded`` pair, as :class:`~andrey.Knowledge`
        describes; these marks, and the check that drops a claim the PAG does not meet with a
        :class:`~andrey.KnowledgeWarning` and lists it in ``metadata["unmet_knowledge"]``, assume no
        selection bias.

    Returns
    -------
    StructureOutput
        Wraps a ``pag`` :class:`~andrey.core.GraphStructure` over the ``d`` data variables;
        ``metadata["algorithm"]`` is ``"fci_jci"``. Endpoints carry arrowheads, tails, or circle
        marks (``o``); a bidirected edge (``<->``) marks a probable hidden common cause.
        ``metadata["index_edges"]`` lists the index's edges as ``[variable, kind]`` pairs, by name
        or 0-based position, in column order: ``"directed"`` for ``index -> variable``,
        ``"partially_directed"`` for ``index o-> variable``, and ``"bidirected"`` for
        ``index <-> variable``; it is empty for a single environment. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read, by name or 0-based
        position. With ``collider_rule`` ``"conservative"`` or ``"majority"``,
        ``metadata["ambiguous_triples"]`` lists every unshielded triple no rule reads as a collider
        or a non-collider, as ``[x, z, y]``: 0-based column positions, the middle node second and
        ``x < y``, in ascending order; the index, in a triple through it, is numbered
        ``n_variables``. With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``categorical`` is not a list
        of column names and positions, ``knowledge`` is neither ``None`` nor a
        :class:`~andrey.Knowledge`, or ``env`` is ``None`` with one table.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``env`` is given with each
        domain's rows apart; if those rows' tables differ in kind or columns, or one has no rows or
        holds ``inf``; if ``env`` does not hold one value per row of ``data``, holds ``NaN`` /
        ``inf`` or a missing label, mixes strings and numbers, or names a column ``data`` lacks; if
        ``data`` has too few rows for a test, if ``data`` or ``categorical`` does not suit the
        test, as for :func:`~andrey.fci`, or, with a categorical or mixed-data test, if a numeric
        ``env`` holds a value that is not a whole number. With ``"fisherz"`` or ``"student_t"``,
        also if a column of ``data``, or ``env`` with more than one domain, repeats another up to
        scale and shift, before any test runs; one message names every group of copies and says to
        keep one column per group. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    Notes
    -----
    **When to use.** FCI-JCI suits independent rows from several domains, sites, batches, or
    experiments, with each row's domain known, where hidden common causes may act on the data
    variables and on the domain itself [1]_. It assumes no selection bias and a distribution
    faithful to the graph over the data and the domain. It returns a PAG over the data variables and
    the domain's edges. When no hidden common cause acts, use :func:`~andrey.cdnod`; when the rows
    come from one domain, :func:`~andrey.fci`.

    The index is appended to the data as a last column, the context variable of Joint Causal
    Inference [1]_, and FCI's adjacency search and possible-d-separation step run over the data and
    the index, as for :func:`~andrey.fci`. Pooling makes the domain a common cause of every variable
    it shifts; with the index among the variables, a test can condition on it, so two variables the
    domain shifts together keep no edge unless they depend within the domains.

    Before each collider pass, every edge the index keeps gets an arrowhead at its data variable,
    because no data variable causes the index (JCI Assumption 1). With ``confounded_index=False``,
    the edge also gets a tail at the index, because no hidden variable causes both (JCI Assumption
    2), so the edge is ``index -> variable`` and the index is never a collider; otherwise the end at
    the index stays a circle until a rule orients it. JCI assumes no selection bias, so Zhang's
    rules R1-R4 and R8-R10 [3]_ orient the rest, without R5-R7; they change only circles, so the
    index's marks stay. An index edge then orients edges between data variables: from
    ``index *-> x o-o y`` with ``y`` not adjacent to the index, rule R1 gives ``x -> y``. Mooij et
    al. prove the procedure sound under Assumption 1, and sound and complete under Assumptions 1
    and 2 with one context variable, where their Assumption 3, on how several context variables
    relate, does not apply [1]_.

    A variable keeps an edge to the index when its distribution changes across the domains and no
    other variable explains the change: the domains change its causal module, the distribution given
    its causes; or a path through hidden variables joins it to the index, as when the domains change
    a hidden cause of it; or, with ``confounded_index=True``, a hidden variable causes both.
    ``metadata["index_edges"]`` lists those variables, the index's neighbors; their descendants
    change in distribution too, through them, and are not listed. An edge ``index -> x`` alone does
    not make ``x`` a direct target of the change; Mooij et al. give the condition that does [1]_.

    ``"fisherz"`` and ``"student_t"`` test the index as a number, by linear correlation, so they
    find a change in a variable's mean that follows the index's values, as for
    :func:`~andrey.cdnod`: with three or more domains coded ``0, 1, 2``, a shift in domain ``1``
    alone can go undetected. ``"chisq"``, ``"gsq"``, ``"degen_gauss"``, and ``"cond_gauss"`` take
    each index value as a category, so they find a shift in any domain, whatever the coding. A
    change in a variable's noise variance alone, or in the size of an effect whose cause has mean
    zero, leaves the means unchanged, so the tests other than the nonlinear ones do not find it.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value; ``env`` must be complete. Each
    test reads only the rows where all of its variables are observed (test-wise deletion) [6]_ [7]_,
    the index among them, as for :func:`~andrey.fci`; ``metadata["rows_tested"]`` gives the fewest
    and the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A pair no tested query could
    decide, such as one never observed together, is undecided: it is left out of the graph and
    listed in ``metadata["undecided_pairs"]``, and a triple whose ends are undecided is neither a
    collider nor a non-collider to any rule; a test whose ``env`` is constant on its rows, which
    come from one environment, is not tested. A drawn edge an untested conditioning set might have
    separated is listed in ``metadata["possibly_separable"]``: a flag of what the data could not
    check, not a finding that the edge is false; under ``possibly_separable="ambiguous"``, a triple
    with a flagged side is neither a collider nor a non-collider to any rule. On merged cohorts,
    pairs measured only in different cohorts are undecided. A continuous column constant on a test's
    rows raises ``ValueError``. With missing values, the skeleton search runs in one process.
    Test-wise deletion is valid when values are missing completely at random, which means that
    whether a value is missing depends on no value in the data; the rows a test reads are then a
    random sample of all rows, and otherwise they are a selected sample that more rows do not
    correct.

    **Limitations.** With hidden common causes and one context variable it made as many errors as
    its authors' code and fewer than the empty graph on every dataset. Its possible-d-separation
    step tests subsets of each set, so its cost grows fast with the number of variables, and it
    takes one environment variable, so domains that change different variables share one index. The
    `0.15.0 release card <https://andrey.abel.ai/docs/benchmarks/0.15.0.html#fci-jci>`__ has the
    numbers.

    References
    ----------
    .. [1] Mooij, J. M., Magliacane, S., and Claassen, T. (2020). Joint causal inference from
       multiple contexts. JMLR 21(99), 1-108.
    .. [2] Spirtes, P., Glymour, C., and Scheines, R. (2000). Causation, Prediction, and Search,
       2nd ed. MIT Press.
    .. [3] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [4] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3921-3962.
    .. [5] Ramsey, J., Zhang, J., and Spirtes, P. (2006). Adjacency-faithfulness and conservative
       causal inference. UAI, 401-408.
    .. [6] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [7] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import fci_jci
    >>> rng = np.random.default_rng(1)
    >>> env = np.repeat([0.0, 1.0], 500)
    >>> x = 2.0 * env + rng.standard_normal(1000)
    >>> y = x + rng.standard_normal(1000)
    >>> out = fci_jci(np.column_stack([x, y]), env)
    >>> out.structure.oriented_edges()
    [(0, 1, 'directed')]
    >>> out.metadata["index_edges"]
    [[0, 'partially_directed']]
    """
    from andrey.constraint.fci_jci import fci_jci as _fci_jci
    from andrey.core.ties import resolve_in_column_order, resolve_possible_dsep

    if env is None:
        raise TypeError(
            "fci_jci needs env, the environment of each row, or data as each environment's rows "
            "apart"
        )
    data, c = env_index(data, env, method="fci_jci")
    assert c is not None  # env is not None
    assert indep_test is not None  # method_spec picks it from the column types
    # A mixed-data test takes the index as categorical, so every data column may be continuous.
    X, mask, labels, constraints = fit_inputs(
        data,
        "indep_test",
        indep_test,
        categorical,
        knowledge,
        method="fci_jci",
        missing=True,
        index_categorical=True,
    )
    if data_kind(indep_test) != "continuous":
        check_categories(c, None, f"indep_test={indep_test!r}", name="env")
    # The search appends the index as column d, so a copy of it is named as the index.
    d, column = X.shape[1], column_namer(labels)
    namer = lambda j: "env" if j == d else column(j)  # noqa: E731
    ambiguous: set[tuple[int, int, int]] = set()
    with copies_named(labels, namer), rows_tested() as rows:
        pag, index_edges = _fci_jci(
            X,
            c,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            collider_rule=collider_rule,
            categorical=mask,
            possible_dsep=resolve_possible_dsep(possible_dsep),
            confounded_index=confounded_index,
            max_depth=max_depth,
            constraints=constraints,
            in_column_order=resolve_in_column_order(),
            seed=seed,
            _ambiguous=ambiguous,
        )
    variable = label_of(labels)
    metadata = {
        "index_edges": [[variable(j), kind] for j, kind in index_edges],
        **categorical_metadata(mask, data),
        **_ambiguous_metadata(collider_rule, ambiguous),
        **rows_metadata(
            rows,
            lambda j: "env" if j == d else variable(j),
            _drawn(pag, [j for j, _ in index_edges]),
        ),
    }
    return structure_output(pag, labels=labels, metadata=metadata)


@method_spec("icp")
def icp(
    data: npt.ArrayLike,
    env: npt.ArrayLike | str | None = None,
    *,
    target: str | int | Sequence[str | int] | None = None,
    targets: Mapping | npt.ArrayLike | str | None = None,
    alpha: float = 0.05,
    invariance_test: str = "approximate",
    max_parents: int | None = None,
) -> StructureOutput | list[StructureOutput]:
    """Learn one variable's causal parents from data in several environments with ICP.

    Invariant causal prediction [1]_ tests, for every set of the other variables, whether the
    regression of the target on that set stays the same in every environment. The target's causal
    parents pass that test whenever the environments leave the target's own mechanism alone, so the
    variables in every accepted set are causal parents with high probability. The result is a DAG
    with an edge into the target from each such variable.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables), or a list or mapping of them
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``. A column that ``env`` or ``targets`` names is not a
        variable: it leaves the data before the fit, and ``target``, ``targets``, and the result's
        positions count the other columns.
        A list of DataFrames or 2-D arrays gives each environment's rows apart, labeled by
        position ``0, 1, ...``, and a mapping labels each by its key: the rows are stacked in that
        order, ``env`` stays ``None``, and ``targets``, when given, maps those labels to their
        variables.
    env : array-like of shape (n_samples,) or (n_samples, 1), str, or None, default=None
        The environment of each row: one label per row, strings or finite numbers, not both, or the
        name of a column of the DataFrame ``data`` that holds them. Every distinct label is one
        environment, and at least two are needed. Which variables an environment changed, and how,
        does not matter, as long as it left the target's mechanism alone. ``None`` takes the
        environments from ``data`` given as each environment's rows apart, or else from ``targets``
        given as a column or a 0/1 matrix: rows with the same targets are one environment.
    target : str, int, list of them, or None
        Required. The variable whose parents to find: a column label (a string) or a 0-based
        position (an integer), or a list of them, each answered in turn. ``None`` raises
        ``ValueError``.
    targets : mapping, str, array-like of shape (n_samples, n_variables), or None, default=None
        The variables intervened on in each environment, as for :func:`~andrey.gies`, by column
        label or 0-based position: a mapping ``{label: [variables]}`` from labels of ``env``, with
        ``[]`` for an observational environment and ``None`` or a missing label for unknown
        targets; the name of a column whose cells give each row's targets; or a 0/1 matrix. ICP
        reads only whether an environment intervened on ``target``: the rows of such an
        environment are left out of the fit, and an environment whose targets are unknown is kept.
        ``None`` keeps every environment.
    alpha : float, default=0.05
        Significance level of every set's invariance test, in ``(0, 1)``: the estimated parents are
        a subset of the true parents with probability at least ``1 - alpha``. Smaller values accept
        more sets and give fewer parents.
    invariance_test : {"approximate", "exact"}, default="approximate"
        The invariance test of each set. ``"approximate"`` fits one regression on every row and
        compares each environment's residuals with the other rows' by Welch's t-test of their means
        and the F-test of their variances. ``"exact"`` fits the regression on the other rows,
        predicts each environment's rows, and tests the prediction errors with Chow's F-test, exact
        for Gaussian noise. ``"exact"`` favors that exact level and fewer false parents when an
        intervention that ``targets`` does not name hits the target; it gives up its guarantee when
        the noise is not Gaussian. ``"approximate"`` favors fewer false parents with a hidden common
        cause; it gives up the exact level. See Notes.
    max_parents : int or None, default=None
        A cap on the target's parents, at least ``0``: only sets of at most ``max_parents``
        variables are tested, so no estimate gives the target more than ``max_parents`` parents.
        The guarantee then holds when the target has at most ``max_parents`` parents. ``None``
        tests every set, ``2^(n_variables - 1)`` of them.

    Returns
    -------
    StructureOutput or list of StructureOutput
        With a list of targets, a list of each target's result, in list order, each equal to a call
        for that target alone. Otherwise, wraps a ``dag``
        :class:`~andrey.core.GraphStructure` over every column, with an edge ``x -> target`` from
        each estimated parent ``x`` and no other edge; an absent edge says nothing about the pair.
        ``metadata["algorithm"]`` is ``"icp"``; ``metadata["target"]`` names the target and
        ``metadata["parents"]`` the estimated parents, the variables in every accepted set, by
        column label or 0-based position. ``metadata["p_values"]`` gives one value per column,
        ``None`` at the target: the largest p-value of the tested sets without the column, so a
        variable is a parent exactly when its value is at most ``alpha``, and every value is 1 when
        no set is accepted. ``metadata["confidence_intervals"]`` gives ``[low, high]`` per column,
        ``None`` at the target, for its coefficient in the target's equation; ``None`` when no set
        is accepted. ``metadata["accepted_sets"]`` lists the accepted sets, by size then by
        position; ``metadata["best_p_value"]`` is the largest p-value of any tested set;
        ``metadata["model_rejected"]`` is ``True`` when no set is accepted; and
        ``metadata["environments_left_out"]`` lists the labels of the environments whose ``targets``
        hold ``target``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is ``None`` or
        names a column ``data`` lacks; if ``data`` is not a 2-D numeric matrix with at least two
        rows, no ``NaN`` or ``inf``, and no constant column; if a column repeats another up to scale
        and shift or, with more rows than columns, is a linear function of other columns, one
        message naming every group of copies; if ``data`` holds each environment's rows apart with
        ``env`` or with ``targets`` other than a mapping, or its tables differ in kind or columns,
        or one has no rows; if ``env`` does not hold one label per row, misses a label or holds
        ``inf``, mixes strings and numbers, or names a column ``data`` lacks; if ``targets`` names a
        variable or an environment ``data`` or ``env`` lacks, or gives two rows of one environment
        different targets; if fewer than two environments remain once those that intervened on
        ``target`` are left out, as with one table and neither ``env`` nor ``targets``, or one has
        fewer than 3 rows; if a column is constant on the rows that remain; or if a regression has
        as many coefficients as rows: under ``"exact"``, on the rows outside each environment, where
        a variable constant there also raises.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** ICP suits continuous data from several environments, such as observation and
    experiments, when you want the causal parents of one target with an error guarantee. Its claim
    [1]_: when the target is a linear function of its parents plus noise whose distribution is the
    same in every environment, so that the environments change the other variables in any way but
    leave the target's mechanism alone, the estimated parents are a subset of the true parents with
    probability at least ``1 - alpha``, without faithfulness or any condition on how strong the
    changes are; the exact test needs Gaussian noise. Weak or few changes give fewer parents, not
    wrong ones. It tests every set of the other variables, so it suits a small number of them, or
    more with ``max_parents``. For the whole graph from interventional data, use
    :func:`~andrey.gies` when every environment's intervened variables are known and
    :func:`~andrey.cdnod` when they are not.

    **How it works.** For every set ``S`` of the other variables, of at most ``max_parents`` of
    them, the null hypothesis is that the target's regression on ``S``, with an intercept, has the
    same coefficients and the same residual distribution in every environment [1]_. A set is
    accepted when its p-value exceeds ``alpha``, and the estimate is the intersection of the
    accepted sets. When no set is accepted, the model is rejected and the estimate is empty: an
    intervention on the target, a hidden common cause of the target and a variable, or a nonlinear
    effect can leave no set invariant.

    **Tests.** Under ``"approximate"`` (the paper's Method II), each environment's residuals of
    the regression on every row are compared with the other rows' residuals by Welch's t-test and
    the two-sided F-test of the variances; the environment's p-value is twice the smaller of the
    two. Under ``"exact"`` (Method I), the regression on the other rows predicts the environment's
    rows, and Chow's statistic [2]_ is ``(RSS_all - RSS_out) / n_e`` over
    ``RSS_out / (n_out - |S| - 1)``, which is ``F(n_e, n_out - |S| - 1)`` under the null with
    Gaussian noise, where ``RSS_all`` and ``RSS_out`` are the residual sums of squares of the
    regressions on every row and on the ``n_out`` rows outside the environment's ``n_e``. A set's
    p-value is the smallest environment's times the number of environments, the Bonferroni
    correction, capped at 1; with two environments the approximate test's two comparisons are one
    test, so its factor is 1. Every set's regression reads the environments' Gram matrices,
    computed once, so a set costs the same at any number of rows and neither test subsamples.

    **Intervals.** For an accepted set of ``s`` variables, each coefficient's interval is the
    pooled estimate plus or minus ``t_{1 - alpha/(2s), n - s - 1}`` standard errors, and a variable
    outside the set has the interval ``[0, 0]``; ``metadata["confidence_intervals"]`` is the union
    over the accepted sets. Together the intervals hold every coefficient of the target's equation
    with probability at least ``1 - 2 alpha`` [1]_.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every invariance test compares
    the regression of ``target`` on a set of the other variables across the environments. Drop the
    incomplete rows first.

    **Limitations.** On linear-Gaussian data whose environments leave the target's mechanism alone,
    the share of fits with a false parent stayed well below ``alpha``. When an intervention also
    hits the target and ``targets`` does not name it, a child of the target can enter the estimate
    in more fits than ``alpha`` allows, and a hidden common cause of the target and another variable
    can add a false parent. Without ``max_parents`` it tests every subset of the other variables,
    doubling the cost with each added variable. The
    `0.14.0 release card <https://andrey.abel.ai/docs/benchmarks/0.14.0.html#icp>`__ has the
    numbers.

    References
    ----------
    .. [1] Peters, J., Buhlmann, P., and Meinshausen, N. (2016). Causal inference by using
       invariant prediction: identification and confidence intervals. Journal of the Royal
       Statistical Society Series B 78(5), 947-1012.
    .. [2] Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear
       regressions. Econometrica 28(3), 591-605.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from andrey import icp
    >>> rng = np.random.default_rng(0)
    >>> env = np.repeat(["observed", "shifted"], 500)
    >>> x = rng.standard_normal(1000) + 2.0 * (env == "shifted")  # the shift moves x only
    >>> y = x + rng.standard_normal(1000)
    >>> z = y + rng.standard_normal(1000)
    >>> data = pd.DataFrame({"x": x, "y": y, "z": z})
    >>> out = icp(data, env, target="y")
    >>> out.metadata["parents"], out.structure.oriented_edges()
    (['x'], [('x', 'y', 'directed')])
    """
    from andrey.api._adapt import dag_from_adjacency, data_matrix
    from andrey.constraint.icp import icp as _icp
    from andrey.core.environments import (
        check_constant_columns,
        check_environment_rows,
        normalize_env_targets,
    )
    from andrey.core.knowledge import as_node, column_position
    from andrey.core.score import check_columns

    given = data
    data, codes, env_labels, by_code = normalize_env_targets(
        data, env, targets, require_known_targets=False, method="icp"
    )
    X = data_matrix(data, method="icp")
    labels = column_labels(data)
    d = X.shape[1]
    if target_columns(target, labels, d, "icp") is not None:
        return [
            cast(
                "StructureOutput",
                icp(
                    given,
                    env,
                    target=each,
                    targets=targets,
                    alpha=alpha,
                    invariance_test=invariance_test,
                    max_parents=max_parents,
                ),
            )
            for each in cast("Sequence[str | int]", target)
        ]
    if target is None:
        raise ValueError(
            "icp: pass target, the variable whose parents to find, as a column label or a 0-based "
            "position"
        )
    t = column_position(as_node(target, "target"), labels, d, "icp: target")
    name = column_namer(labels)
    left_out = [k for k, hit in by_code.items() if hit is not None and t in hit]
    keep = ~np.isin(codes, left_out)
    kept = [k for k in range(len(env_labels)) if k not in left_out]
    if len(kept) < 2:
        why = (
            f"targets intervenes on target {name(t)} in environments "
            f"{[env_labels[k] for k in left_out]}, whose rows are left out, so "
            if left_out
            else ""
        )
        raise ValueError(
            f"icp: {why}{len(kept)} environment remains; ICP compares the target's regression "
            "across environments, so it needs at least two"
        )
    X, codes = X[keep], np.searchsorted(kept, codes[keep])
    environments = [env_labels[k] for k in kept]
    check_environment_rows(codes, environments, "icp", 3)
    check_constant_columns(X, name, "icp", "the rows of the environments that remain")
    with copies_named(labels):
        check_columns(np.atleast_2d(np.cov(X, rowvar=False)), len(X), name)
    largest = d - 1 if max_parents is None else min(max_parents, d - 1)
    _check_rows(X, codes, t, largest, invariance_test, environments, name)
    result = _icp(
        X,
        codes,
        t,
        alpha=alpha,
        invariance_test=invariance_test,
        max_parents=max_parents,
        environments=environments,
    )
    node = label_of(labels)
    adjacency = np.zeros((d, d), dtype=np.int8)
    adjacency[list(result.parents), t] = 1
    p_values = [None if j == t else float(v) for j, v in enumerate(result.p_values)]
    intervals = None
    if result.intervals is not None:
        intervals = [
            None if j == t else [float(a), float(b)] for j, (a, b) in enumerate(result.intervals)
        ]
    metadata = {
        "target": node(t),
        "parents": [node(j) for j in result.parents],
        "p_values": p_values,
        "confidence_intervals": intervals,
        "accepted_sets": [[node(j) for j in s] for s in result.accepted],
        "best_p_value": result.best_p_value,
        "model_rejected": not result.accepted,
        "environments_left_out": [env_labels[k] for k in left_out],
    }
    return structure_output(dag_from_adjacency(adjacency), labels=labels, metadata=metadata)


def _check_rows(
    X: np.ndarray,
    codes: np.ndarray,
    target: int,
    largest: int,
    invariance_test: str,
    environments: list,
    name: Callable[[int], str],
) -> None:
    """Raise ``ValueError`` when a regression of ICP has too few rows for its coefficients.

    A set of ``largest`` variables has ``largest + 1`` coefficients with the intercept, and its
    residual variance needs one more row: on every row, and under ``"exact"`` on the rows outside
    each environment, where every variable must also vary.
    """
    need = largest + 2
    if len(X) < need:
        raise ValueError(
            f"icp: sets of {largest} variables need at least {need} rows, and data has {len(X)}; "
            "lower max_parents or give more rows"
        )
    if invariance_test != "exact":
        return
    for e, label in enumerate(environments):
        outside = X[codes != e]
        if len(outside) < need:
            raise ValueError(
                f"icp: the exact test fits sets of {largest} variables on the rows outside each "
                f"environment, which needs {need} rows, and the rows outside {label!r} are "
                f"{len(outside)}; lower max_parents or use invariance_test='approximate'"
            )
        flat = [
            name(j) for j in np.flatnonzero(np.ptp(outside, axis=0) == 0).tolist() if j != target
        ]
        if flat:
            raise ValueError(
                f"icp: the exact test fits each set on the rows outside each environment, and "
                f"columns {', '.join(flat)} are constant on the rows outside {label!r}; use "
                "invariance_test='approximate'"
            )
