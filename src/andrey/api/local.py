"""Target-centered facade adapters: the edges at one variable and their directions.

``mb_by_mb`` and ``mmb_by_mmb`` run :func:`andrey.constraint.sequential.sequential` with the
choices of their papers and return the target's edges as a CPDAG or a PAG holds them; ``cmb``
runs :func:`andrey.constraint.cmb.cmb` and returns the target's edges with the directions it
decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    check_total_conditioning_rows,
    copies_named,
    edge_structure,
    fit_inputs,
    label_of,
    structure_output,
    target_columns,
)
from andrey.api._local import fit_metadata, target_column, target_metadata
from andrey.api._method import method_spec
from andrey.core import ARROW, TAIL, GraphStructure, backend
from andrey.core.ci import counting_tests
from andrey.core.ci_missing import rows_tested

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

    from andrey.constraint.blanket import LocalSearch
    from andrey.constraint.cmb import CMBResult
    from andrey.constraint.sequential import Rules
    from andrey.core import StructureOutput
    from andrey.core.knowledge import Knowledge
    from andrey.core.structure import Kind


def _edge_sets(edges: Iterable[tuple[int, int, int]]) -> dict[str, list[int]]:
    """Return the target's parents, children, and undirected neighbors among ``edges``, by position.

    Each edge is ``(neighbor, mark at the target, mark at the neighbor)``, ascending by neighbor.
    """
    sets: dict[str, list[int]] = {"parents": [], "children": [], "undirected": []}
    for j, near, far in edges:
        if near == ARROW and far == TAIL:
            sets["parents"].append(j)
        elif near == TAIL and far == ARROW:
            sets["children"].append(j)
        else:
            sets["undirected"].append(j)
    return sets


@dataclass(frozen=True)
class _Edges:
    """The edges at a target as a sequential search found them.

    ``marks`` holds each neighbor's mark at the target and at itself, ascending by neighbor,
    ``blanket`` the target's Markov blanket, and ``learned`` the variables whose blankets the search
    learned.
    """

    marks: list[tuple[int, int, int]]
    blanket: list[int]
    learned: list[int]


def _sequential_edges(
    blanket: str, rules: Rules, local_depth: int | None, search: LocalSearch, t: int
) -> _Edges:
    """Return ``t``'s edges by :func:`~andrey.constraint.sequential.sequential` with ``rules``.

    The run keeps its blankets and local structures in a cache of its own.
    """
    from andrey.constraint.sequential import sequential

    result = sequential(search, t, blanket, rules, local_depth=local_depth)
    network = result.network
    return _Edges(network.edges_at(t), list(result.cache.blankets[t]), list(result.learned))


def _star(marks: Iterable[tuple[int, int, int]], t: int, d: int, kind: Kind) -> GraphStructure:
    """Return the structure over all ``d`` variables of the edges at ``t`` alone.

    The edges are the ``(neighbor, mark at t, mark at the neighbor)`` of :func:`_edge_sets`.
    """
    return edge_structure(d, [(t, v, near, far) for v, near, far in marks], kind=kind)


def _cmb_codes(blanket: str, decided: np.ndarray | None, search: LocalSearch, t: int) -> CMBResult:
    """Return ``t``'s identities by :func:`~andrey.constraint.cmb.cmb`."""
    from andrey.constraint.cmb import cmb as identities

    return identities(search, t, blanket, decided=decided)


@method_spec("mb_by_mb")
def mb_by_mb(
    data: npt.ArrayLike,
    *,
    target: str | int | Sequence[str | int] | None = None,
    blanket: str = "iamb_forward",
    alpha: float = 0.05,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    local_max_depth: int | None = 5,
    knowledge: Knowledge | None = None,
) -> StructureOutput | list[StructureOutput]:
    """Learn the edges at one variable and their directions, its CPDAG's, with MB-by-MB.

    MB-by-MB [1]_ finds the direct causes and the direct effects of a target without learning the
    whole graph. It learns the target's Markov blanket and the local structure over it, then the
    blanket and local structure of a neighbor, and so on outward, one blanket at a time, until no
    further blanket can direct an edge at the target. The answer holds the edges the target has in
    the CPDAG, the causal graphs the tests cannot tell apart, with their directions.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, list of them, or None, default=None
        Required: the variable to answer for, a column label (a string) or a 0-based position (an
        integer). A list of them answers each one with one search, which makes a step two targets
        share once, on ``andrey.config.n_jobs`` worker processes for a partial-correlation
        ``indep_test``; with missing values, each target has a separate search. ``None`` raises
        ``ValueError``.
    blanket : str, default="iamb_forward"
        The Markov blanket learner run for each variable the search visits: ``"iamb_forward"``,
        IAMB's grow phase without its shrink, the learner of the paper's runs [1]_, which keeps a
        variable added in error rather than lose a member, ``"total_conditioning"``, each variable
        the test finds dependent on it given all the other variables [5]_, or
        :func:`~andrey.iamb`, :func:`~andrey.inter_iamb`, :func:`~andrey.gs`,
        :func:`~andrey.fast_iamb`, :func:`~andrey.hiton_mb`, :func:`~andrey.mmmb`, or
        :func:`~andrey.pcmb` by name.
    alpha : float, default=0.05
        Significance level of the conditional-independence test, in ``(0, 1)``. The paper runs
        0.01 and 0.05 [1]_.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`. The
        paper's runs use the likelihood-ratio test on categorical data, ``"gsq"`` [1]_.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one: over 35,808 true
        independences of random discrete networks (3 to 5 categories, 0 to 3 conditioning variables,
        500 to 5,000 rows), a test at 0.05 rejected 4.9% (``"chisq"``) and 3.8% (``"gsq"``). With
        ``"gsq"`` it gives up time, a sum of hypergeometric terms per cell, and rejects below the
        nominal rate in the sparsest tables (0.8% at 3,125 cells). ``"nonempty"`` favors the degrees
        of freedom of Spirtes, Glymour and Scheines (1993), read from the table alone, with no
        hypergeometric sum; it gives up the level: 11.1% and 21.7% of the same true independences
        were rejected. Their time and their power were not measured against each other. Only
        ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. ``"testwise"`` favors every row each test can read; it gives up its guarantee
        outside values missing completely at random. With PC at 20 variables and 1,000 rows, 10 to
        30% of the values missing completely at random, test-wise deletion made median errors of 6.5
        to 7.5 where dropping the incomplete rows made 8.5 and 20.0 on 124 and 10 complete rows and
        had no complete row at 30% (complete data: 4.0; 0.11.0 release card). Where values are
        missing for a reason that other variables carry, on Erdos-Renyi graphs of 10 to 100
        variables with one observed parent per indicator, ``pc``'s ``"mvpc"`` made fewer errors than
        test-wise deletion in 14 of 72 settings at ``alpha=0.01`` and 28 at 0.05, and more in none.
        Not measured with this method. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, the blanket
        learner's and the local structures', at least ``0``; ``None``, the default, sets no limit.
        Under a cap, the blanket learner stops growing at ``max_depth + 1`` members.
    local_max_depth : int or None, default=5
        The maximum size of the subsets the local structures test, at least ``0`` and never above
        ``max_depth``; ``None`` sets no limit. The paper leaves the cap open and its runs use 5
        [1]_, the default, on the local structures alone, as its complexity bounds only them by the
        cap: the same cap on the blanket learner stops its grow phase before the blanket is whole,
        so the errors rise with the sample.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's blanket from the start, and no test removes it. A pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, never shares an edge. Every edge
        a tier or an edge forbidden one way directs is directed so as each local structure joins
        the growing graph, over any v-structure's arrowhead, and Meek's rules, R4 included, direct
        the edges that follow. ``required_paths`` and ``forbidden_paths`` raise
        ``NotImplementedError``.

    Returns
    -------
    StructureOutput or list of StructureOutput
        With a list of targets, a list holding each target's result, in its order, as that target
        alone gives it, except that ``metadata["n_tests"]`` counts only the tests no earlier
        target's result counts. Otherwise, wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`
        over every column holding only the
        edges at ``target``: ``x -> target`` for a direct cause, ``target -> x`` for a direct
        effect, and ``x - target`` where the CPDAG leaves the direction undecided. An absent edge
        away from the target means nothing. ``metadata["target"]`` names the target, and
        ``metadata["parents"]``, ``metadata["children"]``, and ``metadata["undirected"]`` list
        those columns, by column label or 0-based position. ``metadata["blankets_learned"]`` lists
        the variables whose Markov blanket the search learned, in order, the target first, so its
        length is the number of blankets the answer took. ``metadata["algorithm"]`` is
        ``"mb_by_mb"``. Under a mixed-data setting, ``metadata["categorical"]`` lists the
        categorical columns it read. With a randomized ``indep_test``, ``metadata["seed"]`` is the
        seed used. ``metadata["n_tests"]`` is the number of conditional-independence tests the fit
        evaluated; a query answered again from a kept answer is not a new test.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is ``None`` or
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. With ``blanket="total_conditioning"``, a partial-correlation
        test, and no ``max_depth`` below ``d - 2``, also if ``data`` has fewer than ``d + 2`` rows
        with every variable observed, ``d`` its number of columns. Also if ``knowledge`` names a
        column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.
    NotImplementedError
        If ``knowledge`` holds ``required_paths`` or ``forbidden_paths``; the message names the
        methods that honor them.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** MB-by-MB suits i.i.d. observational data with no unmeasured common causes, in
    which every independence follows from the graph (faithfulness), when you want the direct causes
    and effects of one variable rather than the whole graph [1]_. Its paper runs it on categorical
    data with the likelihood-ratio test; the default Fisher-Z test assumes continuous data with
    linear effects and Gaussian noise, and the other tests read categorical or mixed data or detect
    nonlinear effects, as :func:`~andrey.pc` describes. It is designed for large graphs in which
    each variable has a small Markov blanket: it learns only the blankets the answer needs, often
    two or three [1]_. It returns the edges at the target with the directions the CPDAG gives them;
    for the whole CPDAG, use :func:`~andrey.pc`, and for the target's neighbors without directions,
    :func:`~andrey.mmpc`.

    **How it works.** A waitlist starts with the target [1]_. Each step takes the waitlist's head
    ``x``, learns its Markov blanket ``MB(x)`` with the ``blanket`` learner, and appends the members
    neither waiting nor done to the waitlist. It then learns the local structure over ``x`` and its
    blanket: ``x`` and a member are adjacent unless some subset of the rest of the blanket
    separates them, every other pair is tested against every subset of the other variables there,
    and a triple ``a - b - c`` with ``a`` and ``c`` separated by a set without ``b`` is a
    v-structure ``a -> b <- c`` [1]_. When the blanket and ``x`` lie inside those of a variable
    done earlier, or every member of the blanket is done, the local structure is read from what is
    already learned instead [1]_. Only the edges at ``x`` and the v-structures that contain ``x``
    join the growing graph: the rest of the local structure may come from variables outside the
    blanket [1]_. Every independence a test of the search found is kept, a blanket learner's
    included, and a pair it separates holds no edge [1]_: after each step, such an edge leaves the
    growing graph, and a v-structure that named it no longer directs its other edge.
    Meek's rules [2]_ then direct the edges that follow, reading a missing edge as an absence only
    where a test separated the pair, and a triple as a non-collider only where the separating set
    holds its middle [1]_. Last, every waitlist variable that each path to the target in the
    growing graph reaches only through a directed edge is dropped, since its structure cannot
    direct an edge at the target; the search stops when the waitlist is empty [1]_. With exact
    tests, the edges at the target and their directions are those of the true CPDAG [1]_. An edge
    that two v-structures direct both ways is left undirected.

    Every choice between candidates of equal strength goes to the column whose rank, computed from
    the data, comes first, as for :func:`~andrey.iamb`, and a pair separated at some size keeps the
    set with the largest p-value, so the answer does not depend on the column order unless two
    columns are identical.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [3]_ [4]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A continuous column constant on a
    test's rows raises ``ValueError``. Test-wise deletion is valid when values are missing
    completely at random, which means that whether a value is missing depends on no value in the
    data; the rows a test reads are then a random sample of all rows, and otherwise they are a
    selected sample that more rows do not correct.

    **Limitations.** At its paper's settings (Alarm and Hailfinder samples at 500 to 5,000 rows, a
    45 x 45 chessboard) it made fewer errors at the target than the empty answer, with a small
    fraction of the tests of one :func:`~andrey.pc` fit; that fit made fewer errors in a few cells,
    among them Alarm at 5,000 rows with alpha 0.05 and Hailfinder at 500 rows with alpha 0.05, and
    on the chessboard. The `local discovery release card
    <https://andrey.abel.ai/docs/benchmarks/local-discovery.html#mb-by-mb>`_ has the numbers. Each
    local structure tests every pair of a blanket against every subset of the rest, up to
    ``local_max_depth`` members, so a blanket of ``m`` variables takes up to about ``m**2 * 2**m``
    tests without the cap. Under the default ``local_max_depth=5``, a pair that only a larger set
    separates keeps its edge. The default blanket learner never drops a variable it added, so a
    large sample, in which more variables test dependent, gives larger blankets, and each test of
    the blanket learner conditions on the whole set so far. ``"total_conditioning"`` with a
    partial-correlation test needs at least two more rows than variables, rows with every variable
    observed, and raises ``ValueError`` below that. The numbers are a reference, not a hard limit:
    they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Wang, C., Zhou, Y., Zhao, Q., and Geng, Z. (2014). Discovering and orienting the edges
       connected to a target variable in a DAG via a sequential local learning approach.
       Computational Statistics and Data Analysis 77, 252-266.
    .. [2] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [3] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [4] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.
    .. [5] Pellet, J.-P., and Elisseeff, A. (2008). Using Markov blankets for causal structure
       learning. Journal of Machine Learning Research 9, 1295-1342.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import mb_by_mb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = mb_by_mb(np.column_stack([a, b, c]), target=2)
    >>> out.metadata["parents"], out.metadata["children"]
    ([0, 1], [])
    """
    from andrey.constraint._parallel_targets import answers
    from andrey.constraint.blanket import local_search
    from andrey.constraint.sequential import MB_BY_MB

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="mb_by_mb", missing=True
    )
    d = X.shape[1]
    targets = target_columns(target, labels, d, "mb_by_mb")
    t = target_column(target, labels, d, "mb_by_mb") if targets is None else -1
    check_total_conditioning_rows(X, blanket, indep_test, method="mb_by_mb", max_depth=max_depth)
    name = label_of(labels)
    answer = partial(_sequential_edges, blanket, MB_BY_MB, local_max_depth)

    def output(t: int, edges: _Edges, n: int, rows: Mapping[int, int]) -> StructureOutput:
        metadata = target_metadata(t, _edge_sets(edges.marks), name)
        metadata["blankets_learned"] = [name(x) for x in edges.learned]
        metadata |= fit_metadata(mask, data, rows, n)
        return structure_output(_star(edges.marks, t, d, "cpdag"), labels=labels, metadata=metadata)

    with copies_named(labels):
        search_of = partial(
            local_search,
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            categorical=mask,
            seed=seed,
            max_depth=max_depth,
            constraints=constraints,
        )
        if targets is None:
            with rows_tested() as rows, counting_tests() as tests:
                edges = answer(search_of(), t)
            return output(t, edges, tests.n, rows)
        found = answers(search_of, answer, targets, backend.worker_count())
    return [output(t, *answer_) for t, answer_ in zip(targets, found, strict=True)]


def _pag_sets(edges: Iterable[tuple[int, int, int]]) -> dict[str, list[int]]:
    """Return the target's parents, children, bidirected, and undetermined neighbors in ``edges``.

    The PAG's edges at the target fall in one each, by position; the edges are those of
    :func:`_edge_sets`.
    """
    sets: dict[str, list[int]] = {
        "parents": [],
        "children": [],
        "bidirected": [],
        "undetermined": [],
    }
    for j, near, far in edges:
        if near == ARROW and far == ARROW:
            sets["bidirected"].append(j)
        elif near == ARROW and far == TAIL:
            sets["parents"].append(j)
        elif near == TAIL and far == ARROW:
            sets["children"].append(j)
        else:
            sets["undetermined"].append(j)
    return sets


@method_spec("mmb_by_mmb")
def mmb_by_mmb(
    data: npt.ArrayLike,
    *,
    target: str | int | Sequence[str | int] | None = None,
    blanket: str = "total_conditioning",
    alpha: float = 0.01,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    local_max_depth: int | None = 10,
    knowledge: Knowledge | None = None,
) -> StructureOutput | list[StructureOutput]:
    """Learn the edges at one variable and their marks, its PAG's, with MMB-by-MMB.

    MMB-by-MMB [1]_ finds the direct causes and the direct effects of a target when unmeasured
    common causes may exist, without learning the whole graph. It learns the target's Markov
    blanket and the local structure over it, then the blanket and local structure of a neighbor,
    and so on outward, one blanket at a time, until the marks at the target are decided or no
    further blanket can decide them. The answer holds the edges the target has in the PAG, the
    graph of marks every causal model that the tests cannot tell apart shares, hidden common
    causes allowed.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, list of them, or None, default=None
        Required: the variable to answer for, a column label (a string) or a 0-based position (an
        integer). A list of them answers each one with one search, which makes a step two targets
        share once, on ``andrey.config.n_jobs`` worker processes for a partial-correlation
        ``indep_test``; with missing values, each target has a separate search. ``None`` raises
        ``ValueError``.
    blanket : str, default="total_conditioning"
        The Markov blanket learner run for each variable the search visits:
        ``"total_conditioning"``, the paper's [1]_, keeps each variable the test finds dependent
        on it given all the other variables [2]_, or :func:`~andrey.gs`, :func:`~andrey.iamb`,
        :func:`~andrey.inter_iamb`, or :func:`~andrey.fast_iamb` by name, which find the same
        blanket with hidden variables [3]_ with tests that condition on fewer variables.
    alpha : float, default=0.01
        Significance level of the conditional-independence tests, in ``(0, 1)``. The paper states
        none; 0.01 is the level of its authors' code. ``"total_conditioning"`` tests at
        ``alpha / (10 * floor(d / 10))`` over ``d`` columns, ``alpha`` itself below 10 columns, as
        that code does; see **How it works.** in Notes.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`. The
        paper's runs use the Fisher-Z test on linear-Gaussian data, ``"fisherz"`` [1]_.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one: over 35,808 true
        independences of random discrete networks (3 to 5 categories, 0 to 3 conditioning variables,
        500 to 5,000 rows), a test at 0.05 rejected 4.9% (``"chisq"``) and 3.8% (``"gsq"``). With
        ``"gsq"`` it gives up time, a sum of hypergeometric terms per cell, and rejects below the
        nominal rate in the sparsest tables (0.8% at 3,125 cells). ``"nonempty"`` favors the degrees
        of freedom of Spirtes, Glymour and Scheines (1993), read from the table alone, with no
        hypergeometric sum; it gives up the level: 11.1% and 21.7% of the same true independences
        were rejected. Their time and their power were not measured against each other. Only
        ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. ``"testwise"`` favors every row each test can read; it gives up its guarantee
        outside values missing completely at random. With PC at 20 variables and 1,000 rows, 10 to
        30% of the values missing completely at random, test-wise deletion made median errors of 6.5
        to 7.5 where dropping the incomplete rows made 8.5 and 20.0 on 124 and 10 complete rows and
        had no complete row at 30% (complete data: 4.0; 0.11.0 release card). Where values are
        missing for a reason that other variables carry, on Erdos-Renyi graphs of 10 to 100
        variables with one observed parent per indicator, ``pc``'s ``"mvpc"`` made fewer errors than
        test-wise deletion in 14 of 72 settings at ``alpha=0.01`` and 28 at 0.05, and more in none.
        Not measured with this method. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, the blanket
        learner's and the local structures', at least ``0``; ``None``, the default, sets no limit.
        ``"total_conditioning"`` conditions every test on the other ``d - 2`` variables, so below
        that it runs no test and keeps every variable in the blanket; the other learners stop
        growing at ``max_depth + 1`` members.
    local_max_depth : int or None, default=10
        The maximum size of the subsets the local structures test, at least ``0`` and never above
        ``max_depth``; ``None`` sets no limit. The paper leaves the cap open; 10 is the cap of its
        authors' code, which bounds only the local structures by it.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's blanket from the start, and no test removes it. A pair the knowledge makes
        non-adjacent, forbidden both ways or outside ``allowed``, never shares an edge. As each
        local structure joins the growing graph, a tier or an edge forbidden one way puts an
        arrowhead at the variable it keeps from being a cause, a required edge a tail at its cause,
        and the orientation rules keep those marks. ``required_paths`` and ``forbidden_paths``
        raise ``NotImplementedError``.

    Returns
    -------
    StructureOutput or list of StructureOutput
        With a list of targets, a list holding each target's result, in its order, as that target
        alone gives it, except that ``metadata["n_tests"]`` counts only the tests no earlier
        target's result counts. Otherwise, wraps a ``pag`` :class:`~andrey.core.GraphStructure`
        over every column holding only the
        edges at ``target``, with their marks: ``x -> target`` for a direct cause, ``target -> x``
        for a direct effect, ``x <-> target`` where a hidden common cause joins them and neither
        causes the other, and a circle where the PAG leaves the mark undecided. An absent edge
        away from the target means nothing. ``metadata["target"]`` names the target;
        ``metadata["parents"]``, ``metadata["children"]``, and ``metadata["bidirected"]`` list the
        columns of those edges, and ``metadata["undetermined"]`` those whose edge has a circle at
        either end, by column label or 0-based position. ``metadata["markov_blanket"]`` lists the
        target's Markov blanket as the ``blanket`` learner found it.
        ``metadata["blankets_learned"]`` lists the variables whose Markov blanket the search
        learned, in order, the target first, so its length is the number of blankets the answer
        took. ``metadata["algorithm"]`` is ``"mmb_by_mmb"``. Under a mixed-data setting,
        ``metadata["categorical"]`` lists the categorical columns it read. With a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["n_tests"]`` is the
        number of conditional-independence tests the fit evaluated; a query answered again from a
        kept answer is not a new test. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is ``None`` or
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. With ``blanket="total_conditioning"``, a partial-correlation
        test, and no ``max_depth`` below ``d - 2``, also if ``data`` has fewer than ``d + 2`` rows
        with every variable observed, ``d`` its number of columns. Also if ``knowledge`` names a
        column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.
    NotImplementedError
        If ``knowledge`` holds ``required_paths`` or ``forbidden_paths``; the message names the
        methods that honor them.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** MMB-by-MMB suits i.i.d. observational data in which unmeasured common causes
    may exist but the rows were not selected by the variables' values, and in which every
    independence follows from the graph (faithfulness), when you want the direct causes and effects
    of one variable rather than the whole graph [1]_. Its paper runs it on continuous
    linear-Gaussian data with the Fisher-Z test and the total-conditioning blanket learner, which
    reads such data; the other tests read categorical or mixed data or detect nonlinear effects,
    as :func:`~andrey.pc` describes, with a blanket learner whose tests condition on fewer
    variables, such as ``"iamb"``. It is designed for large graphs in which each variable has a
    small Markov blanket: it learns the blankets near the target, not the whole graph [1]_. It
    returns the edges at the target with the marks the PAG gives them; for the whole PAG, use
    :func:`~andrey.fci`, and with no unmeasured common causes, :func:`~andrey.mb_by_mb`.

    **How it works.** A waitlist starts with the target [1]_. Each step takes the waitlist's head
    ``x`` and learns its Markov blanket ``MB(x)`` with the ``blanket`` learner: with hidden
    variables, the blanket also holds the variables joined to ``x`` or to its children by a path of
    hidden common causes, and their parents [3]_. It then learns the local structure over ``x`` and
    its blanket with the search of :func:`~andrey.pc`: ``x`` and a member are adjacent unless some
    subset of the rest of the blanket separates them, every other pair is tested against the
    subsets of either end's neighbors, and a triple ``a - b - c`` with ``a`` and ``c`` separated
    by a set without ``b`` is a v-structure ``a *-> b <-* c`` [1]_. Since a member and ``x`` are
    adjacent exactly when no subset of the rest of the member's own blanket separates them either
    [1]_, the pair is tested on the smaller of the two blankets, as MBCS* limits the search [3]_;
    the method learns the blanket of each member to compare them. When the blanket and ``x`` lie
    inside those of a variable done earlier, or every member of the blanket is done, the local
    structure is read from what is already learned instead [1]_. Only the edges at ``x`` and the
    v-structures that contain ``x`` join the growing graph, since the rest of the local structure
    may come from variables outside the blanket [1]_. The FCI orientation rules without selection
    bias [4]_ then decide the marks that follow, reading a missing edge as an absence only where a
    test separated the pair, and a triple as a non-collider only where the separating set holds
    its middle. The variables adjacent to ``x`` that are neither waiting nor done join the
    waitlist [1]_. The search stops at the first of three rules [1]_: every mark at the target's
    edges is decided (R1); the waitlist is empty (R2); every path from the target that begins with
    an edge with a circle meets an arrowhead pointing away from the target before it reaches a
    waitlist variable (R3).

    Where its authors' implementation goes beyond the paper, the method follows it, with two
    safeguards, each of which makes the answer correct where the paper's steps as written can be
    wrong. A v-structure at ``x`` joins the growing graph only when the variables off ``x`` confirm
    it from their own blankets, its far edge present in the far variable's own local structure or
    its two ends separated there by a set without ``x``, since the local search can keep an edge
    that only a larger subset of the blanket separates and bring it in through a v-structure. The
    stop rules are read only once more blankets have been learned than the target has neighbors
    after its own step, since R3 can stop before a discriminating path decides a circle at the
    target.

    The paper runs total conditioning without stating its level [1]_. The
    ``"total_conditioning"`` learner tests the ``d - 1`` candidates of each blanket at
    ``alpha / (10 * floor(d / 10))`` over ``d`` columns, ``alpha`` itself below 10 columns, as the
    authors' code does; the other learners and the local structures test at ``alpha``. At a flat
    level the scan admits more variables outside the blanket as ``d`` grows, and each one doubles
    the subsets a local structure tests.

    Every choice between candidates of equal strength goes to the column whose rank, computed from
    the data, comes first, as for :func:`~andrey.iamb`, and a pair separated at some size keeps the
    set with the largest p-value, so the answer does not depend on the column order unless two
    columns are identical.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [5]_ [6]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A continuous column constant on a
    test's rows raises ``ValueError``. Test-wise deletion is valid when values are missing
    completely at random, which means that whether a value is missing depends on no value in the
    data; the rows a test reads are then a random sample of all rows, and otherwise they are a
    selected sample that more rows do not correct.

    **Limitations.** On the paper's four networks with hidden nodes, at 1,000 and 5,000 rows and the
    paper's targets, its edges and marks were better than those of the authors' code in five of
    eight settings and no worse in the rest, and its errors at the target were below the empty
    answer's in all eight; the `local discovery release card
    <https://andrey.abel.ai/docs/benchmarks/local-discovery.html#mmb-by-mmb>`_ has the numbers. Two
    adjacent variables that both have very wide blankets can still take long.
    ``"total_conditioning"`` needs continuous data with linear effects, and with a
    partial-correlation test at least two more rows than variables, rows with every variable
    observed, or it raises ``ValueError``; on other data or with fewer rows, choose another learner.
    The numbers are a reference, not a hard limit: they change with the data generator, the density,
    the method's settings, and the machine.

    References
    ----------
    .. [1] Xie, F., Li, Z., Wu, P., Zeng, Y., Liu, C., and Geng, Z. (2024). Local causal structure
       learning in the presence of latent variables. Proceedings of the 41st International
       Conference on Machine Learning, PMLR 235, 54511-54530.
    .. [2] Pellet, J.-P., and Elisseeff, A. (2008). Using Markov blankets for causal structure
       learning. Journal of Machine Learning Research 9, 1295-1342.
    .. [3] Pellet, J.-P., and Elisseeff, A. (2008). Finding latent causes in causal networks: an
       efficient approach based on Markov blankets. Advances in Neural Information Processing
       Systems 21, 1249-1256.
    .. [4] Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the
       presence of latent confounders and selection bias. Artificial Intelligence 172(16-17),
       1873-1896.
    .. [5] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [6] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import mmb_by_mmb
    >>> rng = np.random.default_rng(0)
    >>> a, b, e = rng.standard_normal((3, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = mmb_by_mmb(np.column_stack([a, b, c, c + e]), target=2)
    >>> out.metadata["children"], out.metadata["undetermined"]
    ([3], [0, 1])
    """
    from andrey.constraint._parallel_targets import answers
    from andrey.constraint.blanket import local_search
    from andrey.constraint.sequential import MMB_BY_MMB

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="mmb_by_mmb", missing=True
    )
    d = X.shape[1]
    targets = target_columns(target, labels, d, "mmb_by_mmb")
    t = target_column(target, labels, d, "mmb_by_mmb") if targets is None else -1
    check_total_conditioning_rows(X, blanket, indep_test, method="mmb_by_mmb", max_depth=max_depth)
    name = label_of(labels)
    answer = partial(_sequential_edges, blanket, MMB_BY_MMB, local_max_depth)

    def output(t: int, edges: _Edges, n: int, rows: Mapping[int, int]) -> StructureOutput:
        sets = {"markov_blanket": edges.blanket, **_pag_sets(edges.marks)}
        metadata = target_metadata(t, sets, name)
        metadata["blankets_learned"] = [name(x) for x in edges.learned]
        metadata |= fit_metadata(mask, data, rows, n)
        return structure_output(_star(edges.marks, t, d, "pag"), labels=labels, metadata=metadata)

    with copies_named(labels):
        search_of = partial(
            local_search,
            X,
            alpha=alpha,
            total_alpha=alpha / (10 * (d // 10)) if d >= 10 else alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            categorical=mask,
            seed=seed,
            max_depth=max_depth,
            constraints=constraints,
        )
        if targets is None:
            with rows_tested() as rows, counting_tests() as tests:
                edges = answer(search_of(), t)
            return output(t, edges, tests.n, rows)
        found = answers(search_of, answer, targets, backend.worker_count())
    return [output(t, *answer_) for t, answer_ in zip(targets, found, strict=True)]


@method_spec("cmb")
def cmb(
    data: npt.ArrayLike,
    *,
    target: str | int | Sequence[str | int] | None = None,
    blanket: str = "hiton_mb",
    alpha: float = 0.02,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput | list[StructureOutput]:
    """Learn the edges at one variable and their directions with CMB.

    CMB (Causal Markov Blanket discovery) [1]_ finds the direct causes and the direct effects of a
    target without learning the whole graph. It learns the target's parents and children and its
    Markov blanket, labels each neighbor from how its dependence on the other neighbors changes
    when the target joins the conditioning set, retests an inconclusive pair given one member's
    blanket, and visits each undecided neighbor in turn to learn the target's identity there. The
    answer holds the edges at the target with the directions the tests decide.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    target : str, int, list of them, or None, default=None
        Required: the variable to answer for, a column label (a string) or a 0-based position (an
        integer). A list of them answers each one with one search, which makes a step two targets
        share once, on ``andrey.config.n_jobs`` worker processes for a partial-correlation
        ``indep_test``; with missing values, each target has a separate search. ``None`` raises
        ``ValueError``.
    blanket : {"hiton_mb", "mmmb", "pcmb"}, default="hiton_mb"
        The Markov blanket learner run for each variable CMB visits: :func:`~andrey.hiton_mb`, the
        learner of the paper's runs [1]_, :func:`~andrey.mmmb`, or :func:`~andrey.pcmb`. CMB reads
        a variable's parents and children apart from its spouses, which these learners return.
    alpha : float, default=0.02
        Significance level of the conditional-independence test, in ``(0, 1)``. The paper's runs
        use 0.02 [1]_, the default.
    indep_test : str or None, default=None
        Conditional-independence test, any of :func:`~andrey.pc`'s: ``"fisherz"``,
        ``"student_t"``, or ``"normal_scores"`` for continuous data, ``"chisq"`` or ``"gsq"`` for
        categorical data, ``"degen_gauss"`` or ``"cond_gauss"`` for mixed data, or a nonlinear
        test. ``None`` picks from the declared column types, as for :func:`~andrey.pc`. The
        paper's runs use a mutual-information test on categorical data, ``"gsq"`` [1]_.
        ``metadata["indep_test"]`` names the test the fit used: the one picked from the
        declared column types when this is ``None``.
    degrees_of_freedom : {"mean", "nonempty"}, default="mean"
        Degrees of freedom of the categorical tests, ``"chisq"`` and ``"gsq"``. ``"mean"`` is the
        mean of the statistic when ``y`` is shuffled within each configuration of the conditioning
        set; ``"nonempty"`` is ``(r_x - 1) * (r_y - 1)`` per configuration over the values of ``x``
        and ``y`` present in it, the table's empty rows and columns struck. The two differ only on
        sparse tables. There ``"mean"`` favors a level near the nominal one: over 35,808 true
        independences of random discrete networks (3 to 5 categories, 0 to 3 conditioning variables,
        500 to 5,000 rows), a test at 0.05 rejected 4.9% (``"chisq"``) and 3.8% (``"gsq"``). With
        ``"gsq"`` it gives up time, a sum of hypergeometric terms per cell, and rejects below the
        nominal rate in the sparsest tables (0.8% at 3,125 cells). ``"nonempty"`` favors the degrees
        of freedom of Spirtes, Glymour and Scheines (1993), read from the table alone, with no
        hypergeometric sum; it gives up the level: 11.1% and 21.7% of the same true independences
        were rejected. Their time and their power were not measured against each other. Only
        ``"chisq"`` and ``"gsq"`` read it: passing it with another test raises ``ValueError``.
    categorical : sequence of str or int, or None, default=None
        The categorical columns of ``data``, for a mixed-data test, as for :func:`~andrey.pc`. A
        continuous test raises ``ValueError`` on a declared column.
    missing : {"testwise"} or None, default=None
        How missing values (``NaN``) are handled. ``"testwise"`` deletes rows per test: each test
        reads only the rows where all of its variables are observed. ``None`` picks the method's
        default, ``"testwise"``. On data with missing values, ``metadata["missing"]`` records the
        handling used. ``"testwise"`` favors every row each test can read; it gives up its guarantee
        outside values missing completely at random. With PC at 20 variables and 1,000 rows, 10 to
        30% of the values missing completely at random, test-wise deletion made median errors of 6.5
        to 7.5 where dropping the incomplete rows made 8.5 and 20.0 on 124 and 10 complete rows and
        had no complete row at 30% (complete data: 4.0; 0.11.0 release card). Where values are
        missing for a reason that other variables carry, on Erdos-Renyi graphs of 10 to 100
        variables with one observed parent per indicator, ``pc``'s ``"mvpc"`` made fewer errors than
        test-wise deletion in 14 of 72 settings at ``alpha=0.01`` and 28 at 0.05, and more in none.
        Not measured with this method. See **Missing values.** in Notes.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, and ``"gpdc"`` make; the other tests ignore it. With those
        tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test, the blanket
        learner's and CMB's own, at least ``0``; ``None``, the default, sets no limit, as the paper
        sets none. A pair whose test with the target would condition on more variables runs
        neither of its two tests and decides nothing.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Each end of a required edge is in
        the other's parents and children from the start, and no test removes it. A pair the
        knowledge makes non-adjacent never shares an edge. At each variable CMB visits, the
        direction a tier, a required edge, or an edge forbidden one way gives an edge is fixed
        before any test, and the tests label the rest from it. ``required_paths`` and
        ``forbidden_paths`` raise ``NotImplementedError``.

    Returns
    -------
    StructureOutput or list of StructureOutput
        With a list of targets, a list holding each target's result, in its order, as that target
        alone gives it, except that ``metadata["n_tests"]`` counts only the tests no earlier
        target's result counts. Otherwise, wraps a ``cpdag`` :class:`~andrey.core.GraphStructure`
        over every column holding only the
        edges at ``target``: ``x -> target`` for a direct cause, ``target -> x`` for a direct
        effect, and ``x - target`` where the direction is left undecided. An absent edge away from
        the target means nothing. ``metadata["target"]`` names the target, and
        ``metadata["parents"]``, ``metadata["children"]``, and ``metadata["undirected"]`` list
        those columns, by column label or 0-based position; ``metadata["markov_blanket"]`` lists
        the target's Markov blanket as ``blanket`` learned it. ``metadata["blankets_learned"]``
        lists the variables whose Markov blanket the fit learned, in order, the target first, so
        its length is the number of blankets the answer took. ``metadata["algorithm"]`` is
        ``"cmb"``. Under a mixed-data setting, ``metadata["categorical"]`` lists the categorical
        columns it read. With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["n_tests"]`` is the number of conditional-independence tests the fit evaluated,
        the blanket learner's included; a query answered again from a kept answer is not a new
        test. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is neither a column
        label nor a position, ``categorical`` is not a list of column names and positions, or
        ``knowledge`` is neither ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``target`` is ``None`` or
        names a column ``data`` lacks, or if ``data`` or ``categorical`` does not suit the test, as
        for :func:`~andrey.pc`. With ``"fisherz"`` or ``"student_t"``, also if a column repeats
        another up to scale and shift. Also if ``knowledge`` names a column ``data`` lacks or
        contradicts itself, as :class:`~andrey.Knowledge` lists.
    NotImplementedError
        If ``knowledge`` holds ``required_paths`` or ``forbidden_paths``; the message names the
        methods that honor them.

    Warns
    -----
    DataTypeWarning
        As for :func:`~andrey.pc`, when a mixed-data test finds no declared categorical column or a
        setting reads columns that look like category codes as numbers; once per process.

    Notes
    -----
    **When to use.** CMB suits i.i.d. observational data with no unmeasured common causes and no
    selection bias, in which every independence follows from the graph (faithfulness), when you
    want the direct causes and effects of one variable rather than the whole graph [1]_. Its paper
    runs it on categorical data with a mutual-information test at 0.02; the default Fisher-Z test
    assumes continuous data with linear effects and Gaussian noise, and the other tests read
    categorical or mixed data or detect nonlinear effects, as :func:`~andrey.pc` describes. It is
    designed for large graphs in which each variable has a small Markov blanket: it learns few
    blankets beyond the target's, two to three on average in the paper's runs [1]_. For the whole
    CPDAG, use :func:`~andrey.pc`; for the target's edges with the CPDAG's directions by a search
    that grows a local graph, :func:`~andrey.mb_by_mb`.

    **How it works.** Each neighbor of a variable gets an identity: parent, child, or undecided
    [1]_. Two neighbors ``x`` and ``y`` of the target ``t`` are tested given a set ``Z`` and given
    ``Z`` with ``t``. Independent, then dependent once ``t`` joins, makes both parents: ``t`` is a
    collider between them. Dependent, then independent, makes at least one a child: once one is a
    parent, the other is a child. A pair that neither decides is inconclusive [1]_. The first
    round takes ``Z`` empty. Each inconclusive pair is then retested given the blanket of its
    member of lower rank, less ``t`` and the other member, until no neighbor is inconclusive
    [1]_. A neighbor still inconclusive with two parents becomes a parent when it lies in the set
    that separated them (Meek's rule R3 [2]_): ``t`` is a collider between the two parents, so no
    descendant of ``t`` lies in that set. The remaining inconclusive neighbors become undecided.
    Last, each undecided neighbor ``x`` is visited in turn and the same steps run there, without
    returning to a variable whose own steps are still open [1]_: ``t`` a child of ``x`` makes
    ``x`` a parent of ``t`` and the other member of each pair that made one of the two a child a
    child; ``t`` a parent of ``x`` makes ``x`` a child. The paper's steps have no acyclicity rule;
    Andrey adds Meek's rule R2 [2]_ to the visit, so that an edge the CPDAG directs only because
    the opposite direction would close a cycle is directed too. A visited ``x`` directs each edge
    of ``t`` that closes a triangle with it: ``t -> x -> w`` makes ``w`` a child of ``t``,
    ``w -> x -> t`` makes ``w`` a parent, and a direction already found for ``w`` directs ``x``
    the same way. The rule reads only directions the steps found, so it adds no test and no edge.
    A label once given is never replaced by a weaker one: a parent or child stays so. A neighbor
    still undecided at the end keeps an undirected edge.

    Every choice between candidates of equal strength goes to the column whose rank, computed from
    the data, comes first, as for :func:`~andrey.iamb`, and pairs are tested in rank order, so the
    answer does not depend on the column order unless two columns are identical.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Each test reads only the rows
    where all of its variables are observed (test-wise deletion) [3]_ [4]_, so it keeps rows that
    deleting every incomplete row first would drop; ``metadata["rows_tested"]`` gives the fewest and
    the median rows a test read. A test with fewer rows than it needs (``|S| + 4`` rows for
    ``"fisherz"`` and ``"normal_scores"``, ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and
    2 for most other tests), or with no degrees of freedom left, is not tested: it never separates a
    pair and never confirms a link, and ``metadata["not_tested"]`` counts the conditioning sets not
    tested, including sets skipped because no row can test them. A continuous column constant on a
    test's rows raises ``ValueError``. Test-wise deletion is valid when values are missing
    completely at random, which means that whether a value is missing depends on no value in the
    data; the rows a test reads are then a random sample of all rows, and otherwise they are a
    selected sample that more rows do not correct.

    **Limitations.** Measured on samples of the Alarm network (37 variables) at 1,000 rows with
    G-square tests at alpha 0.02, every variable a target, as its paper ran it: its edges at the
    target were nearer the true CPDAG's than those of :func:`~andrey.pc` on the same data and than
    the empty answer, on 20 datasets and again on 20 fresh ones. The `local discovery release card
    <https://andrey.abel.ai/docs/benchmarks/local-discovery.html#cmb>`_ has the numbers. With exact
    tests and exact blankets, every edge and every direction it returns is the true CPDAG's, but it
    can leave undirected an edge the CPDAG directs: one whose direction follows from a separating
    set its steps never test, or from directions its visits do not find. Over every target of 100
    random graphs of 6 to 20 variables, that happened at 2%, 5%, and 20% of targets at an average of
    1.5, 2.5, and 4 neighbors per variable, and on Alarm to 0.09 edges per target. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Gao, T., and Ji, Q. (2015). Local causal discovery of direct causes and effects.
       Advances in Neural Information Processing Systems 28, 2512-2520.
    .. [2] Meek, C. (1995). Causal inference and causal explanation with background knowledge.
       UAI, 403-410.
    .. [3] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [4] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import cmb
    >>> rng = np.random.default_rng(0)
    >>> a, b = rng.standard_normal((2, 1000))
    >>> c = a + b + 0.5 * rng.standard_normal(1000)
    >>> out = cmb(np.column_stack([a, b, c]), target=2)
    >>> out.metadata["parents"], out.metadata["children"]
    ([0, 1], [])
    """
    from andrey.constraint._parallel_targets import answers
    from andrey.constraint.blanket import local_search
    from andrey.constraint.cmb import CHILD, PARENT

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = fit_inputs(
        data, "indep_test", indep_test, categorical, knowledge, method="cmb", missing=True
    )
    d = X.shape[1]
    targets = target_columns(target, labels, d, "cmb")
    t = target_column(target, labels, d, "cmb") if targets is None else -1
    name = label_of(labels)
    decided = None if constraints is None else constraints.decided()
    answer = partial(_cmb_codes, blanket, decided)

    def output(t: int, result: CMBResult, n: int, rows: Mapping[int, int]) -> StructureOutput:
        marks = [
            (x, ARROW if code == PARENT else TAIL, ARROW if code == CHILD else TAIL)
            for x, code in sorted(result.codes.items())
        ]
        sets = {"markov_blanket": result.blanket, **_edge_sets(marks)}
        metadata = target_metadata(t, sets, name)
        metadata["blankets_learned"] = [name(x) for x in result.learned]
        metadata |= fit_metadata(mask, data, rows, n)
        return structure_output(_star(marks, t, d, "cpdag"), labels=labels, metadata=metadata)

    with copies_named(labels):
        search_of = partial(
            local_search,
            X,
            alpha=alpha,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            categorical=mask,
            seed=seed,
            max_depth=max_depth,
            constraints=constraints,
        )
        if targets is None:
            with rows_tested() as rows, counting_tests() as tests:
                result = answer(search_of(), t)
            return output(t, result, tests.n, rows)
        found = answers(search_of, answer, targets, backend.worker_count())
    return [output(t, *answer_) for t, answer_ in zip(targets, found, strict=True)]
