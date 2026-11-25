"""Temporal-family facades: the LiNGAM, PCMCI, Granger, and DYNOTEARS time-series methods.

VARMA-LiNGAM, Longitudinal LiNGAM, PCMCI, PCMCI+, J-PCMCI+, LPCMCI, Granger, and DYNOTEARS each
run the native engine and return one uniform ``StructureOutput`` wrapping a
:class:`~andrey.core.TemporalStructure` (a per-lag stack of graphs, plus the lossless coefficient
weights for the LiNGAM, Granger, and DYNOTEARS methods). The mapping onto the public core contract
lives in the private ``_temporal`` adapters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from andrey.api._method import drop_unmet, method_spec
from andrey.core.backend import one_blas_thread
from andrey.core.ci_missing import RowCounts, named_pairs, rows_metadata, rows_tested
from andrey.core.knowledge import Knowledge, KnowledgeError, resolve_lagged
from andrey.core.score import copies_error, copy_groups
from andrey.core.structure import NULL
from andrey.core.warning_policy import warn_search_limit

from ._adapt import (
    DROP_DATASETS,
    DROP_UNITS,
    KEEP_STEPS,
    categorical_metadata,
    checked_data,
    column_labels,
    column_mask,
    column_namer,
    copies_named,
    data_matrix,
    int_in_range,
    label_of,
    shared_labels,
    with_labels,
)
from ._temporal import (
    adapt_dynotears,
    adapt_granger,
    adapt_longitudinal_lingam,
    adapt_lpcmci,
    adapt_pcmci,
    adapt_pcmciplus,
    adapt_rpcmci,
    adapt_varma_lingam,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt

    from andrey.core import RegimeOutput, StructureOutput
    from andrey.core.knowledge import LaggedConstraints


@method_spec("varma_lingam")
def varma_lingam(
    data: npt.ArrayLike,
    *,
    order: tuple[int, int] = (1, 1),
    criterion: str | None = None,
    prune: bool = False,
    structural_ma: bool = False,
    max_iter: int = 100,
    measure: str = "pwling",
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a temporal causal structure from one multivariate time series with VARMA-LiNGAM.

    VARMA-LiNGAM [1]_ fits a vector autoregressive moving-average model, then applies LiNGAM to
    the residuals to fix a causal order over the instantaneous (contemporaneous) effects [2]_. Use
    it when a single multivariate series carries both same-step and lagged causal influence.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    order : tuple of int, default=(1, 1)
        The ``(p, q)`` autoregressive and moving-average lag counts, each at least ``0`` and not
        both ``0``.
    criterion : {"aic", "bic", "hqic"} or None, default=None
        Order-selection criterion. When set, ``(p, q)`` is chosen by that information criterion
        over ``0 <= p <= order[0]`` and ``0 <= q <= order[1]``, excluding ``(0, 0)``; ``None``
        uses ``order`` directly.
    prune : bool, default=False
        Re-estimate the weights on the causal order: adaptive Lasso sets small weights to zero and
        least squares refits the rest. With ``False`` every lagged weight is nonzero, so every lag
        graph links every pair of variables, self-loops included.
    structural_ma : bool, default=False
        How the moving-average weights are expressed: ``True`` maps them onto the independent
        noise terms, ``(I - B0) Theta_j (I - B0)^-1`` [1]_; ``False`` uses ``(I - B0) Theta_j``.
        ``B0`` is the lag-0 weight matrix and ``Theta_j`` the fitted moving-average coefficients.
        Ignored when ``prune=True``.
    max_iter : int, default=100
        Most iterations of the likelihood fit, at least ``1``. The optimizer's function evaluations
        are limited to 150 per iteration; each iteration spends about one per parameter, so a fit
        of more than about 150 parameters, such as ``order=(1, 1)`` on 8 variables, can use them up
        first. With ``criterion`` set, every fit in the grid gets these limits. A fit that reaches
        either without converging warns with :class:`~andrey.SearchLimitWarning`; raise
        ``max_iter`` to let it run on.
    measure : {"pwling", "skew", "tanh"}, default="pwling"
        How the instantaneous order search, DirectLiNGAM on the innovations, picks the most
        exogenous variable, as in :func:`~andrey.direct_lingam` [3]_: ``"pwling"`` approximates the
        pairwise likelihood ratios by entropies, and ``"tanh"`` and ``"skew"`` by a correlation of
        nonlinear functions of the standardized innovations.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over the autoregressive lags ``0`` to ``order[0]``: required and forbidden
        links, the pairs allowed to be linked, and temporal tiers, all honored. A link
        ``(cause, effect, lag)`` is the link from ``cause`` at time ``t - lag`` to ``effect`` at
        time ``t``, lag ``0`` being the same step; a forbidden ``(cause, effect)`` pair forbids it
        at every lag, and a required link names its lag. At lag ``0``, DirectLiNGAM on the
        innovations honors the knowledge as :func:`~andrey.direct_lingam` does: the causal order
        puts the cause of each required link, and each earlier tier, first, and the regression
        drops forbidden parents and keeps required ones unpenalized. With ``prune=True``, each
        variable's regression also drops its forbidden lagged regressors and keeps its required
        ones unpenalized; a required lagged link is in the graph even where its estimated weight
        is ``0``. With ``prune=False`` every lagged weight stays, so a required lagged link is in
        the graph, and a forbidden lagged link raises. Tiers forbid every link from a
        later tier into an earlier one, and a pair left out of ``allowed`` has no link at any
        lag; a variable's links to its own past stay allowed. The moving-average weights are not
        constrained.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure`.
        ``structure.lag_weights[k][i, j]`` is the autoregressive weight of edge ``i -> j`` at lag
        ``k``, lag 0 being instantaneous; ``structure.lag_weights_ma[k]`` holds the moving-average
        weights at lag ``k + 1``, in the same orientation. The lag graphs follow the autoregressive
        weights. ``ordering`` is the instantaneous causal order, and ``metadata["algorithm"]`` is
        ``"varma_lingam"``. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``order`` is ``(0, 0)``, or if
        ``data`` has fewer than 2 columns. Also if ``knowledge`` names a column ``data`` lacks,
        contradicts itself, requires a link without a lag or beyond ``order[0]``, forbids a lagged
        link while ``prune=False``, or requires a link at a lag beyond the ``p`` that ``criterion``
        selected.

    Warns
    -----
    SearchLimitWarning
        When a fit takes ``max_iter`` iterations, or the function evaluations they allow, without
        converging; raise ``max_iter`` to let it run on.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.
    BoundaryWarning
        When the data are larger than the largest size whose fits finished within the time and
        memory caps of the published benchmark, at the measured setting nearest the data's rows, and
        fits of the next size ran past a cap. It warns once per process and names that size and the
        release card; the size is a reference, not a hard limit.

    Notes
    -----
    **When to use.** VARMA-LiNGAM suits one continuous, observational multivariate time series in
    which each variable is a linear function of the other variables at the same step, the past
    values of every variable, and past noise terms, with weights that do not change over time; the
    noise terms are independent and non-Gaussian, the same-step effects form no cycle, and no
    unmeasured variable causes two measured ones [1]_ [2]_. Its likelihood fit estimates a weight
    for every pair of variables at every lag, so it suits a modest number of variables over a long
    series. It returns a temporal structure of autoregressive weights at lags ``0`` (the same step)
    to ``p``, with the moving-average weights alongside and the same-step causal order; with
    ``prune=False`` every lag graph links every pair of variables. For Gaussian noise, use
    :func:`~andrey.pcmciplus`; when unmeasured common causes are plausible,
    :func:`~andrey.lpcmci`; for many units measured at a few time points,
    :func:`~andrey.longitudinal_lingam`; and for i.i.d. rows rather than a series,
    :func:`~andrey.direct_lingam`.

    The likelihood fit runs on the series with each column centered and divided by its standard
    deviation, as the structural VAR model that VARMA-LiNGAM extends takes zero-mean series [2]_,
    and its matrices and residuals are mapped back to the data's units, so shifting or rescaling a
    column changes the fit only through rounding.

    DirectLiNGAM orders the innovations of the fitted model. Each pairwise regression in its
    ordering takes the least-squares slope, so every residual it scores is uncorrelated with its
    regressor in the sample.

    With ``prune=True`` each variable is regressed on its causal predecessors at the same step,
    the lagged values of every variable, and the lagged independent noise terms. Adaptive Lasso
    selects the regressors, and an ordinary least-squares fit with an intercept on the selected
    ones gives their weights, so the kept weights are not shrunk toward zero. The noise terms come
    from a lag-0 matrix refit the same way. The regressions start at time step ``max(p, q)``: the
    first steps serve only as lagged values. No lagged value wraps around from the end of the
    series, so no step is regressed on a later one.

    **Choosing the measure.** ``"skew"`` and ``"tanh"`` suit the noise they suit in
    :func:`~andrey.direct_lingam`, whose ``measure`` entry gives the numbers; here they read the
    standardized innovations.

    **Column order.** The fit runs on the columns sorted by a rank computed from the data, as for
    :func:`~andrey.hc`, so reordering the columns relabels the result and changes nothing else,
    unless two columns are identical.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the VARMA fit and the LiNGAM
    step on its residuals require every time step to be complete.
    Use :func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests
    delete rows per test and keep every time step, as dropping a step would join the steps around
    it.

    **Limitations.** It assumes linear effects and non-Gaussian noise, and with ``prune=False``
    every lag graph is complete. With ``prune=True``, the median F1 score of its directed
    cross-links, which is 0 for the empty graph, was at least 0.66 through 40 variables, or 20 with
    series five times as long, where fits of the next size ran past 1 h on 4 cores (`0.10.0
    release card
    <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#methods-on-their-own-axes>`_). With
    ``order=(1, 1)``, its median errors stayed below the empty graph's from 300 time steps and
    exceeded them at 100 steps from 10 variables on (`0.5.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.5.0.html#varma-lingam>`_). The likelihood fit can stop
    short of convergence without a warning, as after a failed line search, which a larger
    ``max_iter`` does not prevent, and where it stops depends on the numerical libraries and the
    CPU's BLAS kernels, so the same series can give a different graph on another machine. With
    ``criterion`` set, it fits one model for every ``(p, q)`` in the grid. The numbers are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Kawahara, Y., Shimizu, S., and Washio, T. (2011). Analyzing relationships among ARMA
       processes based on non-Gaussianity of external influences. Neurocomputing 74(12-13),
       2212-2221.
    .. [2] Hyvarinen, A., Zhang, K., Shimizu, S., and Hoyer, P. O. (2010). Estimation of a
       structural vector autoregression model using non-Gaussianity. JMLR 11, 1709-1731.
    .. [3] Hyvarinen, A., and Smith, S. M. (2013). Pairwise likelihood ratios for estimation of
       non-Gaussian structural equation models. JMLR 14, 111-152.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import varma_lingam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-1, 1, 300)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.uniform(-1, 1, 300)
    >>> out = varma_lingam(np.column_stack([x, y]), order=(1, 1), prune=True)
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    """
    from andrey.temporal.varma import varma_lingam as _native

    X = data_matrix(data, method="varma_lingam", fix=KEEP_STEPS)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    if tuple(order) == (0, 0):
        raise ValueError("varma_lingam: order must have p or q above 0, got (0, 0)")
    labels = column_labels(data)
    constraints = resolve_lagged(knowledge, X.shape[1], labels, order[0], method="varma_lingam")
    causal_order, psis, omegas = _native(
        X,
        order=order,
        criterion=criterion,
        prune=prune,
        structural_ma=structural_ma,
        max_iter=max_iter,
        measure=measure,
        constraints=constraints,
        name=column_namer(labels),
    )
    required = None if constraints is None else constraints.required
    return with_labels(adapt_varma_lingam(causal_order, psis, omegas, required), labels)


@method_spec("longitudinal_lingam")
def longitudinal_lingam(
    data: Sequence[npt.ArrayLike],
    *,
    tau_max: int = 1,
    measure: str = "pwling",
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a temporal causal structure from panel data with Longitudinal LiNGAM.

    Longitudinal LiNGAM [1]_ handles the same units measured at several time points: at each
    occasion it regresses out the past ``tau_max`` time points, then applies DirectLiNGAM to orient
    the instantaneous effects. Use it for repeated-measures / panel designs rather than one long
    series.

    Parameters
    ----------
    data : sequence of array-like, each of shape (n_samples, n_variables)
        One dataset per time point, over the same units and identical shape. Each is coerced to
        ``float64``. DataFrames must name their columns alike; the names become the structure's
        ``labels``.
    tau_max : int, default=1
        The largest lag: the number of past time points regressed out before the instantaneous
        fit, from ``1`` to the number of time points minus one.
    measure : {"pwling", "skew", "tanh"}, default="pwling"
        How the instantaneous order search picks the most exogenous variable, as in
        :func:`~andrey.direct_lingam` [2]_: ``"pwling"`` approximates the pairwise likelihood ratios
        by entropies, and ``"tanh"`` and ``"skew"`` by a correlation of nonlinear functions of the
        standardized columns.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over lags ``0`` to ``tau_max``, the same at every time point: required and
        forbidden links, the pairs allowed to be linked, and temporal tiers, all honored. A link
        ``(cause, effect, lag)`` is the link from ``cause`` at time point ``t - lag`` to
        ``effect`` at time point ``t``, lag ``0`` being the same time point; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag. At
        lag ``0``, each time point's DirectLiNGAM honors the knowledge as
        :func:`~andrey.direct_lingam` does: the causal order puts the cause of each required
        link, and each earlier tier, first, and the regression drops forbidden parents and keeps
        required ones unpenalized. A forbidden lagged link is left out of the least-squares fit
        of its effect's lagged weights, so its weight is ``0``; every other lagged weight stays,
        and a required lagged link is in the graph even where its least-squares weight is ``0``.
        Tiers forbid every link from a later tier into an earlier one, and a pair left out of
        ``allowed`` has no link at any lag; a variable's links to its own past stay allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure`. The first time point has no
        past, so it is not estimated. ``structure.times`` lists the estimated time points, ``1`` to
        ``T - 1``; ``structure.time_weights[t][k][i, j]`` is the weight of edge ``i -> j`` at lag
        ``k`` for the t-th of them (lag 0 is instantaneous), and ``structure.lag_weights`` repeats
        the last time point's. A lag that reaches before the first time point is ``NaN``. Lagged
        weights are not pruned, so every lagged graph links every pair of variables, self-loops
        included; judge lagged effects by weight size. ``metadata["causal_orders"]`` holds one
        causal order per time point, the first one empty; they need not agree, so ``ordering`` is
        ``None``. ``metadata["algorithm"]`` is ``"longitudinal_lingam"``. ``metadata["tie_break"]``
        is the tie-break rule the fit ran under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` has fewer than two
        time points, if ``tau_max`` is at least the number of time points, if the per-time arrays
        differ in shape, or if the DataFrames' column names differ. Also if ``knowledge`` names a
        column the data lacks, contradicts itself, or requires a link without a lag or beyond
        ``tau_max``.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** Longitudinal LiNGAM suits continuous, observational panel data: the same units
    measured at several time points, given as one dataset per time point [1]_. At each time point,
    each variable is a linear function of the other variables at that time point and of every
    variable at the past ``tau_max`` time points, plus independent non-Gaussian noise; the
    same-time effects form no cycle, no unmeasured variable causes two measured ones, and the
    effects may change from one time point to the next [1]_. Each time point's fit treats the units
    as its samples, so it is designed for many units, while the time points can be few. It returns
    a temporal structure with weights at lags ``0`` to ``tau_max`` for each time point after the
    first, in which every lagged graph links every pair of variables, and one same-time causal
    order per time point. For one long series rather than many units, use
    :func:`~andrey.varma_lingam`, or :func:`~andrey.pcmciplus` when the noise is Gaussian; for the
    i.i.d. rows of a single time point, use :func:`~andrey.direct_lingam`.

    At each time point DirectLiNGAM orders the residuals of the lag regressions. Each pairwise
    regression in its ordering takes the least-squares slope, so every residual it scores is
    uncorrelated with its regressor in the sample, and listing every unit twice leaves the causal
    orders unchanged.

    **Choosing the measure.** ``"skew"`` and ``"tanh"`` suit the noise they suit in
    :func:`~andrey.direct_lingam`, whose ``measure`` entry gives the numbers; here they read the
    standardized columns of each time point.

    **Column order.** The fit runs on the variables sorted by a rank computed from the data over all
    time points, as for :func:`~andrey.hc`, so reordering the columns relabels the result and
    changes nothing else, unless two columns are identical.

    **Missing values.** A ``NaN`` in a dataset raises ``ValueError``: each time point's fit compares
    the residuals of every variable over the same units, so a unit with a missing value has no place
    in it. Drop the units that hold one from every dataset first, which suits values missing
    completely at random.

    **Limitations.** It assumes linear effects and non-Gaussian noise. Each time point's fit uses
    only the units' rows at that time, so its accuracy depends on the number of units, not on the
    number of time points. On simulated panels of 20 time points, the median F1 score of its
    directed cross-links, which is 0 for the empty graph, was 0.94 to 1 at every size through 40
    variables, the largest planned, with 500 and 2,000 units (`0.10.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#methods-on-their-own-axes>`_). The numbers
    are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

    References
    ----------
    .. [1] Kadowaki, K., Shimizu, S., and Washio, T. (2013). Estimation of causal structures in
       longitudinal data using non-Gaussianity. IEEE International Workshop on Machine Learning
       for Signal Processing (MLSP), 1-6.
    .. [2] Hyvarinen, A., and Smith, S. M. (2013). Pairwise likelihood ratios for estimation of
       non-Gaussian structural equation models. JMLR 14, 111-152.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import longitudinal_lingam
    >>> rng = np.random.default_rng(0)
    >>> panel = [rng.uniform(-1, 1, (200, 3)) for _ in range(4)]
    >>> out = longitudinal_lingam(panel, tau_max=1)
    >>> out.structure.type
    'temporal'
    """
    from andrey.temporal.longitudinal import LongitudinalLiNGAM

    panel = [
        data_matrix(X, name=f"data[{t}]", method="longitudinal_lingam", fix=DROP_UNITS)
        for t, X in enumerate(data)
    ]
    if len(panel) < 2:
        raise ValueError(f"data must hold at least two time points, got {len(panel)}")
    tau_max = int_in_range(tau_max, "tau_max", 1, len(panel) - 1)
    labels = shared_labels(column_labels(X) for X in data)
    constraints = resolve_lagged(
        knowledge, panel[0].shape[1], labels, tau_max, method="longitudinal_lingam"
    )
    model = LongitudinalLiNGAM(n_lags=tau_max, measure=measure, constraints=constraints).fit(panel)
    required = None if constraints is None else constraints.required
    out = adapt_longitudinal_lingam(model.adjacency_matrices_, model.causal_orders_, required)
    return with_labels(out, labels)


@method_spec("pcmci")
def pcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    pc1_alpha: float = 0.2,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the lagged causal links of one multivariate time series with PCMCI.

    PCMCI [1]_ works in two steps. The PC1 step selects, for each variable, a few candidate parents
    among the lagged values of all variables. The momentary conditional independence (MCI) step
    then tests each lagged link given the selected parents of both its ends. Conditioning on the
    cause's own past removes the autocorrelation that makes a plain lagged-correlation test report
    false links. Use it on one series whose causal effects take at least one time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag tested, at least ``1``: links at lags ``1`` to ``tau_max`` are tested.
    pc1_alpha : float, default=0.2
        Significance level of the PC1 step, in ``(0, 1)``: a candidate parent is dropped when a test
        gives it a p-value above ``pc1_alpha``. A larger value keeps more parents as conditions for
        the MCI step.
    alpha : float, default=0.05
        Significance level of the MCI step, in ``(0, 1)``, the level that decides the returned
        links: a link is in the graph when its p-value is at most ``alpha``.
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the MCI p-values are corrected for the number of links tested. ``"fdr_bh"`` replaces
        them with their Benjamini-Hochberg adjusted values [4]_ over all the links together, every
        lag and every ordered pair of variables, a variable's link to its own past included. A link
        is then in the graph when its adjusted p-value is at most ``alpha``, which holds the false
        discovery rate, the expected share of false links among the returned links, at or below
        ``alpha`` when the tests are independent or positively dependent. The adjusted p-value is
        never below the tested one, so ``"fdr_bh"`` can remove links and never adds one, and
        ``metadata["p_values"]`` holds the adjusted values. ``"none"`` keeps each p-value as tested,
        so about ``alpha`` of the absent links are in the graph. A link the knowledge forbids is no
        test: it is not counted and keeps p-value ``1``; a required link is counted with p-value
        ``0``. ``"fdr_bh"`` favors fewer extra links; it gives up true links whose p-value is not
        small enough for the number of tests. ``"none"`` favors true links; it gives up control of
        the extra ones, which grow with the number of variables. On simulated linear-Gaussian VAR
        series of 5 to 80 variables, 500 and 1,000 time steps, and ``tau_max`` of 1 and 2 (20
        cells), ``"fdr_bh"`` had fewer mean errors (missing plus extra lagged links) than ``"none"``
        in 16 cells and more in 1, with 5 variables; at 80 variables, 1,000 steps, and
        ``tau_max=2``, ``"none"`` had 779 errors, above the empty graph's 481, with 616 extra links,
        and ``"fdr_bh"`` had 252, with 13 extra and 239 missed (`0.12.0 release card
        <https://andrey.abel.ai/docs/benchmarks/0.12.0.html#fdr-control-in-pcmci-and-granger>`__).
        See Notes.
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for linear effects in continuous data: ``"fisherz"`` applies
        Fisher's z-transform and the normal distribution, ``"student_t"`` the Student-t
        distribution. ``"normal_scores"`` applies ``"fisherz"`` to the normal scores of each column
        of the lagged values, for monotone effects. ``"chisq"`` (Pearson's chi-square) and ``"gsq"``
        (G-square, the likelihood ratio) test categorical series, coded by whole numbers.
        ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test
        mixed series, continuous and categorical columns, by a likelihood ratio. Seven more tests
        detect nonlinear dependence in continuous data, at a far higher cost per test: ``"cmiknn"``,
        a nearest-neighbor conditional mutual information test, and ``"gpdc"``, Gaussian-process
        regression with a distance correlation test, the nonlinear tests the method was published
        with; ``"kci"``, the kernel test; ``"rcot"``, its random-feature approximation; and
        ``"gcm"``, ``"wgcm"``, and ``"pcm"``, the generalized, weighted generalized, and projected
        covariance measures of regression residuals. ``None`` picks from the declared column types:
        ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are, else
        ``"fisherz"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values. ``metadata["indep_test"]`` names
        the test used, including the choice for ``None``.
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
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. Every lagged value of a column has its type.
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
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test of the PC1
        step, at least ``0``: round ``p`` of PC1 conditions on the ``p`` strongest other
        candidates, and the rounds stop after ``p = max_depth``; ``None`` sets no limit. A smaller
        cap runs fewer tests and can keep candidate parents that a larger conditioning set would
        remove. The MCI step conditions on the selected parents of both ends, however many.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables: required and forbidden links, the pairs allowed to be linked, and temporal
        tiers, all honored. A link ``(cause, effect, lag)`` is the link from ``cause`` at time
        ``t - lag`` to ``effect`` at time ``t``; a forbidden ``(cause, effect)`` pair forbids it at
        every lag, and a required link names its lag, from ``1`` to ``tau_max``. No test runs on
        a forbidden or a required link: a forbidden link is in no conditioning set and not in the
        graph, and a required one is a parent in every PC1 round and in the graph. Tiers forbid
        every link from a later tier into an earlier one, and a pair left out of ``allowed`` has
        no link at any lag; a variable's links to its own past stay allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to ``tau_max``
        and no lag 0. ``structure.lag(k)`` is a ``digraph`` whose edge ``i -> j`` is the link from
        variable ``i`` at time ``t - k`` to variable ``j`` at time ``t``; a self-loop is a
        variable's dependence on its own past. The structure has no ``lag_weights``.
        ``metadata["p_values"][k - 1][i][j]`` and ``metadata["statistics"][k - 1][i][j]`` hold the
        MCI p-value, adjusted under ``fdr_method="fdr_bh"``, and test statistic of that link (the
        partial correlation under ``"fisherz"`` and ``"student_t"``; see Notes), for every pair,
        significant or not; a link the knowledge forbids reads p-value 1 and statistic 0, and a
        required one reads 0 and 1. ``metadata["parents"][j]`` lists the ``[i, k]`` pairs (variable
        ``i`` at lag ``k``) that PC1 selected for variable ``j``, strongest first. Under a
        mixed-data test, ``metadata["categorical"]`` lists the categorical columns it read, by name
        or 0-based position; with a randomized ``indep_test``, ``metadata["seed"]`` is the seed
        used. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is ``"pcmci"``.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a usable 2-D
        matrix with at least ``2 * tau_max * (n_variables + 1) + 3`` rows, the length at which every
        test keeps a degree of freedom. With ``"fisherz"`` or ``"student_t"``, also if a variable
        repeats another up to scale and shift, at the same step or up to ``2 * tau_max`` steps
        apart, before any test runs; one message names every group of copies, a lagged value as
        ``'x' at lag k``, and says to keep one column per group. With ``"chisq"`` or ``"gsq"``, also
        if ``data`` holds a value that is not a whole number; with ``"degen_gauss"`` or
        ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks; and, with a continuous
        test, if a column is declared categorical, by ``categorical`` or by its type. Also if
        ``knowledge`` names a column ``data`` lacks, contradicts itself, or requires a link without
        a lag or outside lags ``1`` to ``tau_max``.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    Notes
    -----
    **When to use.** PCMCI suits one observational multivariate time series that is stationary (its
    causal links and noise do not change over time), with no unmeasured variable that causes two
    measured ones, in which every independence follows from the graph (faithfulness), and in which
    every effect takes at least one time step [1]_. With a partial-correlation test, the default for
    continuous data, it assumes linear effects; ``"cmiknn"``, ``"gpdc"``, ``"kci"``, ``"rcot"``,
    ``"gcm"``, ``"wgcm"``, and ``"pcm"`` detect nonlinear effects at a far higher cost per test, and
    the categorical and mixed-data tests take other data. It is designed for many variables, each
    strongly dependent on its own past (autocorrelated) and with few lagged parents [1]_, and
    returns a temporal structure of lagged links only: a directed graph at each lag ``1`` to
    ``tau_max``. When effects can act within one time step, use :func:`~andrey.pcmciplus`; when
    unmeasured common causes are plausible, use :func:`~andrey.lpcmci`.

    Every test reads the lagged values as the columns of one design. ``"fisherz"`` and
    ``"student_t"`` report the partial correlation, ``"chisq"`` and ``"gsq"`` their chi-square
    statistic, ``"degen_gauss"`` and ``"cond_gauss"`` the likelihood ratio, ``"cmiknn"`` its
    estimate of the conditional mutual information, ``"gpdc"`` the squared distance correlation,
    ``"rcot"`` its statistic, ``"gcm"`` its normalized covariance of residual products, ``"wgcm"``
    and ``"pcm"`` theirs or ``0``, whichever is larger, and ``"kci"`` its statistic divided by its
    mean under independence. A candidate's strength is its smallest absolute statistic so far,
    comparable only within one test. A test registered with
    :func:`~andrey.core.ci.register_indep_test` that reports only p-values is asked one query at a
    time, its candidates are ranked by p-value, smallest first, and its statistic reads
    ``-log10(p)``. The default, ``"fisherz"``, takes the p-value from Fisher's z-transform of the
    partial correlation. ``"student_t"`` takes it from the Student-t distribution with
    ``n - |S| - 2`` degrees of freedom, for ``|S|`` conditions; its p-values differ slightly, so a
    link close to ``alpha`` can flip. The partial correlation correlates the two tested values'
    least-squares residuals given the conditions. It is computed from the correlation matrix, and
    from the residuals themselves when the conditions are collinear to working precision, as lagged
    copies of a strongly autocorrelated series can be.

    **Choosing fdr_method.** The MCI step tests ``n_variables ** 2 * tau_max`` links, each at level
    ``alpha``. With ``"none"`` about ``alpha`` of the absent links are returned by chance, so the
    extra links grow with the number of variables (see ``fdr_method``). ``"fdr_bh"`` [4]_ sorts the
    ``m`` p-values of the links and gives the one of rank ``k`` the smallest value, over its rank
    and every higher one, of ``p * m / rank``, at most ``1``; equal p-values get equal results. It
    returns fewer links than ``"none"``: the false ones, and also true ones whose p-value is not
    small enough for the number of tests. The links are thresholded at ``alpha`` on the adjusted
    p-values. The PC1 step is not corrected. The guarantee of [4]_ needs independent or positively
    dependent tests, which the MCI tests of one series are not shown to be.

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``,
    where every lag a test can condition on exists: the MCI test of a link at lag ``k`` conditions
    on the cause's parents, which reach back ``k + tau_max`` steps. PC1 runs in rounds ``p = 0, 1,
    ...``: each round tests every remaining candidate given the ``p`` strongest other candidates,
    one conditioning set per candidate, then drops those that test independent. A candidate's
    strength is the smallest absolute partial correlation it has had so far. Equal strengths are
    ordered by each variable's rank, as in :func:`~andrey.hc`; the :doc:`column-order guide
    </docs/guides/column-order>` says what the rank keeps.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so tests can use different time steps;
    ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read. A test with
    fewer rows than it needs (``|S| + 4`` rows for ``"fisherz"`` and ``"normal_scores"``,
    ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and 2 for most other tests), or with no
    degrees of freedom left, is not tested: it never separates a pair and never confirms a link, and
    ``metadata["not_tested"]`` counts the conditioning sets not tested. A link no tested query could
    decide, as one between variables never observed together, is undecided: it is left out of the
    graph, has no p-value (``NaN``), and is listed in ``metadata["undecided_pairs"]``, and no test
    conditions on a lagged parent without a tested result. A test conditions only on the candidates,
    strongest first, that leave enough rows for the test, so its rows do not mix variables no row
    observes together. A drawn link an untested conditioning set might have separated, one a test
    left a candidate out of or one with an end an undecided link joins, is listed in
    ``metadata["possibly_separable"]``: a flag of what the data could not check, not a finding that
    the link is false. A continuous column constant on a test's rows raises ``ValueError``. The
    categorical, mixed-data, and nonlinear tests delete rows the same way. Test-wise deletion is
    valid when values are missing completely at random, which means that whether a value is missing
    depends on no value in the data; the rows a test reads are then a random sample of all rows, and
    otherwise they are a selected sample that more rows do not correct.

    **Limitations.** With ``fdr_method="none"`` on simulated linear-Gaussian series and
    ``tau_max=2``, its median errors (lag-resolved SHD) stayed below the empty graph's through 20
    variables, with series of either length, and not from 40 (`0.10.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#methods-on-their-own-axes>`_); the
    ``fdr_method`` entry gives the default's errors. It assumes a stationary series, no hidden
    common causes, and, with a partial-correlation test, linear effects; ``"cmiknn"`` and ``"gpdc"``
    lift that assumption at a far higher cost per test and are not benchmarked yet. It does not test
    same-step (lag 0) links: an effect within one time step is missed and can show up as a false
    lagged link. An MCI test conditions on up to ``2 * n_variables * tau_max - 1`` lagged values, so
    the series needs many more time steps than that. The MCI step alone runs ``n_variables ** 2 *
    tau_max`` tests. Missing values are not benchmarked yet. The numbers are a reference, not a hard
    limit: they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting
       and quantifying causal associations in large nonlinear time series datasets. Science
       Advances 5(11), eaau4996.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.
    .. [4] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pcmci
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(500)
    >>> out = pcmci(np.column_stack([x, y]), tau_max=2, alpha=0.01)
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    >>> out.structure.lag(2).oriented_edges()
    []
    """
    from andrey.temporal.pcmci import design_column_name
    from andrey.temporal.pcmci import pcmci as _native

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = series_inputs(
        data,
        indep_test,
        categorical,
        knowledge,
        method="pcmci",
        tau_max=tau_max,
        # An MCI test conditions on up to 2 * n * tau_max - 1 lagged columns of the last
        # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
        min_rows=lambda n: 2 * tau_max * (n + 1) + 3,
        contemporaneous=False,
    )
    with (
        copies_named(labels, design_column_name(X.shape[1], column_namer(labels))),
        rows_tested() as rows,
    ):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=pc1_alpha,
            alpha_level=alpha,
            fdr_method=fdr_method,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            constraints=constraints,
            max_depth=max_depth,
            categorical=mask,
            seed=seed,
        )
    undecided = np.zeros((tau_max + 1, *result.p_values.shape[1:]), dtype=bool)
    undecided[1:] = np.isnan(result.p_values)
    drawn = np.zeros_like(undecided)
    drawn[1:] = result.graph
    metadata = {
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        **undecided_links(undecided, labels),
        **possibly_separable_links(rows, drawn, undecided, labels, constraints),
    }
    return with_labels(adapt_pcmci(result, metadata), labels)


@method_spec("pcmciplus")
def pcmciplus(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.01,
    fdr_method: str = "fdr_bh",
    collider_rule: str = "majority",
    possibly_separable: str = "orient",
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the lagged and same-step causal links of one multivariate time series with PCMCI+.

    PCMCI+ [1]_ extends PCMCI to links within one time step. It first selects, for each variable, a
    few candidate parents among the lagged values of all variables. It then tests every lagged
    candidate and every pair of variables at the same step, given subsets of the same-step
    neighbors plus the selected lagged parents of both ends, and orients the same-step links with
    the collider and orientation rules of the PC algorithm. Lagged links always point forward in
    time. Use it on one series whose causal effects can act within a time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag tested, at least ``1``: links at lags ``0`` to ``tau_max`` are tested.
    alpha : float, default=0.01
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``alpha``. The default is the level of the method's published experiments
        [1]_.
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the p-values of the lagged links are corrected for the number of links tested.
        ``"fdr_bh"`` replaces the p-values of lags ``1`` to ``tau_max`` with their
        Benjamini-Hochberg adjusted values [4]_ over all those links together, every lag and every
        ordered pair of variables, a variable's link to its own past included; the same-step
        p-values stay as tested. The adjusted p-value is never below the tested one. ``"none"``
        keeps each p-value as tested. The links themselves do not change with either: the skeleton
        phase decides them at ``alpha``, so ``fdr_method`` only changes ``metadata["p_values"]``,
        and under ``"fdr_bh"`` a lagged link of the graph whose adjusted p-value is above ``alpha``
        is one the correction would not keep. A link the knowledge forbids is no test: it is not
        counted and keeps p-value ``1``; a required link is counted with p-value ``0``. See Notes.
    collider_rule : {"sepsets", "conservative", "majority"}, default="majority"
        How an unshielded triple ``a - b - c`` with a same-step link ``b - c`` is judged a
        collider ``a -> b <- c``. ``"sepsets"`` calls it a collider when ``b`` is not in the one
        separating set the skeleton search found. ``"majority"`` and ``"conservative"`` test ``a``
        and ``c`` again given every subset of their same-step neighbors and look at the subsets
        that separate them: if there is at least one, ``"majority"`` calls the triple a collider
        when fewer than half hold ``b``, and ``"conservative"`` when none does; a triple neither
        rule can decide is ambiguous and orients nothing. The default, ``"majority"``, gives
        orientations that do not depend on the order of the variables [1]_.
    possibly_separable : {"orient", "ambiguous"}, default="orient"
        How the collider phase reads a same-step triple with a link listed in
        ``metadata["possibly_separable"]``, on data with missing values. ``"orient"`` reads it as
        any other triple. ``"ambiguous"`` makes it neither a collider nor a non-collider: no
        collider vote reads it, and no rule orients through it. ``"orient"`` favors same-step
        arrowheads; it gives up protection from false colliders. ``"ambiguous"`` favors no false
        collider; it gives up the same-step arrowheads. On lagged series whose modules are observed
        in different periods, so that no row bridges them, ``"ambiguous"`` took the same-step
        arrowheads per fit from 1.5-1.9 to 0; what that does to accuracy was not measured.
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for linear effects in continuous data: ``"fisherz"`` applies
        Fisher's z-transform and the normal distribution, ``"student_t"`` the Student-t
        distribution. ``"normal_scores"`` applies ``"fisherz"`` to the normal scores of each column
        of the lagged values, for monotone effects. ``"chisq"`` (Pearson's chi-square) and ``"gsq"``
        (G-square, the likelihood ratio) test categorical series, coded by whole numbers.
        ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test
        mixed series, continuous and categorical columns, by a likelihood ratio. Seven more tests
        detect nonlinear dependence in continuous data, at a far higher cost per test: ``"cmiknn"``,
        a nearest-neighbor conditional mutual information test, and ``"gpdc"``, Gaussian-process
        regression with a distance correlation test, the nonlinear tests the method was published
        with; ``"kci"``, the kernel test; ``"rcot"``, its random-feature approximation; and
        ``"gcm"``, ``"wgcm"``, and ``"pcm"``, the generalized, weighted generalized, and projected
        covariance measures of regression residuals. ``None`` picks from the declared column types:
        ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are, else
        ``"fisherz"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values. ``metadata["indep_test"]`` names
        the test used, including the choice for ``None``.
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
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. Every lagged value of a column has its type.
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
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test of the lagged
        skeleton (phase 1, PC1 as in :func:`~andrey.pcmci`), at least ``0``; ``None`` sets no
        limit. A smaller cap runs fewer tests and can keep lagged parents that a larger
        conditioning set would remove. It does not cap the skeleton and collider phases, which
        add the lagged parents of both ends to every conditioning set; ``max_p_global`` caps those.
    max_p_global : int or None, default=None
        The most conditions a test of the skeleton and collider phases takes beyond the default
        conditions, at least ``0``: every such test conditions on the lagged parents of both ends,
        the default conditions, and on a subset of at most ``max_p_global`` same-step neighbors; the
        cap also bounds the subsets the collider vote reads. Unlike ``max_depth``, it does not count
        the default conditions, so a test can condition on more than ``max_p_global`` values.
        ``None`` sets no limit. The number of tests grows exponentially with the same-step links at
        one variable; a cap bounds it and can keep links that a larger conditioning set would
        remove.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables: required and forbidden links, the pairs allowed to be linked, and temporal
        tiers, all honored. A link ``(cause, effect, lag)`` is the link from ``cause`` at time
        ``t - lag`` to ``effect`` at time ``t``, lag ``0`` being the same step; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``0`` to ``tau_max``. The lagged and skeleton phases test neither: a required link stays
        and a forbidden one is never linked. The collider phase tests a pair kept apart for the
        subsets that separate it, as it does any unlinked pair; with ``collider_rule="sepsets"``,
        which reads the skeleton phase's separating set, a triple over such a pair is ambiguous.
        At the same step, a pair forbidden in both directions is not linked, and a link the
        knowledge allows in one direction only is directed that way before the colliders, which
        never reverse it. Tiers forbid every link from a later tier into an earlier one, and a
        pair left out of ``allowed`` has no link at any lag; a variable's links to its own past
        stay allowed. The rule phase applies R1 to R3 only, so a same-step link that Meek's rule R4
        would direct from the knowledge can stay undirected.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``0`` to ``tau_max``.
        ``structure.lag(0)`` is a ``cpdag`` of the same-step links: ``i -> j`` is oriented, and
        ``i -- j`` is a link whose direction the data cannot tell or whose orientations conflict;
        ``metadata["conflicts"]`` lists the latter as ``[i, 0, j]`` links (variable ``i`` at lag
        ``0`` and ``j``), ``i < j``. ``structure.lag(k)``, ``k >= 1``, is a ``digraph`` whose edge
        ``i -> j`` is the link from variable ``i`` at time ``t - k`` to variable ``j`` at time
        ``t``; a self-loop is a variable's dependence on its own past. The structure has no
        ``lag_weights``. ``metadata["p_values"][k][i][j]`` is the largest p-value among the
        skeleton-phase tests of the link between variable ``i`` at ``t - k`` and variable ``j`` at
        ``t`` (the first step's tests for a lagged link that step removed; for ``k >= 1`` its
        Benjamini-Hochberg adjusted value under ``fdr_method="fdr_bh"``), and
        ``metadata["statistics"][k][i][j]`` the statistic of that test (see Notes); lag ``0`` is
        symmetric with ``1`` and ``0`` on the diagonal. A link the knowledge forbids, never tested,
        reads p-value 1 and statistic 0, and a required one reads 0 and 1. Under a mixed-data test,
        ``metadata["categorical"]`` lists the categorical columns it read; with a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["parents"][j]`` lists the
        ``[i, k]`` pairs (variable ``i`` at lag ``k``) the first step selected for variable ``j``,
        strongest first, and ``metadata["ambiguous_triples"]`` the ``[i, k, m, j]`` triples
        (variable ``i`` at lag ``k``, then ``m`` and ``j`` at lag ``0``) the collider rule left
        undecided. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is ``"pcmciplus"``.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a usable 2-D
        matrix with at least ``2 * tau_max * (n_variables + 1) + n_variables + 1`` rows, the length
        at which every test keeps a degree of freedom. With ``"fisherz"`` or ``"student_t"``, also
        if a variable repeats another up to scale and shift, at the same step or up to
        ``2 * tau_max`` steps apart, before any test runs; one message names every group of copies,
        a lagged value as ``'x' at lag k``, and says to keep one column per group. With ``"chisq"``
        or ``"gsq"``, also if ``data`` holds a value that is not a whole number; with
        ``"degen_gauss"`` or ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks;
        and, with a continuous test, if a column is declared categorical, by ``categorical`` or by
        its type. Also if ``knowledge`` names a column ``data`` lacks, contradicts itself, requires
        a link without a lag or beyond ``tau_max``, or directs same-step links into a directed cycle
        over the links the skeleton phase kept.

    Warns
    -----
    DataTypeWarning
        When a mixed-data test finds no declared categorical column and so runs as the
        linear-Gaussian form, or when a setting reads columns that look like category codes as
        numbers; each message warns once per process.

    Notes
    -----
    **When to use.** PCMCI+ suits one observational multivariate time series that is stationary (its
    causal links and noise do not change over time), with no unmeasured variable that causes two
    measured ones, and in which every independence follows from the graph (faithfulness); its
    effects can act within one time step as well as across steps, and the same-step effects form no
    cycle [1]_. Its tests, and the data each suits, are those of :func:`~andrey.pcmci`. It is
    designed for many variables, each strongly dependent on its own past (autocorrelated) [1]_; its
    tests grow exponentially with the same-step neighbors of one variable, so it suits sparse
    same-step links. It returns a temporal structure over lags ``0`` to ``tau_max``: a CPDAG of the
    same-step links and a directed graph of the lagged links at each lag ``1`` to ``tau_max``. When
    unmeasured common causes are plausible, use :func:`~andrey.lpcmci`; when the noise is
    non-Gaussian and every same-step link should be oriented, use :func:`~andrey.varma_lingam`.

    The method runs in four phases. The lagged phase selects the lagged parents of each variable
    as PCMCI's PC1 step does, with equal strengths ordered by each variable's rank. The skeleton
    phase starts from those lagged links and from a link between every two variables at the same
    step. Round ``p = 0, 1, ...`` tests each linked pair given each subset of ``p`` same-step
    neighbors of the later end, with the lagged parents of both ends added, and removes the link
    at the first p-value above ``alpha``. A same-step pair is tested from both ends, first given
    the neighbors of its end of higher rank, and the neighbor sets stay fixed during a round and
    are sorted by strength, equal strengths by rank. Every test reads the variables in rank order,
    so a link's tests, its reported p-value and statistic (those of the largest of its tests), and
    the separating set it stores do not depend on the order of the variables. The collider phase
    orients the unshielded triples by ``collider_rule``, and the rule phase applies Meek's rules R1
    to R3 to the same-step links until none applies. Each collider or rule pass collects its
    orientations before applying them, and marks a link that gets both orientations as a conflict,
    so the orientations do not depend on the order of the variables either.

    **Choosing fdr_method.** The skeleton phase tests each lagged link at level ``alpha`` and keeps
    the links whose tests all stay at or below it, so absent links are kept by chance, more of them
    as the number of variables grows, whatever ``fdr_method`` is. ``"fdr_bh"`` [4]_ sorts the ``m``
    lagged-link p-values and gives the one of rank ``k`` the smallest value, over its rank and every
    higher one, of ``p * m / rank``, at most ``1``; equal p-values get equal results. The reported
    p-values then show which lagged links of the graph the correction would keep, where ``"none"``
    reports them as tested. The graph stays as the skeleton phase decided it. The guarantee of [4]_
    needs independent or positively dependent tests, which these tests are not shown to be.

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``.
    Every test reads the lagged values as the columns of one design. ``"fisherz"`` and
    ``"student_t"`` report the partial correlation, ``"chisq"`` and ``"gsq"`` their chi-square
    statistic, ``"degen_gauss"`` and ``"cond_gauss"`` the likelihood ratio, ``"cmiknn"`` its
    estimate of the conditional mutual information, ``"gpdc"`` the squared distance correlation,
    ``"rcot"`` its statistic, ``"gcm"`` its normalized covariance of residual products, ``"wgcm"``
    and ``"pcm"`` theirs or ``0``, whichever is larger, and ``"kci"`` its statistic divided by its
    mean under independence. A candidate's strength is its smallest absolute statistic so far,
    comparable only within one test. A test registered with
    :func:`~andrey.core.ci.register_indep_test` that reports only p-values is asked one query at a
    time, its candidates are ranked by p-value, smallest first, and its statistic reads
    ``-log10(p)``. The default test, ``"fisherz"``, takes the p-value from Fisher's z-transform of
    the partial correlation, and ``"student_t"`` from the Student-t distribution with
    ``n - |S| - 2`` degrees of freedom, for ``|S|`` conditions. The partial correlation correlates
    the two tested values' least-squares residuals given the conditions. It is computed from the
    correlation matrix, and from the residuals themselves when the conditions are collinear to
    working precision, as lagged copies of a strongly autocorrelated series can be.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so tests can use different time steps;
    ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read. A test with
    fewer rows than it needs (``|S| + 4`` rows for ``"fisherz"`` and ``"normal_scores"``,
    ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and 2 for most other tests), or with no
    degrees of freedom left, is not tested: it never separates a pair and never confirms a link, and
    ``metadata["not_tested"]`` counts the conditioning sets not tested. A link no tested query could
    decide, as one between variables never observed together, is undecided: it is left out of the
    graph, has no p-value (``NaN``), and is listed in ``metadata["undecided_pairs"]``, and no test
    conditions on a lagged parent without a tested result. A test conditions only on the candidates,
    strongest first, that leave enough rows for the test, so its rows do not mix variables no row
    observes together. A drawn link an untested conditioning set might have separated, one a test
    left a candidate out of or one with an end an undecided link joins, is listed in
    ``metadata["possibly_separable"]``: a flag of what the data could not check, not a finding that
    the link is false; under ``possibly_separable="ambiguous"``, no collider vote reads a triple
    with a flagged side. A continuous column constant on a test's rows raises ``ValueError``. The
    categorical, mixed-data, and nonlinear tests delete rows the same way. Test-wise deletion is
    valid when values are missing completely at random, which means that whether a value is missing
    depends on no value in the data; the rows a test reads are then a random sample of all rows, and
    otherwise they are a selected sample that more rows do not correct.

    **Limitations.** On simulated linear-Gaussian series with lagged and same-step links and
    ``tau_max=5``, its median errors (endpoint SHD) stayed below the empty graph's at every size
    through 160 variables, the largest the data generator could draw (`0.10.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#methods-on-their-own-axes>`_). It assumes a
    stationary series, no hidden common causes, and, with a partial-correlation test, linear
    effects; ``"cmiknn"`` and ``"gpdc"`` lift that assumption at a far higher cost per test and are
    not benchmarked yet. The skeleton and collider phases test every subset of a variable's
    same-step neighbors, so the number of tests grows exponentially with the same-step links at one
    variable. A test conditions on up to ``2 * n_variables * tau_max + n_variables - 2`` values, so
    the series needs many more time steps than that. Missing values are not benchmarked yet. The
    numbers are a reference, not a hard limit: they change with the data generator, the density, the
    method's settings, and the machine.

    References
    ----------
    .. [1] Runge, J. (2020). Discovering contemporaneous and lagged causal relations in
       autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on
       Uncertainty in Artificial Intelligence (UAI), PMLR 124, 1388-1397.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.
    .. [4] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import pcmciplus
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = 0.8 * x + rng.standard_normal(500)
    >>> z = np.r_[0.0, 0.8 * y[:-1]] + rng.standard_normal(500)
    >>> out = pcmciplus(np.column_stack([x, y, z]))
    >>> out.structure.lag(0).oriented_edges()
    [(0, 1, 'undirected')]
    >>> out.structure.lag(1).oriented_edges()
    [(1, 2, 'directed')]
    """
    from andrey.temporal.pcmci import design_column_name
    from andrey.temporal.pcmciplus import pcmciplus as _native

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = series_inputs(
        data,
        indep_test,
        categorical,
        knowledge,
        method="pcmciplus",
        tau_max=tau_max,
        # A test conditions on up to 2 * n * tau_max + n - 2 columns of the last
        # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
        min_rows=lambda n: 2 * tau_max * (n + 1) + n + 1,
    )
    with (
        copies_named(labels, design_column_name(X.shape[1], column_namer(labels))),
        rows_tested() as rows,
    ):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=alpha,
            fdr_method=fdr_method,
            collider_rule=collider_rule,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            constraints=constraints,
            max_depth=max_depth,
            max_p_global=max_p_global,
            categorical=mask,
            seed=seed,
        )
    metadata = {
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        **undecided_links(result.undecided, labels),
        **possibly_separable_links(rows, result.adjacency, result.undecided, labels, constraints),
    }
    return with_labels(adapt_pcmciplus(result, metadata), labels)


@method_spec("jpcmciplus")
def jpcmciplus(
    data: Sequence[npt.ArrayLike],
    *,
    time_contexts: Sequence[str | int] | None = None,
    space_contexts: Sequence[str | int] | None = None,
    time_dummy: bool | None = None,
    tau_max: int = 1,
    alpha: float = 0.01,
    fdr_method: str = "fdr_bh",
    collider_rule: str = "majority",
    indep_test: str | None = None,
    seed: int | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the links of time series from several datasets with shared contexts with J-PCMCI+.

    J-PCMCI+ [1]_ extends PCMCI+ [2]_ to datasets of the same variables, such as sites, subjects,
    or regimes, whose differences come from context variables. A time context varies over time and
    is the same in every dataset at each step; a space context is fixed within a dataset. A context
    can be measured or not: a one-hot dummy for the dataset and one for the time step stand in for
    the contexts nobody measured, so that a hidden context does not link the variables it drives.
    The method finds the links from the contexts into the system variables, then the links between
    the system variables, and returns the graph over the system variables.

    Parameters
    ----------
    data : sequence of array-like, each of shape (n_samples, n_variables)
        Two or more datasets over the same columns, one row per time step; their lengths may
        differ. Each is coerced to ``float64``. DataFrames must name their columns alike; the names
        of the system variables become the structure's ``labels``. The columns that
        ``time_contexts`` and ``space_contexts`` do not list are the system variables.
    time_contexts : sequence of str or int, or None, default=None
        The columns that hold time contexts: column labels (strings) or 0-based positions. Each
        must hold the same values in every dataset at each step, counted from each dataset's first
        row. ``None`` lists none.
    space_contexts : sequence of str or int, or None, default=None
        The columns that hold space contexts, listed as in ``time_contexts``. Each must be constant
        within a dataset. ``None`` lists none.
    time_dummy : bool or None, default=None
        Whether the fit adds the dummy of the time step, which stands in for hidden time contexts.
        It needs datasets of one length, aligned in time, with more than one step after the first
        ``2 * tau_max``. ``None``, the default, adds it when the datasets allow it; ``True``
        requires it and raises ``ValueError`` when they do not; ``False`` leaves it out, for
        datasets that are not aligned in time or to save its one column per step. The dummy of the
        dataset is always used. ``metadata["time_dummy"]`` says whether the fit used it. The dummy
        favors a graph without the links a hidden time context adds; it gives up time and memory, a
        column per step in the conditions of every test that takes it. Leaving it out favors long
        series and more system variables; it gives up that stand-in for hidden time contexts. At 5
        system variables and 1,600 steps, a fit took 7 s with it and 0.05 s without; on one core
        with 10 datasets of 100 steps, a fit took 11.5 s with it and 1.5 s without at 15 system
        variables, and at 20 it ran out of 12 GB with it and took 47 s without.
    tau_max : int, default=1
        The largest lag tested, at least ``1``: links at lags ``0`` to ``tau_max`` are tested.
    alpha : float, default=0.01
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``alpha``. The default is :func:`~andrey.pcmciplus`'s; the published
        experiments of the method use ``0.05`` [1]_.
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the p-values of the lagged links between system variables are corrected for the number
        of links tested. ``"fdr_bh"`` replaces them with their Benjamini-Hochberg adjusted values
        [3]_ over all those links together, as in :func:`~andrey.pcmciplus`, and ``"none"`` keeps
        them as tested. The links do not change: the skeleton passes decide them at ``alpha``, so
        ``fdr_method`` only changes ``metadata["p_values"]``.
    collider_rule : {"sepsets", "conservative", "majority"}, default="majority"
        How an unshielded triple ``a - b - c`` with a same-step collider candidate ``b - c`` between
        system variables is judged, as in :func:`~andrey.pcmciplus`; ``a`` can be a context or a
        dummy. The default, ``"majority"``, gives orientations that do not depend on the order of
        the variables.
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value, a test of continuous data, as in
        :func:`~andrey.pcmciplus`: ``"fisherz"`` and ``"student_t"`` test partial correlation, and
        ``"kci"``, ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` detect
        nonlinear dependence at a far higher cost per test. The dummies enter a test as numeric
        columns, so a categorical or mixed-data test raises ``ValueError``. ``"normal_scores"`` is
        not offered: ranked over every dataset's rows, a column's normal scores mix the datasets'
        distributions. ``None`` picks ``"fisherz"``. ``metadata["indep_test"]`` names the test the
        fit used.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of ``indep_test``, which
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_depth : int or None, default=None
        The maximum conditioning-set size of every conditional-independence test of the lagged
        phase (PC1 as in :func:`~andrey.pcmci`), at least ``0``; ``None`` sets no limit. It does
        not cap the later passes, which add the parents of both ends to every conditioning set;
        ``max_p_global`` caps those.
    max_p_global : int or None, default=None
        The most conditions a test of the skeleton passes and the collider phase takes beyond the
        default conditions, at least ``0``: every such test conditions on the parents of both
        ends, the default conditions, and on a subset of at most ``max_p_global`` same-step
        neighbors; the cap also bounds the subsets the collider vote reads. ``None`` sets no limit.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, over the system variables, read as
        links between lagged variables as in :func:`~andrey.pcmciplus`: required and forbidden
        links, the pairs allowed to be linked, and temporal tiers, all honored. A position counts
        the system variables. The contexts bring their own knowledge: no system variable causes a
        context, and no two contexts are linked.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` over the system variables, as
        :func:`~andrey.pcmciplus` returns it: ``structure.lag(0)`` is a ``cpdag`` and
        ``structure.lag(k)``, ``k >= 1``, a ``digraph``. ``metadata["p_values"]``,
        ``metadata["statistics"]``, ``metadata["parents"]``, ``metadata["conflicts"]``, and
        ``metadata["ambiguous_triples"]`` read as there, over the system variables by position.
        ``metadata["context_links"]`` lists the links from contexts into system variables as
        ``[context, lag, variable]``, a space context at lag ``0``, with a context by name or
        position in ``data`` and a variable by name or position among the system variables.
        ``metadata["index_edges"]`` lists the links from the dummy of the dataset as ``[variable,
        "directed"]`` pairs: the variables that change with the dataset in a way the measured
        contexts do not explain. ``metadata["time_index_edges"]`` does the same for the dummy of
        the time step, and ``metadata["time_dummy"]`` says whether the fit used it. A dummy link is
        not a causal link. There is no ``metadata["changing_variables"]``: the dummy links stand in
        for the hidden contexts, and the links of the measured ones are in
        ``metadata["context_links"]``. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is
        ``"jpcmciplus"``. With a randomized ``indep_test``, ``metadata["seed"]`` is the seed used.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``time_contexts`` or
        ``space_contexts`` is not a list of column names and positions, or ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` holds fewer than two
        datasets, a dataset is not a 2-D table, the datasets differ in their columns, or a dataset
        has at most ``2 * tau_max`` rows; if ``time_dummy=True`` and the datasets differ in length
        or have one step after the first ``2 * tau_max``; if a test conditions on so many columns of
        lagged values, contexts, and dummy levels that the rows leave it no degree of freedom; if a
        column is both a time and a space context or a context names a column ``data`` lacks, or
        fewer than two system variables remain; if a time context differs between datasets at some
        step, a space context varies within a dataset, or a space context repeats another up to
        scale and shift, as any two do over two datasets; or if ``data`` holds ``NaN``. With
        ``"fisherz"`` or ``"student_t"``, also if a variable repeats another up to scale and shift,
        at the same step or up to ``2 * tau_max`` steps apart, or a test finds a system variable in
        the span of its conditions, as with one that sums others or a pulse every dataset shares
        that is not a time context. Also if ``knowledge`` names a column that is not a system
        variable, such as a context, contradicts itself, requires a link without a lag or beyond
        ``tau_max``, or directs same-step links into a directed cycle over the links the fit kept.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** J-PCMCI+ suits several datasets of the same variables, each a stationary time
    series, that differ through context variables: no system variable causes a context, no hidden
    variable causes two system variables except a context, and no hidden context stands between a
    measured context and a system variable or causes both [1]_. Every context must be a function of
    the dataset or of the time step that does not tell them all apart. The method suits few system
    variables: its context and dummy passes keep every same-step link between system variables, so
    their tests grow exponentially with those variables, and the dummies add a column per dataset
    and per step. It returns the temporal structure of :func:`~andrey.pcmciplus` over the system
    variables and lists the context and dummy links in ``metadata``. For one series, use
    :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci` when hidden common causes are not contexts;
    for i.i.d. rows from several environments, :func:`~andrey.cdnod` or :func:`~andrey.fci_jci`.

    The method runs PCMCI+'s lagged phase on the system variables and time contexts, then the
    skeleton in three passes: the links from the contexts into the system, given the lagged parents;
    the links from the dummies into the system, given those and the context parents; and the links
    between system variables, given those and the dummy parents. Links from contexts and dummies
    point into the system, and PCMCI+'s collider and rule phases use them to orient the same-step
    links between system variables.

    Every test reads the datasets' lagged rows stacked, so a test reads ``M * (T - 2 * tau_max)``
    rows for ``M`` datasets of ``T`` steps, and no lag window crosses from one dataset into the
    next. A dummy has one level per dataset or per step. A test of a dummy tests each level
    against the rest and multiplies the smallest p-value by the number of levels, Bonferroni's
    correction, so a dummy link needs one level that stands out; two levels are one test.
    Conditioning on a dummy takes all its levels but one, which centers each variable within each
    dataset or step, and a context is fixed given its dummy, so it leaves the conditions.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the stacked design reads every
    step. No method that accepts missing values reads several datasets with their contexts, as
    :func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, and :func:`~andrey.lpcmci` read one series, so
    drop the datasets that hold one.

    **Limitations.** On the linear model of its authors' experiment script [1]_, 5 system variables
    and 4 contexts, all or half of them observed, its mean errors stayed below the empty graph's in
    every setting (`0.15.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.15.0.html#j-pcmci>`_). It assumes faithfulness, so a
    missed dummy link can leave the links a hidden context adds. A dummy link needs one level that
    stands out: a dataset effect far larger than the noise can hide behind the lagged values, and
    the time dummy misses a smooth effect spread over the steps, which leaves false links; list such
    a context if it is known. The lagged phase first tests each candidate without conditions, so an
    autocorrelated target can hide a context link from it for good, and with few steps against many
    datasets false positives stay high as the datasets grow [1]_. A time context that the lags of
    the time contexts determine, such as a sine and cosine of one period, or a sine or a quadratic
    trend alone from ``tau_max=2``, cannot be told apart from them: only the variables its context
    links reach can be trusted, not the context or the lag they name. It tests and reports no link
    between contexts [1]_. Under ``"majority"`` and ``"conservative"``, hidden contexts leave most
    same-step links undirected, as a context cannot orient a link into a variable that has the
    context's dummy as a parent. The cost grows fastest with the system variables (see
    ``time_dummy``). The numbers are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Gunther, W., Ninad, U., and Runge, J. (2023). Causal discovery for time series from
       multiple datasets with latent contexts. Proceedings of the 39th Conference on Uncertainty in
       Artificial Intelligence (UAI), PMLR 216, 766-776.
    .. [2] Runge, J. (2020). Discovering contemporaneous and lagged causal relations in
       autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on
       Uncertainty in Artificial Intelligence (UAI), PMLR 124, 1388-1397.
    .. [3] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import jpcmciplus
    >>> rng = np.random.default_rng(0)
    >>> def dataset(level):
    ...     x = level + rng.standard_normal(200)
    ...     y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(200)
    ...     return np.column_stack([x, y])
    >>> out = jpcmciplus([dataset(level) for level in (0, 3, 6, 9)], time_dummy=False)
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    >>> [variable for variable, _ in out.metadata["index_edges"]]
    [0]
    """
    from andrey.temporal.jpcmciplus import jpcmciplus as _native
    from andrey.temporal.pcmci import design_column_name

    assert indep_test is not None  # method_spec picks it from the column types
    inputs = context_inputs(data, time_contexts, space_contexts, tau_max, time_dummy)
    check_context_knowledge(knowledge, inputs)
    constraints = resolve_lagged(
        knowledge, len(inputs.system), inputs.system_labels, tau_max, method="jpcmciplus"
    )
    with copies_named(
        inputs.series_labels,
        design_column_name(inputs.series[0].shape[1], column_namer(inputs.series_labels)),
    ):
        result = _native(
            inputs.series,
            inputs.spaces,
            len(inputs.system),
            time_dummy=inputs.time_dummy,
            tau_max=tau_max,
            pc_alpha=alpha,
            fdr_method=fdr_method,
            collider_rule=collider_rule,
            indep_test=indep_test,
            constraints=constraints,
            max_depth=max_depth,
            max_p_global=max_p_global,
            seed=seed,
        )
    context, variable = label_of(inputs.labels), label_of(inputs.system_labels)
    metadata = {
        "context_links": [
            [context(inputs.contexts[k]), lag, variable(j)] for k, lag, j in result.context_links
        ],
        "index_edges": [[variable(j), "directed"] for j in result.space_links],
        "time_index_edges": [[variable(j), "directed"] for j in result.time_links],
        "time_dummy": inputs.time_dummy,
        **undecided_links(result.system.undecided, inputs.system_labels),
    }
    return with_labels(adapt_pcmciplus(result.system, metadata), inputs.system_labels)


@method_spec("lpcmci")
def lpcmci(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.05,
    n_preliminary_iterations: int = 1,
    indep_test: str | None = None,
    degrees_of_freedom: str = "mean",
    categorical: Sequence[str | int] | None = None,
    missing: str | None = None,
    seed: int | None = None,
    max_p_global: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a time-series partial ancestral graph that allows hidden common causes with LPCMCI.

    LPCMCI [1]_ finds the lagged and same-step links of one multivariate time series when some
    variables that drive several observed ones are not measured. Like FCI it returns a partial
    ancestral graph: an arrowhead says an end is not a cause of the other, a tail that it is, and a
    circle leaves it open, so a link with arrowheads at both ends is a hidden common cause. It
    orients links while it removes them and conditions every test on the parents found so far,
    which keeps true links in strongly autocorrelated series. Use it on one series when an
    unmeasured variable may confound the measured ones.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag, at least ``1``: the graph covers the time steps ``t - tau_max`` to ``t``,
        and links at lags ``0`` to ``tau_max`` are tested.
    alpha : float, default=0.05
        Significance level of every test, in ``(0, 1)``: a link is removed when a test gives it a
        p-value above ``alpha``.
    n_preliminary_iterations : int, default=1
        How many preliminary rounds run before the final one, at least ``0``. Each round learns
        parents that the next one conditions on from its first test.
    indep_test : str or None, default=None
        The conditional-independence test behind every p-value. ``"fisherz"`` and ``"student_t"``
        test partial correlation, for linear effects in continuous data: ``"fisherz"`` applies
        Fisher's z-transform and the normal distribution, ``"student_t"`` the Student-t
        distribution. ``"normal_scores"`` applies ``"fisherz"`` to the normal scores of each column
        of the lagged values, for monotone effects. ``"chisq"`` (Pearson's chi-square) and ``"gsq"``
        (G-square, the likelihood ratio) test categorical series, coded by whole numbers.
        ``"degen_gauss"`` (degenerate Gaussian) and ``"cond_gauss"`` (conditional Gaussian) test
        mixed series, continuous and categorical columns, by a likelihood ratio. Seven more tests
        detect nonlinear dependence in continuous data, at a far higher cost per test: ``"cmiknn"``,
        a nearest-neighbor conditional mutual information test, and ``"gpdc"``, Gaussian-process
        regression with a distance correlation test, the nonlinear tests the method was published
        with; ``"kci"``, the kernel test; ``"rcot"``, its random-feature approximation; and
        ``"gcm"``, ``"wgcm"``, and ``"pcm"``, the generalized, weighted generalized, and projected
        covariance measures of regression residuals. ``None`` picks from the declared column types:
        ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are, else
        ``"fisherz"``. A column is declared categorical by ``categorical`` or by a DataFrame's
        ``category``, ``bool``, or text type, never by its values. ``metadata["indep_test"]`` names
        the test used, including the choice for ``None``.
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
        The categorical columns of ``data``, for a mixed-data test: column labels (strings) or
        0-based positions. ``None`` takes a DataFrame's ``category``, ``bool``, and text columns,
        never a column by its values. ``[]`` marks none. A categorical test reads every column as
        categorical, and a continuous test raises ``ValueError`` on a declared column, named here or
        by its type. Every lagged value of a column has its type.
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
        only ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``, ``"wgcm"``, and ``"pcm"`` make; the
        other tests ignore it. With those tests, ``metadata["seed"]`` records the seed used.
    max_p_global : int or None, default=None
        The most conditions a test takes beyond the default conditions, at least ``0``: every test
        of the removal phases conditions on the parents of both ends, the default conditions, and on
        a searched subset of at most ``max_p_global`` neighbors; the cap also bounds the subsets the
        majority vote reads. Unlike ``max_depth`` on :func:`~andrey.pc` or :func:`~andrey.pcmci`, it
        does not count the default conditions, so a test can condition on more than ``max_p_global``
        nodes. ``None`` sets no limit, as in the paper. The number of tests grows exponentially with
        the links at one node, so on many strongly related series an uncapped run may not finish. A
        cap bounds the number of tests, though a capped run can still take long, and can keep links
        that a larger conditioning set would remove.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables: required and forbidden links, the pairs allowed to be linked, and temporal
        tiers, all honored. A link ``(cause, effect, lag)`` is the link from ``cause`` at time
        ``t - lag`` to ``effect`` at time ``t``, lag ``0`` being the same step; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``0`` to ``tau_max``. A forbidden lagged link, and a same-step pair forbidden in both
        directions, is never linked. A same-step link forbidden in one direction carries an
        arrowhead at the cause, as in :func:`~andrey.fci`: the cause is not an ancestor of the
        effect. A required link is ``-->`` from the start and no test or rule removes it. No rule
        changes a mark the knowledge sets. Tiers forbid every link from a later tier into an
        earlier one, and a pair left out of ``allowed`` has no link at any lag; a variable's links
        to its own past stay allowed. ``required_paths``, ``forbidden_paths``, and ``confounded``
        also name each link with its lag: on a link the search keeps, a required path makes it
        ``-->`` from the cause and a forbidden path puts an arrowhead at the cause, while the tests
        can still remove it; no test removes a ``confounded`` link. A claim the PAG over the time
        window does not meet is dropped, as for :func:`~andrey.fci`: the fit warns with
        :class:`~andrey.KnowledgeWarning`, and ``metadata["unmet_knowledge"]`` lists each dropped
        claim as ``[part, a, b, lag]``.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``0`` to ``tau_max``.
        ``structure.lag(0)`` is a ``pag`` of the same-step links: ``i -> j``, ``i o-> j``,
        ``i <-> j`` (a hidden common cause), and ``i o-o j``. ``structure.lag(k)``, ``k >= 1``, is a
        ``digraph`` of the lagged links whose earlier end is a tail, the definite causes: its edge
        ``i -> j`` says variable ``i`` at time ``t - k`` causes variable ``j`` at time ``t``, and a
        self-loop is a variable's effect on its own future. The structure has no ``lag_weights``.
        ``metadata["links"][k][i][j]`` holds every link between variable ``i`` at ``t - k`` and
        variable ``j`` at ``t`` as three characters, ``""`` when there is none: the mark at ``i``
        (``-`` tail, ``<`` arrowhead, ``o`` circle, ``x`` conflict), ``-``, and the mark at ``j``
        (``>``, ``-``, ``o``, or ``x``), such as ``"o->"``, ``"<->"``, or ``"-->"``; lag ``0`` lists
        both orders. ``metadata["conflicts"]`` lists the ``[i, k, j]`` links with a conflict mark,
        which ``structure.lag(0)`` shows as a circle and ``structure.lag(k)`` leaves out.
        ``metadata["p_values"][k][i][j]`` is the largest p-value of the final round's tests of that
        pair, and ``metadata["statistics"][k][i][j]`` the statistic of that test (see Notes); lag
        ``0`` is symmetric with ``1`` and ``0`` on the diagonal and on pairs no test ran on, and a
        required link, never tested, reads ``0`` and ``1``. Under a mixed-data test,
        ``metadata["categorical"]`` lists the categorical columns it read; with a randomized
        ``indep_test``, ``metadata["seed"]`` is the seed used. ``metadata["parents"][j]`` lists the
        ``[i, k]`` pairs (variable ``i`` at lag ``k``) the last preliminary round found to be
        parents of variable ``j``. ``ordering`` is ``None``, and ``metadata["algorithm"]`` is
        ``"lpcmci"``. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a usable 2-D
        matrix with at least ``n_variables * (tau_max + 1) + 2 * tau_max + 1`` rows, the length at
        which every test keeps a degree of freedom. With ``"fisherz"`` or ``"student_t"``, also if a
        variable repeats another up to scale and shift, at the same step or up to ``2 * tau_max``
        steps apart, before any test runs; one message names every group of copies, a lagged value
        as ``'x' at lag k``, and says to keep one column per group. With ``"chisq"`` or ``"gsq"``,
        also if ``data`` holds a value that is not a whole number; with ``"degen_gauss"`` or
        ``"cond_gauss"``, if ``categorical`` names a column ``data`` lacks; and, with a continuous
        test, if a column is declared categorical, by ``categorical`` or by its type. Also if
        ``knowledge`` names a column ``data`` lacks, contradicts itself, or requires a link without
        a lag or beyond ``tau_max``.

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
    **When to use.** LPCMCI suits one observational multivariate time series that is stationary (its
    causal links and noise do not change over time), in which every independence follows from the
    graph (faithfulness), and in which unmeasured variables may cause several measured ones; it
    assumes no selection bias, that is, no time step is kept or dropped by the values of the
    variables [1]_. Its effects can act within one time step as well as across steps. Its tests, and
    the data each suits, are those of :func:`~andrey.pcmci`. It is designed for series in which each
    variable strongly depends on its own past (autocorrelated) [1]_, and suits few links at each
    variable, since its tests grow exponentially with them; it returns a temporal structure over
    lags ``0`` to ``tau_max``: a PAG of the same-step links, a directed graph of the definite lagged
    causes at each lag ``1`` to ``tau_max``, and the marks of every link in ``metadata["links"]``.
    When every common cause is measured, use :func:`~andrey.pcmciplus`, or :func:`~andrey.pcmci`
    when every effect also takes at least one time step.

    The graph starts complete, with every lagged link ``o->`` (time order rules out the later end
    as a cause of the earlier one) and every same-step link ``o-o``. A round removes links in
    steps ``p = 0, 1, ...``: it tests a link given ``p`` of the neighbors of its later end that are
    not known non-causes of it, plus the parents of both ends, and tests the autodependency links
    first, then the cross links lag by lag. A same-step link is also tested from its other end.
    After each step that removes a link, orientation rules mark new tails and arrowheads on the
    lagged links and the round restarts at ``p = 0``, as in the paper, so that the separating sets
    found later stay weakly minimal. In these passes the rules R9' and R10' orient lagged links
    only, but their paths may run through same-step links, directed or not, as the paper's rules
    allow. Once nothing is left to test, the full set of rules runs, including the collider
    rule, which first tests the two links of a triple again given each stored separating set of
    the outer pair plus the parents of their ends. A preliminary round keeps only the parents it
    found and starts over from the complete graph. The final round also tests the links that may
    join two variables neither of which causes the other, given subsets of a wider neighborhood.
    Whether a node lies outside the separating set of two others is a majority vote over the
    distinct separating sets stored and found among the neighbors of the two. A mark that rules
    orient both ways is a conflict. The candidates of a step are ordered by statistics frozen when
    its group of links starts, so the order in which a group's links are tested changes nothing.
    Equal statistics are ordered by each variable's rank, as in :func:`~andrey.hc`; the
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps. The collider
    rule stores every separating set its tests find, each made weakly minimal by dropping first the
    node whose removal leaves the largest p-value, so the stored sets do not depend on the order of
    the variables.

    Without missing values, all tests use the same time steps, the last ``n_samples - 2 * tau_max``,
    and every node a test conditions on lies within ``tau_max`` steps of the later end. Every test
    reads the lagged values as the columns of one design. ``"fisherz"`` and ``"student_t"`` report
    the partial correlation, ``"chisq"`` and ``"gsq"`` their chi-square statistic, ``"degen_gauss"``
    and ``"cond_gauss"`` the likelihood ratio, ``"cmiknn"`` its estimate of the conditional mutual
    information, ``"gpdc"`` the squared distance correlation, ``"rcot"`` its statistic, ``"gcm"``
    its normalized covariance of residual products, ``"wgcm"`` and ``"pcm"`` theirs or ``"0"``,
    whichever is larger, and ``"kci"`` its statistic divided by its mean under independence. A
    candidate's strength is its smallest absolute statistic so far, comparable only within one test.
    A test registered with :func:`~andrey.core.ci.register_indep_test` that reports only p-values is
    asked one query at a time, its candidates are ranked by p-value, smallest first, and its
    statistic reads ``-log10(p)``. A test's partial correlation is computed from the correlation
    matrix, and from the least-squares residuals when the conditions are collinear to working
    precision. The defaults match the authors' published settings. The earlier end of a lagged link
    is searched with the empty set only, since time order already rules out the later end as its
    cause.

    **Missing values.** A ``NaN`` in ``data`` marks a missing value. Every row of the lagged design
    that reads it holds it, and each test reads only the design rows where all of its lagged
    variables are observed (test-wise deletion) [2]_ [3]_, so tests can use different time steps;
    ``metadata["rows_tested"]`` gives the fewest and the median design rows a test read. A test with
    fewer rows than it needs (``|S| + 4`` rows for ``"fisherz"`` and ``"normal_scores"``,
    ``|S| + 3`` for ``"student_t"`` and ``"cond_gauss"``, and 2 for most other tests), or with no
    degrees of freedom left, is not tested: it never separates a pair and never confirms a link, and
    ``metadata["not_tested"]`` counts the conditioning sets not tested. A link no tested query could
    decide, as one between variables never observed together, is undecided: it is left out of the
    graph, has no p-value (``NaN``), and is listed in ``metadata["undecided_pairs"]``; no rule reads
    a triple with an undecided side, and no test conditions on a node whose link to either end of
    the tested pair is undecided. A drawn link an untested conditioning set might have separated,
    one with an end an undecided link joins included, is listed in
    ``metadata["possibly_separable"]``: a flag of what the data could not check, not a finding that
    the link is false. A continuous column constant on a test's rows raises ``ValueError``. The
    categorical, mixed-data, and nonlinear tests delete rows the same way. Test-wise deletion is
    valid when values are missing completely at random, which means that whether a value is missing
    depends on no value in the data; the rows a test reads are then a random sample of all rows, and
    otherwise they are a selected sample that more rows do not correct.

    **Limitations.** On simulated linear-Gaussian series with 30% of the series unobserved and
    ``tau_max=3``, its median errors (endpoint SHD against the PAG of the observed series) stayed
    below the empty graph's through 40 variables, the unobserved ones counted, where some fits
    already ran past 1 h on 4 cores (`0.10.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.10.0.html#methods-on-their-own-axes>`_). It assumes a
    stationary series, no selection bias, and, with a partial-correlation test, linear effects;
    ``"cmiknn"`` and ``"gpdc"`` lift that assumption at a far higher cost per test and are not
    benchmarked yet. Hidden common causes are allowed, but the graph covers only ``tau_max + 1``
    time steps, and a hidden cause that acts through earlier steps can add links within that window.
    The removal and rule steps test subsets of a node's neighbors, so the number of tests grows
    exponentially with the links at one node, and a test conditions on up to ``n_variables *
    (tau_max + 1) - 2`` values, so the series needs many more time steps than that. Missing values
    are not benchmarked yet. The numbers are a reference, not a hard limit: they change with the
    data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Gerhardus, A., and Runge, J. (2020). High-recall causal discovery for autocorrelated
       time series with latent confounders. Advances in Neural Information Processing Systems 33
       (NeurIPS 2020), 12615-12625.
    .. [2] Strobl, E. V., Visweswaran, S., and Spirtes, P. L. (2018). Fast causal inference with
       non-random missingness by test-wise deletion. International Journal of Data Science and
       Analytics 6(1), 47-62.
    .. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
       discovery in the presence of missing data. Proceedings of Machine Learning Research 89,
       1762-1770.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import lpcmci
    >>> rng = np.random.default_rng(0)
    >>> hidden = rng.standard_normal(1000)
    >>> x, y = np.zeros(1000), np.zeros(1000)
    >>> for t in range(1, 1000):
    ...     x[t] = 0.5 * x[t - 1] + 0.7 * hidden[t] + rng.standard_normal()
    ...     y[t] = 0.6 * y[t - 1] + 0.6 * hidden[t] + rng.standard_normal()
    >>> out = lpcmci(np.column_stack([x, y]), tau_max=2)
    >>> out.metadata["links"][0][0][1]
    '<->'
    >>> out.structure.lag(1).oriented_edges()
    [(0, 0, 'directed'), (1, 1, 'directed')]
    """
    from andrey.temporal.lpcmci import lpcmci as _native
    from andrey.temporal.lpcmci import unmet_claims
    from andrey.temporal.pcmci import design_column_name

    assert indep_test is not None  # method_spec picks it from the column types
    X, mask, labels, constraints = series_inputs(
        data,
        indep_test,
        categorical,
        knowledge,
        method="lpcmci",
        tau_max=tau_max,
        # A test conditions on up to n * (tau_max + 1) - 2 nodes of the window, over the last
        # n_samples - 2 * tau_max steps; this keeps a degree of freedom in every test.
        min_rows=lambda n: n * (tau_max + 1) + 2 * tau_max + 1,
    )
    with (
        copies_named(labels, design_column_name(X.shape[1], column_namer(labels))),
        rows_tested() as rows,
    ):
        result = _native(
            X,
            tau_max=tau_max,
            pc_alpha=alpha,
            n_preliminary_iterations=n_preliminary_iterations,
            indep_test=indep_test,
            degrees_of_freedom=degrees_of_freedom,
            max_p_global=max_p_global,
            constraints=constraints,
            categorical=mask,
            seed=seed,
        )
    undecided = np.zeros(result.p_values.shape, dtype=bool)
    for i, j, lag in result.undecided:
        undecided[lag, i, j] = True
    metadata = {
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
        **undecided_links(undecided, labels),
        **possibly_separable_links(
            rows, result.start_marks != NULL, undecided, labels, constraints
        ),
    }
    out = with_labels(adapt_lpcmci(result, metadata), labels)
    if constraints is None or constraints.required_paths is None:
        return out
    return drop_unmet(out, unmet_claims(result, constraints), "lpcmci")


@method_spec("granger")
def granger(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    alpha: float = 0.05,
    fdr_method: str = "fdr_bh",
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn the lagged causal graph of a multivariate time series by Granger causality.

    A variable Granger-causes another when its past improves the prediction of the other's
    present [1]_. Each variable is regressed by least squares on the past ``tau_max`` steps of every
    variable, a vector autoregression [2]_, and each lagged coefficient is tested against zero
    given every other lagged variable: the F-test of the regression with and without that
    regressor, the square of the coefficient's t-ratio. A lagged edge is a coefficient whose
    p-value is at most ``alpha``. Use it for one multivariate series whose effects take at least
    one time step.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag: each regression uses the past ``tau_max`` time steps, at least ``1``.
    alpha : float, default=0.05
        Significance level of the test of each lagged coefficient, in ``(0, 1)``: a larger value
        returns more edges.
    fdr_method : {"none", "fdr_bh"}, default="fdr_bh"
        How the p-values are corrected for the number of links tested. ``"fdr_bh"`` replaces them
        with their Benjamini-Hochberg adjusted values [3]_ over all the links together, every lag
        and every ordered pair of variables, a variable's link to its own past included. A link is
        then an edge when its adjusted p-value is at most ``alpha``, which holds the false discovery
        rate, the expected share of false edges among the returned edges, at or below ``alpha`` when
        the tests are independent or positively dependent. The adjusted p-value is never below the
        tested one, so ``"fdr_bh"`` can remove edges and never adds one, and
        ``metadata["p_values"]`` holds the adjusted values. ``"none"`` keeps each p-value as tested,
        so about ``alpha`` of the absent links are edges. A link the knowledge forbids is no test:
        it is not counted and keeps p-value ``1``; a required link is counted with p-value ``0``.
        ``"fdr_bh"`` favors fewer extra edges; it gives up true edges whose p-value is not small
        enough for the number of tests. ``"none"`` favors true edges; it gives up control of the
        extra ones, which grow with the number of variables. On simulated linear-Gaussian VAR series
        of 5 to 80 variables, 500 and 1,000 time steps, and ``tau_max`` of 1 and 2 (20 cells),
        ``"fdr_bh"`` had fewer mean errors (missing plus extra lagged edges) than ``"none"`` in 15
        cells and more in 1, with 5 variables; at 80 variables, 1,000 steps, and ``tau_max=2``,
        ``"none"`` had 768 errors, above the empty graph's 481, with 603 extra edges, and
        ``"fdr_bh"`` had 256, with 13 extra and 243 missed (`0.12.0 release card
        <https://andrey.abel.ai/docs/benchmarks/0.12.0.html#fdr-control-in-pcmci-and-granger>`__).
        See Notes.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over lags ``1`` to ``tau_max``: required and forbidden links, the pairs allowed
        to be linked, and temporal tiers, all honored. A link ``(cause, effect, lag)`` is the link
        from ``cause`` at time ``t - lag`` to ``effect`` at time ``t``; a forbidden
        ``(cause, effect)`` pair forbids it at every lag, and a required link names its lag, from
        ``1`` to ``tau_max`` (the method learns no same-step links). Each variable's regression
        leaves out its forbidden lagged regressors, so the others are tested given the ones that
        stay, and a required link is in the graph whatever its test says, with the regression's
        coefficient as its weight. Tiers forbid every link from a later tier into an earlier one,
        and a pair left out of ``allowed`` has no link at any lag; a variable's links to its own
        past stay allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``1`` to ``tau_max``
        and no lag 0. ``structure.lag(k)`` is a ``digraph`` with the edge ``i -> j`` when
        ``x_i(t - k)`` is significant in the regression of ``x_j(t)``, self-loops included;
        ``structure.lag_weights[k - 1][i, j]`` is that regression coefficient, in the data's units,
        and ``0`` where there is no edge. ``ordering`` is ``None``, and ``metadata["algorithm"]``
        is ``"granger"``. ``metadata["p_values"][k - 1][i][j]`` is the p-value of the link
        ``i -> j`` at lag ``k``, for every pair, significant or not, adjusted under
        ``fdr_method="fdr_bh"``; a link the knowledge forbids reads p-value 1, and a required one 0.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` has fewer than
        ``(n_variables + 1) * tau_max + 2`` rows, so that the regression has no degree of freedom
        left, or fewer than 2 columns, or if a regression is singular, as when one variable is a
        linear combination of others. Also if ``knowledge`` names a column ``data`` lacks,
        contradicts itself, or requires a link without a lag or beyond ``tau_max``.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** Granger causality suits one continuous, observational multivariate time series
    that is stationary (its links and noise do not change over time), in which each variable is a
    linear function of the past ``tau_max`` steps of every variable plus noise [1]_ [2]_. Its edges
    read as causes only when every effect takes at least one time step and every common cause is a
    column of ``data`` [1]_. Each regression takes every lagged variable at once, so it suits a
    moderate number of variables over a series many times longer than ``n_variables * tau_max``
    steps. It returns a temporal structure of lagged links only: a directed graph at each lag ``1``
    to ``tau_max``, weighted by the regression coefficients. When effects can act within one time
    step, use :func:`~andrey.pcmciplus`; when unmeasured common causes are plausible,
    :func:`~andrey.lpcmci`; and for nonlinear lagged effects, :func:`~andrey.pcmci` with a
    nonlinear test.

    An edge says that the past of one variable helps predict another beyond the past of every
    variable in ``data``.

    The test of each coefficient is the F-test of one regressor in Granger's regression [1]_, the
    test a vector autoregression reports for a coefficient [2]_. Its p-value takes the coefficient's
    t-ratio against the Student ``t`` distribution with the regression's residual degrees of
    freedom, which is exact for Gaussian noise. Each link is tested at ``alpha``, and
    ``fdr_method="none"`` makes no correction for the ``n_variables**2 * tau_max`` tests, so on a
    series with many variables about ``alpha`` of the absent links appear as edges, which the
    default ``"fdr_bh"`` avoids; with ``"none"``, lower ``alpha`` or judge the edges by their
    p-values and weights. The regressions are fitted on standardized variables, so the edges do not
    depend on the units of any variable; the coefficients are converted back to the data's units.
    The fit is deterministic and has no random component.

    **Choosing fdr_method.** The method tests ``n_variables ** 2 * tau_max`` links, each at level
    ``alpha``. With ``"none"`` about ``alpha`` of the absent links are returned by chance, so the
    extra edges grow with the number of variables (see ``fdr_method``). ``"fdr_bh"`` [3]_ sorts the
    ``m`` p-values of the links and gives the one of rank ``k`` the smallest value, over its rank
    and every higher one, of ``p * m / rank``, at most ``1``; equal p-values get equal results. It
    returns fewer edges than ``"none"``: the false ones, and also true ones whose p-value is not
    small enough for the number of tests. The guarantee of [3]_ needs independent or positively
    dependent tests, which the coefficient tests of one vector autoregression are not shown to be.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: each variable's regression
    reads every row of the lagged design, so a step with a missing value has no place in it. Use
    :func:`~andrey.pcmci`, :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests delete
    rows per test and keep every time step, as dropping a step would join the steps around it.

    **Limitations.** It assumes linear effects and tests no same-step effects. On simulated
    linear-Gaussian VAR series with 2 expected parents of each variable at each lag, from 5 to 160
    variables, its errors stayed below the empty graph's in every setting with the default
    ``fdr_method="fdr_bh"``; with ``"none"`` they exceeded them in 5 of the 20 settings up to 80
    variables (see ``fdr_method``) and at 160. The work grows with ``tau_max`` and with the square
    of the number of variables. The numbers are a reference, not a hard limit: they change with the
    data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Granger, C. W. J. (1969). Investigating causal relations by econometric models and
       cross-spectral methods. Econometrica 37(3), 424-438.
    .. [2] Lutkepohl, H. (2005). New Introduction to Multiple Time Series Analysis. Springer.
    .. [3] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import granger
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(2000)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(2000)
    >>> out = granger(np.column_stack([x, y]))
    >>> round(float(out.structure.lag_weights[0][0, 1]), 1)
    0.8
    >>> bool(out.metadata["p_values"][0][0][1] < 0.05)
    True
    """
    from andrey.temporal.granger import granger_f_test

    X = data_matrix(data, min_rows=2, method="granger", fix=KEEP_STEPS)
    if X.shape[1] < 2:
        raise ValueError(f"data needs at least 2 columns (variables), got {X.shape[1]}")
    min_rows = (X.shape[1] + 1) * tau_max + 2  # a degree of freedom left in every regression
    check_series_rows(X, tau_max, min_rows)
    labels = column_labels(data)
    constraints = resolve_lagged(
        knowledge, X.shape[1], labels, tau_max, method="granger", contemporaneous=False
    )
    required = None if constraints is None else constraints.required[1:]
    peak = np.abs(X).max(axis=0)  # divide first, so tiny or huge units cannot underflow the std
    spread = (X / peak).std(axis=0)  # data_matrix rejects a constant column
    coeff, p_values = granger_f_test(
        (X / peak - (X / peak).mean(axis=0)) / spread,
        maxlag=tau_max,
        alpha=alpha,
        fdr_method=fdr_method,
        forbidden=None if constraints is None else constraints.forbidden[1:],
        required=required,
    )
    scale = spread * peak
    # coeff[i, j + d*k] weights standardized x_j(t-k-1) in standardized x_i(t): rescale to units.
    # Units far apart can overflow the ratio; a zero weight stays zero rather than becoming NaN.
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = scale[:, None] / np.tile(scale, tau_max)[None, :]
        coeff = np.where(coeff == 0, 0.0, coeff * ratio)
    return with_labels(adapt_granger(coeff, tau_max, required, p_values), labels)


@method_spec("dynotears")
def dynotears(
    data: npt.ArrayLike,
    *,
    tau_max: int = 1,
    lambda1: float = 0.05,
    w_threshold: float = 0.3,
    lag_threshold: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    standardize: bool = False,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a weighted temporal causal structure from one multivariate time series with DYNOTEARS.

    DYNOTEARS [1]_ fits a linear structural vector autoregression, same-step and lagged effects
    together, by continuous optimization: the least-squares loss with an ``l1`` penalty, under the
    smooth acyclicity constraint of NOTEARS [2]_ on the same-step weights. Use it for one
    multivariate series in which effects act both within a time step and from earlier steps.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become the structure's ``labels``.
    tau_max : int, default=1
        The largest lag, at least ``1`` (``p`` in [1]_): each step is fit on the same step and the
        ``tau_max`` steps before it. ``1`` is the order of the paper's experiments.
    lambda1 : float, default=0.05
        Weight of the ``l1`` penalty on every same-step and lagged weight, at least ``0``. Larger
        values give sparser graphs. ``0.05`` is the paper's value for 500 time steps; it used
        ``0.2`` for 50.
    w_threshold : float, default=0.3
        A same-step edge is kept when its weight magnitude reaches this value, at least ``0``.
    lag_threshold : float, default=0.1
        A lagged edge is kept when its weight is not ``0`` and its magnitude reaches this value,
        at least ``0``. The paper used ``0.1`` for 500 time steps and ``0.2`` for 50.
    max_iter : int, default=100
        Most dual steps of the augmented Lagrangian, at least ``1``. Stopping on it rather than on
        ``h_tol`` or ``rho_max`` warns with :class:`~andrey.SearchLimitWarning`.
    h_tol : float, default=1e-8
        The fit stops when the acyclicity function of the same-step weights falls to this value,
        above ``0``.
    rho_max : float, default=1e16
        The fit stops when the penalty weight on the acyclicity function reaches this value, above
        ``1``, the weight the fit starts from.
    standardize : bool, default=False
        Scale each column to unit variance first, so the fit does not read the columns' units; the
        weights, ``lag_weights`` included, are then those of the scaled columns. ``True`` favors a
        fit that no change of units moves; it gives up the gain the fit takes from the data's units
        on series whose variance grows along the causal order. ``False`` favors the data's units,
        the setting the paper's simulations measure; it gives up independence from the units. On the
        paper's design with Gaussian noise and 500 time steps, at 5 to 50 variables, the errors
        (SHD) were 0.15 to 1.75 in the data's units, against the empty graph's 13.5 to 144.6, and
        8.30 to 82.60 with z-scored columns, against 14.6 to 147.2.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`, read as links between lagged
        variables over lags ``0`` to ``tau_max``: required and forbidden links, the pairs allowed
        to be linked, and temporal tiers, all honored. A link ``(cause, effect, lag)`` is the link
        from ``cause`` at time ``t - lag`` to ``effect`` at time ``t``, lag ``0`` being the same
        step; a forbidden ``(cause, effect)`` pair forbids it at every lag, and a required link
        names its lag. A forbidden link is held at zero in the fit; a required link is left out of
        the penalty and the threshold, is in the graph even where its weight is ``0``, and at lag
        ``0`` is never removed by the cycle repair. Tiers forbid every link from a later tier into
        an earlier one, and a pair left out of ``allowed`` has no link at any lag; a variable's
        links to its own past stay allowed.

    Returns
    -------
    StructureOutput
        Wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` with lags ``0`` to ``tau_max``.
        ``structure.lag(0)`` is a ``dag`` of the same-step edges and ``structure.lag(k)``,
        ``k >= 1``, a ``digraph`` with the edge ``i -> j`` from ``x_i(t - k)`` to ``x_j(t)``,
        self-loops included. ``structure.lag_weights[k][i, j]`` is the coefficient of that edge in
        the data's units, or the scaled columns' with ``standardize=True``, and ``0`` where there
        is no edge. ``ordering`` is ``None``. ``metadata["algorithm"]`` is ``"dynotears"``,
        ``metadata["h_final"]`` the acyclicity function of the same-step weights before the
        threshold, and ``metadata["edges_removed"]`` the same-step edges the threshold kept that
        the cycle repair dropped. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a finite
        numeric matrix without a constant column, or has fewer than ``tau_max + 2`` rows, so that
        fewer than 2 steps have ``tau_max`` steps before them. Also if a column repeats another up
        to scale and shift, as the direction between copies cannot be told apart; one message names
        every group of copies and says to keep one column per group. Also if ``knowledge`` names a
        column ``data`` lacks, contradicts itself, or requires a link without a lag or beyond
        ``tau_max``.

    Warns
    -----
    SearchLimitWarning
        When the fit stops on ``max_iter`` rather than on ``h_tol`` or ``rho_max``.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** DYNOTEARS suits one continuous, observational, stationary multivariate time
    series from a linear structural vector autoregression: each variable is a weighted sum of the
    other variables at the same step and of every variable at the past ``tau_max`` steps, plus
    noise, with weights that do not change over time, same-step effects that form no cycle, and no
    unmeasured variable that causes two measured ones [1]_. Its ``l1`` penalty and both thresholds
    act on the weights in the data's units, so the columns should be in comparable units [4]_ [5]_:
    a cause recorded in thousandths multiplies its weight by 1,000, and the penalty can then remove
    a true edge, or every edge, without a warning; convert such columns to comparable units first.
    ``standardize=True`` scales them to unit variance instead, which falls outside the setting the
    paper's simulations measure (see ``standardize``). It fits ``(tau_max + 1) * d * d`` weights
    under an acyclicity function that takes a matrix exponential, so it suits a moderate number of
    variables, a few lags, and a long series. It returns a temporal structure with a DAG at lag
    ``0`` and the graphs at lags ``1`` to ``tau_max``, self-lags included, with the weight of each
    edge. For nonlinear effects, use :func:`~andrey.pcmciplus` with a nonlinear test such as
    ``"gpdc"`` or ``"rcot"``, and for many lags, :func:`~andrey.pcmciplus` with its default test;
    when unmeasured common causes are plausible, use :func:`~andrey.lpcmci`; and for i.i.d. rows
    rather than a series, :func:`~andrey.notears`.

    The model is ``x_t = x_t W + x_{t-1} A_1 + ... + x_{t-p} A_p + z_t`` for the row vector
    ``x_t`` of the variables at step ``t`` and ``p = tau_max``, where ``W`` is acyclic and the
    noise ``z_t`` is independent within and across steps [1]_. The fit minimizes

    ``0.5 / n * ||X - X W - Y A||**2 + lambda1 * (|W|_1 + |A|_1)`` subject to
    ``tr(exp(W * W)) - d = 0``,

    where the ``n`` rows of ``X`` are the steps with ``p`` steps before them, ``Y`` holds the same
    rows' ``p`` earlier steps side by side, and ``A`` stacks the ``A_k``. Every column of ``X`` and
    of ``Y`` is centered on its own mean first, which is the same as fitting an intercept, so a
    series with a nonzero mean adds no weight, to the solver's precision. The paper's loss fits no
    intercept, as its model takes noise of mean zero [1]_, while its simulations draw exponential
    noise of scale 1, whose mean is 1: the centering absorbs that mean, and Limitations gives what
    it costs when the mean is zero. The paper's experiments give ``W`` and ``A`` the same penalty,
    and ``lambda1`` is that one value. A lagged edge points forward in time and closes no cycle, so
    the constraint reads ``W`` alone; a lagged self-loop, a variable's effect on its own future, is
    an edge like any other. As in NOTEARS, the weights are written as ``W = W+ - W-`` with both
    parts non-negative and the diagonal of ``W`` held at zero, and the augmented Lagrangian
    ``f + 0.5 * rho * h**2 + alpha * h`` is minimized by L-BFGS-B from the zero matrix. After each
    solve ``rho`` grows tenfold while ``h`` has not fallen to a quarter of its previous value, and
    then ``alpha += rho * h``, until ``h <= h_tol`` or ``rho >= rho_max``. Same-step weights below
    ``w_threshold`` and lagged ones below ``lag_threshold`` are then removed, and then the weakest
    remaining same-step edges until no cycle is left. The defaults of ``lambda1``, ``w_threshold``,
    ``lag_threshold``, and ``tau_max`` are the paper's values for its simulations with 500 time
    steps; ``max_iter``, ``h_tol``, and ``rho_max`` are those of NOTEARS [2]_, whose method the
    paper follows.

    [1]_ shows the model identifiable when the noise is non-Gaussian, or Gaussian with unit
    variance. The likelihood path that :func:`~andrey.notears` follows with ``robust=True`` [3]_
    is shown for independent samples only, so this method has no such option.

    The fit runs on the columns sorted by a rank computed from the data, as for
    :func:`~andrey.hc`, so under the default tie-break reordering the columns relabels the result
    and changes nothing else, unless two columns are identical. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order, and how ``andrey.config(tie_break="position")`` sorts the columns by position
    instead. The fit draws no random numbers and runs on one BLAS thread, so the host's thread count
    does not change it.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: the loss fits every step on the
    steps before it, so a step with a missing value has no place in it. Use :func:`~andrey.pcmci`,
    :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests delete rows per test and keep
    every time step, as dropping a step would join the steps around it.

    **Limitations.** The ``l1`` penalty and the weight thresholds read the data's units, as in
    NOTEARS [4]_ [5]_: rescaling a column changes the fit, and on series whose variance grows along
    the causal order the fit gains from that order, which ``standardize=True`` gives up. On the
    paper's design [1]_ with 500 time steps, its errors at the defaults stayed far below the empty
    graph's from 5 to 100 variables. At 50 steps the defaults made more errors than the empty graph
    from 20 variables on, so set ``lambda1=0.2`` and ``lag_threshold=0.2``, the paper's settings for
    50 steps, for a short series. Its centering, which the paper's loss leaves out, costs errors on
    short series whose noise has mean zero; the `0.14.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.14.0.html#dynotears>`_ has the numbers. The time grows
    faster than the cube of the number of variables: at 500 steps on one core, the median fit took
    about 30 s at 50 variables and 299 to 385 s at 100. The numbers are a reference, not a hard
    limit: they change with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., Beaumont, P.,
       Georgatzis, K., and Aragam, B. (2020). DYNOTEARS: Structure learning from time-series
       data. AISTATS, PMLR 108, 1595-1605.
    .. [2] Zheng, X., Aragam, B., Ravikumar, P., and Xing, E. P. (2018). DAGs with NO TEARS:
       Continuous optimization for structure learning. NeurIPS 31, 9472-9483.
    .. [3] Deng, C., Bello, K., Ravikumar, P., and Aragam, B. (2024). Markov equivalence and
       consistency in differentiable structure learning. NeurIPS 37.
    .. [4] Reisach, A. G., Seiler, C., and Weichwald, S. (2021). Beware of the simulated DAG!
       Causal discovery benchmarks may be easy to game. NeurIPS 34.
    .. [5] Kaiser, M. and Sipos, M. (2022). Unsuitability of NOTEARS for causal graph discovery
       when dealing with dimensional quantities. Neural Processing Letters 54(3), 1587-1595.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import dynotears
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = np.r_[0.0, 0.8 * x[:-1]] + rng.standard_normal(500)
    >>> out = dynotears(np.column_stack([x, y]))
    >>> out.structure.lag(1).oriented_edges()
    [(0, 1, 'directed')]
    """
    from andrey.temporal.dynotears import dynotears as _native

    X = data_matrix(data, method="dynotears", fix=KEEP_STEPS)
    if X.shape[0] < tau_max + 2:
        raise ValueError(
            f"dynotears: data needs at least tau_max + 2 = {tau_max + 2} rows (time steps), so "
            f"that 2 steps have tau_max={tau_max} steps before them, got {X.shape[0]}"
        )
    labels = column_labels(data)
    if X.shape[1] > 1:
        groups = copy_groups(np.corrcoef(X, rowvar=False), X.shape[0])
        if groups:
            raise copies_error(groups, column_namer(labels))
    constraints = resolve_lagged(knowledge, X.shape[1], labels, tau_max, method="dynotears")
    with one_blas_thread():
        result = _native(
            X,
            tau_max=tau_max,
            lambda1=lambda1,
            w_threshold=w_threshold,
            lag_threshold=lag_threshold,
            max_iter=max_iter,
            h_tol=h_tol,
            rho_max=rho_max,
            standardize=standardize,
            constraints=constraints,
        )
    return with_labels(adapt_dynotears(result), labels)


@method_spec("rpcmci")
def rpcmci(
    data: npt.ArrayLike,
    *,
    n_regimes: int = 2,
    max_transitions: int = 1,
    switch_threshold: float = 0.5,
    tau_max: int = 1,
    pc1_alpha: float = 0.2,
    alpha: float = 0.01,
    fdr_method: str = "none",
    indep_test: str = "student_t",
    n_starts: int = 50,
    max_iter: int = 20,
    seed: int | None = None,
) -> RegimeOutput:
    """Learn the regimes of one time series and the lagged causal links of each with Regime-PCMCI.

    Regime-PCMCI [1]_ takes a series whose lagged causal graph switches between a few persistent
    regimes. It alternates two steps: PCMCI [2]_ learns each regime's graph from its assigned time
    steps, and a linear program reassigns the steps to the regimes whose linear model, fitted on the
    graph's links, predicts them best, with a cap on how often each regime may start or end. It
    returns each time step's regime and one lagged graph per regime.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        A multivariate time series, one row per time step. Coerced to ``float64``. A DataFrame's
        column names become each regime structure's ``labels``.
    n_regimes : int, default=2
        The number of regimes, at least ``2``.
    max_transitions : int, default=1
        The maximum switches on or off per regime over the series, at least ``1``: the cap on
        ``sum_t |gamma_k(t + 1) - gamma_k(t)|`` for each regime ``k``. For regimes that last ``m``
        steps on average, about ``n_samples / (m * n_regimes)`` [1]_. The default, ``1``, the
        smallest that lets a regime start or end, allows one change point per regime; set it from
        the time scale of the regimes.
    switch_threshold : float, default=0.5
        The weight above which a time step joins a regime's PCMCI step and model fit, in
        ``(0, 1)``: step ``t`` joins regime ``k`` when ``gamma_k(t) > switch_threshold``, with the
        weight rounded to 9 decimals. The first weights are drawn in ``[0, 1)``, so ``1`` would
        leave every regime empty.
    tau_max : int, default=1
        The largest lag tested, at least ``1``: links at lags ``1`` to ``tau_max`` are tested.
    pc1_alpha : float, default=0.2
        Significance level of each regime's PC1 step, in ``(0, 1)``: a candidate parent is dropped
        when a test gives it a p-value above ``pc1_alpha``.
    alpha : float, default=0.01
        Significance level of each regime's MCI step, in ``(0, 1)``, the level that decides the
        returned links: a link is in a regime's graph when its p-value is at most ``alpha``.
    fdr_method : {"none", "fdr_bh"}, default="none"
        How each regime's MCI p-values are corrected for the number of links tested. ``"none"``
        keeps each p-value as tested; ``"fdr_bh"`` replaces them with their Benjamini-Hochberg
        adjusted values [3]_ over all of the regime's links together, and a link is then in the
        graph when its adjusted p-value is at most ``alpha``. ``"none"`` favors true links; it gives
        up control of the extra ones. ``"fdr_bh"`` favors fewer extra links; it gives up some true
        ones. Not measured with Regime-PCMCI; :func:`~andrey.pcmci` states the trade as measured
        there.
    indep_test : {"student_t", "fisherz"}, default="student_t"
        The conditional-independence test behind every p-value, a partial-correlation test for
        linear effects: ``"student_t"`` takes the p-value from the Student-t distribution,
        ``"fisherz"`` from Fisher's z-transform and the normal distribution.
        ``metadata["indep_test"]`` names it. ``"student_t"`` favors an exact level on independent
        Gaussian rows and a test that runs with one row fewer (``|S| + 3`` against ``|S| + 4`` for
        ``|S|`` conditions); it gives up agreement with the p-values of :func:`~andrey.pcmci`'s
        default, ``"fisherz"``. ``"fisherz"`` favors that agreement; it gives up the exact level on
        a test with few rows. The two are nearly the same test: on independent Gaussian rows with
        two conditions, Fisher's z at ``alpha=0.01`` rejects a true independence with probability
        1.13% on 20 rows, 1.05% on 50, and 1.02% on 100, calculated from the null distribution of
        the partial correlation, against 1% for ``"student_t"``. Their effect on Regime-PCMCI's
        graphs was not measured.
    n_starts : int, default=50
        The number of starts, at least ``1``, each with a random regime assignment; the start with
        the lowest objective is returned.
    max_iter : int, default=20
        Maximum iterations per start, at least ``1``; one iteration learns every regime's graph and
        then reassigns the steps.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Start ``0`` draws its first regime weights
        from ``seed``, start ``a > 0`` from ``SeedSequence(seed, spawn_key=(a,))``, so each start
        depends on ``seed`` and ``a`` only. ``metadata["seed"]`` records the seed used.

    Returns
    -------
    RegimeOutput
        ``regimes`` holds each time step's largest-weight regime, ``0`` to ``n_regimes - 1``, and
        ``weights`` the ``(n_regimes, n_samples)`` weights the linear program returned. The first
        ``tau_max`` steps have no prediction, so their regime carries no evidence. ``outputs[k]``
        wraps a ``temporal`` :class:`~andrey.core.TemporalStructure` of regime ``k`` with lags ``1``
        to ``tau_max`` and no lag 0, as :func:`~andrey.pcmci` returns it: ``structure.lag(j)`` is a
        ``digraph`` whose edge ``i -> h`` is the link from variable ``i`` at time ``t - j`` to
        variable ``h`` at time ``t``, and its ``metadata["p_values"]``, ``metadata["statistics"]``,
        and ``metadata["parents"]`` hold the regime's MCI p-values, test statistics, and PC1
        parents. Each regime's graph is learned in the last iteration from steps selected by the
        weights before the last reassignment. ``metadata["objective"]`` is the linear program's
        optimal value, ``metadata["start"]`` the index of the returned start,
        ``metadata["n_iterations"]`` its iterations, ``metadata["converged"]`` whether it stopped on
        its own rule rather than on ``max_iter``, and ``metadata["failed_starts"]`` the number of
        starts that failed (see Notes). ``metadata["algorithm"]`` is ``"rpcmci"`` and
        ``metadata["seed"]`` the seed used, on the result and on each regime's output.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a usable 2-D
        matrix with at least ``2 * tau_max * (n_variables + 1) + 3`` rows, the length at which one
        regime's tests keep a degree of freedom, or holds a missing value;
        if a variable repeats another up to scale and shift, at the same step or up to
        ``2 * tau_max`` steps apart, before any test runs, with one message naming every group of
        copies; and if every start fails.

    Warns
    -----
    ExperimentalWarning
        On the first call in a process: the method is not supported (see Limitations), and its
        API may change without deprecation.
    SearchLimitWarning
        When the returned start ran ``max_iter`` iterations without stopping on its own rule.
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** Regime-PCMCI suits one observational multivariate time series whose lagged
    causal links, linear in the variables, switch between a few regimes that each last many time
    steps, such as a seasonal or circulation regime in climate data [1]_. Within a regime, it makes
    PCMCI's assumptions: no unmeasured common cause, faithfulness, and every effect taking at least
    one time step. It is designed for few regimes and variables, with a known number of regimes
    and a known time scale of their switching [1]_, and returns the regime of each time step and a
    directed graph at each lag ``1`` to ``tau_max`` per regime. When the links do not change over
    time, use :func:`~andrey.pcmci`; when the regimes are known, run :func:`~andrey.pcmci` on each,
    or use :func:`~andrey.jpcmciplus` on series from known contexts.

    A start draws its regime weights uniformly in ``[0, 1)`` and repeats two steps [1]_. First, for
    each regime ``k``, PCMCI runs on the steps ``t`` with ``gamma_k(t) > switch_threshold``: each
    test reads those steps as the time of its effect and the lagged values of every earlier step.
    Each variable is regressed by least squares, with an intercept, on its MCI parents in the regime
    over the same steps, and the fit predicts every step from ``tau_max`` on. Second, with
    ``c_k(t)`` the squared residuals of regime ``k`` at step ``t`` summed over the variables, each
    residual divided by the variable's standard deviation over the regime's steps, the weights
    become the solution of the linear program that minimizes ``sum_k sum_t gamma_k(t) c_k(t)`` over
    weights in ``[0, 1]`` that sum to ``1`` at every step and change by at most ``max_transitions``
    in total in each regime. The dual simplex method solves it exactly and returns a vertex, whose
    weights are mostly ``0`` or ``1``. A start stops when the weights no longer change, or from its
    sixth iteration on when their summed absolute change is at most
    ``2 * n_regimes * n_samples // 100``, as the authors' code does, and otherwise after
    ``max_iter`` iterations. A start fails when a regime is left with fewer than 6 time steps, the
    limit of the authors' code, a variable constant over its steps, or a test that cannot run on its
    steps, such as one with more conditions than steps; failed starts are skipped. The returned
    start has the lowest successful objective, breaking ties at 9 decimals per time step by the
    earliest start.

    The paper's threshold for joining a regime is ``0.5`` [1]_, the default. The paper's Eq. 11
    reads ``>=``, and Andrey follows the authors' code, which uses ``>``, so a step the linear
    program weighs half in each of two regimes joins neither fit. Each weight is rounded to 9
    decimals before the comparison, so this holds when the solver returns half with an error in its
    last bits. Dividing each residual by its regime's standard deviation follows the authors' code,
    so rescaling a column changes no result. Equal PC1 strengths are ordered by each variable's
    rank, as in :func:`~andrey.pcmci`, and the residuals enter the cost in rank order, so reordering
    columns only relabels the result.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: a missing value would remove
    its step from every regime's fit and leave its regime unknown. Use :func:`~andrey.pcmci`,
    :func:`~andrey.pcmciplus`, or :func:`~andrey.lpcmci`, whose tests delete rows per test and keep
    every time step, as dropping a step would join the steps around it.

    **Limitations.** Not benchmarked yet; the tests run it on 2 to 3 variables, 2 to 3 regimes, and
    up to 1,000 time steps, and the `0.19.0 release card
    <https://andrey.abel.ai/docs/benchmarks/0.19.0.html#regime-pcmci>`_ compares it with its
    authors' code. The nonconvex objective can leave a regime holding steps from several true
    regimes, or too few to fit; more ``n_starts`` guard against it at a cost that grows linearly
    with them [1]_. Regimes whose models differ little, such as one link's strength alone, are hard
    to tell apart [1]_. On a series without regimes, a regime, most often a small one, can hold a
    spurious link. The number of regimes and ``max_transitions`` are not learned. Each iteration
    runs PCMCI once per regime and solves one linear program over ``2 * n_regimes * n_samples``
    variables. The numbers are a reference, not a hard limit: they change with the data generator,
    the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Saggioro, E., de Wiljes, J., Kretschmer, M., and Runge, J. (2020). Reconstructing
       regime-dependent causal relationships from observational time series. Chaos 30(11),
       113115.
    .. [2] Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., and Sejdinovic, D. (2019). Detecting
       and quantifying causal associations in large nonlinear time series datasets. Science
       Advances 5(11), eaau4996.
    .. [3] Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a
       practical and powerful approach to multiple testing. Journal of the Royal Statistical
       Society B 57(1), 289-300.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import rpcmci
    >>> rng = np.random.default_rng(0)
    >>> x = np.zeros((600, 2))
    >>> for t in range(1, 600):
    ...     sign = 1.0 if (t // 100) % 2 == 0 else -1.0
    ...     x[t] = [0.2 * x[t - 1, 0], 0.2 * x[t - 1, 1] + sign * 0.8 * x[t - 1, 0]]
    ...     x[t] += rng.standard_normal(2)
    >>> out = rpcmci(x, max_transitions=6, n_starts=10, seed=0)
    >>> out.n_regimes, out.regimes.shape
    (2, (600,))
    """
    from andrey.temporal.pcmci import design_column_name, design_test
    from andrey.temporal.rpcmci import rpcmci as _native

    X = data_matrix(data, method="rpcmci", fix=KEEP_STEPS)
    labels = column_labels(data)
    n_vars = X.shape[1]
    check_series_rows(X, tau_max, 2 * tau_max * (n_vars + 1) + 3)  # pcmci's, for one regime
    with copies_named(labels, design_column_name(n_vars, column_namer(labels))):
        design_test(indep_test, X, tau_max)  # copies of the whole series raise here, named
    result = _native(
        X,
        n_regimes=n_regimes,
        max_transitions=max_transitions,
        switch_threshold=switch_threshold,
        n_starts=n_starts,
        max_iter=max_iter,
        tau_max=tau_max,
        pc_alpha=pc1_alpha,
        alpha_level=alpha,
        fdr_method=fdr_method,
        indep_test=indep_test,
        seed=seed if seed is not None else 0,
    )
    if not result.converged:
        warn_search_limit("rpcmci", "max_iter", max_iter, "iterations")
    return adapt_rpcmci(result, labels)


def series_inputs(
    data: npt.ArrayLike,
    indep_test: str,
    categorical: Sequence[str | int] | None,
    knowledge: Knowledge | None,
    *,
    method: str,
    tau_max: int,
    min_rows: Callable[[int], int],
    contemporaneous: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...] | None, LaggedConstraints | None]:
    """Return a series' matrix, its categorical-column mask, its column labels, and knowledge.

    For PCMCI, PCMCI+, and LPCMCI: the series is checked for ``indep_test`` and keeps ``NaN`` as
    a missing value, which the tests delete per test (:func:`~andrey.api._adapt.checked_data`); it
    needs ``min_rows(n)`` rows over ``n`` variables (:func:`check_series_rows`); and ``knowledge``
    is resolved over lags ``0`` to ``tau_max``, with ``contemporaneous`` as
    :func:`~andrey.core.knowledge.resolve_lagged` takes it. Raises what those raise, in that order.
    """
    X, mask = checked_data(data, categorical, "indep_test", indep_test, method=method, missing=True)
    check_series_rows(X, tau_max, min_rows(X.shape[1]))
    labels = column_labels(data)
    constraints = resolve_lagged(
        knowledge, X.shape[1], labels, tau_max, method=method, contemporaneous=contemporaneous
    )
    return X, mask, labels, constraints


class ContextInputs(NamedTuple):
    """The checked inputs of :func:`jpcmciplus`.

    ``series`` holds each dataset's system variables, then its time contexts, and ``spaces`` the
    ``(n_datasets, n_space)`` values of the space contexts. ``system`` and ``contexts`` are column
    positions in ``data``: the time contexts, then the space contexts. ``labels`` names every column
    and ``system_labels`` and ``series_labels`` those of ``system`` and ``series``, or ``None``.
    ``time_dummy`` says whether the fit adds the dummy of the time step.
    """

    series: list[np.ndarray]
    spaces: np.ndarray
    system: list[int]
    contexts: list[int]
    labels: tuple[str, ...] | None
    system_labels: tuple[str, ...] | None
    series_labels: tuple[str, ...] | None
    time_dummy: bool


def context_inputs(
    data: Sequence[npt.ArrayLike],
    time_contexts: Sequence[str | int] | None,
    space_contexts: Sequence[str | int] | None,
    tau_max: int,
    time_dummy: bool | None,
) -> ContextInputs:
    """Check the datasets of :func:`jpcmciplus` and split their columns by kind.

    The data check runs on the datasets stacked, as a space context is constant within a dataset.
    ``time_dummy`` becomes whether the fit adds the dummy of the time step. Raises ``ValueError`` as
    the facade's Raises entry lists, and ``TypeError`` when a context argument is not a list of
    columns (:func:`~andrey.api._adapt.column_mask`).
    """
    tables = [np.asarray(X) for X in data]
    if len(tables) < 2:
        raise ValueError(f"data must hold at least two datasets, got {len(tables)}")
    for k, table in enumerate(tables):
        if table.ndim != 2:
            raise ValueError(f"data[{k}] must be a 2-D table, got {table.ndim} dimensions")
    if len({table.shape[1] for table in tables}) > 1:
        widths = [table.shape[1] for table in tables]
        raise ValueError(f"every dataset in data needs the same columns, got {widths}")
    labels = shared_labels(column_labels(X) for X in data)
    stacked = np.vstack(tables)
    if labels is not None:  # the data check names the columns
        import pandas as pd

        stacked = pd.DataFrame(stacked, columns=list(labels))
    stacked = data_matrix(stacked, method="jpcmciplus", fix=DROP_DATASETS)
    lengths = [len(table) for table in tables]
    blocks = np.split(stacked, np.cumsum(lengths)[:-1])
    d = stacked.shape[1]
    in_time = column_mask(
        [] if time_contexts is None else time_contexts, labels, d, "time_contexts"
    )
    in_space = column_mask(
        [] if space_contexts is None else space_contexts, labels, d, "space_contexts"
    )
    if (in_time & in_space).any():
        raise ValueError("a column is both a time context and a space context")
    system = np.flatnonzero(~in_time & ~in_space).tolist()
    time, space = np.flatnonzero(in_time).tolist(), np.flatnonzero(in_space).tolist()
    if len(system) < 2:
        raise ValueError(f"data needs at least 2 system variables, got {len(system)}")
    steps = min(lengths) - 2 * tau_max
    if steps < 1:
        raise ValueError(
            f"every dataset needs more than 2 * tau_max = {2 * tau_max} rows, got {min(lengths)}"
        )
    name = column_namer(labels)
    longest = max(blocks, key=len)
    for k, block in enumerate(blocks):
        for j in time:
            if not np.array_equal(block[:, j], longest[: len(block), j]):
                raise ValueError(
                    f"time context {name(j)} must hold the same values in every dataset at each "
                    f"step, and data[{k}] differs"
                )
        for j in space:
            if (block[:, j] != block[0, j]).any():
                raise ValueError(f"space context {name(j)} varies within data[{k}]")
    if len(space) > 1:
        groups = copy_groups(np.corrcoef(stacked[:, space], rowvar=False), len(stacked))
        if groups:
            raise copies_error([[space[k] for k in group] for group in groups], name)
    aligned = len(set(lengths)) == 1
    if time_dummy and not (aligned and steps > 1):
        needs = (
            f"datasets of one length, got {lengths} rows"
            if not aligned
            else f"more than one step after the first 2 * tau_max = {2 * tau_max}, got {steps}"
        )
        raise ValueError(
            f"time_dummy=True needs {needs}; pass time_dummy=None to add it only when the "
            "datasets allow it"
        )
    time_dummy = (aligned and steps > 1) if time_dummy is None else time_dummy
    series_columns = [*system, *time]

    def named(columns: list[int]) -> tuple[str, ...] | None:
        return None if labels is None else tuple(labels[j] for j in columns)

    return ContextInputs(
        series=[block[:, series_columns] for block in blocks],
        spaces=np.array([block[0, space] for block in blocks]),
        system=system,
        contexts=[*time, *space],
        labels=labels,
        system_labels=named(system),
        series_labels=named(series_columns),
        time_dummy=time_dummy,
    )


def check_context_knowledge(knowledge: Knowledge | None, inputs: ContextInputs) -> None:
    """Raise :class:`~andrey.core.knowledge.KnowledgeError` when ``knowledge`` names a context.

    The knowledge of :func:`jpcmciplus` is over the system variables, and a position counts them,
    so a column label names a context and a position past the system variables names none.
    """
    if not isinstance(knowledge, Knowledge):
        return
    d = len(inputs.system)
    n_time = inputs.series[0].shape[1] - d
    labels = inputs.labels
    kinds = (
        {}
        if labels is None
        else {labels[j]: "time" if k < n_time else "space" for k, j in enumerate(inputs.contexts)}
    )
    parts = {
        "required": [edge[:2] for edge in knowledge.required],
        "forbidden": [edge[:2] for edge in knowledge.forbidden],
        "allowed": list(knowledge.allowed or ()),
        "tiers": list(knowledge.tiers),
    }
    for part, groups in parts.items():
        for node in (node for nodes in groups for node in nodes):
            if isinstance(node, str) and node in kinds:
                raise KnowledgeError(
                    f"knowledge names {node!r}, a {kinds[node]} context; jpcmciplus reads "
                    "knowledge over the system variables, and the contexts bring their own: no "
                    "system variable causes a context, and no two contexts are linked"
                )
            if isinstance(node, int) and not 0 <= node < d:
                raise KnowledgeError(
                    f"knowledge {part} lists column {node}, but jpcmciplus reads knowledge over "
                    f"the {d} system variables, at positions 0 to {d - 1}"
                )


def check_series_rows(X: np.ndarray, tau_max: int, min_rows: int) -> None:
    """Raise ``ValueError`` when the series ``X`` has fewer than ``min_rows`` rows (time steps)."""
    if X.shape[0] < min_rows:
        raise ValueError(
            f"data needs at least {min_rows} rows (time steps) for tau_max={tau_max} over "
            f"{X.shape[1]} variables, got {X.shape[0]}"
        )


def undecided_links(undecided: np.ndarray | None, labels: tuple[str, ...] | None) -> dict:
    """``{"undecided_pairs": ...}`` for the links ``undecided[lag, i, j]`` marks, or ``{}``.

    A link ``X^i_{t-lag} -> X^j_t`` is the pair of its ends, a lagged end named as
    :func:`~andrey.temporal.pcmci.design_column_name` names it (``"A at lag 1"``), a column
    without a label by its 0-based position as text, listed as
    :func:`~andrey.core.ci_missing.named_pairs` lists pairs; a lag-0 link appears once.
    """
    if undecided is None or not undecided.any():
        return {}
    return {"undecided_pairs": _named_links(_link_pairs(undecided), undecided.shape[1], labels)}


def possibly_separable_links(
    rows: RowCounts,
    drawn: np.ndarray,
    undecided: np.ndarray | None,
    labels: tuple[str, ...] | None,
    constraints: LaggedConstraints | None = None,
) -> dict:
    """Return ``{"possibly_separable": ...}``, the drawn links a set not tested might separate.

    ``drawn[lag, i, j]`` marks the links drawn, and the flagged ones are those of
    :func:`~andrey.temporal.pcmci.flagged_links`, excluding links required by ``constraints``. Links
    follow :func:`undecided_links`; ``{}`` when none is flagged.
    """
    from andrey.temporal.pcmci import flagged_links

    required = None if constraints is None else constraints.required
    flagged = _link_pairs(flagged_links(rows, drawn, undecided, required))
    return {"possibly_separable": _named_links(flagged, drawn.shape[1], labels)} if flagged else {}


def _link_pairs(links: np.ndarray) -> set[tuple[int, int]]:
    """Return the links ``links[lag, i, j]`` marks as pairs of design columns.

    A link's pair is ``lag * n + i`` and ``j``, the smaller first; a lag-0 link appears once.
    """
    n = links.shape[1]
    lags, sources, targets = np.nonzero(links)
    causes = (lags * n + sources).tolist()
    return {(min(a, b), max(a, b)) for a, b in zip(causes, targets.tolist())}


def _named_links(
    pairs: set[tuple[int, int]], n: int, labels: tuple[str, ...] | None
) -> list[list[str]]:
    """Return ``pairs`` of design columns over ``n`` variables by name.

    A lagged end is named as :func:`~andrey.temporal.pcmci.design_column_name` names it (``"A at
    lag 1"``), a column without a label by its 0-based position as text, and the pairs are listed
    as :func:`~andrey.core.ci_missing.named_pairs` lists them.
    """
    from andrey.temporal.pcmci import design_column_name

    name = design_column_name(n, str if labels is None else labels.__getitem__)
    return [list(pair) for pair in dict.fromkeys(map(tuple, named_pairs(pairs, name)))]
