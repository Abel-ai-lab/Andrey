"""Functional-causal-model facades: CAM, RESIT, CPCM, SCORE, NoGAM, DAS, and CAM-UV."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from andrey.api._adapt import (
    column_labels,
    column_namer,
    dag_from_adjacency,
    data_matrix,
    structure_output,
)
from andrey.api._marks import marks_from_signed
from andrey.api._method import method_spec
from andrey.core import ARROW, CIRCLE, GraphStructure
from andrey.core.backend import one_blas_thread
from andrey.core.knowledge import Knowledge, resolve
from andrey.core.score import check_columns, copies_error, copy_groups
from andrey.fcm.gam import BasisCache

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput

_log = logging.getLogger(__name__)

#: Fewest distinct values of a column: a smooth term's basis has at least three functions.
MIN_DISTINCT = 3


def _smooth_columns(X: np.ndarray, labels: tuple[str, ...] | None, method: str) -> None:
    """Raise ``ValueError`` when a column cannot carry a smooth term or repeats another.

    A column with fewer than :data:`MIN_DISTINCT` distinct values, such as a binary one, has no
    smooth basis, and a column that repeats another up to scale and shift explains it with no
    residual. The messages name ``method`` and every such column.
    """
    name = column_namer(labels)
    few = [j for j in range(X.shape[1]) if np.unique(X[:, j]).size < MIN_DISTINCT]
    if few:
        names = [name(j) for j in few]
        if len(names) == 1:
            columns, verb, pronoun = f"column {names[0]}", "has", "it"
        else:
            columns = f"columns {', '.join(names[:-1])} and {names[-1]}"
            verb, pronoun = "have", "them"
        raise ValueError(
            f"data {columns} {verb} fewer than {MIN_DISTINCT} distinct values; {method} fits a "
            f"smooth function of each variable, which needs at least {MIN_DISTINCT}, so drop "
            f"{pronoun} before the fit"
        )
    if X.shape[1] > 1:
        groups = copy_groups(np.corrcoef(X, rowvar=False), X.shape[0])
        if groups:
            raise copies_error(groups, name)


def _required_rows(X: np.ndarray, required: np.ndarray, labels: tuple[str, ...] | None) -> None:
    """Raise ``ValueError`` when a column's required parents have more coefficients than rows."""
    name = column_namer(labels)
    bases = BasisCache(X)
    n = X.shape[0]
    for j in np.flatnonzero(required.any(axis=0)).tolist():
        parents = np.flatnonzero(required[:, j]).tolist()
        need = bases.n_coef(parents)
        if need >= n:
            raise ValueError(
                f"cam: knowledge requires {len(parents)} parents of column {name(j)}, whose "
                f"additive model has {need} coefficients, but the fit needs more rows than "
                f"coefficients and data has {n} rows"
            )


@method_spec("cam")
def cam(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.001,
    max_parents: int | None = None,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with CAM, the causal additive model (nonlinear additive effects).

    CAM [1]_ models each variable as a sum of smooth functions of its parents plus Gaussian noise,
    ``x_j = sum_k f_jk(x_k) + e_j``, which identifies the DAG when the functions are nonlinear
    [6]_ [7]_. It preselects each variable's possible parents by boosting, adds edges greedily by
    the drop in the log residual variance until the DAG is complete within them, and prunes the
    edges whose smooth terms are not significant. It returns a DAG and a causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.001
        Significance level of the pruning test, in ``(0, 1)``: a parent stays when its smooth term's
        p-value is at most ``alpha``. ``0.001`` is the paper's value at every sample size. A larger
        value keeps more of the true edges a small sample tests weakly, and more false edges with
        them: with few rows, a larger value found a few more true adjacencies and more false ones.
    max_parents : int or None, default=None
        A cap on every node's parents, at least ``0``: the order search adds no edge that gives a
        node more, so no node of the returned DAG has more than ``max_parents`` parents. A cap can
        stop the search short of the best-scoring DAG. ``None`` sets no cap, as in the paper.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. Forbidden edges, edges into an
        earlier tier, and pairs outside the allowed set are never possible parents; the boosting
        that preselects the possible parents reads the data alone. The search starts from the
        required edges and the pruning never removes one. The causal order puts the cause of each
        required edge, and every earlier tier, first.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"cam"``. ``ordering`` is a causal order of the search's DAG before pruning, as a tuple of
        column indices, causes first. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a usable 2-D
        matrix: not numeric, holding ``NaN`` or ``inf``, or with a constant column, a column of
        fewer than 3 distinct values, or a column that repeats another up to scale and shift, each
        named in the message. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists; if its required edges give a node more than
        ``max_parents`` parents; or if a column's required parents need more coefficients than
        ``data`` has rows, naming the column.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** CAM suits i.i.d. continuous observational data whose effects are nonlinear and
    additive in the parents, with additive Gaussian noise and no unmeasured common causes [1]_;
    under that model the DAG is identified [6]_ [7]_, and it returns the DAG with a causal order.
    Its preselection of each variable's possible parents is designed for many variables with few
    parents each [1]_. With linear effects and Gaussian noise only the Markov equivalence class is
    identified, and the orientation of the edges it leaves undirected is arbitrary: use
    :func:`~andrey.pc` or :func:`~andrey.ges`. Noise whose scale depends on the parents
    (location-scale data) can reverse edges, because the model's noise variance is fixed.

    **The steps.** The paper's settings, in its three steps [1]_:

    1. *Neighbor preselection.* Each variable is regressed on all the others by componentwise L2
       boosting [2]_, 100 iterations of step size 0.1, with one P-spline base learner per other
       variable [3]_: a cubic B-spline basis on 20 equidistant interior knots over the variable's
       range, a second-order difference penalty, and the penalty weight that gives the learner 4
       degrees of freedom. The possible parents are the ten variables picked most often, among
       those picked at least three times. When more than ten qualify, the ones picked more often
       than the eleventh stay, so a tie at the tenth place keeps fewer than ten rather than
       choosing by column order. When more than ten tie at the top count, none is picked more
       often than the eleventh, and the variable has no possible parents.
    2. *Order search.* From the empty graph, the search adds the edge ``i -> j`` whose fit most
       lowers the log residual variance of ``j``, among the possible parents that close no cycle,
       whatever the sign of the change, and refits only ``j``'s remaining candidates, until none
       is left. Each fit is an additive model of penalized thin plate regression splines with ten
       basis functions per parent and smoothing chosen by GCV [4]_; with fewer than 30 rows per
       parent each term gets ``ceil(n_samples / (3 n_parents))`` functions, at least 3, and a
       variable with fewer distinct values than that gets one per distinct value. The paper states
       no cap on a variable's parents, so ``max_parents`` sets none by default; a candidate whose
       model would have as many coefficients as rows is never added.
    3. *Pruning.* Each variable is fit on its parents in that DAG, and a parent stays when its
       smooth term's p-value [5]_ is at most ``alpha``, the paper's rule.

    The rescaling of a column changes no step: the base learners and the smooth terms read a
    variable through an affine map to a fixed interval, and each gain is a ratio of residual
    variances. The fit draws no random numbers and runs on one BLAS thread.

    **Ties.** Gains are compared rounded to 9 decimal places, as :func:`~andrey.hc` compares its
    score changes per row, so two gains that differ only in their rounding tie. A tie goes to the
    edge whose cause, then effect, has the lower rank computed from the data, as for
    :func:`~andrey.hc`, and so do the variables the causal order could place next; the fit runs on
    the columns in that rank order, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` says what the
    rank keeps under a new column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** On simulated additive-noise data with many rows and sparse graphs, its errors
    stay below the empty graph's into the thousands of variables, though they grow faster than the
    variables, nearly all extra edges, and large fits are slow: the neighbor preselection's cost
    grows with the square of the number of variables. With few rows per variable it keeps false
    edges: on independent columns with fewer rows than variables it made more errors than the empty
    graph. Outside its model the result can be wrong: noise whose scale changes with the parents
    reverses many edges, linear Gaussian data leave the direction of an edge arbitrary where the
    Markov equivalence class leaves it undirected, a hidden common cause joins the variables it
    drives, interacting effects such as the product of two causes are missed or reversed, and a
    strongly skewed cause can be reversed. A column with fewer than 3 distinct values, binary
    columns included, raises ``ValueError``, as do text labels; integer category codes are read as
    numbers, with a :class:`~andrey.DataTypeWarning`. The
    `0.13.0 release card <https://andrey.abel.ai/docs/benchmarks/0.13.0.html#cam>`_ compares it with
    its authors' code. The measured results are a reference, not a hard limit: they change with the
    data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: Causal additive models,
       high-dimensional order search and penalized regression. Annals of Statistics 42(6),
       2526-2556.
    .. [2] Buhlmann, P., and Hothorn, T. (2007). Boosting algorithms: Regularization, prediction
       and model fitting. Statistical Science 22(4), 477-505.
    .. [3] Eilers, P. H. C., and Marx, B. D. (1996). Flexible smoothing with B-splines and
       penalties. Statistical Science 11(2), 89-121.
    .. [4] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.
    .. [5] Wood, S. N. (2013). On p-values for smooth components of an extended generalized
       additive model. Biometrika 100(1), 221-228.
    .. [6] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2009). Nonlinear
       causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [7] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. Journal of Machine Learning Research 15, 2009-2053.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import cam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.standard_normal(300)
    >>> out = cam(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.cam import cam as _cam

    X = data_matrix(data, method="cam")
    labels = column_labels(data)
    _smooth_columns(X, labels, "cam")
    constraints = resolve(knowledge, X.shape[1], labels, method="cam")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        _required_rows(X, constraints.required, labels)
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _cam(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
            max_parents=max_parents,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


#: Fewest rows of the HSIC test, whose null variance has the factor ``(n - 4)(n - 5)``.
MIN_ROWS = 6


def _regression_rows(X: np.ndarray, method: str, parents: int | None = None) -> None:
    """Raise ``ValueError`` when a fit on its most parents has as many coefficients as rows.

    A fit of one variable on ``m`` others, each smooth term of at least 3 basis functions, 2
    coefficients after its centering constraint, plus the intercept, has ``2m + 1`` coefficients,
    which need ``2m + 2`` rows. ``m`` is ``parents``, at most ``d - 1``; ``None`` means ``d - 1``.
    """
    n, d = X.shape
    m = d - 1 if parents is None else min(parents, d - 1)
    if n < 2 * (m + 1):
        others = f"the other {d - 1}" if m == d - 1 else f"{m} others"
        raise ValueError(
            f"data needs at least {2 * (m + 1)} rows (samples) over {d} variables, got {n}: "
            f"{method} can regress a variable on {others} with a smooth term of at least 2 "
            "coefficients each, plus an intercept, which needs more rows than coefficients"
        )


@method_spec("resit")
def resit(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with RESIT, regression with subsequent independence tests (additive noise).

    RESIT [1]_ models each variable as a function of its parents plus noise independent of its
    non-descendants, ``x_j = f_j(parents) + e_j``, which identifies the DAG when the functions are
    nonlinear or the noise is not Gaussian [1]_ [3]_. It finds a causal order from the sink up,
    taking as the next sink the variable whose additive-model residual is most independent of the
    others by an HSIC test [2]_, then removes each parent whose removal leaves the residual
    independent of the variable's predecessors. It returns a DAG and a causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of the edge-removal tests, in ``(0, 1)``: a parent goes when the residual
        without it is independent of the variable's predecessors at a p-value above ``alpha``. The
        causal order does not read it. ``0.05`` is the paper's value.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A variable is the next sink only
        when no remaining variable is in a later tier or is the effect of a required edge out of
        it. Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are in
        no regression and never in the result; each test still reads every remaining variable or
        predecessor. The edge removal never removes a required edge.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"resit"``. ``ordering`` is the causal order of the sink search, as a tuple of column
        indices, causes first. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
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
        matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than 6 rows or fewer than twice
        as many rows as columns, or with a constant column, a column of fewer than 3 distinct
        values, or a column that repeats another up to scale and shift, each named in the message.
        Also if ``knowledge`` names a column ``data`` lacks or contradicts itself, as
        :class:`~andrey.Knowledge` lists.
    FloatingPointError
        If an HSIC test gives a non-finite value, rather than let it steer a choice.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** RESIT suits i.i.d. continuous observational data whose effects are nonlinear
    and additive in the parents, with additive noise of any distribution and no unmeasured common
    causes [1]_; it reads the noise through independence tests, not a likelihood, so the noise need
    not be Gaussian, but noise whose scale depends on the parents breaks the independence of the
    residuals. Each test of the causal order reads all the remaining variables jointly, so it is
    designed for a handful of variables [1]_. It returns a DAG and a causal order. With linear
    effects and Gaussian noise only the Markov equivalence class is identified: use
    :func:`~andrey.pc` or :func:`~andrey.ges`; for more variables with Gaussian noise, use
    :func:`~andrey.cam`, and with hidden variables, :func:`~andrey.camuv`.

    **The steps.** The paper's settings, in its two phases [1]_:

    1. *Causal order.* Each remaining variable is regressed on the other remaining ones and its
       residual tested against them jointly; the variable whose residual has the largest p-value
       is the next sink, and the variables still remaining are its possible parents.
    2. *Edge removal.* Each variable's possible parents are tried one at a time, in the rank
       order of **Ties**: a parent goes when the residual of the fit on the other current parents
       has a p-value above ``alpha`` against all the variable's predecessors in the order. With one
       parent left, the residual without it is the variable itself.

    Each regression is an additive model of penalized thin plate regression splines with ten
    basis functions per term and smoothing chosen by GCV [4]_, with
    ``ceil(n_samples / (3 n_parents))`` functions per term, at least 3, when there are fewer than
    30 rows per term, and at most one per distinct value. Each test is the HSIC test [2]_ with
    Gaussian kernels on the residual and on the other side's joint squared distances, each
    bandwidth the median of the positive squared distances over every pair of rows, and the null
    approximated by a gamma law matched to its mean and variance.

    The kernels read the columns centered and divided by their standard deviations, and each smooth
    term reads a variable through an affine map to a fixed interval, so rescaling or shifting a
    column changes no step; the centering also keeps a large offset from canceling the distances'
    digits. The fit draws no random numbers and runs on one BLAS thread.

    **Ties.** p-values are compared as their logarithms rounded to 9 decimal places, as
    :func:`~andrey.hc` compares its score changes per row, so two p-values that differ only in
    their rounding tie. The logarithms are computed directly, so p-values below the smallest
    double, about ``1e-308``, which strongly dependent residuals reach with a few thousand rows,
    keep their order. A tie goes to the variable with the lower rank computed from the data, as for
    :func:`~andrey.hc`, and the edge removal tries the parents in that rank order; the fit runs on
    the columns in that rank order, so the result does not depend on the column order unless two
    columns are identical. The :doc:`column-order guide </docs/guides/column-order>` says what the
    rank keeps under a new column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Its accuracy holds only on small graphs: on Gaussian additive-noise data with
    many rows its errors come near the empty graph's on moderate graphs and exceed them on larger
    ones. Its cost grows faster than the cube of the number of variables, so it suits small graphs,
    and its time and memory grow with the square of ``n_samples``. With many rows every removal test
    can reject: on the Sachs data it kept every possible edge. Noise whose scale changes with the
    parents and linear Gaussian data lie outside its model, and there its errors were above the
    empty graph's in most measured settings. A cause with a long tail can reverse its edge;
    replacing the cause by its ranks before the fit corrected this in the measured pairs. A column
    with fewer than 3 distinct values, binary columns included, raises ``ValueError``. The
    `0.13.0 release card <https://andrey.abel.ai/docs/benchmarks/0.13.0.html#resit>`_ compares it
    with its authors' code. The measured results are a reference, not a hard limit: they change with
    the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. Journal of Machine Learning Research 15, 2009-2053.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., and Smola, A. J. (2007).
       A kernel statistical test of independence. NeurIPS 20, 585-592.
    .. [3] Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., and Scholkopf, B. (2009). Nonlinear
       causal discovery with additive noise models. NeurIPS 21, 689-696.
    .. [4] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import resit
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.standard_normal(300)
    >>> out = resit(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.resit import resit as _resit

    X = data_matrix(data, method="resit", min_rows=MIN_ROWS)
    labels = column_labels(data)
    _smooth_columns(X, labels, "resit")
    _regression_rows(X, "resit")
    constraints = resolve(knowledge, X.shape[1], labels, method="resit")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _resit(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


@method_spec("cpcm_graph")
def cpcm_graph(
    data: npt.ArrayLike,
    *,
    family: str | list[str] = "gaussian",
    edge_penalty: float = 2.0,
    seed: int | None = None,
) -> StructureOutput:
    """Learn a DAG with CPCM, conditionally parametric causal models, by a penalized score.

    CPCM [1]_ models each variable given its parents by a known parametric family whose parameters
    are smooth functions of the parents, such as a Gaussian whose mean and scale change with them,
    an exponential or Pareto law whose rate does, or a Poisson count. A DAG's residuals are each
    variable's values mapped through its fitted distribution function; the score of a DAG is minus
    the logarithm of the p-value of a test that the residuals are jointly independent [2]_, plus a
    penalty per edge. The DAG of lowest score is returned.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample, at least 2 columns. Coerced to ``float64``. A
        DataFrame's column names become the structure's ``labels``.
    family : str or list of str, default="gaussian"
        The family of each variable given its parents, one name for every column or a list of one
        name per column: ``"gaussian"`` (mean and scale change with the parents),
        ``"gaussian_fixed_scale"`` (the mean alone), ``"exponential"``, ``"pareto"`` (scale 1,
        values greater than 1), ``"poisson"``, ``"gamma"`` (mean and shape), ``"gamma_fixed_shape"``
        (the mean alone), ``"gumbel"`` (location and scale), ``"gumbel_fixed_scale"`` (the
        location alone), or ``"negative_binomial"`` (the mean, one size for all rows). A
        variable without parents is not modeled. See **Families.** in :func:`~andrey.cpcm`.
    edge_penalty : float, default=2.0
        Added to the score per edge, at least ``0``. ``2`` is the paper's value. A larger penalty
        keeps fewer edges.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed for the random draws of the count families,
        ``"poisson"`` and ``"negative_binomial"``; the other families draw nothing.

    Returns
    -------
    StructureOutput
        Contains a :class:`~andrey.core.GraphStructure` of kind ``dag``. ``metadata["algorithm"]``
        is ``"cpcm_graph"``, ``metadata["p_value"]`` the p-value of the joint independence test of
        the returned DAG's residuals, ``metadata["score"]`` its score, ``metadata["search"]``
        ``"exact"`` or ``"greedy"``, and ``metadata["families"]`` the family of each column.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` has fewer than 2
        columns, holds ``NaN`` or ``inf``, or has fewer than ``4 d + 12`` rows for ``d`` columns, if
        ``family`` names an unknown family or does not give one name per column, or if a column
        holds a value outside its family's support, such as a non-integer under ``"poisson"`` or a
        value of at most 1 under ``"pareto"``. Also if a column's residual given a parent set the
        search reads is constant, as when the column is an exact function of them, or if a fit ends
        away from a maximum; the message names the columns.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** CPCM learns a small DAG from i.i.d. observations when each variable given its
    parents follows a known parametric family, with no hidden common cause [1]_. It is designed for
    effects whose whole distribution changes with their causes, such as counts, positive or
    heavy-tailed variables, or a mix where each column gets its own family; the paper claims it for
    small graphs [1]_. On the paper's graphs, with the families its rule picks from each column's
    support, it makes as few errors as the code published with the paper and runs faster. For a
    pair, :func:`~andrey.cpcm` runs the paper's pair test; for nonlinear effects with additive
    noise, use :func:`~andrey.resit` or :func:`~andrey.cam`.

    Each variable with parents is fitted by its family on them and mapped through the fitted
    distribution function, as in :func:`~andrey.cpcm`; a variable without parents is its own
    residual. The score of a DAG is ``-log p + edge_penalty * edges``, ``p`` the gamma dHSIC test
    of the residuals' joint independence with Gaussian kernels whose bandwidth is the median
    heuristic over all rows [2]_. Up to 4 variables every DAG is scored, the paper's exact search,
    and the first lowest score wins, the DAGs ordered by their number of edges. Above, the paper's
    RESIT-greedy search: a causal order from the sink up, each sink the variable whose residual on
    the remaining ones is most independent of them by the gamma HSIC test, the remaining ones read
    jointly with each column scaled to unit standard deviation, then the complete DAG of the order
    loses, one at a time, the edge whose removal lowers the score most, while one does [1]_ [3]_.
    The columns are read in rank order, so the result does not depend on their order.

    **Units.** As for :func:`~andrey.cpcm`: the result does not depend on any column's units under
    the Gaussian, Gumbel, exponential, and gamma families, of which the last two read its origin;
    ``"pareto"`` and the count families read the units, since they fix a scale.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** On the paper's graphs its graph had as many errors as the code published with
    the paper and fewer than the empty graph; the paper's largest graphs were not measured.
    :func:`~andrey.pc` with ``indep_test="kci"`` was close in accuracy there, and faster. The
    `0.19.0 release card <https://andrey.abel.ai/docs/benchmarks/0.19.0.html#cpcm-graph-search>`_
    has the numbers. The exact search scores every DAG and runs up to 4 variables; above, the greedy
    search refits a variable for each parent set it tries, and memory grows with the square of
    ``n_samples``. Under the count families, large means that vary strongly with the parents make
    the true graph's test reject too. The family must be chosen: the paper's choice of a family from
    each column's support, and its sequential choice between one- and two-parameter families, are
    not offered. The measured results are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Bodik, J., and Chavez-Demoulin, V. (2025). Identifiability of causal graphs under
       non-additive conditionally parametric causal models. JMLR 26(264), 1-55.
    .. [2] Pfister, N., Buhlmann, P., Scholkopf, B., and Peters, J. (2018). Kernel-based tests for
       joint independence. JRSS-B 80(1), 5-31.
    .. [3] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. JMLR 15, 2009-2053.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import cpcm_graph
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(0, 1, 300)
    >>> y = rng.exponential(1 + x**2)
    >>> z = rng.exponential(1 + np.sqrt(y))
    >>> family = ["gaussian_fixed_scale", "exponential", "exponential"]
    >>> out = cpcm_graph(np.column_stack([x, y, z]), family=family, seed=0)
    >>> out.structure.oriented_edges()
    [(0, 1, 'directed'), (1, 2, 'directed')]
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.cpcm import Residuals, resolve_families, search
    from andrey.fcm.cpcm import score as cpcm_score

    X = data_matrix(data, method="cpcm_graph")
    labels = column_labels(data)
    min_rows = 4 * X.shape[1] + 12
    if X.shape[0] < min_rows:
        raise ValueError(f"data needs at least {min_rows} rows (samples), got {X.shape[0]}")
    if X.shape[1] < 2:
        raise ValueError(f"cpcm_graph needs at least 2 columns, got {X.shape[1]}")
    name = column_namer(labels)
    families = resolve_families(family, X, "cpcm_graph", name)
    assert seed is not None  # method_spec resolves it
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    res = Residuals(
        X[:, order],
        [families[j] for j in order],
        seed,
        method="cpcm_graph",
        name=lambda j: name(int(order[j])),
    )
    with one_blas_thread():
        parents, _, log_p = search(res, edge_penalty)
        value = cpcm_score(res, parents, edge_penalty)
    dag = np.zeros((X.shape[1], X.shape[1]), dtype=np.int8)
    for v, pa in enumerate(parents):
        for p in pa:
            dag[order[p], order[v]] = 1
    return structure_output(
        dag_from_adjacency(dag),
        labels=labels,
        metadata={
            "p_value": math.exp(log_p),
            "score": value,
            "search": "exact" if X.shape[1] <= 4 else "greedy",
            "families": [f.name for f in families],
        },
    )


@method_spec("score_matching")
def score_matching(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.001,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with SCORE, a causal order from the Jacobian of the score (additive noise).

    SCORE [1]_ models each variable as a nonlinear function of its parents plus Gaussian noise,
    ``x_j = f_j(parents) + e_j``. Under that model the score of the joint density,
    ``grad log p(x)``, marks the leaves: the derivative of its entry ``j`` in ``x_j`` is the same
    at every point exactly when ``j`` has no children [1]_. SCORE estimates those derivatives at
    the rows by a kernel estimator built on Stein's identity [2]_, takes the variable whose
    derivative varies least as the next leaf, repeats on the others, and prunes the complete DAG
    of the order by the smooth-term tests of CAM [3]_. It returns a DAG and a causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.001
        Significance level of the pruning test, in ``(0, 1)``: a parent stays when its smooth term's
        p-value is at most ``alpha``. The causal order does not read it. ``0.001`` is the value of
        the paper's experiments.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A variable is the next leaf only
        when no remaining variable is in a later tier or is the effect of a required edge out of
        it. Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are in
        no pruning fit and never in the result; the score estimate still reads every remaining
        variable. The pruning never removes a required edge.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"score_matching"``. ``ordering`` is the causal order of the leaf search, as a tuple of
        column indices, causes first. ``metadata["tie_break"]`` is the tie-break rule the fit ran
        under, ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value. Also if ``knowledge`` is neither
        ``None`` nor a :class:`~andrey.Knowledge`.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value. Also if ``data`` is not a usable 2-D
        matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than twice as many rows as
        columns, or with a constant column, a column of fewer than 3 distinct values, or a column
        that repeats another up to scale and shift, each named in the message; or if more than
        half of the pairs of rows are equal on the remaining variables at a step, which leaves the
        kernel no bandwidth. Also if ``knowledge`` names a column ``data`` lacks or contradicts
        itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** SCORE suits i.i.d. continuous observational data whose effects are nonlinear in
    every parent, with additive Gaussian noise and no unmeasured common causes [1]_; under that
    model the DAG is identified [4]_, and it returns the DAG with a causal order. Its order search
    fits no regression: each step is one kernel estimate over all the remaining variables, so its
    cost grows with the number of variables only linearly, and with the cube of the rows. A link
    that is linear in a parent leaves that parent's derivative constant too, so it can be taken for
    a leaf: with linear effects and Gaussian noise use :func:`~andrey.pc` or :func:`~andrey.ges`,
    which return the Markov equivalence class. For additive noise that is not Gaussian use
    :func:`~andrey.resit`, and for the same model searched by regressions, :func:`~andrey.cam`.

    **The steps.** The paper's two phases [1]_:

    1. *Causal order.* The score's Jacobian has the diagonal entries
       ``d^2 log p(x) / d x_j^2``. They are estimated at every row of the remaining variables by
       the paper's extension of the Stein gradient estimator [2]_ to second derivatives (its
       Eq. 12), with the Gaussian kernel ``exp(-||x - y||^2 / (2 s^2))``, the bandwidth ``s`` the
       median distance between the rows, and one ridge term for both of its regressions. The
       variable whose entry has the smallest variance over the rows is the next leaf; it is
       removed, and the estimate, bandwidth included, is made again on the others. The causal
       order is the leaves in reverse.
    2. *Pruning.* Each variable is fit on all the variables before it in the order, an additive
       model of penalized thin plate regression splines with ten basis functions per parent and
       smoothing chosen by GCV [5]_, with ``ceil(n_samples / (3 n_parents))`` functions per term,
       at least 3, when there are fewer than 30 rows per term, and at most one per distinct
       value. A parent stays when its smooth term's p-value is at most ``alpha``, the pruning of
       :func:`~andrey.cam` [3]_.

    The choices the paper leaves open. It states no value of the ridge term: it is ``0.001``, the
    value in the code published with the paper, added to the kernel matrix as the paper's Eq. 12
    writes it, a matrix with ones on its diagonal. The median distance is taken over the pairs of
    distinct rows. The variances are compared as estimated, as the paper's Algorithm 1 states, not
    divided by the entries' means.

    One kernel reads all the remaining columns, so the estimate depends on their units, which the
    model does not: a leaf's entry is constant in any units. The columns are therefore centered
    and divided by their standard deviations before the order search, so rescaling or shifting a
    column changes no step; the pruning reads each variable through an affine map to a fixed
    interval. The paper's experiments run on the columns as simulated. The fit draws no random
    numbers and runs on one BLAS thread.

    **Ties.** The variances are compared rounded to 9 decimal places, as :func:`~andrey.hc`
    compares its score changes per row, so two variances that differ only in their rounding tie. A
    tie goes to the variable with the lower rank computed from the data, as for
    :func:`~andrey.hc`; the fit runs on the columns in that rank order, so the result does not
    depend on the column order unless two columns are identical. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** On data simulated as in the paper its errors stay below the empty graph's at
    every measured size, but on sparse graphs, where the order search orders fewer edges correctly
    on the standardized columns it reads than on the columns as simulated, :func:`~andrey.cam` was
    more accurate. Where links have a strong linear part, such as ``z + sin z``, the search ranks
    the sinks last: its order was close to a random one, and its graph could be farther from the
    truth than the empty graph; more rows or other kernel settings did not change the order, and
    :func:`~andrey.cam` suits such data. Relations poorly resolved by a single kernel bandwidth can
    make causes appear as leaves, and the leaf rule is exact only for Gaussian noise;
    :func:`~andrey.nogam` and :func:`~andrey.resit` drop that assumption. The order search's time
    grows with the cube of ``n_samples`` and its memory with the square. The pruning needs at least
    twice as many rows as columns: fewer rows, or a column with fewer than 3 distinct values, binary
    columns included, raise ``ValueError``. The
    `0.17.0 release card <https://andrey.abel.ai/docs/benchmarks/0.17.0.html#score-and-nogam>`_ has
    the numbers. The measured results are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Rolland, P., Cevher, V., Kleindessner, M., Russell, C., Janzing, D., Scholkopf, B., and
       Locatello, F. (2022). Score matching enables causal discovery of nonlinear additive noise
       models. ICML, PMLR 162, 18741-18753.
    .. [2] Li, Y., and Turner, R. E. (2018). Gradient estimators for implicit models. ICLR.
    .. [3] Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: Causal additive models,
       high-dimensional order search and penalized regression. Annals of Statistics 42(6),
       2526-2556.
    .. [4] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. Journal of Machine Learning Research 15, 2009-2053.
    .. [5] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import score_matching
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(300)
    >>> y = np.sin(2 * x) + 0.3 * rng.standard_normal(300)
    >>> out = score_matching(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.score_matching import score_matching as _score_matching

    X = data_matrix(data, method="score_matching")
    labels = column_labels(data)
    _smooth_columns(X, labels, "score_matching")
    _regression_rows(X, "score_matching")
    constraints = resolve(knowledge, X.shape[1], labels, method="score_matching")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _score_matching(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


@method_spec("nogam")
def nogam(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.001,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with NoGAM, a causal order from the score and residuals (additive noise).

    NoGAM [1]_ models each variable as a nonlinear function of its parents plus noise of any
    distribution, ``x_j = f_j(parents) + e_j``. Under that model the entry ``j`` of the score of
    the joint density, ``d log p(x) / d x_j``, is a function of the variable's noise alone exactly
    when ``j`` has no children, and that noise is what a regression of ``x_j`` on all the other
    variables leaves [1]_. NoGAM estimates the score at the rows by a kernel estimator built on
    Stein's identity [2]_, predicts each entry from its variable's regression residual, takes the
    variable predicted with the smallest error as the next leaf, repeats on the others, and prunes
    the complete DAG of the order by the smooth-term tests of CAM [3]_. It returns a DAG and a
    causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.001
        Significance level of the pruning test, in ``(0, 1)``: a parent stays when its smooth term's
        p-value is at most ``alpha``. The causal order does not read it. ``0.001`` is the value of
        the paper's experiments.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A variable is the next leaf only
        when no remaining variable is in a later tier or is the effect of a required edge out of
        it. Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are in
        no pruning fit and never in the result; the score estimate and the residual regressions
        still read every remaining variable. The pruning never removes a required edge.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"nogam"``. ``ordering`` is the causal order of the leaf search, as a tuple of column
        indices, causes first. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
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
        matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than 5 rows or fewer than twice
        as many rows as columns, or with a constant column, a column of fewer than 3 distinct
        values, or a column that repeats another up to scale and shift, each named in the message;
        or if more than half of the pairs of rows are equal on the remaining variables at a step,
        which leaves the score estimate's kernel no bandwidth. Also if ``knowledge`` names a column
        ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** NoGAM suits i.i.d. continuous observational data whose effects are nonlinear,
    with additive noise of any distribution and no unmeasured common causes [1]_; under that model
    the DAG is identified [4]_, and it returns the DAG with a causal order. It does not assume
    Gaussian noise, as :func:`~andrey.score_matching` and :func:`~andrey.cam` do, and its order
    search needs no independence test, as :func:`~andrey.resit`'s does. With linear effects and
    Gaussian noise only the Markov equivalence class is identified: use :func:`~andrey.pc` or
    :func:`~andrey.ges`.

    **The steps.** The paper's two phases [1]_:

    1. *Causal order.* On the remaining variables, three estimates are made at every row. The
       score ``grad log p(x)``, by the Stein gradient estimator [2]_ with the Gaussian kernel
       ``exp(-||x - y||^2 / (2 s^2))``, the bandwidth ``s`` the median distance between the rows.
       Each variable's residual, from its kernel ridge regression on all the other remaining
       variables. And each score entry's prediction from its own variable's residual, by another
       kernel ridge regression. The variable whose score entry has the smallest mean squared
       prediction error is the next leaf; it is removed, and the estimates are made again on the
       others. The causal order is the leaves in reverse.
    2. *Pruning.* Each variable is fit on all the variables before it in the order, an additive
       model of penalized thin plate regression splines with ten basis functions per parent and
       smoothing chosen by GCV [5]_, with ``ceil(n_samples / (3 n_parents))`` functions per term,
       at least 3, when there are fewer than 30 rows per term, and at most one per distinct
       value. A parent stays when its smooth term's p-value is at most ``alpha``, the pruning of
       :func:`~andrey.cam` [3]_.

    The kernel ridge regressions take the settings of the paper's experiments: the kernel
    ``exp(-gamma ||x - y||^2)`` with ``gamma = 0.1``, the ridge term ``0.01``, no intercept, and
    predictions by K-fold cross-validation, each row predicted by the model fit on the other
    folds. ``gamma`` is the value the paper states; the authors' implementation sets it to the
    ridge term's value, ``0.01``.

    The choices the paper leaves open. It names no number of folds: there are 5, each a block of
    consecutive rows, so the order of the rows can change the result. It states the
    cross-validation for the score entries; the residuals are predicted across the same folds.
    It states no ridge term for the score estimate: it is ``0.001``, the value in the authors'
    implementation, added to the kernel matrix as the estimator's definition writes it, a matrix
    with ones on its diagonal; the median distance is taken over the pairs of distinct rows. The
    errors are compared as estimated, not divided by the entries' variances.

    The score estimate and the kernel ridge regressions read the columns' units, which the model
    does not. The columns are therefore centered and divided by their standard deviations before
    the order search, so rescaling or shifting a column changes no step; the pruning reads each
    variable through an affine map to a fixed interval. The paper's experiments run on the columns
    as simulated. The fit draws no random numbers and runs on one BLAS thread.

    **Ties.** The errors are compared rounded to 9 decimal places, as :func:`~andrey.hc` compares
    its score changes per row, so two errors that differ only in their rounding tie. A tie goes to
    the variable with the lower rank computed from the data, as for :func:`~andrey.hc`; the fit
    runs on the columns in that rank order, so the result does not depend on the column order
    unless two columns are identical. The :doc:`column-order guide </docs/guides/column-order>`
    says what the rank keeps under a new column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** On data simulated as in the paper, under Gaussian, Laplace, and Gumbel noise,
    its errors stay below the empty graph's on small graphs; larger graphs were not measured. On
    sparse graphs, where the order search orders fewer edges correctly on the standardized columns
    it reads than on the columns as simulated, :func:`~andrey.cam` was at least as accurate under
    every noise. Where links have a strong linear part, such as ``z + sin z``, the search ranks the
    sinks last: its order was close to a random one, and its graph could be farther from the truth
    than the empty graph; more rows changed the order little, and :func:`~andrey.cam` suits such
    data. The order search's time grows with the square of ``n_variables`` and the cube of
    ``n_samples``, and its memory with the square of ``n_samples``. The pruning needs at least twice
    as many rows as columns: fewer rows, or a column with fewer than 3 distinct values, binary
    columns included, raise ``ValueError``. The
    `0.17.0 release card <https://andrey.abel.ai/docs/benchmarks/0.17.0.html#score-and-nogam>`_ has
    the numbers. The measured results are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Montagna, F., Noceti, N., Rosasco, L., Zhang, K., and Locatello, F. (2023). Causal
       discovery with score matching on additive models with arbitrary noise. CLeaR, PMLR 213,
       726-751.
    .. [2] Li, Y., and Turner, R. E. (2018). Gradient estimators for implicit models. ICLR.
    .. [3] Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: Causal additive models,
       high-dimensional order search and penalized regression. Annals of Statistics 42(6),
       2526-2556.
    .. [4] Peters, J., Mooij, J. M., Janzing, D., and Scholkopf, B. (2014). Causal discovery with
       continuous additive noise models. Journal of Machine Learning Research 15, 2009-2053.
    .. [5] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import nogam
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.uniform(-1, 1, 300)
    >>> out = nogam(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.nogam import FOLDS
    from andrey.fcm.nogam import nogam as _nogam

    X = data_matrix(data, method="nogam", min_rows=FOLDS)
    labels = column_labels(data)
    _smooth_columns(X, labels, "nogam")
    _regression_rows(X, "nogam")
    constraints = resolve(knowledge, X.shape[1], labels, method="nogam")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _nogam(
            X[:, order],
            alpha=alpha,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


@method_spec("das")
def das(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.001,
    max_parents: int | None = 20,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a DAG with DAS, SCORE's causal order with edges screened from the score's Jacobian.

    DAS [1]_ (Discovery At Scale) assumes the model of :func:`~andrey.score_matching`: each
    variable is a nonlinear function of its parents plus Gaussian noise, ``x_j = f_j(parents) +
    e_j``. It finds SCORE's causal order [2]_ and then reads the edges from the same estimate of
    the score's Jacobian: for a leaf ``l``, the entry ``d^2 log p(x) / d x_l d x_j`` is zero at
    every point exactly when ``j`` is not a parent [1]_. Each leaf keeps the variables before it
    whose entries are clearly above zero, at most ``max_parents`` of them, and the smooth-term
    tests of CAM [3]_ prune those candidates only, not every variable before it in the order,
    which makes the pruning fast for many variables. It returns a DAG and a causal order.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.001
        Significance level of the pruning test, in ``(0, 1)``: a candidate parent stays when its
        smooth term's p-value is at most ``alpha``. Neither the causal order nor the edge screen
        reads it. ``0.001`` is the cutoff of the SCORE experiments the paper follows.
    max_parents : int or None, default=20
        A cap on every node's parents, at least ``0``: the edge screen keeps at most
        ``max_parents`` candidate parents per variable, required edges included, so no node of the
        returned DAG has more. ``20`` is the paper's value. ``None`` sets no cap.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A variable is the next leaf only
        when no remaining variable is in a later tier or is the effect of a required edge out of
        it. Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are
        never candidates, but the edge screen still ranks them and may take one as its reference;
        the score estimate still reads every remaining variable. A required edge is a candidate
        that the pruning never removes.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure`; ``metadata["algorithm"]`` is
        ``"das"``. ``ordering`` is the causal order of the leaf search, as a tuple of column
        indices, causes first; it is the order :func:`~andrey.score_matching` finds on the same
        data. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
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
        matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than ``2 m + 2`` rows for
        pruning fits on ``m = min(max_parents, n_variables - 1)`` parents, or with a constant
        column, a column of fewer than 3 distinct values, or a column that repeats another up to
        scale and shift, each named in the message; or if more than half of the pairs of rows are
        equal on the remaining variables at a step, which leaves the kernel no bandwidth. Also if
        ``knowledge`` names a column ``data`` lacks, contradicts itself, as
        :class:`~andrey.Knowledge` lists, or requires more than ``max_parents`` parents of a
        variable.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** DAS suits i.i.d. continuous observational data whose effects are nonlinear in
    every parent, with additive Gaussian noise and no unmeasured common causes [1]_, when there are
    too many variables for :func:`~andrey.score_matching`'s pruning: it finds the same causal order
    and prunes at most ``max_parents`` candidates per variable instead of every variable before it.
    The paper shows a DAG close to SCORE's from a fit several times faster on all but small graphs.
    It is the fastest of the methods here for this model. :func:`~andrey.cam` made fewer errors on
    the same data in nearly every measured setting and took longer on all but the smallest sparse
    graphs; use it when time allows. With fewer variables :func:`~andrey.score_matching` prunes
    every edge its order allows; with linear effects and Gaussian noise use :func:`~andrey.pc` or
    :func:`~andrey.ges`, and for additive noise that is not Gaussian, :func:`~andrey.nogam` or
    :func:`~andrey.resit`.

    **The steps.** The paper's three phases [1]_:

    1. *Causal order.* The leaf search of :func:`~andrey.score_matching`: the diagonal of the
       score's Jacobian is estimated at every row of the remaining variables by the Stein
       estimator of [2]_ (Gaussian kernel at the median distance between the rows), the variable
       whose entry varies least is the next leaf, and the estimate is made again on the others.
    2. *Edge screen.* At the step that picks a leaf, the variables left are the leaf and those
       before it in the order, so the leaf has no children among them, and the leaf's row of the
       step's estimate gives ``d^2 log p / d x_l d x_j`` at every row. The absolute median of the
       entry ranks the variables before the leaf. The ``max_parents + 1`` largest are kept and
       the smallest of them is the reference, standing for a zero entry; each of the others stays
       as a candidate parent when a one-sided Welch test of equal means of the absolute entries
       against the reference's gives a p-value at most ``0.01``, the paper's level.
    3. *Pruning.* Each variable is fit on its candidate parents, an additive model of penalized
       thin plate regression splines with ten basis functions per parent and smoothing chosen by
       GCV [4]_, as in :func:`~andrey.score_matching`, and a candidate stays when its smooth term's
       p-value is at most ``alpha`` [3]_.

    The choices the paper leaves open. It states no pruning cutoff and follows the experiments of
    [2]_, whose cutoff ``0.001`` is the default. The five largest medians stay without a test, the
    value of the ``min_parents`` of the code published with the paper: the paper's screen never
    keeps the reference, so without them a variable all of whose predecessors are parents would
    lose the weakest one, and a chain of two variables would have no edge. That code counts the
    leaf's own entry among the ranked ones, so it usually keeps four untested; here the ranking
    holds only the predecessors, and five stay. The ranking follows that code, the absolute
    median of the entries, where the paper states the mean of their absolute values. That code
    estimates the Jacobian once on all the variables; DAS here reads each leaf's row at the step
    that removes it, as the paper's Algorithm 1 does. A test whose two samples are both constant
    keeps no edge. The estimator's ridge term and bandwidth are those of
    :func:`~andrey.score_matching`, whose Notes state them.

    The order search reads the columns centered and divided by their standard deviations, as
    :func:`~andrey.score_matching` does, so rescaling or shifting a column changes no step; the
    pruning reads each variable through an affine map to a fixed interval. The paper's
    experiments run on the columns as simulated. The fit draws no random numbers and runs on one
    BLAS thread.

    **Ties.** The variances and the medians are compared rounded to 9 decimal places, as
    :func:`~andrey.hc` compares its score changes per row, so two values that differ only in
    their rounding tie. A tie goes to the variable with the lower rank computed from the data, as
    for :func:`~andrey.hc`; the fit runs on the columns in that rank order, so the result does not
    depend on the column order unless two columns are identical. The
    :doc:`column-order guide </docs/guides/column-order>` says what the rank keeps under a new
    column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** On the papers' simulation with z-scored or rescaled columns, DAS made no more
    errors than the code published with the paper, and fewer than the empty graph at every measured
    size, large graphs included. It reads standardized columns, so the result does not depend on
    their units; on raw columns, whose variance grows along the causal order, the code published
    with the paper made fewer errors on sparse graphs. The
    `0.19.0 release card <https://andrey.abel.ai/docs/benchmarks/0.19.0.html#das>`_ has the numbers.
    Its SCORE order has the limits stated by :func:`~andrey.score_matching`: near a random order
    where links have a strong linear part, and exact only for Gaussian noise. Pruning cannot recover
    an edge the order reversed or a true parent the screen dropped. The order search's time grows
    with the cube of ``n_samples`` and its memory with the square. A column with fewer than 3
    distinct values, binary columns included, raises ``ValueError``. The measured results are a
    reference, not a hard limit: they change with the data generator, the density, the method's
    settings, and the machine.

    References
    ----------
    .. [1] Montagna, F., Noceti, N., Rosasco, L., Zhang, K., and Locatello, F. (2023). Scalable
       causal discovery with score matching. CLeaR, PMLR 213, 752-771.
    .. [2] Rolland, P., Cevher, V., Kleindessner, M., Russell, C., Janzing, D., Scholkopf, B., and
       Locatello, F. (2022). Score matching enables causal discovery of nonlinear additive noise
       models. ICML, PMLR 162, 18741-18753.
    .. [3] Buhlmann, P., Peters, J., and Ernest, J. (2014). CAM: Causal additive models,
       high-dimensional order search and penalized regression. Annals of Statistics 42(6),
       2526-2556.
    .. [4] Wood, S. N. (2003). Thin plate regression splines. Journal of the Royal Statistical
       Society B 65(1), 95-114.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import das
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(300)
    >>> y = np.sin(2 * x) + 0.3 * rng.standard_normal(300)
    >>> out = das(np.column_stack([x, y]))
    >>> out.structure.kind, out.ordering
    ('dag', (0, 1))
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.das import das as _das

    X = data_matrix(data, method="das")
    labels = column_labels(data)
    _smooth_columns(X, labels, "das")
    _regression_rows(X, "das", max_parents)
    constraints = resolve(knowledge, X.shape[1], labels, method="das")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = precedence = None
    if constraints is not None:
        most = int(constraints.required.sum(axis=0).max(initial=0))
        if max_parents is not None and most > max_parents:
            raise ValueError(
                f"das: knowledge requires {most} parents of a variable, more than "
                f"max_parents={max_parents}"
            )
        ranked = constraints.permuted(order)
        forbidden, required, precedence = ranked.forbidden, ranked.required, ranked.precedence()
    with one_blas_thread():
        result = _das(
            X[:, order],
            alpha=alpha,
            max_parents=max_parents,
            forbidden=forbidden,
            required=required,
            precedence=precedence,
        )
    back = np.argsort(order)
    dag = result.dag[np.ix_(back, back)]
    return structure_output(dag_from_adjacency(dag), labels=labels, ordering=order[result.order])


@method_spec("camuv")
def camuv(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.01,
    max_depth: int = 2,
    knowledge: Knowledge | None = None,
) -> StructureOutput:
    """Learn a PAG of nonlinear additive effects with hidden variables by CAM-UV.

    CAM-UV [1]_ models each variable, observed or hidden, as a sum of nonlinear functions of its
    direct causes plus independent noise. Where a hidden variable lies between two observed ones,
    as a common cause or on a causal path, their relation cannot be identified by regression and
    independence tests, and CAM-UV marks the pair instead of guessing. It finds each variable's
    direct causes and those pairs.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.01
        Significance level of every HSIC independence test, in ``(0, 1)``: of the sink tests, the
        tests that remove a candidate parent, and the tests that mark a pair. ``0.01`` is the
        paper's value.
    max_depth : int, default=2
        The maximum conditioning-set size of a sink test, at least ``1``: the most variables
        tested together as a variable's new candidate parents, so the search runs over sets of up
        to ``max_depth + 1`` variables. Each sink test also regresses on the variable's candidates
        found so far, however many. Larger values find parents that only show together and take
        longer. The default ``2`` is the paper's setting, sets of at most 3 variables.
    knowledge : Knowledge or None, default=None
        Background knowledge, a :class:`~andrey.Knowledge`: required and forbidden edges, the pairs
        allowed to be adjacent, and temporal tiers, all honored. A forbidden edge, an edge into an
        earlier tier, and a pair outside ``allowed`` never make a variable a candidate parent, so a
        variable is not tried as the sink of a set that holds a forbidden parent of it, and none
        is in the result. A required edge's cause is a candidate parent from the start, and no
        step removes it; its pair is never marked. A pair forbidden in both directions is never
        marked; on a marked pair, a forbidden ``i -> j`` whose reverse is not forbidden, such as an
        edge into an earlier tier, puts an arrowhead at ``i``, as in :func:`~andrey.fci`.

    Returns
    -------
    StructureOutput
        A :class:`~andrey.core.GraphStructure` of kind ``pag``: ``i -> j`` for a direct cause and
        ``i o-o j`` for a pair joined through a hidden variable, whose relation the method cannot
        determine, with an arrowhead where ``knowledge`` sets one. ``metadata["algorithm"]`` is
        ``"camuv"``, ``metadata["latent_pairs"]`` lists the marked pairs as sorted ``[i, j]`` lists,
        and ``metadata["cycle_edges"]`` lists as ``[i, j]`` each edge ``i -> j`` dropped from a
        directed cycle. ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
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
        matrix: not numeric, holding ``NaN`` or ``inf``, with fewer than 6 rows or fewer than twice
        as many rows as columns, or with a constant column, a column of fewer than 3 distinct
        values, a column that repeats another up to scale and shift, or a column that is a linear
        function of others, such as a total, each named in the message. Also if ``knowledge`` names
        a column ``data`` lacks or contradicts itself, as :class:`~andrey.Knowledge` lists.
    FloatingPointError
        If an HSIC test gives a non-finite value, rather than let it steer a choice.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes,
        which the fit reads as numbers. Each message warns once per process.

    Notes
    -----
    **When to use.** CAM-UV suits i.i.d. continuous observational data whose effects are nonlinear
    and additive in the causes, with independent noise of any distribution, where any number of
    causes may be unmeasured, none included [1]_; with none unmeasured every ``o-o`` pair it marks
    is false. With many of the nodes hidden its directed edges were no better than the empty graph's
    (**Limitations**). Its kernel independence tests compare every pair of rows, and its search
    tests sets of up to ``max_depth + 1`` variables, so it suits a moderate number of variables and
    rows. It returns a PAG: ``i -> j`` for a direct cause and ``i o-o j`` for a pair that shares a
    hidden common cause or of which one causes the other through a hidden variable, which the method
    does not tell apart. With linear effects, a path through a hidden variable can be regressed away
    and the pairs are identified differently: use :func:`~andrey.rcd` for non-Gaussian noise and
    :func:`~andrey.fci` for Gaussian noise. Without hidden variables, :func:`~andrey.cam` and
    :func:`~andrey.resit` also give a causal order.

    **The steps.** Every pair of variables is first tested for dependence; then come the paper's
    three steps, with its settings [1]_:

    1. *Candidate parents.* Over every set of 2, then up to ``max_depth + 1`` variables, each
       member dependent on every other member is regressed on the others and its candidate
       parents, and its residual is tested against the others' residuals on their own candidates,
       jointly. The member of largest p-value is a sink of the set when that p-value is above
       ``alpha`` and its residual on its candidates alone is dependent on each other member's,
       every p-value below ``alpha``; the other members then become its candidates. After a sweep
       that adds a candidate, the search starts over from sets of 2. A set that holds a pair
       already related, one a candidate of the other, is skipped: the sink test identifies a sink
       only when each member's candidates lie outside the set.
    2. *Parents.* Each variable's candidates are tried one at a time, in the rank order of
       **Ties**: a candidate goes when the variable's residual on its other candidates is
       independent of the candidate's residual on its own, at a p-value above ``alpha``.
    3. *Hidden paths.* A dependent pair with no edge whose residuals, each on its parents, are
       dependent at a p-value below ``alpha`` is marked ``o-o``.

    The two dependence screens are not part of the paper's algorithm, which tests every member of a
    set and every pair. Under the model they add no conclusion: two independent variables share no
    noise, so their residuals are independent too, and neither a sink test nor step 3 would accept
    them. With tests at ``alpha`` they skip most sink tests: on 40 simulated datasets of 20 observed
    variables and 1,000 rows from Erdos-Renyi graphs of mean degree 2, ten each with nonlinear
    additive effects and 2, 4, or no hidden variables and ten with linear Gaussian effects and none,
    a fit ran 12 times faster with them, and its directed errors were lower on 34 of the 40 and
    higher on 1. A parent whose dependence the test misses, such as a weak parent beside a strong
    one, is not found.

    The paper does not say what to do when the parents form a directed cycle, which its model rules
    out and failed tests can leave: every edge on a directed cycle is dropped, except a required
    one, and step 3 tests the pair; :func:`~andrey.rcd` raises instead when required edges close a
    cycle with the parents it found. ``metadata["cycle_edges"]`` lists the dropped edges, and an
    INFO log names them. On those 40 datasets and on 30 more with one hidden variable per ten
    observed ones, at 10 to 50 observed variables, the drop fired in 15 of the 70 fits, and 74 of
    the 78 edges it dropped came back as ``o-o``; 24 of the 78 were true edges. The child's
    residual on its reduced parents also enters its other pairs, which can mark them too.

    Each regression is an additive model of penalized thin plate regression splines with ten basis
    functions per term and smoothing chosen by GCV [3]_, with ``ceil(n_samples / (3 n_terms))``
    functions per term, at least 3, when there are fewer than 30 rows per term, and at most one per
    distinct value. Each test is the HSIC test [2]_ with Gaussian kernels, each bandwidth the median
    of the positive squared distances over every pair of rows, a side of several residuals read
    through their joint distances, and the null approximated by a gamma law matched to its mean and
    variance.

    Every step reads the columns divided by their standard deviations, so rescaling a column
    changes no step; the paper's joint kernel reads raw residuals and so weighs each by its units,
    which is the same on standardized data. The kernels read the residuals centered, which keeps an
    offset from canceling the distances' digits. The fit draws no random numbers and runs on one
    BLAS thread.

    **Ties.** p-values are compared as their logarithms rounded to 9 decimal places, as
    :func:`~andrey.hc` compares its score changes per row, so two p-values that differ only in
    their rounding tie. The logarithms are computed directly, so p-values below the smallest
    double, about ``1e-308``, which strongly dependent residuals reach with a few thousand rows,
    keep their order. A tie goes to the variable with the lower rank computed from the data, as for
    :func:`~andrey.hc`, and the sets run in that rank order and the parents are tried in it; the fit
    runs on the columns in that rank order, so the result does not depend on the column order unless
    two columns are identical. The :doc:`column-order guide </docs/guides/column-order>` says what
    the rank keeps under a new column order.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** With many of the nodes hidden it made no fewer errors than the empty graph on
    small graphs; with no variable hidden, every pair it marks ``o-o`` is false; and with linear
    Gaussian effects it made more errors than the empty graph. With few hidden variables its errors
    stay below the empty graph's on graphs of a few dozen observed variables and rise back toward it
    as the graph grows, reaching it on graphs of hundreds of variables. A cause with a long tail, or
    a single extreme value, can turn an edge into an ``o-o`` mark; replacing a long-tailed cause by
    its ranks before the fit found the edge again in most fits. A column with fewer than 3 distinct
    values, binary columns included, raises ``ValueError``, and integer category codes are read as
    numbers, with a :class:`~andrey.DataTypeWarning`. Its cost grows with the square of the number
    of variables, so large graphs are slow, and the time and memory of its kernel tests grow with
    the square of ``n_samples``. The
    `0.14.0 release card <https://andrey.abel.ai/docs/benchmarks/0.14.0.html#cam-uv>`_ has the
    numbers. The measured results are a reference, not a hard limit: they change with the data
    generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Maeda, T. N., and Shimizu, S. (2021). Causal additive models with unobserved variables.
       UAI, PMLR 161, 97-106.
    .. [2] Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., and Smola, A. J. (2007).
       A kernel statistical test of independence. NeurIPS 20, 585-592.
    .. [3] Wood, S. N. (2004). Stable and efficient multiple smoothing parameter estimation for
       generalized additive models. Journal of the American Statistical Association 99, 673-686.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import camuv
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, 300)
    >>> y = np.sin(2 * x) + 0.3 * rng.uniform(-1, 1, 300)
    >>> out = camuv(np.column_stack([x, y]))
    >>> out.structure.kind, out.structure.oriented_edges()
    ('pag', [(0, 1, 'directed')])
    """
    from andrey.core.ties import tie_ranks
    from andrey.fcm.camuv import camuv as _camuv

    X = data_matrix(data, method="camuv", min_rows=MIN_ROWS)
    labels = column_labels(data)
    _smooth_columns(X, labels, "camuv")
    _regression_rows(X, "camuv")
    if X.shape[1] > 1:  # a column that is a linear function of others has no noise of its own
        check_columns(np.cov(X, rowvar=False), X.shape[0], column_namer(labels))
    constraints = resolve(knowledge, X.shape[1], labels, method="camuv")
    # The fit runs on the columns in rank order, so no step reads their positions.
    order = np.argsort(tie_ranks(X), kind="stable")
    forbidden = required = None
    if constraints is not None:
        ranked = constraints.permuted(order)
        forbidden, required = ranked.forbidden, ranked.required
    with one_blas_thread():
        result = _camuv(
            X[:, order], alpha=alpha, max_depth=max_depth, forbidden=forbidden, required=required
        )
    back = np.argsort(order)
    edges, pairs, dropped = (
        m[np.ix_(back, back)] for m in (result.edges, result.pairs, result.cycle_edges)
    )
    cycle_edges = [[int(i), int(j)] for i, j in zip(*np.nonzero(dropped), strict=True)]
    if cycle_edges:
        name = column_namer(labels)
        _log.info(
            "camuv: dropped %d edge(s) on directed cycles, then tested each pair for a hidden "
            "path: %s",
            len(cycle_edges),
            ", ".join(f"{name(i)} -> {name(j)}" for i, j in cycle_edges),
        )
    # B[i, j] != 0 is the edge j -> i, and a NaN pair is o-o.
    B = np.where(pairs, np.nan, edges.T.astype(np.float64))
    marks = marks_from_signed(B, nan_mark=CIRCLE)
    if constraints is not None:
        # On a marked pair, a forbidden i -> j whose reverse is not forbidden: an arrowhead at i.
        one_way = constraints.forbidden & ~constraints.forbidden.T
        marks[(marks == CIRCLE) & one_way] = ARROW
    structure = GraphStructure.from_numpy(marks, kind="pag")
    latent = [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(pairs)), strict=True)]
    metadata = {"latent_pairs": latent, "cycle_edges": cycle_edges}
    return structure_output(structure, labels=labels, metadata=metadata)


@method_spec("skewd_mv")
def skewd_mv(
    data: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    indep_test: str | None = "kci",
    tuning_iter: int = 20,
    max_iter: int = 3000,
    max_fits: int | None = 100,
    optimizer: str = "cmaes",
    seed: int | None = None,
) -> StructureOutput:
    """Learn a DAG of a skew-normal location-scale model with SkewD-MV.

    SkewD-MV [1]_ estimates the Markov equivalence class with PC [2]_ and returns the DAG of the
    class whose skew-normal location-scale models, one per variable given its parents, have the
    largest likelihood. It is the graph version of :func:`~andrey.skewd`'s likelihood criterion.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed data, one row per sample. Coerced to ``float64``. A DataFrame's column names
        become the structure's ``labels``.
    alpha : float, default=0.05
        Significance level of PC's conditional-independence tests, in ``(0, 1)``.
    indep_test : str or None, default="kci"
        PC's conditional-independence test for continuous data, as in :func:`~andrey.pc`:
        ``"fisherz"``, ``"student_t"``, ``"kci"``, ``"rcot"``, ``"cmiknn"``, ``"gpdc"``, ``"gcm"``,
        ``"wgcm"``, or ``"pcm"``, but not ``"normal_scores"``: the likelihood stage reads the
        values, which ranks would hide. The default is the kernel test [3]_, the authors' choice.
        ``None`` picks ``"fisherz"``.
    tuning_iter : int, default=20
        The penalty settings, at least 1, that the Bayesian optimization scores by
        cross-validation for each conditional model; the first half, rounded up, are Latin
        hypercube points.
    max_iter : int, default=3000
        The most iterations, at least 1, of the expectation conditional maximization that refines
        each conditional model.
    max_fits : int or None, default=100
        The most conditional models, at least 1, the class's DAGs may need before the call refuses
        to list more of them (``ValueError``); 100 is the order up to which the paper expects the
        method to apply [1]_. ``None`` sets no cap, and the call can then run for days.
        While it is set, a group of undirected edges with more than 65,536 orientations is
        refused too, and so are groups that share a cycle and have more than 65,536 acyclic
        combinations.
    optimizer : {"cmaes", "lbfgs"}, default="cmaes"
        The optimizer of each search of a conditional model's penalized likelihood, as in
        :func:`~andrey.skewd`: ``"cmaes"`` is CMA-ES, the paper's heuristic; ``"lbfgs"`` is
        L-BFGS-B with the analytic gradient, from the same starts within the same bounds.
        ``"cmaes"`` favors the paper's optimizer; it gives up time. ``"lbfgs"`` favors time; it
        gives up the paper's optimizer. On the paper's simulated pairs and graphs of 1,000 rows
        (graphs of 6 and 10 variables; 20 seeds each, one worker), ``"lbfgs"`` decided worse than
        ``"cmaes"`` in no setting by a paired bootstrap, at a median of 5.8 to 14.9 times less time
        per pair and 2.4 times less per graph, and its penalized likelihood was lower in 306 of the
        738 fitted models and higher in 233.
    seed : int or None, default=None
        ``None``, the default, is the configured seed, ``ANDREY_SEED`` or ``0`` when it is unset,
        not a fresh random one, so a fit repeats. Seed of PC's randomized tests and of the random
        draws of each fit.

    Returns
    -------
    StructureOutput
        Contains a :class:`~andrey.core.GraphStructure` of kind ``dag``.
        ``metadata["algorithm"]`` is ``"skewd_mv"``, ``metadata["candidates"]`` the number of DAGs
        in the class, and ``metadata["fits"]`` the number of conditional models fitted.
        ``metadata["indep_test"]`` names the test the fit used and ``metadata["seed"]`` the seed.
        ``metadata["tie_break"]`` is the tie-break rule the fit ran under,
        ``andrey.config.tie_break``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, such as a string for a number; the message names the
        method, the argument, what it accepts, and the value.
    ValueError
        If an argument lies outside its range or is not one of its choices; the message names the
        method, the argument, what it accepts, and the value.

    Warns
    -----
    DataTypeWarning
        When a column holds at most 10 distinct whole numbers and so may hold category codes, which
        the fit reads as numbers. Each message warns once per process.
    SearchLimitWarning
        When a conditional model's refinement stops at ``max_iter`` rather than on its own rule.

    Notes
    -----
    **When to use.** SkewD-MV learns a DAG from continuous variables observed i.i.d. under a
    location-scale noise model, each variable ``y = f(pa) + g(pa) n`` with ``g > 0``, skew-normal
    noise ``n`` independent of its parents, and no hidden common cause; it also assumes the
    faithfulness PC needs, so that the true DAG lies in the estimated class [1]_. It is designed for
    small graphs whose noise scale changes with the parents, which the additive models of
    :func:`~andrey.cam` and :func:`~andrey.resit` do not allow. It favors accuracy on such graphs
    and gives up time. On the paper's skew-normal graphs it made as few errors as the code published
    with the paper, and fewer than the empty graph and than :func:`~andrey.pc` (KCI),
    :func:`~andrey.cam`, :func:`~andrey.nogam`, :func:`~andrey.score_matching`, and
    :func:`~andrey.resit`, all five faster; ``andrey.config(n_jobs)`` runs its conditional models
    and their searches on threads, which cuts the wall time.

    Each column is standardized first [1]_. PC, order-independent [2]_, gives the class. Its DAGs
    orient the undirected edges with no directed cycle and no new collider [1]_, and are every
    combination of one orientation of each connected group of undirected edges, so each group is
    listed on its own, one DAG at a time, and the listing stops with an error once the groups need
    more than ``max_fits`` conditional models or one group has more than 65,536 orientations, so a
    class too large to score is never listed in full. The edges are oriented in a maximum
    cardinality search order, so the listing's time does not depend on the column order. When a
    group has no such orientation, as conflicting colliders can leave, its candidates are every
    acyclic orientation of its edges. Each variable with an undirected edge is fitted given its
    parents in each orientation, once per distinct parent set: the standard Gaussian log-likelihood
    for a variable without parents, and otherwise :func:`~andrey.skewd`'s skew-normal fit with
    additive B-splines of the parents, 10 location and 7 scale columns per parent and a penalty pair
    per parent chosen by a 5-fold cross-validation [1]_. A constant moved between two parents'
    location coefficients changes no fitted value, so the fit leaves it out. Only these variables'
    terms differ between the DAGs, so their sum decides (Proposition 5.1 of [1]_), and each group
    contributes its own terms: the returned DAG holds the orientation of each group with the largest
    sum, the first on a tie, which is the DAG of the class with the largest sum. Where a conflicting
    graph lets the groups' best orientations close a directed cycle across its directed edges, the
    search takes the acyclic combination with the largest sum, and a graph with none raises an
    error. The defaults, 20 candidates of the Bayesian optimization and 3,000 iterations, are those
    of the paper's runs. The columns are fitted in the order of their values' ranks, so the DAG does
    not depend on the column order; ties follow ``andrey.config.tie_break``.

    ``n_jobs`` above 1 (``andrey.config`` or ``ANDREY_N_JOBS``) runs the conditional models, and
    the searches of each, on that many threads. Each search draws from its own random
    stream, split from ``seed`` in a fixed order, so the DAG and its likelihoods are the same bit
    for bit with any number of workers. The models are fitted on one BLAS thread with any number of
    workers, so the result does not depend on the machine's BLAS thread count either.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: every fit reads every row of
    its variables, so a row with a missing value has no place in it. Drop the incomplete rows
    yourself, which suits values missing completely at random, or use :func:`~andrey.pc` or
    :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Its cost grows with the number of distinct parent sets the class allows,
    exponentially in a variable's undirected edges, and it refuses a class that needs more than
    ``max_fits`` (100) conditional models. The
    `0.19.0 release card <https://andrey.abel.ai/docs/benchmarks/0.19.0.html#skewd-mv>`_ has the
    measured times. An edge PC misses or misorients stays wrong: the likelihood only chooses among
    the class's DAGs, and PC's skeleton missed or added some edges on the measured graphs. The
    measured results are a reference, not a hard limit: they change with the data generator, the
    density, the method's settings, and the machine.

    References
    ----------
    .. [1] Klippert, D., and Marx, A. (2026). Skewness-robust causal discovery in location-scale
       noise models. ICML, PMLR 306.
    .. [2] Colombo, D., and Maathuis, M. H. (2014). Order-independent constraint-based causal
       structure learning. JMLR 15, 3741-3782.
    .. [3] Zhang, K., Peters, J., Janzing, D., and Scholkopf, B. (2011). Kernel-based conditional
       independence test and application in causal discovery. UAI, 804-813.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import skewd_mv
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(300)
    >>> y = np.tanh(2 * x) + (0.4 + 0.2 * np.tanh(x)) * rng.standard_normal(300)
    >>> z = y + 0.5 * rng.standard_normal(300)
    >>> data = np.column_stack([x, y, z])
    >>> out = skewd_mv(data, indep_test="fisherz", tuning_iter=1, max_iter=20, seed=0)
    >>> out.structure.kind, out.metadata["candidates"]
    ('dag', 3)
    """
    from andrey.api.constraint import pc
    from andrey.core import backend
    from andrey.core.ties import tie_ranks
    from andrey.core.warning_policy import warn_search_limit
    from andrey.fcm.skewd import (
        LISTED_DAGS,
        Budget,
        Fit,
        class_candidates,
        class_size,
        fit,
        merge_components,
        search_pool,
    )

    X = data_matrix(data, method="skewd_mv", min_rows=MIN_ROWS)
    labels = column_labels(data)
    X = X / np.abs(X).max(axis=0)  # divide first, so tiny or huge units cannot overflow the moments
    order = np.argsort(tie_ranks(X), kind="stable")
    X = X[:, order]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    assert seed is not None  # method_spec resolves it
    skeleton = pc(X, alpha=alpha, indep_test=indep_test, seed=seed).structure
    assert isinstance(skeleton, GraphStructure)
    marks = skeleton.to_numpy()
    components, best, fits = class_candidates(marks, max_fits)
    candidates = class_size(best, components, None if max_fits is None else LISTED_DAGS)
    if candidates == 0:  # known from the graph alone, so before any model is fitted
        raise ValueError(
            "skewd_mv: PC's graph leaves no DAG that keeps its directed edges and adds no collider"
        )
    budget = Budget(
        knots=8,
        folds=5,
        tuning_iter=tuning_iter,
        initial=-(-tuning_iter // 2),
        max_iter=max_iter,
        optimizer=optimizer,
    )
    n = X.shape[0]
    terms: dict[tuple[int, tuple[int, ...]], float] = {}
    converged = True
    workers = backend.worker_count()

    def model(each: tuple[int, tuple[int, ...]]) -> Fit:
        y, parents = each
        return fit(X[:, list(parents)], X[:, y], budget, seed, run)

    # One BLAS thread: a threaded product rounds differently from a serial one, so the result
    # would depend on the machine's thread count.
    with one_blas_thread(), search_pool(workers) as run:
        if workers > 1 and len(fits) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(workers, len(fits))) as pool:
                results = list(pool.map(model, fits))
        else:
            results = [model(each) for each in fits]
    for (y, parents), result in zip(fits, results, strict=True):
        terms[(y, parents)] = n * result.log_likelihood
        converged &= result.converged
    if not converged:
        warn_search_limit("skewd_mv", "max_iter", max_iter, "ECM iterations")
    gaussian = -0.5 * n * np.log(2 * np.pi) - 0.5 * (X**2).sum(axis=0)  # a variable with no parent

    def term(dag: np.ndarray, y: int) -> float:
        parents = tuple(np.flatnonzero(dag[:, y]).tolist())
        return terms[(y, parents)] if parents else float(gaussian[y])

    # A DAG's likelihood is the sum over its variables, and only the variables of an undirected
    # component change with that component's orientation, so the best DAG holds each component's
    # best orientation (the first, on a tie).
    tables = [
        [sum(term(member, y) for y in nodes) for member in members] for nodes, members in components
    ]
    best = merge_components(best, components, tables)
    back = np.argsort(order)
    return structure_output(
        dag_from_adjacency(best[np.ix_(back, back)]),
        labels=labels,
        metadata={"candidates": candidates, "fits": len(fits)},
    )
