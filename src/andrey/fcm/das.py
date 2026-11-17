"""DAS: SCORE's causal order, then an edge screen read from the Jacobian of the score.

DAS (Discovery At Scale, Montagna et al. 2023) assumes SCORE's model, a nonlinear additive noise
model with Gaussian noise, ``x_j = f_j(parents) + e_j``. Its Lemma 1: for a leaf ``l``, the
off-diagonal entry ``d^2 log p / d x_l d x_j`` of the score's Jacobian is
``f_l'(x_j) / sigma_l^2`` when ``j`` is a parent and zero otherwise, so the mean of its absolute
value is positive exactly at the parents. The method runs three phases:

1. **Causal order.** SCORE's leaf search (:func:`andrey.fcm.score_matching.leaf_order` on the
   variances of :mod:`andrey.fcm.stein`'s diagonal estimate), unchanged.
2. **Edge screen** (the paper's Algorithm 1). At the step that picks leaf ``l``, the remaining
   variables are ``l`` and the variables before it in the order, so ``l`` is a leaf of their
   joint density, and row ``l`` of the same step's estimate gives the entries the lemma reads.
   The absolute median of the entry over the rows ranks the variables before ``l``; the ``K + 1``
   largest are kept, the smallest of them is the reference, and each of the others stays when a
   one-sided Welch test of equal means of the absolute entries against the reference's has a
   p-value at most :data:`LEVEL`.
3. **Pruning.** The DAG of the screened edges is pruned by the smooth-term tests of CAM,
   :func:`andrey.fcm.pruning.prune`, at most ``K`` candidate parents per fit instead of every
   variable before it in the order.

The choices the paper leaves open, and where its text and the authors' code differ:

- The test level :data:`LEVEL` is ``0.01`` and ``K`` (:data:`MAX_PARENTS`) is 20, the values of the
  paper's experiments; its text keeps an edge at a p-value "<= 0.01".
- The pruning cutoff :data:`ALPHA` is ``0.001``. The paper states none and follows the experiments
  of Rolland et al. (2022), whose cutoff it is.
- The ``MIN_PARENTS`` largest medians stay without a test. The paper has no such step; ``5`` is the
  authors' code's. Without it the reference is a parent whenever every variable before the leaf
  is one, and the smallest of them is never kept: a chain of two variables would have no edge.
- The ranking is the absolute median of the entries, as in the authors' code; the paper's text
  states the mean of their absolute values.
- The authors' code estimates the Jacobian once, on all the variables. The paper's lemma reads the
  leaf's row in the joint density of the variables left at its step, as here.
- A test whose two samples are both constant has no p-value, and its edge is not kept.
- The medians are compared rounded to ``KEY_DP`` decimal places, as the leaf search compares its
  variances; a tie goes to the earlier column.

The fit draws no random numbers.

References
----------
Montagna, Noceti, Rosasco, Zhang, Locatello. "Scalable causal discovery with score matching."
CLeaR, PMLR 213:752-771, 2023.

Rolland, Cevher, Kleindessner, Russell, Janzing, Scholkopf, Locatello. "Score matching enables
causal discovery of nonlinear additive noise models." ICML, PMLR 162:18741-18753, 2022.

Buhlmann, Peters, Ernest. "CAM: Causal additive models, high-dimensional order search and
penalized regression." Annals of Statistics 42(6):2526-2556, 2014.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import ttest_ind

from andrey.core.ties import tie_key
from andrey.core.units import unit_factors
from andrey.fcm.pruning import prune
from andrey.fcm.score_matching import leaf_order
from andrey.fcm.stein import Jacobian, jacobian

_log = logging.getLogger(__name__)

#: Cutoff of the pruning tests.
ALPHA = 0.001
#: Basis functions per smooth term of the pruning fits.
K = 10
#: Most candidate parents the screen keeps per variable, the paper's ``K``.
MAX_PARENTS = 20
#: Level of the screen's Welch tests.
LEVEL = 0.01
#: The largest medians the screen keeps without a test (the authors' code).
MIN_PARENTS = 5


def screen(entries: np.ndarray, slots: int | None) -> list[int]:
    """Pick a leaf's candidate parents from the entries of its Jacobian row.

    Parameters
    ----------
    entries : np.ndarray
        ``(n, m)`` entries of the leaf's row at the rows, one column per variable before the leaf
        that may be a parent.
    slots : int or None
        Most columns to keep, the paper's ``K``; ``None`` keeps any number.

    Returns
    -------
    list[int]
        The kept columns, largest absolute median first.
    """
    m = entries.shape[1]
    magnitudes = np.abs(entries)
    medians = np.abs(np.median(entries, axis=0))
    ranked = sorted(range(m), key=lambda k: (-tie_key(float(medians[k])), k))
    top = ranked if slots is None else ranked[: slots + 1]
    if not top:
        return []
    reference = top[-1]
    # The reference stands for a zero entry, so it is never kept when it lies beyond the slots.
    beyond = slots is not None and len(top) > slots
    kept = []
    for rank, k in enumerate(top):
        if k == reference and beyond:
            continue
        if rank < MIN_PARENTS:
            kept.append(k)
            continue
        if k == reference:
            continue
        p_value = ttest_ind(
            magnitudes[:, k], magnitudes[:, reference], equal_var=False, alternative="greater"
        ).pvalue
        if p_value <= LEVEL:
            kept.append(k)
    return kept


@dataclass(frozen=True)
class DASResult:
    """A DAS fit: the pruned DAG, the causal order (causes first), and the screened candidates."""

    dag: np.ndarray
    order: np.ndarray
    screened: np.ndarray


def das(
    X: np.ndarray,
    *,
    alpha: float = ALPHA,
    max_parents: int | None = MAX_PARENTS,
    forbidden: np.ndarray | None = None,
    required: np.ndarray | None = None,
    precedence: np.ndarray | None = None,
) -> DASResult:
    """Fit DAS to the ``(n, d)`` matrix ``X``.

    ``precedence`` goes to :func:`~andrey.fcm.score_matching.leaf_order`; a ``forbidden`` edge is
    never a candidate, and a ``required`` one is a candidate the pruning never removes, which takes
    one of the leaf's ``max_parents`` slots. Ties go to the earlier column, so a caller that orders
    the columns by rank breaks them by rank.
    """
    d = X.shape[1]
    unit = (X - X.mean(axis=0)) * unit_factors(X, standardize=True)
    candidates = np.zeros((d, d), dtype=bool) if required is None else required.astype(bool)
    step: dict[str, Jacobian] = {}

    def criterion(columns: np.ndarray) -> np.ndarray:
        step["estimate"] = estimate = jacobian(columns, method="das")
        return estimate.diagonal.var(axis=0)

    def on_leaf(remaining: list[int], leaf: int) -> None:
        pool = [
            (position, k)
            for position, k in enumerate(remaining)
            if k != leaf
            and not candidates[k, leaf]
            and (forbidden is None or not forbidden[k, leaf])
        ]
        if not pool:
            return
        row = step["estimate"].row(remaining.index(leaf))
        entries = row[:, [position for position, _ in pool]]
        slots = (
            None if max_parents is None else max(max_parents - int(candidates[:, leaf].sum()), 0)
        )
        kept = screen(entries, slots)
        candidates[[pool[i][1] for i in kept], leaf] = True
        _log.debug("leaf %d: %d of %d kept", leaf, len(kept), len(pool))

    order = leaf_order(unit, criterion, precedence, on_leaf=on_leaf)
    dag = prune(X, candidates, cutoff=alpha, k=K).astype(bool)
    if required is not None:
        dag |= required
    return DASResult(dag=dag, order=order, screened=candidates)
