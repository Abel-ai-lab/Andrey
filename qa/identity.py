"""Check that a change returns the same answers, bit for bit, by hashing each method's full output.

Every supported method of the quality gate (:mod:`qa.quality_gate`) is fitted on the first draws of
its gate datasets and of one larger dataset (:data:`LARGER`), and UT-IGSP and GnIES on the gate's
smaller interventional dataset (:data:`UNKNOWN_TARGETS`), on the numpy backend with one worker and
one BLAS thread. The options 0.19.0 added (:data:`OPTIONS`) have fits of their own on data where
they change the answer. A fit's hash covers its whole output: the graph, the edge weights, the
causal order, and every metadata value, such as a score, a p-value, or a list of nodes. Floats
enter as their bytes, never as printed text, NaN as one bit pattern, and mappings in key order, so
equal hashes mean equal bits. A second hash covers only the endpoint marks of the graphs, and each
part of the output has its own digest (:func:`parts`), so a difference reads as a changed graph, as
changed bits under the same graph, or as changed metadata values alone, named by key.

The manifest holds one row per fit, keyed by method, dataset, seed, permutation, and worker count,
under a header naming the commit, the imported Andrey package's directory and commit, the Python,
NumPy, and SciPy versions, the loaded BLAS, the CPU, and the thread settings. ``--compare`` counts
the rows of two manifests that differ. A fit that
queries a CI test or a local score also records two deterministic counts (:class:`Counts`), and a
compare names every row where one changed and fails on any change, a fall as well as a rise, unless
``--allow-fewer-queries`` accepts a fall.

``--shuffle K`` refits each draw (:class:`Fit`) on ``K`` seeded column orders, moving the targets
of ICP and IGSP and the contexts of J-PCMCI+ with the columns, and maps each answer back to the
original columns before hashing it;
``--workers N`` refits the methods that run worker processes or threads (:data:`WORKER_METHODS`) on
``N`` workers, with every GES pass, every skeleton pass of conditioning size two or more, and every
possible-d-sep search sent to the pool. Both report each refit whose answer differs from the first
fit. A different worker count must not change a bit, so any difference fails it. A column order
fails the check when a graph or a causal order differs (:data:`STRUCTURE`) or when an edge weight or
metadata value differs by more than rounding (:data:`RTOL`, or :data:`ATOL` near zero). The last
bits of such values, such as p-values, depend on the order a sum runs in, so a refit that differs
only by rounding is reported as information.

    uv run python -m qa.identity -o before.json             # fit the corpus, write a manifest
    uv run python -m qa.identity --compare before.json after.json  # exit 1 on any difference
    uv run python -m qa.identity --shuffle 3                # exit 1 if an order changes a graph
    uv run python -m qa.identity GES GFCI --workers 4       # exit 1 if 4 workers change one
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_limits

import andrey
from andrey import GraphStructure, StructureOutput, TemporalStructure, data
from andrey.core import ci, ci_cmiknn, ci_gpdc, ci_kernel, ci_missing, score, score_delta
from andrey.search import hc, xges
from qa import quality_gate as qg

# ==================================================================================================
# The corpus -- each method's public call, on the gate's datasets and one larger dataset.
# ==================================================================================================

#: Draws per dataset, the first of the gate's ten seeds.
SEEDS = 2
#: The significance level of the CI-test methods, passed to each; the counts read it.
ALPHA = 0.05


@dataclass(frozen=True)
class Fit:
    """A method's public call on one draw, returning its full output.

    ``blocks`` is the number of side-by-side copies of the variables in the data: GIES's samples
    and intervention indicators, MultiGroupDirectLiNGAM's groups; a column order moves each copy
    the same way. ``refit`` is set for the methods whose draws hold more than the variables, ICP,
    IGSP, FCI-JCI, and J-PCMCI+: ``refit(x, rng)`` draws a column order from ``rng``, fits the draw
    with its columns in that order, moving the targets and contexts the call names along with them
    while each row's environment stays, and returns the answer mapped back to the original columns
    (:func:`unpermute`). ``shuffled`` is ``False`` for a method whose draws a column order does not
    apply to. ``alpha`` is the significance level of a CI-test method.
    """

    call: Callable[[np.ndarray], Any]
    blocks: int = 1
    shuffled: bool = True
    alpha: float | None = None
    refit: Callable[[np.ndarray, np.random.Generator], Any] | None = None


_CDNOD = qg.AndreyCDNOD().params()
_GROUPS = qg.AndreyMultiGroupDirectLiNGAM().params()["n_groups"]
_VARMA = qg.AndreyVarmaLiNGAM().params()
_DYNOTEARS = qg.DYNOTEARS.params()


def _domains(x: np.ndarray) -> Any:
    index = qg.regime_models.domain_index(len(x), _CDNOD["n_domains"])
    return andrey.cdnod(x, index, alpha=_CDNOD["alpha"], indep_test=_CDNOD["indep_test"])


def _environments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """The samples of a GIES draw, each row's environment, and the observational one's label.

    An environment is the rows of one target set, labeled ``0, 1, ...`` in the order of its first
    row, so the labels do not depend on the column order.
    """
    samples, mask = qg.unstack(x)
    _, first, inverse = np.unique(mask, axis=0, return_index=True, return_inverse=True)
    env = np.argsort(np.argsort(first))[inverse.reshape(-1)]
    return samples, env, int(env[np.flatnonzero(~mask.any(axis=1))[0]])


def _fci_jci(x: np.ndarray) -> StructureOutput:
    """FCI-JCI with the draw's last column as the environment, as the gate fits it."""
    return andrey.fci_jci(x[:, :-1], x[:, -1], alpha=ALPHA)


def _ut_igsp(x: np.ndarray) -> StructureOutput:
    """UT-IGSP with the observational environment known and every setting's targets unknown."""
    samples, env, observational = _environments(x)
    return andrey.ut_igsp(samples, env, targets={observational: []}, seed=0)


def _gnies(x: np.ndarray) -> StructureOutput:
    """GnIES with every environment's targets unknown."""
    samples, env, _ = _environments(x)
    return andrey.gnies(samples, env)


def _pcmci(x: np.ndarray) -> StructureOutput:
    """PCMCI with the gate adapter's settings; the adapter names the PC1 level ``pc_alpha``, as
    tigramite does."""
    params = qg.AndreyPCMCI("pcmci", "student_t", tau_max=2).params()
    params["pc1_alpha"] = params.pop("pc_alpha")
    return andrey.pcmci(x, **params)


def _jpcmciplus(x: np.ndarray) -> StructureOutput:
    """J-PCMCI+ at the gate's settings on a draw of its datasets, the contexts at the gate's
    positions."""
    contexts = qg.context_columns(x)
    return andrey.jpcmciplus(
        list(x),
        time_contexts=contexts["time_contexts"],
        space_contexts=contexts["space_contexts"],
        time_dummy=True,
        tau_max=2,
        alpha=ALPHA,
    )


def _icp_refit(x: np.ndarray, rng: np.random.Generator) -> StructureOutput:
    """ICP on a draw with its variables in a drawn order: each row's environment (the first column)
    stays, and the target, the last variable, moves to where the order puts it."""
    n = x.shape[1] - 1
    order = rng.permutation(n)
    moved = np.ascontiguousarray(x[:, [0, *(order + 1)]])
    target = int(np.flatnonzero(order == n - 1)[0])
    return unpermute(andrey.icp(moved[:, 1:], moved[:, 0], target=target), order)


def _igsp_refit(x: np.ndarray, rng: np.random.Generator) -> StructureOutput:
    """IGSP on a stacked draw (:func:`qg.known_target_stack`) with its variables in a drawn order:
    the samples and the matrix marking each setting's targets move alike, and each row's
    environment, the column between them, stays."""
    p = (x.shape[1] - 1) // 2
    order = rng.permutation(p)
    moved = np.ascontiguousarray(x[:, [*order, p, *(order + p + 1)]])
    return unpermute(qg.igsp_fit(moved), order)


def _fci_jci_refit(x: np.ndarray, rng: np.random.Generator) -> StructureOutput:
    """FCI-JCI on a draw with its variables in a drawn order; each row's environment, the last
    column, stays."""
    order = rng.permutation(x.shape[1] - 1)
    moved = np.ascontiguousarray(x[:, [*order, x.shape[1] - 1]])
    return unpermute(_fci_jci(moved), order)


def _jpcmciplus_refit(x: np.ndarray, rng: np.random.Generator) -> StructureOutput:
    """J-PCMCI+ on a draw with its columns, contexts among them, in a drawn order: the context lists
    name the contexts' new columns, and the answer's system variables follow their order among
    the columns."""
    contexts = qg.context_columns(x)
    order = rng.permutation(x.shape[-1])
    new = np.argsort(order)  # the column each original column moves to
    moved = np.ascontiguousarray(x[..., order])
    out = andrey.jpcmciplus(
        list(moved),
        time_contexts=[int(new[c]) for c in contexts["time_contexts"]],
        space_contexts=[int(new[c]) for c in contexts["space_contexts"]],
        time_dummy=True,
        tau_max=2,
        alpha=ALPHA,
    )
    context = {c for kind in contexts.values() for c in kind}
    return unpermute(out, np.array([c for c in order if c not in context]))


def _longitudinal(x):
    """Longitudinal LiNGAM with the gate adapter's settings; the adapter names the lag ``n_lags``,
    as lingam does."""
    params = qg.AndreyLongitudinalLiNGAM().params()
    return andrey.longitudinal_lingam(list(x), tau_max=params["n_lags"], measure=params["measure"])


#: The gate's settings, with each method's full output: the benchmark adapters' parameters for the
#: methods the gate fits through an adapter, which returns only the graph.
FITS: dict[str, Fit] = {
    "PC": Fit(functools.partial(andrey.pc, alpha=ALPHA), alpha=ALPHA),
    "FCI": Fit(functools.partial(andrey.fci, alpha=ALPHA), alpha=ALPHA),
    "GFCI": Fit(functools.partial(andrey.gfci, alpha=ALPHA), alpha=ALPHA),
    "CDNOD": Fit(_domains, alpha=_CDNOD["alpha"]),
    "FCI-JCI": Fit(_fci_jci, alpha=ALPHA, refit=_fci_jci_refit),
    "GES": Fit(andrey.ges),
    "GIES": Fit(qg.gies_fit, blocks=2),
    "XGES": Fit(andrey.xges),
    "AGES": Fit(andrey.ages),
    "HC": Fit(andrey.hc),
    "Tabu": Fit(andrey.tabu),
    "IAMB": Fit(functools.partial(andrey.iamb, alpha=ALPHA), alpha=ALPHA),
    "InterIAMB": Fit(functools.partial(andrey.inter_iamb, alpha=ALPHA), alpha=ALPHA),
    "MMPC": Fit(functools.partial(andrey.mmpc, alpha=ALPHA), alpha=ALPHA),
    "MMHC": Fit(functools.partial(andrey.mmhc, alpha=ALPHA), alpha=ALPHA),
    "RSMAX2": Fit(
        functools.partial(andrey.rsmax2, restrict="pc", maximize="tabu", alpha=ALPHA), alpha=ALPHA
    ),
    "CAM": Fit(andrey.cam),
    "RESIT": Fit(andrey.resit),
    "ScoreMatching": Fit(andrey.score_matching),
    "NoGAM": Fit(andrey.nogam),
    "DAS": Fit(andrey.das),
    "ExactSearch": Fit(andrey.exact_search),
    "CALM": Fit(functools.partial(andrey.calm, seed=0)),
    "NOTEARS": Fit(andrey.notears),
    "DAGMA": Fit(andrey.dagma),
    "BOSS": Fit(functools.partial(andrey.boss, seed=0)),
    "GRaSP": Fit(functools.partial(andrey.grasp, seed=0)),
    "BOSS-FCI": Fit(functools.partial(andrey.boss_fci, seed=0, alpha=ALPHA), alpha=ALPHA),
    "GRaSP-FCI": Fit(functools.partial(andrey.grasp_fci, seed=0, alpha=ALPHA), alpha=ALPHA),
    "DirectLiNGAM": Fit(andrey.direct_lingam),
    "ICA-LiNGAM": Fit(functools.partial(andrey.ica_lingam, seed=0)),
    "MultiGroupDirectLiNGAM": Fit(
        lambda x: andrey.multi_group_direct_lingam(qg.regime_models.split_groups(x, _GROUPS)),
        blocks=_GROUPS,
    ),
    "RCD": Fit(andrey.rcd),
    "BottomUpParceLiNGAM": Fit(andrey.bottom_up_parce_lingam),
    "GIN": Fit(andrey.gin),
    "ANM": Fit(andrey.anm),
    "PNL": Fit(andrey.pnl),
    "LOCI": Fit(andrey.loci),
    "CPCM": Fit(functools.partial(andrey.cpcm, family="gaussian")),
    "CPCMGraph": Fit(functools.partial(andrey.cpcm_graph, family="gaussian_fixed_scale")),
    "SkewD": Fit(functools.partial(andrey.skewd, **qg.SKEWD_BUDGET)),
    "SkewDMV": Fit(functools.partial(andrey.skewd_mv, **qg.SKEWD_BUDGET)),
    "PCMCI": Fit(_pcmci),
    "PCMCIplus": Fit(lambda x: andrey.pcmciplus(x, **qg.andrey_pcmciplus_adapter(5).params())),
    "LPCMCI": Fit(lambda x: andrey.lpcmci(x, **qg.andrey_lpcmci_adapter(3).params())),
    "J-PCMCI+": Fit(_jpcmciplus, alpha=ALPHA, refit=_jpcmciplus_refit),
    "VARMA-LiNGAM": Fit(
        lambda x: andrey.varma_lingam(
            x, order=tuple(_VARMA["order"]), criterion=_VARMA["criterion"], prune=_VARMA["prune"]
        )
    ),
    "LongitudinalLiNGAM": Fit(_longitudinal),
    "Granger": Fit(lambda x: andrey.granger(x, tau_max=qg.GRANGER_LAGS)),
    "UT-IGSP": Fit(_ut_igsp, blocks=2),
    "GnIES": Fit(_gnies, blocks=2),
    "ICP": Fit(qg.icp_fit, refit=_icp_refit),
    "IGSP": Fit(qg.igsp_fit, refit=_igsp_refit),
    "CAM-UV": Fit(andrey.camuv),
    "DYNOTEARS": Fit(lambda x: andrey.dynotears(x, **_DYNOTEARS)),
}


def _target_list(name: str) -> Fit:
    """``name`` with every variable a target, in one call."""
    method = getattr(andrey, name)
    return Fit(
        lambda x: method(x, alpha=ALPHA, target=list(range(x.shape[1]))),
        shuffled=False,
        alpha=ALPHA,
    )


#: The set learners, by their names here and in the package.
SET_LEARNERS = {"IAMB": "iamb", "InterIAMB": "inter_iamb", "MMPC": "mmpc"}
#: The set learners, each fitted with a list of targets that one search answers, on the datasets
#: of the method it names, with every variable a target.
TARGET_LISTS: dict[str, str] = {f"{name}-targets": name for name in SET_LEARNERS}
FITS |= {f"{name}-targets": _target_list(SET_LEARNERS[name]) for name in SET_LEARNERS}

# --------------------------------------------------------------------------------------------------
# The options of 0.19.0 -- each fitted outside the gate, on data where it changes the answer.
# --------------------------------------------------------------------------------------------------

#: The columns each of four cohorts measures, in blocks of equal rows. Columns 0-3 are in every
#: cohort; 4-6 only in the first two, 7-9 only in the middle two, and 10-13 only in the last two. No
#: row measures a column of 4-6 with one of 10-13, and no row bridges the cohorts, so a test given
#: a column from another cohort reads no rows. The columns of a model are in a seeded order, so
#: which columns a cohort measures bears no relation to the graph.
COHORT_COLUMNS: tuple[tuple[int, ...], ...] = (
    (*range(7),),
    (*range(10),),
    (*range(4), *range(7, 14)),
    (*range(4), *range(10, 14)),
)
#: The columns of a gapped series: the first is not measured in the first half of the steps and the
#: second not in the second half, so no step measures both.
SERIES_GAPS = ((0, slice(None, 300)), (1, slice(300, None)))
#: A draw is sparse in the sense of ``degrees_of_freedom``'s docstring, which separates the two
#: rules on sparse tables: each variable has 3 to 5 categories and a table of two variables given
#: two more has 81 to 625 cells for 250 rows.
SPARSE_DISCRETE = {"d": 12, "n": 250, "categories": (3, 5)}


def cohort_draws(n: int) -> tuple[qg.Draw, ...]:
    """Return a draw per seed of ``n`` rows stacked from the cohorts of :data:`COHORT_COLUMNS`.

    The values of a model of the gate's ``linear_gauss_er`` regime, with each cohort's rows missing
    the columns it does not measure.
    """
    d = 1 + max(max(columns) for columns in COHORT_COLUMNS)
    cohort = cohort_index(n)[:, 0]
    out = []
    for seed in qg.SEEDS:
        sample = data.sample_scm(**qg.regime_kwargs("linear_gauss_er", d, n, seed))
        x = sample.data.copy()
        for k, columns in enumerate(COHORT_COLUMNS):
            x[np.ix_(cohort == k, np.setdiff1d(np.arange(d), columns))] = np.nan
        out.append(qg.Draw(x, sample.graph))
    return tuple(out)


def cohort_index(n: int) -> np.ndarray:
    """Return the ``(n, 1)`` cohort of each row of a draw of :func:`cohort_draws`."""
    return qg.regime_models.domain_index(n, len(COHORT_COLUMNS))


def gapped_series_draws(d: int, n: int) -> tuple[qg.Draw, ...]:
    """Return a draw per seed of the gate's ``pcmciplus_paper`` series with the gaps of
    :data:`SERIES_GAPS`."""
    out = []
    for draw in qg.series_draws("pcmciplus_paper", d, n):
        x = draw.data.copy()
        for column, steps in SERIES_GAPS:
            x[steps, column] = np.nan
        out.append(qg.Draw(x, draw.truth))
    return tuple(out)


def sparse_discrete_draws() -> tuple[qg.Draw, ...]:
    """Return a draw per seed of a discrete network of :data:`SPARSE_DISCRETE`, its categories'
    codes as the columns."""
    out = (data.sample_discrete(**SPARSE_DISCRETE, seed=seed) for seed in qg.SEEDS)
    return tuple(qg.Draw(sample.data, sample.graph) for sample in out)


#: The datasets of the option fits, which the gate does not draw.
OPTION_DRAWS: dict[str, Callable[[], tuple[qg.Draw, ...]]] = {
    "cohorts_d14_n2000": functools.partial(cohort_draws, 2000),
    "gapped_series_d6_n600": functools.partial(gapped_series_draws, 6, 600),
    "discrete_sparse_d12_n250": sparse_discrete_draws,
}


@dataclass(frozen=True)
class OptionFit:
    """A supported method's call with one of the options 0.19.0 added, on data where it matters.

    ``call(x, **options)`` fits a draw of ``dataset``; ``call(x)`` is the same fit at the option's
    default. ``method`` names the supported method it is a call of, and ``alpha`` is the
    significance level of a CI-test method.
    """

    method: str
    call: Callable[..., Any]
    options: dict[str, str]
    dataset: str
    alpha: float | None = None

    @property
    def fit(self) -> Fit:
        return Fit(functools.partial(self.call, **self.options), alpha=self.alpha)


def _cohorts(x: np.ndarray) -> np.ndarray:
    """The cohort of each row of ``x``, 1-D."""
    return cohort_index(len(x))[:, 0]


def _ambiguous_fits() -> dict[str, OptionFit]:
    """The searches that read ``possibly_separable``, each at ``"ambiguous"`` on cohorts that no row
    bridges, PCMCI+ on a gapped series; MMPC returns the CPDAG that reads it."""
    options = {"possibly_separable": "ambiguous"}
    cohorts = "cohorts_d14_n2000"
    calls: dict[str, tuple[str, Callable[..., Any], str]] = {
        "PC": ("PC", lambda x, **o: andrey.pc(x, alpha=ALPHA, **o), cohorts),
        "FCI": ("FCI", lambda x, **o: andrey.fci(x, alpha=ALPHA, **o), cohorts),
        "CDNOD": (
            "CDNOD",
            lambda x, **o: andrey.cdnod(
                x, cohort_index(len(x)), alpha=_CDNOD["alpha"], indep_test=_CDNOD["indep_test"], **o
            ),
            cohorts,
        ),
        "FCI-JCI": (
            "FCI-JCI",
            lambda x, **o: andrey.fci_jci(x, _cohorts(x), alpha=ALPHA, **o),
            cohorts,
        ),
        "IAMB": ("IAMB", lambda x, **o: andrey.iamb(x, alpha=ALPHA, **o), cohorts),
        "InterIAMB": ("InterIAMB", lambda x, **o: andrey.inter_iamb(x, alpha=ALPHA, **o), cohorts),
        "MMPC": (
            "MMPC",
            lambda x, **o: andrey.mmpc(x, alpha=ALPHA, structure="cpdag", **o),
            cohorts,
        ),
        "PCMCIplus": (
            "PCMCIplus",
            lambda x, **o: andrey.pcmciplus(x, **qg.andrey_pcmciplus_adapter(5).params(), **o),
            "gapped_series_d6_n600",
        ),
    }
    return {
        f"{name}-ambiguous": OptionFit(
            method, call, options, dataset, None if name == "PCMCIplus" else ALPHA
        )
        for name, (method, call, dataset) in calls.items()
    }


def _nonempty_fits() -> dict[str, OptionFit]:
    """The constraint-based and set-learning searches at ``degrees_of_freedom="nonempty"`` with each
    categorical test, on sparse discrete data."""
    options = {"degrees_of_freedom": "nonempty"}
    methods = {
        "PC": andrey.pc,
        "FCI": andrey.fci,
        "IAMB": andrey.iamb,
        "InterIAMB": andrey.inter_iamb,
        "MMPC": andrey.mmpc,
    }
    return {
        f"{name}-{test}-nonempty": OptionFit(
            name,
            functools.partial(search, alpha=ALPHA, indep_test=test),
            options,
            "discrete_sparse_d12_n250",
            ALPHA,
        )
        for name, search in methods.items()
        for test in ("chisq", "gsq")
    }


#: The fits of the options 0.19.0 added, each named for its method and option:
#: ``possibly_separable="ambiguous"`` (:func:`_ambiguous_fits`) and
#: ``degrees_of_freedom="nonempty"`` (:func:`_nonempty_fits`).
OPTIONS: dict[str, OptionFit] = {**_ambiguous_fits(), **_nonempty_fits()}
FITS |= {name: option.fit for name, option in OPTIONS.items()}

#: The methods that run on ``andrey.config(n_jobs=...)``: GES, GFCI's GES phase, the skeleton
#: searches of PC, FCI, FCI-JCI, GFCI, CD-NOD, and RSMAX2's PC phase, and the possible-d-sep
#: searches of FCI, FCI-JCI, and GFCI in worker processes, PNL's two directions on threads, and the
#: lists of targets of :data:`TARGET_LISTS`, and the option fits of those methods.
_WORKERS = ("PC", "FCI", "FCI-JCI", "GFCI", "CDNOD", "RSMAX2", "GES", "PNL")
WORKER_METHODS = (
    *_WORKERS,
    *TARGET_LISTS,
    *(name for name, option in OPTIONS.items() if option.method in _WORKERS),
)

#: The factor on the longitudinal LiNGAM panel's coefficients at ``d = 10``. At the published
#: ranges the panel grows several-fold per time point; at this factor the spectral radius of every
#: step's lagged dynamics stays below 1 and the panel stays bounded.
PANEL_SCALE = 0.3

#: One dataset per method beyond the gate's sizes, drawn as the gate draws its regime, at a size
#: that keeps the whole corpus quick on one core; the panel's coefficients take
#: :data:`PANEL_SCALE`.
LARGER_DATASETS: dict[str, Callable[[], tuple[qg.Draw, ...]]] = {
    "linear_gauss_er_d100": functools.partial(qg.regime_draws, "linear_gauss_er", 100, 5000),
    "latent_gauss_er_d100": functools.partial(qg.regime_draws, "latent_gauss_er", 100, 5000),
    "lingam_sf_d70": functools.partial(qg.regime_draws, "lingam_sf", 70, 3500),
    "latent_lingam_er_d17": functools.partial(qg.regime_draws, "latent_lingam_er", 17, 2000),
    "linear_gauss_er_d16": functools.partial(qg.regime_draws, "linear_gauss_er", 16, 800),
    "anm_er_n1000_d30": functools.partial(qg.regime_draws, "anm_er_n1000", 30, 1000),
    "anm_er_n1000_d12": functools.partial(qg.regime_draws, "anm_er_n1000", 12, 1000),
    "anm_er_n1000_d6": functools.partial(qg.regime_draws, "anm_er_n1000", 6, 1000),
    "score_paper_standardized_d15_er1": functools.partial(
        qg.order_draws, "score_paper_standardized", 15, "er1"
    ),
    "score_paper_laplace_standardized_d12_er1": functools.partial(
        qg.order_draws, "score_paper_laplace_standardized", 12, "er1"
    ),
    "notears_paper_standardized_d60": functools.partial(qg.linear_sem_draws, "standardized", 60),
    "notears_paper_raw_d40": functools.partial(qg.linear_sem_draws, "raw", 40),
    "ages_paper_qs0.5_qw0.5_n10000": functools.partial(qg.weak_edge_draws, 0.5, 0.5, 10000),
    "interventional_gauss_er_d100": functools.partial(qg.intervention_draws, 100, 5000),
    "domain_shift_er_d70": functools.partial(qg.model_draws, "domain_shift_er", 70, 2000),
    "fci_jci_x100": functools.partial(qg.index_pag_draws, "X100"),
    "multi_group_er_d30": functools.partial(qg.model_draws, "multi_group_er", 30, 1000),
    "latent_measurement_d16": functools.partial(qg.model_draws, "latent_measurement", 16, 1000),
    "anm_pair_n1500": functools.partial(qg.pair_draws, "anm_pair", 1500),
    "pnl_pair_n1500": functools.partial(qg.pair_draws, "pnl_pair", 1500),
    "lsnm_pair_n1500": functools.partial(qg.pair_draws, "lsnm_pair", 1500),
    "pcmci_paper_d20_n1000": functools.partial(qg.series_draws, "pcmci_paper", 20, 1000),
    "pcmciplus_paper_d40_n1000": functools.partial(qg.series_draws, "pcmciplus_paper", 40, 1000),
    "lpcmci_paper_d20_n1000": functools.partial(qg.series_draws, "lpcmci_paper", 20, 1000),
    "varma_lingam_paper_d8_n300": functools.partial(qg.series_draws, "varma_lingam_paper", 8, 300),
    "longitudinal_lingam_stable_d10_n20_m500": functools.partial(
        qg.panel_draws, 10, 20, 500, scale=PANEL_SCALE
    ),
    "var2_lagged_d10_n1000": functools.partial(qg.granger_draws, 10, 1000),
    "icp_do_d16": functools.partial(qg.invariance_draws, 16, 200),
    "known_targets_ten_settings_gauss_er_d50": functools.partial(
        qg.known_target_draws, "O1_perfect_k10_p50_n1000"
    ),
    "latent_anm_er_tenth_d33_n1000": functools.partial(
        qg.hidden_cause_draws, "latent_anm_er_tenth", 33, 1000
    ),
    "dynotears_paper_d40_n1000": functools.partial(qg.series_draws, "dynotears_paper", 40, 1000),
    "jpcmciplus_half_m200_t10": functools.partial(qg.context_draws, "half-M200-T10"),
}
LARGER: dict[str, str] = {
    **dict.fromkeys(
        (
            "PC",
            "GES",
            "XGES",
            "HC",
            "Tabu",
            "IAMB",
            "InterIAMB",
            "MMPC",
            "MMHC",
            "RSMAX2",
            "BOSS",
            "GRaSP",
        ),
        "linear_gauss_er_d100",
    ),
    **dict.fromkeys(("FCI", "GFCI", "BOSS-FCI", "GRaSP-FCI"), "latent_gauss_er_d100"),
    **dict.fromkeys(("DirectLiNGAM", "ICA-LiNGAM"), "lingam_sf_d70"),
    **dict.fromkeys(("RCD", "BottomUpParceLiNGAM"), "latent_lingam_er_d17"),
    "ExactSearch": "linear_gauss_er_d16",
    "CAM": "anm_er_n1000_d30",
    "RESIT": "anm_er_n1000_d12",
    "ScoreMatching": "score_paper_standardized_d15_er1",
    "NoGAM": "score_paper_laplace_standardized_d12_er1",
    "DAS": "score_paper_standardized_d15_er1",
    "CALM": "linear_gauss_er_d15",
    "NOTEARS": "notears_paper_standardized_d60",
    "DAGMA": "notears_paper_raw_d40",
    "AGES": "ages_paper_qs0.5_qw0.5_n10000",
    "CDNOD": "domain_shift_er_d70",
    "FCI-JCI": "fci_jci_x100",
    "GIES": "interventional_gauss_er_d100",
    "MultiGroupDirectLiNGAM": "multi_group_er_d30",
    "GIN": "latent_measurement_d16",
    "ANM": "anm_pair_n1500",
    "PNL": "pnl_pair_n1500",
    "LOCI": "lsnm_pair_n1500",
    "CPCM": "lsnm_pair_n1500",
    "CPCMGraph": "anm_er_n1000_d6",
    "SkewD": "lsnm_pair_n1500",
    "SkewDMV": "anm_er_n1000_d6",
    "PCMCI": "pcmci_paper_d20_n1000",
    "PCMCIplus": "pcmciplus_paper_d40_n1000",
    "LPCMCI": "lpcmci_paper_d20_n1000",
    "J-PCMCI+": "jpcmciplus_half_m200_t10",
    "VARMA-LiNGAM": "varma_lingam_paper_d8_n300",
    "LongitudinalLiNGAM": "longitudinal_lingam_stable_d10_n20_m500",
    "Granger": "var2_lagged_d10_n1000",
    "ICP": "icp_do_d16",
    "IGSP": "known_targets_ten_settings_gauss_er_d50",
    "CAM-UV": "latent_anm_er_tenth_d33_n1000",
    "DYNOTEARS": "dynotears_paper_d40_n1000",
}


#: The methods that learn unknown targets, outside the gate, with their datasets: the gate's smaller
#: interventional dataset, whose environments are its target sets.
UNKNOWN_TARGETS: dict[str, tuple[str, ...]] = dict.fromkeys(
    ("UT-IGSP", "GnIES"), qg.INTERVENTION_DATA[:1]
)


def datasets(method: str) -> tuple[str, ...]:
    """Return the datasets ``method`` is fitted on: its gate datasets, then its larger dataset."""
    if method in OPTIONS:
        return (OPTIONS[method].dataset,)
    method = TARGET_LISTS.get(method, method)
    if method in UNKNOWN_TARGETS:
        return UNKNOWN_TARGETS[method]
    return (*qg.SUPPORTED[method].datasets, LARGER[method])


@functools.cache
def draws(dataset: str) -> tuple[qg.Draw, ...]:
    """Return the draws of a gate, larger, or option dataset, generated once per process."""
    if dataset in OPTION_DRAWS:
        return OPTION_DRAWS[dataset]()
    return LARGER_DATASETS[dataset]() if dataset in LARGER_DATASETS else qg.draws(dataset)


def corpus(methods: Iterable[str], seeds: int) -> Iterator[tuple[str, str, int, np.ndarray]]:
    """Yield ``(method, dataset, seed, data)`` for the first ``seeds`` draws of each dataset."""
    for method in methods:
        for dataset in datasets(method):
            for seed, draw in enumerate(draws(dataset)[:seeds]):
                yield method, dataset, seed, np.ascontiguousarray(draw.data)


# ==================================================================================================
# Hashes -- the full output, its graphs, and the output mapped back from a column order.
# ==================================================================================================


def digest(obj: Any) -> str:
    """Return the SHA-256 of ``obj``'s bytes: see :func:`_feed`."""
    h = hashlib.sha256()
    _feed(h, obj)
    return h.hexdigest()


def _feed(h: Any, obj: Any) -> None:
    """Feed ``obj`` to ``h`` as tagged, length-prefixed bytes.

    A float or an array enters as its little-endian bytes after its type and shape, with every NaN
    as one bit pattern; an integer as its decimal digits; a mapping in key order; a dataclass, such
    as a result or a graph, as its class name and every field. Any other type raises
    ``TypeError``, so nothing enters by its printed form.
    """
    if obj is None:
        h.update(b"N")
    elif isinstance(obj, (bool, np.bool_)):
        h.update(b"T" if obj else b"F")
    elif isinstance(obj, (int, np.integer)):
        _sized(h, b"I", str(int(obj)).encode())
    elif isinstance(obj, (float, np.floating, np.ndarray)):
        a = np.asarray(obj)
        if a.dtype.hasobject:
            raise TypeError("cannot hash an object array")
        a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
        if a.dtype.kind in "fc":
            a = np.where(np.isnan(a), np.nan, a).astype(a.dtype)
        _sized(h, b"A", f"{a.dtype.str}{a.shape}".encode())
        _sized(h, b"", a.tobytes())
    elif isinstance(obj, str):
        _sized(h, b"S", obj.encode())
    elif isinstance(obj, Mapping):
        h.update(b"M" + struct.pack("<Q", len(obj)))
        for key in sorted(obj):
            _feed(h, key)
            _feed(h, obj[key])
    elif isinstance(obj, (list, tuple)):
        h.update(b"L" + struct.pack("<Q", len(obj)))
        for item in obj:
            _feed(h, item)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        _sized(h, b"C", type(obj).__qualname__.encode())
        for field in dataclasses.fields(obj):
            _sized(h, b"K", field.name.encode())
            _feed(h, getattr(obj, field.name))
    else:
        raise TypeError(f"cannot hash a {type(obj).__name__}")


def _sized(h: Any, tag: bytes, payload: bytes) -> None:
    h.update(tag + struct.pack("<Q", len(payload)) + payload)


def graphs(obj: Any) -> Iterator[GraphStructure]:
    """Yield every graph in ``obj``: a result's graph, a time series's graph at each lag and
    occasion, and each group's graph in a list of results."""
    if isinstance(obj, GraphStructure):
        yield obj
    elif isinstance(obj, (StructureOutput, TemporalStructure)):
        for field in dataclasses.fields(obj):
            yield from graphs(getattr(obj, field.name))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from graphs(item)


def graph_digest(out: Any) -> str:
    """Return the digest of the kind, endpoint marks, and node types of every graph in ``out``."""
    return digest([(g.kind, g.to_numpy(), g.node_types) for g in graphs(out)])


def parts(out: Any) -> dict[str, str]:
    """Return the digest of each part of ``out``: its graphs, their kinds, their edges (marks and
    node types), its structure with any lag weights, its causal order, its edge weights, and each
    metadata value; a list of results prefixes each result's parts with its position."""
    if isinstance(out, list):
        return {f"{i}.{name}": d for i, o in enumerate(out) for name, d in parts(o).items()}
    return {
        "graph": graph_digest(out),
        "kind": digest([g.kind for g in graphs(out)]),
        "edges": digest([(g.to_numpy(), g.node_types) for g in graphs(out)]),
        "structure": digest(out.structure),
        "ordering": digest(out.ordering),
        "weights": digest(out.weighted_adjacency),
        **{f"metadata.{key}": digest(value) for key, value in out.metadata.items()},
    }


#: Metadata with a value for each lag and ordered pair of variables, ``[k][i][j]``.
LAG_MATRICES = ("p_values", "statistics", "links")
#: Metadata that lists node indices among other values. The column-shuffle comparison leaves it
#: out: each method lays out its lists in its own way.
NODE_LISTS = (
    "ambiguous_triples",
    "causal_orders",
    "changing_variables",
    "cluster_order",
    "clusters",
    "confounded_pairs",
    "conflicts",
    "cycle_edges",
    "dag",
    "accepted_sets",
    "context_links",
    "index_edges",
    "latent_pairs",
    "never_observed_together",
    "parents",
    "possibly_separable",
    "partial_order",
    "time_index_edges",
    "undecided_pairs",
    "unordered",
)
#: ICP's metadata with a value for each variable, ``[i]``.
PER_VARIABLE = ("p_values", "confidence_intervals")
#: The keys of a pair's two directions, first column to second and back, which a swap exchanges.
DIRECTIONS = ("forward", "backward")


def permuted(x: np.ndarray, order: np.ndarray, blocks: int) -> np.ndarray:
    """Return ``x`` with each of its ``blocks`` copies of the variables in column ``order``, in C
    order as the corpus is: some fits' bits depend on the memory layout."""
    copies = [b[..., order] for b in np.split(x, blocks, axis=-1)]
    return np.ascontiguousarray(np.concatenate(copies, axis=-1))


def unpermute(out: Any, order: np.ndarray) -> Any:
    """Return ``out``, fitted on the data's columns in ``order``, with each node moved back to its
    original column: the graphs, the time series's weights, the causal order, the edge weights,
    and the metadata in :data:`LAG_MATRICES`. Hidden nodes after the observed ones keep their
    places, a swapped pair's :data:`DIRECTIONS` trade places, ICP's target and its metadata in
    :data:`PER_VARIABLE` move back, UT-IGSP's targets move back with each setting's list sorted
    again, and the metadata in :data:`NODE_LISTS` is left out."""
    if isinstance(out, list):
        return [unpermute(o, order) for o in out]
    order = np.concatenate([order, np.arange(len(order), out.structure.n_nodes)])
    back = np.argsort(order)
    weights = out.weighted_adjacency
    swapped = len(order) == 2 and order[0] == 1
    metadata = {}
    icp = out.metadata.get("algorithm") == "icp"
    ut_igsp = out.metadata.get("algorithm") == "ut_igsp"
    for key, value in out.metadata.items():
        if key in NODE_LISTS:
            continue
        if icp and key == "target":
            value = int(order[value])
        elif icp and key in PER_VARIABLE:
            value = None if value is None else [value[i] for i in back]
        elif ut_igsp and key == "targets":
            value = {label: sorted(int(order[t]) for t in found) for label, found in value.items()}
        elif isinstance(value, Mapping) and set(value) == set(DIRECTIONS):
            forward, backward = DIRECTIONS
            value = {forward: value[backward], backward: value[forward]} if swapped else value
        elif key in LAG_MATRICES:
            value = np.asarray(value)[:, back][:, :, back].tolist()
        metadata[key] = value
    return StructureOutput.new(
        _moved(out.structure, back),
        ordering=None if out.ordering is None else [int(order[i]) for i in out.ordering],
        weighted_adjacency=None if weights is None else weights[np.ix_(back, back)],
        metadata=metadata,
    )


def _moved(value: Any, back: np.ndarray) -> Any:
    """Return ``value`` with node ``back[i]`` as node ``i``: a graph's marks and node types, and a
    time series's graphs and ``(..., n, n)`` weights. Labels stay: an array's columns are named by
    position."""
    if isinstance(value, GraphStructure):
        types = value.node_types
        return GraphStructure.from_numpy(
            value.to_numpy()[np.ix_(back, back)],
            kind=value.kind,
            labels=value.labels,
            node_types=None if types is None else np.asarray(types)[back],
            allow_self_loops=True,
        )
    if isinstance(value, TemporalStructure):
        moved = {f.name: _moved(getattr(value, f.name), back) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **moved)
    if isinstance(value, tuple):
        return tuple(_moved(v, back) for v in value)
    if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[-2:] == (len(back),) * 2:
        return value[..., back, :][..., back]
    return value


# ==================================================================================================
# Query counts -- the CI-test and local-score queries one fit makes.
# ==================================================================================================


class Counts:
    """Deterministic counts of the CI-test and local-score queries made inside :func:`counting`.

    ``ci_after_separation`` counts the CI-test queries of a pair made after a query of that pair
    returned ``p > alpha``, its first separating set; it needs ``alpha``. ``ci_sets`` and
    ``score_sets`` count the distinct conditioning sets of each size: a CI test's conditioning set
    and a local score's parent set. ``ci_queries`` and ``score_queries`` count every query, one
    per p-value or per parent set scored.
    """

    def __init__(self, alpha: float | None) -> None:
        self.alpha = alpha
        self.busy = False  # inside a counted call, whose own queries are not counted again
        self.ci_queries = self.after_separation = self.score_queries = 0
        self.separated: set[tuple[int, int]] = set()
        self.ci_sets: defaultdict[int, set[frozenset[int]]] = defaultdict(set)
        self.score_sets: defaultdict[int, set[frozenset[int]]] = defaultdict(set)

    def ci(self, queries: Iterable[tuple[int, int, Iterable[int], float]]) -> None:
        for x, y, condition_set, p in queries:
            pair = (min(int(x), int(y)), max(int(x), int(y)))
            self.ci_queries += 1
            self.after_separation += pair in self.separated
            if self.alpha is not None and p > self.alpha:
                self.separated.add(pair)
            conditions = frozenset(int(z) for z in condition_set)
            self.ci_sets[len(conditions)].add(conditions)

    def score(self, parent_sets: Iterable[Iterable[int]]) -> None:
        for parent_set in parent_sets:
            parents = frozenset(int(p) for p in parent_set)
            self.score_queries += 1
            self.score_sets[len(parents)].add(parents)

    def summary(self) -> dict[str, Any]:
        """Return the counts with a query, the sets by size as ``{"size": count}``."""
        out: dict[str, Any] = {}
        if self.ci_queries:
            out["ci_queries"] = self.ci_queries
            if self.alpha is not None:
                out["ci_after_separation"] = self.after_separation
            out["ci_sets"] = {str(k): len(v) for k, v in sorted(self.ci_sets.items())}
        if self.score_queries:
            out["score_queries"] = self.score_queries
            out["score_sets"] = {str(k): len(v) for k, v in sorted(self.score_sets.items())}
        return out


def _moves(base: Iterable[int], moves: Iterable[tuple[str, int]]) -> list[list[int]]:
    """The parent set ``base`` and ``base`` after each ``("add" | "remove", u)`` move."""
    base = list(base)
    return [base, *([*base, u] if op == "add" else [p for p in base if p != u] for op, u in moves)]


def _rows(x: Any, y: Any, conditions: Any, p_values: Any) -> list[tuple[int, int, Any, float]]:
    """The queries of a call that answers many, one per p-value: query ``i`` asks ``x[i]`` and
    ``y[i]`` given row ``i`` of the ``(N, m)`` array ``conditions``."""
    sets = np.asarray(conditions).reshape(len(x), -1) if len(x) else []
    return list(zip(x, y, sets, p_values))


#: The CI tests' query methods, each with the ``(x, y, conditioning set, p-value)`` of every test
#: one call makes, read from the call's result, the test, and the call's arguments.
CI_QUERIES: dict[str, Callable[..., Iterable[tuple[int, int, Any, float]]]] = {
    "__call__": lambda p, test, x, y, conditions=(): [(x, y, conditions, p)],
    "statistic_pvalue": lambda rp, test, x, y, condition_set=(): [(x, y, condition_set, rp[1])],
    "statistic_pvalues": lambda rp, test, x, y, conditions: _rows(x, y, conditions, rp[1]),
    "residual_statistic_pvalues": lambda rp, test, data, x, y, conditions: _rows(
        x, y, conditions, rp[1]
    ),
    "_bounded_pvalue": lambda p, test, x, y, conditions, *settings, **options: [
        (x, y, conditions, p)
    ],
    "batched_call": lambda ps, test, x, y, sets: [(x, y, s, p) for s, p in zip(sets, ps)],
    # The sets answered, up to the first separating one; the sets after it are not queries.
    "separating_prefix": lambda ps, test, x, y, sets, alpha: [
        (x, y, s, p) for s, p in zip(sets, ps)
    ],
    "first_order_pvalues": lambda ps, test, x, y, zs: [(x, y, (z,), p) for z, p in zip(zs, ps)],
    "statistic_pvalues_given": lambda rp, test, x, ys, condition_set: [
        (x, y, condition_set, p) for y, p in zip(ys, rp[1])
    ],
    "pvalues_given": lambda ps, test, target, xs, condition_set=(): [
        (target, x, condition_set, p) for x, p in zip(xs, ps)
    ],
    "_canonical_pvalues": lambda ps, test, xs, ys, sets: list(zip(xs, ys, sets, ps)),
    "marginal_pvalues": lambda ps, test: [
        (x, y, (), ps[x, y]) for x, y in zip(*np.triu_indices(len(ps), 1))
    ],
}
CI_CLASSES = (
    ci.PartialCorrelationTest,
    ci.ChiSquare,
    ci.DGTest,
    ci.CGTest,
    ci_kernel.KCI,
    ci_kernel.RCoT,
    ci_cmiknn.CMIknn,
    ci_gpdc.GPDC,
    # The tests that delete rows per test, on data with missing values: a query of theirs counts
    # in the data's columns, and the test they build for the rows of its block does not count again.
    ci_missing.TestwiseDeletion,
    ci_missing.PartialCorrelationDeletion,
    ci_missing.GaussianDeletion,
    ci_missing.BoundedDeletion,
)
#: The local scores' query methods and the score-delta functions, each with the parent sets one
#: call scores, read from the call's result, the score, and the call's arguments. A delta scores
#: a base set and each move from it.
SCORE_QUERIES: dict[str, Callable[..., Iterable[Iterable[int]]]] = {
    "score": lambda result, s, node, parents: [parents],
    "score_many": lambda result, s, node, parent_sets: parent_sets,
    "delta": lambda result, s, node, base, u, op: _moves(base, [(op, u)]),
    "delta_many_with_base": lambda result, s, node, base, moves, **options: _moves(base, moves),
    "score_many_with_base": lambda result, s, node, base, moves, **options: _moves(base, moves),
    "add_deltas": lambda result, s, node, base, **options: _moves(
        base, [("add", u) for u in range(len(s.cov)) if u != node and u not in set(base)]
    ),
    "remove_deltas": lambda result, s, node, base, **options: _moves(
        base, [("remove", u) for u in base]
    ),
}
SCORE_CLASSES = (
    score.BICScore,
    score.BDeuScore,
    score.CategoricalBICScore,
    score.DGScore,
    score.CGScore,
    score_delta.DeltaBICScore,
    score_delta.FullScoreDelta,
)
#: The modules that name the score-delta functions ``add_deltas`` and ``remove_deltas``.
DELTA_MODULES = (score_delta, hc, xges)

#: The counts of the running :func:`counting` block, or ``None``.
_active: Counts | None = None


@contextmanager
def counting(alpha: float | None) -> Generator[Counts]:
    """Count the CI-test and local-score queries made inside the block.

    Each query method of :data:`CI_CLASSES` and :data:`SCORE_CLASSES`, and each name of a
    score-delta function in :data:`DELTA_MODULES`, is wrapped for the block and restored after it.
    A call made inside another counted call is part of it and is not counted again. Only this
    process counts, so the queries of worker processes are not seen.
    """
    global _active
    patches = list(_patches())
    for owner, name, wrapped in patches:
        setattr(owner, name, _counted(wrapped, *_RECORDERS[name]))
    _active = Counts(alpha)
    try:
        yield _active
    finally:
        _active = None
        for owner, name, wrapped in patches:
            setattr(owner, name, wrapped)


def _in_data_columns(
    queries: Callable[..., Iterable[tuple[int, int, Any, float]]],
) -> Callable[..., list[tuple[int, int, Any, float]]]:
    """Return ``queries`` reading each query of a test over reordered columns as its columns of the
    data: a deletion test that ``from_columns`` maps to another matrix, as FCI's possible-d-sep
    search does in rank order, records its queries as pairs of that matrix's columns."""

    def read(
        result: Any, test: Any, *args: Any, **kwargs: Any
    ) -> list[tuple[int, int, Any, float]]:
        rows = list(queries(result, test, *args, **kwargs))
        source = getattr(test, "_source", None)
        if source is None:
            return rows
        return [
            (int(source[int(x)]), int(source[int(y)]), [int(source[int(z)]) for z in zs], p)
            for x, y, zs, p in rows
        ]

    return read


_RECORDERS: dict[str, tuple[Callable[..., Iterable[Any]], Callable[[Counts, Any], None]]] = {
    **{name: (_in_data_columns(queries), Counts.ci) for name, queries in CI_QUERIES.items()},
    **{name: (queries, Counts.score) for name, queries in SCORE_QUERIES.items()},
}


def _patches() -> Iterator[tuple[Any, str, Callable[..., Any]]]:
    """Yield ``(owner, name, function)`` for each query method or function to count, once each:
    the class that defines a method, and each module that names a score-delta function."""
    owners = set()
    for classes, names in ((CI_CLASSES, CI_QUERIES), (SCORE_CLASSES, SCORE_QUERIES)):
        for cls in classes:
            for name in names:
                owner = next((c for c in cls.__mro__ if name in vars(c)), object)
                if owner is not object:
                    owners.add((owner, name))
    for owner, name in owners:
        yield owner, name, vars(owner)[name]
    for module in DELTA_MODULES:
        for name in ("add_deltas", "remove_deltas"):
            if hasattr(module, name):
                yield module, name, getattr(module, name)


def _counted(
    function: Callable[..., Any],
    queries: Callable[..., Iterable[Any]],
    record: Callable[[Counts, Any], None],
) -> Callable[..., Any]:
    """Wrap ``function`` to record the queries of each outermost call inside :func:`counting`. An
    iterator argument is read into a list first, so the queries can read it too."""

    @functools.wraps(function)
    def counted(*args: Any, **kwargs: Any) -> Any:
        counts = _active
        if counts is None or counts.busy:
            return function(*args, **kwargs)
        args = tuple(list(a) if isinstance(a, Iterator) else a for a in args)
        counts.busy = True
        try:
            result = function(*args, **kwargs)
        finally:
            counts.busy = False
        record(counts, queries(result, *args, **kwargs))
        return result

    return counted


# ==================================================================================================
# The run and the manifest
# ==================================================================================================

#: The environment variables that set thread and worker counts, recorded in the header, and
#: ``ANDREY_SHARED_SCAN_MIN_WORK``, which picks between two paths whose p-values agree to rounding.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ANDREY_N_JOBS",
    "ANDREY_GES_PARALLEL_MIN_WORK",
    "ANDREY_SHARED_SCAN_MIN_WORK",
)
ROOT = Path(__file__).resolve().parents[1]


def commit() -> dict[str, Any]:
    """Return the checked-out commit, whether tracked files differ from it, and the imported
    Andrey package's directory with the commit of its git work tree, or ``None`` outside one."""
    package = Path(andrey.__file__).resolve().parent
    return {
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "uncommitted_changes": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "andrey_path": str(package),
        "andrey_commit": _git("rev-parse", "HEAD", cwd=package) or None,
    }


def header(code: dict[str, Any], seeds: int, shuffle: int, workers: int) -> dict[str, Any]:
    """Return ``code`` (:func:`commit`, read before the fits), the versions, the BLAS and OpenMP
    libraries loaded by now, the CPU, and the thread settings."""
    blas = [
        {key: value for key, value in pool.items() if key != "filepath"}
        for pool in threadpoolctl.threadpool_info()
    ]
    return {
        **code,
        "andrey": andrey.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": sorted(blas, key=lambda pool: json.dumps(pool, sort_keys=True)),
        "cpu": _cpu_model(),
        "machine": f"{platform.system()} {platform.machine()}",
        "threads": {
            "backend": "numpy",
            "blas_threads": 1,
            "n_jobs": 1,
            **{name: os.environ.get(name) for name in THREAD_VARIABLES},
        },
        "corpus": {"seeds": seeds, "shuffle": shuffle, "workers": workers},
        # A refit on more workers sends every GES pass, every skeleton pass, and every
        # possible-d-sep search to the worker pool.
        "refit_ges_parallel_min_work": "0" if workers > 1 else None,
        "refit_skeleton_min_pool_queries": 0 if workers > 1 else None,
        "refit_possible_dsep_min_pool_subsets": 0 if workers > 1 else None,
    }


def _git(*args: str, cwd: Path = ROOT) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    except OSError:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


@contextmanager
def _environ(name: str, value: str) -> Generator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


def row(method: str, dataset: str, seed: int, permutation: int, workers: int, out: Any) -> dict:
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "permutation": permutation,
        "workers": workers,
        "hash": digest(out),
        "graph_hash": graph_digest(out),
        "parts": parts(out),
    }


#: The parts of an output (:func:`parts`) that a column order must not change: its graphs, their
#: kinds and edges, and its causal order. The edge weights and metadata values are not among them.
STRUCTURE = ("graph", "kind", "edges", "ordering")
#: The kinds of difference between a refit and the first fit, as :func:`run` reports them:
#: ``"graph"``, a different graph; ``"structure"``, the same graphs and a different kind, edge
#: marks, or causal order; ``"bits"``, only different bits in a refit on workers; ``"values"``,
#: edge weights or metadata values of a column-order refit that differ by more than rounding;
#: ``"digits"``, edge weights or metadata values of a column-order refit that differ by rounding
#: only; ``"documented"``, a column-order difference the column-order guide documents
#: (:data:`DOCUMENTED`). The last two do not fail the check.
FAILING = ("graph", "structure", "bits", "values")
#: The column-order differences the column-order guide documents, by method: the parts of an output
#: (:func:`parts`) a column order may change, and why. A column-order refit that differs only in
#: those parts is expected.
DOCUMENTED: dict[str, tuple[frozenset[str], str]] = {
    "ICA-LiNGAM": (
        frozenset({"graph", "edges", "structure", "ordering", "weights"}),
        "FastICA can reach another solution under a new column order",
    ),
}
#: The relative and absolute differences that count as rounding between two values.
RTOL, ATOL = 1e-6, 1e-12


def _close(a: Any, b: Any) -> bool:
    """Whether ``a`` and ``b`` are the same values up to :data:`RTOL` and :data:`ATOL`: numbers and
    arrays by closeness, NaN with NaN, anything else, such as text, by equality."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_close(a[key], b[key]) for key in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        kinds = a.dtype.kind, b.dtype.kind
        if kinds[0] in "iu" and kinds[1] in "iu":
            return bool(np.array_equal(a, b))
        if kinds == ("f", "f"):
            return a.shape == b.shape and bool(np.allclose(a, b, RTOL, ATOL, equal_nan=True))
        if kinds[0] in "fiu" or kinds[1] in "fiu":
            return False  # an integer array and a float array are not the same values
        return _close(a.tolist(), b.tolist())
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    truth = bool | np.bool_
    if isinstance(a, truth) or isinstance(b, truth):  # True and 1 are not the same value
        return isinstance(a, truth) and isinstance(b, truth) and bool(a) == bool(b)
    real = (float, int, np.floating, np.integer)
    if isinstance(a, real) and isinstance(b, real):
        if isinstance(a, int | np.integer) or isinstance(b, int | np.integer):
            return isinstance(a, int | np.integer) and isinstance(b, int | np.integer) and a == b
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL) or (math.isnan(a) and math.isnan(b))
    return bool(a == b)


def _rounded(first: Any, refit: Any) -> bool:
    """Whether the edge weights, the lag and time weights of a time series's structure, and the
    metadata of ``refit`` equal ``first``'s up to rounding (:func:`_close`), and its lags and times
    are the same."""
    if isinstance(first, list):
        return len(first) == len(refit) and all(_rounded(a, b) for a, b in zip(first, refit))
    a, b = first.structure, refit.structure
    axes = all(getattr(a, name, None) == getattr(b, name, None) for name in ("lags", "times"))
    weights = all(
        _close(getattr(a, name, None), getattr(b, name, None))
        for name in ("lag_weights", "lag_weights_ma", "time_weights")
    )
    return (
        axes
        and weights
        and _close(first.weighted_adjacency, refit.weighted_adjacency)
        and _close(dict(first.metadata), dict(refit.metadata))
    )


def classify(first: Any, refit: Any, *, column_order: bool) -> tuple[str, list[str]] | None:
    """Return the kind of difference between ``refit`` and ``first`` and the names of the parts
    that differ (:func:`parts`), or ``None`` when no part differs; ``column_order`` is whether the
    refit is on another column order, not on workers. A column order's difference in weights or
    metadata is ``"digits"`` when it is rounding and ``"values"`` when it is more."""
    a, b = parts(first), parts(refit)
    changed = sorted(name for name in a.keys() | b.keys() if a.get(name) != b.get(name))
    if not changed:
        return None
    structural = {name.split(".", 1)[-1] if name[0].isdigit() else name for name in changed}
    if "graph" in structural:
        return "graph", changed
    if structural & set(STRUCTURE):
        return "structure", changed
    if not column_order:
        return "bits", changed
    return ("digits" if _rounded(first, refit) else "values"), changed


def documented(method: str, changed: list[str]) -> str | None:
    """Return why a column-order refit of ``method`` that differs in the parts ``changed`` is
    expected (:data:`DOCUMENTED`), or ``None`` when it is not."""
    allowed, reason = DOCUMENTED.get(method, (frozenset(), ""))
    return reason if changed and set(changed) <= allowed else None


def run(
    methods: list[str], seeds: int, shuffle: int = 0, workers: int = 1
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fit ``methods`` on the corpus; return the manifest's rows and the refits that differ.

    Each draw is fitted once with its queries counted. ``shuffle`` refits it on that many column
    orders, the k-th drawn from ``(seed, k)``, unless its :class:`Fit` is not ``shuffled``, and
    hashes each answer mapped back (:func:`unpermute`). ``workers`` above 1 refits
    :data:`WORKER_METHODS` on that many workers, with every GES pass, every skeleton pass of
    conditioning size two or more, and every possible-d-sep search sent to the worker pool. A refit
    differs when one of its
    :func:`parts` differs from the first fit's, mapped back the same way for a column order; each
    difference is reported as ``(kind, description)``, naming the parts that changed, with the kind
    of :func:`classify`.
    """
    rows, findings = [], []
    with andrey.config(backend="numpy", n_jobs=1), threadpool_limits(limits=1):
        for method, dataset, seed, x in corpus(methods, seeds):
            fit = FITS[method]
            with counting(fit.alpha) as counts:
                out = fit.call(x)
            rows.append({**row(method, dataset, seed, 0, 1, out), **counts.summary()})
            print(f"{method:24s} {dataset:40s} seed {seed}  {_brief(rows[-1])}", flush=True)
            refits = []  # (permutation, workers, the first fit hashed as the refit is, the refit)
            n_vars = x.shape[-1] // fit.blocks
            for k in range(1, shuffle + 1 if fit.shuffled else 1):
                rng = np.random.default_rng([seed, k])
                if fit.refit is None:
                    order = rng.permutation(n_vars)
                    refit = unpermute(fit.call(permuted(x, order, fit.blocks)), order)
                    first = unpermute(out, np.arange(n_vars))
                else:
                    refit = fit.refit(x, rng)
                    first = unpermute(out, np.arange(out.structure.n_nodes))
                refits.append((k, 1, first, refit))
            if workers > 1 and method in WORKER_METHODS:
                with (
                    andrey.config(n_jobs=workers),
                    _environ("ANDREY_GES_PARALLEL_MIN_WORK", "0"),
                    _environ("ANDREY_SKELETON_PARALLEL_MIN_WORK", "0"),
                    _environ("ANDREY_POSSIBLE_DSEP_PARALLEL_MIN_WORK", "0"),
                    _environ("ANDREY_RCOT_PARALLEL_MIN_WORK", "0"),
                ):
                    refits.append((0, workers, out, fit.call(x)))
            for permutation, n_workers, first, refit in refits:
                rows.append(row(method, dataset, seed, permutation, n_workers, refit))
                if found := classify(first, refit, column_order=permutation > 0):
                    kind, changed = found
                    line = f"{_label(rows[-1])}: {', '.join(changed)}"
                    if kind in FAILING and permutation > 0 and (why := documented(method, changed)):
                        kind, line = "documented", f"{line} ({why})"
                    findings.append((kind, line))
    return rows, findings


def _label(r: dict) -> str:
    refit = f"permutation {r['permutation']}" if r["permutation"] else f"{r['workers']} workers"
    return f"{r['method']} {r['dataset']} seed {r['seed']} {refit}"


def _brief(r: dict) -> str:
    keys = ("ci_queries", "ci_after_separation", "score_queries")
    return ", ".join(f"{key} {r[key]}" for key in keys if key in r)


def write(path: Path, head: dict, rows: list[dict]) -> None:
    """Write the manifest, one row per line, so a diff shows one line per changed fit."""
    lines = ",\n".join(f"    {json.dumps(r)}" for r in rows)
    head_text = json.dumps(head, indent=2).replace("\n", "\n  ")
    path.write_text(f'{{\n  "header": {head_text},\n  "rows": [\n{lines}\n  ]\n}}\n')


# ==================================================================================================
# Compare
# ==================================================================================================

KEY = ("method", "dataset", "seed", "permutation", "workers")


def _counters(r: dict) -> dict[str, int]:
    """Flatten a row's counts: each total, and each set size as ``ci_sets[2]``."""
    out = {}
    for key, value in r.items():
        if key == "parts":
            continue
        if isinstance(value, dict):
            out.update({f"{key}[{size}]": n for size, n in value.items()})
        elif key in ("ci_queries", "ci_after_separation", "score_queries"):
            out[key] = value
    return out


def compare(before: dict, after: dict, *, allow_fewer: bool = False) -> int:
    """Print how manifest ``after`` differs from ``before``; return 1 if a row's hash differs, a row
    is in one only, or a count rose or fell, else 0. A count that fell passes with ``allow_fewer``:
    it is either a real algorithmic change or a query the counting no longer sees.

    A row whose graph differs keeps its edges when its :func:`parts` show that only the kind
    changed. A row whose graph is unchanged differs in its bits, or, when its :func:`parts` show
    that only metadata values changed, in its metadata, and the compare names the metadata keys."""
    for key in sorted(before["header"].keys() | after["header"].keys()):
        if before["header"].get(key) != after["header"].get(key):
            print(f"header {key}: {before['header'].get(key)} -> {after['header'].get(key)}")
    old = {tuple(r[k] for k in KEY): r for r in before["rows"]}
    new = {tuple(r[k] for k in KEY): r for r in after["rows"]}
    graph, kind, bits, metadata, raised, lowered = [], [], [], [], [], []
    keys: set[str] = set()
    for key in sorted(old.keys() & new.keys(), key=str):
        a, b = old[key], new[key]
        label = " ".join(map(str, key))
        if a["graph_hash"] != b["graph_hash"]:
            changed = kind_change(a.get("parts", {}), b.get("parts", {}))
            if changed is None:
                graph.append(label)
            else:
                kind.append(f"{label} {', '.join(changed)}".rstrip())
                keys.update(changed)
        elif a["hash"] != b["hash"]:
            changed = metadata_keys(a.get("parts", {}), b.get("parts", {}))
            if changed:
                metadata.append(f"{label} {', '.join(changed)}")
                keys.update(changed)
            else:
                bits.append(label)
        ca, cb = _counters(a), _counters(b)
        for name in sorted(ca.keys() | cb.keys()):
            change = f"{label} {name}: {ca.get(name, 0)} -> {cb.get(name, 0)}"
            if cb.get(name, 0) > ca.get(name, 0):
                raised.append(change)
            elif cb.get(name, 0) < ca.get(name, 0):
                lowered.append(change)
    only = sorted(" ".join(map(str, k)) for k in old.keys() ^ new.keys())
    for title, lines in (
        ("graph differs", graph),
        ("only the kind differs", kind),
        ("only bits differ", bits),
        ("only metadata differs", metadata),
        ("in one manifest only", only),
        ("count rose", raised),
        ("count fell", lowered),
    ):
        for line in lines:
            print(f"{title}: {line}")
    named = f" ({', '.join(sorted(keys))})" if keys else ""
    print(
        f"\n{len(old.keys() & new.keys())} rows in both: {len(graph)} with a different graph, "
        f"{len(bits)} with only different bits, {len(metadata)} with only different metadata, "
        f"{len(kind)} with only a different kind{named}; {len(only)} in one only; "
        f"{len(raised)} counts rose, {len(lowered)} fell."
    )
    differs = graph or kind or bits or metadata or only or raised
    return 1 if differs or (lowered and not allow_fewer) else 0


def metadata_keys(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """Return the metadata keys whose digests differ between two rows' :func:`parts`, or ``[]``
    when another part differs too or the rows record no parts."""
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if not changed or any("metadata." not in name for name in changed):
        return []
    return sorted({name.split("metadata.", 1)[1] for name in changed})


def kind_change(before: Mapping[str, str], after: Mapping[str, str]) -> list[str] | None:
    """Return the metadata keys that differ when two rows' :func:`parts` differ in the graphs'
    kind alone, their edges, order, and weights the same; else ``None``."""
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    other = {name.rsplit(".", 1)[-1] for name in changed if "metadata." not in name}
    if "kind" not in other or not other <= {"graph", "structure", "kind"}:
        return None
    return sorted({name.split("metadata.", 1)[1] for name in changed if "metadata." in name})


# ==================================================================================================
# Command line
# ==================================================================================================


#: How each kind of difference reads in the report.
LABELS = {
    "graph": "graph differs",
    "structure": "structure differs",
    "bits": "only bits differ",
    "values": "weights or metadata differ by more than rounding",
    "digits": "weights or metadata differ by rounding only (not a failure)",
    "documented": "differs as the column-order guide documents (not a failure)",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Hash every supported method's full output.")
    ap.add_argument("methods", nargs="*", help="methods to fit (default: all supported)")
    ap.add_argument("-o", "--out", type=Path, help="write the manifest to this file")
    ap.add_argument("--seeds", type=int, default=SEEDS, help="draws per dataset")
    ap.add_argument("--shuffle", type=int, default=0, metavar="K", help="refit on K column orders")
    ap.add_argument("--workers", type=int, default=1, metavar="N", help="refit on N workers")
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    ap.add_argument(
        "--allow-fewer-queries", action="store_true", help="with --compare, pass when a count fell"
    )
    args = ap.parse_args(argv)

    if args.compare:
        before, after = (json.loads(path.read_text()) for path in args.compare)
        return compare(before, after, allow_fewer=args.allow_fewer_queries)
    unknown = sorted(set(args.methods) - set(FITS))
    if unknown:
        ap.error(f"unknown methods {unknown}; known: {list(FITS)}")
    code = commit()
    rows, findings = run(args.methods or list(FITS), args.seeds, args.shuffle, args.workers)
    if args.out:
        write(args.out, header(code, args.seeds, args.shuffle, args.workers), rows)
    for kind, line in findings:
        print(f"{LABELS[kind]}: {line}")
    refits = sum(r["permutation"] > 0 or r["workers"] > 1 for r in rows)
    count = {kind: sum(k == kind for k, _ in findings) for kind in LABELS}
    print(
        f"\n{len(rows)} fits hashed. Of {refits} refits, {count['graph']} have a different graph, "
        f"{count['structure']} a different structure, {count['values']} different values, and "
        f"{count['bits']} only different bits; {count['digits']} differ only by rounding in their "
        "weights or metadata under a column order, which does not fail the check. Differences the "
        f"column-order guide documents, which do not fail it either: {count['documented']}."
    )
    return 1 if any(kind in FAILING for kind, _ in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
