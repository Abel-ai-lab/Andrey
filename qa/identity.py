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
``N`` workers, with every GES pass, every skeleton pass of conditioning size two or more, every
possible-d-sep search, and every full-graph fit of IAMB, Inter-IAMB, and MMPC sent to the pool. Both
report each refit whose answer differs from the first fit. A different worker count must not
change a bit, so any difference fails it. A column order
fails the check when a graph or a causal order differs (:data:`STRUCTURE`) or when an edge weight or
metadata value differs by more than rounding (:data:`RTOL`, or :data:`ATOL` near zero). The last
bits of such values, such as p-values, depend on the order a sum runs in, so a refit that differs
only by rounding is reported as information.

``--checkpoints`` (with ``-o``) also records, for each draw's first fit, a digest of the
algorithm's state at its stages (:class:`Checkpoints`), the stages the package passes to its
private sink, ``andrey.core._checkpoint.TAKE``, which :func:`checkpointing` sets for the fit. The
manifest row gains two fields, the pieces' bytes go to a sidecar file next to the manifest, and
``--compare`` of two such manifests names, for each fit that differs, the first stage and piece
where the builds part, and groups the differences by fingerprint (:func:`report_checkpoints`). The
exit status is the one the output hashes and counts decide.

    uv run python -m qa.identity -o before.json             # fit the corpus, write a manifest
    uv run python -m qa.identity --compare before.json after.json  # exit 1 on any difference
    uv run python -m qa.identity --shuffle 3                # exit 1 if an order changes a graph
    uv run python -m qa.identity GES GFCI --workers 4       # exit 1 if 4 workers change one
    uv run python -m qa.identity PC -o a.json --checkpoints # a manifest and its sidecar
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_limits

import andrey
from andrey import GraphStructure, StructureOutput, TemporalStructure, data
from andrey.core import ci, ci_cmiknn, ci_gpdc, ci_kernel, ci_missing, score, score_delta
from andrey.data import missingness
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

# --------------------------------------------------------------------------------------------------
# Structural EM -- fitted outside the gate, on a gate dataset with values removed at random.
# --------------------------------------------------------------------------------------------------

#: The gate dataset of the structural-EM fits, and the share of its values removed completely at
#: random (:func:`andrey.data.missingness.mcar`) with the draw's seed 0, the rate the EM tests use.
EM_DATASET, EM_RATE = "linear_gauss_er_d10", 0.3
#: The structural-EM fits, one with a search over graphs and one with a search over weights: each
#: method's ``missing="em"`` on a draw of :data:`EM_DATASET` with values removed. A column order
#: moves the draw's missing values, so they are not refitted under ``--shuffle``.
EM_FITS: dict[str, Callable[..., StructureOutput]] = {
    "GES-em": andrey.ges,
    "NOTEARS-em": andrey.notears,
}


def _with_missing(
    search: Callable[..., StructureOutput],
) -> Callable[[np.ndarray], StructureOutput]:
    def fit(x: np.ndarray) -> StructureOutput:
        mask = missingness.draw(x, missingness.mcar(EM_RATE), seed=0).mask
        return search(np.where(mask, np.nan, x), missing="em")

    return fit


FITS |= {name: Fit(_with_missing(search), shuffled=False) for name, search in EM_FITS.items()}

#: The methods that run on ``andrey.config(n_jobs=...)``: GES, GES in each structural-EM round of
#: ``GES-em``, GFCI's GES phase, the skeleton searches of PC, FCI, FCI-JCI, GFCI, CD-NOD, and
#: RSMAX2's PC phase, the possible-d-sep searches of FCI, FCI-JCI, and GFCI, and the full-graph
#: searches of IAMB, Inter-IAMB, and MMPC, which MMHC and RSMAX2's restrict phase run, in worker
#: processes, PNL's two directions on threads, and the lists of targets of :data:`TARGET_LISTS`,
#: and the option fits of those methods.
_WORKERS = (
    "PC",
    "FCI",
    "FCI-JCI",
    "GFCI",
    "CDNOD",
    "RSMAX2",
    "GES",
    "GES-em",
    "PNL",
    "IAMB",
    "InterIAMB",
    "MMPC",
    "MMHC",
)
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
    if method in EM_FITS:
        return (EM_DATASET,)
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
    per p-value or per parent set scored; a test that answers stacked queries up to the first
    separating one (``pvalues_until_separated``) counts the p-values it returns, not the queries
    it computed ahead and dropped.
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
    # The stacked queries answered, up to the first separating one; the queries after it are
    # computed ahead and dropped, so they are not queries.
    "pvalues_until_separated": lambda ps, test, x, y, conditions, alpha: _rows(
        x, y, conditions, ps
    ),
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
    ci_missing.ChiSquareDeletion,
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
# Checkpoints -- a digest of each algorithm's state at its stages, for ``--checkpoints``.
# ==================================================================================================

#: How the elements of a piece's array read, by form. ``pairs``: a symmetric adjacency as its upper
#: triangle in ``np.triu_indices(d, 1)`` order, one element per pair. ``matrix``: row-major, one
#: element per entry. ``sets``: a mapping from keys to collections as the rows ``[*key, k,
#: *members]`` in key order, a separating set or the marks an orientation round implied, as the
#: round leaves them: the marks it applied are gone. ``values``: a mapping from keys to numbers as
#: the rows ``[*key, value]``. ``list``: a sequence flattened in order, a string as its length
#: and its characters. ``scalar``: one number, an integer too wide for 64 bits as its 64-bit limbs.
#: ``text``: the bytes of a string. The width of a piece is its row length for ``matrix`` and
#: ``values``, the node count for ``pairs``, and the length of a key for ``sets``.
FORMS = ("pairs", "matrix", "sets", "values", "list", "scalar", "text")


@dataclass(frozen=True)
class Piece:
    """One piece of a record: its name, its form (:data:`FORMS`) and width, the digest of its
    array, and the canonical 1-D array itself."""

    name: str
    form: str
    width: int
    digest: str
    array: np.ndarray


@dataclass(frozen=True)
class Record:
    """One call of the sink: the stage, its level (the conditioning size of a skeleton search, a
    round of structural EM) when the algorithm has one, and the pieces of the state."""

    stage: str
    level: int | None
    pieces: tuple[Piece, ...]


class Label(NamedTuple):
    """What the manifest and the sidecar keep of a piece: its record's position in the fit, the
    record's stage and level, the piece's name, form, width, and digest, and its array's length."""

    record: int
    stage: str
    level: int | None
    piece: str
    form: str
    width: int
    digest: str
    length: int


def _wide(value: int) -> list[int]:
    """``value`` as 64-bit limbs, little-endian: one limb for a value that fits in 64 bits."""
    size = (value.bit_length() // 64 + 1) * 8
    return np.frombuffer(value.to_bytes(size, "little", signed=True), dtype="<i8").tolist()


def _flat(value: Any) -> list[Any]:
    """Flatten ``value`` to numbers in order: a set or frozenset sorted, a string as its length
    and the code points, an integer as its limbs (:func:`_wide`)."""
    out: list[Any] = []

    def walk(item: Any) -> None:
        if isinstance(item, str):
            out.append(len(item))
            out.extend(ord(c) for c in item)
        elif isinstance(item, (set, frozenset)):
            for member in sorted(item):
                walk(member)
        elif isinstance(item, (list, tuple)):
            for member in item:
                walk(member)
        elif isinstance(item, np.ndarray):
            walk(item.ravel().tolist())
        elif isinstance(item, (bool, np.bool_, int, np.integer)):
            out.extend(_wide(int(item)))
        else:
            out.append(float(item))

    walk(value)
    return out


def _numbers(values: list[Any]) -> np.ndarray:
    """``values`` as one int64 array, or a float64 array when any is a float."""
    floats = any(isinstance(v, float) for v in values)
    return np.array(values, dtype=np.float64 if floats else np.int64)


def _keys(mapping: Mapping[Any, Any]) -> tuple[list[Any], int]:
    """The keys of ``mapping`` in order, and their length: a pair or an integer."""
    keys = sorted(mapping, key=lambda k: tuple(int(e) for e in k) if isinstance(k, tuple) else (k,))
    return keys, 0 if not keys else (len(keys[0]) if isinstance(keys[0], tuple) else 1)


def _mapping(mapping: Mapping[Any, Any]) -> tuple[str, int, np.ndarray]:
    keys, arity = _keys(mapping)
    rows: list[Any] = []
    numeric = bool(keys) and all(
        isinstance(mapping[k], (int, float, np.integer, np.floating)) for k in keys
    )
    for key in keys:
        head = _flat(key)
        if numeric:
            rows.extend((*head, float(mapping[key])))
        else:
            members = _flat(mapping[key])
            rows.extend((*head, len(members), *members))
    if numeric:
        return "values", arity + 1, np.array(rows, dtype=np.float64)
    return "sets", arity, np.array(rows, dtype=np.int64)


def _canonical(name: str, value: Any) -> tuple[str, int, np.ndarray]:
    """Return a piece's form, width, and canonical 1-D array, copied, so that a later change to
    the search's state does not reach the record."""
    if name == "parents":  # the bitset rows of HC and Tabu: row u holds the children of u
        d = len(value)
        bits = [[(row >> v) & 1 for v in range(d)] for row in value]
        return "matrix", d, np.array(bits, dtype=np.int64).ravel()
    if isinstance(value, Mapping):
        return _mapping(value)
    if isinstance(value, str):
        return "text", 0, np.frombuffer(value.encode(), dtype=np.uint8).copy()
    if callable(value):
        return "text", 0, np.frombuffer(value.__name__.encode(), dtype=np.uint8).copy()
    if isinstance(value, np.ndarray) and value.ndim == 2:
        kind = "f" if value.dtype.kind == "f" else "i"
        a = np.array(value, dtype=np.float64 if kind == "f" else np.int64)
        if name == "adjacency" and a.shape[0] == a.shape[1] and np.array_equal(a, a.T):
            return "pairs", a.shape[0], a[np.triu_indices(a.shape[0], 1)]
        return "matrix", a.shape[1], a.ravel()
    if isinstance(value, (bool, int, float, np.number, np.bool_)):
        return "scalar", 0, _numbers(_flat(value))
    return "list", 0, _numbers(_flat(value))


def _piece(name: str, form: str, width: int, array: np.ndarray) -> Piece:
    return Piece(name, form, width, digest(array), array)


class Checkpoints:
    """The recorder of :func:`checkpointing`: called as ``take(stage, level, **pieces)``, it
    serializes each piece at once (:func:`_canonical`), since the search rewrites its state in
    place, and keeps the :class:`Record`. A piece that is ``None`` is left out. An array of more
    than two dimensions is flattened, so its shape follows it as a piece of its own,
    ``<name>.shape``.

    A skeleton search's ``skeleton.size`` record holds the whole adjacency but only the pairs the
    size removed, with their separating sets and p-values, read from the cumulative mappings;
    ``skeleton.end`` resets that, so a second search in one fit starts from no pairs.
    """

    def __init__(self) -> None:
        self.records: list[Record] = []
        self._removed: set[Any] = set()

    def __call__(self, stage: str, level: int | None = None, **pieces: Any) -> None:
        if stage == "skeleton.size":
            pieces = self._removed_at_this_size(pieces)
        elif stage == "skeleton.end":
            self._removed = set()
        kept = []
        for name, value in pieces.items():
            if value is not None:
                kept.append(_piece(name, *_canonical(name, value)))
                if isinstance(value, np.ndarray) and value.ndim > 2:
                    shape = np.array(value.shape, dtype=np.int64)
                    kept.append(_piece(f"{name}.shape", "list", 0, shape))
        self.records.append(Record(stage, level, tuple(kept)))

    def _removed_at_this_size(self, pieces: dict[str, Any]) -> dict[str, Any]:
        sepsets, pvalues = pieces["sepsets"], pieces["pvalues"]
        new = [pair for pair in sepsets if pair not in self._removed]
        self._removed.update(new)
        return {
            **pieces,
            "sepsets": {pair: sepsets[pair] for pair in new},
            "pvalues": {pair: pvalues[pair] for pair in new if pair in pvalues},
        }


@contextmanager
def checkpointing(enabled: bool) -> Generator[Checkpoints | None]:
    """Record the stages of the fit run inside the block: yield the recorder and set the package's
    sink to it, reset on exit, even after an exception. With ``enabled`` false, yield ``None`` and
    touch nothing: the package module the sink lives in is not imported, so a build without it
    runs the plain harness."""
    if not enabled:
        yield None
        return
    sink = importlib.import_module("andrey.core._checkpoint")
    recorder = Checkpoints()
    token = sink.TAKE.set(recorder)
    try:
        yield recorder
    finally:
        sink.TAKE.reset(token)


def labels(records: Iterable[Record]) -> list[Label]:
    """One :class:`Label` per piece of ``records``, in order."""
    return [
        Label(i, r.stage, r.level, p.name, p.form, p.width, p.digest, int(p.array.size))
        for i, r in enumerate(records)
        for p in r.pieces
    ]


def checkpoint_fields(records: list[Record]) -> dict[str, Any]:
    """The two fields a manifest row gains: the SHA-256 of the fit's labels in order, and the number
    of records. A fit with no record has neither."""
    if not records:
        return {}
    text = json.dumps([list(label) for label in labels(records)])
    return {
        "checkpoints": hashlib.sha256(text.encode()).hexdigest(),
        "checkpoint_records": len(records),
    }


def sidecar_name(manifest: Path) -> str:
    """The file name of the sidecar of ``manifest``, which holds the pieces' arrays."""
    return f"{manifest.name}.checkpoints.npz"


def _array_key(key: tuple) -> str:
    return "|".join(map(str, key))


def write_checkpoints(path: Path, records: Mapping[tuple, list[Record]]) -> None:
    """Write the sidecar: per fit, keyed by the row's :data:`KEY` values, ``labels``, a JSON list of
    :class:`Label`, and one array per piece."""
    arrays: dict[str, np.ndarray] = {}
    for key, fit in records.items():
        name = _array_key(key)
        arrays[f"{name}/labels"] = np.array(json.dumps([list(label) for label in labels(fit)]))
        for i, array in enumerate(p.array for r in fit for p in r.pieces):
            arrays[f"{name}/{i}"] = array
    # The array names are the keyword arguments.
    np.savez_compressed(path, **arrays)  # ty: ignore[invalid-argument-type]


@dataclass(frozen=True)
class Stages:
    """The stages of one fit: its labels and the array of each piece, by position."""

    labels: list[Label]
    array: Callable[[int], np.ndarray]


def stages_of(records: list[Record]) -> Stages:
    """The stages of a fit as :func:`run` recorded them."""
    arrays = [p.array for r in records for p in r.pieces]
    return Stages(labels(records), arrays.__getitem__)


def read_stages(path: Path, rows: Iterable[dict]) -> dict[tuple, Stages]:
    """The stages of each fit of ``rows`` that has records, read lazily from the sidecar at
    ``path``."""
    npz = np.load(path, allow_pickle=False)
    out = {}
    for r in rows:
        key = tuple(r[k] for k in KEY)
        name = _array_key(key)
        if "checkpoints" in r:
            if f"{name}/labels" not in npz:
                raise ValueError(f"the sidecar has no checkpoints for the fit {name}")
            text = str(npz[f"{name}/labels"])
            if hashlib.sha256(text.encode()).hexdigest() != r["checkpoints"]:
                raise ValueError(
                    f"the sidecar's checkpoints of the fit {name} are not the manifest's"
                )
            fit = [Label(*entry) for entry in json.loads(text)]
            out[key] = Stages(fit, functools.partial(_load, npz, name))
    return out


def _load(npz: Any, name: str, i: int) -> np.ndarray:
    return npz[f"{name}/{i}"]


#: What reading a sidecar raises when it is not an archive, is cut short, lacks a member, or holds a
#: member that does not decompress.
SIDECAR_ERRORS = (ValueError, OSError, EOFError, KeyError, zipfile.BadZipFile, zlib.error)


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


def header(
    code: dict[str, Any], seeds: int, shuffle: int, workers: int, checkpoints: str | None = None
) -> dict[str, Any]:
    """Return ``code`` (:func:`commit`, read before the fits), the versions, the BLAS and OpenMP
    libraries loaded by now, the CPU, and the thread settings; with ``checkpoints``, the name of
    the sidecar that holds the pieces of the stages (:func:`sidecar_name`)."""
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
        # A refit on more workers sends every GES pass, every skeleton pass, every possible-d-sep
        # search, and every full-graph fit of IAMB, Inter-IAMB, and MMPC to the worker pool.
        "refit_ges_parallel_min_work": "0" if workers > 1 else None,
        "refit_skeleton_min_pool_queries": 0 if workers > 1 else None,
        "refit_possible_dsep_min_pool_subsets": 0 if workers > 1 else None,
        "refit_blanket_parallel_min_work": 0 if workers > 1 else None,
        **({} if checkpoints is None else {"checkpoints": checkpoints}),
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
    methods: list[str],
    seeds: int,
    shuffle: int = 0,
    workers: int = 1,
    checkpoints: dict[tuple, list[Record]] | None = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Fit ``methods`` on the corpus; return the manifest's rows and the refits that differ.

    Each draw is fitted once with its queries counted. ``shuffle`` refits it on that many column
    orders, the k-th drawn from ``(seed, k)``, unless its :class:`Fit` is not ``shuffled``, and
    hashes each answer mapped back (:func:`unpermute`). ``workers`` above 1 refits
    :data:`WORKER_METHODS` on that many workers, with every GES pass, every skeleton pass of
    conditioning size two or more, every possible-d-sep search, and every full-graph fit of IAMB,
    Inter-IAMB, and MMPC sent to the worker pool. A refit
    differs when one of its
    :func:`parts` differs from the first fit's, mapped back the same way for a column order; each
    difference is reported as ``(kind, description)``, naming the parts that changed, with the kind
    of :func:`classify`.

    With a dict as ``checkpoints``, each draw's first fit, one worker and no column order, also
    records the stages of the algorithm (:func:`checkpointing`): the dict gets the records by the
    row's :data:`KEY` values, and the row the two fields of :func:`checkpoint_fields`. The refits
    run with the sink unset.
    """
    rows, findings = [], []
    with andrey.config(backend="numpy", n_jobs=1), threadpool_limits(limits=1):
        for method, dataset, seed, x in corpus(methods, seeds):
            fit = FITS[method]
            with counting(fit.alpha) as counts, checkpointing(checkpoints is not None) as recorder:
                out = fit.call(x)
            rows.append({**row(method, dataset, seed, 0, 1, out), **counts.summary()})
            if recorder is not None and recorder.records:
                rows[-1] |= checkpoint_fields(recorder.records)
                assert checkpoints is not None
                checkpoints[tuple(rows[-1][k] for k in KEY)] = recorder.records
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
                    _environ("ANDREY_BLANKET_PARALLEL_MIN_WORK", "0"),
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


def compare(
    before: dict,
    after: dict,
    *,
    allow_fewer: bool = False,
    stages: tuple[Mapping[tuple, Stages], Mapping[tuple, Stages]] | None = None,
) -> int:
    """Print how manifest ``after`` differs from ``before``; return 1 if a row's hash differs, a row
    is in one only, or a count rose or fell, else 0. A count that fell passes with ``allow_fewer``:
    it is either a real algorithmic change or a query the counting no longer sees.

    With the ``stages`` of both manifests' fits, read from their sidecars or recorded by
    :func:`run`, it also reports, after the lines above, the first stage and piece where each
    differing fit parts (:func:`report_checkpoints`). The report does not change the return value.

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
    with_records = [any("checkpoints" in r for r in m["rows"]) for m in (before, after)]
    if stages is not None:
        report_checkpoints(old, new, stages)
    elif any(with_records):
        reason = "only one manifest has them" if not all(with_records) else "no sidecar was read"
        print(f"checkpoints: {reason}, so there is no checkpoint report")
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


# --------------------------------------------------------------------------------------------------
# The first stage where two builds part (``--checkpoints``)
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Difference:
    """Where two fits' stages first part, in the first piece whose label or digest differs.

    ``kind`` is ``"value"``, the same piece with different elements; ``"bits"``, a float piece
    whose differing elements all agree up to :data:`RTOL` and :data:`ATOL`; ``"sequence"``, another
    stage or piece at that position, or the same bytes under another record, form, or width;
    ``"missing"``, a fit that ends first; or ``"after"``, fits whose stages are all equal and whose
    outputs differ, so they part after the last stage (:func:`_after_the_last`). ``index`` is the
    first element whose bytes differ, ``where`` reads it by the piece's form, ``change`` shows the
    two values, and ``lengths`` are the arrays' lengths (:data:`FORMS`).
    """

    record: int
    stage: str
    level: int | None
    piece: str
    kind: str
    index: int | None = None
    lengths: tuple[int, int] = (0, 0)
    where: str = ""
    change: str = ""

    def fingerprint(self, method: str) -> tuple:
        """The method, stage, level, piece, kind, first differing element, and both lengths."""
        return (method, self.stage, self.level, self.piece, self.kind, self.index, *self.lengths)

    def stage_text(self) -> str:
        return self.stage + ("" if self.level is None else f" level {self.level}")


def _element(label: Label, array: np.ndarray, index: int) -> str:
    """Read element ``index`` of a piece's array by the piece's form."""
    form, width = label.form, label.width
    if form == "pairs" and index < width * (width - 1) // 2:
        a, b = (int(v[index]) for v in np.triu_indices(width, 1))
        return f"pair ({a}, {b})"
    if form == "matrix" and width:
        return "entry ({}, {})".format(*divmod(index, width))
    if form == "values" and width > 1:
        at = index // width * width
        key = ", ".join(str(int(v)) for v in array[at : at + width - 1])
        return f"pair ({key})" if width == 3 else f"key ({key})"
    if form == "sets":
        at = 0
        while at < array.size:
            end = at + width + 1 + int(array[at + width])
            if index < end:
                key = ", ".join(str(int(v)) for v in array[at : at + width])
                return f"pair ({key})" if width == 2 else f"key ({key})"
            at = end
    return {"scalar": "value", "text": f"character {index}"}.get(form, f"position {index}")


def _show(array: np.ndarray, index: int) -> str:
    if index >= array.size:
        return "none"
    value = array[index].item()
    return str(value) if isinstance(value, int) else f"{value:.17g}"


def _bytes_differ(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which of the first ``min(len)`` elements of ``x`` and ``y`` differ in their bytes, with every
    NaN as one bit pattern."""
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    if x.dtype != y.dtype:  # a changed dtype is a changed representation from the first element
        return np.ones(n, dtype=bool)
    if x.dtype.kind == "f":
        x, y = (np.where(np.isnan(a), np.nan, a).view(np.uint64) for a in (x, y))
    return x != y


def first_difference(before: Stages, after: Stages) -> Difference | None:
    """Walk two fits' labels in order to the first position where the stage, level, or piece, or
    the digest, differs; ``None`` when the fits agree."""
    for i in range(max(len(before.labels), len(after.labels))):
        a = before.labels[i] if i < len(before.labels) else None
        b = after.labels[i] if i < len(after.labels) else None
        if a is None or b is None:
            side = a or b
            assert side is not None
            lengths = (a.length if a else 0, b.length if b else 0)
            return Difference(
                side.record, side.stage, side.level, side.piece, "missing", None, lengths
            )
        if (a.stage, a.level, a.piece) != (b.stage, b.level, b.piece):
            here = f"{a.stage} {a.piece}"
            there = f"{b.stage} {b.piece}"
            return Difference(
                a.record, a.stage, a.level, a.piece, "sequence", None, where=f"{here} -> {there}"
            )
        if a.digest == b.digest:
            if (a.record, a.form, a.width, a.length) == (b.record, b.form, b.width, b.length):
                continue
            layout = (
                f"{a.stage} {a.piece} (record {a.record}, {a.form}, width {a.width}) -> "
                f"{b.stage} {b.piece} (record {b.record}, {b.form}, width {b.width})"
            )
            return Difference(
                a.record, a.stage, a.level, a.piece, "sequence", None, (a.length, b.length), layout
            )
        x, y = before.array(i), after.array(i)
        differs = _bytes_differ(x, y)
        index = int(np.flatnonzero(differs)[0]) if differs.any() else min(x.size, y.size)
        close = (
            x.size == y.size
            and x.dtype == y.dtype == np.float64
            and np.allclose(x[differs], y[differs], RTOL, ATOL, equal_nan=True)
        )
        source, array = (a, x) if index < x.size else (b, y)
        return Difference(
            a.record,
            a.stage,
            a.level,
            a.piece,
            "bits" if close else "value",
            index,
            (a.length, b.length),
            _element(source, array, index),
            f"{_show(x, index)} -> {_show(y, index)}",
        )
    return None


def _after_the_last(a: dict, b: dict) -> Difference:
    """The difference of two fits with equal stages and different outputs: the stage ``output``,
    and as its piece the :func:`parts` of the output that differ, or ``hash`` when the rows record
    none. The steps a method takes after its last stage, such as the conversion of a DAG to its
    CPDAG, show here."""
    before, after = a.get("parts", {}), b.get("parts", {})
    changed = sorted(n for n in before.keys() | after.keys() if before.get(n) != after.get(n))
    return Difference(
        a.get("checkpoint_records", 0), "output", None, ", ".join(changed) or "hash", "after"
    )


def _records(before: int, after: int) -> str:
    """Which record of how many: the count when both fits have as many, else each fit's."""
    return f"of {before}" if before == after else f"({before} before, {after} after)"


def _describe(d: Difference) -> str:
    where = f"{d.stage_text()}, {d.piece}"
    if d.kind == "after":
        return f"the output differs in {d.piece} (after)"
    if d.kind == "sequence":
        return f"{where}: the builds record another stage here, {d.where} (sequence)"
    if d.kind == "missing":
        return f"{where}: in one fit only (missing)"
    return (
        f"{where}: element {d.index} = {d.where}, {d.change}; "
        f"{d.lengths[0]} elements before, {d.lengths[1]} after ({d.kind})"
    )


def _text(fingerprint: tuple) -> str:
    method, stage, level, piece, kind, index, before, after = fingerprint
    at = stage if level is None else f"{stage} level {level}"
    return f"{method} {at} {piece} {kind} {'-' if index is None else index} {before} {after}"


def report_checkpoints(
    old: Mapping[tuple, dict], new: Mapping[tuple, dict], stages: tuple[Mapping, Mapping]
) -> None:
    """Print, for each row in both manifests whose checkpoints differ and whose output differs, the
    first stage and piece that differ; for each row whose checkpoints are all equal and whose output
    differs, the parts of the output that differ (:func:`_after_the_last`); the rows grouped by
    fingerprint and by stage; and, as information, the rows whose output is equal. Later stages
    follow from the first and are not reported. A sidecar piece that cannot be read ends the report
    with a line saying so. The exit status of the compare does not read it."""
    empty = Stages([], lambda i: np.empty(0, dtype=np.int64))
    found, equal = [], []
    try:
        for key in sorted(old.keys() & new.keys(), key=str):
            a, b = old[key], new[key]
            counts = (a.get("checkpoint_records", 0), b.get("checkpoint_records", 0))
            if a.get("checkpoints") == b.get("checkpoints"):
                if "checkpoints" in a and a["hash"] != b["hash"]:
                    found.append((key, counts, _after_the_last(a, b)))
                continue
            difference = first_difference(stages[0].get(key, empty), stages[1].get(key, empty))
            if difference is not None:
                (equal if a["hash"] == b["hash"] else found).append((key, counts, difference))
    except SIDECAR_ERRORS as error:
        print(f"checkpoints not compared: a sidecar piece cannot be read ({error!r})")
        return
    for key, counts, d in found:
        label = " ".join(map(str, key))
        at = f"after all {counts[0]} records" if d.kind == "after" else None
        at = at or f"record {d.record + 1} {_records(*counts)}"
        print(f"first checkpoint difference: {label}: {at}, {_describe(d)}")
    prints = Counter(d.fingerprint(key[0]) for key, _, d in found)
    for fingerprint, count in sorted(prints.items(), key=lambda kv: (-kv[1], _text(kv[0]))):
        print(f"fingerprint {count} rows: {_text(fingerprint)}")
    where: defaultdict[tuple, defaultdict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    rows: Counter[tuple] = Counter()
    for key, _, d in found:
        group = (d.stage_text(), d.piece, d.kind)
        where[group][key[0]].add(key[1])
        rows[group] += 1
    for group, count in sorted(rows.items(), key=lambda kv: (-kv[1], kv[0])):
        methods = ", ".join(
            f"{m} ({', '.join(sorted(ds))})" for m, ds in sorted(where[group].items())
        )
        print(f"by stage {count} rows: {group[0]} {group[1]} {group[2]} -- {methods}")
    for key, _, d in equal:
        label = " ".join(map(str, key))
        print(
            f"checkpoints differ, output equal: {label}: record {d.record + 1}, "
            f"{d.stage_text()} {d.piece} ({d.kind})"
        )


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


def _sidecar_stages(
    paths: list[Path], manifests: list[dict]
) -> tuple[dict[tuple, Stages], dict[tuple, Stages]] | None:
    """The stages of the fits of both manifests, read from the sidecars their headers name, next
    to each manifest; ``None`` unless both name one that exists."""
    names = [m["header"].get("checkpoints") for m in manifests]
    if not all(names):
        return None
    sidecars = [path.with_name(name) for path, name in zip(paths, names, strict=True)]
    if not all(s.is_file() for s in sidecars):
        return None
    try:
        before, after = (
            read_stages(s, m["rows"]) for s, m in zip(sidecars, manifests, strict=True)
        )
    except SIDECAR_ERRORS as error:
        print(f"checkpoints not compared: {error}")
        return None
    return before, after


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
    ap.add_argument(
        "--checkpoints",
        action="store_true",
        help="record the stages of each fit, for --compare to name where builds part (needs -o)",
    )
    args = ap.parse_args(argv)

    if args.compare:
        manifests = [json.loads(path.read_text()) for path in args.compare]
        return compare(
            *manifests,
            allow_fewer=args.allow_fewer_queries,
            stages=_sidecar_stages(args.compare, manifests),
        )
    if args.checkpoints:
        if not args.out:
            ap.error("--checkpoints needs -o: the pieces of the stages go next to the manifest")
        try:
            importlib.import_module("andrey.core._checkpoint")
        except ImportError:
            ap.error(f"the Andrey imported from {Path(andrey.__file__).parent} has no checkpoints")
    unknown = sorted(set(args.methods) - set(FITS))
    if unknown:
        ap.error(f"unknown methods {unknown}; known: {list(FITS)}")
    code = commit()
    records: dict[tuple, list[Record]] | None = {} if args.checkpoints else None
    rows, findings = run(
        args.methods or list(FITS), args.seeds, args.shuffle, args.workers, records
    )
    if args.out:
        name = sidecar_name(args.out) if args.checkpoints else None
        write(args.out, header(code, args.seeds, args.shuffle, args.workers, name), rows)
        if records is not None and name is not None:
            write_checkpoints(args.out.with_name(name), records)
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
