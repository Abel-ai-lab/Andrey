"""CALM, ANM, LOCI, PNL, and the test phase of GFCI, BOSS-FCI, and GRaSP-FCI on one thread.

Each runs on one BLAS thread, and CALM on one torch thread, so a result is the one-thread result on
any host. A child process started with several BLAS and torch threads fits each method and returns
its output bits and the threads it saw; the one-thread child's output is the reference.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

import andrey
from andrey.core import ci

#: BLAS, OpenMP, and torch threads of the child that must give the one-thread result.
MANY_THREADS = 4

_CHILD = r"""
import pickle, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
try:
    import torch
except ImportError:
    torch = None
from threadpoolctl import threadpool_info
import andrey
import andrey.data
from andrey.core import ci_kernel
from andrey.irregular import anm as anm_module, loci as loci_module, pnl as pnl_module
from andrey.search import calm as calm_module


def counts():
    pools = threadpool_info()
    return (
        max((p["num_threads"] for p in pools if p["user_api"] == "blas"), default=1),
        max((p["num_threads"] for p in pools if p["user_api"] == "openmp"), default=1),
        1 if torch is None else torch.get_num_threads(),
    )


inside, trace = [], []


def record(module, name):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        inside.append((name, counts()))
        return original(*args, **kwargs)

    setattr(module, name, wrapper)


record(anm_module, "anm")
record(loci_module, "fit_direction")
record(pnl_module, "_run_tasks")
record(pnl_module, "_device_tests")
record(calm_module, "_optimize_gated_dag")
statistic_pvalue = ci_kernel.KCI._statistic_pvalue


def traced(self, x, y, condition):
    out = statistic_pvalue(self, x, y, condition)
    trace.append((x, y, tuple(condition), counts()[0], out[1]))
    return out


ci_kernel.KCI._statistic_pvalue = traced


def digest(out):
    weights = out.weighted_adjacency
    return (
        repr(out.metadata),
        out.structure.to_numpy().tobytes(),
        None if weights is None else np.asarray(weights).tobytes(),
    )


rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 400)
cubic = np.column_stack([x, x**3 + 0.3 * rng.uniform(-1, 1, 400)])
z = rng.standard_normal(1000)
scale = np.column_stack(
    [z, np.sin(2 * z) + z + (0.6 + 0.5 * np.tanh(z)) * rng.standard_normal(1000)]
)
u = rng.uniform(-1, 1, 1000)
post = np.column_stack([u, np.tanh(u**3 + 0.3 * rng.uniform(-1, 1, 1000))])
chain = rng.standard_normal((300, 6))
chain[:, 1] += chain[:, 0]
chain[:, 2] += 0.8 * chain[:, 1]
holes = chain.copy()
holes[rng.random(holes.shape) < 0.1] = np.nan
kci_data = np.asarray(
    andrey.data.sample_scm(d=6, n=200, seed=0, functional="additive_noise", density=2.0).data,
    float,
)

cases = {
    "anm": lambda: andrey.anm(cubic),
    "loci": lambda: andrey.loci(scale),
    "loci_likelihood": lambda: andrey.loci(scale, criterion="likelihood"),
    "pnl": lambda: andrey.pnl(post),
    "calm": lambda: andrey.calm(chain, subproblem_iter=60, max_iter=4),
    "calm_no_moral_graph": lambda: andrey.calm(
        chain, subproblem_iter=60, max_iter=4, use_moral_graph=False
    ),
    "calm_em": lambda: andrey.calm(holes, subproblem_iter=40, max_iter=3, missing="em"),
}
for name in ("gfci", "boss_fci", "grasp_fci"):
    cases[name] = lambda name=name: getattr(andrey, name)(kci_data, indep_test="kci")
if torch is None:  # CALM needs torch
    cases = {name: fit for name, fit in cases.items() if not name.startswith("calm")}

andrey.ges(chain)  # a score search first, the order in which the BLAS limit alone missed torch
outputs, seen = {}, {}
for name, fit in cases.items():
    if sys.argv[1:] and name not in sys.argv[1:]:
        continue
    inside.clear()
    trace.clear()
    outputs[name] = digest(fit())
    seen[name] = (list(inside), list(trace))
if not sys.argv[1:]:
    with andrey.config(n_jobs=2):
        outputs["pnl_two_workers"] = digest(andrey.pnl(post))
pickle.dump((counts(), outputs, seen), sys.stdout.buffer)
"""


def _child(env_threads: dict[str, int], *cases: str):
    """The threads the child started with, its output bits per case, and what it saw per case.

    ``env_threads`` sets the child's thread variables; the child fits ``cases``, or every case."""
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    }
    env.update({name: str(count) for name, count in env_threads.items()})
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, *cases],
        capture_output=True,
        env=env,
        timeout=1200,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    return pickle.loads(proc.stdout)


def _threads(count: int) -> dict[str, int]:
    return dict.fromkeys(("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), count)


@pytest.fixture(scope="module")
def one_thread():
    return _child(_threads(1))


@pytest.fixture(scope="module")
def many_threads():
    started, outputs, seen = _child(_threads(MANY_THREADS))
    if started[0] <= 1 and started[2] <= 1:
        pytest.skip("this build runs no BLAS or torch pool on more than one thread")
    return started, outputs, seen


def test_a_child_with_many_threads_gives_the_one_thread_bits(one_thread, many_threads):
    _, reference, _ = one_thread
    _, outputs, _ = many_threads
    assert outputs.keys() == reference.keys()
    assert [name for name in outputs if outputs[name] != reference[name]] == []
    assert outputs["pnl_two_workers"] == outputs["pnl"]


FITS = ["anm", "loci", "loci_likelihood", "pnl", "calm", "calm_no_moral_graph", "calm_em"]
GFCI_FAMILY = ["gfci", "boss_fci", "grasp_fci"]


@pytest.mark.parametrize("name", FITS)
def test_a_fit_sees_one_blas_thread_and_calm_one_torch_thread(name, many_threads):
    if name.startswith("calm"):
        pytest.importorskip("torch")
    inside = many_threads[2][name][0]
    assert inside
    # Where the library has no pool to count, the count is 1. Only CALM computes in torch.
    for _, (blas, openmp, torch_threads) in inside:
        assert (blas, openmp) == (1, 1)
        if name.startswith("calm"):
            assert torch_threads == 1


def test_calm_sees_one_torch_thread_when_only_mkl_has_many():
    """The BLAS limit alone leaves torch's threads where ``MKL_NUM_THREADS`` alone puts them."""
    pytest.importorskip("torch")
    _, _, seen = _child({"MKL_NUM_THREADS": MANY_THREADS}, "calm")
    assert {threads for _, threads in seen["calm"][0]} == {(1, 1, 1)}


@pytest.mark.parametrize("name", GFCI_FAMILY)
def test_a_kci_query_of_the_gfci_family_has_the_one_thread_p_value(name, one_thread, many_threads):
    queries, expected = many_threads[2][name][1], one_thread[2][name][1]
    assert len(queries) > 20
    assert [(*query[:3], query[4]) for query in queries] == [
        (*query[:3], query[4]) for query in expected
    ]
    assert {query[3] for query in queries} == {1}


def _latent_data(seed: int) -> np.ndarray:
    """Twelve variables of a random linear graph, three of them hidden: discriminating paths,
    unshielded triples, and non-adjacent pairs that need separating sets."""
    rng = np.random.default_rng(seed)
    d, n = 12, 500
    weights = np.triu(rng.uniform(0.5, 1.0, (d, d)) * (rng.random((d, d)) < 0.3), 1)
    data = np.zeros((n, d))
    for j in range(d):
        data[:, j] = data @ weights[:, j] + rng.standard_normal(n)
    return np.delete(data, rng.choice(d, 3, replace=False), axis=1)


def _blas_threads() -> int:
    """The most threads a BLAS library runs; 0 where threadpoolctl lists none, as on macOS."""
    return max(
        (info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"), default=0
    )


@pytest.mark.parametrize("name", ["gfci", "boss_fci", "grasp_fci"])
@pytest.mark.parametrize("pin_blas", [None, True, False])
def test_the_gfci_family_pins_every_phase_as_the_test_sets(name, pin_blas, monkeypatch):
    """Separating sets, collider votes, and R4 query a test on one BLAS thread when it sets
    ``pin_blas``, as the adjacency search does; a test that sets it false, or sets nothing, keeps
    every thread."""
    seen = []
    phases = set()

    class Recording:
        def __init__(self, data):
            self._test = ci.FisherZ(data)

        def __call__(self, x, y, condition_set=()):
            frame = sys._getframe(1)
            while frame is not None:
                phases.add(frame.f_code.co_name)
                frame = frame.f_back
            seen.append(_blas_threads())
            return self._test(x, y, condition_set)

    if pin_blas is not None:
        Recording.pin_blas = pin_blas
    monkeypatch.setitem(ci._CI_REGISTRY, "recording", Recording)
    with threadpool_limits(limits=2, user_api="blas"):
        if _blas_threads() < 2:
            pytest.skip("threadpoolctl finds no BLAS library that runs more than one thread")
        getattr(andrey, name)(
            _latent_data(2), indep_test="recording", collider_rule="majority", seed=0
        )
    assert {"_derive_sepsets", "vote_triples", "_rule_r4"} <= phases
    assert seen and set(seen) == {1 if pin_blas else 2}
