"""The SkewD lane: its pair regimes, the adapters, and the authors' code at the reference boundary.

Facts about the regimes and Andrey's adapters are checked in bench-env. The authors' adapter is
checked there against a stand-in for its code, and by running the code itself in ``.venv-skewd``
at tiny budgets, skipped when that environment or the checkout has not been built.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import skewd_pairs
from scipy.stats import skewnorm

from andrey import GraphStructure
from andrey_bench import datasets, runner
from andrey_bench.adapters import klippert_skewd
from andrey_bench.adapters.andrey_loci import AndreyLOCI, AndreyLOCILikelihood
from andrey_bench.adapters.andrey_skewd import AndreySkewD, AndreySkewDLikelihood
from andrey_bench.adapters.klippert_skewd import (
    KlippertSkewD,
    KlippertSkewDLikelihood,
    edge_marks,
)

SKEWD_PYTHON = Path(runner.BENCH_ROOT) / ".venv-skewd" / "bin" / "python"
SKEWD_CODE = Path(os.environ.get(klippert_skewd.CODE_ENV) or Path(runner.BENCH_ROOT) / ".skewd")
needs_skewd_env = pytest.mark.skipif(
    not (SKEWD_PYTHON.exists() and (SKEWD_CODE / "src" / "skewd.py").is_file()),
    reason="skewd env not built - run benchmarks/start_skewd.sh",
)

ANDREY = [AndreySkewD(), AndreySkewDLikelihood()]
AUTHORS = [KlippertSkewD(), KlippertSkewDLikelihood()]
#: Budgets that end each search of Andrey's fit after one candidate and five iterations.
TINY_ANDREY = {"tuning_iter": 1, "max_iter": 5}
#: The authors' arguments at budgets of a few seconds.
TINY_AUTHORS = {
    "cma_iter": 20,
    "n_cv_splits_boptim": 2,
    "ecm_max_iter": 5,
    "ecm_max_iter_rho": 5,
    "n_calls_boptim": 4,
    "n_inits_boptim": 3,
}


def _skew_pair(n: int = 300, seed: int = 0) -> np.ndarray:
    """A pair ``x -> y`` with skew-normal noise of shape 5 whose scale changes with the cause."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = skewnorm.rvs(5, size=n, random_state=rng)
    return np.column_stack([x, np.tanh(2 * x) + x + (0.4 + 0.2 * np.tanh(x)) * noise])


def _edges(marks: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i, j, _ in GraphStructure.from_numpy(marks, kind="dag").oriented_edges()]


# --- regimes ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", skewd_pairs.REGIMES)
def test_every_regime_draws_two_columns_and_one_edge_either_way(regime: str) -> None:
    directions = set()
    for key in skewd_pairs.regime_keys(regime, n=200, seeds=8):
        data, marks, kind, columns = datasets.generate_one(key)
        assert data.shape == (200, 2) and kind == "dag" and sorted(columns) == [0, 1]
        [edge] = _edges(marks)
        directions.add(edge)
    assert directions == {(0, 1), (1, 0)}  # the seed draws the direction


def test_the_regimes_are_gaussian_laplace_and_gumbel_noise_z_scored_and_rescaled() -> None:
    cells = set()
    for regime in skewd_pairs.REGIMES:
        (key,) = skewd_pairs.regime_keys(regime, n=500, seeds=1)
        call = datasets.generation_kwargs(key)
        assert (call["functional"], call["d"], call["density"]) == ("location_scale", 2, 1.0)
        cells.add((call["noise"], call["scale"]))
    assert cells == {
        (noise, scale)
        for noise in ("gaussian", "gumbel", "laplace")
        for scale in ("standardize", "rescale")
    }
    assert skewd_pairs.SIZES == [500, 1000]


def test_the_driver_prepares_every_regime_without_fitting(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(datasets.DATA_STORE_ENV, raising=False)
    argv = ["--out", str(tmp_path), "--prepare-only", "--sizes", "60", "--seeds", "2"]
    assert skewd_pairs.main(argv) == 0
    assert len(list((tmp_path / "data").glob("ds_*.npz"))) == 2 * len(skewd_pairs.REGIMES)


def test_only_the_gumbel_regimes_have_skewed_noise() -> None:
    skewed = {r for r in skewd_pairs.REGIMES if skewd_pairs.noise_skewed(r)}
    assert skewed == {"lsnm_gumbel_pair", "lsnm_gumbel_rescaled_pair"}


# --- adapters --------------------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", ANDREY + AUTHORS, ids=lambda a: a.name)
def test_an_adapter_survives_the_record_and_the_worker(adapter) -> None:
    assert json.loads(json.dumps(adapter.params(), sort_keys=True)) == adapter.params()
    assert pickle.loads(pickle.dumps(adapter)).name == adapter.name
    assert (adapter.algorithm, adapter.output_type) == ("skewd", "dag")


def test_every_solution_is_in_the_driver_and_the_authors_run_in_their_own_environment() -> None:
    assert set(skewd_pairs.SOLUTIONS) == {a.name for a in ANDREY + AUTHORS} | {
        a.name for a in (AndreyLOCI(), AndreyLOCILikelihood())
    }
    assert {a.python_env for a in AUTHORS} == {"ANDREY_BENCH_SKEWD_PYTHON"}
    assert not any(hasattr(a, "python_env") for a in ANDREY)


def test_each_andrey_arm_has_the_authors_arm_of_its_criterion() -> None:
    pairs = {a.name: b.name for a, b in zip(ANDREY, AUTHORS)}
    assert skewd_pairs.COMPARATOR == pairs
    for mine, theirs in zip(ANDREY, AUTHORS):
        assert mine.params()["criterion"] == theirs.params()["criterion"]
        assert set(skewd_pairs.COMPARATOR) <= set(skewd_pairs.SOLUTIONS)


def test_the_authors_arms_record_the_campaigns_arguments_and_the_commit() -> None:
    for adapter in AUTHORS:
        params = adapter.params()
        assert {k: params[k] for k in klippert_skewd.CAMPAIGN} == {
            "cma_iter": 5000,
            "n_cv_splits_boptim": 8,
            "ecm_max_iter": 5000,
            "ecm_max_iter_rho": 100,
            "n_calls_boptim": 60,
            "n_inits_boptim": 40,
        }
        assert (params["commit"], params["standardize"]) == (klippert_skewd.COMMIT, True)


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
@pytest.mark.parametrize("adapter", ANDREY, ids=lambda a: a.name)
def test_andreys_arm_finds_the_planted_direction_at_tiny_budgets(adapter) -> None:
    """The budgets reach `fit` through its `params`; the setup's own are fixed."""
    adapter.setup()
    pair = _skew_pair()
    for data, edge in ((pair, (0, 1)), (pair[:, ::-1], (1, 0))):
        native = np.asarray(adapter.fit(data, TINY_ANDREY))
        structure = adapter.to_structure(native)
        assert (structure.kind, structure.n_nodes) == ("dag", 2)
        assert structure.oriented_edges() == [(*edge, "directed")]


@pytest.mark.parametrize(
    ("score", "direction", "edges"),
    [(0.3, 1, [(0, 1)]), (-1e-9, 0, [(1, 0)]), (0.0, 0, []), (float("nan"), 0, [])],
)
def test_the_authors_direction_decides_the_edge_and_no_score_gives_none(
    score, direction, edges
) -> None:
    assert _edges(edge_marks(score, direction)) == edges


def _stand_in(root: Path) -> Path:
    """A ``src`` package whose ``skewd`` records its call and returns the code's eight values."""
    package = root / "src"
    package.mkdir()
    for name in klippert_skewd.REQUIRED_FILES:
        (root / name).write_text("")
    (package / "skewd.py").write_text(
        "CALLS = []\n"
        "def skewd(x, y, **arguments):\n"
        "    CALLS.append((x, y, arguments))\n"
        "    return 0.3, 1, -0.2, 0, {}, {}, {}, {}\n"
    )
    return root


@pytest.mark.parametrize("adapter", AUTHORS, ids=lambda a: a.name)
def test_the_authors_arm_standardizes_passes_the_campaign_and_reads_its_criterion(
    adapter, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv(klippert_skewd.CODE_ENV, str(_stand_in(tmp_path)))
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    rng = np.random.default_rng(1)
    data = np.column_stack([5 + 3 * rng.standard_normal(50), -2 + 0.1 * rng.standard_normal(50)])
    adapter.setup()
    marks = adapter.fit(data, adapter.params())
    from src.skewd import CALLS  # ty: ignore[unresolved-import]  # the stand-in above

    [warm_up, (x, y, arguments)] = CALLS
    assert warm_up[2] == klippert_skewd.WARMUP
    for column in (x, y):
        assert column.shape == (50, 1)
        assert abs(column.mean()) < 1e-12 and column.std() == pytest.approx(1)
    assert arguments == klippert_skewd.CAMPAIGN
    # The independence score (0.3, direction 1) and the likelihood score (-0.2, direction 0).
    assert _edges(marks) == ([(0, 1)] if adapter.criterion == "independence" else [(1, 0)])


def test_the_authors_arm_takes_its_arguments_from_the_params(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(klippert_skewd.CODE_ENV, str(_stand_in(tmp_path)))
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    adapter = KlippertSkewD()
    adapter.fit(_skew_pair(60), {**adapter.params(), **TINY_AUTHORS})
    from src.skewd import CALLS  # ty: ignore[unresolved-import]  # the stand-in above

    assert CALLS[-1][2] == TINY_AUTHORS


def test_the_authors_arm_names_a_missing_code_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(klippert_skewd.CODE_ENV, raising=False)
    with pytest.raises(SystemExit, match=klippert_skewd.CODE_ENV):
        klippert_skewd.code_dir()
    monkeypatch.setenv(klippert_skewd.CODE_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match=r"ANDREY_BENCH_SKEWD.*src/skewd\.py"):
        klippert_skewd.code_dir()


@needs_skewd_env
@pytest.mark.parametrize("adapter", AUTHORS, ids=lambda a: a.name)
def test_the_authors_code_runs_in_its_environment_at_tiny_budgets(adapter, monkeypatch) -> None:
    """The fit returns a two-node DAG; budgets this small support no claim about its direction."""
    monkeypatch.setenv(adapter.python_env, str(SKEWD_PYTHON))
    monkeypatch.setenv(klippert_skewd.CODE_ENV, str(SKEWD_CODE))
    params = {**adapter.params(), **TINY_AUTHORS}
    (measures,) = runner.run_task(
        adapter, _skew_pair(200), params, cap_wall_s=600.0, repeats=1, warmup=0
    )
    assert measures["status"] == "ok", measures.get("stderr")
    structure = adapter.to_structure(measures["adj"])
    assert (structure.kind, structure.n_nodes) == ("dag", 2)
