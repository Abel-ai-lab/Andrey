"""Every cache key holds every setting that changes the result.

A cache that answers a call with the entry of an earlier call that differed in a setting its key
leaves out returns a stale result. The module-level caches are listed in ``MODULE_CACHES``, so a
new one fails here until its key is checked; the tests below check each cache whose result can
change while its key does not: the library pools a BLAS limit reaches, a conditioning threshold,
the environment columns of a test, and the rows of a score.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import pkgutil
import sys
import types
from collections import OrderedDict

import numpy as np
import pytest

import andrey
from andrey.constraint import _skeleton
from andrey.core import backend, ci_missing, mixed, score, score_delta
from andrey.core.ci import make_indep_test
from andrey.search import operators


def module_caches() -> set[str]:
    """The ``lru_cache`` and ``cache`` functions defined at the top of an andrey module."""
    found = set()
    for info in pkgutil.walk_packages(andrey.__path__, "andrey."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # a module of an extra that is not installed
            continue
        for name, obj in vars(module).items():
            if (
                hasattr(obj, "cache_clear")
                and hasattr(obj, "cache_info")
                and getattr(obj, "__module__", None) == module.__name__
            ):
                found.add(f"{module.__name__}.{name}")
    return found


#: Each module-level cache and what its key is. A function of its arguments only holds them all
#: (``test_a_cache_misses_when_any_argument_changes``); the others read the machine, never a setting
#: of a call: what the process has installed or imported.
MODULE_CACHES = {
    "andrey.api._method.spec_of": "name",
    "andrey.api._boundary.measured": "the generated table, which is source code",
    "andrey.constraint._skeleton._combination_table": "m, r",
    "andrey.core.backend._thread_pools": "the number of imported modules",
    "andrey.core.backend.torch": "whether torch is installed",
    "andrey.core.ci_missing._apart": "k",
    "andrey.core.mixed._triangle": "d",
    "andrey.core.score.ebic_penalty": "n_parents, n_variables, ebic_gamma",
    "andrey.search._parallel._forkserver_starts": "the temporary directory, fixed for the process",
    "andrey.search.operators._position_plan": "m",
}


def test_every_module_level_cache_is_listed():
    assert module_caches() == set(MODULE_CACHES)


#: A function of its arguments only: a call, then each argument changed alone.
ARGUMENT_CACHES = [
    pytest.param(score.ebic_penalty, (2, 10, 0.5), {0: 3, 1: 12, 2: 1.0}, id="ebic_penalty"),
    pytest.param(_skeleton._combination_table, (6, 2), {0: 7, 1: 3}, id="combination_table"),
    pytest.param(ci_missing._apart, (4,), {0: 5}, id="apart"),
    pytest.param(mixed._triangle, (3,), {0: 4}, id="triangle"),
    pytest.param(operators._position_plan, (3,), {0: 4}, id="position_plan"),
]


def same(a: object, b: object) -> bool:
    return bool(np.all(np.asarray(a, dtype=object) == np.asarray(b, dtype=object)))


@pytest.mark.parametrize(("cached", "base", "changes"), ARGUMENT_CACHES)
def test_a_cache_misses_when_any_argument_changes(cached, base, changes):
    cached.cache_clear()
    cached(*base)
    for position, value in changes.items():
        args = (*base[:position], value, *base[position + 1 :])
        misses = cached.cache_info().misses
        found = cached(*args)
        name = list(inspect.signature(cached.__wrapped__).parameters)[position]
        assert cached.cache_info().misses == misses + 1, f"changing {name} hit the cache"
        assert same(found, cached.__wrapped__(*args))
    cached.cache_clear()


@pytest.fixture
def pools():
    """The BLAS pool finder cleared, before and after, so a fake one never outlives its test."""
    backend._thread_pools.cache_clear()
    yield
    backend._thread_pools.cache_clear()


def test_the_blas_limit_reaches_a_library_imported_after_the_first_fit(monkeypatch, pools):
    # A fit limits the BLAS and OpenMP pools it finds to one thread, so its result does not depend
    # on the machine's thread count. A library imported later brings pools the first search never
    # saw, so they are found again once an import has happened.
    import threadpoolctl

    found = []

    class Controller:
        def __init__(self):
            found.append(len(sys.modules))

        def limit(self, limits):
            return contextlib.nullcontext()

    monkeypatch.setattr(threadpoolctl, "ThreadpoolController", Controller)
    for _ in range(2):
        with backend.one_blas_thread():
            pass
    assert len(found) == 1  # nothing was imported in between: the pools found stay
    monkeypatch.setitem(sys.modules, "a_library_imported_later", types.ModuleType("later"))
    with backend.one_blas_thread():
        pass
    assert len(found) == 2


def test_a_conditioning_threshold_is_applied_to_an_inverse_cached_under_another():
    # The inverse of a set's covariance is cached with its conditioning proxy, so a later call
    # with another threshold applies its own gate.
    cov = np.array([[1.0, 0.99999], [0.99999, 1.0]])  # a proxy near 4e5
    cache: OrderedDict = OrderedDict()
    get = score_delta._get_inv_with_cond
    loose, ill = get(cache, (0, 1), cov, [0, 1], 1e12)
    assert not ill and loose is not None
    assert get(cache, (0, 1), cov, [0, 1], 1e3) == (None, True)
    again, ill = get(cache, (0, 1), cov, [0, 1], 1e12)
    assert not ill and np.array_equal(again, loose)
    cache.clear()
    assert get(cache, (0, 1), cov, [0, 1], 1e3) == (None, True)  # the strict gate first
    assert not get(cache, (0, 1), cov, [0, 1], 1e12)[1]


@pytest.mark.parametrize("name", ["gsq", "chisq"])
def test_marking_environment_columns_drops_the_block_tests_a_query_cached(name):
    # Rows that observe all three columns have one value of column 2, so once it is an environment
    # indicator its queries are not tested. The answer cached before the mark must not outlive it.
    rng = np.random.default_rng(0)
    x = rng.integers(0, 3, 300).astype(float)
    y = (x + rng.integers(0, 2, 300)) % 3
    indicator = np.r_[np.zeros(150), np.ones(150)]
    data = np.column_stack([x, y, indicator])
    data[:150, 1] = np.nan
    test = make_indep_test(name, data)
    assert test(0, 1, [2]) < 1e-6  # tested, and cached
    test.mark_environments([2])
    assert np.isnan(test(0, 1, [2]))


def _codes(rows: int = 400) -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, rows)
    y = (x + rng.integers(0, 2, rows)) % 3
    return np.column_stack([x, y]).astype(float)


@pytest.mark.parametrize("name", ["bdeu", "discrete_bic"])
def test_a_score_on_some_rows_keeps_its_own_scores_apart_from_the_whole(name):
    from andrey.core.score import make_score

    data = _codes()
    whole = make_score(name, data)
    before = whole.score(1, [0])  # cached in the whole score
    rows = np.arange(len(data)) < 100
    part = whole.on_rows(rows)
    assert part.score(1, [0]) != before  # not the whole's cached answer
    if name == "bdeu":  # its terms read the rows only; the BIC keeps the whole's log(n)
        assert part.score(1, [0]) == make_score(name, data[rows]).score(1, [0])
    assert whole.score(1, [0]) == before  # and the whole's cache holds no answer of the part's
