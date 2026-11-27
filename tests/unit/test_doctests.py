"""Run the documented examples of the public modules as doctests.

A targeted runner, not a global ``--doctest-modules``: it executes the ``andrey.api`` facade
examples (the runnable ``Examples`` the docstring standard guarantees) and those of the other public
modules the API reference shows, so collection stays bounded to the public surface and does not
pull in every internal module's docstring.
"""

from __future__ import annotations

import doctest
import importlib
import pkgutil

import pytest

import andrey
import andrey.api

_FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


def _facade_modules() -> list[str]:
    """Public facade modules under ``andrey.api`` (the private ``_`` adapters carry no examples)."""
    return [
        f"andrey.api.{info.name}"
        for info in pkgutil.iter_modules(andrey.api.__path__)
        if not info.name.startswith("_")
    ]


# The other public modules whose examples the API reference shows.
DOCUMENTED = [
    "andrey.core.environments",
    "andrey.core.knowledge",
    "andrey.core.structure",
    "andrey.data",
    "andrey.data.discrete",
    "andrey.data.interventions",
    "andrey.data.measurement",
    "andrey.data.missingness",
    "andrey.data.mixed",
    "andrey.data.real",
    "andrey.data.scm",
    "andrey.data.sortability",
    "andrey.markov",
    "andrey.metrics.graph",
    "andrey.metrics.latent",
    "andrey.metrics.order",
    "andrey.metrics.pairwise",
    "andrey.metrics.resolve",
    "andrey.metrics.temporal",
    "andrey.metrics.weights",
    "andrey.resample",
    "andrey.viz",
    "andrey.viz.mpl",
]


@pytest.mark.parametrize("modname", _facade_modules() + DOCUMENTED)
def test_documented_examples_run(modname):
    mod = importlib.import_module(modname)
    results = doctest.testmod(mod, optionflags=_FLAGS, verbose=False)
    assert results.attempted > 0, f"{modname}: no doctests found -- the module lost its Examples"
    assert results.failed == 0, (
        f"{modname}: {results.failed} of {results.attempted} doctests failed"
    )


def test_every_public_module_with_examples_is_covered():
    covered = set(_facade_modules()) | set(DOCUMENTED)
    missing = []
    finder = doctest.DocTestFinder()
    for info in pkgutil.walk_packages(andrey.__path__, "andrey.", onerror=lambda _: None):
        if any(part.startswith("_") for part in info.name.split(".")):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            # An optional dependency can make a public module unavailable in this environment.
            continue
        if (
            any(test.name == info.name and test.examples for test in finder.find(module))
            and info.name not in covered
        ):
            missing.append(info.name)
    assert not missing, (
        "public modules have module docstring examples but are in neither _facade_modules() nor "
        f"DOCUMENTED: {sorted(missing)}"
    )
