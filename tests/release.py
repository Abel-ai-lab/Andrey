"""The ``release`` mark on single cases of a parametrized test.

A ``release`` test is a heavy accuracy check: the release freeze runs it, and pull-request CI
deselects it with ``-m "not release"`` (``DEVELOPMENT.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest

_ParameterSet = type(pytest.param(None))


def release_cases(cases: Iterable[Any], heavy: set[str], id: Callable[[Any], str] = str) -> list:
    """``cases`` as test parameters, the ones whose id is in ``heavy`` marked ``release``.

    A case is a value, a tuple of values for a test with several arguments, or a ``pytest.param``,
    which keeps its id and marks; ``id`` names the others. A name in ``heavy`` that no case has
    raises, so a renamed case cannot leave its mark behind.
    """
    out = []
    for case in cases:
        if not isinstance(case, _ParameterSet):
            values = case if isinstance(case, tuple) else (case,)
            case = pytest.param(*values, id=id(case))
        if case.id in heavy:
            marks = [*case.marks, pytest.mark.release]
            case = pytest.param(*case.values, id=case.id, marks=marks)
        out.append(case)
    unknown = heavy - {case.id for case in out}
    if unknown:
        raise ValueError(f"no case has the id {sorted(unknown)}")
    return out
