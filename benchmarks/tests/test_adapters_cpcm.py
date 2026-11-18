"""The CPCM adapters' names, which a run_id holds in place of their params."""

from __future__ import annotations

import pytest

from andrey_bench.adapters.andrey_cpcm import AndreyCPCMGraph
from andrey_bench.adapters.bodik_cpcm import BodikCPCM


@pytest.mark.parametrize(
    ("make", "default"),
    [
        (lambda value: AndreyCPCMGraph(edge_penalty=value), 2.0),
        (lambda value: BodikCPCM(lambda_=value), 2.0),
    ],
)
def test_settings_that_differ_in_any_digit_get_different_names(make, default):
    values = [1.0000001, 1.0000002, 1.0, 1e-20, default + 1e-12]
    names = [make(value).name for value in values]
    assert len(set(names)) == len(values)
    assert make(default).name not in names
    assert make(1).name == make(1.0).name
