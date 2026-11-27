"""The separating sets and p-values held as a mask (:class:`MaskedMapping`).

The size-0 pass of the adjacency search and MMPC's forward phase hold the pairs and candidates
the empty set separates as one boolean mask, and the p-values of those pairs as the marginal
p-value matrix. The mapping must read as the dict the search would fill entry by entry: the same
entries, in the same order, under every lookup and change the searches and their callers make.
The memory checks bound what the searches keep by a few bytes per pair, where a dict entry per
pair takes about a hundred.
"""

from __future__ import annotations

import importlib
import pickle
import tracemalloc

import numpy as np
import pytest

from andrey.constraint._skeleton import MaskedMapping
from andrey.constraint.blanket import local_search
from andrey.constraint.pc import adjacency_search
from andrey.core.ci import make_indep_test


def _pair_case(seed: int = 0, d: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """A strictly upper-triangular mask with about half its bits set, and a p-value matrix."""
    rng = np.random.default_rng(seed)
    mask = np.triu(rng.random((d, d)) < 0.5, 1)
    return mask, rng.random((d, d))


def _as_dict(mask: np.ndarray, values: np.ndarray | None = None) -> dict:
    """The dict the search fills: each marked key in row-major order, with its value."""
    found = np.nonzero(mask)
    keys = found[0].tolist() if mask.ndim == 1 else list(zip(*(a.tolist() for a in found)))
    if values is None:
        return dict.fromkeys(keys, ())
    return dict(zip(keys, values[found].tolist(), strict=True))


def _same(mapping: MaskedMapping, want: dict) -> None:
    assert list(mapping.items()) == list(want.items())
    assert list(mapping.copy().items()) == list(want.items())
    assert len(mapping) == len(want)
    assert mapping == want and want == mapping


@pytest.mark.parametrize("with_values", [False, True])
def test_pair_mapping_reads_and_changes_as_the_dict(with_values: bool) -> None:
    mask, pmat = _pair_case()
    values = pmat if with_values else None
    mapping, want = MaskedMapping(mask, values), _as_dict(mask, values)
    _same(mapping, want)
    marked = next(iter(want))
    x, y = marked
    unmarked = next((a, b) for a in range(9) for b in range(a + 1, 9) if not mask[a, b])
    # Lookups under both key orders, NumPy integers, and keys no bit marks.
    assert mapping[marked] == want[marked] and mapping.get((y, x)) is None
    assert (np.int64(x), np.int64(y)) in mapping and (y, x) not in mapping
    for absent in [unmarked, (-1, 3), (3, 9), (x, x), (x,), (x, y, 0), x, "ab", None, 2.5]:
        assert absent not in mapping and mapping.get(absent, "none") == "none"
        with pytest.raises(KeyError):
            mapping[absent]
    with pytest.raises(TypeError):
        [x, y] in mapping  # noqa: B015  # an unhashable key, as for a dict
    # Changes: over a marked key, both key orders, update, setdefault, pop, and del.
    for target in (mapping, want):
        target[marked] = (4,)
        target[(y, x)] = (5,)
        target.update({unmarked: (6,), (0, 0): (7,)})
        target.setdefault(marked, (8,))
        target.setdefault((1, 0), (9,))
    _same(mapping, want)
    second = list(want)[1]
    assert mapping.pop(second) == want.pop(second)
    assert mapping.pop(second, "gone") == want.pop(second, "gone") == "gone"
    del mapping[marked], want[marked]
    del mapping[unmarked], want[unmarked]
    _same(mapping, want)
    mapping[marked], want[marked] = (1,), (1,)
    _same(mapping, want)
    with pytest.raises(KeyError):
        del mapping[unmarked]
    assert pickle.loads(pickle.dumps(mapping)) == want


def test_pvalues_are_the_matrix_floats() -> None:
    mask, pmat = _pair_case(1)
    xs, ys = np.nonzero(mask)
    mapping = MaskedMapping(mask, pmat)
    got = [mapping[x, y] for x, y in zip(xs.tolist(), ys.tolist())]
    assert all(type(p) is float for p in got)
    assert np.array_equal(np.array(got), pmat[xs, ys])


def test_a_shared_read_only_mask_is_copied_before_a_write() -> None:
    mask, pmat = _pair_case(2)
    mask.flags.writeable = False
    sepsets, pvalues = MaskedMapping(mask), MaskedMapping(mask, pmat)
    first = next(iter(sepsets))
    del sepsets[first]
    assert first not in sepsets and first in pvalues and mask[first]


def test_int_mapping_marks_an_empty_mapping() -> None:
    mapping = MaskedMapping(np.zeros(8, dtype=bool))
    mapping.mark([5, 1, 6])
    want = {1: (), 5: (), 6: ()}
    _same(mapping, want)
    assert np.int64(5) in mapping and 8 not in mapping and -1 not in mapping
    mapping.setdefault(2, (0,))
    mapping.setdefault(5, (0,))
    want.setdefault(2, (0,))
    _same(mapping, want)
    with pytest.raises(ValueError, match="empty"):
        mapping.mark([3])


@pytest.mark.parametrize("indep_test", ["fisherz", "chisq"])
@pytest.mark.parametrize("record_pvalues", [True, False])
def test_search_returns_read_as_the_dicts_of_the_generic_scan(
    indep_test: str, record_pvalues: bool
) -> None:
    # The generic scan, which a test without the vectorized size-0 pass takes, fills plain dicts
    # pair by pair: the mask's returns hold the same entries in the same order.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((300, 14))
    x[:, 1:4] += x[:, [0]]
    if indep_test == "chisq":
        x = np.digitize(x, [-0.5, 0.5]).astype(float)
    test = make_indep_test(indep_test, x)
    fast = adjacency_search(test, 14, 0.05, record_pvalues=record_pvalues)
    with pytest.MonkeyPatch.context() as patch:
        pc_module = importlib.import_module("andrey.constraint.pc")
        patch.setattr(pc_module, "prescreen_capable", lambda test: False)
        generic = adjacency_search(test, 14, 0.05, record_pvalues=record_pvalues)
    assert isinstance(fast[1], MaskedMapping)
    assert np.array_equal(fast[0], generic[0])
    assert list(fast[1].items()) == list(generic[1].items())
    assert list(fast[2].items()) == list(generic[2].items())
    assert bool(fast[2]) == record_pvalues


def _kept_bytes(build) -> int:
    """The traced bytes ``build()``'s return keeps."""
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        kept = build()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    del kept
    return after - before


def test_adjacency_search_keeps_a_few_bytes_per_pair() -> None:
    # Independent columns: the size-0 pass removes nearly every pair. The returns hold the
    # skeleton, the mask, and the p-value matrix: about 10 d^2 bytes.
    d = 600
    x = np.random.default_rng(4).standard_normal((200, d))
    test = make_indep_test("fisherz", x)
    kept = _kept_bytes(lambda: adjacency_search(test, d, 0.05, max_depth=0, record_pvalues=True))
    assert kept < 16 * d * d, f"{kept / 2**20:.1f} MiB kept at d={d}"


def test_mmpc_runs_keep_a_few_bytes_per_candidate() -> None:
    # Independent columns: each target's forward phase separates nearly every candidate given the
    # empty set, and max_depth=0 tests no other set. Each run keeps a mask of d bytes and a fixed
    # overhead.
    d = 300
    x = np.random.default_rng(5).standard_normal((200, d))
    search = local_search(x, alpha=0.05, indep_test="fisherz", max_depth=0)
    search.run("mmpc", 0)  # the first run's lazy imports are not the runs' cost
    kept = _kept_bytes(lambda: [search.run("mmpc", t) for t in range(1, d)])
    assert kept < 8 * d * d, f"{kept / 2**20:.1f} MiB kept at d={d}"
