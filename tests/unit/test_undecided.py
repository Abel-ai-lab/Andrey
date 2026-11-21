"""Undecided pairs and possibly separable edges on merged cohorts.

A pair that no test with rows could decide, as a pair never observed together, is left out of the
graph and listed in ``metadata["undecided_pairs"]``; no orientation reads it. A drawn edge that a
conditioning set not tested might have separated is listed in ``metadata["possibly_separable"]``.
The hand cases are three variables ``X``, ``Y`` and a third, with weights 0.8, root noise 1 and
child noise 0.6, merged from cohorts that each observe some of them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.core import ARROW

# case: (edges as (parent, child) over 0 = X, 1 = Y, 2 = third, cohort patterns, cohort shares)
HAND = {
    "fork": ([(2, 0), (2, 1)], [(1, 0, 1), (0, 1, 1)], (0.5, 0.5)),
    "collider": ([(0, 2), (1, 2)], [(1, 0, 1), (0, 1, 1)], (0.5, 0.5)),
    "chain_shared": ([(0, 2), (2, 1)], [(1, 0, 1), (0, 1, 1), (1, 1, 1)], (0.45, 0.45, 0.10)),
    "pairwise_only": ([(2, 0), (2, 1)], [(1, 1, 0), (1, 0, 1), (0, 1, 1)], (1 / 3, 1 / 3, 1 / 3)),
    "chain_apart": ([(2, 0), (0, 1)], [(1, 0, 1), (0, 1, 1)], (0.5, 0.5)),
}


def _hand(case: str, n: int = 1000, extra: int = 0, seed: int = 0) -> pd.DataFrame:
    """A hand case over ``n`` rows merged from its cohorts, plus ``extra`` rows that observe every
    column, in shuffled order."""
    edges, patterns, shares = HAND[case]
    rng = np.random.default_rng(seed)
    total = n + extra
    x = np.zeros((total, 3))
    done: list[int] = []
    while len(done) < 3:  # each column after its parents
        child = next(
            c for c in range(3) if c not in done and all(a in done for a, b in edges if b == c)
        )
        done.append(child)
        parents = [a for a, b in edges if b == child]
        noise = 0.6 if parents else 1.0
        x[:, child] = 0.8 * x[:, parents].sum(axis=1) + noise * rng.standard_normal(total)
    counts = [round(share * n) for share in shares]
    counts[-1] = n - sum(counts[:-1])
    observed = np.vstack(
        [np.tile(np.array(p, dtype=bool), (k, 1)) for p, k in zip(patterns, counts)]
        + [np.ones((extra, 3), dtype=bool)]
    )
    order = rng.permutation(total)
    third = "W" if case == "pairwise_only" else "Z"
    return pd.DataFrame(np.where(observed, x, np.nan)[order], columns=["X", "Y", third])


def _skeleton(out) -> set[tuple[str, str]]:
    return {tuple(sorted((a, b))) for a, b, _ in out.structure.oriented_edges()}


def _pairs(out, key: str) -> set[tuple[str, str]]:
    return {tuple(pair) for pair in out.metadata.get(key, [])}


FITS = {"pc": andrey.pc, "fci": andrey.fci}


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("method", list(FITS))
@pytest.mark.parametrize(
    ("case", "extra", "drawn", "undecided", "flagged"),
    [
        # X and Y are never observed together: X - Y is undecided whatever the truth, and the
        # other column could have separated each drawn pair.
        ("fork", 0, {("X", "Z"), ("Y", "Z")}, {("X", "Y")}, {("X", "Z"), ("Y", "Z")}),
        ("collider", 0, {("X", "Z"), ("Y", "Z")}, {("X", "Y")}, {("X", "Z"), ("Y", "Z")}),
        ("fork", 3, {("X", "Z"), ("Y", "Z")}, {("X", "Y")}, {("X", "Z"), ("Y", "Z")}),
        # Every set is tested on the rows that observe all three.
        ("chain_shared", 0, {("X", "Z"), ("Y", "Z")}, set(), set()),
        # Each pair is observed, never all three: the separating set W of X and Y is not tested.
        (
            "pairwise_only",
            0,
            {("W", "X"), ("W", "Y"), ("X", "Y")},
            set(),
            {("W", "X"), ("W", "Y"), ("X", "Y")},
        ),
        # Z -> X -> Y: the separating set X of Y and Z is never observed with Y.
        ("chain_apart", 0, {("X", "Z"), ("Y", "Z")}, {("X", "Y")}, {("X", "Z"), ("Y", "Z")}),
    ],
)
def test_a_hand_case_gives_its_decided_graph_undecided_pairs_and_flags(
    method, case, extra, drawn, undecided, flagged
):
    out = FITS[method](_hand(case, extra=extra), max_depth=3)
    assert _skeleton(out) == drawn
    assert _pairs(out, "undecided_pairs") == undecided
    assert _pairs(out, "possibly_separable") == flagged
    assert not _skeleton(out) & undecided


@pytest.mark.parametrize("method", list(FITS))
def test_four_rows_observing_every_column_decide_the_pair_on_those_rows_alone(method):
    """The bridging cost: the four rows test X and Y with no power, so they separate them and the
    fork reads as a collider, while each edge stays flagged."""
    out = FITS[method](_hand("fork", extra=4), max_depth=3)
    assert "undecided_pairs" not in out.metadata or not out.metadata["undecided_pairs"]
    assert _skeleton(out) == {("X", "Z"), ("Y", "Z")}
    assert _pairs(out, "possibly_separable") == {("X", "Z"), ("Y", "Z")}


def _undecided_end() -> pd.DataFrame:
    """``A -> X <- B`` and ``Z -> X``: one cohort observes ``A``, ``B``, ``X``, the other ``X`` and
    ``Z``, so ``A`` and ``B`` are never observed with ``Z``."""
    rng = np.random.default_rng(0)
    n = 2000
    a, b, z = rng.normal(size=(3, n))
    x = 0.8 * a + 0.8 * b + 0.8 * z + 0.6 * rng.normal(size=n)
    df = pd.DataFrame({"A": a, "B": b, "X": x, "Z": z})
    df.loc[: n // 2 - 1, "Z"] = np.nan
    df.loc[n // 2 :, ["A", "B"]] = np.nan
    return df


@pytest.mark.parametrize("method", list(FITS))
def test_no_rule_reads_an_undecided_pair_as_apart(method):
    out = FITS[method](_undecided_end())
    assert _pairs(out, "undecided_pairs") == {("A", "Z"), ("B", "Z")}
    assert _skeleton(out) == {("A", "X"), ("B", "X"), ("X", "Z")}
    # The decided collider A -> X <- B stands; A - X - Z is no unshielded triple to Meek's or FCI's
    # first rule, which would otherwise orient X -> Z.
    assert out.structure.endpoints("A", "X")[1] == ARROW
    assert out.structure.endpoints("B", "X")[1] == ARROW
    assert ARROW not in out.structure.endpoints("X", "Z")


def _cohort_series(seed: int = 0, steps: int = 3000) -> pd.DataFrame:
    """A VAR(1) of six variables: ``C0``-``C2`` observed throughout, ``M1`` in the first two
    thirds, and ``M2a`` and ``M2b`` in the last third, so no row observes ``M1`` with either."""
    rng = np.random.default_rng(seed)
    a = np.zeros((6, 6))
    a[np.arange(6), np.arange(6)] = 0.4
    a[0, 1] = a[1, 2] = a[0, 3] = a[3, 2] = a[0, 4] = a[4, 5] = 0.3
    x = np.zeros((steps, 6))
    for t in range(1, steps):
        x[t] = x[t - 1] @ a + rng.normal(size=6)
    df = pd.DataFrame(x, columns=["C0", "C1", "C2", "M1", "M2a", "M2b"])
    df.loc[2 * steps // 3 :, "M1"] = np.nan
    df.loc[: 2 * steps // 3 - 1, ["M2a", "M2b"]] = np.nan
    return df


def _drawn_links(out) -> set[tuple[str, str, int]]:
    """The lagged links drawn, as ``(cause, effect, lag)``."""
    found = set()
    for lag in out.structure.lags:
        if lag == 0:
            continue
        marks = out.structure.lag(lag).to_numpy()
        labels = list(out.structure.labels)
        for i, cause in enumerate(labels):
            for j, effect in enumerate(labels):
                if (i == j and marks[i, i] == 2) or (i != j and marks[j, i] == 2):
                    found.add((cause, effect, lag))
    return found


def test_pcmci_and_pcmciplus_leave_undecided_links_out_alike():
    df = _cohort_series()
    lagged = {}
    for method in (andrey.pcmci, andrey.pcmciplus):
        out = method(df, tau_max=1, indep_test="fisherz")
        undecided = [tuple(pair) for pair in out.metadata["undecided_pairs"]]
        assert undecided, method.__name__
        drawn = _drawn_links(out)
        for a, b in undecided:  # a lagged undecided link is not drawn
            if a.endswith(" at lag 1"):
                assert (a.split()[0], b, 1) not in drawn
            if b.endswith(" at lag 1"):
                assert (b.split()[0], a, 1) not in drawn
        lagged[method.__name__] = {p for p in undecided if "lag" in p[0] + p[1]}
        # Links between columns observed together are found: their tests condition only on
        # the parents that keep rows, so no block mixes M1 with M2a or M2b.
        assert {("C0", "C1", 1), ("C1", "C2", 1), ("C0", "M1", 1)} <= drawn
    assert lagged["pcmci"] == lagged["pcmciplus"]


def _merged_cohorts(seed: int = 0) -> pd.DataFrame:
    """Eight cohorts of 1,000 to 4,000 rows over 40 columns: 12 that every cohort measures, four
    modules of 6 that three cohorts each measure, and 4 that one cohort measures; 10% of the cells
    are missing at random besides."""
    rng = np.random.default_rng(seed)
    sizes = (4000, 3500, 3000, 2500, 2500, 2000, 1500, 1000)
    modules = ((0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 0))
    d = 12 + 6 * len(modules) + 4
    order = rng.permutation(d)
    w = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 3.0 / (d - 1):
                w[order[i], order[j]] = rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
    n = sum(sizes)
    x = np.zeros((n, d))
    for j in order:
        x[:, j] = x @ w[:, j] + rng.normal(size=n)
    observed = np.zeros((n, d), dtype=bool)
    start = 0
    for k, size in enumerate(sizes):
        rows = slice(start, start + size)
        observed[rows, :12] = True
        for m, cohorts in enumerate(modules):
            if k in cohorts:
                observed[rows, 12 + 6 * m : 18 + 6 * m] = True
        if k < 4:
            observed[rows, 36 + k] = True
        start += size
    observed &= rng.random((n, d)) >= 0.10
    return pd.DataFrame(np.where(observed, x, np.nan)[rng.permutation(n)])


def test_uncapped_pc_finishes_on_merged_cohorts_and_leaves_out_every_pair_never_observed():
    df = _merged_cohorts()
    out = andrey.pc(df)
    never = {tuple(pair) for pair in out.metadata["never_observed_together"]}
    assert never and never <= _pairs(out, "undecided_pairs")
    assert not _skeleton(out) & never


def test_complete_data_have_no_undecided_pair_and_no_flag():
    df = _hand("fork").fillna(0.0)
    complete = pd.DataFrame(np.random.default_rng(1).normal(size=(500, 3)), columns=df.columns)
    for method in (andrey.pc, andrey.fci, andrey.iamb, andrey.mmpc):
        metadata = method(complete).metadata
        assert "undecided_pairs" not in metadata and "possibly_separable" not in metadata


@pytest.mark.parametrize("method", ["pc", "fci", "iamb", "mmpc"])
def test_undecided_pairs_and_flags_do_not_depend_on_the_column_order(method):
    df = _undecided_end()
    fit = getattr(andrey, method)
    out = fit(df)
    moved = fit(df[["Z", "X", "B", "A"]])
    assert _skeleton(moved) == _skeleton(out)
    for key in ("undecided_pairs", "possibly_separable"):
        assert moved.metadata.get(key) == out.metadata.get(key), key
