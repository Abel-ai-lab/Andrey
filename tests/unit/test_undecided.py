"""Undecided pairs and possibly separable edges on merged cohorts.

A pair that no test with rows could decide, as a pair never observed together, is left out of the
graph and listed in ``metadata["undecided_pairs"]``; no orientation reads it. A drawn edge that a
conditioning set not tested might have separated is listed in ``metadata["possibly_separable"]``;
with ``possibly_separable="ambiguous"``, no rule reads a triple with such a side, one whose set not
tested lies within an end's other adjacencies, as a collider or a non-collider.
The hand cases are three variables ``X``, ``Y`` and a third, with weights 0.8, root noise 1 and
child noise 0.6, merged from cohorts that each observe some of them.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.constraint.fci import _iter_dsep_choices, _left_out
from andrey.constraint.fci import fci as fci_core
from andrey.constraint.pc import pc as pc_core
from andrey.core import ARROW, NULL, ci_missing

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


def _arrowheads(out) -> set[tuple[str, str]]:
    """The ``(a, b)`` edges with an arrowhead at ``b``."""
    heads = set()
    for a, b, _ in out.structure.oriented_edges():
        x, y = out.structure.endpoints(a, b)
        heads |= {(a, b)} if y == ARROW else set()
        heads |= {(b, a)} if x == ARROW else set()
    return heads


def _ambiguous(method: str, df: pd.DataFrame, **settings):
    """The triples ``method``'s core search reads as neither a collider nor a non-collider, by name
    with the ends in name order; the flagged pairs whose rules it vetoes, those with a set not
    tested within an end's adjacencies; its boolean adjacency; and the names."""
    core = {"pc": pc_core, "fci": fci_core}[method]
    sink: set[tuple[int, int, int]] = set()
    with ci_missing.rows_tested() as rows, ci_missing.reading_flags():
        graph = core(df.to_numpy(), _ambiguous=sink, **settings)
    names = [str(c) for c in df.columns]
    drawn = graph.to_numpy() != NULL
    flagged = {tuple(sorted((names[a], names[b]))) for a, b in rows.separable_within(drawn)}
    triples = {(min(names[a], names[c]), names[b], max(names[a], names[c])) for a, b, c in sink}
    return triples, flagged, drawn, names


FITS = {"pc": andrey.pc, "fci": andrey.fci}
AMBIGUOUS = {"possibly_separable": "ambiguous"}


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
    # No case has a triple with decided ends and unflagged sides, so none has an arrowhead when
    # flagged sides read as ambiguous; the graph and the lists stay.
    on = FITS[method](_hand(case, extra=extra), max_depth=3, **AMBIGUOUS)
    assert not _arrowheads(on)
    assert _skeleton(on) == drawn and on.metadata.get("not_tested") == out.metadata.get(
        "not_tested"
    )


@pytest.mark.parametrize("method", list(FITS))
def test_four_rows_observing_every_column_decide_the_pair_on_those_rows_alone(method):
    """The bridging cost: the four rows test X and Y with no power, so they separate them and the
    fork reads as a collider, while each edge stays flagged. Read as ambiguous, the flagged sides,
    which no row tests given the third column, make the fork no collider."""
    df = _hand("fork", extra=4)
    out = FITS[method](df, max_depth=3)
    assert "undecided_pairs" not in out.metadata or not out.metadata["undecided_pairs"]
    assert _skeleton(out) == {("X", "Z"), ("Y", "Z")}
    assert _pairs(out, "possibly_separable") == {("X", "Z"), ("Y", "Z")}
    assert _arrowheads(out) == {("X", "Z"), ("Y", "Z")}
    on = FITS[method](df, max_depth=3, **AMBIGUOUS)
    assert _skeleton(on) == _skeleton(out)
    assert not _arrowheads(on)
    triples, *_ = _ambiguous(method, df, max_depth=3)
    assert triples == {("X", "Z", "Y")}


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
    # Z, never observed with A or B, could have separated A - X and B - X: read as ambiguous, the
    # flagged sides make A - X - B, whose ends a test decided apart, no collider.
    assert _pairs(out, "possibly_separable") == {("A", "X"), ("B", "X"), ("X", "Z")}
    triples, *_ = _ambiguous(method, _undecided_end())
    assert ("A", "X", "B") in triples
    assert not _arrowheads(FITS[method](_undecided_end(), **AMBIGUOUS))


@pytest.mark.parametrize("method", list(FITS))
@pytest.mark.parametrize("seed", [0, 1])
def test_no_rule_reads_a_triple_with_a_flagged_side(method, seed):
    """On merged cohorts, read as ambiguous, every unshielded triple with a flagged side whose set
    not tested lies within an end's adjacencies is ambiguous, whether its ends are decided apart or
    undecided."""
    df = _merged_cohorts(seed)
    triples, flagged, drawn, names = _ambiguous(method, df, max_depth=2)
    expected = set()
    for b in range(len(names)):
        around = np.flatnonzero(drawn[b]).tolist()
        for a in around:
            for c in around:
                ends = (min(names[a], names[c]), max(names[a], names[c]))
                sides = {tuple(sorted((names[a], names[b]))), tuple(sorted((names[c], names[b])))}
                if a != c and not drawn[a, c] and sides & flagged:
                    expected.add((ends[0], names[b], ends[1]))
    assert expected and expected <= triples


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


@pytest.mark.parametrize("method", ["pcmci", "pcmciplus", "lpcmci"])
def test_a_required_link_beside_an_undecided_link_is_not_flagged(method):
    df = _cohort_series()
    knowledge = andrey.Knowledge(required=[("C0", "M1", 1)])
    out = getattr(andrey, method)(df, tau_max=1, knowledge=knowledge)
    assert out.metadata["undecided_pairs"]
    assert ("C0 at lag 1", "M1") not in _pairs(out, "possibly_separable")


def test_lpcmci_draws_no_undecided_link():
    out = andrey.lpcmci(_cohort_series(), tau_max=1, indep_test="fisherz")
    undecided = [tuple(pair) for pair in out.metadata["undecided_pairs"]]
    assert undecided
    labels = list(out.structure.labels)
    for a, b in undecided:
        (x, lag_x), (y, lag_y) = (
            (name.split()[0], int(name.split()[-1]) if "lag" in name else 0) for name in (a, b)
        )
        marks = out.structure.lag(abs(lag_x - lag_y)).to_numpy()
        i, j = labels.index(x), labels.index(y)
        assert marks[i, j] == 0 and marks[j, i] == 0, (a, b)


def _separator_series(periods: list[tuple[int, int, int]], n: int = 3000) -> pd.DataFrame:
    """Z drives X one step later and Y two steps later, so X at lag 1 and Y are joined through Z at
    lag 2 alone; equal consecutive periods observe the columns (X, Y, Z) each pattern marks."""
    rng = np.random.default_rng(0)
    x = np.zeros((n, 3))
    for t in range(2, n):
        z = 0.5 * x[t - 1, 2] + rng.standard_normal()
        cause = 0.3 * x[t - 1, 0] + 0.7 * x[t - 1, 2] + rng.standard_normal()
        effect = 0.3 * x[t - 1, 1] + 0.7 * x[t - 2, 2] + rng.standard_normal()
        x[t] = (cause, effect, z)
    period = np.arange(n) * len(periods) // n
    observed = np.array(periods, dtype=bool)[period]
    return pd.DataFrame(np.where(observed, x, np.nan), columns=["X", "Y", "Z"])


@pytest.mark.parametrize("method", ["pcmci", "pcmciplus", "lpcmci"])
def test_a_link_a_separator_never_observed_with_it_might_remove_is_flagged(method):
    df = _separator_series([(0, 1, 1), (1, 1, 0)])  # X and Z never observed together
    fit = getattr(andrey, method)
    out = fit(df, tau_max=2)
    assert ["X at lag 1", "Y"] in out.metadata["possibly_separable"]
    assert ["X", "Z at lag 1"] in out.metadata["undecided_pairs"]
    moved = fit(df[["Z", "Y", "X"]], tau_max=2)
    for key in ("undecided_pairs", "possibly_separable", "not_tested"):
        assert moved.metadata[key] == out.metadata[key], key


@pytest.mark.parametrize("method", ["pcmci", "pcmciplus", "lpcmci"])
def test_a_link_whose_separator_no_test_could_condition_on_is_flagged(method):
    # Every pair shares rows, so nothing is undecided: the flag comes from the tests that left Z at
    # lag 2 out of their conditions.
    df = _separator_series([(1, 1, 0), (0, 1, 1), (1, 0, 1)])
    fit = getattr(andrey, method)
    out = fit(df, tau_max=2)
    assert not out.metadata.get("undecided_pairs")
    assert ["X at lag 1", "Y"] in out.metadata["possibly_separable"]
    moved = fit(df[["Z", "Y", "X"]], tau_max=2)
    for key in ("possibly_separable", "not_tested"):
        assert moved.metadata[key] == out.metadata[key], key


def test_the_notice_names_the_possibly_separable_edges(caplog):
    df = _undecided_end()
    with caplog.at_level(logging.WARNING, logger="andrey"):
        logging.disable(logging.NOTSET)
        andrey.pc(df)
    (notice,) = [r.getMessage() for r in caplog.records if r.name == "andrey.api._method"]
    assert 'metadata["possibly_separable"]' in notice
    assert "3 drawn edges (A and X, B and X, X and Z) are flagged." in notice


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


def test_left_out_counts_the_dsep_subsets_a_pruned_search_skips():
    rng = np.random.default_rng(0)
    for _ in range(500):
        dsep = rng.permutation(12)[: rng.integers(0, 9)].tolist()
        kept = [v for v in dsep if rng.random() < 0.6]
        within = tuple({v for v in range(12) if rng.random() < 0.3} for _ in range(3))
        depth = [None, 1, 2, 3][rng.integers(4)]
        choices = list(_iter_dsep_choices(dsep, *within[:2], depth, within[2]))
        held = [c for c in choices if set(c) <= set(kept)]
        found = held[rng.integers(len(held))] if held and rng.random() < 0.7 else None
        reached = choices[: choices.index(found)] if found is not None else choices
        skipped = sum(not set(c) <= set(kept) for c in reached)
        assert _left_out(dsep, kept, within, depth, found) == skipped


def test_fci_asks_no_possible_dsep_set_no_row_can_test_and_keeps_its_answer(monkeypatch):
    df = _merged_cohorts()
    pruned = andrey.fci(df, max_depth=2)
    every = lambda self: np.ones((self._observed.shape[1],) * 2, dtype=bool)  # noqa: E731
    for kind in (ci_missing.TestwiseDeletion, ci_missing.GaussianDeletion):
        monkeypatch.setattr(kind, "together", every)
    asked = andrey.fci(df, max_depth=2)
    assert pruned.structure.oriented_edges() == asked.structure.oriented_edges()
    assert pruned.metadata == asked.metadata


@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_a_blanket_beside_an_undecided_candidate_is_flagged(method):
    # Y is never observed with X, and could separate X and Z.
    outs = getattr(andrey, method)(_hand("fork"), target=["X", "Y", "Z"])
    for out, edge in zip(outs, [("X", "Z"), ("Y", "Z")]):
        assert _skeleton(out) == {edge}
        assert _pairs(out, "undecided_pairs") == {("X", "Y")}
        assert _pairs(out, "possibly_separable") == {edge}
    assert _pairs(outs[2], "possibly_separable") == {("X", "Z"), ("Y", "Z")}
    # At depth 0 no conditioning set is asked, so none is missed.
    assert "possibly_separable" not in getattr(andrey, method)(_hand("fork"), max_depth=0).metadata


@pytest.mark.parametrize("method", ["iamb", "inter_iamb", "mmpc"])
def test_a_required_member_beside_an_undecided_candidate_stays_decided(method):
    knowledge = andrey.Knowledge(required=[("Z", "X")])
    out = getattr(andrey, method)(_hand("fork"), target="X", knowledge=knowledge)
    assert _skeleton(out) == {("X", "Z")}
    assert _pairs(out, "undecided_pairs") == {("X", "Y")}


def test_bootstrap_keeps_the_undecided_pairs_and_flags_of_its_fits():
    out = andrey.bootstrap("pc", _hand("fork"), n_resamples=4, missing="testwise", seed=0)
    assert _pairs(out, "undecided_pairs") == {("X", "Y")}
    assert _pairs(out, "possibly_separable") == {("X", "Z"), ("Y", "Z")}
    assert out.frequency("absent")[0, 1] == 1.0


def _shared_rows_cohorts(seed: int = 100, n: int = 2000) -> pd.DataFrame:
    """X0 -> X1, X0 -> X2, X1 -> X2, and X3 apart, whose sign picks the cohort: X2 is missing where
    X3 > 0, and X0 or X1, each on half of the other rows; three rows observe every column, so every
    pair shares rows."""
    rng = np.random.default_rng([seed, 11])
    x0 = rng.standard_normal(n)
    x1 = 0.8 * x0 + 0.6 * rng.standard_normal(n)
    x2 = 0.6 * x0 + 0.6 * x1 + 0.6 * rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    observed = np.ones((n, 4), dtype=bool)
    upper = x3 > 0
    observed[upper, 2] = False
    half = rng.random(n) < 0.5
    observed[~upper & half, 1] = False
    observed[~upper & ~half, 0] = False
    observed[rng.choice(n, 3, replace=False)] = True
    data = np.where(observed, np.column_stack([x0, x1, x2, x3]), np.nan)
    return pd.DataFrame(data, columns=["X0", "X1", "X2", "X3"])


def test_mvpc_keeps_the_edges_its_first_search_decided():
    # The correction's queries fall back to deletion tests without rows: they decide nothing.
    df = _shared_rows_cohorts()
    first = andrey.pc(df)
    out = andrey.pc(df, missing="mvpc")
    corrected = out.metadata["corrected_edges"]
    truth = {("X0", "X1"), ("X0", "X2"), ("X1", "X2")}
    assert _skeleton(first) == truth and not out.metadata["undecided_pairs"]
    assert corrected["uncorrected"]
    assert corrected["removed"] == []
    assert _skeleton(out) == truth


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


def _collider_beside_cohorts(n: int = 4000) -> pd.DataFrame:
    """``A -> C <- B``, observed in every row, with ``C -> D`` and ``C -> E``: one cohort observes
    ``D``, the other ``E``, so ``D`` and ``E`` are never observed together."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, n))
    c = 0.8 * a + 0.8 * b + 0.6 * rng.normal(size=n)
    d = 0.8 * c + 0.6 * rng.normal(size=n)
    e = 0.8 * c + 0.6 * rng.normal(size=n)
    df = pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e})
    df.loc[: n // 2 - 1, "E"] = np.nan
    df.loc[n // 2 :, "D"] = np.nan
    return df


@pytest.mark.parametrize("method", list(FITS))
def test_a_flagged_side_orients_nothing_and_a_decided_collider_stands(method):
    df = _collider_beside_cohorts()
    out = FITS[method](df, max_depth=1)
    assert _skeleton(out) == {("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")}
    assert _pairs(out, "undecided_pairs") == {("D", "E")}
    # E could have separated C and D, and D could have separated C and E.
    assert _pairs(out, "possibly_separable") == {("C", "D"), ("C", "E")}
    # Read as any other triple, the collider A -> C <- B orients C -> D and C -> E.
    assert _arrowheads(out) == {("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")}
    # Read as ambiguous: A - C - B has decided ends and unflagged sides; A - C - D and the others
    # with a flagged side read neither way, so no rule orients C - D or C - E.
    on = FITS[method](df, max_depth=1, **AMBIGUOUS)
    assert _skeleton(on) == _skeleton(out)
    assert _arrowheads(on) == {("A", "C"), ("B", "C")}
    moved = FITS[method](df[["E", "D", "C", "B", "A"]], max_depth=1, **AMBIGUOUS)
    assert _skeleton(moved) == _skeleton(on)
    assert _arrowheads(moved) == _arrowheads(on)
    for key in ("undecided_pairs", "possibly_separable", "not_tested"):
        assert moved.metadata[key] == on.metadata[key] == out.metadata[key], key


@pytest.mark.parametrize(
    "fit",
    [
        lambda df, env, **kw: andrey.cdnod(df, env=env, max_depth=1, **kw),
        lambda df, env, **kw: andrey.fci_jci(df, env=env, max_depth=1, **kw),
        lambda df, env, **kw: andrey.mmpc(df, structure="cpdag", max_depth=1, **kw),
    ],
    ids=["cdnod", "fci_jci", "mmpc"],
)
def test_the_other_orienting_searches_orient_nothing_from_a_flagged_side(fit):
    """Meek's first rule, or FCI's, orients C -> D and C -> E from the decided collider; read as
    ambiguous, A - C - D and B - C - E have a flagged side, and neither is oriented."""
    df = _collider_beside_cohorts()
    env = np.random.default_rng(7).integers(0, 2, len(df))
    out = fit(df, env)
    on = fit(df, env, **AMBIGUOUS)
    for graph in (out, on):
        assert _skeleton(graph) == {("A", "C"), ("B", "C"), ("C", "D"), ("C", "E")}
        assert _pairs(graph, "possibly_separable") == {("C", "D"), ("C", "E")}
    assert {("C", "D"), ("C", "E")} <= _arrowheads(out)
    assert _arrowheads(on) == {("A", "C"), ("B", "C")}


def test_pcmciplus_reads_no_collider_from_a_flagged_side():
    from andrey.temporal.pcmciplus import colliders

    # X0 - X1 - X2 at lag 0, X0 and X2 separated by the empty set: a collider at X1.
    adjacency = np.zeros((2, 3, 3), dtype=bool)
    adjacency[0, 0, 1] = adjacency[0, 1, 0] = adjacency[0, 1, 2] = adjacency[0, 2, 1] = True
    sepsets = {(0, 0, 2): (), (2, 0, 0): ()}
    args = (None, 3, ((), (), ()), adjacency, sepsets, 0.05, "sepsets")
    marks, ambiguous = colliders(*args)
    assert (marks == ARROW).sum() == 2 and not ambiguous  # X0 -> X1 <- X2
    flagged = np.zeros_like(adjacency)
    flagged[0, 0, 1] = flagged[0, 1, 0] = True
    marks, ambiguous = colliders(*args, flagged=flagged)
    assert ARROW not in marks and (0, 0, 1, 2) in ambiguous


def test_a_search_called_directly_reads_its_own_flags():
    """Outside any counting block, as when the engine is called directly, the flags still come
    from the search's own tests; outside a reading block, no flag is read."""
    X = _hand("fork", extra=4).to_numpy()
    for core in (pc_core, fci_core):
        sink: set[tuple[int, int, int]] = set()
        with ci_missing.reading_flags():
            core(X, max_depth=3, _ambiguous=sink)
        assert (0, 2, 1) in sink
        sink.clear()
        core(X, max_depth=3, _ambiguous=sink)
        assert (0, 2, 1) not in sink


def test_a_flagged_side_is_vetoed_only_by_a_set_within_an_ends_adjacencies():
    """A set not tested that holds a column adjacent to neither end of its pair cannot be the
    separating set the search missed; one within an end's other adjacencies, or one not known,
    can. The sets are kept only while flags are read."""
    # Pairs (0, 1), (1, 2), (2, 3) are drawn; 4 and 5 are adjacent to 3 only.
    drawn = np.zeros((6, 6), dtype=bool)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5)):
        drawn[a, b] = drawn[b, a] = True
    tested = [(0, 1), (1, 2), (2, 3)]
    rows = ci_missing.RowCounts()
    rows.add([50] * 3, [0.01] * 3, *zip(*tested))
    rows.add([1], [ci_missing.NOT_TESTED], [0], [1], [[4]])
    assert not rows.untested_sets
    with ci_missing.reading_flags():
        rows.add([1], [ci_missing.NOT_TESTED], [0], [1], [[4]])  # 4 is beside neither 0 nor 1
        rows.add([1], [ci_missing.NOT_TESTED], [1], [2], [[0]])  # 0 is beside 1
        rows.add([1], [ci_missing.NOT_TESTED], [2], [3], [[4, 5]])  # 4 and 5 are beside 3
        assert sorted(rows.possibly_separable(drawn)) == tested
        assert sorted(rows.separable_within(drawn)) == [(1, 2), (2, 3)]
        # A set not known is kept as the empty set, within every end's adjacencies.
        rows.add([1], [ci_missing.NOT_TESTED], [0], [1])
        assert sorted(rows.separable_within(drawn)) == tested
