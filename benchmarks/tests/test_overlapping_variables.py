"""The benchmark on datasets with overlapping variables: generators, plan, fits, and summary.

The fits run at a tiny size through ``run.run``, which forks one process per fit, so the arms that
patch Andrey (p = 1 and p = 0) never leak into this process. The fits of the recorded rows are
deterministic: one thread, fixed streams, and no wall time in the comparison. The rows of the p = 1
and p = 0 arms are those recorded for the 0.19.0 card, so the test fails when a change to Andrey
stops the patch from making a search answer as the usual answers do.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from overlapping_variables import classes, fit, generate, run, score, summary


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:12]


def test_the_cohort_variants_hold_the_rows_they_name():
    """No row observes every column without bridging rows, and 2% do with them."""
    pats = generate.cohort_patterns("S3")
    assert pats.shape == (3, generate.dims("S3")) and generate.dims("S3") == 20
    for variant, complete in (("b0s0", 0), ("b0s10", 0), ("b2s0", 100), ("b5s0", 250)):
        ds = generate.cohort("S3", variant, 0)
        assert ds["data"].shape == (5000, 20)
        assert int((~np.isnan(ds["data"])).all(axis=1).sum()) == complete
    complete = generate.cohort("S3", "complete", 0)
    assert not np.isnan(complete["data"]).any()
    again = generate.cohort("S3", "b0s10", 0)
    assert np.array_equal(again["data"], generate.cohort("S3", "b0s10", 0)["data"], equal_nan=True)
    assert not np.array_equal(
        again["data"], generate.cohort("S3", "b0s10", 1)["data"], equal_nan=True
    )


def test_the_hand_cases_and_the_fixed_fork_have_two_cohorts_that_never_meet():
    for ds in (generate.hand("H1", 1000, 0), generate.fixed_fork()):
        both = (~np.isnan(ds["data"])).all(axis=1)
        x_y = (~np.isnan(ds["data"][:, :2])).all(axis=1)
        assert not both.any() and not x_y.any()
        assert (~np.isnan(ds["data"][:, 2])).all()  # the third column is in every row
    assert (~np.isnan(generate.hand("H5k3", 1000, 0)["data"])).all(axis=1).sum() == 3


def test_the_pair_classes_agree_with_a_search_of_every_separating_set():
    checked = classes.check_rule(n_graphs=40)
    assert checked["checked"] > 100 and checked["mismatches"] == 0


def test_the_plan_lists_each_fit_once():
    for size, seeds, count in (("small", None, 186), ("card", None, 5250)):
        specs = run.plan(size, seeds)
        assert len(specs) == count
        assert len({fit.key(s) for s in specs}) == count
    assert run.parse_seeds("0-2,7") == (0, 1, 2, 7)
    narrowed = run.plan("card", (3,), ("M8",), ("cohort",))
    assert {s["design"] for s in narrowed} == {"cohort"}
    assert {s["cell"].split("_")[0] for s in narrowed} == {"M8"} and {
        s["seed"] for s in narrowed
    } == {3}


def _spec(design, cell, arm, method, depth=3, seed=0):
    return {
        "design": design,
        "cell": cell,
        "arm": arm,
        "method": method,
        "seed": seed,
        "depth": depth,
    }


SPECS = (
    [_spec("hand", "H1_n1000", arm, "pc", d) for arm in "ab" for d in (3, 0)]
    + [_spec("hand", "H1_n1000", "c", "pc"), _spec("hand", "H4_n1000", "a", "pc")]
    + [_spec("hand", "H4_n1000", "a", "fci"), _spec("hand", "H1_fixed", "a", "pc")]
    + [_spec("cohort", "S3_b0s10", arm, "pc") for arm in ("a", "amb", "b", "c", "dcover")]
    + [_spec("cohort", "S3_b0s0", "c", "pc"), _spec("cohort", "S3_complete", "f", "pc")]
    + [_spec("lagged", "nolap5", arm, "pcmci", None) for arm in "ab"]
    + [_spec("lagged", "nolap5", "b", "pcmciplus", None)]
    + [_spec("lagged", "complete", "f", "pcmci", None)]
)

#: Fits recorded for the 0.19.0 release card. Edges and lists are exact for the hand cases; the
#: cohort and lagged rows are the counts of edges (links), undecided pairs, and flagged edges, a
#: digest of the edges (links), and for PCMCI+ a digest of the lag-0 edges. The p = 1 rows (`b`) and
#: p = 0 rows (`c`) come from the run behind the card, which recorded no flagged edges for them
#: (``None``).
RECORDED = {
    "hand|H1_n1000|a|pc|0|3": ([[0, 2, 1, 1], [1, 2, 1, 1]], [[0, 1]], [[0, 2], [1, 2]]),
    "hand|H1_n1000|a|pc|0|0": ([[0, 2, 1, 1], [1, 2, 1, 1]], [[0, 1]], []),
    "hand|H1_n1000|b|pc|0|3": ([], [[0, 1]], None),
    "hand|H1_n1000|b|pc|0|0": ([[0, 2, 1, 2], [1, 2, 1, 2]], [[0, 1]], None),
    "hand|H1_n1000|c|pc|0|3": ([[0, 1, 1, 1], [0, 2, 1, 1], [1, 2, 1, 1]], [[0, 1]], None),
    "hand|H4_n1000|a|pc|0|3": ([[0, 2, 1, 1], [1, 2, 1, 1]], [[0, 1]], [[0, 2], [1, 2]]),
    "hand|H4_n1000|a|fci|0|3": ([[0, 2, 3, 3], [1, 2, 3, 3]], [[0, 1]], [[0, 2], [1, 2]]),
    "hand|H1_fixed|a|pc|0|3": ([[0, 2, 1, 1], [1, 2, 1, 1]], [[0, 1]], [[0, 2], [1, 2]]),
}
RECORDED_COUNTS = {
    "cohort|S3_b0s10|a|pc|0|3": (24, 37, 24, "438eeab61d9b"),
    "cohort|S3_b0s10|amb|pc|0|3": (24, 37, 24, "038a235d5c40"),
    "cohort|S3_b0s10|b|pc|0|3": (14, 37, None, "2732273877dc"),
    "cohort|S3_b0s10|c|pc|0|3": (61, 37, None, "db29eba5dd6c"),
    "cohort|S3_b0s0|c|pc|0|3": (63, 37, None, "ba6ebdba2c61"),
    "cohort|S3_b0s10|dcover|pc|0|3": (13, 0, None, "9d7597bf5d1e"),
    "cohort|S3_complete|f|pc|0|3": (26, 0, 0, "d1ca3aacbc92"),
    "lagged|nolap5|a|pcmci|0|None": (28, 36, 27, "ddc88938b065"),
    "lagged|nolap5|b|pcmci|0|None": (28, 0, None, "d45e9b2384f7"),
    "lagged|nolap5|b|pcmciplus|0|None": (23, 0, None, "005888563186"),
    "lagged|complete|f|pcmci|0|None": (30, 0, 0, "6b4ed2894746"),
}


#: A digest of the lag-0 edges of the PCMCI+ fit recorded for the card.
RECORDED_LAG0 = {"lagged|nolap5|b|pcmciplus|0|None": "8ae66b45e0cd"}


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    path = tmp_path_factory.mktemp("overlap") / "records.jsonl"
    assert run.run(SPECS, str(path), procs=4, cap=900) == len(SPECS)
    assert run.run(SPECS, str(path), procs=4, cap=900) == 0  # a restart skips the finished fits
    with open(path) as fh:
        return {r["key"]: r for r in map(json.loads, fh)}


def test_every_record_has_the_schema_of_its_design(records):
    assert set(records) == {fit.key(s) for s in SPECS}
    for rec in records.values():
        assert rec["status"] == "ok", rec.get("error")
        assert {"key", "design", "cell", "arm", "method", "seed", "depth", "d", "wall"} <= set(rec)
        assert isinstance(rec["wall"], float) and rec["wall"] >= 0
        assert "undecided" in rec and "possibly" in rec
        if rec["design"] == "lagged":
            assert all(len(link) == 3 and link[0] >= 1 for link in rec["links"])
            assert all(len(edge) == 4 for edge in rec["lag0"])
        else:
            assert all(len(e) == 4 and e[0] < e[1] < rec["d"] for e in rec["edges"])
            assert all(a < b for a, b in rec["undecided"])
        assert score.score(rec)["shd"] >= 0


def test_the_hand_cases_return_what_the_arms_say(records):
    """Andrey draws the two edges and lists the pair; p = 1 removes both edges, or at depth 0
    turns the pair into a collider; p = 0 draws the pair."""
    for key, (edges, undecided, possibly) in RECORDED.items():
        rec = records[key]
        assert rec["edges"] == edges
        assert rec["undecided"] == undecided
        if possibly is not None:
            assert rec["possibly"] == possibly
    assert records["hand|H1_n1000|b|pc|0|3"]["edges"] == []


def test_the_recorded_rows_of_the_cohort_and_lagged_fits_are_reproduced(records):
    for key, (n_edges, n_undecided, n_possibly, digest) in RECORDED_COUNTS.items():
        rec = records[key]
        graph = rec["links"] if rec["design"] == "lagged" else rec["edges"]
        assert (len(graph), len(rec["undecided"])) == (n_edges, n_undecided), key
        if n_possibly is not None:
            assert len(rec["possibly"]) == n_possibly, key
        assert _digest(graph) == digest, key
        if key in RECORDED_LAG0:
            assert _digest(rec["lag0"]) == RECORDED_LAG0[key], key


def test_andrey_draws_no_pair_without_rows_and_lists_every_one(records):
    for rec in records.values():
        if rec["arm"] != "a" or rec["design"] == "lagged":
            continue
        row = score.score(rec)
        assert (row["ii_drawn"], row["und_drawn_n"], row["ii_unlisted"]) == (0, 0, 0)
    a = score.score(records["cohort|S3_b0s10|a|pc|0|3"])
    assert a["shd"] < a["shd_empty"]  # fewer errors than the empty graph
    assert a["ul_n"] == 37 and a["eadj_rec"] > 0.8


def test_the_summary_writes_the_tables_of_the_cells_it_holds(records):
    fig = summary.figures(list(records.values()))
    row = fig["shd_table"]["small"]
    assert all(row[arm] is not None for arm in ("empty", "a", "b", "c", "dcover", "f"))
    assert row["a"][0] < row["empty"][0] < row["c"][0]  # p = 0 draws every pair never observed
    text = summary.tables(fig)
    assert text.count("| small |") == 2
    assert fig["undecided_pairs_listed_not_drawn"] == {"fits": 6, "exceptions": 0}
    assert fig["lagged"]["pcmci|nolap5|a"]["undecided_drawn"][0] == 0


def test_the_bootstrap_interval_and_the_verdict():
    mean, lo, hi = summary.boot(list(range(20)))
    assert lo < mean == 9.5 < hi
    assert summary.boot([]) is None and summary.boot([3.0, 3.0]) == (3.0, 3.0, 3.0)
    assert summary.verdict([-5.0, -6.0, -4.0, -5.5], 1.0, higher_better=False) == "better"
    assert summary.verdict([5.0, 6.0, 4.0, 5.5], 1.0, higher_better=False) == "worse"
    assert summary.verdict([0.1, -0.1, 0.0, 0.05], 1.0, higher_better=False) == "not worse"


def test_the_card_order_names_every_seed_of_its_two_cells_once():
    assert set(summary.CARD_ORDER) == {
        ("cohort", "M8_b0s10", "pc", 3, "a"),
        ("cohort", "L6_b0s10", "pc", 3, "a"),
    }
    for order in summary.CARD_ORDER.values():
        assert sorted(order) == list(range(20))
