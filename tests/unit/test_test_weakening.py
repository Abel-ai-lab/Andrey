"""The weakened-test check flags each kind of weakening and passes added and tightened tests."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = "tests/unit/test_example.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location(
        "test_weakening", ROOT / "scripts/test_weakening.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reasons(check, old: str | None, new: str | None, path: str = PATH) -> list[str]:
    """The reasons the check flags the change of one file from ``old`` to ``new``."""
    old = None if old is None else textwrap.dedent(old)
    new = None if new is None else textwrap.dedent(new)
    return [finding.reason for finding in check.analyze([check.Change(path, path, old, new)])]


FLAGGED = {
    "assert removed": (
        """
        def test_fit():
            result = fit()
            assert result.n_edges == 3
            assert result.score > 0
        """,
        """
        def test_fit():
            result = fit()
            assert result.n_edges == 3
        """,
        "assertion removed",
    ),
    "assert call removed": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b)
            np.testing.assert_array_equal(c, d)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b)
        """,
        "assertion removed",
    ),
    "raises removed": (
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
            with pytest.raises(TypeError):
                fit(1)
        """,
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
        """,
        "pytest.raises check removed",
    ),
    "raises lost its match": (
        """
        def test_fit():
            with pytest.raises(ValueError, match="alpha"):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
        """,
        "lost its match",
    ),
    "raises widened": (
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(Exception):
                fit(None)
        """,
        "widened to Exception",
    ),
    "approx assertion rewritten": (
        """
        def test_fit():
            assert score == pytest.approx(0.5)
        """,
        """
        def test_fit():
            assert score is not None
        """,
        "pytest.approx assertion removed or rewritten",
    ),
    "rtol keyword": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b, rtol=1e-8)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b, rtol=1e-4)
        """,
        "rtol loosened from 1e-08 to 0.0001",
    ),
    "atol positional": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b, 1e-8, 0)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b, 1e-8, 1e-3)
        """,
        "atol loosened",
    ),
    "tolerance added over the default": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b, rtol=1e-3)
        """,
        "rtol loosened from 1e-07 to 0.001",
    ),
    "decimal smaller": (
        """
        def test_fit():
            np.testing.assert_almost_equal(a, b, decimal=8)
        """,
        """
        def test_fit():
            np.testing.assert_almost_equal(a, b, decimal=3)
        """,
        "decimal loosened from 8 to 3",
    ),
    "approx rel": (
        """
        def test_fit():
            assert a == pytest.approx(b, rel=1e-9)
        """,
        """
        def test_fit():
            assert a == pytest.approx(b, rel=1e-3)
        """,
        "rel loosened",
    ),
    "approx abs given": (
        """
        def test_fit():
            assert a == pytest.approx(b, abs=1e-9)
        """,
        """
        def test_fit():
            assert a == pytest.approx(b, abs=1e-2)
        """,
        "abs loosened",
    ),
    "exact turned into approx": (
        """
        def test_fit():
            assert a == b
        """,
        """
        def test_fit():
            assert a == pytest.approx(b, abs=1e-3)
        """,
        "exact comparison relaxed to pytest.approx",
    ),
    "upper bound": (
        """
        def test_fit():
            assert abs(a - b) < 1e-6
        """,
        """
        def test_fit():
            assert abs(a - b) < 1e-3
        """,
        "bound loosened from 1e-06 to 0.001",
    ),
    "lower bound": (
        """
        def test_fit():
            assert score > 0.9
        """,
        """
        def test_fit():
            assert score > 0.5
        """,
        "bound loosened from 0.9 to 0.5",
    ),
    "bound with the number first": (
        """
        def test_fit():
            assert 0.9 <= score
        """,
        """
        def test_fit():
            assert 0.5 <= score
        """,
        "bound loosened",
    ),
    "bound written as a power": (
        """
        def test_fit():
            assert err < 10**-9
        """,
        """
        def test_fit():
            assert err < 10**-3
        """,
        "bound loosened",
    ),
    "chained upper bound": (
        """
        def test_fit():
            assert 0 <= score <= 1
        """,
        """
        def test_fit():
            assert 0 <= score <= 2
        """,
        "bound loosened from 1 to 2",
    ),
    "chained lower bound": (
        """
        def test_fit():
            assert 0.5 < score < 1
        """,
        """
        def test_fit():
            assert 0.1 < score < 1
        """,
        "bound loosened from 0.5 to 0.1",
    ),
    "strict bound made non-strict": (
        """
        def test_fit():
            assert score < 1
            assert score > 0
        """,
        """
        def test_fit():
            assert score <= 1
            assert score >= 0
        """,
        "loosened from strict to non-strict at 1",
    ),
    "strict bound moved and made non-strict": (
        """
        def test_fit():
            assert score < 1
        """,
        """
        def test_fit():
            assert score <= 2
        """,
        "bound loosened from 1 to 2",
    ),
    "lower strict bound moved and made non-strict": (
        """
        def test_fit():
            assert score > 1
        """,
        """
        def test_fit():
            assert score >= 0
        """,
        "bound loosened from 1 to 0",
    ),
    "range of parametrize cases shortened": (
        """
        @pytest.mark.parametrize("seed", range(6))
        def test_fit(seed):
            assert fit(seed)
        """,
        """
        @pytest.mark.parametrize("seed", range(1))
        def test_fit(seed):
            assert fit(seed)
        """,
        "5 parametrize case(s) removed",
    ),
    "range with a start shortened": (
        """
        SEEDS = list(range(2, 8))

        @pytest.mark.parametrize("seed", SEEDS)
        def test_fit(seed):
            assert fit(seed)
        """,
        """
        SEEDS = list(range(2, 4))

        @pytest.mark.parametrize("seed", SEEDS)
        def test_fit(seed):
            assert fit(seed)
        """,
        "4 parametrize case(s) removed",
    ),
    "list cut to a range": (
        """
        @pytest.mark.parametrize("seed", [0, 1, 2, 3])
        def test_fit(seed):
            assert fit(seed)
        """,
        """
        @pytest.mark.parametrize("seed", range(2))
        def test_fit(seed):
            assert fit(seed)
        """,
        "2 parametrize case(s) removed",
    ),
    "xfail in a case list an existing test uses": (
        """
        CASES = [pytest.param(1), pytest.param(2)]

        @pytest.mark.parametrize("n", CASES)
        def test_fit(n):
            assert fit(n)
        """,
        """
        CASES = [pytest.param(1), pytest.param(2, marks=pytest.mark.xfail)]

        @pytest.mark.parametrize("n", CASES)
        def test_fit(n):
            assert fit(n)
        """,
        "pytest.mark.xfail added to a test that existed",
    ),
    "module constant": (
        """
        RTOL = 1e-8
        """,
        """
        RTOL = 1e-4
        """,
        "RTOL loosened",
    ),
    "tolerance table": (
        """
        SPEC = {"PC": {"numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}}}}
        """,
        """
        SPEC = {"PC": {"numeric": {"score": {"atol": 1e-3, "rtol": 1e-6}}}}
        """,
        "SPEC.PC.numeric.score.atol loosened",
    ),
    "default of a parameter": (
        """
        def close(a, b, atol=1e-8):
            return abs(a - b) <= atol
        """,
        """
        def close(a, b, atol=1e-3):
            return abs(a - b) <= atol
        """,
        "atol loosened",
    ),
    "skip marker": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skip(reason="broken")
        def test_fit():
            assert fit()
        """,
        "pytest.mark.skip added",
    ),
    "skipif marker": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif(sys.platform == "linux", reason="slow")
        def test_fit():
            assert fit()
        """,
        "pytest.mark.skipif added",
    ),
    "xfail marker": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            @pytest.mark.xfail
            def test_fit(self):
                assert fit()
        """,
        "pytest.mark.xfail added",
    ),
    "skip call": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            if not fit():
                pytest.skip("not today")
            assert fit()
        """,
        "pytest.skip() added",
    ),
    "release marker": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.release
        def test_fit():
            assert fit()
        """,
        "pytest.mark.release added",
    ),
    "module marker": (
        """
        def test_fit():
            assert fit()
        """,
        """
        pytestmark = pytest.mark.skip
        def test_fit():
            assert fit()
        """,
        "pytest.mark.skip added",
    ),
    "skip in a parametrize case": (
        """
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [1, pytest.param(2, marks=pytest.mark.xfail)])
        def test_fit(n):
            assert fit(n)
        """,
        "pytest.mark.xfail added",
    ),
    "deleted test": (
        """
        def test_fit():
            assert fit()

        def test_other():
            assert other()
        """,
        """
        def test_fit():
            assert fit()
        """,
        "test deleted",
    ),
    "deleted method": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()

            def test_other(self):
                assert other()
        """,
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        "test deleted",
    ),
    "removed parametrize case": (
        """
        @pytest.mark.parametrize("n", [1, 2, 3])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n):
            assert fit(n)
        """,
        "1 parametrize case(s) removed",
    ),
    "removed case of a module list": (
        """
        CASES = [(1, 2), (3, 4), (5, 6)]

        @pytest.mark.parametrize("a, b", CASES)
        def test_fit(a, b):
            assert fit(a, b)
        """,
        """
        CASES = [(1, 2), (3, 4)]

        @pytest.mark.parametrize("a, b", CASES)
        def test_fit(a, b):
            assert fit(a, b)
        """,
        "1 parametrize case(s) removed",
    ),
    "removed fixture param": (
        """
        @pytest.fixture(params=["a", "b"])
        def kind(request):
            return request.param
        """,
        """
        @pytest.fixture(params=["a"])
        def kind(request):
            return request.param
        """,
        "1 parametrize case(s) removed",
    ),
    "removed parametrize decorator": (
        """
        @pytest.mark.parametrize("n", [1, 2])
        @pytest.mark.parametrize("m", [1, 2])
        def test_fit(n, m):
            assert fit(n, m)
        """,
        """
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n, m=1):
            assert fit(n, m)
        """,
        "parametrization of m removed",
    ),
    "file that does not parse": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit(:
            assert fit()
        """,
        "does not parse",
    ),
}


@pytest.mark.parametrize("name", FLAGGED)
def test_each_weakening_is_flagged(check, name):
    old, new, expected = FLAGGED[name]
    found = reasons(check, old, new)
    assert any(expected in reason for reason in found), found


def test_deleting_a_file_flags_its_tests(check):
    old = """
    def test_fit():
        assert fit()

    def helper():
        return 1
    """
    assert reasons(check, old, None) == ["test deleted"]


NOT_FLAGGED = {
    "added test": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()

        @pytest.mark.skipif(sys.platform == "win32", reason="posix only")
        def test_new():
            torch = pytest.importorskip("torch")
            assert torch is not None
            assert fit() == pytest.approx(1, abs=1e-1)
        """,
    ),
    "added test in a class": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            def test_fit(self):
                assert fit()

            @pytest.mark.xfail
            def test_new(self):
                assert fit() < 1
        """,
    ),
    "added assert": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()
            assert other()
            with pytest.raises(ValueError):
                fit(None)
        """,
    ),
    "tighter keyword": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b, rtol=1e-4)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b, rtol=1e-9)
        """,
    ),
    "tighter than the default": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b)
            assert x == pytest.approx(y)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=0)
            assert x == pytest.approx(y, rel=1e-9)
        """,
    ),
    "tighter decimal": (
        """
        def test_fit():
            np.testing.assert_almost_equal(a, b, decimal=3)
        """,
        """
        def test_fit():
            np.testing.assert_almost_equal(a, b, decimal=7)
        """,
    ),
    "tighter bounds": (
        """
        def test_fit():
            assert abs(a - b) < 1e-3
            assert score > 0.5
            assert 0.5 <= score
        """,
        """
        def test_fit():
            assert abs(a - b) < 1e-6
            assert score > 0.9
            assert 0.9 <= score
        """,
    ),
    "tighter constant and table": (
        """
        RTOL = 1e-4
        SPEC = {"PC": {"numeric": {"score": {"atol": 1e-3}}}}

        def close(a, b, atol=1e-3):
            return abs(a - b) <= atol
        """,
        """
        RTOL = 1e-8
        SPEC = {"PC": {"numeric": {"score": {"atol": 1e-6}}}}

        def close(a, b, atol=1e-8):
            return abs(a - b) <= atol
        """,
    ),
    "approx turned exact": (
        """
        def test_fit():
            assert a == pytest.approx(b, rel=1e-3)
        """,
        """
        def test_fit():
            assert a == b
        """,
    ),
    "markers removed": (
        """
        @pytest.mark.release
        @pytest.mark.skip
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()
        """,
    ),
    "tighter chained bound": (
        """
        def test_fit():
            assert 0 <= score <= 2
        """,
        """
        def test_fit():
            assert 0 <= score <= 1
        """,
    ),
    "non-strict bound made strict": (
        """
        def test_fit():
            assert score <= 1
            assert score >= 0
        """,
        """
        def test_fit():
            assert score < 1
            assert score > 0
        """,
    ),
    "range of parametrize cases grown": (
        """
        @pytest.mark.parametrize("seed", range(3))
        def test_fit(seed):
            assert fit(seed)
        """,
        """
        @pytest.mark.parametrize("seed", range(6))
        def test_fit(seed):
            assert fit(seed)
        """,
    ),
    "xfail in a case list only an added test uses": (
        """
        def test_fit():
            assert fit()
        """,
        """
        NEW_CASES = [pytest.param(1), pytest.param(2, marks=pytest.mark.xfail)]

        def test_fit():
            assert fit()

        @pytest.mark.parametrize("n", NEW_CASES)
        def test_new(n):
            assert fit(n)
        """,
    ),
    "xfail in a case list only an added method uses": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        CASES = [pytest.param(1, marks=pytest.mark.skip)]

        class TestFit:
            def test_fit(self):
                assert fit()

            @pytest.mark.parametrize("n", CASES)
            def test_new(self, n):
                assert fit(n)
        """,
    ),
    "added parametrize cases": (
        """
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [1, 2, 3])
        @pytest.mark.parametrize("m", [1, 2])
        def test_fit(n, m):
            assert fit(n, m)
        """,
    ),
    "merged parametrize decorators": (
        """
        @pytest.mark.parametrize("robust", [False, True])
        @pytest.mark.parametrize("name", NAMES)
        def test_fit(name, robust):
            assert fit(name, robust)
        """,
        """
        @pytest.mark.parametrize(("name", "robust"), CASES)
        def test_fit(name, robust):
            assert fit(name, robust)
        """,
    ),
    "unchanged": (
        """
        @pytest.mark.skip
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n):
            assert fit(n) < 1e-6
            np.testing.assert_allclose(a, b, rtol=1e-8)
        """,
        """
        # A comment and a docstring change nothing.
        @pytest.mark.skip
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n):
            '''Fit.'''
            assert fit(n) < 1e-6
            np.testing.assert_allclose(a, b, rtol=1e-8)
        """,
    ),
    "renamed test": (
        """
        def test_fit():
            '''Fit.'''
            assert fit() == 3
        """,
        """
        def test_fit_returns_three():
            assert fit() == 3
        """,
    ),
    "assert moved inside a with block": (
        """
        def test_fit():
            assert fit() == 3
        """,
        """
        def test_fit():
            with warnings.catch_warnings():
                assert fit() == 3
        """,
    ),
}


@pytest.mark.parametrize("name", NOT_FLAGGED)
def test_added_tightened_and_unchanged_tests_are_not_flagged(check, name):
    old, new = NOT_FLAGGED[name]
    assert reasons(check, old, new) == []


def test_a_new_file_is_not_flagged(check):
    new = """
    pytestmark = pytest.mark.skip
    RTOL = 1e-1

    @pytest.mark.xfail
    def test_fit():
        pytest.importorskip("torch")
    """
    assert reasons(check, None, new) == []


def test_a_test_moved_to_another_file_is_not_flagged(check):
    body = """
    def test_fit():
        assert fit() == 3
        np.testing.assert_allclose(a, b, rtol=1e-8)
    """
    other = """
    def test_other():
        assert other()
    """
    a, b = "tests/unit/test_a.py", "tests/unit/test_b.py"
    changes = [
        check.Change(a, a, textwrap.dedent(body + other), textwrap.dedent(other)),
        check.Change(b, b, "x = 1\n", textwrap.dedent(body)),
    ]
    assert check.analyze(changes) == []


def test_a_test_that_moved_and_loosened_is_flagged(check):
    old = """
    def test_fit():
        np.testing.assert_allclose(a, b, rtol=1e-8)
    """
    new = """
    def test_fit():
        np.testing.assert_allclose(a, b, rtol=1e-2)
    """
    changes = [
        check.Change("tests/unit/test_a.py", "tests/unit/test_a.py", textwrap.dedent(old), ""),
        check.Change(
            "tests/unit/test_b.py", "tests/unit/test_b.py", "x = 1\n", textwrap.dedent(new)
        ),
    ]
    found = [finding.reason for finding in check.analyze(changes)]
    assert any("rtol loosened" in reason for reason in found), found


def test_a_test_moved_into_a_new_file_is_not_flagged(check):
    body = """
    def test_fit():
        assert fit() == 3
    """
    other = """
    def test_other():
        assert other()
    """
    a, b = "tests/unit/test_a.py", "tests/unit/test_b.py"
    changes = [
        check.Change(a, a, textwrap.dedent(body + other), textwrap.dedent(other)),
        check.Change(b, b, None, textwrap.dedent(body)),
    ]
    assert check.analyze(changes) == []


def test_a_test_moved_into_a_new_file_and_loosened_is_flagged(check):
    old = "def test_fit():\n    np.testing.assert_allclose(a, b, rtol=1e-8)\n"
    new = "def test_fit():\n    np.testing.assert_allclose(a, b, rtol=1e-2)\n"
    other = "def test_other():\n    assert other()\n"
    a, b = "tests/unit/test_a.py", "tests/unit/test_b.py"
    changes = [check.Change(a, a, old + other, other), check.Change(b, b, None, new)]
    found = [finding.reason for finding in check.analyze(changes)]
    assert any("rtol loosened" in reason for reason in found), found


def test_a_file_renamed_out_of_the_collected_names_loses_its_tests(check):
    source = "def test_fit():\n    assert fit()\n\n\ndef helper():\n    return 1\n"
    old, new = "tests/unit/test_fit.py", "tests/unit/fit.py"
    found = check.analyze([check.Change(new, old, source, source)])
    assert [(f.path, f.unit, f.reason) for f in found] == [
        (old, "test_fit", f"test no longer collected: the file was renamed to {new}")
    ]


@pytest.mark.parametrize("new", ["tests/unit/fit_test.py", "tests/unit/test_other.py"])
def test_a_file_renamed_within_the_collected_names_is_not_flagged(check, new):
    source = "def test_fit():\n    assert fit()\n"
    assert check.analyze([check.Change(new, "tests/unit/test_fit.py", source, source)]) == []


def test_a_file_renamed_into_the_collected_names_is_not_flagged(check):
    source = "def test_fit():\n    assert fit()\n"
    assert (
        check.analyze([check.Change("tests/unit/test_fit.py", "tests/unit/fit.py", source, source)])
        == []
    )


def test_a_flag_names_the_file_the_test_and_the_code(check):
    old = """
    def test_fit():
        assert abs(a - b) < 1e-6
    """
    new = """
    def test_fit():
        assert abs(a - b) < 1e-3
    """
    change = check.Change(PATH, PATH, textwrap.dedent(old), textwrap.dedent(new))
    text = check.report(check.analyze([change]))
    assert f"{PATH}:3: test_fit: bound loosened" in text
    assert "- assert abs(a - b) < 1e-6" in text
    assert "+ assert abs(a - b) < 1e-3" in text


BASELINE = "tests/recovery/baselines/PC/gauss.json"


def test_a_changed_baseline_value_is_flagged(check):
    found = check.baseline_findings(
        BASELINE, '{"graph": [[0, 1]], "score": 1.5}', '{"graph": [[0, 1]], "score": 1.6}'
    )
    assert [f.reason for f in found] == ["baseline re-recorded (changed: score)"]


def test_a_removed_baseline_value_is_flagged(check):
    found = check.baseline_findings(BASELINE, '{"graph": [1, 2], "score": 1.5}', '{"graph": [1]}')
    assert [f.reason for f in found] == ["baseline re-recorded (changed: graph[1], score)"]


def test_a_baseline_with_an_added_value_is_not_flagged(check):
    assert check.baseline_findings(BASELINE, '{"graph": [1]}', '{"graph": [1], "score": 1.6}') == []
    assert (
        check.baseline_findings("qa/quality_baselines.json", '{"PC": 1}', '{"PC": 1, "GES": 2}')
        == []
    )


def test_a_new_or_unchanged_baseline_is_not_flagged(check):
    assert check.baseline_findings(BASELINE, None, "{}") == []
    assert check.baseline_findings(BASELINE, "{}", "{}") == []


def test_a_deleted_baseline_is_flagged(check):
    assert [f.reason for f in check.baseline_findings(BASELINE, "{}", None)] == ["baseline deleted"]


def test_a_changed_binary_baseline_is_flagged(check):
    found = check.baseline_findings("tests/recovery/baselines/x.npz", "a", "b")
    assert [f.reason for f in found] == ["baseline re-recorded"]


# The same checks over a real repository: the file statuses, the baselines, and the exit code.


def git(repo: Path, *args: str) -> None:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t"}
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(repo)}
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        env=env,
        capture_output=True,
    )


def write(repo: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    git(tmp_path, "init", "-q", "-b", "main")
    write(
        tmp_path,
        {
            "tests/unit/test_a.py": """
                def test_a():
                    assert a() == 1
                    assert b() == 2
                    assert c() == 3
                """,
            "tests/unit/test_b.py": """
                def test_b():
                    assert b() == 2
                    assert b() == 3
                    assert b() == 4
                """,
            "tests/unit/test_c.py": """
                def test_c():
                    assert c() == 3
                """,
            "tests/recovery/baselines/PC/gauss.json": '{"score": 1.5}',
            "tests/recovery/baselines/PC/old.json": '{"score": 2.5}',
            "tests/unit/data.json": '{"x": 1}',
            "src/andrey/__init__.py": "x = 1\n",
        },
    )
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "checkout", "-q", "-b", "feature")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def commit(repo: Path) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")


def test_statuses_and_baselines_of_a_real_change(check, repo):
    write(
        repo,
        {
            "tests/unit/test_a.py": """
                def test_a():
                    assert a() == 1
                    assert b() == 2
                """,
            "tests/unit/test_d.py": """
                @pytest.mark.skip
                def test_d():
                    assert d()
                """,
            "tests/recovery/baselines/PC/gauss.json": '{"score": 1.6}',
            "tests/recovery/baselines/PC/new.json": '{"score": 1.0}',
            "tests/unit/data.json": '{"x": 2}',
            "src/andrey/__init__.py": "x = 2\n",
        },
    )
    (repo / "tests/recovery/baselines/PC/old.json").unlink()
    git(repo, "mv", "tests/unit/test_c.py", "tests/unit/test_e.py")
    (repo / "tests/unit/test_b.py").unlink()
    commit(repo)
    changes, baselines = check.load("main", "HEAD", str(repo))
    found = {(f.path, f.reason) for f in [*check.analyze(changes), *baselines]}
    assert found == {
        ("tests/unit/test_a.py", "assertion removed"),
        ("tests/unit/test_b.py", "test deleted"),
        ("tests/recovery/baselines/PC/gauss.json", "baseline re-recorded (changed: score)"),
        ("tests/recovery/baselines/PC/old.json", "baseline deleted"),
    }


def test_a_renamed_file_is_compared_with_its_old_content(check, repo):
    git(repo, "mv", "tests/unit/test_a.py", "tests/unit/test_z.py")
    write(
        repo, {"tests/unit/test_z.py": "def test_a():\n    assert a() == 1\n    assert b() == 2\n"}
    )
    commit(repo)
    changes, baselines = check.load("main", "HEAD", str(repo))
    found = check.analyze(changes)
    assert [(f.path, f.reason) for f in found] == [("tests/unit/test_z.py", "assertion removed")]
    assert baselines == []


def test_a_baseline_moved_out_of_its_folder_is_a_deletion(check, repo):
    (repo / "tests/recovery/data").mkdir()
    git(repo, "mv", "tests/recovery/baselines/PC/gauss.json", "tests/recovery/data/gauss.json")
    commit(repo)
    changes, baselines = check.load("main", "HEAD", str(repo))
    assert [(f.path, f.reason) for f in baselines] == [
        ("tests/recovery/baselines/PC/gauss.json", "baseline deleted")
    ]
    assert changes == []


def test_a_baseline_moved_within_its_folder_is_not_flagged(check, repo):
    (repo / "tests/recovery/baselines/FCI").mkdir()
    git(repo, "mv", "tests/recovery/baselines/PC/gauss.json", "tests/recovery/baselines/FCI/g.json")
    commit(repo)
    assert check.load("main", "HEAD", str(repo)) == ([], [])


def test_a_change_outside_tests_and_qa_is_not_read(check, repo):
    write(repo, {"src/andrey/__init__.py": "x = 3\n", "tests/unit/data.json": '{"x": 9}'})
    commit(repo)
    assert check.main(["--base", "main"]) == 0


SINCE = "2026-11-20T12:00:00Z"
FLAGGED_CHANGE = {"tests/unit/test_a.py": "def test_a():\n    assert a() == 1\n"}


def when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def test_a_flagged_change_fails_without_the_label(check, repo, monkeypatch, capsys):
    write(repo, FLAGGED_CHANGE)
    commit(repo)
    monkeypatch.setattr(check, "label_applied", lambda repository, pull_request, label: None)
    argv = ["--base", "main", "--pr", "7", "--since", SINCE, "--repo", "o/r"]
    assert check.main(argv) == 1
    output = capsys.readouterr().out
    assert "tests/unit/test_a.py:1: test_a: 2 assertions removed" in output
    assert (
        "::error file=tests/unit/test_a.py,line=1,title=Weakened test::2 assertions removed"
        in output
    )
    assert "test-change-approved" in output
    assert "::warning" not in output


def test_a_flagged_change_passes_with_a_label_set_after_the_head_changed(
    check, repo, monkeypatch, capsys
):
    write(repo, FLAGGED_CHANGE)
    commit(repo)
    seen = []

    def label_applied(repository, pull_request, label):
        seen.append((repository, pull_request, label))
        return when("2026-11-20T12:30:00Z")

    monkeypatch.setattr(check, "label_applied", label_applied)
    assert check.main(["--base", "main", "--pr", "7", "--since", SINCE, "--repo", "o/r"]) == 0
    assert seen == [("o/r", "7", "test-change-approved")]
    assert "2 assertions removed" in capsys.readouterr().out


@pytest.mark.parametrize("applied", ["2026-11-20T11:59:59Z", SINCE])
def test_a_label_set_before_the_head_changed_does_not_approve(
    check, repo, monkeypatch, capsys, applied
):
    write(repo, FLAGGED_CHANGE)
    commit(repo)
    monkeypatch.setattr(check, "label_applied", lambda *args: when(applied))
    argv = ["--base", "main", "--pr", "7", "--since", SINCE, "--repo", "o/r"]
    assert check.main(argv) == 1
    output = capsys.readouterr().out
    assert "::warning::The label test-change-approved was set before" in output
    assert "title=Weakened test::2 assertions removed" in output


def test_a_pull_request_needs_the_time_of_its_last_head_change(check, repo, capsys):
    write(repo, FLAGGED_CHANGE)
    commit(repo)
    with pytest.raises(SystemExit) as stop:
        check.main(["--base", "main", "--pr", "7", "--repo", "o/r"])
    assert stop.value.code == 2
    assert "--pr needs --since" in capsys.readouterr().err


def test_a_flagged_change_without_a_pull_request_fails(check, repo):
    write(repo, FLAGGED_CHANGE)
    commit(repo)
    assert check.main(["--base", "main"]) == 1


def test_a_clean_change_never_asks_for_the_label(check, repo, monkeypatch):
    write(repo, {"tests/unit/test_f.py": "def test_f():\n    assert f()\n"})
    commit(repo)

    def label_applied(repository, pull_request, label):
        raise AssertionError("a clean change needs no label")

    monkeypatch.setattr(check, "label_applied", label_applied)
    argv = ["--base", "main", "--pr", "7", "--since", SINCE, "--repo", "o/r"]
    assert check.main(argv) == 0


@pytest.mark.parametrize("stderr", ["HTTP 403", b"HTTP 403"])
def test_a_failed_label_lookup_fails_the_check(check, repo, monkeypatch, capsys, stderr):
    write(repo, FLAGGED_CHANGE)
    commit(repo)

    def label_applied(repository, pull_request, label):
        raise subprocess.CalledProcessError(1, ["gh"], stderr=stderr)

    monkeypatch.setattr(check, "label_applied", label_applied)
    argv = ["--base", "main", "--pr", "7", "--since", SINCE, "--repo", "o/r"]
    assert check.main(argv) == 2
    output = capsys.readouterr().out
    assert "could not run" in output
    assert "HTTP 403" in output
    assert "b'" not in output


def test_a_malformed_time_fails_the_check(check, repo, capsys):
    write(repo, FLAGGED_CHANGE)
    commit(repo)
    assert check.main(["--base", "main", "--pr", "7", "--since", "yesterday"]) == 2
    assert "could not run" in capsys.readouterr().out


def events(*rows: tuple[str, str, str]) -> str:
    """What ``gh api`` prints for the events of a pull request: event, label, time per line."""
    return "".join("\t".join(row) + "\n" for row in rows)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([], None),
        ([("labeled", "chore", "2026-11-20T10:00:00Z")], None),
        ([("assigned", "", "2026-11-20T10:00:00Z")], None),
        (
            [("labeled", "test-change-approved", "2026-11-20T10:00:00Z")],
            "2026-11-20T10:00:00Z",
        ),
        (
            [
                ("labeled", "test-change-approved", "2026-11-20T10:00:00Z"),
                ("unlabeled", "test-change-approved", "2026-11-20T11:00:00Z"),
            ],
            None,
        ),
        (
            [
                ("labeled", "test-change-approved", "2026-11-20T10:00:00Z"),
                ("unlabeled", "test-change-approved", "2026-11-20T11:00:00Z"),
                ("labeled", "chore", "2026-11-20T11:30:00Z"),
                ("labeled", "test-change-approved", "2026-11-20T12:00:00Z"),
            ],
            "2026-11-20T12:00:00Z",
        ),
    ],
)
def test_the_label_time_is_the_last_time_it_was_set(check, monkeypatch, rows, expected):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=events(*rows))

    monkeypatch.setattr(check.subprocess, "run", run)
    applied = check.label_applied("o/r", "7", "test-change-approved")
    assert applied == (when(expected) if expected else None)
    assert calls[0][:4] == ["gh", "api", "--paginate", "repos/o/r/issues/7/events"]


def test_an_unknown_base_fails_the_check(check, repo, capsys):
    assert check.main(["--base", "no-such-branch"]) == 2
    assert "could not run" in capsys.readouterr().out
