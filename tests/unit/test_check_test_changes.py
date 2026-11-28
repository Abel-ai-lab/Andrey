"""The check of test changes flags each skipped, loosened, or removed test and passes the rest."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PATH = "tests/unit/test_example.py"


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location(
        "check_test_changes", ROOT / "scripts/check_test_changes.py"
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
    "parametrize case replaced by another": (
        """
        @pytest.mark.parametrize("kind", ["edge", "common"])
        def test_fit(kind):
            assert fit(kind)
        """,
        """
        @pytest.mark.parametrize("kind", ["common", "other"])
        def test_fit(kind):
            assert fit(kind)
        """,
        "1 parametrize case(s) removed",
    ),
    "places of assertAlmostEqual left to the default": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, places=2)
        """,
        "places loosened from 7 to 2",
    ),
    "skip through a module alias": (
        """
        SKIP = pytest.mark.skip(reason="slow")

        @SKIP
        def test_other():
            assert other()

        def test_fit():
            assert fit()
        """,
        """
        SKIP = pytest.mark.skip(reason="slow")

        @SKIP
        def test_other():
            assert other()

        @SKIP
        def test_fit():
            assert fit()
        """,
        "pytest.mark.skip added to a test that existed",
    ),
    "skip through an imported mark": (
        """
        from pytest import mark

        def test_fit():
            assert fit()
        """,
        """
        from pytest import mark

        @mark.skip
        def test_fit():
            assert fit()
        """,
        "pytest.mark.skip added to a test that existed",
    ),
    "pytest.fail removed": (
        """
        def test_fit():
            if reached():
                pytest.fail("unreachable")
            assert fit()
        """,
        """
        def test_fit():
            assert fit()
        """,
        "assertion removed",
    ),
    "raises accepts one more exception": (
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises((ValueError, TypeError)):
                fit(None)
        """,
        "pytest.raises accepts TypeError as well",
    ),
    "raises in a tuple widened to Exception": (
        """
        def test_fit():
            with pytest.raises((ValueError, TypeError)):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises((ValueError, TypeError, Exception)):
                fit(None)
        """,
        "pytest.raises accepts Exception as well",
    ),
    "parametrized fixture deleted": (
        """
        @pytest.fixture(params=["a", "b", "c"])
        def kind(request):
            return request.param

        def test_fit(kind):
            assert fit(kind)
        """,
        """
        def test_fit():
            assert fit("a")
        """,
        "parametrized fixture deleted",
    ),
    "test deleted from a class pytest collects": (
        """
        class TestFit:
            def test_a(self):
                assert a()

            def test_b(self):
                assert b()
        """,
        """
        class TestFit:
            def test_a(self):
                assert a()
        """,
        "test deleted",
    ),
    "test deleted from a TestCase": (
        """
        class Fits(unittest.TestCase):
            def test_a(self):
                self.assertTrue(a())

            def test_b(self):
                self.assertTrue(b())
        """,
        """
        class Fits(unittest.TestCase):
            def test_a(self):
                self.assertTrue(a())
        """,
        "test deleted",
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
    "numeric parametrize case replaced": (
        """
        @pytest.mark.parametrize("n", [0, 1])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [0, 2])
        def test_fit(n):
            assert fit(n)
        """,
        "1 parametrize case(s) removed",
    ),
    "range of parametrize cases shifted": (
        """
        @pytest.mark.parametrize("seed", range(3))
        def test_fit(seed):
            assert fit(seed)
        """,
        """
        @pytest.mark.parametrize("seed", range(1, 4))
        def test_fit(seed):
            assert fit(seed)
        """,
        "1 parametrize case(s) removed",
    ),
    "math.isclose loosened from its default": (
        """
        def test_fit():
            assert math.isclose(a, b)
        """,
        """
        def test_fit():
            assert math.isclose(a, b, rel_tol=0.5)
        """,
        "rel_tol loosened",
    ),
    "raises match set to None": (
        """
        def test_fit():
            with pytest.raises(ValueError, match="specific"):
                fit()
        """,
        """
        def test_fit():
            with pytest.raises(ValueError, match=None):
                fit()
        """,
        "pytest.raises lost its match",
    ),
    "skipif condition made true": (
        """
        @pytest.mark.skipif(False, reason="slow")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif(True, reason="slow")
        def test_fit():
            assert fit()
        """,
        "skipif condition made always true",
    ),
    "skipif condition replaced by true": (
        """
        @pytest.mark.skipif(sys.platform == "win32", reason="windows")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif(condition=True, reason="windows")
        def test_fit():
            assert fit()
        """,
        "skipif condition made always true",
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
    "delta replaces the default places": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=1)
        """,
        "places loosened from 7 to delta=1",
    ),
    "delta replaces explicit places": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, places=3)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=0.01)
        """,
        "places loosened from 3 to delta=0.01",
    ),
    "delta enlarged": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=0.01)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=1)
        """,
        "places loosened from delta=0.01 to delta=1",
    ),
    "test deleted from a TestCase through a base class": (
        """
        class Base(unittest.TestCase):
            pass

        class Fits(Base):
            def test_a(self):
                assert a()

            def test_b(self):
                assert b()
        """,
        """
        class Base(unittest.TestCase):
            pass

        class Fits(Base):
            def test_a(self):
                assert a()
        """,
        "test deleted",
    ),
    "test deleted from a TestCase imported under another name": (
        """
        from unittest import TestCase as Case

        class Fits(Case):
            def test_a(self):
                assert a()

            def test_b(self):
                assert b()
        """,
        """
        from unittest import TestCase as Case

        class Fits(Case):
            def test_a(self):
                assert a()
        """,
        "test deleted",
    ),
    "test deleted from a class with a base imported from another module": (
        """
        from helpers import Base

        class Fits(Base):
            def test_a(self):
                assert a()

            def test_b(self):
                assert b()
        """,
        """
        from helpers import Base

        class Fits(Base):
            def test_a(self):
                assert a()
        """,
        "test deleted",
    ),
    "skip called by an imported name": (
        """
        from pytest import skip

        def test_fit():
            assert fit()
        """,
        """
        from pytest import skip

        def test_fit():
            skip("disabled")
            assert fit()
        """,
        "pytest.skip() added",
    ),
    "skip called through an alias of pytest": (
        """
        import pytest as pt

        def test_fit():
            assert fit()
        """,
        """
        import pytest as pt

        def test_fit():
            pt.skip("disabled")
            assert fit()
        """,
        "pytest.skip() added",
    ),
    "skip marker through an alias of mark": (
        """
        from pytest import mark as m

        def test_fit():
            assert fit()
        """,
        """
        from pytest import mark as m

        @m.skip
        def test_fit():
            assert fit()
        """,
        "pytest.mark.skip added",
    ),
    "pytest.fail removed through an imported name": (
        """
        from pytest import fail

        def test_fit():
            if not fit():
                fail("no fit")
        """,
        """
        from pytest import fail

        def test_fit():
            fit()
        """,
        "assertion removed",
    ),
    "raises match turned into the empty string": (
        """
        def test_fit():
            with pytest.raises(ValueError, match="specific"):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(ValueError, match=""):
                fit(None)
        """,
        "pytest.raises lost its match",
    ),
    "raises match turned into a pattern that matches everything": (
        """
        def test_fit():
            with pytest.raises(ValueError, match="specific"):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(ValueError, match=".*"):
                fit(None)
        """,
        "pytest.raises lost its match",
    ),
    "raises tuple member replaced by Exception": (
        """
        def test_fit():
            with pytest.raises((ValueError, TypeError)):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises((ValueError, Exception)):
                fit(None)
        """,
        "pytest.raises widened to Exception",
    ),
    "xfail made non-strict": (
        """
        @pytest.mark.xfail(strict=True)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(strict=False)
        def test_fit():
            assert fit()
        """,
        "xfail made non-strict",
    ),
    "xfail without its strict": (
        """
        @pytest.mark.xfail(reason="slow", strict=True)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(reason="slow")
        def test_fit():
            assert fit()
        """,
        "xfail made non-strict",
    ),
    "xfail without its raises": (
        """
        @pytest.mark.xfail(raises=ValueError)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail
        def test_fit():
            assert fit()
        """,
        "xfail lost its raises",
    ),
    "self.fail removed": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                if fit() is None:
                    self.fail("unexpected path")
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                fit()
        """,
        "assertion removed",
    ),
    "xfail raises widened to more exceptions": (
        """
        @pytest.mark.xfail(raises=ValueError)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(raises=(ValueError, TypeError))
        def test_fit():
            assert fit()
        """,
        "xfail accepts TypeError as well",
    ),
    "xfail raises widened to Exception": (
        """
        @pytest.mark.xfail(raises=ValueError)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(raises=Exception)
        def test_fit():
            assert fit()
        """,
        "xfail widened to Exception",
    ),
    "xfail raises turned into None": (
        """
        @pytest.mark.xfail(raises=ValueError)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(raises=None)
        def test_fit():
            assert fit()
        """,
        "xfail lost its raises",
    ),
    "math.isclose through an alias of math": (
        """
        import math as m

        def test_fit():
            assert m.isclose(a, b)
        """,
        """
        import math as m

        def test_fit():
            assert m.isclose(a, b, rel_tol=0.5)
        """,
        "rel_tol loosened from 1e-09 to 0.5",
    ),
    "math.isclose imported by name": (
        """
        from math import isclose

        def test_fit():
            assert isclose(a, b)
        """,
        """
        from math import isclose

        def test_fit():
            assert isclose(a, b, rel_tol=0.5)
        """,
        "rel_tol loosened from 1e-09 to 0.5",
    ),
    "parametrize with keyword arguments": (
        """
        @pytest.mark.parametrize(argnames="x", argvalues=[1, 2, 3])
        def test_fit(x):
            assert fit(x)
        """,
        """
        @pytest.mark.parametrize(argnames="x", argvalues=[1])
        def test_fit(x):
            assert fit(x)
        """,
        "2 parametrize case(s) removed",
    ),
    "underscored assertion helper removed": (
        """
        def test_fit():
            _assert_same(a, b)
            _assert_order_free(c)
        """,
        """
        def test_fit():
            _assert_same(a, b)
        """,
        "assertion removed",
    ),
    "unittest skip added": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        class TestFit(unittest.TestCase):
            @unittest.skip("off")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        "unittest.skip added",
    ),
    "unittest expectedFailure added": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        class TestFit(unittest.TestCase):
            @unittest.expectedFailure
            def test_fit(self):
                self.assertTrue(fit())
        """,
        "unittest.expectedFailure added",
    ),
    "self.skipTest added": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.skipTest("off")
                self.assertTrue(fit())
        """,
        "self.skipTest() added",
    ),
    "unpacked tolerance constants": (
        """
        RTOL, ATOL = 1e-6, 1e-12
        """,
        """
        RTOL, ATOL = 0.5, 1e-12
        """,
        "RTOL loosened from 1e-06 to 0.5",
    ),
    "chained tolerance constants": (
        """
        RTOL = TOL = 1e-6
        """,
        """
        RTOL = TOL = 1e-2
        """,
        "loosened from 1e-06 to 0.01",
    ),
    "raises removed through an aliased import": (
        """
        from pytest import raises as expect

        def test_fit():
            with expect(ValueError):
                fit(None)
            with expect(TypeError):
                fit(1)
        """,
        """
        from pytest import raises as expect

        def test_fit():
            with expect(ValueError):
                fit(None)
        """,
        "pytest.raises check removed",
    ),
    "raises removed through a module alias": (
        """
        import pytest as pt

        def test_fit():
            with pt.raises(ValueError):
                fit(None)
            with pt.raises(TypeError):
                fit(1)
        """,
        """
        import pytest as pt

        def test_fit():
            with pt.raises(ValueError):
                fit(None)
        """,
        "pytest.raises check removed",
    ),
    "assertion helper removed through an aliased import": (
        """
        from numpy.testing import assert_allclose as close

        def test_fit():
            close(a, b)
            close(c, d)
        """,
        """
        from numpy.testing import assert_allclose as close

        def test_fit():
            close(a, b)
        """,
        "assertion removed",
    ),
    "unittest skipIf condition made true": (
        """
        import unittest

        class TestFit(unittest.TestCase):
            @unittest.skipIf(False, "never")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        import unittest

        class TestFit(unittest.TestCase):
            @unittest.skipIf(True, "never")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        "skipIf condition made always true",
    ),
    "unittest skipUnless condition made false": (
        """
        from unittest import TestCase, skipUnless

        class TestFit(TestCase):
            @skipUnless(True, "needs it")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        from unittest import TestCase, skipUnless

        class TestFit(TestCase):
            @skipUnless(False, "needs it")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        "skipUnless condition made always false",
    ),
    "class disabled with __test__": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            __test__ = False

            def test_fit(self):
                assert fit()
        """,
        "test no longer collected",
    ),
    "unittest class stops deriving from TestCase": (
        """
        import unittest

        class FitTest(unittest.TestCase):
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        import unittest

        class FitTest:
            def test_fit(self):
                self.assertTrue(fit())
        """,
        "test no longer collected",
    ),
    "approx rel set to None": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=1e-9)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=None)
        """,
        "rel loosened from 1e-09 to 1e-06",
    ),
    "approx abs set to None": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, abs=1e-14)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, abs=None)
        """,
        "rel loosened from 0 to 1e-06",
    ),
    "raises lost its check": (
        """
        def test_fit():
            with pytest.raises(OSError, check=lambda e: e.errno == errno.EACCES):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(OSError):
                fit(None)
        """,
        "pytest.raises lost its check",
    ),
    "parametrized case changed along with its id": (
        """
        @pytest.mark.parametrize("n", [pytest.param(1, id="one"), pytest.param(2, id="two")])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [pytest.param(1, id="one"), pytest.param(3, id="two")])
        def test_fit(n):
            assert fit(n)
        """,
        "1 parametrize case(s) removed",
    ),
    "warning filter relaxed to ignore": (
        """
        @pytest.mark.filterwarnings("error::RuntimeWarning")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.filterwarnings("ignore::RuntimeWarning")
        def test_fit():
            assert fit()
        """,
        "error filter removed or changed",
    ),
    "warning filter that errors removed": (
        """
        @pytest.mark.filterwarnings("error")
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()
        """,
        "error filter removed or changed",
    ),
    "warning ignore filter added": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.filterwarnings("ignore::RuntimeWarning")
        def test_fit():
            assert fit()
        """,
        "filter that lets a warning pass added",
    ),
    "warning default filter added through a module name": (
        """
        from pytest import mark

        def test_fit():
            assert fit()
        """,
        """
        from pytest import mark

        @mark.filterwarnings("default")
        def test_fit():
            assert fit()
        """,
        "filter that lets a warning pass added",
    ),
    "plain test class gains a constructor": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            def __init__(self):
                self.x = 1

            def test_fit(self):
                assert fit()
        """,
        "test no longer collected",
    ),
    "approx nan_ok made true": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=False)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=True)
        """,
        "nan_ok loosened from False to True",
    ),
    "approx nan_ok added": (
        """
        def test_fit():
            assert score == pytest.approx(1.0)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=True)
        """,
        "nan_ok loosened from False to True",
    ),
    "equal_nan left to its default of true": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b, equal_nan=False)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b)
        """,
        "equal_nan loosened from False to True",
    ),
    "warns widened to Warning": (
        """
        def test_fit():
            with pytest.warns(DeprecationWarning):
                fit()
        """,
        """
        def test_fit():
            with pytest.warns(Warning):
                fit()
        """,
        "widened to Warning",
    ),
    "assertRaises widened": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(ValueError):
                    fit(None)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(Exception):
                    fit(None)
        """,
        "widened to Exception",
    ),
    "assertRaisesRegex lost its pattern": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    fit(None)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(ValueError):
                    fit(None)
        """,
        "lost its match",
    ),
    "assertRaises removed": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(ValueError):
                    fit(None)
                with self.assertRaises(TypeError):
                    fit(1)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(ValueError):
                    fit(None)
        """,
        "pytest.raises check removed",
    ),
    "xfail condition made true": (
        """
        @pytest.mark.xfail(False, reason="flaky")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(True, reason="flaky")
        def test_fit():
            assert fit()
        """,
        "xfail condition made always true",
    ),
    "xfail condition dropped": (
        """
        @pytest.mark.xfail(condition=False, reason="flaky")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(reason="flaky")
        def test_fit():
            assert fit()
        """,
        "xfail condition made always true",
    ),
    "test deleted from a helper class that a test class inherits": (
        """
        class Helper:
            def test_inherited(self):
                assert fit()

        class TestChild(Helper):
            pass
        """,
        """
        class Helper:
            pass

        class TestChild(Helper):
            pass
        """,
        "test deleted",
    ),
    "__test__ = False set on a module": (
        """
        def test_fit():
            assert fit()
        """,
        """
        __test__ = False

        def test_fit():
            assert fit()
        """,
        "__test__ = False added",
    ),
    "__test__ = False set on a function": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()

        test_fit.__test__ = False
        """,
        "__test__ = False added",
    ),
    "__test__ set to False with setattr": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()

        setattr(test_fit, "__test__", False)
        """,
        "__test__ = False added",
    ),
    "string skipif condition made true": (
        """
        @pytest.mark.skipif("sys.platform == 'win32'", reason="windows")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif(True, reason="windows")
        def test_fit():
            assert fit()
        """,
        "skipif condition made always true",
    ),
    "string skipif condition made the string True": (
        """
        @pytest.mark.skipif("sys.platform == 'win32'", reason="windows")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif("True", reason="windows")
        def test_fit():
            assert fit()
        """,
        "skipif condition made always true",
    ),
    "string xfail condition made true": (
        """
        @pytest.mark.xfail("sys.platform == 'win32'", reason="flaky")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(condition=True, reason="flaky")
        def test_fit():
            assert fit()
        """,
        "xfail condition made always true",
    ),
    "raises widened from Exception to BaseException": (
        """
        def test_fit():
            with pytest.raises(Exception):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(BaseException):
                fit(None)
        """,
        "widened to BaseException",
    ),
    "xfail raises widened from Exception to BaseException": (
        """
        @pytest.mark.xfail(raises=Exception, strict=True)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(raises=BaseException, strict=True)
        def test_fit():
            assert fit()
        """,
        "xfail widened to BaseException",
    ),
    "test made a fixture": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.fixture
        def test_fit():
            assert fit()
        """,
        "test no longer collected: it became a fixture",
    ),
    "test made a fixture with arguments, imported by name": (
        """
        def test_fit():
            assert fit()
        """,
        """
        from pytest import fixture

        @fixture(scope="module")
        def test_fit():
            assert fit()
        """,
        "test no longer collected: it became a fixture",
    ),
    "empty parametrization added": (
        """
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [])
        def test_fit(n):
            assert fit(n)
        """,
        "parametrization with no case added",
    ),
    "parametrization over an empty range added": (
        """
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", range(0))
        def test_fit(n):
            assert fit(n)
        """,
        "parametrization with no case added",
    ),
    "parametrization with cases not stated made empty": (
        """
        @pytest.mark.parametrize("n", CASES)
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", ())
        def test_fit(n):
            assert fit(n)
        """,
        "parametrization with no case added",
    ),
    "fixture with no params added": (
        """
        @pytest.fixture
        def data():
            return load()
        """,
        """
        @pytest.fixture(params=[])
        def data(request):
            return load(request.param)
        """,
        "parametrization with no case added",
    ),
    "xfail made not to run": (
        """
        @pytest.mark.xfail(reason="flaky", strict=True)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(reason="flaky", strict=True, run=False)
        def test_fit():
            assert fit()
        """,
        "xfail made not to run the test",
    ),
    "setup_module that skips added": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def setup_module():
            pytest.skip("no gpu")

        def test_fit():
            assert fit()
        """,
        "setup_module that skips tests added",
    ),
    "setup_function that skips added": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def setup_function(function):
            pytest.importorskip("torch")

        def test_fit():
            assert fit()
        """,
        "setup_function that skips tests added",
    ),
    "setup_method that skips added to a class": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            def setup_method(self, method):
                pytest.skip("slow")

            def test_fit(self):
                assert fit()
        """,
        "setup_method that skips tests added",
    ),
    "skip added to a setup_module that existed": (
        """
        def setup_module():
            seed()
        """,
        """
        def setup_module():
            pytest.skip("slow")
        """,
        "pytest.skip() added",
    ),
}


@pytest.mark.parametrize("name", FLAGGED)
def test_each_change_that_skips_loosens_or_removes_a_test_is_flagged(check, name):
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
    "parametrize cases reordered": (
        """
        @pytest.mark.parametrize("kind", ["edge", "common"])
        def test_fit(kind):
            assert fit(kind)
        """,
        """
        @pytest.mark.parametrize("kind", ["common", "edge"])
        def test_fit(kind):
            assert fit(kind)
        """,
    ),
    "raises exception swapped": (
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(TypeError):
                fit(None)
        """,
    ),
    "raises exceptions narrowed": (
        """
        def test_fit():
            with pytest.raises((ValueError, TypeError)):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(ValueError):
                fit(None)
        """,
    ),
    "method named test of a class pytest does not collect": (
        """
        class Helper:
            def test_value(self):
                return 1

            def other(self):
                return 2
        """,
        """
        class Helper:
            def other(self):
                return 2
        """,
    ),
    "skip alias used by an added test only": (
        """
        def test_fit():
            assert fit()
        """,
        """
        SKIP = pytest.mark.skip(reason="slow")

        def test_fit():
            assert fit()

        @SKIP
        def test_new():
            assert new()
        """,
    ),
    "skip alias used as before": (
        """
        SKIP = pytest.mark.skip(reason="slow")

        @SKIP
        def test_fit():
            assert fit()
        """,
        """
        SKIP = pytest.mark.skip(reason="slow")

        @SKIP
        def test_fit():
            assert fit()
            assert fit() == 1
        """,
    ),
    "pytest.fail kept": (
        """
        def test_fit():
            if reached():
                pytest.fail("unreachable")
        """,
        """
        def test_fit():
            if reached():
                pytest.fail("never")
            assert fit()
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
    "numeric parametrize case added": (
        """
        @pytest.mark.parametrize("n", [0, 1])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [0, 1, 2])
        def test_fit(n):
            assert fit(n)
        """,
    ),
    "range of parametrize cases written out": (
        """
        @pytest.mark.parametrize("seed", range(3))
        def test_fit(seed):
            assert fit(seed)
        """,
        """
        @pytest.mark.parametrize("seed", [0, 1, 2.0])
        def test_fit(seed):
            assert fit(seed)
        """,
    ),
    "math.isclose tightened": (
        """
        def test_fit():
            assert math.isclose(a, b, rel_tol=1e-6)
        """,
        """
        def test_fit():
            assert math.isclose(a, b)
        """,
    ),
    "raises match kept": (
        """
        def test_fit():
            with pytest.raises(ValueError, match="specific"):
                fit()
        """,
        """
        def test_fit():
            with pytest.raises(ValueError, match="more specific"):
                fit()
        """,
    ),
    "skipif condition kept false": (
        """
        @pytest.mark.skipif(False, reason="slow")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif(sys.platform == "win32", reason="slow")
        def test_fit():
            assert fit()
        """,
    ),
    "delta tightened": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=1)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=0.01)
        """,
    ),
    "delta tighter than the places it replaces": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, places=2)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=0.001)
        """,
    ),
    "delta of None is no delta": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertAlmostEqual(a, b, delta=None)
        """,
    ),
    "method named test of a class that derives from a plain class": (
        """
        class Base:
            pass

        class Helper(Base):
            def test_value(self):
                return 1

            def other(self):
                return 2
        """,
        """
        class Base:
            pass

        class Helper(Base):
            def other(self):
                return 2
        """,
    ),
    "skip of a local function named skip": (
        """
        def skip(reason):
            return reason

        def test_fit():
            assert fit()
        """,
        """
        def skip(reason):
            return reason

        def test_fit():
            skip("not pytest's")
            assert fit()
        """,
    ),
    "skip imported from another module": (
        """
        from helpers import skip

        def test_fit():
            assert fit()
        """,
        """
        from helpers import skip

        def test_fit():
            skip("not pytest's")
            assert fit()
        """,
    ),
    "raises match changed to another pattern": (
        """
        def test_fit():
            with pytest.raises(ValueError, match="a|b"):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(ValueError, match="a|c"):
                fit(None)
        """,
    ),
    "raises tuple member replaced by another exception": (
        """
        def test_fit():
            with pytest.raises((ValueError, TypeError)):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises((ValueError, KeyError)):
                fit(None)
        """,
    ),
    "raises tuple that was broad already": (
        """
        def test_fit():
            with pytest.raises((ValueError, Exception)):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises((TypeError, Exception)):
                fit(None)
        """,
    ),
    "xfail made strict": (
        """
        @pytest.mark.xfail(strict=False)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(strict=True)
        def test_fit():
            assert fit()
        """,
    ),
    "xfail that stays strict": (
        """
        @pytest.mark.xfail(strict=True, raises=ValueError)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(strict=True, raises=ValueError, reason="slow")
        def test_fit():
            assert fit()
        """,
    ),
    "self.fail added": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                fit()
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                if fit() is None:
                    self.fail("unexpected path")
        """,
    ),
    "fail of an object that is not the test": (
        """
        def test_fit():
            report.fail("a")
            assert fit()
        """,
        """
        def test_fit():
            assert fit()
        """,
    ),
    "xfail raises narrowed": (
        """
        @pytest.mark.xfail(raises=(ValueError, TypeError))
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(raises=ValueError)
        def test_fit():
            assert fit()
        """,
    ),
    "math.isclose through an alias tightened": (
        """
        import math as m

        def test_fit():
            assert m.isclose(a, b, rel_tol=1e-3)
        """,
        """
        import math as m

        def test_fit():
            assert m.isclose(a, b)
        """,
    ),
    "parametrize with keyword arguments and more cases": (
        """
        @pytest.mark.parametrize(argnames="x", argvalues=[1])
        def test_fit(x):
            assert fit(x)
        """,
        """
        @pytest.mark.parametrize(argnames="x", argvalues=[1, 2])
        def test_fit(x):
            assert fit(x)
        """,
    ),
    "underscored assertion helper added": (
        """
        def test_fit():
            _assert_same(a, b)
        """,
        """
        def test_fit():
            _assert_same(a, b)
            _assert_order_free(c)
        """,
    ),
    "unpacked tolerance constants tightened": (
        """
        RTOL, ATOL = 0.5, 1e-12
        """,
        """
        RTOL, ATOL = 1e-6, 1e-12
        """,
    ),
    "raises through an import that is not pytest's": (
        """
        from tools import expect

        def test_fit():
            expect(fit(None))
            expect(fit(1))
        """,
        """
        from tools import expect

        def test_fit():
            expect(fit(None))
        """,
    ),
    "aliased raises kept": (
        """
        from pytest import raises as expect

        def test_fit():
            with expect(ValueError, match="alpha"):
                fit(None)
        """,
        """
        from pytest import raises as expect

        def test_fit():
            with expect(ValueError, match="alpha"):
                fit(None)
            with expect(TypeError):
                fit(1)
        """,
    ),
    "unittest skipIf condition made false": (
        """
        import unittest

        class TestFit(unittest.TestCase):
            @unittest.skipIf(True, "never")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        import unittest

        class TestFit(unittest.TestCase):
            @unittest.skipIf(False, "never")
            def test_fit(self):
                self.assertTrue(fit())
        """,
    ),
    "unittest skipUnless condition made true": (
        """
        from unittest import TestCase, skipUnless

        class TestFit(TestCase):
            @skipUnless(False, "needs it")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        from unittest import TestCase, skipUnless

        class TestFit(TestCase):
            @skipUnless(True, "needs it")
            def test_fit(self):
                self.assertTrue(fit())
        """,
    ),
    "unittest skip condition not a literal": (
        """
        import sys
        from unittest import TestCase, skipIf

        class TestFit(TestCase):
            @skipIf(sys.platform == "win32", "posix only")
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        import sys
        from unittest import TestCase, skipIf

        class TestFit(TestCase):
            @skipIf(sys.platform == "darwin", "posix only")
            def test_fit(self):
                self.assertTrue(fit())
        """,
    ),
    "class that opts into collection with __test__": (
        """
        class TestFit:
            __test__ = False

            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
    ),
    "class that stays disabled with __test__": (
        """
        class TestFit:
            __test__ = False

            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            __test__ = False

            def test_fit(self):
                assert fit()
                assert fit() > 0
        """,
    ),
    "__test__ set to True": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            __test__ = True

            def test_fit(self):
                assert fit()
        """,
    ),
    "approx rel tightened from None": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=None)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=1e-9)
        """,
    ),
    "approx rel from a value to None that is the default": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=1e-3)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=None)
        """,
    ),
    "approx rel None unchanged": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=None, abs=1e-3)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, rel=None, abs=1e-3)
        """,
    ),
    "raises check kept": (
        """
        def test_fit():
            with pytest.raises(OSError, check=lambda e: e.errno == errno.EACCES):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(OSError, check=lambda e: e.errno == errno.EACCES):
                fit(None)
            with pytest.raises(ValueError):
                fit(1)
        """,
    ),
    "raises check added": (
        """
        def test_fit():
            with pytest.raises(OSError):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(OSError, check=lambda e: e.errno == errno.EACCES):
                fit(None)
        """,
    ),
    "parametrized case renamed by its id": (
        """
        @pytest.mark.parametrize("n", [pytest.param(1, id="old"), pytest.param(2)])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [pytest.param(1, id="new"), pytest.param(2, id="two")])
        def test_fit(n):
            assert fit(n)
        """,
    ),
    "warning filter tightened to error": (
        """
        @pytest.mark.filterwarnings("ignore::RuntimeWarning")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.filterwarnings("error::RuntimeWarning")
        def test_fit():
            assert fit()
        """,
    ),
    "warning error filter added": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.filterwarnings("error")
        def test_fit():
            assert fit()
        """,
    ),
    "warning filters reordered": (
        """
        @pytest.mark.filterwarnings("error::RuntimeWarning")
        @pytest.mark.filterwarnings("ignore::UserWarning")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.filterwarnings("ignore::UserWarning")
        @pytest.mark.filterwarnings("error::RuntimeWarning")
        def test_fit():
            assert fit()
        """,
    ),
    "warning filter on an added test": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()

        @pytest.mark.filterwarnings("ignore")
        def test_new():
            assert fit()
        """,
    ),
    "TestCase class gains a constructor": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                self.assertTrue(fit())
        """,
        """
        class TestFit(unittest.TestCase):
            def __init__(self, *args):
                super().__init__(*args)

            def test_fit(self):
                self.assertTrue(fit())
        """,
    ),
    "plain test class that already had a constructor": (
        """
        class TestFit:
            def __init__(self):
                self.x = 1

            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            def __init__(self):
                self.x = 2

            def test_fit(self):
                assert fit()
        """,
    ),
    "approx nan_ok made false": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=True)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=False)
        """,
    ),
    "approx nan_ok unchanged": (
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=True)
        """,
        """
        def test_fit():
            assert score == pytest.approx(1.0, nan_ok=True, rel=1e-9)
        """,
    ),
    "equal_nan made false": (
        """
        def test_fit():
            np.testing.assert_allclose(a, b)
        """,
        """
        def test_fit():
            np.testing.assert_allclose(a, b, equal_nan=False)
        """,
    ),
    "warns narrowed": (
        """
        def test_fit():
            with pytest.warns(Warning):
                fit()
        """,
        """
        def test_fit():
            with pytest.warns(DeprecationWarning):
                fit()
        """,
    ),
    "assertRaises narrowed": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(Exception):
                    fit(None)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaises(ValueError):
                    fit(None)
        """,
    ),
    "assertRaisesRegex kept": (
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    fit(None)
        """,
        """
        class TestFit(unittest.TestCase):
            def test_fit(self):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    fit(None)
                with self.assertRaises(TypeError):
                    fit(1)
        """,
    ),
    "xfail condition made false": (
        """
        @pytest.mark.xfail(True, reason="flaky")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(False, reason="flaky")
        def test_fit():
            assert fit()
        """,
    ),
    "xfail condition not a literal": (
        """
        @pytest.mark.xfail(sys.platform == "win32", reason="posix only")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(sys.platform == "darwin", reason="posix only")
        def test_fit():
            assert fit()
        """,
    ),
    "test deleted from a helper class that no test class inherits": (
        """
        class Helper:
            def test_inherited(self):
                assert fit()

        class TestChild:
            pass
        """,
        """
        class Helper:
            pass

        class TestChild:
            pass
        """,
    ),
    "__test__ = True set on a module": (
        """
        def test_fit():
            assert fit()
        """,
        """
        __test__ = True

        def test_fit():
            assert fit()
        """,
    ),
    "__test__ = False kept on a module": (
        """
        __test__ = False

        def test_fit():
            assert fit()
        """,
        """
        __test__ = False

        def test_fit():
            assert fit()
            assert fit() == 1
        """,
    ),
    "__test__ = False kept on a function": (
        """
        def test_fit():
            assert fit()

        test_fit.__test__ = False
        """,
        """
        def test_fit():
            assert fit()

        test_fit.__test__ = False
        """,
    ),
    "__test__ = False removed": (
        """
        __test__ = False

        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()
        """,
    ),
    "string skipif condition changed to another string": (
        """
        @pytest.mark.skipif("sys.platform == 'win32'", reason="windows")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif("sys.platform == 'darwin'", reason="mac")
        def test_fit():
            assert fit()
        """,
    ),
    "string skipif condition kept": (
        """
        @pytest.mark.skipif("sys.platform == 'win32'", reason="windows")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif("sys.platform == 'win32'", reason="windows")
        def test_fit():
            assert fit()
            assert fit() == 1
        """,
    ),
    "string skipif condition made False": (
        """
        @pytest.mark.skipif("sys.platform == 'win32'", reason="windows")
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.skipif("False", reason="windows")
        def test_fit():
            assert fit()
        """,
    ),
    "raises narrowed from BaseException to Exception": (
        """
        def test_fit():
            with pytest.raises(BaseException):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(Exception):
                fit(None)
        """,
    ),
    "raises kept at BaseException": (
        """
        def test_fit():
            with pytest.raises(BaseException):
                fit(None)
        """,
        """
        def test_fit():
            with pytest.raises(BaseException):
                fit(None)
            assert fit(1)
        """,
    ),
    "xfail raises narrowed from BaseException to Exception": (
        """
        @pytest.mark.xfail(raises=BaseException, strict=True)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(raises=Exception, strict=True)
        def test_fit():
            assert fit()
        """,
    ),
    "helper made a fixture": (
        """
        def make():
            return fit()

        def test_fit():
            assert fit()
        """,
        """
        @pytest.fixture
        def make():
            return fit()

        def test_fit():
            assert fit()
        """,
    ),
    "fixture added next to a test": (
        """
        def test_fit():
            assert fit()
        """,
        """
        @pytest.fixture
        def data():
            return load()

        def test_fit():
            assert fit()
        """,
    ),
    "fixture that was named like a test stays one": (
        """
        @pytest.fixture
        def test_data():
            return load()
        """,
        """
        @pytest.fixture(scope="module")
        def test_data():
            return load()
        """,
    ),
    "parametrization with a case added": (
        """
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [1, 2])
        def test_fit(n):
            assert fit(n)
        """,
    ),
    "empty parametrization kept": (
        """
        @pytest.mark.parametrize("n", [])
        def test_fit(n):
            assert fit(n)
        """,
        """
        @pytest.mark.parametrize("n", [])
        def test_fit(n):
            assert fit(n)
            assert fit(n) == 1
        """,
    ),
    "empty parametrization on a new test": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def test_fit():
            assert fit()

        @pytest.mark.parametrize("n", [])
        def test_fit_more(n):
            assert fit(n)
        """,
    ),
    "xfail made to run": (
        """
        @pytest.mark.xfail(reason="flaky", run=False)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(reason="flaky", run=True)
        def test_fit():
            assert fit()
        """,
    ),
    "xfail with run=False kept": (
        """
        @pytest.mark.xfail(reason="flaky", run=False)
        def test_fit():
            assert fit()
        """,
        """
        @pytest.mark.xfail(reason="flaky", run=False)
        def test_fit():
            assert fit()
            assert fit() == 1
        """,
    ),
    "setup_module without a skip added": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def setup_module():
            seed()

        def test_fit():
            assert fit()
        """,
    ),
    "setup_method without a skip added to a class": (
        """
        class TestFit:
            def test_fit(self):
                assert fit()
        """,
        """
        class TestFit:
            def setup_method(self, method):
                seed()

            def test_fit(self):
                assert fit()
        """,
    ),
    "setup_module that skips kept": (
        """
        def setup_module():
            pytest.skip("slow")
        """,
        """
        def setup_module():
            pytest.skip("slow")
            seed()
        """,
    ),
    "helper named like a setup that skips": (
        """
        def test_fit():
            assert fit()
        """,
        """
        def setup_data():
            pytest.skip("slow")

        def test_fit():
            assert fit()
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


def test_a_parametrized_fixture_that_moved_is_not_deleted(check):
    fixture = textwrap.dedent("""
    @pytest.fixture(params=["a", "b", "c"])
    def kind(request):
        return request.param
    """)
    test = "def test_fit(kind):\n    assert fit(kind)\n"
    a, conftest = "tests/unit/test_a.py", "tests/unit/conftest.py"
    moved = [
        check.Change(a, a, fixture + test, test),
        check.Change(conftest, conftest, None, fixture),
    ]
    assert check.analyze(moved) == []
    gone = [check.Change(a, a, fixture + test, test)]
    assert [f.reason for f in check.analyze(gone)] == ["parametrized fixture deleted"]


def test_a_test_moved_out_of_the_test_folders_is_lost(check):
    source = "def test_fit():\n    assert fit()\n"
    old, new = "tests/unit/test_fit.py", "qa/test_fit.py"
    found = check.analyze([check.Change(new, old, source, source)])
    assert [(f.path, f.unit, f.reason) for f in found] == [
        (old, "test_fit", f"test no longer collected: the file was renamed to {new}")
    ]


def test_a_parametrized_fixture_that_moved_and_lost_cases_is_flagged(check):
    fixture = textwrap.dedent("""
    @pytest.fixture(params=["a", "b", "c"])
    def kind(request):
        return request.param
    """)
    fewer = fixture.replace('["a", "b", "c"]', '["a"]')
    test = "def test_fit(kind):\n    assert fit(kind)\n"
    a, conftest = "tests/unit/test_a.py", "tests/unit/conftest.py"
    moved = [
        check.Change(a, a, fixture + test, test),
        check.Change(conftest, conftest, None, fewer),
    ]
    assert [f.reason for f in check.analyze(moved)] == ["2 parametrize case(s) removed"]


def test_a_parametrized_fixture_that_moved_with_its_cases_is_not_flagged(check):
    fixture = textwrap.dedent("""
    @pytest.fixture(params=["a", "b", "c"])
    def kind(request):
        return request.param
    """)
    more = fixture.replace('["a", "b", "c"]', '["a", "b", "c", "d"]')
    test = "def test_fit(kind):\n    assert fit(kind)\n"
    a, conftest = "tests/unit/test_a.py", "tests/unit/conftest.py"
    moved = [
        check.Change(a, a, fixture + test, test),
        check.Change(conftest, conftest, None, more),
    ]
    assert check.analyze(moved) == []


FIT_ONCE = "def test_fit():\n    assert fit('a')\n"


def test_a_function_of_the_same_name_does_not_hide_a_deleted_fixture(check):
    fixture = textwrap.dedent("""
    @pytest.fixture(params=["a", "b", "c"])
    def kind(request):
        return request.param
    """)
    a, helpers = "tests/unit/test_a.py", "tests/unit/test_helpers.py"
    changes = [
        check.Change(a, a, fixture + "def test_fit(kind):\n    assert fit(kind)\n", FIT_ONCE),
        check.Change(helpers, helpers, "x = 1\n", "def kind():\n    return 1\n"),
    ]
    assert [f.reason for f in check.analyze(changes)] == ["parametrized fixture deleted"]


def test_a_fixture_that_moved_without_its_params_is_flagged(check):
    fixture = textwrap.dedent("""
    @pytest.fixture(params=["a", "b", "c"])
    def kind(request):
        return request.param
    """)
    plain = "@pytest.fixture\ndef kind(request):\n    return 'a'\n"
    a, conftest = "tests/unit/test_a.py", "tests/unit/conftest.py"
    changes = [
        check.Change(a, a, fixture + "def test_fit(kind):\n    assert fit(kind)\n", FIT_ONCE),
        check.Change(conftest, conftest, None, plain),
    ]
    assert [f.reason for f in check.analyze(changes)] == ["parametrized fixture deleted"]


def test_a_test_moved_into_a_file_pytest_does_not_collect_is_lost(check):
    source = "def test_fit():\n    assert fit()\n"
    a, helpers = "tests/unit/test_a.py", "tests/unit/helpers.py"
    changes = [check.Change(a, a, source, "x = 1\n"), check.Change(helpers, helpers, None, source)]
    assert [f.reason for f in check.analyze(changes)] == ["test deleted"]


def test_a_class_renamed_and_skipped_is_flagged(check):
    old = "class TestOld:\n    def test_fit(self):\n        assert fit()\n"
    skipped = "@pytest.mark.skip\n" + old.replace("TestOld", "TestNew")
    renamed = old.replace("TestOld", "TestNew")
    a = "tests/unit/test_a.py"
    found = check.analyze([check.Change(a, a, old, skipped)])
    assert [f.reason for f in found] == ["pytest.mark.skip added to a test that existed"]
    assert check.analyze([check.Change(a, a, old, renamed)]) == []


def test_a_test_moved_into_a_skipped_module_is_flagged(check):
    source = "def test_fit():\n    assert fit()\n"
    a, b = "tests/unit/test_a.py", "tests/unit/test_b.py"
    changes = [
        check.Change(a, a, source, "x = 1\n"),
        check.Change(b, b, None, "pytestmark = pytest.mark.skip\n\n" + source),
    ]
    found = check.analyze(changes)
    assert [f.reason for f in found] == ["pytest.mark.skip added to a test that existed"]


CONFTEST = "tests/unit/conftest.py"
AUTOUSE_SKIP = """
import pytest

@pytest.fixture(autouse=True)
def gpu():
    pytest.skip("no gpu")
"""


@pytest.mark.parametrize(
    "source",
    [
        AUTOUSE_SKIP,
        "import pytest\n\n@pytest.fixture(autouse=True)\n"
        'def gpu():\n    pytest.importorskip("torch")\n',
        'import pytest\n\n@pytest.fixture(scope="session", autouse=True)\n'
        "def gpu(request):\n    request.applymarker(pytest.mark.skip)\n",
        "from pytest import fixture, skip\n\n@fixture(autouse=True)\ndef gpu():\n    skip('x')\n",
    ],
)
def test_a_new_conftest_that_skips_every_test_is_flagged(check, source):
    found = reasons(check, None, source, CONFTEST)
    assert len(found) == 1 and found[0].endswith("that skips tests added"), found


HOOK = "def pytest_collection_modifyitems(items):\n    items[:] = [i for i in items if i.name]\n"


def test_a_pytest_hook_that_the_change_adds_or_edits_is_flagged(check):
    new = "def pytest_collection_modifyitems(items):\n    items[:] = items[:1]\n"
    assert reasons(check, None, HOOK, CONFTEST) == ["pytest hook added or changed"]
    assert reasons(check, "x = 1\n", "x = 1\n\n" + HOOK, CONFTEST) == [
        "pytest hook added or changed"
    ]
    assert reasons(check, HOOK, new, CONFTEST) == ["pytest hook added or changed"]


@pytest.mark.parametrize("path", [CONFTEST, "tests/unit/test_a.py"])
def test_a_skipping_autouse_fixture_added_to_an_existing_file_is_flagged(check, path):
    old = "def helper():\n    return 1\n"
    found = reasons(check, old, old + AUTOUSE_SKIP, path)
    assert found == ["autouse fixture that skips tests added"]


def test_an_existing_fixture_made_autouse_that_skips_is_flagged(check):
    old = "import pytest\n\n@pytest.fixture\ndef gpu():\n    pytest.skip('no gpu')\n"
    new = old.replace("@pytest.fixture", "@pytest.fixture(autouse=True)")
    assert reasons(check, old, new, CONFTEST) == ["autouse fixture that skips tests added"]


def test_a_skip_added_to_an_existing_autouse_fixture_is_flagged_once(check):
    old = "import pytest\n\n@pytest.fixture(autouse=True)\ndef gpu():\n    return 1\n"
    new = old.replace("return 1", "pytest.skip('no gpu')")
    assert reasons(check, old, new, CONFTEST) == ["pytest.skip() added to a test that existed"]


@pytest.mark.parametrize(
    ("old", "new", "path"),
    [
        (
            None,
            "import pytest\n\n@pytest.fixture(autouse=True)\ndef seed():\n    return 1\n",
            CONFTEST,
        ),
        (
            None,
            "import pytest\n\n@pytest.fixture\ndef gpu():\n    pytest.skip('no gpu')\n",
            CONFTEST,
        ),
        (None, AUTOUSE_SKIP, "tests/unit/test_new.py"),
        (
            "def helper():\n    return 1\n",
            "import pytest\n\n@pytest.fixture(autouse=False)\ndef gpu():\n    pytest.skip('x')\n",
            CONFTEST,
        ),
        (AUTOUSE_SKIP, AUTOUSE_SKIP, CONFTEST),
        (HOOK, HOOK, CONFTEST),
        (None, HOOK, "tests/unit/test_new.py"),
    ],
    ids=[
        "autouse without a skip",
        "skip in a fixture that is not autouse",
        "autouse skip in a new test file",
        "autouse=False",
        "unchanged autouse skip",
        "unchanged hook",
        "hook in a new test file",
    ],
)
def test_other_fixtures_and_hooks_are_not_flagged(check, old, new, path):
    assert reasons(check, old, new, path) == []


QA = "qa/validate.py"
VALIDATE = """
def helper():
    return 1

def main():
    result = run()
    report(result)
    return 0
"""

# Changes to a ``conftest.py`` or to ``qa/`` that skip or remove checks, with the file they are in.
FLAGGED_AT = {
    "collect_ignore in a new conftest": (
        None,
        'collect_ignore = ["test_slow.py"]\n',
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "collect_ignore_glob in a new conftest": (
        None,
        'collect_ignore_glob = ["*_gpu.py"]\n',
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "collect_ignore in a new conftest, annotated": (
        None,
        'collect_ignore: list[str] = ["test_slow.py"]\n',
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "collect_ignore set to a computed value": (
        "x = 1\n",
        "x = 1\ncollect_ignore = sorted(glob('*.py'))\n",
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "entry added to collect_ignore": (
        'collect_ignore = ["a.py"]\n',
        'collect_ignore = ["a.py", "b.py"]\n',
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "entry appended to collect_ignore": (
        'collect_ignore = ["a.py"]\n',
        'collect_ignore = ["a.py"]\ncollect_ignore.append("b.py")\n',
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "entries added to collect_ignore_glob with +=": (
        "collect_ignore_glob = []\n",
        'collect_ignore_glob = []\ncollect_ignore_glob += ["*_slow.py"]\n',
        CONFTEST,
        "collect_ignore or collect_ignore_glob entry added",
    ),
    "function of qa deleted": (
        VALIDATE,
        "def helper():\n    return 1\n",
        QA,
        "function of qa/ deleted",
    ),
    "function of qa emptied to pass": (
        VALIDATE,
        VALIDATE.replace("    result = run()\n    report(result)\n    return 0\n", "    pass\n"),
        QA,
        "function of qa/ emptied",
    ),
    "function of qa emptied to a docstring and a return": (
        VALIDATE,
        VALIDATE.replace(
            "    result = run()\n    report(result)\n    return 0\n",
            '    """Validate."""\n    return 0\n',
        ),
        QA,
        "function of qa/ emptied",
    ),
    "method of a class of qa deleted": (
        (
            "class Check:\n    def run(self):\n        return go()\n\n"
            "    def stop(self):\n        halt()\n"
        ),
        "class Check:\n    def run(self):\n        return go()\n",
        QA,
        "function of qa/ deleted",
    ),
    "file of qa deleted": (VALIDATE, None, QA, "function of qa/ deleted"),
}
NOT_FLAGGED_AT = {
    "empty collect_ignore in a new conftest": (None, "collect_ignore = []\n", CONFTEST),
    "collect_ignore kept": (
        'collect_ignore = ["a.py"]\n',
        'collect_ignore = ["a.py"]\nx = 1\n',
        CONFTEST,
    ),
    "collect_ignore entry removed": (
        'collect_ignore = ["a.py", "b.py"]\n',
        'collect_ignore = ["a.py"]\n',
        CONFTEST,
    ),
    "collect_ignore entries reordered": (
        'collect_ignore = ["a.py", "b.py"]\n',
        'collect_ignore = ["b.py", "a.py"]\n',
        CONFTEST,
    ),
    "collect_ignore in a test file": (
        "x = 1\n",
        'x = 1\ncollect_ignore = ["a.py"]\n',
        "tests/unit/test_a.py",
    ),
    "unrelated list in a new conftest": (None, 'skip_names = ["a.py"]\n', CONFTEST),
    "function of qa added": (VALIDATE, VALIDATE + "\ndef extra():\n    return 2\n", QA),
    "function of qa edited": (VALIDATE, VALIDATE.replace("report(result)", "log(result)"), QA),
    "function of qa renamed": (VALIDATE, VALIDATE.replace("def main", "def run_all"), QA),
    "function of qa that was empty already": (
        "def main():\n    pass\n",
        'def main():\n    """Validate."""\n',
        QA,
    ),
    "function of qa that returns more": (
        "def main():\n    return 0\n",
        "def main():\n    run()\n    return 0\n",
        QA,
    ),
    "helper of tests deleted": (
        "def helper():\n    return 1\n\ndef test_a():\n    assert helper()\n",
        "def test_a():\n    assert 1\n",
        "tests/unit/test_a.py",
    ),
    "new file of qa": (None, VALIDATE, QA),
}


@pytest.mark.parametrize("name", FLAGGED_AT)
def test_each_conftest_or_qa_change_that_skips_or_removes_checks_is_flagged(check, name):
    old, new, path, expected = FLAGGED_AT[name]
    found = reasons(check, old, new, path)
    assert any(expected in reason for reason in found), found


@pytest.mark.parametrize("name", NOT_FLAGGED_AT)
def test_other_conftest_and_qa_changes_are_not_flagged(check, name):
    old, new, path = NOT_FLAGGED_AT[name]
    assert reasons(check, old, new, path) == []


def test_a_function_of_qa_that_moved_to_another_file_is_not_flagged(check):
    helper = "def helper():\n    run()\n    return 1\n"
    changes = [
        check.Change(
            "qa/a.py",
            "qa/a.py",
            helper + "\ndef keep():\n    return 2\n",
            "def keep():\n    return 2\n",
        ),
        check.Change("qa/b.py", "qa/b.py", None, helper),
    ]
    assert check.analyze(changes) == []


def test_a_function_of_qa_that_moved_and_was_emptied_is_flagged(check):
    helper = "def helper():\n    run()\n    return 1\n"
    changes = [
        check.Change("qa/a.py", "qa/a.py", helper, "x = 1\n"),
        check.Change("qa/b.py", "qa/b.py", None, "def helper():\n    pass\n"),
    ]
    found = [f.reason for f in check.analyze(changes)]
    assert found == ["function of qa/ emptied"], found


def test_a_flag_for_a_collect_ignore_names_the_file_and_the_statement(check):
    change = check.Change(CONFTEST, CONFTEST, None, 'collect_ignore = ["test_slow.py"]\n')
    text = check.report(check.analyze([change]))
    assert f"{CONFTEST}:1: <module>: collect_ignore or collect_ignore_glob entry added" in text
    assert '+ collect_ignore = ["test_slow.py"]' in text


def test_a_marker_on_a_class_is_reported_once(check):
    methods = (
        "    def test_a(self):\n        assert a()\n\n    def test_b(self):\n        assert b()\n"
    )
    old, new = "class TestFit:\n" + methods, "@pytest.mark.skip\nclass TestFit:\n" + methods
    found = check.analyze([check.Change(PATH, PATH, old, new)])
    assert [f.reason for f in found] == ["pytest.mark.skip added to a test that existed"]


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


def test_a_python_file_renamed_to_another_kind_of_file_loses_its_tests(check, repo):
    git(repo, "mv", "tests/unit/test_c.py", "tests/unit/test_c.txt")
    commit(repo)
    changes, baselines = check.load("main", "HEAD", str(repo))
    found = check.analyze(changes)
    reason = "test no longer collected: the file was renamed to tests/unit/test_c.txt"
    assert [(f.path, f.unit, f.reason) for f in found] == [
        ("tests/unit/test_c.py", "test_c", reason)
    ]
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


def test_a_binary_baseline_that_changed_one_invalid_byte_is_flagged(check, repo):
    path = repo / "tests/recovery/baselines/PC/x.npz"
    git(repo, "checkout", "-q", "main")
    path.write_bytes(b"\x93NUMPY\x80data")
    commit(repo)
    git(repo, "checkout", "-q", "feature")
    git(repo, "merge", "-q", "--ff-only", "main")
    path.write_bytes(b"\x93NUMPY\x81data")
    commit(repo)
    changes, baselines = check.load("main", "HEAD", str(repo))
    assert [(f.path, f.reason) for f in baselines] == [
        ("tests/recovery/baselines/PC/x.npz", "baseline re-recorded")
    ]
    assert changes == []


def test_a_change_outside_tests_and_qa_is_not_read(check, repo):
    write(repo, {"src/andrey/__init__.py": "x = 3\n", "tests/unit/data.json": '{"x": 9}'})
    commit(repo)
    assert check.main(["--base", "main"]) == 0


# The check and the workflow job that runs it are guarded: a pull request cannot loosen the check
# it is judged by without a maintainer reading the change.

WORKFLOW = textwrap.dedent("""
    name: CI
    jobs:
      lint:
        steps:
          - uses: actions/checkout@v7
          - name: ty
            run: ty check
      # A required check, by its name.
      check-test-changes:
        if: github.event_name == 'pull_request'
        runs-on: ubuntu-24.04
        steps:
          - uses: actions/checkout@v7
          - name: Changes that skip, loosen, or remove a test need approval
            # Judges the pull request's tests.
            if: github.base_ref != 'main'
            run: |
              git show HEAD^1:scripts/check_test_changes.py > "$RUNNER_TEMP/check_test_changes.py"
              python -I "$RUNNER_TEMP/check_test_changes.py" --base HEAD^1
      other:
        steps:
          - run: echo hi
    """)
PYPROJECT = '[project]\nname = "x"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
CHECK = "print('the check')\n"
WORKFLOW_FILE = ".github/workflows/ci.yml"
CHECK_FILE = "scripts/check_test_changes.py"


@pytest.fixture
def guarded(repo):
    """The repository whose base branch already has the check and the step that runs it."""
    git(repo, "checkout", "-q", "main")
    write(repo, {CHECK_FILE: CHECK, WORKFLOW_FILE: WORKFLOW, "pyproject.toml": PYPROJECT})
    commit(repo)
    git(repo, "checkout", "-q", "feature")
    git(repo, "reset", "-q", "--hard", "main")
    return repo


GUARDED = {
    "the check edited": {CHECK_FILE: "import sys\nsys.exit(0)\n"},
    "the step edited": {WORKFLOW_FILE: WORKFLOW.replace("--base HEAD^1", "--base HEAD")},
    "the step renamed": {
        WORKFLOW_FILE: WORKFLOW.replace("Changes that skip, loosen, or remove a test", "Tests")
    },
    "the step condition edited": {
        WORKFLOW_FILE: WORKFLOW.replace("github.base_ref != 'main'", "false")
    },
    "the step pointed at the pull request's copy": {
        WORKFLOW_FILE: WORKFLOW.replace('"$RUNNER_TEMP/check_test_changes.py"', CHECK_FILE)
    },
    "the step removed": {WORKFLOW_FILE: WORKFLOW.replace("check_test_changes", "other")},
    "the job condition edited": {
        WORKFLOW_FILE: WORKFLOW.replace("github.event_name == 'pull_request'", "false")
    },
    "the job runner edited": {WORKFLOW_FILE: WORKFLOW.replace("ubuntu-24.04", "macos-14")},
    "the job renamed": {WORKFLOW_FILE: WORKFLOW.replace("  check-test-changes:", "  check-tests:")},
    "the job removed": {
        WORKFLOW_FILE: WORKFLOW[: WORKFLOW.index("  # A required check")]
        + WORKFLOW[WORKFLOW.index("  other:") :]
    },
}
NOT_GUARDED = {
    "an unrelated edit of the workflow": {
        WORKFLOW_FILE: WORKFLOW.replace("ty check", "ty check .")
    },
    "a reworded comment of the step": {
        WORKFLOW_FILE: WORKFLOW.replace("Judges the pull request's tests.", "Judges tests.")
    },
    "a reworded comment of the job": {
        WORKFLOW_FILE: WORKFLOW.replace("A required check, by its name.", "Required.")
    },
    "an edit of another job": {WORKFLOW_FILE: WORKFLOW.replace("echo hi", "echo hello")},
    "a job added": {WORKFLOW_FILE: WORKFLOW + "  extra:\n    steps:\n      - run: echo x\n"},
}


@pytest.mark.parametrize("name", GUARDED)
def test_a_change_to_the_check_or_its_step_needs_approval(check, guarded, capsys, name):
    write(guarded, GUARDED[name])
    commit(guarded)
    assert check.main(["--base", "main"]) == 1
    output = capsys.readouterr().out
    assert "1 flagged change(s)" in output
    assert "this check changed" in output
    assert "title=Test change needs approval::" in output


@pytest.mark.parametrize(
    "options",
    [
        'addopts = "--ignore=tests/unit/test_a.py"',
        'testpaths = ["tests/unit"]',
        'filterwarnings = ["ignore"]',
    ],
)
def test_a_change_to_the_pytest_options_that_choose_the_tests_needs_approval(
    check, guarded, capsys, options
):
    write(guarded, {"pyproject.toml": PYPROJECT.replace('testpaths = ["tests"]', options)})
    commit(guarded)
    assert check.main(["--base", "main"]) == 1
    assert "pytest options that choose or judge the tests changed" in capsys.readouterr().out


def test_other_changes_to_pyproject_need_no_approval(check, guarded):
    text = PYPROJECT.replace('name = "x"', 'name = "y"') + 'markers = ["slow: a slow test"]\n'
    write(guarded, {"pyproject.toml": text})
    commit(guarded)
    assert check.main(["--base", "main"]) == 0


def test_deleting_the_check_needs_approval(check, guarded, capsys):
    (guarded / CHECK_FILE).unlink()
    commit(guarded)
    assert check.main(["--base", "main"]) == 1
    assert f"{CHECK_FILE}: this check was deleted" in capsys.readouterr().out


@pytest.mark.parametrize("name", NOT_GUARDED)
def test_other_changes_to_the_workflow_need_no_approval(check, guarded, name):
    write(guarded, NOT_GUARDED[name])
    commit(guarded)
    assert check.main(["--base", "main"]) == 0


def test_a_change_to_the_check_passes_with_a_label_set_after_the_head_changed(
    check, guarded, monkeypatch
):
    write(guarded, GUARDED["the check edited"])
    commit(guarded)
    argv = ["--base", "main", "--pr", "7", "--since", "2026-11-20T12:00:00Z", "--repo", "o/r"]
    monkeypatch.setattr(check, "label_applied", lambda *args: when("2026-11-20T12:30:00Z"))
    assert check.main(argv) == 0
    monkeypatch.setattr(check, "label_applied", lambda *args: when("2026-11-20T11:00:00Z"))
    assert check.main(argv) == 1


def test_a_pull_request_that_adds_the_check_and_its_step_is_not_flagged(check, repo):
    write(repo, {CHECK_FILE: CHECK, WORKFLOW_FILE: WORKFLOW})
    commit(repo)
    assert check.main(["--base", "main"]) == 0


def test_the_job_of_this_repository_runs_the_check_of_the_base_branch(check):
    text = (ROOT / WORKFLOW_FILE).read_text()
    [(_, job)] = check.workflow_jobs(text)
    assert job.startswith("  check-test-changes:")
    assert 'git show HEAD^1:scripts/check_test_changes.py > "$detector"' in job
    assert 'python -I "$detector"' in job
    assert "python scripts/check_test_changes.py" not in job


def test_the_job_of_this_repository_falls_back_to_the_script_under_its_old_name(check):
    """A base from before the rename has the script as ``scripts/test_weakening.py``."""
    text = (ROOT / WORKFLOW_FILE).read_text()
    [(_, job)] = check.workflow_jobs(text)
    new, old = "HEAD^1:scripts/check_test_changes.py", "HEAD^1:scripts/test_weakening.py"
    assert 'git show HEAD^1:scripts/test_weakening.py > "$detector"' in job
    assert job.index(new) < job.index(old)


def test_the_job_of_this_repository_is_a_gated_check_that_reports_on_every_pull_request(check):
    workflow = yaml.safe_load((ROOT / WORKFLOW_FILE).read_text())
    job = workflow["jobs"]["check-test-changes"]
    assert "needs" not in job
    assert "base_ref" not in job["if"]
    assert "check-test-changes" in workflow["jobs"]["ci-gate"]["needs"]
    assert [s["if"] for s in job["steps"] if "if" in s] == ["github.base_ref != 'main'"]


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
        "::error file=tests/unit/test_a.py,line=1,"
        "title=Test change needs approval::2 assertions removed" in output
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
    assert "title=Test change needs approval::2 assertions removed" in output


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
