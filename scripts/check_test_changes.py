"""Flag a pull request that skips, loosens, or removes an existing test, for a person to approve.

A change that must make a check pass can drop an assertion, loosen a tolerance, skip the test, or
record the new output as the baseline, and each of these passes the check without making the code
right. This script compares the base and the head of ``tests/`` and ``qa/`` by their syntax trees,
function by function, and flags:

- a removed ``assert`` statement, ``assert_*``, ``pytest.fail`` or ``self.fail`` call,
  ``pytest.raises`` check, or ``pytest.approx`` assertion, and a ``pytest.raises``,
  ``pytest.warns``, or ``assertRaises`` that lost its ``match`` (or matches every message) or its
  ``check``, widened to ``Exception`` (``Warning`` for a warning), or lists more exceptions;
- a looser tolerance or bound: a larger ``rtol``, ``atol``, ``abs``, ``rel``, ``tol``, or ``delta``
  (a ``delta`` of ``assertAlmostEqual`` replaces its ``places``), a smaller ``decimal`` or
  ``places``, a looser bound in a comparison, ``nan_ok`` or ``equal_nan`` made true, an exact
  comparison turned into ``pytest.approx``, in a call, a default, a module constant, or a tolerance
  table;
- a new ``skip``, ``skipif``, or ``xfail`` marker or call, or a new ``release`` marker (a pull
  request into a release branch skips those tests), on a test that existed in the base, written out,
  through a module name bound to one, or through an alias of ``pytest``; a conditional skip
  (``skipif``, unittest's ``skipIf`` and ``skipUnless``) or ``xfail`` condition made to apply
  always, and an ``xfail`` made non-strict or without its ``raises``; an autouse fixture that skips,
  and a ``pytest_*`` hook, added or changed in a ``conftest.py`` or a test file that existed; a
  ``__test__ = False``, a constructor, or a lost ``TestCase`` base that stops pytest collecting a
  class; a ``filterwarnings`` marker that stops raising a warning or lets one pass;
- a deleted test function or parametrized fixture, a test file renamed out of the names pytest
  collects (or into another kind of file), or a removed ``parametrize`` case (a literal list, tuple,
  set, or ``range``; a case that another replaced counts as removed);
- a re-recorded baseline file: a changed or deleted file of ``baselines/`` or named for one, and a
  JSON baseline whose recorded values changed;
- a change to this script or to the workflow job that runs it (see below), or to the pytest options
  of ``pyproject.toml`` that choose the tests or judge them (``addopts``, ``testpaths``,
  ``norecursedirs``, ``python_files``, ``python_classes``, ``python_functions``, ``xfail_strict``,
  ``filterwarnings``).

An added test, a tightened tolerance, and a test that only moved or was renamed are not flagged. It
cannot see an assertion rewritten to a weaker one, a tolerance passed through a variable, a check
that moved into a helper, or an existing test pointed at a new baseline file, so a pass is
necessary and not sufficient.

The script prints each flagged change with its reason and exits 1 unless the pull request carries
the label ``test-change-approved``, which a maintainer sets after reading the flags. Run it in a
checkout with the history of ``--base``; a pull request's merge commit is ``HEAD`` and its first
parent ``HEAD^1``. ``--pr`` reads the label's events from GitHub with ``gh`` (``GH_TOKEN`` and
``GITHUB_REPOSITORY`` set). The label approves only the head it was set after: one set before
``--since``, the time the head last changed, is stale, so a push withdraws the approval and a
maintainer removes the label and sets it again after reading the new flags. It exits 2 when it
cannot decide.

CI runs the copy of this script that the base branch has, never the pull request's own, so a pull
request cannot loosen the check it is judged by: it flags a change to this file, and to the
workflow job that runs it, like any other change to a test. The workflow file itself is the pull
request's, so a change to the rest of it that stops the job is for a reviewer to catch: the job is
a required status check, and ``.github/`` and this script have code owners.

    python3 scripts/check_test_changes.py --base HEAD^1 --pr 123 --since 2026-11-20T12:00:00Z
"""

from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import json
import math
import operator
import os
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from typing import Any, NamedTuple

LABEL = "test-change-approved"
ROOTS = ("tests", "qa")
# This script and the workflow whose job runs it: a change to either needs the label too.
DETECTOR = "scripts/check_test_changes.py"
WORKFLOW = ".github/workflows/ci.yml"
# The pytest options (``pyproject.toml``) that choose which tests run, and how a failure counts.
PYTEST_OPTIONS = (
    "addopts",
    "testpaths",
    "norecursedirs",
    "python_files",
    "python_classes",
    "python_functions",
    "xfail_strict",
    "filterwarnings",
)
UNITTEST_SKIPS = frozenset({"skip", "skipIf", "skipUnless", "expectedFailure"})
# The conditional skips, and the value of the condition that skips the test.
SKIP_WHEN = {
    "pytest.mark.skipif": True,
    "unittest.skipIf": True,
    "unittest.skipUnless": False,
    "pytest.mark.xfail": True,
}

LARGER_IS_LOOSER = frozenset(
    {"rtol", "atol", "rel_tol", "abs_tol", "rel", "abs", "tol", "tolerance", "eps", "epsilon"}
    | {"delta", "nulp", "maxulp"}
)
SMALLER_IS_LOOSER = frozenset({"decimal", "places", "significant", "precision"})
TOLERANCES = LARGER_IS_LOOSER | SMALLER_IS_LOOSER
# A constant or a parameter named for a tolerance: RTOL, GPU_ATOL, rtol, tol_loose.
TOLERANCE_NAME = re.compile(
    r"(?:^|_)(?:rtol|atol|tol|tols|tolerance|tolerances|eps|epsilon|delta|decimal|places)(?:$|_)",
    re.IGNORECASE,
)
# Positional tolerance arguments, by callee, and the defaults of the tolerances a call leaves out,
# read from the signatures of NumPy 2.4 and pytest 9.1.
POSITIONAL = {
    "approx": {1: "rel", 2: "abs"},
    "assert_allclose": {2: "rtol", 3: "atol"},
    "isclose": {2: "rtol", 3: "atol"},
    "allclose": {2: "rtol", 3: "atol"},
    "assert_almost_equal": {2: "decimal"},
    "assert_array_almost_equal": {2: "decimal"},
    "assert_approx_equal": {2: "significant"},
    "assert_array_almost_equal_nulp": {2: "nulp"},
    "assert_array_max_ulp": {2: "maxulp"},
    "assertAlmostEqual": {2: "places", 4: "delta"},
}
DEFAULTS = {
    "assert_allclose": {"rtol": 1e-7, "atol": 0.0},
    "isclose": {"rtol": 1e-5, "atol": 1e-8},
    "math.isclose": {"rel_tol": 1e-9, "abs_tol": 0.0},
    "allclose": {"rtol": 1e-5, "atol": 1e-8},
    "assert_almost_equal": {"decimal": 7},
    "assert_array_almost_equal": {"decimal": 6},
    "assert_approx_equal": {"significant": 7},
    "assert_array_almost_equal_nulp": {"nulp": 1},
    "assert_array_max_ulp": {"maxulp": 1},
    "assertAlmostEqual": {"places": 7},
}
APPROX_DEFAULT_REL, APPROX_DEFAULT_ABS = 1e-6, 1e-12
# The flags that make a comparison accept NaN as equal to NaN, and each callee's default.
NAN_FLAGS = {
    "nan_ok": {"approx": False},
    "equal_nan": {"assert_allclose": True, "isclose": False, "allclose": False},
}
RAISES = frozenset(
    {"raises", "warns", "deprecated_call"}
    | {"assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex"}
)
MARKERS = frozenset({"skip", "skipif", "xfail", "release"})
SKIP_CALLS = frozenset({"skip", "xfail", "importorskip"})
BROAD = frozenset({"Exception", "BaseException", "Warning"})
ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
MAX_LINES = 6
# The file names pytest collects: its default ``python_files``, which pyproject.toml leaves unset.
COLLECTED = ("test_*.py", "*_test.py")
# The folders pytest collects from, the ``testpaths`` of pyproject.toml.
TESTPATHS = ("tests",)

Flag = Callable[..., None]


class Finding(NamedTuple):
    """One flagged change: where, why, and the code before and after."""

    path: str
    line: int
    unit: str
    reason: str
    before: str = ""
    after: str = ""


class Change(NamedTuple):
    """One changed file of the pull request, with its source before and after (None if absent)."""

    path: str
    old_path: str
    old: str | None
    new: str | None


class Unit(NamedTuple):
    """A function, or the module or class code outside every function, with its syntax nodes."""

    path: str
    qualname: str
    node: ast.AST
    nodes: list[ast.AST]
    source: str
    constants: dict[str, ast.AST]
    aliases: dict[str, str] = {}
    collected: bool = True
    imports: dict[str, str] = {}
    inherited: list[ast.AST] = []  # the markers of the enclosing classes and module

    @property
    def is_function(self) -> bool:
        """Whether the unit is a function (not a module or class body)."""
        return isinstance(self.node, ast.FunctionDef | ast.AsyncFunctionDef)

    @property
    def is_test(self) -> bool:
        """Whether the unit is a function named ``test...`` that pytest collects."""
        name = self.qualname.rsplit(".", 1)[-1]
        return self.is_function and self.collected and name.startswith("test")


class Check(NamedTuple):
    """An assertion-like node: an assert, an ``assert_*`` or ``pytest.fail`` call, or a raises."""

    kind: str
    node: ast.AST
    key: str


class Knob(NamedTuple):
    """A numeric tolerance or bound, the context that identifies it, and which way is looser."""

    context: tuple
    label: str
    value: float
    larger_is_looser: bool
    strict: bool = False
    shown: str = ""  # how a report writes the value when it is not the number itself


def _callee(node: ast.AST) -> str:
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else ""


def _tolerance_callee(call: ast.Call, imports: dict[str, str]) -> str:
    """Return the callee whose tolerance knobs a call has: ``math.isclose`` apart from NumPy's."""
    if _dotted(call.func, imports) in {"math.isclose", "cmath.isclose"}:
        return "math.isclose"
    return _callee(call)


def _number(node: ast.AST | None) -> float | None:
    """Return the value of a numeric literal or of arithmetic on literals, else None."""
    if isinstance(node, ast.Constant):
        number = isinstance(node.value, int | float) and not isinstance(node.value, bool)
        return float(node.value) if number else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value = _number(node.operand)
        return value if value is None or isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in ARITHMETIC:
        left, right = _number(node.left), _number(node.right)
        if left is None or right is None:
            return None
        try:
            value = ARITHMETIC[type(node.op)](left, right)
        except (ArithmeticError, ValueError):
            return None
        return float(value) if isinstance(value, int | float) else None
    return None


class _Exact(ast.NodeTransformer):
    """Rewrite a node with every number as a float, so ``1`` and ``1.0`` are one case."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        number = _number(node)
        return node if number is None else ast.Constant(value=number, kind=None)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if _callee(node) == "param":
            # The ``id`` of a case names it in the report and changes nothing it runs.
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "id"]
        return node


def _case(node: ast.AST) -> str:
    """Return the node with its numbers kept: one case of a parametrization before and after."""
    return ast.dump(_Exact().visit(copy.deepcopy(node)))


class _Mask(ast.NodeTransformer):
    """Rewrite a node without its numbers, tolerances, and ``approx`` wrappers."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if _number(node) is not None:
            return ast.Constant(value=0, kind=None)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        # A bound that turns strict or non-strict is still the same check.
        strictness = {ast.Lt: ast.LtE, ast.Gt: ast.GtE}
        node.ops = [strictness.get(type(op), type(op))() for op in node.ops]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        name = _callee(node)
        if name == "approx" and node.args:
            return node.args[0]
        node.keywords = [
            keyword
            for keyword in node.keywords
            if keyword.arg not in TOLERANCES and keyword.arg not in NAN_FLAGS
        ]
        if name in POSITIONAL:
            node.args = node.args[: min(POSITIONAL[name])]
        return node


def _key(node: ast.AST) -> str:
    """Return the node without its numbers and tolerances: one check before and after an edit."""
    return ast.dump(_Mask().visit(copy.deepcopy(node)))


def _position(node: ast.AST) -> tuple[int, int]:
    return getattr(node, "lineno", 0), getattr(node, "col_offset", 0)


def _text(unit: Unit, node: ast.AST) -> str:
    return ast.get_source_segment(unit.source, node) or ""


def _clip(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:MAX_LINES] + (["..."] if len(lines) > MAX_LINES else []))


def _scope_nodes(root: ast.AST, definitions: list[ast.AST]) -> Iterator[ast.AST]:
    """Yield the nodes of a module or class outside its functions and nested classes."""
    stack = list(ast.iter_child_nodes(root))
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            definitions.append(node)
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _imports(tree: ast.AST) -> dict[str, str]:
    """Return what each imported name stands for: ``{"pt": "pytest", "sk": "pytest.skip"}``."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                found[alias.asname or alias.name] = f"{module}.{alias.name}"
    return found


def _dotted(node: ast.AST, imports: dict[str, str]) -> str:
    """Return the dotted name of a name or attribute chain, its first name read through imports."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    return ".".join([imports.get(node.id, node.id), *reversed(parts)])


def _marker_kind(node: ast.AST, imports: dict[str, str] | None = None) -> str:
    """Return ``pytest.mark.skip`` and the like for ``pytest.mark.skip`` or ``mark.skip``."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr in MARKERS
        and (
            (isinstance(node.value, ast.Attribute) and node.value.attr == "mark")
            or (isinstance(node.value, ast.Name) and node.value.id == "mark")
        )
    ):
        return f"pytest.mark.{node.attr}"
    dotted = _dotted(node, imports or {}).split(".")
    if len(dotted) == 3 and dotted[:2] == ["pytest", "mark"] and dotted[2] in MARKERS:
        return ".".join(dotted)
    if len(dotted) == 2 and dotted[0] == "unittest" and dotted[1] in UNITTEST_SKIPS:
        return ".".join(dotted)
    return ""


def _test_flag(node: ast.ClassDef) -> bool | None:
    """Return the ``__test__`` a class body sets to a literal ``True`` or ``False``, else None."""
    flag = None
    for statement in node.body:
        if not isinstance(statement, ast.Assign | ast.AnnAssign):
            continue
        for name, value in _assigned(statement):
            if name == "__test__" and isinstance(value, ast.Constant):
                flag = value.value if isinstance(value.value, bool) else flag
    return flag


def _constructs(node: ast.ClassDef, classes: dict[str, ast.ClassDef], seen: frozenset[str]) -> bool:
    """Whether a class, or a base in the file, defines ``__init__`` or ``__new__``."""
    seen = seen | {node.name}
    if any(
        isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
        and statement.name in {"__init__", "__new__"}
        for statement in node.body
    ):
        return True
    return any(
        isinstance(base, ast.Name)
        and base.id in classes
        and base.id not in seen
        and _constructs(classes[base.id], classes, seen)
        for base in node.bases
    )


def _collected_class(
    node: ast.ClassDef,
    classes: dict[str, ast.ClassDef],
    imports: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> bool:
    """Whether pytest collects the methods of a class.

    A class named ``Test...`` is collected, and so is one that derives from a ``TestCase``, directly
    or through classes of the file. A base imported from another module cannot be read, so it counts
    as one that may derive from a ``TestCase``. A literal ``__test__ = False`` in the body stops the
    collection of a class, and a ``True`` collects one whatever its name; a class that is not a
    ``TestCase`` and defines ``__init__`` or ``__new__`` is not collected.
    """
    seen = seen | {node.name}
    flag = _test_flag(node)
    if flag is not None:
        return flag
    unittest = any(_dotted(base, imports).endswith("TestCase") for base in node.bases)
    if node.name.startswith("Test"):
        # pytest does not collect a plain class with a constructor.
        return unittest or not _constructs(node, classes, frozenset())
    for base in node.bases:
        if _dotted(base, imports).endswith("TestCase"):
            return True
        root = base
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name):
            continue
        if isinstance(base, ast.Name) and base.id in classes:
            if base.id not in seen and _collected_class(classes[base.id], classes, imports, seen):
                return True
        elif root.id in imports:
            return True
    return False


def collect(path: str, source: str) -> dict[str, Unit]:
    """Every unit of a source file, by qualified name; ``<module>`` holds the module's own code."""
    tree = ast.parse(source)
    constants = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name):
            constants[target.id] = value
    imports = _imports(tree)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    # A module name bound to a marker, such as ``SKIP = pytest.mark.skip(reason="slow")``.
    aliases = {}
    for name, value in constants.items():
        kind = _marker_kind(value.func if isinstance(value, ast.Call) else value, imports)
        if kind:
            aliases[name] = kind
    # The helper classes whose methods a collected class of the file inherits and pytest collects.
    exposed: set[str] = set()
    pending = [c for c in classes.values() if _collected_class(c, classes, imports)]
    while pending:
        for base in pending.pop().bases:
            if isinstance(base, ast.Name) and base.id in classes and base.id not in exposed:
                exposed.add(base.id)
                pending.append(classes[base.id])
    units: dict[str, Unit] = {}

    def add(
        root: ast.AST, qualname: str, collected: bool = True, inherited: tuple[ast.AST, ...] = ()
    ) -> None:
        definitions: list[ast.AST] = []
        if isinstance(root, ast.ClassDef | ast.Module):
            nodes = list(_scope_nodes(root, definitions))
        else:
            nodes = list(ast.walk(root))
        units[qualname] = Unit(
            path,
            qualname,
            root,
            nodes,
            source,
            constants,
            aliases,
            collected,
            imports,
            list(inherited),
        )
        # What a test inherits from its class and module: their decorators and ``pytestmark``.
        above = inherited
        if isinstance(root, ast.ClassDef | ast.Module):
            marks = [
                node
                for statement in root.body
                if isinstance(statement, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in statement.targets)
                for node in ast.walk(statement.value)
            ]
            decorators = [n for d in getattr(root, "decorator_list", []) for n in ast.walk(d)]
            above = (*inherited, *decorators, *marks)
        for child in definitions:
            name = getattr(child, "name", "")
            inside = (
                isinstance(child, ast.ClassDef)
                and child.name not in exposed
                and not _collected_class(child, classes, imports)
            )
            add(
                child,
                name if qualname == "<module>" else f"{qualname}.{name}",
                collected and not inside,
                above,
            )

    add(tree, "<module>")
    return units


def _pytest_call(node: ast.AST, imports: dict[str, str]) -> str:
    """Return ``fail`` for the call ``pytest.fail(...)`` and so on, else an empty string.

    The call may be written through an alias (``pt.fail``) or imported by name (``fail``).
    """
    func = node.func if isinstance(node, ast.Call) else None
    if func is None:
        return ""
    dotted = _dotted(func, imports).split(".")
    if len(dotted) == 2 and dotted[0] == "pytest":
        return dotted[1]
    if isinstance(func, ast.Attribute) and _callee(func.value) == "pytest":
        return func.attr
    return ""


def _self_call(node: ast.AST) -> str:
    """Return ``fail`` for the call ``self.fail(...)`` of a ``unittest.TestCase`` and so on."""
    func = node.func if isinstance(node, ast.Call) else None
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.attr if func.value.id == "self" else ""
    return ""


def _imported_callee(node: ast.Call, imports: dict[str, str]) -> str:
    """Return the imported name of a call: ``raises`` for ``expect`` after an aliased import."""
    return _dotted(node.func, imports).rsplit(".", 1)[-1] or _callee(node)


def _checks(unit: Unit) -> list[Check]:
    checks = []
    for node in sorted(unit.nodes, key=_position):
        name = _imported_callee(node, unit.imports) if isinstance(node, ast.Call) else ""
        if name in RAISES:
            checks.append(Check("raises", node, "raises"))
        elif (
            isinstance(node, ast.Assert)
            or name.lstrip("_").startswith("assert")
            or _pytest_call(node, unit.imports) == "fail"
            or _self_call(node) == "fail"
        ):
            checks.append(Check("assert", node, _key(node)))
    return checks


def _count_approx(node: ast.AST) -> int:
    return sum(isinstance(n, ast.Call) and _callee(n) == "approx" for n in ast.walk(node))


def _nan_knobs(call: ast.Call, name: str) -> list[Knob]:
    """Return whether a comparison accepts NaN: looser when ``nan_ok`` or ``equal_nan`` is true."""
    knobs = []
    for flag, defaults in NAN_FLAGS.items():
        if name not in defaults:
            continue
        given = next((k.value for k in call.keywords if k.arg == flag), None)
        if given is None:
            value = defaults[name]
        elif isinstance(given, ast.Constant) and isinstance(given.value, bool):
            value = given.value
        else:
            continue
        knobs.append(Knob(("call", name, flag), flag, float(value), True, shown=str(value)))
    return knobs


def _call_knobs(call: ast.Call, imports: dict[str, str]) -> list[Knob]:
    name = _tolerance_callee(call, imports)
    return [*_tolerance_knobs(call, name, imports), *_nan_knobs(call, name)]


def _tolerance_knobs(call: ast.Call, name: str, imports: dict[str, str]) -> list[Knob]:
    given: dict[str, float] = {}
    opaque: set[str] = set()
    positional = POSITIONAL.get(name, {})
    named = [(positional[i], arg) for i, arg in enumerate(call.args) if i in positional]
    named += [(kw.arg, kw.value) for kw in call.keywords if kw.arg in TOLERANCES]
    for tolerance, value in named:
        if _is_none(value) and (tolerance == "delta" or name == "approx"):
            continue  # left out: ``approx`` reads a ``None`` tolerance as the default
        number = _number(value)
        if number is None:
            opaque.add(tolerance)
        else:
            given[tolerance] = number
    if name == "approx":
        if opaque:
            return []
        absolute = given.get("abs")
        relative = given.get("rel")
        if relative is None:
            relative = 0.0 if absolute is not None else APPROX_DEFAULT_REL
        given = {"rel": relative, "abs": APPROX_DEFAULT_ABS if absolute is None else absolute}
    elif name == "assertAlmostEqual" and ("delta" in given or "delta" in opaque):
        # A delta replaces the places: read it as the places that allow the same difference.
        if "delta" in opaque:
            return []
        delta = given["delta"]
        places = -math.log10(2 * delta) if delta > 0 else math.inf
        return [Knob(("call", name, "places"), "places", places, False, shown=f"delta={delta:g}")]
    else:
        given = {**{k: v for k, v in DEFAULTS.get(name, {}).items() if k not in opaque}, **given}
    return [
        Knob(("call", name, tolerance), tolerance, value, tolerance in LARGER_IS_LOOSER)
        for tolerance, value in given.items()
    ]


def _compare_knobs(node: ast.Compare) -> list[Knob]:
    """Return the bounds of a comparison; a chain such as ``0 <= x <= 1`` holds one per link."""
    knobs = []
    operands = [node.left, *node.comparators]
    for op, left, right in zip(node.ops, operands, operands[1:]):
        if not isinstance(op, ast.Lt | ast.LtE | ast.Gt | ast.GtE):
            continue
        upper, strict = isinstance(op, ast.Lt | ast.LtE), isinstance(op, ast.Lt | ast.Gt)
        low, high = _number(left), _number(right)
        if high is not None and low is None:
            value, other, larger_is_looser = high, left, upper
        elif low is not None and high is None:
            value, other, larger_is_looser = low, right, not upper
        else:
            continue
        knobs.append(
            Knob(("bound", larger_is_looser, _key(other)), "bound", value, larger_is_looser, strict)
        )
    return knobs


def _check_knobs(node: ast.AST, imports: dict[str, str]) -> list[Knob]:
    knobs = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            knobs += _call_knobs(child, imports)
        elif isinstance(child, ast.Compare):
            knobs += _compare_knobs(child)
    return knobs


def _dict_knobs(node: ast.Dict, path: tuple[str, ...]) -> Iterator[Knob]:
    for key, value in zip(node.keys, node.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        here = (*path, key.value)
        if isinstance(value, ast.Dict):
            yield from _dict_knobs(value, here)
        elif key.value in TOLERANCES and (number := _number(value)) is not None:
            yield Knob(("key", here), ".".join(here), number, key.value in LARGER_IS_LOOSER)


def _assigned(node: ast.Assign | ast.AnnAssign) -> list[tuple[str, ast.expr]]:
    """Return the (name, value) pairs an assignment binds, chained and unpacked ones included."""
    if isinstance(node, ast.AnnAssign):
        named = isinstance(node.target, ast.Name) and node.value is not None
        return [(node.target.id, node.value)] if named else []  # type: ignore[union-attr]
    pairs = []
    for target in node.targets:
        if isinstance(target, ast.Name):
            pairs.append((target.id, node.value))
        elif (
            isinstance(target, ast.Tuple | ast.List)
            and isinstance(node.value, ast.Tuple | ast.List)
            and len(target.elts) == len(node.value.elts)
        ):
            pairs += [
                (t.id, v) for t, v in zip(target.elts, node.value.elts) if isinstance(t, ast.Name)
            ]
    return pairs


def _own_knobs(unit: Unit) -> list[Knob]:
    """Return the tolerances a unit holds outside its checks: defaults, constants, and tables."""
    knobs = []
    for node in sorted(unit.nodes, key=_position):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            a = node.args
            positional = [*a.posonlyargs, *a.args]
            pairs = list(zip(positional[len(positional) - len(a.defaults) :], a.defaults))
            pairs += [(arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
            for arg, default in pairs:
                number = _number(default)
                if number is not None and TOLERANCE_NAME.search(arg.arg):
                    larger = not re.search(r"decimal|places", arg.arg, re.IGNORECASE)
                    knobs.append(Knob(("default", node.name, arg.arg), arg.arg, number, larger))
        elif isinstance(node, ast.Assign | ast.AnnAssign) and not unit.is_function:
            for name, value in _assigned(node):
                number = _number(value)
                if number is not None and TOLERANCE_NAME.search(name):
                    larger = not re.search(r"decimal|places", name, re.IGNORECASE)
                    knobs.append(Knob(("name", name), name, number, larger))
                elif isinstance(value, ast.Dict):
                    knobs += [
                        Knob(
                            ("table", name, *k.context[1]),
                            f"{name}.{k.label}",
                            k.value,
                            k.larger_is_looser,
                        )
                        for k in _dict_knobs(value, ())
                    ]
    return knobs


def _shown(knob: Knob) -> str:
    return knob.shown or f"{knob.value:g}"


def _looser(before: list[Knob], after: list[Knob]) -> list[str]:
    """Return why a tolerance or bound of ``after`` is looser than its match in ``before``."""
    by_context: dict[tuple, list[Knob]] = defaultdict(list)
    for knob in before:
        by_context[knob.context].append(knob)
    seen: Counter[tuple] = Counter()
    reasons = []
    for knob in after:
        index = seen[knob.context]
        seen[knob.context] += 1
        if index >= len(by_context[knob.context]):
            continue
        old = by_context[knob.context][index]
        looser = knob.value > old.value if knob.larger_is_looser else knob.value < old.value
        if looser:
            reasons.append(f"{knob.label} loosened from {_shown(old)} to {_shown(knob)}")
        elif knob.value == old.value and old.strict and not knob.strict:
            reasons.append(f"{knob.label} loosened from strict to non-strict at {knob.value:g}")
    return reasons


def _is_none(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _first_arg(call: ast.Call) -> ast.expr | None:
    return call.args[0] if call.args else None


def _exception_names(node: ast.expr | None) -> list[str]:
    """Return the names of the exceptions an expression lists, "" for one it does not name."""
    elements = node.elts if isinstance(node, ast.Tuple | ast.List) else [node]
    return [_callee(e) if e is not None else "" for e in elements]


def _listed(node: ast.expr | None) -> frozenset[str]:
    """Return the exception names an expression lists that the source writes out."""
    return frozenset(name for name in _exception_names(node) if name)


def _accepted(node: ast.expr | None) -> frozenset[str] | None:
    """Return the exception names an expression lists, or None when it does not name them all."""
    names = _exception_names(node)
    return frozenset(names) if node is not None and all(names) else None


def _widened(before: ast.expr | None, after: ast.expr | None) -> list[str]:
    """Return why ``after`` accepts more exceptions than ``before``: a broad one, or more names."""
    reasons = []
    broad = _listed(after) & BROAD
    if broad and not _listed(before) & BROAD:
        reasons.append(f"widened to {', '.join(sorted(broad))}")
    old, new = _accepted(before), _accepted(after)
    if old is not None and new is not None and old < new:
        reasons.append(f"accepts {', '.join(sorted(new - old))} as well")
    return reasons


# Messages a pattern is tested on: one that matches all of them limits nothing (``""``, ``.*``).
PROBES = ("", "x", "Error: 1 2")


def _limits_message(value: ast.expr) -> bool:
    """Whether a ``match`` of ``pytest.raises`` limits the message: not ``None``, not match-all."""
    if _is_none(value):
        return False
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        try:
            return not all(re.search(value.value, probe) for probe in PROBES)
        except re.error:
            return True
    return True


def _matches(call: ast.Call) -> bool:
    """Whether a ``pytest.raises`` checks the message: it has a ``match`` that limits it."""
    if _callee(call).endswith("Regex") and len(call.args) > 1 and _limits_message(call.args[1]):
        return True
    return any(
        k.arg in {"match", "expected_regex"} and _limits_message(k.value) for k in call.keywords
    )


def _checked(call: ast.Call) -> bool:
    """Whether a ``pytest.raises`` tests the exception with a ``check`` predicate."""
    return any(k.arg == "check" and not _is_none(k.value) for k in call.keywords)


def _raises_reasons(before: ast.Call, after: ast.Call) -> list[str]:
    reasons = []
    if _matches(before) and not _matches(after):
        reasons.append("pytest.raises lost its match")
    if _checked(before) and not _checked(after):
        reasons.append("pytest.raises lost its check")
    reasons += [
        f"pytest.raises {reason}" for reason in _widened(_first_arg(before), _first_arg(after))
    ]
    return reasons


def _exempt_nodes(unit: Unit, exempt: frozenset[str]) -> set[int]:
    """Return the ids of the nodes of the module-level assignments to the exempt constants."""
    body = unit.node.body if isinstance(unit.node, ast.Module) else []
    nodes: set[int] = set()
    for statement in body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id in exempt:
            nodes.update(id(node) for node in ast.walk(statement))
    return nodes


def _markers(unit: Unit, exempt: frozenset[str] = frozenset()) -> dict[str, list[ast.AST]]:
    """Return a unit's markers by kind: written out, or used through a module name bound to one."""
    found: dict[str, list[ast.AST]] = defaultdict(list)
    skipped = _exempt_nodes(unit, exempt)
    for node in sorted([*unit.nodes, *unit.inherited], key=_position):
        if id(node) in skipped:
            continue
        if kind := _marker_kind(node, unit.imports):
            found[kind].append(node)
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in unit.aliases
        ):
            found[unit.aliases[node.id]].append(node)
        elif (call := _pytest_call(node, unit.imports)) in SKIP_CALLS:
            found[f"pytest.{call}()"].append(node)
        elif _self_call(node) == "skipTest":
            found["self.skipTest()"].append(node)
    return found


def _integers(arguments: list[ast.expr]) -> list[int] | None:
    numbers = [_number(argument) for argument in arguments]
    if all(number is not None and number.is_integer() for number in numbers):
        return [int(number) for number in numbers if number is not None]
    return None


def _elements(
    node: ast.AST | None, constants: dict[str, ast.AST], depth: int = 0
) -> Counter[str] | None:
    """Return the cases a parametrize argument lists, when the source states them, else None.

    Each case is its syntax tree, numbers included: a seed, a bound, or a tolerance that changes is
    a case removed and another added.
    """
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        if any(isinstance(element, ast.Starred) for element in node.elts):
            return None
        return Counter(_case(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return Counter(_case(key) for key in node.keys if key is not None)
    if isinstance(node, ast.Name) and node.id in constants and depth < 2:
        return _elements(constants[node.id], constants, depth + 1)
    if isinstance(node, ast.Call) and _callee(node) in {"list", "tuple", "set"} and node.args:
        return _elements(node.args[0], constants, depth + 1)
    if isinstance(node, ast.Call) and _callee(node) == "range" and 1 <= len(node.args) <= 3:
        integers = _integers(node.args)
        try:
            if integers is None:
                return None
            return Counter(_case(ast.Constant(value=i)) for i in range(*integers))
        except (ValueError, OverflowError):
            return None
    return None


def _argnames(node: ast.AST) -> tuple[str, ...]:
    """Return the argument names a parametrize call lists."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return tuple(sorted(name.strip() for name in node.value.split(",") if name.strip()))
    if isinstance(node, ast.List | ast.Tuple):
        names = [
            e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if len(names) == len(node.elts):
            return tuple(sorted(names))
    return (ast.dump(node),)


def _parametrizations(
    unit: Unit,
) -> list[tuple[tuple[str, ...], Counter[str] | None, ast.Call]]:
    """List the parametrizations of a unit: argument names, the cases when known, and the call."""
    found = []
    for node in sorted(unit.nodes, key=_position):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node)
        keywords = {k.arg: k.value for k in node.keywords}
        if name == "parametrize":
            names = node.args[0] if node.args else keywords.get("argnames")
            values = node.args[1] if len(node.args) > 1 else keywords.get("argvalues")
            if names is not None:
                found.append((_argnames(names), _elements(values, unit.constants), node))
        elif name == "fixture":
            params = keywords.get("params")
            if params is not None:
                found.append((("params",), _elements(params, unit.constants), node))
    return found


def _pair(
    old: list[Check], new: list[Check]
) -> tuple[list[tuple[Check, Check]], list[Check], list[Check]]:
    """Match the checks that are the same up to numbers and tolerances; return what is left."""
    kept: dict[str, list[Check]] = defaultdict(list)
    for check in new:
        kept[check.key].append(check)
    paired, lost = [], []
    for check in old:
        if kept[check.key]:
            paired.append((check, kept[check.key].pop(0)))
        else:
            lost.append(check)
    return paired, lost, [check for checks in kept.values() for check in checks]


def _check_findings(old: Unit, new: Unit, flag: Flag) -> None:
    """Flag the checks a unit lost and the tolerances and bounds of its checks that loosened."""
    old_checks, new_checks = _checks(old), _checks(new)
    paired, lost, gained = _pair(old_checks, new_checks)
    removed = False
    for kind, noun in (("assert", "assertion"), ("raises", "pytest.raises check")):
        net = sum(c.kind == kind for c in old_checks) - sum(c.kind == kind for c in new_checks)
        if net > 0:
            removed |= kind == "assert"
            texts = [_clip(_text(old, c.node)) for c in lost if c.kind == kind]
            flag(None, f"{noun} removed" if net == 1 else f"{net} {noun}s removed", texts)
    approx_lost = sum(_count_approx(c.node) for c in lost)
    if not removed and approx_lost > sum(_count_approx(c.node) for c in gained):
        texts = [_clip(_text(old, c.node)) for c in lost if _count_approx(c.node)]
        flag(None, "pytest.approx assertion removed or rewritten", texts)
    for before, after in paired:
        reasons = _raises_reasons(before.node, after.node) if before.kind == "raises" else []
        if _count_approx(after.node) > _count_approx(before.node):
            reasons.append("exact comparison relaxed to pytest.approx")
        reasons += _looser(
            _check_knobs(before.node, old.imports), _check_knobs(after.node, new.imports)
        )
        if reasons:
            flag(
                after.node,
                "; ".join(reasons),
                [_clip(_text(old, before.node))],
                [_clip(_text(new, after.node))],
            )
    for reason in _looser(_own_knobs(old), _own_knobs(new)):
        flag(None, reason)


def _marker_uses(unit: Unit, kind: str) -> list[ast.expr]:
    """List the uses of one marker in a unit in order: ``pytest.mark.xfail(...)`` or a bare one."""
    nodes = sorted(unit.nodes, key=_position)
    called = {id(node.func) for node in nodes if isinstance(node, ast.Call)}
    return [
        node
        for node in nodes
        if isinstance(node, ast.expr)
        and (
            (isinstance(node, ast.Call) and _marker_kind(node.func, unit.imports) == kind)
            or (id(node) not in called and _marker_kind(node, unit.imports) == kind)
        )
    ]


def _keywords(use: ast.expr) -> dict[str, ast.expr]:
    return {k.arg: k.value for k in use.keywords if k.arg} if isinstance(use, ast.Call) else {}


def _skipifs(unit: Unit, kind: str) -> list[tuple[ast.expr, bool]]:
    """List a unit's conditional skips of a kind in order, with whether the condition skips."""
    found = []
    for node in _marker_uses(unit, kind):
        if isinstance(node, ast.Call):
            condition = node.args[0] if node.args else _keywords(node).get("condition")
            skips = isinstance(condition, ast.Constant) and bool(condition.value) == SKIP_WHEN[kind]
            # An ``xfail`` with no condition always applies, and so does a bare one.
            found.append((node, skips or (condition is None and kind.endswith("xfail"))))
        elif kind.endswith("xfail"):
            found.append((node, True))
    return found


def _strict(use: ast.expr) -> bool | None:
    """Whether an ``xfail`` marker fails a test that passes: None if ``strict`` is not literal.

    A marker that does not say is not strict: ``pyproject.toml`` leaves ``xfail_strict`` unset.
    """
    value = _keywords(use).get("strict")
    if value is None:
        return False
    return bool(value.value) if isinstance(value, ast.Constant) else None


def _xfail_reasons(before: ast.expr, after: ast.expr) -> list[str]:
    reasons = []
    if _strict(before) is True and _strict(after) is False:
        reasons.append("xfail made non-strict")
    old, new = _keywords(before).get("raises"), _keywords(after).get("raises")
    if old is not None and (new is None or _is_none(new)):
        reasons.append("xfail lost its raises")
    elif old is not None and new is not None:
        reasons += [f"xfail {reason}" for reason in _widened(old, new)]
    return reasons


def _filters(unit: Unit) -> list[str]:
    """List the literal warning filters of a unit's ``filterwarnings`` markers, in order."""
    found = []
    for node in sorted(unit.nodes, key=_position):
        if isinstance(node, ast.Call) and (
            _dotted(node.func, unit.imports).endswith("mark.filterwarnings")
        ):
            found += [
                arg.value
                for arg in node.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            ]
    return found


def _is_error(warning_filter: str) -> bool:
    """Whether a warning filter (``action:message:category:module:line``) raises the warning."""
    action = warning_filter.split(":", 1)[0].strip()
    return bool(action) and "error".startswith(action)


def _filter_findings(old: Unit, new: Unit, flag: Flag) -> None:
    """Flag a ``filterwarnings`` marker that no longer raises a warning, or lets one pass."""
    before, after = Counter(_filters(old)), Counter(_filters(new))
    dropped = [f for f in before - after if _is_error(f)]
    added = [f for f in after - before if not _is_error(f)]
    if dropped:
        flag(None, "pytest.mark.filterwarnings error filter removed or changed", dropped)
    if added:
        flag(None, "pytest.mark.filterwarnings filter that lets a warning pass added", [], added)


def _marker_findings(old: Unit, new: Unit, flag: Flag, exempt: frozenset[str]) -> None:
    """Flag the skip, xfail, and release markers a unit gained, but not in an exempt constant.

    A ``skipif`` that stays and whose condition becomes a literal true also skips the test, and an
    ``xfail`` that stays and turns non-strict or loses its ``raises`` also lets a failure pass.
    """
    for kind, skips in SKIP_WHEN.items():
        reason = f"{kind.rsplit('.', 1)[-1]} condition made always {str(skips).lower()}"
        for (_, was), (call, now) in zip(_skipifs(old, kind), _skipifs(new, kind)):
            if now and not was:
                flag(call, reason, [], [_clip(_text(new, call))])
    xfail = "pytest.mark.xfail"
    for before_use, after_use in zip(_marker_uses(old, xfail), _marker_uses(new, xfail)):
        if reasons := _xfail_reasons(before_use, after_use):
            flag(
                after_use,
                "; ".join(reasons),
                [_clip(_text(old, before_use))],
                [_clip(_text(new, after_use))],
            )
    _filter_findings(old, new, flag)
    before = _markers(old, exempt)
    for name, nodes in _markers(new, exempt).items():
        extra = len(nodes) - len(before.get(name, []))
        for node in nodes[len(nodes) - extra :] if extra > 0 else []:
            flag(node, f"{name} added to a test that existed", [], [_clip(_text(new, node))])


def _parametrize_findings(old: Unit, new: Unit, flag: Flag) -> None:
    """Flag the cases a unit's parametrizations lost and an argument left without one."""
    pool: dict[tuple[str, ...], list[tuple[Counter[str] | None, ast.Call]]] = defaultdict(list)
    kept = set()
    for names, elements, node in _parametrizations(new):
        pool[names].append((elements, node))
        kept.update(names)
    for names, elements, node in _parametrizations(old):
        if pool[names]:
            after_elements, after_node = pool[names].pop(0)
            if elements is not None and after_elements is not None:
                fewer = sum((elements - after_elements).values())
                if fewer > 0:
                    flag(
                        after_node,
                        f"{fewer} parametrize case(s) removed",
                        [_clip(_text(old, node))],
                        [_clip(_text(new, after_node))],
                    )
        elif missing := [name for name in names if name not in kept]:
            flag(
                None, f"parametrization of {', '.join(missing)} removed", [_clip(_text(old, node))]
            )


def compare(old: Unit, new: Unit, exempt: frozenset[str] = frozenset()) -> list[Finding]:
    """Return the changes in ``new`` that skip, loosen, or remove what the same unit ``old`` runs.

    ``exempt`` names the module constants whose markers are not read: those only added tests use.
    """
    findings: list[Finding] = []

    def flag(
        node: ast.AST | None, reason: str, before: Sequence[str] = (), after: Sequence[str] = ()
    ) -> None:
        line = (_position(node)[0] if node is not None else 0) or _position(new.node)[0]
        findings.append(
            Finding(new.path, line, new.qualname, reason, "\n".join(before), "\n".join(after))
        )

    _check_findings(old, new, flag)
    _marker_findings(old, new, flag, exempt)
    _parametrize_findings(old, new, flag)
    return findings


def _body_key(unit: Unit) -> str:
    """Return a function's arguments and body, without its name, decorators, and docstring."""
    node = unit.node
    assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return ast.dump(node.args) + "".join(ast.dump(statement) for statement in body)


def _collected(path: str) -> bool:
    """Whether pytest collects tests from a file: in its test folders, under a name it collects."""
    name = path.rsplit("/", 1)[-1]
    return any(path.startswith(f"{root}/") for root in TESTPATHS) and any(
        fnmatch.fnmatch(name, pattern) for pattern in COLLECTED
    )


def _only_added_tests(old_units: dict[str, Unit], new_units: dict[str, Unit]) -> frozenset[str]:
    """Return the module constants of a file that tests added by the change use and no code else."""
    module = new_units.get("<module>")
    if module is None:
        return frozenset()
    users: dict[str, list[bool]] = defaultdict(list)
    for name, unit in new_units.items():
        loaded = {
            n.id for n in unit.nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        for constant in loaded & module.constants.keys():
            users[constant].append(name != "<module>" and name not in old_units)
    return frozenset(constant for constant, added in users.items() if all(added))


def _is_fixture(unit: Unit) -> bool:
    """Whether a unit is a function with a parametrized fixture."""
    return unit.is_function and any(n == ("params",) for n, _, _ in _parametrizations(unit))


def _autouse(unit: Unit) -> bool:
    """Whether a unit is a fixture that every test of its scope uses: ``fixture(autouse=...)``."""
    if not unit.is_function:
        return False
    for decorator in getattr(unit.node, "decorator_list", []):
        if (
            isinstance(decorator, ast.Call)
            and _dotted(decorator.func, unit.imports).rsplit(".", 1)[-1] == "fixture"
        ):
            for keyword in decorator.keywords:
                off = isinstance(keyword.value, ast.Constant) and not keyword.value.value
                if keyword.arg == "autouse" and not off:
                    return True
    return False


def _hook(unit: Unit) -> bool:
    """Whether a unit is a pytest hook: a function named ``pytest_...``."""
    return unit.is_function and unit.qualname.rsplit(".", 1)[-1].startswith("pytest_")


def _skipping(units: dict[str, Unit], old_units: dict[str, Unit]) -> list[Finding]:
    """Flag the autouse fixtures that skip tests and the pytest hooks that the change added.

    An autouse fixture or a hook of a ``conftest.py`` reaches the tests that already exist, so a
    skip in one skips them without a marker on any test, and a hook can deselect, skip, or pass
    them in ways a reader of the tests does not see: a hook that is new or whose code changed is
    flagged. An autouse fixture that skipped before is compared with its old self, like any unit.
    """
    findings = []
    for name, unit in units.items():
        was = old_units.get(name)
        if _hook(unit):
            same = was is not None and _hook(was) and ast.dump(was.node) == ast.dump(unit.node)
            reason = "" if same else "pytest hook added or changed"
        elif _autouse(unit) and not (was is not None and _autouse(was)):
            skips = any(_markers(unit._replace(inherited=[])).values())
            reason = "autouse fixture that skips tests added" if skips else ""
        else:
            reason = ""
        if reason:
            line = _position(unit.node)[0]
            findings.append(
                Finding(unit.path, line, unit.qualname, reason, "", _clip(_text(unit, unit.node)))
            )
    return findings


def analyze(changes: Sequence[Change]) -> list[Finding]:
    """Return the changes that skip, loosen, or remove a test in the changed Python files."""
    findings: list[Finding] = []
    deleted: list[tuple[Unit, str]] = []
    fixtures: list[Unit] = []
    added: list[Unit] = []
    added_fixtures: list[Unit] = []
    for change in changes:
        if change.old is None:
            # A new file has nothing to compare with, but a test moved into it is not a deletion.
            try:
                new_file = collect(change.path, change.new or "")
            except (SyntaxError, ValueError):
                continue
            added += [u for u in new_file.values() if u.is_test and _collected(change.path)]
            added_fixtures += [u for u in new_file.values() if _is_fixture(u)]
            if change.path.rsplit("/", 1)[-1] == "conftest.py":
                findings += _skipping(new_file, {})
            continue
        try:
            old_units = collect(change.old_path, change.old)
        except (SyntaxError, ValueError):
            continue
        try:
            new_units = collect(change.path, change.new or "")
        except (SyntaxError, ValueError) as error:
            line = getattr(error, "lineno", None) or 1
            findings.append(Finding(change.path, line, "", f"does not parse: {error}"))
            continue
        reason = "test deleted"
        if _collected(change.old_path) and not _collected(change.path):
            # Renamed out of the names pytest collects: the tests are gone, not moved.
            reason = f"test no longer collected: the file was renamed to {change.path}"
            new_units = {}
        exempt = _only_added_tests(old_units, new_units)
        findings += _skipping(new_units, old_units)
        for name, unit in old_units.items():
            if name in new_units:
                now = new_units[name]
                if unit.is_test and now.is_function and not now.is_test:
                    # Same function, but pytest no longer collects its class.
                    reason = "test no longer collected: pytest stops collecting its class"
                    line = _position(now.node)[0]
                    findings.append(
                        Finding(now.path, line, name, reason, _clip(_text(unit, unit.node)))
                    )
                findings += compare(unit, now, exempt if name == "<module>" else frozenset())
            elif unit.is_test:
                deleted.append((unit, reason))
            elif _is_fixture(unit):
                fixtures.append(unit)
        added += [
            u
            for n, u in new_units.items()
            if n not in old_units and u.is_test and _collected(change.path)
        ]
        added_fixtures += [u for n, u in new_units.items() if n not in old_units and _is_fixture(u)]
    for unit in fixtures:
        name = unit.qualname.rsplit(".", 1)[-1]
        # A fixture that moved is an added parametrized fixture of the same name; a function of that
        # name that is not one is no move.
        same = [a for a in added_fixtures if a.qualname.rsplit(".", 1)[-1] == name]
        if len(same) == 1:
            findings += compare(unit, same[0])
        else:
            line = _position(unit.node)[0]
            reason = "parametrized fixture deleted"
            findings.append(
                Finding(unit.path, line, unit.qualname, reason, _clip(_text(unit, unit.node)))
            )
    for unit, reason in deleted:
        key = _body_key(unit)
        name = unit.qualname.rsplit(".", 1)[-1]
        match = next((a for a in added if _body_key(a) == key), None)
        if match is None:
            same_name = [a for a in added if a.qualname.rsplit(".", 1)[-1] == name]
            match = same_name[0] if len(same_name) == 1 else None
        if match is None:
            line = _position(unit.node)[0]
            findings.append(
                Finding(unit.path, line, unit.qualname, reason, _clip(_text(unit, unit.node)))
            )
        else:
            added.remove(match)
            findings += compare(unit, match)
    # A marker on a class or module counts for each of its tests: the same line is reported once.
    unique: dict[tuple, Finding] = {}
    for finding in findings:
        unique.setdefault((finding.path, finding.line, finding.reason, finding.after), finding)
    return list(unique.values())


def _is_baseline(path: str) -> bool:
    parts = path.split("/")
    named = "baselines" in parts[:-1] or "baseline" in parts[-1].lower()
    return named and parts[0] in ROOTS and not path.endswith((".py", ".md"))


def _changed_values(old: Any, new: Any, path: str = "") -> list[str]:
    """Return the paths in ``old`` whose value is gone or different in ``new``."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [
            leaf
            for key, value in old.items()
            for leaf in (
                _changed_values(value, new[key], f"{path}.{key}" if path else str(key))
                if key in new
                else [f"{path}.{key}" if path else str(key)]
            )
        ]
    if isinstance(old, list) and isinstance(new, list):
        leaves = [
            leaf
            for index, (a, b) in enumerate(zip(old, new))
            for leaf in _changed_values(a, b, f"{path}[{index}]")
        ]
        return leaves + [f"{path}[{index}]" for index in range(len(new), len(old))]
    return [] if old == new else [path or "(value)"]


def baseline_findings(path: str, old: str | bytes | None, new: str | bytes | None) -> list[Finding]:
    """Flag a baseline that was deleted, or changed (a JSON one: its old values changed)."""
    if old is None:
        return []
    if new is None:
        return [Finding(path, 0, "", "baseline deleted")]
    if old == new:
        return []
    if path.endswith(".json"):
        try:
            leaves = _changed_values(json.loads(old), json.loads(new))
        except ValueError:
            leaves = ["(file)"]
        if not leaves:
            return []
        shown = ", ".join(leaves[:5]) + (f", and {len(leaves) - 5} more" if len(leaves) > 5 else "")
        return [Finding(path, 0, "", f"baseline re-recorded (changed: {shown})")]
    return [Finding(path, 0, "", "baseline re-recorded")]


def _git(*args: str, root: str | None = None) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True, cwd=root).stdout


def _show(root: str, revision: str, path: str) -> bytes:
    return _git("show", f"{revision}:{path}", root=root)


def _text_of(blob: bytes | None) -> str | None:
    return None if blob is None else blob.decode("utf-8", "replace")


def changed_files(base: str, head: str, root: str) -> list[tuple[str, str, str, str]]:
    """List the changed files of ``tests/`` and ``qa/``: status, old path, new path, merge base."""
    merge_base = _git("merge-base", base, head, root=root).decode().strip()
    raw = _git("diff", "--name-status", "-z", "-M", merge_base, head, "--", *ROOTS, root=root)
    tokens = raw.decode("utf-8", "replace").split("\0")
    files, index = [], 0
    while index < len(tokens) and tokens[index]:
        status = tokens[index]
        width = 2 if status[0] in "RC" else 1
        paths = tokens[index + 1 : index + 1 + width]
        files.append((status[0], paths[0], paths[-1], merge_base))
        index += 1 + width
    return files


def load(base: str, head: str, root: str) -> tuple[list[Change], list[Finding]]:
    """Read the changed Python and baseline files; return the Python changes and baseline flags."""
    changes: list[Change] = []
    findings: list[Finding] = []
    for status, old_path, path, merge_base in changed_files(base, head, root):
        old = None if status in "AC" else _show(root, merge_base, old_path)
        new = None if status == "D" else _show(root, head, path)
        if old is not None and _is_baseline(old_path) and not _is_baseline(path):
            findings += baseline_findings(old_path, old, None)  # renamed out of the baselines
        elif path.endswith(".py") and path.split("/")[0] in ROOTS:
            changes.append(Change(path, old_path, _text_of(old), _text_of(new)))
        elif old is not None and old_path.endswith(".py") and old_path.split("/")[0] in ROOTS:
            # A Python file renamed to another kind of file: its tests are gone.
            changes.append(Change(path, old_path, _text_of(old), None))
        elif _is_baseline(path):
            findings += baseline_findings(path, old, new)
    return changes, findings


def _exists(root: str, revision: str, path: str) -> bool:
    return bool(_git("ls-tree", "--name-only", revision, "--", path, root=root).strip())


def _pytest_options(root: str, revision: str) -> dict[str, Any]:
    """Return the ``[tool.pytest.ini_options]`` of ``pyproject.toml`` at a revision."""
    if not _exists(root, revision, "pyproject.toml"):
        return {}
    try:
        text = _show(root, revision, "pyproject.toml").decode("utf-8", "replace")
        return tomllib.loads(text).get("tool", {}).get("pytest", {}).get("ini_options", {})
    except tomllib.TOMLDecodeError:
        return {}


def workflow_jobs(workflow: str) -> list[tuple[int, str]]:
    """Return the jobs of a workflow file that run this script, found by content: (line, text).

    A job is the mapping under its name in ``jobs:``, up to the next line at or left of its name.
    Its text leaves out blank lines and comments, so a reworded comment is not a change, and keeps
    the job's own condition, runner, and permissions, which decide whether its steps run.
    """
    lines = workflow.splitlines()
    name = DETECTOR.rsplit("/", 1)[-1][:-3]
    heads = [i for i, line in enumerate(lines) if re.fullmatch(r"jobs:\s*", line)]
    if not heads:
        return []

    def content(i: int) -> bool:
        return bool(lines[i].strip()) and not lines[i].lstrip().startswith("#")

    def indent(i: int) -> int:
        return len(lines[i]) - len(lines[i].lstrip())

    rest = [i for i in range(heads[0] + 1, len(lines)) if content(i)]
    if not rest:
        return []
    column = indent(rest[0])
    starts = [i for i in rest if indent(i) == column]
    jobs = []
    for start, stop in zip(starts, [*starts[1:], len(lines)]):
        end = next((i for i in rest if i > start and indent(i) < column), len(lines))
        block = [lines[i].rstrip() for i in range(start, min(stop, end)) if content(i)]
        if any(name in line for line in block):
            jobs.append((start + 1, "\n".join(block)))
    return jobs


def guard_findings(base: str, head: str, root: str) -> list[Finding]:
    """Flag a change to this script or to the workflow job that runs it.

    CI runs the copy of this script on the base branch, so a pull request cannot loosen the check
    it is judged by without a maintainer reading the change. A file the base does not have yet is
    not a change.
    """
    merge_base = _git("merge-base", base, head, root=root).decode().strip()
    findings = []
    if _exists(root, merge_base, DETECTOR):
        if not _exists(root, head, DETECTOR):
            findings.append(Finding(DETECTOR, 0, "", "this check was deleted"))
        elif _show(root, head, DETECTOR) != _show(root, merge_base, DETECTOR):
            findings.append(Finding(DETECTOR, 0, "", "this check changed"))
    changed = [
        key
        for key in PYTEST_OPTIONS
        if _pytest_options(root, merge_base).get(key) != _pytest_options(root, head).get(key)
    ]
    if changed:
        reason = f"pytest options that choose or judge the tests changed: {', '.join(changed)}"
        findings.append(Finding("pyproject.toml", 0, "", reason))
    if _exists(root, merge_base, WORKFLOW):
        before = workflow_jobs(_show(root, merge_base, WORKFLOW).decode("utf-8", "replace"))
        text = (
            _show(root, head, WORKFLOW).decode("utf-8", "replace")
            if _exists(root, head, WORKFLOW)
            else ""
        )
        after = workflow_jobs(text)
        if before and [job for _, job in before] != [job for _, job in after]:
            line = after[0][0] if after else 0
            reason = "the workflow job that runs this check changed or was removed"
            findings.append(Finding(WORKFLOW, line, "", reason))
    return findings


def report(findings: Sequence[Finding]) -> str:
    """Format the flags: where, why, and the code before and after."""
    lines = []
    for finding in findings:
        where = f"{finding.path}:{finding.line}" if finding.line else finding.path
        unit = f" {finding.unit}:" if finding.unit else ""
        lines.append(f"{where}:{unit} {finding.reason}")
        lines += [f"    - {line}" for line in finding.before.splitlines()]
        lines += [f"    + {line}" for line in finding.after.splitlines()]
    return "\n".join(lines)


def _time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def label_applied(repository: str, pull_request: str, label: str) -> datetime | None:
    """Return when a pull request got the label it carries now, read from GitHub; else None."""
    path = f"repos/{repository}/issues/{pull_request}/events"
    jq = '.[] | [.event, (.label.name // ""), .created_at] | @tsv'
    out = subprocess.run(
        ["gh", "api", "--paginate", path, "--jq", jq], check=True, capture_output=True, text=True
    ).stdout
    applied = None
    for line in out.splitlines():
        event, name, when = line.split("\t")
        if name == label and event in {"labeled", "unlabeled"}:
            applied = _time(when) if event == "labeled" else None
    return applied


def _detail(error: Exception) -> str:
    """Return the standard error of a failed command, which is text or bytes."""
    detail = getattr(error, "stderr", None) or ""
    return (detail.decode("utf-8", "replace") if isinstance(detail, bytes) else detail).strip()


def main(argv: Sequence[str] = ()) -> int:
    """Print the flagged changes; exit 0 when none or approved, 1 when not, 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the revision the change starts from")
    parser.add_argument("--head", default="HEAD", help="the revision to check (default HEAD)")
    parser.add_argument("--pr", help="the pull request whose label approves the change")
    parser.add_argument(
        "--since",
        help="the time the pull request's head last changed; a label set before it is stale",
    )
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--label", default=LABEL, help=f"the approving label (default {LABEL})")
    args = parser.parse_args(argv)
    if args.pr and not args.since:
        parser.error("--pr needs --since, the time of the pull request's last head change")
    try:
        root = _git("rev-parse", "--show-toplevel").decode().strip()
        changes, findings = load(args.base, args.head, root)
        findings = [*analyze(changes), *findings, *guard_findings(args.base, args.head, root)]
        if not findings:
            print(
                f"No change between {args.base} and {args.head} skips, loosens, or removes a test."
            )
            return 0
        print(f"{len(findings)} flagged change(s) in tests/ and qa/:\n\n{report(findings)}\n")
        applied = label_applied(args.repo, args.pr, args.label) if args.pr else None
        since = _time(args.since) if args.since else None
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"::error::The check of test changes could not run: {error} {_detail(error)}")
        return 2
    if applied is not None and since is not None:
        if applied > since:
            print(f"::notice::The label {args.label} approves these changes.")
            return 0
        print(
            f"::warning::The label {args.label} was set before the pull request's head last "
            "changed, so it does not approve these changes."
        )
    for finding in findings:
        location = f"file={finding.path},line={max(finding.line, 1)}"
        print(f"::error {location},title=Test change needs approval::{finding.reason}")
    print(
        "::error::The pull request skips, loosens, or removes tests, or re-records baselines. A "
        f"maintainer who has read the flags above approves it by setting the label {args.label} "
        "(remove it first if it is set), after the last push; then re-run this job."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
