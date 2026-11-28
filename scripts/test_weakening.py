"""Flag a pull request that weakens a test, so that a person approves the change.

A change that must make a check pass can drop an assertion, loosen a tolerance, skip the test, or
record the new output as the baseline, and each of these passes the check without making the code
right. This script compares the base and the head of ``tests/`` and ``qa/`` by their syntax trees,
function by function, and flags:

- a removed ``assert`` statement, ``assert_*`` or ``pytest.fail`` call, ``pytest.raises`` check, or
  ``pytest.approx`` assertion, and a ``pytest.raises`` that lost its ``match``, widened to
  ``Exception``, or lists more exceptions;
- a looser tolerance or bound: a larger ``rtol``, ``atol``, ``abs``, ``rel``, ``tol``, or ``delta``,
  a smaller ``decimal`` or ``places``, a looser bound in a comparison, an exact comparison turned
  into ``pytest.approx``, in a call, a default, a module constant, or a tolerance table;
- a new ``skip``, ``skipif``, or ``xfail`` marker or call, or a new ``release`` marker (a pull
  request into a release branch skips those tests), on a test that existed in the base, written out
  or through a module name bound to one;
- a deleted test function or parametrized fixture, a test file renamed out of the names pytest
  collects, or a removed ``parametrize`` case (a literal list, tuple, set, or ``range``; a case
  that another replaced counts as removed);
- a re-recorded baseline file: a changed or deleted file of ``baselines/`` or named for one, and a
  JSON baseline whose recorded values changed.

An added test, a tightened tolerance, and a test that only moved or was renamed are not flagged. It
cannot see an assertion rewritten to a weaker one, a tolerance passed through a variable, or a
check that moved into a helper, so a pass is necessary and not sufficient.

The script prints each flagged change with its reason and exits 1 unless the pull request carries
the label ``test-change-approved``, which a maintainer sets after reading the flags. Run it in a
checkout with the history of ``--base``; a pull request's merge commit is ``HEAD`` and its first
parent ``HEAD^1``. ``--pr`` reads the label's events from GitHub with ``gh`` (``GH_TOKEN`` and
``GITHUB_REPOSITORY`` set). The label approves only the head it was set after: one set before
``--since``, the time the head last changed, is stale, so a push withdraws the approval and a
maintainer removes the label and sets it again after reading the new flags. It exits 2 when it
cannot decide.

    python3 scripts/test_weakening.py --base HEAD^1 --pr 123 --since 2026-11-20T12:00:00Z
"""

from __future__ import annotations

import argparse
import ast
import copy
import fnmatch
import json
import operator
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from typing import Any, NamedTuple

LABEL = "test-change-approved"
ROOTS = ("tests", "qa")

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
    "assertAlmostEqual": {2: "places"},
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
RAISES = frozenset({"raises", "warns", "deprecated_call"})
MARKERS = frozenset({"skip", "skipif", "xfail", "release"})
SKIP_CALLS = frozenset({"skip", "xfail", "importorskip"})
BROAD = frozenset({"Exception", "BaseException"})
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


def _callee(node: ast.AST) -> str:
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else ""


def _tolerance_callee(call: ast.Call) -> str:
    """Return the callee a call's tolerance knobs are read for: ``math.isclose`` apart from NumPy's."""
    func = call.func
    if (
        isinstance(func, ast.Attribute)
        and func.attr == "isclose"
        and isinstance(func.value, ast.Name)
        and func.value.id in {"math", "cmath"}
    ):
        return f"math.{func.attr}"
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
        node.keywords = [keyword for keyword in node.keywords if keyword.arg not in TOLERANCES]
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


def _marker_kind(node: ast.AST) -> str:
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
    return ""


def _collected_class(node: ast.ClassDef) -> bool:
    """Whether pytest collects the methods of a class: named ``Test...`` or a ``TestCase``."""
    bases = {_callee(base) for base in node.bases}
    return node.name.startswith("Test") or "TestCase" in bases


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
    # A module name bound to a marker, such as ``SKIP = pytest.mark.skip(reason="slow")``.
    aliases = {}
    for name, value in constants.items():
        kind = _marker_kind(value.func if isinstance(value, ast.Call) else value)
        if kind:
            aliases[name] = kind
    units: dict[str, Unit] = {}

    def add(root: ast.AST, qualname: str, collected: bool = True) -> None:
        definitions: list[ast.AST] = []
        if isinstance(root, ast.ClassDef | ast.Module):
            nodes = list(_scope_nodes(root, definitions))
        else:
            nodes = list(ast.walk(root))
        units[qualname] = Unit(path, qualname, root, nodes, source, constants, aliases, collected)
        for child in definitions:
            name = getattr(child, "name", "")
            inside = isinstance(child, ast.ClassDef) and not _collected_class(child)
            add(
                child,
                name if qualname == "<module>" else f"{qualname}.{name}",
                collected and not inside,
            )

    add(tree, "<module>")
    return units


def _pytest_call(node: ast.AST) -> str:
    """Return ``fail`` for the call ``pytest.fail(...)`` and so on, else an empty string."""
    func = node.func if isinstance(node, ast.Call) else None
    if isinstance(func, ast.Attribute) and _callee(func.value) == "pytest":
        return func.attr
    return ""


def _checks(unit: Unit) -> list[Check]:
    checks = []
    for node in sorted(unit.nodes, key=_position):
        name = _callee(node) if isinstance(node, ast.Call) else ""
        if (
            isinstance(node, ast.Assert)
            or name.startswith("assert")
            or _pytest_call(node) == "fail"
        ):
            checks.append(Check("assert", node, _key(node)))
        elif name in RAISES:
            checks.append(Check("raises", node, "raises"))
    return checks


def _count_approx(node: ast.AST) -> int:
    return sum(isinstance(n, ast.Call) and _callee(n) == "approx" for n in ast.walk(node))


def _call_knobs(call: ast.Call) -> list[Knob]:
    name = _tolerance_callee(call)
    given: dict[str, float] = {}
    opaque: set[str] = set()
    positional = POSITIONAL.get(name, {})
    named = [(positional[i], arg) for i, arg in enumerate(call.args) if i in positional]
    named += [(kw.arg, kw.value) for kw in call.keywords if kw.arg in TOLERANCES]
    for tolerance, value in named:
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


def _check_knobs(node: ast.AST) -> list[Knob]:
    knobs = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            knobs += _call_knobs(child)
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
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) != 1 or not isinstance(targets[0], ast.Name) or node.value is None:
                continue
            name, number = targets[0].id, _number(node.value)
            if number is not None and TOLERANCE_NAME.search(name):
                larger = not re.search(r"decimal|places", name, re.IGNORECASE)
                knobs.append(Knob(("name", name), name, number, larger))
            elif isinstance(node.value, ast.Dict):
                knobs += [
                    Knob(
                        ("table", name, *k.context[1]),
                        f"{name}.{k.label}",
                        k.value,
                        k.larger_is_looser,
                    )
                    for k in _dict_knobs(node.value, ())
                ]
    return knobs


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
            reasons.append(f"{knob.label} loosened from {old.value:g} to {knob.value:g}")
        elif knob.value == old.value and old.strict and not knob.strict:
            reasons.append(f"{knob.label} loosened from strict to non-strict at {knob.value:g}")
    return reasons


def _first_name(call: ast.Call) -> str:
    return _callee(call.args[0]) if call.args else ""


def _accepted(call: ast.Call) -> frozenset[str] | None:
    """Return the exception names a ``pytest.raises`` lists, or None when it does not name them."""
    first = call.args[0] if call.args else None
    names = [
        _callee(n) for n in (first.elts if isinstance(first, ast.Tuple | ast.List) else [first])
    ]
    return frozenset(names) if first is not None and all(names) else None


def _matches(call: ast.Call) -> bool:
    """Whether a ``pytest.raises`` checks the message: it has a ``match`` that is not ``None``."""
    return any(
        k.arg == "match" and not (isinstance(k.value, ast.Constant) and k.value.value is None)
        for k in call.keywords
    )


def _raises_reasons(before: ast.Call, after: ast.Call) -> list[str]:
    reasons = []
    if _matches(before) and not _matches(after):
        reasons.append("pytest.raises lost its match")
    if _first_name(after) in BROAD and _first_name(before) not in BROAD:
        reasons.append(f"pytest.raises widened to {_first_name(after)}")
    old, new = _accepted(before), _accepted(after)
    if old is not None and new is not None and old < new:
        reasons.append(f"pytest.raises accepts {', '.join(sorted(new - old))} as well")
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
    for node in sorted(unit.nodes, key=_position):
        if id(node) in skipped:
            continue
        if kind := _marker_kind(node):
            found[kind].append(node)
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in unit.aliases
        ):
            found[unit.aliases[node.id]].append(node)
        elif (call := _pytest_call(node)) in SKIP_CALLS:
            found[f"pytest.{call}()"].append(node)
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
        if name == "parametrize" and node.args:
            values = node.args[1] if len(node.args) > 1 else None
            values = values or next((k.value for k in node.keywords if k.arg == "argvalues"), None)
            found.append((_argnames(node.args[0]), _elements(values, unit.constants), node))
        elif name == "fixture":
            params = next((k.value for k in node.keywords if k.arg == "params"), None)
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
        reasons += _looser(_check_knobs(before.node), _check_knobs(after.node))
        if reasons:
            flag(
                after.node,
                "; ".join(reasons),
                [_clip(_text(old, before.node))],
                [_clip(_text(new, after.node))],
            )
    for reason in _looser(_own_knobs(old), _own_knobs(new)):
        flag(None, reason)


def _skipifs(unit: Unit) -> list[tuple[ast.Call, bool]]:
    """List the ``skipif`` markers of a unit in order, with whether the condition is a literal true."""
    found = []
    for node in sorted(unit.nodes, key=_position):
        if isinstance(node, ast.Call) and _marker_kind(node.func) == "pytest.mark.skipif":
            keywords = {k.arg: k.value for k in node.keywords}
            condition = node.args[0] if node.args else keywords.get("condition")
            found.append((node, isinstance(condition, ast.Constant) and bool(condition.value)))
    return found


def _marker_findings(old: Unit, new: Unit, flag: Flag, exempt: frozenset[str]) -> None:
    """Flag the skip, xfail, and release markers a unit gained, but not in an exempt constant.

    A ``skipif`` that stays and whose condition becomes a literal true also skips the test.
    """
    for (_, was), (call, now) in zip(_skipifs(old), _skipifs(new)):
        if now and not was:
            flag(call, "skipif condition made always true", [], [_clip(_text(new, call))])
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
    """Return the weakenings in ``new`` against the same unit ``old``.

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
    """Whether pytest collects tests from a file: under its test folders, with a name it collects."""
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


def analyze(changes: Sequence[Change]) -> list[Finding]:
    """Return the weakenings in the changed Python files of a pull request."""
    findings: list[Finding] = []
    deleted: list[tuple[Unit, str]] = []
    fixtures: list[Unit] = []
    added: list[Unit] = []
    added_fixtures: list[Unit] = []
    names: set[str] = (
        set()
    )  # every function of the new files, to tell a moved fixture from a gone one
    for change in changes:
        if change.old is None:
            # A new file has nothing to compare with, but a test moved into it is not a deletion.
            try:
                new_file = collect(change.path, change.new or "")
            except (SyntaxError, ValueError):
                continue
            added += [u for u in new_file.values() if u.is_test]
            added_fixtures += [u for u in new_file.values() if _is_fixture(u)]
            names.update(u.qualname.rsplit(".", 1)[-1] for u in new_file.values() if u.is_function)
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
        names.update(u.qualname.rsplit(".", 1)[-1] for u in new_units.values() if u.is_function)
        exempt = _only_added_tests(old_units, new_units)
        for name, unit in old_units.items():
            if name in new_units:
                findings += compare(
                    unit, new_units[name], exempt if name == "<module>" else frozenset()
                )
            elif unit.is_test:
                deleted.append((unit, reason))
            elif _is_fixture(unit):
                fixtures.append(unit)
        added += [u for n, u in new_units.items() if n not in old_units and u.is_test]
        added_fixtures += [u for n, u in new_units.items() if n not in old_units and _is_fixture(u)]
    for unit in fixtures:
        name = unit.qualname.rsplit(".", 1)[-1]
        if name not in names:
            line = _position(unit.node)[0]
            reason = "parametrized fixture deleted"
            findings.append(
                Finding(unit.path, line, unit.qualname, reason, _clip(_text(unit, unit.node)))
            )
        elif len(same := [a for a in added_fixtures if a.qualname.rsplit(".", 1)[-1] == name]) == 1:
            findings += compare(unit, same[0])
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
    return findings


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
        elif _is_baseline(path):
            findings += baseline_findings(path, old, new)
    return changes, findings


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
        findings = [*analyze(changes), *findings]
        if not findings:
            print(f"No weakened test between {args.base} and {args.head}.")
            return 0
        print(f"{len(findings)} flagged change(s) in tests/ and qa/:\n\n{report(findings)}\n")
        applied = label_applied(args.repo, args.pr, args.label) if args.pr else None
        since = _time(args.since) if args.since else None
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        print(f"::error::The weakened-test check could not run: {error} {_detail(error)}")
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
        print(f"::error {location},title=Weakened test::{finding.reason}")
    print(
        f"::error::The pull request weakens tests or re-records baselines. A maintainer who has "
        f"read the flags above approves it by setting the label {args.label} (remove it first if "
        f"it is set), after the last push; then re-run this job."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
