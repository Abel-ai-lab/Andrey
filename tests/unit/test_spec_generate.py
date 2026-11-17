"""The generated agent surfaces are complete, serializable, and consistent with the registry."""

from __future__ import annotations

import json

import pytest

from andrey.spec import (
    ENV_VARS,
    REGISTRY,
    SCHEMA_VERSION,
    get_spec,
    help_schema,
    render_skill,
    to_json_schema,
)
from andrey.spec.generate import _example_token, _param_schema, grammar_default
from andrey.spec.models import ParamSpec

_SPEC_NAMES = sorted(REGISTRY)

#: The JSON Schema type of each parameter type.
JSON_TYPE = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "int_pair": "array",
    "str_list": "array",
    "str_or_list": "string|array",
    "knowledge": "object",
    "targets": "object",
}


def types_of(schema: dict) -> set[str]:
    return set(schema["type"]) if isinstance(schema["type"], list) else {schema["type"]}


def test_help_schema_is_json_serializable_and_complete():
    ctx = help_schema()
    json.dumps(ctx)  # must not raise
    assert set(ctx["methods"]) == set(REGISTRY)
    assert ctx["schema_version"] == SCHEMA_VERSION
    assert len(ctx["environment"]["variables"]) == len(ENV_VARS)
    assert ctx["environment"]["precedence"]
    assert ctx["cli"]["exit_codes"]["0"]


def test_help_schema_single_method_carries_env_and_header():
    ctx = help_schema("pc")
    assert ctx["method"]["name"] == "pc"
    assert "methods" not in ctx
    assert ctx["environment"]["variables"]  # env travels with a single-method context too


@pytest.mark.parametrize("name", _SPEC_NAMES)
def test_method_schema_matches_spec(name):
    spec = get_spec(name)
    schema = to_json_schema(spec)
    json.dumps(schema)

    # Param properties are exactly the spec params, and no other param is accepted.
    props = schema["params"]["properties"]
    assert set(props) == {p.name for p in spec.params}
    assert schema["params"]["additionalProperties"] is False

    for p in spec.params:
        entry = props[p.name]
        assert entry["default"] == json.loads(json.dumps(p.default)) and "description" in entry
        expected_types = {*JSON_TYPE[p.type].split("|"), *(["null"] if p.nullable else [])}
        assert types_of(entry) == expected_types
        if p.choices:
            expected = [*p.choices, None] if p.nullable else list(p.choices)
            if p.type == "str_or_list":
                assert entry["anyOf"][0] == {"enum": expected}
                assert entry["items"]["enum"] == list(p.choices)
            else:
                assert entry["enum"] == expected

    assert schema["status"] == spec.status
    assert schema["data_kinds"] == list(spec.data_kinds)

    # Data inputs surface a flag; the CLI example runs this method.
    assert [i["name"] for i in schema["data"]["inputs"]] == [i.name for i in spec.data.inputs]
    assert schema["cli"].startswith(f"andrey run {name}")


def test_skill_lists_every_method_and_env_var():
    skill = render_skill()
    for name, spec in REGISTRY.items():
        kind = spec.output.graph_kind or spec.output.structure_type
        status = ", experimental" if spec.status == "experimental" else ""
        assert f"- `{name}` ({kind}{status}): " in skill
    for v in ENV_VARS:
        assert v.name in skill
    # The guide names the executable discovery and introspection commands.
    assert "andrey config" in skill
    assert "andrey list" in skill and "--help --json" in skill


def test_skill_calls_each_command_as_the_contract_does():
    """The agent guide gives the invocation of run, bootstrap, and markov-check that
    ``--help --json`` gives, every flag included."""
    skill = render_skill()
    cli = help_schema()["cli"]
    for invocation in (
        cli["invocation"],
        cli["bootstrap"]["invocation"],
        cli["markov_check"]["invocation"],
    ):
        assert f"`{invocation}`" in skill


def test_skill_is_an_installable_skill_that_explains_every_edge_type():
    import yaml

    from andrey.core.structure import EDGE_GLYPHS
    from andrey.spec.generate import EDGE_MEANINGS

    skill = render_skill()
    front = yaml.safe_load(skill.split("---\n")[1])
    assert front["name"] == "andrey" and front["description"].strip()
    # Every edge type a result can carry has a meaning, and the guide shows each with its symbol.
    assert set(EDGE_MEANINGS) == set(EDGE_GLYPHS)
    for name, glyph in EDGE_GLYPHS.items():
        assert name in skill and glyph in skill


@pytest.mark.parametrize("nullable", [False, True])
def test_str_or_list_is_one_string_or_a_list_of_at_least_one(nullable):
    """The schema takes a string or a list of strings, and null where the parameter is nullable;
    the value passes its check unchanged, for the method to read."""
    default = None if nullable else "a"
    param = ParamSpec(name="names", type="str_or_list", default=default, doc="", nullable=nullable)
    entry = _param_schema(param)
    assert types_of(entry) == {"string", "array", *(["null"] if nullable else [])}
    assert entry["items"] == {"type": "string"} and entry["minItems"] == 1
    assert entry["default"] == default
    for value in ("a", ["a", "b"], ("a", "b")):
        assert param.check(value, "method") is value


@pytest.mark.parametrize("nullable", [False, True])
def test_str_or_list_choices_hold_for_one_value_and_for_each_item(nullable):
    """With choices, one value is a choice and a list holds at least one, each a choice: the
    check raises otherwise, and the schema says the same."""
    param = ParamSpec(
        name="names",
        type="str_or_list",
        default="a",
        doc="",
        choices=("a", "b"),
        nullable=nullable,
    )
    for value in ("a", ["a", "b"], ("b",)):
        assert param.check(value, "method") is value
    for bad in ("c", ["a", "c"], []):
        with pytest.raises(ValueError, match="names must be one of 'a', 'b', or a list of them"):
            param.check(bad, "method")
    for bad in (3, ["a", 3]):
        with pytest.raises(TypeError, match="names must be one of"):
            param.check(bad, "method")
    with pytest.raises(ValueError, match="a str_or_list parameter takes no choice aliases"):
        ParamSpec(**{**param.model_dump(), "choice_aliases": (("c", "a"),)})
    entry = _param_schema(param)
    assert "enum" not in entry  # a list is none of the choices
    assert entry["anyOf"] == [
        {"enum": ["a", "b", *([None] if nullable else [])]},
        {"type": "array"},
    ]
    assert entry["items"] == {"type": "string", "enum": ["a", "b"]} and entry["minItems"] == 1


def test_a_str_or_list_default_reads_back_in_the_parameter_grammar():
    """A string default is written bare and a list default as a quoted bracket list."""
    one = ParamSpec(name="names", type="str_or_list", default="a", doc="")
    several = ParamSpec(name="names", type="str_or_list", default=("a", "b"), doc="")
    assert (grammar_default(one), _example_token(one)) == ("a", "names=a")
    assert (grammar_default(several), _example_token(several)) == ("[a,b]", "'names=[a,b]'")
