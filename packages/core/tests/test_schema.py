import pytest
from stanok_core.schema import (
    ToolSchemaError,
    canonical_json,
    digest_spec,
    validate_tool_schema,
)
from stanok_core.tools import Effect, JsonObject, ToolSpec

_CLOSED: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"inn": {"type": "string"}},
    "required": ["inn"],
}


def test_accepts_closed_object() -> None:
    validate_tool_schema(_CLOSED)


def test_accepts_bounded_array() -> None:
    validate_tool_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        }
    )


def test_accepts_ref_into_defs() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"c": {"$ref": "#/$defs/Client"}},
        "required": ["c"],
        "$defs": {"Client": {"type": "object", "additionalProperties": False, "properties": {}}},
    }
    validate_tool_schema(schema)


def test_accepts_anyof_of_scalars() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"id": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        "required": ["id"],
    }
    validate_tool_schema(schema)


@pytest.mark.parametrize(
    "bad_prop",
    [
        True,  # boolean schema
        {},  # empty / unconstrained
        {"type": ["object", "null"]},  # array-valued type
        {"required": ["x"]},  # object keywords without type
        {"additionalProperties": True},  # open object, untyped
        {"patternProperties": {".*": {"type": "string"}}},  # defeats additionalProperties:false
        {"unevaluatedProperties": True},
        {"allOf": [{"type": "string"}]},  # combinator not cleared for v1
        {"oneOf": [{"type": "string"}]},  # oneOf not in v1 grammar
        {"type": "object", "additionalProperties": False, "properties": {"y": {"type": "string"}}},
        {"type": "array"},  # array without items = deep-open
        {"$ref": "https://evil.example/s.json"},  # remote ref
        {"$ref": "#/$defs/Missing"},  # dangling local ref
        {"type": "string", "format": "regex"},  # format not in allowlist
        {"type": "string", "enum": [1, 2]},  # enum values not of declared type
    ],
)
def test_rejects_open_or_unknown_forms(bad_prop: object) -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": bad_prop},
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_optional_property() -> None:
    # a property absent from required (default / default_factory) masks a missing argument
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_non_object_root() -> None:
    with pytest.raises(ToolSchemaError):
        validate_tool_schema({"type": "array", "items": {"type": "string"}})


def test_rejects_embedded_dialect() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_boolean_def_target() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"$ref": "#/$defs/Open"}},
        "required": ["x"],
        "$defs": {"Open": True},
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_deep_ref_chain() -> None:
    # no cycle, but A0 -> A1 -> ... -> A40 is deep recursion the structural preflight can't see
    defs: dict[str, object] = {}
    for i in range(40):
        defs[f"A{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/A{i + 1}"}},
            "required": ["n"],
        }
    defs["A40"] = {"type": "object", "additionalProperties": False, "properties": {}}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/A0"}},
        "required": ["a"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_unsafe_def_name() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "$defs": {"bad/name": {"type": "string"}},
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_too_deep_schema() -> None:
    node: dict[str, object] = {"type": "string"}
    for _ in range(40):
        node = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": node},
            "required": ["n"],
        }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(node)


def test_rejects_oversized_schema() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string", "description": "z" * 200_000}},
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def _spec() -> ToolSpec:
    return ToolSpec(
        name="search_client",
        version=1,
        description="Find a client",
        effect=Effect.read,
        required_permission="customer.read",
        input_schema=_CLOSED,
    )


def test_digest_deterministic_and_prefixed() -> None:
    d = digest_spec(_spec())
    assert d == digest_spec(_spec())
    assert d.startswith("sha256:")
    assert len(d) == len("sha256:") + 64


def test_digest_changes_with_content() -> None:
    assert digest_spec(_spec()) != digest_spec(_spec().model_copy(update={"version": 2}))


def test_canonical_json_sorts_keys() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
