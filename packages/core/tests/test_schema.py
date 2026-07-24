import sys

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
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 16}
            },
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


def test_rejects_string_pattern() -> None:
    # arbitrary regex is a ReDoS vector on the untrusted boundary (K2a)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string", "pattern": "(a+)+$"}},
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_array_without_maxitems() -> None:
    # an array without maxItems is not bounded (V1)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "array", "items": {"type": "string"}}},
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_array_maxitems_over_limit() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "array", "items": {"type": "string"}, "maxItems": 10_000_000}
        },
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_anyof_ref_fanout() -> None:
    # acyclic and within node/depth/byte limits, but anyOf+$ref reuse forces exponential work (K2b)
    defs: dict[str, object] = {}
    for i in range(18):
        defs[f"L{i}"] = {"anyOf": [{"$ref": f"#/$defs/L{i + 1}"}, {"$ref": f"#/$defs/L{i + 1}"}]}
    defs["L18"] = {"type": "integer"}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"$ref": "#/$defs/L0"}},
        "required": ["x"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_deep_shared_tail_chain() -> None:
    # real ref depth ~52 via a shared tail; the old DFS depth counter undercounted it (K2c)
    defs: dict[str, object] = {}
    for i in range(30):
        defs[f"T{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/T{i + 1}"}},
            "required": ["n"],
        }
    defs["T30"] = {"type": "object", "additionalProperties": False, "properties": {}}
    for i in range(20):
        defs[f"S{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/S{i + 1}"}},
            "required": ["n"],
        }
    defs["S20"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"n": {"$ref": "#/$defs/T0"}},
        "required": ["n"],
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/S0"}},
        "required": ["a"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_long_ref_chain_rejected_without_recursionerror() -> None:
    # the budget pass evaluates $defs iteratively (reverse-topo), so a long shallow ref chain is
    # rejected as ToolSchemaError even with a tight recursion limit — never a leaked RecursionError
    defs: dict[str, object] = {}
    for i in range(100):
        defs[f"D{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/D{i + 1}"}},
            "required": ["n"],
        }
    defs["D100"] = {"type": "integer"}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/D0"}},
        "required": ["a"],
        "$defs": defs,
    }
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(120)
    try:
        with pytest.raises(ToolSchemaError):
            validate_tool_schema(schema)
    finally:
        sys.setrecursionlimit(old_limit)


def test_very_deep_schema_rejected_not_recursionerror() -> None:
    # preflight bounds depth iteratively before serialization, so this is ToolSchemaError, never a
    # leaked RecursionError (V2)
    node: dict[str, object] = {"type": "string"}
    for _ in range(200):
        node = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": node},
            "required": ["n"],
        }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(node)


def test_rejects_overlong_array_anyof_amplification() -> None:
    # in-limit schema, but items=anyOf of hundreds of branches amplifies work on an overlong array
    branches = [{"type": "string", "const": f"v{i}"} for i in range(280)]
    items = {"anyOf": [*branches, {"type": "integer"}]}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"xs": {"type": "array", "maxItems": 1, "items": items}},
        "required": ["xs"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_uniqueitems() -> None:
    # uniqueItems on composite items is an O(n^2) scan — the keyword is not in the v1 profile
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "xs": {
                "type": "array",
                "maxItems": 4096,
                "uniqueItems": True,
                "items": {"type": "integer"},
            }
        },
        "required": ["xs"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_large_enum_under_array() -> None:
    # a huge enum scanned per array element (each of up to 1024 elements) is the amplified case
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "xs": {
                "type": "array",
                "maxItems": 4096,
                "items": {"type": "string", "enum": [f"v{i}" for i in range(900)]},
            }
        },
        "required": ["xs"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_wide_object_reused_under_anyof() -> None:
    # empty {} still triggers every `required` check; anyOf over a wide object multiplies that per
    # array element even though each branch and the payload stay within the node limits
    wide_props: dict[str, object] = {f"p{i}": {"type": "integer"} for i in range(280)}
    wide = {
        "type": "object",
        "additionalProperties": False,
        "properties": wide_props,
        "required": list(wide_props),
    }
    items = {"anyOf": [{"$ref": "#/$defs/Wide"} for _ in range(30)] + [{"type": "integer"}]}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"xs": {"type": "array", "maxItems": 4096, "items": items}},
        "required": ["xs"],
        "$defs": {"Wide": wide},
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_wrapped_anyof_expanded_depth() -> None:
    # each def wraps the next ref in singleton anyOf; ref depth stays small but the validator would
    # descend ~300 nested anyOf levels — bounded by expanded validation depth
    defs: dict[str, object] = {}
    for c in range(30):
        inner: object = {"$ref": f"#/$defs/D{c + 1}"} if c < 29 else {"type": "integer"}
        for _ in range(10):
            inner = {"anyOf": [inner]}
        defs[f"D{c}"] = inner
    defs["D30"] = {"type": "integer"}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/D0"}},
        "required": ["a"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_accepts_reused_ref_when_not_nested() -> None:
    # a $ref reused across sibling fields is linear, not exponential — must still pass
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {"$ref": "#/$defs/Id"},
            "b": {"$ref": "#/$defs/Id"},
            "c": {"$ref": "#/$defs/Id"},
        },
        "required": ["a", "b", "c"],
        "$defs": {"Id": {"type": "string"}},
    }
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
