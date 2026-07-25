import json
import sys

import pytest
from stanok_core.schema import (
    ToolSchemaError,
    _check_instance_bounds,
    canonical_json,
    digest_spec,
    validate_tool_schema,
)
from stanok_core.tools import Effect, JsonObject, ToolSpec

_CLOSED: JsonObject = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"inn": {"type": "string", "maxLength": 64}},
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
                "tags": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 64},
                    "maxItems": 16,
                }
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
        "properties": {
            "id": {
                "anyOf": [
                    {"type": "string", "maxLength": 64},
                    {"type": "integer", "minimum": 0, "maximum": 9},
                ]
            }
        },
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
        {
            "patternProperties": {".*": {"type": "string", "maxLength": 64}}
        },  # defeats additionalProperties:false
        {"unevaluatedProperties": True},
        {"allOf": [{"type": "string", "maxLength": 64}]},  # combinator not cleared for v1
        {"oneOf": [{"type": "string", "maxLength": 64}]},  # oneOf not in v1 grammar
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"y": {"type": "string", "maxLength": 64}},
        },
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
        "properties": {
            "a": {"type": "string", "maxLength": 64},
            "b": {"type": "string", "maxLength": 64},
        },
        "required": ["a"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_non_object_root() -> None:
    with pytest.raises(ToolSchemaError):
        validate_tool_schema({"type": "array", "items": {"type": "string", "maxLength": 64}})


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
        "$defs": {"bad/name": {"type": "string", "maxLength": 64}},
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_too_deep_schema() -> None:
    node: dict[str, object] = {"type": "string", "maxLength": 64}
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
        "properties": {"x": {"type": "array", "items": {"type": "string", "maxLength": 64}}},
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError):
        validate_tool_schema(schema)


def test_rejects_array_maxitems_over_limit() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {
                "type": "array",
                "items": {"type": "string", "maxLength": 64},
                "maxItems": 10_000_000,
            }
        },
        "required": ["x"],
    }
    with pytest.raises(ToolSchemaError, match="must set maxItems"):
        validate_tool_schema(schema)


def test_rejects_anyof_ref_fanout() -> None:
    # acyclic and within node/depth/byte limits, but anyOf+$ref reuse forces exponential work (K2b)
    defs: dict[str, object] = {}
    for i in range(18):
        defs[f"L{i}"] = {"anyOf": [{"$ref": f"#/$defs/L{i + 1}"}, {"$ref": f"#/$defs/L{i + 1}"}]}
    defs["L18"] = {"type": "integer", "minimum": 0, "maximum": 9}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"$ref": "#/$defs/L0"}},
        "required": ["x"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError, match="would enter more than"):
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
    with pytest.raises(ToolSchemaError, match="expanded validation depth"):
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
    defs["D100"] = {"type": "integer", "minimum": 0, "maximum": 9}
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
    node: dict[str, object] = {"type": "string", "maxLength": 64}
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
    # in-limit schema, but every one of the branches is tried against every one of the elements
    branches = [{"type": "string", "const": f"v{i}"} for i in range(280)]
    items = {"anyOf": [*branches, {"type": "integer", "minimum": 0, "maximum": 9}]}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"xs": {"type": "array", "maxItems": 4094, "items": items}},
        "required": ["xs"],
    }
    with pytest.raises(ToolSchemaError, match="would enter more than"):
        validate_tool_schema(schema)


def test_rejects_uniqueitems() -> None:
    # uniqueItems on composite items is an O(n^2) scan — the keyword is not in the v1 profile
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "xs": {
                "type": "array",
                "maxItems": 4094,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 0, "maximum": 9},
            }
        },
        "required": ["xs"],
    }
    with pytest.raises(ToolSchemaError, match="uniqueItems"):
        validate_tool_schema(schema)


def test_rejects_large_enum_under_array() -> None:
    # a huge enum scanned per array element (each of up to 1024 elements) is the amplified case
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "xs": {
                "type": "array",
                "maxItems": 4094,
                "items": {"type": "string", "enum": [f"v{i}" for i in range(900)]},
            }
        },
        "required": ["xs"],
    }
    with pytest.raises(ToolSchemaError, match="would take more than"):
        validate_tool_schema(schema)


def test_rejects_wide_object_reused_under_anyof() -> None:
    # empty {} still triggers every `required` check; anyOf over a wide object multiplies that per
    # array element even though each branch and the payload stay within the node limits
    wide_props: dict[str, object] = {
        f"p{i}": {"type": "integer", "minimum": 0, "maximum": 9} for i in range(100)
    }
    wide = {
        "type": "object",
        "additionalProperties": False,
        "properties": wide_props,
        "required": list(wide_props),
    }
    items = {
        "anyOf": [{"$ref": "#/$defs/Wide"} for _ in range(30)]
        + [{"type": "integer", "minimum": 0, "maximum": 9}]
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"xs": {"type": "array", "maxItems": 4094, "items": items}},
        "required": ["xs"],
        "$defs": {"Wide": wide},
    }
    with pytest.raises(ToolSchemaError, match="would enter more than"):
        validate_tool_schema(schema)


def test_rejects_wrapped_anyof_expanded_depth() -> None:
    # each def wraps the next ref in singleton anyOf; ref depth stays small, but every wrapper is
    # one more descent charged against the same argument node, so they add up and are refused
    defs: dict[str, object] = {}
    for c in range(30):
        inner: object = (
            {"$ref": f"#/$defs/D{c + 1}"}
            if c < 29
            else {"type": "integer", "minimum": 0, "maximum": 9}
        )
        for _ in range(10):
            inner = {"anyOf": [inner]}
        defs[f"D{c}"] = inner
    defs["D30"] = {"type": "integer", "minimum": 0, "maximum": 9}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/D0"}},
        "required": ["a"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError, match="expanded validation depth"):
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
        "$defs": {"Id": {"type": "string", "maxLength": 64}},
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


def test_rejects_unbounded_string() -> None:
    # a string with no maxLength/enum/const has no largest accepted value, so the schema could
    # publish a contract whose calls never fit the byte budget - the same reasoning as maxItems (V1)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"note": {"type": "string"}},
        "required": ["note"],
    }
    with pytest.raises(ToolSchemaError, match="larger than 65536 bytes"):
        validate_tool_schema(schema)


def test_accepts_string_bounded_by_enum_without_maxlength() -> None:
    # enum and const bound the value just as well as maxLength does
    validate_tool_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "stage": {"type": "string", "enum": ["new", "won", "lost"]},
                "kind": {"type": "string", "const": "deal"},
            },
            "required": ["stage", "kind"],
        }
    )


def test_rejects_schema_whose_largest_call_exceeds_the_byte_budget() -> None:
    # every element is bounded and the node count fits, but 4094 x 36 chars cannot be delivered:
    # the menu would advertise maxItems the runtime always refuses
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ids": {
                "type": "array",
                "maxItems": 4094,
                "items": {"type": "string", "format": "uuid", "maxLength": 36},
            }
        },
        "required": ["ids"],
    }
    with pytest.raises(ToolSchemaError, match="larger than 65536 bytes"):
        validate_tool_schema(schema)


def test_accepts_the_same_array_once_it_fits_the_byte_budget() -> None:
    validate_tool_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ids": {
                    "type": "array",
                    "maxItems": 290,
                    "items": {"type": "string", "format": "uuid", "maxLength": 36},
                }
            },
            "required": ["ids"],
        }
    )


def test_rejects_expanded_depth_even_when_each_node_is_cheap() -> None:
    # a $ref/object chain costs only two descents per argument node however long it gets, so nothing
    # but the expanded-depth bound stands between it and a RecursionError inside jsonschema
    defs: dict[str, object] = {"C40": {"type": "integer", "minimum": 0, "maximum": 9}}
    for i in range(40):
        defs[f"C{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/C{i + 1}"}},
            "required": ["n"],
        }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/C0"}},
        "required": ["a"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError, match="expanded validation depth"):
        validate_tool_schema(schema)


def test_rejects_ref_chain_that_multiplies_descents_per_element() -> None:
    # Found by review: amplification used to count only anyOf branches, so a chain of $ref hops -
    # each of which is a real descent charged against the same argument node - slipped through and a
    # fully valid document then ran out of runtime budget (a B1 recurrence, via the other counter)
    defs: dict[str, object] = {"D15": {"type": "integer", "minimum": 0, "maximum": 0}}
    for i in range(15):
        defs[f"D{i}"] = {"$ref": f"#/$defs/D{i + 1}"}
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "a": {
                "type": "array",
                "maxItems": 4094,
                "items": {"anyOf": [{"$ref": "#/$defs/D0"}]},
            }
        },
        "required": ["a"],
        "$defs": defs,
    }
    with pytest.raises(ToolSchemaError, match="would enter more than"):
        validate_tool_schema(schema)


def test_accepts_a_deep_ref_chain_that_stays_cheap_per_node() -> None:
    # the same shape without the array: nesting alone must not be mistaken for per-node work
    defs: dict[str, object] = {"D15": {"type": "integer", "minimum": 0, "maximum": 0}}
    for i in range(15):
        defs[f"D{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/D{i + 1}"}},
            "required": ["n"],
        }
    validate_tool_schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {"a": {"$ref": "#/$defs/D0"}},
            "required": ["a"],
            "$defs": defs,
        }
    )


def test_rejects_branches_that_walk_a_longer_sibling_array() -> None:
    # A branch is run against the value the MATCHING branch produced. jsonschema walks every element
    # before maxItems can reject it, so the short branch pays for the long value, not for its own
    # declaration. Only reproducible in stored key order, where "items" sorts before "maxItems".
    item = {"anyOf": [{"type": "integer", "minimum": i, "maximum": i} for i in range(16)]}
    schema = json.loads(
        canonical_json(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "a": {
                        "anyOf": [
                            {"type": "array", "maxItems": 1, "items": item},
                            {"type": "array", "maxItems": 4094, "items": item},
                        ]
                    }
                },
                "required": ["a"],
            }
        )
    )
    with pytest.raises(ToolSchemaError, match="would enter more than"):
        validate_tool_schema(schema)


def test_charges_a_branch_for_a_wider_siblings_keys() -> None:
    # same mechanism on objects: the narrow branch's additionalProperties scan walks the keys of
    # whatever value the wide branch accepts, so it must be charged for that width
    narrow = {"type": "object", "additionalProperties": False, "properties": {}, "required": []}
    wide_props = {f"p{i}": {"type": "integer", "minimum": 0, "maximum": 9} for i in range(40)}
    wide = {
        "type": "object",
        "additionalProperties": False,
        "properties": wide_props,
        "required": list(wide_props),
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"u": {"anyOf": [narrow, wide]}},
        "required": ["u"],
    }
    validate_tool_schema(schema)  # cheap enough to accept - but it must be charged honestly
    bounds = _check_instance_bounds(schema, {})
    # the narrow branch alone would score ~0; both branches scan the 40-key value
    assert bounds.steps >= 2 * len(wide_props)
