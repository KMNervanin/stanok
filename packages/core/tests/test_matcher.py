import sys
import threading
import time
from itertools import islice
from typing import Any

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from stanok_core.matcher import ArgumentTooComplex, _BoundedValidator, failing_paths
from stanok_core.schema import MAX_DESCENTS, MAX_VALIDATION_STEPS

_SIMPLE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "inn": {"type": "string", "minLength": 10, "maxLength": 12},
        "when": {"type": "string", "format": "date", "maxLength": 32},
        "count": {"type": "integer", "minimum": 0, "maximum": 10},
        "kind": {"anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]},
        "tags": {"type": "array", "maxItems": 4, "items": {"type": "string", "maxLength": 64}},
        "nested": {"$ref": "#/$defs/Inner"},
    },
    "required": ["inn", "when", "count", "kind", "tags", "nested"],
    "$defs": {
        "Inner": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        }
    },
}

_VALUES: list[Any] = [
    None,
    True,
    False,
    0,
    1,
    -1,
    2.0,
    2.5,
    "",
    "a",
    "7707083893",
    "2026-07-25",
    "not-a-date",
    [],
    ["a"],
    ["a", "b", "c", "d", "e"],
    [1],
    {},
    {"flag": True},
    {"flag": 1},
    {"flag": True, "extra": 1},
]


def _instances() -> list[dict[str, Any]]:
    """One instance per (property, value) pair over a valid baseline, plus edge shapes."""
    base = {
        "inn": "7707083893",
        "when": "2026-07-25",
        "count": 3,
        "kind": "a",
        "tags": ["x"],
        "nested": {"flag": True},
    }
    out: list[dict[str, Any]] = [dict(base), {}, {"unexpected": 1}]
    for prop in base:
        for value in _VALUES:
            out.append({**base, prop: value})
    for value in _VALUES:
        out.append({**base, "surprise": value})
    return out


@pytest.mark.parametrize("instance", _instances())
def test_verdict_matches_stock_jsonschema(instance: dict[str, Any]) -> None:
    # the bounded validator changes HOW the check runs, never WHAT the schema means: input_schema
    # stays the normative truth, so its verdict must be identical to the reference implementation
    stock = Draft202012Validator(_SIMPLE, format_checker=FormatChecker()).is_valid(instance)
    assert (failing_paths(_SIMPLE, instance) == []) is stock


def test_reports_path_of_the_failing_field() -> None:
    paths = failing_paths(_SIMPLE, {**_instances()[0], "nested": {"flag": "no"}})
    assert paths == ["nested/flag"]


def test_paths_never_carry_argument_values() -> None:
    secret = "SECRET_VALUE_12345"
    paths = failing_paths(_SIMPLE, {**_instances()[0], "inn": secret})
    assert paths and all(secret not in path for path in paths)


def _fanout_schema(levels: int) -> dict[str, Any]:
    """anyOf[$ref x2] chain: acyclic, tiny, and exponential to expand. schema.py refuses it at
    registration, so it can only reach the matcher if a bound is broken - which is what the runtime
    budget is for."""
    defs: dict[str, Any] = {f"D{levels}": {"type": "array", "maxItems": 0, "items": {"type": "x"}}}
    for i in range(levels - 1, -1, -1):
        defs[f"D{i}"] = {"anyOf": [{"$ref": f"#/$defs/D{i + 1}"}, {"$ref": f"#/$defs/D{i + 1}"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"$ref": "#/$defs/D0"}},
        "required": ["x"],
        "$defs": defs,
    }


def test_exponential_fanout_hits_the_budget_instead_of_hanging() -> None:
    # K2b: before the fix this same shape ran 3.7 s at 11 levels and grew x2 per level
    started = time.perf_counter()
    with pytest.raises(ArgumentTooComplex):
        failing_paths(_fanout_schema(40), {"x": ["s"] * 1000})
    assert time.perf_counter() - started < 5.0


def test_long_ref_chain_becomes_argument_too_complex() -> None:
    # a chain deep enough to exhaust the interpreter stack inside jsonschema's ref resolution must
    # still leave as a fail-closed rejection, never a bare RecursionError
    defs: dict[str, Any] = {"D400": {"type": "integer"}}
    for i in range(400):
        defs[f"D{i}"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"n": {"$ref": f"#/$defs/D{i + 1}"}},
            "required": ["n"],
        }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"a": {"$ref": "#/$defs/D0"}},
        "required": ["a"],
        "$defs": defs,
    }
    instance: dict[str, Any] = {"a": {}}
    node = instance["a"]
    for _ in range(400):
        node["n"] = {}
        node = node["n"]
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(300)
    try:
        with pytest.raises(ArgumentTooComplex):
            failing_paths(schema, instance)
    finally:
        sys.setrecursionlimit(old_limit)


def test_bounded_keywords_refuse_to_run_without_a_budget() -> None:
    # using the extended validator outside failing_paths must fail closed, not run unmetered
    validator = _BoundedValidator(_SIMPLE, format_checker=FormatChecker())
    with pytest.raises(ArgumentTooComplex):
        list(validator.iter_errors({"inn": "7707083893"}))


def test_budget_is_fresh_for_every_call() -> None:
    # a budget leaking across calls would let cheap calls starve a later legitimate one
    valid = _instances()[0]
    for _ in range(50):
        assert failing_paths(_SIMPLE, valid) == []


def test_concurrent_validations_do_not_share_a_budget() -> None:
    valid = _instances()[0]
    results: list[object] = []

    def run() -> None:
        try:
            for _ in range(20):
                results.append(failing_paths(_SIMPLE, valid))
        except ArgumentTooComplex as exc:  # pragma: no cover - only on a shared-budget regression
            results.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [[]] * 160


def _charges(schema: dict[str, Any], instance: Any) -> tuple[int, int]:
    """What matcher.py actually spends on `instance`, with the budget lifted out of the way."""
    from stanok_core.matcher import _FORMAT_CHECKER, _budget

    huge = 10**9
    cell = [huge, huge]
    token = _budget.set(cell)
    try:
        list(
            islice(
                _BoundedValidator(dict(schema), format_checker=_FORMAT_CHECKER).iter_errors(
                    instance
                ),
                3,
            )
        )
    finally:
        _budget.reset(token)
    return huge - cell[0], huge - cell[1]


def _largest_accepted(node: dict[str, Any], defs: dict[str, Any]) -> Any:
    """The biggest document the subschema accepts - what the registration bound claims to cover."""
    if "$ref" in node:
        return _largest_accepted(defs[node["$ref"].rsplit("/", 1)[1]], defs)
    if (
        "anyOf" in node
    ):  # the costly branch is the last one: everything before it is tried and fails
        return _largest_accepted(node["anyOf"][-1], defs)
    kind = node.get("type")
    if kind == "object":
        return {k: _largest_accepted(v, defs) for k, v in node.get("properties", {}).items()}
    if kind == "array":
        return [_largest_accepted(node["items"], defs)] * node["maxItems"]
    if "enum" in node:
        return node["enum"][-1]
    if kind == "string":
        return "x" * min(node.get("maxLength", 1), 8)
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    return node.get("maximum", 0)


def _union_layers(branches: int, levels: int) -> dict[str, Any]:
    """Nested discriminated unions: the shape whose real cost is (branches ** levels). "a" sorts
    before the discriminator "z", so every branch validates the shared child before rejecting."""
    defs: dict[str, Any] = {
        f"U{levels}": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"leaf": {"type": "integer", "minimum": 0, "maximum": 9}},
            "required": ["leaf"],
        }
    }
    for i in range(levels - 1, -1, -1):
        defs[f"U{i}"] = {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "a": {"$ref": f"#/$defs/U{i + 1}"},
                        "z": {"type": "integer", "const": k},
                    },
                    "required": ["a", "z"],
                }
                for k in range(branches)
            ]
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {"root": {"$ref": "#/$defs/U0"}},
        "required": ["root"],
        "$defs": defs,
    }


def _int(value: int) -> dict[str, Any]:
    return {"type": "integer", "minimum": value, "maximum": value}


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": list(props),
    }


_CORPUS: list[dict[str, Any]] = [
    _SIMPLE,
    # a nested union where NO branch accepts the value, inside an outer union that does: the
    # envelope has to be inherited from the outer union, not recomputed locally
    _obj(
        {
            "u": {
                "anyOf": [
                    _obj(
                        {
                            "a": {
                                "anyOf": [
                                    {"type": "array", "maxItems": 1, "items": _int(0)},
                                    {"type": "array", "maxItems": 1, "items": _int(1)},
                                ]
                            },
                            "z": {"type": "string", "const": "A"},
                        }
                    ),
                    _obj(
                        {
                            "a": {"type": "array", "maxItems": 500, "items": _int(2)},
                            "z": {"type": "string", "const": "B"},
                        }
                    ),
                ]
            }
        }
    ),
    # one def reached from branches that see different envelopes: its measurement cannot be reused
    {
        **_obj(
            {
                "u": {
                    "anyOf": [
                        {"$ref": "#/$defs/Short"},
                        {"type": "array", "maxItems": 500, "items": _int(2)},
                    ]
                }
            }
        ),
        "$defs": {"Short": {"type": "array", "maxItems": 1, "items": _int(0)}},
    },
    # the long array sits deep inside the def, so the envelope search has to expand the $ref
    {
        **_obj(
            {
                "u": {
                    "anyOf": [
                        _obj({"inner": {"type": "array", "maxItems": 1, "items": _int(0)}}),
                        {"$ref": "#/$defs/Deep"},
                    ]
                }
            }
        ),
        "$defs": {"Deep": _obj({"inner": {"type": "array", "maxItems": 800, "items": _int(0)}})},
    },
    _union_layers(6, 5),
    _union_layers(3, 3),
    _union_layers(16, 1),
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "xs": {
                "type": "array",
                "maxItems": 2000,
                "items": {"type": "string", "enum": [f"v{i}" for i in range(255)]},
            }
        },
        "required": ["xs"],
    },
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "xs": {
                "type": "array",
                "maxItems": 4094,
                "items": {"anyOf": [{"type": "integer", "const": i} for i in range(8)]},
            }
        },
        "required": ["xs"],
    },
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {f"p{i}": {"type": "integer", "minimum": 0, "maximum": 9} for i in range(60)},
        "required": [f"p{i}" for i in range(60)],
    },
]


@pytest.mark.parametrize("schema", _CORPUS, ids=range(len(_CORPUS)))
def test_registration_bound_is_an_upper_bound_on_real_work(schema: dict[str, Any]) -> None:
    """Registration's promise must cover what the runtime actually spends.

    Two earlier versions of the bound scored a per-node figure and multiplied it by the node budget;
    both passed every other test in this suite while under-counting nested unions by orders of
    magnitude, because nothing compared the model against a measurement. This checks one document
    per schema - the largest one, taking the last (costliest) branch of every union. Documents that
    match an earlier branch are covered by the fuzzer, which enumerates them."""
    from stanok_core.schema import _check_instance_bounds, validate_tool_schema

    validate_tool_schema(schema)  # only accepted schemas make a promise to keep
    predicted = _check_instance_bounds(schema, schema.get("$defs", {}))
    document = _largest_accepted(schema, schema.get("$defs", {}))
    steps, descents = _charges(schema, document)

    assert steps <= predicted.steps, f"steps: spent {steps}, promised {predicted.steps}"
    assert descents <= predicted.descents, (
        f"descents: spent {descents}, promised {predicted.descents}"
    )
    # and the promise must stay inside the budgets the runtime enforces, or the call would be
    # refused despite matching its own schema (B1)
    assert predicted.steps <= MAX_VALIDATION_STEPS
    assert predicted.descents <= MAX_DESCENTS


def test_every_descending_keyword_is_metered() -> None:
    # The budget only bounds work it is charged for. If a jsonschema upgrade teaches one of the
    # profile's keywords to descend into a subschema, or the profile gains a keyword that already
    # does, that descent would run free - so the two sets have to stay equal, not merely overlap.
    from stanok_core.matcher import _METERED
    from stanok_core.schema import ALLOWED_KEYWORDS

    descending = {
        keyword
        for keyword in ALLOWED_KEYWORDS
        if keyword in Draft202012Validator.VALIDATORS
        and "descend" in Draft202012Validator.VALIDATORS[keyword].__code__.co_names
    }
    # $ref reaches descend indirectly, through Validator._validate_reference
    descending |= {"$ref"}
    assert descending <= _METERED, f"descends unmetered: {sorted(descending - _METERED)}"
    # the rest are metered for the bulk scan they run, not for descending; naming them keeps the
    # set honest, so a keyword cannot be added to _METERED without saying which cost it covers
    assert _METERED - descending == {"required", "enum"}
