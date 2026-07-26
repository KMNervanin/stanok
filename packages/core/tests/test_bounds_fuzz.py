"""Property test over generated schemas: whatever the profile accepts must stay executable.

Two properties, checked on every schema the profile lets through:

1. the registration bound covers what the runtime really spends - if `_check_instance_bounds`
   promises less than matcher.py charges, the budget can refuse a call the schema allows (B1);
2. the largest document a schema accepts fits the runtime caps and validates without running out
   of budget.

The generator builds nested discriminated unions and sibling arrays with different maxItems on
purpose: those are the shapes whose cost multiplies, and random structure reaches them with
probability ~0.14 per level, which is why three rounds of review missed them. Schemas are put
through `canonical_json` first, because that sorted form is what the registry stores and what
jsonschema then applies keyword by keyword - one defect only ever reproduced in that order.
"""

import json
import random
from collections.abc import Iterator
from itertools import islice, product
from typing import Any, NamedTuple

import pytest
from jsonschema import Draft202012Validator
from stanok_core.matcher import _FORMAT_CHECKER, _BoundedValidator, _budget
from stanok_core.schema import (
    MAX_INSTANCE_BYTES,
    MAX_INSTANCE_NODES,
    ToolSchemaError,
    _check_instance_bounds,
    canonical_json,
    validate_tool_schema,
)

_HUGE = 10**9

_Schema = dict[str, Any]


def _scalar(rnd: random.Random) -> _Schema:
    kind = rnd.choice(["string", "integer", "number", "boolean", "null", "enum", "format"])
    if kind == "enum":
        return {"type": "string", "enum": [f"v{i}" for i in range(rnd.randint(1, 200))]}
    if kind == "format":
        # maxLength must leave room for the format's own sample, or every witness built from this
        # subschema is invalid and the round silently checks nothing
        return {
            "type": "string",
            "maxLength": rnd.choice([48, 64]),
            "format": rnd.choice(["email", "date", "uuid", "ipv4"]),
        }
    if kind == "string":
        return {"type": "string", "maxLength": rnd.choice([4, 32, 512])}
    if kind in ("integer", "number"):
        return {"type": kind, "minimum": 0, "maximum": rnd.choice([9, 10**6])}
    return {"type": kind}


def _closed(props: dict[str, _Schema]) -> _Schema:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": list(props),
    }


def _subschema(rnd: random.Random, depth: int, defs: dict[str, _Schema]) -> _Schema:
    roll = rnd.random()
    if depth <= 0 or roll < 0.28:
        return _scalar(rnd)
    if roll < 0.46:
        # discriminated union: "a" sorts before the discriminator "z", so every branch validates
        # the shared child before rejecting - the shape whose cost is (width ** depth)
        child = _subschema(rnd, depth - 1, defs)
        return {
            "anyOf": [
                _closed({"a": child, "z": {"type": "integer", "minimum": k, "maximum": k}})
                for k in range(rnd.randint(2, 7))
            ]
        }
    if roll < 0.62:
        return _closed(
            {f"p{i}": _subschema(rnd, depth - 1, defs) for i in range(rnd.randint(0, 4))}
        )
    if roll < 0.80:
        return {
            "type": "array",
            "maxItems": rnd.choice([0, 1, 3, 64, 1000, 4094]),
            "items": _subschema(rnd, depth - 1, defs),
        }
    if roll < 0.84:
        # branches whose maxItems differ: the short one still walks the long sibling's value
        child = _subschema(rnd, depth - 1, defs)
        return {
            "anyOf": [
                {"type": "array", "items": child, "maxItems": rnd.choice([0, 1, 8, 500, 4094])}
                for _ in range(rnd.randint(2, 4))
            ]
        }
    if roll < 0.90:
        return {"anyOf": [_subschema(rnd, depth - 1, defs) for _ in range(rnd.randint(1, 6))]}
    name = f"R{len(defs)}"
    defs[name] = _subschema(rnd, depth - 1, defs)
    return {"$ref": f"#/$defs/{name}"}


def _widest_scalar(node: _Schema) -> Any:
    """The largest value the scalar accepts, not merely a valid one: a short string or a small
    number can be swallowed by an earlier anyOf branch, so the costly branches are never tried."""
    if "const" in node:
        return node["const"]
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return max(enum, key=lambda value: len(canonical_json(value)))
    kind = node.get("type")
    fmt = node.get("format")
    if fmt is not None:
        return {
            "email": "a@b.co",
            "date": "2026-07-25",
            "uuid": "0" * 8 + "-0000-0000-0000-" + "0" * 12,
            "ipv4": "127.0.0.1",
        }[fmt]
    if kind == "string":
        return "x" * node.get("maxLength", 4)
    if kind == "boolean":
        return True
    if kind == "null":
        return None
    endpoints = [node[k] for k in ("minimum", "maximum") if k in node]
    return max(endpoints or [0], key=lambda value: len(canonical_json(value)))


def _documents(node: _Schema, defs: dict[str, _Schema], cap: int) -> Iterator[Any]:
    """Documents the subschema accepts, one per combination of branch choices.

    Taking only the last branch of every union - the obvious shortcut - misses the documents that
    match an earlier one, and those can cost more: a union of arrays is worst at whichever branch
    declares the largest maxItems, which need not be the last.
    """
    ref = node.get("$ref")
    if isinstance(ref, str):
        yield from _documents(defs[ref.rsplit("/", 1)[1]], defs, cap)
        return
    branches = node.get("anyOf")
    if isinstance(branches, list):
        for branch in branches:
            yield from _documents(branch, defs, cap)
        return
    kind = node.get("type")
    if kind == "object":
        props: dict[str, _Schema] = node.get("properties", {})
        if not props:
            yield {}
            return
        names = list(props)
        per_property = [list(islice(_documents(props[n], defs, cap), cap)) for n in names]
        for combination in islice(product(*per_property), cap):
            yield dict(zip(names, combination, strict=True))
        return
    if kind == "array":
        for item in islice(_documents(node["items"], defs, cap), cap):
            yield [item] * node["maxItems"]
        return
    yield _widest_scalar(node)


def _charges(schema: _Schema, instance: Any) -> tuple[int, int]:
    """Steps and descents matcher.py really spends, with the budget lifted out of the way."""
    cell = [_HUGE, _HUGE]
    token = _budget.set(cell)
    try:
        validator = _BoundedValidator(dict(schema), format_checker=_FORMAT_CHECKER)
        list(islice(validator.iter_errors(instance), 3))
    finally:
        _budget.reset(token)
    return _HUGE - cell[0], _HUGE - cell[1]


def _node_count(value: Any) -> int:
    count = 0
    stack: list[Any] = [value]
    while stack:
        node = stack.pop()
        count += 1
        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return count


class _Report(NamedTuple):
    rounds: int
    accepted: int
    documents: int
    unions: int  # accepted schemas containing a union, the shapes whose cost multiplies
    peak_steps: int  # so a run that measured nothing cannot look like a run that proved something
    failures: list[str]


def _campaign(seed: int, rounds: int, per_schema: int) -> _Report:
    rnd = random.Random(seed)
    reference = Draft202012Validator.FORMAT_CHECKER
    accepted = documents = unions = peak = 0
    failures: list[str] = []
    for _ in range(rounds):
        defs: dict[str, _Schema] = {}
        props = {
            f"f{i}": _subschema(rnd, rnd.randint(1, 5), defs) for i in range(rnd.randint(1, 3))
        }
        schema: _Schema = _closed(props)
        if defs:
            schema["$defs"] = dict(defs)
        schema = json.loads(canonical_json(schema))  # the form the registry stores
        try:
            validate_tool_schema(schema)
        except ToolSchemaError:
            continue
        accepted += 1
        schema_defs = schema.get("$defs", {})
        if "anyOf" in json.dumps(schema):
            unions += 1
        predicted = _check_instance_bounds(schema, schema_defs)
        validator = Draft202012Validator(schema, format_checker=reference)
        for document in islice(_documents(schema, schema_defs, per_schema), per_schema):
            # every witness is built branch by branch from the schema itself, so an invalid one
            # means the generator drifted - fail instead of skipping the round in silence
            if not validator.is_valid(document):
                failures.append(f"witness rejected by the reference for {json.dumps(schema)[:300]}")
                continue
            documents += 1
            steps, descents = _charges(schema, document)
            peak = max(peak, steps)
            if steps > predicted.steps or descents > predicted.descents:
                failures.append(
                    f"bound under-counts: spent {steps}/{descents}, promised "
                    f"{predicted.steps}/{predicted.descents} for {json.dumps(schema)[:300]}"
                )
            nodes = _node_count(document)
            size = len(canonical_json(document))
            if nodes > MAX_INSTANCE_NODES or size > MAX_INSTANCE_BYTES:
                failures.append(
                    f"accepted schema admits {nodes} nodes / {size} bytes for "
                    f"{json.dumps(schema)[:300]}"
                )
    return _Report(rounds, accepted, documents, unions, peak, failures)


def _assert_the_run_proved_something(report: _Report) -> None:
    """A green run must also be a run that happened: enough registrable schemas, enough of them
    with the shapes that multiply, and charges big enough that a broken counter would show."""
    assert report.accepted > report.rounds // 10, f"almost nothing registrable: {report}"
    assert report.unions > report.accepted // 20, f"unions barely generated: {report}"
    assert report.peak_steps > 1000, f"only trivial documents were measured: {report}"
    assert not report.failures, "\n".join(report.failures[:3])


@pytest.mark.parametrize("seed", [1, 2])
def test_bounds_hold_on_generated_schemas(seed: int) -> None:
    _assert_the_run_proved_something(_campaign(seed, rounds=200, per_schema=6))


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(1, 13))
def test_bounds_hold_on_a_long_campaign(seed: int) -> None:
    """The full sweep. Deselected by default; run with `uv run pytest -m slow`."""
    _assert_the_run_proved_something(_campaign(seed, rounds=1200, per_schema=40))
