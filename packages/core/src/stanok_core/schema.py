"""Tool-argument schema validation (Stanok authoring profile) and canonical spec digest.

The profile is a positive grammar (default-deny): a subschema is accepted only if it is one of a
few known-safe shapes — a scalar type, a closed object, a bounded array, a local $ref, or an anyOf
of those. Everything else is rejected until conformance proves the backend supports it. The profile
also bounds size/depth and forbids $ref cycles so a hostile schema can't exhaust the validator
(AD-8, AD-9, AD-10).
"""

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from typing import Any, NamedTuple

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaError

from stanok_core.tools import ToolSpec

_SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})
_ALL_TYPES = _SCALAR_TYPES | frozenset({"object", "array"})
_ANNOTATIONS = frozenset({"title", "description"})
ALLOWED_FORMATS = frozenset(
    {"email", "uri", "uuid", "date", "date-time", "hostname", "ipv4", "ipv6"}
)

_NUMERIC = frozenset(
    {"enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
)
_KEYWORDS_BY_TYPE: dict[str, frozenset[str]] = {
    # `pattern` is intentionally excluded: an author-supplied regex is a ReDoS vector on the
    # untrusted boundary (CPython `re` backtracks and holds the GIL, so a timeout can't preempt it).
    # Reintroduce only behind a linear engine (e.g. re2). Bounded strings use enum/const/format/len.
    "string": frozenset({"enum", "const", "format", "minLength", "maxLength"}),
    "number": _NUMERIC,
    "integer": _NUMERIC,
    "boolean": frozenset({"enum", "const"}),
    "null": frozenset(),
    "object": frozenset({"properties", "required", "additionalProperties"}),
    # `uniqueItems` is excluded: on non-hashable (composite) items jsonschema falls back to an
    # O(n^2) pairwise scan, so an in-limit array turns into millions of comparisons. Dedup, if
    # needed, belongs in domain logic (Step 3), not in the untrusted-boundary schema.
    "array": frozenset({"items", "minItems", "maxItems"}),
}

# Every keyword the profile can let through, so matcher.py can prove it meters each one that
# descends into a subschema.
ALLOWED_KEYWORDS = (
    frozenset({"type", "$ref", "anyOf", "$defs"})
    | _ANNOTATIONS
    | frozenset().union(*_KEYWORDS_BY_TYPE.values())
)

_DEF_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_REF_RE = re.compile(r"#/\$defs/([A-Za-z_][A-Za-z0-9_.-]*)\Z")
_MAX_DEPTH = 32
_MAX_NODES = 1000
_MAX_BYTES = 128 * 1024
# Normative limits of the boundary. They are enforced twice, from both ends: matcher.py measures the
# real work of a call against them, and a schema is accepted here only if its own worst case fits
# inside them. That second half is what keeps the published contract executable - a schema-valid
# call can never be turned away for exceeding a limit its own schema was allowed to promise (B1).
MAX_INSTANCE_NODES = 4096  # largest argument tree accepted, at registration and at runtime
MAX_INSTANCE_BYTES = 64 * 1024  # largest canonical encoding of an argument tree
# The two runtime budgets, calibrated by measurement rather than derived from a per-node figure -
# an earlier version defined them as MAX_INSTANCE_NODES x a per-node cost, which is not an upper
# bound at all (see _Bounds). A descent costs ~5 us, since it builds a sub-validator; the bulk
# scans that `steps` counts run in C and cost ~0.04 us each. So descents are what bounds wall time:
# 65536 of them is ~0.33 s, and the step budget adds ~40 ms on top. Both were measured on
# pydantic 2.13.4 / jsonschema 4.26.0 and want re-measuring if either is upgraded.
MAX_DESCENTS = 65_536
MAX_VALIDATION_STEPS = 1_048_576

# Upper bound on one code point in canonical JSON: 4 bytes of UTF-8, or 6 for a \uXXXX escape.
_BYTES_PER_CHAR = 6
# Room for a float's mantissa: canonical JSON writes repr(float), never longer than this.
_FLOAT_BYTES = 24

_MAX_ARRAY_ITEMS = MAX_INSTANCE_NODES - 2  # leaves room for the root object and the array itself
_MAX_EXPANDED_DEPTH = 64


class ToolSchemaError(ValueError):
    """A tool input_schema violates the Stanok authoring profile."""


def validate_tool_schema(schema: Mapping[str, Any]) -> None:
    """Validate a tool input schema against the Stanok authoring profile. Raises ToolSchemaError."""
    if not isinstance(schema, Mapping):
        raise ToolSchemaError("$: schema must be an object")
    # Structural bound FIRST, and iteratively: cap depth/nodes before json.dumps or check_schema
    # recurse into a hostile schema and raise a RecursionError we couldn't attribute to the input.
    _preflight_complexity(schema)
    try:
        encoded = canonical_json(schema)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ToolSchemaError(f"schema is not JSON-serializable: {exc}") from exc
    if len(encoded) > _MAX_BYTES:
        raise ToolSchemaError(f"schema too large (> {_MAX_BYTES} bytes)")
    try:
        Draft202012Validator.check_schema(schema)
    except JsonSchemaError as exc:
        raise ToolSchemaError(f"invalid JSON Schema: {exc.message}") from exc
    if "$schema" in schema:
        raise ToolSchemaError("$: put dialect in ToolSpec.schema_dialect, not in input_schema")

    raw_defs = schema.get("$defs")
    def_names: set[str] = set()
    if raw_defs is not None:
        if not isinstance(raw_defs, Mapping):
            raise ToolSchemaError("$.$defs: must be an object")
        for name in raw_defs:
            if not _DEF_NAME_RE.fullmatch(name):
                raise ToolSchemaError(f"$.$defs: unsafe definition name {name!r}")
        def_names = set(raw_defs)

    if schema.get("type") != "object":
        raise ToolSchemaError("$: root schema must declare type: object")
    _check_node(schema, "$", def_names, at_root=True)
    for name, sub in (raw_defs or {}).items():
        _check_node(sub, f"$.$defs.{name}", def_names)
    if raw_defs:
        _check_no_ref_cycles(raw_defs)
    _check_instance_bounds(schema, raw_defs if isinstance(raw_defs, Mapping) else {})


def _preflight_complexity(schema: Any) -> None:
    count = 0
    stack: list[tuple[Any, int]] = [(schema, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_DEPTH:
            raise ToolSchemaError(f"schema nesting too deep (> {_MAX_DEPTH})")
        count += 1
        if count > _MAX_NODES:
            raise ToolSchemaError(f"schema too large (> {_MAX_NODES} nodes)")
        if isinstance(node, Mapping):
            for value in node.values():
                stack.append((value, depth + 1))
        elif isinstance(node, list):
            for value in node:
                stack.append((value, depth + 1))


def _check_node(node: Any, path: str, def_names: set[str], *, at_root: bool = False) -> None:
    if isinstance(node, bool):
        raise ToolSchemaError(f"{path}: boolean schema is not allowed")
    if not isinstance(node, Mapping):
        raise ToolSchemaError(f"{path}: schema must be an object")
    keys = set(node)

    if "$ref" in node:
        _check_ref(node, path, keys, def_names)
        return
    if "anyOf" in node:
        _check_any_of(node, path, keys, def_names)
        return

    node_type = node.get("type")
    if not isinstance(node_type, str) or node_type not in _ALL_TYPES:
        raise ToolSchemaError(f"{path}: node must declare a single supported type")

    allowed = {"type"} | _ANNOTATIONS | _KEYWORDS_BY_TYPE[node_type]
    if at_root:
        allowed.add("$defs")
    extra = keys - allowed
    if extra:
        raise ToolSchemaError(f"{path}: unsupported keywords for type {node_type}: {sorted(extra)}")

    if node_type in _SCALAR_TYPES:
        _check_scalar_values(node, path, node_type)
    elif node_type == "object":
        _check_object(node, path, def_names)
    else:  # array
        items = node.get("items")
        if items is None:
            raise ToolSchemaError(f"{path}: array must declare items")
        max_items = node.get("maxItems")
        # An array without maxItems is not bounded (V1); type() excludes bool (a subclass of int).
        if type(max_items) is not int or not 0 <= max_items <= _MAX_ARRAY_ITEMS:
            raise ToolSchemaError(f"{path}: array must set maxItems in [0, {_MAX_ARRAY_ITEMS}]")
        _check_node(items, f"{path}.items", def_names)


def _check_scalar_values(node: Mapping[str, Any], path: str, node_type: str) -> None:
    if "enum" in node or "const" in node:
        # enum length is bounded by _check_validation_budget: a scalar enum is scanned per value,
        # so its length is exactly its per-value amplification and is capped there.
        type_validator = Draft202012Validator({"type": node_type})
        for i, value in enumerate(node.get("enum", []) or []):
            if not type_validator.is_valid(value):
                raise ToolSchemaError(f"{path}.enum[{i}]: value is not a {node_type}")
        if "const" in node and not type_validator.is_valid(node["const"]):
            raise ToolSchemaError(f"{path}.const: value is not a {node_type}")
    fmt = node.get("format")
    if fmt is not None and fmt not in ALLOWED_FORMATS:
        raise ToolSchemaError(f"{path}.format: {fmt!r} is not an allowed format")


def _check_object(node: Mapping[str, Any], path: str, def_names: set[str]) -> None:
    if node.get("additionalProperties") is not False:
        raise ToolSchemaError(f"{path}: object must set additionalProperties: false")
    props = node.get("properties", {})
    if not isinstance(props, Mapping):
        raise ToolSchemaError(f"{path}.properties: must be an object")
    for name, sub in props.items():
        _check_node(sub, f"{path}.properties.{name}", def_names)
    required = node.get("required", [])
    if not isinstance(required, list) or not all(isinstance(r, str) for r in required):
        raise ToolSchemaError(f"{path}.required: must be a list of strings")
    # Every property must be required. An optional property (default / default_factory) would let
    # a missing argument be silently masked; nullable is expressed as `T | None`, still required.
    if set(required) != set(props):
        optional = sorted(set(props) - set(required))
        unknown = sorted(set(required) - set(props))
        raise ToolSchemaError(
            f"{path}: every property must be required; optional: {optional}, unknown: {unknown}"
        )


def _check_ref(node: Mapping[str, Any], path: str, keys: set[str], def_names: set[str]) -> None:
    extra = keys - {"$ref"} - _ANNOTATIONS
    if extra:
        raise ToolSchemaError(f"{path}: $ref must stand alone, extra keys: {sorted(extra)}")
    ref = node["$ref"]
    match = _REF_RE.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        raise ToolSchemaError(f"{path}: $ref must be '#/$defs/<name>', got {ref!r}")
    if match.group(1) not in def_names:
        raise ToolSchemaError(f"{path}: $ref to unknown $defs/{match.group(1)}")


def _check_any_of(node: Mapping[str, Any], path: str, keys: set[str], def_names: set[str]) -> None:
    extra = keys - {"anyOf"} - _ANNOTATIONS
    if extra:
        raise ToolSchemaError(f"{path}: anyOf must stand alone, extra keys: {sorted(extra)}")
    branches = node["anyOf"]
    if not isinstance(branches, list) or not branches:
        raise ToolSchemaError(f"{path}.anyOf: must be a non-empty list")
    for i, branch in enumerate(branches):
        _check_node(branch, f"{path}.anyOf[{i}]", def_names)


def _check_no_ref_cycles(raw_defs: Mapping[str, Any]) -> None:
    """Reject $ref cycles only, with an explicit stack so a long chain can't overflow Python's
    recursion. Reference-chain depth and fan-out cost are bounded separately over the resulting DAG
    (_check_validation_budget); a DFS depth counter here would undercount, because it never
    re-descends through a node already marked fully explored, so it must not be relied on."""
    graph: dict[str, set[str]] = {}
    for name, sub in raw_defs.items():
        refs: set[str] = set()
        _collect_refs(sub, refs)
        graph[name] = {r for r in refs if r in raw_defs}
    color: dict[str, int] = {}  # 1 = gray (on current path), 2 = black (fully explored)
    for start in graph:
        if color.get(start):
            continue
        color[start] = 1
        stack: list[tuple[str, Iterator[str]]] = [(start, iter(graph[start]))]
        while stack:
            name, targets = stack[-1]
            descended = False
            for target in targets:
                hue = color.get(target)
                if hue == 1:
                    raise ToolSchemaError(f"$.$defs: reference cycle through {target!r}")
                if hue is None:
                    color[target] = 1
                    stack.append((target, iter(graph[target])))
                    descended = True
                    break
            if not descended:
                color[name] = 2
                stack.pop()


def _collect_refs(node: Any, acc: set[str]) -> None:
    if isinstance(node, Mapping):
        ref = node.get("$ref")
        if isinstance(ref, str):
            match = _REF_RE.fullmatch(ref)
            if match:
                acc.add(match.group(1))
        for value in node.values():
            _collect_refs(value, acc)
    elif isinstance(node, list):
        for value in node:
            _collect_refs(value, acc)


def _sat(value: int, ceiling: int) -> int:
    """Clamp one past the ceiling: exact below it, always rejected at or above it. Keeps an
    exponential worst case from growing unbounded before we refuse the schema."""
    return value if value <= ceiling else ceiling + 1


class _Bounds(NamedTuple):
    """What one subschema can demand of the validator.

    `steps` and `descents` are TOTALS: the whole amount matcher.py charges while checking the
    largest document this subschema accepts, counted the same way it charges. They are not per-node
    figures. An earlier version propagated a per-node maximum and multiplied it by the node budget;
    that is not an upper bound, because `anyOf` branches sharing a child re-validate that child once
    per branch, so a node nested under L levels of B-way union is visited B**L times. Measured:
    B=6, L=6 charges 494550 steps on a 91-byte document while the per-node view scored it at 37."""

    nodes: int
    size: int
    steps: int
    descents: int
    depth: int
    keys: int  # widest object a value here can be, for the additionalProperties scan
    items: int  # longest array a value here can be, for the per-element walk


# Used where a subschema is unusable or unreachable, so any such path fails closed.
_UNSATISFIABLE = _Bounds(
    nodes=MAX_INSTANCE_NODES + 1,
    size=MAX_INSTANCE_BYTES + 1,
    steps=MAX_VALIDATION_STEPS + 1,
    descents=MAX_DESCENTS + 1,
    depth=0,
    keys=0,
    items=0,
)


def _enforce_bounds(bounds: _Bounds) -> None:
    if bounds.depth > _MAX_EXPANDED_DEPTH:
        raise ToolSchemaError(f"$: expanded validation depth too deep (> {_MAX_EXPANDED_DEPTH})")
    if bounds.descents > MAX_DESCENTS:
        raise ToolSchemaError(
            f"$: validating one call would enter more than {MAX_DESCENTS} subschemas; "
            "narrow the unions, shorten $ref chains, or lower maxItems"
        )
    if bounds.steps > MAX_VALIDATION_STEPS:
        raise ToolSchemaError(
            f"$: validating one call would take more than {MAX_VALIDATION_STEPS} steps; "
            "reduce nested unions, object width, enum length, or maxItems"
        )
    if bounds.nodes > MAX_INSTANCE_NODES:
        raise ToolSchemaError(
            f"$: schema admits arguments larger than {MAX_INSTANCE_NODES} nodes; "
            "lower maxItems or drop fields, so every schema-valid call stays executable"
        )
    if bounds.size > MAX_INSTANCE_BYTES:
        raise ToolSchemaError(
            f"$: schema admits arguments larger than {MAX_INSTANCE_BYTES} bytes; "
            "bound every string with maxLength (or enum/const) and every number with "
            "minimum and maximum, and lower maxItems, so every schema-valid call stays executable"
        )


def _defs_reverse_topo(raw_defs: Mapping[str, Any]) -> list[str]:
    """Reverse-topological order of $defs (each def after all defs it references), computed with an
    explicit stack so a long chain can't overflow Python's recursion. The graph is acyclic
    (_check_no_ref_cycles ran); back-edges into a def already being processed are simply skipped."""
    graph: dict[str, set[str]] = {}
    for name, sub in raw_defs.items():
        refs: set[str] = set()
        _collect_refs(sub, refs)
        graph[name] = {r for r in refs if r in raw_defs}
    order: list[str] = []
    done: set[str] = set()
    for root in graph:
        if root in done:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            name, exiting = stack.pop()
            if exiting:
                if name not in done:
                    done.add(name)
                    order.append(name)
                continue
            if name in done:
                continue
            stack.append((name, True))
            for target in graph[name]:
                if target not in done:
                    stack.append((target, False))
    return order


def _check_instance_bounds(schema: Mapping[str, Any], raw_defs: Mapping[str, Any]) -> _Bounds:
    """Bound what a schema can demand of the validator - by totalling it, not estimating (K2b/B1).

    Two earlier versions of this pass scored a PER-NODE figure and multiplied it by the node budget.
    Both were wrong in the same way: the multiplier is not the node count. `anyOf` branches that
    share a child re-check that child once per branch, so nesting L levels of B-way union makes one
    deep node cost B**L, which no per-node figure can express. This version instead adds up exactly
    what matcher.py charges, mirroring it keyword for keyword, and each quantity is a real total:

    * nodes, size - the largest argument tree the schema can ACCEPT, in nodes and in canonical
      bytes. Every object is closed with all properties required, every array carries maxItems and
      every scalar is bounded, so both are exact. Requiring them to fit the runtime caps is what
      keeps the contract published to the model executable: a schema-valid call can no longer be
      refused for exceeding a limit its own schema was allowed to promise (B1).
    * steps, descents - the work of validating that largest accepted document. `own` is what this
      keyword charges once; children add their own totals, an array multiplies its items by
      maxItems, and an anyOf ADDS UP its branches because each is tried against the same value.
      Requiring these to fit the runtime budgets is the other half of the same guarantee, and the
      budgets are calibrated from measurement (see MAX_VALIDATION_STEPS).
    * depth - nesting once $refs are followed, so a long chain cannot drive jsonschema's reference
      resolution into a RecursionError.

    $defs are measured leaves-first (reverse topological order) so a `$ref` is a memo lookup, never
    a recursive descent into another def: measure() then recurses only through inline structure,
    already capped at _MAX_DEPTH, so this pass cannot itself raise RecursionError on a long chain.
    """
    memo: dict[str, _Bounds] = {}
    # widest object and longest array anywhere inside each def, refs expanded
    reach: dict[str, tuple[int, int]] = {}

    def widest(node: Mapping[str, Any], of_arrays: bool) -> int:
        """Largest value any subschema in here accepts: keys of an object, or length of an array.

        A branch of an anyOf is run against the value the MATCHING branch produced, not against one
        of its own - so it scans that value's keys and walks that value's elements, however narrow
        its own declaration is. The document is globally valid, so the value at a position is
        bounded by whatever sibling accepts it, which is what this returns."""
        ref = node.get("$ref")
        if isinstance(ref, str):
            match = _REF_RE.fullmatch(ref)
            if match is None or match.group(1) not in reach:
                return 0
            return reach[match.group(1)][1 if of_arrays else 0]
        branches = node.get("anyOf")
        if isinstance(branches, list):
            per_branch = (widest(b, of_arrays) for b in branches if isinstance(b, Mapping))
            return max(per_branch, default=0)
        node_type = node.get("type")
        if node_type == "object":
            props = node.get("properties", {})
            props = props if isinstance(props, Mapping) else {}
            inner = max(
                (widest(v, of_arrays) for v in props.values() if isinstance(v, Mapping)), default=0
            )
            return inner if of_arrays else max(len(props), inner)
        if node_type == "array":
            items = node.get("items", {})
            inner = widest(items, of_arrays) if isinstance(items, Mapping) else 0
            max_items = node.get("maxItems")
            own = max_items if type(max_items) is int else 0
            return max(own, inner) if of_arrays else inner
        return 0

    def measure(node: Mapping[str, Any], floor_keys: int = 0, floor_items: int = 0) -> _Bounds:
        ref = node.get("$ref")
        if isinstance(ref, str):
            match = _REF_RE.fullmatch(ref)
            if match is None:  # unreachable: _check_ref ran first. Fail closed if it ever is.
                return _UNSATISFIABLE
            target = memo.get(match.group(1), _UNSATISFIABLE)
            if floor_keys or floor_items:
                # the def was measured with no envelope; under one, every object inside it scans a
                # wider value and every array walks a longer one, so it has to be measured again
                sub = raw_defs.get(match.group(1))
                if isinstance(sub, Mapping):
                    target = measure(sub, floor_keys, floor_items)
            return _Bounds(
                nodes=target.nodes,
                size=target.size,
                steps=_sat(1 + target.steps, MAX_VALIDATION_STEPS),
                descents=_sat(1 + target.descents, MAX_DESCENTS),
                depth=target.depth + 1,
                keys=target.keys,
                items=target.items,
            )
        branches = node.get("anyOf")
        if isinstance(branches, list):
            # a branch that does not match still scans additionalProperties over the value the
            # matching branch produced, so every branch is charged for the widest value here
            wide_keys = max(floor_keys, widest(node, of_arrays=False))
            wide_items = max(floor_items, widest(node, of_arrays=True))
            measured = [
                measure(b, wide_keys, wide_items) for b in branches if isinstance(b, Mapping)
            ]
            # every branch is tried against the same value, so their work adds up
            return _Bounds(
                nodes=max((m.nodes for m in measured), default=1),
                size=max((m.size for m in measured), default=2),
                steps=_sat(len(measured) + sum(m.steps for m in measured), MAX_VALIDATION_STEPS),
                descents=_sat(len(measured) + sum(m.descents for m in measured), MAX_DESCENTS),
                depth=max((m.depth for m in measured), default=0) + 1,
                keys=wide_keys,
                items=wide_items,
            )
        node_type = node.get("type")
        if node_type == "object":
            props = node.get("properties", {})
            props = props if isinstance(props, Mapping) else {}
            required = node.get("required", [])
            required = required if isinstance(required, list) else []
            # charged once per visit: the additionalProperties scan over the value's keys (which
            # equal the declared ones for an accepted value), the properties descent, and one
            # check per required key - the last happens even when the value arrives empty
            nodes, size = 1, 2
            scanned = max(len(props), floor_keys)  # additionalProperties walks the value's keys
            steps, descents, depth = len(props) + scanned + len(required), 1, 0
            for name, sub in props.items():
                if isinstance(sub, Mapping):
                    m = measure(sub, floor_keys, floor_items)
                    nodes = _sat(nodes + m.nodes, MAX_INSTANCE_NODES)
                    # "name": value, - quotes, colon and separator around each property
                    size = _sat(size + len(name) * _BYTES_PER_CHAR + 4 + m.size, MAX_INSTANCE_BYTES)
                    steps = _sat(steps + m.steps, MAX_VALIDATION_STEPS)
                    descents = _sat(descents + m.descents, MAX_DESCENTS)
                    depth = max(depth, m.depth)
            return _Bounds(nodes, size, steps, descents, depth + 1, len(props), 0)
        if node_type == "array":
            items = node.get("items", {})
            m = (
                measure(items, floor_keys, floor_items)
                if isinstance(items, Mapping)
                else _UNSATISFIABLE
            )
            max_items = node.get("maxItems")
            accepted = max_items if type(max_items) is int else 0
            # every element of the value is walked before maxItems can reject it, so a branch pays
            # for the length the position can actually hold, not for its own declaration
            count = max(accepted, floor_items)
            return _Bounds(
                nodes=_sat(1 + accepted * m.nodes, MAX_INSTANCE_NODES),
                size=_sat(2 + accepted * (m.size + 1), MAX_INSTANCE_BYTES),
                # the items descent is charged once per element, and so is each element's own work
                steps=_sat(count + count * m.steps, MAX_VALIDATION_STEPS),
                descents=_sat(1 + count * m.descents, MAX_DESCENTS),
                depth=m.depth + 1,
                keys=0,
                items=accepted,
            )
        enum = node.get("enum")
        return _Bounds(
            nodes=1,
            size=_scalar_size(node, node_type),
            steps=len(enum) if isinstance(enum, list) else 0,  # only `enum` is charged on a scalar
            descents=0,
            depth=1,
            keys=0,
            items=0,
        )

    for name in _defs_reverse_topo(raw_defs):
        sub = raw_defs.get(name)
        if isinstance(sub, Mapping):
            reach[name] = (widest(sub, of_arrays=False), widest(sub, of_arrays=True))
        memo[name] = measure(sub) if isinstance(sub, Mapping) else _UNSATISFIABLE
        _enforce_bounds(memo[name])
    root = measure(schema)
    _enforce_bounds(root)
    return root


def _scalar_size(node: Mapping[str, Any], node_type: Any) -> int:
    """Largest canonical encoding of a scalar the subschema accepts, saturating when unbounded."""
    if "const" in node:
        return _sat(len(canonical_json(node["const"])), MAX_INSTANCE_BYTES)
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return _sat(max(len(canonical_json(v)) for v in enum), MAX_INSTANCE_BYTES)
    if node_type == "string":
        max_length = node.get("maxLength")
        if type(max_length) is not int:  # an unbounded string makes the schema unsatisfiable
            return MAX_INSTANCE_BYTES + 1
        return _sat(2 + max_length * _BYTES_PER_CHAR, MAX_INSTANCE_BYTES)
    if node_type == "boolean":
        return 5
    if node_type == "null":
        return 4
    # A JSON number carries no length limit of its own, and `number` accepts integers too, so both
    # ends must be pinned before its encoding can be bounded at all.
    lower = [
        node[k] for k in ("minimum", "exclusiveMinimum") if isinstance(node.get(k), int | float)
    ]
    upper = [
        node[k] for k in ("maximum", "exclusiveMaximum") if isinstance(node.get(k), int | float)
    ]
    if not (lower and upper):
        return MAX_INSTANCE_BYTES + 1
    widest = max(len(canonical_json(v)) for v in (*lower, *upper))
    # a float inside those bounds still needs room for its mantissa
    return _sat(widest if node_type == "integer" else max(widest, _FLOAT_BYTES), MAX_INSTANCE_BYTES)


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, UTF-8, no NaN/Infinity, compact."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest_spec(spec: ToolSpec) -> str:
    """Content digest over the whole ToolSpec (name, version, effect, permission, schema, ...)."""
    payload = canonical_json(spec.model_dump(mode="json"))
    return "sha256:" + hashlib.sha256(payload).hexdigest()
