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
from typing import Any

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

_DEF_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_REF_RE = re.compile(r"#/\$defs/([A-Za-z_][A-Za-z0-9_.-]*)\Z")
_MAX_DEPTH = 32
_MAX_NODES = 1000
_MAX_BYTES = 128 * 1024
_MAX_ARRAY_ITEMS = 4096
# Validation work has two bounds. (1) Cost of validating ONE maximally-failing value, per node: a
# scalar is 1 (+ enum length), an object is 1 + its required count + the sum of its properties (all
# `required` keys are checked even against an empty object), an array is 1 + its item cost, an
# `anyOf` is the sum over its branches (so a wide object reused via `$ref` under `anyOf` adds up),
# a `$ref` is its target. Capped by _MAX_SCHEMA_COST. (2) An array's items are validated once per
# element and the element count is attacker-controlled up to the instance-node cap (validation.py),
# so each array's item cost is separately capped by _MAX_ARRAY_ITEM_AMP. Expanded depth caps every
# nesting level so a deep/wrapped schema can't drive deep recursion. All three want a real benchmark
# before they are treated as final.
_MAX_SCHEMA_COST = 8192
_MAX_ARRAY_ITEM_AMP = 64
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
    _check_validation_budget(schema, raw_defs if isinstance(raw_defs, Mapping) else {})


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
            raise ToolSchemaError(
                f"{path}: array must set maxItems in [0, {_MAX_ARRAY_ITEMS}]"
            )
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


def _sat(x: int) -> int:
    """Saturating clamp so exponential worst cases don't grow unbounded before we reject them."""
    return x if x < _MAX_SCHEMA_COST else _MAX_SCHEMA_COST


def _enforce_budget(depth: int, cost: int) -> None:
    if depth > _MAX_EXPANDED_DEPTH:
        raise ToolSchemaError(f"$: expanded validation depth too deep (> {_MAX_EXPANDED_DEPTH})")
    if cost >= _MAX_SCHEMA_COST:
        raise ToolSchemaError(
            f"$: single-value validation cost too high (>= {_MAX_SCHEMA_COST}); "
            "reduce anyOf fan-out, $ref reuse, object width, or nesting"
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


def _check_validation_budget(schema: Mapping[str, Any], raw_defs: Mapping[str, Any]) -> None:
    """Bound the single-value validation cost and the expanded validation depth (K2).

    Cost is the number of subschema checks jsonschema runs to validate ONE maximally-failing value:
    a scalar enum re-checks up to len(enum) values; an object checks every `required` key (even
    against `{}`) and descends into every property; an `anyOf` validates every branch in full (so a
    wide object reused via `$ref` under `anyOf` adds up); an array is its item cost. Capping the
    root's cost stops a small payload from forcing one huge validation. Separately, an array's items
    are validated once per element and the element count is attacker-controlled up to the instance
    cap (validation.py), so each array's item cost is bounded by _MAX_ARRAY_ITEM_AMP; together with
    that cap, an array of N elements can cost at most N x _MAX_ARRAY_ITEM_AMP. Expanded depth counts
    every nesting level (object/array/anyOf/$ref) so a long chain or deeply wrapped ref is refused.

    $defs are evaluated leaves-first (reverse topological order) so a `$ref` is a memo lookup, never
    a recursive descent into another def: node_measure then recurses only through inline structure,
    already capped at _MAX_DEPTH, so this pass can't itself raise RecursionError on a long chain.
    """
    cost_memo: dict[str, int] = {}
    depth_memo: dict[str, int] = {}

    def node_measure(node: Mapping[str, Any]) -> tuple[int, int]:
        """Return (single-value validation cost, expanded_depth) for one node."""
        ref = node.get("$ref")
        if isinstance(ref, str):
            match = _REF_RE.fullmatch(ref)
            if match is None:
                return 1, 1
            name = match.group(1)
            return cost_memo.get(name, 1), depth_memo.get(name, 0) + 1
        branches = node.get("anyOf")
        if isinstance(branches, list):
            cost, max_d = 1, 0
            for branch in branches:
                if isinstance(branch, Mapping):
                    c, d = node_measure(branch)
                    cost, max_d = _sat(cost + c), max(max_d, d)
            return cost, max_d + 1
        node_type = node.get("type")
        if node_type == "object":
            props = node.get("properties", {})
            required = node.get("required", [])
            cost = 1 + (len(required) if isinstance(required, list) else 0)
            max_d = 0
            if isinstance(props, Mapping):
                for sub in props.values():
                    if isinstance(sub, Mapping):
                        c, d = node_measure(sub)
                        cost, max_d = _sat(cost + c), max(max_d, d)
            return cost, max_d + 1
        if node_type == "array":
            items = node.get("items", {})
            c, d = node_measure(items) if isinstance(items, Mapping) else (1, 0)
            if c >= _MAX_ARRAY_ITEM_AMP:
                # items are validated once per attacker-controlled element — cap the per-item cost
                raise ToolSchemaError(
                    f"$: array item too costly to validate per element (>= {_MAX_ARRAY_ITEM_AMP})"
                )
            return _sat(1 + c), d + 1
        enum = node.get("enum")
        if isinstance(enum, list):  # a scalar enum is scanned per value
            return _sat(1 + len(enum)), 1
        return 1, 1  # scalar leaf

    for name in _defs_reverse_topo(raw_defs):
        sub = raw_defs.get(name)
        cost, depth = node_measure(sub) if isinstance(sub, Mapping) else (1, 1)
        cost_memo[name], depth_memo[name] = cost, depth
        _enforce_budget(depth, cost)
    root_cost, root_depth = node_measure(schema)
    _enforce_budget(root_depth, root_cost)


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, UTF-8, no NaN/Infinity, compact."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest_spec(spec: ToolSpec) -> str:
    """Content digest over the whole ToolSpec (name, version, effect, permission, schema, ...)."""
    payload = canonical_json(spec.model_dump(mode="json"))
    return "sha256:" + hashlib.sha256(payload).hexdigest()
