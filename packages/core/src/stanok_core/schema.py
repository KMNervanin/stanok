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
from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaError

from stanok_core.tools import ToolSpec

_SCALAR_TYPES = frozenset({"string", "number", "integer", "boolean", "null"})
_ALL_TYPES = _SCALAR_TYPES | frozenset({"object", "array"})
_ANNOTATIONS = frozenset({"title", "description"})
_ALLOWED_FORMATS = frozenset(
    {"email", "uri", "uuid", "date", "date-time", "hostname", "ipv4", "ipv6"}
)

_NUMERIC = frozenset(
    {"enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
)
_KEYWORDS_BY_TYPE: dict[str, frozenset[str]] = {
    "string": frozenset({"enum", "const", "format", "minLength", "maxLength", "pattern"}),
    "number": _NUMERIC,
    "integer": _NUMERIC,
    "boolean": frozenset({"enum", "const"}),
    "null": frozenset(),
    "object": frozenset({"properties", "required", "additionalProperties"}),
    "array": frozenset({"items", "minItems", "maxItems", "uniqueItems"}),
}

_DEF_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_REF_RE = re.compile(r"#/\$defs/([A-Za-z_][A-Za-z0-9_.-]*)\Z")
_MAX_DEPTH = 32
_MAX_NODES = 1000
_MAX_REF_DEPTH = 32
_MAX_BYTES = 128 * 1024


class ToolSchemaError(ValueError):
    """A tool input_schema violates the Stanok authoring profile."""


def validate_tool_schema(schema: Mapping[str, Any]) -> None:
    """Validate a tool input schema against the Stanok authoring profile. Raises ToolSchemaError."""
    if not isinstance(schema, Mapping):
        raise ToolSchemaError("$: schema must be an object")
    try:
        encoded = canonical_json(schema)
    except (ValueError, TypeError) as exc:
        raise ToolSchemaError(f"schema is not JSON-serializable: {exc}") from exc
    if len(encoded) > _MAX_BYTES:
        raise ToolSchemaError(f"schema too large (> {_MAX_BYTES} bytes)")
    _preflight_complexity(schema)
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


def _preflight_complexity(schema: Any) -> None:
    count = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal count
        if depth > _MAX_DEPTH:
            raise ToolSchemaError(f"schema nesting too deep (> {_MAX_DEPTH})")
        count += 1
        if count > _MAX_NODES:
            raise ToolSchemaError(f"schema too large (> {_MAX_NODES} nodes)")
        if isinstance(node, Mapping):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(schema, 0)


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
        _check_node(items, f"{path}.items", def_names)


def _check_scalar_values(node: Mapping[str, Any], path: str, node_type: str) -> None:
    if "enum" in node or "const" in node:
        type_validator = Draft202012Validator({"type": node_type})
        for i, value in enumerate(node.get("enum", []) or []):
            if not type_validator.is_valid(value):
                raise ToolSchemaError(f"{path}.enum[{i}]: value is not a {node_type}")
        if "const" in node and not type_validator.is_valid(node["const"]):
            raise ToolSchemaError(f"{path}.const: value is not a {node_type}")
    fmt = node.get("format")
    if fmt is not None and fmt not in _ALLOWED_FORMATS:
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
    missing = set(required) - set(props)
    if missing:
        raise ToolSchemaError(f"{path}.required: names absent from properties: {sorted(missing)}")


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
    graph: dict[str, set[str]] = {}
    for name, sub in raw_defs.items():
        refs: set[str] = set()
        _collect_refs(sub, refs)
        graph[name] = refs
    color: dict[str, int] = {}

    def visit(name: str, depth: int) -> None:
        if depth > _MAX_REF_DEPTH:
            raise ToolSchemaError(f"$.$defs: reference chain too deep (> {_MAX_REF_DEPTH})")
        color[name] = 1  # gray: on the current path
        for target in graph.get(name, ()):
            if color.get(target) == 1:
                raise ToolSchemaError(f"$.$defs: reference cycle through {target!r}")
            if color.get(target) is None and target in graph:
                visit(target, depth + 1)
        color[name] = 2  # black: fully explored

    for name in graph:
        if color.get(name) is None:
            visit(name, 0)


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


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, UTF-8, no NaN/Infinity, compact."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def digest_spec(spec: ToolSpec) -> str:
    """Content digest over the whole ToolSpec (name, version, effect, permission, schema, ...)."""
    payload = canonical_json(spec.model_dump(mode="json"))
    return "sha256:" + hashlib.sha256(payload).hexdigest()
