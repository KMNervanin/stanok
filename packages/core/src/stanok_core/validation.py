"""Validate an untrusted ToolCallProposal into a trusted ValidatedToolCall (or CallRejected).

Order: resolve the exact offered version -> coarse authorization -> schema validation -> (domain
normalization, Step 3+). Every step fails closed. The model's tool name is bound to the exact
ToolRef from the menu shown that turn, never 'latest'; trusted ids (call_id) are server-assigned.

Rejection detail carries only JSON paths, never argument values (AD-8) - jsonschema error messages
embed the offending values, so we take error.absolute_path only.
"""

import json
from collections.abc import Mapping
from itertools import islice
from uuid import uuid4

from jsonschema import Draft202012Validator, FormatChecker

from stanok_core.registry import ToolRegistry, ToolRegistryError
from stanok_core.schema import ALLOWED_FORMATS, canonical_json
from stanok_core.tools import CallRejected, ToolCallProposal, ToolRef, ValidatedToolCall

_FORMAT_CHECKER = FormatChecker()

# Fail fast: an allowed format without a real checker would silently pass (AD-8).
# The nongpl checkers (uri, hostname, date-time, ...) require jsonschema[format-nongpl].
_missing_checkers = frozenset(ALLOWED_FORMATS) - set(_FORMAT_CHECKER.checkers)
if _missing_checkers:
    raise RuntimeError(
        f"missing jsonschema format checkers {sorted(_missing_checkers)}; "
        "install jsonschema[format-nongpl]"
    )

_MAX_ARGS_BYTES = 64 * 1024
_MAX_ARGS_NODES = 1024
_MAX_ARGS_DEPTH = 64
_MAX_ERRORS = 3


def _instance_too_complex(value: object) -> bool:
    """Bound the untrusted argument tree by node count and depth before validation. Worst-case
    validation work is (instance nodes) x (schema amplification, capped in schema.py); capping the
    node count here is the other half, so a small-bytes payload can't be blown up by the schema
    (K2). Iterative, so a deeply nested payload can't overflow the stack here either."""
    count = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_ARGS_DEPTH:
            return True
        count += 1
        if count > _MAX_ARGS_NODES:
            return True
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, list):
            stack.extend((v, depth + 1) for v in node)
    return False


def validate_proposal(
    proposal: ToolCallProposal,
    offered: Mapping[str, ToolRef],
    registry: ToolRegistry,
    granted_permissions: frozenset[str],
) -> ValidatedToolCall | CallRejected:
    """Turn an untrusted proposal into a trusted call, or reject it.

    `offered` is the server-owned menu shown to the model this turn (name -> exact ToolRef); it must
    hold at most one version per name and `key == ref.name`."""
    ref = offered.get(proposal.tool)
    if ref is None or ref.name != proposal.tool:
        return CallRejected(reason="unknown_tool", detail="tool not in the offered menu")
    try:
        registered = registry.resolve_ref(ref)
    except ToolRegistryError:
        return CallRejected(reason="tool_unavailable", detail="tool unavailable or stale")

    spec = registered.spec
    if spec.required_permission not in granted_permissions:
        return CallRejected(reason="forbidden", detail="missing required permission")

    if _instance_too_complex(proposal.arguments):
        return CallRejected(reason="invalid_arguments", detail="arguments too complex")

    # Encode once, then validate exactly what we return (detached from the proposal).
    try:
        encoded = canonical_json(proposal.arguments)
    except (ValueError, TypeError, RecursionError):
        return CallRejected(reason="invalid_arguments", detail="arguments not JSON-serializable")
    if len(encoded) > _MAX_ARGS_BYTES:
        return CallRejected(reason="invalid_arguments", detail="arguments too large")
    snapshot = json.loads(encoded)

    validator = Draft202012Validator(dict(spec.input_schema), format_checker=_FORMAT_CHECKER)
    errors = list(islice(validator.iter_errors(snapshot), _MAX_ERRORS))
    if errors:
        paths = "; ".join("/".join(str(p) for p in e.absolute_path) or "(root)" for e in errors)
        return CallRejected(reason="invalid_arguments", detail=f"schema errors at: {paths}"[:200])

    # Store the exact canonical bytes we validated; ValidatedToolCall.arguments re-parses a fresh
    # copy on read, so the trusted snapshot can't be mutated before dispatch (K1).
    return ValidatedToolCall(call_id=uuid4(), tool=ref, arguments_json=encoded)
