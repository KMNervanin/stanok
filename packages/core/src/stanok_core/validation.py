"""Validate an untrusted ToolCallProposal into a trusted ValidatedToolCall (or CallRejected).

Order: resolve the exact offered version -> coarse authorization -> schema validation -> (domain
normalization, Step 3+). Every step fails closed. The model's tool name is bound to the exact
ToolRef from the menu shown that turn, never 'latest'; trusted ids (call_id) are server-assigned.

Schema validation itself runs under a measured work budget and returns paths only, never argument
values (AD-8) - see matcher.py.
"""

import json
from collections.abc import Mapping
from uuid import uuid4

from stanok_core.matcher import ArgumentTooComplex, failing_paths
from stanok_core.registry import ToolRegistry, ToolRegistryError
from stanok_core.schema import MAX_INSTANCE_BYTES, MAX_INSTANCE_NODES, canonical_json
from stanok_core.tools import CallRejected, ToolCallProposal, ToolRef, ValidatedToolCall

_MAX_ARGS_DEPTH = 64


def _instance_too_complex(value: object) -> bool:
    """Bound the untrusted argument tree by node count and depth before validation.

    The work of validating a call is bounded by the budgets matcher.py meters, not by any figure
    derived from this count - see _Bounds in schema.py for why a per-node model cannot bound it.
    What this cap does is keep the runtime and the profile talking about the same tree: it is the
    same MAX_INSTANCE_NODES a schema must be satisfiable within, so no schema-valid call can trip
    it (B1). Iterative, so a deeply nested payload can't overflow the stack here either."""
    count = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_ARGS_DEPTH:
            return True
        count += 1
        if count > MAX_INSTANCE_NODES:
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
    if len(encoded) > MAX_INSTANCE_BYTES:
        return CallRejected(reason="invalid_arguments", detail="arguments too large")
    snapshot = json.loads(encoded)

    try:
        paths = failing_paths(spec.input_schema, snapshot)
    except ArgumentTooComplex:
        # The schema was accepted only if its own worst case fits the budget, so this means a bound
        # is wrong somewhere - refuse the call rather than keep working.
        return CallRejected(reason="invalid_arguments", detail="arguments too complex to validate")
    if paths:
        return CallRejected(
            reason="invalid_arguments", detail=f"schema errors at: {'; '.join(paths)}"[:200]
        )

    # Store the exact canonical bytes we validated; ValidatedToolCall.arguments re-parses a fresh
    # copy on read, so the trusted snapshot can't be mutated before dispatch (K1).
    return ValidatedToolCall(call_id=uuid4(), tool=ref, arguments_json=encoded)
