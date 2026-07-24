"""Trusted tool-boundary contracts.

Untrusted model proposal -> server-side validation -> trusted call -> typed outcome.
The model picks a tool and proposes arguments; call_id, version, digest and outcome
are assigned by our code (AD-8).

Types only. Schema validation, registry resolution and building a
ValidatedToolCall are a separate step.
"""

import json
from enum import StrEnum
from typing import Annotated, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue

# Tool arguments are always a JSON object of valid JSON values, never arbitrary Python objects.
JsonObject = dict[str, JsonValue]

SchemaDialect = Literal["https://json-schema.org/draft/2020-12/schema"]
SchemaProfile = Literal["stanok.tool-args.v1"]

SCHEMA_DIALECT_2020_12: SchemaDialect = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_PROFILE_V1: SchemaProfile = "stanok.tool-args.v1"

# Constrained identifiers so empty/garbage values fail at parse time.
# ToolName is a resolved name (pattern); RawToolName is the model's raw, unresolved proposal.
ToolName = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_]*$")]
RawToolName = Annotated[str, Field(min_length=1, max_length=100)]
Permission = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.]*$")]
ErrorCode = Annotated[str, Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.]*$")]
Version = Annotated[int, Field(gt=0)]
SpecDigest = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
ShortText = Annotated[str, Field(min_length=1, max_length=200)]
Description = Annotated[str, Field(min_length=1, max_length=1000)]


class ContractModel(BaseModel):
    """Base for boundary contracts: no extra fields, immutable, non-finite numbers rejected.

    hide_input_in_errors keeps offending argument values out of ValidationError *messages* (AD-7/8).
    It does NOT scrub the `input` field of `.errors()`, so the ingress parser (Step 3) must still
    map validation failures to value-free loc/type and never forward raw `.errors()` outward."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, hide_input_in_errors=True
    )


class Effect(StrEnum):
    """Coarse tool effect. Not sufficient as the only policy signal:
    a read tool over PII still needs an explicit permission (see ToolSpec.required_permission)."""

    read = "read"
    write = "write"


class ToolSpec(ContractModel):
    """Tool passport: the menu shown to the model and the NORMATIVE argument schema.

    input_schema is server-side truth; never silently narrow it to the backend."""

    name: ToolName
    version: Version
    description: Description
    effect: Effect
    required_permission: Permission
    input_schema: JsonObject
    schema_dialect: SchemaDialect = SCHEMA_DIALECT_2020_12
    schema_profile: SchemaProfile = SCHEMA_PROFILE_V1


class ToolRef(ContractModel):
    """Reference to a specific tool version. Goes into the audit log and the trusted call.

    Registry must keep an immutable spec snapshot per digest (else it is unresolvable)."""

    name: ToolName
    version: Version
    spec_digest: SpecDigest


class ToolCallProposal(ContractModel):
    """Untrusted model PROPOSAL. Not yet checked against schema, permissions, registry (AD-8)."""

    tool: RawToolName
    arguments: JsonObject


class ValidatedToolCall(ContractModel):
    """Trusted call after validation. call_id and tool are server-assigned, not from the model.

    The trusted payload must not change between validation and execution. pydantic `frozen` is not
    deep, so we hold the exact canonical JSON bytes that were validated and rebuild a fresh parsed
    copy on each `arguments` access — a caller mutating the returned dict cannot alter the snapshot,
    and dispatch executes exactly `arguments_json`."""

    call_id: UUID
    tool: ToolRef
    arguments_json: bytes

    @property
    def arguments(self) -> JsonObject:
        return cast(JsonObject, json.loads(self.arguments_json))


class CallRejected(ContractModel):
    """A proposal rejected before dispatch; detail is bounded, value-free (no raw args)."""

    reason: Literal["unknown_tool", "tool_unavailable", "invalid_arguments", "forbidden"]
    detail: ShortText


class ToolError(ContractModel):
    """Structured error, safe for the audit log and for the model.
    No raw exception text and no document content (AD-7, AD-8)."""

    code: ErrorCode
    message: ShortText
    retryable: bool = False


class ToolSucceeded(ContractModel):
    # `data` is parsed from JSON as part of ExecutionOutcome, so it stays a plain object here; its
    # canonical-bytes freezing lands with the dispatcher that first builds it (Step 3), same as
    # ValidatedToolCall.arguments_json — no code constructs a ToolSucceeded yet.
    status: Literal["succeeded"] = "succeeded"
    call_id: UUID
    data: JsonObject


class ToolFailed(ContractModel):
    status: Literal["failed"] = "failed"
    call_id: UUID
    error: ToolError


class ToolUnknown(ContractModel):
    """Written, but unknown if it applied (AD-6). Reconciliation only, never auto-retry."""

    status: Literal["unknown"] = "unknown"
    call_id: UUID
    reconciliation_ref: ShortText


# Execution outcome — exactly one of three states, discriminated by `status`.
# Contradictory combinations (success with an error, etc.) are unrepresentable by design.
ExecutionOutcome = Annotated[
    ToolSucceeded | ToolFailed | ToolUnknown, Field(discriminator="status")
]
