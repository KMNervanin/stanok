from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError
from stanok_core.tools import (
    Effect,
    ExecutionOutcome,
    ToolCallProposal,
    ToolError,
    ToolFailed,
    ToolRef,
    ToolSpec,
    ToolSucceeded,
)

_OUTCOME: TypeAdapter[ExecutionOutcome] = TypeAdapter(ExecutionOutcome)
_VALID_DIGEST = "sha256:" + "a" * 64


def test_toolspec_has_default_dialect_and_profile() -> None:
    spec = ToolSpec(
        name="search_client",
        version=1,
        description="Find a client in Bitrix24 by identifiers",
        effect=Effect.read,
        required_permission="customer.read",
        input_schema={"type": "object", "additionalProperties": False},
    )
    assert spec.schema_profile == "stanok.tool-args.v1"
    assert spec.effect is Effect.read


def test_proposal_forbids_extra_fields() -> None:
    # an extra field from the model must not be silently swallowed (AD-8)
    with pytest.raises(ValidationError):
        ToolCallProposal(tool="search_client", arguments={}, sneaky=1)  # type: ignore[call-arg]


def test_outcome_dispatches_by_status() -> None:
    cid = str(uuid4())
    ok = _OUTCOME.validate_python({"status": "succeeded", "call_id": cid, "data": {"found": True}})
    assert isinstance(ok, ToolSucceeded)
    fail = _OUTCOME.validate_python(
        {
            "status": "failed",
            "call_id": cid,
            "error": {"code": "not_found", "message": "no match", "retryable": False},
        }
    )
    assert isinstance(fail, ToolFailed)


def test_success_cannot_carry_error() -> None:
    # a contradictory result (success carrying an error) is unrepresentable
    with pytest.raises(ValidationError):
        ToolSucceeded(call_id=uuid4(), error=ToolError(code="x", message="y"))  # type: ignore[call-arg]


def test_rejects_nonpositive_version() -> None:
    with pytest.raises(ValidationError):
        ToolRef(name="search_client", version=0, spec_digest=_VALID_DIGEST)


def test_rejects_malformed_digest() -> None:
    with pytest.raises(ValidationError):
        ToolRef(name="search_client", version=1, spec_digest="deadbeef")


def test_rejects_non_finite_numbers() -> None:
    # NaN/Infinity are not JSON — the untrusted boundary rejects them
    with pytest.raises(ValidationError):
        ToolCallProposal(tool="search_client", arguments={"score": float("inf")})


def test_validation_error_message_hides_argument_values() -> None:
    # K3: hide_input_in_errors keeps offending values out of the ValidationError message
    secret = "LEAKED_SECRET_VALUE"
    with pytest.raises(ValidationError) as exc:
        ToolCallProposal(tool="search_client", arguments={}, sneaky=secret)  # type: ignore[call-arg]
    assert secret not in str(exc.value)
