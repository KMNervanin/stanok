from uuid import UUID

from stanok_core.registry import ToolArgs, ToolRegistry
from stanok_core.tools import CallRejected, Effect, ToolCallProposal, ToolRef, ValidatedToolCall
from stanok_core.validation import validate_proposal


class SearchArgs(ToolArgs):
    inn: str


_GRANTED = frozenset({"customer.read"})


def _setup() -> tuple[ToolRegistry, dict[str, ToolRef]]:
    reg = ToolRegistry()
    ref = reg.register(
        "search_client", 1, "Find a client", Effect.read, "customer.read", SearchArgs
    )
    return reg, {ref.name: ref}


def test_valid_proposal_becomes_validated_call() -> None:
    reg, offered = _setup()
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": "7700000000"})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, ValidatedToolCall)
    assert out.tool == offered["search_client"]
    assert out.arguments == {"inn": "7700000000"}
    assert isinstance(out.call_id, UUID)


def test_validated_payload_is_immutable() -> None:
    # K1: mutating the returned dict must not alter the trusted snapshot executed later
    reg, offered = _setup()
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": "7700000000"})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, ValidatedToolCall)
    out.arguments["inn"] = "MUTATED"
    out.arguments["injected"] = "x"
    assert out.arguments == {"inn": "7700000000"}  # snapshot unchanged


def test_validated_payload_detached_from_proposal() -> None:
    # the snapshot is taken from canonical bytes, independent of the (mutable) proposal object
    reg, offered = _setup()
    args = {"inn": "7700000000"}
    proposal = ToolCallProposal(tool="search_client", arguments=args)
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, ValidatedToolCall)
    assert out.arguments_json == b'{"inn":"7700000000"}'


def test_rejects_overcomplex_arguments() -> None:
    # a small-bytes but high-node-count payload is capped before validation, so a hostile schema
    # can't amplify it into millions of comparisons (K2)
    reg, offered = _setup()
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": list(range(5000))})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "invalid_arguments"


def test_unknown_tool_rejected() -> None:
    reg, offered = _setup()
    out = validate_proposal(ToolCallProposal(tool="nope", arguments={}), offered, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "unknown_tool"


def test_forbidden_before_schema() -> None:
    # authz runs before schema: an unauthorized caller gets forbidden, not a schema oracle
    reg, offered = _setup()
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": 123})  # also schema-invalid
    out = validate_proposal(proposal, offered, reg, frozenset())
    assert isinstance(out, CallRejected)
    assert out.reason == "forbidden"


def test_invalid_arguments_wrong_type() -> None:
    reg, offered = _setup()
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": 123})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "invalid_arguments"


def test_invalid_arguments_missing_required() -> None:
    reg, offered = _setup()
    proposal = ToolCallProposal(tool="search_client", arguments={})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "invalid_arguments"


def test_stale_ref_is_tool_unavailable() -> None:
    reg, _ = _setup()
    stale = ToolRef(name="search_client", version=1, spec_digest="sha256:" + "0" * 64)
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": "x"})
    out = validate_proposal(proposal, {"search_client": stale}, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "tool_unavailable"


def test_detail_never_leaks_argument_values() -> None:
    reg, offered = _setup()
    secret = "SECRET_VALUE_12345"
    proposal = ToolCallProposal(tool="search_client", arguments={"inn": 999, "leaked": secret})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "invalid_arguments"
    assert secret not in out.detail
