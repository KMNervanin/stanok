from typing import Annotated
from uuid import UUID

from pydantic import Field
from stanok_core.registry import ToolArgs, ToolRegistry
from stanok_core.schema import MAX_INSTANCE_NODES
from stanok_core.tools import CallRejected, Effect, ToolCallProposal, ToolRef, ValidatedToolCall
from stanok_core.validation import validate_proposal


class SearchArgs(ToolArgs):
    inn: str = Field(max_length=12)


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


class BoundedListArgs(ToolArgs):
    # bounded elements as well as a bounded length: both budgets the profile publishes
    xs: list[Annotated[int, Field(ge=0, le=9999)]] = Field(max_length=MAX_INSTANCE_NODES - 2)


def _setup_list() -> tuple[ToolRegistry, dict[str, ToolRef]]:
    reg = ToolRegistry()
    ref = reg.register("bulk", 1, "Bulk lookup", Effect.read, "customer.read", BoundedListArgs)
    return reg, {ref.name: ref}


def test_published_maxitems_is_executable() -> None:
    # B1: whatever maxItems the menu shows the model must actually be callable. This used to break
    # at 1023 elements against a published maxItems of 4096.
    reg, offered = _setup_list()
    properties = reg.resolve_ref(offered["bulk"]).spec.input_schema["properties"]
    assert isinstance(properties, dict)
    xs_schema = properties["xs"]
    assert isinstance(xs_schema, dict)
    published = MAX_INSTANCE_NODES - 2
    assert xs_schema["maxItems"] == published
    for count in (1022, 1023, published):
        proposal = ToolCallProposal(tool="bulk", arguments={"xs": list(range(count))})
        out = validate_proposal(proposal, offered, reg, _GRANTED)
        assert isinstance(out, ValidatedToolCall), f"{count} items rejected: {out}"


def test_over_published_maxitems_is_rejected() -> None:
    reg, offered = _setup_list()
    over = MAX_INSTANCE_NODES - 1
    proposal = ToolCallProposal(tool="bulk", arguments={"xs": list(range(over))})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, CallRejected)
    assert out.reason == "invalid_arguments"


class BoundedTextArgs(ToolArgs):
    ids: list[Annotated[str, Field(max_length=36)]] = Field(max_length=290)


def test_published_string_array_is_executable_to_its_last_element() -> None:
    # B1, byte dimension: a call filling the published maxItems with maximum-length strings must
    # still go through - the fuzzer found this rejected as "arguments too large"
    reg = ToolRegistry()
    ref = reg.register("bulk_ids", 1, "Bulk ids", Effect.read, "customer.read", BoundedTextArgs)
    offered = {ref.name: ref}
    proposal = ToolCallProposal(tool="bulk_ids", arguments={"ids": ["x" * 36] * 290})
    out = validate_proposal(proposal, offered, reg, _GRANTED)
    assert isinstance(out, ValidatedToolCall), out
