import pytest
from pydantic import Field, RootModel
from stanok_core.registry import (
    RegisteredTool,
    ToolArgs,
    ToolRegistry,
    ToolRegistryError,
    spec_from_args,
)
from stanok_core.schema import ToolSchemaError, digest_spec
from stanok_core.tools import Effect, ToolRef


class SearchArgs(ToolArgs):
    inn: str
    kpp: str | None  # nullable, required, no default


def _register(reg: ToolRegistry) -> ToolRef:
    return reg.register(
        "search_client", 1, "Find a client", Effect.read, "customer.read", SearchArgs
    )


def test_register_and_resolve_ref() -> None:
    reg = ToolRegistry()
    ref = _register(reg)
    assert ref.name == "search_client"
    assert ref.version == 1
    assert ref.spec_digest.startswith("sha256:")
    got = reg.resolve_ref(ref)
    assert isinstance(got, RegisteredTool)
    assert got.spec.name == "search_client"
    assert got.args_model is SearchArgs
    assert got.spec.required_permission == "customer.read"


def test_schema_derived_from_model() -> None:
    reg = ToolRegistry()
    schema = reg.resolve_ref(_register(reg)).spec.input_schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "inn" in schema["properties"]  # type: ignore[operator]


def test_resolved_spec_cannot_mutate_registry() -> None:
    reg = ToolRegistry()
    ref = _register(reg)
    reg.resolve_ref(ref).spec.input_schema["properties"].clear()  # type: ignore[union-attr]
    fresh = reg.resolve_ref(ref).spec.input_schema["properties"]
    assert "inn" in fresh  # type: ignore[operator]


def test_rejects_defaults_in_args() -> None:
    class BadArgs(ToolArgs):
        x: int = 5  # default falls outside the authoring profile

    reg = ToolRegistry()
    with pytest.raises(ToolSchemaError):
        reg.register("bad", 1, "d", Effect.read, "customer.read", BadArgs)


def test_rejects_optional_via_default_factory() -> None:
    class FactoryArgs(ToolArgs):
        tags: list[str] = Field(default_factory=list)  # optional, no JSON default -> masks absence

    reg = ToolRegistry()
    with pytest.raises(ToolSchemaError):
        reg.register("bad", 1, "d", Effect.read, "customer.read", FactoryArgs)


def test_rejects_duplicate_name_version() -> None:
    reg = ToolRegistry()
    _register(reg)
    with pytest.raises(ToolRegistryError):
        _register(reg)


def test_rejects_non_basemodel() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolRegistryError):
        reg.register("x", 1, "d", Effect.read, "customer.read", dict)  # type: ignore[arg-type]


def test_rejects_rootmodel() -> None:
    class RootArgs(RootModel[dict[str, str]]):
        pass

    reg = ToolRegistry()
    with pytest.raises(ToolRegistryError):
        reg.register("x", 1, "d", Effect.read, "customer.read", RootArgs)


def test_resolve_ref_digest_mismatch_fails_closed() -> None:
    reg = ToolRegistry()
    _register(reg)
    stale = ToolRef(name="search_client", version=1, spec_digest="sha256:" + "0" * 64)
    with pytest.raises(ToolRegistryError):
        reg.resolve_ref(stale)


def test_resolve_ref_unknown_tool() -> None:
    reg = ToolRegistry()
    ref = ToolRef(name="search_client", version=9, spec_digest="sha256:" + "a" * 64)
    with pytest.raises(ToolRegistryError):
        reg.resolve_ref(ref)


def test_digest_stable_across_registrations() -> None:
    assert _register(ToolRegistry()).spec_digest == _register(ToolRegistry()).spec_digest


def test_digest_matches_digest_spec() -> None:
    reg = ToolRegistry()
    entry = reg.resolve_ref(_register(reg))
    assert entry.digest == digest_spec(entry.spec)


def test_spec_from_args_validates_eagerly() -> None:
    spec = spec_from_args("search_client", 1, "d", Effect.read, "customer.read", SearchArgs)
    assert spec.name == "search_client"
    assert spec.input_schema["additionalProperties"] is False
