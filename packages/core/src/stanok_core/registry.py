"""Tool registry: build specs from pydantic arg models, validate, and hold immutable snapshots.

A tool is registered by giving its metadata and a pydantic args model; the registry derives the
input schema from that model (single source of truth), validates it against the Stanok authoring
profile, and stores a canonical byte snapshot. resolve() rebuilds a fresh ToolSpec from those bytes
every time, so a caller can never mutate registry state (AD-8, AD-9).
"""

import hashlib
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, RootModel

from stanok_core.schema import canonical_json, validate_tool_schema
from stanok_core.tools import Effect, ToolRef, ToolSpec


class ToolRegistryError(ValueError):
    """Registration or resolution failed."""


class ToolArgs(BaseModel):
    """Recommended base for tool argument models: forbids extra fields.

    extra=forbid makes pydantic emit additionalProperties: false, which the authoring profile
    requires. Use plain types and `T | None` (no `= None`) for nullable fields; defaults and
    default_factory are rejected so a missing argument is never silently masked."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class RegisteredTool:
    """Immutable registry entry: a canonical spec snapshot plus its args model and digest.

    `spec` is rebuilt from the byte snapshot on every access, so mutating it cannot affect the
    registry. args_model is authoring/provenance only - it is NOT used for runtime validation
    (custom validators need not appear in the JSON Schema). handler/reconciler arrive in Step 3+."""

    spec_json: bytes
    args_model: type[BaseModel]
    digest: str

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec.model_validate_json(self.spec_json)


def spec_from_args(
    name: str,
    version: int,
    description: str,
    effect: Effect,
    required_permission: str,
    args_model: type[BaseModel],
) -> ToolSpec:
    """Build a ToolSpec whose input schema is derived from the args model (single source of truth).

    Raises ToolRegistryError for an unusable model, or ToolSchemaError (with a JSON path) if the
    generated schema falls outside the authoring profile - e.g. a default, an optional field, a
    tuple, or an unsupported format. Fix the model, not the schema."""
    _check_args_model(args_model)
    schema = args_model.model_json_schema(mode="validation")
    spec = ToolSpec(
        name=name,
        version=version,
        description=description,
        effect=effect,
        required_permission=required_permission,
        input_schema=schema,
    )
    validate_tool_schema(spec.input_schema)  # fail fast for the author; register re-checks
    return spec


def _check_args_model(args_model: type[BaseModel]) -> None:
    if not (isinstance(args_model, type) and issubclass(args_model, BaseModel)):
        raise ToolRegistryError("args_model must be a BaseModel subclass")
    if issubclass(args_model, RootModel):
        raise ToolRegistryError("RootModel is not allowed as an args model")


class ToolRegistry:
    """Holds registered tools keyed by (name, version) as immutable canonical snapshots."""

    def __init__(self) -> None:
        self._tools: dict[tuple[str, int], RegisteredTool] = {}

    def register(
        self,
        name: str,
        version: int,
        description: str,
        effect: Effect,
        required_permission: str,
        args_model: type[BaseModel],
    ) -> ToolRef:
        spec = spec_from_args(name, version, description, effect, required_permission, args_model)
        key = (spec.name, spec.version)
        if key in self._tools:
            raise ToolRegistryError(f"tool already registered: {key}")
        spec_json = canonical_json(spec.model_dump(mode="json"))
        digest = "sha256:" + hashlib.sha256(spec_json).hexdigest()
        entry = RegisteredTool(spec_json=spec_json, args_model=args_model, digest=digest)
        validate_tool_schema(entry.spec.input_schema)  # trust boundary on the exact stored form
        self._tools[key] = entry
        return ToolRef(name=spec.name, version=spec.version, spec_digest=digest)

    def resolve(self, name: str, version: int) -> RegisteredTool | None:
        return self._tools.get((name, version))

    def resolve_ref(self, ref: ToolRef) -> RegisteredTool:
        """Resolve a trusted ToolRef, failing closed if the digest no longer matches.

        A mismatch means a stale ref or configuration drift, not an AD-6 unknown - reject before
        dispatch, do not run reconciliation."""
        entry = self._tools.get((ref.name, ref.version))
        if entry is None:
            raise ToolRegistryError(f"unknown tool: {(ref.name, ref.version)}")
        if entry.digest != ref.spec_digest:
            raise ToolRegistryError(
                f"spec digest mismatch for {(ref.name, ref.version)}: stale ToolRef"
            )
        return entry
