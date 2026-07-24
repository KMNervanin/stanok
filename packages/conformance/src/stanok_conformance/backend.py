"""The one seam between the conformance harness and a model.

A backend takes a case's scenario and returns the model's turn already parsed into a RawResponse
(a real backend parses the model's structured output; the StubBackend replays a scripted response so
the harness can be built and run without a GPU). Exactly one backend, no abstractions "for later" —
when a real runtime is up it implements the same `respond` and the cases and runner are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from stanok_core.tools import JsonObject

if TYPE_CHECKING:
    from stanok_conformance.cases import Case


@dataclass(frozen=True)
class RawResponse:
    """One model turn, parsed into shape. Either a tool call (`tool` set) or a free-text answer
    (`text` set); `malformed` means the backend could not parse the model output at all — the runner
    must treat that as caught, not as a crash. `citations` are the model's raw source references."""

    tool: str | None = None
    arguments: JsonObject | None = None
    text: str | None = None
    citations: tuple[dict[str, str], ...] = ()
    malformed: bool = False


class Backend(Protocol):
    def respond(self, case: Case) -> RawResponse: ...


@dataclass(frozen=True)
class StubBackend:
    """Replays each case's scripted `model_response`. Dev scaffolding: it lets the harness run green
    without a model, and a real backend later ignores the script and calls the model instead."""

    label: str = "stub"

    def respond(self, case: Case) -> RawResponse:
        return case.model_response
