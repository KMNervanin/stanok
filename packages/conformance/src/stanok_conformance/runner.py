"""Run a case: take the backend's response, push it through the real core boundary, check the
required outcome. The runner never trusts the model — a tool call becomes a ToolCallProposal and
goes through validate_proposal exactly as in production; garbage is caught and reported, not raised.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError
from stanok_core.citation import Citation, ungrounded_citations
from stanok_core.registry import ToolRegistry
from stanok_core.tools import CallRejected, ToolCallProposal, ToolRef, ValidatedToolCall
from stanok_core.validation import validate_proposal

from stanok_conformance.backend import Backend, RawResponse
from stanok_conformance.cases import Case, Expect


@dataclass(frozen=True)
class CaseOutcome:
    """Normalised result of interpreting one model turn through the core boundary."""

    kind: str  # "validated" | "rejected" | "answer" | "malformed"
    call: ValidatedToolCall | None = None
    rejection: CallRejected | None = None
    text: str = ""
    ungrounded: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    outcome: CaseOutcome
    note: str


def run_case(
    case: Case, backend: Backend, registry: ToolRegistry, refs: dict[str, ToolRef]
) -> CaseResult:
    outcome = _interpret(case, backend.respond(case), registry, refs)
    passed, note = _check(case.expect, outcome)
    return CaseResult(case.id, case.category, passed, outcome, note)


def run_all(
    cases: tuple[Case, ...], backend: Backend, registry: ToolRegistry, refs: dict[str, ToolRef]
) -> list[CaseResult]:
    return [run_case(c, backend, registry, refs) for c in cases]


def _interpret(
    case: Case, resp: RawResponse, registry: ToolRegistry, refs: dict[str, ToolRef]
) -> CaseOutcome:
    if resp.malformed:
        return CaseOutcome(kind="malformed", detail="backend could not parse model output")
    if resp.tool is None:  # free-text answer: refusal, disambiguation, or cited summary
        citations = _parse_citations(resp.citations)
        if citations is None:
            return CaseOutcome(kind="malformed", detail="citation payload not well-formed")
        ungrounded = tuple(ungrounded_citations(citations, case.offered_citation_ids))
        return CaseOutcome(kind="answer", text=resp.text or "", ungrounded=ungrounded)
    try:  # a tool call is untrusted: build the proposal, then validate exactly as in production
        proposal = ToolCallProposal(tool=resp.tool, arguments=resp.arguments or {})
    except ValidationError:
        return CaseOutcome(kind="malformed", detail="tool call not well-formed")
    offered = {name: refs[name] for name in case.offered_tools if name in refs}
    result = validate_proposal(proposal, offered, registry, case.granted)
    if isinstance(result, ValidatedToolCall):
        return CaseOutcome(kind="validated", call=result)
    return CaseOutcome(kind="rejected", rejection=result)


def _parse_citations(raw: tuple[dict[str, str], ...]) -> list[Citation] | None:
    parsed: list[Citation] = []
    for item in raw:
        try:
            parsed.append(Citation(**item))
        except (ValidationError, TypeError):
            return None
    return parsed


def _check(expect: Expect, outcome: CaseOutcome) -> tuple[bool, str]:
    if outcome.kind != expect.kind:
        return False, f"kind {outcome.kind!r} != expected {expect.kind!r} ({outcome.detail})"
    if expect.kind == "validated":
        call = outcome.call
        if call is None:
            return False, "no validated call"
        if expect.tool is not None and call.tool.name != expect.tool:
            return False, f"tool {call.tool.name!r} != expected {expect.tool!r}"
        if expect.arguments is not None and call.arguments != expect.arguments:
            return False, "arguments differ from expected"
    elif expect.kind == "rejected":
        rejection = outcome.rejection
        if rejection is None:
            return False, "no rejection"
        if expect.reason is not None and rejection.reason != expect.reason:
            return False, f"reason {rejection.reason!r} != expected {expect.reason!r}"
    elif expect.kind == "answer" and expect.grounded is not None:
        grounded = not outcome.ungrounded
        if grounded != expect.grounded:
            return False, f"grounded={grounded} != {expect.grounded}; {outcome.ungrounded}"
    return True, "ok"


def summarize(results: list[CaseResult]) -> str:
    passed = sum(r.passed for r in results)
    lines = [
        f"{'PASS' if r.passed else 'FAIL'}  {r.category:<10} {r.case_id:<28} {r.note}"
        for r in results
    ]
    return "\n".join(lines) + f"\n\n{passed}/{len(results)} passed"
