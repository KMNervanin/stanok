from dataclasses import replace

from stanok_conformance.backend import RawResponse, StubBackend
from stanok_conformance.cases import CASES, Case, Expect, build_registry
from stanok_conformance.runner import run_all, run_case
from stanok_core.citation import Citation, ungrounded_citations


def _run_one(case: Case) -> bool:
    reg, refs = build_registry()
    return run_case(case, StubBackend(), reg, refs).passed


def test_all_cases_pass_with_conforming_stub() -> None:
    reg, refs = build_registry()
    results = run_all(CASES, StubBackend(), reg, refs)
    failed = [r for r in results if not r.passed]
    assert not failed, "\n".join(f"{r.case_id}: {r.note}" for r in failed)
    assert len(results) == len(CASES)


def test_every_plan_category_is_covered() -> None:
    categories = {c.category for c in CASES}
    plan = {"selection", "args", "schema", "ambiguity", "refusal", "citation", "bad_output"}
    assert plan <= categories


def test_case_ids_unique() -> None:
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_harness_catches_wrong_tool_selection() -> None:
    base = next(c for c in CASES if c.id == "select_client_by_inn")
    picked_other = RawResponse(tool="create_task", arguments={"company_id": "1", "title": "x"})
    wrong = replace(base, model_response=picked_other)
    assert _run_one(wrong) is False  # picked a different tool than required


def test_harness_catches_distorted_argument() -> None:
    base = next(c for c in CASES if c.id == "arg_inn_exact")
    dropped_digit = RawResponse(tool="search_client", arguments={"inn": "770708389"})
    distorted = replace(base, model_response=dropped_digit)
    assert _run_one(distorted) is False  # INN lost a digit


def test_harness_flags_ungrounded_citation() -> None:
    base = next(c for c in CASES if c.id == "citation_grounded")
    ungrounded = replace(
        base,
        model_response=RawResponse(
            text="…", citations=({"source_id": "src:deal-99", "quote": "выдумано"},)
        ),
    )
    assert _run_one(ungrounded) is False  # cited a source never offered


def test_runner_does_not_crash_on_garbage() -> None:
    base = next(c for c in CASES if c.id == "select_client_by_inn")
    garbage = replace(
        base, model_response=RawResponse(malformed=True), expect=Expect(kind="malformed")
    )
    result = _run_one(garbage)
    assert result is True  # classified as malformed, no exception


def test_citation_grounding_helper() -> None:
    offered = frozenset({"src:deal-1", "src:deal-2"})
    cites = [
        Citation(source_id="src:deal-1", quote="ok"),
        Citation(source_id="src:deal-9", quote="invented"),
    ]
    assert ungrounded_citations(cites, offered) == ["src:deal-9"]
    assert ungrounded_citations([cites[0]], offered) == []
