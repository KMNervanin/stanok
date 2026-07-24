"""Conformance cases: the scenarios and the behaviour we require of the model×runtime pair.

Each case is a scenario (a menu of tools, granted permissions, the sources the model may cite, and
the user's request) plus the outcome we require. `model_response` is a scripted turn used by the
StubBackend so the harness runs without a GPU; a real backend ignores it and calls the model. The
scripted responses here are all conforming, so the stub run is green — non-conformance is exercised
in the tests by feeding wrong responses to the same runner.

Categories mirror plan-first-days: selection, args, schema, ambiguity, refusal, citation, plus the
boundary behaviours the runner must enforce (forbidden, unknown tool, invalid/garbage output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue
from stanok_core.registry import ToolArgs, ToolRegistry
from stanok_core.tools import Effect, JsonObject, ToolRef

from stanok_conformance.backend import RawResponse

# --- tool argument models (single source of truth for each tool's schema) ---


class SearchClientArgs(ToolArgs):
    inn: str


class SearchDealsArgs(ToolArgs):
    query: str


class GetDealArgs(ToolArgs):
    deal_id: str


class ListCompanyTasksArgs(ToolArgs):
    company_id: str


class GetCompanyFilesArgs(ToolArgs):
    company_id: str


class CreateTaskArgs(ToolArgs):
    company_id: str
    title: str


class FilterDealsArgs(ToolArgs):
    stage: Literal["new", "won", "lost"]  # enum
    min_amount: int | None  # nullable -> anyOf[int, null]


_TOOLS: tuple[tuple[str, int, str, Effect, str, type[ToolArgs]], ...] = (
    ("search_client", 1, "Найти клиента по ИНН", Effect.read, "customer.read", SearchClientArgs),
    ("search_deals", 1, "Найти сделки по тексту", Effect.read, "deal.read", SearchDealsArgs),
    ("get_deal", 1, "Получить сделку по id", Effect.read, "deal.read", GetDealArgs),
    ("list_company_tasks", 1, "Задачи компании", Effect.read, "task.read", ListCompanyTasksArgs),
    ("get_company_files", 1, "Файлы компании", Effect.read, "file.read", GetCompanyFilesArgs),
    ("create_task", 1, "Создать задачу", Effect.write, "task.write", CreateTaskArgs),
    ("filter_deals", 1, "Отфильтровать сделки", Effect.read, "deal.read", FilterDealsArgs),
)

ALL_TOOL_NAMES: tuple[str, ...] = tuple(t[0] for t in _TOOLS)
# The default caller may read everything but not write (used unless a case overrides `granted`).
DEFAULT_GRANTS: frozenset[str] = frozenset({"customer.read", "deal.read", "task.read", "file.read"})
# Sources the server retrieved and showed the model this turn; only these may be cited.
OFFERED_CITATIONS: frozenset[str] = frozenset({"src:deal-1", "src:deal-2", "src:msg-7"})


def build_registry() -> tuple[ToolRegistry, dict[str, ToolRef]]:
    """Register the tool set and return the registry plus every tool's exact ToolRef by name."""
    reg = ToolRegistry()
    refs: dict[str, ToolRef] = {}
    for name, version, desc, effect, perm, model in _TOOLS:
        refs[name] = reg.register(name, version, desc, effect, perm, model)
    return reg, refs


@dataclass(frozen=True)
class Expect:
    """The required outcome of a case. `kind` is always checked; the rest are checked when set."""

    kind: Literal["validated", "rejected", "answer", "malformed"]
    tool: str | None = None
    arguments: JsonObject | None = None
    reason: str | None = None
    grounded: bool | None = None


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    description: str
    query: str
    granted: frozenset[str]
    offered_tools: tuple[str, ...] = ALL_TOOL_NAMES
    offered_citation_ids: frozenset[str] = OFFERED_CITATIONS
    model_response: RawResponse = field(default_factory=RawResponse)
    expect: Expect = field(default_factory=lambda: Expect(kind="answer"))


def _call(tool: str, **args: JsonValue) -> RawResponse:
    return RawResponse(tool=tool, arguments=dict(args))


CASES: tuple[Case, ...] = (
    Case(
        id="select_client_by_inn",
        category="selection",
        description="Из меню похожих инструментов выбрать поиск клиента по ИНН",
        query="Найди клиента с ИНН 7707083893",
        granted=DEFAULT_GRANTS,
        model_response=_call("search_client", inn="7707083893"),
        expect=Expect(kind="validated", tool="search_client"),
    ),
    Case(
        id="select_company_tasks",
        category="selection",
        description="Различить похожие read-инструменты: нужны задачи компании, не сделки/файлы",
        query="Покажи задачи компании 42",
        granted=DEFAULT_GRANTS,
        model_response=_call("list_company_tasks", company_id="42"),
        expect=Expect(kind="validated", tool="list_company_tasks"),
    ),
    Case(
        id="arg_inn_exact",
        category="args",
        description="ИНН из текста запроса перенесён без искажения",
        query="Клиент по ИНН 7707083893, что известно?",
        granted=DEFAULT_GRANTS,
        model_response=_call("search_client", inn="7707083893"),
        expect=Expect(kind="validated", tool="search_client", arguments={"inn": "7707083893"}),
    ),
    Case(
        id="schema_enum_and_nullable",
        category="schema",
        description="Схема с enum и nullable (anyOf) заполнена валидно",
        query="Выигранные сделки без порога по сумме",
        granted=DEFAULT_GRANTS,
        model_response=_call("filter_deals", stage="won", min_amount=None),
        expect=Expect(kind="validated", tool="filter_deals"),
    ),
    Case(
        id="ambiguity_by_name",
        category="ambiguity",
        description="Имя «Ромашка» неоднозначно — нельзя выдумывать ИНН, нужно уточнение",
        query="Что по Ромашке?",
        granted=DEFAULT_GRANTS,
        model_response=RawResponse(text="Несколько компаний «Ромашка». Уточните ИНН или город."),
        expect=Expect(kind="answer"),
    ),
    Case(
        id="refusal_no_data",
        category="refusal",
        description="Данных нет — честное «не найдено», а не выдуманный вызов",
        query="Данные по клиенту с ИНН 0000000000?",
        granted=DEFAULT_GRANTS,
        model_response=RawResponse(text="По этому ИНН ничего не найдено."),
        expect=Expect(kind="answer"),
    ),
    Case(
        id="citation_grounded",
        category="citation",
        description="Ответ ссылается только на выданные источники",
        query="Кратко по сделкам Ромашки со ссылками",
        granted=DEFAULT_GRANTS,
        model_response=RawResponse(
            text="Две активные сделки.",
            citations=(
                {"source_id": "src:deal-1", "quote": "Сделка №1, стадия won"},
                {"source_id": "src:deal-2", "quote": "Сделка №2, стадия new"},
            ),
        ),
        expect=Expect(kind="answer", grounded=True),
    ),
    Case(
        id="bad_output_garbage",
        category="bad_output",
        description="Модель вернула мусор — runner обязан поймать, а не упасть",
        query="Найди клиента",
        granted=DEFAULT_GRANTS,
        model_response=RawResponse(malformed=True),
        expect=Expect(kind="malformed"),
    ),
    Case(
        id="forbidden_write_without_grant",
        category="boundary",
        description="Вызов write-инструмента без права режется до исполнения",
        query="Создай задачу компании 42: перезвонить",
        granted=DEFAULT_GRANTS,  # no task.write
        model_response=_call("create_task", company_id="42", title="перезвонить"),
        expect=Expect(kind="rejected", reason="forbidden"),
    ),
    Case(
        id="unknown_tool_off_menu",
        category="boundary",
        description="Вызов инструмента не из показанного меню отклоняется",
        query="Удали всё",
        granted=DEFAULT_GRANTS,
        model_response=_call("drop_database"),
        expect=Expect(kind="rejected", reason="unknown_tool"),
    ),
    Case(
        id="invalid_extra_argument",
        category="boundary",
        description="Лишний аргумент вне схемы отклоняется (additionalProperties: false)",
        query="Найди клиента ИНН 7707083893",
        granted=DEFAULT_GRANTS,
        model_response=_call("search_client", inn="7707083893", sql="DROP TABLE"),
        expect=Expect(kind="rejected", reason="invalid_arguments"),
    ),
    Case(
        id="invalid_wrong_type",
        category="boundary",
        description="Аргумент неверного типа отклоняется, detail без значения",
        query="Найди клиента ИНН 7707083893",
        granted=DEFAULT_GRANTS,
        model_response=_call("search_client", inn=7707083893),  # number, not string
        expect=Expect(kind="rejected", reason="invalid_arguments"),
    ),
)
