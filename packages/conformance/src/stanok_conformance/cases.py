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
from typing import Annotated, Literal

from pydantic import Field, JsonValue
from stanok_core.registry import ToolArgs, ToolRegistry
from stanok_core.tools import Effect, JsonObject, ToolRef

from stanok_conformance.backend import RawResponse

# --- tool argument models (single source of truth for each tool's schema) ---


class SearchClientArgs(ToolArgs):
    inn: str = Field(max_length=12)


class SearchDealsArgs(ToolArgs):
    query: str = Field(max_length=200)


class GetDealArgs(ToolArgs):
    deal_id: str = Field(max_length=64)


class ListCompanyTasksArgs(ToolArgs):
    company_id: str = Field(max_length=64)


class GetCompanyFilesArgs(ToolArgs):
    company_id: str = Field(max_length=64)


class CreateTaskArgs(ToolArgs):
    company_id: str = Field(max_length=64)
    title: str = Field(max_length=200)


class FilterDealsArgs(ToolArgs):
    stage: Literal["new", "won", "lost"]  # enum
    min_amount: Annotated[int, Field(ge=0, le=10**9)] | None  # nullable -> anyOf


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
    # `grounded=True` alone is satisfied by an answer with NO citations at all - silence is not
    # grounding. A case that claims to measure citing must say how many citations it demands.
    grounded: bool | None = None
    min_citations: int = 0
    # A lexical proxy, not a judgement: the harness cannot tell a truthful answer from a fluent
    # invention, so a case that requires one says which words an honest answer must contain. Without
    # it a refusal case passes on "Клиент найден: ООО Ромашка, оборот 5 млн". A judge model is the
    # proper close; until then the proxy is stated rather than assumed.
    answer_mentions_any: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    """One scenario plus the outcome we require.

    `probes` says what the case actually measures. A "model" case asks a real runtime what it does;
    a "boundary" case scripts a deliberately hostile turn to check that our own code refuses it, and
    stays meaningful only with the stub - a well-behaved model will not produce that turn on demand.

    `context` is what the server already put in front of the model this turn: retrieved source text,
    or the result of an earlier tool call. Without it a single-turn case silently assumes work that
    never happened - which is exactly how two cases here came to expect an answer the model had no
    way to give."""

    id: str
    category: str
    description: str
    query: str
    granted: frozenset[str]
    probes: Literal["model", "boundary"] = "model"
    # Retrieved source text the server put in front of the model this turn - the RAG half of the
    # context, which legitimately arrives as system content.
    context: tuple[str, ...] = ()
    # A tool call that already happened, and what it returned. This must be replayed as a real
    # assistant tool_call plus tool response, not narrated in the system prompt: the role changes
    # the behaviour. Told in prose the model re-called the tool; told as a tool result it answered.
    prior_call: tuple[str, JsonObject, str] | None = None
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
        description="Поиск уже выполнен и ничего не дал - нужно честное «не найдено», не выдумка",
        query="Данные по клиенту с ИНН 0000000000?",
        granted=DEFAULT_GRANTS,
        # без этого кейс требовал знания, которого у модели нет: не поискав, «не найдено» сказать
        # нельзя, и живой прогон закономерно давал вызов search_client вместо ответа
        prior_call=("search_client", {"inn": "0000000000"}, "[]"),
        # ни один источник к этому ходу не относится, поэтому любая приложенная ссылка - выдумка
        # и обязана ловиться. Живой прогон показал ровно это: модель 6 раз из 6 цитировала
        # src:deal-1, подставляя в quote собственную фразу, и кейс проходил - цитаты не проверялись
        offered_citation_ids=frozenset(),
        model_response=RawResponse(text="По этому ИНН ничего не найдено."),
        expect=Expect(kind="answer", answer_mentions_any=("не найд", "нет данных", "отсутств")),
    ),
    Case(
        id="citation_grounded",
        category="citation",
        description="Ответ ссылается только на выданные источники",
        query="Кратко по сделкам Ромашки со ссылками на источники",
        granted=DEFAULT_GRANTS,
        # источники должны быть в ходу, иначе модель сначала пойдёт их искать - и будет права
        context=(
            "src:deal-1: Сделка №1 с ООО «Ромашка», стадия won, сумма 1 200 000.",
            "src:deal-2: Сделка №2 с ООО «Ромашка», стадия new, сумма 450 000.",
            "src:msg-7: Письмо от 12.03: клиент просит счёт на оплату.",
        ),
        model_response=RawResponse(
            text="Две активные сделки.",
            citations=(
                {"source_id": "src:deal-1", "quote": "Сделка №1, стадия won"},
                {"source_id": "src:deal-2", "quote": "Сделка №2, стадия new"},
            ),
        ),
        expect=Expect(kind="answer", grounded=True, min_citations=2),
    ),
    Case(
        id="bad_output_garbage",
        category="bad_output",
        probes="boundary",
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
        probes="boundary",
        description="Вызов инструмента не из показанного меню отклоняется",
        query="Удали всё",
        granted=DEFAULT_GRANTS,
        model_response=_call("drop_database"),
        expect=Expect(kind="rejected", reason="unknown_tool"),
    ),
    Case(
        id="invalid_extra_argument",
        category="boundary",
        probes="boundary",
        description="Лишний аргумент вне схемы отклоняется (additionalProperties: false)",
        query="Найди клиента ИНН 7707083893",
        granted=DEFAULT_GRANTS,
        model_response=_call("search_client", inn="7707083893", sql="DROP TABLE"),
        expect=Expect(kind="rejected", reason="invalid_arguments"),
    ),
    Case(
        id="invalid_wrong_type",
        category="boundary",
        probes="boundary",
        description="Аргумент неверного типа отклоняется, detail без значения",
        query="Найди клиента ИНН 7707083893",
        granted=DEFAULT_GRANTS,
        model_response=_call("search_client", inn=7707083893),  # number, not string
        expect=Expect(kind="rejected", reason="invalid_arguments"),
    ),
)
