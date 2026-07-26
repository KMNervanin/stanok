"""A real backend: one turn of a local model served by llama.cpp's OpenAI-compatible server.

Implements the same `respond` seam as StubBackend, so the cases and the runner are unchanged. The
model is given the turn's menu as OpenAI-style tools and decides for itself whether to call one or
answer in words - which is the behaviour the conformance set exists to measure. Nothing here trusts
the result: the runner still pushes whatever comes back through validate_proposal.

Note what this backend can and cannot measure. Cases that script a deliberately hostile turn (a tool
off the menu, an injected argument, a wrong type, garbage) test the boundary, not the model; a
well-behaved model simply will not produce them, so those cases stay meaningful only with the stub.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import HTTPException
from typing import TYPE_CHECKING, Any

from stanok_core.registry import ToolRegistry
from stanok_core.tools import JsonObject, ToolRef

from stanok_conformance.backend import RawResponse

if TYPE_CHECKING:
    from stanok_conformance.cases import Case

_SYSTEM = (
    "Ты - ассистент внутри корпоративной системы. Тебе доступны инструменты; вызывай инструмент "
    "только когда запрос однозначно требует данных из системы и всех аргументов хватает. "
    "Если запрос неоднозначен или данных нет - не выдумывай аргументы, ответь словами и уточни, "
    "чего не хватает. Не сообщай о выполнении того, чего не делал."
)
_CITATION_RULE = (
    "\nЕсли приводишь источники, бери source_id ТОЛЬКО из этого списка и копируй его дословно, "
    "целиком, вместе с префиксом до двоеточия: {ids}. Сокращать, обрезать префикс или "
    "придумывать свой идентификатор нельзя - такая ссылка будет отброшена. "
    'Заверши ответ отдельной строкой вида ИСТОЧНИКИ: [{{"source_id": "...", "quote": "..."}}] '
    "- строго JSON-массив. Если источники не нужны, строку не добавляй."
)
_CITATION_MARK = "ИСТОЧНИКИ:"


@dataclass(frozen=True)
class LlamaServerBackend:
    """One turn against llama-server. `base_url` points at the OpenAI-compatible endpoint."""

    registry: ToolRegistry
    refs: dict[str, ToolRef]
    base_url: str = "http://127.0.0.1:8080"
    timeout: float = 180.0
    temperature: float = 0.0
    label: str = "llama-server"

    def respond(self, case: Case) -> RawResponse:
        try:
            message = self._complete(case)
        except urllib.error.HTTPError as exc:
            # the runtime answered, and answered badly - that is a broken measurement, not a
            # missing one, and must never be reported as "no runtime is running"
            raise BackendBroken(f"{self.base_url} ответил {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendUnavailable(f"{self.base_url}: {exc}") from exc
        except (ValueError, KeyError, IndexError, HTTPException) as exc:
            raise BackendBroken(f"{self.base_url}: неожиданный ответ: {exc}") from exc
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls:
            # more than one call is a turn this harness cannot score; dropping the rest silently
            # would hide, say, a write call riding along behind a harmless read
            if len(calls) > 1 or not isinstance(calls[0], dict):
                return RawResponse(malformed=True)
            return _as_tool_call(calls[0])
        content = message.get("content")
        return _as_answer(content if isinstance(content, str) else "")

    def _complete(self, case: Case) -> dict[str, Any]:
        system = _SYSTEM
        if case.context:
            # what the server already showed the model this turn: retrieved sources, or the result
            # of an earlier tool call. Without it the model would have to call a tool to find out.
            system += "\n\nУже известно в этом ходу:\n" + "\n".join(case.context)
        if case.offered_citation_ids:
            system += _CITATION_RULE.format(ids=", ".join(sorted(case.offered_citation_ids)))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": case.query},
        ]
        if case.prior_call is not None:
            # replay it the way production would: the assistant asked, the tool answered. Narrating
            # this in the system prompt instead measured our prose, not the model - told in prose it
            # re-called the tool, told as a tool result it answered from the result.
            name, arguments, result = case.prior_call
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_prior",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": "call_prior", "content": result})
        body = {
            "messages": messages,
            "tools": self._tools(case),
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": 512,
            # Qwen3 thinks by default; the contract is about the call it makes, not its monologue
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        choice: dict[str, Any] = payload["choices"][0]["message"]
        return choice

    def _tools(self, case: Case) -> list[dict[str, Any]]:
        """The menu, exactly as the server would show it: name, description, published schema."""
        tools: list[dict[str, Any]] = []
        for name in case.offered_tools:
            ref = self.refs.get(name)
            if ref is None:
                continue
            spec = self.registry.resolve_ref(ref).spec
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": dict(spec.input_schema),
                    },
                }
            )
        return tools


class BackendUnavailable(RuntimeError):
    """The runtime could not be reached. Not a conformance failure - a missing measurement."""


class BackendBroken(RuntimeError):
    """The runtime replied, but not with a turn. A failed measurement, never a skipped one."""


def _as_tool_call(call: dict[str, Any]) -> RawResponse:
    function = call.get("function") or {}
    name = function.get("name")
    raw = function.get("arguments")
    if not isinstance(name, str):
        return RawResponse(malformed=True)
    if isinstance(raw, dict):
        arguments: Any = raw
    else:
        try:
            arguments = json.loads(raw or "{}")
        # ValueError, not JSONDecodeError: a literal past CPython's int-to-str digit limit raises
        # the base class, and the grammar does not bound how many digits a model may emit
        except (TypeError, ValueError):
            return RawResponse(malformed=True)
    if not isinstance(arguments, dict):
        return RawResponse(malformed=True)
    payload: JsonObject = arguments
    return RawResponse(tool=name, arguments=payload)


def _as_answer(content: str) -> RawResponse:
    split = _split_citations(content)
    if split is None:  # the model wrote a citation block we could not read
        return RawResponse(malformed=True)
    text, citations = split
    return RawResponse(text=text, citations=citations)


def _split_citations(content: str) -> tuple[str, tuple[dict[str, str], ...]] | None:
    """Pull the citation block off an answer, or return None if it is not readable.

    Every unreadable shape is malformed, never an empty citation set: silently dropping citations
    turns into `ungrounded=()`, which reads as "every source checked out" - a measurement that
    flatters the model exactly where it must not. The marker has to start a line, as the prompt
    demands, so the word appearing inside prose does not swallow the real block.
    """
    lines = content.splitlines()
    marked = [i for i, line in enumerate(lines) if line.lstrip().startswith(_CITATION_MARK)]
    if not marked:
        return content.strip(), ()
    if len(marked) > 1:  # which one is the answer's? unreadable rather than guessed
        return None
    index = marked[0]
    text = "\n".join(lines[:index]).strip()
    tail = lines[index].lstrip()[len(_CITATION_MARK) :] + "\n".join(lines[index + 1 :])
    try:
        parsed = json.loads(tail.strip() or "[]")
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    items: list[dict[str, str]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            return None
        items.append({str(k): str(v) for k, v in entry.items()})
    return text, tuple(items)
