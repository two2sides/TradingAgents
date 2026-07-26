"""Shared helpers for strict, provider-aware structured agent output.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, bind the provider's reliable structured method. DeepSeek
   thinking models use JSON mode; providers that can force schema tools keep
   function calling.
2. At invocation, validate into the requested Pydantic model and render only
   validated data back to markdown. A failed attempt gets one structured
   correction retry. Exhaustion raises ``StructuredOutputError``; free text is
   never accepted as a decision.

Centralising the pattern keeps agent factories small and prevents unvalidated
text from silently weakening downstream decisions and credibility records.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from tradingagents.llm_clients.capabilities import get_capabilities
from tradingagents.extensions.decision.credibility.models import StructuredInvocationResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
_MAX_STRUCTURED_ATTEMPTS = 2

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


class StructuredOutputError(RuntimeError):
    """Raised when an agent cannot produce a schema-valid result."""


def _model_name(llm: Any) -> str | None:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _append_instruction(prompt: Any, instruction: str) -> Any:
    """Append an output instruction without mutating the caller's prompt."""
    if isinstance(prompt, str):
        return f"{prompt.rstrip()}\n\n{instruction}"
    if isinstance(prompt, list):
        return [*prompt, HumanMessage(content=instruction)]
    if hasattr(prompt, "to_messages"):
        return [*prompt.to_messages(), HumanMessage(content=instruction)]
    return f"{prompt}\n\n{instruction}"


def _json_schema_instruction(schema: type[BaseModel]) -> str:
    schema_json = json.dumps(
        schema.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "OUTPUT CONTRACT (mandatory): Return exactly one valid JSON object and "
        "nothing else. Do not use Markdown, code fences, commentary, or tool calls. "
        "The object must validate against the JSON Schema below. Include every "
        "required field; use null or an empty list only where the schema permits it.\n"
        f"JSON Schema:\n{schema_json}"
    )


@dataclass(frozen=True)
class StructuredBinding:
    """Provider-aware structured runnable plus its schema contract."""

    runnable: Any
    schema: type[BaseModel]
    method: str

    def prepare_prompt(self, prompt: Any) -> Any:
        if self.method == "json_mode":
            return _append_instruction(prompt, _json_schema_instruction(self.schema))
        return prompt

    def invoke(self, prompt: Any) -> Any:
        return self.runnable.invoke(self.prepare_prompt(prompt))


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Bind a provider-native structured call with raw response visibility.

    DeepSeek V4 uses JSON mode because it cannot be forced to call a schema
    tool. Other providers keep their declared preferred method. Unsupported
    providers return ``None`` and will fail closed at invocation time.
    """
    model_name = _model_name(llm)
    method = (
        get_capabilities(model_name).preferred_structured_method
        if model_name
        else "function_calling"
    )
    try:
        runnable = llm.with_structured_output(
            schema,
            method=method,
            include_raw=True,
        )
        return StructuredBinding(runnable=runnable, schema=schema, method=method)
    except (NotImplementedError, AttributeError) as exc:
        logger.error("%s: provider does not support structured output (%s)", agent_name, exc)
        return None


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Compatibility wrapper: structured-only despite the legacy name."""
    return invoke_structured_with_metadata(
        structured_llm, plain_llm, prompt, render, agent_name
    ).text


def invoke_structured_with_metadata(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> StructuredInvocationResult:
    """Require a schema-valid result, retrying structured output once.

    There is deliberately no free-text fallback. A non-structured answer must
    never flow into downstream decision nodes as if it had passed validation.
    """
    del plain_llm  # retained in the signature for call-site compatibility
    if structured_llm is None:
        raise StructuredOutputError(
            f"{agent_name}: structured output is unavailable for this provider"
        )

    last_error: Exception | None = None
    base_prompt = prompt
    for attempt in range(1, _MAX_STRUCTURED_ATTEMPTS + 1):
        attempt_prompt = base_prompt
        if attempt > 1:
            detail = str(last_error or "unknown validation failure")
            attempt_prompt = _append_instruction(
                base_prompt,
                "CORRECTION REQUIRED: The previous response did not satisfy the "
                f"structured output contract ({detail[:500]}). Return the complete "
                "schema-valid object now; do not omit required fields.",
            )
        try:
            raw_result = structured_llm.invoke(attempt_prompt)
            result = raw_result
            parsing_error = None
            if (
                isinstance(raw_result, dict)
                and {"raw", "parsed", "parsing_error"}.issubset(raw_result)
            ):
                result = raw_result.get("parsed")
                parsing_error = raw_result.get("parsing_error")
            if parsing_error is not None:
                raise StructuredOutputError(
                    f"structured output validation failed: {parsing_error}"
                )
            if result is None:
                raise StructuredOutputError("structured output returned no parsed result")
            if not isinstance(result, BaseModel):
                raise StructuredOutputError(
                    f"structured output returned {type(result).__name__}, expected Pydantic model"
                )
            return StructuredInvocationResult(
                text=render(result),
                mode="STRUCTURED",
                agent_name=agent_name,
                schema_name=type(result).__name__,
                attempts=attempt,
                parsed=result.model_dump(mode="json"),
            )
        except Exception as exc:
            last_error = exc
            logger.warning(
                "%s: structured-output attempt %d/%d failed (%s)",
                agent_name,
                attempt,
                _MAX_STRUCTURED_ATTEMPTS,
                exc,
            )

    raise StructuredOutputError(
        f"{agent_name}: failed to produce schema-valid output after "
        f"{_MAX_STRUCTURED_ATTEMPTS} attempts: {last_error}"
    ) from last_error
