"""Evidence capture shared by ToolNode, direct calls and custom tool loops."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from tradingagents.extensions.contracts import TraceEvent

from .models import SCHEMA_VERSION, stable_id

_SECRET_RE = re.compile(r"(api[_-]?key|token|secret|password|authorization)", re.I)


def redact_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): ("[REDACTED]" if _SECRET_RE.search(str(k)) else redact_arguments(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_arguments(v) for v in value]
    return value


def _status_for(result: Any, error: Exception | None) -> str:
    if error is not None:
        return "ERROR"
    text = str(result or "").upper()
    if not text.strip() or "NO_DATA_AVAILABLE" in text or "UNAVAILABLE" in text:
        return "EMPTY"
    if "TOOL ERROR:" in text:
        return "ERROR"
    return "SUCCESS"


def make_observation(
    *,
    run_id: str,
    producer_node: str,
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
    tool_call_id: str,
    attempt: int = 1,
    started_at: datetime | None = None,
    error: Exception | None = None,
    post_run_recomputed: bool = False,
) -> dict[str, Any]:
    started_at = started_at or datetime.now(timezone.utc)
    ended_at = datetime.now(timezone.utc)
    content = str(result or "")
    artifact_id = stable_id("artifact", content)
    event_id = stable_id(
        "event",
        {
            "run_id": run_id,
            "producer_node": producer_node,
            "tool_call_id": tool_call_id,
            "attempt": attempt,
        },
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "event_id": event_id,
        "tool_call_id": tool_call_id,
        "producer_node": producer_node,
        "tool_name": tool_name,
        "arguments_redacted": redact_arguments(arguments),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "attempt": attempt,
        "status": _status_for(result, error),
        "artifact_id": artifact_id,
        "artifact_media_type": "application/json"
        if content.lstrip().startswith(("{", "["))
        else "text/plain",
        "artifact_content": content,
        "canonical_hash": artifact_id.removeprefix("artifact_"),
        "post_run_recomputed": post_run_recomputed,
        "error": str(error) if error else None,
    }
    event = TraceEvent(
        timestamp=ended_at,
        source=producer_node,
        event_type="EVIDENCE_OBSERVATION",
        summary=f"{tool_name}: {payload['status']}",
        payload=payload,
    )
    return event.model_dump(mode="json")


def invoke_evidenced(
    *,
    run_id: str,
    producer_node: str,
    tool_name: str,
    arguments: dict[str, Any],
    call: Callable[[], Any],
    tool_call_id: str | None = None,
    attempt: int = 1,
    post_run_recomputed: bool = False,
) -> tuple[Any, dict[str, Any]]:
    tool_call_id = tool_call_id or stable_id(
        "call", {"run_id": run_id, "node": producer_node, "tool": tool_name, "args": arguments}
    )
    started_at = datetime.now(timezone.utc)
    error: Exception | None = None
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001
        error = exc
        result = f"tool error: {exc}"
    event = make_observation(
        run_id=run_id,
        producer_node=producer_node,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        tool_call_id=tool_call_id,
        attempt=attempt,
        started_at=started_at,
        error=error,
        post_run_recomputed=post_run_recomputed,
    )
    return result, event


def build_audited_tool_node(tools: list[Any], producer_node: str):
    """Return a LangGraph node that records ToolNode results as TraceEvents."""
    node = ToolNode(tools)

    def audited(state: dict[str, Any]) -> dict[str, Any]:
        output = node.invoke(state)
        messages = output.get("messages", []) if isinstance(output, dict) else []
        calls: dict[str, dict[str, Any]] = {}
        for message in reversed(state.get("messages", [])):
            for call in getattr(message, "tool_calls", None) or []:
                calls[str(call.get("id"))] = call
            if calls:
                break

        events = []
        rewritten = []
        run_id = state.get("run_id", "unknown-run")
        for message in messages:
            if not isinstance(message, ToolMessage):
                rewritten.append(message)
                continue
            call_id = str(message.tool_call_id)
            call = calls.get(call_id, {})
            name = call.get("name") or getattr(message, "name", None) or "unknown_tool"
            args = call.get("args") or {}
            event = make_observation(
                run_id=run_id,
                producer_node=producer_node,
                tool_name=name,
                arguments=args,
                result=message.content,
                tool_call_id=call_id,
            )
            events.append(event)
            artifact_id = event["payload"]["artifact_id"]
            content = f"[artifact_id={artifact_id}]\n{message.content}"
            rewritten.append(
                ToolMessage(
                    content=content,
                    tool_call_id=message.tool_call_id,
                    name=getattr(message, "name", None),
                )
            )
        return {"messages": rewritten, "audit_events": events}

    audited.tools_by_name = node.tools_by_name
    return audited


def artifact_payload(event: dict[str, Any]) -> tuple[str, str] | None:
    payload = event.get("payload") or {}
    artifact_id = payload.get("artifact_id")
    if not artifact_id:
        return None
    content = payload.get("artifact_content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    return str(artifact_id), content
