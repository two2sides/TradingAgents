"""Minimal ReAct tool loop for agents that are not wired to LangGraph ToolNodes."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import HumanMessage, ToolMessage
from tradingagents.extensions.decision.credibility import invoke_evidenced


def invoke_with_tools(
    llm: Any,
    tools: list[Any],
    user_prompt: str,
    *,
    max_tool_rounds: int = 2,
) -> str:
    """Run up to ``max_tool_rounds`` of tool calls; return final text content."""
    content, _ = invoke_with_tools_trace(
        llm, tools, user_prompt, max_tool_rounds=max_tool_rounds
    )
    return content


def invoke_with_tools_trace(
    llm: Any,
    tools: list[Any],
    user_prompt: str,
    *,
    max_tool_rounds: int = 2,
) -> tuple[str, set[str]]:
    """Run a tool loop and return final text plus actually executed tool names."""
    if not tools:
        return str(llm.invoke(user_prompt).content), set()

    by_name: dict[str, Callable[..., str]] = {t.name: t.invoke for t in tools}
    llm_bound = llm.bind_tools(tools)
    messages: list[Any] = [HumanMessage(content=user_prompt)]
    used_tools: set[str] = set()

    for _ in range(max_tool_rounds + 1):
        response = llm_bound.invoke(messages)
        if not getattr(response, "tool_calls", None):
            return str(response.content or ""), used_tools

        messages.append(response)
        for call in response.tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            fn = by_name.get(name)
            if fn is None:
                result = f"unknown tool: {name}"
            elif name in used_tools:
                result = (
                    f"tool call budget exhausted: {name} already executed "
                    "in this tool loop"
                )
            else:
                used_tools.add(name)
                try:
                    result = fn(args)
                except Exception as exc:  # noqa: BLE001
                    result = f"tool error: {exc}"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    return str(response.content or ""), used_tools


def invoke_with_tools_audit(
    llm: Any,
    tools: list[Any],
    user_prompt: str,
    *,
    run_id: str,
    producer_node: str,
    max_tool_rounds: int = 2,
) -> tuple[str, set[str], list[dict]]:
    """Run the custom tool loop and return text, used names and TraceEvents."""
    if not tools:
        return str(llm.invoke(user_prompt).content), set(), []

    by_name: dict[str, Callable[..., str]] = {t.name: t.invoke for t in tools}
    llm_bound = llm.bind_tools(tools)
    messages: list[Any] = [HumanMessage(content=user_prompt)]
    used_tools: set[str] = set()
    events: list[dict] = []

    for _ in range(max_tool_rounds + 1):
        response = llm_bound.invoke(messages)
        if not getattr(response, "tool_calls", None):
            return str(response.content or ""), used_tools, events
        messages.append(response)
        for call in response.tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            fn = by_name.get(name)
            if fn is None:
                result = f"unknown tool: {name}"
            elif name in used_tools:
                result = f"tool call budget exhausted: {name} already executed in this tool loop"
            else:
                used_tools.add(name)
                result, event = invoke_evidenced(
                    run_id=run_id,
                    producer_node=producer_node,
                    tool_name=name,
                    arguments=args,
                    call=lambda fn=fn, args=args: fn(args),
                    tool_call_id=str(call["id"]),
                )
                events.append(event)
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return str(response.content or ""), used_tools, events
