"""Shared Bull/Bear debate context and tool-augmented LLM invocation."""

from __future__ import annotations
import re

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.tool_loop import invoke_with_tools_audit
from tradingagents.extensions.decision.tools.debate_tools import (
    DEBATE_REACT_TOOLS,
)
from tradingagents.extensions.decision.credibility.models import (
    ArtifactFieldRef,
    DebateTurnRecord,
)


def available_debate_tools(used_tools: list[str] | set[str]) -> list:
    """Return tools not yet executed; this is the per-run call-budget gate."""
    used = set(used_tools)
    return [tool for tool in DEBATE_REACT_TOOLS if tool.name not in used]


def _fundamentals_label(asset_type: str) -> str:
    if asset_type == "stock":
        return "Company fundamentals report"
    return "Asset fundamentals report (may be unavailable for crypto)"


def build_debate_user_prompt(
    *,
    side: str,
    state: dict,
    investment_debate_state: dict,
) -> str:
    """Build the user prompt for bull or bear (side is 'bull' or 'bear')."""
    asset_type = state.get("asset_type", "stock")
    target_label = "stock" if asset_type == "stock" else "asset"
    instrument_context = get_instrument_context_from_state(state)
    history = investment_debate_state.get("history", "")
    current_response = investment_debate_state.get("current_response", "")
    trade_date = state["trade_date"]
    symbol = state["company_of_interest"]
    claim_lines = [
        f"- {claim.get('claim_id')}: {claim.get('text')}"
        for claim in state.get("claims", [])
        if claim.get("importance") in {"CRITICAL", "MAJOR"}
    ]
    claim_ledger = "\n".join(claim_lines[-20:]) or "- no structured upstream claims available"

    used_tools = set(investment_debate_state.get("used_tools", []))
    available_names = [tool.name for tool in available_debate_tools(used_tools)]
    availability = (
        "\n".join(f"- {name}" for name in available_names)
        if available_names
        else "- none (both debate tools were already used earlier in this run)"
    )

    tools_blurb = f"""
Only these non-overlapping debate tools remain available in this run:
{availability}

Each tool may execute **at most once for the entire Bull/Bear debate**, not once
per turn. Already-used tools are removed from later turns. Call an available
tool only when it materially strengthens your rebuttal; otherwise rely on the
upstream analyst reports.
You may reuse a value already present in the conversation, but do not claim
that you called the tool again.
Preserve exact field semantics: ``max_drawdown_in_window`` is the maximum
drawdown over the returned ``lookback_bars``; never relabel it as a 20-day or
52-week statistic. ``days_since_window_high`` is not a calendar-day count.
Do not invent historical analogues, past return figures, valuation figures, or
support/resistance levels that are absent from the upstream reports or tools.
When rebutting the opponent, reference their specific claims and counter with tool-backed data.
"""

    if side == "bull":
        role = f"""You are a Bull Analyst advocating for investing in the {target_label}.
Emphasize growth, competitive advantages, positive trend/momentum, and constructive news.
Use drawdown stats to argue recovery potential when relevant."""
        opponent = f"Last bear argument: {current_response}"
    else:
        role = f"""You are a Bear Analyst making the case against investing in the {target_label}.
Emphasize valuation risk, negative momentum, drawdowns, weak relative strength, and adverse news.
Use benchmark alpha and drawdown stats to challenge bullish narratives."""
        opponent = f"Last bull argument: {current_response}"

    return f"""{role}

{tools_blurb}

{instrument_context}

Market research report: {state.get("market_report", "")}
Social media sentiment report: {state.get("sentiment_report", "")}
Latest world affairs news: {state.get("news_report", "")}
{_fundamentals_label(asset_type)}: {state.get("fundamentals_report", "")}

Conversation history: {history}
{opponent}

Structured claim ledger:
{claim_ledger}

Target one claim ID when possible. Deliver a compelling {side} argument in
conversational debate style with specific evidence. A role/persona is not evidence.
""" + get_language_instruction()


def run_debate_turn(
    llm, *, side: str, state: dict, investment_debate_state: dict
) -> tuple[str, list[str], list[dict]]:
    """Run one turn and return argument, cumulative usage and evidence events."""
    prompt = build_debate_user_prompt(
        side=side, state=state, investment_debate_state=investment_debate_state
    )
    already_used = set(investment_debate_state.get("used_tools", []))
    available_tools = available_debate_tools(already_used)

    # Memory recall tool — not counted against the debate tool budget
    provider = state.get("memory_provider")
    if provider:
        from tradingagents.extensions.memory.tools import create_memory_recall_tool

        role = "bull_researcher" if side == "bull" else "bear_researcher"
        available_tools = available_tools + [
            create_memory_recall_tool(
                provider,
                state["company_of_interest"],
                state["trade_date"],
                role,
            )
        ]

    content, used_now, events = invoke_with_tools_audit(
        llm,
        available_tools,
        prompt,
        run_id=state.get("run_id", "legacy-run"),
        producer_node=f"{side}_researcher_tools",
        max_tool_rounds=1,
    )
    prefix = "Bull Analyst" if side == "bull" else "Bear Analyst"
    cumulative = sorted(already_used | used_now)
    return f"{prefix}: {content}", cumulative, events


def build_debate_turn_record(
    *, side: str, count_before: int, argument: str, events: list[dict]
) -> dict:
    refs = []
    for event in events:
        payload = event.get("payload") or {}
        if payload.get("artifact_id") and payload.get("event_id"):
            refs.append(
                ArtifactFieldRef(
                    observation_event_id=payload["event_id"],
                    artifact_id=payload["artifact_id"],
                    selector="/",
                    schema_name=payload.get("tool_name") or "unstructured",
                    schema_version=payload.get("schema_version") or "1.0",
                )
            )
    return DebateTurnRecord(
        side=side,
        cycle=count_before // 2 + 1,
        target_claim_id=(
            re.search(r"\bclaim_[0-9a-f]{8,}\b", argument, re.I).group(0)
            if re.search(r"\bclaim_[0-9a-f]{8,}\b", argument, re.I)
            else None
        ),
        stance=argument,
        added_evidence_refs=refs,
        complete=True,
    ).model_dump(mode="json")
