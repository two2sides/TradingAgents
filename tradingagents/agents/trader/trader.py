"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_with_metadata,
)
from tradingagents.agents.utils.rating import parse_rating_strict
from tradingagents.extensions.decision.credibility.models import DecisionSnapshot, stable_id
from tradingagents.extensions.decision.credibility.claims import claims_from_invocation


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        run_id = state.get("run_id", "legacy-run")
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
                ),
            },
        ]

        invocation = invoke_structured_with_metadata(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )
        trader_plan = invocation.text
        parsed = parse_rating_strict(trader_plan, expected_label="action")
        snapshot = DecisionSnapshot(
            snapshot_id=stable_id("snapshot", {"run_id": run_id, "stage": "trader"}),
            run_id=run_id,
            stage="trader",
            decision_type="TraderAction",
            value=parsed.parsed,
            parsed=parsed.parsed is not None,
            source=parsed.source,
            error=parsed.error,
            alignment="NOT_COMPARABLE",
            upstream_decision_ref=next(
                (
                    item.get("snapshot_id")
                    for item in reversed(state.get("decision_snapshots", []))
                    if item.get("stage") == "research_manager"
                ),
                None,
            ),
        )
        claims = claims_from_invocation(
            run_id=run_id,
            agent="Trader",
            stage="trader",
            invocation=invocation,
            audit_events=state.get("audit_events", []),
            trade_date=state.get("trade_date", ""),
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
            "structured_invocations": [invocation.model_dump(mode="json")],
            "decision_snapshots": [snapshot.model_dump(mode="json")],
            "claims": claims,
        }

    return functools.partial(trader_node, name="Trader")
