"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
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


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        run_id = state.get("run_id", "legacy-run")

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry

**Output requirement (mandatory):** The first content line of your answer MUST be exactly:
`**Rating**: <Buy|Overweight|Hold|Underweight|Sell>`
Use the English rating token even if the rest of the narrative is in another language.

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.{get_language_instruction()}"""

        invocation = invoke_structured_with_metadata(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )
        final_trade_decision = invocation.text
        parsed = parse_rating_strict(final_trade_decision, expected_label="rating")
        research_snapshot = next(
            (
                item
                for item in reversed(state.get("decision_snapshots", []))
                if item.get("stage") == "research_manager"
            ),
            None,
        )
        alignment = "NOT_COMPARABLE"
        if research_snapshot and parsed.parsed and research_snapshot.get("value"):
            alignment = (
                "ADOPTED"
                if parsed.parsed == research_snapshot.get("value")
                else "MODIFIED"
            )
        snapshot = DecisionSnapshot(
            snapshot_id=stable_id("snapshot", {"run_id": run_id, "stage": "portfolio_manager"}),
            run_id=run_id,
            stage="portfolio_manager",
            decision_type="PortfolioRating",
            value=parsed.parsed,
            parsed=parsed.parsed is not None,
            source=parsed.source,
            error=parsed.error,
            alignment=alignment,
            upstream_decision_ref=research_snapshot.get("snapshot_id")
            if research_snapshot
            else None,
            change_reason_refs=[
                item["constraint_id"]
                for item in risk_debate_state.get("constraints", [])
                if item.get("constraint_id")
            ]
            if alignment == "MODIFIED"
            else [],
        )
        claims = claims_from_invocation(
            run_id=run_id,
            agent="Portfolio Manager",
            stage="portfolio_manager",
            invocation=invocation,
            audit_events=state.get("audit_events", []),
            trade_date=state.get("trade_date", ""),
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
            "constraints": risk_debate_state.get("constraints", []),
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "structured_invocations": [invocation.model_dump(mode="json")],
            "decision_snapshots": [snapshot.model_dump(mode="json")],
            "claims": claims,
        }

    return portfolio_manager_node
