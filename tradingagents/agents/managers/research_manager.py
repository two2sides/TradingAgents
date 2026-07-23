"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_with_metadata,
)
from tradingagents.agents.utils.rating import parse_rating_strict
from tradingagents.extensions.decision.credibility.models import DecisionSnapshot, stable_id
from tradingagents.extensions.decision.credibility.claims import claims_from_invocation


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        run_id = state.get("run_id", "legacy-run")
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]
        claim_ledger = "\n".join(
            f"- {claim.get('claim_id')}: {claim.get('text')} "
            f"[{claim.get('verification_status')}]"
            for claim in state.get("claims", [])
            if claim.get("importance") in {"CRITICAL", "MAJOR"}
        ) or "- no structured claims available"

        memory_context = state.get("memory_research_manager", "")
        memory_block = (
            f"- Lessons from past debate outcomes:\n{memory_context}\n"
            if memory_context
            else ""
        )

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one):
- **Buy**: Strongest bullish conviction; maximum permitted exposure
- **Overweight**: Constructive view; high exposure
- **Hold**: Genuinely balanced view; neutral exposure
- **Underweight**: Cautious view; small defensive exposure
- **Sell**: Strong bearish conviction; zero exposure

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced.

---

{memory_block}
**Debate History:**
{history}

**Structured Claim Ledger:**
{claim_ledger}

Populate accepted_claim_ids, rejected_claim_ids, and unresolved_claim_ids using
only IDs shown above. Preserve unresolved minority evidence instead of forcing
false consensus.""" + NO_EXTERNAL_TOOLS + get_language_instruction()

        invocation = invoke_structured_with_metadata(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )
        investment_plan = invocation.text
        parsed = parse_rating_strict(investment_plan, expected_label="recommendation")
        snapshot = DecisionSnapshot(
            snapshot_id=stable_id("snapshot", {"run_id": run_id, "stage": "research_manager"}),
            run_id=run_id,
            stage="research_manager",
            decision_type="PortfolioRating",
            value=parsed.parsed,
            parsed=parsed.parsed is not None,
            source=parsed.source,
            error=parsed.error,
        )
        claims = claims_from_invocation(
            run_id=run_id,
            agent="Research Manager",
            stage="research_manager",
            invocation=invocation,
            audit_events=state.get("audit_events", []),
            trade_date=state.get("trade_date", ""),
        )
        parsed_plan = invocation.parsed or {}

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
            "used_tools": investment_debate_state.get("used_tools", []),
            "tool_events": investment_debate_state.get("tool_events", []),
            "debate_turns": investment_debate_state.get("debate_turns", []),
            "no_novelty_cycles": investment_debate_state.get("no_novelty_cycles", 0),
            "accepted_claim_ids": parsed_plan.get("accepted_claim_ids", []),
            "rejected_claim_ids": parsed_plan.get("rejected_claim_ids", []),
            "unresolved_claim_ids": parsed_plan.get("unresolved_claim_ids", []),
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
            "structured_invocations": [invocation.model_dump(mode="json")],
            "decision_snapshots": [snapshot.model_dump(mode="json")],
            "claims": claims,
        }

    return research_manager_node
