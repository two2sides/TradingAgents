from tradingagents.agents.researchers.debate_common import (
    build_debate_turn_record,
    run_debate_turn,
)

# Output-language enforcement is centralized in debate_common via
# get_language_instruction().


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        argument, used_tools, tool_events = run_debate_turn(
            llm, side="bull", state=state, investment_debate_state=investment_debate_state
        )
        turn = build_debate_turn_record(
            side="bull",
            count_before=investment_debate_state["count"],
            argument=argument,
            events=tool_events,
        )

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            "used_tools": used_tools,
            "tool_events": investment_debate_state.get("tool_events", []) + tool_events,
            "debate_turns": investment_debate_state.get("debate_turns", []) + [turn],
            "no_novelty_cycles": investment_debate_state.get("no_novelty_cycles", 0),
            "accepted_claim_ids": investment_debate_state.get("accepted_claim_ids", []),
            "rejected_claim_ids": investment_debate_state.get("rejected_claim_ids", []),
            "unresolved_claim_ids": investment_debate_state.get("unresolved_claim_ids", []),
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "audit_events": tool_events,
        }

    return bull_node
