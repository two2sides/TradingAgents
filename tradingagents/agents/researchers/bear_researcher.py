from tradingagents.agents.researchers.debate_common import (
    build_debate_turn_record,
    run_debate_turn,
)

# Output-language enforcement is centralized in debate_common via
# get_language_instruction().


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        argument, used_tools, tool_events = run_debate_turn(
            llm, side="bear", state=state, investment_debate_state=investment_debate_state
        )
        turn = build_debate_turn_record(
            side="bear",
            count_before=investment_debate_state["count"],
            argument=argument,
            events=tool_events,
        )
        prior_turns = investment_debate_state.get("debate_turns", [])
        all_turns = prior_turns + [turn]
        current_cycle = turn["cycle"]
        current_ids = {
            ref["artifact_id"]
            for item in all_turns
            if item.get("cycle") == current_cycle
            for ref in item.get("added_evidence_refs", [])
        }
        prior_ids = {
            ref["artifact_id"]
            for item in all_turns
            if item.get("cycle", 0) < current_cycle
            for ref in item.get("added_evidence_refs", [])
        }
        has_novelty = bool(current_ids - prior_ids)
        no_novelty_cycles = (
            0
            if has_novelty
            else investment_debate_state.get("no_novelty_cycles", 0) + 1
        )

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "bear_history": bear_history + "\n" + argument,
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
            "used_tools": used_tools,
            "tool_events": investment_debate_state.get("tool_events", []) + tool_events,
            "debate_turns": all_turns,
            "no_novelty_cycles": no_novelty_cycles,
            "accepted_claim_ids": investment_debate_state.get("accepted_claim_ids", []),
            "rejected_claim_ids": investment_debate_state.get("rejected_claim_ids", []),
            "unresolved_claim_ids": investment_debate_state.get("unresolved_claim_ids", []),
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "audit_events": tool_events,
        }

    return bear_node
