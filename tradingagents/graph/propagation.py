# TradingAgents/graph/propagation.py

from typing import Any
from uuid import uuid4

from tradingagents.agents.utils.agent_states import (
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        asset_type: str = "stock",
        past_context: str = "",
        instrument_context: str = "",
        **extra_state,
    ) -> dict[str, Any]:
        """Create the initial state for the agent graph.

        ``instrument_context`` is the deterministic ticker-identity string
        resolved once at run start (see
        ``TradingAgentsGraph.resolve_instrument_context``). When empty, agents
        fall back to ticker-only context via
        ``get_instrument_context_from_state``.

        ``extra_state`` kwargs are merged directly into the returned dict,
        allowing callers to inject per-agent memory fields without modifying
        the propagation signature for each new agent role.
        """
        state = {
            "messages": [("human", company_name)],
            "run_id": f"run_{uuid4().hex}",
            "audit_events": [],
            "structured_invocations": [],
            "decision_snapshots": [],
            "claims": [],
            "audit_scope_reasons": [],
            "company_of_interest": company_name,
            "asset_type": asset_type,
            "instrument_context": instrument_context,
            "trade_date": str(trade_date),
            "past_context": past_context,
            "investment_debate_state": InvestDebateState(
                {
                    "bull_history": "",
                    "bear_history": "",
                    "history": "",
                    "current_response": "",
                    "judge_decision": "",
                    "count": 0,
                    "used_tools": [],
                    "tool_events": [],
                    "debate_turns": [],
                    "no_novelty_cycles": 0,
                    "accepted_claim_ids": [],
                    "rejected_claim_ids": [],
                    "unresolved_claim_ids": [],
                }
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "aggressive_history": "",
                    "conservative_history": "",
                    "neutral_history": "",
                    "history": "",
                    "latest_speaker": "",
                    "current_aggressive_response": "",
                    "current_conservative_response": "",
                    "current_neutral_response": "",
                    "judge_decision": "",
                    "count": 0,
                    "constraints": [],
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }
        state.update(extra_state)
        return state

    def get_graph_args(self, callbacks: list | None = None) -> dict[str, Any]:
        """Get arguments for the graph invocation.

        Args:
            callbacks: Optional list of callback handlers for tool execution tracking.
                       Note: LLM callbacks are handled separately via LLM constructor.
        """
        config = {"recursion_limit": self.max_recur_limit}
        if callbacks:
            config["callbacks"] = callbacks
        return {
            "stream_mode": "values",
            "config": config,
        }
