"""Session and resource helpers shared by Streamlit pages."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from tradingagents.extensions.paper_trading import (
    RatingAllocationPolicy,
    SQLiteRunStore,
    TradingAgentsGraphDecisionProvider,
)


@dataclass(frozen=True, slots=True)
class AgentRuntime:
    """Cached heavyweight resources for the real TradingAgents mode."""

    decision_provider: Any
    memory_provider: Any
    details: dict[str, Any]


@st.cache_resource(show_spinner=False)
def get_run_store() -> SQLiteRunStore:
    configured = os.getenv("TRADINGAGENTS_RUN_STORE")
    return SQLiteRunStore(Path(configured)) if configured else SQLiteRunStore()


@st.cache_resource(show_spinner=False, max_entries=4)
def get_agent_runtime(
    selected_analysts: tuple[str, ...],
    max_position_weight: float,
) -> AgentRuntime:
    """Build the graph and B's persistent RAG provider only when real mode is used."""

    load_dotenv()
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.extensions.memory import EnhancedMemoryProvider
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    config = deepcopy(DEFAULT_CONFIG)
    config["memory_provider"] = None
    graph = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        debug=False,
        config=config,
    )
    memory_provider = EnhancedMemoryProvider(
        config,
        llm_client=graph.quick_thinking_llm,
    )
    decision_provider = TradingAgentsGraphDecisionProvider(
        graph,
        RatingAllocationPolicy(max_position_weight=max_position_weight),
    )
    return AgentRuntime(
        decision_provider=decision_provider,
        memory_provider=memory_provider,
        details={
            "llm_provider": config["llm_provider"],
            "quick_model": config["quick_think_llm"],
            "deep_model": config["deep_think_llm"],
            "analysts": list(selected_analysts),
            "memory_embedding": config["memory_embedding"],
            "memory_embedding_model": config["memory_embedding_model"],
            "allocation_policy": decision_provider.policy.to_dict(),
        },
    )


def select_run(run_id: str) -> None:
    st.session_state["selected_run_id"] = run_id


def selected_run_id() -> str | None:
    return st.session_state.get("selected_run_id")


__all__ = [
    "AgentRuntime",
    "get_agent_runtime",
    "get_run_store",
    "select_run",
    "selected_run_id",
]
