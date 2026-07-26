"""Memory recall tool — lets agents query historical decisions on demand.

Each agent that has a tool-calling loop can include this tool.  The LLM
decides *when* to call it by supplying a natural-language description of
the pattern or thesis it wants to validate.

The tool is a real LangChain ``StructuredTool`` (with ``.name`` / ``.invoke``)
so it works with:

- analyst ``bind_tools`` + ``tool.name`` prompt wiring
- LangGraph ``ToolNode`` execution
- Bull/Bear custom ``invoke_with_tools_audit`` loops
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Tool descriptions tailored to each role, so the LLM knows when to reach
# for memory.  Kept outside the factory so they are importable for testing.
_ROLE_TOOL_DESCRIPTIONS: dict[str, str] = {
    "market_analyst": (
        "Search past trading decisions for market conditions similar to the "
        "current one. Use this when you observe an unusual technical pattern "
        "(divergence, volume anomaly, breakout/breakdown) and want to check "
        "how reliable that signal has been historically. Returns formatted "
        "lessons from the most relevant past decisions with their outcomes."
    ),
    "fundamentals_analyst": (
        "Search past trading decisions for similar valuation or fundamental "
        "setups. Use this when PE, revenue growth, margins, or debt levels "
        "resemble a past situation and you want to see how that thesis played "
        "out. Returns relevant historical decisions with outcome data."
    ),
    "news_analyst": (
        "Search past decisions for similar macroeconomic or news-driven "
        "contexts. Use this when the current macro event (rate decision, "
        "policy change, geopolitical development) mirrors a past situation "
        "and you want to understand how the market digested it previously."
    ),
    "bull_researcher": (
        "Search past decisions to validate a bullish thesis. Use this to "
        "check whether a particular bullish argument (growth catalyst, "
        "undervaluation, competitive advantage) has been correct or incorrect "
        "in similar past situations. Returns lessons from verified outcomes."
    ),
    "bear_researcher": (
        "Search past decisions to validate a bearish thesis. Use this to "
        "check whether a particular bearish argument (downside risk, "
        "overvaluation, headwind) was actually borne out in similar past "
        "situations. Returns lessons from verified outcomes."
    ),
}

_DEFAULT_TOOL_DESCRIPTION = (
    "Search past trading decisions relevant to the current analysis. "
    "Provide a short natural-language query describing the pattern or thesis "
    "to validate. Returns formatted lessons from relevant past decisions."
)

# Request-scoped binding used by ToolNode execution (symbol/date/role/provider).
_recall_context: dict[str, Any] = {
    "provider": None,
    "symbol": None,
    "trade_date": None,
    "role": "market_analyst",
}


def _build_memory_query(symbol: str, as_of_str: str, role: str) -> Any:
    """Build a minimal MemoryQuery for tool-based retrieval."""
    from tradingagents.extensions.contracts import (
        MarketSnapshot,
        MemoryQuery,
        PortfolioState,
    )

    try:
        as_of = datetime.fromisoformat(str(as_of_str))
    except (ValueError, TypeError):
        as_of = datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    return MemoryQuery(
        symbol=symbol,
        as_of=as_of,
        market=MarketSnapshot(symbol=symbol, as_of=as_of),
        portfolio=PortfolioState(as_of=as_of, cash=0, total_equity=0),
        limit=3,
        metadata={"agent_role": role},
    )


def set_memory_recall_context(
    provider: Any,
    symbol: str,
    trade_date: str,
    role: str,
) -> None:
    """Bind request-scoped values for ToolNode / bind_tools execution."""
    _recall_context["provider"] = provider
    _recall_context["symbol"] = symbol
    _recall_context["trade_date"] = trade_date
    _recall_context["role"] = role or "market_analyst"
    description = _ROLE_TOOL_DESCRIPTIONS.get(
        _recall_context["role"], _DEFAULT_TOOL_DESCRIPTION
    )
    recall_historical_decisions.description = description


def clear_memory_recall_context() -> None:
    """Clear request-scoped recall binding."""
    _recall_context["provider"] = None
    _recall_context["symbol"] = None
    _recall_context["trade_date"] = None
    _recall_context["role"] = "market_analyst"
    recall_historical_decisions.description = _DEFAULT_TOOL_DESCRIPTION


def _resolve_provider() -> Any:
    from tradingagents.extensions.memory import get_active_provider

    return _recall_context.get("provider") or get_active_provider()


def _recall_historical_decisions_impl(query: str) -> str:
    """Search past decisions relevant to the current analysis context."""
    if not query or not str(query).strip():
        return "[Memory] No query provided."

    provider = _resolve_provider()
    symbol = _recall_context.get("symbol")
    trade_date = _recall_context.get("trade_date")
    role = _recall_context.get("role") or "market_analyst"

    if provider is None or not symbol or not trade_date:
        return "[Memory] Recall temporarily unavailable."

    try:
        memory_query = _build_memory_query(str(symbol), str(trade_date), str(role))
        # Thread the LLM's query text into metadata so the retrieval
        # pipeline can use it for embedding alongside the role template.
        memory_query.metadata["llm_query"] = str(query).strip()
        ctx = provider.retrieve(memory_query)
        formatted = provider.format_context_for_prompt(ctx)

        if not formatted:
            return (
                "[Memory] No relevant past decisions found for "
                f"{symbol} ({role})."
            )
        return formatted
    except Exception:
        logger.debug(
            "Memory recall failed for %s (%s)", symbol, role, exc_info=True,
        )
        return "[Memory] Recall temporarily unavailable."


recall_historical_decisions = StructuredTool.from_function(
    func=_recall_historical_decisions_impl,
    name="recall_historical_decisions",
    description=_DEFAULT_TOOL_DESCRIPTION,
)


def create_memory_recall_tool(provider: Any, symbol: str, trade_date: str, role: str):
    """Return a LangChain tool the LLM can invoke to recall history.

    Args:
        provider: An ``EnhancedMemoryProvider`` (or compatible) instance.
        symbol: The ticker being analysed (from state).
        trade_date: The current analysis date as an ISO-ish string.
        role: The agent role key (e.g. ``"market_analyst"``).

    Returns:
        A LangChain ``StructuredTool`` named ``recall_historical_decisions``.
    """
    set_memory_recall_context(provider, symbol, trade_date, role)
    return recall_historical_decisions
