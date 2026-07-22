"""Memory recall tool — lets agents query historical decisions on demand.

Each agent that has a tool-calling loop can include this tool.  The LLM
decides *when* to call it by supplying a natural-language description of
the pattern or thesis it wants to validate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

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


def create_memory_recall_tool(provider: Any, symbol: str, trade_date: str, role: str):
    """Return a callable tool that the LLM can invoke to recall history.

    Args:
        provider: An ``EnhancedMemoryProvider`` instance.
        symbol: The ticker being analysed (from state).
        trade_date: The current analysis date as an ISO-ish string.
        role: The agent role key (e.g. ``"market_analyst"``).

    Returns:
        A function with signature ``(query: str) -> str``, suitable for
        inclusion in a LangChain tool list.
    """
    description = _ROLE_TOOL_DESCRIPTIONS.get(role, _ROLE_TOOL_DESCRIPTIONS["market_analyst"])

    def recall_historical_decisions(query: str) -> str:
        """Search past decisions relevant to the current analysis context."""
        if not query or not query.strip():
            return "[Memory] No query provided."

        try:
            memory_query = _build_memory_query(symbol, trade_date, role)
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

    # Attach metadata so the LLM sees the tool description
    recall_historical_decisions.__name__ = "recall_historical_decisions"
    recall_historical_decisions.__doc__ = description

    return recall_historical_decisions
