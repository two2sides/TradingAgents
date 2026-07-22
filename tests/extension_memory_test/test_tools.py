"""Tests for memory recall tool factory (extensions/memory/tools.py)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from tradingagents.extensions.memory.tools import (
    _build_memory_query,
    _ROLE_TOOL_DESCRIPTIONS,
    create_memory_recall_tool,
)


# ── Tool factory ───────────────────────────────────────────────────────

class TestCreateMemoryRecallTool:
    def test_returns_callable(self):
        provider = MagicMock()
        provider.retrieve.return_value = MagicMock(items=[])
        provider.format_context_for_prompt.return_value = ""

        tool = create_memory_recall_tool(provider, "AAPL", "2026-07-22", "market_analyst")
        assert callable(tool)

    def test_tool_has_correct_name(self):
        provider = MagicMock()
        provider.retrieve.return_value = MagicMock(items=[])
        provider.format_context_for_prompt.return_value = ""

        tool = create_memory_recall_tool(provider, "AAPL", "2026-07-22", "market_analyst")
        assert tool.__name__ == "recall_historical_decisions"

    def test_tool_has_docstring(self):
        provider = MagicMock()
        provider.retrieve.return_value = MagicMock(items=[])
        provider.format_context_for_prompt.return_value = ""

        tool = create_memory_recall_tool(provider, "AAPL", "2026-07-22", "market_analyst")
        assert tool.__doc__ is not None
        assert len(tool.__doc__) > 50

    def test_signature_accepts_query(self):
        provider = MagicMock()
        provider.retrieve.return_value = MagicMock(items=[])
        provider.format_context_for_prompt.return_value = ""

        tool = create_memory_recall_tool(provider, "AAPL", "2026-07-22", "news_analyst")
        sig = inspect.signature(tool)
        assert "query" in sig.parameters

    def test_calls_retrieve_on_invoke(self):
        from tradingagents.extensions.contracts import MemoryContext

        ctx = MemoryContext(
            as_of=__import__("datetime").datetime(2026, 7, 22),
            items=[],
            summary="One relevant memory.",
        )
        provider = MagicMock()
        provider.retrieve.return_value = ctx
        provider.format_context_for_prompt.return_value = "[2026-01-05 | AAPL] Test memory."

        tool = create_memory_recall_tool(provider, "AAPL", "2026-07-22", "bull_researcher")
        result = tool("Is this bull thesis reliable?")

        provider.retrieve.assert_called_once()
        args = provider.retrieve.call_args[0][0]
        # Symbol must be normalised and role must be in metadata
        assert args.symbol == "AAPL"
        assert args.metadata["agent_role"] == "bull_researcher"
        assert "[2026-01-05 | AAPL]" in result

    def test_empty_query_graceful(self):
        provider = MagicMock()

        tool = create_memory_recall_tool(provider, "NVDA", "2026-07-22", "market_analyst")
        result = tool("")

        assert "[Memory]" in result
        provider.retrieve.assert_not_called()

    def test_provider_error_graceful(self):
        provider = MagicMock()
        provider.retrieve.side_effect = RuntimeError("DB down")

        tool = create_memory_recall_tool(provider, "NVDA", "2026-07-22", "market_analyst")
        result = tool("test query")

        assert "[Memory]" in result
        assert "unavailable" in result.lower() or "temporarily" in result.lower()

    def test_empty_result_returns_placeholder(self):
        provider = MagicMock()
        provider.retrieve.return_value = MagicMock(items=[])
        provider.format_context_for_prompt.return_value = ""

        tool = create_memory_recall_tool(provider, "MSFT", "2026-07-22", "fundamentals_analyst")
        result = tool("PE compression similar to 2024")

        assert "No relevant past decisions" in result or "[Memory]" in result

    def test_each_role_has_different_description(self):
        """Descriptions must differ so each agent gets role-appropriate guidance."""
        descs = set(_ROLE_TOOL_DESCRIPTIONS.values())
        assert len(descs) == len(_ROLE_TOOL_DESCRIPTIONS), "All role descriptions must be unique"


# ── Query construction ─────────────────────────────────────────────────

class TestBuildMemoryQuery:
    def test_symbol_normalised_to_uppercase(self):
        q = _build_memory_query("aapl", "2026-07-22", "market_analyst")
        assert q.symbol == "AAPL"

    def test_agent_role_in_metadata(self):
        q = _build_memory_query("AAPL", "2026-07-22", "bear_researcher")
        assert q.metadata["agent_role"] == "bear_researcher"

    def test_limit_is_three(self):
        q = _build_memory_query("AAPL", "2026-07-22", "market_analyst")
        assert q.limit == 3

    def test_bad_date_falls_back_to_now(self):
        q = _build_memory_query("AAPL", "not-a-date", "market_analyst")
        assert q.as_of is not None


# ── Role-to-tool mapping ───────────────────────────────────────────────

class TestToolRoleCoverage:
    """All 5 tool-based roles must have a description."""

    @pytest.mark.parametrize("role", [
        "market_analyst",
        "fundamentals_analyst",
        "news_analyst",
        "bull_researcher",
        "bear_researcher",
    ])
    def test_role_has_description(self, role):
        assert role in _ROLE_TOOL_DESCRIPTIONS, f"{role} missing tool description"
        assert len(_ROLE_TOOL_DESCRIPTIONS[role]) > 30

    @pytest.mark.parametrize("role", [
        "portfolio_manager",
        "research_manager",
        "sentiment_analyst",
        "trader",
        "risk_aggressive",
        "risk_conservative",
        "risk_neutral",
    ])
    def test_prompt_only_roles_have_no_description(self, role):
        """Prompt-injected roles should NOT have tool descriptions."""
        assert role not in _ROLE_TOOL_DESCRIPTIONS, (
            f"{role} should not be in _ROLE_TOOL_DESCRIPTIONS — it uses prompt injection, not tools"
        )
