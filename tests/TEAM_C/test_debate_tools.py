"""Tests for Bull/Bear debate tools."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.agents.researchers.debate_common import available_debate_tools
from tradingagents.extensions.decision.tools.debate_tools import (
    DEBATE_REACT_TOOLS,
    compare_to_benchmark,
    get_drawdown_risk_stats,
)

NOW = datetime(2024, 6, 1, 23, 59, tzinfo=timezone.utc)


def _fake_ohlcv(n: int = 90, drift: float = 0.003):
    rows = []
    price = 100.0
    for i in range(n):
        day = NOW - timedelta(days=n - i)
        open_p = price
        close_p = price * (1.0 + drift)
        rows.append(
            {
                "Date": day.replace(tzinfo=None),
                "Open": open_p,
                "High": close_p * 1.01,
                "Low": open_p * 0.99,
                "Close": close_p,
                "Volume": 1_000_000 + i * 1000,
            }
        )
        price = close_p
    return pd.DataFrame(rows)


def test_only_two_non_overlapping_debate_tools_registered():
    assert [tool.name for tool in DEBATE_REACT_TOOLS] == [
        "compare_to_benchmark_for_ticker",
        "get_drawdown_risk_stats_for_ticker",
    ]


def test_used_tool_is_unavailable_in_later_debate_turns():
    first = available_debate_tools([])
    assert len(first) == 2
    second = available_debate_tools(["compare_to_benchmark_for_ticker"])
    assert [tool.name for tool in second] == ["get_drawdown_risk_stats_for_ticker"]
    final = available_debate_tools(
        ["compare_to_benchmark_for_ticker", "get_drawdown_risk_stats_for_ticker"]
    )
    assert final == []


@patch("tradingagents.dataflows.stockstats_utils.load_ohlcv", return_value=_fake_ohlcv())
def test_benchmark_alpha_windows(mock_load):
    out = compare_to_benchmark("AAPL", "2024-06-01", benchmark="SPY")
    assert "windows" in out
    assert "20d" in out["windows"]
    assert out["windows"]["20d"]["alpha"] is not None


@patch("tradingagents.dataflows.stockstats_utils.load_ohlcv", return_value=_fake_ohlcv())
def test_drawdown_stats(mock_load):
    dd = get_drawdown_risk_stats("AAPL", "2024-06-01")
    assert dd["last_close"] > 0
    assert dd["current_drawdown_from_high"] <= 0
