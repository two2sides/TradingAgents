"""Ticker+date OHLCV tools for legacy ReAct Market Analyst.

These wrap deterministic numerical helpers so agents call ``symbol`` + ``trade_date``
instead of shipping raw bar JSON.
"""

from __future__ import annotations

import json
from typing import Annotated

from tradingagents.extensions.decision.tools.ohlcv_tools import (
    analyze_multi_horizon_ohlcv,
    detect_price_gap,
    detect_volume_anomaly,
)

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover
    tool = None


def _snapshot(symbol: str, trade_date: str):
    from tradingagents.extensions.decision.tools.market_snapshot import build_market_snapshot

    return build_market_snapshot(symbol, trade_date)


def _multi_horizon_impl(symbol: str, trade_date: str) -> str:
    market = _snapshot(symbol, trade_date)
    return json.dumps(analyze_multi_horizon_ohlcv(market), ensure_ascii=False, default=str)


def _volume_impl(symbol: str, trade_date: str) -> str:
    market = _snapshot(symbol, trade_date)
    return json.dumps(detect_volume_anomaly(market), ensure_ascii=False, default=str)


def _gap_impl(symbol: str, trade_date: str) -> str:
    market = _snapshot(symbol, trade_date)
    return json.dumps(detect_price_gap(market), ensure_ascii=False, default=str)


if tool is not None:

    @tool
    def analyze_multi_horizon_ohlcv_for_ticker(
        symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
        trade_date: Annotated[str, "analysis date YYYY-MM-DD"],
    ) -> str:
        """Deterministic multi-horizon OHLCV features (short/medium/long returns, SMA ratios, vol)."""
        return _multi_horizon_impl(symbol, trade_date)

    @tool
    def detect_volume_anomaly_for_ticker(
        symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
        trade_date: Annotated[str, "analysis date YYYY-MM-DD"],
    ) -> str:
        """Deterministic volume z-score vs recent history (abnormally high/low/normal)."""
        return _volume_impl(symbol, trade_date)

    @tool
    def detect_price_gap_for_ticker(
        symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
        trade_date: Annotated[str, "analysis date YYYY-MM-DD"],
    ) -> str:
        """Deterministic open-gap detection vs previous close."""
        return _gap_impl(symbol, trade_date)

else:  # pragma: no cover

    def analyze_multi_horizon_ohlcv_for_ticker(symbol: str, trade_date: str) -> str:
        return _multi_horizon_impl(symbol, trade_date)

    def detect_volume_anomaly_for_ticker(symbol: str, trade_date: str) -> str:
        return _volume_impl(symbol, trade_date)

    def detect_price_gap_for_ticker(symbol: str, trade_date: str) -> str:
        return _gap_impl(symbol, trade_date)


MARKET_REACT_TOOLS = [
    analyze_multi_horizon_ohlcv_for_ticker,
    detect_volume_anomaly_for_ticker,
    detect_price_gap_for_ticker,
]
