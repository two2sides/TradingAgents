"""Numerical OHLCV tools for ReAct agents and DecisionProvider internals."""

from __future__ import annotations

import json
from typing import Annotated, Any

from tradingagents.extensions.contracts import MarketBar, MarketSnapshot
from tradingagents.extensions.decision.quant_engine import (
    compute_quant_signal,
    multi_horizon_ohlcv_summary,
)

try:
    from langchain_core.tools import tool as langchain_tool
except ImportError:  # pragma: no cover - langchain always present in this project
    langchain_tool = None


def analyze_multi_horizon_ohlcv(market: MarketSnapshot) -> dict[str, Any]:
    """Return short/medium/long horizon numerical features from bars."""
    return multi_horizon_ohlcv_summary(market)


def detect_volume_anomaly(market: MarketSnapshot, window: int = 20) -> dict[str, Any]:
    signal = compute_quant_signal(market)
    z = signal.features.get("volume_z_20")
    label = "normal"
    if z is not None:
        if z >= 2.0:
            label = "abnormally_high"
        elif z <= -2.0:
            label = "abnormally_low"
    return {
        "symbol": market.symbol,
        "volume_z": z,
        "label": label,
        "window": window,
        "warnings": signal.warnings,
    }


def detect_price_gap(market: MarketSnapshot) -> dict[str, Any]:
    signal = compute_quant_signal(market)
    gap = signal.features.get("gap_1")
    label = "none"
    if gap is not None:
        if gap >= 0.02:
            label = "gap_up"
        elif gap <= -0.02:
            label = "gap_down"
    return {
        "symbol": market.symbol,
        "gap": gap,
        "label": label,
        "warnings": signal.warnings,
    }


def bars_from_records(records: list[dict[str, Any]], symbol: str, as_of_iso: str) -> MarketSnapshot:
    """Helper for tool callers that pass JSON-serializable bar dicts."""
    from datetime import datetime

    as_of = datetime.fromisoformat(as_of_iso)
    bars = [
        MarketBar(
            timestamp=datetime.fromisoformat(r["timestamp"]),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r.get("volume", 0)),
        )
        for r in records
    ]
    return MarketSnapshot(symbol=symbol, as_of=as_of, bars=bars)


def _tool_multi_horizon(
    symbol: Annotated[str, "ticker symbol"],
    as_of_iso: Annotated[str, "ISO-8601 as_of timestamp"],
    bars_json: Annotated[str, "JSON list of OHLCV dicts with timestamp/open/high/low/close/volume"],
) -> str:
    """Compute short/medium/long horizon OHLCV numerical features from bars JSON."""
    records = json.loads(bars_json)
    market = bars_from_records(records, symbol, as_of_iso)
    return json.dumps(analyze_multi_horizon_ohlcv(market), ensure_ascii=False, default=str)


def _tool_volume_anomaly(
    symbol: Annotated[str, "ticker symbol"],
    as_of_iso: Annotated[str, "ISO-8601 as_of timestamp"],
    bars_json: Annotated[str, "JSON list of OHLCV dicts"],
) -> str:
    """Detect whether the latest volume is abnormally high or low vs its history."""
    records = json.loads(bars_json)
    market = bars_from_records(records, symbol, as_of_iso)
    return json.dumps(detect_volume_anomaly(market), ensure_ascii=False, default=str)


def _tool_price_gap(
    symbol: Annotated[str, "ticker symbol"],
    as_of_iso: Annotated[str, "ISO-8601 as_of timestamp"],
    bars_json: Annotated[str, "JSON list of OHLCV dicts"],
) -> str:
    """Detect an opening gap relative to the previous close."""
    records = json.loads(bars_json)
    market = bars_from_records(records, symbol, as_of_iso)
    return json.dumps(detect_price_gap(market), ensure_ascii=False, default=str)


if langchain_tool is not None:
    analyze_multi_horizon_ohlcv_tool = langchain_tool(_tool_multi_horizon)
    detect_volume_anomaly_tool = langchain_tool(_tool_volume_anomaly)
    detect_price_gap_tool = langchain_tool(_tool_price_gap)
else:  # pragma: no cover
    analyze_multi_horizon_ohlcv_tool = _tool_multi_horizon
    detect_volume_anomaly_tool = _tool_volume_anomaly
    detect_price_gap_tool = _tool_price_gap


REACT_OHLCV_TOOLS = [
    analyze_multi_horizon_ohlcv_tool,
    detect_volume_anomaly_tool,
    detect_price_gap_tool,
]
