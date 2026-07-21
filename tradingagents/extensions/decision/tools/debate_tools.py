"""Non-overlapping deterministic tools for Bull / Bear researchers.

Market trend/volume/gap and news sentiment stay with upstream analysts. Debate
researchers only receive capabilities not already exposed upstream:
relative benchmark performance and drawdown/recovery statistics.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import numpy as np

from tradingagents.extensions.decision.tools.market_snapshot import build_market_snapshot

try:
    from langchain_core.tools import tool as langchain_tool
except ImportError:  # pragma: no cover
    langchain_tool = None


def _resolve_benchmark(symbol: str) -> str:
    from tradingagents.default_config import DEFAULT_CONFIG

    explicit = DEFAULT_CONFIG.get("benchmark_ticker")
    if explicit:
        return str(explicit)
    sym = symbol.upper()
    for suffix, bench in (DEFAULT_CONFIG.get("benchmark_map") or {}).items():
        if suffix and sym.endswith(suffix):
            return bench
    return "SPY"


def compare_to_benchmark(
    symbol: str,
    trade_date: str,
    benchmark: str | None = None,
) -> dict[str, Any]:
    """Relative performance vs regional benchmark (default SPY for US)."""
    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    bench = (benchmark or _resolve_benchmark(symbol)).upper()
    stock_df = load_ohlcv(symbol, trade_date)
    bench_df = load_ohlcv(bench, trade_date)
    out: dict[str, Any] = {
        "symbol": symbol.upper(),
        "benchmark": bench,
        "trade_date": trade_date,
        "windows": {},
    }
    if stock_df is None or stock_df.empty or bench_df is None or bench_df.empty:
        out["error"] = "insufficient_data"
        return out

    def _window_return(df, window: int) -> float | None:
        tail = df.tail(window + 1)
        if len(tail) < window + 1:
            return None
        start = float(tail.iloc[0]["Close"])
        end = float(tail.iloc[-1]["Close"])
        if start <= 0:
            return None
        return float(end / start - 1.0)

    for label, window in (("5d", 5), ("20d", 20), ("60d", 60)):
        rs = _window_return(stock_df, window)
        rb = _window_return(bench_df, window)
        if rs is None or rb is None:
            out["windows"][label] = {"stock": rs, "benchmark": rb, "alpha": None}
        else:
            out["windows"][label] = {
                "stock": rs,
                "benchmark": rb,
                "alpha": float(rs - rb),
            }
    return out


def get_drawdown_risk_stats(symbol: str, trade_date: str, lookback: int = 126) -> dict[str, Any]:
    """Pullback / drawdown stats — ammunition for bear; recovery for bull."""
    market = build_market_snapshot(symbol, trade_date, lookback=lookback)
    closes = np.asarray([b.close for b in market.bars], dtype=float)
    if len(closes) < 5:
        return {"symbol": symbol.upper(), "error": "insufficient_bars"}

    running_max = np.maximum.accumulate(closes)
    drawdowns = closes / running_max - 1.0
    current_dd = float(drawdowns[-1])
    max_dd = float(drawdowns.min())
    high_idx = int(np.argmax(closes))
    days_since_high = len(closes) - 1 - high_idx

    low_20 = float(np.min(closes[-20:])) if len(closes) >= 20 else float(np.min(closes))
    low_60 = float(np.min(closes[-60:])) if len(closes) >= 60 else float(np.min(closes))
    last = float(closes[-1])

    return {
        "symbol": symbol.upper(),
        "trade_date": trade_date,
        "last_close": last,
        "current_drawdown_from_high": current_dd,
        "max_drawdown_in_window": max_dd,
        "days_since_window_high": days_since_high,
        "bounce_from_20d_low_pct": float(last / low_20 - 1.0) if low_20 > 0 else None,
        "bounce_from_60d_low_pct": float(last / low_60 - 1.0) if low_60 > 0 else None,
        "lookback_bars": len(closes),
    }


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _benchmark_impl(symbol: str, trade_date: str, benchmark: str = "") -> str:
    bench = benchmark.strip() or None
    return _json(compare_to_benchmark(symbol, trade_date, benchmark=bench))


def _drawdown_impl(symbol: str, trade_date: str) -> str:
    return _json(get_drawdown_risk_stats(symbol, trade_date))


if langchain_tool is not None:

    @langchain_tool
    def compare_to_benchmark_for_ticker(
        symbol: Annotated[str, "ticker symbol"],
        trade_date: Annotated[str, "analysis date YYYY-MM-DD"],
        benchmark: Annotated[str, "benchmark ticker, default SPY or regional index"] = "",
    ) -> str:
        """Stock vs benchmark returns and alpha over 5d/20d/60d windows."""
        return _benchmark_impl(symbol, trade_date, benchmark)

    @langchain_tool
    def get_drawdown_risk_stats_for_ticker(
        symbol: Annotated[str, "ticker symbol"],
        trade_date: Annotated[str, "analysis date YYYY-MM-DD"],
    ) -> str:
        """Drawdown from recent high, max drawdown, bounce from 20d/60d lows."""
        return _drawdown_impl(symbol, trade_date)

else:  # pragma: no cover

    def compare_to_benchmark_for_ticker(symbol: str, trade_date: str, benchmark: str = "") -> str:
        return _benchmark_impl(symbol, trade_date, benchmark)

    def get_drawdown_risk_stats_for_ticker(symbol: str, trade_date: str) -> str:
        return _drawdown_impl(symbol, trade_date)

DEBATE_REACT_TOOLS = [
    compare_to_benchmark_for_ticker,
    get_drawdown_risk_stats_for_ticker,
]
