"""Build as_of-safe MarketSnapshot for analyst OHLCV tools."""

from __future__ import annotations

from datetime import datetime, timezone

from tradingagents.extensions.contracts import MarketBar, MarketSnapshot


def _parse_trade_date(trade_date: str) -> datetime:
    """Interpret YYYY-MM-DD as UTC end-of-day for as_of semantics."""
    day = datetime.strptime(trade_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return day.replace(hour=23, minute=59, second=59)


def pd_timestamp_to_aware(value) -> datetime:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return ts.to_pydatetime().astimezone(timezone.utc)


def build_market_snapshot(symbol: str, trade_date: str, lookback: int = 90) -> MarketSnapshot:
    """Load as_of-safe bars via the project OHLCV loader (yahoo_chart primary)."""
    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    as_of = _parse_trade_date(trade_date)
    df = load_ohlcv(symbol, trade_date)
    if df is None or df.empty:
        return MarketSnapshot(symbol=symbol, as_of=as_of, bars=[])

    tail = df.tail(lookback)
    bars: list[MarketBar] = []
    for _, row in tail.iterrows():
        ts = pd_timestamp_to_aware(row["Date"])
        if ts > as_of:
            continue
        bars.append(
            MarketBar(
                timestamp=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"])
                if "Volume" in row and row["Volume"] == row["Volume"]
                else 0.0,
            )
        )
    return MarketSnapshot(symbol=symbol, as_of=as_of, bars=bars)
