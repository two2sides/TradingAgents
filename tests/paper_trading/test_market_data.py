"""Tests for time-safe historical market access."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradingagents.dataflows.errors import VendorRateLimitError
from tradingagents.extensions.contracts import MarketBar
from tradingagents.extensions.paper_trading.demo import generate_demo_market_data
from tradingagents.extensions.paper_trading.market_data import (
    HistoricalMarketDataProvider,
    MarketDataRateLimited,
    MarketDataUnavailable,
)

START = datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)


def make_bars(prices: list[float]) -> list[MarketBar]:
    return [
        MarketBar(
            timestamp=START + timedelta(days=index),
            open=price,
            high=price + 1,
            low=price - 1,
            close=price + 0.5,
            volume=1_000 + index,
        )
        for index, price in enumerate(prices)
    ]


def test_snapshot_is_right_bounded_and_execution_quote_is_strictly_later():
    provider = HistoricalMarketDataProvider({"aapl": make_bars([100, 101, 102, 103])})
    decision_at = START + timedelta(days=2)

    snapshot = provider.get_snapshot("AAPL", decision_at, lookback=2)
    quote = provider.get_execution_quote("aapl", decision_at)

    assert [bar.open for bar in snapshot.bars] == [101, 102]
    assert all(bar.timestamp <= decision_at for bar in snapshot.bars)
    assert quote.timestamp == START + timedelta(days=3)
    assert quote.price == 103


def test_provider_rejects_missing_future_execution_data():
    provider = HistoricalMarketDataProvider({"AAPL": make_bars([100, 101])})

    with pytest.raises(MarketDataUnavailable, match="execution bar"):
        provider.get_execution_quote("AAPL", START + timedelta(days=1))


def test_common_calendar_uses_only_shared_symbol_timestamps():
    aapl = make_bars([100, 101, 102, 103])
    msft = make_bars([200, 201, 202, 203])
    msft.pop(1)
    provider = HistoricalMarketDataProvider({"AAPL": aapl, "MSFT": msft})

    calendar = provider.common_calendar(
        ["AAPL", "MSFT"],
        START,
        START + timedelta(days=3),
    )

    assert calendar == [START, START + timedelta(days=2), START + timedelta(days=3)]
    assert provider.get_quote_at("MSFT", calendar[1]).price == 202


def test_dataframe_adapter_normalizes_columns_and_naive_timestamps():
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [100, 200],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    provider = HistoricalMarketDataProvider.from_frames({"aapl": frame})
    snapshot = provider.get_snapshot("AAPL", datetime(2024, 1, 3), 10)

    assert len(snapshot.bars) == 2
    assert snapshot.bars[-1].timestamp.tzinfo == timezone.utc
    assert snapshot.metadata["source"] == "dataframe"


def test_built_in_demo_data_is_deterministic_and_needs_no_network():
    end = START + timedelta(days=15)

    first = generate_demo_market_data(["AAPL", "MSFT"], START, end)
    second = generate_demo_market_data(["AAPL", "MSFT"], START, end)

    assert first.export_bars(["AAPL", "MSFT"], START, end) == second.export_bars(
        ["AAPL", "MSFT"], START, end
    )
    assert first.source == "built-in-demo"
    assert len(first.common_calendar(["AAPL", "MSFT"], START, end)) >= 10


def test_yfinance_retries_transient_rate_limit(monkeypatch):
    calls = []
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000, 1_100],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True),
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            calls.append((self.symbol, kwargs))
            if len(calls) < 3:
                raise RuntimeError("Too Many Requests. Rate limited.")
            return frame

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    provider = HistoricalMarketDataProvider.from_yfinance(
        ["AAPL"],
        START,
        START + timedelta(days=2),
        max_attempts=3,
        retry_delay=0,
    )

    assert len(calls) == 3
    assert provider.source == "yfinance-adjusted"


def test_yfinance_reports_persistent_rate_limit(monkeypatch):
    class FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            raise RuntimeError("HTTP 429 Too Many Requests")

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    with pytest.raises(MarketDataRateLimited, match="after 2 attempts"):
        HistoricalMarketDataProvider.from_yfinance(
            ["AAPL"],
            START,
            START + timedelta(days=2),
            max_attempts=2,
            retry_delay=0,
        )


def test_yahoo_chart_uses_shared_cache_path_and_retries_rate_limit(monkeypatch):
    calls = []
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000, 1_100],
        }
    )

    def fake_load_ohlcv(symbol, end_date):
        calls.append((symbol, end_date))
        if len(calls) == 1:
            raise VendorRateLimitError("temporary chart throttle")
        return frame

    monkeypatch.setattr(
        "tradingagents.dataflows.stockstats_utils.load_ohlcv",
        fake_load_ohlcv,
    )

    window_start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    provider = HistoricalMarketDataProvider.from_yahoo_chart(
        ["AAPL"],
        window_start,
        window_start + timedelta(days=2),
        retry_delay=0,
    )

    assert len(calls) == 2
    assert provider.source == "yahoo-chart-cached"
    assert len(provider.get_snapshot("AAPL", window_start + timedelta(days=2), 10).bars) == 2
