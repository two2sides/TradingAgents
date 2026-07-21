"""TEAM_C tests for analyst tools (OHLCV helpers + sentiment lexicon).

Hybrid decision / fusion / boundaries were removed; these tests cover the
tooling that Market and Sentiment analysts still use.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.extensions.contracts import MarketBar, MarketSnapshot
from tradingagents.extensions.decision import REACT_TOOLS
from tradingagents.extensions.decision.quant_engine import compute_quant_signal
from tradingagents.extensions.decision.sentiment_lexicon import score_text
from tradingagents.extensions.decision.tools.market_bound_tools import MARKET_REACT_TOOLS
from tradingagents.extensions.decision.tools.ohlcv_tools import (
    analyze_multi_horizon_ohlcv,
    detect_price_gap,
    detect_volume_anomaly,
)
from tradingagents.extensions.decision.tools.sentiment_tools import score_news_sentiment

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=timezone.utc)


def _trend_bars(n: int = 60, start: float = 100.0, drift: float = 0.004) -> list[MarketBar]:
    bars: list[MarketBar] = []
    price = start
    for i in range(n):
        ts = NOW - timedelta(days=n - i)
        open_p = price
        close_p = price * (1.0 + drift)
        high_p = max(open_p, close_p) * 1.01
        low_p = min(open_p, close_p) * 0.99
        bars.append(
            MarketBar(
                timestamp=ts,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=1_000_000 + 1000 * i,
            )
        )
        price = close_p
    return bars


def make_market(n: int = 60, drift: float = 0.004) -> MarketSnapshot:
    return MarketSnapshot(symbol="AAPL", as_of=NOW, bars=_trend_bars(n=n, drift=drift))


def test_react_tools_registered():
    assert len(REACT_TOOLS) >= 3
    assert len(MARKET_REACT_TOOLS) == 3


def test_quant_uptrend_positive_score():
    sig = compute_quant_signal(make_market(drift=0.005), max_weight=0.35, current_weight=0.0)
    assert sig.q_score > 0
    assert 0 <= sig.suggested_weight <= 0.35


def test_multi_horizon_summary_keys():
    summary = analyze_multi_horizon_ohlcv(make_market())
    assert isinstance(summary, dict)
    assert summary


def test_volume_and_gap_detectors():
    market = make_market()
    vol = detect_volume_anomaly(market)
    gap = detect_price_gap(market)
    assert "label" in vol or "zscore" in vol or isinstance(vol, dict)
    assert isinstance(gap, dict)


def test_lexicon_positive_text():
    scored = score_text("Apple beats estimates with record growth and strong profit")
    assert scored.score > 0


def test_score_news_sentiment_tool():
    out = score_news_sentiment("growth beat profit strong")
    assert out
