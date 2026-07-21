"""Tests for non-yfinance default vendors (chart OHLCV + Google News)."""

from tradingagents.dataflows.interface import VENDOR_METHODS
from tradingagents.dataflows.errors import VendorRateLimitError
from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.default_config import DEFAULT_CONFIG
from yfinance.exceptions import YFRateLimitError


def test_stooq_and_yahoo_chart_registered():
    assert "stooq" in VENDOR_METHODS["get_stock_data"]
    assert "yahoo_chart" in VENDOR_METHODS["get_stock_data"]


def test_default_vendors_avoid_yfinance_package():
    vendors = DEFAULT_CONFIG["data_vendors"]
    assert vendors["core_stock_apis"] == "yahoo_chart"
    assert vendors["technical_indicators"] == "stockstats"
    assert vendors["news_data"] == "google_news"
    assert vendors["fundamental_data"] == "sec_edgar"
    assert "yfinance" not in vendors["core_stock_apis"]
    assert "yfinance" not in vendors["news_data"]
    assert "yfinance" not in vendors["fundamental_data"]
    assert "google_news" in VENDOR_METHODS["get_news"]
    assert "stockstats" in VENDOR_METHODS["get_indicators"]
    assert "sec_edgar" in VENDOR_METHODS["get_fundamentals"]
    assert DEFAULT_CONFIG.get("tool_vendors") == {}


def test_yf_retry_raises_vendor_rate_limit(monkeypatch):
    calls = {"n": 0}

    def always_limited():
        calls["n"] += 1
        raise YFRateLimitError()

    import tradingagents.dataflows.stockstats_utils as ssu

    monkeypatch.setattr(ssu.time, "sleep", lambda _s: None)

    try:
        yf_retry(always_limited, max_retries=1, base_delay=0.01)
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised = exc

    assert isinstance(raised, VendorRateLimitError)
    assert calls["n"] == 2
