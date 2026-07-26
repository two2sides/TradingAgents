"""Regression tests for Google News vendor optional-arg handling.

LLM tool calls often omit ``look_back_days`` / ``limit`` as ``None``. That used
to crash ``get_global_news_google`` with ``NoneType // int`` and abort the graph
as FAILED_SAFE.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

import tradingagents.dataflows.google_news as gnews
from tradingagents.agents.utils.news_data_tools import get_global_news
from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG


def _sample_article(title: str = "Markets rally on rate cut hopes") -> dict:
    return {
        "title": title,
        "summary": "Summary text",
        "publisher": "Example Wire",
        "link": "https://example.com/a",
        "pub_date": datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
    }


@pytest.fixture
def google_news_config(monkeypatch):
    cfg = {**DEFAULT_CONFIG, "data_vendors": {**DEFAULT_CONFIG["data_vendors"], "news_data": "google_news"}}
    set_config(cfg)
    yield cfg
    set_config(DEFAULT_CONFIG)


@pytest.mark.unit
def test_global_news_accepts_none_limit_and_lookback(monkeypatch, google_news_config):
    calls: list[tuple] = []

    def fake_fetch(query, *, when="7d", max_items=25):
        calls.append((query, when, max_items))
        return [_sample_article(f"Headline for {query[:20]}")]

    monkeypatch.setattr(gnews, "_fetch_rss", fake_fetch)

    out = gnews.get_global_news_google("2024-01-02", look_back_days=None, limit=None)

    assert "Vendor: google_news" in out
    assert "Global market news" in out
    assert "Headline for" in out
    assert calls, "expected RSS fetches"
    # Config defaults: lookback 7 → when=7d; article limit 10
    assert all(when == "7d" for _, when, _ in calls)
    assert all(isinstance(n, int) and n >= 3 for _, _, n in calls)


@pytest.mark.unit
def test_global_news_none_limit_does_not_raise_floordiv(monkeypatch, google_news_config):
    """Exact failure mode from production logs."""
    monkeypatch.setattr(
        gnews,
        "_fetch_rss",
        lambda *a, **k: [_sample_article()],
    )
    # Previously: TypeError: unsupported operand type(s) for //: 'NoneType' and 'int'
    out = gnews.get_global_news_google("2024-01-02", look_back_days=7, limit=None)
    assert "NO_DATA_AVAILABLE" not in out or "Markets rally" in out


@pytest.mark.unit
def test_tool_layer_resolves_none_before_vendor(monkeypatch, google_news_config):
    seen: dict = {}

    def fake_vendor(curr_date, look_back_days, limit):
        seen["args"] = (curr_date, look_back_days, limit)
        return "# ok"

    import tradingagents.dataflows.interface as iface

    monkeypatch.setitem(iface.VENDOR_METHODS["get_global_news"], "google_news", fake_vendor)

    out = get_global_news.func("2024-01-02", None, None)
    assert out == "# ok"
    assert seen["args"][0] == "2024-01-02"
    assert seen["args"][1] == google_news_config["global_news_lookback_days"]
    assert seen["args"][2] == google_news_config["global_news_article_limit"]


@pytest.mark.unit
def test_positive_int_rejects_garbage():
    with pytest.raises(ValueError, match="limit"):
        gnews._positive_int("abc", default=10, name="limit")


@pytest.mark.unit
def test_company_news_tolerates_missing_company_limit(monkeypatch, google_news_config):
    monkeypatch.setattr(
        gnews,
        "_fetch_rss",
        lambda *a, **k: [_sample_article("META earnings beat")],
    )
    cfg = {**google_news_config, "news_article_limit": 5}
    cfg.pop("company_news_limit", None)
    set_config(cfg)

    out = gnews.get_news_google("META", "2024-01-01", "2024-01-08")
    assert "META" in out
    assert "META earnings beat" in out
