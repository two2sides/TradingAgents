"""Network probes for Sentiment enrichment sources (StockTwits / Reddit).

These are **manual / optional** checks — not part of the default CI gate.
Run explicitly before deciding to re-enable a source in Sentiment Analyst::

    python -m pytest tests/TEAM_C/test_sentiment_source_probes.py -v -s

Neither source requires an API key for the public endpoints we use.
Failures are usually User-Agent / WAF / per-IP rate limits, not missing keys.
"""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

PROJECT_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ST_URL = "https://api.stocktwits.com/api/2/streams/symbol/AAPL.json"
REDDIT_RSS = (
    "https://www.reddit.com/r/stocks/search.rss?"
    "q=AAPL&restrict_sr=on&sort=new&t=week&limit=5"
)
REDDIT_JSON = (
    "https://www.reddit.com/r/stocks/search.json?"
    "q=AAPL&restrict_sr=on&sort=new&t=week&limit=5"
)


def _probe(url: str, *, ua: str, accept: str = "*/*") -> tuple[str, int | None, str]:
    """Return ``(status_label, http_code_or_None, detail)``."""
    req = Request(url, headers={"User-Agent": ua, "Accept": accept})
    try:
        with urlopen(req, timeout=15) as resp:
            body = resp.read(120)
            return "OK", resp.status, f"bytes={len(body)}"
    except HTTPError as exc:
        return "HTTP_ERROR", exc.code, exc.reason or ""
    except URLError as exc:
        return "NETWORK", None, str(exc.reason if hasattr(exc, "reason") else exc)
    except TimeoutError:
        return "TIMEOUT", None, "timed out"
    except OSError as exc:
        return "OS_ERROR", None, str(exc)


@pytest.mark.network
def test_probe_stocktwits_project_ua_vs_browser_ua():
    """StockTwits: no API key; often 403 on bot UA, may work with browser UA."""
    proj = _probe(ST_URL, ua=PROJECT_UA, accept="application/json")
    browser = _probe(ST_URL, ua=BROWSER_UA, accept="application/json")
    print("\n[StockTwits]")
    print(f"  project UA -> {proj}")
    print(f"  browser UA -> {browser}")
    print("  need_api_key: NO (public symbol stream)")
    print(
        "  typical_cause: WAF/bot filter on User-Agent (403), "
        "not a missing API key. China GFW may also block, but 403 usually means "
        "reached server then rejected."
    )
    # Soft assert: at least one path documented; do not fail CI if both blocked.
    assert proj[0] in {"OK", "HTTP_ERROR", "NETWORK", "TIMEOUT", "OS_ERROR"}
    assert browser[0] in {"OK", "HTTP_ERROR", "NETWORK", "TIMEOUT", "OS_ERROR"}


@pytest.mark.network
def test_probe_reddit_rss_and_json():
    """Reddit: no API key for RSS; JSON search often WAF 403; RSS may 429."""
    rss = _probe(REDDIT_RSS, ua=PROJECT_UA)
    js = _probe(REDDIT_JSON, ua=PROJECT_UA, accept="application/json")
    print("\n[Reddit]")
    print(f"  RSS  (project UA) -> {rss}")
    print(f"  JSON (project UA) -> {js}")
    print("  need_api_key: NO for public RSS/JSON search we use")
    print(
        "  typical_cause: JSON often 403 (WAF); RSS works intermittently but "
        "429 Too Many Requests when the same IP hits too often (multi-subreddit "
        "loops make this worse). Official Reddit API would need OAuth app keys "
        "— that is a different, heavier path."
    )
    assert rss[0] in {"OK", "HTTP_ERROR", "NETWORK", "TIMEOUT", "OS_ERROR"}
    assert js[0] in {"OK", "HTTP_ERROR", "NETWORK", "TIMEOUT", "OS_ERROR"}


@pytest.mark.network
def test_probe_project_fetchers_once():
    """Call the real fetch helpers once and print a short preview."""
    from tradingagents.dataflows.reddit import fetch_reddit_posts
    from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages

    st = fetch_stocktwits_messages("AAPL", limit=3)
    rd = fetch_reddit_posts("AAPL", limit_per_sub=3)
    print("\n[Project fetchers]")
    print(f"  stocktwits preview: {st[:160]!r}")
    print(f"  reddit preview:     {rd[:160]!r}")
    assert isinstance(st, str) and len(st) > 0
    assert isinstance(rd, str) and len(rd) > 0
