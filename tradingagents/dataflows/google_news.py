"""Google News RSS vendor — ticker and global headlines without Yahoo/yfinance.

Uses the public Google News RSS search endpoint (no API key). Intended as the
default ``news_data`` source when Yahoo Finance is unreachable.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode
from urllib.request import Request, urlopen

from .config import get_config
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search"
_UA = (
    "Mozilla/5.0 (compatible; TradingAgents/0.3; research; +https://github.com/)"
)
_DEFAULT_COMPANY_ALIASES = {
    "AAPL": ["Apple"],
    "AMZN": ["Amazon"],
    "GOOGL": ["Google", "Alphabet"],
    "META": ["Meta Platforms"],
    "MSFT": ["Microsoft"],
    "NVDA": ["Nvidia"],
    "TSLA": ["Tesla", "Elon Musk"],
}


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", unescape(cleaned)).strip()


def _fetch_rss(query: str, *, when: str = "7d", max_items: int = 25) -> list[dict]:
    params = urlencode(
        {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
            "when": when,
        }
    )
    url = f"{_RSS_URL}?{params}"
    req = Request(url, headers={"User-Agent": _UA})
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("Google News RSS failed for %r: %s", query, exc)
        return []

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        logger.warning("Google News RSS parse failed for %r: %s", query, exc)
        return []

    items: list[dict] = []
    for item in root.findall("./channel/item"):
        title = _strip_html(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        source_el = item.find("source")
        publisher = (source_el.text or "Google News").strip() if source_el is not None else "Google News"
        description = _strip_html(item.findtext("description") or "")
        pub_date = None
        if pub:
            try:
                pub_date = parsedate_to_datetime(pub)
            except (TypeError, ValueError, IndexError):
                pub_date = None
        if not title:
            continue
        items.append(
            {
                "title": title,
                "summary": description,
                "publisher": publisher,
                "link": link,
                "pub_date": pub_date,
            }
        )
        if len(items) >= max_items:
            break
    return items


def _format_articles(articles: list[dict], *, header: str) -> str:
    if not articles:
        return (
            f"{header}\n\n"
            "NO_DATA_AVAILABLE: Google News RSS returned no articles. "
            "Do not invent headlines."
        )
    lines = [header, ""]
    for i, art in enumerate(articles, 1):
        when = ""
        if art.get("pub_date") is not None:
            when = art["pub_date"].strftime("%Y-%m-%d %H:%M")
        lines.append(f"### {i}. {art['title']}")
        lines.append(f"- Publisher: {art.get('publisher') or 'Unknown'}")
        if when:
            lines.append(f"- Published: {when}")
        if art.get("link"):
            lines.append(f"- URL: {art['link']}")
        if art.get("summary"):
            lines.append(f"- Summary: {art['summary'][:500]}")
        lines.append("")
    return "\n".join(lines).strip()


def get_news_google(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Company / ticker news via Google News RSS between start_date and end_date."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    canonical = normalize_symbol(ticker)
    # Prefer recent window; Google RSS ``when`` is coarse (h/d).
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = max(1, (end - start).days)
    except ValueError:
        days = 7
    when = "1d" if days <= 1 else ("7d" if days <= 7 else "30d")

    cfg = get_config()
    configured_aliases = (cfg.get("news_query_aliases") or {}).get(canonical, [])
    aliases = list(dict.fromkeys([*_DEFAULT_COMPANY_ALIASES.get(canonical, []), *configured_aliases]))
    max_items = int(cfg.get("company_news_limit", 40))
    queries = [
        f'"{canonical}" stock OR shares OR earnings',
        f'"{canonical}" company results OR guidance OR regulation',
    ]
    if aliases:
        alias_expr = " OR ".join(f'"{alias}"' for alias in aliases)
        queries.extend(
            [
                f"({alias_expr}) stock OR earnings OR deliveries",
                f"({alias_expr}) guidance OR lawsuit OR regulation OR product",
            ]
        )

    articles: list[dict] = []
    seen: set[tuple[str, str]] = set()
    per_query = max(10, max_items // max(1, len(queries)))
    for query in queries:
        for article in _fetch_rss(query, when=when, max_items=per_query):
            key = (
                re.sub(r"\W+", " ", article["title"].lower()).strip(),
                (article.get("publisher") or "").lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            articles.append(article)
            if len(articles) >= max_items:
                break
        if len(articles) >= max_items:
            break

    # Keep items inside the requested window when pubDate is available.
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    filtered = []
    for art in articles:
        pd = art.get("pub_date")
        if pd is None:
            filtered.append(art)
            continue
        naive = pd.replace(tzinfo=None) if pd.tzinfo else pd
        if start_dt <= naive < end_dt:
            filtered.append(art)
    if not filtered:
        filtered = articles  # fall back to unfiltered RSS set

    header = (
        f"# News for {canonical} ({start_date} → {end_date})\n"
        f"# Vendor: google_news"
    )
    return _format_articles(filtered, header=header)


def get_global_news_google(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 25,
) -> str:
    """Broad market news via Google News RSS using configured query themes."""
    datetime.strptime(curr_date, "%Y-%m-%d")
    cfg = get_config()
    queries = cfg.get("global_news_queries") or [
        "Federal Reserve interest rates inflation",
        "stock market earnings economic outlook",
    ]
    when = "7d" if look_back_days <= 7 else "30d"
    collected: list[dict] = []
    seen_titles: set[str] = set()
    per_query = max(3, limit // max(1, len(queries)))
    for q in queries:
        for art in _fetch_rss(str(q), when=when, max_items=per_query):
            key = art["title"].lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            collected.append(art)
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break

    start = (
        datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    ).strftime("%Y-%m-%d")
    header = (
        f"# Global market news ({start} → {curr_date})\n"
        f"# Vendor: google_news"
    )
    return _format_articles(collected[:limit], header=header)
