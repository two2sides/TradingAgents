"""Lightweight Yahoo Chart API client (no yfinance crumb / cookie flow).

``yfinance`` frequently hits HTTP 429 via its crumb session. The public chart
endpoint often still works with a normal browser User-Agent, so this module
fetches OHLCV JSON directly and converts it to the same CSV / DataFrame shapes
the rest of the data layer expects.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import requests

from .errors import NoMarketDataError, VendorRateLimitError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
}


def fetch_yahoo_chart_ohlcv(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV via Yahoo's chart JSON endpoint."""
    canonical = normalize_symbol(symbol)
    params: dict[str, str | int] = {"interval": "1d", "events": "div,splits"}

    if start_date or end_date:
        # period1/period2 are unix seconds; end is exclusive-ish — add one day buffer
        start = pd.Timestamp(start_date or "1980-01-01", tz="UTC")
        end = pd.Timestamp(end_date or pd.Timestamp.today().strftime("%Y-%m-%d"), tz="UTC")
        end = end + pd.Timedelta(days=1)
        params["period1"] = int(start.timestamp())
        params["period2"] = int(end.timestamp())
    else:
        params["range"] = "5y"

    try:
        resp = requests.get(
            _CHART_URL.format(symbol=canonical),
            params=params,
            headers=_HEADERS,
            timeout=20,
        )
    except requests.RequestException as exc:
        raise NoMarketDataError(symbol, canonical, f"yahoo chart request failed: {exc}") from exc

    if resp.status_code == 429:
        raise VendorRateLimitError("Yahoo chart endpoint rate limited")
    if resp.status_code != 200:
        raise NoMarketDataError(
            symbol, canonical, f"yahoo chart HTTP {resp.status_code}"
        )

    try:
        payload = resp.json()
    except ValueError as exc:
        raise NoMarketDataError(symbol, canonical, "yahoo chart returned non-JSON") from exc

    result = (payload.get("chart") or {}).get("result")
    if not result:
        err = (payload.get("chart") or {}).get("error")
        raise NoMarketDataError(symbol, canonical, f"yahoo chart empty result: {err}")

    block = result[0]
    timestamps = block.get("timestamp") or []
    quote = ((block.get("indicators") or {}).get("quote") or [{}])[0]
    if not timestamps or not quote:
        raise NoMarketDataError(symbol, canonical, "yahoo chart missing quote arrays")

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "Open": quote.get("open"),
            "High": quote.get("high"),
            "Low": quote.get("low"),
            "Close": quote.get("close"),
            "Volume": quote.get("volume"),
        }
    )
    df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    for col in ("Open", "High", "Low", "Close"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    if df.empty:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"no yahoo chart rows between {start_date or '...'} and {end_date or '...'}",
        )

    logger.info("Loaded %s via Yahoo chart API (%s rows)", canonical, len(df))
    return df.reset_index(drop=True)


def get_yahoo_chart_stock(symbol: str, start_date: str, end_date: str) -> str:
    """Vendor entrypoint matching ``get_stock_data`` CSV text contract."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    canonical = normalize_symbol(symbol)
    df = fetch_yahoo_chart_ohlcv(symbol, start_date, end_date)
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    header = (
        f"# Stock data for {label} from {start_date} to {end_date}\n"
        f"# Vendor: yahoo_chart\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + df.to_csv(index=False)
