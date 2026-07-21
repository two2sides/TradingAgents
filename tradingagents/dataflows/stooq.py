"""Stooq OHLCV vendor — default free daily bars (no API key).

Stooq serves daily equity bars as CSV. Coverage is strongest for US tickers
(``aapl.us``). Used as the primary ``get_stock_data`` / ``load_ohlcv`` source so
CLI runs do not depend on Yahoo Finance rate limits.
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from .errors import NoMarketDataError
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_STOQQ_URL = "https://stooq.com/q/d/l/"


def _stooq_symbol(symbol: str) -> str:
    """Map a Yahoo-style symbol to Stooq's daily CSV ticker."""
    canonical = normalize_symbol(symbol).upper()
    if "." in canonical and canonical.lower().endswith((".us", ".uk", ".de", ".hk", ".jp")):
        return canonical.lower()
    if "." not in canonical and "-" not in canonical:
        return f"{canonical.lower()}.us"
    raise NoMarketDataError(
        symbol, canonical, "symbol not supported by stooq equity daily CSV"
    )


def fetch_stooq_ohlcv(symbol: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    """Download daily OHLCV as a DataFrame (Date/Open/High/Low/Close/Volume)."""
    canonical = normalize_symbol(symbol)
    stooq_sym = _stooq_symbol(symbol)

    try:
        resp = requests.get(
            _STOQQ_URL,
            params={"s": stooq_sym, "i": "d"},
            timeout=20,
            headers={"User-Agent": "TradingAgents/0.3 (research; stooq-primary)"},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise NoMarketDataError(symbol, canonical, f"stooq request failed: {exc}") from exc

    text = resp.text.strip()
    if not text or text.lower().startswith("<!"):
        raise NoMarketDataError(symbol, canonical, "stooq returned empty or HTML body")

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        raise NoMarketDataError(symbol, canonical, f"stooq CSV parse failed: {exc}") from exc

    if df.empty or "Date" not in df.columns:
        raise NoMarketDataError(symbol, canonical, "stooq CSV missing Date column or rows")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)

    df = df.sort_values("Date").reset_index(drop=True)

    if start_date:
        df = df[df["Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.Timestamp(end_date)]

    if df.empty:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"no stooq rows between {start_date or '...'} and {end_date or '...'}",
        )

    logger.info("Loaded %s via Stooq (%s, %s rows)", symbol, stooq_sym, len(df))
    return df


def get_stooq_stock(
    symbol: str,
    start_date: str,
    end_date: str,
) -> str:
    """Return OHLCV CSV text for ``symbol`` between ``start_date`` and ``end_date``."""
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")
    canonical = normalize_symbol(symbol)
    df = fetch_stooq_ohlcv(symbol, start_date, end_date)
    csv_body = df.to_csv(index=False)
    label = canonical if canonical == symbol.upper() else f"{canonical} (from {symbol})"
    stooq_sym = _stooq_symbol(symbol)
    header = (
        f"# Stock data for {label} from {start_date} to {end_date}\n"
        f"# Vendor: stooq ({stooq_sym})\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + csv_body
