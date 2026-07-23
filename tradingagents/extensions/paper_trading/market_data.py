"""Time-safe historical market data for paper trading and backtests."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

from tradingagents.extensions.contracts import ExecutionQuote, MarketBar, MarketSnapshot


class MarketDataUnavailable(LookupError):
    """Raised when a requested historical observation does not exist."""


class MarketDataRateLimited(MarketDataUnavailable):
    """Raised after a historical data vendor remains rate limited after retries."""


def is_rate_limit_error(exc: BaseException) -> bool:
    """Recognize common HTTP/vendor rate-limit exception shapes."""

    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return (
        "ratelimit" in name
        or "rate limit" in message
        or "rate limited" in message
        or "too many requests" in message
        or "http 429" in message
    )


def as_utc(value: datetime) -> datetime:
    """Normalize daily-market timestamps so comparisons stay deterministic."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def normalize_symbol(symbol: str) -> str:
    value = symbol.strip().upper()
    if not value:
        raise ValueError("symbol must not be empty")
    return value


class HistoricalMarketDataProvider:
    """Immutable in-memory bar store implementing ``MarketDataProvider``.

    All timestamps are normalized to UTC at the boundary.  ``get_snapshot``
    uses a right-bounded lookup, while ``get_execution_quote`` uses a strict
    right lookup, so a decision can never execute on its own bar.
    """

    def __init__(
        self,
        bars: Mapping[str, Sequence[MarketBar]],
        *,
        source: str = "memory",
    ) -> None:
        if not bars:
            raise ValueError("at least one symbol is required")
        self.source = source
        normalized: dict[str, tuple[MarketBar, ...]] = {}
        timestamps: dict[str, tuple[datetime, ...]] = {}

        for raw_symbol, raw_bars in bars.items():
            symbol = normalize_symbol(raw_symbol)
            converted = tuple(
                sorted(
                    (
                        bar.model_copy(update={"timestamp": as_utc(bar.timestamp)})
                        for bar in raw_bars
                    ),
                    key=lambda bar: bar.timestamp,
                )
            )
            if not converted:
                raise ValueError(f"no bars supplied for {symbol}")
            times = tuple(bar.timestamp for bar in converted)
            if len(times) != len(set(times)):
                raise ValueError(f"duplicate bar timestamps for {symbol}")
            normalized[symbol] = converted
            timestamps[symbol] = times

        self._bars = normalized
        self._timestamps = timestamps

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._bars))

    def get_snapshot(self, symbol: str, as_of: datetime, lookback: int) -> MarketSnapshot:
        if lookback < 1:
            raise ValueError("lookback must be positive")
        normalized = normalize_symbol(symbol)
        timestamp = as_utc(as_of)
        bars = self._require_symbol(normalized)
        end = bisect_right(self._timestamps[normalized], timestamp)
        if end == 0:
            raise MarketDataUnavailable(f"no {normalized} market data at or before {timestamp}")
        selected = bars[max(0, end - lookback) : end]
        return MarketSnapshot(
            symbol=normalized,
            as_of=timestamp,
            bars=list(selected),
            metadata={"source": self.source, "lookback": lookback},
        )

    def get_execution_quote(self, symbol: str, after: datetime) -> ExecutionQuote:
        normalized = normalize_symbol(symbol)
        timestamp = as_utc(after)
        bars = self._require_symbol(normalized)
        index = bisect_right(self._timestamps[normalized], timestamp)
        if index >= len(bars):
            raise MarketDataUnavailable(f"no {normalized} execution bar after {timestamp}")
        bar = bars[index]
        return ExecutionQuote(
            symbol=normalized,
            timestamp=bar.timestamp,
            price=bar.open,
            metadata={"source": self.source, "price_field": "open"},
        )

    def get_quote_at(self, symbol: str, timestamp: datetime) -> ExecutionQuote:
        """Return an opening quote for an exact synchronized replay bar."""

        normalized = normalize_symbol(symbol)
        at = as_utc(timestamp)
        bars = self._require_symbol(normalized)
        index = bisect_left(self._timestamps[normalized], at)
        if index >= len(bars) or bars[index].timestamp != at:
            raise MarketDataUnavailable(f"no {normalized} execution bar at {at}")
        return ExecutionQuote(
            symbol=normalized,
            timestamp=at,
            price=bars[index].open,
            metadata={"source": self.source, "price_field": "open"},
        )

    def common_calendar(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> list[datetime]:
        """Return timestamps shared by all requested symbols, inclusively."""

        normalized = [normalize_symbol(symbol) for symbol in symbols]
        if not normalized:
            return []
        start_at, end_at = as_utc(start), as_utc(end)
        calendars = [set(self._timestamps[symbol]) for symbol in normalized]
        common = set.intersection(*calendars)
        return sorted(timestamp for timestamp in common if start_at <= timestamp <= end_at)

    def close_prices(self, symbols: Sequence[str], as_of: datetime) -> dict[str, float]:
        """Return the latest close at or before ``as_of`` for each symbol."""

        timestamp = as_utc(as_of)
        prices: dict[str, float] = {}
        for raw_symbol in symbols:
            symbol = normalize_symbol(raw_symbol)
            bars = self._require_symbol(symbol)
            index = bisect_right(self._timestamps[symbol], timestamp) - 1
            if index < 0:
                raise MarketDataUnavailable(f"no {symbol} close at or before {timestamp}")
            prices[symbol] = bars[index].close
        return prices

    def bars_between(
        self,
        symbol: str,
        start_exclusive: datetime,
        end_inclusive: datetime,
    ) -> tuple[MarketBar, ...]:
        normalized = normalize_symbol(symbol)
        bars = self._require_symbol(normalized)
        times = self._timestamps[normalized]
        left = bisect_right(times, as_utc(start_exclusive))
        right = bisect_right(times, as_utc(end_inclusive))
        return bars[left:right]

    def export_bars(
        self,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, list[dict[str, Any]]]:
        """Export a deduplicated run data set for persistence and replay."""

        start_at, end_at = as_utc(start), as_utc(end)
        result: dict[str, list[dict[str, Any]]] = {}
        for raw_symbol in symbols:
            symbol = normalize_symbol(raw_symbol)
            result[symbol] = [
                bar.model_dump(mode="json")
                for bar in self._require_symbol(symbol)
                if start_at <= bar.timestamp <= end_at
            ]
        return result

    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, Any],
        *,
        source: str = "dataframe",
    ) -> HistoricalMarketDataProvider:
        """Build a provider from pandas-like OHLCV DataFrames."""

        parsed: dict[str, list[MarketBar]] = {}
        for raw_symbol, frame in frames.items():
            symbol = normalize_symbol(raw_symbol)
            columns = {str(column).lower(): column for column in frame.columns}
            missing = {"open", "high", "low", "close"} - columns.keys()
            if missing:
                raise ValueError(f"{symbol} frame is missing columns: {sorted(missing)}")
            symbol_bars: list[MarketBar] = []
            for index, row in frame.iterrows():
                values = [row[columns[name]] for name in ("open", "high", "low", "close")]
                if any(value is None or value != value for value in values):
                    continue
                timestamp = index.to_pydatetime() if hasattr(index, "to_pydatetime") else index
                if not isinstance(timestamp, datetime):
                    timestamp = datetime.fromisoformat(str(timestamp))
                volume_column = columns.get("volume")
                volume = row[volume_column] if volume_column is not None else 0
                if volume is None or volume != volume:
                    volume = 0
                symbol_bars.append(
                    MarketBar(
                        timestamp=as_utc(timestamp),
                        open=float(values[0]),
                        high=float(values[1]),
                        low=float(values[2]),
                        close=float(values[3]),
                        volume=max(0, float(volume)),
                    )
                )
            parsed[symbol] = symbol_bars
        return cls(parsed, source=source)

    @classmethod
    def from_yfinance(
        cls,
        symbols: Sequence[str],
        start: datetime,
        end: datetime,
        *,
        max_attempts: int = 3,
        retry_delay: float = 1.0,
    ) -> HistoricalMarketDataProvider:
        """Download adjusted daily OHLCV bars from yfinance.

        Adjusted bars avoid artificial jumps from splits in the one-week MVP.
        Corporate-action-aware share accounting remains outside the current
        scope and is documented in the implementation plan. Transient Yahoo
        rate limits are retried with a short linear backoff.
        """

        import yfinance as yf

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_delay < 0:
            raise ValueError("retry_delay must not be negative")

        frames: dict[str, Any] = {}
        start_at, end_at = as_utc(start), as_utc(end)
        for raw_symbol in symbols:
            symbol = normalize_symbol(raw_symbol)
            for attempt in range(1, max_attempts + 1):
                try:
                    frame = yf.Ticker(symbol).history(
                        start=start_at.date().isoformat(),
                        end=(end_at + timedelta(days=1)).date().isoformat(),
                        auto_adjust=True,
                        actions=False,
                    )
                    break
                except Exception as exc:
                    if not is_rate_limit_error(exc):
                        raise
                    if attempt == max_attempts:
                        raise MarketDataRateLimited(
                            f"yfinance remained rate limited for {symbol} "
                            f"after {max_attempts} attempts"
                        ) from exc
                    sleep(retry_delay * attempt)
            if frame.empty:
                raise MarketDataUnavailable(f"yfinance returned no bars for {symbol}")
            frames[symbol] = frame
        return cls.from_frames(frames, source="yfinance-adjusted")

    def _require_symbol(self, symbol: str) -> tuple[MarketBar, ...]:
        try:
            return self._bars[symbol]
        except KeyError as exc:
            raise MarketDataUnavailable(f"unknown market symbol {symbol}") from exc


__all__ = [
    "HistoricalMarketDataProvider",
    "MarketDataRateLimited",
    "MarketDataUnavailable",
    "as_utc",
    "is_rate_limit_error",
]
