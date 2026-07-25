"""yfinance treats ``end`` as exclusive; we must request one extra day so the
requested end_date (and the current day) is actually included.

Regressions for #986 (current-day OHLCV excluded) and #987 (requested end_date
row omitted).
"""
import pandas as pd
import pytest

import tradingagents.dataflows.stockstats_utils as su
import tradingagents.dataflows.y_finance as yfin
from tradingagents.dataflows.config import set_config


@pytest.mark.unit
def test_get_yfin_requests_inclusive_end(monkeypatch):
    captured = {}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, start, end):
            captured["start"] = start
            captured["end"] = end
            idx = pd.to_datetime(["2025-05-08", "2025-05-09"])
            return pd.DataFrame(
                {"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0],
                 "Close": [1.0, 2.0], "Volume": [1, 2]},
                index=idx,
            )

    monkeypatch.setattr(yfin.yf, "Ticker", FakeTicker)
    out = yfin.get_YFin_data_online("AAPL", "2025-05-01", "2025-05-09")

    # end is requested one day past end_date so 2025-05-09 is included (#987).
    assert captured["end"] == "2025-05-10"
    # Header still reflects the requested range, not the internal +1 day.
    assert "to 2025-05-09" in out


@pytest.mark.unit
def test_load_ohlcv_includes_intraday_timestamp_on_requested_day(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    captured = {}

    def fake_download(symbol, start, end):
        captured["end"] = end
        requested_day = pd.Timestamp.today().normalize()
        return pd.DataFrame(
            {
                "Date": [requested_day + pd.Timedelta(hours=13, minutes=30)],
                "Open": [100.0],
                "High": [100.0],
                "Low": [100.0],
                "Close": [100.0],
                "Volume": [1],
            },
        )

    monkeypatch.setattr(
        "tradingagents.dataflows.yahoo_chart.fetch_yahoo_chart_ohlcv",
        fake_download,
    )
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    result = su.load_ohlcv("AAPL", today)

    assert captured["end"] == today
    assert len(result) == 1
    assert result.iloc[0]["Date"].date() == pd.Timestamp(today).date()
