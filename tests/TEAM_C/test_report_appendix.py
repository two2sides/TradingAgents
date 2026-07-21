"""Tests for deterministic report appendix."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.report_appendix import (
    build_appendix_stats,
    render_appendix_markdown,
    write_report_appendix,
)


@patch("tradingagents.report_appendix.build_appendix_stats")
@patch("tradingagents.report_appendix.render_price_volume_chart", return_value=None)
def test_write_report_appendix_creates_md(mock_chart, mock_stats, tmp_path):
    mock_stats.return_value = {
        "symbol": "AAPL",
        "trade_date": "2024-06-01",
        "scorecard": {"trend_label": "bullish", "features": {}},
        "benchmark": {"windows": {}},
        "drawdown": {},
        "news_sentiment": {"headline_count": 0},
    }
    md, _ = write_report_appendix(
        {"trade_date": "2024-06-01"},
        "AAPL",
        tmp_path,
    )
    assert md is not None
    assert "附录" in md
    assert (tmp_path / "appendix" / "statistics.md").exists()


def test_render_appendix_includes_table():
    md = render_appendix_markdown(
        {
            "symbol": "AAPL",
            "trade_date": "2024-06-01",
            "scorecard": {
                "trend_label": "mixed",
                "q_score": 0.1,
                "features": {"ret_20": 0.05},
                "multi_horizon": {
                    "horizons": {
                        "short": {"return": 0.01, "sma_ratio": 0.02, "realized_vol": 0.01},
                        "medium": {"return": 0.05, "sma_ratio": 0.03, "realized_vol": 0.02},
                        "long": {"return": 0.1, "sma_ratio": 0.04, "realized_vol": 0.015},
                    }
                },
            },
            "benchmark": {
                "windows": {
                    "5d": {"stock": 0.01, "benchmark": 0.005, "alpha": 0.005},
                    "20d": {"stock": 0.05, "benchmark": 0.03, "alpha": 0.02},
                    "60d": {"stock": 0.1, "benchmark": 0.08, "alpha": 0.02},
                }
            },
            "drawdown": {
                "current_drawdown_from_high": -0.05,
                "max_drawdown_in_window": -0.12,
                "days_since_window_high": 10,
                "bounce_from_20d_low_pct": 0.08,
            },
            "news_sentiment": {
                "headline_count": 5,
                "positive": 2,
                "negative": 1,
                "neutral": 2,
                "net_sentiment": "positive",
            },
        },
        chart_rel_path="appendix/price_volume.png",
    )
    assert "VI. 附录" in md
    assert "price_volume.png" in md
    assert "Alpha" in md
