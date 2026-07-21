"""Deterministic statistical appendix and price/volume charts for saved reports."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tradingagents.extensions.decision.quant_engine import compute_quant_signal
from tradingagents.extensions.decision.tools.debate_tools import (
    compare_to_benchmark,
    get_drawdown_risk_stats,
)
from tradingagents.extensions.decision.tools.market_snapshot import build_market_snapshot
from tradingagents.extensions.decision.tools.ohlcv_tools import analyze_multi_horizon_ohlcv
from tradingagents.extensions.decision.credibility import invoke_evidenced

logger = logging.getLogger(__name__)


def _pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def build_appendix_stats(symbol: str, trade_date: str) -> dict[str, Any]:
    """Collect reproducible stats for the report appendix."""
    market = build_market_snapshot(symbol, trade_date)
    signal = compute_quant_signal(market)
    scorecard = {
        "q_score": signal.q_score,
        "features": signal.features,
        "multi_horizon": analyze_multi_horizon_ohlcv(market),
    }
    benchmark = compare_to_benchmark(symbol, trade_date)
    drawdown = get_drawdown_risk_stats(symbol, trade_date)
    return {
        "symbol": symbol.upper(),
        "trade_date": trade_date,
        "scorecard": scorecard,
        "benchmark": benchmark,
        "drawdown": drawdown,
    }


def render_appendix_markdown(stats: dict[str, Any], chart_rel_path: str | None) -> str:
    """Render appendix section as markdown."""
    sc = stats.get("scorecard") or {}
    feats = sc.get("features") or {}
    horizons = (sc.get("multi_horizon") or {}).get("horizons") or {}
    bench = stats.get("benchmark") or {}
    dd = stats.get("drawdown") or {}

    lines = [
        "## VI. 附录 — 确定性统计与图表（供读者自行研判）",
        "",
        "本节由代码在报告保存时自动生成，**不经过 LLM**。"
        "数值来自 as-of 安全的 OHLCV / 新闻词典，可与正文交叉核对。",
        "",
        f"- **标的**: `{stats.get('symbol')}` | **分析日**: `{stats.get('trade_date')}`",
        f"- **q_score（统计摘要）**: `{sc.get('q_score', 'n/a')}`",
        "",
        "### 多周期收益与均线",
        "",
        "| 窗口 | 收益率 | 相对均线 | 实现波动率 |",
        "|------|--------|----------|------------|",
    ]

    for label, key in (("5日", "short"), ("20日", "medium"), ("60日", "long")):
        h = horizons.get(key) or {}
        lines.append(
            f"| {label} | {_pct(h.get('return'))} | "
            f"{_pct(h.get('sma_ratio'))} | {_pct(h.get('realized_vol'))} |"
        )

    lines.extend(
        [
            "",
            "### 相对基准（Alpha）",
            "",
            "| 窗口 | 标的 | 基准 | Alpha |",
            "|------|------|------|-------|",
        ]
    )
    for label, key in (("5日", "5d"), ("20日", "20d"), ("60日", "60d")):
        w = (bench.get("windows") or {}).get(key) or {}
        lines.append(
            f"| {label} | {_pct(w.get('stock'))} | {_pct(w.get('benchmark'))} | "
            f"{_pct(w.get('alpha'))} |"
        )

    lines.extend(
        [
            "",
            "### 回撤与反弹",
            "",
            f"- 现价相对窗口高点回撤: **{_pct(dd.get('current_drawdown_from_high'))}**",
            f"- 窗口内最大回撤: **{_pct(dd.get('max_drawdown_in_window'))}**",
            f"- 距窗口高点交易日: **{dd.get('days_since_window_high', 'n/a')}**",
            f"- 较 20 日低点反弹: **{_pct(dd.get('bounce_from_20d_low_pct'))}**",
            "",
            "### 特征快照",
            "",
            "```json",
        ]
    )
    import json

    lines.append(json.dumps(feats, ensure_ascii=False, indent=2, default=str))
    lines.append("```")

    if chart_rel_path:
        lines.extend(
            [
                "",
                "### 价量时序图",
                "",
                f"![价量时序图]({chart_rel_path})",
            ]
        )

    return "\n".join(lines)


def render_price_volume_chart(
    symbol: str,
    trade_date: str,
    save_dir: Path,
    *,
    lookback: int = 120,
) -> str | None:
    """Save price+volume chart; return path relative to report root."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from tradingagents.dataflows.stockstats_utils import load_ohlcv
    except ImportError:
        logger.warning("matplotlib not installed; skipping price/volume chart")
        return None

    df = load_ohlcv(symbol, trade_date)
    if df is None or df.empty:
        return None

    tail = df.tail(lookback).copy()
    if tail.empty:
        return None

    dates = tail["Date"]
    if hasattr(dates.iloc[0], "to_pydatetime"):
        dates = dates.apply(lambda x: x.to_pydatetime())

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "price_volume.png"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax1.plot(dates, tail["Close"], color="#2563eb", linewidth=1.5, label="Close")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{symbol.upper()} — OHLCV (as of {trade_date})")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(tail["Close"], tail["Open"])]
    ax2.bar(dates, tail["Volume"], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return "appendix/price_volume.png"


def write_report_appendix(
    final_state: dict,
    ticker: str,
    save_path: Path,
) -> tuple[str | None, dict[str, Any] | None]:
    """Write appendix files; return (markdown, stats dict)."""
    trade_date = final_state.get("trade_date")
    if not trade_date:
        return None, None

    try:
        stats, event = invoke_evidenced(
            run_id=final_state.get("run_id", "unknown-run"),
            producer_node="report_appendix",
            tool_name="build_appendix_stats",
            arguments={"ticker": ticker, "trade_date": str(trade_date)},
            call=lambda: build_appendix_stats(ticker, str(trade_date)),
            post_run_recomputed=True,
        )
        final_state.setdefault("audit_events", []).append(event)
        appendix_dir = save_path / "appendix"
        chart_rel = render_price_volume_chart(
            ticker, str(trade_date), appendix_dir
        )
        md = render_appendix_markdown(stats, chart_rel)
        appendix_dir.mkdir(parents=True, exist_ok=True)
        (appendix_dir / "statistics.md").write_text(md, encoding="utf-8")
        return md, stats
    except Exception:
        logger.exception("Failed to build report appendix for %s", ticker)
        return None, None
