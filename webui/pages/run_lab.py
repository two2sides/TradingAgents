"""Create and execute a historical experiment."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import streamlit as st

from tradingagents.extensions.contracts import BacktestRequest, ExecutionConfig
from tradingagents.extensions.paper_trading import (
    BacktestApplicationService,
    DemoMemoryProvider,
    HistoricalMarketDataProvider,
    MovingAverageDecisionProvider,
    generate_demo_market_data,
)
from webui.components.progress import StreamlitProgressObserver
from webui.components.style import (
    format_money,
    format_percent,
    render_badges,
    render_callout,
    render_hero,
    render_metric_grid,
    render_section,
)
from webui.state import get_run_store, select_run


@st.cache_resource(show_spinner=False)
def _load_yfinance(
    symbols: tuple[str, ...],
    start_iso: str,
    end_iso: str,
) -> HistoricalMarketDataProvider:
    return HistoricalMarketDataProvider.from_yfinance(
        symbols,
        datetime.fromisoformat(start_iso),
        datetime.fromisoformat(end_iso),
    )


def _as_day_start(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _as_day_end(value: date) -> datetime:
    return datetime.combine(value, time.max, tzinfo=timezone.utc)


def render() -> None:
    render_hero(
        "EXPERIMENT CONTROL",
        "Run a time-safe replay",
        "配置一次可复现的历史实验；Agent 决策、执行报价和账户变化都会进入审计档案。",
        accent="DEMO READY",
    )
    render_badges(
        [
            ("NEXT OPEN EXECUTION", "cyan"),
            ("LONG ONLY", "amber"),
            ("DETERMINISTIC DEMO", "green"),
        ]
    )
    render_callout(
        "Provider boundary",
        "当前页面默认使用可解释的演示 DecisionProvider。C 的真实实现接入后只需替换 Provider，不修改回测器、Broker 或页面。",
        tone="amber",
    )

    today = date.today()
    default_start = today - timedelta(days=150)
    render_section("Experiment specification", "费用和滑点均会进入真实成交与账户账本。", index="01")
    with st.form("backtest-request", border=True):
        top = st.columns([1.5, 1, 1])
        with top[0]:
            symbols_text = st.text_input(
                "Symbols", value="AAPL", help="逗号分隔，建议演示时 1–3 个。"
            )
        with top[1]:
            start_date = st.date_input("Start", value=default_start, max_value=today)
        with top[2]:
            end_date = st.date_input("End", value=today, max_value=today)

        middle = st.columns(4)
        with middle[0]:
            initial_cash = st.number_input(
                "Initial cash", min_value=1_000.0, value=100_000.0, step=5_000.0
            )
        with middle[1]:
            decision_interval = st.number_input(
                "Decision interval · bars", min_value=1, max_value=30, value=5
            )
        with middle[2]:
            lookback = st.number_input(
                "Context lookback · bars", min_value=8, max_value=250, value=60
            )
        with middle[3]:
            outcome_horizon = st.number_input(
                "Outcome horizon · bars", min_value=1, max_value=60, value=5
            )

        costs = st.columns(4)
        with costs[0]:
            commission_percent = st.number_input(
                "Commission · %", min_value=0.0, max_value=5.0, value=0.05, step=0.01
            )
        with costs[1]:
            slippage_bps = st.number_input(
                "Slippage · bps", min_value=0.0, max_value=500.0, value=10.0, step=1.0
            )
        with costs[2]:
            minimum_fee = st.number_input("Minimum fee", min_value=0.0, value=0.0, step=0.5)
        with costs[3]:
            source = st.selectbox(
                "Market source", ["Built-in deterministic demo", "yfinance daily"]
            )

        label = st.text_input("Run label", value="Decision Lab experiment")
        submitted = st.form_submit_button(
            "Launch historical replay", type="primary", width="stretch"
        )

    if not submitted:
        render_section("Execution contract", "提交后页面将实时显示事件流水线。", index="02")
        st.code(
            "T close snapshot → MemoryProvider → DecisionProvider\n"
            "→ T+1 common-bar open quote → LedgerBroker → mark-to-market",
            language=None,
        )
        return

    symbols = tuple(
        dict.fromkeys(item.strip().upper() for item in symbols_text.split(",") if item.strip())
    )
    if not symbols:
        st.error("请至少输入一个股票代码。")
        return
    if len(symbols) > 5:
        st.error("一周版本最多同时回放 5 个标的。")
        return
    start_at, end_at = _as_day_start(start_date), _as_day_end(end_date)
    if start_at >= end_at:
        st.error("结束日期必须晚于开始日期。")
        return

    request = BacktestRequest(
        symbols=list(symbols),
        start=start_at,
        end=end_at,
        initial_cash=initial_cash,
        lookback=int(lookback),
        decision_interval_bars=int(decision_interval),
        outcome_horizon_bars=int(outcome_horizon),
        execution=ExecutionConfig(
            commission_rate=commission_percent / 100,
            slippage_rate=slippage_bps / 10_000,
            minimum_fee=minimum_fee,
        ),
        metadata={"ui_source": source},
    )
    fetch_start = start_at - timedelta(days=max(45, int(lookback) * 2))
    render_section("Live run pipeline", "每条事件同时写入本地运行档案。", index="02")
    progress = StreamlitProgressObserver()
    try:
        if source.startswith("Built-in"):
            provider = generate_demo_market_data(symbols, fetch_start, end_at)
        else:
            with st.spinner("Downloading adjusted daily bars from yfinance…"):
                provider = _load_yfinance(symbols, fetch_start.isoformat(), end_at.isoformat())
        service = BacktestApplicationService(provider, get_run_store())
        stored = service.run_and_store(
            request,
            MovingAverageDecisionProvider(),
            DemoMemoryProvider(),
            label=label,
            observer=progress,
        )
    except Exception as exc:
        st.error(f"回测未完成：{exc}")
        return

    select_run(stored.run_id)
    result = stored.result
    if result is None:
        st.error("运行已保存，但缺少结果对象。")
        return
    st.success(f"运行已归档：{stored.run_id[:12]}")
    render_metric_grid(
        [
            {
                "label": "Final equity",
                "value": format_money(result.metrics.get("final_equity")),
                "tone": "cyan",
            },
            {
                "label": "Total return",
                "value": format_percent(result.metrics.get("total_return"), signed=True),
                "tone": "positive" if result.metrics.get("total_return", 0) >= 0 else "negative",
            },
            {
                "label": "Max drawdown",
                "value": format_percent(result.metrics.get("max_drawdown")),
                "tone": "negative",
            },
            {"label": "Trades", "value": int(result.metrics.get("fill_count", 0)), "tone": "amber"},
            {
                "label": "Fees",
                "value": format_money(result.metrics.get("total_fees")),
                "tone": "neutral",
            },
        ]
    )
    st.info("结果已设为当前运行。请从左侧进入 Decision Replay 查看交易点和完整证据链。")


__all__ = ["render"]
