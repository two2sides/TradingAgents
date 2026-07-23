"""Create and execute a historical experiment."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

import streamlit as st

from tradingagents.extensions.contracts import BacktestRequest, ExecutionConfig
from tradingagents.extensions.paper_trading import (
    BacktestApplicationService,
    DemoMemoryProvider,
    HistoricalMarketDataProvider,
    MarketDataRateLimited,
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
from webui.state import get_agent_runtime, get_run_store, select_run

DEMO_MODE = "Fast demo"
AGENT_MODE = "TradingAgents + RAG"
ANALYST_OPTIONS = {
    "Market": "market",
    "Social": "social",
    "News": "news",
    "Fundamentals": "fundamentals",
}


def _run_error_details(exc: Exception, *, real_mode: bool) -> tuple[str, str | None]:
    """Return a precise user-facing error and an optional next action."""

    message = str(exc).strip() or type(exc).__name__
    if isinstance(exc, MarketDataRateLimited):
        return (
            f"行情服务限流：{message}",
            "稍后重试，或把 Market source 改为 Built-in execution sandbox。",
        )
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        missing = getattr(exc, "name", None)
        package = f" `{missing}`" if missing else ""
        return (
            f"运行依赖缺失{package}：{message}",
            "执行 `uv sync --extra dev --extra webui --extra memory` 后重启 WebUI。",
        )
    if real_mode and (
        "rate limit" in message.lower()
        or "rate limited" in message.lower()
        or "too many requests" in message.lower()
        or "429" in message
    ):
        return (
            f"模型或外部数据服务限流：{message}",
            "等待服务配额恢复后重试；已经完成的运行档案不会受影响。",
        )
    return f"回测未完成：{message}", None


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
        accent="TWO ENGINES",
    )
    engine = st.segmented_control(
        "Decision engine",
        [DEMO_MODE, AGENT_MODE],
        default=DEMO_MODE,
        required=True,
        key="decision-engine",
        width="stretch",
        persist_state="page",
    )
    real_mode = engine == AGENT_MODE
    if real_mode:
        render_badges(
            [
                ("TRADINGAGENTS GRAPH", "cyan"),
                ("ENHANCED RAG MEMORY", "green"),
                ("NEXT OPEN EXECUTION", "amber"),
            ]
        )
        render_callout(
            "Real provider path",
            "每个决策点都会运行完整多 Agent 图。B 的记忆先由回测器按历史时点检索，再以只读上下文注入图；最终评级通过显式仓位策略交给 Broker。",
            tone="cyan",
        )
    else:
        render_badges(
            [
                ("NEXT OPEN EXECUTION", "cyan"),
                ("LONG ONLY", "amber"),
                ("DETERMINISTIC DEMO", "green"),
            ]
        )
        render_callout(
            "Fast, offline baseline",
            "演示引擎使用确定性行情、内存记忆和可解释均线策略，适合快速展示执行、审计与 What-if；它不会调用 LLM。",
            tone="amber",
        )

    today = date.today()
    default_start = today - timedelta(days=21 if real_mode else 150)
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
                "Decision interval · bars",
                min_value=1,
                max_value=60,
                value=30 if real_mode else 5,
                key=f"decision-interval-{engine}",
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
                "Market source",
                ["yfinance daily", "Built-in execution sandbox"]
                if real_mode
                else ["Built-in deterministic demo", "yfinance daily"],
                help=(
                    "Sandbox 使用确定性执行行情，适合在 Yahoo 限流时验证 Agent、"
                    "记忆和 Broker；它不代表真实历史价格。"
                    if real_mode
                    else None
                ),
            )

        selected_analyst_labels: list[str] = []
        max_position_percent = 35.0
        if real_mode:
            agent_columns = st.columns([2, 1])
            with agent_columns[0]:
                selected_analyst_labels = st.multiselect(
                    "Agent analysts",
                    list(ANALYST_OPTIONS),
                    default=["Market"],
                    help="每个被选中的分析师都会参与每个决策点的完整图运行。",
                )
            with agent_columns[1]:
                max_position_percent = st.number_input(
                    "Position entry cap · %",
                    min_value=1.0,
                    max_value=100.0,
                    value=35.0,
                    step=5.0,
                    help="Buy 的单标的建仓上限；多标的时还会应用 1/N 分散上限。",
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
        if real_mode:
            st.caption(
                "建议首次使用 1 个标的、约 21–30 天窗口、30 bars 决策间隔；"
                "完整 Agent 图远慢于演示策略。"
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
    if real_mode and not selected_analyst_labels:
        st.error("真实 Agent 模式至少选择一个分析师。")
        return
    start_at, end_at = _as_day_start(start_date), _as_day_end(end_date)
    if start_at >= end_at:
        st.error("结束日期必须晚于开始日期。")
        return

    estimated_bars = max(1, math.ceil((end_at - start_at).days * 5 / 7))
    estimated_agent_calls = math.ceil(estimated_bars / int(decision_interval)) * len(symbols)
    if real_mode and estimated_agent_calls > 12:
        st.error(
            f"当前配置预计需要约 {estimated_agent_calls} 次完整 Agent 图调用。"
            "一周版单次实验上限为 12 次；请缩短窗口、增加决策间隔或减少标的。"
        )
        return

    analyst_ids = tuple(ANALYST_OPTIONS[label] for label in selected_analyst_labels)
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
        metadata={
            "ui_source": source,
            "decision_engine": "tradingagents_rag" if real_mode else "deterministic_demo",
            "selected_analysts": list(analyst_ids),
            "estimated_agent_calls": estimated_agent_calls if real_mode else 0,
        },
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
        if real_mode:
            with st.status(
                "Preparing TradingAgents graph and RAG memory…",
                expanded=True,
            ) as runtime_status:
                runtime_status.write(
                    "Loading configured LLM clients, persistent Chroma store, "
                    "and the embedding model."
                )
                runtime = get_agent_runtime(
                    analyst_ids,
                    max_position_percent / 100,
                )
                runtime_status.write(
                    f"Provider: {runtime.details['llm_provider']} · "
                    f"quick: {runtime.details['quick_model']} · "
                    f"deep: {runtime.details['deep_model']}"
                )
                runtime_status.write(f"Estimated full graph calls: {estimated_agent_calls}")
                runtime_status.update(
                    label="TradingAgents + RAG runtime ready",
                    state="complete",
                    expanded=False,
                )
            decision_provider = runtime.decision_provider
            memory_provider = runtime.memory_provider
        else:
            decision_provider = MovingAverageDecisionProvider()
            memory_provider = DemoMemoryProvider()
        stored = service.run_and_store(
            request,
            decision_provider,
            memory_provider,
            label=label,
            observer=progress,
        )
    except Exception as exc:
        message, action = _run_error_details(exc, real_mode=real_mode)
        st.error(message)
        if action:
            st.info(action, icon=":material/lightbulb:")
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
