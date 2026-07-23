"""Interactive decision audit and execution what-if workspace."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from tradingagents.extensions.contracts import ExecutionConfig
from tradingagents.extensions.paper_trading import (
    BacktestApplicationService,
    build_decision_replay,
)
from webui.components.charts import (
    allocation_figure,
    candlestick_figure,
    drawdown_figure,
    equity_figure,
    price_and_target_figure,
)
from webui.components.progress import StreamlitProgressObserver
from webui.components.style import (
    format_money,
    format_percent,
    render_badges,
    render_callout,
    render_empty,
    render_hero,
    render_metric_grid,
)
from webui.state import get_run_store, select_run, selected_run_id


def _tone(status: str) -> str:
    return {
        "SUCCESS": "success",
        "FILLED": "success",
        "COMPLETED": "success",
        "DEGRADED": "degraded",
        "PARTIAL": "degraded",
        "FAILED_SAFE": "failed",
        "REJECTED": "failed",
    }.get(status, "cyan")


def _selected_decision_from_chart(event: Any) -> str | None:
    try:
        points = event.selection.points
    except (AttributeError, KeyError):
        return None
    if not points:
        return None
    custom = points[-1].get("customdata")
    if isinstance(custom, (list, tuple)):
        return str(custom[0]) if custom else None
    return str(custom) if custom else None


def _render_overview(result) -> None:
    event = st.plotly_chart(
        equity_figure(result),
        width="stretch",
        key="replay-equity-chart",
        on_select="rerun",
        selection_mode="points",
        config={"displaylogo": False, "scrollZoom": True},
    )
    selected = _selected_decision_from_chart(event)
    if selected:
        st.session_state["selected_decision_id"] = selected
    left, right = st.columns([1.55, 1])
    with left:
        st.plotly_chart(
            allocation_figure(result),
            width="stretch",
            config={"displaylogo": False},
        )
    with right:
        st.plotly_chart(
            drawdown_figure(result),
            width="stretch",
            config={"displaylogo": False},
        )
    if result.decisions:
        symbol = st.selectbox(
            "Price/target lens",
            sorted({item.intent.symbol for item in result.decisions}),
            key="target-lens-symbol",
        )
        st.plotly_chart(
            price_and_target_figure(result, symbol),
            width="stretch",
            config={"displaylogo": False},
        )


def _render_portfolio(portfolio, title: str) -> None:
    if portfolio is None:
        st.caption(f"{title}: unavailable")
        return
    st.markdown(f"**{title}** · `{portfolio.as_of:%Y-%m-%d}`")
    st.caption(
        f"Cash {format_money(portfolio.cash)} · Equity {format_money(portfolio.total_equity)}"
    )
    if portfolio.positions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Symbol": symbol,
                        "Qty": position.quantity,
                        "Avg cost": position.average_cost,
                        "Mark": position.market_price,
                        "Value": position.market_value,
                        "Weight": position.weight,
                    }
                    for symbol, position in portfolio.positions.items()
                ]
            ),
            hide_index=True,
            width="stretch",
            column_config={"Weight": st.column_config.ProgressColumn(format="percent")},
        )
    else:
        st.caption("No open positions")


def _render_decision_audit(result) -> None:
    items = build_decision_replay(result)
    if not items:
        render_empty("没有决策记录", "本次运行没有产生可以审计的 DecisionEnvelope。")
        return
    item_by_id = {item.decision.intent.decision_id: item for item in items}
    ids = list(item_by_id)
    queued = st.session_state.get("selected_decision_id")
    default_index = ids.index(queued) if queued in ids else 0
    selected = st.selectbox(
        "Decision point",
        ids,
        index=default_index,
        format_func=lambda decision_id: (
            f"{item_by_id[decision_id].decision.intent.as_of:%Y-%m-%d} · "
            f"{item_by_id[decision_id].decision.intent.symbol} · "
            f"target {item_by_id[decision_id].decision.intent.target_weight:.0%}"
        ),
    )
    st.session_state["selected_decision_id"] = selected
    item = item_by_id[selected]
    decision = item.decision
    intent = decision.intent
    execution = item.execution
    badges = [
        (decision.status, _tone(decision.status)),
        (intent.symbol, "cyan"),
        (f"TARGET {intent.target_weight:.0%}", "amber"),
    ]
    if execution is not None:
        badges.append((execution.status, _tone(execution.status)))
    rating = intent.metadata.get("rating")
    if rating:
        badges.append((str(rating).upper(), "green"))
    render_badges(badges)
    confidence_label = (
        "Translation integrity"
        if intent.metadata.get("confidence_semantics")
        == "deterministic rating-translation integrity"
        else "Confidence"
    )
    render_metric_grid(
        [
            {"label": "Target weight", "value": f"{intent.target_weight:.1%}", "tone": "amber"},
            {
                "label": confidence_label,
                "value": f"{intent.confidence:.1%}",
                "tone": "cyan",
            },
            {
                "label": "Before",
                "value": format_percent(
                    item.portfolio_before.weight_for(intent.symbol)
                    if item.portfolio_before
                    else None
                ),
                "tone": "neutral",
            },
            {
                "label": "Achieved",
                "value": format_percent(execution.achieved_weight if execution else None),
                "tone": "positive"
                if execution and execution.status in {"FILLED", "NO_ACTION"}
                else "negative",
            },
            {
                "label": "Fees",
                "value": format_money(execution.fees if execution else None),
                "tone": "neutral",
            },
        ]
    )
    render_callout("Agent rationale", intent.rationale, tone="amber")
    allocation = intent.metadata.get("allocation")
    if isinstance(allocation, dict) and allocation.get("rule"):
        render_callout(
            "Rating → allocation",
            (
                f"{allocation.get('rating')} · {allocation.get('rule')} · "
                f"diversification cap {float(allocation.get('diversification_cap', 0)):.1%}"
            ),
            tone="cyan",
        )
    if intent.warnings:
        render_callout("Decision warnings", " · ".join(intent.warnings), tone="red")

    chart_column, evidence_column = st.columns([1.55, 1])
    with chart_column:
        if item.market is not None:
            last_fill = execution.fills[-1] if execution and execution.fills else None
            st.plotly_chart(
                candlestick_figure(
                    item.market,
                    fill_time=last_fill.timestamp if last_fill else None,
                    fill_price=last_fill.price if last_fill else None,
                ),
                width="stretch",
                config={"displaylogo": False, "scrollZoom": True},
            )
        else:
            st.info("该运行没有保存行情上下文。")
    with evidence_column:
        st.markdown("**Execution evidence**")
        if execution is None:
            st.caption("No execution report")
        elif execution.fills:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Side": fill.side,
                            "Qty": fill.quantity,
                            "Price": fill.price,
                            "Fee": fill.fee,
                            "At": fill.timestamp.strftime("%Y-%m-%d"),
                        }
                        for fill in execution.fills
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption(execution.rejection_reason or execution.status)
        _render_portfolio(item.portfolio_before, "Portfolio before")
        _render_portfolio(item.portfolio_after, "Portfolio after")

    memory_tab, dossier_tab, trace_tab, ledger_tab, diagnostics_tab = st.tabs(
        [
            "Retrieved memory",
            "Agent dossier",
            "Agent trace",
            "Ledger evidence",
            "Diagnostics",
        ]
    )
    with memory_tab:
        if item.memory is None:
            st.caption("No saved memory context")
        else:
            st.caption(item.memory.summary or "Memory provider returned no summary.")
            if item.memory.items:
                for memory_item in item.memory.items:
                    with st.container(border=True):
                        st.markdown(
                            f"`{memory_item.available_at:%Y-%m-%d}` · {memory_item.content}"
                        )
            else:
                st.caption("No time-safe prior outcomes were available at this point.")
    with dossier_tab:
        _render_agent_dossier(decision)
    with trace_tab:
        if decision.trace:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Time": event.timestamp,
                            "Source": event.source,
                            "Type": event.event_type,
                            "Summary": event.summary,
                        }
                        for event in decision.trace
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Provider emitted no trace events.")
    with ledger_tab:
        if item.ledger_entries:
            st.dataframe(pd.DataFrame(item.ledger_entries), hide_index=True, width="stretch")
        else:
            st.caption("This decision did not change the account ledger.")
    with diagnostics_tab:
        st.json(decision.diagnostics or {"message": "No diagnostics emitted."})


def _render_agent_dossier(decision) -> None:
    diagnostics = decision.diagnostics or {}
    reports = diagnostics.get("agent_reports")
    if not isinstance(reports, dict) or not any(reports.values()):
        st.caption("This provider did not emit a multi-agent report dossier.")
        return

    render_badges(
        [
            (f"RATING {diagnostics.get('rating') or 'UNKNOWN'}", "amber"),
            (f"GRAPH SIGNAL {diagnostics.get('graph_signal') or 'UNKNOWN'}", "cyan"),
            (
                f"PARSE {str(diagnostics.get('rating_parse_source') or 'UNKNOWN').upper()}",
                "green",
            ),
        ]
    )
    audit_counts = [
        ("Decision snapshots", len(diagnostics.get("decision_snapshots") or [])),
        ("Structured calls", len(diagnostics.get("structured_invocations") or [])),
        ("Claims", len(diagnostics.get("claims") or [])),
        ("Audit events", len(diagnostics.get("audit_events") or [])),
    ]
    with st.container(horizontal=True):
        for label, value in audit_counts:
            st.metric(label, value, border=True)

    stages = [
        ("market", "Market analyst", ":material/candlestick_chart:"),
        ("fundamentals", "Fundamentals analyst", ":material/account_balance:"),
        ("sentiment", "Sentiment analyst", ":material/forum:"),
        ("news", "News analyst", ":material/newspaper:"),
        ("research_plan", "Research manager", ":material/science:"),
        ("trader_plan", "Trader", ":material/swap_horiz:"),
        ("final_decision", "Portfolio manager", ":material/gavel:"),
    ]
    for key, label, icon in stages:
        content = str(reports.get(key) or "").strip()
        if not content:
            continue
        with st.expander(
            label,
            expanded=key == "final_decision",
            icon=icon,
        ):
            st.markdown(content)


def _render_what_if(store, stored) -> None:
    result = stored.result
    if result is None:
        return
    render_callout(
        "Execution-only counterfactual",
        "这里复用原 Agent 目标仓位，只重新计算资金、整数股、费用和滑点。它不会假装是一次新的 Agent 推理。",
        tone="amber",
    )
    with st.form("what-if-form"):
        columns = st.columns(4)
        with columns[0]:
            initial_cash = st.number_input(
                "Initial cash",
                min_value=1_000.0,
                value=float(stored.request.initial_cash),
                step=5_000.0,
                key="what-if-cash",
            )
        with columns[1]:
            commission = st.number_input(
                "Commission · %",
                min_value=0.0,
                max_value=5.0,
                value=float(stored.request.execution.commission_rate * 100),
                step=0.01,
                key="what-if-commission",
            )
        with columns[2]:
            slippage = st.number_input(
                "Slippage · bps",
                min_value=0.0,
                max_value=500.0,
                value=float(stored.request.execution.slippage_rate * 10_000),
                step=1.0,
                key="what-if-slippage",
            )
        with columns[3]:
            minimum_fee = st.number_input(
                "Minimum fee",
                min_value=0.0,
                value=float(stored.request.execution.minimum_fee),
                step=0.5,
                key="what-if-minimum-fee",
            )
        label = st.text_input("Scenario label", value=f"Cost stress · {stored.label}")
        submitted = st.form_submit_button("Run execution what-if", type="primary", width="stretch")
    if not submitted:
        return
    observer = StreamlitProgressObserver()
    try:
        service = BacktestApplicationService(None, store)
        scenario = service.run_what_if_and_store(
            stored.run_id,
            execution=ExecutionConfig(
                commission_rate=commission / 100,
                slippage_rate=slippage / 10_000,
                minimum_fee=minimum_fee,
            ),
            initial_cash=initial_cash,
            label=label,
            observer=observer,
        )
    except Exception as exc:
        st.error(f"What-if 未完成：{exc}")
        return
    select_run(scenario.run_id)
    scenario_result = scenario.result
    st.success(f"What-if 已保存：{scenario.run_id[:12]}")
    if scenario_result:
        render_metric_grid(
            [
                {
                    "label": "Return delta",
                    "value": format_percent(
                        scenario_result.metrics.get("total_return", 0)
                        - result.metrics.get("total_return", 0),
                        signed=True,
                    ),
                    "tone": "cyan",
                },
                {
                    "label": "Fee delta",
                    "value": format_money(
                        scenario_result.metrics.get("total_fees", 0)
                        - result.metrics.get("total_fees", 0)
                    ),
                    "tone": "amber",
                },
                {
                    "label": "Agent calls reused",
                    "value": scenario_result.metadata.get("agent_calls_reused", 0),
                    "tone": "green",
                },
            ]
        )


def _render_events(stored) -> None:
    if not stored.events:
        render_empty("没有事件记录", "旧运行或外部导入的结果可能没有 RunEvent。")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Time": event.timestamp,
                    "Stage": event.stage,
                    "Progress": event.progress,
                    "Message": event.message,
                }
                for event in stored.events
            ]
        ),
        hide_index=True,
        width="stretch",
        column_config={"Progress": st.column_config.ProgressColumn(min_value=0, max_value=1)},
    )


def render() -> None:
    render_hero(
        "AUDIT WORKSPACE",
        "Decision Replay",
        "沿着净值和交易时间轴，检查当时行情、检索记忆、Agent 目标、实际成交与账户结果。",
        accent="INTERACTIVE",
    )
    store = get_run_store()
    summaries = [item for item in store.list_runs() if item.status == "COMPLETED"]
    if not summaries:
        render_empty("没有可回放的运行", "请先在 Run Experiment 页面完成一次历史回放。")
        return
    run_ids = [item.run_id for item in summaries]
    current = selected_run_id()
    default_index = run_ids.index(current) if current in run_ids else 0
    labels = {item.run_id: item.label for item in summaries}
    selected = st.selectbox(
        "Archived run",
        run_ids,
        index=default_index,
        format_func=lambda run_id: f"{labels[run_id]} · {run_id[:8]}",
    )
    select_run(selected)
    stored = store.get_run(selected)
    result = stored.result
    if result is None:
        st.error("该运行没有结果对象。")
        return

    top_left, top_right = st.columns([4, 1])
    with top_left:
        render_badges(
            [
                (
                    str(result.metadata.get("run_kind", "FULL")),
                    "amber" if result.metadata.get("run_kind") == "EXECUTION_WHAT_IF" else "cyan",
                ),
                (str(result.metadata.get("market_data_source", "UNKNOWN")), "green"),
                (f"{len(result.decisions)} DECISIONS", "cyan"),
                (stored.status, _tone(stored.status)),
            ]
        )
    with top_right:
        st.download_button(
            "Export run JSON",
            data=json.dumps(store.export_run(selected), ensure_ascii=False, indent=2),
            file_name=f"tradingagents-run-{selected[:10]}.json",
            mime="application/json",
            width="stretch",
        )
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
                "label": "Alpha",
                "value": format_percent(
                    next(
                        (
                            value
                            for key, value in result.metrics.items()
                            if key.startswith("alpha_vs_")
                        ),
                        None,
                    ),
                    signed=True,
                ),
                "tone": "amber",
            },
            {
                "label": "Max drawdown",
                "value": format_percent(result.metrics.get("max_drawdown")),
                "tone": "negative",
            },
            {
                "label": "Sharpe",
                "value": f"{result.metrics.get('sharpe', 0):.2f}",
                "tone": "neutral",
            },
            {
                "label": "Total fees",
                "value": format_money(result.metrics.get("total_fees")),
                "tone": "neutral",
            },
        ]
    )
    if result.metadata.get("run_kind") == "EXECUTION_WHAT_IF":
        render_callout(
            "Reused Agent decisions",
            str(result.metadata.get("what_if_disclaimer", "Execution-layer what-if.")),
            tone="amber",
        )
    if result.warnings:
        with st.expander(
            f"Data and runtime warnings · {len(result.warnings)}", icon=":material/warning:"
        ):
            for warning in result.warnings:
                st.warning(warning)

    overview, decisions, what_if, timeline = st.tabs(
        ["Portfolio timeline", "Decision audit", "Execution What-if", "Run events"]
    )
    with overview:
        _render_overview(result)
    with decisions:
        _render_decision_audit(result)
    with what_if:
        _render_what_if(store, stored)
    with timeline:
        _render_events(stored)


__all__ = ["render"]
