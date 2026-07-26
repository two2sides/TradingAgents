"""Interactive decision audit and execution what-if workspace."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

import pandas as pd
import streamlit as st

from tradingagents.extensions.contracts import ExecutionConfig
from tradingagents.extensions.decision.credibility.profile import build_audit_profile
from tradingagents.extensions.decision.credibility.verifier import run_verifier
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

logger = logging.getLogger(__name__)

_AUDIT_METRICS = (
    ("evidence_coverage", "证据覆盖"),
    ("numeric_verification_rate", "数值核验"),
    ("temporal_integrity", "时间完整性"),
    ("tool_success_rate", "工具可用性"),
    ("structured_output_integrity", "结构化输出"),
    ("handoff_explanation_coverage", "交接解释"),
    ("debate_novelty_rate", "辩论新颖性"),
)


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


def _audit_tone(status: str) -> str:
    return {
        "AUDIT_PASSED": "success",
        "AUDIT_DEGRADED": "degraded",
        "AUDIT_FAILED": "failed",
    }.get(status, "cyan")


def _audit_metric_cards(profile: dict[str, Any]) -> list[dict[str, Any]]:
    counts_by_metric = profile.get("metric_counts") or {}
    cards = []
    for key, label in _AUDIT_METRICS:
        value = profile.get(key)
        counts = counts_by_metric.get(key) or {}
        failed = int(counts.get("failed_count") or 0)
        unknown = int(counts.get("unknown_count") or 0)
        eligible = int(counts.get("eligible_count") or 0)
        supported = int(counts.get("supported_count") or 0)
        excluded = int(counts.get("excluded_count") or 0)
        if value is None:
            tone = "neutral"
        elif failed:
            tone = "negative"
        elif unknown or value < 1:
            tone = "amber"
        else:
            tone = "positive"
        cards.append(
            {
                "label": label,
                "value": "N/A" if value is None else f"{float(value):.1%}",
                "delta": (
                    f"通过 {supported}/{eligible} · 排除 {excluded} · 未知 {unknown}"
                ),
                "tone": tone,
            }
        )
    return cards


def _render_run_credibility_summary(result) -> None:
    profiles = [
        (item.diagnostics or {}).get("audit_profile")
        for item in result.decisions
        if isinstance((item.diagnostics or {}).get("audit_profile"), dict)
    ]
    if not profiles:
        return
    status_counts = Counter(str(profile.get("status") or "UNKNOWN") for profile in profiles)
    render_badges(
        [
            ("可信度审计仅提供建议，不改变交易执行", "cyan"),
            (f"已画像 {len(profiles)}/{len(result.decisions)}", "green"),
        ]
    )
    render_metric_grid(
        [
            {
                "label": "审计通过",
                "value": status_counts.get("AUDIT_PASSED", 0),
                "tone": "positive",
            },
            {
                "label": "建议改进",
                "value": status_counts.get("AUDIT_DEGRADED", 0),
                "tone": "amber",
            },
            {
                "label": "需优先复核",
                "value": status_counts.get("AUDIT_FAILED", 0),
                "tone": "negative",
            },
            {
                "label": "无画像",
                "value": len(result.decisions) - len(profiles),
                "tone": "neutral",
            },
        ]
    )


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
    _render_run_credibility_summary(result)


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

    credibility_tab, memory_tab, dossier_tab, trace_tab, ledger_tab, diagnostics_tab = st.tabs(
        [
            "可信度建议",
            "Retrieved memory",
            "Agent dossier",
            "Agent trace",
            "Ledger evidence",
            "Diagnostics",
        ],
        key=f"decision-evidence-{intent.decision_id}",
        on_change="rerun",
    )
    if credibility_tab.open:
        with credibility_tab:
            _render_credibility(decision)
    if memory_tab.open:
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
                provenance = (decision.diagnostics or {}).get("memory_provenance") or []
                if provenance:
                    st.markdown("**Memory provenance**")
                    st.dataframe(
                        pd.DataFrame(provenance),
                        hide_index=True,
                        width="stretch",
                    )
    if dossier_tab.open:
        with dossier_tab:
            _render_agent_dossier(decision)
    if trace_tab.open:
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
    if ledger_tab.open:
        with ledger_tab:
            if item.ledger_entries:
                st.dataframe(pd.DataFrame(item.ledger_entries), hide_index=True, width="stretch")
            else:
                st.caption("This decision did not change the account ledger.")
    if diagnostics_tab.open:
        with diagnostics_tab:
            st.json(decision.diagnostics or {"message": "No diagnostics emitted."})


def _render_credibility(decision) -> None:
    diagnostics = decision.diagnostics or {}
    profile, findings, claims, rebuilt = _resolve_credibility_view(decision)
    if not isinstance(profile, dict):
        error = diagnostics.get("audit_error")
        if error:
            render_callout(
                "可信度画像暂不可用",
                f"审计投影发生错误：{error}。原始交易结论和执行保持不变。",
                tone="red",
            )
        else:
            render_callout(
                "旧运行暂无可信度画像",
                "该决策仍可查看原始 Claims、Audit events 与 Diagnostics；"
                "重新运行后会生成完整画像。交易执行不受影响。",
                tone="cyan",
            )
        return

    status = str(profile.get("status") or "UNKNOWN")
    scope = str(profile.get("audit_scope") or "UNKNOWN")
    render_badges(
        [
            (status, _audit_tone(status)),
            (f"SCOPE {scope}", "amber" if scope == "PARTIAL" else "green"),
            ("ADVISORY ONLY", "cyan"),
            ("PM RATING UNCHANGED", "green"),
            *(([("REBUILT FROM STORED DIAGNOSTICS", "amber")]) if rebuilt else []),
        ]
    )
    render_callout(
        "如何理解",
        "这里衡量证据、时间、结构化输出与交接完整性，不是上涨概率，"
        "不会自动阻止交易、改变仓位或覆盖 Portfolio Manager Rating。",
        tone="cyan",
    )
    render_metric_grid(_audit_metric_cards(profile))

    advisories = [str(item) for item in profile.get("advisories") or [] if item]
    st.markdown("#### 改进建议")
    if advisories:
        for advisory in advisories:
            st.markdown(f"- {advisory}")
    else:
        st.success("当前适用规则没有生成额外改进建议。")

    st.markdown("#### 核验发现")
    if findings:
        severity_counts = Counter(
            str(item.get("severity") or "UNKNOWN") for item in findings
        )
        render_badges(
            [
                (f"CRITICAL {severity_counts.get('CRITICAL', 0)}", "red"),
                (f"WARNING {severity_counts.get('WARNING', 0)}", "amber"),
                (f"INFO {severity_counts.get('INFO', 0)}", "cyan"),
            ]
        )
        finding_rows = []
        for finding in findings:
            finding_rows.append(
                {
                    "级别": finding.get("severity"),
                    "规则": finding.get("code"),
                    "阶段": finding.get("stage"),
                    "说明": finding.get("message"),
                    "位置": finding.get("location") or "—",
                    "预期": _display_value(finding.get("expected")),
                    "实际": _display_value(finding.get("actual")),
                    "Claim ID": finding.get("claim_id") or "—",
                    "Artifact ID": finding.get("artifact_id") or "—",
                }
            )
        st.dataframe(
            pd.DataFrame(finding_rows),
            hide_index=True,
            width="stretch",
        )
    else:
        st.success("未发现适用规则能够确定识别的问题。")

    snapshots = [
        item
        for item in diagnostics.get("decision_snapshots") or []
        if isinstance(item, dict)
    ]
    if snapshots:
        st.markdown("#### 决策交接链")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "阶段": item.get("stage"),
                        "决策": item.get("value"),
                        "解析成功": bool(item.get("parsed")),
                        "与上游关系": item.get("alignment"),
                        "上游引用": item.get("upstream_decision_ref") or "—",
                        "改变理由": " · ".join(
                            str(ref) for ref in item.get("change_reason_refs") or []
                        )
                        or "—",
                    }
                    for item in snapshots
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    important_claims = [
        item
        for item in claims
        if item.get("importance") in {"CRITICAL", "MAJOR"}
    ]
    if important_claims:
        claim_expander = st.expander(
            f"关键主张与证据定位（{len(important_claims)}）",
            expanded=False,
            icon=":material/account_tree:",
        )
        if claim_expander.open:
            with claim_expander:
                rows = []
                for claim in important_claims[:200]:
                    refs = claim.get("evidence_refs") or []
                    rows.append(
                        {
                            "状态": claim.get("verification_status"),
                            "Agent": claim.get("agent"),
                            "阶段": claim.get("stage"),
                            "类型": claim.get("claim_type"),
                            "主张": claim.get("text"),
                            "证据数": len(refs),
                            "证据定位": " · ".join(
                                f"{ref.get('artifact_id')}#{ref.get('selector') or '/'}"
                                for ref in refs
                                if isinstance(ref, dict)
                            )
                            or "—",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                if len(important_claims) > 200:
                    st.caption("为保持界面流畅，仅显示前 200 条关键主张。")


def _resolve_credibility_view(
    decision,
) -> tuple[dict[str, Any] | None, list[dict], list[dict], bool]:
    """Return persisted audit views, or rebuild legacy views without writing data."""

    diagnostics = decision.diagnostics or {}
    profile = diagnostics.get("audit_profile")
    findings = [
        item for item in diagnostics.get("audit_findings") or [] if isinstance(item, dict)
    ]
    claims = [
        item for item in diagnostics.get("claims") or [] if isinstance(item, dict)
    ]
    if isinstance(profile, dict):
        return profile, findings, claims, False
    if not any(
        diagnostics.get(key)
        for key in (
            "audit_events",
            "claims",
            "structured_invocations",
            "decision_snapshots",
        )
    ):
        return None, findings, claims, False

    reports = diagnostics.get("agent_reports") or {}
    final_state = {
        "run_id": diagnostics.get("run_id")
        or decision.intent.metadata.get("graph_run_id")
        or decision.intent.decision_id,
        "trade_date": diagnostics.get("trade_date")
        or decision.intent.as_of.date().isoformat(),
        "audit_events": diagnostics.get("audit_events") or [],
        "claims": claims,
        "structured_invocations": diagnostics.get("structured_invocations") or [],
        "decision_snapshots": diagnostics.get("decision_snapshots") or [],
        "investment_debate_state": {
            "debate_turns": diagnostics.get("debate_turns") or []
        },
        "market_report": reports.get("market") if isinstance(reports, dict) else "",
        "news_report": reports.get("news") if isinstance(reports, dict) else "",
        "fundamentals_report": (
            reports.get("fundamentals") if isinstance(reports, dict) else ""
        ),
    }
    try:
        events, audited_claims, audited_findings = run_verifier(final_state)
        rebuilt_profile = build_audit_profile(
            final_state, events, audited_claims, audited_findings
        ).model_dump(mode="json")
    except Exception:
        logger.exception(
            "Could not rebuild credibility view for %s",
            decision.intent.decision_id,
        )
        return None, findings, claims, False
    return rebuilt_profile, audited_findings, audited_claims, True


def _display_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


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
        report_expander = st.expander(
            label,
            expanded=key == "final_decision",
            icon=icon,
            key=f"agent-report-{decision.intent.decision_id}-{key}",
            on_change="rerun",
        )
        if report_expander.open:
            with report_expander:
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
        logger.exception("Execution what-if failed parent_run_id=%s", stored.run_id)
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
        warnings_expander = st.expander(
            f"Data and runtime warnings · {len(result.warnings)}",
            icon=":material/warning:",
            key=f"run-warnings-{stored.run_id}",
            on_change="rerun",
        )
        if warnings_expander.open:
            with warnings_expander:
                for warning in result.warnings:
                    st.warning(warning)

    overview, decisions, what_if, timeline = st.tabs(
        ["Portfolio timeline", "Decision audit", "Execution What-if", "Run events"],
        key="replay-workspace-tabs",
        on_change="rerun",
    )
    if overview.open:
        with overview:
            _render_overview(result)
    if decisions.open:
        with decisions:
            _render_decision_audit(result)
    if what_if.open:
        with what_if:
            _render_what_if(store, stored)
    if timeline.open:
        with timeline:
            _render_events(stored)


__all__ = ["render"]
