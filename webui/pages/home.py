"""Decision Lab landing page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from webui.components.style import (
    format_percent,
    render_badges,
    render_callout,
    render_empty,
    render_hero,
    render_metric_grid,
    render_section,
)
from webui.state import get_run_store, select_run


def render() -> None:
    render_hero(
        "MULTI-AGENT STRATEGY OBSERVATORY",
        "Decision Lab",
        "把每一次 Agent 判断还原成可验证的行情、记忆、仓位、成交与账户结果。",
        accent="AUDIT READY",
    )
    store = get_run_store()
    summaries = store.list_runs(limit=100)
    completed = [item for item in summaries if item.status == "COMPLETED"]
    best = max(
        (item.total_return for item in completed if item.total_return is not None),
        default=None,
    )
    latest = summaries[0].created_at.strftime("%m-%d %H:%M") if summaries else "—"
    render_metric_grid(
        [
            {"label": "Archived runs", "value": len(summaries), "tone": "cyan"},
            {"label": "Completed", "value": len(completed), "tone": "positive"},
            {
                "label": "Best return",
                "value": format_percent(best, signed=True),
                "tone": "positive" if best is not None and best >= 0 else "negative",
            },
            {"label": "Latest activity", "value": latest, "tone": "amber"},
        ]
    )

    render_section(
        "From opinion to evidence",
        "Decision Lab 让回测过程成为可以逐点检查的实验记录。",
        index="01",
    )
    columns = st.columns(4)
    steps = [
        ("TIME-SAFE DATA", "锁定决策时刻，仅暴露当时已经出现的 K 线与记忆。"),
        ("AGENT TARGET", "记录目标仓位、置信度、理由、降级状态与诊断事件。"),
        ("LEDGER EXECUTION", "下一根开盘成交，显式处理滑点、费用、整数股与现金约束。"),
        ("REPLAY & COMPARE", "点击交易点审计证据，或复用决策进行执行层 What-if。"),
    ]
    for column, (title, body) in zip(columns, steps, strict=True):
        with column:
            render_callout(title, body, tone="cyan" if title != "LEDGER EXECUTION" else "amber")

    render_section(
        "Recent experiments",
        "运行档案保存在本地 SQLite，重启页面后仍可继续审阅。",
        index="02",
    )
    if not summaries:
        render_empty(
            "还没有运行档案",
            "打开 Run Experiment，使用内置行情即可在几秒内生成第一份可回放结果。",
        )
        return

    render_badges(
        [
            (f"{len(completed)} COMPLETED", "green"),
            (f"{sum(item.status == 'FAILED' for item in summaries)} FAILED", "red"),
            ("LOCAL SQLITE", "cyan"),
        ]
    )
    rows = [
        {
            "Run": item.run_id[:10],
            "Label": item.label,
            "Status": item.status,
            "Symbols": ", ".join(item.symbols),
            "Window": f"{item.start:%Y-%m-%d} → {item.end:%Y-%m-%d}",
            "Return": format_percent(item.total_return, signed=True),
            "Max DD": format_percent(item.max_drawdown),
            "Created": item.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for item in summaries[:12]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    completed_ids = [item.run_id for item in completed]
    if completed_ids:
        labels = {item.run_id: item.label for item in completed}
        selected = st.selectbox(
            "Queue a run for Decision Replay",
            completed_ids,
            format_func=lambda run_id: f"{labels[run_id]} · {run_id[:8]}",
        )
        if st.button("Open in replay workspace", type="primary", width="stretch"):
            select_run(selected)
            st.success("已选中运行。请从左侧进入 Decision Replay。")


__all__ = ["render"]
