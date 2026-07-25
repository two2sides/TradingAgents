"""Side-by-side experiment comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from webui.components.charts import comparison_figure
from webui.components.style import (
    format_money,
    format_percent,
    render_badges,
    render_empty,
    render_hero,
    render_metric_grid,
    render_section,
)
from webui.state import get_run_store


def _decision_map(result):
    return {
        (item.intent.symbol, item.intent.as_of): item.intent.target_weight
        for item in result.decisions
    }


def render() -> None:
    render_hero(
        "EXPERIMENT COMPARISON",
        "Compare runs, not screenshots",
        "归一化比较净值、风险、费用和每个时间点的目标仓位差异。",
        accent="A / B",
    )
    store = get_run_store()
    summaries = [item for item in store.list_runs() if item.status == "COMPLETED"]
    if len(summaries) < 2:
        render_empty("至少需要两次运行", "完成一个原始回测和一个 What-if 后即可进入对比。")
        return

    options = [item.run_id for item in summaries]
    labels = {item.run_id: item.label for item in summaries}
    selectors = st.columns(2)
    with selectors[0]:
        left_id = st.selectbox(
            "Run A",
            options,
            index=0,
            format_func=lambda run_id: f"{labels[run_id]} · {run_id[:8]}",
        )
    with selectors[1]:
        right_id = st.selectbox(
            "Run B",
            options,
            index=1,
            format_func=lambda run_id: f"{labels[run_id]} · {run_id[:8]}",
        )
    if left_id == right_id:
        st.warning("请选择两个不同运行。")
        return
    left, right = store.get_run(left_id), store.get_run(right_id)
    if left.result is None or right.result is None:
        st.error("选择的运行缺少结果对象。")
        return
    left_kind = str(left.result.metadata.get("run_kind", "FULL"))
    right_kind = str(right.result.metadata.get("run_kind", "FULL"))
    render_badges(
        [
            (f"A · {left_kind}", "cyan"),
            (f"B · {right_kind}", "amber"),
            (f"{', '.join(left.request.symbols)}", "green"),
        ]
    )
    left_return = left.result.metrics.get("total_return", 0)
    right_return = right.result.metrics.get("total_return", 0)
    render_metric_grid(
        [
            {
                "label": "A return",
                "value": format_percent(left_return, signed=True),
                "tone": "cyan",
            },
            {
                "label": "B return",
                "value": format_percent(right_return, signed=True),
                "tone": "amber",
            },
            {
                "label": "B − A",
                "value": format_percent(right_return - left_return, signed=True),
                "tone": "positive" if right_return >= left_return else "negative",
            },
            {
                "label": "A fees",
                "value": format_money(left.result.metrics.get("total_fees")),
                "tone": "neutral",
            },
            {
                "label": "B fees",
                "value": format_money(right.result.metrics.get("total_fees")),
                "tone": "neutral",
            },
        ]
    )

    render_section(
        "Normalized performance", "两次运行均从 0% 起点绘制，初始资金不同也可比较。", index="01"
    )
    st.plotly_chart(
        comparison_figure(
            [(f"A · {left.label}", left.result), (f"B · {right.label}", right.result)]
        ),
        width="stretch",
        config={"displaylogo": False, "scrollZoom": True},
    )

    render_section("Metric attribution", "数值差异用于定位收益、风险和成本变化。", index="02")
    metric_names = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "sharpe",
        "sortino",
        "turnover",
        "total_fees",
        "fill_count",
        "rejection_count",
    ]
    metric_rows = []
    for name in metric_names:
        a_value = left.result.metrics.get(name)
        b_value = right.result.metrics.get(name)
        metric_rows.append(
            {
                "Metric": name,
                "Run A": a_value,
                "Run B": b_value,
                "B − A": (b_value - a_value)
                if a_value is not None and b_value is not None
                else None,
            }
        )
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")

    render_section(
        "Decision divergence", "按 symbol + as_of 对齐目标仓位，而不是按列表位置猜测。", index="03"
    )
    left_decisions, right_decisions = _decision_map(left.result), _decision_map(right.result)
    keys = sorted(set(left_decisions) | set(right_decisions), key=lambda item: (item[1], item[0]))
    divergence = [
        {
            "Date": timestamp,
            "Symbol": symbol,
            "Target A": left_decisions.get((symbol, timestamp)),
            "Target B": right_decisions.get((symbol, timestamp)),
            "Delta": (
                right_decisions[(symbol, timestamp)] - left_decisions[(symbol, timestamp)]
                if (symbol, timestamp) in left_decisions and (symbol, timestamp) in right_decisions
                else None
            ),
        }
        for symbol, timestamp in keys
    ]
    st.dataframe(
        pd.DataFrame(divergence),
        hide_index=True,
        width="stretch",
        column_config={
            "Target A": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Target B": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent"),
            "Delta": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if all(row["Delta"] == 0 for row in divergence if row["Delta"] is not None):
        st.info("两个运行的目标仓位完全一致；结果差异来自执行参数或初始资金。")


__all__ = ["render"]
