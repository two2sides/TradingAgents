"""Plotly figures shared by the replay and comparison pages."""

from __future__ import annotations

from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradingagents.extensions.contracts import BacktestResult, MarketSnapshot

CYAN = "#55D6E6"
AMBER = "#F2B84B"
GREEN = "#58D68D"
RED = "#FF6B7A"
MUTED = "#6F8A9B"
GRID = "rgba(133, 174, 198, 0.10)"
PAPER = "rgba(0,0,0,0)"


def _base_layout(figure: go.Figure, *, height: int, title: str | None = None) -> go.Figure:
    figure.update_layout(
        height=height,
        title={"text": title, "font": {"size": 14, "color": "#DDEAF0"}, "x": 0.02}
        if title
        else None,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        margin={"l": 22, "r": 22, "t": 48 if title else 24, "b": 26},
        font={"color": "#91A8B7", "family": "Inter, Arial, sans-serif", "size": 11},
        hovermode="x unified",
        hoverlabel={"bgcolor": "#10202D", "bordercolor": "#274355", "font_color": "#E8F0F5"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, showspikes=True, spikecolor=CYAN)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    return figure


def equity_figure(result: BacktestResult) -> go.Figure:
    figure = go.Figure()
    curve = result.equity_curve
    figure.add_trace(
        go.Scatter(
            x=[point.timestamp for point in curve],
            y=[point.total_equity for point in curve],
            mode="lines",
            name="Agent portfolio",
            line={"color": CYAN, "width": 2.7},
            fill="tozeroy",
            fillcolor="rgba(85,214,230,0.055)",
            hovertemplate="%{y:,.2f}<extra>Agent portfolio</extra>",
        )
    )
    palette = [AMBER, "#9A8CFF", "#78B6FF"]
    for index, (name, benchmark) in enumerate(result.benchmark_curves.items()):
        figure.add_trace(
            go.Scatter(
                x=[point.timestamp for point in benchmark],
                y=[point.total_equity for point in benchmark],
                mode="lines",
                name=name.replace("_", " "),
                line={"color": palette[index % len(palette)], "width": 1.35, "dash": "dot"},
                hovertemplate=f"%{{y:,.2f}}<extra>{name}</extra>",
            )
        )

    equity_by_time = {point.timestamp: point.total_equity for point in curve}
    for side, color, symbol in (("BUY", GREEN, "triangle-up"), ("SELL", RED, "triangle-down")):
        points = []
        for execution in result.executions:
            for fill in execution.fills:
                if fill.side == side:
                    closest = min(
                        curve,
                        key=lambda point: abs((point.timestamp - fill.timestamp).total_seconds()),
                    )
                    points.append((fill, execution.decision_id, equity_by_time[closest.timestamp]))
        if points:
            figure.add_trace(
                go.Scatter(
                    x=[item[0].timestamp for item in points],
                    y=[item[2] for item in points],
                    mode="markers",
                    name=side,
                    marker={
                        "color": color,
                        "size": 11,
                        "symbol": symbol,
                        "line": {"width": 1, "color": "#071018"},
                    },
                    customdata=[[item[1]] for item in points],
                    text=[
                        f"{item[0].symbol} · {item[0].quantity} @ {item[0].price:.2f}"
                        for item in points
                    ],
                    hovertemplate="%{text}<br>Click to audit<extra>" + side + "</extra>",
                )
            )
    return _base_layout(
        figure, height=470, title="Portfolio equity · click a trade marker to audit"
    )


def drawdown_figure(result: BacktestResult) -> go.Figure:
    peak = 0.0
    values = []
    for point in result.equity_curve:
        peak = max(peak, point.total_equity)
        values.append(point.total_equity / peak - 1 if peak else 0)
    figure = go.Figure(
        go.Scatter(
            x=[point.timestamp for point in result.equity_curve],
            y=values,
            mode="lines",
            line={"color": RED, "width": 1.8},
            fill="tozeroy",
            fillcolor="rgba(255,107,122,0.12)",
            hovertemplate="%{y:.2%}<extra>Drawdown</extra>",
        )
    )
    figure.update_yaxes(tickformat=".0%")
    return _base_layout(figure, height=245, title="Drawdown profile")


def allocation_figure(result: BacktestResult) -> go.Figure:
    symbols = sorted(
        {symbol for portfolio in result.portfolio_history for symbol in portfolio.positions}
    )
    figure = go.Figure()
    colors = [CYAN, AMBER, "#9A8CFF", "#78B6FF", GREEN]
    for index, symbol in enumerate(symbols):
        figure.add_trace(
            go.Scatter(
                x=[portfolio.as_of for portfolio in result.portfolio_history],
                y=[portfolio.weight_for(symbol) for portfolio in result.portfolio_history],
                mode="lines",
                stackgroup="positions",
                name=symbol,
                line={"width": 0.8, "color": colors[index % len(colors)]},
                hovertemplate="%{y:.1%}<extra>" + symbol + "</extra>",
            )
        )
    cash_weights = [
        portfolio.cash / portfolio.total_equity if portfolio.total_equity else 0
        for portfolio in result.portfolio_history
    ]
    figure.add_trace(
        go.Scatter(
            x=[portfolio.as_of for portfolio in result.portfolio_history],
            y=cash_weights,
            mode="lines",
            stackgroup="positions",
            name="CASH",
            line={"width": 0.8, "color": MUTED},
            hovertemplate="%{y:.1%}<extra>Cash</extra>",
        )
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    return _base_layout(figure, height=300, title="Capital allocation")


def candlestick_figure(
    market: MarketSnapshot,
    *,
    fill_time=None,
    fill_price: float | None = None,
) -> go.Figure:
    bars = market.bars
    figure = go.Figure(
        go.Candlestick(
            x=[bar.timestamp for bar in bars],
            open=[bar.open for bar in bars],
            high=[bar.high for bar in bars],
            low=[bar.low for bar in bars],
            close=[bar.close for bar in bars],
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            name=market.symbol,
        )
    )
    if fill_time is not None and fill_price is not None:
        figure.add_trace(
            go.Scatter(
                x=[fill_time],
                y=[fill_price],
                mode="markers",
                name="Fill",
                marker={"color": AMBER, "size": 12, "symbol": "diamond"},
            )
        )
    figure.update_layout(xaxis_rangeslider_visible=False)
    return _base_layout(
        figure, height=360, title=f"{market.symbol} · information visible at decision time"
    )


def comparison_figure(results: Sequence[tuple[str, BacktestResult]]) -> go.Figure:
    figure = go.Figure()
    colors = [CYAN, AMBER, "#9A8CFF", GREEN]
    for index, (label, result) in enumerate(results):
        if not result.equity_curve:
            continue
        base = result.equity_curve[0].total_equity
        figure.add_trace(
            go.Scatter(
                x=[point.timestamp for point in result.equity_curve],
                y=[point.total_equity / base - 1 for point in result.equity_curve],
                mode="lines",
                name=label,
                line={"color": colors[index % len(colors)], "width": 2.4},
                hovertemplate="%{y:+.2%}<extra>" + label + "</extra>",
            )
        )
    figure.update_yaxes(tickformat="+.0%")
    return _base_layout(figure, height=470, title="Normalized performance comparison")


def price_and_target_figure(result: BacktestResult, symbol: str) -> go.Figure:
    raw_bars = result.metadata.get("market_bars", {}).get(symbol, [])
    targets = [item for item in result.decisions if item.intent.symbol == symbol]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    if raw_bars:
        figure.add_trace(
            go.Scatter(
                x=[item["timestamp"] for item in raw_bars],
                y=[item["close"] for item in raw_bars],
                name=f"{symbol} close",
                line={"color": CYAN, "width": 1.8},
            ),
            secondary_y=False,
        )
    figure.add_trace(
        go.Scatter(
            x=[item.intent.as_of for item in targets],
            y=[item.intent.target_weight for item in targets],
            name="Target weight",
            mode="lines+markers",
            line={"color": AMBER, "shape": "hv", "width": 1.8},
            marker={"size": 6},
        ),
        secondary_y=True,
    )
    figure.update_yaxes(title_text="Price", secondary_y=False)
    figure.update_yaxes(title_text="Target", tickformat=".0%", range=[0, 1], secondary_y=True)
    return _base_layout(figure, height=350, title=f"{symbol} price and Agent target")


__all__ = [
    "allocation_figure",
    "candlestick_figure",
    "comparison_figure",
    "drawdown_figure",
    "equity_figure",
    "price_and_target_figure",
]
