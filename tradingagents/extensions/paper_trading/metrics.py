"""Deterministic performance metrics and buy-and-hold baselines."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from statistics import fmean, stdev

from tradingagents.extensions.contracts import EquityPoint, ExecutionReport

from .market_data import HistoricalMarketDataProvider

TRADING_DAYS_PER_YEAR = 252


def calculate_metrics(
    equity_curve: Sequence[EquityPoint],
    executions: Sequence[ExecutionReport],
) -> dict[str, float]:
    """Calculate stable, UI-ready metrics without optional dependencies."""

    if not equity_curve:
        return {}
    ordered = sorted(equity_curve, key=lambda point: point.timestamp)
    equities = [point.total_equity for point in ordered]
    initial, final = equities[0], equities[-1]
    total_return = final / initial - 1 if initial else 0.0
    elapsed_days = max(0.0, (ordered[-1].timestamp - ordered[0].timestamp).total_seconds() / 86400)
    if elapsed_days > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (365.25 / elapsed_days) - 1
    else:
        annualized_return = total_return

    returns = [
        current / previous - 1
        for previous, current in zip(equities, equities[1:], strict=False)
        if previous
    ]
    volatility = stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR) if len(returns) > 1 else 0
    sharpe = (
        fmean(returns) / stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if len(returns) > 1 and stdev(returns) > 0
        else 0
    )
    downside = [min(value, 0) for value in returns]
    downside_deviation = (
        math.sqrt(fmean(value * value for value in downside)) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if downside
        else 0
    )
    sortino = (
        fmean(returns) * TRADING_DAYS_PER_YEAR / downside_deviation if downside_deviation > 0 else 0
    )

    peak = equities[0]
    max_drawdown = 0.0
    for equity in equities:
        peak = max(peak, equity)
        drawdown = equity / peak - 1 if peak else 0
        max_drawdown = min(max_drawdown, drawdown)
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

    fills = [fill for execution in executions for fill in execution.fills]
    traded_notional = sum(fill.quantity * fill.price for fill in fills)
    average_equity = fmean(equities)
    total_fees = sum(execution.fees for execution in executions)
    return {
        "initial_equity": float(initial),
        "final_equity": float(final),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "turnover": float(traded_notional / average_equity) if average_equity else 0,
        "total_fees": float(total_fees),
        "execution_count": float(len(executions)),
        "fill_count": float(len(fills)),
        "rejection_count": float(sum(item.status == "REJECTED" for item in executions)),
    }


def build_buy_and_hold_curves(
    provider: HistoricalMarketDataProvider,
    symbols: Sequence[str],
    calendar: Sequence[datetime],
    initial_cash: float,
) -> dict[str, list[EquityPoint]]:
    """Build frictionless per-symbol and equal-weight buy-and-hold curves."""

    if not calendar:
        return {}
    first_prices = provider.close_prices(symbols, calendar[0])
    per_symbol: dict[str, list[EquityPoint]] = {}
    ratios_by_time: list[list[float]] = []

    for timestamp in calendar:
        prices = provider.close_prices(symbols, timestamp)
        ratios = [prices[symbol] / first_prices[symbol] for symbol in symbols]
        ratios_by_time.append(ratios)
        for symbol, ratio in zip(symbols, ratios, strict=True):
            per_symbol.setdefault(f"BUY_HOLD:{symbol}", []).append(
                EquityPoint(timestamp=timestamp, cash=0, total_equity=initial_cash * ratio)
            )

    if len(symbols) > 1:
        per_symbol["EQUAL_WEIGHT_BUY_HOLD"] = [
            EquityPoint(
                timestamp=timestamp,
                cash=0,
                total_equity=initial_cash * fmean(ratios),
            )
            for timestamp, ratios in zip(calendar, ratios_by_time, strict=True)
        ]
    return per_symbol


__all__ = ["build_buy_and_hold_curves", "calculate_metrics"]
