"""End-to-end tests for deterministic historical replay."""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.extensions.contracts import BacktestRequest, ExecutionConfig, MarketBar
from tradingagents.extensions.paper_trading import (
    DemoMemoryProvider,
    EventCollector,
    HistoricalBacktestRunner,
    HistoricalMarketDataProvider,
    InsufficientMarketBars,
    MovingAverageDecisionProvider,
    validate_backtest_calendar,
)
from tradingagents.extensions.paper_trading.metrics import calculate_metrics

START = datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)


def make_provider() -> HistoricalMarketDataProvider:
    closes = [100, 101, 102, 104, 103, 106, 108, 107, 110, 112]
    bars = [
        MarketBar(
            timestamp=START + timedelta(days=index),
            open=close - 0.5,
            high=close + 1,
            low=close - 1,
            close=close,
            volume=10_000,
        )
        for index, close in enumerate(closes)
    ]
    return HistoricalMarketDataProvider({"AAPL": bars}, source="test-bars")


def make_request() -> BacktestRequest:
    return BacktestRequest(
        symbols=["AAPL"],
        start=START,
        end=START + timedelta(days=9),
        initial_cash=10_000,
        lookback=8,
        decision_interval_bars=2,
        outcome_horizon_bars=2,
        execution=ExecutionConfig(commission_rate=0.001, slippage_rate=0.001),
    )


def test_backtest_builds_auditable_result_and_time_safe_contexts():
    provider = make_provider()
    memory = DemoMemoryProvider()
    observer = EventCollector()

    result = HistoricalBacktestRunner(provider).run(
        make_request(),
        MovingAverageDecisionProvider(),
        memory,
        observer,
    )

    assert len(result.equity_curve) == 10
    assert len(result.portfolio_history) == 10
    assert len(result.decisions) == 5
    assert len(result.executions) == 5
    assert len(result.benchmark_curves["BUY_HOLD:AAPL"]) == 10
    assert result.metrics["final_equity"] == result.equity_curve[-1].total_equity
    assert result.metadata["run_kind"] == "FULL"
    assert result.metadata["market_data_source"] == "test-bars"
    assert any(entry["event_type"] == "FILL" for entry in result.metadata["ledger_entries"])
    assert observer.events[0].stage == "PREPARING"
    assert observer.events[-1].stage == "COMPLETED"
    assert observer.events[-1].progress == 1
    assert any(event.stage == "DECISION_STARTED" for event in observer.events)

    for decision in result.decisions:
        context = result.metadata["decision_contexts"][decision.intent.decision_id]
        assert all(
            datetime.fromisoformat(bar["timestamp"]) <= decision.intent.as_of
            for bar in context["market"]["bars"]
        )
    assert memory.outcomes


def test_backtest_is_deterministic_for_fixed_providers():
    request = make_request()
    first = HistoricalBacktestRunner(make_provider()).run(
        request,
        MovingAverageDecisionProvider(),
        DemoMemoryProvider(),
    )
    second = HistoricalBacktestRunner(make_provider()).run(
        request,
        MovingAverageDecisionProvider(),
        DemoMemoryProvider(),
    )

    assert first == second


def test_backtest_calendar_reports_actual_bars_for_too_short_window():
    request = make_request().model_copy(update={"end": START})

    with pytest.raises(InsufficientMarketBars) as caught:
        validate_backtest_calendar(make_provider(), request)

    assert caught.value.symbols == ("AAPL",)
    assert caught.value.available_bars == (START,)
    assert "found 1" in str(caught.value)
    assert START.date().isoformat() in str(caught.value)


def test_decision_provider_failure_preserves_position_and_run_continues():
    class BrokenDecisionProvider:
        def decide(self, request):
            raise RuntimeError("simulated model outage")

    result = HistoricalBacktestRunner(make_provider()).run(
        make_request(),
        BrokenDecisionProvider(),
        DemoMemoryProvider(),
    )

    assert all(decision.status == "FAILED_SAFE" for decision in result.decisions)
    assert all(execution.status == "NO_ACTION" for execution in result.executions)
    assert result.equity_curve[-1].total_equity == 10_000
    assert any("simulated model outage" in warning for warning in result.warnings)


def test_metrics_report_drawdown_fees_and_turnover():
    result = HistoricalBacktestRunner(make_provider()).run(
        make_request(),
        MovingAverageDecisionProvider(),
        DemoMemoryProvider(),
    )

    metrics = calculate_metrics(result.equity_curve, result.executions)

    assert metrics["total_fees"] > 0
    assert metrics["turnover"] > 0
    assert metrics["max_drawdown"] <= 0
    assert metrics["fill_count"] >= 1
    assert metrics["total_return"] == pytest.approx(
        result.equity_curve[-1].total_equity / result.equity_curve[0].total_equity - 1
    )
