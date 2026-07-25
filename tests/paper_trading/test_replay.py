"""Execution-only what-if tests."""

from datetime import datetime, timedelta, timezone

from tradingagents.extensions.contracts import BacktestRequest, ExecutionConfig, MarketBar
from tradingagents.extensions.paper_trading import (
    DemoMemoryProvider,
    HistoricalBacktestRunner,
    HistoricalMarketDataProvider,
    MovingAverageDecisionProvider,
    build_decision_replay,
    run_execution_what_if,
)

START = datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)


def make_original():
    bars = [
        MarketBar(
            timestamp=START + timedelta(days=index),
            open=100 + index,
            high=102 + index,
            low=99 + index,
            close=101 + index,
            volume=10_000,
        )
        for index in range(10)
    ]
    provider = HistoricalMarketDataProvider({"AAPL": bars})
    request = BacktestRequest(
        symbols=["AAPL"],
        start=START,
        end=START + timedelta(days=9),
        initial_cash=10_000,
        lookback=8,
        decision_interval_bars=2,
        execution=ExecutionConfig(commission_rate=0.001, slippage_rate=0),
    )
    result = HistoricalBacktestRunner(provider).run(
        request,
        MovingAverageDecisionProvider(),
        DemoMemoryProvider(),
    )
    return request, result


def test_execution_what_if_reuses_targets_and_changes_only_execution_assumptions():
    original_request, original_result = make_original()

    request, result = run_execution_what_if(
        original_request,
        original_result,
        execution=ExecutionConfig(commission_rate=0.02, slippage_rate=0.01),
        parent_run_id="parent-1",
    )

    assert [item.intent.target_weight for item in result.decisions] == [
        item.intent.target_weight for item in original_result.decisions
    ]
    assert all(item.intent.decision_id.startswith("what-if:") for item in result.decisions)
    assert request.execution.commission_rate == 0.02
    assert result.metrics["total_fees"] > original_result.metrics["total_fees"]
    assert result.metadata["run_kind"] == "EXECUTION_WHAT_IF"
    assert result.metadata["parent_run_id"] == "parent-1"
    assert result.metadata["agent_calls_reused"] == len(original_result.decisions)


def test_replay_view_model_joins_decision_context_execution_and_ledger():
    _, result = make_original()

    items = build_decision_replay(result)

    assert len(items) == len(result.decisions)
    assert all(item.market is not None for item in items)
    assert all(item.memory is not None for item in items)
    assert all(item.portfolio_before is not None for item in items)
    assert all(item.execution is not None for item in items)
    assert any(item.ledger_entries for item in items)
