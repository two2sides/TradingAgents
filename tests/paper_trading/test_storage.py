"""SQLite run archive tests."""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.extensions.contracts import (
    BacktestRequest,
    BacktestResult,
    EquityPoint,
    MarketBar,
    RunEvent,
)
from tradingagents.extensions.paper_trading import (
    BacktestApplicationService,
    DemoMemoryProvider,
    HistoricalMarketDataProvider,
    MovingAverageDecisionProvider,
)
from tradingagents.extensions.paper_trading.storage import SQLiteRunStore

NOW = datetime(2024, 1, 10, tzinfo=timezone.utc)


def make_request() -> BacktestRequest:
    return BacktestRequest(
        symbols=["AAPL"],
        start=NOW - timedelta(days=5),
        end=NOW,
        initial_cash=100_000,
    )


def make_result() -> BacktestResult:
    return BacktestResult(
        equity_curve=[EquityPoint(timestamp=NOW, cash=10_000, total_equity=105_000)],
        metrics={"total_return": 0.05, "max_drawdown": -0.02},
        metadata={"ledger_entries": [{"event_type": "DEPOSIT"}]},
    )


def test_store_round_trips_completed_run_events_and_metadata(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    event = RunEvent(
        timestamp=NOW,
        stage="COMPLETED",
        message="done",
        progress=1,
    )

    run_id = store.save_completed(
        make_request(),
        make_result(),
        events=[event],
        label="AAPL demo",
    )
    restored = store.get_run(run_id)

    assert restored.status == "COMPLETED"
    assert restored.label == "AAPL demo"
    assert restored.request == make_request()
    assert restored.result == make_result()
    assert restored.events == (event,)
    assert restored.result.metadata["ledger_entries"][0]["event_type"] == "DEPOSIT"


def test_store_lists_exports_fails_and_deletes_runs(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    completed_id = store.save_completed(make_request(), make_result())
    failed_id = store.create_run(make_request(), label="failed demo")
    store.fail_run(failed_id, "simulated failure")

    summaries = store.list_runs()
    exported = store.export_run(completed_id)

    assert {summary.run_id for summary in summaries} == {completed_id, failed_id}
    completed = next(summary for summary in summaries if summary.run_id == completed_id)
    assert completed.total_return == pytest.approx(0.05)
    assert exported["schema_version"] == 1
    assert exported["result"]["metrics"]["max_drawdown"] == pytest.approx(-0.02)
    assert store.get_run(failed_id).error == "simulated failure"

    store.delete_run(completed_id)
    with pytest.raises(KeyError, match="unknown run"):
        store.get_run(completed_id)


def test_application_service_persists_success_and_failure(tmp_path):
    bars = [
        MarketBar(
            timestamp=NOW - timedelta(days=5 - index),
            open=100 + index,
            high=102 + index,
            low=99 + index,
            close=101 + index,
            volume=1_000,
        )
        for index in range(6)
    ]
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")
    service = BacktestApplicationService(
        HistoricalMarketDataProvider({"AAPL": bars}),
        store,
    )

    stored = service.run_and_store(
        make_request(),
        MovingAverageDecisionProvider(),
        DemoMemoryProvider(),
    )

    assert stored.status == "COMPLETED"
    assert stored.result is not None
    assert stored.events[0].stage == "PREPARING"
    assert stored.events[-1].stage == "COMPLETED"

    invalid_window = make_request().model_copy(
        update={
            "start": NOW + timedelta(days=10),
            "end": NOW + timedelta(days=12),
        }
    )
    with pytest.raises(ValueError, match="at least two"):
        service.run_and_store(
            invalid_window,
            MovingAverageDecisionProvider(),
            DemoMemoryProvider(),
        )
    failed = next(summary for summary in store.list_runs() if summary.status == "FAILED")
    assert "at least two" in store.get_run(failed.run_id).error
