"""Streamlit shell and Plotly view smoke tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from tradingagents.extensions.contracts import BacktestRequest
from tradingagents.extensions.paper_trading import (
    DemoMemoryProvider,
    HistoricalBacktestRunner,
    MovingAverageDecisionProvider,
    SQLiteRunStore,
    generate_demo_market_data,
)
from webui.components.charts import allocation_figure, drawdown_figure, equity_figure


def test_decision_lab_pages_and_builtin_run_render(monkeypatch, tmp_path):
    store_path = tmp_path / "ui-runs.sqlite3"
    monkeypatch.setenv("TRADINGAGENTS_RUN_STORE", str(store_path))

    app = AppTest.from_file(Path("webui/app.py"), default_timeout=20).run()

    assert not app.exception
    assert any("Decision Lab" in item.value for item in app.markdown)

    run_app = AppTest.from_file(Path("webui/views/run.py"), default_timeout=30).run()
    assert not run_app.exception
    launch = next(button for button in run_app.button if button.label == "Launch historical replay")
    launch.click()
    run_app.run(timeout=30)
    assert not run_app.exception
    assert run_app.success

    stored = SQLiteRunStore(store_path).get_run(SQLiteRunStore(store_path).list_runs()[0].run_id)
    assert stored.result is not None

    replay_app = AppTest.from_file(Path("webui/views/replay.py"), default_timeout=30).run()
    assert not replay_app.exception
    assert any("Decision Replay" in item.value for item in replay_app.markdown)

    SQLiteRunStore(store_path).save_completed(
        stored.request,
        stored.result,
        label="Comparison copy",
    )
    compare_app = AppTest.from_file(Path("webui/views/compare.py"), default_timeout=30).run()
    assert not compare_app.exception
    assert any("Compare runs" in item.value for item in compare_app.markdown)


def test_replay_charts_accept_a_complete_backtest_result():
    start = datetime(2024, 1, 2, tzinfo=timezone.utc)
    end = start + timedelta(days=60)
    provider = generate_demo_market_data(["AAPL"], start, end)
    result = HistoricalBacktestRunner(provider).run(
        BacktestRequest(
            symbols=["AAPL"],
            start=start,
            end=end,
            initial_cash=100_000,
            lookback=20,
        ),
        MovingAverageDecisionProvider(),
        DemoMemoryProvider(),
    )

    equity = equity_figure(result)
    allocation = allocation_figure(result)
    drawdown = drawdown_figure(result)

    assert len(equity.data) >= 2
    assert any(trace.name in {"BUY", "SELL"} for trace in equity.data)
    assert allocation.data
    assert drawdown.data[0].fill == "tozeroy"
