"""Application service used by WebUI and future CLI entry points."""

from __future__ import annotations

from tradingagents.extensions.contracts import BacktestRequest, ExecutionConfig
from tradingagents.extensions.protocols import DecisionProvider, MemoryProvider, RunObserver

from .backtest import HistoricalBacktestRunner
from .market_data import HistoricalMarketDataProvider
from .observers import CompositeRunObserver, EventCollector
from .replay import run_execution_what_if
from .storage import RunStoreObserver, SQLiteRunStore, StoredRun


class BacktestApplicationService:
    """Own run lifecycle and keep persistence policy out of the WebUI."""

    def __init__(
        self,
        market_data: HistoricalMarketDataProvider | None,
        store: SQLiteRunStore,
    ) -> None:
        self.market_data = market_data
        self.store = store

    def run_and_store(
        self,
        request: BacktestRequest,
        decision_provider: DecisionProvider,
        memory_provider: MemoryProvider,
        *,
        label: str | None = None,
        observer: RunObserver | None = None,
    ) -> StoredRun:
        if self.market_data is None:
            raise ValueError("market_data is required for a full backtest")
        run_id = self.store.create_run(request, label=label)
        store_observer = RunStoreObserver(self.store, run_id)
        combined: RunObserver = (
            CompositeRunObserver([store_observer, observer]) if observer else store_observer
        )
        try:
            result = HistoricalBacktestRunner(self.market_data).run(
                request,
                decision_provider,
                memory_provider,
                combined,
            )
            self.store.complete_run(run_id, result)
        except Exception as exc:
            self.store.fail_run(run_id, str(exc))
            raise
        return self.store.get_run(run_id)

    def run_what_if_and_store(
        self,
        parent_run_id: str,
        *,
        execution: ExecutionConfig | None = None,
        initial_cash: float | None = None,
        label: str | None = None,
        observer: RunObserver | None = None,
    ) -> StoredRun:
        parent = self.store.get_run(parent_run_id)
        if parent.result is None:
            raise ValueError("parent run has no completed result")
        collector = EventCollector()
        combined: RunObserver = (
            CompositeRunObserver([collector, observer]) if observer else collector
        )
        request, result = run_execution_what_if(
            parent.request,
            parent.result,
            execution=execution,
            initial_cash=initial_cash,
            parent_run_id=parent_run_id,
            observer=combined,
        )
        run_id = self.store.save_completed(
            request,
            result,
            events=collector.events,
            label=label or f"What-if · {parent.label}",
        )
        return self.store.get_run(run_id)


__all__ = ["BacktestApplicationService"]
