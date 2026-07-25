"""Public ports implemented by the three extension workstreams.

The protocols contain no construction or implementation policy.  Callers
should accept these ports through dependency injection so mocks and real
implementations can be exchanged without changing orchestration code.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from .contracts import (
    BacktestRequest,
    BacktestResult,
    DecisionEnvelope,
    DecisionOutcome,
    DecisionRecord,
    DecisionRequest,
    ExecutionQuote,
    ExecutionReport,
    MarketSnapshot,
    MemoryContext,
    MemoryQuery,
    MemoryReference,
    PortfolioState,
    RunEvent,
    TradeIntent,
)


@runtime_checkable
class MarketDataProvider(Protocol):
    """A: expose decision-time data and a later executable quote."""

    def get_snapshot(
        self,
        symbol: str,
        as_of: datetime,
        lookback: int,
    ) -> MarketSnapshot: ...

    def get_execution_quote(
        self,
        symbol: str,
        after: datetime,
    ) -> ExecutionQuote: ...


@runtime_checkable
class Broker(Protocol):
    """A: own portfolio state and execute a final target-weight intent."""

    def get_portfolio(self, as_of: datetime) -> PortfolioState: ...

    def rebalance(
        self,
        intent: TradeIntent,
        quote: ExecutionQuote,
    ) -> ExecutionReport: ...


@runtime_checkable
class MemoryProvider(Protocol):
    """B: retrieve time-safe context and manage the memory lifecycle."""

    def retrieve(self, query: MemoryQuery) -> MemoryContext: ...

    def record_decision(self, record: DecisionRecord) -> MemoryReference: ...

    def record_outcome(
        self,
        reference: MemoryReference,
        outcome: DecisionOutcome,
    ) -> None: ...


@runtime_checkable
class DecisionProvider(Protocol):
    """C: turn the public decision context into one executable intent."""

    def decide(self, request: DecisionRequest) -> DecisionEnvelope: ...


@runtime_checkable
class RunObserver(Protocol):
    """Receive optional progress events without coupling a runner to its UI."""

    def on_event(self, event: RunEvent) -> None: ...


@runtime_checkable
class BacktestRunner(Protocol):
    """A: orchestrate historical time without knowing B or C internals."""

    def run(
        self,
        request: BacktestRequest,
        decision_provider: DecisionProvider,
        memory_provider: MemoryProvider,
        observer: RunObserver | None = None,
    ) -> BacktestResult: ...


__all__ = [
    "BacktestRunner",
    "Broker",
    "DecisionProvider",
    "MarketDataProvider",
    "MemoryProvider",
    "RunObserver",
]
