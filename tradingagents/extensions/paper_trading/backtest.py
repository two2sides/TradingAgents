"""Historical event loop that composes A, B, and C through public protocols."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tradingagents.extensions.contracts import (
    BacktestRequest,
    BacktestResult,
    DecisionEnvelope,
    DecisionOutcome,
    DecisionRecord,
    DecisionRequest,
    EquityPoint,
    ExecutionQuote,
    ExecutionReport,
    MarketSnapshot,
    MemoryContext,
    MemoryQuery,
    MemoryReference,
    PortfolioState,
    RunEvent,
    TraceEvent,
    TradeIntent,
)
from tradingagents.extensions.protocols import DecisionProvider, MemoryProvider, RunObserver

from .broker import LedgerBroker
from .market_data import HistoricalMarketDataProvider, MarketDataUnavailable, as_utc
from .metrics import build_buy_and_hold_curves, calculate_metrics

logger = logging.getLogger(__name__)


class InsufficientMarketBars(ValueError):
    """Raised when NEXT_OPEN replay has fewer than two common market bars."""

    def __init__(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        available_bars: list[datetime],
    ) -> None:
        self.symbols = tuple(symbols)
        self.start = start
        self.end = end
        self.available_bars = tuple(available_bars)
        available = (
            ", ".join(timestamp.date().isoformat() for timestamp in available_bars)
            if available_bars
            else "none"
        )
        super().__init__(
            "NEXT_OPEN backtest needs at least two common market bars; "
            f"found {len(available_bars)} for {', '.join(symbols)} between "
            f"{start.date().isoformat()} and {end.date().isoformat()} "
            f"(available: {available})"
        )


def validate_backtest_calendar(
    market_data: HistoricalMarketDataProvider,
    request: BacktestRequest,
) -> list[datetime]:
    """Return the synchronized replay calendar or a detailed validation error."""

    calendar = market_data.common_calendar(request.symbols, request.start, request.end)
    if len(calendar) < 2:
        raise InsufficientMarketBars(
            request.symbols,
            request.start,
            request.end,
            calendar,
        )
    return calendar


@dataclass(slots=True)
class _PendingOutcome:
    due_index: int
    reference: MemoryReference
    symbol: str
    decision_at: datetime
    starting_price: float
    target_weight: float


@dataclass(slots=True)
class _PreparedDecision:
    envelope: DecisionEnvelope
    market: MarketSnapshot
    portfolio_before: PortfolioState
    quote: ExecutionQuote | None


class HistoricalBacktestRunner:
    """Deterministic daily-bar runner with next-bar execution."""

    def __init__(self, market_data: HistoricalMarketDataProvider) -> None:
        self.market_data = market_data

    def run(
        self,
        request: BacktestRequest,
        decision_provider: DecisionProvider,
        memory_provider: MemoryProvider,
        observer: RunObserver | None = None,
    ) -> BacktestResult:
        calendar = validate_backtest_calendar(self.market_data, request)

        broker = LedgerBroker(
            initial_cash=request.initial_cash,
            opened_at=calendar[0],
            execution=request.execution,
        )
        warnings: list[str] = []
        decisions: list[DecisionEnvelope] = []
        executions: list[ExecutionReport] = []
        portfolio_history: list[PortfolioState] = []
        equity_curve: list[EquityPoint] = []
        contexts: dict[str, dict[str, Any]] = {}
        pending_outcomes: list[_PendingOutcome] = []
        seen_decision_ids: set[str] = set()
        decision_indices = set(range(0, len(calendar) - 1, request.decision_interval_bars))

        self._emit(observer, calendar[0], "PREPARING", "Historical replay prepared", 0)
        for index, timestamp in enumerate(calendar):
            self._resolve_outcomes(
                index=index,
                timestamp=timestamp,
                pending=pending_outcomes,
                memory_provider=memory_provider,
                warnings=warnings,
                observer=observer,
            )
            prices = self.market_data.close_prices(request.symbols, timestamp)
            broker.mark_to_market(timestamp, prices)
            self._emit(
                observer,
                timestamp,
                "MARK_TO_MARKET",
                f"Valued portfolio on {timestamp.date().isoformat()}",
                index / max(1, len(calendar) - 1),
                {"prices": prices},
            )

            if index in decision_indices:
                prepared = self._prepare_decisions(
                    request=request,
                    timestamp=timestamp,
                    execution_at=calendar[index + 1],
                    broker=broker,
                    decision_provider=decision_provider,
                    memory_provider=memory_provider,
                    warnings=warnings,
                    seen_decision_ids=seen_decision_ids,
                    contexts=contexts,
                    observer=observer,
                )
                prepared.sort(key=self._execution_order)
                for item in prepared:
                    execution = self._execute(item, broker, warnings, observer)
                    executions.append(execution)
                    decisions.append(item.envelope)
                    reference = self._record_decision(
                        memory_provider=memory_provider,
                        envelope=item.envelope,
                        market=item.market,
                        portfolio_before=item.portfolio_before,
                        execution=execution,
                        warnings=warnings,
                    )
                    due_index = index + request.outcome_horizon_bars
                    if reference is not None and due_index < len(calendar):
                        pending_outcomes.append(
                            _PendingOutcome(
                                due_index=due_index,
                                reference=reference,
                                symbol=item.envelope.intent.symbol,
                                decision_at=timestamp,
                                starting_price=item.market.bars[-1].close,
                                target_weight=item.envelope.intent.target_weight,
                            )
                        )

            portfolio = broker.get_portfolio(timestamp)
            portfolio_history.append(portfolio)
            equity_curve.append(
                EquityPoint(
                    timestamp=timestamp,
                    cash=portfolio.cash,
                    total_equity=portfolio.total_equity,
                )
            )

        benchmarks = build_buy_and_hold_curves(
            self.market_data,
            request.symbols,
            calendar,
            request.initial_cash,
        )
        metrics = calculate_metrics(equity_curve, executions)
        for name, curve in benchmarks.items():
            benchmark_metrics = calculate_metrics(curve, [])
            metrics[f"alpha_vs_{name}"] = (
                metrics["total_return"] - benchmark_metrics["total_return"]
            )

        self._emit(observer, calendar[-1], "COMPLETED", "Historical replay completed", 1)
        return BacktestResult(
            decisions=decisions,
            executions=executions,
            equity_curve=equity_curve,
            portfolio_history=portfolio_history,
            benchmark_curves=benchmarks,
            metrics=metrics,
            warnings=warnings,
            metadata={
                "run_kind": "FULL",
                "market_data_source": self.market_data.source,
                "calendar": [timestamp.isoformat() for timestamp in calendar],
                "decision_contexts": contexts,
                "ledger_entries": [entry.to_dict() for entry in broker.ledger.entries],
                "market_bars": self.market_data.export_bars(
                    request.symbols, calendar[0], calendar[-1]
                ),
            },
        )

    def _prepare_decisions(
        self,
        *,
        request: BacktestRequest,
        timestamp: datetime,
        execution_at: datetime,
        broker: LedgerBroker,
        decision_provider: DecisionProvider,
        memory_provider: MemoryProvider,
        warnings: list[str],
        seen_decision_ids: set[str],
        contexts: dict[str, dict[str, Any]],
        observer: RunObserver | None,
    ) -> list[_PreparedDecision]:
        portfolio = broker.get_portfolio(timestamp)
        prepared: list[_PreparedDecision] = []
        for symbol in request.symbols:
            market = self.market_data.get_snapshot(symbol, timestamp, request.lookback)
            memory = self._retrieve_memory(
                symbol=symbol,
                timestamp=timestamp,
                market=market,
                portfolio=portfolio,
                memory_provider=memory_provider,
                warnings=warnings,
            )
            decision_request = DecisionRequest(
                symbol=symbol,
                as_of=timestamp,
                market=market,
                portfolio=portfolio,
                memory=memory,
                metadata={
                    "backtest": True,
                    "universe_size": len(request.symbols),
                    "universe_symbols": list(request.symbols),
                },
            )
            envelope = self._decide_safely(
                request=decision_request,
                provider=decision_provider,
                seen_decision_ids=seen_decision_ids,
                warnings=warnings,
            )
            seen_decision_ids.add(envelope.intent.decision_id)
            contexts[envelope.intent.decision_id] = {
                "market": market.model_dump(mode="json"),
                "portfolio_before": portfolio.model_dump(mode="json"),
                "memory": memory.model_dump(mode="json"),
            }
            try:
                quote = self.market_data.get_quote_at(symbol, execution_at)
            except MarketDataUnavailable as exc:
                quote = None
                self._warn(warnings, f"{symbol} {timestamp.date()}: {exc}")
            prepared.append(
                _PreparedDecision(
                    envelope=envelope,
                    market=market,
                    portfolio_before=portfolio,
                    quote=quote,
                )
            )
            self._emit(
                observer,
                timestamp,
                "DECISION",
                f"Decision ready for {symbol}",
                None,
                {
                    "symbol": symbol,
                    "decision_id": envelope.intent.decision_id,
                    "status": envelope.status,
                    "target_weight": envelope.intent.target_weight,
                },
            )
        return prepared

    def _retrieve_memory(
        self,
        *,
        symbol: str,
        timestamp: datetime,
        market: MarketSnapshot,
        portfolio: PortfolioState,
        memory_provider: MemoryProvider,
        warnings: list[str],
    ) -> MemoryContext:
        try:
            memory = memory_provider.retrieve(
                MemoryQuery(
                    symbol=symbol,
                    as_of=timestamp,
                    market=market,
                    portfolio=portfolio,
                )
            )
            if as_utc(memory.as_of) > timestamp:
                raise ValueError("memory provider returned a future as_of")
            return memory
        except Exception as exc:
            message = f"{symbol} {timestamp.date()}: memory degraded: {exc}"
            self._warn(warnings, message)
            return MemoryContext(as_of=timestamp, warnings=[message])

    def _decide_safely(
        self,
        *,
        request: DecisionRequest,
        provider: DecisionProvider,
        seen_decision_ids: set[str],
        warnings: list[str],
    ) -> DecisionEnvelope:
        current_weight = request.portfolio.weight_for(request.symbol)
        try:
            envelope = provider.decide(request)
            intent = envelope.intent
            if intent.symbol != request.symbol or as_utc(intent.as_of) != request.as_of:
                raise ValueError("decision intent does not match request symbol/as_of")
            if intent.decision_id in seen_decision_ids:
                raise ValueError(f"duplicate decision_id {intent.decision_id}")
            if envelope.status == "FAILED_SAFE" and intent.target_weight != current_weight:
                raise ValueError("FAILED_SAFE decision must preserve current weight")
            return envelope
        except Exception as exc:
            message = f"{request.symbol} {request.as_of.date()}: decision failed safe: {exc}"
            self._warn(warnings, message)
            return DecisionEnvelope(
                intent=TradeIntent(
                    decision_id=f"failed-safe-{request.symbol}-{request.as_of.isoformat()}",
                    symbol=request.symbol,
                    as_of=request.as_of,
                    target_weight=current_weight,
                    confidence=0,
                    rationale="Decision provider failed; preserve the current position.",
                    warnings=[message],
                ),
                status="FAILED_SAFE",
                trace=[
                    TraceEvent(
                        timestamp=request.as_of,
                        source="HistoricalBacktestRunner",
                        event_type="DECISION_FAILED_SAFE",
                        summary=message,
                    )
                ],
            )

    def _execute(
        self,
        item: _PreparedDecision,
        broker: LedgerBroker,
        warnings: list[str],
        observer: RunObserver | None,
    ) -> ExecutionReport:
        intent = item.envelope.intent
        if item.quote is None:
            reason = "no execution quote available after decision"
            return ExecutionReport(
                decision_id=intent.decision_id,
                status="REJECTED",
                requested_target_weight=intent.target_weight,
                achieved_weight=item.portfolio_before.weight_for(intent.symbol),
                rejection_reason=reason,
            )
        try:
            report = broker.rebalance(intent, item.quote)
        except Exception as exc:
            reason = f"broker rejected execution with internal error: {exc}"
            self._warn(warnings, f"{intent.symbol} {intent.as_of.date()}: {reason}")
            report = ExecutionReport(
                decision_id=intent.decision_id,
                status="REJECTED",
                requested_target_weight=intent.target_weight,
                achieved_weight=item.portfolio_before.weight_for(intent.symbol),
                rejection_reason=reason,
            )
        self._emit(
            observer,
            item.quote.timestamp,
            "EXECUTION",
            f"Execution {report.status} for {intent.symbol}",
            None,
            {
                "symbol": intent.symbol,
                "decision_id": intent.decision_id,
                "status": report.status,
                "fees": report.fees,
            },
        )
        return report

    def _record_decision(
        self,
        *,
        memory_provider: MemoryProvider,
        envelope: DecisionEnvelope,
        market: MarketSnapshot,
        portfolio_before: PortfolioState,
        execution: ExecutionReport,
        warnings: list[str],
    ) -> MemoryReference | None:
        try:
            return memory_provider.record_decision(
                DecisionRecord(
                    intent=envelope.intent,
                    portfolio_before=portfolio_before,
                    market_at_decision=market,
                    execution=execution,
                )
            )
        except Exception as exc:
            self._warn(warnings, f"could not persist decision {envelope.intent.decision_id}: {exc}")
            return None

    def _resolve_outcomes(
        self,
        *,
        index: int,
        timestamp: datetime,
        pending: list[_PendingOutcome],
        memory_provider: MemoryProvider,
        warnings: list[str],
        observer: RunObserver | None,
    ) -> None:
        ready = [item for item in pending if item.due_index == index]
        for item in ready:
            bars = self.market_data.bars_between(item.symbol, item.decision_at, timestamp)
            if not bars:
                continue
            realized = bars[-1].close / item.starting_price - 1
            adverse = min([bar.low / item.starting_price - 1 for bar in bars] + [0.0])
            try:
                memory_provider.record_outcome(
                    item.reference,
                    DecisionOutcome(
                        observed_at=timestamp,
                        holding_period_return=realized,
                        max_adverse_move=adverse,
                        portfolio_impact=item.target_weight * realized,
                    ),
                )
                self._emit(
                    observer,
                    timestamp,
                    "OUTCOME",
                    f"Outcome recorded for {item.symbol}",
                    None,
                    {"holding_period_return": realized},
                )
            except Exception as exc:
                self._warn(warnings, f"could not persist outcome {item.reference.memory_id}: {exc}")
            pending.remove(item)

    @staticmethod
    def _execution_order(item: _PreparedDecision) -> tuple:
        quote_missing = item.quote is None
        quote_time = item.quote.timestamp if item.quote is not None else datetime.min
        current = item.portfolio_before.weight_for(item.envelope.intent.symbol)
        sell_first = 0 if item.envelope.intent.target_weight < current else 1
        return quote_missing, quote_time, sell_first, item.envelope.intent.symbol

    @staticmethod
    def _warn(warnings: list[str], message: str) -> None:
        if message not in warnings:
            warnings.append(message)
            logger.warning(message)

    @staticmethod
    def _emit(
        observer: RunObserver | None,
        timestamp: datetime,
        stage: str,
        message: str,
        progress: float | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if observer is None:
            return
        try:
            observer.on_event(
                RunEvent(
                    timestamp=timestamp,
                    stage=stage,
                    message=message,
                    progress=progress,
                    payload=payload or {},
                )
            )
        except Exception:
            # A display or logging callback must never invalidate a backtest.
            logger.exception("Run observer failed stage=%s message=%s", stage, message)
            return


__all__ = [
    "HistoricalBacktestRunner",
    "InsufficientMarketBars",
    "validate_backtest_calendar",
]
