"""Shared data contracts for the TradingAgents team extensions.

This module intentionally contains data and validation only.  It must not
depend on a broker, a memory backend, a LangGraph implementation, or a UI.
The three implementation areas communicate exclusively through these models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DecisionStatus = Literal["SUCCESS", "DEGRADED", "FAILED_SAFE"]
ExecutionPolicy = Literal["NEXT_OPEN"]
ExecutionStatus = Literal["FILLED", "PARTIAL", "REJECTED", "NO_ACTION"]
OrderSide = Literal["BUY", "SELL"]


def _clean_symbol(value: str) -> str:
    symbol = value.strip().upper()
    if not symbol:
        raise ValueError("symbol must not be empty")
    return symbol


def _is_after(left: datetime, right: datetime, field_names: str) -> bool:
    """Compare timestamps and turn naive/aware mismatches into a clear error."""
    try:
        return left > right
    except TypeError as exc:
        raise ValueError(f"{field_names} must use compatible timezone awareness") from exc


class ContractModel(BaseModel):
    """Base configuration shared by every public contract."""

    model_config = ConfigDict(extra="forbid")


class MarketBar(ContractModel):
    """One OHLCV bar known to the system at ``timestamp``."""

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_ohlc(self) -> MarketBar:
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be greater than or equal to open, low, and close")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be less than or equal to open, high, and close")
        return self


class MarketSnapshot(ContractModel):
    """Market information that was available no later than ``as_of``."""

    symbol: str
    as_of: datetime
    bars: list[MarketBar] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)

    @model_validator(mode="after")
    def validate_bar_times(self) -> MarketSnapshot:
        previous: datetime | None = None
        for bar in self.bars:
            if _is_after(bar.timestamp, self.as_of, "bar.timestamp and as_of"):
                raise ValueError("market snapshot contains a bar later than as_of")
            if previous is not None and _is_after(previous, bar.timestamp, "bar timestamps"):
                raise ValueError("market bars must be sorted by timestamp ascending")
            previous = bar.timestamp
        return self


class Position(ContractModel):
    """Read-only view of a long-only position."""

    symbol: str
    quantity: int = Field(ge=0)
    average_cost: float = Field(ge=0)
    market_price: float = Field(ge=0)
    market_value: float = Field(ge=0)
    weight: float = Field(ge=0, le=1)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)


class PortfolioState(ContractModel):
    """Read-only account snapshot.  Only a Broker may create the next state."""

    as_of: datetime
    cash: float = Field(ge=0)
    total_equity: float = Field(ge=0)
    positions: dict[str, Position] = Field(default_factory=dict)

    @field_validator("positions")
    @classmethod
    def validate_position_keys(cls, value: dict[str, Position]) -> dict[str, Position]:
        normalized: dict[str, Position] = {}
        for key, position in value.items():
            symbol = _clean_symbol(key)
            if symbol != position.symbol:
                raise ValueError(
                    f"position key {key!r} does not match position symbol {position.symbol!r}"
                )
            normalized[symbol] = position
        return normalized

    def weight_for(self, symbol: str) -> float:
        """Return the current weight for ``symbol``, or zero when not held."""
        position = self.positions.get(_clean_symbol(symbol))
        return position.weight if position is not None else 0.0


class ExecutionQuote(ContractModel):
    """Price offered by the execution environment after a decision."""

    symbol: str
    timestamp: datetime
    price: float = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)


class Fill(ContractModel):
    """One simulated fill produced by the Broker."""

    symbol: str
    timestamp: datetime
    side: OrderSide
    quantity: int = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)


class TradeIntent(ContractModel):
    """C's final instruction to A; target weight is the sole execution source."""

    decision_id: str = Field(min_length=1)
    symbol: str
    as_of: datetime
    target_weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)


class TraceEvent(ContractModel):
    """Implementation-neutral diagnostic event for tests and the WebUI."""

    timestamp: datetime
    source: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ExecutionReport(ContractModel):
    """Observable result of attempting to rebalance to a target weight."""

    decision_id: str = Field(min_length=1)
    status: ExecutionStatus
    requested_target_weight: float = Field(ge=0, le=1)
    achieved_weight: float = Field(ge=0, le=1)
    fills: list[Fill] = Field(default_factory=list)
    fees: float = Field(ge=0, default=0)
    rejection_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_status_details(self) -> ExecutionReport:
        if self.status == "REJECTED" and not self.rejection_reason:
            raise ValueError("a rejected execution must include rejection_reason")
        if self.status == "NO_ACTION" and self.fills:
            raise ValueError("a no-action execution cannot contain fills")
        return self


class MemoryItem(ContractModel):
    """One retrieved experience, available to decisions from ``available_at``."""

    memory_id: str = Field(min_length=1)
    symbol: str
    decision_at: datetime
    available_at: datetime
    content: str
    score: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)


class MemoryQuery(ContractModel):
    """State visible to B when it retrieves prior experience."""

    symbol: str
    as_of: datetime
    market: MarketSnapshot
    portfolio: PortfolioState
    limit: int = Field(default=5, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)

    @model_validator(mode="after")
    def validate_context(self) -> MemoryQuery:
        if self.market.symbol != self.symbol:
            raise ValueError("memory query symbol must match market snapshot symbol")
        if _is_after(self.market.as_of, self.as_of, "market.as_of and query.as_of"):
            raise ValueError("memory query market snapshot is later than query as_of")
        if _is_after(self.portfolio.as_of, self.as_of, "portfolio.as_of and query.as_of"):
            raise ValueError("memory query portfolio is later than query as_of")
        return self


class MemoryContext(ContractModel):
    """B's complete, time-safe response to a memory query."""

    as_of: datetime
    items: list[MemoryItem] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_future_memories(self) -> MemoryContext:
        for item in self.items:
            if _is_after(item.available_at, self.as_of, "memory.available_at and as_of"):
                raise ValueError("memory context contains information unavailable at as_of")
        return self


class MemoryReference(ContractModel):
    """Opaque handle returned by B and used to attach a later outcome."""

    memory_id: str = Field(min_length=1)


class DecisionRecord(ContractModel):
    """Information known when a decision is persisted."""

    intent: TradeIntent
    portfolio_before: PortfolioState
    market_at_decision: MarketSnapshot
    execution: ExecutionReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionOutcome(ContractModel):
    """Outcome learned after the original decision time."""

    observed_at: datetime
    holding_period_return: float | None = None
    max_adverse_move: float | None = None
    portfolio_impact: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionRequest(ContractModel):
    """All public inputs C may use to produce a final intent."""

    symbol: str
    as_of: datetime
    market: MarketSnapshot
    portfolio: PortfolioState
    memory: MemoryContext
    mode: str = "enhanced"
    metadata: dict[str, Any] = Field(default_factory=dict)

    _normalize_symbol = field_validator("symbol")(_clean_symbol)

    @model_validator(mode="after")
    def validate_context(self) -> DecisionRequest:
        if self.market.symbol != self.symbol:
            raise ValueError("decision symbol must match market snapshot symbol")
        for label, timestamp in (
            ("market.as_of", self.market.as_of),
            ("portfolio.as_of", self.portfolio.as_of),
            ("memory.as_of", self.memory.as_of),
        ):
            if _is_after(timestamp, self.as_of, f"{label} and decision.as_of"):
                raise ValueError(f"{label} is later than decision as_of")
        return self


class DecisionEnvelope(ContractModel):
    """C's final result plus optional implementation-neutral diagnostics."""

    intent: TradeIntent
    status: DecisionStatus
    trace: list[TraceEvent] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class EquityPoint(ContractModel):
    """One point on a backtest or paper-account equity curve."""

    timestamp: datetime
    cash: float = Field(ge=0)
    total_equity: float = Field(ge=0)


class ExecutionConfig(ContractModel):
    """Public execution assumptions shared by backtests and the WebUI.

    Commission and slippage rates are decimal fractions.  For example,
    ``0.001`` means 0.1 percent.
    """

    commission_rate: float = Field(default=0.0005, ge=0, le=1)
    slippage_rate: float = Field(default=0.001, ge=0, le=1)
    minimum_fee: float = Field(default=0, ge=0)
    execution_policy: ExecutionPolicy = "NEXT_OPEN"


class BacktestRequest(ContractModel):
    """Implementation-neutral request to run a historical simulation."""

    symbols: list[str] = Field(min_length=1)
    start: datetime
    end: datetime
    initial_cash: float = Field(gt=0)
    lookback: int = Field(default=60, ge=1)
    decision_interval_bars: int = Field(default=1, ge=1)
    outcome_horizon_bars: int = Field(default=5, ge=1)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbols")
    @classmethod
    def normalize_symbols(cls, value: list[str]) -> list[str]:
        symbols = [_clean_symbol(symbol) for symbol in value]
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must not contain duplicates")
        return symbols

    @model_validator(mode="after")
    def validate_window(self) -> BacktestRequest:
        if _is_after(self.start, self.end, "backtest start and end"):
            raise ValueError("backtest start must not be later than end")
        return self


class RunEvent(ContractModel):
    """One implementation-neutral progress event emitted during a run."""

    timestamp: datetime
    stage: str = Field(min_length=1)
    message: str
    progress: float | None = Field(default=None, ge=0, le=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class BacktestResult(ContractModel):
    """Public result consumed by evaluation code and the WebUI."""

    decisions: list[DecisionEnvelope] = Field(default_factory=list)
    executions: list[ExecutionReport] = Field(default_factory=list)
    equity_curve: list[EquityPoint] = Field(default_factory=list)
    portfolio_history: list[PortfolioState] = Field(default_factory=list)
    benchmark_curves: dict[str, list[EquityPoint]] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "BacktestRequest",
    "BacktestResult",
    "ContractModel",
    "DecisionEnvelope",
    "DecisionOutcome",
    "DecisionRecord",
    "DecisionStatus",
    "EquityPoint",
    "ExecutionConfig",
    "ExecutionPolicy",
    "ExecutionQuote",
    "ExecutionReport",
    "ExecutionStatus",
    "Fill",
    "MarketBar",
    "MarketSnapshot",
    "MemoryContext",
    "MemoryItem",
    "MemoryQuery",
    "MemoryReference",
    "OrderSide",
    "PortfolioState",
    "Position",
    "RunEvent",
    "TraceEvent",
    "TradeIntent",
]
