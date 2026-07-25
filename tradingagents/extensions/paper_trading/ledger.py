"""Append-only account ledger used by the simulated broker.

The ledger is deliberately independent from market-data, agent, and UI code.
Every portfolio state is rebuilt from immutable entries, which makes a run
replayable and gives the WebUI an audit trail for each cash/position change.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Literal

from tradingagents.extensions.contracts import PortfolioState, Position

LedgerEventType = Literal["DEPOSIT", "FILL", "MARK", "REJECTION"]

ZERO = Decimal("0")


def as_decimal(value: Decimal | float | int | str) -> Decimal:
    """Convert external numeric input without inheriting binary-float noise."""

    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


def is_after(left: datetime, right: datetime) -> bool:
    try:
        return left > right
    except TypeError as exc:
        raise ValueError("ledger timestamps must use compatible timezone awareness") from exc


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One immutable fact in an account's history."""

    sequence: int
    timestamp: datetime
    event_type: LedgerEventType
    cash_delta: Decimal = ZERO
    symbol: str | None = None
    quantity_delta: int = 0
    price: Decimal | None = None
    fee: Decimal = ZERO
    decision_id: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("ledger sequence must be positive")
        if self.symbol is not None:
            object.__setattr__(self, "symbol", normalize_symbol(self.symbol))
        object.__setattr__(self, "cash_delta", as_decimal(self.cash_delta))
        object.__setattr__(self, "fee", as_decimal(self.fee))
        if self.price is not None:
            object.__setattr__(self, "price", as_decimal(self.price))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        if self.fee < ZERO:
            raise ValueError("ledger fee must not be negative")
        if self.event_type in {"FILL", "MARK"} and (
            self.symbol is None or self.price is None or self.price <= ZERO
        ):
            raise ValueError(f"{self.event_type} requires a symbol and positive price")
        if self.event_type == "FILL" and self.quantity_delta == 0:
            raise ValueError("FILL requires a non-zero quantity_delta")
        if self.event_type == "MARK" and (
            self.cash_delta != ZERO or self.quantity_delta != 0 or self.fee != ZERO
        ):
            raise ValueError("MARK cannot change cash, quantity, or fees")
        if self.event_type == "DEPOSIT" and self.cash_delta <= ZERO:
            raise ValueError("DEPOSIT requires a positive cash_delta")
        if self.event_type == "REJECTION" and not self.reason:
            raise ValueError("REJECTION requires a reason")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for storage and the WebUI."""

        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "cash_delta": str(self.cash_delta),
            "symbol": self.symbol,
            "quantity_delta": self.quantity_delta,
            "price": str(self.price) if self.price is not None else None,
            "fee": str(self.fee),
            "decision_id": self.decision_id,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class AccountLedger:
    """Append-only long-only account ledger with deterministic replay."""

    def __init__(self, initial_cash: float, opened_at: datetime) -> None:
        cash = as_decimal(initial_cash)
        if cash <= ZERO:
            raise ValueError("initial_cash must be positive")
        self._entries: list[LedgerEntry] = [
            LedgerEntry(
                sequence=1,
                timestamp=opened_at,
                event_type="DEPOSIT",
                cash_delta=cash,
                metadata={"source": "initial_cash"},
            )
        ]

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Expose an immutable view of the audit trail."""

        return tuple(self._entries)

    def _append(self, **values: Any) -> LedgerEntry:
        timestamp = values["timestamp"]
        if is_after(self._entries[-1].timestamp, timestamp):
            raise ValueError("ledger entries must be appended in timestamp order")
        entry = LedgerEntry(sequence=len(self._entries) + 1, **values)
        self._entries.append(entry)
        return entry

    def record_mark(self, timestamp: datetime, symbol: str, price: float) -> LedgerEntry:
        return self._append(
            timestamp=timestamp,
            event_type="MARK",
            symbol=symbol,
            price=as_decimal(price),
        )

    def record_fill(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        price: float,
        fee: float,
        decision_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        if quantity <= 0:
            raise ValueError("fill quantity must be positive")
        normalized = normalize_symbol(symbol)
        fill_price = as_decimal(price)
        fill_fee = as_decimal(fee)
        if fill_price <= ZERO or fill_fee < ZERO:
            raise ValueError("fill price must be positive and fee must not be negative")

        before = self.snapshot(timestamp)
        notional = fill_price * quantity
        if side == "BUY":
            cash_delta = -(notional + fill_fee)
            quantity_delta = quantity
            if as_decimal(before.cash) + cash_delta < ZERO:
                raise ValueError("fill would make cash negative")
        else:
            held = before.positions.get(normalized)
            if held is None or held.quantity < quantity:
                raise ValueError("fill would make position quantity negative")
            cash_delta = notional - fill_fee
            quantity_delta = -quantity
            if as_decimal(before.cash) + cash_delta < ZERO:
                raise ValueError("sell proceeds and cash cannot cover the fee")

        return self._append(
            timestamp=timestamp,
            event_type="FILL",
            cash_delta=cash_delta,
            symbol=normalized,
            quantity_delta=quantity_delta,
            price=fill_price,
            fee=fill_fee,
            decision_id=decision_id,
            metadata=metadata or {},
        )

    def record_rejection(
        self,
        *,
        timestamp: datetime,
        symbol: str,
        decision_id: str,
        reason: str,
    ) -> LedgerEntry:
        return self._append(
            timestamp=timestamp,
            event_type="REJECTION",
            symbol=symbol,
            decision_id=decision_id,
            reason=reason,
        )

    def snapshot(self, as_of: datetime) -> PortfolioState:
        """Rebuild the account using only entries available by ``as_of``."""

        cash = ZERO
        quantities: dict[str, int] = {}
        average_costs: dict[str, Decimal] = {}
        marks: dict[str, Decimal] = {}

        for entry in self._entries:
            if is_after(entry.timestamp, as_of):
                continue
            cash += entry.cash_delta
            if entry.symbol is None or entry.price is None:
                continue

            if entry.event_type == "MARK":
                marks[entry.symbol] = entry.price
                continue
            if entry.event_type != "FILL":
                continue

            symbol = entry.symbol
            old_quantity = quantities.get(symbol, 0)
            new_quantity = old_quantity + entry.quantity_delta
            if new_quantity < 0:
                raise ValueError(f"ledger replay produced a negative position for {symbol}")

            if entry.quantity_delta > 0:
                old_cost = average_costs.get(symbol, ZERO) * old_quantity
                acquired_cost = entry.price * entry.quantity_delta + entry.fee
                average_costs[symbol] = (old_cost + acquired_cost) / new_quantity
            elif new_quantity == 0:
                average_costs.pop(symbol, None)

            if new_quantity:
                quantities[symbol] = new_quantity
            else:
                quantities.pop(symbol, None)
            marks[symbol] = entry.price

        market_values = {
            symbol: as_decimal(marks.get(symbol, average_costs[symbol])) * quantity
            for symbol, quantity in quantities.items()
        }
        total_equity = cash + sum(market_values.values(), ZERO)
        if cash < ZERO or total_equity < ZERO:
            raise ValueError("ledger replay violated non-negative account constraints")

        positions = {
            symbol: Position(
                symbol=symbol,
                quantity=quantity,
                average_cost=float(average_costs[symbol]),
                market_price=float(marks.get(symbol, average_costs[symbol])),
                market_value=float(market_values[symbol]),
                weight=float(market_values[symbol] / total_equity) if total_equity else 0,
            )
            for symbol, quantity in sorted(quantities.items())
        }
        return PortfolioState(
            as_of=as_of,
            cash=float(cash),
            total_equity=float(total_equity),
            positions=positions,
        )


__all__ = ["AccountLedger", "LedgerEntry", "LedgerEventType", "as_decimal"]
