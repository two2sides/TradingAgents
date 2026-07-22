"""Target-weight simulated broker backed by an append-only ledger."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import ROUND_FLOOR, Decimal

from tradingagents.extensions.contracts import (
    ExecutionConfig,
    ExecutionQuote,
    ExecutionReport,
    Fill,
    PortfolioState,
    TradeIntent,
)

from .ledger import ZERO, AccountLedger, as_decimal


class LedgerBroker:
    """Long-only Broker that rebalances an account to an Agent target weight."""

    def __init__(
        self,
        *,
        initial_cash: float,
        opened_at: datetime,
        execution: ExecutionConfig | None = None,
    ) -> None:
        self.execution = execution or ExecutionConfig()
        self.ledger = AccountLedger(initial_cash=initial_cash, opened_at=opened_at)

    def mark_to_market(self, timestamp: datetime, prices: Mapping[str, float]) -> PortfolioState:
        """Append observable marks and return the resulting account state."""

        for symbol in sorted(prices):
            self.ledger.record_mark(timestamp, symbol, prices[symbol])
        return self.get_portfolio(timestamp)

    def get_portfolio(self, as_of: datetime) -> PortfolioState:
        return self.ledger.snapshot(as_of)

    def rebalance(self, intent: TradeIntent, quote: ExecutionQuote) -> ExecutionReport:
        rejection = self._validate_order_context(intent, quote)
        if rejection is not None:
            self.ledger.record_rejection(
                timestamp=quote.timestamp,
                symbol=intent.symbol,
                decision_id=intent.decision_id,
                reason=rejection,
            )
            portfolio = self.get_portfolio(quote.timestamp)
            return ExecutionReport(
                decision_id=intent.decision_id,
                status="REJECTED",
                requested_target_weight=intent.target_weight,
                achieved_weight=portfolio.weight_for(intent.symbol),
                rejection_reason=rejection,
            )

        self.ledger.record_mark(quote.timestamp, quote.symbol, quote.price)
        before = self.get_portfolio(quote.timestamp)
        raw_price = as_decimal(quote.price)
        current = before.positions.get(intent.symbol)
        current_quantity = current.quantity if current is not None else 0

        approximate_side = (
            "BUY" if intent.target_weight > before.weight_for(intent.symbol) else "SELL"
        )
        execution_price = self._slipped_price(raw_price, approximate_side)
        target_value = as_decimal(before.total_equity) * as_decimal(intent.target_weight)
        target_quantity = int(
            (target_value / execution_price).to_integral_value(rounding=ROUND_FLOOR)
        )
        quantity_delta = target_quantity - current_quantity

        if quantity_delta == 0:
            if current_quantity == 0 and intent.target_weight > 0:
                reason = "insufficient cash to buy one share including fees"
                self.ledger.record_rejection(
                    timestamp=quote.timestamp,
                    symbol=intent.symbol,
                    decision_id=intent.decision_id,
                    reason=reason,
                )
                return ExecutionReport(
                    decision_id=intent.decision_id,
                    status="REJECTED",
                    requested_target_weight=intent.target_weight,
                    achieved_weight=0,
                    rejection_reason=reason,
                    metadata={
                        "requested_quantity": 1,
                        "executed_quantity": 0,
                        "raw_quote": quote.price,
                    },
                )
            return ExecutionReport(
                decision_id=intent.decision_id,
                status="NO_ACTION",
                requested_target_weight=intent.target_weight,
                achieved_weight=before.weight_for(intent.symbol),
                metadata={
                    "requested_quantity": 0,
                    "executed_quantity": 0,
                    "raw_quote": quote.price,
                },
            )

        side = "BUY" if quantity_delta > 0 else "SELL"
        execution_price = self._slipped_price(raw_price, side)
        requested_quantity = abs(quantity_delta)
        execution_quantity = requested_quantity
        status = "FILLED"

        if side == "BUY":
            execution_quantity = self._affordable_quantity(
                desired=requested_quantity,
                price=execution_price,
                cash=as_decimal(before.cash),
            )
            if execution_quantity == 0:
                reason = "insufficient cash to buy one share including fees"
                self.ledger.record_rejection(
                    timestamp=quote.timestamp,
                    symbol=intent.symbol,
                    decision_id=intent.decision_id,
                    reason=reason,
                )
                return ExecutionReport(
                    decision_id=intent.decision_id,
                    status="REJECTED",
                    requested_target_weight=intent.target_weight,
                    achieved_weight=before.weight_for(intent.symbol),
                    rejection_reason=reason,
                    metadata={
                        "requested_quantity": requested_quantity,
                        "executed_quantity": 0,
                        "raw_quote": quote.price,
                    },
                )
            if execution_quantity < requested_quantity:
                status = "PARTIAL"

        fee = self._fee(execution_price * execution_quantity)
        if side == "SELL" and as_decimal(before.cash) + execution_price * execution_quantity < fee:
            reason = "available cash and sell proceeds cannot cover fees"
            self.ledger.record_rejection(
                timestamp=quote.timestamp,
                symbol=intent.symbol,
                decision_id=intent.decision_id,
                reason=reason,
            )
            return ExecutionReport(
                decision_id=intent.decision_id,
                status="REJECTED",
                requested_target_weight=intent.target_weight,
                achieved_weight=before.weight_for(intent.symbol),
                rejection_reason=reason,
            )

        fill = Fill(
            symbol=intent.symbol,
            timestamp=quote.timestamp,
            side=side,
            quantity=execution_quantity,
            price=float(execution_price),
            fee=float(fee),
            metadata={
                "raw_quote": quote.price,
                "slippage_rate": self.execution.slippage_rate,
                "execution_policy": self.execution.execution_policy,
            },
        )
        self.ledger.record_fill(
            timestamp=fill.timestamp,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            fee=fill.fee,
            decision_id=intent.decision_id,
            metadata=fill.metadata,
        )
        # Value the account at the observable quote, not at our adverse fill
        # price, so slippage appears immediately as a loss in account equity.
        self.ledger.record_mark(quote.timestamp, quote.symbol, quote.price)
        after = self.get_portfolio(quote.timestamp)
        return ExecutionReport(
            decision_id=intent.decision_id,
            status=status,
            requested_target_weight=intent.target_weight,
            achieved_weight=after.weight_for(intent.symbol),
            fills=[fill],
            fees=fill.fee,
            metadata={
                "requested_quantity": requested_quantity,
                "executed_quantity": execution_quantity,
                "raw_quote": quote.price,
                "execution_price": float(execution_price),
            },
        )

    def _validate_order_context(
        self,
        intent: TradeIntent,
        quote: ExecutionQuote,
    ) -> str | None:
        if intent.symbol != quote.symbol:
            return f"quote symbol {quote.symbol} does not match intent symbol {intent.symbol}"
        try:
            if quote.timestamp <= intent.as_of:
                return "execution quote must be later than the decision timestamp"
        except TypeError:
            return "quote and decision timestamps must use compatible timezone awareness"
        return None

    def _slipped_price(self, price: Decimal, side: str) -> Decimal:
        rate = as_decimal(self.execution.slippage_rate)
        return price * (Decimal("1") + rate if side == "BUY" else Decimal("1") - rate)

    def _fee(self, notional: Decimal) -> Decimal:
        if notional <= ZERO:
            return ZERO
        proportional = notional * as_decimal(self.execution.commission_rate)
        return max(as_decimal(self.execution.minimum_fee), proportional)

    def _affordable_quantity(self, *, desired: int, price: Decimal, cash: Decimal) -> int:
        if desired <= 0 or cash <= ZERO:
            return 0
        candidate = min(desired, int((cash / price).to_integral_value(rounding=ROUND_FLOOR)))
        while candidate > 0 and price * candidate + self._fee(price * candidate) > cash:
            candidate -= 1
        return candidate


__all__ = ["LedgerBroker"]
