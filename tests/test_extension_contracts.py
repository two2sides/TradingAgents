"""Contract tests shared by the three extension workstreams."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tradingagents.extensions.contracts import (
    BacktestRequest,
    DecisionRequest,
    ExecutionReport,
    MarketBar,
    MarketSnapshot,
    MemoryContext,
    MemoryItem,
    PortfolioState,
    Position,
    TradeIntent,
)
from tradingagents.extensions.protocols import DecisionProvider

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=timezone.utc)


def make_market(symbol: str = "AAPL", as_of: datetime = NOW) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        as_of=as_of,
        bars=[
            MarketBar(
                timestamp=as_of - timedelta(days=1),
                open=100,
                high=103,
                low=99,
                close=102,
                volume=1_000_000,
            )
        ],
    )


def make_portfolio(as_of: datetime = NOW) -> PortfolioState:
    return PortfolioState(
        as_of=as_of,
        cash=80_000,
        total_equity=100_000,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                quantity=200,
                average_cost=95,
                market_price=100,
                market_value=20_000,
                weight=0.2,
            )
        },
    )


def test_contracts_round_trip_through_json():
    intent = TradeIntent(
        decision_id="decision-1",
        symbol="aapl",
        as_of=NOW,
        target_weight=0.45,
        confidence=0.7,
        rationale="Public-contract smoke test.",
    )

    restored = TradeIntent.model_validate_json(intent.model_dump_json())

    assert restored == intent
    assert restored.symbol == "AAPL"


def test_market_snapshot_rejects_future_bar():
    with pytest.raises(ValidationError, match="later than as_of"):
        MarketSnapshot(
            symbol="AAPL",
            as_of=NOW,
            bars=[
                MarketBar(
                    timestamp=NOW + timedelta(minutes=1),
                    open=100,
                    high=101,
                    low=99,
                    close=100,
                    volume=10,
                )
            ],
        )


def test_market_snapshot_rejects_unsorted_bars():
    later = MarketBar(
        timestamp=NOW - timedelta(days=1),
        open=100,
        high=101,
        low=99,
        close=100,
        volume=10,
    )
    earlier = later.model_copy(update={"timestamp": NOW - timedelta(days=2)})

    with pytest.raises(ValidationError, match="sorted"):
        MarketSnapshot(symbol="AAPL", as_of=NOW, bars=[later, earlier])


def test_portfolio_reports_zero_weight_for_unheld_symbol():
    portfolio = make_portfolio()

    assert portfolio.weight_for("aapl") == pytest.approx(0.2)
    assert portfolio.weight_for("MSFT") == 0


def test_trade_intent_rejects_out_of_range_target():
    with pytest.raises(ValidationError):
        TradeIntent(
            decision_id="decision-1",
            symbol="AAPL",
            as_of=NOW,
            target_weight=1.1,
            confidence=0.7,
            rationale="Invalid target.",
        )


def test_rejected_execution_requires_reason():
    with pytest.raises(ValidationError, match="rejection_reason"):
        ExecutionReport(
            decision_id="decision-1",
            status="REJECTED",
            requested_target_weight=0.8,
            achieved_weight=0.2,
        )


def test_memory_context_rejects_information_from_the_future():
    item = MemoryItem(
        memory_id="memory-1",
        symbol="AAPL",
        decision_at=NOW - timedelta(days=10),
        available_at=NOW + timedelta(days=1),
        content="This outcome is not available yet.",
    )

    with pytest.raises(ValidationError, match="unavailable at as_of"):
        MemoryContext(as_of=NOW, items=[item])


def test_decision_request_rejects_mismatched_market_symbol():
    with pytest.raises(ValidationError, match="must match market snapshot"):
        DecisionRequest(
            symbol="MSFT",
            as_of=NOW,
            market=make_market("AAPL"),
            portfolio=make_portfolio(),
            memory=MemoryContext(as_of=NOW),
        )


def test_backtest_request_normalizes_symbols_and_rejects_duplicates():
    request = BacktestRequest(
        symbols=["aapl", "msft"],
        start=NOW - timedelta(days=30),
        end=NOW,
        initial_cash=100_000,
    )
    assert request.symbols == ["AAPL", "MSFT"]

    with pytest.raises(ValidationError, match="duplicates"):
        BacktestRequest(
            symbols=["AAPL", "aapl"],
            start=NOW - timedelta(days=30),
            end=NOW,
            initial_cash=100_000,
        )


def test_decision_provider_supports_structural_typing():
    class StubDecisionProvider:
        def decide(self, request):
            raise NotImplementedError

    assert isinstance(StubDecisionProvider(), DecisionProvider)
