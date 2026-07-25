"""Account-ledger and simulated-broker tests."""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.extensions.contracts import ExecutionConfig, ExecutionQuote, TradeIntent
from tradingagents.extensions.paper_trading import AccountLedger, LedgerBroker

OPENED_AT = datetime(2024, 1, 2, 16, 0, tzinfo=timezone.utc)
DECISION_AT = OPENED_AT + timedelta(days=1)
EXECUTION_AT = DECISION_AT + timedelta(days=1)


def make_intent(target_weight: float, symbol: str = "AAPL") -> TradeIntent:
    return TradeIntent(
        decision_id=f"decision-{target_weight}",
        symbol=symbol,
        as_of=DECISION_AT,
        target_weight=target_weight,
        confidence=0.8,
        rationale="Deterministic broker test.",
    )


def make_quote(price: float = 100, symbol: str = "AAPL") -> ExecutionQuote:
    return ExecutionQuote(symbol=symbol, timestamp=EXECUTION_AT, price=price)


def test_ledger_replays_only_events_available_at_snapshot_time():
    ledger = AccountLedger(initial_cash=10_000, opened_at=OPENED_AT)
    ledger.record_fill(
        timestamp=EXECUTION_AT,
        symbol="AAPL",
        side="BUY",
        quantity=20,
        price=100,
        fee=2,
        decision_id="decision-1",
    )
    ledger.record_mark(EXECUTION_AT + timedelta(days=1), "AAPL", 110)

    before_fill = ledger.snapshot(DECISION_AT)
    after_fill = ledger.snapshot(EXECUTION_AT)
    after_mark = ledger.snapshot(EXECUTION_AT + timedelta(days=1))

    assert before_fill.cash == 10_000
    assert before_fill.positions == {}
    assert after_fill.cash == 7_998
    assert after_fill.positions["AAPL"].quantity == 20
    assert after_fill.positions["AAPL"].average_cost == pytest.approx(100.1)
    assert after_mark.total_equity == pytest.approx(10_198)


def test_ledger_entries_are_exposed_as_an_immutable_tuple():
    ledger = AccountLedger(initial_cash=10_000, opened_at=OPENED_AT)

    assert isinstance(ledger.entries, tuple)
    with pytest.raises(AttributeError):
        ledger.entries.append("not-an-entry")
    with pytest.raises(TypeError):
        ledger.entries[0].metadata["source"] = "changed"


def test_ledger_refuses_fills_that_break_account_constraints():
    ledger = AccountLedger(initial_cash=1_000, opened_at=OPENED_AT)

    with pytest.raises(ValueError, match="cash negative"):
        ledger.record_fill(
            timestamp=EXECUTION_AT,
            symbol="AAPL",
            side="BUY",
            quantity=11,
            price=100,
            fee=0,
            decision_id="too-large",
        )

    with pytest.raises(ValueError, match="quantity negative"):
        ledger.record_fill(
            timestamp=EXECUTION_AT,
            symbol="AAPL",
            side="SELL",
            quantity=1,
            price=100,
            fee=0,
            decision_id="short-sale",
        )


def test_broker_buys_at_slipped_price_and_charges_commission():
    broker = LedgerBroker(
        initial_cash=10_000,
        opened_at=OPENED_AT,
        execution=ExecutionConfig(commission_rate=0.001, slippage_rate=0.01),
    )

    report = broker.rebalance(make_intent(0.5), make_quote(100))
    portfolio = broker.get_portfolio(EXECUTION_AT)

    assert report.status == "FILLED"
    assert report.fills[0].side == "BUY"
    assert report.fills[0].quantity == 49
    assert report.fills[0].price == pytest.approx(101)
    assert report.fees == pytest.approx(4.949)
    assert portfolio.cash == pytest.approx(5_046.051)
    assert portfolio.positions["AAPL"].quantity == 49


def test_broker_partially_fills_when_target_is_unaffordable_after_fees():
    broker = LedgerBroker(
        initial_cash=1_000,
        opened_at=OPENED_AT,
        execution=ExecutionConfig(commission_rate=0, slippage_rate=0, minimum_fee=10),
    )

    report = broker.rebalance(make_intent(1), make_quote(100))

    assert report.status == "PARTIAL"
    assert report.metadata["requested_quantity"] == 10
    assert report.metadata["executed_quantity"] == 9
    assert broker.get_portfolio(EXECUTION_AT).cash == pytest.approx(90)


def test_broker_rejects_buy_when_one_share_is_unaffordable():
    broker = LedgerBroker(
        initial_cash=50,
        opened_at=OPENED_AT,
        execution=ExecutionConfig(commission_rate=0, slippage_rate=0, minimum_fee=1),
    )

    report = broker.rebalance(make_intent(1), make_quote(100))

    assert report.status == "REJECTED"
    assert "insufficient cash" in report.rejection_reason
    assert broker.get_portfolio(EXECUTION_AT).cash == 50
    assert broker.ledger.entries[-1].event_type == "REJECTION"


def test_broker_can_sell_down_to_a_lower_target():
    broker = LedgerBroker(
        initial_cash=10_000,
        opened_at=OPENED_AT,
        execution=ExecutionConfig(commission_rate=0, slippage_rate=0),
    )
    broker.rebalance(make_intent(0.5), make_quote(100))
    later_decision = make_intent(0.2).model_copy(
        update={"decision_id": "sell-down", "as_of": EXECUTION_AT}
    )
    later_quote = make_quote(120).model_copy(update={"timestamp": EXECUTION_AT + timedelta(days=1)})

    report = broker.rebalance(later_decision, later_quote)
    portfolio = broker.get_portfolio(later_quote.timestamp)

    assert report.status == "FILLED"
    assert report.fills[0].side == "SELL"
    assert portfolio.positions["AAPL"].quantity < 50
    assert portfolio.weight_for("AAPL") == pytest.approx(report.achieved_weight)


def test_broker_rejects_a_quote_from_the_decision_time_or_wrong_symbol():
    broker = LedgerBroker(initial_cash=10_000, opened_at=OPENED_AT)
    same_time_quote = make_quote().model_copy(update={"timestamp": DECISION_AT})

    same_time = broker.rebalance(make_intent(0.5), same_time_quote)
    wrong_symbol = broker.rebalance(make_intent(0.5), make_quote(symbol="MSFT"))

    assert same_time.status == "REJECTED"
    assert "later" in same_time.rejection_reason
    assert wrong_symbol.status == "REJECTED"
    assert "does not match" in wrong_symbol.rejection_reason


def test_broker_returns_no_action_when_integer_target_is_already_met():
    broker = LedgerBroker(
        initial_cash=1_000,
        opened_at=OPENED_AT,
        execution=ExecutionConfig(commission_rate=0, slippage_rate=0),
    )
    broker.rebalance(make_intent(0.5), make_quote(100))
    next_intent = make_intent(0.5).model_copy(
        update={"decision_id": "same-target", "as_of": EXECUTION_AT}
    )
    next_quote = make_quote(100).model_copy(update={"timestamp": EXECUTION_AT + timedelta(days=1)})

    report = broker.rebalance(next_intent, next_quote)

    assert report.status == "NO_ACTION"
    assert report.fills == []
