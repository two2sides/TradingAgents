"""Tests for the B/C integration boundary owned by paper trading."""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.extensions.contracts import (
    DecisionRequest,
    MarketBar,
    MarketSnapshot,
    MemoryContext,
    MemoryItem,
    MemoryQuery,
    PortfolioState,
    Position,
)
from tradingagents.extensions.paper_trading import (
    RatingAllocationPolicy,
    TradingAgentsGraphDecisionProvider,
)
from tradingagents.extensions.protocols import DecisionProvider

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=timezone.utc)


def make_request(*, current_weight: float = 0.2) -> DecisionRequest:
    market = MarketSnapshot(
        symbol="AAPL",
        as_of=NOW,
        bars=[
            MarketBar(
                timestamp=NOW - timedelta(days=1),
                open=100,
                high=103,
                low=99,
                close=102,
                volume=1_000_000,
            )
        ],
    )
    position_value = 100_000 * current_weight
    portfolio = PortfolioState(
        as_of=NOW,
        cash=100_000 - position_value,
        total_equity=100_000,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                quantity=int(position_value / 100),
                average_cost=95,
                market_price=100,
                market_value=position_value,
                weight=current_weight,
            )
        }
        if current_weight
        else {},
    )
    memory = MemoryContext(
        as_of=NOW,
        items=[
            MemoryItem(
                memory_id="memory-1",
                symbol="AAPL",
                decision_at=NOW - timedelta(days=20),
                available_at=NOW - timedelta(days=10),
                content="A prior breakout failed after weak volume confirmation.",
                score=0.9,
            )
        ],
        summary="One relevant, already-resolved decision.",
    )
    return DecisionRequest(
        symbol="AAPL",
        as_of=NOW,
        market=market,
        portfolio=portfolio,
        memory=memory,
        metadata={"backtest": True, "universe_size": 1},
    )


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        ("Buy", 0.35),
        ("Overweight", 0.2625),
        ("Hold", 0.2),
        ("Underweight", 0.0875),
        ("Sell", 0.0),
    ],
)
def test_rating_policy_has_explicit_directional_bands(rating, expected):
    decision = RatingAllocationPolicy().resolve(
        rating,
        current_weight=0.2,
        universe_size=1,
    )

    assert decision.target_weight == pytest.approx(expected)
    assert decision.diversification_cap == pytest.approx(0.35)


def test_rating_policy_diversifies_entries_and_never_opens_on_underweight():
    policy = RatingAllocationPolicy(max_position_weight=0.8)

    buy = policy.resolve("Buy", current_weight=0, universe_size=4)
    underweight = policy.resolve("Underweight", current_weight=0, universe_size=4)

    assert buy.target_weight == pytest.approx(0.25)
    assert underweight.target_weight == 0


def test_graph_adapter_injects_request_memory_and_restores_graph_state():
    original_provider = object()
    original_log = object()

    class FakeGraph:
        def __init__(self):
            self.memory_provider = original_provider
            self.memory_log = original_log
            self.seen_context = None

        def propagate(self, symbol, trade_date, asset_type="stock"):
            assert self.memory_provider is not original_provider
            assert self.memory_log is not original_log
            assert self.memory_log.get_pending_entries() == []
            self.seen_context = self.memory_provider.retrieve(
                MemoryQuery(
                    symbol=symbol,
                    as_of=NOW,
                    market=MarketSnapshot(symbol=symbol, as_of=NOW),
                    portfolio=PortfolioState(as_of=NOW, cash=0, total_equity=0),
                    metadata={"agent_role": "portfolio_manager"},
                )
            )
            return (
                {
                    "run_id": "graph-run-1",
                    "final_trade_decision": (
                        "**Rating**: Buy\n\n**Executive Summary**: Build exposure in stages."
                    ),
                    "decision_snapshots": [
                        {
                            "stage": "portfolio_manager",
                            "value": "Buy",
                            "parsed": True,
                        }
                    ],
                },
                "Buy",
            )

    graph = FakeGraph()
    provider = TradingAgentsGraphDecisionProvider(graph)

    result = provider.decide(make_request())

    assert isinstance(provider, DecisionProvider)
    assert result.status == "SUCCESS"
    assert result.intent.target_weight == pytest.approx(0.35)
    assert result.intent.metadata["rating"] == "Buy"
    assert result.diagnostics["agent_reports"]["final_decision"].startswith("**Rating**")
    assert graph.seen_context.items[0].memory_id == "memory-1"
    assert graph.memory_provider is original_provider
    assert graph.memory_log is original_log


def test_graph_adapter_fails_safe_when_rating_is_not_explicit():
    class AmbiguousGraph:
        def propagate(self, symbol, trade_date, asset_type="stock"):
            return {"final_trade_decision": "The evidence is mixed."}, "Hold"

    result = TradingAgentsGraphDecisionProvider(AmbiguousGraph()).decide(make_request())

    assert result.status == "FAILED_SAFE"
    assert result.intent.target_weight == pytest.approx(0.2)
    assert result.intent.confidence == 0
    assert "not explicit" in result.intent.warnings[0]


def test_graph_adapter_fails_safe_and_restores_state_after_graph_error():
    original_provider = object()

    class BrokenGraph:
        memory_provider = original_provider

        def propagate(self, symbol, trade_date, asset_type="stock"):
            raise RuntimeError("model unavailable")

    graph = BrokenGraph()
    result = TradingAgentsGraphDecisionProvider(graph).decide(make_request())

    assert result.status == "FAILED_SAFE"
    assert result.intent.target_weight == pytest.approx(0.2)
    assert "model unavailable" in result.intent.warnings[0]
    assert graph.memory_provider is original_provider
