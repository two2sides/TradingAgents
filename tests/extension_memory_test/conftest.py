"""Shared fixtures for extension memory tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingagents.extensions.contracts import (
    DecisionOutcome,
    DecisionRecord,
    ExecutionReport,
    MarketBar,
    MarketSnapshot,
    MemoryQuery,
    PortfolioState,
    Position,
    TradeIntent,
)

NOW = datetime(2026, 7, 21, 16, 0, tzinfo=timezone.utc)
NOW_MINUS_10D = NOW.replace(day=11)


# ── Shared builders ────────────────────────────────────────────────────


def make_market(symbol: str = "AAPL", as_of: datetime | None = None, n_bars: int = 10) -> MarketSnapshot:
    """Build a realistic MarketSnapshot with ascending OHLCV bars."""
    if as_of is None:
        as_of = NOW
    bars = []
    base = 100.0
    for i in range(n_bars):
        t = as_of - (n_bars - 1 - i) * (datetime(2026, 1, 1, 1) - datetime(2026, 1, 1, 0))
        open_p = base + i * 0.5
        close_p = open_p + 1.2
        bars.append(MarketBar(
            timestamp=t,
            open=open_p,
            high=close_p + 1.5,
            low=open_p - 0.8,
            close=close_p,
            volume=1_000_000 + i * 50_000,
        ))
    return MarketSnapshot(symbol=symbol, as_of=as_of, bars=bars)


def make_portfolio(as_of: datetime | None = None) -> PortfolioState:
    """Build a realistic PortfolioState with one held position."""
    if as_of is None:
        as_of = NOW
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
            ),
        },
    )


def make_memory_query(
    symbol: str = "AAPL",
    as_of: datetime | None = None,
    agent_role: str = "portfolio_manager",
) -> MemoryQuery:
    """Build a realistic MemoryQuery for a given agent role."""
    if as_of is None:
        as_of = NOW
    return MemoryQuery(
        symbol=symbol,
        as_of=as_of,
        market=make_market(symbol, as_of),
        portfolio=make_portfolio(as_of),
        limit=5,
        metadata={"agent_role": agent_role},
    )


def make_trade_intent(
    symbol: str = "AAPL",
    decision_id: str = "decision-1",
    rationale: str | None = None,
    confidence: float = 0.7,
    target_weight: float = 0.3,
) -> TradeIntent:
    if rationale is None:
        rationale = (
            "Strong buy on valuation. PE compression combined with 20% revenue "
            "growth creates an attractive entry point. Technical indicators show "
            "MACD bullish crossover with increasing volume. Risk: potential "
            "regulatory headwinds in EU markets could pressure margins."
        )
    return TradeIntent(
        decision_id=decision_id,
        symbol=symbol,
        as_of=NOW,
        target_weight=target_weight,
        confidence=confidence,
        rationale=rationale,
    )


def make_decision_record(
    symbol: str = "AAPL",
    decision_id: str = "decision-1",
    rationale: str | None = None,
) -> DecisionRecord:
    return DecisionRecord(
        intent=make_trade_intent(symbol=symbol, decision_id=decision_id, rationale=rationale),
        portfolio_before=make_portfolio(),
        market_at_decision=make_market(symbol),
    )


def make_outcome(
    holding_period_return: float = 0.05,
    max_adverse_move: float = -0.03,
    portfolio_impact: float = 0.01,
) -> DecisionOutcome:
    return DecisionOutcome(
        observed_at=NOW,
        holding_period_return=holding_period_return,
        max_adverse_move=max_adverse_move,
        portfolio_impact=portfolio_impact,
    )


# ── Dependency checks ──────────────────────────────────────────────────

_chromadb_available: bool | None = None
_embedder_available: bool | None = None


def has_chromadb() -> bool:
    global _chromadb_available
    if _chromadb_available is None:
        try:
            import chromadb  # noqa: F401
            _chromadb_available = True
        except ImportError:
            _chromadb_available = False
    return _chromadb_available


def has_embedder() -> bool:
    global _embedder_available
    if _embedder_available is None:
        try:
            from tradingagents.extensions.memory.embedder import (
                LocalEmbeddingBackend,
            )
            # Try actual instantiation, not just import
            LocalEmbeddingBackend()
            _embedder_available = True
        except Exception:
            _embedder_available = False
    return _embedder_available


@pytest.fixture(scope="session")
def chromadb_store(tmp_path_factory):
    """Session-scoped ChromaDB MemoryStore for integration tests."""
    if not has_chromadb():
        pytest.skip("chromadb not installed")
    from tradingagents.extensions.memory.store import MemoryStore

    db_path = str(tmp_path_factory.mktemp("chromadb_test"))
    return MemoryStore(path=db_path)


@pytest.fixture(scope="session")
def memory_embedder():
    """Session-scoped MemoryEmbedder for integration tests."""
    if not has_embedder():
        pytest.skip("sentence-transformers not installed")
    from tradingagents.extensions.memory.embedder import MemoryEmbedder

    return MemoryEmbedder()
