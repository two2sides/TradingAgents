"""Tests for chunker — decision text splitting and tag classification."""

from __future__ import annotations

from tradingagents.extensions.contracts import DecisionRecord, MarketSnapshot, TradeIntent
from tradingagents.extensions.memory.chunker import (
    DecisionChunker,
    _classify_tags,
    _format_market_context,
    _format_portfolio_context,
)
from .conftest import (
    make_decision_record,
    make_market,
    make_outcome,
    make_portfolio,
    make_trade_intent,
    NOW,
)


# ── Chunking ────────────────────────────────────────────────────────────

class TestDecisionChunker:
    def test_split_produces_three_chunk_types(self):
        record = make_decision_record()
        chunks = DecisionChunker.split(record)
        types = {c["type"] for c in chunks}
        assert types >= {"thesis", "market_context", "portfolio_context"}

    def test_thesis_chunk_contains_rationale(self):
        record = make_decision_record(rationale="Buy on earnings growth and PE expansion.")
        chunks = DecisionChunker.split(record)
        thesis = next(c for c in chunks if c["type"] == "thesis")
        assert "earnings growth" in thesis["content"]

    def test_market_context_includes_symbol_and_price(self):
        record = make_decision_record()
        chunks = DecisionChunker.split(record)
        mkt = next(c for c in chunks if c["type"] == "market_context")
        assert "AAPL" in mkt["content"]
        assert "Price:" in mkt["content"]

    def test_portfolio_context_includes_cash(self):
        record = make_decision_record()
        chunks = DecisionChunker.split(record)
        pf = next(c for c in chunks if c["type"] == "portfolio_context")
        assert "Cash:" in pf["content"]

    def test_empty_market_bars_produces_fallback_context(self):
        intent = make_trade_intent()
        record = DecisionRecord(
            intent=intent,
            portfolio_before=make_portfolio(),
            market_at_decision=MarketSnapshot(symbol="AAPL", as_of=NOW),
        )
        chunks = DecisionChunker.split(record)
        mkt = next(c for c in chunks if c["type"] == "market_context")
        assert "No OHLCV data" in mkt["content"]

    def test_thesis_truncated_at_limit(self):
        long_rationale = "buy " * 500  # ~2000 chars
        record = make_decision_record(rationale=long_rationale)
        chunks = DecisionChunker.split(record)
        thesis = next(c for c in chunks if c["type"] == "thesis")
        assert len(thesis["content"]) <= 803  # 800 limit + "..."


# ── Tag classification ─────────────────────────────────────────────────

class TestTagClassification:
    def test_valuation_tags(self):
        record = make_decision_record(
            rationale="PE ratio is low, the stock is undervalued based on DCF valuation."
        )
        tags = _classify_tags(record)
        assert "valuation" in tags

    def test_technical_tags(self):
        record = make_decision_record(
            rationale="MACD crossover with RSI at 35, volume spike confirms the breakout."
        )
        tags = _classify_tags(record)
        assert "technical" in tags or "volume" in tags

    def test_bullish_tags(self):
        record = make_decision_record(
            rationale="Bullish outlook: strong growth catalyst from new product launch."
        )
        tags = _classify_tags(record)
        assert "bull_thesis" in tags

    def test_bearish_tags(self):
        record = make_decision_record(
            rationale="Bearish: significant downside risk from regulatory headwinds."
        )
        tags = _classify_tags(record)
        assert "bear_thesis" in tags

    def test_no_keywords_returns_general(self):
        record = make_decision_record(rationale="Market conditions are mixed.")
        tags = _classify_tags(record)
        assert tags == ["general"]

    def test_tags_no_duplicates_in_result(self):
        # Each tag should appear at most once
        record = make_decision_record(
            rationale="PE PE PE valuation valuation"
        )
        tags = _classify_tags(record)
        assert len(tags) == len(set(tags))


# ── Reflection chunk ────────────────────────────────────────────────────

class TestReflectionChunk:
    def test_build_reflection_with_outcome_data(self):
        record = make_decision_record(rationale="Buy on momentum.")
        outcome = make_outcome(holding_period_return=0.08, max_adverse_move=-0.04)
        chunk = DecisionChunker.build_reflection_chunk(
            record, outcome, "The thesis held — momentum was confirmed."
        )
        assert chunk is not None
        assert chunk["type"] == "reflection"
        assert "+8.00%" in chunk["content"]
        assert "-4.00%" in chunk["content"]

    def test_empty_reflection_returns_none(self):
        record = make_decision_record()
        outcome = make_outcome()
        chunk = DecisionChunker.build_reflection_chunk(record, outcome, "")
        assert chunk is None

    def test_reflection_without_return_data(self):
        from tradingagents.extensions.contracts import DecisionOutcome

        record = make_decision_record()
        outcome = DecisionOutcome(observed_at=NOW)
        chunk = DecisionChunker.build_reflection_chunk(
            record, outcome, "Inconclusive."
        )
        assert chunk is not None
        assert "not yet measured" in chunk["content"]
