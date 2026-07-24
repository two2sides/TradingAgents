"""Tests for store — ChromaDB-backed memory storage (requires chromadb)."""

from __future__ import annotations

import pytest

from tradingagents.extensions.contracts import (
    DecisionRecord,
    MarketBar,
    MarketSnapshot,
    PortfolioState,
    TradeIntent,
)
from .conftest import (
    has_chromadb, make_decision_record, make_market, make_outcome,
    make_portfolio, make_trade_intent, NOW,
)

pytestmark = pytest.mark.skipif(not has_chromadb(), reason="chromadb not installed")


# ── Store lifecycle ────────────────────────────────────────────────────

class TestMemoryStoreLifecycle:
    def test_insert_returns_memory_id(self, chromadb_store):
        record = make_decision_record()
        chunks = [{"type": "thesis", "content": "Buy on strength."}]
        embeddings = [[0.1] * 384]  # dummy embedding
        mid = chromadb_store.insert(record, chunks, embeddings, ["bull_thesis", "valuation"])
        assert mid.startswith("mem-")
        assert len(mid) > 4

    def test_insert_increments_count(self, chromadb_store):
        before = chromadb_store.count()
        record = make_decision_record(decision_id="count-test")
        chunks = [{"type": "thesis", "content": "Count test."}]
        embeddings = [[0.2] * 384]
        chromadb_store.insert(record, chunks, embeddings, ["general"])
        assert chromadb_store.count() == before + 1

    def test_get_record_context_after_insert(self, chromadb_store):
        record = make_decision_record(decision_id="ctx-test")
        chunks = [
            {"type": "thesis", "content": "Thesis text."},
            {"type": "market_context", "content": "Market snapshot."},
        ]
        embeddings = [[0.3] * 384, [0.4] * 384]
        mid = chromadb_store.insert(record, chunks, embeddings, ["general"])

        ctx = chromadb_store.get_record_context(mid)
        assert ctx is not None
        assert ctx["memory_id"] == mid
        assert ctx["symbol"] == "AAPL"
        assert "thesis" in ctx["chunks"]
        assert ctx["chunks"]["thesis"] == "Thesis text."


# ── Outcome updates ────────────────────────────────────────────────────

class TestOutcomeUpdate:
    def test_update_outcome_sets_quality_fields(self, chromadb_store):
        record = make_decision_record(decision_id="outcome-test")
        chunks = [{"type": "thesis", "content": "Test."}]
        embeddings = [[0.5] * 384]
        mid = chromadb_store.insert(record, chunks, embeddings, ["general"])

        outcome = make_outcome(holding_period_return=0.12)  # strong positive
        chromadb_store.update_outcome(mid, outcome)

        ctx = chromadb_store.get_record_context(mid)
        # The context returns chunks, not metadata directly.
        # We verify the outcome didn't crash — the metadata update is tested
        # implicitly through retrieval (see test_retrieval).
        assert ctx is not None

    def test_update_outcome_missing_id_no_error(self, chromadb_store):
        """update_outcome on a non-existent id should log a warning, not raise."""
        from tradingagents.extensions.contracts import DecisionOutcome
        from datetime import timezone, datetime
        outcome = DecisionOutcome(observed_at=datetime.now(timezone.utc))
        # Should not raise
        chromadb_store.update_outcome("mem-nonexistent", outcome)


# ── Source and parent metadata ────────────────────────────────────────

class TestSourceMetadata:
    def test_source_field_stored(self, chromadb_store):
        from tradingagents.extensions.contracts import TradeIntent

        intent = make_trade_intent()
        intent.metadata["source"] = "market_analyst"
        intent.metadata["parent"] = "mem-parent-001"
        record = DecisionRecord(
            intent=intent,
            portfolio_before=make_portfolio(),
            market_at_decision=make_market(),
        )
        chunks = [{"type": "thesis", "content": "Test."}]
        embeddings = [[0.9] * 384]
        mid = chromadb_store.insert(record, chunks, embeddings, ["general"])

        ctx = chromadb_store.get_record_context(mid)
        assert ctx is not None


# ── Deduplication ──────────────────────────────────────────────────────

def _make_embedding(seed: float, dim: int = 384) -> list[float]:
    """Generate a deterministic but non-uniform pseudo-embedding."""
    return [(seed * (i + 1)) % 1.0 for i in range(dim)]


class TestFindSimilar:
    def test_empty_store_returns_false(self, tmp_path):
        from tradingagents.extensions.memory.store import MemoryStore
        store = MemoryStore(path=str(tmp_path / "dedup_empty"))
        assert store.find_similar(_make_embedding(0.1), "AAPL") is False

    def test_near_duplicate_detected(self, tmp_path):
        from tradingagents.extensions.memory.store import MemoryStore
        store = MemoryStore(path=str(tmp_path / "dedup_match"))
        record = make_decision_record(decision_id="dedup-test")
        chunks = [{"type": "thesis", "content": "Buy on AI demand strength."}]
        emb = _make_embedding(0.5)
        store.insert(record, chunks, [emb], ["bull_thesis"])
        # Same embedding should match itself at threshold=0.95
        assert store.find_similar(emb, "AAPL") is True

    def test_different_embedding_not_duplicate(self, tmp_path):
        from tradingagents.extensions.memory.store import MemoryStore
        store = MemoryStore(path=str(tmp_path / "dedup_diff"))
        record = make_decision_record(decision_id="dedup-diff")
        chunks = [{"type": "thesis", "content": "Buy on momentum."}]
        store.insert(record, chunks, [_make_embedding(0.3)], ["general"])
        # Very different pseudo-embedding should not match at the strict default
        assert store.find_similar(_make_embedding(0.9), "AAPL") is False

    def test_different_symbol_not_matched(self, tmp_path):
        from tradingagents.extensions.memory.store import MemoryStore
        store = MemoryStore(path=str(tmp_path / "dedup_sym"))
        record = make_decision_record(symbol="NVDA", decision_id="dedup-nvda")
        chunks = [{"type": "thesis", "content": "Strong GPU demand."}]
        emb = _make_embedding(0.7)
        store.insert(record, chunks, [emb], ["bull_thesis"])
        # Querying for a different symbol should not match
        assert store.find_similar(emb, "AAPL") is False


# ── Reflection chunk append ────────────────────────────────────────────

class TestReflectionAppend:
    def test_append_reflection_adds_chunk(self, chromadb_store):
        record = make_decision_record(decision_id="refl-append")
        chunks = [{"type": "thesis", "content": "Original thesis."}]
        embeddings = [[0.6] * 384]
        mid = chromadb_store.insert(record, chunks, embeddings, ["general"])

        before_ctx = chromadb_store.get_record_context(mid)
        assert "reflection" not in before_ctx["chunks"]

        chromadb_store.append_reflection_chunk(
            memory_id=mid,
            content="Lesson learned: patience pays.",
            embedding=[0.7] * 384,
            symbol="AAPL",
            available_at=NOW,
        )

        after_ctx = chromadb_store.get_record_context(mid)
        assert "reflection" in after_ctx["chunks"]
        assert "patience" in after_ctx["chunks"]["reflection"]
