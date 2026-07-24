"""Tests for EnhancedMemoryProvider — MemoryProvider protocol compliance."""

from __future__ import annotations

import pytest

from tradingagents.extensions.contracts import (
    MemoryContext,
    MemoryQuery,
    MemoryReference,
)
from tradingagents.extensions.protocols import MemoryProvider
from .conftest import (
    has_chromadb,
    has_embedder,
    make_decision_record,
    make_memory_query,
    make_outcome,
    NOW,
)


# ── Protocol compliance (no heavy deps) ────────────────────────────────

class TestProtocolCompliance:
    def test_enhanced_memory_provider_is_memory_provider(self):
        """Structural subtyping: the class must satisfy the protocol."""
        from tradingagents.extensions.memory import EnhancedMemoryProvider

        # Even without instantiation, the class should be recognized
        # as a structural subtype of MemoryProvider.
        # We verify by checking method signatures match.
        assert hasattr(EnhancedMemoryProvider, "retrieve")
        assert hasattr(EnhancedMemoryProvider, "record_decision")
        assert hasattr(EnhancedMemoryProvider, "record_outcome")

    def test_method_signatures(self):
        """All three protocol methods must accept the declared parameter types."""
        import inspect
        from tradingagents.extensions.memory import EnhancedMemoryProvider

        sig_retrieve = inspect.signature(EnhancedMemoryProvider.retrieve)
        assert "query" in sig_retrieve.parameters

        sig_record = inspect.signature(EnhancedMemoryProvider.record_decision)
        assert "record" in sig_record.parameters

        sig_outcome = inspect.signature(EnhancedMemoryProvider.record_outcome)
        assert "reference" in sig_outcome.parameters
        assert "outcome" in sig_outcome.parameters


# ── Formatting helpers (no heavy deps) ─────────────────────────────────

class TestFormatContext:
    def test_format_empty_context(self):
        from tradingagents.extensions.memory import EnhancedMemoryProvider

        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        ctx = MemoryContext(as_of=NOW, items=[])
        result = provider.format_context_for_prompt(ctx)
        assert result == ""

    def test_format_context_with_items(self):
        from tradingagents.extensions.memory import EnhancedMemoryProvider
        from tradingagents.extensions.contracts import MemoryItem

        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        item = MemoryItem(
            memory_id="mem-1",
            symbol="AAPL",
            decision_at=NOW,
            available_at=NOW,
            content="Buy on technical breakout. Outcome: +5.2%",
            score=0.85,
        )
        ctx = MemoryContext(
            as_of=NOW,
            items=[item],
            summary="Relevant past experience found.",
        )
        result = provider.format_context_for_prompt(ctx)
        assert "Relevant past experience" in result
        assert "AAPL" in result
        assert "+5.2%" in result
        assert "0.85" in result or "relevance" in result.lower()

    def test_format_all_for_state(self):
        from tradingagents.extensions.memory import EnhancedMemoryProvider
        from tradingagents.extensions.contracts import MemoryItem

        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        item = MemoryItem(
            memory_id="mem-2",
            symbol="NVDA",
            decision_at=NOW,
            available_at=NOW,
            content="Market analyst pattern match.",
            score=0.72,
        )

        contexts = {
            "portfolio_manager": MemoryContext(as_of=NOW, items=[item]),
            "market_analyst": MemoryContext(as_of=NOW, items=[]),
        }
        result = provider.format_all_for_state(contexts)

        assert "memory_portfolio_manager" in result
        assert len(result["memory_portfolio_manager"]) > 0
        assert "memory_market_analyst" not in result  # empty context skipped


# ── Full integration (requires chromadb + embedder) ────────────────────

@pytest.mark.skipif(
    not (has_chromadb() and has_embedder()),
    reason="chromadb and sentence-transformers required",
)
class TestProviderIntegration:
    def test_retrieve_via_provider(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.provider import EnhancedMemoryProvider

        config = {"memory_db_path": chromadb_store.db_path}
        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        provider.config = config
        provider.store = chromadb_store
        provider.embedder = memory_embedder
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever
        provider.retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        provider.chunker = __import__(
            "tradingagents.extensions.memory.chunker", fromlist=["DecisionChunker"]
        ).DecisionChunker()

        # Populate with one decision
        record = make_decision_record(symbol="MSFT", decision_id="prov-test")
        provider.record_decision(record)

        query = make_memory_query(symbol="MSFT", agent_role="portfolio_manager")
        ctx = provider.retrieve(query)
        assert isinstance(ctx, MemoryContext)
        assert ctx.as_of is not None

    def test_record_decision_and_outcome_cycle(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.provider import EnhancedMemoryProvider

        config = {"memory_db_path": chromadb_store.db_path}
        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        provider.config = config
        provider.store = chromadb_store
        provider.embedder = memory_embedder
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever
        provider.retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        provider.chunker = __import__(
            "tradingagents.extensions.memory.chunker", fromlist=["DecisionChunker"]
        ).DecisionChunker()
        provider._llm = None  # skip reflection generation

        # Phase 1: record decision
        record = make_decision_record(symbol="GOOGL", decision_id="cycle-test")
        ref = provider.record_decision(record)
        assert isinstance(ref, MemoryReference)
        assert ref.memory_id.startswith("mem-")

        # Phase 2: record outcome
        outcome = make_outcome(holding_period_return=0.10)
        provider.record_outcome(ref, outcome)
        # Should not raise; reflection skipped because _llm is None

    def test_dedup_skips_near_duplicate(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.provider import EnhancedMemoryProvider

        config = {"memory_db_path": chromadb_store.db_path}
        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        provider.config = config
        provider.store = chromadb_store
        provider.embedder = memory_embedder
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever
        provider.retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        provider.chunker = __import__(
            "tradingagents.extensions.memory.chunker", fromlist=["DecisionChunker"]
        ).DecisionChunker()
        provider._llm = None

        # Insert first decision
        record1 = make_decision_record(symbol="TSLA", decision_id="dedup-1",
                                       rationale="Strong buy on EV dominance and margin expansion.")
        ref1 = provider.record_decision(record1)
        assert ref1.memory_id.startswith("mem-")

        # Insert identical decision — should be dedup'd
        record2 = make_decision_record(symbol="TSLA", decision_id="dedup-2",
                                       rationale="Strong buy on EV dominance and margin expansion.")
        ref2 = provider.record_decision(record2)
        assert ref2.memory_id == "mem-dup", f"Expected mem-dup, got {ref2.memory_id}"

    def test_intermediate_record_with_source(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.provider import EnhancedMemoryProvider

        config = {"memory_db_path": chromadb_store.db_path}
        provider = EnhancedMemoryProvider.__new__(EnhancedMemoryProvider)
        provider.config = config
        provider.store = chromadb_store
        provider.embedder = memory_embedder
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever
        provider.retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        provider.chunker = __import__(
            "tradingagents.extensions.memory.chunker", fromlist=["DecisionChunker"]
        ).DecisionChunker()
        provider._llm = None

        # Store PM decision
        pm_record = make_decision_record(symbol="META", decision_id="intermediate-pm")
        pm_ref = provider.record_decision(pm_record)

        # Store intermediate analysis with parent link
        from tradingagents.extensions.contracts import DecisionRecord, TradeIntent
        from .conftest import make_portfolio, make_market, make_trade_intent

        market_intent = make_trade_intent(symbol="META",
                                          rationale="MACD bullish crossover with volume confirmation.")
        market_intent.metadata["source"] = "market_analyst"
        market_intent.metadata["parent"] = pm_ref.memory_id
        market_record = DecisionRecord(
            intent=market_intent,
            portfolio_before=make_portfolio(),
            market_at_decision=make_market("META"),
        )
        market_ref = provider.record_decision(market_record)
        assert market_ref.memory_id.startswith("mem-")
        assert market_ref.memory_id != pm_ref.memory_id

        # Propagate outcome — should update both parent and child
        outcome = make_outcome(holding_period_return=0.15)
        provider.record_outcome(pm_ref, outcome)
        # Child should now have outcome data
        child_ctx = chromadb_store.get_record_context(market_ref.memory_id)
        assert child_ctx is not None
