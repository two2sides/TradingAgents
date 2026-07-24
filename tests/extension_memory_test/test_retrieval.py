"""Tests for retrieval — hybrid RAG pipeline (requires chromadb + embedder)."""

from __future__ import annotations

import pytest

from tradingagents.extensions.memory.agent_profiles import get_profile
from .conftest import (
    has_chromadb,
    has_embedder,
    make_decision_record,
    make_memory_query,
    NOW,
)

pytestmark = pytest.mark.skipif(
    not (has_chromadb() and has_embedder()),
    reason="chromadb and sentence-transformers required",
)


# ── Helpers ────────────────────────────────────────────────────────────

def _populate_store(store, embedder, n: int = 3, symbol: str = "AAPL"):
    """Insert *n* decision records into the store with real embeddings."""
    from tradingagents.extensions.memory.chunker import DecisionChunker, _classify_tags

    chunker = DecisionChunker()
    ids = []
    for i in range(n):
        record = make_decision_record(
            symbol=symbol,
            decision_id=f"retrieve-{symbol}-{i}",
        )
        chunks = chunker.split(record)
        if not chunks:
            continue
        texts = [c["content"] for c in chunks]
        embeddings = embedder.embed(texts)
        tags = _classify_tags(record)
        mid = store.insert(record, chunks, embeddings, tags)
        ids.append(mid)
    return ids


# ── Retrieval pipeline ─────────────────────────────────────────────────

class TestRetrievalPipeline:
    def test_retrieve_returns_memory_context(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever

        ids = _populate_store(chromadb_store, memory_embedder, n=3)
        assert len(ids) > 0, "Store should have inserted records"

        retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        query = make_memory_query(symbol="AAPL", agent_role="portfolio_manager")
        profile = get_profile("portfolio_manager")

        result = retriever.retrieve(query, profile)
        assert result.as_of == query.as_of
        assert len(result.items) > 0, "Should return at least one memory item"

    def test_retrieve_respects_max_items(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever

        _populate_store(chromadb_store, memory_embedder, n=5)
        retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        query = make_memory_query(symbol="AAPL", agent_role="portfolio_manager")
        profile = get_profile("portfolio_manager")

        result = retriever.retrieve(query, profile)
        assert len(result.items) <= profile.max_items

    def test_retrieve_items_have_scores(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever

        _populate_store(chromadb_store, memory_embedder, n=3)
        retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        query = make_memory_query(symbol="AAPL", agent_role="portfolio_manager")
        profile = get_profile("portfolio_manager")

        result = retriever.retrieve(query, profile)
        for item in result.items:
            assert item.score is not None
            assert 0 <= item.score <= 1.0, f"Score {item.score} out of [0,1]"

    def test_empty_store_returns_empty_context(self, tmp_path, memory_embedder):
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever
        from tradingagents.extensions.memory.store import MemoryStore

        # Truly fresh store — not the shared session fixture
        store = MemoryStore(path=str(tmp_path / "empty_db"))
        retriever = AgentAwareRetriever(store, memory_embedder)
        query = make_memory_query(symbol="AAPL")
        profile = get_profile("portfolio_manager")

        result = retriever.retrieve(query, profile)
        assert result.items == []

    def test_retrieve_returns_summary_string(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever

        _populate_store(chromadb_store, memory_embedder, n=2)
        retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        query = make_memory_query(symbol="AAPL")
        profile = get_profile("portfolio_manager")

        result = retriever.retrieve(query, profile)
        assert isinstance(result.summary, str)
        if result.items:
            assert len(result.summary) > 0


# ── Role-specific retrieval ────────────────────────────────────────────

class TestRoleSpecificRetrieval:
    def test_market_analyst_retrieval(self, chromadb_store, memory_embedder):
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever

        # Populate with decisions tagged for technical patterns
        from tradingagents.extensions.memory.chunker import DecisionChunker, _classify_tags

        record = make_decision_record(
            symbol="NVDA",
            decision_id="tech-test",
            rationale="MACD bullish crossover with rising RSI. Volume confirms breakout above resistance.",
        )
        chunker = DecisionChunker()
        chunks = chunker.split(record)
        embeddings = memory_embedder.embed([c["content"] for c in chunks])
        tags = _classify_tags(record)
        chromadb_store.insert(record, chunks, embeddings, tags)

        retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        query = make_memory_query(symbol="NVDA", agent_role="market_analyst")
        profile = get_profile("market_analyst")

        result = retriever.retrieve(query, profile)
        # Should work without error — market analyst profile favors
        # market_context and technical-tagged chunks
        assert isinstance(result.items, list)

    def test_bear_researcher_weights_outcome_more(self, chromadb_store, memory_embedder):
        br = get_profile("bear_researcher")
        ma = get_profile("market_analyst")
        # Bear researcher cares most about whether the thesis was correct
        assert br.outcome_weight > ma.outcome_weight

    def test_source_boost_prefers_same_source_records(self, chromadb_store, memory_embedder):
        """Records from the same source as the querier should rank higher."""
        from tradingagents.extensions.memory.retrieval import AgentAwareRetriever
        from tradingagents.extensions.memory.chunker import DecisionChunker, _classify_tags

        chunker = DecisionChunker()

        # Insert a record with source=market_analyst (should be boosted)
        source_record = make_decision_record(
            symbol="AMD",
            decision_id="source-amd",
            rationale="RSI divergence with volume spike — potential reversal signal.",
        )
        source_record.intent.metadata["source"] = "market_analyst"
        source_chunks = chunker.split(source_record)
        source_embs = memory_embedder.embed([c["content"] for c in source_chunks])
        source_tags = _classify_tags(source_record)
        source_mid = chromadb_store.insert(source_record, source_chunks, source_embs, source_tags)

        # Insert a record for the same symbol WITHOUT source (no boost)
        no_source_record = make_decision_record(
            symbol="AMD",
            decision_id="nosource-amd",
            rationale="General market overview — sector rotation into semis.",
        )
        no_source_chunks = chunker.split(no_source_record)
        no_source_embs = memory_embedder.embed([c["content"] for c in no_source_chunks])
        no_source_tags = _classify_tags(no_source_record)
        chromadb_store.insert(no_source_record, no_source_chunks, no_source_embs, no_source_tags)

        retriever = AgentAwareRetriever(chromadb_store, memory_embedder)
        query = make_memory_query(symbol="AMD", agent_role="market_analyst")
        profile = get_profile("market_analyst")

        result = retriever.retrieve(query, profile)
        assert len(result.items) >= 1

        # The same-source record should appear in results
        source_ids = {
            item.memory_id for item in result.items
            if item.metadata.get("source") == "market_analyst"
        }
        assert source_mid in source_ids, "Market analyst source record should be retrieved"
