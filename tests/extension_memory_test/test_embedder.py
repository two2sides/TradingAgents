"""Tests for embedder — embedding model wrapper (requires sentence-transformers)."""

from __future__ import annotations

import pytest

from .conftest import has_embedder

pytestmark = pytest.mark.skipif(not has_embedder(), reason="sentence-transformers not installed")


class TestMemoryEmbedder:
    def test_local_embedder_initializes(self):
        from tradingagents.extensions.memory.embedder import MemoryEmbedder

        emb = MemoryEmbedder({"memory_embedding": "local"})
        assert emb.dim > 0

    def test_embed_returns_correct_dimensions(self):
        from tradingagents.extensions.memory.embedder import MemoryEmbedder

        emb = MemoryEmbedder({"memory_embedding": "local"})
        texts = ["The market is bullish on tech stocks.", "Valuation concerns persist."]
        vecs = emb.embed(texts)
        assert len(vecs) == 2
        assert all(len(v) == emb.dim for v in vecs)

    def test_embed_query_returns_single_vector(self):
        from tradingagents.extensions.memory.embedder import MemoryEmbedder

        emb = MemoryEmbedder({"memory_embedding": "local"})
        vec = emb.embed_query("NVDA strong buy on AI demand")
        assert len(vec) == emb.dim
        assert isinstance(vec[0], float)

    def test_embed_normalized_vectors(self):
        """Embeddings should be L2-normalized (unit vectors)."""
        from tradingagents.extensions.memory.embedder import MemoryEmbedder

        emb = MemoryEmbedder({"memory_embedding": "local"})
        vec = emb.embed_query("test")
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 0.01

    def test_embed_empty_list_returns_empty(self):
        from tradingagents.extensions.memory.embedder import MemoryEmbedder

        emb = MemoryEmbedder({"memory_embedding": "local"})
        assert emb.embed([]) == []

    def test_unknown_backend_raises(self):
        from tradingagents.extensions.memory.embedder import MemoryEmbedder

        with pytest.raises(ValueError, match="Unsupported"):
            MemoryEmbedder({"memory_embedding": "nonexistent_backend"})
