"""Embedding model wrapper for memory chunk vectorisation.

Supports local (sentence-transformers) and API-based (OpenAI) backends,
selected via config.  The local default has zero API cost and works offline.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingBackend(Protocol):
    """Minimal protocol for an embedding service."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        ...


class LocalEmbeddingBackend:
    """sentence-transformers (e.g. all-MiniLM-L6-v2) — offline, 384-dim."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        logger.info("Loading embedding model %s (this may take a moment on first run)...", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_embedding_dimension()
        logger.info("Embedding model loaded — %d dimensions.", self.dim)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        result = self._model.encode(texts, normalize_embeddings=normalize_embeddings)
        return result.tolist()


class OpenAIEmbeddingBackend:
    """OpenAI text-embedding API — requires OPENAI_API_KEY, 1536-dim (3-small)."""

    def __init__(self, model_name: str = "text-embedding-3-small", base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )
        self._client = OpenAI(base_url=base_url)
        self._model = model_name
        # Known dimensions; actual dims are returned by the API per-request
        self.dim = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(
            model_name, 1536
        )

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        embeddings = [d.embedding for d in response.data]
        if normalize_embeddings:
            embeddings = [_normalize(emb) for emb in embeddings]
        return embeddings


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalise a vector in-place-ish (returns new list)."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class MemoryEmbedder:
    """Thin wrapper that delegates to a configured backend.

    Usage::

        emb = MemoryEmbedder({"memory_embedding": "local"})
        vecs = emb.embed(["some text", "more text"])
        query_vec = emb.embed_query("a question")
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        backend_name = cfg.get("memory_embedding", "local")

        if backend_name == "local" or backend_name == "sentence-transformers":
            model = cfg.get("memory_embedding_model", "all-MiniLM-L6-v2")
            self._backend: EmbeddingBackend = LocalEmbeddingBackend(model)
        elif backend_name == "openai":
            model = cfg.get("memory_embedding_model", "text-embedding-3-small")
            base_url = cfg.get("backend_url")
            self._backend = OpenAIEmbeddingBackend(model, base_url=base_url)
        else:
            raise ValueError(f"Unsupported embedding backend: {backend_name}")

        self.dim = self._backend.dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (documents / chunks)."""
        if not texts:
            return []
        return self._backend.encode(texts, normalize_embeddings=True)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.  Returns a single vector."""
        result = self._backend.encode([text], normalize_embeddings=True)
        return result[0]
