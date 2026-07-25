"""ChromaDB-backed storage for trading memory.

Each memory is stored as multiple chunks (thesis, market_context,
portfolio_context, reflection).  Chunks share a ``memory_id`` so they can
be de-duplicated during retrieval.  ChromaDB provides ANN vector search
with metadata filtering out of the box.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tradingagents.extensions.contracts import (
        DecisionOutcome,
        DecisionRecord,
        MemoryItem,
    )

logger = logging.getLogger(__name__)

# ChromaDB metadata keys used in where-clause filters
_KEY_MEMORY_ID = "memory_id"
_KEY_CHUNK_TYPE = "chunk_type"
_KEY_SYMBOL = "symbol"
_KEY_DECISION_AT = "decision_at"
_KEY_AVAILABLE_AT = "available_at"
_KEY_TAGS = "agent_tags"
_KEY_CONFIDENCE = "confidence"
_KEY_SOURCE = "source"
_KEY_PARENT = "parent"


def _to_ts(dt: datetime) -> float:
    """Convert a datetime to Unix timestamp (float) for ChromaDB metadata.

    ChromaDB's ``$lte`` / ``$gte`` only accept numeric operands.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _sanitize_meta(meta: dict) -> dict:
    """Drop None values so ChromaDB never sees a non-metadata type."""
    return {k: v for k, v in meta.items() if v is not None}


class MemoryStore:
    """ChromaDB-backed persistent store for trading memory chunks."""

    def __init__(self, path: str | None = None, config: dict | None = None):
        cfg = config or {}
        db_path = path or cfg.get(
            "memory_db_path",
            cfg.get("data_cache_dir", ".tradingagents") + "/memory_db",
        )

        self._db_path = db_path

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for the enhanced memory store. "
                "Install with: pip install chromadb"
            )

        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="trading_memories",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "MemoryStore initialised at %s — %d chunks in collection.",
            db_path,
            self._collection.count(),
        )

    @property
    def collection(self):
        """Direct ChromaDB collection access for retrieval layer."""
        return self._collection

    @property
    def db_path(self) -> str:
        return self._db_path

    # ── Write path ──────────────────────────────────────────────────────

    def insert(
        self,
        record: DecisionRecord,
        chunks: list[dict],
        embeddings: list[list[float]],
        agent_tags: list[str],
    ) -> str:
        """Persist a new decision as multiple chunk rows.

        Returns the shared ``memory_id``.
        """
        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        decision_ts = _to_ts(record.intent.as_of)
        tags_str = "|".join(agent_tags) if agent_tags else "general"

        ids: list[str] = []
        docs: list[str] = []
        embs: list[list[float]] = []
        source = (record.intent.metadata or {}).get("source", "")
        parent = (record.intent.metadata or {}).get("parent", "")
        metas: list[dict] = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{memory_id}-{chunk['type']}"
            ids.append(chunk_id)
            docs.append(chunk["content"])
            embs.append(embeddings[i])
            metas.append({
                _KEY_MEMORY_ID: memory_id,
                _KEY_CHUNK_TYPE: chunk["type"],
                _KEY_SYMBOL: record.intent.symbol,
                _KEY_DECISION_AT: decision_ts,
                _KEY_AVAILABLE_AT: decision_ts,
                _KEY_TAGS: tags_str,
                _KEY_CONFIDENCE: record.intent.confidence,
                _KEY_SOURCE: source,
                _KEY_PARENT: parent,
                "outcome_raw": None,
                "outcome_alpha": None,
                "outcome_quality": None,
            })

        self._collection.add(ids=ids, embeddings=embs, documents=docs,
                            metadatas=[_sanitize_meta(m) for m in metas])
        logger.debug("Inserted memory %s with %d chunks.", memory_id, len(chunks))
        return memory_id

    def update_outcome(self, memory_id: str, outcome: DecisionOutcome) -> None:
        """Update all chunks of *memory_id* with outcome metadata."""
        existing = self._collection.get(where={_KEY_MEMORY_ID: memory_id})
        if not existing or not existing["ids"]:
            logger.warning("update_outcome: no chunks found for %s", memory_id)
            return

        quality = _outcome_quality(outcome)
        new_metas = []
        for meta in existing["metadatas"]:
            meta = dict(meta)  # copy — ChromaDB returns immutable dicts
            meta["outcome_raw"] = outcome.holding_period_return
            meta["outcome_alpha"] = (
                outcome.holding_period_return  # alpha is computed externally
            )
            meta["outcome_quality"] = quality
            new_metas.append(meta)

        self._collection.update(ids=existing["ids"],
                                metadatas=[_sanitize_meta(m) for m in new_metas])
        logger.debug("Updated outcome for %s (quality=%.2f).", memory_id, quality)

    def append_reflection_chunk(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        symbol: str,
        available_at: datetime,
        agent_tags: str = "reflection",
    ) -> None:
        """Add a reflection chunk to an existing memory.

        ``available_at`` is the time from which this reflection is safe to retrieve.
        """
        chunk_id = f"{memory_id}-reflection"
        now_ts = datetime.now(timezone.utc).timestamp()
        self._collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[_sanitize_meta({
                _KEY_MEMORY_ID: memory_id,
                _KEY_CHUNK_TYPE: "reflection",
                _KEY_SYMBOL: symbol,
                _KEY_DECISION_AT: now_ts,
                _KEY_AVAILABLE_AT: _to_ts(available_at),
                _KEY_TAGS: agent_tags,
                _KEY_CONFIDENCE: 0.0,
                "outcome_raw": None,
                "outcome_alpha": None,
                "outcome_quality": None,
            })],
        )
        logger.debug("Appended reflection chunk to %s.", memory_id)

    # ── Read path ───────────────────────────────────────────────────────

    def get_record_context(self, memory_id: str) -> dict[str, Any] | None:
        """Return all chunks for *memory_id* as a dict for reflection generation."""
        result = self._collection.get(where={_KEY_MEMORY_ID: memory_id})
        if not result or not result["ids"]:
            return None

        chunks_by_type: dict[str, str] = {}
        symbol = ""
        for i, ctype in enumerate(result["metadatas"]):
            chunks_by_type[ctype[_KEY_CHUNK_TYPE]] = result["documents"][i]
            symbol = ctype[_KEY_SYMBOL]

        return {
            "memory_id": memory_id,
            "symbol": symbol,
            "chunks": chunks_by_type,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def propagate_outcome(
        self, parent_id: str, outcome_raw: float | None, outcome_quality: float | None
    ) -> None:
        """Propagate outcome metadata to all child records of *parent_id*."""
        try:
            children = self._collection.get(where={_KEY_PARENT: parent_id})
        except Exception:
            return

        if not children or not children["ids"]:
            return

        new_metas = []
        for meta in children["metadatas"]:
            meta = dict(meta)
            meta["outcome_raw"] = outcome_raw
            meta["outcome_alpha"] = outcome_raw
            meta["outcome_quality"] = outcome_quality
            new_metas.append(meta)

        self._collection.update(
            ids=children["ids"],
            metadatas=[_sanitize_meta(m) for m in new_metas],
        )
        logger.debug(
            "Propagated outcome to %d child records of %s.",
            len(children["ids"]), parent_id,
        )

    def find_similar(
        self,
        embedding: list[float],
        symbol: str,
        threshold: float = 0.95,
    ) -> bool:
        """Check whether a very similar memory already exists for *symbol*.

        Returns True if any existing chunk's cosine similarity to *embedding*
        exceeds *threshold*, meaning the new memory would be near-duplicate.
        """
        if self._collection.count() == 0:
            return False
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=3,
                where={"symbol": symbol},
                include=["distances"],
            )
        except Exception:
            return False

        if not results or not results.get("distances") or not results["distances"][0]:
            return False

        # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance/2
        min_dist = min(results["distances"][0])
        similarity = 1.0 - min_dist / 2.0
        return similarity >= threshold


def _outcome_quality(outcome: DecisionOutcome) -> float | None:
    """Heuristic that scores outcome quality from holding-period return.

    Returns a value in [0, 1], or None when the return is unavailable.
    This is used to boost high-quality memories in retrieval ranking.
    """
    ret = outcome.holding_period_return
    if ret is None:
        return None
    if ret > 0.10:
        return 1.0
    if ret > 0.05:
        return 0.85
    if ret > 0.0:
        return 0.65
    if ret > -0.05:
        return 0.40
    if ret > -0.10:
        return 0.20
    return 0.10
