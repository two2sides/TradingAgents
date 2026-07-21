"""Hybrid retrieval pipeline: ANN vector search + metadata filtering + weighted reranking.

The pipeline:
  1. Embed the agent-specific query text
  2. ANN search in ChromaDB (coarse top-K)
  3. Metadata filter (symbol, time-safety, chunk types, tags)
  4. Weighted rerank (similarity × recency × outcome_quality)
  5. Deduplicate by memory_id, return top items
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tradingagents.extensions.contracts import MemoryItem

if TYPE_CHECKING:
    from tradingagents.extensions.contracts import MemoryContext, MemoryQuery
    from tradingagents.extensions.memory.agent_profiles import AgentMemoryProfile
    from tradingagents.extensions.memory.store import MemoryStore
    from tradingagents.extensions.memory.embedder import MemoryEmbedder

# — Time-safety window: chunks whose available_at is within this many seconds
#   of as_of are considered safe.  Keeps the where-clause simple.
_TIME_EPSILON = 1.0  # seconds


def _parse_iso(ts_str: str) -> datetime:
    """Parse an ISO timestamp string into a timezone-aware datetime."""
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _days_decay(dt_str: str, as_of: datetime) -> float:
    """Exponential recency decay based on days elapsed.

    Returns a value in (0, 1] where 1.0 = same-day and ~0.05 after 60 days.
    """
    try:
        decision_dt = _parse_iso(dt_str)
    except (ValueError, TypeError):
        return 0.1
    days = (as_of - decision_dt).total_seconds() / 86400.0
    if days < 0:
        return 0.0  # future decision — shouldn't happen after time filtering
    return math.exp(-0.05 * days)


def _outcome_score(outcome_raw, outcome_alpha, outcome_quality) -> float:
    """Convert stored outcome fields to a quality score in [0, 1].

    If outcome_quality was precomputed by the store, use it directly.
    Otherwise fall back to the raw return value.
    """
    if outcome_quality is not None:
        try:
            return float(outcome_quality)
        except (TypeError, ValueError):
            pass
    if outcome_raw is not None:
        try:
            raw = float(outcome_raw)
            if raw > 0.10:
                return 1.0
            if raw > 0.05:
                return 0.85
            if raw > 0.0:
                return 0.65
            if raw > -0.05:
                return 0.40
            if raw > -0.10:
                return 0.20
            return 0.10
        except (TypeError, ValueError):
            pass
    return 0.5  # unknown outcome — neutral


def _tag_match(chunk_tags: str, interest_tags: list[str]) -> bool:
    """Return True when *chunk_tags* contains any of *interest_tags*.

    chunk_tags is a pipe-delimited string like ``"valuation|earnings|bull_thesis"``.
    An interest tag of ``"*"`` matches everything.
    """
    if not interest_tags or interest_tags == ["*"]:
        return True
    chunk_set = set(chunk_tags.split("|"))
    return bool(chunk_set.intersection(interest_tags))


class AgentAwareRetriever:
    """Orchestrates the full retrieval pipeline for a given agent profile."""

    def __init__(self, store: MemoryStore, embedder: MemoryEmbedder):
        self._store = store
        self._embedder = embedder

    def retrieve(
        self,
        query: MemoryQuery,
        profile: AgentMemoryProfile,
    ) -> MemoryContext:
        """Run the full pipeline and return a MemoryContext."""
        from tradingagents.extensions.contracts import MemoryContext

        kwargs = profile.to_retrieval_kwargs(query)
        query_text = kwargs.pop("query_text")
        query_emb = self._embedder.embed_query(query_text)

        # ── Step 1: ANN coarse retrieval ──
        n_fetch = max(kwargs["max_items"] * 4, 10)

        # Build ChromaDB where clause for time-safety and symbol filtering
        where_parts: list[dict] = [
            {  # available_at must not be later than as_of
                "available_at": {"$lte": kwargs["as_of"].isoformat()}
            }
        ]
        if not kwargs["cross_ticker"]:
            where_parts.append({"symbol": kwargs["symbol"]})

        where = {"$and": where_parts} if len(where_parts) > 1 else where_parts[0]

        try:
            results = self._store.collection.query(
                query_embeddings=[query_emb],
                n_results=min(n_fetch, self._store.count()),
                where=where,
                include=["embeddings", "documents", "metadatas", "distances"],
            )
        except Exception:
            # ChromaDB may raise if the collection is empty or where-clause
            # matches nothing; this is harmless.
            return MemoryContext(as_of=query.as_of, items=[])

        if not results or not results["ids"] or not results["ids"][0]:
            return MemoryContext(as_of=query.as_of, items=[])

        # ── Step 2: Post-retrieval filtering (tags, chunk types, cross-ticker) ──
        candidates = _filter_and_score(results, kwargs)
        if not candidates:
            return MemoryContext(as_of=query.as_of, items=[])

        # ── Step 3: Deduplicate by memory_id, keeping highest score ──
        seen: dict[str, tuple[float, dict]] = {}
        for score, item_data in candidates:
            mid = item_data["memory_id"]
            if mid not in seen or score > seen[mid][0]:
                seen[mid] = (score, item_data)

        # ── Step 4: Sort and take top-k ──
        ranked = sorted(seen.values(), key=lambda x: x[0], reverse=True)
        top = ranked[: kwargs["max_items"]]

        # ── Step 5: Build MemoryItems ──
        items: list[MemoryItem] = []
        for score, data in top:
            try:
                decision_at = _parse_iso(data.get("decision_at", kwargs["as_of"].isoformat()))
            except (ValueError, TypeError):
                decision_at = kwargs["as_of"]
            try:
                available_at = _parse_iso(data.get("available_at", kwargs["as_of"].isoformat()))
            except (ValueError, TypeError):
                available_at = kwargs["as_of"]

            items.append(MemoryItem(
                memory_id=data["memory_id"],
                symbol=data.get("symbol", kwargs["symbol"]),
                decision_at=decision_at,
                available_at=available_at,
                content=_assemble_item_content(data),
                score=round(score, 4),
                metadata={
                    "chunk_types": data.get("chunk_types", []),
                    "tags": data.get("tags", ""),
                    "outcome_raw": data.get("outcome_raw"),
                    "outcome_alpha": data.get("outcome_alpha"),
                },
            ))

        summary = _build_summary(items, profile.role)
        return MemoryContext(
            as_of=query.as_of,
            items=items,
            summary=summary,
        )


def _filter_and_score(
    chroma_results: dict,
    kwargs: dict,
) -> list[tuple[float, dict]]:
    """Filter ChromaDB results by tags/chunk_types and compute weighted scores.

    Returns a list of ``(score, item_data)`` tuples.
    """
    ids_list = chroma_results["ids"][0]
    docs_list = chroma_results["documents"][0]
    metas_list = chroma_results["metadatas"][0]
    distances = chroma_results["distances"][0]

    sim_weight = kwargs["sim_weight"]
    recency_weight = kwargs["recency_weight"]
    outcome_weight = kwargs["outcome_weight"]
    as_of = kwargs["as_of"]
    chunk_types = kwargs.get("chunk_types", [])
    interest_tags = kwargs.get("interest_tags", [])
    query_symbol = kwargs["symbol"]
    cross_ticker = kwargs["cross_ticker"]

    candidates: list[tuple[float, dict]] = []

    for i in range(len(ids_list)):
        meta = metas_list[i] or {}
        doc = docs_list[i] or ""
        chunk_type = meta.get("chunk_type", "")

        # Filter by chunk type
        if chunk_types and chunk_type not in chunk_types:
            continue

        # Filter by interest tags
        tags = meta.get("agent_tags", "")
        if not _tag_match(tags, interest_tags):
            continue

        # Cross-ticker filtering (already applied in ChromaDB where, but double-check)
        if not cross_ticker and meta.get("symbol", "") != query_symbol:
            continue

        # Convert ChromaDB cosine distance → similarity
        # For cosine space, distance ∈ [0, 2]; similarity = 1 - distance/2
        distance = distances[i]
        cos_sim = 1.0 - (distance / 2.0)
        cos_sim = max(0.0, min(1.0, cos_sim))

        # Recency score
        decision_at = meta.get("decision_at", "")
        recency = _days_decay(decision_at, as_of)

        # Outcome score
        outcome = _outcome_score(
            meta.get("outcome_raw"),
            meta.get("outcome_alpha"),
            meta.get("outcome_quality"),
        )

        # Weighted fusion
        score = (
            sim_weight * cos_sim
            + recency_weight * recency
            + outcome_weight * outcome
        )

        candidates.append((score, {
            "memory_id": meta.get("memory_id", f"unknown-{i}"),
            "symbol": meta.get("symbol", ""),
            "decision_at": decision_at,
            "available_at": meta.get("available_at", ""),
            "chunk_types": [chunk_type],
            "tags": tags,
            "confidence": meta.get("confidence"),
            "outcome_raw": meta.get("outcome_raw"),
            "outcome_alpha": meta.get("outcome_alpha"),
            "content": doc,
        }))

    return candidates


def _assemble_item_content(data: dict) -> str:
    """Assemble readable content from chunk data."""
    content = data.get("content", "")
    outcome = data.get("outcome_raw")
    confidence = data.get("confidence")

    suffix_parts = []
    if outcome is not None:
        try:
            suffix_parts.append(f"[Return: {float(outcome):+.2%}]")
        except (TypeError, ValueError):
            pass
    if confidence is not None:
        try:
            suffix_parts.append(f"[Confidence: {float(confidence):.0%}]")
        except (TypeError, ValueError):
            pass

    if suffix_parts:
        content = content + "\n" + " ".join(suffix_parts)

    return content


def _build_summary(items: list[MemoryItem], role: str) -> str:
    """Generate a short natural-language summary of retrieved memories."""
    if not items:
        return ""

    n = len(items)
    symbols = list(dict.fromkeys(item.symbol for item in items))  # unique, order-preserving
    symbol_str = ", ".join(symbols[:3])
    if len(symbols) > 3:
        symbol_str += f" and {len(symbols) - 3} more"

    outcomes = [
        item.metadata.get("outcome_raw")
        for item in items
        if item.metadata.get("outcome_raw") is not None
    ]
    pos = sum(1 for o in outcomes if o > 0)
    neg = sum(1 for o in outcomes if o < 0)

    parts = [f"{n} relevant past experiences retrieved for {role} ({symbol_str})."]
    if outcomes:
        parts.append(f"{pos} positive, {neg} negative outcomes among those with known results.")
    if n - len(outcomes) > 0:
        parts.append(f"{n - len(outcomes)} entries have pending outcomes.")

    return " ".join(parts)
