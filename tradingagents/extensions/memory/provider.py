"""EnhancedMemoryProvider — B-team's RAG implementation of the MemoryProvider protocol.

This is the main entry point.  It wires together:
  - ChromaDB storage (``MemoryStore``)
  - Embedding generation (``MemoryEmbedder``)
  - Semantic chunking (``DecisionChunker``)
  - Role-aware hybrid retrieval (``AgentAwareRetriever`` + ``AgentMemoryProfile``)

Usage by Team A::

    from tradingagents.extensions.memory import EnhancedMemoryProvider

    provider = EnhancedMemoryProvider(config, llm_client=quick_think_llm)
    context = provider.retrieve(query)
    ref     = provider.record_decision(record)
    provider.record_outcome(ref, outcome)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from tradingagents.extensions.contracts import (
    DecisionOutcome,
    DecisionRecord,
    MemoryContext,
    MemoryQuery,
    MemoryReference,
)

from .agent_profiles import get_profile
from .chunker import DecisionChunker
from .embedder import MemoryEmbedder
from .retrieval import AgentAwareRetriever
from .store import MemoryStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Reflection prompt ───────────────────────────────────────────────────────
# Kept here (not in chunker) because it uses the LLM, which is provider-level.

_REFLECTION_PROMPT = (
    "You are a trading analyst reviewing a past decision now that the outcome is known.\n"
    "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
    "Cover in order:\n"
    "1. Was the directional call correct? (cite the return figures)\n"
    "2. Which part of the investment thesis held or failed?\n"
    "3. One concrete lesson to apply to the next similar analysis.\n\n"
    "Be specific and terse. Your output will be stored in a memory database "
    "and retrieved by future analysts, so every word must earn its place."
)


class EnhancedMemoryProvider:
    """B-team implementation of the MemoryProvider protocol with RAG retrieval."""

    def __init__(
        self,
        config: dict | None = None,
        llm_client: Any = None,
    ):
        """Initialise the memory provider.

        Args:
            config: TradingAgents config dict.  Relevant keys:
                - ``memory_db_path``: path to ChromaDB persistent store
                - ``memory_embedding``: ``"local"`` (default) or ``"openai"``
                - ``memory_embedding_model``: model name for the chosen backend
                - ``data_cache_dir``: fallback path for the DB when
                  ``memory_db_path`` is unset
            llm_client: A LangChain chat model used to generate reflections.
                When None, reflections are skipped (graceful degradation).
        """
        cfg = config or {}
        self.config = cfg
        self._llm = llm_client

        # Initialise the three backbone services
        self.store = MemoryStore(config=cfg)
        self.embedder = MemoryEmbedder(cfg)
        self.retriever = AgentAwareRetriever(self.store, self.embedder)
        self.chunker = DecisionChunker()

        logger.info(
            "EnhancedMemoryProvider ready — store has %d chunks.",
            self.store.count(),
        )

    # ── MemoryProvider protocol ─────────────────────────────────────────────

    def retrieve(self, query: MemoryQuery) -> MemoryContext:
        """Return time-safe, role-aware memories for *query*.

        The agent role is read from ``query.metadata["agent_role"]``, falling
        back to ``"portfolio_manager"`` when absent.
        """
        role = query.metadata.get("agent_role", "portfolio_manager")
        profile = get_profile(role)
        logger.debug(
            "Retrieving memories for %s as %s (max %d items).",
            query.symbol, role, profile.max_items,
        )
        return self.retriever.retrieve(query, profile)

    def record_decision(self, record: DecisionRecord) -> MemoryReference:
        """Persist a decision into the memory store.

        Splits the record into semantic chunks, embeds each, classifies tags,
        and writes everything to ChromaDB in one batch.
        """
        # 1. Chunk
        chunks = self.chunker.split(record)
        if not chunks:
            logger.warning("record_decision: no chunks generated — skipping.")
            return MemoryReference(memory_id="mem-empty")

        # 2. Embed
        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed(texts)

        # 3. Classify tags
        from .chunker import _classify_tags
        agent_tags = _classify_tags(record)

        # 4. Store
        memory_id = self.store.insert(record, chunks, embeddings, agent_tags)
        logger.info(
            "record_decision: stored %s (%d chunks, tags=%s).",
            memory_id, len(chunks), agent_tags,
        )

        return MemoryReference(memory_id=memory_id)

    def record_outcome(
        self,
        reference: MemoryReference,
        outcome: DecisionOutcome,
    ) -> None:
        """Update the decision with its realised outcome and a reflection chunk.

        If an LLM client was provided, generates a reflection before persisting.
        """
        memory_id = reference.memory_id
        if memory_id == "mem-empty":
            return

        # 1. Update outcome metadata on existing chunks
        self.store.update_outcome(memory_id, outcome)

        # 2. Retrieve original record context for reflection generation
        record_ctx = self.store.get_record_context(memory_id)
        if record_ctx is None:
            logger.warning("record_outcome: memory %s not found.", memory_id)
            return

        # 3. Generate reflection (best-effort)
        reflection_text = self._generate_reflection(record_ctx, outcome)
        if reflection_text:
            ref_chunk = self.chunker.build_reflection_chunk(
                _dummy_record(record_ctx), outcome, reflection_text,
            )
            if ref_chunk:
                ref_emb = self.embedder.embed_query(ref_chunk["content"])
                self.store.append_reflection_chunk(
                    memory_id=memory_id,
                    content=ref_chunk["content"],
                    embedding=ref_emb,
                    symbol=record_ctx["symbol"],
                    available_at=outcome.observed_at.isoformat(),
                )
                logger.info("record_outcome: reflection appended to %s.", memory_id)

    # ── Formatting helpers (for agent prompt injection) ──────────────────────

    def format_context_for_prompt(self, context: MemoryContext) -> str:
        """Convert a MemoryContext into a string suitable for agent prompts.

        This mirrors the output shape of ``TradingMemoryLog.get_past_context()``
        so existing agent code needs no changes to consume the enhanced memory.
        """
        if not context.items:
            return ""

        parts: list[str] = []
        if context.summary:
            parts.append(f"[Memory Summary] {context.summary}")

        parts.append("Relevant past decisions (most relevant first):")
        for item in context.items:
            date_str = (
                item.decision_at.strftime("%Y-%m-%d")
                if item.decision_at
                else "unknown date"
            )
            score_str = f"{item.score:.2f}" if item.score is not None else "n/a"
            tag_line = f"[{date_str} | {item.symbol} | relevance={score_str}]"
            parts.append(f"{tag_line}\n{item.content}")

        return "\n\n".join(parts)

    def format_all_for_state(
        self, contexts: dict[str, MemoryContext]
    ) -> dict[str, str]:
        """Convert per-agent MemoryContexts into prompt-ready strings.

        Keys are role names (e.g. ``"market_analyst"``); values are the
        formatted string for that agent's state field.

        Use in the graph like::

            state.update(provider.format_all_for_state(contexts))
        """
        return {
            f"memory_{role}": self.format_context_for_prompt(ctx)
            for role, ctx in contexts.items()
            if ctx.items
        }

    # ── Internal ────────────────────────────────────────────────────────────

    def _generate_reflection(
        self, record_ctx: dict, outcome: DecisionOutcome
    ) -> str:
        """Generate a concise reflection using the configured LLM."""
        if self._llm is None:
            return ""

        thesis = record_ctx.get("chunks", {}).get("thesis", "No thesis recorded.")
        ret_str = (
            f"{outcome.holding_period_return:+.2%}"
            if outcome.holding_period_return is not None
            else "unknown"
        )
        impact_str = (
            f"{outcome.portfolio_impact:+.2%}"
            if outcome.portfolio_impact is not None
            else "unknown"
        )

        messages = [
            ("system", _REFLECTION_PROMPT),
            (
                "human",
                (
                    f"Holding-period return: {ret_str}\n"
                    f"Portfolio impact: {impact_str}\n"
                    f"Max adverse move: {outcome.max_adverse_move}\n\n"
                    f"Original decision thesis:\n{thesis[:1200]}"
                ),
            ),
        ]

        try:
            return self._llm.invoke(messages).content
        except Exception:
            logger.exception("Reflection generation failed for memory — skipping.")
            return ""


def _dummy_record(record_ctx: dict) -> DecisionRecord:
    """Build a minimal DecisionRecord from stored context for the chunker."""
    symbol = record_ctx.get("symbol", "UNKNOWN")
    thesis = record_ctx.get("chunks", {}).get("thesis", "")
    market = record_ctx.get("chunks", {}).get("market_context", "")

    # Minimal required fields for the chunker's reflection builder
    from tradingagents.extensions.contracts import (
        DecisionRecord,
        MarketBar,
        MarketSnapshot,
        PortfolioState,
        TradeIntent,
    )

    now = datetime.now(timezone.utc)
    return DecisionRecord(
        intent=TradeIntent(
            decision_id=record_ctx.get("memory_id", "unknown"),
            symbol=symbol,
            as_of=now,
            target_weight=0.0,
            confidence=0.5,
            rationale=thesis,
        ),
        portfolio_before=PortfolioState(as_of=now, cash=0, total_equity=0),
        market_at_decision=MarketSnapshot(symbol=symbol, as_of=now),
    )
