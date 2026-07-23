"""Adapters that connect the public paper-trading ports to the default agent graph.

The default :class:`TradingAgentsGraph` returns a five-tier portfolio rating,
while the paper-trading broker executes a target weight.  This module owns that
translation explicitly so neither the graph nor the broker has to know about
the other implementation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any

from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating_strict
from tradingagents.extensions.contracts import (
    DecisionEnvelope,
    DecisionOutcome,
    DecisionRecord,
    DecisionRequest,
    MemoryContext,
    MemoryQuery,
    MemoryReference,
    TraceEvent,
    TradeIntent,
)

_RATINGS = frozenset(RATINGS_5_TIER)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    """Explainable result of translating one rating into one target weight."""

    rating: str
    current_weight: float
    target_weight: float
    diversification_cap: float
    rule: str


@dataclass(frozen=True, slots=True)
class RatingAllocationPolicy:
    """Convert C's five-tier rating into a long-only target weight.

    ``max_position_weight`` is an increase/entry cap, not a forced risk
    liquidation threshold.  A bullish rating therefore never sells an
    existing position merely because appreciation moved it above the cap.
    Bearish ratings never open a new position.
    """

    max_position_weight: float = 0.35
    overweight_fraction: float = 0.75
    underweight_fraction: float = 0.25
    version: str = "rating-allocation-v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("max_position_weight", self.max_position_weight),
            ("overweight_fraction", self.overweight_fraction),
            ("underweight_fraction", self.underweight_fraction),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.underweight_fraction >= self.overweight_fraction:
            raise ValueError("underweight_fraction must be less than overweight_fraction")

    def resolve(
        self,
        rating: str,
        *,
        current_weight: float,
        universe_size: int = 1,
    ) -> AllocationDecision:
        """Return a directional, diversified target for one canonical rating."""

        if rating not in _RATINGS:
            raise ValueError(f"unsupported portfolio rating {rating!r}")
        if not 0 <= current_weight <= 1:
            raise ValueError("current_weight must be between 0 and 1")
        if universe_size < 1:
            raise ValueError("universe_size must be at least 1")

        diversification_cap = min(self.max_position_weight, 1 / universe_size)
        if rating == "Buy":
            target = max(current_weight, diversification_cap)
            rule = "increase toward the diversified position cap"
        elif rating == "Overweight":
            target = max(current_weight, diversification_cap * self.overweight_fraction)
            rule = "increase toward the overweight allocation band"
        elif rating == "Hold":
            target = current_weight
            rule = "preserve the current position"
        elif rating == "Underweight":
            target = min(current_weight, diversification_cap * self.underweight_fraction)
            rule = "reduce toward the underweight allocation band"
        else:
            target = 0.0
            rule = "exit the position"

        return AllocationDecision(
            rating=rating,
            current_weight=current_weight,
            target_weight=min(1.0, max(0.0, target)),
            diversification_cap=diversification_cap,
            rule=rule,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe policy metadata for audits and the WebUI."""

        return asdict(self)


class _RequestMemoryBridge:
    """Expose A's already-retrieved context to graph agents without persisting twice."""

    def __init__(self, context: MemoryContext) -> None:
        self.context = context

    def retrieve(self, query: MemoryQuery) -> MemoryContext:
        items = [item for item in self.context.items if item.available_at <= query.as_of]
        metadata = dict(self.context.metadata)
        metadata.update(
            {
                "bridge": "paper-trading-request",
                "agent_role": query.metadata.get("agent_role", "portfolio_manager"),
            }
        )
        return MemoryContext(
            as_of=query.as_of,
            items=items[: query.limit],
            summary=self.context.summary,
            warnings=list(self.context.warnings),
            metadata=metadata,
        )

    def format_context_for_prompt(self, context: MemoryContext) -> str:
        parts: list[str] = []
        if context.summary:
            parts.append(f"[Memory Summary] {context.summary}")
        if context.items:
            parts.append("Relevant past decisions (most relevant first):")
        for item in context.items:
            date_text = item.decision_at.strftime("%Y-%m-%d")
            score_text = f"{item.score:.2f}" if item.score is not None else "n/a"
            parts.append(f"[{date_text} | {item.symbol} | relevance={score_text}]\n{item.content}")
        return "\n\n".join(parts)

    def record_decision(self, record: DecisionRecord) -> MemoryReference:
        return MemoryReference(memory_id=f"request-bridge-{record.intent.decision_id}")

    def record_outcome(
        self,
        reference: MemoryReference,
        outcome: DecisionOutcome,
    ) -> None:
        return None


class _NoOpGraphMemoryLog:
    """Disable graph-owned markdown memory during A-owned historical replay."""

    def get_pending_entries(self) -> list[dict[str, Any]]:
        return []

    def get_past_context(self, ticker: str, n_same: int = 5, n_cross: int = 3) -> str:
        return ""

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
    ) -> None:
        return None

    def update_with_outcome(self, **kwargs: Any) -> None:
        return None


class TradingAgentsGraphDecisionProvider:
    """Thin ``DecisionProvider`` adapter for the repository's default graph.

    The adapter makes A the sole owner of B's memory lifecycle:

    - A retrieves one time-safe :class:`MemoryContext`.
    - The graph receives a request-scoped, read-only view of that context.
    - Graph-internal markdown/RAG writes are disabled for the duration.
    - A records the final executable intent and its later outcome exactly once.

    Calls are serialized because a ``TradingAgentsGraph`` stores its active
    state on the instance and is not safe to mutate concurrently.
    """

    def __init__(
        self,
        graph: Any,
        policy: RatingAllocationPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.policy = policy or RatingAllocationPolicy()
        self._lock = RLock()

    def decide(self, request: DecisionRequest) -> DecisionEnvelope:
        """Run the graph and translate its strict final rating into an intent."""

        current_weight = request.portfolio.weight_for(request.symbol)
        try:
            with self._lock, self._bind_request_context(request.memory):
                final_state, graph_signal = self.graph.propagate(
                    request.symbol,
                    request.as_of.date().isoformat(),
                    asset_type=str(request.metadata.get("asset_type", "stock")),
                )
        except Exception as exc:
            return self._failed_safe(
                request,
                f"TradingAgents graph failed: {type(exc).__name__}: {exc}",
            )

        if not isinstance(final_state, dict):
            return self._failed_safe(request, "TradingAgents graph returned a non-mapping state")

        final_text = str(final_state.get("final_trade_decision") or "").strip()
        parsed = parse_rating_strict(final_text, expected_label="rating")
        if parsed.parsed is None:
            return self._failed_safe(
                request,
                f"Portfolio Manager rating was not explicit: {parsed.error}",
                diagnostics=self._diagnostics(final_state, graph_signal, None, parsed.source),
            )

        universe_size = _positive_int(request.metadata.get("universe_size"), default=1)
        allocation = self.policy.resolve(
            parsed.parsed,
            current_weight=current_weight,
            universe_size=universe_size,
        )
        warnings = list(request.memory.warnings)
        status = "SUCCESS"
        canonical_signal = str(graph_signal).strip().title()
        if canonical_signal not in _RATINGS or canonical_signal != parsed.parsed:
            status = "DEGRADED"
            warnings.append(
                "Graph signal disagreed with the explicit Portfolio Manager rating; "
                "the explicit rating was used."
            )

        decision_id = f"tradingagents-{request.symbol}-{request.as_of.isoformat()}"
        intent = TradeIntent(
            decision_id=decision_id,
            symbol=request.symbol,
            as_of=request.as_of,
            target_weight=allocation.target_weight,
            confidence=1.0,
            rationale=final_text,
            warnings=warnings,
            metadata={
                "provider": "TradingAgentsGraphDecisionProvider",
                "rating": parsed.parsed,
                "rating_parse_source": parsed.source,
                "allocation": asdict(allocation),
                "allocation_policy": self.policy.to_dict(),
                "confidence_semantics": "deterministic rating-translation integrity",
                "graph_run_id": final_state.get("run_id"),
            },
        )
        return DecisionEnvelope(
            intent=intent,
            status=status,
            trace=[
                TraceEvent(
                    timestamp=request.as_of,
                    source="TradingAgentsGraphDecisionProvider",
                    event_type="MEMORY_CONTEXT_INJECTED",
                    summary=f"Injected {len(request.memory.items)} time-safe memory items",
                    payload={"memory_items": len(request.memory.items)},
                ),
                TraceEvent(
                    timestamp=request.as_of,
                    source="RatingAllocationPolicy",
                    event_type="RATING_MAPPED",
                    summary=(
                        f"{parsed.parsed} mapped from {current_weight:.1%} "
                        f"to {allocation.target_weight:.1%}"
                    ),
                    payload=asdict(allocation),
                ),
            ],
            diagnostics=self._diagnostics(
                final_state,
                graph_signal,
                parsed.parsed,
                parsed.source,
            ),
        )

    @contextmanager
    def _bind_request_context(self, memory: MemoryContext) -> Iterator[None]:
        previous_provider = getattr(self.graph, "memory_provider", _MISSING)
        previous_log = getattr(self.graph, "memory_log", _MISSING)
        self.graph.memory_provider = _RequestMemoryBridge(memory)
        if previous_log is not _MISSING:
            self.graph.memory_log = _NoOpGraphMemoryLog()
        try:
            yield
        finally:
            if previous_provider is _MISSING:
                delattr(self.graph, "memory_provider")
            else:
                self.graph.memory_provider = previous_provider
            if previous_log is not _MISSING:
                self.graph.memory_log = previous_log

    @staticmethod
    def _diagnostics(
        final_state: dict[str, Any],
        graph_signal: Any,
        rating: str | None,
        parse_source: str,
    ) -> dict[str, Any]:
        return {
            "graph_signal": str(graph_signal),
            "rating": rating,
            "rating_parse_source": parse_source,
            "agent_reports": {
                "market": str(final_state.get("market_report") or ""),
                "fundamentals": str(final_state.get("fundamentals_report") or ""),
                "sentiment": str(final_state.get("sentiment_report") or ""),
                "news": str(final_state.get("news_report") or ""),
                "research_plan": str(final_state.get("investment_plan") or ""),
                "trader_plan": str(final_state.get("trader_investment_plan") or ""),
                "final_decision": str(final_state.get("final_trade_decision") or ""),
            },
            "decision_snapshots": _json_list(final_state.get("decision_snapshots")),
            "claims": _json_list(final_state.get("claims")),
            "audit_events": _json_list(final_state.get("audit_events")),
            "structured_invocations": _json_list(final_state.get("structured_invocations")),
        }

    @staticmethod
    def _failed_safe(
        request: DecisionRequest,
        message: str,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> DecisionEnvelope:
        current_weight = request.portfolio.weight_for(request.symbol)
        timestamp = request.as_of
        return DecisionEnvelope(
            intent=TradeIntent(
                decision_id=f"tradingagents-failed-safe-{request.symbol}-{timestamp.isoformat()}",
                symbol=request.symbol,
                as_of=timestamp,
                target_weight=current_weight,
                confidence=0.0,
                rationale="Preserve the current position because the agent result was unusable.",
                warnings=[message],
                metadata={
                    "provider": "TradingAgentsGraphDecisionProvider",
                    "confidence_semantics": "deterministic rating-translation integrity",
                },
            ),
            status="FAILED_SAFE",
            trace=[
                TraceEvent(
                    timestamp=timestamp,
                    source="TradingAgentsGraphDecisionProvider",
                    event_type="DECISION_FAILED_SAFE",
                    summary=message,
                )
            ],
            diagnostics=diagnostics or {"error": message},
        )


def _positive_int(value: Any, *, default: int) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError):
        return default
    return converted if converted > 0 else default


def _json_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


__all__ = [
    "AllocationDecision",
    "RatingAllocationPolicy",
    "TradingAgentsGraphDecisionProvider",
]
