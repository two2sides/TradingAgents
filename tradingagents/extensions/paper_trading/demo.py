"""Deterministic providers for offline tests and the no-key WebUI demo."""

from __future__ import annotations

from tradingagents.extensions.contracts import (
    DecisionEnvelope,
    DecisionOutcome,
    DecisionRecord,
    DecisionRequest,
    MemoryContext,
    MemoryItem,
    MemoryQuery,
    MemoryReference,
    TraceEvent,
    TradeIntent,
)


class DemoMemoryProvider:
    """Small in-memory B-compatible provider used only by demo mode."""

    def __init__(self) -> None:
        self.records: dict[str, DecisionRecord] = {}
        self.outcomes: dict[str, DecisionOutcome] = {}

    def retrieve(self, query: MemoryQuery) -> MemoryContext:
        items: list[MemoryItem] = []
        for memory_id, record in reversed(self.records.items()):
            outcome = self.outcomes.get(memory_id)
            if record.intent.symbol != query.symbol or outcome is None:
                continue
            if outcome.observed_at > query.as_of:
                continue
            realized = outcome.holding_period_return
            realized_text = "unknown" if realized is None else f"{realized:+.2%}"
            items.append(
                MemoryItem(
                    memory_id=memory_id,
                    symbol=query.symbol,
                    decision_at=record.intent.as_of,
                    available_at=outcome.observed_at,
                    content=(
                        f"目标仓位 {record.intent.target_weight:.0%}；"
                        f"持有期结果 {realized_text}。{record.intent.rationale}"
                    ),
                    score=min(1.0, abs(realized or 0)),
                    metadata={"holding_period_return": realized},
                )
            )
            if len(items) >= query.limit:
                break
        return MemoryContext(
            as_of=query.as_of,
            items=items,
            summary=f"找到 {len(items)} 条在当前时点已经揭晓结果的历史决策。",
            metadata={"provider": "demo-memory"},
        )

    def record_decision(self, record: DecisionRecord) -> MemoryReference:
        memory_id = f"demo-memory-{len(self.records) + 1}"
        self.records[memory_id] = record
        return MemoryReference(memory_id=memory_id)

    def record_outcome(self, reference: MemoryReference, outcome: DecisionOutcome) -> None:
        if reference.memory_id not in self.records:
            raise KeyError(f"unknown memory reference {reference.memory_id}")
        self.outcomes[reference.memory_id] = outcome


class MovingAverageDecisionProvider:
    """Explainable, deterministic C-compatible provider for demo mode.

    It is intentionally not presented as the team's final decision algorithm.
    Its purpose is to make A's engine and UI fully runnable before C's provider
    is available.
    """

    def __init__(
        self,
        *,
        fast_window: int = 3,
        slow_window: int = 8,
        bullish_weight: float = 0.8,
        neutral_weight: float = 0.35,
        bearish_weight: float = 0.05,
    ) -> None:
        if fast_window < 1 or slow_window <= fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.bullish_weight = bullish_weight
        self.neutral_weight = neutral_weight
        self.bearish_weight = bearish_weight

    def decide(self, request: DecisionRequest) -> DecisionEnvelope:
        closes = [bar.close for bar in request.market.bars]
        fast_values = closes[-self.fast_window :]
        slow_values = closes[-self.slow_window :]
        fast = sum(fast_values) / len(fast_values)
        slow = sum(slow_values) / len(slow_values)
        momentum = fast / slow - 1 if slow else 0

        if momentum > 0.01:
            target, regime = self.bullish_weight, "BULLISH"
        elif momentum < -0.01:
            target, regime = self.bearish_weight, "BEARISH"
        else:
            target, regime = self.neutral_weight, "NEUTRAL"
        confidence = min(0.95, 0.55 + abs(momentum) * 8)
        status = "SUCCESS" if len(closes) >= self.slow_window else "DEGRADED"
        warnings = [] if status == "SUCCESS" else ["历史窗口不足，演示策略使用现有全部 K 线。"]

        intent = TradeIntent(
            decision_id=f"demo-{request.symbol}-{request.as_of.isoformat()}",
            symbol=request.symbol,
            as_of=request.as_of,
            target_weight=target,
            confidence=confidence,
            rationale=(
                f"演示均线状态为 {regime}：fast={fast:.2f}，slow={slow:.2f}，动量={momentum:+.2%}。"
            ),
            warnings=warnings,
            metadata={"provider": "moving-average-demo", "regime": regime},
        )
        return DecisionEnvelope(
            intent=intent,
            status=status,
            trace=[
                TraceEvent(
                    timestamp=request.as_of,
                    source="MovingAverageDecisionProvider",
                    event_type="SIGNAL_COMPUTED",
                    summary=f"{regime} regime selected target weight {target:.0%}",
                    payload={
                        "fast_average": fast,
                        "slow_average": slow,
                        "momentum": momentum,
                        "memory_items": len(request.memory.items),
                    },
                )
            ],
            diagnostics={"fast": fast, "slow": slow, "momentum": momentum},
        )


__all__ = ["DemoMemoryProvider", "MovingAverageDecisionProvider"]
