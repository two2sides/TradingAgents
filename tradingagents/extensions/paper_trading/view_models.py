"""UI-neutral view models for decision audit and replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradingagents.extensions.contracts import (
    BacktestResult,
    DecisionEnvelope,
    ExecutionReport,
    MarketSnapshot,
    MemoryContext,
    PortfolioState,
)


@dataclass(frozen=True, slots=True)
class DecisionReplayItem:
    """All public evidence needed to explain one saved decision."""

    decision: DecisionEnvelope
    execution: ExecutionReport | None
    market: MarketSnapshot | None
    memory: MemoryContext | None
    portfolio_before: PortfolioState | None
    portfolio_after: PortfolioState | None
    ledger_entries: tuple[dict[str, Any], ...]


def build_decision_replay(result: BacktestResult) -> list[DecisionReplayItem]:
    """Join normalized result fields with saved audit metadata by decision ID."""

    executions = {item.decision_id: item for item in result.executions}
    raw_contexts = result.metadata.get("decision_contexts", {})
    raw_ledger = result.metadata.get("ledger_entries", [])
    items: list[DecisionReplayItem] = []

    for decision in result.decisions:
        decision_id = decision.intent.decision_id
        context = raw_contexts.get(decision_id, {}) if isinstance(raw_contexts, dict) else {}
        execution = executions.get(decision_id)
        effective_at = decision.intent.as_of
        if execution is not None and execution.fills:
            effective_at = execution.fills[-1].timestamp
        portfolio_after = next(
            (
                portfolio
                for portfolio in result.portfolio_history
                if portfolio.as_of >= effective_at
            ),
            None,
        )
        matching_entries = tuple(
            entry
            for entry in raw_ledger
            if isinstance(entry, dict) and entry.get("decision_id") == decision_id
        )
        items.append(
            DecisionReplayItem(
                decision=decision,
                execution=execution,
                market=(
                    MarketSnapshot.model_validate(context["market"])
                    if context.get("market")
                    else None
                ),
                memory=(
                    MemoryContext.model_validate(context["memory"])
                    if context.get("memory")
                    else None
                ),
                portfolio_before=(
                    PortfolioState.model_validate(context["portfolio_before"])
                    if context.get("portfolio_before")
                    else None
                ),
                portfolio_after=portfolio_after,
                ledger_entries=matching_entries,
            )
        )
    return items


__all__ = ["DecisionReplayItem", "build_decision_replay"]
