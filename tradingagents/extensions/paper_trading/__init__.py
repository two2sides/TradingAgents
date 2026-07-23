"""A-owned paper-trading implementations with lazy public exports.

The package is imported by every WebUI page.  Keeping this module lightweight
prevents optional Agent, LangChain, Transformers, and Torch dependencies from
being initialized before the user starts a real Agent experiment.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AccountLedger": ("ledger", "AccountLedger"),
    "AllocationDecision": ("integrations", "AllocationDecision"),
    "BacktestApplicationService": ("service", "BacktestApplicationService"),
    "CompositeRunObserver": ("observers", "CompositeRunObserver"),
    "DemoMemoryProvider": ("demo", "DemoMemoryProvider"),
    "DecisionReplayItem": ("view_models", "DecisionReplayItem"),
    "EventCollector": ("observers", "EventCollector"),
    "HistoricalBacktestRunner": ("backtest", "HistoricalBacktestRunner"),
    "HistoricalMarketDataProvider": ("market_data", "HistoricalMarketDataProvider"),
    "LedgerBroker": ("broker", "LedgerBroker"),
    "LedgerEntry": ("ledger", "LedgerEntry"),
    "LoggingRunObserver": ("observers", "LoggingRunObserver"),
    "MarketDataRateLimited": ("market_data", "MarketDataRateLimited"),
    "MarketDataUnavailable": ("market_data", "MarketDataUnavailable"),
    "MovingAverageDecisionProvider": ("demo", "MovingAverageDecisionProvider"),
    "RatingAllocationPolicy": ("integrations", "RatingAllocationPolicy"),
    "ReplayDecisionProvider": ("replay", "ReplayDecisionProvider"),
    "RunStoreObserver": ("storage", "RunStoreObserver"),
    "SQLiteRunStore": ("storage", "SQLiteRunStore"),
    "TradingAgentsGraphDecisionProvider": (
        "integrations",
        "TradingAgentsGraphDecisionProvider",
    ),
    "build_decision_replay": ("view_models", "build_decision_replay"),
    "generate_demo_market_data": ("demo", "generate_demo_market_data"),
    "run_execution_what_if": ("replay", "run_execution_what_if"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public implementation only when a caller actually needs it."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
