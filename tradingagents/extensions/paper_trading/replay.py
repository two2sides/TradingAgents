"""Decision reuse for clearly-labelled execution-layer what-if experiments."""

from __future__ import annotations

from tradingagents.extensions.contracts import (
    BacktestRequest,
    BacktestResult,
    DecisionEnvelope,
    DecisionRequest,
    ExecutionConfig,
    MarketBar,
    TraceEvent,
)
from tradingagents.extensions.protocols import RunObserver

from .backtest import HistoricalBacktestRunner
from .demo import DemoMemoryProvider
from .market_data import HistoricalMarketDataProvider, as_utc


class ReplayDecisionProvider:
    """Return saved targets at their original symbol and decision timestamp."""

    def __init__(self, decisions: list[DecisionEnvelope]) -> None:
        self._decisions = {
            (item.intent.symbol, as_utc(item.intent.as_of)): item for item in decisions
        }
        if len(self._decisions) != len(decisions):
            raise ValueError("saved decisions must have unique symbol/as_of pairs")

    def decide(self, request: DecisionRequest) -> DecisionEnvelope:
        key = (request.symbol, as_utc(request.as_of))
        try:
            original = self._decisions[key]
        except KeyError as exc:
            raise LookupError(f"no saved decision for {request.symbol} at {request.as_of}") from exc
        intent = original.intent.model_copy(
            update={
                "decision_id": f"what-if:{original.intent.decision_id}",
                "rationale": f"[执行层 What-if 复用] {original.intent.rationale}",
                "metadata": {
                    **original.intent.metadata,
                    "reused_decision_id": original.intent.decision_id,
                },
            }
        )
        return DecisionEnvelope(
            intent=intent,
            status=original.status,
            trace=[
                *original.trace,
                TraceEvent(
                    timestamp=request.as_of,
                    source="ReplayDecisionProvider",
                    event_type="DECISION_REUSED",
                    summary="Reused the original target weight without a new Agent call.",
                    payload={"original_decision_id": original.intent.decision_id},
                ),
            ],
            diagnostics={**original.diagnostics, "execution_what_if": True},
        )


def run_execution_what_if(
    original_request: BacktestRequest,
    original_result: BacktestResult,
    *,
    execution: ExecutionConfig | None = None,
    initial_cash: float | None = None,
    parent_run_id: str | None = None,
    observer: RunObserver | None = None,
) -> tuple[BacktestRequest, BacktestResult]:
    """Re-run only account execution while preserving saved Agent targets."""

    raw_bars = original_result.metadata.get("market_bars")
    if not isinstance(raw_bars, dict) or not raw_bars:
        raise ValueError("original result does not contain replayable market_bars")
    bars = {
        symbol: [MarketBar.model_validate(item) for item in symbol_bars]
        for symbol, symbol_bars in raw_bars.items()
    }
    provider = HistoricalMarketDataProvider(bars, source="saved-run")
    request_data = original_request.model_dump()
    request_data.update(
        {
            "execution": execution or original_request.execution,
            "initial_cash": (
                original_request.initial_cash if initial_cash is None else initial_cash
            ),
        }
    )
    request = BacktestRequest.model_validate(request_data)
    replayed = HistoricalBacktestRunner(provider).run(
        request,
        ReplayDecisionProvider(original_result.decisions),
        DemoMemoryProvider(),
        observer,
    )
    result = replayed.model_copy(
        update={
            "metadata": {
                **replayed.metadata,
                "run_kind": "EXECUTION_WHAT_IF",
                "parent_run_id": parent_run_id,
                "agent_calls_reused": len(original_result.decisions),
                "what_if_disclaimer": (
                    "Targets were reused from the parent run; this is not a full Agent rerun."
                ),
            }
        }
    )
    return request, result


__all__ = ["ReplayDecisionProvider", "run_execution_what_if"]
