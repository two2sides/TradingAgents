"""Reusable run observers that remain independent from Streamlit."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from tradingagents.extensions.contracts import RunEvent
from tradingagents.extensions.protocols import RunObserver


class EventCollector:
    """Collect progress events for tests, persistence, or later replay."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def on_event(self, event: RunEvent) -> None:
        self.events.append(event)


class CompositeRunObserver:
    """Fan one event out to multiple independent observers."""

    def __init__(self, observers: Iterable[RunObserver]) -> None:
        self._observers = tuple(observers)

    def on_event(self, event: RunEvent) -> None:
        for observer in self._observers:
            observer.on_event(event)


class LoggingRunObserver:
    """Write implementation-neutral run events to the configured terminal log."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or "-"
        self._logger = logging.getLogger(__name__)

    def on_event(self, event: RunEvent) -> None:
        level = logging.DEBUG if event.stage == "MARK_TO_MARKET" else logging.INFO
        self._logger.log(
            level,
            "run_event run_id=%s stage=%s at=%s progress=%s message=%s payload=%s",
            self.run_id,
            event.stage,
            event.timestamp.isoformat(),
            f"{event.progress:.3f}" if event.progress is not None else "-",
            event.message,
            event.payload or {},
        )


__all__ = ["CompositeRunObserver", "EventCollector", "LoggingRunObserver"]
