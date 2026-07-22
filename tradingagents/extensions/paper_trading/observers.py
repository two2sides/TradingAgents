"""Reusable run observers that remain independent from Streamlit."""

from __future__ import annotations

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


__all__ = ["CompositeRunObserver", "EventCollector"]
