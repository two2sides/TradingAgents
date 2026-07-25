"""Streamlit adapter for the implementation-neutral RunObserver protocol."""

from __future__ import annotations

import streamlit as st

from tradingagents.extensions.contracts import RunEvent


class StreamlitProgressObserver:
    def __init__(self) -> None:
        self._progress = st.progress(0.0, text="Preparing replay…")
        self._status = st.empty()
        self._log = st.empty()
        self._lines: list[str] = []

    def on_event(self, event: RunEvent) -> None:
        if event.progress is not None:
            self._progress.progress(
                event.progress,
                text=f"{event.stage.replace('_', ' ').title()} · {event.message}",
            )
        self._status.caption(f"{event.timestamp:%Y-%m-%d} · {event.stage} · {event.message}")
        self._lines.append(f"{event.timestamp:%Y-%m-%d}  {event.stage:<16} {event.message}")
        self._log.code("\n".join(self._lines[-8:]), language=None)


__all__ = ["StreamlitProgressObserver"]
