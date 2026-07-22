"""Session and resource helpers shared by Streamlit pages."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from tradingagents.extensions.paper_trading import SQLiteRunStore


@st.cache_resource(show_spinner=False)
def get_run_store() -> SQLiteRunStore:
    configured = os.getenv("TRADINGAGENTS_RUN_STORE")
    return SQLiteRunStore(Path(configured)) if configured else SQLiteRunStore()


def select_run(run_id: str) -> None:
    st.session_state["selected_run_id"] = run_id


def selected_run_id() -> str | None:
    return st.session_state.get("selected_run_id")


__all__ = ["get_run_store", "select_run", "selected_run_id"]
