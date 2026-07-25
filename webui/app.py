"""TradingAgents Decision Lab Streamlit entry point."""

from __future__ import annotations

import logging
from time import monotonic

import streamlit as st
from dotenv import load_dotenv

from webui.logging_config import configure_terminal_logging

load_dotenv()
configure_terminal_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    started_at = monotonic()
    logger.debug("Streamlit script rerun started")
    st.set_page_config(
        page_title="TradingAgents · Decision Lab",
        page_icon=":material/finance_mode:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "TradingAgents Decision Lab · auditable paper trading and strategy replay",
        },
    )

    from webui.components.style import apply_global_style

    apply_global_style()
    with st.sidebar:
        st.markdown(
            """
            <div class="ta-brand">
              <div class="ta-brand-mark">TRADINGAGENTS</div>
              <div class="ta-brand-name">Decision Lab</div>
              <div class="ta-brand-sub">Replay · Audit · Compare</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    page = st.navigation(
        {
            "WORKSPACE": [
                st.Page(
                    "views/overview.py",
                    title="Overview",
                    icon=":material/grid_view:",
                    url_path="overview",
                    default=True,
                ),
                st.Page(
                    "views/run.py",
                    title="Run Experiment",
                    icon=":material/play_circle:",
                    url_path="run",
                ),
                st.Page(
                    "views/replay.py",
                    title="Decision Replay",
                    icon=":material/manage_search:",
                    url_path="replay",
                ),
                st.Page(
                    "views/compare.py",
                    title="Compare Runs",
                    icon=":material/compare_arrows:",
                    url_path="compare",
                ),
            ]
        }
    )
    page.run()
    with st.sidebar:
        st.caption("LOCAL-FIRST · SQLITE ARCHIVE")
        st.caption("Execution policy · NEXT_OPEN")
    logger.debug(
        "Streamlit script rerun rendered duration_ms=%.0f", (monotonic() - started_at) * 1000
    )


if __name__ == "__main__":
    main()
