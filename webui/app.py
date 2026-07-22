"""TradingAgents Decision Lab Streamlit entry point."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="TradingAgents · Decision Lab",
        page_icon="◈",
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


if __name__ == "__main__":
    main()
