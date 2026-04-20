"""
Dashboard Home - Main App Entry Point

This is the landing page for the Streamlit app.

It serves as:
- a homepage for users
- navigation guide for fund members
- high-level system status overview
"""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.db.crud import load_portfolio_activity, load_position_state
from src.db.session import init_db, session_scope
from src.utils.ui import apply_app_theme


def render_header() -> None:
    st.title("Dashboard Home")
    st.markdown(
        """
        Welcome to the Claremont McKenna Student Investment Fund Dashboard.
        """
    )
    st.divider()


def render_system_status() -> None:
    st.subheader("SYSTEM STATUS")

    with session_scope() as session:
        position_state = load_position_state(session)
        activity = load_portfolio_activity(session)

    col1, col2 = st.columns(2)
    col1.metric(
        "Active Positions",
        f"{len(position_state):,}" if not position_state.empty else "0",
    )
    col2.metric(
        "Activity Records",
        f"{len(activity):,}" if not activity.empty else "0",
    )

    if position_state.empty:
        st.warning(
            "No portfolio data loaded. Start by uploading a **Portfolio Snapshot**."
        )
    else:
        st.success("Portfolio data loaded successfully.")

    st.divider()


def render_quick_start() -> None:
    st.subheader("QUICK START")
    st.markdown(
        """
        **Step 1:** Upload a Portfolio Snapshot  
        Go to **Upload** and add the latest portfolio snapshot.

        **Step 2:** Upload Trade Receipts  
        Add trade receipts as they occur to keep portfolio snapshot up-to-date.

        **Step 3:** Explore the Dashboard  
        Monitor performance in Portfolio View and Sector View.
        """
    )
    st.divider()


def render_notes() -> None:
    with st.expander("NOTES ON CURRENT VERSION"):
        st.markdown(
            """
            - Portfolio state is reconstructed from snapshots plus trades
            - This website is maintained by Kiefer Tierling. Contact me at kiefer.tierling@gmail.com
            in the event of any bugs or outages.
            """
        )


def main() -> None:
    st.set_page_config(
        page_title="Dashboard Home",
        layout="wide",
    )

    apply_app_theme()
    init_db()

    render_header()
    render_system_status()
    render_quick_start()
    render_notes()


if __name__ == "__main__":
    main()
