"""
CMCSIF Portfolio Tracker - Main App Entry Point

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
    st.title("Fund Dashboard Home")
    st.markdown(
        """
        Welcome to the **Claremont McKenna Student Investment Fund Dashboard**.

        This platform provides:
        - Real-time portfolio tracking
        - Team-level performance visibility
        - Trade and activity monitoring
        - Centralized portfolio data management
        """
    )
    st.divider()


def render_navigation_guide() -> None:
    st.subheader("NAVIGATION GUIDE")
    st.markdown(
        """
        **Core Pages:**

        - **Total Fund View**  
        View total portfolio AUM, allocation across teams, and top holdings.

        - **Team View**  
        Drill into each sector sleeve like Healthcare, TMT, and Consumer.

        - **Holdings**  
        Review the current portfolio with filters and position-level detail.

        - **Portfolio Activity**  
        Track trades, reconciliations, and cash movements over time.

        - **Upload**  
        Upload portfolio snapshots and trade receipts.
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
        Go to **Upload** and add the latest file from the endowment office.

        **Step 2:** Upload Trade Receipts  
        Add monthly trades as they occur.

        **Step 3:** Explore the Dashboard  
        Monitor performance in **Total Fund View** and **Team View**.
        """
    )
    st.divider()


def render_notes() -> None:
    with st.expander("NOTES ON CURRENT VERSION"):
        st.markdown(
            """
            - Portfolio state is reconstructed from snapshots plus trades
            - Historical AUM charts currently reflect current holdings backfilled with prices
            - Full historical position tracking will be added next
            """
        )


def main() -> None:
    st.set_page_config(
        page_title="Fund Dashboard Home",
        layout="wide",
    )

    apply_app_theme()
    init_db()

    render_header()
    render_navigation_guide()
    render_system_status()
    render_quick_start()
    render_notes()


if __name__ == "__main__":
    main()
