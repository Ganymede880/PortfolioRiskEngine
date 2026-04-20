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
from src.utils.ui import apply_app_theme, render_top_nav


def render_graphic_pattern() -> None:
    st.markdown(
        """
        <div style="
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            min-height: 220px;
            margin: 0.35rem 0 1.4rem 0;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background:
                radial-gradient(circle at 18% 28%, rgba(45, 194, 189, 0.22), transparent 24%),
                radial-gradient(circle at 78% 24%, rgba(122, 130, 171, 0.24), transparent 26%),
                radial-gradient(circle at 62% 72%, rgba(18, 102, 79, 0.26), transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.9));
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.22);
        ">
            <div style="
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(rgba(148, 163, 184, 0.10) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148, 163, 184, 0.10) 1px, transparent 1px);
                background-size: 34px 34px;
                mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.18));
                -webkit-mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.18));
            "></div>
            <div style="
                position: absolute;
                width: 380px;
                height: 380px;
                right: -82px;
                top: -108px;
                border-radius: 50%;
                border: 1px solid rgba(191, 219, 254, 0.16);
                box-shadow:
                    0 0 0 38px rgba(191, 219, 254, 0.06),
                    0 0 0 96px rgba(45, 194, 189, 0.06);
            "></div>
            <div style="
                position: absolute;
                width: 220px;
                height: 220px;
                left: 9%;
                bottom: -118px;
                transform: rotate(18deg);
                border-radius: 36px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02));
            "></div>
            <div style="
                position: relative;
                z-index: 2;
                padding: 1.6rem 1.8rem;
                max-width: 640px;
                color: #E2E8F0;
            ">
                <div style="
                    display: inline-block;
                    padding: 0.32rem 0.72rem;
                    margin-bottom: 0.8rem;
                    border-radius: 999px;
                    background: rgba(15, 118, 110, 0.20);
                    border: 1px solid rgba(94, 234, 212, 0.20);
                    font-size: 0.8rem;
                    letter-spacing: 0.08em;
                    font-weight: 700;
                ">CMCSIF PORTFOLIO TRACKER</div>
                <div style="
                    font-size: 2rem;
                    line-height: 1.08;
                    font-weight: 700;
                    margin-bottom: 0.55rem;
                ">A live map of the fund, built for quick reads and effective decisions.</div>
                <div style="
                    font-size: 1rem;
                    line-height: 1.55;
                    color: rgba(226, 232, 240, 0.86);
                    max-width: 540px;
                ">
                    Explore exposures, attribution, risk, holdings, and activity from one place.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )





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

def render_notes() -> None:
    with st.expander("NOTES ON CURRENT VERSION"):
        st.markdown(
            """
            - Portfolio state is reconstructed from snapshots plus trades
            - This website is maintained by Kiefer Tierling. Contact me at kiefer.tierling@gmail.com
            in the event of any outages.
            """
        )


def main() -> None:
    st.set_page_config(layout="wide")

    apply_app_theme()
    render_top_nav()

    init_db()

    render_graphic_pattern()
    render_system_status()
    render_notes()


if __name__ == "__main__":
    main()
