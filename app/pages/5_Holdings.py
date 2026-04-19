"""
Holdings page for the CMCSIF Portfolio Tracker (v2).

This page is built on reconstructed position state and supports:
- current holdings detail
- team filter
- long/short/cash filter
- ticker search
- summary metrics
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on the Python path when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices
from src.db.crud import load_position_state
from src.db.session import session_scope
from src.utils.ui import apply_app_theme, left_align_dataframe


COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
COL_PRICE = "price"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"


def _format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def _format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_holdings_snapshot() -> pd.DataFrame:
    """
    Load reconstructed latest position state and attach latest prices.
    """
    with session_scope() as session:
        position_state_df = load_position_state(session)

    if position_state_df.empty:
        return pd.DataFrame()

    tickers = (
        position_state_df[COL_TICKER]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("CASH")]
        .unique()
        .tolist()
    )

    latest_prices_df, _ = fetch_latest_prices(tickers)
    snapshot_df = build_current_portfolio_snapshot(position_state_df, latest_prices_df)

    return snapshot_df


def render_empty_state() -> None:
    st.title("Holdings")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def _apply_holdings_filter_theme() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSelectbox label p,
        [data-testid="stSidebar"] .stTextInput label,
        [data-testid="stSidebar"] .stTextInput label p,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label p,
        [data-testid="stSidebar"] [data-testid="stTextInput"] label,
        [data-testid="stSidebar"] [data-testid="stTextInput"] label p {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div > div,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] p,
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] [role="combobox"],
        [data-testid="stSidebar"] [role="listbox"],
        [data-testid="stSidebar"] [role="option"],
        [data-testid="stSidebar"] [data-baseweb="select"] [aria-selected="true"],
        [data-testid="stSidebar"] [data-testid="stTextInput"] input,
        [data-testid="stSidebar"] input {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [role="combobox"],
        [data-testid="stSidebar"] [data-testid="stTextInput"] input,
        [data-testid="stSidebar"] input {
            background: #f8fafc !important;
            border-color: #cbd5e1 !important;
        }

        [data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder,
        [data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
        }

        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] div,
        div[data-baseweb="popover"] [role="option"] span,
        div[data-baseweb="popover"] [role="option"] p {
            background: #f8fafc !important;
            color: #0f172a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_filters(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    teams = (
        snapshot_df[COL_TEAM]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )

    ordered_teams = [team for team in settings.display_team_order if team in teams]
    fallback_teams = [team for team in teams if team not in ordered_teams]
    team_options = ["All"] + ordered_teams + sorted(fallback_teams)

    side_options = ["All", "LONG", "SHORT", "CASH"]

    selected_team = st.sidebar.selectbox("Team", options=team_options)
    selected_side = st.sidebar.selectbox("Position Side", options=side_options)
    ticker_search = st.sidebar.text_input("Search Ticker", value="").strip().upper()

    filtered = snapshot_df.copy()

    if selected_team != "All":
        filtered = filtered.loc[filtered[COL_TEAM] == selected_team].copy()

    if selected_side != "All":
        filtered = filtered.loc[filtered[COL_POSITION_SIDE] == selected_side].copy()

    if ticker_search:
        filtered = filtered.loc[
            filtered[COL_TICKER]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.contains(ticker_search, na=False)
        ].copy()

    return filtered.reset_index(drop=True)


def render_summary_metrics(filtered_df: pd.DataFrame) -> None:
    total_market_value = float(
        pd.to_numeric(filtered_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)
    ) if not filtered_df.empty else 0.0

    gross_exposure = float(
        pd.to_numeric(filtered_df[COL_MARKET_VALUE], errors="coerce").abs().sum(skipna=True)
    ) if not filtered_df.empty else 0.0

    unique_tickers = int(filtered_df[COL_TICKER].nunique(dropna=True)) if not filtered_df.empty else 0
    unique_teams = int(filtered_df[COL_TEAM].nunique(dropna=True)) if not filtered_df.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtered Market Value", _format_currency(total_market_value))
    c2.metric("Gross Exposure", _format_currency(gross_exposure))
    c3.metric("Tickers", f"{unique_tickers:,}")
    c4.metric("Teams", f"{unique_teams:,}")


def render_holdings_table(filtered_df: pd.DataFrame) -> None:
    st.subheader("Holdings Detail")

    if filtered_df.empty:
        st.warning("No holdings matched the current filters.")
        return

    display_cols = [
        col for col in [
            COL_TICKER,
            COL_TEAM,
            COL_POSITION_SIDE,
            COL_SHARES,
            COL_PRICE,
            COL_MARKET_VALUE,
            COL_WEIGHT,
        ]
        if col in filtered_df.columns
    ]

    display_df = filtered_df[display_cols].copy().sort_values(
        by=[COL_TEAM, COL_MARKET_VALUE],
        ascending=[True, False],
    )

    if COL_SHARES in display_df.columns:
        display_df[COL_SHARES] = display_df[COL_SHARES].map(_format_number)
    if COL_PRICE in display_df.columns:
        display_df[COL_PRICE] = display_df[COL_PRICE].map(_format_currency)
    if COL_MARKET_VALUE in display_df.columns:
        display_df[COL_MARKET_VALUE] = display_df[COL_MARKET_VALUE].map(_format_currency)
    if COL_WEIGHT in display_df.columns:
        display_df[COL_WEIGHT] = display_df[COL_WEIGHT].map(_format_percent)

    st.dataframe(left_align_dataframe(display_df), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Holdings", layout="wide")
    apply_app_theme()
    _apply_holdings_filter_theme()
    st.title("Holdings")

    snapshot_df = get_holdings_snapshot()

    if snapshot_df.empty:
        render_empty_state()
        return

    filtered_df = render_filters(snapshot_df)
    render_summary_metrics(filtered_df)
    st.divider()
    render_holdings_table(filtered_df)


if __name__ == "__main__":
    main()
