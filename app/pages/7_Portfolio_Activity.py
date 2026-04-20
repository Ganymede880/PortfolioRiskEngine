"""
Portfolio Activity page for the CMCSIF Portfolio Tracker.

This page displays a unified activity feed built from:
- trade receipts
- reconciliation events
- cash ledger entries
- snapshot checkpoint events

It is designed to answer:
- what changed in the portfolio?
- when did it change?
- was it driven by a trade, reconciliation, or cash movement?
"""

from __future__ import annotations
from src.db.session import init_db

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.db import crud
from src.db.session import session_scope
from src.config.settings import settings
from src.utils.ui import apply_app_theme, left_align_dataframe


def _apply_portfolio_activity_filter_theme() -> None:
    return None


def _format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    return f"(${abs(numeric_value):,.2f})" if numeric_value < 0 else f"${numeric_value:,.2f}"


def _format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):,.2f}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _prepare_activity_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format activity feed for display.
    """
    if df.empty:
        return df

    display_df = df.copy()

    if "activity_date" in display_df.columns:
        display_df["activity_date"] = pd.to_datetime(
            display_df["activity_date"],
            errors="coerce",
        ).dt.date

    if "quantity" in display_df.columns:
        display_df["quantity"] = display_df["quantity"].map(_format_number)

    if "price" in display_df.columns:
        display_df["price"] = display_df["price"].map(_format_currency)

    if "cash_impact" in display_df.columns:
        display_df["cash_impact"] = display_df["cash_impact"].map(_format_currency)

    display_df = display_df.where(pd.notna(display_df), "None")

    return display_df


def _filter_activity(
    activity_df: pd.DataFrame,
    selected_team: str,
    selected_activity_types: list[str],
    ticker_search: str,
) -> pd.DataFrame:
    """
    Apply sidebar filters to the activity feed.
    """
    if activity_df.empty:
        return activity_df

    filtered = activity_df.copy()

    if selected_team != "All":
        filtered = filtered.loc[filtered["team"] == selected_team].copy()

    if selected_activity_types:
        filtered = filtered.loc[
            filtered["activity_type"].isin(selected_activity_types)
        ].copy()

    if ticker_search:
        filtered = filtered.loc[
            filtered["ticker"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.contains(ticker_search.upper(), na=False)
        ].copy()

    return filtered.reset_index(drop=True)


def _render_summary_metrics(activity_df: pd.DataFrame) -> None:
    """
    Render top-line summary cards for the filtered activity feed.
    """
    total_events = len(activity_df)

    trade_events = 0
    if "activity_type" in activity_df.columns:
        trade_events = int((activity_df["activity_type"] == "TRADE").sum())

    reconciliation_events = 0
    if "activity_type" in activity_df.columns:
        reconciliation_events = int(
            (activity_df["activity_type"] == "RECONCILIATION").sum()
        )

    total_cash_impact = 0.0
    if "cash_impact" in activity_df.columns:
        cash_metric_df = activity_df.copy()
        if "activity_type" in cash_metric_df.columns:
            cash_metric_df = cash_metric_df.loc[
                cash_metric_df["activity_type"].astype(str).eq("TRADE")
                | cash_metric_df["activity_type"].astype(str).str.startswith("CASH_")
            ].copy()
        total_cash_impact = float(
            pd.to_numeric(cash_metric_df["cash_impact"], errors="coerce").sum(skipna=True)
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Events", f"{total_events:,}")
    col2.metric("Trades", f"{trade_events:,}")
    col3.metric("Reconciliations", f"{reconciliation_events:,}")
    col4.metric("Net Cash Impact", _format_currency(total_cash_impact))


def _render_filters(activity_df: pd.DataFrame) -> tuple[str, list[str], str]:
    """
    Render sidebar filters and return selected values.
    """
    st.sidebar.header("Filters")

    teams = []
    if "team" in activity_df.columns:
        teams = (
            activity_df["team"]
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

    selected_team = st.sidebar.selectbox("Pod", options=team_options)

    activity_type_options = []
    if "activity_type" in activity_df.columns:
        activity_type_options = sorted(
            activity_df["activity_type"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    selected_activity_types = st.sidebar.multiselect(
        "Activity Types",
        options=activity_type_options,
        default=[],
    )

    ticker_search = st.sidebar.text_input("Search Ticker", value="").strip()

    return selected_team, selected_activity_types, ticker_search


def main() -> None:
    st.set_page_config(page_title="Portfolio Activity", layout="wide")
    apply_app_theme()
    _apply_portfolio_activity_filter_theme()
    init_db()
    st.title("Portfolio Activity")

    with session_scope() as session:
        activity_df = crud.load_portfolio_activity(session=session)

    if activity_df.empty:
        st.info("No portfolio activity is available yet. Upload snapshots or trade receipts to populate this page.")
        return

    selected_team, selected_activity_types, ticker_search = _render_filters(activity_df)

    filtered_df = _filter_activity(
        activity_df=activity_df,
        selected_team=selected_team,
        selected_activity_types=selected_activity_types,
        ticker_search=ticker_search,
    )

    _render_summary_metrics(filtered_df)
    st.divider()

    st.subheader("Activity Feed")

    if filtered_df.empty:
        st.warning("No activity matched the current filters.")
        return

    display_df = _prepare_activity_display(filtered_df)
    display_df = display_df.rename(
        columns={
            "activity_date": "ACTIVITY DATE",
            "activity_type": "ACTIVITY TYPE",
            "team": "POD",
            "ticker": "TICKER",
            "position_side": "POSITION SIDE",
            "quantity": "QUANTITY",
            "price": "PRICE",
            "cash_impact": "CASH IMPACT",
            "note": "NOTE",
            "source_file": "SOURCE FILE",
        }
    )

    preferred_columns = [
        "ACTIVITY DATE",
        "ACTIVITY TYPE",
        "POD",
        "TICKER",
        "POSITION SIDE",
        "QUANTITY",
        "PRICE",
        "CASH IMPACT",
        "NOTE",
        "SOURCE FILE",
    ]
    display_columns = [col for col in preferred_columns if col in display_df.columns]

    st.dataframe(
        left_align_dataframe(display_df[display_columns]),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
