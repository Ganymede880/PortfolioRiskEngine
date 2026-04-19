"""
Factor Exposure page for the CMCSIF Portfolio Tracker (v2).

MVP implementation:
- built on reconstructed position state
- size proxy via log market cap
- value proxy via inverse price-to-book
- momentum proxy via trailing 252-day return

This remains a rough internal proxy dashboard, not a full risk model.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import load_position_state
from src.db.session import session_scope
from src.utils.ui import apply_app_theme, left_align_dataframe, style_plotly_figure


COL_TEAM = "team"
COL_TICKER = "ticker"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"
COL_POSITION_SIDE = "position_side"

FACTOR_SIZE = "size"
FACTOR_VALUE = "value"
FACTOR_MOMENTUM = "momentum"


def _format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def _format_number(value, decimals: int = 2):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_base_snapshot() -> pd.DataFrame:
    """
    Load latest reconstructed position state and attach live prices.
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


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_factor_inputs(tickers: list[str]) -> pd.DataFrame:
    """
    Pull Yahoo-derived factor inputs.
    """
    rows = []

    price_history_df = fetch_multiple_price_histories(tickers, lookback_days=400)

    momentum_map: dict[str, float] = {}
    if not price_history_df.empty:
        working = price_history_df.copy()
        working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
        working = working.sort_values(["ticker", "date"])

        for ticker, group in working.groupby("ticker"):
            group = group.dropna(subset=["adj_close"]).reset_index(drop=True)
            if len(group) >= 252:
                start_price = group["adj_close"].iloc[-252]
                end_price = group["adj_close"].iloc[-1]

                if pd.notna(start_price) and pd.notna(end_price) and start_price != 0:
                    momentum_map[ticker] = (end_price / start_price) - 1.0
                else:
                    momentum_map[ticker] = np.nan
            else:
                momentum_map[ticker] = np.nan

    for ticker in tickers:
        market_cap = np.nan
        price_to_book = np.nan

        try:
            info = yf.Ticker(ticker).info
            market_cap = info.get("marketCap", np.nan)
            price_to_book = info.get("priceToBook", np.nan)
        except Exception:
            pass

        log_market_cap = np.log(market_cap) if pd.notna(market_cap) and market_cap > 0 else np.nan

        rows.append({
            COL_TICKER: ticker,
            "market_cap": market_cap,
            "log_market_cap": log_market_cap,
            "price_to_book": price_to_book,
            "momentum_252d": momentum_map.get(ticker, np.nan),
        })

    return pd.DataFrame(rows)


def build_exposure_snapshot(snapshot_df: pd.DataFrame, factor_inputs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge current snapshot with factor inputs and compute factor proxies.
    """
    if snapshot_df.empty:
        return pd.DataFrame()

    investable_df = snapshot_df.loc[snapshot_df[COL_TICKER] != "CASH"].copy()

    merged = investable_df.merge(
        factor_inputs_df,
        on=COL_TICKER,
        how="left",
    )

    merged[COL_WEIGHT] = pd.to_numeric(merged[COL_WEIGHT], errors="coerce")
    merged["log_market_cap"] = pd.to_numeric(merged["log_market_cap"], errors="coerce")
    merged["price_to_book"] = pd.to_numeric(merged["price_to_book"], errors="coerce")
    merged["momentum_252d"] = pd.to_numeric(merged["momentum_252d"], errors="coerce")

    merged[FACTOR_SIZE] = merged["log_market_cap"]
    merged[FACTOR_VALUE] = np.where(
        merged["price_to_book"].notna() & (merged["price_to_book"] != 0),
        1.0 / merged["price_to_book"],
        np.nan,
    )
    merged[FACTOR_MOMENTUM] = merged["momentum_252d"]

    return merged


def compute_weighted_factor_summary(exposure_df: pd.DataFrame) -> dict:
    if exposure_df.empty:
        return {
            FACTOR_SIZE: np.nan,
            FACTOR_VALUE: np.nan,
            FACTOR_MOMENTUM: np.nan,
        }

    results = {}
    for factor in [FACTOR_SIZE, FACTOR_VALUE, FACTOR_MOMENTUM]:
        valid = exposure_df[[COL_WEIGHT, factor]].dropna()
        if valid.empty:
            results[factor] = np.nan
            continue

        weight_sum = valid[COL_WEIGHT].sum()
        if weight_sum == 0 or pd.isna(weight_sum):
            results[factor] = np.nan
            continue

        results[factor] = float((valid[COL_WEIGHT] * valid[factor]).sum() / weight_sum)

    return results


def compute_team_factor_summary(exposure_df: pd.DataFrame) -> pd.DataFrame:
    if exposure_df.empty:
        return pd.DataFrame(columns=[COL_TEAM, FACTOR_SIZE, FACTOR_VALUE, FACTOR_MOMENTUM, "team_market_value"])

    rows = []

    for team, group in exposure_df.groupby(COL_TEAM, dropna=False):
        row = {
            COL_TEAM: team,
            "team_market_value": pd.to_numeric(group[COL_MARKET_VALUE], errors="coerce").sum(skipna=True),
        }

        for factor in [FACTOR_SIZE, FACTOR_VALUE, FACTOR_MOMENTUM]:
            valid = group[[COL_WEIGHT, factor]].dropna()
            if valid.empty:
                row[factor] = np.nan
            else:
                weight_sum = valid[COL_WEIGHT].sum()
                row[factor] = (
                    float((valid[COL_WEIGHT] * valid[factor]).sum() / weight_sum)
                    if pd.notna(weight_sum) and weight_sum != 0
                    else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("team_market_value", ascending=False).reset_index(drop=True)


def render_empty_state() -> None:
    st.title("Factor Exposure")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_factor_cards(summary: dict) -> None:
    st.subheader("Total Fund Factor Snapshot")

    c1, c2, c3 = st.columns(3)
    c1.metric("Size Proxy", _format_number(summary.get(FACTOR_SIZE)))
    c2.metric("Value Proxy", _format_number(summary.get(FACTOR_VALUE)))
    c3.metric("Momentum Proxy", _format_percent(summary.get(FACTOR_MOMENTUM)))


def render_team_exposure_table(team_summary_df: pd.DataFrame) -> None:
    st.subheader("Team-Level Exposure")

    if team_summary_df.empty:
        st.info("No team exposure data available.")
        return

    display_df = team_summary_df.copy()

    if "team_market_value" in display_df.columns:
        display_df["team_market_value"] = display_df["team_market_value"].map(_format_currency)
    if FACTOR_SIZE in display_df.columns:
        display_df[FACTOR_SIZE] = display_df[FACTOR_SIZE].map(_format_number)
    if FACTOR_VALUE in display_df.columns:
        display_df[FACTOR_VALUE] = display_df[FACTOR_VALUE].map(_format_number)
    if FACTOR_MOMENTUM in display_df.columns:
        display_df[FACTOR_MOMENTUM] = display_df[FACTOR_MOMENTUM].map(_format_percent)

    st.dataframe(
        left_align_dataframe(display_df),
        use_container_width=True,
        hide_index=True,
    )


def render_factor_scatter(exposure_df: pd.DataFrame) -> None:
    st.subheader("Cross-Sectional View")

    if exposure_df.empty:
        st.info("No exposure data available.")
        return

    plot_df = exposure_df.dropna(subset=[FACTOR_VALUE, FACTOR_MOMENTUM, COL_MARKET_VALUE]).copy()

    if plot_df.empty:
        st.info("Not enough factor data is available for the scatterplot.")
        return

    fig = px.scatter(
        plot_df,
        x=FACTOR_VALUE,
        y=FACTOR_MOMENTUM,
        size=COL_MARKET_VALUE,
        color=COL_TEAM,
        hover_name=COL_TICKER,
        title="Value vs Momentum by Holding",
        labels={
            FACTOR_VALUE: "Value Proxy",
            FACTOR_MOMENTUM: "Momentum Proxy",
            COL_MARKET_VALUE: "Market Value",
        },
    )
    fig = style_plotly_figure(fig, title_text="Value vs Momentum by Holding")
    st.plotly_chart(fig, use_container_width=True)


def render_holdings_factor_table(exposure_df: pd.DataFrame) -> None:
    st.subheader("Holding-Level Factor Detail")

    if exposure_df.empty:
        st.info("No factor detail available.")
        return

    display_cols = [
        col for col in [
            COL_TICKER,
            COL_TEAM,
            COL_POSITION_SIDE,
            COL_MARKET_VALUE,
            COL_WEIGHT,
            "market_cap",
            FACTOR_SIZE,
            FACTOR_VALUE,
            FACTOR_MOMENTUM,
        ]
        if col in exposure_df.columns
    ]

    display_df = exposure_df[display_cols].copy()

    if COL_MARKET_VALUE in display_df.columns:
        display_df[COL_MARKET_VALUE] = display_df[COL_MARKET_VALUE].map(_format_currency)
    if COL_WEIGHT in display_df.columns:
        display_df[COL_WEIGHT] = display_df[COL_WEIGHT].map(_format_percent)
    if "market_cap" in display_df.columns:
        display_df["market_cap"] = display_df["market_cap"].map(_format_currency)
    if FACTOR_SIZE in display_df.columns:
        display_df[FACTOR_SIZE] = display_df[FACTOR_SIZE].map(_format_number)
    if FACTOR_VALUE in display_df.columns:
        display_df[FACTOR_VALUE] = display_df[FACTOR_VALUE].map(_format_number)
    if FACTOR_MOMENTUM in display_df.columns:
        display_df[FACTOR_MOMENTUM] = display_df[FACTOR_MOMENTUM].map(_format_percent)

    st.dataframe(
        left_align_dataframe(display_df),
        use_container_width=True,
        hide_index=True,
    )


def render_methodology_note() -> None:
    with st.expander("Methodology Notes"):
        st.write(
            """
            This MVP factor page uses simple Yahoo-derived proxies:

            - **Size**: log of market capitalization
            - **Value**: inverse of price-to-book
            - **Momentum**: trailing 252-trading-day return

            These are rough indicators, not a full institutional risk model.
            Coverage may be incomplete for some tickers.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Factor Exposure", layout="wide")
    apply_app_theme()
    st.title("Factor Exposure")

    snapshot_df = get_base_snapshot()

    if snapshot_df.empty:
        render_empty_state()
        return

    tickers = (
        snapshot_df[COL_TICKER]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("CASH")]
        .unique()
        .tolist()
    )

    factor_inputs_df = get_factor_inputs(tickers)
    exposure_df = build_exposure_snapshot(snapshot_df, factor_inputs_df)

    whole_fund_summary = compute_weighted_factor_summary(exposure_df)
    team_summary_df = compute_team_factor_summary(exposure_df)

    render_factor_cards(whole_fund_summary)
    st.divider()

    render_team_exposure_table(team_summary_df)
    st.divider()

    render_factor_scatter(exposure_df)
    st.divider()

    render_holdings_factor_table(exposure_df)
    render_methodology_note()


if __name__ == "__main__":
    main()
