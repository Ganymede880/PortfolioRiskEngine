"""
Return calculation utilities for the CMCSIF portfolio tracker.

This module computes:
- holding-level daily returns
- holding-level daily P&L
- team-level daily returns
- portfolio-level daily returns
- cumulative returns from a return series

Assumptions for MVP:
- latest prices come from the live pricing layer
- historical close prices come from Yahoo Finance
- daily return uses close-to-close return
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.utils.constants import (
    COL_MARKET_VALUE,
    COL_PNL,
    COL_PRICE,
    COL_RETURN,
    COL_SHARES,
    COL_TEAM,
    COL_TICKER,
    COL_WEIGHT,
)


def compute_historical_daily_returns(price_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns from historical adjusted close prices.

    Expected input columns:
    - date
    - ticker
    - adj_close

    Returns:
    DataFrame with:
    - date
    - ticker
    - adj_close
    - return
    """
    if price_history_df.empty:
        return pd.DataFrame(columns=["date", COL_TICKER, "adj_close", COL_RETURN])

    working = price_history_df.copy()
    working = working.sort_values([COL_TICKER, "date"]).reset_index(drop=True)

    working["adj_close"] = pd.to_numeric(working["adj_close"], errors="coerce")
    working[COL_RETURN] = (
        working.groupby(COL_TICKER)["adj_close"].pct_change()
    )

    return working[["date", COL_TICKER, "adj_close", COL_RETURN]]


def attach_latest_holding_returns(
    holdings_snapshot_df: pd.DataFrame,
    historical_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the latest available daily return to each holding.

    Uses the most recent row per ticker from historical_returns_df.
    Also computes daily P&L = shares * price * daily_return

    Expected holdings columns:
    - ticker
    - shares
    - price
    - market_value

    Expected historical return columns:
    - ticker
    - date
    - return
    """
    if holdings_snapshot_df.empty:
        return holdings_snapshot_df.copy()

    working = holdings_snapshot_df.copy()

    if historical_returns_df.empty:
        if COL_RETURN not in working.columns:
            working[COL_RETURN] = pd.NA
        if COL_PNL not in working.columns:
            working[COL_PNL] = pd.NA
        return working

    latest_returns = (
        historical_returns_df
        .sort_values(["date", COL_TICKER])
        .dropna(subset=[COL_RETURN])
        .groupby(COL_TICKER, as_index=False)
        .tail(1)[[COL_TICKER, COL_RETURN]]
    )

    working = working.merge(
        latest_returns,
        on=COL_TICKER,
        how="left",
    )

    working[COL_PRICE] = pd.to_numeric(working[COL_PRICE], errors="coerce")
    working[COL_SHARES] = pd.to_numeric(working[COL_SHARES], errors="coerce")
    working[COL_MARKET_VALUE] = pd.to_numeric(working[COL_MARKET_VALUE], errors="coerce")
    working[COL_RETURN] = pd.to_numeric(working[COL_RETURN], errors="coerce")

    working[COL_PNL] = working[COL_MARKET_VALUE] * working[COL_RETURN]

    return working


def compute_team_daily_returns(holdings_with_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute team-level daily return as weighted average of holding returns.

    Uses current portfolio weights within the total fund. For a pure sleeve-level
    return, this is equivalent to:
        sum(daily pnl for team) / sum(team market value)
    """
    if holdings_with_returns_df.empty:
        return pd.DataFrame(columns=[COL_TEAM, "team_market_value", "team_daily_pnl", "team_daily_return"])

    working = holdings_with_returns_df.copy()

    working[COL_MARKET_VALUE] = pd.to_numeric(working[COL_MARKET_VALUE], errors="coerce")
    working[COL_PNL] = pd.to_numeric(working[COL_PNL], errors="coerce")

    grouped = (
        working.groupby(COL_TEAM, dropna=False)
        .agg(
            team_market_value=(COL_MARKET_VALUE, "sum"),
            team_daily_pnl=(COL_PNL, "sum"),
        )
        .reset_index()
    )

    grouped["team_daily_return"] = pd.NA
    valid_mask = grouped["team_market_value"].notna() & (grouped["team_market_value"] != 0)
    grouped.loc[valid_mask, "team_daily_return"] = (
        grouped.loc[valid_mask, "team_daily_pnl"] / grouped.loc[valid_mask, "team_market_value"]
    )

    return grouped


def compute_portfolio_daily_return(holdings_with_returns_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute total portfolio daily P&L and daily return.
    """
    if holdings_with_returns_df.empty:
        return {
            "portfolio_market_value": 0.0,
            "portfolio_daily_pnl": 0.0,
            "portfolio_daily_return": 0.0,
        }

    working = holdings_with_returns_df.copy()

    working[COL_MARKET_VALUE] = pd.to_numeric(working[COL_MARKET_VALUE], errors="coerce")
    working[COL_PNL] = pd.to_numeric(working[COL_PNL], errors="coerce")

    portfolio_market_value = float(working[COL_MARKET_VALUE].sum(skipna=True))
    portfolio_daily_pnl = float(working[COL_PNL].sum(skipna=True))

    if portfolio_market_value == 0:
        portfolio_daily_return = 0.0
    else:
        portfolio_daily_return = portfolio_daily_pnl / portfolio_market_value

    return {
        "portfolio_market_value": portfolio_market_value,
        "portfolio_daily_pnl": portfolio_daily_pnl,
        "portfolio_daily_return": portfolio_daily_return,
    }


def compute_contribution_to_return(holdings_with_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute each holding's contribution to total portfolio return.

    Contribution = holding weight * holding return

    Returns the input DataFrame with an added column:
    - contribution_to_return
    """
    if holdings_with_returns_df.empty:
        result = holdings_with_returns_df.copy()
        result["contribution_to_return"] = pd.Series(dtype="float64")
        return result

    working = holdings_with_returns_df.copy()

    working[COL_WEIGHT] = pd.to_numeric(working[COL_WEIGHT], errors="coerce")
    working[COL_RETURN] = pd.to_numeric(working[COL_RETURN], errors="coerce")

    working["contribution_to_return"] = working[COL_WEIGHT] * working[COL_RETURN]

    return working


def compute_cumulative_return(return_series: pd.Series) -> pd.Series:
    """
    Convert a daily return series into cumulative return series.

    Formula:
        cumulative = (1 + r).cumprod() - 1
    """
    clean_returns = pd.to_numeric(return_series, errors="coerce").fillna(0.0)
    return (1.0 + clean_returns).cumprod() - 1.0


def compute_portfolio_return_series(
    holdings_history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a historical portfolio return series from a holdings-history-style table.

    Expected columns:
    - date
    - market_value
    - pnl

    This is a simple helper for future expansion once we store daily snapshots.
    """
    if holdings_history_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_daily_return", "portfolio_daily_pnl", "portfolio_market_value"])

    working = holdings_history_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working[COL_MARKET_VALUE] = pd.to_numeric(working[COL_MARKET_VALUE], errors="coerce")
    working[COL_PNL] = pd.to_numeric(working[COL_PNL], errors="coerce")

    grouped = (
        working.groupby("date", dropna=False)
        .agg(
            portfolio_market_value=(COL_MARKET_VALUE, "sum"),
            portfolio_daily_pnl=(COL_PNL, "sum"),
        )
        .reset_index()
        .sort_values("date")
    )

    grouped["portfolio_daily_return"] = 0.0
    valid_mask = grouped["portfolio_market_value"].notna() & (grouped["portfolio_market_value"] != 0)
    grouped.loc[valid_mask, "portfolio_daily_return"] = (
        grouped.loc[valid_mask, "portfolio_daily_pnl"] / grouped.loc[valid_mask, "portfolio_market_value"]
    )

    grouped["portfolio_cumulative_return"] = compute_cumulative_return(grouped["portfolio_daily_return"])

    return grouped


def build_return_views(
    holdings_snapshot_df: pd.DataFrame,
    historical_returns_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build all core MVP return outputs in one call.

    Returns:
    - holdings_with_returns
    - team_returns
    - portfolio_return_summary
    """
    holdings_with_returns = attach_latest_holding_returns(
        holdings_snapshot_df,
        historical_returns_df,
    )
    holdings_with_returns = compute_contribution_to_return(holdings_with_returns)

    team_returns = compute_team_daily_returns(holdings_with_returns)
    portfolio_return_summary = compute_portfolio_daily_return(holdings_with_returns)

    return {
        "holdings_with_returns": holdings_with_returns,
        "team_returns": team_returns,
        "portfolio_return_summary": portfolio_return_summary,
    }