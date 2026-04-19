"""
Attribution utilities for the CMCSIF portfolio tracker.

This module computes:
- holding-level contribution summaries
- team-level contribution summaries
- top contributors and detractors
- simple portfolio concentration views

It assumes holdings already have:
- market value
- weight
- daily return
- daily P&L
- contribution to return
"""

from __future__ import annotations

from typing import Dict, Any

import pandas as pd

from src.utils.constants import (
    COL_MARKET_VALUE,
    COL_PNL,
    COL_RETURN,
    COL_TEAM,
    COL_TICKER,
    COL_WEIGHT,
)


CONTRIBUTION_COL = "contribution_to_return"


def summarize_holdings_contribution(holdings_with_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a holding-level attribution table sorted by contribution.

    Expected columns:
    - ticker
    - team
    - market_value
    - weight
    - return
    - pnl
    - contribution_to_return
    """
    if holdings_with_returns_df.empty:
        return pd.DataFrame(
            columns=[
                COL_TICKER,
                COL_TEAM,
                COL_MARKET_VALUE,
                COL_WEIGHT,
                COL_RETURN,
                COL_PNL,
                CONTRIBUTION_COL,
            ]
        )

    working = holdings_with_returns_df.copy()

    for col in [COL_MARKET_VALUE, COL_WEIGHT, COL_RETURN, COL_PNL, CONTRIBUTION_COL]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    sort_col = CONTRIBUTION_COL if CONTRIBUTION_COL in working.columns else COL_PNL
    working = working.sort_values(sort_col, ascending=False).reset_index(drop=True)

    keep_cols = [
        col for col in [
            COL_TICKER,
            COL_TEAM,
            COL_MARKET_VALUE,
            COL_WEIGHT,
            COL_RETURN,
            COL_PNL,
            CONTRIBUTION_COL,
        ]
        if col in working.columns
    ]

    return working[keep_cols]


def summarize_team_contribution(holdings_with_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate attribution to the team level.

    Returns:
    - team
    - team_market_value
    - team_weight
    - team_daily_pnl
    - team_contribution_to_return
    - average_holding_return
    - holding_count
    """
    if holdings_with_returns_df.empty:
        return pd.DataFrame(
            columns=[
                COL_TEAM,
                "team_market_value",
                "team_weight",
                "team_daily_pnl",
                "team_contribution_to_return",
                "average_holding_return",
                "holding_count",
            ]
        )

    working = holdings_with_returns_df.copy()

    for col in [COL_MARKET_VALUE, COL_WEIGHT, COL_RETURN, COL_PNL, CONTRIBUTION_COL]:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    grouped = (
        working.groupby(COL_TEAM, dropna=False)
        .agg(
            team_market_value=(COL_MARKET_VALUE, "sum"),
            team_weight=(COL_WEIGHT, "sum"),
            team_daily_pnl=(COL_PNL, "sum"),
            team_contribution_to_return=(CONTRIBUTION_COL, "sum"),
            average_holding_return=(COL_RETURN, "mean"),
            holding_count=(COL_TICKER, "nunique"),
        )
        .reset_index()
        .sort_values("team_contribution_to_return", ascending=False)
        .reset_index(drop=True)
    )

    return grouped


def get_top_contributors(
    holdings_with_returns_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top N positive contributors to portfolio return.
    """
    if holdings_with_returns_df.empty:
        return pd.DataFrame()

    working = summarize_holdings_contribution(holdings_with_returns_df)

    if CONTRIBUTION_COL in working.columns:
        working = working.sort_values(CONTRIBUTION_COL, ascending=False)
    elif COL_PNL in working.columns:
        working = working.sort_values(COL_PNL, ascending=False)

    return working.head(n).reset_index(drop=True)


def get_top_detractors(
    holdings_with_returns_df: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return the top N negative contributors to portfolio return.
    """
    if holdings_with_returns_df.empty:
        return pd.DataFrame()

    working = summarize_holdings_contribution(holdings_with_returns_df)

    if CONTRIBUTION_COL in working.columns:
        working = working.sort_values(CONTRIBUTION_COL, ascending=True)
    elif COL_PNL in working.columns:
        working = working.sort_values(COL_PNL, ascending=True)

    return working.head(n).reset_index(drop=True)


def summarize_portfolio_concentration(holdings_snapshot_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute simple portfolio concentration statistics.

    Returns:
    - top_5_weight
    - top_10_weight
    - largest_position_weight
    - number_of_positions
    """
    if holdings_snapshot_df.empty or COL_WEIGHT not in holdings_snapshot_df.columns:
        return {
            "top_5_weight": 0.0,
            "top_10_weight": 0.0,
            "largest_position_weight": 0.0,
            "number_of_positions": 0,
        }

    working = holdings_snapshot_df.copy()
    working[COL_WEIGHT] = pd.to_numeric(working[COL_WEIGHT], errors="coerce")
    weights = working[COL_WEIGHT].dropna().sort_values(ascending=False)

    return {
        "top_5_weight": float(weights.head(5).sum()) if not weights.empty else 0.0,
        "top_10_weight": float(weights.head(10).sum()) if not weights.empty else 0.0,
        "largest_position_weight": float(weights.iloc[0]) if not weights.empty else 0.0,
        "number_of_positions": int(len(weights)),
    }


def build_attribution_view(
    holdings_with_returns_df: pd.DataFrame,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Build all core MVP attribution outputs in one call.

    Returns:
    - holdings_attribution
    - team_attribution
    - top_contributors
    - top_detractors
    - concentration_summary
    """
    holdings_attribution = summarize_holdings_contribution(holdings_with_returns_df)
    team_attribution = summarize_team_contribution(holdings_with_returns_df)
    top_contributors = get_top_contributors(holdings_with_returns_df, n=top_n)
    top_detractors = get_top_detractors(holdings_with_returns_df, n=top_n)
    concentration_summary = summarize_portfolio_concentration(holdings_with_returns_df)

    return {
        "holdings_attribution": holdings_attribution,
        "team_attribution": team_attribution,
        "top_contributors": top_contributors,
        "top_detractors": top_detractors,
        "concentration_summary": concentration_summary,
    }