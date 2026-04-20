"""
Portfolio analytics utilities built on top of reconstructed position state.

This module is responsible for:
- attaching prices to reconstructed positions
- computing market value and weights
- building team and master-fund summaries
- separating investable positions from cash
- constructing simple AUM history from daily position state

It assumes the portfolio state engine writes canonical position rows with:
- as_of_date (optional in some functions)
- team
- ticker
- position_side
- shares
- cost_basis_per_share
- total_cost_basis
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# ============================================================================
# Canonical column names
# ============================================================================
COL_DATE = "as_of_date"
COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
COL_PRICE = "price"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"
COL_COST_BASIS_PER_SHARE = "cost_basis_per_share"
COL_TOTAL_COST_BASIS = "total_cost_basis"
CASH_LIKE_TICKERS = {"CASH", "EUR", "GBP", "NOGXX"}


# ============================================================================
# Helpers
# ============================================================================
def _standardize_position_state(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a position-state-like DataFrame has the columns this module expects.
    """
    if position_df.empty:
        return pd.DataFrame(
            columns=[
                COL_DATE,
                COL_TEAM,
                COL_TICKER,
                COL_POSITION_SIDE,
                COL_SHARES,
                COL_COST_BASIS_PER_SHARE,
                COL_TOTAL_COST_BASIS,
            ]
        )

    df = position_df.copy()

    for col in [COL_TEAM, COL_TICKER, COL_POSITION_SIDE]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype(str).str.strip()

    if COL_DATE not in df.columns:
        df[COL_DATE] = pd.NaT
    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    for col in [COL_SHARES, COL_COST_BASIS_PER_SHARE, COL_TOTAL_COST_BASIS]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def _standardize_price_history(price_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a price-history-like DataFrame.

    Accepts either:
    - price_date / close_price / adj_close_price
    or
    - date / close / adj_close
    """
    if price_history_df.empty:
        return pd.DataFrame(columns=["price_date", COL_TICKER, "close_price", "adj_close_price"])

    df = price_history_df.copy()

    if "price_date" not in df.columns and "date" in df.columns:
        df["price_date"] = df["date"]

    if "close_price" not in df.columns and "close" in df.columns:
        df["close_price"] = df["close"]

    if "adj_close_price" not in df.columns and "adj_close" in df.columns:
        df["adj_close_price"] = df["adj_close"]

    df["price_date"] = pd.to_datetime(df["price_date"], errors="coerce")
    df[COL_TICKER] = df[COL_TICKER].astype(str).str.strip()
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["adj_close_price"] = pd.to_numeric(df["adj_close_price"], errors="coerce")

    return df[["price_date", COL_TICKER, "close_price", "adj_close_price"]].copy()


# ============================================================================
# Current snapshot / latest state analytics
# ============================================================================
def attach_latest_prices_to_positions(
    position_state_df: pd.DataFrame,
    latest_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach latest prices to reconstructed positions.

    latest_prices_df is expected to contain:
    - ticker
    - price
    """
    positions = _standardize_position_state(position_state_df)

    if positions.empty:
        return positions

    if latest_prices_df.empty:
        positions[COL_PRICE] = pd.NA
        positions[COL_MARKET_VALUE] = pd.NA
        return positions

    prices = latest_prices_df.copy()
    prices[COL_TICKER] = prices[COL_TICKER].astype(str).str.strip()
    prices[COL_PRICE] = pd.to_numeric(prices[COL_PRICE], errors="coerce")

    merged = positions.merge(
        prices[[COL_TICKER, COL_PRICE]],
        on=COL_TICKER,
        how="left",
    )

    # Cash-like rows should be priced at 1
    cash_mask = (
        merged[COL_POSITION_SIDE].astype(str).str.upper().eq("CASH")
        | merged[COL_TICKER].astype(str).str.upper().isin(CASH_LIKE_TICKERS)
        | merged[COL_TEAM].astype(str).str.upper().eq("CASH")
    )
    merged.loc[cash_mask, COL_PRICE] = 1.0

    # Market value convention:
    # LONG  -> +shares * price
    # SHORT -> -shares * price
    # CASH  ->  shares * 1
    merged[COL_MARKET_VALUE] = pd.NA

    long_mask = merged[COL_POSITION_SIDE].eq("LONG")
    short_mask = merged[COL_POSITION_SIDE].eq("SHORT")
    cash_mask = (
        merged[COL_POSITION_SIDE].astype(str).str.upper().eq("CASH")
        | merged[COL_TICKER].astype(str).str.upper().isin(CASH_LIKE_TICKERS)
        | merged[COL_TEAM].astype(str).str.upper().eq("CASH")
    )

    merged.loc[long_mask, COL_MARKET_VALUE] = (
        pd.to_numeric(merged.loc[long_mask, COL_SHARES], errors="coerce")
        * pd.to_numeric(merged.loc[long_mask, COL_PRICE], errors="coerce")
    )

    merged.loc[short_mask, COL_MARKET_VALUE] = (
        -pd.to_numeric(merged.loc[short_mask, COL_SHARES], errors="coerce")
        * pd.to_numeric(merged.loc[short_mask, COL_PRICE], errors="coerce")
    )

    merged.loc[cash_mask, COL_MARKET_VALUE] = pd.to_numeric(
        merged.loc[cash_mask, COL_SHARES],
        errors="coerce",
    )

    merged[COL_MARKET_VALUE] = pd.to_numeric(merged[COL_MARKET_VALUE], errors="coerce")

    return merged


def compute_portfolio_weights(position_with_prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute portfolio weights using total net market value as denominator.
    """
    df = position_with_prices_df.copy()

    if df.empty:
        df[COL_WEIGHT] = pd.Series(dtype="float64")
        return df

    total_market_value = pd.to_numeric(df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)

    if pd.isna(total_market_value) or total_market_value == 0:
        df[COL_WEIGHT] = pd.NA
        return df

    df[COL_WEIGHT] = pd.to_numeric(df[COL_MARKET_VALUE], errors="coerce") / total_market_value
    return df


def build_current_portfolio_snapshot(
    position_state_df: pd.DataFrame,
    latest_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a priced current portfolio snapshot from reconstructed positions.
    """
    priced = attach_latest_prices_to_positions(position_state_df, latest_prices_df)
    weighted = compute_portfolio_weights(priced)
    return weighted.reset_index(drop=True)


def summarize_by_team(position_snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a priced position snapshot to the team level.

    Returns:
    - team
    - market_value
    - weight
    - position_count
    """
    if position_snapshot_df.empty:
        return pd.DataFrame(columns=[COL_TEAM, COL_MARKET_VALUE, COL_WEIGHT, "position_count"])

    grouped = (
        position_snapshot_df.groupby(COL_TEAM, dropna=False)
        .agg(
            market_value=(COL_MARKET_VALUE, "sum"),
            position_count=(COL_TICKER, "count"),
        )
        .reset_index()
    )

    total_market_value = pd.to_numeric(grouped["market_value"], errors="coerce").sum(skipna=True)
    if pd.isna(total_market_value) or total_market_value == 0:
        grouped[COL_WEIGHT] = pd.NA
    else:
        grouped[COL_WEIGHT] = pd.to_numeric(grouped["market_value"], errors="coerce") / total_market_value

    return grouped.sort_values("market_value", ascending=False).reset_index(drop=True)


def summarize_total_portfolio(position_snapshot_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute top-level summary statistics for the priced portfolio.
    """
    if position_snapshot_df.empty:
        return {
            "total_market_value": 0.0,
            "total_positions": 0,
            "total_teams": 0,
            "priced_positions": 0,
            "unpriced_positions": 0,
            "cash_value": 0.0,
            "gross_exposure": 0.0,
        }

    market_values = pd.to_numeric(position_snapshot_df[COL_MARKET_VALUE], errors="coerce")
    prices = pd.to_numeric(position_snapshot_df[COL_PRICE], errors="coerce")

    cash_mask = (
        position_snapshot_df[COL_POSITION_SIDE].astype(str).str.upper().eq("CASH")
        | position_snapshot_df[COL_TICKER].astype(str).str.upper().isin(CASH_LIKE_TICKERS)
        | position_snapshot_df[COL_TEAM].astype(str).str.upper().eq("CASH")
    )

    gross_exposure = market_values.abs().sum(skipna=True)

    return {
        "total_market_value": float(market_values.sum(skipna=True)),
        "total_positions": int(len(position_snapshot_df)),
        "total_teams": int(position_snapshot_df[COL_TEAM].nunique(dropna=True)),
        "priced_positions": int(prices.notna().sum()),
        "unpriced_positions": int(prices.isna().sum()),
        "cash_value": float(market_values.loc[cash_mask].sum(skipna=True)),
        "gross_exposure": float(gross_exposure),
    }


def find_unpriced_positions(position_snapshot_df: pd.DataFrame) -> List[str]:
    """
    Return tickers with missing prices, excluding CASH.
    """
    if position_snapshot_df.empty:
        return []

    mask = (
        position_snapshot_df[COL_PRICE].isna()
        & ~position_snapshot_df[COL_TICKER].astype(str).str.upper().isin(CASH_LIKE_TICKERS)
    )

    return sorted(
        position_snapshot_df.loc[mask, COL_TICKER]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def build_portfolio_views(
    position_state_df: pd.DataFrame,
    latest_prices_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Build the core current-state portfolio views in one call.
    """
    snapshot = build_current_portfolio_snapshot(position_state_df, latest_prices_df)
    team_summary = summarize_by_team(snapshot)
    portfolio_summary = summarize_total_portfolio(snapshot)
    unpriced_tickers = find_unpriced_positions(snapshot)

    return {
        "holdings_snapshot": snapshot,
        "team_summary": team_summary,
        "portfolio_summary": portfolio_summary,
        "unpriced_tickers": unpriced_tickers,
    }


# ============================================================================
# Historical AUM / position value history
# ============================================================================
def attach_historical_prices_to_position_state(
    position_state_history_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    """
    Attach daily historical prices to daily position state.

    position_state_history_df is expected to contain:
    - as_of_date
    - team
    - ticker
    - position_side
    - shares

    price_history_df is expected to contain:
    - price_date
    - ticker
    - close_price / adj_close_price
    """
    positions = _standardize_position_state(position_state_history_df)
    prices = _standardize_price_history(price_history_df)

    if positions.empty:
        positions[COL_PRICE] = pd.Series(dtype="float64")
        positions[COL_MARKET_VALUE] = pd.Series(dtype="float64")
        return positions

    if prices.empty:
        positions[COL_PRICE] = pd.NA
        positions[COL_MARKET_VALUE] = pd.NA
        return positions

    price_col = "adj_close_price" if use_adjusted_close else "close_price"

    merged = positions.merge(
        prices[["price_date", COL_TICKER, price_col]],
        left_on=[COL_DATE, COL_TICKER],
        right_on=["price_date", COL_TICKER],
        how="left",
    )

    merged = merged.rename(columns={price_col: COL_PRICE})

    cash_mask = (
        merged[COL_POSITION_SIDE].astype(str).str.upper().eq("CASH")
        | merged[COL_TICKER].astype(str).str.upper().isin(CASH_LIKE_TICKERS)
        | merged[COL_TEAM].astype(str).str.upper().eq("CASH")
    )
    merged.loc[cash_mask, COL_PRICE] = 1.0

    merged[COL_MARKET_VALUE] = pd.NA

    long_mask = merged[COL_POSITION_SIDE].eq("LONG")
    short_mask = merged[COL_POSITION_SIDE].eq("SHORT")

    merged.loc[long_mask, COL_MARKET_VALUE] = (
        pd.to_numeric(merged.loc[long_mask, COL_SHARES], errors="coerce")
        * pd.to_numeric(merged.loc[long_mask, COL_PRICE], errors="coerce")
    )

    merged.loc[short_mask, COL_MARKET_VALUE] = (
        -pd.to_numeric(merged.loc[short_mask, COL_SHARES], errors="coerce")
        * pd.to_numeric(merged.loc[short_mask, COL_PRICE], errors="coerce")
    )

    merged.loc[cash_mask, COL_MARKET_VALUE] = pd.to_numeric(
        merged.loc[cash_mask, COL_SHARES],
        errors="coerce",
    )

    merged[COL_MARKET_VALUE] = pd.to_numeric(merged[COL_MARKET_VALUE], errors="coerce")

    return merged.drop(columns=["price_date"], errors="ignore").reset_index(drop=True)


def build_team_aum_history(
    position_state_history_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    team: str,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    """
    Build true historical team AUM from daily position state and historical prices.

    Returns:
    - as_of_date
    - team
    - team_aum
    """
    priced = attach_historical_prices_to_position_state(
        position_state_history_df=position_state_history_df,
        price_history_df=price_history_df,
        use_adjusted_close=use_adjusted_close,
    )

    if priced.empty:
        return pd.DataFrame(columns=[COL_DATE, COL_TEAM, "team_aum"])

    filtered = priced.loc[priced[COL_TEAM] == team].copy()

    history = (
        filtered.groupby([COL_DATE, COL_TEAM], dropna=False)
        .agg(team_aum=(COL_MARKET_VALUE, "sum"))
        .reset_index()
        .sort_values(COL_DATE)
        .reset_index(drop=True)
    )

    return history


def build_master_fund_aum_history(
    position_state_history_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    """
    Build true historical total fund AUM from daily position state and prices.

    Returns:
    - as_of_date
    - total_aum
    """
    priced = attach_historical_prices_to_position_state(
        position_state_history_df=position_state_history_df,
        price_history_df=price_history_df,
        use_adjusted_close=use_adjusted_close,
    )

    if priced.empty:
        return pd.DataFrame(columns=[COL_DATE, "total_aum"])

    history = (
        priced.groupby(COL_DATE, dropna=False)
        .agg(total_aum=(COL_MARKET_VALUE, "sum"))
        .reset_index()
        .sort_values(COL_DATE)
        .reset_index(drop=True)
    )

    return history


def build_team_exposure_history(
    position_state_history_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    use_adjusted_close: bool = True,
) -> pd.DataFrame:
    """
    Build daily AUM history for all teams.

    Returns:
    - as_of_date
    - team
    - team_aum
    """
    priced = attach_historical_prices_to_position_state(
        position_state_history_df=position_state_history_df,
        price_history_df=price_history_df,
        use_adjusted_close=use_adjusted_close,
    )

    if priced.empty:
        return pd.DataFrame(columns=[COL_DATE, COL_TEAM, "team_aum"])

    history = (
        priced.groupby([COL_DATE, COL_TEAM], dropna=False)
        .agg(team_aum=(COL_MARKET_VALUE, "sum"))
        .reset_index()
        .sort_values([COL_DATE, COL_TEAM])
        .reset_index(drop=True)
    )

    return history
