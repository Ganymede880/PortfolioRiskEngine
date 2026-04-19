"""
Normalization utilities for uploaded snapshot and trade receipt files.

This module converts validated raw uploads into canonical internal schemas for:
- authoritative portfolio snapshots
- trade receipts

It assumes the file has already been loaded and validated.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.data.mappings import apply_team_mapping, apply_ticker_mapping
from src.data.validators import (
    SNAPSHOT_COLUMN_ALIASES,
    TRADE_RECEIPT_COLUMN_ALIASES,
    resolve_column_mapping,
)


# ============================================================================
# Shared helpers
# ============================================================================
def _clean_string_series(series: pd.Series) -> pd.Series:
    """
    Standardize a string-like pandas Series by stripping whitespace.
    """
    return series.astype(str).str.strip()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """
    Convert a Series to numeric, coercing invalid values to NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def _coerce_datetime(series: pd.Series) -> pd.Series:
    """
    Convert a Series to pandas datetime, coercing invalid values to NaT.
    """
    return pd.to_datetime(series, errors="coerce")


def _safe_upper_string(series: pd.Series) -> pd.Series:
    """
    Clean and uppercase a string series.
    """
    return _clean_string_series(series).str.upper()


def _replace_nullish_strings(series: pd.Series) -> pd.Series:
    """
    Replace common null-ish string values with pandas NA.
    """
    return series.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "NONE": pd.NA,
            "null": pd.NA,
            "NULL": pd.NA,
        }
    )


def _fill_optional_column(
    output_df: pd.DataFrame,
    source_df: pd.DataFrame,
    resolved_columns: Dict[str, str],
    canonical_name: str,
    coercion_func: Any | None = None,
) -> None:
    """
    Fill a canonical column if its source column exists.
    """
    source_col = resolved_columns.get(canonical_name)
    if source_col is None:
        return

    series = source_df[source_col]
    if coercion_func is not None:
        series = coercion_func(series)

    output_df[canonical_name] = series


# ============================================================================
# Snapshot normalization
# ============================================================================
SNAPSHOT_OUTPUT_COLUMNS: List[str] = [
    "snapshot_date",
    "team",
    "ticker",
    "position_side",
    "shares",
    "cost_basis_per_share",
    "total_cost_basis",
]


def _normalize_position_side(series: pd.Series) -> pd.Series:
    """
    Normalize snapshot position-side labels.

    Canonical values:
    - LONG
    - SHORT
    - CASH
    """
    cleaned = _safe_upper_string(series)

    mapping = {
        "LONG": "LONG",
        "L": "LONG",
        "SHORT": "SHORT",
        "S": "SHORT",
        "CASH": "CASH",
    }

    normalized = cleaned.map(mapping)
    fallback_mask = normalized.isna() & cleaned.notna()
    normalized.loc[fallback_mask] = cleaned.loc[fallback_mask]

    return _replace_nullish_strings(normalized)


def normalize_snapshot_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Normalize a raw snapshot DataFrame into canonical snapshot schema.

    Returns:
    - snapshot: normalized DataFrame
    - unmapped_teams: list[str]
    - suspicious_tickers: list[str]
    - resolved_columns: dict[str, str]
    """
    working_df = df.copy()
    resolved_columns = resolve_column_mapping(working_df, SNAPSHOT_COLUMN_ALIASES)

    normalized_df = pd.DataFrame(index=working_df.index, columns=SNAPSHOT_OUTPUT_COLUMNS)

    # Required fields
    team_source_col = resolved_columns.get("team")
    ticker_source_col = resolved_columns.get("ticker")
    position_source_col = resolved_columns.get("position_side")
    shares_source_col = resolved_columns.get("shares")
    snapshot_date_source_col = resolved_columns.get("snapshot_date")

    if team_source_col is None:
        raise ValueError("Cannot normalize snapshot: team column could not be resolved.")
    if ticker_source_col is None:
        raise ValueError("Cannot normalize snapshot: ticker column could not be resolved.")
    if position_source_col is None:
        raise ValueError("Cannot normalize snapshot: position_side column could not be resolved.")
    if shares_source_col is None:
        raise ValueError("Cannot normalize snapshot: shares column could not be resolved.")
    if snapshot_date_source_col is None:
        raise ValueError("Cannot normalize snapshot: snapshot_date column could not be resolved.")

    # Clean source columns
    working_df[team_source_col] = _clean_string_series(working_df[team_source_col])
    working_df[ticker_source_col] = _clean_string_series(working_df[ticker_source_col])
    working_df[position_source_col] = _clean_string_series(working_df[position_source_col])

    # Team mapping
    team_mapped_df, unmapped_teams = apply_team_mapping(
        working_df,
        source_column=team_source_col,
        output_column="team",
    )
    normalized_df["team"] = _replace_nullish_strings(_clean_string_series(team_mapped_df["team"]))

    # Ticker mapping
    ticker_mapped_df, suspicious_tickers = apply_ticker_mapping(
        working_df,
        source_column=ticker_source_col,
        output_column="ticker",
    )
    normalized_df["ticker"] = _replace_nullish_strings(_clean_string_series(ticker_mapped_df["ticker"]))

    # Position side
    normalized_df["position_side"] = _normalize_position_side(working_df[position_source_col])

    # Required numeric/date fields
    normalized_df["shares"] = _coerce_numeric(working_df[shares_source_col])
    normalized_df["snapshot_date"] = _coerce_datetime(working_df[snapshot_date_source_col])

    # Optional fields
    _fill_optional_column(
        normalized_df,
        working_df,
        resolved_columns,
        "cost_basis_per_share",
        coercion_func=_coerce_numeric,
    )
    _fill_optional_column(
        normalized_df,
        working_df,
        resolved_columns,
        "total_cost_basis",
        coercion_func=_coerce_numeric,
    )

    # Derive total cost basis where possible
    if "total_cost_basis" in normalized_df.columns:
        missing_total_cost_mask = (
            normalized_df["total_cost_basis"].isna()
            & normalized_df["cost_basis_per_share"].notna()
            & normalized_df["shares"].notna()
        )
        normalized_df.loc[missing_total_cost_mask, "total_cost_basis"] = (
            normalized_df.loc[missing_total_cost_mask, "cost_basis_per_share"]
            * normalized_df.loc[missing_total_cost_mask, "shares"]
        )

    # Drop unusable rows
    normalized_df = normalized_df.dropna(
        subset=["snapshot_date", "team", "ticker", "position_side", "shares"]
    ).copy()

    # Reset index
    normalized_df = normalized_df.reset_index(drop=True)

    return {
        "snapshot": normalized_df,
        "unmapped_teams": unmapped_teams,
        "suspicious_tickers": suspicious_tickers,
        "resolved_columns": resolved_columns,
    }


# ============================================================================
# Trade receipt normalization
# ============================================================================
TRADE_OUTPUT_COLUMNS: List[str] = [
    "trade_date",
    "settlement_date",
    "team",
    "ticker",
    "trade_side",
    "quantity",
    "gross_price",
    "commission",
    "fees",
    "net_cash_amount",
    "raw_description",
]


def _normalize_trade_side(series: pd.Series) -> pd.Series:
    """
    Normalize trade-side labels.

    Canonical values:
    - BUY
    - SELL
    - SHORT_SELL
    - COVER
    """
    cleaned = _safe_upper_string(series)

    mapping = {
        "BUY": "BUY",
        "BOT": "BUY",
        "PURCHASE": "BUY",
        "SELL": "SELL",
        "SLD": "SELL",
        "SHORT": "SHORT_SELL",
        "SHORT SELL": "SHORT_SELL",
        "SHORT_SELL": "SHORT_SELL",
        "COVER": "COVER",
        "BUY TO COVER": "COVER",
    }

    normalized = cleaned.map(mapping)
    fallback_mask = normalized.isna() & cleaned.notna()
    normalized.loc[fallback_mask] = cleaned.loc[fallback_mask]

    return _replace_nullish_strings(normalized)


def _derive_net_cash_amount(
    trade_side: pd.Series,
    quantity: pd.Series,
    gross_price: pd.Series,
    commission: pd.Series,
    fees: pd.Series,
) -> pd.Series:
    """
    Derive signed net cash amount when no explicit net consideration is provided.

    Convention:
    - BUY / COVER -> negative cash amount
    - SELL / SHORT_SELL -> positive cash amount
    """
    gross_notional = quantity * gross_price
    total_fees = commission.fillna(0.0) + fees.fillna(0.0)

    result = pd.Series(pd.NA, index=trade_side.index, dtype="object")

    buy_like = trade_side.isin(["BUY", "COVER"])
    sell_like = trade_side.isin(["SELL", "SHORT_SELL"])

    result.loc[buy_like] = -(gross_notional.loc[buy_like] + total_fees.loc[buy_like])
    result.loc[sell_like] = gross_notional.loc[sell_like] - total_fees.loc[sell_like]

    return pd.to_numeric(result, errors="coerce")


def normalize_trade_receipt_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Normalize a raw trade receipt DataFrame into canonical trade schema.

    Returns:
    - trades: normalized DataFrame
    - unmapped_teams: list[str]
    - suspicious_tickers: list[str]
    - resolved_columns: dict[str, str]
    """
    working_df = df.copy()
    resolved_columns = resolve_column_mapping(working_df, TRADE_RECEIPT_COLUMN_ALIASES)

    normalized_df = pd.DataFrame(index=working_df.index, columns=TRADE_OUTPUT_COLUMNS)

    # Required fields
    team_source_col = resolved_columns.get("team")
    trade_side_source_col = resolved_columns.get("trade_side")
    ticker_source_col = resolved_columns.get("ticker")
    quantity_source_col = resolved_columns.get("quantity")
    gross_price_source_col = resolved_columns.get("gross_price")
    settlement_date_source_col = resolved_columns.get("settlement_date")

    if team_source_col is None:
        raise ValueError("Cannot normalize trade receipt: team column could not be resolved.")
    if trade_side_source_col is None:
        raise ValueError("Cannot normalize trade receipt: trade_side column could not be resolved.")
    if ticker_source_col is None:
        raise ValueError("Cannot normalize trade receipt: ticker column could not be resolved.")
    if quantity_source_col is None:
        raise ValueError("Cannot normalize trade receipt: quantity column could not be resolved.")
    if gross_price_source_col is None:
        raise ValueError("Cannot normalize trade receipt: gross_price column could not be resolved.")
    if settlement_date_source_col is None:
        raise ValueError("Cannot normalize trade receipt: settlement_date column could not be resolved.")

    # Clean source columns
    working_df[team_source_col] = _clean_string_series(working_df[team_source_col])
    working_df[trade_side_source_col] = _clean_string_series(working_df[trade_side_source_col])
    working_df[ticker_source_col] = _clean_string_series(working_df[ticker_source_col])

    # Team mapping
    team_mapped_df, unmapped_teams = apply_team_mapping(
        working_df,
        source_column=team_source_col,
        output_column="team",
    )
    normalized_df["team"] = _replace_nullish_strings(_clean_string_series(team_mapped_df["team"]))

    # Ticker mapping
    ticker_mapped_df, suspicious_tickers = apply_ticker_mapping(
        working_df,
        source_column=ticker_source_col,
        output_column="ticker",
    )
    normalized_df["ticker"] = _replace_nullish_strings(_clean_string_series(ticker_mapped_df["ticker"]))

    # Trade side
    normalized_df["trade_side"] = _normalize_trade_side(working_df[trade_side_source_col])

    # Numeric/date fields
    normalized_df["quantity"] = _coerce_numeric(working_df[quantity_source_col]).abs()
    normalized_df["gross_price"] = _coerce_numeric(working_df[gross_price_source_col])
    normalized_df["settlement_date"] = _coerce_datetime(working_df[settlement_date_source_col])

    # Optional fields
    if "trade_date" in resolved_columns:
        normalized_df["trade_date"] = _coerce_datetime(working_df[resolved_columns["trade_date"]])
    else:
        normalized_df["trade_date"] = normalized_df["settlement_date"]

    if "commission" in resolved_columns:
        normalized_df["commission"] = _coerce_numeric(working_df[resolved_columns["commission"]])
    else:
        normalized_df["commission"] = 0.0

    if "fees" in resolved_columns:
        normalized_df["fees"] = _coerce_numeric(working_df[resolved_columns["fees"]])
    else:
        normalized_df["fees"] = 0.0

    if "net_cash_amount" in resolved_columns:
        normalized_df["net_cash_amount"] = _coerce_numeric(working_df[resolved_columns["net_cash_amount"]])
    else:
        normalized_df["net_cash_amount"] = _derive_net_cash_amount(
            trade_side=normalized_df["trade_side"],
            quantity=normalized_df["quantity"],
            gross_price=normalized_df["gross_price"],
            commission=normalized_df["commission"],
            fees=normalized_df["fees"],
        )

    if "raw_description" in resolved_columns:
        normalized_df["raw_description"] = _clean_string_series(working_df[resolved_columns["raw_description"]])
        normalized_df["raw_description"] = _replace_nullish_strings(normalized_df["raw_description"])
    else:
        normalized_df["raw_description"] = pd.NA

    # Drop unusable rows
    normalized_df = normalized_df.dropna(
        subset=["trade_date", "settlement_date", "team", "ticker", "trade_side", "quantity", "gross_price"]
    ).copy()

    # Reset index
    normalized_df = normalized_df.reset_index(drop=True)

    return {
        "trades": normalized_df,
        "unmapped_teams": unmapped_teams,
        "suspicious_tickers": suspicious_tickers,
        "resolved_columns": resolved_columns,
    }


# ============================================================================
# Source tagging wrappers
# ============================================================================
def normalize_snapshot_and_tag_source(
    df: pd.DataFrame,
    source_file: str,
    selected_sheet: str = "",
) -> Dict[str, Any]:
    """
    Normalize a snapshot upload and append source metadata.
    """
    result = normalize_snapshot_dataframe(df)
    snapshot = result["snapshot"].copy()

    snapshot["source_file"] = source_file
    snapshot["selected_sheet"] = selected_sheet

    result["snapshot"] = snapshot
    return result


def normalize_trade_receipt_and_tag_source(
    df: pd.DataFrame,
    source_file: str,
    selected_sheet: str = "",
) -> Dict[str, Any]:
    """
    Normalize a trade receipt upload and append source metadata.
    """
    result = normalize_trade_receipt_dataframe(df)
    trades = result["trades"].copy()

    trades["source_file"] = source_file
    trades["selected_sheet"] = selected_sheet

    result["trades"] = trades
    return result