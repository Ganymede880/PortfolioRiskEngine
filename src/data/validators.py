"""
Validation utilities for uploaded snapshot and trade receipt files.

This module provides:
- shared column resolution helpers
- snapshot-specific validation
- trade-receipt-specific validation
- auto validation dispatch based on upload type
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# ============================================================================
# Column alias maps
# ============================================================================
SNAPSHOT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "team": ["Sector", "sector", "Team", "team"],
    "snapshot_date": ["Date", "date", "As Of Date", "Snapshot Date"],
    "position_side": ["Position", "position", "Side", "side"],
    "ticker": ["Ticker", "ticker", "Symbol", "symbol"],
    "shares": ["Shares", "shares", "Quantity", "quantity"],
    "cost_basis_per_share": ["Cost", "cost", "Price", "price", "Avg Cost", "Average Cost"],
    "total_cost_basis": ["Total Cost", "total cost", "Cost Basis", "cost basis"],
}

TRADE_RECEIPT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "team": ["Sector", "sector", "Team", "team"],
    "trade_side": ["Trade", "trade", "Action", "action", "Side", "side"],
    "ticker": ["Ticker", "ticker", "Symbol", "symbol"],
    "quantity": ["Quantity", "quantity", "Shares", "shares"],
    "gross_price": ["Gross Price", "gross price", "Price", "price", "Execution Price"],
    "commission": ["Commission", "commission"],
    "fees": ["Fees", "fees"],
    "net_cash_amount": ["Net-Net Consideration", "net-net consideration", "Net Consideration", "net consideration"],
    "trade_date": ["Trade Date", "trade date", "Date", "date", "Execution Date"],
    "settlement_date": ["Settlement Date", "settlement date", "Settle Date"],
    "raw_description": ["Description", "description", "Security Description"],
}


SNAPSHOT_REQUIRED_COLUMNS = [
    "team",
    "snapshot_date",
    "position_side",
    "ticker",
    "shares",
]

TRADE_REQUIRED_COLUMNS = [
    "team",
    "trade_side",
    "ticker",
    "quantity",
    "gross_price",
    "settlement_date",
]


# ============================================================================
# Shared helpers
# ============================================================================
def _normalize_column_name(column_name: Any) -> str:
    """
    Convert a column name to a stripped string for comparison.
    """
    return str(column_name).strip()


def _find_matching_column(columns: List[str], aliases: List[str]) -> str | None:
    """
    Return the first matching column from a list of aliases, if present.
    """
    normalized_to_original = {_normalize_column_name(col): col for col in columns}
    for alias in aliases:
        if alias in normalized_to_original:
            return normalized_to_original[alias]
    return None


def resolve_column_mapping(
    df: pd.DataFrame,
    alias_map: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Map canonical internal column names to actual DataFrame column names.
    """
    columns = df.columns.tolist()
    mapping: Dict[str, str] = {}

    for canonical_name, aliases in alias_map.items():
        match = _find_matching_column(columns, aliases)
        if match is not None:
            mapping[canonical_name] = match

    return mapping


def _validate_required_columns(
    df: pd.DataFrame,
    alias_map: Dict[str, List[str]],
    required_columns: List[str],
) -> tuple[List[str], Dict[str, str]]:
    """
    Validate that all required columns are present and return resolved mapping.
    """
    errors: List[str] = []
    resolved_columns = resolve_column_mapping(df, alias_map)

    for required_col in required_columns:
        if required_col not in resolved_columns:
            aliases = alias_map.get(required_col, [required_col])
            errors.append(
                f"Missing required column '{required_col}'. Accepted names: {aliases}"
            )

    return errors, resolved_columns


def _validate_non_empty_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Ensure uploaded DataFrame is not empty.
    """
    if df.empty:
        return ["Uploaded file contains no rows."]
    return []


def _validate_non_blank_values(
    df: pd.DataFrame,
    actual_column_name: str | None,
    field_label: str,
) -> List[str]:
    """
    Ensure a resolved column does not contain blank/null-ish values.
    """
    if actual_column_name is None:
        return []

    series = df[actual_column_name].astype(str).str.strip()
    blank_mask = series.eq("") | series.str.lower().eq("nan")

    if blank_mask.any():
        return [f"{field_label} column contains {int(blank_mask.sum())} blank or invalid value(s)."]

    return []


def _validate_numeric_values(
    df: pd.DataFrame,
    actual_column_name: str | None,
    field_label: str,
) -> List[str]:
    """
    Ensure a resolved column is numeric/coercible for non-empty values.
    """
    if actual_column_name is None:
        return []

    series = df[actual_column_name]
    coerced = pd.to_numeric(series, errors="coerce")
    invalid_mask = series.notna() & coerced.isna()

    if invalid_mask.any():
        return [f"{field_label} column contains {int(invalid_mask.sum())} non-numeric value(s)."]

    return []


def _validate_date_values(
    df: pd.DataFrame,
    actual_column_name: str | None,
    field_label: str,
) -> List[str]:
    """
    Ensure a resolved column is parseable as dates for non-empty values.
    """
    if actual_column_name is None:
        return []

    series = df[actual_column_name]
    parsed = pd.to_datetime(series, errors="coerce")
    invalid_mask = series.notna() & parsed.isna()

    if invalid_mask.any():
        return [f"{field_label} column contains {int(invalid_mask.sum())} invalid date value(s)."]

    return []


def _validate_duplicate_rows(
    df: pd.DataFrame,
    resolved_columns: Dict[str, str],
    subset: List[str],
) -> List[str]:
    """
    Check for duplicate rows based on canonical column subset.
    """
    actual_columns: List[str] = []

    for canonical_col in subset:
        actual_col = resolved_columns.get(canonical_col)
        if actual_col is not None:
            actual_columns.append(actual_col)

    if not actual_columns:
        return []

    duplicate_count = int(df.duplicated(subset=actual_columns).sum())
    if duplicate_count > 0:
        return [f"Found {duplicate_count} duplicate row(s) based on columns: {subset}"]

    return []


# ============================================================================
# Snapshot validation
# ============================================================================
def validate_snapshot_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate an uploaded portfolio snapshot DataFrame.
    """
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(_validate_non_empty_dataframe(df))

    required_errors, resolved_columns = _validate_required_columns(
        df=df,
        alias_map=SNAPSHOT_COLUMN_ALIASES,
        required_columns=SNAPSHOT_REQUIRED_COLUMNS,
    )
    errors.extend(required_errors)

    errors.extend(_validate_non_blank_values(df, resolved_columns.get("team"), "Team/Sector"))
    errors.extend(_validate_non_blank_values(df, resolved_columns.get("position_side"), "Position"))
    errors.extend(_validate_non_blank_values(df, resolved_columns.get("ticker"), "Ticker"))

    errors.extend(_validate_numeric_values(df, resolved_columns.get("shares"), "Shares"))
    errors.extend(_validate_numeric_values(df, resolved_columns.get("cost_basis_per_share"), "Cost"))
    errors.extend(_validate_numeric_values(df, resolved_columns.get("total_cost_basis"), "Total Cost Basis"))

    errors.extend(_validate_date_values(df, resolved_columns.get("snapshot_date"), "Snapshot Date"))

    errors.extend(
        _validate_duplicate_rows(
            df=df,
            resolved_columns=resolved_columns,
            subset=["snapshot_date", "team", "ticker", "position_side"],
        )
    )

    if "cost_basis_per_share" not in resolved_columns:
        warnings.append("No per-share cost basis column detected. Cost basis tracking may be incomplete.")

    if "total_cost_basis" not in resolved_columns:
        warnings.append("No total cost basis column detected. This can still be derived later if needed.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "resolved_columns": resolved_columns,
        "upload_type": "snapshot",
    }


# ============================================================================
# Trade receipt validation
# ============================================================================
def validate_trade_receipt_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate an uploaded trade receipt DataFrame.
    """
    errors: List[str] = []
    warnings: List[str] = []

    errors.extend(_validate_non_empty_dataframe(df))

    required_errors, resolved_columns = _validate_required_columns(
        df=df,
        alias_map=TRADE_RECEIPT_COLUMN_ALIASES,
        required_columns=TRADE_REQUIRED_COLUMNS,
    )
    errors.extend(required_errors)

    errors.extend(_validate_non_blank_values(df, resolved_columns.get("team"), "Team/Sector"))
    errors.extend(_validate_non_blank_values(df, resolved_columns.get("trade_side"), "Trade Side"))
    errors.extend(_validate_non_blank_values(df, resolved_columns.get("ticker"), "Ticker"))

    errors.extend(_validate_numeric_values(df, resolved_columns.get("quantity"), "Quantity"))
    errors.extend(_validate_numeric_values(df, resolved_columns.get("gross_price"), "Gross Price"))
    errors.extend(_validate_numeric_values(df, resolved_columns.get("commission"), "Commission"))
    errors.extend(_validate_numeric_values(df, resolved_columns.get("fees"), "Fees"))
    errors.extend(_validate_numeric_values(df, resolved_columns.get("net_cash_amount"), "Net Cash Amount"))

    errors.extend(_validate_date_values(df, resolved_columns.get("trade_date"), "Trade Date"))
    errors.extend(_validate_date_values(df, resolved_columns.get("settlement_date"), "Settlement Date"))

    errors.extend(
        _validate_duplicate_rows(
            df=df,
            resolved_columns=resolved_columns,
            subset=["team", "trade_side", "ticker", "quantity", "gross_price", "settlement_date"],
        )
    )

    if "trade_date" not in resolved_columns:
        warnings.append("No trade date column detected. Settlement date will be used as the activity date if needed.")

    if "net_cash_amount" not in resolved_columns:
        warnings.append("No net cash amount column detected. Cash effect may need to be derived from price, quantity, and fees.")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "resolved_columns": resolved_columns,
        "upload_type": "trade_receipt",
    }


# ============================================================================
# Auto dispatch
# ============================================================================
def validate_uploaded_dataframe(
    df: pd.DataFrame,
    upload_type: str,
) -> Dict[str, Any]:
    """
    Validate an uploaded DataFrame based on upload type.

    upload_type must be one of:
    - "snapshot"
    - "trade_receipt"
    """
    if upload_type == "snapshot":
        return validate_snapshot_dataframe(df)

    if upload_type in {"trade_receipt", "sector_rebalance", "portfolio_liquidation"}:
        return validate_trade_receipt_dataframe(df)

    return {
        "is_valid": False,
        "errors": [f"Unsupported upload_type: {upload_type}"],
        "warnings": [],
        "resolved_columns": {},
        "upload_type": upload_type,
    }
