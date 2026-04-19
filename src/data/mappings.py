"""
Mapping utilities for uploaded holdings data.

This module loads and applies:
- team mappings (raw sector/team labels -> canonical team names)
- ticker mappings (raw uploaded ticker -> Yahoo Finance ticker)

It is designed to be transparent and safe:
- missing mapping files do not crash the app
- unmapped values are surfaced for review
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.config.settings import settings


TEAM_MAP_REQUIRED_COLUMNS = {"raw_sector", "team"}
TICKER_MAP_REQUIRED_COLUMNS = {"raw_ticker", "yahoo_ticker"}


def _safe_read_csv(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV file if it exists, otherwise return an empty DataFrame.
    """
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_team_map() -> pd.DataFrame:
    """
    Load team mapping CSV from disk.

    Expected columns:
    - raw_sector
    - team
    """
    df = _safe_read_csv(settings.team_map_path)

    if df.empty:
        # Fall back to defaults defined in settings
        fallback_rows = [
            {"raw_sector": raw_value, "team": team_name}
            for raw_value, team_name in settings.raw_to_team_defaults.items()
        ]
        return pd.DataFrame(fallback_rows)

    missing = TEAM_MAP_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"team_map.csv is missing required columns: {sorted(missing)}"
        )

    return df.copy()


def load_ticker_map() -> pd.DataFrame:
    """
    Load ticker mapping CSV from disk.

    Expected columns:
    - raw_ticker
    - yahoo_ticker
    """
    df = _safe_read_csv(settings.ticker_map_path)

    if df.empty:
        return pd.DataFrame(columns=["raw_ticker", "yahoo_ticker"])

    missing = TICKER_MAP_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"ticker_map.csv is missing required columns: {sorted(missing)}"
        )

    return df.copy()


def build_team_mapping_dict(team_map_df: pd.DataFrame) -> Dict[str, str]:
    """
    Convert a team mapping DataFrame into a dictionary.

    Keys and values are stripped strings.
    """
    if team_map_df.empty:
        return {}

    mapping: Dict[str, str] = {}
    for _, row in team_map_df.iterrows():
        raw_value = str(row["raw_sector"]).strip()
        canonical_team = str(row["team"]).strip()
        mapping[raw_value] = canonical_team

    return mapping


def build_ticker_mapping_dict(ticker_map_df: pd.DataFrame) -> Dict[str, str]:
    """
    Convert a ticker mapping DataFrame into a dictionary.

    Blank yahoo_ticker values are ignored.
    """
    if ticker_map_df.empty:
        return {}

    mapping: Dict[str, str] = {}
    for _, row in ticker_map_df.iterrows():
        raw_ticker = str(row["raw_ticker"]).strip()
        yahoo_ticker = str(row["yahoo_ticker"]).strip()

        if yahoo_ticker and yahoo_ticker.lower() != "nan":
            mapping[raw_ticker] = yahoo_ticker

    return mapping


def apply_team_mapping(
    df: pd.DataFrame,
    source_column: str,
    output_column: str = "team",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply raw sector/team mapping to a DataFrame.

    Parameters:
        df: input DataFrame
        source_column: column containing raw sector/team labels
        output_column: destination column for canonical team names

    Returns:
        (mapped_df, unmapped_values)
    """
    team_map_df = load_team_map()
    team_mapping = build_team_mapping_dict(team_map_df)

    mapped_df = df.copy()
    raw_values = mapped_df[source_column].astype(str).str.strip()

    mapped_df[output_column] = raw_values.map(team_mapping)

    # If raw value is already a canonical team, preserve it directly
    canonical_set = set(settings.canonical_teams)
    already_canonical_mask = raw_values.isin(canonical_set)
    mapped_df.loc[already_canonical_mask, output_column] = raw_values[already_canonical_mask]

    unmapped_mask = mapped_df[output_column].isna() & raw_values.ne("") & raw_values.str.lower().ne("nan")
    unmapped_values = sorted(raw_values[unmapped_mask].dropna().unique().tolist())

    return mapped_df, unmapped_values


def apply_ticker_mapping(
    df: pd.DataFrame,
    source_column: str,
    output_column: str = "ticker",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply ticker mapping to a DataFrame.

    Parameters:
        df: input DataFrame
        source_column: column containing raw tickers
        output_column: destination column for Yahoo-compatible tickers

    Returns:
        (mapped_df, unmapped_values)

    Behavior:
    - if a raw ticker exists in ticker_map.csv, use the mapped ticker
    - otherwise keep the original raw ticker
    - also returns suspicious/unmapped tickers for review
    """
    ticker_map_df = load_ticker_map()
    ticker_mapping = build_ticker_mapping_dict(ticker_map_df)

    mapped_df = df.copy()
    raw_tickers = mapped_df[source_column].astype(str).str.strip()

    mapped_df[output_column] = raw_tickers.map(ticker_mapping)

    # Preserve original ticker where no explicit mapping exists
    missing_map_mask = mapped_df[output_column].isna()
    mapped_df.loc[missing_map_mask, output_column] = raw_tickers[missing_map_mask]

    # Flag values that were not explicitly mapped and look unusual
    # This is intentionally simple for MVP.
    suspicious_mask = (
        ~raw_tickers.isin(ticker_mapping.keys())
        & raw_tickers.ne("")
        & raw_tickers.str.lower().ne("nan")
        & ~raw_tickers.str.fullmatch(r"[A-Za-z0-9.\-^=]+", na=False)
    )
    suspicious_values = sorted(raw_tickers[suspicious_mask].dropna().unique().tolist())

    return mapped_df, suspicious_values


def summarize_mapping_status(unmapped_teams: List[str], suspicious_tickers: List[str]) -> Dict[str, object]:
    """
    Return a structured mapping summary for UI display or logs.
    """
    return {
        "has_mapping_issues": bool(unmapped_teams or suspicious_tickers),
        "unmapped_teams": unmapped_teams,
        "suspicious_tickers": suspicious_tickers,
    }