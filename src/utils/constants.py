"""
Project-wide constants.

These are static values used across the application that are not environment-
dependent. This helps avoid hardcoding strings in multiple places and keeps
the codebase consistent and readable.
"""

from typing import Dict, List


# ---------------------------------------------------------------------
# Core column names (canonical internal schema)
# ---------------------------------------------------------------------
COL_TICKER: str = "ticker"
COL_TEAM: str = "team"
COL_SHARES: str = "shares"
COL_PRICE: str = "price"
COL_MARKET_VALUE: str = "market_value"
COL_WEIGHT: str = "weight"
COL_DATE: str = "date"
COL_RETURN: str = "return"
COL_PNL: str = "pnl"


# ---------------------------------------------------------------------
# Standard column order for holdings display
# ---------------------------------------------------------------------
HOLDINGS_DISPLAY_COLUMNS: List[str] = [
    COL_TICKER,
    COL_TEAM,
    COL_SHARES,
    COL_PRICE,
    COL_MARKET_VALUE,
    COL_WEIGHT,
    COL_PNL,
]


# ---------------------------------------------------------------------
# Raw upload column aliases
# These help normalize different possible naming conventions
# ---------------------------------------------------------------------
COLUMN_ALIASES: Dict[str, List[str]] = {
    COL_TICKER: ["Ticker", "ticker", "Symbol", "symbol"],
    COL_SHARES: ["Shares", "shares", "Quantity", "quantity"],
    COL_TEAM: ["Team", "team", "Sector", "sector"],
    COL_PRICE: ["Price", "price", "Last Price", "last_price"],
    COL_MARKET_VALUE: ["Market Value", "market_value", "Value"],
    COL_DATE: ["Date", "date", "As Of Date"],
}


# ---------------------------------------------------------------------
# Factor names (for exposure page)
# ---------------------------------------------------------------------
FACTOR_SIZE: str = "size"
FACTOR_VALUE: str = "value"
FACTOR_MOMENTUM: str = "momentum"

SUPPORTED_FACTORS: List[str] = [
    FACTOR_SIZE,
    FACTOR_VALUE,
    FACTOR_MOMENTUM,
]


# ---------------------------------------------------------------------
# Shared chart palettes
# ---------------------------------------------------------------------
TEAM_COLORS: Dict[str, str] = {
    "Consumer": "#C6D4FF",
    "E&U": "#7A82AB",
    "F&R": "#307473",
    "Healthcare": "#12664F",
    "TMT": "#2DC2BD",
    "M&I": "#3F3047",
    "Cash": "#7A82AB",
}

FACTOR_COLORS: Dict[str, str] = {
    "Market": TEAM_COLORS["Healthcare"],
    "Size": TEAM_COLORS["E&U"],
    "Momentum": TEAM_COLORS["TMT"],
    "Value": TEAM_COLORS["F&R"],
    "Idiosyncratic": TEAM_COLORS["M&I"],
}


# ---------------------------------------------------------------------
# Performance metric names
# ---------------------------------------------------------------------
METRIC_DAILY_RETURN: str = "daily_return"
METRIC_CUM_RETURN: str = "cumulative_return"
METRIC_SHARPE: str = "sharpe"
METRIC_SORTINO: str = "sortino"
METRIC_VOLATILITY: str = "volatility"


# ---------------------------------------------------------------------
# Streamlit display labels (cleaner UI naming)
# ---------------------------------------------------------------------
DISPLAY_LABELS: Dict[str, str] = {
    COL_TICKER: "Ticker",
    COL_TEAM: "Team",
    COL_SHARES: "Shares",
    COL_PRICE: "Price",
    COL_MARKET_VALUE: "Market Value",
    COL_WEIGHT: "Weight",
    COL_PNL: "Daily P&L",
    METRIC_DAILY_RETURN: "Daily Return",
    METRIC_CUM_RETURN: "Cumulative Return",
    METRIC_SHARPE: "Sharpe Ratio",
    METRIC_SORTINO: "Sortino Ratio",
}


# ---------------------------------------------------------------------
# Default numeric formatting
# ---------------------------------------------------------------------
DEFAULT_DECIMALS: int = 4
PERCENT_MULTIPLIER: float = 100.0


# ---------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------
CASH_TICKER_PLACEHOLDER: str = "CASH"
UNKNOWN_TEAM: str = "UNKNOWN"
