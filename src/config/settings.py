"""
Global application settings for the CMCSIF portfolio tracker.

This module centralizes configuration so the rest of the project does not
hard-code important values like team names, folder paths, benchmark symbols,
or refresh behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

# Load environment variables from .env at project root
load_dotenv()


@dataclass
class Settings:
    """
    Container for project-wide configuration values.
    """

    # ---------------------------------------------------------------------
    # Project paths
    # ---------------------------------------------------------------------
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def mappings_dir(self) -> Path:
        return self.data_dir / "mappings"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def sample_uploads_dir(self) -> Path:
        return self.data_dir / "sample_uploads"

    # ---------------------------------------------------------------------
    # App metadata
    # ---------------------------------------------------------------------
    app_name: str = "CMCSIF Portfolio Tracker"
    app_short_name: str = "CMCSIF Tracker"

    # ---------------------------------------------------------------------
    # Portfolio / team structure
    # ---------------------------------------------------------------------
    canonical_teams: List[str] = field(default_factory=lambda: [
        "Consumer",
        "E&U",
        "F&R",
        "Healthcare",
        "TMT",
        "M&I",
        "Cash",
    ])

    display_team_order: List[str] = field(default_factory=lambda: [
        "Consumer",
        "E&U",
        "F&R",
        "Healthcare",
        "TMT",
        "M&I",
        "Cash",
    ])

    raw_to_team_defaults: Dict[str, str] = field(default_factory=lambda: {
        "Consumer": "Consumer",
        "Energy": "E&U",
        "Financials": "F&R",
        "Healthcare": "Healthcare",
        "TMT": "TMT",
        "Technology & Telecommunication Services": "TMT",
        "Industrials": "M&I",
        "Materials": "M&I",
        "Cash": "Cash",
    })

    cash_team_name: str = "Cash"

    # ---------------------------------------------------------------------
    # Upload / file handling
    # ---------------------------------------------------------------------
    supported_file_extensions: List[str] = field(default_factory=lambda: [".csv", ".xlsx", ".xls"])

    # These are likely to evolve once we finalize the exact upload template.
    required_holdings_columns: List[str] = field(default_factory=lambda: [
        "Ticker",
        "Shares",
    ])

    optional_holdings_columns: List[str] = field(default_factory=lambda: [
        "Sector",
        "Date",
        "Price",
        "Market Value",
        "Cost",
        "% of Total",
    ])

    preferred_holdings_sheet_names: List[str] = field(default_factory=lambda: [
        "SIF Portfolio",
        "Export - NT Brokerage Positions",
    ])

    # ---------------------------------------------------------------------
    # Market data / analytics
    # ---------------------------------------------------------------------
    benchmark_ticker: str = "SPY"
    risk_free_rate_annual: float = 0.04
    trading_days_per_year: int = 252
    price_refresh_interval_seconds: int = 300  # 5 minutes
    history_lookback_days: int = 365

    # ---------------------------------------------------------------------
    # Database
    # ---------------------------------------------------------------------
    database_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        "sqlite:///cmcsif_portfolio.db",
    ))

    # ---------------------------------------------------------------------
    # Mapping files
    # ---------------------------------------------------------------------
    @property
    def team_map_path(self) -> Path:
        return self.mappings_dir / "team_map.csv"

    @property
    def ticker_map_path(self) -> Path:
        return self.mappings_dir / "ticker_map.csv"

    # ---------------------------------------------------------------------
    # Helper methods
    # ---------------------------------------------------------------------
    def ensure_directories_exist(self) -> None:
        """
        Create core data directories if they do not already exist.
        Safe to call at app startup.
        """
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.mappings_dir,
            self.cache_dir,
            self.sample_uploads_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def is_valid_team(self, team_name: str) -> bool:
        """
        Return True if the supplied team name is one of the canonical teams.
        """
        return team_name in self.canonical_teams


# Singleton-style settings object used throughout the project
settings = Settings()
