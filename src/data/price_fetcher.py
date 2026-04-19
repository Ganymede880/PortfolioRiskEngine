"""
Market data utilities using Yahoo Finance.

This module is responsible for:
- fetching latest prices for a list of tickers
- fetching historical price series
- basic caching to avoid repeated API calls
- graceful handling of bad/missing tickers

Uses yfinance for MVP purposes.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yfinance as yf

from src.config.settings import settings


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _cache_path_for_ticker(ticker: str) -> Path:
    """
    Return cache file path for a ticker.
    """
    safe_ticker = ticker.replace("/", "-")
    return settings.cache_dir / f"{safe_ticker}.csv"


@contextmanager
def _yfinance_request_context():
    """
    Normalize the yfinance runtime environment for local desktop use.

    Some environments inject dead proxy settings that cause Yahoo requests to
    fail and return empty DataFrames. We temporarily clear those vars here and
    force yfinance to use a writable local cache directory.
    """
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]
    previous_env = {key: os.environ.get(key) for key in proxy_keys}

    try:
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(key, None)

        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"

        yf_cache_dir = settings.cache_dir / "yfinance"
        yf_cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(yf_cache_dir.resolve()))

        yield
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _is_cache_fresh(file_path: Path, max_age_seconds: int) -> bool:
    """
    Check if cache file is fresh enough.
    """
    if not file_path.exists():
        return False

    modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
    age = (datetime.utcnow() - modified_time).total_seconds()
    return age < max_age_seconds


def _normalize_yfinance_history_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize a yfinance history frame to the app's canonical schema.

    Recent yfinance responses may omit `Adj Close`, so we fall back to `Close`.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    normalized = df.reset_index().copy()
    normalized = normalized.rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Close": "close",
            "Adj Close": "adj_close",
        }
    )

    if "date" not in normalized.columns:
        first_col = normalized.columns[0]
        normalized = normalized.rename(columns={first_col: "date"})

    if "close" not in normalized.columns:
        normalized["close"] = pd.NA

    if "adj_close" not in normalized.columns:
        normalized["adj_close"] = normalized["close"]

    normalized["ticker"] = ticker
    return normalized[["date", "ticker", "close", "adj_close"]]


# ---------------------------------------------------------------------
# Latest price fetching
# ---------------------------------------------------------------------
def fetch_latest_prices(tickers: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fetch latest prices for a list of tickers.

    Returns:
        (prices_df, failed_tickers)

    prices_df columns:
        ticker, price, timestamp
    """
    tickers = list(set([t for t in tickers if isinstance(t, str) and t.strip() != ""]))
    if not tickers:
        return pd.DataFrame(columns=["ticker", "price", "timestamp"]), []

    with _yfinance_request_context():
        data = yf.download(
            tickers=tickers,
            period="1d",
            interval="1m",
            progress=False,
            group_by="ticker",
        )

    results = []
    failed_tickers: List[str] = []

    timestamp = datetime.utcnow()

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                # yfinance returns different structure for single ticker
                last_price = data["Close"].dropna().iloc[-1]
            else:
                last_price = data[ticker]["Close"].dropna().iloc[-1]

            results.append({
                "ticker": ticker,
                "price": float(last_price),
                "timestamp": timestamp,
            })

        except Exception:
            failed_tickers.append(ticker)

    prices_df = pd.DataFrame(results)
    return prices_df, failed_tickers


# ---------------------------------------------------------------------
# Historical price fetching
# ---------------------------------------------------------------------
def fetch_price_history(
    ticker: str,
    lookback_days: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical daily prices for a ticker.

    Returns DataFrame with:
        date, ticker, close, adj_close
    """
    if lookback_days is None:
        lookback_days = settings.history_lookback_days

    cache_path = _cache_path_for_ticker(ticker)

    # Use cache if available and fresh
    if use_cache and _is_cache_fresh(cache_path, settings.price_refresh_interval_seconds):
        try:
            return pd.read_csv(cache_path, parse_dates=["date"])
        except Exception:
            pass  # fall back to fetching

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        with _yfinance_request_context():
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
            )

        if df.empty:
            return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

        df = _normalize_yfinance_history_frame(df, ticker)

        # Save to cache
        try:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False)
        except Exception:
            pass

        return df

    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])


# ---------------------------------------------------------------------
# Batch historical fetch
# ---------------------------------------------------------------------
def fetch_multiple_price_histories(
    tickers: List[str],
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    Robust batch historical fetch.

    Improvements:
    - filters invalid tickers (CASH, FX placeholders)
    - normalizes tickers (BRKB → BRK-B)
    - fetches in one batch instead of loop
    - gracefully skips failures
    """
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    # --- Normalize & filter ---
    def _clean_ticker(t: str) -> str:
        t = str(t).strip().upper()

        # Skip non-equity placeholders
        if t in {"", "CASH", "EUR", "GBP"}:
            return ""

        # Fix common Yahoo mismatches
        if t == "BRKB":
            return "BRK-B"
        if t == "BRKA":
            return "BRK-A"

        return t

    cleaned = list(set(filter(None, [_clean_ticker(t) for t in tickers])))

    if not cleaned:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    # --- Fetch in batch ---
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days or settings.history_lookback_days)

    try:
        with _yfinance_request_context():
            data = yf.download(
                tickers=cleaned,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                group_by="ticker",
            )
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    frames = []

    # --- Handle single vs multi ticker structure ---
    if len(cleaned) == 1:
        ticker = cleaned[0]
        df = data.copy()

        if df.empty:
            return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

        frames.append(_normalize_yfinance_history_frame(df, ticker))

    else:
        for ticker in cleaned:
            try:
                df = data[ticker].copy()

                if df.empty:
                    continue

                frames.append(_normalize_yfinance_history_frame(df, ticker))

            except Exception:
                continue

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------
# Utility: merge prices into holdings
# ---------------------------------------------------------------------
def attach_latest_prices(
    holdings_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge latest prices into holdings DataFrame.

    Assumes holdings_df has 'ticker' column.
    """
    if holdings_df.empty or price_df.empty:
        return holdings_df.copy()

    merged = holdings_df.merge(
        price_df[["ticker", "price"]],
        on="ticker",
        how="left",
    )

    return merged


# ---------------------------------------------------------------------
# Utility: identify missing prices
# ---------------------------------------------------------------------
def find_missing_prices(price_df: pd.DataFrame, expected_tickers: List[str]) -> List[str]:
    """
    Return list of tickers for which no price was returned.
    """
    returned = set(price_df["ticker"].unique())
    expected = set(expected_tickers)
    missing = list(expected - returned)
    return sorted(missing)
