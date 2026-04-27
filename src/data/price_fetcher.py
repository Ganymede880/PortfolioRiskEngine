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
import json
import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _cache_path_for_named_payload(name: str, suffix: str) -> Path:
    safe_name = str(name).strip().replace("/", "-").replace("\\", "-")
    return settings.cache_dir / f"{safe_name}{suffix}"


def _read_json_cache(cache_path: Path):
    try:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _write_json_cache(cache_path: Path, payload) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        pass


def _clean_text_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "<na>"}:
        return None
    return text


def _normalize_symbol(value: Any) -> str:
    cleaned = _clean_text_label(value)
    return cleaned.upper() if cleaned else ""


def _read_price_cache(cache_path: Path) -> pd.DataFrame:
    try:
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["date"])
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _write_price_cache(cache_path: Path, df: pd.DataFrame) -> None:
    try:
        if df.empty:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    except Exception:
        pass


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


def _historical_cache_ttl_seconds(lookback_days: int | None = None) -> int:
    """
    Historical daily bars change slowly, so keep them much longer than live quotes.
    """
    requested_days = int(lookback_days or settings.history_lookback_days)
    if requested_days >= 365:
        return 24 * 60 * 60
    if requested_days >= 90:
        return 12 * 60 * 60
    return max(settings.price_refresh_interval_seconds, 6 * 60 * 60)


def _cache_covers_start_date(df: pd.DataFrame, start_date: datetime) -> bool:
    if df.empty or "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    return dates.min() <= pd.Timestamp(start_date).normalize()


def _latest_expected_history_date(reference_time: datetime | None = None) -> pd.Timestamp:
    timestamp = pd.Timestamp(reference_time or datetime.utcnow()).normalize()
    return pd.bdate_range(end=timestamp, periods=1)[0].normalize()


def _cache_covers_end_date(df: pd.DataFrame, end_date: datetime) -> bool:
    if df.empty or "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    return dates.max().normalize() >= _latest_expected_history_date(end_date)


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


def _normalize_sector_key(key: str) -> str:
    return (
        str(key)
        .strip()
        .lower()
        .replace("&", "and")
        .replace("-", " ")
        .replace("/", " ")
        .replace("_", " ")
    )


SP500_SECTOR_ALIASES: Dict[str, list[str]] = {
    "materials": [
        "basic materials",
        "materials",
    ],
    "industrials": [
        "industrials",
    ],
    "financials": [
        "financial services",
        "financials",
    ],
    "real_estate": [
        "real estate",
    ],
    "consumer_cyclical": [
        "consumer cyclical",
        "consumer discretionary",
    ],
    "consumer_defensive": [
        "consumer defensive",
        "consumer staples",
    ],
    "utilities": [
        "utilities",
    ],
    "energy": [
        "energy",
    ],
    "healthcare": [
        "healthcare",
        "health care",
    ],
    "communication_services": [
        "communication services",
        "communications",
    ],
    "technology": [
        "technology",
        "information technology",
    ],
}


def _normalize_security_profile_payload(ticker: str, info: dict[str, Any] | None) -> dict[str, Any]:
    info = info if isinstance(info, dict) else {}
    return {
        "ticker": _normalize_symbol(ticker),
        "quote_type": _clean_text_label(info.get("quoteType")),
        "sector": _clean_text_label(info.get("sector")),
        "industry": _clean_text_label(info.get("industry")),
        "category": _clean_text_label(info.get("category")),
        "fund_family": _clean_text_label(info.get("fundFamily")),
        "long_name": _clean_text_label(info.get("longName")),
        "short_name": _clean_text_label(info.get("shortName")),
    }


def _normalize_top_holdings_frame(raw_top_holdings) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["ticker", "weight"])
    if raw_top_holdings is None:
        return empty

    if isinstance(raw_top_holdings, pd.DataFrame):
        working = raw_top_holdings.reset_index().copy()
    elif isinstance(raw_top_holdings, dict):
        working = pd.DataFrame(list(raw_top_holdings.items()), columns=["ticker", "weight"])
    elif isinstance(raw_top_holdings, list):
        working = pd.DataFrame(raw_top_holdings)
    else:
        return empty

    if working.empty:
        return empty

    symbol_col = next(
        (
            col
            for col in working.columns
            if str(col).strip().lower() in {"symbol", "ticker", "holding", "holdingsymbol"}
        ),
        None,
    )
    if symbol_col is None:
        object_cols = [col for col in working.columns if pd.api.types.is_object_dtype(working[col])]
        symbol_col = object_cols[0] if object_cols else None
    if symbol_col is None:
        return empty

    weight_col = next(
        (
            col
            for col in working.columns
            if "weight" in str(col).strip().lower() or "percent" in str(col).strip().lower()
        ),
        None,
    )
    if weight_col is None:
        numeric_cols = [
            col
            for col in working.columns
            if col != symbol_col and pd.api.types.is_numeric_dtype(working[col])
        ]
        weight_col = numeric_cols[0] if numeric_cols else None
    if weight_col is None:
        return empty

    normalized = working[[symbol_col, weight_col]].copy()
    normalized.columns = ["ticker", "weight"]
    normalized["ticker"] = normalized["ticker"].map(_normalize_symbol)
    normalized["weight"] = pd.to_numeric(normalized["weight"], errors="coerce")
    normalized = normalized.loc[normalized["ticker"].ne("") & normalized["weight"].gt(0)].copy()
    if normalized.empty:
        return empty

    weight_sum = float(normalized["weight"].sum())
    if weight_sum <= 0:
        return empty
    normalized["weight"] = normalized["weight"] / weight_sum
    return normalized.groupby("ticker", as_index=False)["weight"].sum()


def fetch_live_security_profiles(tickers: List[str]) -> pd.DataFrame:
    columns = ["ticker", "quote_type", "sector", "industry", "category", "fund_family", "long_name", "short_name"]
    unique = sorted({_normalize_symbol(ticker) for ticker in tickers if _normalize_symbol(ticker)})
    if not unique:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for ticker in unique:
        cache_path = _cache_path_for_named_payload(f"security_profile_{ticker}", ".json")
        cached = _read_json_cache(cache_path)
        if _is_cache_fresh(cache_path, settings.price_refresh_interval_seconds) and isinstance(cached, dict):
            rows.append(_normalize_security_profile_payload(ticker, cached))
            continue

        payload = None
        try:
            with _yfinance_request_context():
                payload = _normalize_security_profile_payload(ticker, yf.Ticker(ticker).info)
        except Exception:
            payload = None

        if payload and any(payload.get(col) for col in columns if col != "ticker"):
            _write_json_cache(cache_path, payload)
            rows.append(payload)
            continue

        if isinstance(cached, dict):
            rows.append(_normalize_security_profile_payload(ticker, cached))
        else:
            rows.append(_normalize_security_profile_payload(ticker, {}))

    return pd.DataFrame(rows, columns=columns)


def fetch_etf_top_holdings(ticker: str) -> pd.DataFrame:
    normalized_ticker = _normalize_symbol(ticker)
    if not normalized_ticker:
        return pd.DataFrame(columns=["ticker", "weight"])

    cache_path = _cache_path_for_named_payload(f"etf_top_holdings_{normalized_ticker}", ".json")
    cached = _read_json_cache(cache_path)
    if _is_cache_fresh(cache_path, settings.price_refresh_interval_seconds) and isinstance(cached, list):
        return _normalize_top_holdings_frame(cached)

    normalized = pd.DataFrame(columns=["ticker", "weight"])
    try:
        with _yfinance_request_context():
            normalized = _normalize_top_holdings_frame(yf.Ticker(normalized_ticker).funds_data.top_holdings)
    except Exception:
        normalized = pd.DataFrame(columns=["ticker", "weight"])

    if not normalized.empty:
        _write_json_cache(cache_path, normalized.to_dict(orient="records"))
        return normalized

    if isinstance(cached, list):
        return _normalize_top_holdings_frame(cached)
    return normalized


def _coerce_sector_weightings(raw_sector_weightings) -> Dict[str, float]:
    """
    Normalize yfinance fund sector weightings to a simple dict of float weights.
    """
    if raw_sector_weightings is None:
        return {}

    if isinstance(raw_sector_weightings, dict):
        items = raw_sector_weightings.items()
    elif isinstance(raw_sector_weightings, pd.Series):
        items = raw_sector_weightings.to_dict().items()
    elif isinstance(raw_sector_weightings, pd.DataFrame):
        working = raw_sector_weightings.copy()
        if working.empty:
            return {}

        if {"sector", "weight"}.issubset(working.columns):
            items = zip(working["sector"], working["weight"])
        elif working.shape[1] >= 2:
            items = zip(working.iloc[:, 0], working.iloc[:, 1])
        else:
            return {}
    elif isinstance(raw_sector_weightings, list):
        items = []
        for row in raw_sector_weightings:
            if isinstance(row, dict):
                sector = row.get("sector", row.get("name", row.get("label")))
                weight = row.get("weight", row.get("value"))
                items.append((sector, weight))
    else:
        return {}

    normalized: Dict[str, float] = {}
    for key, value in items:
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric_value):
            continue
        normalized[_normalize_sector_key(str(key))] = float(numeric_value)

    return normalized


def _sector_weight_from_aliases(
    sector_weights: Dict[str, float],
    sector_name: str,
) -> float | None:
    total = 0.0
    found_any = False
    for alias in SP500_SECTOR_ALIASES.get(sector_name, []):
        alias_key = _normalize_sector_key(alias)
        if alias_key in sector_weights:
            total += float(sector_weights[alias_key])
            found_any = True
            break
    return total if found_any else None


def fetch_sp500_sector_proxy_weights() -> Dict[str, float | None]:
    """
    Fetch live SPY sector weights keyed by normalized sector bucket.

    These are used as a practical market-cap proxy when constructing
    composite pod benchmarks from sector ETFs.
    """
    default_result: Dict[str, float | None] = {
        "materials": None,
        "industrials": None,
        "financials": None,
        "real_estate": None,
        "consumer_cyclical": None,
        "consumer_defensive": None,
        "utilities": None,
        "energy": None,
        "healthcare": None,
        "communication_services": None,
        "technology": None,
    }
    cache_path = _cache_path_for_named_payload("sp500_sector_proxy_weights", ".json")
    cached = _read_json_cache(cache_path)
    if _is_cache_fresh(cache_path, settings.price_refresh_interval_seconds):
        return cached if isinstance(cached, dict) and cached else default_result

    try:
        with _yfinance_request_context():
            raw_sector_weightings = yf.Ticker("SPY").funds_data.sector_weightings
    except Exception:
        return cached if isinstance(cached, dict) and cached else default_result

    sector_weights = _coerce_sector_weightings(raw_sector_weightings)
    if not sector_weights:
        return cached if isinstance(cached, dict) and cached else default_result

    result: Dict[str, float | None] = {
        sector_name: _sector_weight_from_aliases(sector_weights, sector_name)
        for sector_name in default_result
    }

    if any(value is not None for value in result.values()):
        _write_json_cache(cache_path, result)

    return result


def fetch_sp500_sector_group_weights() -> Dict[str, float | None]:
    """
    Fetch live S&P 500 sector weights from SPY's reported sector exposure and
    aggregate them into the portfolio's pod buckets.

    We use SPY as a practical live proxy for current S&P 500 sector weights.
    """
    default_result: Dict[str, float | None] = {
        "Consumer": None,
        "E&U": None,
        "F&R": None,
        "Healthcare": None,
        "TMT": None,
        "M&I": None,
        "Cash": 0.0,
    }
    cache_path = _cache_path_for_named_payload("sp500_sector_group_weights", ".json")
    cached = _read_json_cache(cache_path)
    if _is_cache_fresh(cache_path, settings.price_refresh_interval_seconds):
        return cached if isinstance(cached, dict) and cached else default_result

    sector_weights = fetch_sp500_sector_proxy_weights()
    if not sector_weights:
        return cached if isinstance(cached, dict) and cached else default_result

    def _sector_weight(*sector_names: str) -> float:
        values = [sector_weights.get(sector_name) for sector_name in sector_names]
        values = [float(value) for value in values if value is not None and not pd.isna(value)]
        return sum(values) if values else float("nan")

    result: Dict[str, float | None] = {
        "Consumer": _sector_weight("consumer_cyclical", "consumer_defensive"),
        "E&U": _sector_weight("utilities", "energy"),
        "F&R": _sector_weight("financials", "real_estate"),
        "Healthcare": _sector_weight("healthcare"),
        "TMT": _sector_weight("communication_services", "technology"),
        "M&I": _sector_weight("materials", "industrials"),
        "Cash": 0.0,
    }

    for team, value in list(result.items()):
        if pd.isna(value):
            result[team] = None

    if any(value is not None for key, value in result.items() if key != "Cash"):
        _write_json_cache(cache_path, result)

    return result


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
    tickers = sorted(tickers)
    ticker_hash = hashlib.sha1(json.dumps(tickers).encode("utf-8")).hexdigest()
    cache_key = _cache_path_for_named_payload(
        f"latest_prices_{ticker_hash}",
        ".json",
    )
    if _is_cache_fresh(cache_key, settings.price_refresh_interval_seconds):
        cached = _read_json_cache(cache_key)
        if isinstance(cached, dict):
            prices_payload = cached.get("prices", [])
            failed_payload = cached.get("failed_tickers", [])
            cached_df = pd.DataFrame(prices_payload)
            if not cached_df.empty and "timestamp" in cached_df.columns:
                cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"], errors="coerce")
            if not cached_df.empty or failed_payload:
                return cached_df, list(failed_payload)

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
    if not prices_df.empty or failed_tickers:
        _write_json_cache(
            cache_key,
            {
                "prices": [
                    {
                        "ticker": row["ticker"],
                        "price": row["price"],
                        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    }
                    for row in results
                ],
                "failed_tickers": failed_tickers,
            },
        )
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
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)
    cache_ttl_seconds = _historical_cache_ttl_seconds(lookback_days)

    # Use cache if available and fresh
    if use_cache and _is_cache_fresh(cache_path, cache_ttl_seconds):
        cached = _read_price_cache(cache_path)
        if _cache_covers_start_date(cached, start_date) and _cache_covers_end_date(cached, end_date):
            return cached

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
        _write_price_cache(cache_path, df)

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

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days or settings.history_lookback_days)
    cache_ttl_seconds = _historical_cache_ttl_seconds(lookback_days)
    cached_frames: list[pd.DataFrame] = []
    missing_tickers: list[str] = []

    for ticker in cleaned:
        cache_path = _cache_path_for_ticker(ticker)
        cached = _read_price_cache(cache_path) if _is_cache_fresh(cache_path, cache_ttl_seconds) else pd.DataFrame()
        if _cache_covers_start_date(cached, start_date) and _cache_covers_end_date(cached, end_date):
            cached_frames.append(cached)
        else:
            missing_tickers.append(ticker)

    if not missing_tickers:
        return pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    try:
        with _yfinance_request_context():
            data = yf.download(
                tickers=missing_tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                progress=False,
                group_by="ticker",
            )
    except Exception:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    frames = list(cached_frames)

    # --- Handle single vs multi ticker structure ---
    if len(missing_tickers) == 1:
        ticker = missing_tickers[0]
        df = data.copy()

        if df.empty:
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

        normalized = _normalize_yfinance_history_frame(df, ticker)
        _write_price_cache(_cache_path_for_ticker(ticker), normalized)
        frames.append(normalized)

    else:
        for ticker in missing_tickers:
            try:
                df = data[ticker].copy()

                if df.empty:
                    continue

                normalized = _normalize_yfinance_history_frame(df, ticker)
                _write_price_cache(_cache_path_for_ticker(ticker), normalized)
                frames.append(normalized)

            except Exception:
                continue

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "ticker"]).drop_duplicates(subset=["date", "ticker"], keep="last")
    return combined.sort_values(["ticker", "date"]).reset_index(drop=True)


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
