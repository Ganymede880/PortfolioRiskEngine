"""
Institutional-style factor analytics for the CMCSIF portfolio tracker.

This module builds a custom live factor model using S&P 500 constituents and
Yahoo Finance data. These are custom internal factors, not official
Fama-French factors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None

from src.analytics.ledger import apply_cash_ledger_entries_to_positions, apply_trades_to_positions
from src.analytics.performance import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    prepare_flow_adjusted_history,
)
from src.data.price_fetcher import _yfinance_request_context, fetch_multiple_price_histories
from src.db.crud import (
    get_portfolio_history_cache_signature,
    load_all_portfolio_snapshots,
    load_cash_ledger,
    load_trade_receipts,
)
from src.db.session import session_scope
from src.config.settings import settings


COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
FACTOR_COLUMNS = ["MKT", "SMB", "MOM", "VAL"]
EXTERNAL_FLOW_ACTIVITY_TYPES = {"SECTOR_REBALANCE", "PORTFOLIO_LIQUIDATION"}
VALID_POSITION_SIDES = {"LONG", "SHORT", "CASH"}
YAHOO_TICKER_REPLACEMENTS = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}
FACTOR_MODEL_DISPLAY_LOOKBACK_DAYS = 365
FACTOR_MODEL_WARMUP_DAYS = 425
FACTOR_MODEL_PRICE_LOOKBACK_DAYS = FACTOR_MODEL_DISPLAY_LOOKBACK_DAYS + FACTOR_MODEL_WARMUP_DAYS
MAX_FACTOR_ANALYTICS_CACHE_ENTRIES = 4
FACTOR_REFERENCE_CACHE_TTL_SECONDS = 24 * 60 * 60
FACTOR_ARTIFACT_CACHE_TTL_SECONDS = 24 * 60 * 60
PORTFOLIO_HISTORY_CACHE_TTL_SECONDS = 12 * 60 * 60
_FACTOR_ANALYTICS_CACHE: dict[str, dict[str, Any]] = {}
_PORTFOLIO_HISTORY_CACHE: dict[str, pd.DataFrame] = {}


@dataclass(frozen=True)
class FactorConstructionConfig:
    weighting_scheme: str = "equal_weight"
    top_quantile: float = 0.10
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    sector_neutral: bool = False
    market_neutral: bool = False
    beta_neutral: bool = False
    regression_windows: tuple[int, ...] = (20, 60, 126, 252)
    nw_lags: int = 5
    cost_per_turnover: float = 0.001
    ewma_halflife: int = 60


@dataclass
class FactorAnalyticsPlatform:
    factor_returns: pd.DataFrame
    regression_summary: pd.DataFrame
    rolling_betas: pd.DataFrame
    portfolio_returns: pd.DataFrame
    latest_holdings_exposure: pd.DataFrame
    holdings_signals: dict[str, Any]
    portfolio_factor_betas: dict[str, Any]
    universe_fundamentals: pd.DataFrame
    factor_diagnostics: pd.DataFrame
    attribution_df: pd.DataFrame
    cumulative_attribution_df: pd.DataFrame
    risk_decomposition_df: pd.DataFrame
    risk_decomposition_reason: str
    holdings_factor_contribution_df: pd.DataFrame
    multi_horizon_exposures: pd.DataFrame
    correlation_matrix: pd.DataFrame
    rolling_correlation_panel: pd.DataFrame
    decile_returns_df: pd.DataFrame
    factor_backtest_summary: pd.DataFrame
    turnover_df: pd.DataFrame
    cost_estimate_df: pd.DataFrame
    factor_regime_df: pd.DataFrame
    residual_return_series: pd.DataFrame
    scenario_template_df: pd.DataFrame
    grid_stress_df: pd.DataFrame
    optimizer_recommendation_df: pd.DataFrame
    optimizer_projected_exposures_df: pd.DataFrame
    visualization_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _empty_analytics_payload(notes: list[str] | None = None) -> dict[str, Any]:
    empty = pd.DataFrame()
    analytics = {
        "factor_returns": empty,
        "regression_summary": empty,
        "rolling_betas": empty,
        "portfolio_returns": empty,
        "holdings_signals": {},
        "portfolio_factor_betas": {},
        "drawdowns": empty,
        "attribution": empty,
        "cumulative_attribution": empty,
        "risk_decomposition": empty,
        "risk_decomposition_reason": "",
        "holdings_contribution": empty,
        "factor_diagnostics": empty,
        "factor_correlations": empty,
        "rolling_factor_correlations": empty,
        "multi_horizon_exposures": empty,
        "decile_backtest": empty,
        "factor_backtest_summary": empty,
        "turnover_cost": empty,
        "scenario_template": empty,
        "optimizer_output": empty,
        "optimizer_projected_exposures": empty,
        "universe_fundamentals": empty,
        "latest_holdings_exposure": empty,
        "residual_return_series": empty,
        "metadata": {},
        "visualization_data": {},
        "notes": notes or [],
    }
    _validate_analytics_payload(analytics)
    return analytics


def _validate_analytics_payload(analytics: dict[str, Any]) -> None:
    required_keys = [
        "attribution",
        "risk_decomposition",
        "factor_diagnostics",
        "factor_correlations",
        "multi_horizon_exposures",
        "decile_backtest",
        "turnover_cost",
    ]
    missing = [key for key in required_keys if key not in analytics]
    if missing:
        raise ValueError(f"Missing analytics keys: {missing}")


def _cache_path_for_named_frame(name: str) -> Path:
    safe_name = str(name).strip().replace("/", "-").replace("\\", "-")
    return settings.cache_dir / f"{safe_name}.csv"


def _cache_path_for_named_blob(name: str) -> Path:
    safe_name = str(name).strip().replace("/", "-").replace("\\", "-").replace(":", "-")
    return settings.cache_dir / f"{safe_name}.pkl"


def _read_frame_cache(cache_path: Path) -> pd.DataFrame:
    try:
        if cache_path.exists():
            return pd.read_csv(cache_path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _write_frame_cache(cache_path: Path, df: pd.DataFrame) -> None:
    try:
        if df.empty:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    except Exception:
        pass


def _is_cache_fresh(cache_path: Path, max_age_seconds: int) -> bool:
    if not cache_path.exists():
        return False
    modified_at = pd.Timestamp(cache_path.stat().st_mtime, unit="s")
    age_seconds = (pd.Timestamp.now() - modified_at).total_seconds()
    return age_seconds < max_age_seconds


def _normalize_holdings_snapshot_for_cache(holdings_snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if holdings_snapshot_df.empty:
        return pd.DataFrame()
    working = holdings_snapshot_df.copy()
    for column in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column]):
            working[column] = pd.to_datetime(working[column], errors="coerce").astype("string")
        elif pd.api.types.is_numeric_dtype(working[column]):
            working[column] = pd.to_numeric(working[column], errors="coerce").round(10)
        else:
            working[column] = working[column].astype(str).str.strip()
    sort_columns = sorted(working.columns.tolist())
    return working.reindex(columns=sort_columns).sort_values(by=sort_columns, kind="stable").reset_index(drop=True)


def _build_factor_analytics_cache_key(
    holdings_snapshot_df: pd.DataFrame,
    config: FactorConstructionConfig,
) -> str:
    normalized = _normalize_holdings_snapshot_for_cache(holdings_snapshot_df)
    if normalized.empty:
        holdings_hash = "empty"
    else:
        hashed = pd.util.hash_pandas_object(normalized.fillna("<NA>"), index=False).values.tobytes()
        holdings_hash = hashlib.sha1(hashed).hexdigest()
    config_hash = hashlib.sha1(repr(config).encode("utf-8")).hexdigest()
    return f"{holdings_hash}:{config_hash}"


def _copy_cached_analytics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(payload)


def _store_factor_analytics_cache(cache_key: str, analytics: dict[str, Any]) -> dict[str, Any]:
    if len(_FACTOR_ANALYTICS_CACHE) >= MAX_FACTOR_ANALYTICS_CACHE_ENTRIES:
        oldest_key = next(iter(_FACTOR_ANALYTICS_CACHE))
        _FACTOR_ANALYTICS_CACHE.pop(oldest_key, None)
    _FACTOR_ANALYTICS_CACHE[cache_key] = _copy_cached_analytics_payload(analytics)
    return _copy_cached_analytics_payload(analytics)


def _persist_factor_analytics_cache(cache_key: str, analytics: dict[str, Any]) -> None:
    cache_path = _cache_path_for_named_blob(
        f"factor_analytics_{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()}"
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(analytics, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _load_persisted_factor_analytics_cache(cache_key: str) -> dict[str, Any] | None:
    cache_path = _cache_path_for_named_blob(
        f"factor_analytics_{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()}"
    )
    if not _is_cache_fresh(cache_path, FACTOR_ARTIFACT_CACHE_TTL_SECONDS):
        return None
    try:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        _validate_analytics_payload(payload)
    except Exception:
        return None
    return _copy_cached_analytics_payload(payload)


def _finalize_factor_analytics_cache(cache_key: str, analytics: dict[str, Any]) -> dict[str, Any]:
    stored = _store_factor_analytics_cache(cache_key, analytics)
    _persist_factor_analytics_cache(cache_key, stored)
    return stored


def _build_portfolio_history_cache_key(lookback_days: int, signature: dict[str, object]) -> str:
    signature_hash = hashlib.sha1(repr(sorted(signature.items())).encode("utf-8")).hexdigest()
    return f"portfolio_history:{lookback_days}:{signature_hash}"


def _persist_portfolio_history_cache(cache_key: str, history_df: pd.DataFrame) -> None:
    cache_path = _cache_path_for_named_blob(
        f"portfolio_history_{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()}"
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(history_df, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass


def _load_persisted_portfolio_history_cache(cache_key: str) -> pd.DataFrame | None:
    cache_path = _cache_path_for_named_blob(
        f"portfolio_history_{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()}"
    )
    if not _is_cache_fresh(cache_path, PORTFOLIO_HISTORY_CACHE_TTL_SECONDS):
        return None
    try:
        with cache_path.open("rb") as fh:
            payload = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(payload, pd.DataFrame):
        return None
    return payload.copy()


def _finalize_portfolio_history_cache(cache_key: str, history_df: pd.DataFrame) -> pd.DataFrame:
    cached_df = history_df.copy()
    _PORTFOLIO_HISTORY_CACHE[cache_key] = cached_df
    _persist_portfolio_history_cache(cache_key, cached_df)
    return cached_df.copy()


def _reason_or_default(df: pd.DataFrame, reason: str) -> str:
    return reason if df.empty else ""


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).strip().upper()
    if not value:
        return ""
    return YAHOO_TICKER_REPLACEMENTS.get(value, value)


def _winsorize_series(series: pd.Series, lower: float, upper: float) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    valid = clean.dropna()
    if valid.empty:
        return clean
    lower_bound = float(valid.quantile(lower))
    upper_bound = float(valid.quantile(upper))
    return clean.clip(lower=lower_bound, upper=upper_bound)


def _zscore_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    mean = clean.mean(skipna=True)
    std = clean.std(ddof=0, skipna=True)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(clean)), index=clean.index, dtype="float64")
    return (clean - mean) / std


def _safe_qcut_rank(series: pd.Series, bucket_count: int) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    valid = clean.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    rank_source = valid.rank(method="first")
    if valid.nunique() < bucket_count:
        deciles = pd.Series(
            np.ceil(rank_source / max(len(valid), 1) * bucket_count),
            index=valid.index,
            dtype="float64",
        )
    else:
        deciles = pd.qcut(rank_source, q=bucket_count, labels=False, duplicates="drop") + 1
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    result.loc[deciles.index] = deciles.astype(int)
    return result


def _with_constant(x: pd.DataFrame) -> pd.DataFrame:
    working = x.copy()
    if "const" not in working.columns:
        working.insert(0, "const", 1.0)
    return working


def _ols_fit(
    y: pd.Series,
    x: pd.DataFrame,
    hac_lags: int,
    min_obs: int | None = None,
) -> dict[str, Any]:
    y_clean = pd.to_numeric(y, errors="coerce")
    x_clean = x.apply(pd.to_numeric, errors="coerce")
    valid_mask = y_clean.notna() & x_clean.notna().all(axis=1)
    y_clean = y_clean.loc[valid_mask]
    x_clean = x_clean.loc[valid_mask]

    required_obs = max(min_obs if min_obs is not None else 15, x_clean.shape[1] + 2)
    if len(y_clean) < required_obs:
        return {"success": False}

    if sm is not None:
        model = sm.OLS(y_clean, x_clean, missing="drop").fit()
        robust = model.get_robustcov_results(cov_type="HAC", maxlags=hac_lags)
        params = pd.Series(robust.params, index=x_clean.columns)
        tvalues = pd.Series(robust.tvalues, index=x_clean.columns)
        pvalues = pd.Series(robust.pvalues, index=x_clean.columns)
        residuals = pd.Series(model.resid, index=y_clean.index)
        return {
            "success": True,
            "params": params,
            "tvalues": tvalues,
            "pvalues": pvalues,
            "rsquared": float(model.rsquared),
            "adj_rsquared": float(model.rsquared_adj),
            "residuals": residuals,
            "residual_volatility": float(residuals.std(ddof=1)),
            "nobs": float(model.nobs),
        }

    x_np = x_clean.to_numpy(dtype="float64")
    y_np = y_clean.to_numpy(dtype="float64")
    beta, _, rank, _ = np.linalg.lstsq(x_np, y_np, rcond=None)
    if rank < x_np.shape[1] or len(y_np) <= x_np.shape[1]:
        return {"success": False}
    fitted = x_np @ beta
    resid = y_np - fitted
    sse = float(np.sum(resid ** 2))
    sst = float(np.sum((y_np - y_np.mean()) ** 2))
    rsquared = float(1.0 - sse / sst) if sst != 0 else np.nan
    adj_rsquared = float(1.0 - (1.0 - rsquared) * ((len(y_np) - 1) / (len(y_np) - x_np.shape[1]))) if len(y_np) > x_np.shape[1] else np.nan
    sigma2 = sse / (len(y_np) - x_np.shape[1])
    try:
        xtx_inv = np.linalg.inv(x_np.T @ x_np)
    except np.linalg.LinAlgError:
        return {"success": False}
    std_err = np.sqrt(np.diag(sigma2 * xtx_inv))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / std_err
    if scipy_stats is not None:
        p_values = 2.0 * (1.0 - scipy_stats.t.cdf(np.abs(t_stats), df=len(y_np) - x_np.shape[1]))
    else:
        p_values = np.full_like(t_stats, np.nan, dtype="float64")
    residuals = pd.Series(resid, index=y_clean.index)
    return {
        "success": True,
        "params": pd.Series(beta, index=x_clean.columns),
        "tvalues": pd.Series(t_stats, index=x_clean.columns),
        "pvalues": pd.Series(p_values, index=x_clean.columns),
        "rsquared": rsquared,
        "adj_rsquared": adj_rsquared,
        "residuals": residuals,
        "residual_volatility": float(residuals.std(ddof=1)),
        "nobs": float(len(y_np)),
    }


@lru_cache(maxsize=2)
def _get_sp500_constituents_cached() -> pd.DataFrame:
    required = ["ticker", "security", "gics_sector", "gics_sub_industry"]
    cache_path = _cache_path_for_named_frame("sp500_constituents")

    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        rename_map = {
            "Symbol": "ticker",
            "symbol": "ticker",
            "Ticker": "ticker",
            "Security": "security",
            "Name": "security",
            "name": "security",
            "GICS Sector": "gics_sector",
            "Sector": "gics_sector",
            "sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "sub_industry": "gics_sub_industry",
        }
        working = working.rename(columns=rename_map)
        for col in required:
            if col not in working.columns:
                working[col] = pd.NA
        working["ticker"] = working["ticker"].astype(str).map(_normalize_ticker)
        working = working.loc[working["ticker"].ne("")].drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        return working[required]

    cached = _read_frame_cache(cache_path)
    if not cached.empty:
        return _finalize(cached)

    csv_sources = [
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
    ]
    for source in csv_sources:
        try:
            finalized = _finalize(pd.read_csv(source))
            if not finalized.empty:
                _write_frame_cache(cache_path, finalized)
                return finalized
        except Exception:
            continue

    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if tables:
            finalized = _finalize(tables[0])
            if not finalized.empty:
                _write_frame_cache(cache_path, finalized)
                return finalized
    except Exception:
        pass

    return pd.DataFrame(columns=required)


def get_sp500_constituents() -> pd.DataFrame:
    return _get_sp500_constituents_cached().copy()


def _fetch_yahoo_fundamentals_rows(tickers: tuple[str, ...]) -> pd.DataFrame:
    ticker_key = hashlib.sha1("|".join(tickers).encode("utf-8")).hexdigest()
    cache_path = _cache_path_for_named_frame(f"fundamentals_{ticker_key}")
    if _is_cache_fresh(cache_path, FACTOR_REFERENCE_CACHE_TTL_SECONDS):
        cached = _read_frame_cache(cache_path)
        if not cached.empty:
            return cached
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        market_cap = np.nan
        trailing_pe = np.nan
        forward_pe = np.nan
        last_price = np.nan
        trailing_eps = np.nan
        forward_eps = np.nan
        shares_outstanding = np.nan

        try:
            with _yfinance_request_context():
                info = yf.Ticker(ticker).info
            market_cap = pd.to_numeric(pd.Series([info.get("marketCap")]), errors="coerce").iloc[0]
            trailing_pe = pd.to_numeric(pd.Series([info.get("trailingPE")]), errors="coerce").iloc[0]
            forward_pe = pd.to_numeric(pd.Series([info.get("forwardPE")]), errors="coerce").iloc[0]
            last_price = pd.to_numeric(pd.Series([info.get("currentPrice", info.get("regularMarketPrice"))]), errors="coerce").iloc[0]
            trailing_eps = pd.to_numeric(pd.Series([info.get("trailingEps")]), errors="coerce").iloc[0]
            forward_eps = pd.to_numeric(pd.Series([info.get("forwardEps")]), errors="coerce").iloc[0]
            shares_outstanding = pd.to_numeric(pd.Series([info.get("sharesOutstanding")]), errors="coerce").iloc[0]
        except Exception:
            pass

        if (pd.isna(market_cap) or market_cap <= 0) and pd.notna(last_price) and pd.notna(shares_outstanding):
            market_cap = float(last_price * shares_outstanding)

        selected_pe = trailing_pe if pd.notna(trailing_pe) and trailing_pe > 0 else forward_pe
        if (pd.isna(selected_pe) or selected_pe <= 0) and pd.notna(last_price):
            if pd.notna(trailing_eps) and trailing_eps > 0:
                selected_pe = float(last_price / trailing_eps)
            elif pd.notna(forward_eps) and forward_eps > 0:
                selected_pe = float(last_price / forward_eps)

        earnings_yield = (1.0 / selected_pe) if pd.notna(selected_pe) and selected_pe > 0 else np.nan
        rows.append(
            {
                "ticker": ticker,
                "market_cap": market_cap,
                "trailing_pe": trailing_pe,
                "forward_pe": forward_pe,
                "selected_pe": selected_pe,
                "earnings_yield": earnings_yield,
            }
        )
    result = pd.DataFrame(rows)
    usable = result[["market_cap", "selected_pe", "earnings_yield"]].notna().any(axis=1).sum() if not result.empty else 0
    if usable > 0:
        _write_frame_cache(cache_path, result)
        return result
    cached = _read_frame_cache(cache_path)
    return cached if not cached.empty else result


@lru_cache(maxsize=8)
def _fetch_fundamentals_cached(tickers: tuple[str, ...]) -> pd.DataFrame:
    return _fetch_yahoo_fundamentals_rows(tickers)


def fetch_live_security_fundamentals(tickers: list[str]) -> pd.DataFrame:
    unique = tuple(sorted({_normalize_ticker(ticker) for ticker in tickers if str(ticker).strip()}))
    if not unique:
        return pd.DataFrame(columns=["ticker", "market_cap", "trailing_pe", "forward_pe", "selected_pe", "earnings_yield"])
    return _fetch_fundamentals_cached(unique).copy()


def _build_price_matrix(price_history_df: pd.DataFrame) -> pd.DataFrame:
    if price_history_df.empty:
        return pd.DataFrame()
    working = price_history_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working["ticker"] = working["ticker"].astype(str).map(_normalize_ticker)
    working["adj_close"] = pd.to_numeric(working.get("adj_close"), errors="coerce")
    working["close"] = pd.to_numeric(working.get("close"), errors="coerce")
    working["px"] = working["adj_close"]
    missing_mask = working["px"].isna()
    working.loc[missing_mask, "px"] = working.loc[missing_mask, "close"]
    working = working.dropna(subset=["date", "ticker", "px"]).copy()
    if working.empty:
        return pd.DataFrame()
    return working.pivot_table(index="date", columns="ticker", values="px", aggfunc="last").sort_index().ffill()


def _compute_position_value(positions_df: pd.DataFrame, price_map: dict[str, float]) -> float:
    if positions_df.empty:
        return 0.0

    total = 0.0
    for _, row in positions_df.iterrows():
        ticker = str(row.get(COL_TICKER, "")).strip().upper()
        side = str(row.get(COL_POSITION_SIDE, "")).strip().upper()
        team = str(row.get(COL_TEAM, "")).strip().upper()
        shares = float(pd.to_numeric(pd.Series([row.get(COL_SHARES)]), errors="coerce").fillna(0.0).iloc[0])
        if ticker in {"CASH", "EUR", "GBP", "NOGXX"} or side == "CASH" or team == "CASH":
            total += shares
        else:
            px = price_map.get(ticker, 0.0)
            total += (-shares * px) if side == "SHORT" else (shares * px)
    return float(total)


def _transition_positions_for_day(
    active_positions_df: pd.DataFrame | None,
    snapshot_for_day: pd.DataFrame | None,
    trades_today: pd.DataFrame | None,
    cash_today: pd.DataFrame | None,
    price_map: dict[str, float],
) -> tuple[pd.DataFrame | None, float, float]:
    net_external_flow = 0.0
    reconciliation_pnl = 0.0

    expected_positions_df = active_positions_df.copy() if active_positions_df is not None else None
    if expected_positions_df is not None and trades_today is not None and not trades_today.empty:
        expected_positions_df, _ = apply_trades_to_positions(expected_positions_df, trades_today)

    if cash_today is not None and not cash_today.empty:
        net_external_flow = float(pd.to_numeric(cash_today["amount"], errors="coerce").fillna(0.0).sum())
        if expected_positions_df is not None:
            expected_positions_df = apply_cash_ledger_entries_to_positions(expected_positions_df, cash_today)

    if snapshot_for_day is not None:
        if expected_positions_df is not None:
            reconciliation_pnl = float(
                _compute_position_value(snapshot_for_day, price_map)
                - _compute_position_value(expected_positions_df, price_map)
            )
        return snapshot_for_day.copy(), net_external_flow, reconciliation_pnl

    return expected_positions_df, net_external_flow, reconciliation_pnl


def _validate_snapshot_history_integrity(snapshots_df: pd.DataFrame) -> None:
    """
    Fail fast when stored snapshots contain invalid position-side values.
    """
    if snapshots_df.empty or COL_POSITION_SIDE not in snapshots_df.columns:
        return

    sides = snapshots_df[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    invalid_mask = ~sides.isin(VALID_POSITION_SIDES)
    if not invalid_mask.any():
        return

    invalid_rows = snapshots_df.loc[invalid_mask].copy()
    preview_cols = [
        col
        for col in ["snapshot_date", "team", "ticker", "position_side", "source_file"]
        if col in invalid_rows.columns
    ]
    preview_records = invalid_rows[preview_cols].head(5).to_dict("records")
    affected_dates = sorted(
        pd.to_datetime(invalid_rows["snapshot_date"], errors="coerce")
        .dropna()
        .dt.strftime("%Y-%m-%d")
        .unique()
        .tolist()
    )
    raise ValueError(
        "Portfolio snapshots contain invalid position_side values, so holdings history cannot be reconstructed. "
        f"Affected snapshot dates: {affected_dates[:5]}. "
        f"Example rows: {preview_records}. "
        "This usually means a snapshot upload mapped the Position column incorrectly."
    )


def _compute_daily_stock_beta(stock_returns: pd.DataFrame, market_returns: pd.Series, lookback: int = 252) -> pd.DataFrame:
    if stock_returns.empty or market_returns.empty:
        return pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype="float64")
    aligned_market = pd.to_numeric(market_returns, errors="coerce").reindex(stock_returns.index)
    market_var = aligned_market.rolling(lookback, min_periods=max(60, lookback // 3)).var()
    beta_matrix = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype="float64")
    for ticker in stock_returns.columns:
        cov = stock_returns[ticker].rolling(lookback, min_periods=max(60, lookback // 3)).cov(aligned_market)
        beta_matrix[ticker] = cov / market_var
    return beta_matrix


def _build_signal_frame_for_date(
    as_of_date: pd.Timestamp,
    universe_df: pd.DataFrame,
    price_matrix: pd.DataFrame,
    beta_matrix: pd.DataFrame,
    config: FactorConstructionConfig,
) -> pd.DataFrame:
    base = universe_df.copy().set_index("ticker")
    if as_of_date not in price_matrix.index:
        prior_dates = price_matrix.index[price_matrix.index <= as_of_date]
        if len(prior_dates) == 0:
            return pd.DataFrame()
        as_of_date = prior_dates[-1]

    momentum_signal = (price_matrix.shift(21).loc[as_of_date] / price_matrix.shift(252).loc[as_of_date]) - 1.0
    signal_df = base.join(momentum_signal.rename("momentum_signal"), how="left")
    signal_df["size_signal_raw"] = -pd.to_numeric(signal_df["market_cap"], errors="coerce")
    signal_df["value_signal_raw"] = pd.to_numeric(signal_df["earnings_yield"], errors="coerce")
    signal_df["momentum_signal_raw"] = pd.to_numeric(signal_df["momentum_signal"], errors="coerce")
    signal_df["beta_to_market"] = pd.to_numeric(beta_matrix.reindex(index=[as_of_date]).iloc[0], errors="coerce") if as_of_date in beta_matrix.index else np.nan

    for factor_name, raw_col in {
        "SMB": "size_signal_raw",
        "VAL": "value_signal_raw",
        "MOM": "momentum_signal_raw",
    }.items():
        wins_col = f"{factor_name.lower()}_signal"
        z_col = f"{factor_name.lower()}_zscore"
        signal_df[wins_col] = _winsorize_series(signal_df[raw_col], config.winsor_lower, config.winsor_upper)
        signal_df[z_col] = _zscore_series(signal_df[wins_col])

    signal_df["date"] = as_of_date
    return signal_df.reset_index()


def _build_group_neutral_weights(
    group_df: pd.DataFrame,
    signal_column: str,
    config: FactorConstructionConfig,
) -> pd.DataFrame:
    valid = group_df.copy()
    valid[signal_column] = pd.to_numeric(valid[signal_column], errors="coerce")
    valid = valid.dropna(subset=[signal_column]).copy()
    if valid.empty:
        return pd.DataFrame(columns=["ticker", "weight", "leg", "signal", "zscore", "beta_to_market"])

    bucket_size = max(int(np.floor(len(valid) * config.top_quantile)), 1)
    ranked = valid.sort_values(signal_column)
    short_leg = ranked.head(bucket_size).copy()
    long_leg = ranked.tail(bucket_size).copy()
    if short_leg.empty or long_leg.empty:
        return pd.DataFrame(columns=["ticker", "weight", "leg", "signal", "zscore", "beta_to_market"])

    if config.weighting_scheme == "market_cap_weight":
        long_base = pd.to_numeric(long_leg["market_cap"], errors="coerce").fillna(0.0)
        short_base = pd.to_numeric(short_leg["market_cap"], errors="coerce").fillna(0.0)
        long_weights = long_base / long_base.sum() if long_base.sum() != 0 else pd.Series(1.0 / len(long_leg), index=long_leg.index)
        short_weights = short_base / short_base.sum() if short_base.sum() != 0 else pd.Series(1.0 / len(short_leg), index=short_leg.index)
    else:
        long_weights = pd.Series(1.0 / len(long_leg), index=long_leg.index)
        short_weights = pd.Series(1.0 / len(short_leg), index=short_leg.index)

    long_leg = long_leg.assign(weight=0.5 * long_weights, leg="long")
    short_leg = short_leg.assign(weight=-0.5 * short_weights, leg="short")
    combined = pd.concat([long_leg, short_leg], ignore_index=True)

    if config.market_neutral or config.beta_neutral:
        combined["beta_to_market"] = pd.to_numeric(combined["beta_to_market"], errors="coerce").fillna(1.0)
        long_beta = abs(float((combined.loc[combined["weight"] > 0, "weight"] * combined.loc[combined["weight"] > 0, "beta_to_market"]).sum()))
        short_beta = abs(float((combined.loc[combined["weight"] < 0, "weight"] * combined.loc[combined["weight"] < 0, "beta_to_market"]).sum()))
        if long_beta > 0 and short_beta > 0:
            combined.loc[combined["weight"] > 0, "weight"] *= 0.5 / long_beta
            combined.loc[combined["weight"] < 0, "weight"] *= 0.5 / short_beta
            gross = combined["weight"].abs().sum()
            if gross != 0:
                combined["weight"] = combined["weight"] / gross

    return combined[["ticker", "weight", "leg", "beta_to_market"]]


def _build_factor_weight_book(
    signal_panel_df: pd.DataFrame,
    config: FactorConstructionConfig,
) -> pd.DataFrame:
    if signal_panel_df.empty:
        return pd.DataFrame(columns=["date", "factor", "ticker", "weight", "leg", "signal", "zscore", "sector", "beta_to_market"])

    rows: list[pd.DataFrame] = []
    factor_signal_columns = {
        "SMB": ("smb_signal", "smb_zscore"),
        "MOM": ("mom_signal", "mom_zscore"),
        "VAL": ("val_signal", "val_zscore"),
    }

    for date, date_df in signal_panel_df.groupby("date", sort=True):
        for factor_name, (signal_col, z_col) in factor_signal_columns.items():
            if config.sector_neutral:
                group_frames = []
                for sector, sector_df in date_df.groupby("gics_sector", dropna=False):
                    weights_df = _build_group_neutral_weights(sector_df, signal_col, config)
                    if weights_df.empty:
                        continue
                    weights_df["sector"] = sector
                    group_frames.append(weights_df)
                if not group_frames:
                    continue
                weights_df = pd.concat(group_frames, ignore_index=True)
                weights_df["weight"] = weights_df["weight"] / weights_df["weight"].abs().sum()
            else:
                weights_df = _build_group_neutral_weights(date_df, signal_col, config)
                if weights_df.empty:
                    continue
                weights_df["sector"] = pd.NA

            enriched = weights_df.merge(
                date_df[["ticker", signal_col, z_col, "gics_sector"]].rename(columns={signal_col: "signal", z_col: "zscore"}),
                on="ticker",
                how="left",
            )
            enriched["sector"] = enriched["sector"].fillna(enriched["gics_sector"])
            enriched["date"] = date
            enriched["factor"] = factor_name
            rows.append(enriched[["date", "factor", "ticker", "weight", "leg", "signal", "zscore", "sector", "beta_to_market"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "factor", "ticker", "weight", "leg", "signal", "zscore", "sector", "beta_to_market"])


def _build_decile_weight_book(
    signal_panel_df: pd.DataFrame,
    config: FactorConstructionConfig,
) -> pd.DataFrame:
    if signal_panel_df.empty:
        return pd.DataFrame(columns=["date", "factor", "ticker", "decile", "weight"])

    rows: list[pd.DataFrame] = []
    factor_signal_columns = {
        "SMB": "smb_signal",
        "MOM": "mom_signal",
        "VAL": "val_signal",
    }

    for date, date_df in signal_panel_df.groupby("date", sort=True):
        for factor_name, signal_col in factor_signal_columns.items():
            working = date_df[["ticker", signal_col, "market_cap"]].copy()
            working[signal_col] = pd.to_numeric(working[signal_col], errors="coerce")
            working = working.dropna(subset=[signal_col]).copy()
            if working.empty:
                continue
            working["decile"] = _safe_qcut_rank(working[signal_col], 10)
            working = working.dropna(subset=["decile"]).copy()
            if working.empty:
                continue
            if config.weighting_scheme == "market_cap_weight":
                working["base_weight"] = pd.to_numeric(working["market_cap"], errors="coerce").fillna(0.0)
                working["weight"] = working.groupby("decile")["base_weight"].transform(lambda s: s / s.sum() if s.sum() != 0 else 1.0 / len(s))
            else:
                working["weight"] = working.groupby("decile")["ticker"].transform(lambda s: 1.0 / len(s))
            working["date"] = date
            working["factor"] = factor_name
            rows.append(working[["date", "factor", "ticker", "decile", "weight"]])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "factor", "ticker", "decile", "weight"])


def _compute_period_returns(
    weight_book_df: pd.DataFrame,
    decile_weight_book_df: pd.DataFrame,
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []

    if weight_book_df.empty or stock_returns.empty:
        return pd.DataFrame(columns=["date"] + FACTOR_COLUMNS), pd.DataFrame()

    rebalance_dates = sorted(pd.to_datetime(weight_book_df["date"], errors="coerce").dropna().unique().tolist())
    for idx, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else stock_returns.index.max()
        holding_dates = stock_returns.index[(stock_returns.index > rebalance_date) & (stock_returns.index <= next_rebalance)]
        if len(holding_dates) == 0:
            continue

        date_weights = weight_book_df.loc[pd.to_datetime(weight_book_df["date"], errors="coerce").eq(rebalance_date)].copy()
        for factor_name in ["SMB", "MOM", "VAL"]:
            factor_weights = date_weights.loc[date_weights["factor"] == factor_name, ["ticker", "weight"]].copy()
            if factor_weights.empty:
                continue
            weights = factor_weights.set_index("ticker")["weight"]
            aligned_returns = stock_returns.reindex(index=holding_dates, columns=weights.index).fillna(0.0)
            factor_series = aligned_returns.mul(weights, axis=1).sum(axis=1)
            for date, value in factor_series.items():
                factor_rows.append({"date": date, "factor": factor_name, "return": float(value)})

        if not decile_weight_book_df.empty:
            date_deciles = decile_weight_book_df.loc[pd.to_datetime(decile_weight_book_df["date"], errors="coerce").eq(rebalance_date)].copy()
            for factor_name in ["SMB", "MOM", "VAL"]:
                factor_deciles = date_deciles.loc[date_deciles["factor"] == factor_name].copy()
                if factor_deciles.empty:
                    continue
                for decile, decile_df in factor_deciles.groupby("decile"):
                    weights = decile_df.set_index("ticker")["weight"]
                    aligned_returns = stock_returns.reindex(index=holding_dates, columns=weights.index).fillna(0.0)
                    decile_series = aligned_returns.mul(weights, axis=1).sum(axis=1)
                    for date, value in decile_series.items():
                        decile_rows.append({"date": date, "factor": factor_name, "decile": int(decile), "return": float(value)})

    factor_returns_df = pd.DataFrame(factor_rows)
    if factor_returns_df.empty:
        factor_wide = pd.DataFrame(columns=["date"] + FACTOR_COLUMNS)
    else:
        factor_wide = factor_returns_df.pivot_table(index="date", columns="factor", values="return", aggfunc="last").reset_index()

    market_df = pd.DataFrame({"date": market_returns.index, "MKT": pd.to_numeric(market_returns, errors="coerce").values})
    factor_wide["date"] = pd.to_datetime(factor_wide["date"], errors="coerce")
    output = market_df.merge(factor_wide, on="date", how="left")
    for factor in ["SMB", "MOM", "VAL"]:
        if factor not in output.columns:
            output[factor] = pd.NA

    decile_returns_df = pd.DataFrame(decile_rows)
    return output.sort_values("date").reset_index(drop=True), decile_returns_df.sort_values(["date", "factor", "decile"]).reset_index(drop=True)


def build_portfolio_return_history(
    lookback_days: int = 450,
    base_price_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    with session_scope() as session:
        cache_signature = get_portfolio_history_cache_signature(session)
        cache_key = _build_portfolio_history_cache_key(lookback_days, cache_signature)
        in_memory_cached = _PORTFOLIO_HISTORY_CACHE.get(cache_key)
        if in_memory_cached is not None:
            return in_memory_cached.copy()

        persisted_cached = _load_persisted_portfolio_history_cache(cache_key)
        if persisted_cached is not None:
            _PORTFOLIO_HISTORY_CACHE[cache_key] = persisted_cached.copy()
            return persisted_cached

        snapshots_df = load_all_portfolio_snapshots(session)
        trades_df = load_trade_receipts(session)
        cash_df = load_cash_ledger(session)

    if snapshots_df.empty:
        return _finalize_portfolio_history_cache(
            cache_key,
            pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "reconciliation_pnl", "portfolio_return", "portfolio_pnl"]),
        )

    snapshots = snapshots_df.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], errors="coerce")
    snapshots[COL_TICKER] = snapshots[COL_TICKER].astype(str).map(_normalize_ticker)
    snapshots[COL_TEAM] = snapshots[COL_TEAM].astype(str).str.strip()
    snapshots[COL_POSITION_SIDE] = snapshots[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    snapshots[COL_SHARES] = pd.to_numeric(snapshots[COL_SHARES], errors="coerce")
    _validate_snapshot_history_integrity(snapshots)
    snapshots = snapshots.dropna(subset=["snapshot_date", COL_TICKER, COL_SHARES]).copy()
    if snapshots.empty:
        return _finalize_portfolio_history_cache(
            cache_key,
            pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "reconciliation_pnl", "portfolio_return", "portfolio_pnl"]),
        )

    today = pd.Timestamp.today().normalize()
    oldest_snapshot_date = snapshots["snapshot_date"].min().normalize()
    start_date = max(oldest_snapshot_date, today - pd.Timedelta(days=lookback_days))
    business_dates = pd.bdate_range(start=start_date, end=today)
    if len(business_dates) == 0:
        return _finalize_portfolio_history_cache(
            cache_key,
            pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "reconciliation_pnl", "portfolio_return", "portfolio_pnl"]),
        )

    trades = trades_df.copy()
    if not trades.empty:
        trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce")
        trades[COL_TICKER] = trades[COL_TICKER].astype(str).map(_normalize_ticker)
        trades = trades.dropna(subset=["trade_date", COL_TICKER]).copy()

    external_cash = cash_df.copy()
    if not external_cash.empty:
        external_cash["activity_date"] = pd.to_datetime(external_cash["activity_date"], errors="coerce")
        external_cash["activity_type"] = external_cash["activity_type"].astype(str).str.strip().str.upper()
        external_cash["amount"] = pd.to_numeric(external_cash["amount"], errors="coerce").fillna(0.0)
        external_cash = external_cash.loc[external_cash["activity_type"].isin(EXTERNAL_FLOW_ACTIVITY_TYPES)].dropna(subset=["activity_date"]).copy()
        if not external_cash.empty:
            cash_dates = external_cash["activity_date"].dt.normalize()
            aligned_idx = business_dates.searchsorted(cash_dates)
            valid_mask = aligned_idx < len(business_dates)
            external_cash = external_cash.loc[valid_mask].copy()
            if not external_cash.empty:
                external_cash["flow_date"] = business_dates.take(aligned_idx[valid_mask]).normalize()

    price_tickers = sorted(
        {
            ticker
            for ticker in pd.concat([snapshots[COL_TICKER], trades[COL_TICKER] if not trades.empty else pd.Series(dtype="object")]).dropna().tolist()
            if ticker not in {"CASH", "EUR", "GBP", "NOGXX"}
        }
    )
    reused_price_history_df = pd.DataFrame()
    missing_price_tickers = price_tickers
    if base_price_history_df is not None and not base_price_history_df.empty:
        reused_price_history_df = base_price_history_df.copy()
        reused_price_history_df["ticker"] = reused_price_history_df["ticker"].astype(str).map(_normalize_ticker)
        available_tickers = set(reused_price_history_df["ticker"].dropna().astype(str).tolist())
        missing_price_tickers = [ticker for ticker in price_tickers if ticker not in available_tickers]

    fetched_price_history_df = (
        fetch_multiple_price_histories(
            missing_price_tickers,
            lookback_days=(today - start_date).days + 30,
        )
        if missing_price_tickers else pd.DataFrame()
    )
    price_history_df = pd.concat(
        [reused_price_history_df, fetched_price_history_df],
        ignore_index=True,
    ) if not reused_price_history_df.empty or not fetched_price_history_df.empty else pd.DataFrame()
    price_matrix = _build_price_matrix(price_history_df).reindex(business_dates).ffill()

    snapshots_by_date = {dt.normalize(): grp.copy() for dt, grp in snapshots.groupby(snapshots["snapshot_date"].dt.normalize())}
    trades_by_date = {dt.normalize(): grp.copy() for dt, grp in trades.groupby(trades["trade_date"].dt.normalize())} if not trades.empty else {}
    cash_by_date = {dt.normalize(): grp.copy() for dt, grp in external_cash.groupby("flow_date")} if not external_cash.empty else {}

    active_positions_df: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []

    for dt in business_dates:
        price_map = {}
        if dt in price_matrix.index:
            row = price_matrix.loc[dt]
            price_map = {str(col).strip().upper(): float(row[col]) for col in price_matrix.columns if pd.notna(row[col])}

        snapshot_for_day = snapshots_by_date.get(dt.normalize())
        if active_positions_df is None:
            eligible = [d for d in snapshots_by_date.keys() if d <= dt]
            if not eligible:
                continue
            if snapshot_for_day is None:
                active_positions_df = snapshots_by_date[max(eligible)].copy()

        active_positions_df, net_external_flow, reconciliation_pnl = _transition_positions_for_day(
            active_positions_df=active_positions_df,
            snapshot_for_day=snapshot_for_day,
            trades_today=trades_by_date.get(dt.normalize()),
            cash_today=cash_by_date.get(dt.normalize()),
            price_map=price_map,
        )
        if active_positions_df is None:
            continue

        portfolio_aum = _compute_position_value(active_positions_df, price_map)
        rows.append(
            {
                "date": dt,
                "portfolio_aum": float(portfolio_aum),
                "net_external_flow": net_external_flow,
                "reconciliation_pnl": reconciliation_pnl,
            }
        )

    history_df = pd.DataFrame(rows)
    if history_df.empty:
        return _finalize_portfolio_history_cache(
            cache_key,
            pd.DataFrame(
                columns=[
                    "date",
                    "portfolio_aum",
                    "net_external_flow",
                    "reconciliation_pnl",
                    "portfolio_return",
                    "portfolio_pnl",
                ]
            ),
        )

    prepared = prepare_flow_adjusted_history(history_df, value_column="portfolio_aum", flow_column="net_external_flow")
    prepared = prepared.rename(columns={"performance_return": "portfolio_return", "performance_pnl": "portfolio_pnl"})
    if "reconciliation_pnl" not in prepared.columns:
        prepared["reconciliation_pnl"] = 0.0
    return _finalize_portfolio_history_cache(
        cache_key,
        prepared[["date", "portfolio_aum", "net_external_flow", "reconciliation_pnl", "portfolio_return", "portfolio_pnl"]],
    )


def _build_regression_suite(
    factor_returns_df: pd.DataFrame,
    portfolio_returns_df: pd.DataFrame,
    config: FactorConstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = portfolio_returns_df.merge(factor_returns_df, on="date", how="inner")
    merged["portfolio_return"] = pd.to_numeric(merged["portfolio_return"], errors="coerce")
    for factor in FACTOR_COLUMNS:
        merged[factor] = pd.to_numeric(merged[factor], errors="coerce")
    merged = merged.dropna(subset=["portfolio_return"] + FACTOR_COLUMNS).sort_values("date").reset_index(drop=True)
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    x = _with_constant(merged[FACTOR_COLUMNS])
    fit = _ols_fit(merged["portfolio_return"], x, hac_lags=config.nw_lags)
    if not fit.get("success"):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), merged, pd.DataFrame()

    portfolio_factor_beta_summary = pd.DataFrame(
        [
            {
                "alpha": fit["params"].get("const", np.nan),
                "beta_mkt": fit["params"].get("MKT", np.nan),
                "beta_smb": fit["params"].get("SMB", np.nan),
                "beta_mom": fit["params"].get("MOM", np.nan),
                "beta_val": fit["params"].get("VAL", np.nan),
                "r_squared": fit["rsquared"],
                "adj_r_squared": fit["adj_rsquared"],
                "residual_vol": fit["residual_volatility"],
                "obs_count": fit["nobs"],
            }
        ]
    )

    regression_summary = pd.DataFrame(
        [
            {"term": "Alpha", "coefficient": fit["params"].get("const", np.nan), "t_stat": fit["tvalues"].get("const", np.nan), "p_value": fit["pvalues"].get("const", np.nan)},
            {"term": "Portfolio Beta to MKT", "coefficient": fit["params"].get("MKT", np.nan), "t_stat": fit["tvalues"].get("MKT", np.nan), "p_value": fit["pvalues"].get("MKT", np.nan)},
            {"term": "Portfolio Beta to SMB", "coefficient": fit["params"].get("SMB", np.nan), "t_stat": fit["tvalues"].get("SMB", np.nan), "p_value": fit["pvalues"].get("SMB", np.nan)},
            {"term": "Portfolio Beta to MOM", "coefficient": fit["params"].get("MOM", np.nan), "t_stat": fit["tvalues"].get("MOM", np.nan), "p_value": fit["pvalues"].get("MOM", np.nan)},
            {"term": "Portfolio Beta to VAL", "coefficient": fit["params"].get("VAL", np.nan), "t_stat": fit["tvalues"].get("VAL", np.nan), "p_value": fit["pvalues"].get("VAL", np.nan)},
            {"term": "R-Squared", "coefficient": fit["rsquared"], "t_stat": np.nan, "p_value": np.nan},
            {"term": "Adj R-Squared", "coefficient": fit["adj_rsquared"], "t_stat": np.nan, "p_value": np.nan},
            {"term": "Residual Volatility", "coefficient": fit["residual_volatility"], "t_stat": np.nan, "p_value": np.nan},
            {"term": "Observations", "coefficient": fit["nobs"], "t_stat": np.nan, "p_value": np.nan},
        ]
    )

    residual_return_series = pd.DataFrame(
        {
            "date": merged.loc[fit["residuals"].index, "date"].values,
            "residual_return": fit["residuals"].values,
        }
    )

    min_regression_obs = max(15, len(FACTOR_COLUMNS) + 2)
    min_display_regression_obs = len(FACTOR_COLUMNS) + 2
    rolling_frames: list[pd.DataFrame] = []
    multi_horizon_frames: list[pd.DataFrame] = []
    for window in config.regression_windows:
        rows: list[dict[str, Any]] = []
        for end_idx in range(window - 1, len(merged)):
            window_df = merged.iloc[end_idx - window + 1 : end_idx + 1].copy()
            fit_window = _ols_fit(window_df["portfolio_return"], _with_constant(window_df[FACTOR_COLUMNS]), hac_lags=config.nw_lags)
            if not fit_window.get("success"):
                continue
            rows.append(
                {
                    "date": window_df["date"].iloc[-1],
                    "window": window,
                    "alpha": fit_window["params"].get("const", np.nan),
                    "beta_mkt": fit_window["params"].get("MKT", np.nan),
                    "beta_smb": fit_window["params"].get("SMB", np.nan),
                    "beta_mom": fit_window["params"].get("MOM", np.nan),
                    "beta_val": fit_window["params"].get("VAL", np.nan),
                }
            )
        if rows:
            frame = pd.DataFrame(rows)
            rolling_frames.append(frame)
            temp = frame.set_index("date")[["beta_mkt", "beta_smb", "beta_mom", "beta_val"]].copy()
            temp.columns = pd.MultiIndex.from_tuples(
                [("MKT", window), ("SMB", window), ("MOM", window), ("VAL", window)],
                names=["factor", "horizon"],
            )
            multi_horizon_frames.append(temp)

    rolling_betas = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame(columns=["date", "window", "alpha", "beta_mkt", "beta_smb", "beta_mom", "beta_val"])
    multi_horizon_exposures = pd.concat(multi_horizon_frames, axis=1).sort_index() if multi_horizon_frames else pd.DataFrame()

    # Build the displayed factor-loading series for the last year, but calibrate
    # each point using prior return history as well so the series does not begin
    # late inside the display window.
    factor_loading_rows: list[dict[str, Any]] = []
    loadings_start_date = pd.to_datetime(merged["date"], errors="coerce").max() - pd.Timedelta(days=365)
    merged_dates = pd.to_datetime(merged["date"], errors="coerce")
    display_end_indices = merged.index[merged_dates >= loadings_start_date].tolist()
    if display_end_indices:
        trailing_window = 252
        for end_idx in display_end_indices:
            start_idx = max(0, end_idx - trailing_window + 1)
            window_df = merged.iloc[start_idx : end_idx + 1].copy()
            fit_window = _ols_fit(
                window_df["portfolio_return"],
                _with_constant(window_df[FACTOR_COLUMNS]),
                hac_lags=config.nw_lags,
                min_obs=min_display_regression_obs,
            )
            if not fit_window.get("success"):
                continue
            factor_loading_rows.append(
                {
                    "date": window_df["date"].iloc[-1],
                    "window": 252,
                    "estimation_method": "trailing_252d_with_prestart_history",
                    "alpha": fit_window["params"].get("const", np.nan),
                    "beta_mkt": fit_window["params"].get("MKT", np.nan),
                    "beta_smb": fit_window["params"].get("SMB", np.nan),
                    "beta_mom": fit_window["params"].get("MOM", np.nan),
                    "beta_val": fit_window["params"].get("VAL", np.nan),
                    "obs_count": fit_window["nobs"],
                }
            )
    factor_loadings = pd.DataFrame(factor_loading_rows)
    if factor_loadings.empty and not rolling_betas.empty:
        factor_loadings = (
            rolling_betas.loc[rolling_betas["window"] == 252]
            .copy()
            .assign(estimation_method="trailing_252d")
            .reset_index(drop=True)
        )
    if not factor_loadings.empty:
        factor_loadings = factor_loadings.drop(columns=["window"], errors="ignore").reset_index(drop=True)

    # Exponentially weighted exposure estimate using recursive weights
    if not merged.empty:
        ewm_rows: list[dict[str, Any]] = []
        halflife = max(config.ewma_halflife, 2)
        alpha = 1.0 - np.exp(np.log(0.5) / halflife)
        for end_idx in range(20, len(merged)):
            window_df = merged.iloc[: end_idx + 1].copy()
            weights = np.power(1.0 - alpha, np.arange(len(window_df) - 1, -1, -1))
            weights = weights / weights.sum()
            x_mat = _with_constant(window_df[FACTOR_COLUMNS]).to_numpy(dtype="float64")
            y_vec = window_df["portfolio_return"].to_numpy(dtype="float64")
            w_sqrt = np.sqrt(weights)
            x_w = x_mat * w_sqrt[:, None]
            y_w = y_vec * w_sqrt
            beta, _, rank, _ = np.linalg.lstsq(x_w, y_w, rcond=None)
            if rank < x_w.shape[1]:
                continue
            ewm_rows.append(
                {
                    "date": window_df["date"].iloc[-1],
                    "window": f"ewm_{halflife}",
                    "alpha": beta[0],
                    "beta_mkt": beta[1],
                    "beta_smb": beta[2],
                    "beta_mom": beta[3],
                    "beta_val": beta[4],
                }
            )
        if ewm_rows:
            rolling_betas = pd.concat([rolling_betas, pd.DataFrame(ewm_rows)], ignore_index=True)

    return regression_summary, factor_loadings, multi_horizon_exposures, residual_return_series, merged, portfolio_factor_beta_summary


def _build_attribution_tables(
    regression_input_df: pd.DataFrame,
    rolling_betas_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if regression_input_df.empty or rolling_betas_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    betas = rolling_betas_df.copy()
    betas["date"] = pd.to_datetime(betas["date"], errors="coerce")
    betas = betas.sort_values("date").dropna(subset=["date"])

    attrib = regression_input_df.merge(betas, on="date", how="left")
    attrib = attrib.sort_values("date").ffill()
    if attrib.empty:
        return pd.DataFrame(), pd.DataFrame()

    attrib["contribution_mkt"] = pd.to_numeric(attrib["beta_mkt"], errors="coerce") * pd.to_numeric(attrib["MKT"], errors="coerce")
    attrib["contribution_smb"] = pd.to_numeric(attrib["beta_smb"], errors="coerce") * pd.to_numeric(attrib["SMB"], errors="coerce")
    attrib["contribution_mom"] = pd.to_numeric(attrib["beta_mom"], errors="coerce") * pd.to_numeric(attrib["MOM"], errors="coerce")
    attrib["contribution_val"] = pd.to_numeric(attrib["beta_val"], errors="coerce") * pd.to_numeric(attrib["VAL"], errors="coerce")
    attrib["explained_return"] = attrib[["contribution_mkt", "contribution_smb", "contribution_mom", "contribution_val"]].sum(axis=1, skipna=True)
    attrib["residual"] = pd.to_numeric(attrib["portfolio_return"], errors="coerce") - attrib["explained_return"]

    contribution_cols = ["contribution_mkt", "contribution_smb", "contribution_mom", "contribution_val", "residual"]
    cumulative = attrib[["date"] + contribution_cols].copy()
    for col in contribution_cols:
        cumulative[col] = pd.to_numeric(cumulative[col], errors="coerce").fillna(0.0).cumsum()

    return attrib[["date", "portfolio_return", "MKT", "SMB", "MOM", "VAL", "beta_mkt", "beta_smb", "beta_mom", "beta_val"] + contribution_cols], cumulative


def _build_factor_diagnostics(
    factor_returns_df: pd.DataFrame,
    weight_book_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
) -> pd.DataFrame:
    if factor_returns_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    working = factor_returns_df.copy().sort_values("date")
    for factor in FACTOR_COLUMNS:
        series = pd.to_numeric(working[factor], errors="coerce").dropna()
        if series.empty:
            continue
        annualized_return = float(series.mean() * 252)
        annualized_vol = float(series.std(ddof=1) * np.sqrt(252)) if len(series) > 1 else np.nan
        sharpe = compute_sharpe_ratio(series)
        max_drawdown = compute_max_drawdown(series)
        hit_rate = float((series > 0).mean())
        factor_turnover = pd.to_numeric(turnover_df.loc[turnover_df["factor"] == factor, "turnover"], errors="coerce").mean() if not turnover_df.empty else np.nan
        avg_leg = float(weight_book_df.loc[weight_book_df["factor"] == factor].groupby(["date", "leg"])["ticker"].nunique().groupby("date").mean().mean()) if not weight_book_df.empty else np.nan
        rolling_sharpe_60 = float(series.rolling(60).mean().iloc[-1] * 252 / (series.rolling(60).std(ddof=1).iloc[-1] * np.sqrt(252))) if len(series) >= 60 and series.rolling(60).std(ddof=1).iloc[-1] not in [0, np.nan] else np.nan
        rolling_sharpe_252 = float(series.rolling(252).mean().iloc[-1] * 252 / (series.rolling(252).std(ddof=1).iloc[-1] * np.sqrt(252))) if len(series) >= 252 and series.rolling(252).std(ddof=1).iloc[-1] not in [0, np.nan] else np.nan
        rows.append(
            {
                "factor": factor,
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_vol,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "hit_rate": hit_rate,
                "turnover": factor_turnover,
                "avg_stocks_per_leg": avg_leg,
                "rolling_sharpe_60d": rolling_sharpe_60,
                "rolling_sharpe_252d": rolling_sharpe_252,
            }
        )
    return pd.DataFrame(rows)


def _build_turnover_tables(weight_book_df: pd.DataFrame, cost_per_turnover: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weight_book_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    grouped = weight_book_df.groupby(["date", "factor", "ticker"], as_index=False).agg(weight=("weight", "sum"))
    pivot = grouped.pivot_table(index=["date", "factor"], columns="ticker", values="weight", aggfunc="sum").fillna(0.0).sort_index()
    turnover_series = pivot.groupby(level="factor").diff().abs().sum(axis=1).fillna(0.0)
    turnover_df = turnover_series.reset_index(name="turnover")
    cost_estimate_df = turnover_df.copy()
    cost_estimate_df["estimated_cost"] = pd.to_numeric(cost_estimate_df["turnover"], errors="coerce").fillna(0.0) * cost_per_turnover
    return turnover_df, cost_estimate_df


def _build_correlation_outputs(factor_returns_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if factor_returns_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    full_corr = factor_returns_df[FACTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").corr()
    rolling_rows: list[dict[str, Any]] = []
    working = factor_returns_df.copy().sort_values("date").reset_index(drop=True)
    for end_idx in range(59, len(working)):
        window_df = working.iloc[end_idx - 59 : end_idx + 1].copy()
        corr = window_df[FACTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").corr()
        for left in FACTOR_COLUMNS:
            for right in FACTOR_COLUMNS:
                rolling_rows.append({"date": window_df["date"].iloc[-1], "factor_x": left, "factor_y": right, "correlation": corr.loc[left, right]})
    return full_corr, pd.DataFrame(rolling_rows)


def _build_backtest_summary(decile_returns_df: pd.DataFrame) -> pd.DataFrame:
    if decile_returns_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for factor, group in decile_returns_df.groupby("factor"):
        pivot = group.pivot_table(index="date", columns="decile", values="return", aggfunc="last").sort_index()
        long_leg = pd.to_numeric(pivot.get(10), errors="coerce")
        short_leg = pd.to_numeric(pivot.get(1), errors="coerce")
        spread = long_leg - short_leg
        rows.append(
            {
                "factor": factor,
                "spread_annualized_return": float(spread.dropna().mean() * 252) if spread.notna().any() else np.nan,
                "spread_sharpe": compute_sharpe_ratio(spread.dropna()) if spread.notna().any() else np.nan,
                "long_annualized_return": float(long_leg.dropna().mean() * 252) if long_leg.notna().any() else np.nan,
                "short_annualized_return": float(short_leg.dropna().mean() * 252) if short_leg.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_risk_decomposition(
    regression_input_df: pd.DataFrame,
    rolling_betas_df: pd.DataFrame,
    residual_return_series: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if regression_input_df.empty:
        return pd.DataFrame(), "N/A: regression inputs are unavailable, so factor covariance and portfolio variance cannot be estimated."
    if rolling_betas_df.empty:
        return pd.DataFrame(), "N/A: portfolio factor betas are unavailable because the regression history is insufficient."

    latest_beta_frame = rolling_betas_df.dropna(subset=["date"]).sort_values("date")
    if latest_beta_frame.empty:
        return pd.DataFrame(), "N/A: no dated portfolio beta estimates are available."

    latest_beta_row = latest_beta_frame.iloc[-1]
    beta_vector = np.array([
        pd.to_numeric(pd.Series([latest_beta_row.get("beta_mkt")]), errors="coerce").fillna(0.0).iloc[0],
        pd.to_numeric(pd.Series([latest_beta_row.get("beta_smb")]), errors="coerce").fillna(0.0).iloc[0],
        pd.to_numeric(pd.Series([latest_beta_row.get("beta_mom")]), errors="coerce").fillna(0.0).iloc[0],
        pd.to_numeric(pd.Series([latest_beta_row.get("beta_val")]), errors="coerce").fillna(0.0).iloc[0],
    ])
    if not np.isfinite(beta_vector).all():
        return pd.DataFrame(), "N/A: the latest portfolio beta estimates contain non-finite values."

    recent_factor_history = regression_input_df.copy().sort_values("date").tail(126)
    recent_factor_history = recent_factor_history[FACTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(recent_factor_history) < 60:
        return pd.DataFrame(), "N/A: factor covariance history is too short; at least 60 overlapping factor-return observations are required."

    factor_cov_df = recent_factor_history.cov()
    if factor_cov_df.empty or factor_cov_df.shape != (len(FACTOR_COLUMNS), len(FACTOR_COLUMNS)):
        return pd.DataFrame(), "N/A: factor covariance estimation failed for the recent factor-return window."
    factor_cov = factor_cov_df.to_numpy(dtype="float64")
    if not np.isfinite(factor_cov).all():
        return pd.DataFrame(), "N/A: factor covariance estimation produced non-finite values."

    factor_var = float(beta_vector.T @ factor_cov @ beta_vector)
    if not np.isfinite(factor_var):
        return pd.DataFrame(), "N/A: factor-attributable variance could not be computed from the latest portfolio betas."

    residual_series = pd.to_numeric(residual_return_series.get("residual_return"), errors="coerce").dropna() if not residual_return_series.empty else pd.Series(dtype="float64")
    residual_var = float(residual_series.tail(126).var(ddof=1)) if len(residual_series) >= 20 else np.nan
    portfolio_var_series = pd.to_numeric(regression_input_df["portfolio_return"], errors="coerce").dropna()
    observed_total_var = float(portfolio_var_series.tail(126).var(ddof=1)) if len(portfolio_var_series) >= 20 else np.nan
    if pd.isna(residual_var) or residual_var < 0:
        residual_var = max(observed_total_var - factor_var, 0.0) if pd.notna(observed_total_var) else np.nan
    if pd.isna(residual_var) or not np.isfinite(residual_var):
        return pd.DataFrame(), "N/A: idiosyncratic variance could not be estimated from residual history or total variance."

    total_var = factor_var + residual_var
    if not np.isfinite(total_var) or abs(total_var) < 1e-10:
        return pd.DataFrame(), "N/A: total portfolio variance is too close to zero for a stable decomposition."

    marginal = factor_cov @ beta_vector
    contributions = beta_vector * marginal
    if not np.isfinite(contributions).all():
        return pd.DataFrame(), "N/A: component variance contributions could not be computed from the covariance matrix."

    recomposed_total = float(np.nansum(contributions) + residual_var)
    if not np.isfinite(recomposed_total) or abs(recomposed_total) < 1e-10:
        return pd.DataFrame(), "N/A: recomposed portfolio variance is unstable."
    if abs(recomposed_total - total_var) > max(1e-8, abs(total_var) * 0.05):
        return pd.DataFrame(), "N/A: risk decomposition is unstable because factor and idiosyncratic variance do not reconcile cleanly."

    rows = []
    for factor, contrib in zip(FACTOR_COLUMNS, contributions):
        rows.append(
            {
                "component": factor,
                "variance_contrib": float(contrib),
            }
        )
    rows.append(
        {
            "component": "Idiosyncratic",
            "variance_contrib": float(residual_var),
        }
    )
    rows.append(
        {
            "component": "Total",
            "variance_contrib": float(total_var),
        }
    )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, "N/A: no risk decomposition rows were generated."

    result["direction"] = np.where(
        pd.to_numeric(result["variance_contrib"], errors="coerce") > 0,
        "Positive",
        np.where(
            pd.to_numeric(result["variance_contrib"], errors="coerce") < 0,
            "Negative",
            "Neutral",
        ),
    )
    result["abs_variance_contrib"] = pd.to_numeric(result["variance_contrib"], errors="coerce").abs()

    display_mask = result["component"].astype(str).ne("Total")
    abs_total = float(result.loc[display_mask, "abs_variance_contrib"].sum())
    if abs_total > 0:
        result.loc[display_mask, "pct_total_risk"] = result.loc[display_mask, "abs_variance_contrib"] / abs_total
    else:
        return pd.DataFrame(), "N/A: absolute component risk is too small to compute percent-of-total-risk weights."

    total_mask = result["component"].astype(str).eq("Total")
    result.loc[total_mask, "pct_total_risk"] = 1.0
    return result, ""


def _build_factor_regime_df(factor_returns_df: pd.DataFrame) -> pd.DataFrame:
    if factor_returns_df.empty:
        return pd.DataFrame()

    working = factor_returns_df.copy().sort_values("date").reset_index(drop=True)
    rows = pd.DataFrame({"date": working["date"]})
    for factor in FACTOR_COLUMNS:
        series = pd.to_numeric(working[factor], errors="coerce")
        rows[f"{factor}_return_20d"] = series.rolling(20).sum()
        rows[f"{factor}_return_60d"] = series.rolling(60).sum()
        rows[f"{factor}_sharpe_20d"] = series.rolling(20).mean() * np.sqrt(252) / series.rolling(20).std(ddof=1)
        rows[f"{factor}_sharpe_60d"] = series.rolling(60).mean() * np.sqrt(252) / series.rolling(60).std(ddof=1)
    rows["factor_dispersion_20d"] = working[FACTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").rolling(20).std(ddof=1).mean(axis=1)
    return rows


def _build_drawdown_df(
    factor_returns_df: pd.DataFrame,
    portfolio_returns_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if not portfolio_returns_df.empty:
        portfolio = portfolio_returns_df.copy().sort_values("date")
        portfolio["date"] = pd.to_datetime(portfolio["date"], errors="coerce")
        series = pd.to_numeric(portfolio["portfolio_return"], errors="coerce").fillna(0.0)
        wealth = (1.0 + series).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        rows.append(pd.DataFrame({"date": portfolio["date"], "series": "Portfolio", "drawdown": drawdown}))

    if not factor_returns_df.empty:
        factors = factor_returns_df.copy().sort_values("date")
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
        for factor in FACTOR_COLUMNS:
            if factor not in factors.columns:
                continue
            series = pd.to_numeric(factors[factor], errors="coerce").fillna(0.0)
            wealth = (1.0 + series).cumprod()
            drawdown = wealth / wealth.cummax() - 1.0
            rows.append(pd.DataFrame({"date": factors["date"], "series": factor, "drawdown": drawdown}))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["date", "series", "drawdown"])


def _compute_latest_holdings_momentum(price_history_df: pd.DataFrame) -> pd.DataFrame:
    if price_history_df.empty:
        return pd.DataFrame(columns=["ticker", "momentum_12_1_fallback"])

    price_matrix = _build_price_matrix(price_history_df)
    if price_matrix.empty or len(price_matrix.index) < 253:
        return pd.DataFrame(columns=["ticker", "momentum_12_1_fallback"])

    momentum_matrix = (price_matrix.shift(21) / price_matrix.shift(252)) - 1.0
    latest_momentum = pd.to_numeric(momentum_matrix.ffill().iloc[-1], errors="coerce").dropna()
    if latest_momentum.empty:
        return pd.DataFrame(columns=["ticker", "momentum_12_1_fallback"])

    return pd.DataFrame(
        {
            "ticker": latest_momentum.index.astype(str),
            "momentum_12_1_fallback": latest_momentum.values,
        }
    )


def _build_holdings_tables(
    holdings_snapshot_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    holdings_fundamentals_df: pd.DataFrame,
    signal_panel_df: pd.DataFrame,
    holdings_price_history_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdings_snapshot_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    holdings = holdings_snapshot_df.copy()
    holdings[COL_TICKER] = holdings[COL_TICKER].astype(str).map(_normalize_ticker)
    holdings = holdings.loc[~holdings[COL_TICKER].isin(["", "CASH", "EUR", "GBP", "NOGXX"])].copy()
    if holdings.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest_signals = signal_panel_df.sort_values("date").groupby("ticker", as_index=False).tail(1) if not signal_panel_df.empty else pd.DataFrame(columns=["ticker"])
    holdings_momentum_df = _compute_latest_holdings_momentum(holdings_price_history_df)
    merged = holdings.merge(holdings_fundamentals_df, left_on=COL_TICKER, right_on="ticker", how="left")
    merged = merged.merge(
        latest_signals[["ticker", "gics_sector", "smb_signal", "smb_zscore", "mom_signal", "mom_zscore", "val_signal", "val_zscore"]],
        on="ticker",
        how="left",
    )
    merged = merged.merge(holdings_momentum_df, on="ticker", how="left")
    if "gics_sector" not in merged.columns:
        merged = merged.merge(universe_df[["ticker", "gics_sector"]], on="ticker", how="left")

    merged["market_cap"] = pd.to_numeric(merged["market_cap"], errors="coerce")
    merged["selected_pe"] = pd.to_numeric(merged["selected_pe"], errors="coerce")
    merged["earnings_yield"] = pd.to_numeric(merged["earnings_yield"], errors="coerce")
    merged["momentum_12_1"] = pd.to_numeric(merged["mom_signal"], errors="coerce")
    merged["momentum_12_1_fallback"] = pd.to_numeric(merged.get("momentum_12_1_fallback"), errors="coerce")
    merged["momentum_12_1"] = merged["momentum_12_1"].combine_first(merged["momentum_12_1_fallback"])

    fallback_market_cap = float(merged["market_cap"].dropna().median()) if merged["market_cap"].notna().any() else np.nan
    fallback_pe = float(merged["selected_pe"].dropna().median()) if merged["selected_pe"].notna().any() else np.nan
    fallback_ey = float(merged["earnings_yield"].dropna().median()) if merged["earnings_yield"].notna().any() else np.nan
    fallback_mom = float(merged["momentum_12_1"].dropna().median()) if merged["momentum_12_1"].notna().any() else 0.0

    merged["market_cap"] = merged["market_cap"].fillna(fallback_market_cap)
    merged["selected_pe"] = merged["selected_pe"].fillna(fallback_pe)
    merged["earnings_yield"] = merged["earnings_yield"].fillna(fallback_ey)
    merged["momentum_12_1"] = merged["momentum_12_1"].fillna(fallback_mom)

    fallback_signal_map = {
        "smb": -pd.to_numeric(merged["market_cap"], errors="coerce"),
        "mom": pd.to_numeric(merged["momentum_12_1"], errors="coerce"),
        "val": pd.to_numeric(merged["earnings_yield"], errors="coerce"),
    }
    for prefix, raw_series in fallback_signal_map.items():
        signal_col = f"{prefix}_signal"
        zscore_col = f"{prefix}_zscore"
        merged[signal_col] = pd.to_numeric(merged.get(signal_col), errors="coerce")
        merged[zscore_col] = pd.to_numeric(merged.get(zscore_col), errors="coerce")

        fallback_signal = _winsorize_series(raw_series, 0.01, 0.99)
        fallback_zscore = _zscore_series(fallback_signal)

        merged[signal_col] = merged[signal_col].where(merged[signal_col].notna(), fallback_signal)
        merged[zscore_col] = merged[zscore_col].where(merged[zscore_col].notna(), fallback_zscore)

    for col in ["smb_signal", "smb_zscore", "mom_signal", "mom_zscore", "val_signal", "val_zscore"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    population = merged.copy()
    population["size_position"] = population["market_cap"].rank(pct=True)
    population["value_position"] = population["earnings_yield"].rank(pct=True)
    population["momentum_position"] = population["momentum_12_1"].rank(pct=True)
    latest_holdings_exposure = merged.assign(
        size_position=population["size_position"].values,
        value_position=population["value_position"].values,
        momentum_position=population["momentum_position"].values,
    )

    contrib = latest_holdings_exposure.copy()
    contrib["size_contribution"] = pd.to_numeric(contrib.get("weight"), errors="coerce").fillna(0.0) * pd.to_numeric(contrib["smb_zscore"], errors="coerce").fillna(0.0)
    contrib["momentum_contribution"] = pd.to_numeric(contrib.get("weight"), errors="coerce").fillna(0.0) * pd.to_numeric(contrib["mom_zscore"], errors="coerce").fillna(0.0)
    contrib["value_contribution"] = pd.to_numeric(contrib.get("weight"), errors="coerce").fillna(0.0) * pd.to_numeric(contrib["val_zscore"], errors="coerce").fillna(0.0)
    contrib["abs_total_contribution"] = contrib[["size_contribution", "momentum_contribution", "value_contribution"]].abs().sum(axis=1)
    holdings_factor_contribution_df = contrib.sort_values("abs_total_contribution", ascending=False).reset_index(drop=True)

    return latest_holdings_exposure, holdings_factor_contribution_df


def _build_holdings_signals_output(
    latest_holdings_signals_df: pd.DataFrame,
    weighted_signal_contributions_df: pd.DataFrame,
) -> dict[str, Any]:
    reason = _reason_or_default(
        latest_holdings_signals_df,
        "N/A: live holdings signal descriptors are unavailable because no eligible non-cash holdings could be matched to signal inputs.",
    )
    return {
        "latest": latest_holdings_signals_df,
        "weighted_signal_contributions": weighted_signal_contributions_df,
        "reason": reason,
    }


def _build_portfolio_factor_betas_output(
    latest_portfolio_factor_betas_df: pd.DataFrame,
    regression_summary_df: pd.DataFrame,
    rolling_portfolio_factor_betas_df: pd.DataFrame,
    multi_horizon_portfolio_factor_betas_df: pd.DataFrame,
) -> dict[str, Any]:
    reason = _reason_or_default(
        latest_portfolio_factor_betas_df,
        "N/A: portfolio factor betas require enough overlapping portfolio and factor return history for regression.",
    )
    return {
        "latest": latest_portfolio_factor_betas_df,
        "regression_summary": regression_summary_df,
        "rolling": rolling_portfolio_factor_betas_df,
        "multi_horizon": multi_horizon_portfolio_factor_betas_df,
        "reason": reason,
    }


def build_scenario_template(latest_beta_row: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["shock_MKT", "shock_SMB", "shock_MOM", "shock_VAL", "expected_portfolio_return"]
    if latest_beta_row is None:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=["shock_MKT", "shock_VAL", "expected_portfolio_return"])

    template = pd.DataFrame(
        [
            {"shock_MKT": -0.05, "shock_SMB": 0.00, "shock_MOM": 0.00, "shock_VAL": 0.00},
            {"shock_MKT": 0.00, "shock_SMB": -0.02, "shock_MOM": 0.02, "shock_VAL": 0.00},
        ]
    )
    for factor in ["MKT", "SMB", "MOM", "VAL"]:
        template[f"beta_{factor}"] = float(pd.to_numeric(pd.Series([latest_beta_row.get(f"beta_{factor.lower()}")]), errors="coerce").fillna(0.0).iloc[0])
    template["expected_portfolio_return"] = (
        template["beta_MKT"] * template["shock_MKT"]
        + template["beta_SMB"] * template["shock_SMB"]
        + template["beta_MOM"] * template["shock_MOM"]
        + template["beta_VAL"] * template["shock_VAL"]
    )
    template = template[columns]

    grid_values = np.linspace(-0.05, 0.05, 11)
    grid_rows = []
    beta_mkt = float(pd.to_numeric(pd.Series([latest_beta_row.get("beta_mkt")]), errors="coerce").fillna(0.0).iloc[0])
    beta_val = float(pd.to_numeric(pd.Series([latest_beta_row.get("beta_val")]), errors="coerce").fillna(0.0).iloc[0])
    for shock_mkt in grid_values:
        for shock_val in grid_values:
            grid_rows.append(
                {
                    "shock_MKT": shock_mkt,
                    "shock_VAL": shock_val,
                    "expected_portfolio_return": beta_mkt * shock_mkt + beta_val * shock_val,
                }
            )
    return template, pd.DataFrame(grid_rows)


def optimize_rebalance_to_target_exposures(
    holdings_factor_contribution_df: pd.DataFrame,
    current_exposures: pd.Series,
    target_exposures: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdings_factor_contribution_df.empty or current_exposures.empty:
        return pd.DataFrame(), pd.DataFrame()

    if target_exposures is None:
        target_exposures = pd.Series({"SMB": 0.0, "MOM": 0.0, "VAL": 0.0})

    x_mat = holdings_factor_contribution_df[["smb_zscore", "mom_zscore", "val_zscore"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    current_weights = pd.to_numeric(holdings_factor_contribution_df.get("weight"), errors="coerce").fillna(0.0).to_numpy(dtype="float64")
    current_vec = np.array([
        float(pd.to_numeric(pd.Series([current_exposures.get("SMB")]), errors="coerce").fillna(0.0).iloc[0]),
        float(pd.to_numeric(pd.Series([current_exposures.get("MOM")]), errors="coerce").fillna(0.0).iloc[0]),
        float(pd.to_numeric(pd.Series([current_exposures.get("VAL")]), errors="coerce").fillna(0.0).iloc[0]),
    ])
    target_vec = np.array([
        float(pd.to_numeric(pd.Series([target_exposures.get("SMB")]), errors="coerce").fillna(0.0).iloc[0]),
        float(pd.to_numeric(pd.Series([target_exposures.get("MOM")]), errors="coerce").fillna(0.0).iloc[0]),
        float(pd.to_numeric(pd.Series([target_exposures.get("VAL")]), errors="coerce").fillna(0.0).iloc[0]),
    ])
    diff = target_vec - current_vec
    if x_mat.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    try:
        # Solve X' * delta = diff in least-squares form, where:
        # - X is holdings x factors
        # - delta is holding-level weight changes
        # - diff is desired factor-exposure change
        exposure_matrix = x_mat.T  # factors x holdings
        delta, _, _, _ = np.linalg.lstsq(exposure_matrix, diff, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        delta = np.zeros_like(current_weights)
    projected_weights = current_weights + delta
    trades_df = holdings_factor_contribution_df[[COL_TICKER, COL_TEAM, "weight"]].copy()
    trades_df["recommended_weight_change"] = delta
    trades_df["projected_weight"] = projected_weights
    projected_exposures = x_mat.T @ projected_weights
    projected_df = pd.DataFrame(
        [
            {"factor": "SMB", "current_exposure": current_vec[0], "target_exposure": target_vec[0], "projected_exposure": projected_exposures[0]},
            {"factor": "MOM", "current_exposure": current_vec[1], "target_exposure": target_vec[1], "projected_exposure": projected_exposures[1]},
            {"factor": "VAL", "current_exposure": current_vec[2], "target_exposure": target_vec[2], "projected_exposure": projected_exposures[2]},
        ]
    )
    return trades_df.sort_values("recommended_weight_change", ascending=False).reset_index(drop=True), projected_df


def build_factor_analytics_platform(
    holdings_snapshot_df: pd.DataFrame,
    config: FactorConstructionConfig | None = None,
) -> dict[str, Any]:
    config = config or FactorConstructionConfig()
    cache_key = _build_factor_analytics_cache_key(holdings_snapshot_df, config)
    cached = _FACTOR_ANALYTICS_CACHE.get(cache_key)
    if cached is not None:
        return _copy_cached_analytics_payload(cached)
    persisted = _load_persisted_factor_analytics_cache(cache_key)
    if persisted is not None:
        return _store_factor_analytics_cache(cache_key, persisted)
    notes = [
        "These are custom live factors built from current Yahoo data and S&P 500 constituents, not official Fama-French factors.",
        "The factor universe uses current S&P 500 constituents, so the system is subject to survivorship bias.",
        "SMB and VAL use live Yahoo market cap and PE data, so they are not point-in-time historical fundamentals.",
    ]

    universe_df = get_sp500_constituents()
    if universe_df.empty:
        return _finalize_factor_analytics_cache(
            cache_key,
            _empty_analytics_payload(notes + ["Unable to fetch S&P 500 constituents."]),
        )

    universe_fundamentals = fetch_live_security_fundamentals(universe_df["ticker"].tolist())
    universe = universe_df.merge(universe_fundamentals, on="ticker", how="left")
    tickers = sorted(set(universe["ticker"].dropna().astype(str).tolist() + ["SPY"]))
    price_history_df = fetch_multiple_price_histories(
        tickers,
        lookback_days=FACTOR_MODEL_PRICE_LOOKBACK_DAYS,
    )
    price_matrix = _build_price_matrix(price_history_df)
    if price_matrix.empty or "SPY" not in price_matrix.columns:
        analytics = _empty_analytics_payload(notes + ["Unable to build price history or SPY market history."])
        analytics["universe_fundamentals"] = universe
        return _finalize_factor_analytics_cache(cache_key, analytics)

    market_returns = pd.to_numeric(price_matrix["SPY"], errors="coerce").pct_change()
    stock_tickers = [ticker for ticker in price_matrix.columns if ticker != "SPY"]
    stock_returns = price_matrix[stock_tickers].pct_change()
    beta_matrix = _compute_daily_stock_beta(stock_returns, market_returns, lookback=252)
    rebalance_dates = price_matrix.resample("ME").last().index

    signal_frames = [
        _build_signal_frame_for_date(date, universe, price_matrix[stock_tickers], beta_matrix, config)
        for date in rebalance_dates
    ]
    signal_panel_df = pd.concat([frame for frame in signal_frames if not frame.empty], ignore_index=True) if signal_frames else pd.DataFrame()
    weight_book_df = _build_factor_weight_book(signal_panel_df, config)
    decile_weight_book_df = _build_decile_weight_book(signal_panel_df, config)
    factor_returns_df, decile_returns_df = _compute_period_returns(weight_book_df, decile_weight_book_df, stock_returns, market_returns)
    portfolio_returns_df = build_portfolio_return_history(
        lookback_days=FACTOR_MODEL_PRICE_LOOKBACK_DAYS,
        base_price_history_df=price_history_df,
    )

    regression_summary_df, rolling_betas_df, multi_horizon_exposures_df, residual_return_series_df, regression_input_df, latest_portfolio_factor_betas_df = _build_regression_suite(
        factor_returns_df=factor_returns_df,
        portfolio_returns_df=portfolio_returns_df,
        config=config,
    )

    attribution_df, cumulative_attribution_df = _build_attribution_tables(regression_input_df, rolling_betas_df)
    turnover_df, cost_estimate_df = _build_turnover_tables(weight_book_df, config.cost_per_turnover)
    factor_diagnostics_df = _build_factor_diagnostics(factor_returns_df, weight_book_df, turnover_df)
    correlation_matrix_df, rolling_correlation_panel_df = _build_correlation_outputs(factor_returns_df)
    factor_backtest_summary_df = _build_backtest_summary(decile_returns_df)
    risk_decomposition_df, risk_decomposition_reason = _build_risk_decomposition(regression_input_df, rolling_betas_df, residual_return_series_df)
    factor_regime_df = _build_factor_regime_df(factor_returns_df)
    drawdowns_df = _build_drawdown_df(factor_returns_df, portfolio_returns_df)

    holdings_tickers = (
        holdings_snapshot_df[COL_TICKER].dropna().astype(str).map(_normalize_ticker).tolist()
        if not holdings_snapshot_df.empty else []
    )
    holdings_ticker_set = sorted({ticker for ticker in holdings_tickers if str(ticker).strip()})
    holdings_fundamentals_df = universe.loc[
        universe["ticker"].astype(str).isin(holdings_ticker_set)
    ].copy()
    missing_holdings_fundamentals = [
        ticker for ticker in holdings_ticker_set
        if ticker not in set(holdings_fundamentals_df["ticker"].astype(str).tolist())
    ]
    if missing_holdings_fundamentals:
        extra_holdings_fundamentals_df = fetch_live_security_fundamentals(missing_holdings_fundamentals)
        holdings_fundamentals_df = pd.concat(
            [holdings_fundamentals_df, extra_holdings_fundamentals_df],
            ignore_index=True,
        )
        if not holdings_fundamentals_df.empty:
            holdings_fundamentals_df = holdings_fundamentals_df.drop_duplicates(
                subset=["ticker"],
                keep="last",
            ).reset_index(drop=True)

    holdings_price_history_df = price_history_df.loc[
        price_history_df["ticker"].astype(str).isin(holdings_ticker_set)
    ].copy() if not price_history_df.empty else pd.DataFrame()
    missing_holdings_price_tickers = [
        ticker for ticker in holdings_ticker_set
        if ticker not in set(holdings_price_history_df["ticker"].astype(str).tolist())
    ]
    if missing_holdings_price_tickers:
        extra_holdings_price_history_df = fetch_multiple_price_histories(
            missing_holdings_price_tickers,
            lookback_days=FACTOR_MODEL_PRICE_LOOKBACK_DAYS,
        )
        holdings_price_history_df = pd.concat(
            [holdings_price_history_df, extra_holdings_price_history_df],
            ignore_index=True,
        )
        if not holdings_price_history_df.empty:
            holdings_price_history_df = holdings_price_history_df.drop_duplicates(
                subset=["date", "ticker"],
                keep="last",
            ).reset_index(drop=True)
    latest_holdings_exposure_df, holdings_factor_contribution_df = _build_holdings_tables(
        holdings_snapshot_df=holdings_snapshot_df,
        universe_df=universe,
        holdings_fundamentals_df=holdings_fundamentals_df,
        signal_panel_df=signal_panel_df,
        holdings_price_history_df=holdings_price_history_df,
    )
    holdings_signals = _build_holdings_signals_output(
        latest_holdings_signals_df=latest_holdings_exposure_df,
        weighted_signal_contributions_df=holdings_factor_contribution_df,
    )
    portfolio_factor_betas = _build_portfolio_factor_betas_output(
        latest_portfolio_factor_betas_df=latest_portfolio_factor_betas_df,
        regression_summary_df=regression_summary_df,
        rolling_portfolio_factor_betas_df=rolling_betas_df,
        multi_horizon_portfolio_factor_betas_df=multi_horizon_exposures_df,
    )

    latest_beta_row = rolling_betas_df.sort_values("date").iloc[-1] if not rolling_betas_df.empty else None
    scenario_template_df, grid_stress_df = build_scenario_template(latest_beta_row)

    current_exposures = pd.Series(
        {
            "SMB": float(pd.to_numeric(holdings_factor_contribution_df.get("size_contribution"), errors="coerce").fillna(0.0).sum()) if not holdings_factor_contribution_df.empty else 0.0,
            "MOM": float(pd.to_numeric(holdings_factor_contribution_df.get("momentum_contribution"), errors="coerce").fillna(0.0).sum()) if not holdings_factor_contribution_df.empty else 0.0,
            "VAL": float(pd.to_numeric(holdings_factor_contribution_df.get("value_contribution"), errors="coerce").fillna(0.0).sum()) if not holdings_factor_contribution_df.empty else 0.0,
        }
    )
    optimizer_recommendation_df, optimizer_projected_exposures_df = optimize_rebalance_to_target_exposures(
        holdings_factor_contribution_df=holdings_factor_contribution_df,
        current_exposures=current_exposures,
    )

    visualization_data = {
        "exposure_heatmap": multi_horizon_exposures_df.copy(),
        "attribution_stacked_series": cumulative_attribution_df.copy(),
        "factor_risk_contribution_pie": risk_decomposition_df.copy(),
        "top_contributors_bar": holdings_factor_contribution_df.copy(),
        "correlation_heatmap": correlation_matrix_df.copy(),
        "decile_spread_chart": decile_returns_df.copy(),
        "factor_drawdown_series": drawdowns_df.copy(),
    }

    metadata = {
        "universe_definition": "Current S&P 500 constituents from public web sources",
        "rebalance_frequency": "Monthly end-of-month",
        "weighting_scheme": config.weighting_scheme,
        "factor_definitions": {
            "MKT": "SPY daily returns",
            "SMB": "Long small-cap, short large-cap using market cap",
            "MOM": "Long high 12-1 momentum, short low 12-1 momentum",
            "VAL": "Long cheap, short expensive using earnings yield = 1 / PE",
        },
        "limitations": [
            "Survivorship bias from current S&P 500 universe",
            "Yahoo Finance data quality and field availability may vary",
            "Live fundamentals are not point-in-time historical fundamentals",
        ],
    }

    module_frames = {
        "factor_returns": factor_returns_df,
        "regression_summary": portfolio_factor_betas["regression_summary"],
        "rolling_betas": portfolio_factor_betas["rolling"],
        "portfolio_returns": portfolio_returns_df,
        "drawdowns": drawdowns_df,
        "attribution": attribution_df,
        "cumulative_attribution": cumulative_attribution_df,
        "risk_decomposition": risk_decomposition_df,
        "risk_decomposition_reason": risk_decomposition_reason,
        "holdings_contribution": holdings_signals["weighted_signal_contributions"],
        "factor_diagnostics": factor_diagnostics_df,
        "factor_correlations": correlation_matrix_df,
        "rolling_factor_correlations": rolling_correlation_panel_df,
        "multi_horizon_exposures": portfolio_factor_betas["multi_horizon"],
        "decile_backtest": decile_returns_df,
        "factor_backtest_summary": factor_backtest_summary_df,
        "turnover_cost": turnover_df.merge(cost_estimate_df, on=["date", "factor"], how="outer") if not turnover_df.empty or not cost_estimate_df.empty else pd.DataFrame(),
        "scenario_template": scenario_template_df,
        "optimizer_output": optimizer_recommendation_df,
        "optimizer_projected_exposures": optimizer_projected_exposures_df,
        "universe_fundamentals": universe,
        "latest_holdings_exposure": holdings_signals["latest"],
        "residual_return_series": residual_return_series_df,
    }

    for key, value in module_frames.items():
        if isinstance(value, pd.DataFrame) and value.empty:
            notes.append(f"{key} returned empty output.")

    analytics = {
        **module_frames,
        "holdings_signals": holdings_signals,
        "portfolio_factor_betas": portfolio_factor_betas,
        "risk_decomposition_reason": risk_decomposition_reason,
        "metadata": metadata,
        "visualization_data": visualization_data,
        "notes": notes,
    }
    _validate_analytics_payload(analytics)
    return _finalize_factor_analytics_cache(cache_key, analytics)


# Backward-compatible alias for older page imports.
build_custom_live_factor_model = build_factor_analytics_platform
