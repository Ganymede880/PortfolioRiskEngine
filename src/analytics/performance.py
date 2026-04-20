"""
Performance metric utilities for the CMCSIF portfolio tracker.

This module computes:
- volatility
- Sharpe ratio
- Sortino ratio
- max drawdown
- rolling volatility / rolling Sharpe helpers

These functions operate on return series and are intentionally independent
from the raw holdings ingestion pipeline.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config.settings import settings


def _clean_return_series(return_series: pd.Series) -> pd.Series:
    """
    Convert a return series to numeric and drop missing values.
    """
    return pd.to_numeric(return_series, errors="coerce").dropna()


def prepare_flow_adjusted_history(
    history_df: pd.DataFrame,
    value_column: str,
    flow_column: str = "net_external_flow",
) -> pd.DataFrame:
    """
    Prepare a dated history with flow-adjusted daily P&L and returns.

    Conventions:
    - external flow > 0 means capital added to the sleeve/fund
    - external flow < 0 means capital withdrawn from the sleeve/fund
    - performance P&L removes those external flows from the change in AUM
    """
    if history_df.empty or value_column not in history_df.columns:
        return pd.DataFrame(
            columns=[
                "date",
                value_column,
                flow_column,
                "previous_value",
                "performance_pnl",
                "performance_return",
            ]
        )

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    if flow_column not in df.columns:
        df[flow_column] = 0.0
    df[flow_column] = pd.to_numeric(df[flow_column], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date", value_column]).sort_values("date").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                value_column,
                flow_column,
                "previous_value",
                "performance_pnl",
                "performance_return",
            ]
        )

    df["previous_value"] = pd.to_numeric(df[value_column], errors="coerce").shift(1)
    df["performance_pnl"] = (
        pd.to_numeric(df[value_column], errors="coerce").diff()
        - pd.to_numeric(df[flow_column], errors="coerce").fillna(0.0)
    )

    df["performance_return"] = np.where(
        df["previous_value"].notna() & df["previous_value"].ne(0),
        ((pd.to_numeric(df[value_column], errors="coerce") - pd.to_numeric(df[flow_column], errors="coerce").fillna(0.0)) / df["previous_value"]) - 1.0,
        np.nan,
    )
    df["performance_return"] = pd.to_numeric(df["performance_return"], errors="coerce")

    return df


def build_flow_adjusted_benchmark_series(
    benchmark_return_series: pd.Series,
    external_flow_series: pd.Series,
    initial_value: float,
) -> pd.Series:
    """
    Build a benchmark AUM series that absorbs the same external cash flows.
    """
    benchmark_returns = pd.to_numeric(benchmark_return_series, errors="coerce")
    external_flows = pd.to_numeric(external_flow_series, errors="coerce").fillna(0.0)

    if benchmark_returns.empty:
        return pd.Series(dtype="float64")

    values: list[float] = []
    current_value = float(initial_value)

    for idx, ret in benchmark_returns.items():
        flow = float(external_flows.get(idx, 0.0)) if hasattr(external_flows, "get") else 0.0
        if pd.isna(ret):
            values.append(np.nan)
            continue
        current_value = (current_value * (1.0 + float(ret))) + flow
        values.append(float(current_value))

    return pd.Series(values, index=benchmark_returns.index, dtype="float64")


def compute_annualized_volatility(return_series: pd.Series) -> float:
    """
    Compute annualized volatility from a daily return series.
    """
    clean = _clean_return_series(return_series)
    if len(clean) < 2:
        return 0.0

    daily_vol = clean.std(ddof=1)
    return float(daily_vol * np.sqrt(settings.trading_days_per_year))


def compute_sharpe_ratio(
    return_series: pd.Series,
    risk_free_rate_annual: float | None = None,
) -> float:
    """
    Compute annualized Sharpe ratio from a daily return series.

    Sharpe = annualized excess return / annualized volatility
    """
    clean = _clean_return_series(return_series)
    if len(clean) < 2:
        return 0.0

    if risk_free_rate_annual is None:
        risk_free_rate_annual = settings.risk_free_rate_annual

    daily_rf = risk_free_rate_annual / settings.trading_days_per_year
    excess_returns = clean - daily_rf

    excess_vol = excess_returns.std(ddof=1)
    if excess_vol == 0 or pd.isna(excess_vol):
        return 0.0

    annualized_excess_return = excess_returns.mean() * settings.trading_days_per_year
    annualized_excess_vol = excess_vol * np.sqrt(settings.trading_days_per_year)

    return float(annualized_excess_return / annualized_excess_vol)


def compute_sortino_ratio(
    return_series: pd.Series,
    risk_free_rate_annual: float | None = None,
) -> float:
    """
    Compute annualized Sortino ratio from a daily return series.

    Sortino uses downside deviation instead of total volatility.
    """
    clean = _clean_return_series(return_series)
    if len(clean) < 2:
        return 0.0

    if risk_free_rate_annual is None:
        risk_free_rate_annual = settings.risk_free_rate_annual

    daily_rf = risk_free_rate_annual / settings.trading_days_per_year
    excess_returns = clean - daily_rf

    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return 0.0

    downside_deviation = downside_returns.std(ddof=1)
    if downside_deviation == 0 or pd.isna(downside_deviation):
        return 0.0

    annualized_excess_return = excess_returns.mean() * settings.trading_days_per_year
    annualized_downside_deviation = downside_deviation * np.sqrt(settings.trading_days_per_year)

    return float(annualized_excess_return / annualized_downside_deviation)


def compute_cumulative_return_series(return_series: pd.Series) -> pd.Series:
    """
    Convert a daily return series into cumulative return series.

    Formula:
        cumulative = (1 + r).cumprod() - 1
    """
    clean = pd.to_numeric(return_series, errors="coerce").fillna(0.0)
    return (1.0 + clean).cumprod() - 1.0


def compute_drawdown_series(return_series: pd.Series) -> pd.Series:
    """
    Compute drawdown series from a daily return series.

    Drawdown is measured relative to the running peak of cumulative wealth.
    """
    clean = pd.to_numeric(return_series, errors="coerce").fillna(0.0)
    wealth_index = (1.0 + clean).cumprod()
    running_peak = wealth_index.cummax()
    drawdown = wealth_index / running_peak - 1.0
    return drawdown


def compute_max_drawdown(return_series: pd.Series) -> float:
    """
    Compute the maximum drawdown from a daily return series.
    """
    drawdown = compute_drawdown_series(return_series)
    if drawdown.empty:
        return 0.0
    return float(drawdown.min())


def compute_rolling_volatility(
    return_series: pd.Series,
    window: int = 21,
) -> pd.Series:
    """
    Compute rolling annualized volatility over a specified window.
    """
    clean = pd.to_numeric(return_series, errors="coerce")
    rolling_daily_vol = clean.rolling(window=window).std(ddof=1)
    return rolling_daily_vol * np.sqrt(settings.trading_days_per_year)


def compute_rolling_sharpe_ratio(
    return_series: pd.Series,
    window: int = 63,
    risk_free_rate_annual: float | None = None,
) -> pd.Series:
    """
    Compute rolling annualized Sharpe ratio over a specified window.
    """
    clean = pd.to_numeric(return_series, errors="coerce")

    if risk_free_rate_annual is None:
        risk_free_rate_annual = settings.risk_free_rate_annual

    daily_rf = risk_free_rate_annual / settings.trading_days_per_year
    excess_returns = clean - daily_rf

    rolling_mean = excess_returns.rolling(window=window).mean() * settings.trading_days_per_year
    rolling_std = excess_returns.rolling(window=window).std(ddof=1) * np.sqrt(settings.trading_days_per_year)

    sharpe = rolling_mean / rolling_std
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)

    return sharpe


def summarize_performance_metrics(
    return_series: pd.Series,
    risk_free_rate_annual: float | None = None,
) -> Dict[str, float]:
    """
    Compute a summary set of core performance metrics for a return series.
    """
    clean = _clean_return_series(return_series)

    if clean.empty:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "cumulative_return": 0.0,
            "observations": 0,
        }

    annualized_return = float(clean.mean() * settings.trading_days_per_year)
    annualized_volatility = compute_annualized_volatility(clean)
    sharpe = compute_sharpe_ratio(clean, risk_free_rate_annual=risk_free_rate_annual)
    sortino = compute_sortino_ratio(clean, risk_free_rate_annual=risk_free_rate_annual)
    max_drawdown = compute_max_drawdown(clean)

    cumulative_series = compute_cumulative_return_series(clean)
    cumulative_return = float(cumulative_series.iloc[-1]) if not cumulative_series.empty else 0.0

    return {
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "cumulative_return": cumulative_return,
        "observations": int(len(clean)),
    }


def build_performance_view(
    return_series_df: pd.DataFrame,
    return_column: str = "portfolio_daily_return",
    date_column: str = "date",
) -> Dict[str, Any]:
    """
    Build a performance view from a DataFrame containing a dated return series.

    Returns:
    - metrics: dict of summary metrics
    - cumulative_returns: DataFrame with date and cumulative return
    - drawdowns: DataFrame with date and drawdown
    - rolling_volatility: DataFrame with date and rolling volatility
    - rolling_sharpe: DataFrame with date and rolling Sharpe
    """
    if return_series_df.empty or return_column not in return_series_df.columns:
        empty_ts = pd.DataFrame(columns=[date_column, "value"])
        return {
            "metrics": summarize_performance_metrics(pd.Series(dtype="float64")),
            "cumulative_returns": empty_ts.copy(),
            "drawdowns": empty_ts.copy(),
            "rolling_volatility": empty_ts.copy(),
            "rolling_sharpe": empty_ts.copy(),
        }

    working = return_series_df.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.sort_values(date_column).reset_index(drop=True)

    returns = pd.to_numeric(working[return_column], errors="coerce")

    cumulative_returns = pd.DataFrame({
        date_column: working[date_column],
        "value": compute_cumulative_return_series(returns),
    })

    drawdowns = pd.DataFrame({
        date_column: working[date_column],
        "value": compute_drawdown_series(returns),
    })

    rolling_volatility = pd.DataFrame({
        date_column: working[date_column],
        "value": compute_rolling_volatility(returns),
    })

    rolling_sharpe = pd.DataFrame({
        date_column: working[date_column],
        "value": compute_rolling_sharpe_ratio(returns),
    })

    metrics = summarize_performance_metrics(returns)

    return {
        "metrics": metrics,
        "cumulative_returns": cumulative_returns,
        "drawdowns": drawdowns,
        "rolling_volatility": rolling_volatility,
        "rolling_sharpe": rolling_sharpe,
    }
