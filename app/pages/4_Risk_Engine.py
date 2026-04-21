"""
Risk Engine page for the CMCSIF Portfolio Tracker.

This page centralizes portfolio risk metrics, risk decomposition, and
scenario analysis using the factor analytics backend.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.analytics.exposure as exposure_module
from src.analytics.ledger import apply_cash_ledger_entries_to_positions, apply_trades_to_positions
from src.analytics.performance import prepare_flow_adjusted_history
from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import load_all_portfolio_snapshots, load_cash_ledger, load_position_state, load_trade_receipts
from src.db.session import session_scope
from src.utils.constants import FACTOR_COLORS, TEAM_COLORS
from src.utils.ui import apply_app_theme, left_align_dataframe, style_plotly_figure, render_top_nav


COL_TICKER = "ticker"
COL_TEAM = "team"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
TEAM_ORDER = ["Consumer", "E&U", "F&R", "Healthcare", "TMT", "M&I"]
EXTERNAL_FLOW_ACTIVITY_TYPES = {"SECTOR_REBALANCE", "PORTFOLIO_LIQUIDATION"}
SCENARIO_LIBRARY = {
    "Custom": {
        "description": "Manual factor shock entry.",
        "shock_mkt": 0.00,
        "shock_smb": 0.00,
        "shock_mom": 0.00,
        "shock_val": 0.00,
    },
    "Systematic De-Risking": {
        "description": "Broad risk-off liquidation where beta, liquidity, and cyclicality all sell off together.",
        "shock_mkt": -0.12,
        "shock_smb": -0.08,
        "shock_mom": -0.06,
        "shock_val": 0.03,
    },
    "Growth Selloff": {
        "description": "Long-duration growth equities reprice sharply lower as rates rise and valuation multiples compress.",
        "shock_mkt": -0.06,
        "shock_smb": -0.02,
        "shock_mom": -0.08,
        "shock_val": 0.10,
    },
    "Value Crash": {
        "description": "Crowded value exposures unwind quickly while growth and momentum leadership reassert themselves.",
        "shock_mkt": 0.04,
        "shock_smb": 0.03,
        "shock_mom": 0.06,
        "shock_val": -0.12,
    },
    "Small Cap Stress": {
        "description": "Financing pressure and falling risk appetite hit smaller and less liquid companies much harder than the broad market.",
        "shock_mkt": -0.05,
        "shock_smb": -0.15,
        "shock_mom": -0.02,
        "shock_val": -0.03,
    },
    "Momentum Crash": {
        "description": "Recent winners reverse violently as crowded momentum positioning unwinds even without a full market selloff.",
        "shock_mkt": 0.02,
        "shock_smb": 0.04,
        "shock_mom": -0.20,
        "shock_val": 0.06,
    },
    "Inflation Shock": {
        "description": "Higher inflation and rising yields pressure broad equities while value and real-asset-linked exposures outperform.",
        "shock_mkt": -0.07,
        "shock_smb": -0.04,
        "shock_mom": -0.05,
        "shock_val": 0.08,
    },
    "Quality Rotation": {
        "description": "Investors rotate away from speculative leadership and toward cheaper, more stable, higher-quality businesses.",
        "shock_mkt": 0.03,
        "shock_smb": 0.02,
        "shock_mom": -0.03,
        "shock_val": 0.05,
    },
    "Liquidity Crunch": {
        "description": "Market depth vanishes and correlations spike, causing a sharp selloff across most risky factor exposures.",
        "shock_mkt": -0.08,
        "shock_smb": -0.12,
        "shock_mom": -0.10,
        "shock_val": -0.06,
    },
    "Factor Crowding Unwind": {
        "description": "Crowded quant and relative-value exposures de-gross rapidly, hurting style factors even if the index is roughly flat.",
        "shock_mkt": 0.00,
        "shock_smb": -0.06,
        "shock_mom": -0.12,
        "shock_val": -0.08,
    },
}

def _format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    return f"(${abs(numeric_value):,.2f})" if numeric_value < 0 else f"${numeric_value:,.2f}"


def _extract_latest_factor_loadings(rolling_betas_df: pd.DataFrame) -> dict[str, float] | None:
    if rolling_betas_df.empty:
        return None

    latest = rolling_betas_df.copy()
    latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
    latest = latest.dropna(subset=["date"]).sort_values("date")
    if latest.empty:
        return None

    row = latest.iloc[-1]
    return {
        "beta_mkt": float(pd.to_numeric(pd.Series([row.get("beta_mkt")]), errors="coerce").fillna(0.0).iloc[0]),
        "beta_smb": float(pd.to_numeric(pd.Series([row.get("beta_smb")]), errors="coerce").fillna(0.0).iloc[0]),
        "beta_mom": float(pd.to_numeric(pd.Series([row.get("beta_mom")]), errors="coerce").fillna(0.0).iloc[0]),
        "beta_val": float(pd.to_numeric(pd.Series([row.get("beta_val")]), errors="coerce").fillna(0.0).iloc[0]),
    }


def _compute_series_beta(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    min_obs: int = 20,
) -> float | None:
    aligned = pd.concat(
        [
            pd.to_numeric(asset_returns, errors="coerce").rename("asset"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
        ],
        axis=1,
    ).dropna()
    if len(aligned) < min_obs:
        return None

    benchmark_var = float(aligned["benchmark"].var(ddof=1))
    if not np.isfinite(benchmark_var) or abs(benchmark_var) <= 1e-12:
        return None

    covariance = float(aligned["asset"].cov(aligned["benchmark"]))
    if not np.isfinite(covariance):
        return None
    return covariance / benchmark_var


def _build_scenario_trade_recommendations(
    snapshot_df: pd.DataFrame,
    holdings_signals_df: pd.DataFrame,
    pod_market_betas: dict[str, float] | None,
    shock_mkt: float,
    shock_smb: float,
    shock_mom: float,
    shock_val: float,
) -> pd.DataFrame:
    """
    Rank current holdings by estimated downside contribution under the selected factor shock.

    Interpretation:
    - market component uses portfolio weight as the holding's market exposure proxy
    - size/value/momentum components use weighted signal contributions when available
    - names with the most negative estimated contribution are the first trim/sell candidates
    """
    if snapshot_df.empty:
        return pd.DataFrame()

    working = snapshot_df.copy()
    working = working.loc[~_cash_like_mask(working)].copy()
    if working.empty:
        return pd.DataFrame()

    working[COL_TICKER] = working[COL_TICKER].astype(str).str.strip().str.upper()
    working[COL_TEAM] = working[COL_TEAM].astype(str).str.strip()
    working[COL_POSITION_SIDE] = working[COL_POSITION_SIDE].astype(str).str.strip().str.upper()

    beta_map = pod_market_betas or {}
    if "weight" in working.columns:
        working["weight"] = pd.to_numeric(working["weight"], errors="coerce").fillna(0.0)
    else:
        working["weight"] = 0.0
    working["pod_beta_mkt"] = working[COL_TEAM].map(beta_map)
    working["pod_beta_mkt"] = pd.to_numeric(working["pod_beta_mkt"], errors="coerce").fillna(1.0)
    working["market_exposure_contribution"] = working["weight"] * working["pod_beta_mkt"]

    working["size_contribution"] = 0.0
    working["momentum_contribution"] = 0.0
    working["value_contribution"] = 0.0

    if not holdings_signals_df.empty:
        signals = holdings_signals_df.copy()
        signals[COL_TICKER] = signals[COL_TICKER].astype(str).str.strip().str.upper()

        merge_cols = [COL_TICKER]
        available_signal_cols = [
            col for col in [
                "size_contribution",
                "momentum_contribution",
                "value_contribution",
            ]
            if col in signals.columns
        ]
        if available_signal_cols:
            working = working.drop(columns=["size_contribution", "momentum_contribution", "value_contribution"], errors="ignore")
            working = working.merge(
                signals[merge_cols + available_signal_cols],
                on=COL_TICKER,
                how="left",
            )
            for col in ["size_contribution", "momentum_contribution", "value_contribution"]:
                if col not in working.columns:
                    working[col] = 0.0
                working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    working["scenario_return_estimate"] = (
        working["market_exposure_contribution"] * shock_mkt
        + working["size_contribution"] * shock_smb
        + working["momentum_contribution"] * shock_mom
        + working["value_contribution"] * shock_val
    )

    working["market_value"] = pd.to_numeric(working.get("market_value"), errors="coerce").fillna(0.0)
    working["estimated_pnl_impact"] = working["market_value"] * working["scenario_return_estimate"]

    working["recommended_action"] = "Hold"
    working.loc[working["scenario_return_estimate"] < 0, "recommended_action"] = "Reduce / Sell"

    recommendations = (
        working.sort_values("estimated_pnl_impact", ascending=True)
        .head(5)
        .copy()
    )

    if recommendations.empty:
        return pd.DataFrame()

    return recommendations.rename(
        columns={
            COL_TICKER: "Ticker",
            COL_TEAM: "Pod",
            COL_POSITION_SIDE: "Position Side",
            "market_exposure_contribution": "Market Exposure Proxy",
            "size_contribution": "Size Exposure Contribution",
            "momentum_contribution": "Momentum Exposure Contribution",
            "value_contribution": "Value Exposure Contribution",
            "scenario_return_estimate": "Scenario Return Estimate",
            "estimated_pnl_impact": "Estimated P&L Impact",
            "recommended_action": "Recommended Action",
        }
    )[
        [
            "Ticker",
            "Pod",
            "Position Side",
            "Recommended Action",
            "Scenario Return Estimate",
            "Estimated P&L Impact",
            "Market Exposure Proxy",
            "Size Exposure Contribution",
            "Momentum Exposure Contribution",
            "Value Exposure Contribution",
        ]
    ]

def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):.2%}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _format_number(value, decimals: int = 2):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):,.{decimals}f}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _render_na_reason(default_message: str, reason: str | None = None) -> None:
    if reason:
        st.info(reason)
    else:
        st.info(default_message)


def _get_team_color(team_name: str) -> str:
    return TEAM_COLORS.get(str(team_name).strip(), "#7A82AB")


def _render_risk_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="risk-dashboard-card">
            <div class="risk-dashboard-card-label">{label}</div>
            <div class="risk-dashboard-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stSelectbox"] {
            max-width: 230px;
        }

        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
            background: linear-gradient(135deg, rgba(20, 52, 110, 0.94), rgba(29, 78, 216, 0.9));
            border: 1px solid rgba(96, 165, 250, 0.34);
            border-radius: 12px;
            min-height: 2.6rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.16);
        }

        div[data-testid="stSelectbox"] [role="combobox"] {
            color: #E2E8F0;
            font-weight: 600;
        }

        .risk-dashboard-card {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.14), rgba(59, 130, 246, 0.18));
            border: 1px solid rgba(14, 116, 144, 0.22);
            border-radius: 16px;
            color: inherit;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .risk-dashboard-card-label {
            color: inherit;
            opacity: 0.8;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }

        .risk-dashboard-card-value {
            color: inherit;
            font-size: 1.65rem;
            font-weight: 600;
            line-height: 1.2;
        }

        html[data-theme="dark"] .risk-dashboard-card-label,
        html[data-theme="dark"] .risk-dashboard-card-value,
        body[data-theme="dark"] .risk-dashboard-card-label,
        body[data-theme="dark"] .risk-dashboard-card-value,
        [data-testid="stAppViewContainer"][data-theme="dark"] .risk-dashboard-card-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .risk-dashboard-card-value {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Risk Engine")


def _compute_batting_average(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean > 0).mean())


def _compute_win_loss_ratio(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return None
    wins = clean.loc[clean > 0]
    losses = clean.loc[clean < 0]
    if wins.empty:
        return 0.0
    if losses.empty:
        return None
    avg_loss = float(losses.mean())
    if avg_loss == 0:
        return None
    return float(float(wins.mean()) / abs(avg_loss))


def _compute_historical_var_95(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(0.05))


def _compute_one_year_max_drawdown(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return None
    cumulative = (1.0 + clean).cumprod()
    trailing = cumulative.iloc[-252:] if len(cumulative) >= 252 else cumulative
    if trailing.empty:
        return None
    drawdown = trailing / trailing.cummax() - 1.0
    return float(drawdown.min()) if not drawdown.empty else None


def _compute_skew(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if len(clean) < 60:
        return None
    return float(clean.skew())


def _compute_tracking_error(active_returns: pd.Series) -> float | None:
    clean = pd.to_numeric(active_returns, errors="coerce").dropna()
    if len(clean) < 60:
        return None
    return float(clean.std(ddof=1) * (252 ** 0.5))


def _compute_information_ratio(active_returns: pd.Series) -> float | None:
    clean = pd.to_numeric(active_returns, errors="coerce").dropna()
    if len(clean) < 60:
        return None
    tracking_error = _compute_tracking_error(clean)
    if tracking_error is None or tracking_error == 0:
        return None
    return float(clean.mean() * 252 / tracking_error)


def _compute_calmar_ratio(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if len(clean) < 60:
        return None
    cumulative = (1.0 + clean).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    if pd.isna(max_dd) or max_dd == 0:
        return None
    annual_return = float(clean.mean() * 252)
    return float(annual_return / abs(max_dd))


def _is_cash_like_row(row: pd.Series) -> bool:
    ticker = str(row.get(COL_TICKER, "")).strip().upper()
    team = str(row.get(COL_TEAM, "")).strip().upper()
    position_side = str(row.get(COL_POSITION_SIDE, "")).strip().upper()
    return ticker in {"CASH", "EUR", "GBP", "NOGXX"} or team == "CASH" or position_side == "CASH"


def _cash_like_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    ticker = df.get(COL_TICKER, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    team = df.get(COL_TEAM, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    position_side = df.get(COL_POSITION_SIDE, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    return ticker.isin({"CASH", "EUR", "GBP", "NOGXX"}) | team.eq("CASH") | position_side.eq("CASH")


def _compute_position_value(df: pd.DataFrame, price_map: dict[str, float]) -> float:
    if df.empty:
        return 0.0
    working = df.copy()
    working[COL_SHARES] = pd.to_numeric(working[COL_SHARES], errors="coerce").fillna(0.0)
    ticker = working.get(COL_TICKER, pd.Series("", index=working.index)).astype(str).str.strip().str.upper()
    position_side = working.get(COL_POSITION_SIDE, pd.Series("", index=working.index)).astype(str).str.strip().str.upper()
    cash_mask = _cash_like_mask(working)
    cash_total = float(working.loc[cash_mask, COL_SHARES].sum()) if cash_mask.any() else 0.0
    investable_mask = ~cash_mask
    if not investable_mask.any():
        return cash_total
    market_values = working.loc[investable_mask, COL_SHARES] * ticker.loc[investable_mask].map(price_map)
    market_values = market_values.where(position_side.loc[investable_mask].ne("SHORT"), -market_values)
    return cash_total + float(pd.to_numeric(market_values, errors="coerce").fillna(0.0).sum())


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


def _build_price_matrix(raw_price_history: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    if len(dates) == 0:
        return pd.DataFrame()
    if raw_price_history.empty:
        return pd.DataFrame(index=dates)

    prices = raw_price_history.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.strip().str.upper()
    prices["close"] = pd.to_numeric(prices.get("close"), errors="coerce")
    prices["adj_close"] = pd.to_numeric(prices.get("adj_close"), errors="coerce")
    prices["px"] = prices["adj_close"]
    missing_mask = prices["px"].isna()
    prices.loc[missing_mask, "px"] = prices.loc[missing_mask, "close"]
    prices = prices.dropna(subset=["date", "ticker", "px"]).copy()
    if prices.empty:
        return pd.DataFrame(index=dates)

    return (
        prices.pivot_table(index="date", columns="ticker", values="px", aggfunc="last")
        .sort_index()
        .reindex(dates)
        .ffill()
    )


def _build_team_history(team: str, snapshots_df: pd.DataFrame) -> pd.DataFrame:
    if snapshots_df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow"])

    with session_scope() as session:
        trades_df = load_trade_receipts(session, team=team)
        cash_df = load_cash_ledger(session, team=team)

    df = snapshots_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df[COL_TEAM] = df[COL_TEAM].astype(str).str.strip()
    df[COL_TICKER] = df[COL_TICKER].astype(str).str.strip().str.upper()
    df[COL_POSITION_SIDE] = df[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    df[COL_SHARES] = pd.to_numeric(df[COL_SHARES], errors="coerce")
    df = df.dropna(subset=["snapshot_date", COL_TICKER, COL_SHARES]).copy()
    df = df.loc[df[COL_TEAM] == team].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow"])

    today = pd.Timestamp.today().normalize()
    oldest_snapshot_date = df["snapshot_date"].min().normalize()
    start_date = max(oldest_snapshot_date, today - pd.Timedelta(days=372))
    business_dates = pd.bdate_range(start=start_date, end=today)
    if len(business_dates) == 0:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow"])

    ticker_normalization = {"BRKB": "BRK-B", "BRKA": "BRK-A"}
    df[COL_TICKER] = df[COL_TICKER].replace(ticker_normalization)

    trades = trades_df.copy()
    if not trades.empty:
        trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce")
        trades[COL_TICKER] = trades[COL_TICKER].astype(str).str.strip().str.upper().replace(ticker_normalization)
        trades = trades.dropna(subset=["trade_date", COL_TICKER]).copy()

    external_cash = cash_df.copy()
    if not external_cash.empty:
        external_cash["activity_date"] = pd.to_datetime(external_cash["activity_date"], errors="coerce")
        external_cash["activity_type"] = external_cash["activity_type"].astype(str).str.strip().str.upper()
        external_cash["amount"] = pd.to_numeric(external_cash["amount"], errors="coerce").fillna(0.0)
        external_cash = external_cash.loc[
            external_cash["activity_type"].isin(EXTERNAL_FLOW_ACTIVITY_TYPES)
        ].dropna(subset=["activity_date"]).copy()
        if not external_cash.empty:
            cash_dates = external_cash["activity_date"].dt.normalize()
            aligned_idx = business_dates.searchsorted(cash_dates)
            valid_mask = aligned_idx < len(business_dates)
            external_cash = external_cash.loc[valid_mask].copy()
            if not external_cash.empty:
                external_cash["flow_date"] = business_dates.take(aligned_idx[valid_mask]).normalize()

    investable_tickers = df.loc[~_cash_like_mask(df), COL_TICKER].dropna().unique().tolist()
    trade_tickers = trades[COL_TICKER].dropna().unique().tolist() if not trades.empty else []
    history_tickers = sorted(set(investable_tickers + trade_tickers))
    raw_price_history = fetch_multiple_price_histories(history_tickers, lookback_days=(today - start_date).days + 30)
    price_matrix = _build_price_matrix(raw_price_history, business_dates)

    snapshot_by_date = {dt.normalize(): grp.copy() for dt, grp in df.groupby("snapshot_date")}
    snapshot_dates = sorted(snapshot_by_date.keys())
    trades_by_date = {dt.normalize(): grp.copy() for dt, grp in trades.groupby(trades["trade_date"].dt.normalize())} if not trades.empty else {}
    cash_by_date = {dt.normalize(): grp.copy() for dt, grp in external_cash.groupby("flow_date")} if not external_cash.empty else {}

    rows = []
    active_positions_df: pd.DataFrame | None = None

    for dt in business_dates:
        price_map = {}
        if dt in price_matrix.index:
            row = price_matrix.loc[dt]
            price_map = {
                str(ticker).strip().upper(): float(row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(row[ticker])
            }

        snapshot_for_day = snapshot_by_date.get(dt.normalize())
        if active_positions_df is None:
            eligible = [d for d in snapshot_dates if d <= dt]
            if not eligible:
                continue
            if snapshot_for_day is None:
                active_positions_df = snapshot_by_date[eligible[-1]].copy()

        active_positions_df, net_external_flow, reconciliation_pnl = _transition_positions_for_day(
            active_positions_df=active_positions_df,
            snapshot_for_day=snapshot_for_day,
            trades_today=trades_by_date.get(dt.normalize()),
            cash_today=cash_by_date.get(dt.normalize()),
            price_map=price_map,
        )
        if active_positions_df is None:
            continue

        team_aum = _compute_position_value(active_positions_df, price_map)

        rows.append(
            {
                "date": dt,
                "team_aum": float(team_aum),
                "net_external_flow": net_external_flow,
                "reconciliation_pnl": reconciliation_pnl,
            }
        )

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=settings.price_refresh_interval_seconds)

def get_team_metrics_data() -> dict[str, pd.DataFrame]:
    with session_scope() as session:
        snapshots_df = load_all_portfolio_snapshots(session)

    if snapshots_df.empty:
        return {"team_metrics": pd.DataFrame(), "correlation_matrix": pd.DataFrame(), "pod_market_betas": pd.DataFrame()}

    team_returns: dict[str, pd.Series] = {}
    for team in TEAM_ORDER:
        history_df = _build_team_history(team, snapshots_df)
        if history_df.empty:
            continue
        prepared = prepare_flow_adjusted_history(
            history_df=history_df,
            value_column="team_aum",
            flow_column="net_external_flow",
        )
        returns = pd.to_numeric(prepared.get("performance_return"), errors="coerce")
        dates = pd.to_datetime(prepared.get("date"), errors="coerce")
        series = pd.Series(returns.values, index=dates).dropna()
        if not series.empty:
            team_returns[team] = series

    if not team_returns:
        return {"team_metrics": pd.DataFrame(), "correlation_matrix": pd.DataFrame(), "pod_market_betas": pd.DataFrame()}

    snapshot_df = get_base_snapshot()
    analytics = get_risk_analytics(snapshot_df)
    factor_returns_df = analytics.get("factor_returns", pd.DataFrame())
    if factor_returns_df.empty or "MKT" not in factor_returns_df.columns:
        return {"team_metrics": pd.DataFrame(), "correlation_matrix": pd.DataFrame(), "pod_market_betas": pd.DataFrame()}

    benchmark_returns = pd.Series(
        pd.to_numeric(factor_returns_df["MKT"], errors="coerce").values,
        index=pd.to_datetime(factor_returns_df["date"], errors="coerce"),
        name="benchmark",
    ).dropna()

    returns_df = pd.concat(team_returns, axis=1)
    returns_df = returns_df.join(benchmark_returns.rename("benchmark"), how="inner").dropna()
    team_columns = [col for col in returns_df.columns if col != "benchmark"]
    if returns_df.empty or not team_columns:
        return {"team_metrics": pd.DataFrame(), "correlation_matrix": pd.DataFrame(), "pod_market_betas": pd.DataFrame()}

    rows = []
    beta_rows = []
    for team in team_columns:
        team_series = pd.to_numeric(returns_df[team], errors="coerce").dropna()
        if len(team_series) < 60:
            continue
        active = pd.to_numeric(returns_df[team] - returns_df["benchmark"], errors="coerce").dropna()
        beta_mkt = _compute_series_beta(returns_df[team], returns_df["benchmark"])
        tracking_error = _compute_tracking_error(active)
        information_ratio = _compute_information_ratio(active)
        calmar_ratio = _compute_calmar_ratio(team_series)
        skew = _compute_skew(team_series)
        rows.append(
            {
                "team": team,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio,
                "calmar_ratio": calmar_ratio,
                "skew": skew,
            }
        )
        if beta_mkt is not None:
            beta_rows.append({"team": team, "beta_mkt": beta_mkt})

    team_metrics_df = pd.DataFrame(rows).set_index("team") if rows else pd.DataFrame(columns=["tracking_error", "information_ratio", "calmar_ratio", "skew"])
    pod_market_betas_df = pd.DataFrame(beta_rows).set_index("team") if beta_rows else pd.DataFrame(columns=["beta_mkt"])
    correlation_matrix = returns_df.drop(columns="benchmark", errors="ignore").corr() if len(team_columns) >= 2 else pd.DataFrame()
    return {
        "team_metrics": team_metrics_df,
        "correlation_matrix": correlation_matrix,
        "pod_market_betas": pod_market_betas_df,
    }


def get_fund_metrics(portfolio_returns_df: pd.DataFrame, factor_returns_df: pd.DataFrame) -> dict[str, float | None]:
    if portfolio_returns_df.empty or factor_returns_df.empty:
        return {
            "tracking_error": None,
            "information_ratio": None,
            "calmar_ratio": None,
            "skew": None,
            "batting_average": None,
            "win_loss_ratio": None,
            "historical_var_95": None,
            "max_drawdown_1y": None,
        }

    portfolio = portfolio_returns_df.copy()
    benchmark = factor_returns_df.copy()
    portfolio["date"] = pd.to_datetime(portfolio["date"], errors="coerce")
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    portfolio["portfolio_return"] = pd.to_numeric(portfolio["portfolio_return"], errors="coerce")
    benchmark["MKT"] = pd.to_numeric(benchmark["MKT"], errors="coerce")

    merged = portfolio[["date", "portfolio_return"]].merge(
        benchmark[["date", "MKT"]],
        on="date",
        how="inner",
    ).dropna(subset=["portfolio_return", "MKT"]).sort_values("date").reset_index(drop=True)

    if len(merged) < 60:
        return {
            "tracking_error": None,
            "information_ratio": None,
            "calmar_ratio": None,
            "skew": None,
            "batting_average": None,
            "win_loss_ratio": None,
            "historical_var_95": None,
            "max_drawdown_1y": None,
        }

    returns = pd.to_numeric(merged["portfolio_return"], errors="coerce").dropna()
    active = pd.to_numeric(merged["portfolio_return"] - merged["MKT"], errors="coerce").dropna()

    return {
        "tracking_error": _compute_tracking_error(active),
        "information_ratio": _compute_information_ratio(active),
        "calmar_ratio": _compute_calmar_ratio(returns),
        "skew": _compute_skew(returns),
        "batting_average": _compute_batting_average(returns),
        "win_loss_ratio": _compute_win_loss_ratio(returns),
        "historical_var_95": _compute_historical_var_95(returns),
        "max_drawdown_1y": _compute_one_year_max_drawdown(returns),
    }


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_base_snapshot() -> pd.DataFrame:
    with session_scope() as session:
        position_state_df = load_position_state(session)

    if position_state_df.empty:
        return pd.DataFrame()

    tickers = (
        position_state_df[COL_TICKER]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("CASH")]
        .unique()
        .tolist()
    )
    latest_prices_df, _ = fetch_latest_prices(tickers)
    return build_current_portfolio_snapshot(position_state_df, latest_prices_df)


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_risk_analytics(snapshot_df: pd.DataFrame) -> dict:
    builder = getattr(
        exposure_module,
        "build_factor_analytics_platform",
        getattr(exposure_module, "build_custom_live_factor_model"),
    )
    result = builder(snapshot_df)
    if isinstance(result, dict):
        return result
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    return {}


def render_empty_state() -> None:
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_team_metrics(team_metrics_df: pd.DataFrame) -> None:
    st.subheader("Pod Risk Metrics")
    if team_metrics_df.empty:
        st.info("Pod risk metrics are unavailable.")
        return

    plot_df = team_metrics_df.copy().reset_index()
    plot_df = plot_df.rename(
        columns={
            "team": "Pod",
            "tracking_error": "Tracking Error",
            "information_ratio": "Information Ratio",
            "calmar_ratio": "Calmar Ratio",
            "skew": "Skew",
        }
    )

    top_left, top_right = st.columns(2)
    with top_left:
        if {"Pod", "Tracking Error"}.issubset(plot_df.columns):
            te_fig = px.bar(
                plot_df,
                x="Pod",
                y="Tracking Error",
                color="Pod",
                color_discrete_map={team: _get_team_color(team) for team in plot_df["Pod"].dropna().astype(str).unique()},
                title="TRACKING ERROR BY POD",
            )
            te_fig.update_layout(yaxis_tickformat=".2%")
            te_fig = style_plotly_figure(te_fig, title_text="TRACKING ERROR BY POD")
            te_fig.update_layout(showlegend=False)
            st.plotly_chart(te_fig, use_container_width=True)
    with top_right:
        if {"Pod", "Calmar Ratio"}.issubset(plot_df.columns):
            calmar_fig = px.bar(
                plot_df,
                x="Pod",
                y="Calmar Ratio",
                color="Pod",
                color_discrete_map={team: _get_team_color(team) for team in plot_df["Pod"].dropna().astype(str).unique()},
                title="CALMAR RATIO BY POD",
            )
            calmar_fig = style_plotly_figure(calmar_fig, title_text="CALMAR RATIO BY POD")
            calmar_fig.update_layout(showlegend=False)
            st.plotly_chart(calmar_fig, use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        if {"Pod", "Information Ratio"}.issubset(plot_df.columns):
            ir_fig = px.bar(
                plot_df,
                x="Pod",
                y="Information Ratio",
                color="Pod",
                color_discrete_map={team: _get_team_color(team) for team in plot_df["Pod"].dropna().astype(str).unique()},
                title="INFORMATION RATIO BY POD",
            )
            ir_fig = style_plotly_figure(ir_fig, title_text="INFORMATION RATIO BY POD")
            ir_fig.update_layout(showlegend=False)
            st.plotly_chart(ir_fig, use_container_width=True)
    with bottom_right:
        if {"Pod", "Skew"}.issubset(plot_df.columns):
            skew_fig = px.bar(
                plot_df,
                x="Pod",
                y="Skew",
                color="Pod",
                color_discrete_map={team: _get_team_color(team) for team in plot_df["Pod"].dropna().astype(str).unique()},
                title="SKEW BY POD",
            )
            skew_fig = style_plotly_figure(skew_fig, title_text="SKEW BY POD")
            skew_fig.update_layout(showlegend=False)
            st.plotly_chart(skew_fig, use_container_width=True)

    display_df = plot_df.copy()
    if "Tracking Error" in display_df.columns:
        display_df["Tracking Error"] = display_df["Tracking Error"].map(_format_percent)
    for col in ["Information Ratio", "Calmar Ratio", "Skew"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_format_number)
    with st.expander("Pod Risk Metrics Table"):
        st.dataframe(left_align_dataframe(display_df), use_container_width=True, hide_index=True)


def render_fund_dashboard(fund_metrics: dict[str, float | None]) -> None:
    st.subheader("Fund Risk Dashboard")
    with st.container(border=True):
        row_1 = st.columns(4)
        with row_1[0]:
            _render_risk_metric_card("Tracking Error", _format_percent(fund_metrics.get("tracking_error")))
        with row_1[1]:
            _render_risk_metric_card("Information Ratio", _format_number(fund_metrics.get("information_ratio")))
        with row_1[2]:
            _render_risk_metric_card("Calmar Ratio", _format_number(fund_metrics.get("calmar_ratio")))
        with row_1[3]:
            _render_risk_metric_card("Skew", _format_number(fund_metrics.get("skew")))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        row_2 = st.columns(4)
        with row_2[0]:
            _render_risk_metric_card("Batting Average", _format_percent(fund_metrics.get("batting_average")))
        with row_2[1]:
            _render_risk_metric_card("Win vs Loss Ratio", _format_number(fund_metrics.get("win_loss_ratio")))
        with row_2[2]:
            _render_risk_metric_card("Daily Historical VaR (95%)", _format_percent(fund_metrics.get("historical_var_95")))
        with row_2[3]:
            _render_risk_metric_card("1 Year Max Drawdown", _format_percent(fund_metrics.get("max_drawdown_1y")))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def render_team_correlation_matrix(correlation_matrix_df: pd.DataFrame) -> None:
    if correlation_matrix_df.empty:
        st.info("Cross-pod correlation data is unavailable.")
        return

    fig = px.imshow(
        correlation_matrix_df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="POD RETURN CORRELATION HEATMAP",
    )
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=None,
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.22,
            yanchor="top",
            len=0.7,
        ),
        margin=dict(t=48, b=110),
        height=520,
    )
    fig = style_plotly_figure(fig, title_text="POD RETURN CORRELATION HEATMAP")
    st.plotly_chart(fig, use_container_width=True)


def render_risk_decomposition(risk_decomposition_df: pd.DataFrame, reason: str | None = None) -> None:
    if risk_decomposition_df.empty:
        _render_na_reason("N/A: risk decomposition is unavailable.", reason)
        return

    plot_df = risk_decomposition_df.loc[
        risk_decomposition_df["component"].astype(str).ne("Total")
    ].copy()
    if plot_df.empty:
        return
    if "pct_total_risk" not in plot_df.columns:
        return

    plot_df["pct_total_risk"] = pd.to_numeric(plot_df["pct_total_risk"], errors="coerce")
    plot_df["abs_variance_contrib"] = pd.to_numeric(plot_df.get("abs_variance_contrib"), errors="coerce")
    plot_df = plot_df.dropna(subset=["pct_total_risk"]).copy()
    plot_df = plot_df.loc[
        plot_df["component"].astype(str).isin(["MKT", "SMB", "MOM", "VAL", "Idiosyncratic"])
    ].copy()
    if plot_df.empty:
        return
    plot_df["component"] = plot_df["component"].replace(
        {
            "MKT": "Market",
            "SMB": "Size",
            "MOM": "Momentum",
            "VAL": "Value",
        }
    )

    fig = px.pie(
        plot_df,
        names="component",
        values="pct_total_risk",
        title="PORTFOLIO RISK CONTRIBUTION",
        hole=0.35,
        color="component",
        color_discrete_map=FACTOR_COLORS,
    )
    fig.update_traces(
        texttemplate="%{label}<br>%{percent}",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Percent of Total Risk: %{value:.2%}<br>"
            "Share of Pie: %{percent}<br>"
            "Absolute Contribution: %{customdata[0]:.2%}<extra></extra>"
        ),
        customdata=plot_df[["abs_variance_contrib"]].fillna(0.0).to_numpy(),
    )
    fig.update_layout(
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        margin=dict(t=48, b=110),
        height=520,
    )
    fig = style_plotly_figure(fig, title_text="PORTFOLIO RISK CONTRIBUTION")
    st.plotly_chart(fig, use_container_width=True)


def _build_scenario_pod_returns(
    snapshot_df: pd.DataFrame,
    holdings_signals_df: pd.DataFrame,
    pod_market_betas: dict[str, float] | None,
    shock_mkt: float,
    shock_smb: float,
    shock_mom: float,
    shock_val: float,
) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["Pod", "Expected Return"])

    working = snapshot_df.copy()
    working = working.loc[~_cash_like_mask(working)].copy()
    if working.empty:
        return pd.DataFrame(columns=["Pod", "Expected Return"])

    working[COL_TICKER] = working[COL_TICKER].astype(str).str.strip().str.upper()
    working[COL_TEAM] = working[COL_TEAM].astype(str).str.strip()
    working["market_value"] = pd.to_numeric(working.get("market_value"), errors="coerce").fillna(0.0)
    working["weight"] = pd.to_numeric(working.get("weight"), errors="coerce").fillna(0.0)

    beta_map = pod_market_betas or {}
    working["pod_beta_mkt"] = working[COL_TEAM].map(beta_map)
    working["pod_beta_mkt"] = pd.to_numeric(working["pod_beta_mkt"], errors="coerce").fillna(1.0)
    working["market_exposure_contribution"] = working["weight"] * working["pod_beta_mkt"]
    working["size_contribution"] = 0.0
    working["momentum_contribution"] = 0.0
    working["value_contribution"] = 0.0

    if not holdings_signals_df.empty:
        signals = holdings_signals_df.copy()
        signals[COL_TICKER] = signals[COL_TICKER].astype(str).str.strip().str.upper()

        merge_cols = [COL_TICKER]
        factor_cols = [col for col in ["size_contribution", "momentum_contribution", "value_contribution"] if col in signals.columns]
        if factor_cols:
            working = working.drop(columns=["size_contribution", "momentum_contribution", "value_contribution"], errors="ignore")
            working = working.merge(
                signals[merge_cols + factor_cols],
                on=COL_TICKER,
                how="left",
            )
            for col in ["size_contribution", "momentum_contribution", "value_contribution"]:
                if col not in working.columns:
                    working[col] = 0.0
                working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)

    working["scenario_return_estimate"] = (
        working["market_exposure_contribution"] * shock_mkt
        + working["size_contribution"] * shock_smb
        + working["momentum_contribution"] * shock_mom
        + working["value_contribution"] * shock_val
    )

    pod_df = (
        working.groupby(COL_TEAM, dropna=False)
        .agg(
            pod_weight=("weight", "sum"),
            pod_beta_mkt=("pod_beta_mkt", "last"),
            portfolio_return_contribution=("scenario_return_estimate", "sum"),
        )
        .reset_index()
    )
    pod_df["pod_weight"] = pd.to_numeric(pod_df["pod_weight"], errors="coerce").fillna(0.0)
    pod_df["portfolio_return_contribution"] = pd.to_numeric(
        pod_df["portfolio_return_contribution"],
        errors="coerce",
    ).fillna(0.0)
    pod_df["Expected Return"] = 0.0
    valid_weight = pod_df["pod_weight"].abs() > 1e-12
    pod_df.loc[valid_weight, "Expected Return"] = (
        pod_df.loc[valid_weight, "portfolio_return_contribution"] / pod_df.loc[valid_weight, "pod_weight"]
    )

    pod_df = pod_df.rename(columns={COL_TEAM: "Pod"})
    pod_df = pod_df[["Pod", "Expected Return", "pod_beta_mkt"]].sort_values("Expected Return", ascending=True).reset_index(drop=True)
    return pod_df

def render_scenario_pod_heatmap(pod_returns_df: pd.DataFrame) -> None:
    if pod_returns_df.empty:
        st.info("Pod scenario return estimates are unavailable.")
        return

    plot_df = pod_returns_df.copy()
    plot_df["Expected Return"] = pd.to_numeric(plot_df["Expected Return"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Expected Return"]).copy()
    if plot_df.empty:
        st.info("Pod scenario return estimates are unavailable.")
        return

    heatmap_df = plot_df.set_index("Pod")[["Expected Return"]].T
    heatmap_df.index = ["Scenario"]
    text_df = heatmap_df.copy().map(_format_percent)

    fig = px.imshow(
        heatmap_df,
        color_continuous_scale="RdBu",
        aspect="auto",
        title="EXPECTED RETURN BY POD",
        zmin=-0.10,
        zmax=0.10,
    )
    fig.update_traces(
        text=text_df.to_numpy(),
        texttemplate="%{text}",
        textfont=dict(size=16),
    )
    fig.update_layout(
        margin=dict(t=80, b=20),
        height=260,
    )
    fig.update_coloraxes(showscale=False, cmid=0.0)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="", showticklabels=False)
    fig = style_plotly_figure(fig, title_text="EXPECTED RETURN BY POD")
    st.plotly_chart(fig, use_container_width=True)

def _compute_scenario_hero_metrics(
    snapshot_df: pd.DataFrame,
    expected_portfolio_return: float,
    shock_mkt: float,
) -> dict[str, float | None]:
    total_market_value = 0.0
    if not snapshot_df.empty and "market_value" in snapshot_df.columns:
        working = snapshot_df.copy()
        working["market_value"] = pd.to_numeric(working["market_value"], errors="coerce").fillna(0.0)
        total_market_value = float(working["market_value"].sum())

    expected_portfolio_pnl = total_market_value * expected_portfolio_return if total_market_value else None
    expected_spy_return = shock_mkt
    active_return_vs_spy = expected_portfolio_return - expected_spy_return

    return {
        "expected_portfolio_return": expected_portfolio_return,
        "expected_portfolio_pnl": expected_portfolio_pnl,
        "expected_spy_return": expected_spy_return,
        "active_return_vs_spy": active_return_vs_spy,
    }

def render_scenario_hero_metrics(hero_metrics: dict[str, float | None]) -> None:
    with st.container(border=True):
        row = st.columns(4)

        with row[0]:
            _render_risk_metric_card(
                "Expected Portfolio Return",
                _format_percent(hero_metrics.get("expected_portfolio_return")),
            )

        with row[1]:
            _render_risk_metric_card(
                "Expected Portfolio P&L",
                _format_currency(hero_metrics.get("expected_portfolio_pnl")),
            )

        with row[2]:
            _render_risk_metric_card(
                "Expected S&P 500 Return",
                _format_percent(hero_metrics.get("expected_spy_return")),
            )

        with row[3]:
            _render_risk_metric_card(
                "Active Return vs S&P 500",
                _format_percent(hero_metrics.get("active_return_vs_spy")),
            )

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

def render_scenario_analysis(
    rolling_betas_df: pd.DataFrame,
    scenario_template_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    holdings_signals_df: pd.DataFrame,
    pod_market_betas_df: pd.DataFrame,
    reason: str | None = None,
) -> None:
    st.subheader("Scenario Library")

    latest_loadings = _extract_latest_factor_loadings(rolling_betas_df)
    if latest_loadings is None:
        _render_na_reason(
            "N/A: scenario analysis uses the latest portfolio factor betas, which are not available yet.",
            reason,
        )
        return

    scenario_names = list(SCENARIO_LIBRARY.keys())
    selector_col, _ = st.columns([0.3, 0.7])
    with selector_col:
        selected_scenario = st.selectbox(
            "Select Scenario",
            options=scenario_names,
            index=0,
            key="risk_engine_scenario_library",
            label_visibility="collapsed",
        )

    scenario = SCENARIO_LIBRARY[selected_scenario]
    st.caption(scenario.get("description", ""))

    use_custom = selected_scenario == "Custom"

    # Use scenario values directly unless user is in Custom mode
    scenario_mkt = float(scenario.get("shock_mkt", 0.0))
    scenario_smb = float(scenario.get("shock_smb", 0.0))
    scenario_mom = float(scenario.get("shock_mom", 0.0))
    scenario_val = float(scenario.get("shock_val", 0.0))

    col1, col2, col3, col4 = st.columns(4)
    custom_mkt_pct = col1.number_input(
        "Market Shock (%)",
        value=scenario_mkt * 100.0,
        step=0.25,
        format="%.2f",
        disabled=not use_custom,
        key="risk_engine_shock_mkt",
    )
    custom_smb_pct = col2.number_input(
        "Size Shock (%)",
        value=scenario_smb * 100.0,
        step=0.25,
        format="%.2f",
        disabled=not use_custom,
        key="risk_engine_shock_smb",
    )
    custom_mom_pct = col3.number_input(
        "Momentum Shock (%)",
        value=scenario_mom * 100.0,
        step=0.25,
        format="%.2f",
        disabled=not use_custom,
        key="risk_engine_shock_mom",
    )
    custom_val_pct = col4.number_input(
        "Value Shock (%)",
        value=scenario_val * 100.0,
        step=0.25,
        format="%.2f",
        disabled=not use_custom,
        key="risk_engine_shock_val",
    )

    if use_custom:
        shock_mkt = float(custom_mkt_pct) / 100.0
        shock_smb = float(custom_smb_pct) / 100.0
        shock_mom = float(custom_mom_pct) / 100.0
        shock_val = float(custom_val_pct) / 100.0
    else:
        shock_mkt = scenario_mkt
        shock_smb = scenario_smb
        shock_mom = scenario_mom
        shock_val = scenario_val

    pod_market_betas = (
        pd.to_numeric(pod_market_betas_df.get("beta_mkt"), errors="coerce").dropna().to_dict()
        if not pod_market_betas_df.empty else {}
    )

    expected_portfolio_return = (
        latest_loadings["beta_mkt"] * shock_mkt
        + latest_loadings["beta_smb"] * shock_smb
        + latest_loadings["beta_mom"] * shock_mom
        + latest_loadings["beta_val"] * shock_val
    )

    # One box only
    hero_metrics = _compute_scenario_hero_metrics(
        snapshot_df=snapshot_df,
        expected_portfolio_return=expected_portfolio_return,
        shock_mkt=shock_mkt,
    )

    render_scenario_hero_metrics(hero_metrics)

    pod_returns_df = _build_scenario_pod_returns(
        snapshot_df=snapshot_df,
        holdings_signals_df=holdings_signals_df,
        pod_market_betas=pod_market_betas,
        shock_mkt=shock_mkt,
        shock_smb=shock_smb,
        shock_mom=shock_mom,
        shock_val=shock_val,
    )

    render_scenario_pod_heatmap(pod_returns_df)

    recommendations_df = _build_scenario_trade_recommendations(
        snapshot_df=snapshot_df,
        holdings_signals_df=holdings_signals_df,
        pod_market_betas=pod_market_betas,
        shock_mkt=shock_mkt,
        shock_smb=shock_smb,
        shock_mom=shock_mom,
        shock_val=shock_val,
    )

    st.markdown("**Top 5 Recommended Trades**")
    if recommendations_df.empty:
        st.info("No scenario-based trade recommendations are available.")
        return

    display_rec_df = recommendations_df.copy()
    for col in [
        "Scenario Return Estimate",
        "Market Exposure Proxy",
        "Size Exposure Contribution",
        "Momentum Exposure Contribution",
        "Value Exposure Contribution",
    ]:
        if col in display_rec_df.columns:
            display_rec_df[col] = display_rec_df[col].map(_format_percent)
    if "Estimated P&L Impact" in display_rec_df.columns:
        display_rec_df["Estimated P&L Impact"] = display_rec_df["Estimated P&L Impact"].map(_format_currency)

    st.dataframe(
        left_align_dataframe(display_rec_df),
        use_container_width=True,
        hide_index=True,
    )


def render_drawdown_chart(attribution_df: pd.DataFrame) -> None:
    st.subheader("Factor Drawdown Analysis")
    if attribution_df.empty:
        st.info("Factor drawdown decomposition is unavailable.")
        return

    plot_df = attribution_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if plot_df.empty:
        st.info("Factor drawdown decomposition is unavailable.")
        return

    contribution_cols = [
        "contribution_mkt",
        "contribution_smb",
        "contribution_mom",
        "contribution_val",
        "residual",
    ]
    required_cols = ["portfolio_return", *contribution_cols]
    missing_cols = [col for col in required_cols if col not in plot_df.columns]
    if missing_cols:
        st.info(
            "Factor drawdown decomposition is unavailable because these required columns are missing: "
            + ", ".join(missing_cols)
        )
        return

    plot_df["portfolio_return"] = pd.to_numeric(plot_df["portfolio_return"], errors="coerce").fillna(0.0)
    for col in contribution_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)

    plot_df["cum_portfolio"] = (1.0 + plot_df["portfolio_return"]).cumprod()
    plot_df["running_hwm"] = plot_df["cum_portfolio"].cummax()
    plot_df["portfolio_drawdown"] = plot_df["cum_portfolio"] / plot_df["running_hwm"] - 1.0
    plot_df["is_hwm"] = plot_df["portfolio_drawdown"].abs() <= 1e-12
    plot_df.loc[plot_df["is_hwm"], "portfolio_drawdown"] = 0.0
    plot_df["drawdown_episode"] = plot_df["is_hwm"].cumsum()

    resettable_cols = {
        "contribution_mkt": "Market",
        "contribution_smb": "Size",
        "contribution_mom": "Momentum",
        "contribution_val": "Value",
        "residual": "Idiosyncratic",
    }
    for col in resettable_cols:
        active_flow = plot_df[col].where(~plot_df["is_hwm"], 0.0)
        plot_df[f"{col}_since_hwm"] = active_flow.groupby(plot_df["drawdown_episode"]).cumsum()
        plot_df.loc[plot_df["is_hwm"], f"{col}_since_hwm"] = 0.0

    plot_df["factor_sum_since_hwm"] = plot_df[[f"{col}_since_hwm" for col in resettable_cols]].sum(axis=1)
    reconcile_error = (plot_df["factor_sum_since_hwm"] - plot_df["portfolio_drawdown"]).abs()
    max_reconcile_error = float(reconcile_error.max()) if not reconcile_error.empty else 0.0

    start_date = plot_df["date"].max() - pd.Timedelta(days=365)
    plot_df = plot_df.loc[plot_df["date"] >= start_date].copy()
    if plot_df.empty:
        st.info("Factor drawdown decomposition is unavailable for the last year.")
        return

    if max_reconcile_error > 0.02:
        st.warning(
            "Drawdown attribution only approximately reconciles to the underwater curve. "
            f"Maximum episode gap over the full history is {max_reconcile_error:.2%}."
        )

    component_map = {
        "contribution_val_since_hwm": "Value",
        "contribution_mkt_since_hwm": "Market",
        "residual_since_hwm": "Idiosyncratic",
        "contribution_smb_since_hwm": "Size",
        "contribution_mom_since_hwm": "Momentum",
    }
    fig = go.Figure()
    for column, label in component_map.items():
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df[column],
                mode="lines",
                name=label,
                line=dict(width=2, color=FACTOR_COLORS[label]),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["portfolio_drawdown"],
            mode="lines",
            name="Portfolio Drawdown",
            line=dict(color="#64748B", width=4),
        )
    )
    fig.update_layout(
        title=dict(text="Drawdown Attribution Since Last High Water Mark", x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="CUMULATIVE CONTRIBUTION SINCE LAST HIGH WATER MARK",
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=90, b=90),
        hovermode="x unified",
    )
    fig = style_plotly_figure(fig, title_text="Drawdown Attribution Since Last High Water Mark")
    st.plotly_chart(fig, use_container_width=True)


def render_notes(notes: list[str]) -> None:
    with st.expander("Methodology Notes"):
        for note in notes:
            st.write(f"- {note}")


def main() -> None:
    st.set_page_config(page_title="Risk Engine", layout="wide")
    apply_app_theme()
    render_top_nav()
    render_header()

    snapshot_df = get_base_snapshot()
    if snapshot_df.empty:
        render_empty_state()
        return

    analytics = get_risk_analytics(snapshot_df)
    team_metrics_data = get_team_metrics_data()
    fund_metrics = get_fund_metrics(
        analytics.get("portfolio_returns", pd.DataFrame()),
        analytics.get("factor_returns", pd.DataFrame()),
    )

    render_fund_dashboard(fund_metrics)
    st.divider()
    render_team_metrics(team_metrics_data.get("team_metrics", pd.DataFrame()))
    st.divider()
    st.subheader("Risk Architecture")
    correlation_col, risk_col = st.columns(2)
    with correlation_col:
        render_team_correlation_matrix(team_metrics_data.get("correlation_matrix", pd.DataFrame()))
    with risk_col:
        render_risk_decomposition(
            analytics.get("risk_decomposition", pd.DataFrame()),
            analytics.get("risk_decomposition_reason"),
        )
    st.divider()
    render_scenario_analysis(
        analytics.get("rolling_betas", pd.DataFrame()),
        analytics.get("scenario_template", pd.DataFrame()),
        snapshot_df,
        analytics.get("holdings_signals", {}).get("weighted_signal_contributions", pd.DataFrame()),
        team_metrics_data.get("pod_market_betas", pd.DataFrame()),
        analytics.get("portfolio_factor_betas", {}).get("reason"),
    )
    st.divider()
    render_drawdown_chart(analytics.get("attribution", pd.DataFrame()))
    render_notes(analytics.get("notes", []))


if __name__ == "__main__":
    main()
