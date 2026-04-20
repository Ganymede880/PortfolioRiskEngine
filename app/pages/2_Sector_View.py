"""
Sector View dashboard page for the CMCSIF Portfolio Tracker.

This page mirrors the Portfolio View styling while focusing on a single pod:
- top sleeve summary metrics
- detailed current holdings table with trailing returns
- allocation and P&L contribution charts
- weekly sleeve AUM vs sector benchmark
"""

from __future__ import annotations

import html
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
from src.analytics.performance import (
    build_flow_adjusted_benchmark_series,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    prepare_flow_adjusted_history,
)
from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import (
    load_all_portfolio_snapshots,
    load_cash_ledger,
    load_position_state,
    load_trade_receipts,
)
from src.db.session import session_scope
from src.analytics.ledger import apply_cash_ledger_entries_to_positions, apply_trades_to_positions
from src.utils.constants import FACTOR_COLORS as SHARED_FACTOR_COLORS
from src.utils.ui import apply_app_theme, left_align_dataframe, render_page_title, render_top_nav



COL_DATE = "as_of_date"
COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
COL_PRICE = "price"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"
COL_COST_BASIS = "cost_basis_per_share"

TEAM_BENCHMARK_TICKERS = {
    "Consumer": "XLY",
    "E&U": "XLU",
    "F&R": "XLF",
    "Healthcare": "XLV",
    "TMT": "XLK",
    "M&I": "XLI",
}

TEAM_COLORS = {
    "Consumer": "#C6D4FF",
    "E&U": "#7A82AB",
    "F&R": "#307473",
    "Healthcare": "#12664F",
    "TMT": "#2DC2BD",
    "M&I": "#3F3047",
    "Cash": "#7A82AB",
}
FACTOR_COLORS = {
    "Benchmark": SHARED_FACTOR_COLORS["Market"],
    "Size": SHARED_FACTOR_COLORS["Size"],
    "Momentum": SHARED_FACTOR_COLORS["Momentum"],
    "Value": SHARED_FACTOR_COLORS["Value"],
    "Idiosyncratic": SHARED_FACTOR_COLORS["Idiosyncratic"],
}
EXTERNAL_FLOW_ACTIVITY_TYPES = {"SECTOR_REBALANCE", "PORTFOLIO_LIQUIDATION"}


def _get_team_color(team_name: str) -> str:
    return TEAM_COLORS.get(str(team_name).strip(), "#7A82AB")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = str(hex_color).strip().lstrip("#")
    if len(cleaned) != 6:
        return 122, 130, 171
    return tuple(int(cleaned[idx: idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = [max(0, min(255, int(channel))) for channel in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    clamped_alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {clamped_alpha:.2f})"


def _blend_hex(hex_color: str, target_rgb: tuple[int, int, int], weight: float) -> str:
    base_rgb = _hex_to_rgb(hex_color)
    clamped_weight = max(0.0, min(1.0, weight))
    blended = tuple(
        round(base_channel * (1.0 - clamped_weight) + target_channel * clamped_weight)
        for base_channel, target_channel in zip(base_rgb, target_rgb)
    )
    return _rgb_to_hex(blended)


def _build_tonal_palette(base_color: str, count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return [base_color]
    weights = [0.0, 0.16, 0.3, 0.42, 0.22, 0.5, 0.36, 0.58, 0.46, 0.64]
    return [
        _blend_hex(base_color, (255, 255, 255), weights[idx % len(weights)])
        for idx in range(count)
    ]


def _format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    return f"(${abs(numeric_value):,.2f})" if numeric_value < 0 else f"${numeric_value:,.2f}"


def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):.2%}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):,.2f}"
    return f"({formatted})" if numeric_value < 0 else formatted


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

    avg_win = float(wins.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0:
        return None
    return float(avg_win / abs(avg_loss))


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

    running_max = trailing.cummax()
    drawdown = trailing / running_max - 1.0
    if drawdown.empty:
        return None
    return float(drawdown.min())


def _apply_team_page_theme() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.8rem;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            color: rgba(226, 232, 240, 0.74);
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #DBEAFE;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: linear-gradient(135deg, rgba(20, 52, 110, 0.98), rgba(29, 78, 216, 0.94)) !important;
            height: 0.2rem !important;
            border-radius: 999px !important;
        }

        .pod-summary-card {
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

        .pod-summary-card-label {
            color: inherit;
            opacity: 0.8;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }

        .pod-summary-card-value {
            color: inherit;
            font-size: 1.65rem;
            font-weight: 600;
            line-height: 1.2;
        }

        html[data-theme="dark"] .pod-summary-card-label,
        html[data-theme="dark"] .pod-summary-card-value,
        body[data-theme="dark"] .pod-summary-card-label,
        body[data-theme="dark"] .pod-summary-card-value,
        [data-testid="stAppViewContainer"][data-theme="dark"] .pod-summary-card-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .pod-summary-card-value {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return None


def _render_pod_summary_card(label: str, value: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="pod-summary-card">
            <div class="pod-summary-card-label">{safe_label}</div>
            <div class="pod-summary-card-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prepare_team_performance_history(history_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_flow_adjusted_history(
        history_df=history_df,
        value_column="team_aum",
        flow_column="net_external_flow",
    )


def _is_cash_like_row(row: pd.Series) -> bool:
    ticker = str(row.get(COL_TICKER, "")).strip().upper()
    team = str(row.get(COL_TEAM, "")).strip().upper()
    position_side = str(row.get(COL_POSITION_SIDE, "")).strip().upper()
    return (
        ticker in {"CASH", "EUR", "GBP", "NOGXX"}
        or team == "CASH"
        or position_side == "CASH"
    )


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


def _build_return_series(history_df: pd.DataFrame, value_column: str) -> pd.Series:
    prepared = _prepare_team_performance_history(history_df)
    if prepared.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(prepared["performance_return"], errors="coerce").dropna()


def _compute_team_turnover(history_df: pd.DataFrame, snapshots_df: pd.DataFrame, team: str) -> float | None:
    if history_df.empty:
        return None

    hist = history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["team_aum"] = pd.to_numeric(hist["team_aum"], errors="coerce")
    hist = hist.dropna(subset=["date", "team_aum"]).sort_values("date").reset_index(drop=True)
    if hist.empty:
        return None

    end_ts = hist["date"].iloc[-1].normalize()
    start_ts = end_ts - pd.Timedelta(days=365)
    trailing_history = hist.loc[hist["date"] >= start_ts].copy()
    average_aum = float(trailing_history["team_aum"].mean()) if not trailing_history.empty else 0.0
    if average_aum == 0:
        return None

    snapshots = snapshots_df.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], errors="coerce")
    snapshots[COL_TICKER] = snapshots[COL_TICKER].astype(str).str.strip().str.upper()
    snapshots[COL_TEAM] = snapshots[COL_TEAM].astype(str).str.strip()
    snapshots[COL_POSITION_SIDE] = snapshots[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    snapshots[COL_SHARES] = pd.to_numeric(snapshots[COL_SHARES], errors="coerce")
    snapshots = snapshots.dropna(subset=["snapshot_date", COL_TICKER, COL_POSITION_SIDE, COL_SHARES]).copy()
    snapshots = snapshots.loc[snapshots[COL_TEAM] == team].copy()
    snapshots = snapshots.loc[~_cash_like_mask(snapshots)].copy()
    if snapshots.empty:
        return 0.0

    snapshot_dates = sorted(snapshots["snapshot_date"].dt.normalize().unique().tolist())
    transition_dates = [dt for dt in snapshot_dates if dt >= start_ts]
    if not transition_dates:
        return 0.0

    pricing_start = min(max([dt for dt in snapshot_dates if dt < transition_dates[0]], default=transition_dates[0]), transition_dates[0])
    pricing_end = snapshot_dates[-1]
    tickers = sorted(snapshots[COL_TICKER].dropna().unique().tolist())
    raw_price_history = fetch_multiple_price_histories(
        tickers=tickers,
        lookback_days=max((pricing_end - pricing_start).days + 30, 30),
    )
    price_matrix = _build_price_matrix(raw_price_history, pd.date_range(start=pricing_start, end=pricing_end, freq="D"))
    snapshot_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in snapshots.groupby(snapshots["snapshot_date"].dt.normalize())
    }

    replaced_value = 0.0
    key_cols = [COL_TICKER, COL_POSITION_SIDE]

    for current_date in transition_dates:
        prior_dates = [dt for dt in snapshot_dates if dt < current_date]
        if not prior_dates:
            continue

        previous_date = prior_dates[-1]
        previous_df = snapshot_by_date.get(previous_date, pd.DataFrame()).copy()
        current_df = snapshot_by_date.get(current_date, pd.DataFrame()).copy()
        if previous_df.empty or current_df.empty:
            continue

        previous_keys = set(map(tuple, previous_df[key_cols].itertuples(index=False, name=None)))
        current_keys = set(map(tuple, current_df[key_cols].itertuples(index=False, name=None)))
        removed_keys = previous_keys - current_keys
        added_keys = current_keys - previous_keys
        if not removed_keys and not added_keys:
            continue

        removed_df = previous_df.loc[previous_df[key_cols].apply(tuple, axis=1).isin(removed_keys)].copy()
        added_df = current_df.loc[current_df[key_cols].apply(tuple, axis=1).isin(added_keys)].copy()

        removed_value = 0.0
        added_value = 0.0

        if previous_date in price_matrix.index:
            row = price_matrix.loc[previous_date]
            for _, pos in removed_df.iterrows():
                px = row.get(str(pos[COL_TICKER]).strip().upper())
                if pd.notna(px):
                    shares = float(pd.to_numeric(pd.Series([pos[COL_SHARES]]), errors="coerce").fillna(0.0).iloc[0])
                    removed_value += abs(shares * float(px))

        if current_date in price_matrix.index:
            row = price_matrix.loc[current_date]
            for _, pos in added_df.iterrows():
                px = row.get(str(pos[COL_TICKER]).strip().upper())
                if pd.notna(px):
                    shares = float(pd.to_numeric(pd.Series([pos[COL_SHARES]]), errors="coerce").fillna(0.0).iloc[0])
                    added_value += abs(shares * float(px))

        replaced_value += min(removed_value, added_value)

    return (replaced_value / average_aum) * 100.0


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_team_view_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    with session_scope() as session:
        position_state_df = load_position_state(session)
        snapshots_df = load_all_portfolio_snapshots(session)

    if position_state_df.empty:
        return pd.DataFrame(), pd.DataFrame()

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
    snapshot_df = build_current_portfolio_snapshot(position_state_df, latest_prices_df)
    return snapshot_df, snapshots_df


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_factor_analytics(snapshot_df: pd.DataFrame):
    builder = getattr(
        exposure_module,
        "build_factor_analytics_platform",
        getattr(exposure_module, "build_custom_live_factor_model"),
    )
    return builder(snapshot_df)


def _get_team_options(snapshot_df: pd.DataFrame) -> list[str]:
    teams = (
        snapshot_df[COL_TEAM]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.ne("Cash")]
        .unique()
        .tolist()
    )
    ordered = [team for team in settings.display_team_order if team in teams]
    fallback = sorted([team for team in teams if team not in ordered])
    return ordered + fallback


def _build_holding_metadata(team: str, snapshots_df: pd.DataFrame) -> pd.DataFrame:
    if snapshots_df.empty:
        return pd.DataFrame(columns=[COL_TICKER, COL_POSITION_SIDE, "inception_date", "oldest_cost_basis"])

    df = snapshots_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df[COL_TEAM] = df[COL_TEAM].astype(str).str.strip()
    df[COL_TICKER] = df[COL_TICKER].astype(str).str.strip().str.upper()
    df[COL_POSITION_SIDE] = df[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    df[COL_COST_BASIS] = pd.to_numeric(df.get(COL_COST_BASIS), errors="coerce")
    df = df.loc[df[COL_TEAM] == team].copy()
    df = df.loc[~_cash_like_mask(df)].copy()
    if df.empty:
        return pd.DataFrame(columns=[COL_TICKER, COL_POSITION_SIDE, "inception_date", "oldest_cost_basis"])

    rows = []
    for (ticker, position_side), group in df.groupby([COL_TICKER, COL_POSITION_SIDE], dropna=False):
        group = group.sort_values("snapshot_date").reset_index(drop=True)
        valid_cost_basis = group[COL_COST_BASIS].dropna()
        rows.append(
            {
                COL_TICKER: ticker,
                COL_POSITION_SIDE: position_side,
                "inception_date": group["snapshot_date"].iloc[0],
                "oldest_cost_basis": float(valid_cost_basis.iloc[0]) if not valid_cost_basis.empty else pd.NA,
            }
        )

    return pd.DataFrame(rows)


def _build_team_history(team: str, snapshots_df: pd.DataFrame, benchmark_ticker: str) -> pd.DataFrame:
    if snapshots_df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow", "benchmark_aum"])

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
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow", "benchmark_aum"])

    today = pd.Timestamp.today().normalize()
    oldest_snapshot_date = df["snapshot_date"].min().normalize()
    start_date = max(oldest_snapshot_date, today - pd.Timedelta(days=372))
    business_dates = pd.bdate_range(start=start_date, end=today)
    if len(business_dates) == 0:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow", "benchmark_aum"])

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

    investable_tickers = (
        df[COL_TICKER]
        .loc[~_cash_like_mask(df)]
        .dropna()
        .unique()
        .tolist()
    )
    trade_tickers = trades[COL_TICKER].dropna().unique().tolist() if not trades.empty else []
    history_tickers = sorted(set(investable_tickers + trade_tickers + ([benchmark_ticker] if benchmark_ticker else [])))
    raw_price_history = fetch_multiple_price_histories(history_tickers, lookback_days=(today - start_date).days + 30)
    price_matrix = _build_price_matrix(raw_price_history, business_dates)

    snapshot_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in df.groupby("snapshot_date")
    }
    snapshot_dates = sorted(snapshot_by_date.keys())
    trades_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in trades.groupby(trades["trade_date"].dt.normalize())
    } if not trades.empty else {}
    cash_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in external_cash.groupby(external_cash["activity_date"].dt.normalize())
    } if not external_cash.empty else {}
    rows = []
    active_positions_df: pd.DataFrame | None = None

    for dt in business_dates:
        net_external_flow = 0.0
        snapshot_for_day = snapshot_by_date.get(dt.normalize())

        if snapshot_for_day is not None:
            active_positions_df = snapshot_for_day.copy()
        else:
            if active_positions_df is None:
                eligible = [d for d in snapshot_dates if d <= dt]
                if not eligible:
                    continue
                active_positions_df = snapshot_by_date[eligible[-1]].copy()

            trades_today = trades_by_date.get(dt.normalize())
            if trades_today is not None and not trades_today.empty:
                active_positions_df, _ = apply_trades_to_positions(
                    base_positions_df=active_positions_df,
                    trades_df=trades_today,
                )

        cash_today = cash_by_date.get(dt.normalize())
        if cash_today is not None and not cash_today.empty:
            net_external_flow = float(pd.to_numeric(cash_today["amount"], errors="coerce").fillna(0.0).sum())
            if snapshot_for_day is None and active_positions_df is not None:
                active_positions_df = apply_cash_ledger_entries_to_positions(
                    positions_df=active_positions_df,
                    cash_entries_df=cash_today,
                )

        if active_positions_df is None:
            continue

        price_map = {}
        if dt in price_matrix.index:
            row = price_matrix.loc[dt]
            price_map = {
                str(ticker).strip().upper(): float(row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(row[ticker])
            }

        team_aum = _compute_position_value(active_positions_df, price_map)
        rows.append({"date": dt, "team_aum": float(team_aum), "net_external_flow": net_external_flow})

    history_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if history_df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow", "benchmark_aum"])

    history_df["benchmark_aum"] = pd.NA
    if benchmark_ticker and benchmark_ticker in price_matrix.columns:
        bench_prices = pd.to_numeric(price_matrix[benchmark_ticker].reindex(history_df["date"]).values, errors="coerce")
        valid = pd.Series(bench_prices).dropna()
        if not valid.empty:
            scaled = pd.Series(index=history_df.index, dtype="float64")
            scaled.iloc[0] = float(history_df["team_aum"].iloc[0])
            if len(history_df) > 1:
                bench_returns = pd.Series(bench_prices).pct_change()
                scaled_tail = build_flow_adjusted_benchmark_series(
                    benchmark_return_series=bench_returns.iloc[1:],
                    external_flow_series=history_df["net_external_flow"].iloc[1:],
                    initial_value=float(history_df["team_aum"].iloc[0]),
                )
                scaled.iloc[1:] = scaled_tail.values
            history_df["benchmark_aum"] = scaled.values

    return history_df


def _compute_period_return(price_matrix: pd.DataFrame, ticker: str, current_date: pd.Timestamp, inception_date: pd.Timestamp, days: int) -> float | None:
    if ticker not in price_matrix.columns or pd.isna(inception_date):
        return None

    target_date = current_date - pd.Timedelta(days=days)
    if inception_date.normalize() > target_date.normalize():
        return None

    series = pd.to_numeric(price_matrix[ticker], errors="coerce").dropna()
    if series.empty or current_date not in series.index:
        return None

    eligible = series.loc[series.index <= target_date]
    if eligible.empty:
        return None

    start_px = float(eligible.iloc[-1])
    end_px = float(series.loc[current_date])
    if start_px == 0:
        return None
    return (end_px / start_px) - 1.0


def _build_team_holdings_view(team_df: pd.DataFrame, snapshots_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    investable_df = team_df.loc[~_cash_like_mask(team_df)].copy()
    if investable_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metadata_df = _build_holding_metadata(str(investable_df[COL_TEAM].iloc[0]), snapshots_df)
    investable_df = investable_df.merge(
        metadata_df,
        on=[COL_TICKER, COL_POSITION_SIDE],
        how="left",
    )

    tickers = investable_df[COL_TICKER].dropna().astype(str).str.strip().str.upper().unique().tolist()
    raw_price_history = fetch_multiple_price_histories(tickers=tickers, lookback_days=400)
    calendar_end = pd.Timestamp.today().normalize()
    price_matrix = _build_price_matrix(raw_price_history, pd.bdate_range(start=calendar_end - pd.Timedelta(days=400), end=calendar_end))
    current_date = price_matrix.index.max() if not price_matrix.empty else pd.NaT

    view_df = investable_df.copy()
    view_df["1_year_return"] = pd.NA
    view_df["6_month_return"] = pd.NA
    view_df["1_month_return"] = pd.NA
    view_df["1_day_return"] = pd.NA
    view_df["pnl_contribution"] = pd.NA

    for idx, row in view_df.iterrows():
        ticker = str(row[COL_TICKER]).strip().upper()
        inception_date = pd.to_datetime(row.get("inception_date"), errors="coerce")
        current_px = float(pd.to_numeric(pd.Series([row.get(COL_PRICE)]), errors="coerce").fillna(0.0).iloc[0])
        shares = float(pd.to_numeric(pd.Series([row.get(COL_SHARES)]), errors="coerce").fillna(0.0).iloc[0])
        side = str(row.get(COL_POSITION_SIDE, "")).strip().upper()

        ret_1y = _compute_period_return(price_matrix, ticker, current_date, inception_date, 365)
        ret_6m = _compute_period_return(price_matrix, ticker, current_date, inception_date, 182)
        ret_1m = _compute_period_return(price_matrix, ticker, current_date, inception_date, 30)
        ret_1d = _compute_period_return(price_matrix, ticker, current_date, inception_date, 1)

        view_df.loc[idx, "1_year_return"] = ret_1y
        view_df.loc[idx, "6_month_return"] = ret_6m
        view_df.loc[idx, "1_month_return"] = ret_1m
        view_df.loc[idx, "1_day_return"] = ret_1d

        if ticker in price_matrix.columns:
            series = pd.to_numeric(price_matrix[ticker], errors="coerce").dropna()
            start_date = inception_date
            if pd.notna(inception_date) and inception_date <= current_date - pd.Timedelta(days=365):
                start_date = current_date - pd.Timedelta(days=365)
            eligible = series.loc[series.index <= pd.to_datetime(start_date)] if pd.notna(start_date) else pd.Series(dtype="float64")
            if not eligible.empty:
                start_px = float(eligible.iloc[-1])
                signed_multiplier = -1.0 if side == "SHORT" else 1.0
                view_df.loc[idx, "pnl_contribution"] = signed_multiplier * shares * (current_px - start_px)

    formatted_df = view_df.rename(
        columns={
            COL_TICKER: "Ticker",
            COL_POSITION_SIDE: "Position",
            COL_SHARES: "Shares",
            COL_PRICE: "Current Price",
            "oldest_cost_basis": "Cost Basis",
            COL_MARKET_VALUE: "Market Value",
            COL_WEIGHT: "Total Portfolio Weight",
            "1_year_return": "1 Year Return",
            "6_month_return": "6 Month Return",
            "1_month_return": "1 Month Return",
            "1_day_return": "1 Day Return",
        }
    ).copy()

    if "Position" in formatted_df.columns:
        formatted_df["Position"] = formatted_df["Position"].str.title()
    for col in ["Shares"]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(_format_number)
    for col in ["Current Price", "Cost Basis", "Market Value"]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(_format_currency)
    for col in ["Total Portfolio Weight", "1 Year Return", "6 Month Return", "1 Month Return", "1 Day Return"]:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(_format_percent)

    chart_df = view_df[[COL_TICKER, COL_TEAM, COL_MARKET_VALUE, "pnl_contribution"]].copy()
    chart_df[COL_MARKET_VALUE] = pd.to_numeric(chart_df[COL_MARKET_VALUE], errors="coerce")
    chart_df["pnl_contribution"] = pd.to_numeric(chart_df["pnl_contribution"], errors="coerce")
    return formatted_df, chart_df


def render_empty_state() -> None:
    render_page_title("Sector View")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_team_dashboard(team_df: pd.DataFrame, history_df: pd.DataFrame, snapshots_df: pd.DataFrame, team: str) -> None:
    st.subheader("Pod Performance Summary")
    team_market_value = float(pd.to_numeric(team_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)) if not team_df.empty else 0.0
    returns = _build_return_series(history_df, "team_aum")
    team_turnover = _compute_team_turnover(history_df, snapshots_df, team)
    sharpe = compute_sharpe_ratio(returns) if not returns.empty else None
    sortino = compute_sortino_ratio(returns) if not returns.empty else None
    batting_average = _compute_batting_average(returns)
    win_loss_ratio = _compute_win_loss_ratio(returns)
    historical_var_95 = _compute_historical_var_95(returns)
    one_year_max_drawdown = _compute_one_year_max_drawdown(returns)

    with st.container(border=True):
        row_1 = st.columns(4)
        with row_1[0]:
            _render_pod_summary_card("Pod Market Value", _format_currency(team_market_value))
        with row_1[1]:
            _render_pod_summary_card("1 Year Portfolio Turnover", _format_percent(team_turnover / 100.0 if team_turnover is not None else None))
        with row_1[2]:
            _render_pod_summary_card("1 Year Sharpe Ratio", _format_number(sharpe))
        with row_1[3]:
            _render_pod_summary_card("1 Year Sortino Ratio", _format_number(sortino))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        row_2 = st.columns(4)
        with row_2[0]:
            _render_pod_summary_card("Batting Average", _format_percent(batting_average))
        with row_2[1]:
            _render_pod_summary_card("Win vs Loss Ratio", _format_number(win_loss_ratio))
        with row_2[2]:
            _render_pod_summary_card("Daily Historical VaR (95%)", _format_percent(historical_var_95))
        with row_2[3]:
            _render_pod_summary_card("1 Year Max Drawdown", _format_percent(one_year_max_drawdown))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def render_holdings_table(formatted_df: pd.DataFrame) -> None:
    st.subheader("CURRENT POD HOLDINGS")
    if formatted_df.empty:
        st.info("No holdings available for this pod.")
        return
    display_cols = [
        "Ticker",
        "Position",
        "Shares",
        "Current Price",
        "Cost Basis",
        "Market Value",
        "Total Portfolio Weight",
        "1 Year Return",
        "6 Month Return",
        "1 Month Return",
        "1 Day Return",
    ]
    st.dataframe(left_align_dataframe(formatted_df[display_cols]), use_container_width=True, hide_index=True)


def render_team_charts(team_df: pd.DataFrame, chart_df: pd.DataFrame, team: str) -> None:
    investable_df = team_df.loc[~_cash_like_mask(team_df)].copy()
    if investable_df.empty:
        st.info("No pod holdings available for charting.")
        return

    base_color = _get_team_color(team)
    pie_colors = _build_tonal_palette(base_color, len(investable_df))

    pie_fig = px.pie(
        investable_df,
        names=COL_TICKER,
        values=COL_MARKET_VALUE,
        title="CURRENT POD ALLOCATION",
        color_discrete_sequence=pie_colors,
    )
    pie_fig.update_layout(
        title=dict(text="CURRENT POD ALLOCATION", x=0.5, xanchor="center"),
        showlegend=False,
        margin=dict(t=100, b=20),
    )

    pnl_df = chart_df.dropna(subset=["pnl_contribution"]).sort_values("pnl_contribution", ascending=False).copy()
    pnl_colors = [
        _blend_hex(base_color, (255, 255, 255), 0.12) if value >= 0
        else _blend_hex(base_color, (255, 255, 255), 0.42)
        for value in pd.to_numeric(pnl_df["pnl_contribution"], errors="coerce").fillna(0.0)
    ]
    bar_fig = px.bar(
        pnl_df,
        x="pnl_contribution",
        y=COL_TICKER,
        orientation="h",
        title="1 YEAR P&L CONTRIBUTION BY POSITION",
    )
    bar_fig.update_traces(marker_color=pnl_colors, marker_line_width=0)
    bar_fig.update_layout(
        title=dict(text="1 YEAR P&L CONTRIBUTION BY POSITION", x=0.5, xanchor="center"),
        xaxis_title="P&L CONTRIBUTION",
        yaxis_title="TICKER",
        showlegend=False,
        margin=dict(t=100, b=20),
    )
    bar_fig.update_yaxes(categoryorder="total ascending")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with col2:
        st.plotly_chart(bar_fig, use_container_width=True)


def render_team_history(history_df: pd.DataFrame, team: str) -> None:
    st.subheader("1 Year Returns vs Benchmark")
    if history_df.empty:
        st.info("No historical pod return data available.")
        return

    plot_df = history_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    start_date = plot_df["date"].max() - pd.Timedelta(days=365)
    plot_df = plot_df.loc[plot_df["date"] >= start_date].copy()
    if plot_df.empty:
        st.info("No historical pod return data available.")
        return

    team_series = pd.to_numeric(plot_df["team_aum"], errors="coerce")
    valid_team = team_series.dropna()
    if valid_team.empty or float(valid_team.iloc[0]) == 0:
        st.info("No historical pod return data available.")
        return
    plot_df["team_return"] = team_series / float(valid_team.iloc[0]) - 1.0

    has_benchmark = False
    if "benchmark_aum" in plot_df.columns:
        benchmark_series = pd.to_numeric(plot_df["benchmark_aum"], errors="coerce")
        valid_benchmark = benchmark_series.dropna()
        if not valid_benchmark.empty and float(valid_benchmark.iloc[0]) != 0:
            plot_df["benchmark_return"] = benchmark_series / float(valid_benchmark.iloc[0]) - 1.0
            has_benchmark = True

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["team_return"],
            mode="lines",
            name=team,
        )
    )
    if has_benchmark:
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df["benchmark_return"],
                mode="lines",
                name=f"{TEAM_BENCHMARK_TICKERS.get(team, 'SPY')} Benchmark",
            )
        )

    fig.update_layout(
        title=dict(text="1 YEAR RETURNS VS SECTOR BENCHMARK", x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="Cumulative Return",
        legend=dict(title=None),
        margin=dict(t=58),
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology Note"):
        st.write(
            f"""
            Pod history is reconstructed from authoritative snapshots for **{team}**,
            then carried forward with intervening trade receipts and any uploaded
            sector-rebalance or portfolio-liquidation cash flows.

            This chart normalizes the pod and benchmark series to 0 at the start of
            the trailing 1-year window while using
            **{TEAM_BENCHMARK_TICKERS.get(team, 'SPY')}** as the sector proxy.
            """
        )


def render_team_factor_loadings(team: str, history_df: pd.DataFrame, analytics: dict) -> None:
    st.subheader("Return Decomposition")

    factor_returns_df = analytics.get("factor_returns", pd.DataFrame())
    if history_df.empty:
        st.info("No pod return history is available for decomposition.")
        return
    if factor_returns_df.empty:
        st.info("No factor return history is available for decomposition.")
        return

    team_history = _prepare_team_performance_history(history_df)
    if team_history.empty or "performance_return" not in team_history.columns:
        st.info("No flow-adjusted pod return history is available for decomposition.")
        return

    benchmark_history = prepare_flow_adjusted_history(
        history_df=history_df,
        value_column="benchmark_aum",
        flow_column="net_external_flow",
    )
    if benchmark_history.empty or "performance_return" not in benchmark_history.columns:
        st.info("Benchmark return history is unavailable for decomposition.")
        return

    team_returns = (
        team_history[["date", "performance_return"]]
        .rename(columns={"performance_return": "team_return"})
        .copy()
    )
    benchmark_returns = (
        benchmark_history[["date", "performance_return"]]
        .rename(columns={"performance_return": "benchmark_return"})
        .copy()
    )
    factor_inputs = factor_returns_df[["date", "SMB", "MOM", "VAL"]].copy()

    regression_df = (
        team_returns
        .merge(benchmark_returns, on="date", how="inner")
        .merge(factor_inputs, on="date", how="inner")
    )
    regression_df["date"] = pd.to_datetime(regression_df["date"], errors="coerce")
    for col in ["team_return", "benchmark_return", "SMB", "MOM", "VAL"]:
        regression_df[col] = pd.to_numeric(regression_df[col], errors="coerce")
    regression_df = regression_df.dropna(subset=["date", "team_return", "benchmark_return", "SMB", "MOM", "VAL"])
    regression_df = regression_df.sort_values("date").reset_index(drop=True)
    if regression_df.empty:
        st.info("No overlapping pod, benchmark, and factor return history is available for decomposition.")
        return

    end_date = regression_df["date"].max()
    start_date = end_date - pd.Timedelta(days=365)
    regression_df = regression_df.loc[regression_df["date"] >= start_date].copy()
    if len(regression_df) < 20:
        st.info("N/A: return decomposition needs more overlapping pod, benchmark, and factor observations in the last year.")
        return

    x = regression_df[["benchmark_return", "SMB", "MOM", "VAL"]].to_numpy(dtype=float)
    y = regression_df["team_return"].to_numpy(dtype=float)
    x_with_const = np.column_stack([np.ones(len(regression_df)), x])
    try:
        coefficients, *_ = np.linalg.lstsq(x_with_const, y, rcond=None)
    except np.linalg.LinAlgError:
        st.info("N/A: return decomposition could not be estimated because the regression inputs are unstable.")
        return

    regression_df["contribution_benchmark"] = coefficients[1] * regression_df["benchmark_return"]
    regression_df["contribution_smb"] = coefficients[2] * regression_df["SMB"]
    regression_df["contribution_mom"] = coefficients[3] * regression_df["MOM"]
    regression_df["contribution_val"] = coefficients[4] * regression_df["VAL"]
    regression_df["explained_return"] = regression_df[
        ["contribution_benchmark", "contribution_smb", "contribution_mom", "contribution_val"]
    ].sum(axis=1)
    regression_df["residual"] = regression_df["team_return"] - regression_df["explained_return"]

    cumulative_df = regression_df[["date"]].copy()
    cumulative_map = {
        "contribution_val": "Value",
        "contribution_benchmark": "Benchmark",
        "residual": "Idiosyncratic",
        "contribution_smb": "Size",
        "contribution_mom": "Momentum",
    }
    for column in cumulative_map:
        cumulative_df[column] = pd.to_numeric(regression_df[column], errors="coerce").fillna(0.0).cumsum()

    fig = go.Figure()
    for column, label in cumulative_map.items():
        fig.add_trace(
            go.Scatter(
                x=cumulative_df["date"],
                y=cumulative_df[column],
                mode="lines",
                name=label,
                line=dict(color=FACTOR_COLORS[label], width=2.5),
                stackgroup="one",
                groupnorm="",
                fillcolor=_hex_to_rgba(FACTOR_COLORS[label], 0.55),
                hovertemplate=f"{label}: %{{y:.2%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="1 YEAR CUMULATIVE RETURN DECOMPOSITION", x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="Cumulative Contribution",
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(t=90, b=90),
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology Note"):
        st.write(
            f"""
            This decomposition estimates each pod's daily return as a function of its
            **{TEAM_BENCHMARK_TICKERS.get(team, 'SPY')}** benchmark return plus the
            shared **Size**, **Momentum**, and **Value** factor return series over the
            trailing 1-year window.

            The plotted lines show cumulative attributed return from **Benchmark**,
            **Size**, **Momentum**, **Value**, and **Idiosyncratic** return, where
            idiosyncratic return is the portion of pod performance not explained by
            the benchmark and style-factor legs.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Sector View", layout="wide")
    apply_app_theme()
    render_top_nav()
    _apply_team_page_theme()
    render_page_title("Sector View")

    snapshot_df, snapshots_df = get_team_view_data()
    if snapshot_df.empty:
        render_empty_state()
        return

    factor_analytics = get_factor_analytics(snapshot_df)
    team_options = _get_team_options(snapshot_df)
    team_tabs = st.tabs(team_options)

    for team_name, team_tab in zip(team_options, team_tabs):
        with team_tab:
            team_df = snapshot_df.loc[snapshot_df[COL_TEAM] == team_name].copy()
            benchmark_ticker = TEAM_BENCHMARK_TICKERS.get(team_name, "SPY")
            history_df = _build_team_history(team_name, snapshots_df, benchmark_ticker)
            formatted_holdings_df, chart_df = _build_team_holdings_view(team_df, snapshots_df)

            render_team_dashboard(team_df, history_df, snapshots_df, team_name)
            st.divider()
            render_holdings_table(formatted_holdings_df)
            st.divider()
            render_team_charts(team_df, chart_df, team_name)
            st.divider()
            render_team_history(history_df, team_name)
            st.divider()
            render_team_factor_loadings(team_name, history_df, factor_analytics)


if __name__ == "__main__":
    main()
