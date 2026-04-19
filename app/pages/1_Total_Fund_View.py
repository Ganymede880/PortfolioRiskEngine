"""
Total Fund View dashboard page for the CMCSIF Portfolio Tracker.

This version includes:
- current top-line portfolio metrics
- performance dashboard (1Y / 1M / 1W / Daily return)
- current team allocation table + pie chart
- weekly AUM chart for the past year (or back to oldest snapshot)
- SPY / QQQ benchmark AUM comparison, scaled to the portfolio's initial AUM

Important current assumption:
- portfolio holdings are carried forward from the latest uploaded snapshot until the
  next uploaded snapshot, unless a newer snapshot replaces them
- trade receipts are not yet applied into the carried historical series on this page
- benchmark lines use SPY and QQQ as practical proxies for S&P 500 and Nasdaq exposure
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.portfolio import (
    build_current_portfolio_snapshot,
    summarize_total_portfolio,
)
from src.analytics.performance import compute_sharpe_ratio, compute_sortino_ratio
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import load_all_portfolio_snapshots, load_position_state
from src.db.session import session_scope


COL_DATE = "as_of_date"
COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
COL_PRICE = "price"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"

BENCHMARK_TICKERS = {
    "S&P 500": "SPY",
    "Nasdaq": "QQQ",
}
TEAM_SP500_WEIGHTS = {
    "Consumer": 0.16,
    "E&U": 0.07,
    "F&R": 0.14,
    "Healthcare": 0.11,
    "TMT": 0.40,
    "M&I": 0.12,
    "Cash": 0.0,
}
TEAM_COLORS = {
    "Consumer": "#38bdf8",
    "E&U": "#f59e0b",
    "F&R": "#10b981",
    "Healthcare": "#f43f5e",
    "TMT": "#8b5cf6",
    "M&I": "#eab308",
    "Cash": "#94a3b8",
}
MAX_RETURN_LOOKBACK_DAYS = 365
RETURN_LOOKBACK_BUFFER_DAYS = 7


def _format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def _format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def _format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def apply_page_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            background: #07111a !important;
            color: #f8fafc !important;
        }

        .block-container {
            background: #07111a !important;
            color: #f8fafc !important;
            padding-top: 1.5rem;
        }

        h1, h2, h3, h4, h5, h6,
        p, span, label, div, li,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] {
            color: #f8fafc !important;
        }

        [data-testid="stMetric"] {
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.1rem 0.15rem;
            box-shadow: none;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.74);
            border-color: rgba(148, 163, 184, 0.35) !important;
            border-radius: 18px;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.summary-panel-marker) {
            background:
                linear-gradient(135deg, rgba(59, 130, 246, 0.48) 0%, rgba(30, 64, 175, 0.34) 42%, rgba(15, 23, 42, 0.88) 78%),
                linear-gradient(180deg, rgba(125, 211, 252, 0.16) 0%, rgba(15, 23, 42, 0.02) 100%);
            border: 1px solid rgba(125, 211, 252, 0.5) !important;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.28);
        }

        [data-testid="stDataFrame"] table,
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] td {
            text-align: left !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_portfolio_return_series(history_df: pd.DataFrame) -> pd.Series:
    """
    Convert portfolio AUM history into a simple daily return series.
    """
    if history_df.empty or "portfolio_aum" not in history_df.columns:
        return pd.Series(dtype="float64")

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["portfolio_aum"] = pd.to_numeric(df["portfolio_aum"], errors="coerce")
    df = df.dropna(subset=["date", "portfolio_aum"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return pd.Series(dtype="float64")

    returns = df["portfolio_aum"].pct_change().replace([float("inf"), float("-inf")], pd.NA)
    return pd.to_numeric(returns, errors="coerce").dropna()


def _is_cash_like_row(row: pd.Series) -> bool:
    ticker = str(row.get(COL_TICKER, "")).strip().upper()
    team = str(row.get(COL_TEAM, "")).strip().upper()
    position_side = str(row.get(COL_POSITION_SIDE, "")).strip().upper()
    return (
        ticker in {"CASH", "EUR", "GBP", "NOGXX"}
        or team == "CASH"
        or position_side == "CASH"
    )


def _compute_one_year_portfolio_turnover(history_df: pd.DataFrame) -> float | None:
    """
    Estimate 1-year turnover as replaced position value divided by average AUM.

    Replaced position value is approximated from authoritative snapshot changes:
    for each snapshot transition in the trailing year, compare investable
    position keys and count the smaller of:
    - market value of positions removed
    - market value of positions added
    """
    if history_df.empty or "date" not in history_df.columns or "portfolio_aum" not in history_df.columns:
        return None

    history = history_df.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["portfolio_aum"] = pd.to_numeric(history["portfolio_aum"], errors="coerce")
    history = history.dropna(subset=["date", "portfolio_aum"]).sort_values("date").reset_index(drop=True)
    if history.empty:
        return None

    end_ts = history["date"].iloc[-1].normalize()
    start_ts = end_ts - pd.Timedelta(days=365)

    trailing_history = history.loc[history["date"] >= start_ts].copy()
    average_aum = float(trailing_history["portfolio_aum"].mean()) if not trailing_history.empty else 0.0
    if average_aum == 0:
        return None

    with session_scope() as session:
        snapshots_df = load_all_portfolio_snapshots(session)

    if snapshots_df.empty:
        return 0.0

    snapshots = snapshots_df.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"], errors="coerce")
    snapshots[COL_TICKER] = snapshots[COL_TICKER].astype(str).str.strip().str.upper()
    snapshots[COL_TEAM] = snapshots[COL_TEAM].astype(str).str.strip()
    snapshots[COL_POSITION_SIDE] = snapshots[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    snapshots[COL_SHARES] = pd.to_numeric(snapshots[COL_SHARES], errors="coerce")
    snapshots = snapshots.dropna(subset=["snapshot_date", COL_TICKER, COL_POSITION_SIDE, COL_SHARES]).copy()
    if snapshots.empty:
        return 0.0

    snapshots = snapshots.loc[~snapshots.apply(_is_cash_like_row, axis=1)].copy()
    if snapshots.empty:
        return 0.0

    ticker_normalization = {
        "BRKB": "BRK-B",
        "BRKA": "BRK-A",
    }
    snapshots[COL_TICKER] = snapshots[COL_TICKER].replace(ticker_normalization)

    snapshot_dates = sorted(snapshots["snapshot_date"].dt.normalize().unique().tolist())
    transition_dates = [dt for dt in snapshot_dates if dt >= start_ts]
    if not transition_dates:
        return 0.0

    oldest_transition_prev_date = max([dt for dt in snapshot_dates if dt < transition_dates[0]], default=transition_dates[0])
    pricing_start = min(oldest_transition_prev_date, transition_dates[0])
    pricing_end = snapshot_dates[-1]

    tickers = sorted(snapshots[COL_TICKER].dropna().unique().tolist())
    raw_price_history = fetch_multiple_price_histories(
        tickers=tickers,
        lookback_days=max((pricing_end - pricing_start).days + 30, 30),
    )
    calendar_dates = pd.date_range(start=pricing_start, end=pricing_end, freq="D")
    price_matrix = _build_price_matrix(raw_price_history, calendar_dates)

    snapshot_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in snapshots.groupby(snapshots["snapshot_date"].dt.normalize())
    }

    replaced_value = 0.0

    for current_date in transition_dates:
        prior_dates = [dt for dt in snapshot_dates if dt < current_date]
        if not prior_dates:
            continue

        previous_date = prior_dates[-1]
        previous_df = snapshot_by_date.get(previous_date, pd.DataFrame()).copy()
        current_df = snapshot_by_date.get(current_date, pd.DataFrame()).copy()
        if previous_df.empty or current_df.empty:
            continue

        key_cols = [COL_TEAM, COL_TICKER, COL_POSITION_SIDE]
        previous_keys = set(map(tuple, previous_df[key_cols].itertuples(index=False, name=None)))
        current_keys = set(map(tuple, current_df[key_cols].itertuples(index=False, name=None)))

        removed_keys = previous_keys - current_keys
        added_keys = current_keys - previous_keys

        if not removed_keys and not added_keys:
            continue

        removed_df = previous_df.loc[
            previous_df[key_cols].apply(tuple, axis=1).isin(removed_keys)
        ].copy()
        added_df = current_df.loc[
            current_df[key_cols].apply(tuple, axis=1).isin(added_keys)
        ].copy()

        previous_price_map: dict[str, float] = {}
        current_price_map: dict[str, float] = {}

        if previous_date in price_matrix.index and len(price_matrix.columns) > 0:
            previous_row = price_matrix.loc[previous_date]
            previous_price_map = {
                str(ticker).strip().upper(): float(previous_row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(previous_row[ticker])
            }

        if current_date in price_matrix.index and len(price_matrix.columns) > 0:
            current_row = price_matrix.loc[current_date]
            current_price_map = {
                str(ticker).strip().upper(): float(current_row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(current_row[ticker])
            }

        removed_value = abs(_apply_position_values(removed_df, previous_price_map))
        added_value = abs(_apply_position_values(added_df, current_price_map))
        replaced_value += min(removed_value, added_value)

    return (replaced_value / average_aum) * 100.0


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_master_fund_snapshot() -> pd.DataFrame:
    """
    Load reconstructed latest position state and attach latest prices.
    """
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
    snapshot_df = build_current_portfolio_snapshot(position_state_df, latest_prices_df)

    return snapshot_df


def _build_complete_team_summary(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build current allocation summary for all configured teams, including cash.
    """
    configured_teams = list(settings.display_team_order)

    rows = []
    for team in configured_teams:
        team_df = snapshot_df.loc[snapshot_df[COL_TEAM] == team].copy()

        if team.lower() == "cash":
            investable_positions = 0
        else:
            investable_positions = int(
                team_df.loc[team_df[COL_TICKER] != "CASH", COL_TICKER].nunique(dropna=True)
            )

        market_value = float(
            pd.to_numeric(team_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)
        ) if not team_df.empty else 0.0

        rows.append({
            COL_TEAM: team,
            COL_MARKET_VALUE: market_value,
            "position_count": investable_positions,
        })

    summary_df = pd.DataFrame(rows)
    total_market_value = pd.to_numeric(summary_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)

    if total_market_value == 0 or pd.isna(total_market_value):
        summary_df[COL_WEIGHT] = 0.0
    else:
        summary_df[COL_WEIGHT] = pd.to_numeric(summary_df[COL_MARKET_VALUE], errors="coerce") / total_market_value

    return summary_df


def _apply_position_values(snapshot_positions_df: pd.DataFrame, price_map: dict[str, float]) -> float:
    """
    Compute total portfolio AUM for a snapshot-like DataFrame using a ticker->price map.

    Rules:
    - CASH rows use shares directly
    - rows with missing price are skipped
    - short positions subtract market value
    """
    if snapshot_positions_df.empty:
        return 0.0

    df = snapshot_positions_df.copy()
    df[COL_SHARES] = pd.to_numeric(df[COL_SHARES], errors="coerce")

    total_aum = 0.0

    for _, row in df.iterrows():
        ticker = str(row.get(COL_TICKER, "")).strip().upper()
        team = str(row.get(COL_TEAM, "")).strip().upper()
        position_side = str(row.get(COL_POSITION_SIDE, "")).strip().upper()
        shares = float(pd.to_numeric(pd.Series([row.get(COL_SHARES)]), errors="coerce").fillna(0.0).iloc[0])

        if ticker in {"CASH", "EUR", "GBP"} or team == "CASH" or position_side == "CASH":
            total_aum += shares
            continue

        px_val = price_map.get(ticker)
        if px_val is None or pd.isna(px_val):
            continue

        if position_side == "LONG":
            total_aum += shares * px_val
        elif position_side == "SHORT":
            total_aum -= shares * px_val
        else:
            total_aum += shares * px_val

    return float(total_aum)


def _build_price_matrix(
    raw_price_history: pd.DataFrame,
    business_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a date x ticker price matrix aligned to the requested business dates.

    Missing or failed prices remain missing here and are interpreted as zero
    contribution later when valuing positions.
    """
    if len(business_dates) == 0:
        return pd.DataFrame()

    if raw_price_history.empty:
        return pd.DataFrame(index=business_dates)

    prices = raw_price_history.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.strip().str.upper()
    prices["close"] = pd.to_numeric(prices.get("close"), errors="coerce")
    prices["adj_close"] = pd.to_numeric(prices.get("adj_close"), errors="coerce")

    prices["px"] = prices["adj_close"]
    missing_adj_mask = prices["px"].isna()
    prices.loc[missing_adj_mask, "px"] = prices.loc[missing_adj_mask, "close"]

    prices = prices.dropna(subset=["date", "ticker", "px"]).copy()
    if prices.empty:
        return pd.DataFrame(index=business_dates)

    return (
        prices.pivot_table(index="date", columns="ticker", values="px", aggfunc="last")
        .sort_index()
        .reindex(business_dates)
        .ffill()
    )


def _determine_history_start_date(
    oldest_snapshot_date: pd.Timestamp,
    today: pd.Timestamp,
) -> pd.Timestamp:
    """
    Include a small buffer before the longest displayed lookback window.

    This lets trailing-return calculations find the nearest available business
    day on or before the target date, even when the exact target is a weekend
    or holiday.
    """
    return max(
        oldest_snapshot_date.normalize(),
        today - pd.Timedelta(days=MAX_RETURN_LOOKBACK_DAYS + RETURN_LOOKBACK_BUFFER_DAYS),
    )


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_master_fund_history() -> pd.DataFrame:
    """
    Build a carried-forward daily portfolio AUM series using uploaded snapshots.

    Logic:
    - start slightly before the longest displayed lookback window
    - never start earlier than the oldest available snapshot date
    - for each business day, use the latest snapshot on or before that day
    - mark that snapshot to market using the latest available price on or before that day
    - carry positions forward until the next uploaded snapshot

    Benchmarks:
    - SPY for S&P 500 proxy
    - QQQ for Nasdaq proxy
    - benchmark AUM is scaled to match the portfolio's initial AUM
    """
    with session_scope() as session:
        all_snapshots_df = load_all_portfolio_snapshots(session)

    if all_snapshots_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "sp500_aum", "nasdaq_aum"])

    df = all_snapshots_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df[COL_TICKER] = df[COL_TICKER].astype(str).str.strip().str.upper()
    df[COL_TEAM] = df[COL_TEAM].astype(str).str.strip()
    df[COL_POSITION_SIDE] = df[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    df[COL_SHARES] = pd.to_numeric(df[COL_SHARES], errors="coerce")

    df = df.dropna(subset=["snapshot_date", COL_TICKER, COL_SHARES]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "sp500_aum", "nasdaq_aum"])

    # Normalize common ticker mismatches
    ticker_normalization = {
        "BRKB": "BRK-B",
        "BRKA": "BRK-A",
    }
    df[COL_TICKER] = df[COL_TICKER].replace(ticker_normalization)

    today = pd.Timestamp.today().normalize()
    oldest_snapshot_date = df["snapshot_date"].min()
    start_date = _determine_history_start_date(oldest_snapshot_date, today)
    business_dates = pd.bdate_range(start=start_date, end=today)

    if len(business_dates) == 0:
        return pd.DataFrame(columns=["date", "portfolio_aum", "sp500_aum", "nasdaq_aum"])

    equity_tickers = (
        df[COL_TICKER]
        .loc[~df[COL_TICKER].isin(["CASH", "EUR", "GBP"])]
        .dropna()
        .unique()
        .tolist()
    )
    benchmark_tickers = list(BENCHMARK_TICKERS.values())
    history_tickers = sorted(set(equity_tickers + benchmark_tickers))

    lookback_days = (today - start_date).days + 30
    raw_price_history = fetch_multiple_price_histories(
        tickers=history_tickers,
        lookback_days=lookback_days,
    )

    price_matrix = _build_price_matrix(raw_price_history, business_dates)

    snapshot_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in df.groupby("snapshot_date")
    }
    snapshot_dates_sorted = sorted(snapshot_by_date.keys())

    history_rows = []

    for dt in business_dates:
        eligible_snapshot_dates = [d for d in snapshot_dates_sorted if d <= dt]
        if not eligible_snapshot_dates:
            continue

        active_snapshot_date = eligible_snapshot_dates[-1]
        active_snapshot_df = snapshot_by_date[active_snapshot_date]

        price_map: dict[str, float] = {}
        if dt in price_matrix.index and len(price_matrix.columns) > 0:
            price_row = price_matrix.loc[dt]
            price_map = {
                str(ticker).strip().upper(): float(price_row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(price_row[ticker])
            }

        portfolio_aum = _apply_position_values(active_snapshot_df, price_map)

        history_rows.append({
            "date": dt,
            "portfolio_aum": float(portfolio_aum),
        })

    history_df = pd.DataFrame(history_rows)
    if history_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "sp500_aum", "nasdaq_aum"])

    history_df = history_df.sort_values("date").reset_index(drop=True)

    initial_aum = float(history_df["portfolio_aum"].iloc[0])

    # Benchmark AUM lines scaled to the portfolio's initial AUM
    for benchmark_name, benchmark_ticker in BENCHMARK_TICKERS.items():
        benchmark_ticker = benchmark_ticker.upper()
        if benchmark_ticker not in price_matrix.columns:
            continue

        aligned = history_df[["date"]].copy()
        aligned["benchmark_price"] = price_matrix[benchmark_ticker].reindex(aligned["date"]).values
        aligned["benchmark_price"] = pd.to_numeric(aligned["benchmark_price"], errors="coerce").ffill()

        valid = aligned["benchmark_price"].dropna()
        if valid.empty:
            continue

        start_price = float(valid.iloc[0])
        if start_price == 0:
            continue

        scaled_aum = initial_aum * (aligned["benchmark_price"] / start_price)

        if benchmark_name == "S&P 500":
            history_df["sp500_aum"] = scaled_aum.values
        elif benchmark_name == "Nasdaq":
            history_df["nasdaq_aum"] = scaled_aum.values

    if "sp500_aum" not in history_df.columns:
        history_df["sp500_aum"] = pd.NA
    if "nasdaq_aum" not in history_df.columns:
        history_df["nasdaq_aum"] = pd.NA

    return history_df


def _compute_trailing_return(history_df: pd.DataFrame, days: int) -> float | None:
    """
    Compute trailing return using the nearest available business-day observation
    on or before the target lookback date.
    """
    if history_df.empty or "portfolio_aum" not in history_df.columns:
        return None

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["portfolio_aum"] = pd.to_numeric(df["portfolio_aum"], errors="coerce")
    df = df.dropna(subset=["date", "portfolio_aum"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return None

    latest_date = df["date"].iloc[-1]
    latest_value = float(df["portfolio_aum"].iloc[-1])

    target_date = latest_date - pd.Timedelta(days=days)
    prior_df = df.loc[df["date"] <= target_date].copy()

    if prior_df.empty:
        return None

    prior_value = float(prior_df["portfolio_aum"].iloc[-1])
    if prior_value == 0:
        return None

    return (latest_value / prior_value) - 1.0


def _compute_daily_return(history_df: pd.DataFrame, current_total_market_value: float) -> float | None:
    """
    Compute daily return using the most recent historical carried AUM prior to today.
    """
    if history_df.empty:
        return None

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["portfolio_aum"] = pd.to_numeric(df["portfolio_aum"], errors="coerce")
    df = df.dropna(subset=["date", "portfolio_aum"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return None

    previous_value = float(df["portfolio_aum"].iloc[-2])
    latest_history_value = float(df["portfolio_aum"].iloc[-1])

    # If the current snapshot date already matches the latest carried date closely,
    # use the last two history points instead of mixing live/current with carried history.
    if latest_history_value != 0:
        return (latest_history_value / previous_value) - 1.0

    if previous_value == 0:
        return None

    return (current_total_market_value / previous_value) - 1.0


def _compute_trailing_pnl(history_df: pd.DataFrame, days: int) -> float | None:
    """
    Compute dollar P&L versus the nearest observation on or before the target date.
    """
    if history_df.empty or "portfolio_aum" not in history_df.columns:
        return None

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["portfolio_aum"] = pd.to_numeric(df["portfolio_aum"], errors="coerce")
    df = df.dropna(subset=["date", "portfolio_aum"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return None

    latest_date = df["date"].iloc[-1]
    latest_value = float(df["portfolio_aum"].iloc[-1])
    target_date = latest_date - pd.Timedelta(days=days)
    prior_df = df.loc[df["date"] <= target_date].copy()
    if prior_df.empty:
        return None

    prior_value = float(prior_df["portfolio_aum"].iloc[-1])
    return latest_value - prior_value


def _compute_daily_pnl(history_df: pd.DataFrame, current_total_market_value: float) -> float | None:
    """
    Compute daily dollar P&L from the last two business-day observations.
    """
    if history_df.empty or "portfolio_aum" not in history_df.columns:
        return None

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["portfolio_aum"] = pd.to_numeric(df["portfolio_aum"], errors="coerce")
    df = df.dropna(subset=["date", "portfolio_aum"]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return None

    previous_value = float(df["portfolio_aum"].iloc[-2])
    latest_history_value = float(df["portfolio_aum"].iloc[-1])

    if latest_history_value != 0:
        return latest_history_value - previous_value

    return current_total_market_value - previous_value


def _latest_history_date(history_df: pd.DataFrame) -> pd.Timestamp | None:
    if history_df.empty or "date" not in history_df.columns:
        return None

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return None
    return df["date"].iloc[-1]


def render_empty_state() -> None:
    st.title("Total Fund View")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_header_metrics(snapshot_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    """
    Top-level metrics using explicit, robust definitions.
    """

    if snapshot_df.empty:
        total_long_exposure = 0.0
        cash_value = 0.0
        gross_exposure = 0.0
        position_count = 0
        average_position_size = 0.0
    else:
        df = snapshot_df.copy()

        df[COL_MARKET_VALUE] = pd.to_numeric(df[COL_MARKET_VALUE], errors="coerce")

        # --- Cash ---
        cash_mask = (
            df[COL_TEAM].astype(str).str.upper().eq("CASH")
            | df[COL_TICKER].astype(str).str.upper().isin(["CASH", "EUR", "GBP", "NOGXX"])
            | df[COL_POSITION_SIDE].astype(str).str.upper().eq("CASH")
        )

        cash_value = df.loc[cash_mask, COL_MARKET_VALUE].sum(skipna=True)

        non_cash_df = df.loc[~cash_mask].copy()
        total_long_exposure = float(
            non_cash_df.loc[
                pd.to_numeric(non_cash_df[COL_MARKET_VALUE], errors="coerce") > 0,
                COL_MARKET_VALUE,
            ].sum(skipna=True)
        )
        gross_exposure = non_cash_df[COL_MARKET_VALUE].abs().sum(skipna=True)

        position_count = int(
            df.loc[~cash_mask, COL_TICKER].nunique(dropna=True)
        )
        average_position_size = gross_exposure / position_count if position_count else 0.0

    trailing_returns = _build_portfolio_return_series(history_df)
    one_year_sharpe = compute_sharpe_ratio(trailing_returns) if not trailing_returns.empty else None
    one_year_sortino = compute_sortino_ratio(trailing_returns) if not trailing_returns.empty else None
    one_year_turnover = _compute_one_year_portfolio_turnover(history_df)

    with st.container(border=True):
        st.markdown('<div class="summary-panel-marker"></div>', unsafe_allow_html=True)
        row_1 = st.columns(4)
        row_1[0].metric("Total Long Exposure", _format_currency(total_long_exposure))
        row_1[1].metric("Cash", _format_currency(cash_value))
        row_1[2].metric("Gross Exposure", _format_currency(gross_exposure))
        row_1[3].metric("Positions", f"{position_count:,}")

        row_2 = st.columns(4)
        row_2[0].metric("Average Position Size", _format_currency(average_position_size))
        row_2[1].metric("1 Year Portfolio Turnover", _format_percent(one_year_turnover / 100.0 if one_year_turnover is not None else None))
        row_2[2].metric("1 Year Sharpe Ratio", _format_number(one_year_sharpe))
        row_2[3].metric("1 Year Sortino Ratio", _format_number(one_year_sortino))


def render_performance_dashboard(snapshot_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    st.subheader("Performance Dashboard")

    current_total_market_value = float(
        pd.to_numeric(snapshot_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)
    ) if not snapshot_df.empty else 0.0

    ret_1y = _compute_trailing_return(history_df, 365)
    ret_1m = _compute_trailing_return(history_df, 30)
    ret_1w = _compute_trailing_return(history_df, 7)
    ret_daily = _compute_daily_return(history_df, current_total_market_value)
    pnl_1y = _compute_trailing_pnl(history_df, 365)
    pnl_1m = _compute_trailing_pnl(history_df, 30)
    pnl_1w = _compute_trailing_pnl(history_df, 7)
    pnl_daily = _compute_daily_pnl(history_df, current_total_market_value)
    latest_business_date = _latest_history_date(history_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1 Year Return", _format_percent(ret_1y))
    c2.metric("1 Month Return", _format_percent(ret_1m))
    c3.metric("1 Week Return", _format_percent(ret_1w))
    c4.metric("Daily Return", _format_percent(ret_daily))

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("1 Year P&L", _format_currency(pnl_1y))
    p2.metric("1 Month P&L", _format_currency(pnl_1m))
    p3.metric("1 Week P&L", _format_currency(pnl_1w))
    p4.metric("Daily P&L", _format_currency(pnl_daily))

    if latest_business_date is not None:
        st.caption(
            f"Daily return and daily P&L are as of {latest_business_date.strftime('%A, %B %d, %Y')}, "
            "the latest business day in the portfolio history."
        )


def render_team_allocation(snapshot_df: pd.DataFrame) -> None:
    st.subheader("CURRENT ALLOCATION BY TEAM")

    team_summary = _build_complete_team_summary(snapshot_df)
    if team_summary.empty:
        st.info("No team allocation data available.")
        return

    display_df = team_summary.copy()
    display_df["sp500_weight"] = display_df[COL_TEAM].map(TEAM_SP500_WEIGHTS).fillna(0.0)
    display_df["sector_tilt"] = (
        pd.to_numeric(display_df[COL_WEIGHT], errors="coerce").fillna(0.0)
        - pd.to_numeric(display_df["sp500_weight"], errors="coerce").fillna(0.0)
    )

    formatted_df = display_df.rename(
        columns={
            COL_TEAM: "Team",
            COL_MARKET_VALUE: "Market Value",
            COL_WEIGHT: "Current Weight",
            "position_count": "Positions",
            "sp500_weight": "S&P 500 Weight",
            "sector_tilt": "Sector Tilt",
        }
    ).copy()
    formatted_df["Market Value"] = formatted_df["Market Value"].map(_format_currency)
    formatted_df["Current Weight"] = formatted_df["Current Weight"].map(_format_percent)
    formatted_df["S&P 500 Weight"] = formatted_df["S&P 500 Weight"].map(_format_percent)
    formatted_df["Sector Tilt"] = formatted_df["Sector Tilt"].map(_format_percent)
    formatted_df["Positions"] = pd.to_numeric(formatted_df["Positions"], errors="coerce").fillna(0).astype(int).map(lambda x: f"{x:,}")

    st.dataframe(
        formatted_df[
            ["Team", "Market Value", "Current Weight", "S&P 500 Weight", "Sector Tilt", "Positions"]
        ].style.set_properties(
            **{
                "text-align": "left",
                "background-color": "#07111a",
                "color": "#f8fafc",
            }
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "left"),
                        ("background-color", "#07111a"),
                        ("color", "#f8fafc"),
                    ],
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    plot_df = team_summary.loc[
        pd.to_numeric(team_summary[COL_MARKET_VALUE], errors="coerce").fillna(0.0) != 0
    ].copy()

    if plot_df.empty:
        st.info("No allocation data available for charting.")
        return

    holdings_plot_df = snapshot_df.copy()
    holdings_plot_df[COL_MARKET_VALUE] = pd.to_numeric(holdings_plot_df[COL_MARKET_VALUE], errors="coerce")
    holdings_plot_df = holdings_plot_df.loc[~holdings_plot_df.apply(_is_cash_like_row, axis=1)].copy()
    holdings_plot_df = holdings_plot_df.sort_values(COL_MARKET_VALUE, ascending=False).head(10)
    combo_fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=("CURRENT PORTFOLIO ALLOCATION", "TOP 10 HOLDINGS BY MARKET VALUE"),
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.12,
    )

    combo_fig.add_trace(
        go.Pie(
            labels=plot_df[COL_TEAM],
            values=pd.to_numeric(plot_df[COL_MARKET_VALUE], errors="coerce").fillna(0.0),
            showlegend=False,
            marker=dict(colors=[TEAM_COLORS.get(team, "#94a3b8") for team in plot_df[COL_TEAM]]),
            textfont=dict(color="#f8fafc"),
            hoverlabel=dict(font=dict(color="#f8fafc")),
            sort=False,
        ),
        row=1,
        col=1,
    )

    for _, row in holdings_plot_df.iterrows():
        team = str(row.get(COL_TEAM, "")).strip()
        combo_fig.add_trace(
            go.Bar(
                x=[float(pd.to_numeric(pd.Series([row.get(COL_MARKET_VALUE)]), errors="coerce").fillna(0.0).iloc[0])],
                y=[str(row.get(COL_TICKER, ""))],
                orientation="h",
                name=team,
                showlegend=False,
                marker=dict(color=TEAM_COLORS.get(team, "#94a3b8")),
                hoverlabel=dict(font=dict(color="#f8fafc")),
            ),
            row=1,
            col=2,
        )

    combo_fig.update_annotations(font=dict(color="#f8fafc", size=16), yshift=12)
    combo_fig.update_xaxes(title_text="MARKET VALUE", color="#f8fafc", row=1, col=2)
    combo_fig.update_yaxes(title_text="TICKER", color="#f8fafc", row=1, col=2, categoryorder="total ascending")
    combo_fig.update_layout(
        paper_bgcolor="#07111a",
        plot_bgcolor="#07111a",
        font=dict(color="#f8fafc"),
        showlegend=False,
        margin=dict(t=125, b=40),
    )

    st.plotly_chart(combo_fig, use_container_width=True)

    legend_html = "".join(
        f"""
        <div style="display:flex; align-items:center; gap:0.45rem; margin:0.15rem 0.9rem 0.15rem 0;">
            <span style="width:12px; height:12px; background:{TEAM_COLORS.get(team, '#94a3b8')}; display:inline-block;"></span>
            <span style="color:#f8fafc;">{team}</span>
        </div>
        """
        for team in settings.display_team_order
    )
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:center; margin-top:0.35rem;">
            {legend_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_aum_chart(history_df: pd.DataFrame) -> None:
    st.subheader("Weekly AUM History")

    if history_df.empty:
        st.info("No historical AUM data available.")
        return

    plot_df = history_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date")

    if plot_df.empty:
        st.info("No historical AUM data available.")
        return

    weekly_df = (
        plot_df.set_index("date")
        .resample("W-FRI")
        .last()
        .dropna(subset=["portfolio_aum"], how="all")
        .reset_index()
    )

    if weekly_df.empty:
        st.info("No historical AUM data available.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly_df["date"],
            y=weekly_df["portfolio_aum"],
            mode="lines+markers",
            name="Portfolio",
        )
    )

    if "sp500_aum" in weekly_df.columns and pd.to_numeric(weekly_df["sp500_aum"], errors="coerce").notna().any():
        fig.add_trace(
            go.Scatter(
                x=weekly_df["date"],
                y=weekly_df["sp500_aum"],
                mode="lines",
                name="S&P 500 (SPY)",
            )
        )

    if "nasdaq_aum" in weekly_df.columns and pd.to_numeric(weekly_df["nasdaq_aum"], errors="coerce").notna().any():
        fig.add_trace(
            go.Scatter(
                x=weekly_df["date"],
                y=weekly_df["nasdaq_aum"],
                mode="lines",
                name="Nasdaq (QQQ)",
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AUM",
        hovermode="x unified",
        paper_bgcolor="#07111a",
        plot_bgcolor="#07111a",
        font=dict(color="#f8fafc"),
        legend=dict(font=dict(color="#f8fafc"), title=dict(font=dict(color="#f8fafc"))),
        margin=dict(t=58),
        annotations=[
            dict(
                text="PORTFOLIO VS BENCHMARK",
                x=0.5,
                xref="paper",
                y=1.08,
                yref="paper",
                showarrow=False,
                xanchor="center",
                font=dict(color="#f8fafc", size=16),
            )
        ],
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology Note"):
        st.write(
            """
            Portfolio AUM is carried forward from the latest uploaded snapshot on or before each date.
            If no newer snapshot exists, the prior snapshot's holdings are carried forward.

            Benchmark AUM lines show what the portfolio's starting AUM would have become if it had
            been invested in SPY or QQQ at the beginning of the displayed history.
            """
        )


def render_full_holdings_snapshot(snapshot_df: pd.DataFrame) -> None:
    st.subheader("CURRENT PORTFOLIO SNAPSHOT")

    if snapshot_df.empty:
        st.info("No holdings available.")
        return

    display_cols = [
        col for col in [
            COL_TICKER,
            COL_TEAM,
            COL_POSITION_SIDE,
            COL_SHARES,
            COL_PRICE,
            COL_MARKET_VALUE,
            COL_WEIGHT,
        ]
        if col in snapshot_df.columns
    ]

    display_df = snapshot_df[display_cols].copy().sort_values(
        by=[COL_TEAM, COL_MARKET_VALUE],
        ascending=[True, False],
    )
    display_df = display_df.rename(
        columns={
            COL_TICKER: "Ticker",
            COL_TEAM: "Team",
            COL_POSITION_SIDE: "Position Side",
            COL_SHARES: "Shares",
            COL_PRICE: "Price",
            COL_MARKET_VALUE: "Market Value",
            COL_WEIGHT: "Weight",
        }
    )

    if "Shares" in display_df.columns:
        display_df["Shares"] = display_df["Shares"].map(_format_number)
    if "Price" in display_df.columns:
        display_df["Price"] = display_df["Price"].map(_format_currency)
    if "Market Value" in display_df.columns:
        display_df["Market Value"] = display_df["Market Value"].map(_format_currency)
    if "Weight" in display_df.columns:
        display_df["Weight"] = display_df["Weight"].map(_format_percent)

    st.dataframe(
        display_df.style.set_properties(**{"text-align": "left"}).set_table_styles(
            [{"selector": "th", "props": [("text-align", "left")]}]
        ),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="Total Fund View", layout="wide")
    apply_page_theme()
    st.title("Total Fund View")

    snapshot_df = get_master_fund_snapshot()

    if snapshot_df.empty:
        render_empty_state()
        return

    history_df = get_master_fund_history()

    render_header_metrics(snapshot_df, history_df)
    st.divider()

    render_performance_dashboard(snapshot_df, history_df)
    st.divider()

    render_team_allocation(snapshot_df)
    st.divider()

    render_aum_chart(history_df)


if __name__ == "__main__":
    main()
