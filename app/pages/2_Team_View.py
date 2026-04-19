"""
Team View dashboard page for the CMCSIF Portfolio Tracker.

This page mirrors the Total Fund View styling while focusing on a single pod:
- top sleeve summary metrics
- detailed current holdings table with trailing returns
- allocation and P&L contribution charts
- weekly sleeve AUM vs sector benchmark
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.performance import compute_sharpe_ratio, compute_sortino_ratio
from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import load_all_portfolio_snapshots, load_position_state
from src.db.session import session_scope
from src.utils.ui import apply_app_theme, left_align_dataframe


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
    "Consumer": "#38bdf8",
    "E&U": "#f59e0b",
    "F&R": "#10b981",
    "Healthcare": "#f43f5e",
    "TMT": "#8b5cf6",
    "M&I": "#eab308",
    "Cash": "#94a3b8",
}


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


def _apply_team_page_theme() -> None:
    apply_app_theme()
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSelectbox label p,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] label p {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="select"] > div > div,
        [data-testid="stSidebar"] [data-baseweb="select"] div,
        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] span,
        [data-testid="stSidebar"] [data-baseweb="select"] p,
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] [role="combobox"],
        [data-testid="stSidebar"] [role="listbox"],
        [data-testid="stSidebar"] [role="option"],
        [data-testid="stSidebar"] [data-baseweb="select"] [aria-selected="true"] {
            color: #0f172a !important;
            fill: #0f172a !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [role="combobox"],
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            background: #f8fafc !important;
            border-color: #cbd5e1 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] input::placeholder {
            color: #334155 !important;
            -webkit-text-fill-color: #334155 !important;
        }

        div[data-baseweb="popover"] [role="listbox"],
        div[data-baseweb="popover"] [role="option"],
        div[data-baseweb="popover"] [role="option"] div,
        div[data-baseweb="popover"] [role="option"] span,
        div[data-baseweb="popover"] [role="option"] p {
            background: #f8fafc !important;
            color: #0f172a !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.team-summary-marker) {
            background:
                linear-gradient(135deg, rgba(59, 130, 246, 0.48) 0%, rgba(30, 64, 175, 0.34) 42%, rgba(15, 23, 42, 0.88) 78%),
                linear-gradient(180deg, rgba(125, 211, 252, 0.16) 0%, rgba(15, 23, 42, 0.02) 100%);
            border: 1px solid rgba(125, 211, 252, 0.5) !important;
            box-shadow: 0 10px 30px rgba(2, 6, 23, 0.28);
        }
        </style>
        """,
        unsafe_allow_html=True,
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
    if history_df.empty or value_column not in history_df.columns:
        return pd.Series(dtype="float64")

    df = history_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=["date", value_column]).sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        return pd.Series(dtype="float64")

    return pd.to_numeric(df[value_column].pct_change(), errors="coerce").dropna()


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
    snapshots = snapshots.loc[~snapshots.apply(_is_cash_like_row, axis=1)].copy()
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


def _get_team_selector(snapshot_df: pd.DataFrame) -> str:
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
    return st.sidebar.selectbox("Select Team", options=ordered + fallback)


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
    df = df.loc[~df.apply(_is_cash_like_row, axis=1)].copy()
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
        return pd.DataFrame(columns=["date", "team_aum", "benchmark_aum"])

    df = snapshots_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df[COL_TEAM] = df[COL_TEAM].astype(str).str.strip()
    df[COL_TICKER] = df[COL_TICKER].astype(str).str.strip().str.upper()
    df[COL_POSITION_SIDE] = df[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    df[COL_SHARES] = pd.to_numeric(df[COL_SHARES], errors="coerce")
    df = df.dropna(subset=["snapshot_date", COL_TICKER, COL_SHARES]).copy()
    df = df.loc[df[COL_TEAM] == team].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "benchmark_aum"])

    today = pd.Timestamp.today().normalize()
    oldest_snapshot_date = df["snapshot_date"].min().normalize()
    start_date = max(oldest_snapshot_date, today - pd.Timedelta(days=372))
    business_dates = pd.bdate_range(start=start_date, end=today)
    if len(business_dates) == 0:
        return pd.DataFrame(columns=["date", "team_aum", "benchmark_aum"])

    investable_tickers = (
        df[COL_TICKER]
        .loc[~df.apply(_is_cash_like_row, axis=1)]
        .dropna()
        .unique()
        .tolist()
    )
    history_tickers = sorted(set(investable_tickers + ([benchmark_ticker] if benchmark_ticker else [])))
    raw_price_history = fetch_multiple_price_histories(history_tickers, lookback_days=(today - start_date).days + 30)
    price_matrix = _build_price_matrix(raw_price_history, business_dates)

    snapshot_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in df.groupby("snapshot_date")
    }
    snapshot_dates = sorted(snapshot_by_date.keys())
    rows = []

    for dt in business_dates:
        eligible = [d for d in snapshot_dates if d <= dt]
        if not eligible:
            continue

        active_date = eligible[-1]
        active_df = snapshot_by_date[active_date]
        price_map = {}
        if dt in price_matrix.index:
            row = price_matrix.loc[dt]
            price_map = {
                str(ticker).strip().upper(): float(row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(row[ticker])
            }

        team_aum = 0.0
        for _, pos in active_df.iterrows():
            ticker = str(pos[COL_TICKER]).strip().upper()
            side = str(pos[COL_POSITION_SIDE]).strip().upper()
            shares = float(pd.to_numeric(pd.Series([pos[COL_SHARES]]), errors="coerce").fillna(0.0).iloc[0])
            if _is_cash_like_row(pos):
                team_aum += shares
                continue
            px_val = price_map.get(ticker, 0.0)
            if side == "SHORT":
                team_aum -= shares * px_val
            else:
                team_aum += shares * px_val
        rows.append({"date": dt, "team_aum": float(team_aum)})

    history_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if history_df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "benchmark_aum"])

    history_df["benchmark_aum"] = pd.NA
    if benchmark_ticker and benchmark_ticker in price_matrix.columns:
        bench_prices = pd.to_numeric(price_matrix[benchmark_ticker].reindex(history_df["date"]).values, errors="coerce")
        valid = pd.Series(bench_prices).dropna()
        if not valid.empty:
            start_price = float(valid.iloc[0])
            if start_price != 0:
                history_df["benchmark_aum"] = history_df["team_aum"].iloc[0] * (bench_prices / start_price)

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
    investable_df = team_df.loc[~team_df.apply(_is_cash_like_row, axis=1)].copy()
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
    st.title("Team View")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_team_dashboard(team_df: pd.DataFrame, history_df: pd.DataFrame, snapshots_df: pd.DataFrame, team: str) -> None:
    team_market_value = float(pd.to_numeric(team_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)) if not team_df.empty else 0.0
    returns = _build_return_series(history_df, "team_aum")
    team_turnover = _compute_team_turnover(history_df, snapshots_df, team)
    sharpe = compute_sharpe_ratio(returns) if not returns.empty else None
    sortino = compute_sortino_ratio(returns) if not returns.empty else None

    with st.container(border=True):
        st.markdown('<div class="team-summary-marker"></div>', unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].metric("Team Market Value", _format_currency(team_market_value))
        cols[1].metric("1 Year Portfolio Turnover", _format_percent(team_turnover / 100.0 if team_turnover is not None else None))
        cols[2].metric("1 Year Sharpe Ratio", _format_number(sharpe))
        cols[3].metric("1 Year Sortino Ratio", _format_number(sortino))


def render_holdings_table(formatted_df: pd.DataFrame) -> None:
    st.subheader("CURRENT TEAM HOLDINGS")
    if formatted_df.empty:
        st.info("No holdings available for this team.")
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
    investable_df = team_df.loc[~team_df.apply(_is_cash_like_row, axis=1)].copy()
    if investable_df.empty:
        st.info("No team holdings available for charting.")
        return

    pie_fig = px.pie(
        investable_df,
        names=COL_TICKER,
        values=COL_MARKET_VALUE,
        title="CURRENT TEAM ALLOCATION",
        color_discrete_sequence=[TEAM_COLORS.get(team, "#38bdf8")],
    )
    pie_fig.update_layout(
        paper_bgcolor="#07111a",
        plot_bgcolor="#07111a",
        font=dict(color="#f8fafc"),
        title=dict(text="CURRENT TEAM ALLOCATION", font=dict(color="#f8fafc", size=16), x=0.5, xanchor="center"),
        showlegend=False,
        margin=dict(t=100, b=20),
    )

    pnl_df = chart_df.dropna(subset=["pnl_contribution"]).sort_values("pnl_contribution", ascending=False).copy()
    bar_fig = px.bar(
        pnl_df,
        x="pnl_contribution",
        y=COL_TICKER,
        orientation="h",
        title="1 YEAR P&L CONTRIBUTION BY POSITION",
        color_discrete_sequence=[TEAM_COLORS.get(team, "#38bdf8")],
    )
    bar_fig.update_layout(
        paper_bgcolor="#07111a",
        plot_bgcolor="#07111a",
        font=dict(color="#f8fafc"),
        title=dict(text="1 YEAR P&L CONTRIBUTION BY POSITION", font=dict(color="#f8fafc", size=16), x=0.5, xanchor="center"),
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
    st.subheader("WEEKLY TEAM AUM VS SECTOR BENCHMARK")
    if history_df.empty:
        st.info("No historical team AUM data available.")
        return

    weekly_df = (
        history_df.set_index("date")
        .resample("W-FRI")
        .last()
        .dropna(subset=["team_aum"], how="all")
        .reset_index()
    )
    if weekly_df.empty:
        st.info("No historical team AUM data available.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weekly_df["date"],
            y=weekly_df["team_aum"],
            mode="lines+markers",
            name=team,
            line=dict(color=TEAM_COLORS.get(team, "#38bdf8")),
        )
    )
    if "benchmark_aum" in weekly_df.columns and pd.to_numeric(weekly_df["benchmark_aum"], errors="coerce").notna().any():
        fig.add_trace(
            go.Scatter(
                x=weekly_df["date"],
                y=weekly_df["benchmark_aum"],
                mode="lines",
                name=f"{TEAM_BENCHMARK_TICKERS.get(team, 'SPY')} Benchmark",
                line=dict(color="#94a3b8"),
            )
        )

    fig.update_layout(
        paper_bgcolor="#07111a",
        plot_bgcolor="#07111a",
        font=dict(color="#f8fafc"),
        title=dict(text="TEAM VS SECTOR BENCHMARK", font=dict(color="#f8fafc", size=16), x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="AUM",
        legend=dict(font=dict(color="#f8fafc"), title=None),
        margin=dict(t=58),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology Note"):
        st.write(
            f"""
            Weekly team AUM is reconstructed by carrying each authoritative snapshot for **{team}**
            forward until the next snapshot and marking those positions to market with historical prices.

            The benchmark line is scaled to the sleeve's starting AUM using **{TEAM_BENCHMARK_TICKERS.get(team, 'SPY')}**
            as a practical sector proxy.
            """
        )


def main() -> None:
    st.set_page_config(page_title="Team View", layout="wide")
    _apply_team_page_theme()
    st.title("Team View")

    snapshot_df, snapshots_df = get_team_view_data()
    if snapshot_df.empty:
        render_empty_state()
        return

    selected_team = _get_team_selector(snapshot_df)
    team_df = snapshot_df.loc[snapshot_df[COL_TEAM] == selected_team].copy()
    benchmark_ticker = TEAM_BENCHMARK_TICKERS.get(selected_team, "SPY")
    history_df = _build_team_history(selected_team, snapshots_df, benchmark_ticker)
    formatted_holdings_df, chart_df = _build_team_holdings_view(team_df, snapshots_df)

    st.caption(f"Viewing sleeve: {selected_team}")
    render_team_dashboard(team_df, history_df, snapshots_df, selected_team)
    st.divider()
    render_holdings_table(formatted_holdings_df)
    st.divider()
    render_team_charts(team_df, chart_df, selected_team)
    st.divider()
    render_team_history(history_df, selected_team)


if __name__ == "__main__":
    main()
