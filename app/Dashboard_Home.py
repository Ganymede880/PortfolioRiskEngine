"""
Dashboard Home - Executive landing page for the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.ledger import apply_cash_ledger_entries_to_positions, apply_trades_to_positions
from src.analytics.exposure import build_portfolio_return_history
from src.analytics.performance import compute_sharpe_ratio, prepare_flow_adjusted_history
from src.analytics.portfolio import build_current_portfolio_snapshot, summarize_by_team, summarize_total_portfolio
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import (
    get_latest_position_state_date,
    load_all_portfolio_snapshots,
    load_cash_ledger,
    load_position_state,
    load_trade_receipts,
)
from src.db.session import init_db, session_scope
from src.utils.ui import (
    apply_app_theme,
    apply_summary_ui_theme,
    render_summary_card,
    render_summary_status_banner,
    render_top_nav,
)


COL_DATE = "as_of_date"
COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
COL_MARKET_VALUE = "market_value"
MAX_RETURN_LOOKBACK_DAYS = 365
RETURN_LOOKBACK_BUFFER_DAYS = 7
EXTERNAL_FLOW_ACTIVITY_TYPES = {"SECTOR_REBALANCE", "PORTFOLIO_LIQUIDATION"}


def _format_currency(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    return f"(${abs(numeric_value):,.2f})" if numeric_value < 0 else f"${numeric_value:,.2f}"


def _format_percent(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):.2%}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _format_number(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):,.2f}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.Timestamp(value).strftime("%b %d, %Y")


def _value_class(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    if float(value) > 0:
        return "positive"
    if float(value) < 0:
        return "negative"
    return ""


def render_graphic_pattern(data_as_of_text: str) -> None:
    st.markdown(
        f"""
        <div style="
            position: relative;
            overflow: hidden;
            border-radius: 24px;
            min-height: 220px;
            margin: 0.35rem 0 0.65rem 0;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background:
                radial-gradient(circle at 18% 28%, rgba(45, 194, 189, 0.22), transparent 24%),
                radial-gradient(circle at 78% 24%, rgba(122, 130, 171, 0.24), transparent 26%),
                radial-gradient(circle at 62% 72%, rgba(18, 102, 79, 0.26), transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.9));
            box-shadow: 0 20px 60px rgba(15, 23, 42, 0.22);
        ">
            <div style="
                position: absolute;
                inset: 0;
                background-image:
                    linear-gradient(rgba(148, 163, 184, 0.10) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(148, 163, 184, 0.10) 1px, transparent 1px);
                background-size: 34px 34px;
                mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.18));
                -webkit-mask-image: linear-gradient(to bottom, rgba(0, 0, 0, 0.92), rgba(0, 0, 0, 0.18));
            "></div>
            <div style="
                position: absolute;
                width: 380px;
                height: 380px;
                right: -82px;
                top: -108px;
                border-radius: 50%;
                border: 1px solid rgba(191, 219, 254, 0.16);
                box-shadow:
                    0 0 0 38px rgba(191, 219, 254, 0.06),
                    0 0 0 96px rgba(45, 194, 189, 0.06);
            "></div>
            <div style="
                position: absolute;
                width: 220px;
                height: 220px;
                left: 9%;
                bottom: -118px;
                transform: rotate(18deg);
                border-radius: 36px;
                border: 1px solid rgba(255, 255, 255, 0.10);
                background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02));
            "></div>
            <div style="
                position: relative;
                z-index: 2;
                padding: 1.55rem 1.8rem;
                max-width: 660px;
                color: #E2E8F0;
            ">
                <div style="
                    display: inline-block;
                    padding: 0.32rem 0.72rem;
                    margin-bottom: 0.75rem;
                    border-radius: 999px;
                    background: rgba(15, 118, 110, 0.20);
                    border: 1px solid rgba(94, 234, 212, 0.20);
                    font-size: 0.8rem;
                    letter-spacing: 0.08em;
                    font-weight: 700;
                ">CMCSIF PORTFOLIO TRACKER</div>
                <div style="
                    font-size: 1.88rem;
                    line-height: 1.07;
                    font-weight: 700;
                    margin-bottom: 0.48rem;
                ">A live map of the fund, built for quick reads and effective decisions.</div>
                <div style="
                    font-size: 0.98rem;
                    line-height: 1.48;
                    color: rgba(226, 232, 240, 0.86);
                    max-width: 520px;
                ">
                    Explore exposures, attribution, risk, holdings, and activity from one place.
                </div>
                <div style="
                    margin-top: 0.95rem;
                    font-size: 0.84rem;
                    color: rgba(226, 232, 240, 0.72);
                    letter-spacing: 0.03em;
                ">{data_as_of_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ownership_banner() -> None:
    st.markdown(
        """
        <div style="
            position: relative;
            overflow: hidden;
            margin: 0.65rem 0 0.2rem 0;
            padding: 1.05rem 1.2rem;
            border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background:
                radial-gradient(circle at 16% 22%, rgba(45, 194, 189, 0.18), transparent 24%),
                radial-gradient(circle at 82% 30%, rgba(122, 130, 171, 0.16), transparent 26%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(17, 24, 39, 0.86));
            box-shadow: 0 18px 46px rgba(15, 23, 42, 0.16);
            color: rgba(226, 232, 240, 0.94);
        ">
            <div style="
                position: absolute;
                width: 220px;
                height: 220px;
                right: -88px;
                top: -120px;
                border-radius: 50%;
                box-shadow:
                    0 0 0 30px rgba(191, 219, 254, 0.05),
                    0 0 0 76px rgba(45, 194, 189, 0.05);
                border: 1px solid rgba(191, 219, 254, 0.14);
            "></div>
            <div style="position: relative; z-index: 1;">
                <div style="
                    display: inline-block;
                    padding: 0.26rem 0.62rem;
                    border-radius: 999px;
                    margin-bottom: 0.52rem;
                    background: rgba(15, 118, 110, 0.18);
                    border: 1px solid rgba(94, 234, 212, 0.18);
                    font-size: 0.74rem;
                    letter-spacing: 0.08em;
                    font-weight: 700;
                ">PLATFORM STEWARDSHIP</div>
                <div style="
                    font-size: 1.08rem;
                    line-height: 1.2;
                    font-weight: 700;
                    margin-bottom: 0.25rem;
                ">Built and maintained by Kiefer Tierling</div>
                <div style="
                    font-size: 0.92rem;
                    line-height: 1.45;
                    color: rgba(226, 232, 240, 0.74);
                    max-width: 520px;
                ">Find me on Github at Ganymede880.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cash_like_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    ticker = df.get(COL_TICKER, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    team = df.get(COL_TEAM, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    position_side = df.get(COL_POSITION_SIDE, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    return ticker.isin({"CASH", "EUR", "GBP", "NOGXX"}) | team.eq("CASH") | position_side.eq("CASH")


def _apply_position_values(snapshot_positions_df: pd.DataFrame, price_map: dict[str, float]) -> float:
    if snapshot_positions_df.empty:
        return 0.0

    df = snapshot_positions_df.copy()
    df[COL_SHARES] = pd.to_numeric(df[COL_SHARES], errors="coerce").fillna(0.0)
    ticker = df.get(COL_TICKER, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    position_side = df.get(COL_POSITION_SIDE, pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    cash_mask = _cash_like_mask(df)

    cash_total = float(df.loc[cash_mask, COL_SHARES].sum()) if cash_mask.any() else 0.0
    investable_mask = ~cash_mask
    if not investable_mask.any():
        return cash_total

    market_values = df.loc[investable_mask, COL_SHARES] * ticker.loc[investable_mask].map(price_map)
    market_values = market_values.where(position_side.loc[investable_mask].ne("SHORT"), -market_values)
    return cash_total + float(pd.to_numeric(market_values, errors="coerce").fillna(0.0).sum())


def _transition_positions_for_day(
    active_positions_df: pd.DataFrame | None,
    snapshot_for_day: pd.DataFrame | None,
    trades_today: pd.DataFrame | None,
    cash_today: pd.DataFrame | None,
    price_map: dict[str, float],
) -> tuple[pd.DataFrame | None, float]:
    net_external_flow = 0.0
    expected_positions_df = active_positions_df.copy() if active_positions_df is not None else None

    if expected_positions_df is not None and trades_today is not None and not trades_today.empty:
        expected_positions_df, _ = apply_trades_to_positions(
            base_positions_df=expected_positions_df,
            trades_df=trades_today,
        )

    if cash_today is not None and not cash_today.empty:
        net_external_flow = float(pd.to_numeric(cash_today["amount"], errors="coerce").fillna(0.0).sum())
        if expected_positions_df is not None:
            expected_positions_df = apply_cash_ledger_entries_to_positions(
                positions_df=expected_positions_df,
                cash_entries_df=cash_today,
            )

    if snapshot_for_day is not None:
        return snapshot_for_day.copy(), net_external_flow

    return expected_positions_df, net_external_flow


def _build_price_matrix(raw_price_history: pd.DataFrame, business_dates: pd.DatetimeIndex) -> pd.DataFrame:
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


def _determine_history_start_date(oldest_snapshot_date: pd.Timestamp, today: pd.Timestamp) -> pd.Timestamp:
    return max(
        oldest_snapshot_date.normalize(),
        today - pd.Timedelta(days=MAX_RETURN_LOOKBACK_DAYS + RETURN_LOOKBACK_BUFFER_DAYS),
    )


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_home_snapshot_data() -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    with session_scope() as session:
        latest_state_date = get_latest_position_state_date(session)
        position_state_df = load_position_state(session, as_of_date=latest_state_date)
        trades_df = pd.DataFrame()
        cash_df = pd.DataFrame()

        if latest_state_date is not None:
            start_date = pd.to_datetime(latest_state_date) + pd.Timedelta(days=1)
            end_date = pd.Timestamp.today().normalize().date()
            trades_df = load_trade_receipts(
                session=session,
                start_date=start_date.date(),
                end_date=end_date,
            )
            cash_df = load_cash_ledger(
                session=session,
                start_date=start_date.date(),
                end_date=end_date,
            )

    latest_trade_date = pd.to_datetime(trades_df["trade_date"], errors="coerce").max() if not trades_df.empty else pd.NaT

    if position_state_df.empty:
        return pd.DataFrame(), pd.to_datetime(latest_state_date, errors="coerce"), pd.to_datetime(latest_trade_date, errors="coerce")

    carried_positions_df = position_state_df.copy()
    if not trades_df.empty:
        carried_positions_df, _ = apply_trades_to_positions(
            base_positions_df=carried_positions_df,
            trades_df=trades_df,
        )

    if not cash_df.empty:
        carried_positions_df = apply_cash_ledger_entries_to_positions(
            positions_df=carried_positions_df,
            cash_entries_df=cash_df,
        )

    tickers = (
        carried_positions_df[COL_TICKER]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("") & s.str.upper().ne("CASH")]
        .unique()
        .tolist()
    )
    latest_prices_df, _ = fetch_latest_prices(tickers)
    snapshot_df = build_current_portfolio_snapshot(carried_positions_df, latest_prices_df)
    return (
        snapshot_df,
        pd.to_datetime(latest_state_date, errors="coerce"),
        pd.to_datetime(latest_trade_date, errors="coerce"),
    )


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_home_history() -> pd.DataFrame:
    history_df = build_portfolio_return_history(
        lookback_days=MAX_RETURN_LOOKBACK_DAYS + RETURN_LOOKBACK_BUFFER_DAYS,
    )
    if history_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow"])
    return history_df[["date", "portfolio_aum", "net_external_flow"]].copy()


def _prepare_portfolio_history(history_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_flow_adjusted_history(
        history_df=history_df,
        value_column="portfolio_aum",
        flow_column="net_external_flow",
    )


def _compute_trailing_return(history_df: pd.DataFrame, days: int) -> float | None:
    prepared = _prepare_portfolio_history(history_df)
    if prepared.empty:
        return None

    df = prepared.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        return None

    latest_date = df["date"].iloc[-1]
    target_date = latest_date - pd.Timedelta(days=days)
    prior_candidates = df.loc[df["date"] <= target_date].copy()
    if prior_candidates.empty:
        return None

    prior_date = prior_candidates["date"].iloc[-1]
    period_returns = pd.to_numeric(
        df.loc[df["date"] > prior_date, "performance_return"],
        errors="coerce",
    ).dropna()
    if period_returns.empty:
        return None
    return float((1.0 + period_returns).prod() - 1.0)


def _compute_live_daily_metrics(snapshot_df: pd.DataFrame) -> tuple[float | None, float | None]:
    if snapshot_df.empty or COL_MARKET_VALUE not in snapshot_df.columns:
        return None, None

    current_total_market_value = float(
        pd.to_numeric(snapshot_df[COL_MARKET_VALUE], errors="coerce").fillna(0.0).sum()
    )
    if current_total_market_value == 0:
        return None, None

    investable_df = snapshot_df.loc[~_cash_like_mask(snapshot_df)].copy()
    if investable_df.empty:
        return 0.0, 0.0

    tickers = (
        investable_df[COL_TICKER]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )
    if not tickers:
        return None, None

    history_prices_df = fetch_multiple_price_histories(tickers=tickers, lookback_days=10)
    if history_prices_df.empty:
        return None, None

    price_history = history_prices_df.copy()
    price_history["date"] = pd.to_datetime(price_history["date"], errors="coerce")
    price_history["ticker"] = price_history["ticker"].astype(str).str.strip().str.upper()
    price_history["adj_close"] = pd.to_numeric(price_history.get("adj_close"), errors="coerce")
    price_history["close"] = pd.to_numeric(price_history.get("close"), errors="coerce")
    price_history["px"] = price_history["adj_close"]
    missing_adj_mask = price_history["px"].isna()
    price_history.loc[missing_adj_mask, "px"] = price_history.loc[missing_adj_mask, "close"]
    price_history = price_history.dropna(subset=["date", "ticker", "px"]).sort_values(["ticker", "date"]).copy()
    if price_history.empty:
        return None, None

    today = pd.Timestamp.today().normalize()
    prior_close_candidates = price_history.loc[price_history["date"].dt.normalize() < today, "date"]
    prior_close_date = prior_close_candidates.max() if not prior_close_candidates.empty else price_history["date"].max()
    if pd.isna(prior_close_date):
        return None, None

    prior_close_map = (
        price_history.loc[price_history["date"].dt.normalize() <= pd.Timestamp(prior_close_date).normalize()]
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")["px"]
        .to_dict()
    )
    if not prior_close_map:
        return None, None

    prior_close_total_market_value = _apply_position_values(snapshot_df, prior_close_map)
    if prior_close_total_market_value == 0:
        return None, None

    daily_pnl = current_total_market_value - prior_close_total_market_value
    daily_return = daily_pnl / prior_close_total_market_value
    return float(daily_return), float(daily_pnl)


def _compute_current_drawdown(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return None
    wealth = (1.0 + clean).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    if drawdown.empty:
        return None
    return float(drawdown.iloc[-1])


def _compute_historical_var_95(return_series: pd.Series) -> float | None:
    clean = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(0.05))


def _build_home_metrics(snapshot_df: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, object]:
    portfolio_summary = summarize_total_portfolio(snapshot_df) if not snapshot_df.empty else {}
    prepared_history = _prepare_portfolio_history(history_df)
    return_series = (
        pd.to_numeric(prepared_history["performance_return"], errors="coerce").dropna()
        if not prepared_history.empty and "performance_return" in prepared_history.columns
        else pd.Series(dtype="float64")
    )

    daily_return, daily_pnl = _compute_live_daily_metrics(snapshot_df)
    if daily_return is None and not return_series.empty:
        daily_return = float(return_series.iloc[-1])
    if daily_pnl is None and not prepared_history.empty and "performance_pnl" in prepared_history.columns:
        valid_daily_pnl = pd.to_numeric(prepared_history["performance_pnl"], errors="coerce").dropna()
        if not valid_daily_pnl.empty:
            daily_pnl = float(valid_daily_pnl.iloc[-1])

    total_market_value = portfolio_summary.get("total_market_value")
    cash_value = portfolio_summary.get("cash_value")
    active_positions = 0
    if not snapshot_df.empty:
        active_positions = int(
            snapshot_df.loc[~_cash_like_mask(snapshot_df), COL_TICKER]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        )

    return {
        "total_market_value": total_market_value,
        "daily_pnl": daily_pnl,
        "cash_value": cash_value,
        "active_positions": active_positions,
        "one_year_return": _compute_trailing_return(history_df, 365),
        "current_drawdown": _compute_current_drawdown(return_series),
        "one_year_sharpe": compute_sharpe_ratio(return_series) if len(return_series) >= 2 else None,
        "historical_var_95": _compute_historical_var_95(return_series),
        "portfolio_summary": portfolio_summary,
    }


def _build_portfolio_status_insights(snapshot_df: pd.DataFrame, metrics: dict[str, object]) -> list[str]:
    one_year_return = metrics.get("one_year_return")
    if one_year_return is None or pd.isna(one_year_return):
        insight_1 = "One-year portfolio performance is not available yet."
    else:
        direction = "up" if float(one_year_return) >= 0 else "down"
        insight_1 = f"Portfolio is {direction} {_format_percent(one_year_return)} over the last year."

    concentration_insight = "Largest current top-level risk is not available from the current dataset."
    if not snapshot_df.empty:
        team_summary = summarize_by_team(snapshot_df)
        team_summary = team_summary.loc[
            team_summary[COL_TEAM].astype(str).str.strip().str.upper().ne("CASH")
        ].copy()
        if not team_summary.empty:
            team_summary["weight"] = pd.to_numeric(team_summary["weight"], errors="coerce")
            team_summary = team_summary.dropna(subset=["weight"]).sort_values("weight", ascending=False)
            if not team_summary.empty:
                top_team = str(team_summary.iloc[0][COL_TEAM]).strip()
                top_weight = float(team_summary.iloc[0]["weight"])
                concentration_insight = (
                    f"Largest current top-level risk is concentration in {top_team} at {_format_percent(top_weight)} of portfolio value."
                )

    total_market_value = metrics.get("total_market_value")
    cash_value = metrics.get("cash_value")
    active_positions = int(metrics.get("active_positions", 0) or 0)
    cash_weight = None
    if total_market_value not in (None, 0) and cash_value is not None and not pd.isna(total_market_value):
        total_mv = float(total_market_value)
        if total_mv != 0:
            cash_weight = float(cash_value) / total_mv

    cash_text = _format_percent(cash_weight) if cash_weight is not None else "N/A"
    insight_3 = f"Cash balance is {cash_text} of the portfolio and there are {active_positions:,} active positions."

    return [insight_1, concentration_insight, insight_3]


def render_kpi_section(title: str, cards: list[tuple[str, str, str]]) -> None:
    st.markdown(f'<div class="summary-section-kicker">{title}</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for column, (label, value, value_class) in zip(columns, cards):
        with column:
            render_summary_card(label, value, value_class)


def _count_loaded_pages() -> int:
    pages_dir = Path(__file__).resolve().parent / "pages"
    page_count = len([path for path in pages_dir.glob("*.py") if path.is_file()])
    return page_count + 1  # Include the Home dashboard itself.


def render_system_status_section(metrics: dict[str, object], latest_trade_date: pd.Timestamp | None) -> None:
    status_cards = [
        ("Live AUM", _format_currency(metrics.get("total_market_value")), ""),
        ("Securities Loaded", f"{int(metrics.get('active_positions', 0) or 0):,}", ""),
        ("Last Recorded Trade", _format_date(latest_trade_date), ""),
        ("Pages Fully Loaded", f"{_count_loaded_pages():,}", ""),
    ]
    render_kpi_section("System Status", status_cards)


def render_system_health_bar(metrics: dict[str, object]) -> None:
    portfolio_summary = metrics.get("portfolio_summary", {})
    unpriced_positions = int(portfolio_summary.get("unpriced_positions", 0) or 0)
    pages_loaded = _count_loaded_pages()
    total_market_value = metrics.get("total_market_value")
    is_live = (
        pages_loaded > 0
        and total_market_value is not None
        and not pd.isna(total_market_value)
        and float(total_market_value) > 0
        and unpriced_positions == 0
    )

    border_color = "rgba(74, 222, 128, 0.34)" if is_live else "rgba(248, 113, 113, 0.34)"
    glow_color = "rgba(34, 197, 94, 0.18)" if is_live else "rgba(239, 68, 68, 0.16)"
    badge_bg = "rgba(22, 163, 74, 0.18)" if is_live else "rgba(220, 38, 38, 0.16)"
    badge_border = "rgba(134, 239, 172, 0.26)" if is_live else "rgba(252, 165, 165, 0.24)"
    title = "All Systems Live." if is_live else "Systems Not Fully Loaded."
    detail = (
        f"{pages_loaded:,} pages online and pricing fully loaded."
        if is_live
        else f"{unpriced_positions:,} position{'s' if unpriced_positions != 1 else ''} still need pricing or data is incomplete."
    )

    st.markdown(
        f"""
        <div style="
            margin: 0.85rem 0 0 0;
            padding: 0.9rem 1.05rem;
            border-radius: 18px;
            border: 1px solid {border_color};
            background:
                radial-gradient(circle at 12% 50%, {glow_color}, transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(17, 24, 39, 0.88));
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.14);
            color: rgba(226, 232, 240, 0.94);
        ">
            <div style="
                display: inline-block;
                padding: 0.24rem 0.58rem;
                border-radius: 999px;
                margin-bottom: 0.44rem;
                background: {badge_bg};
                border: 1px solid {badge_border};
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                font-weight: 700;
            ">SYSTEM HEALTH</div>
            <div style="font-size: 1rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.16rem;">{title}</div>
            <div style="font-size: 0.91rem; line-height: 1.42; color: rgba(226, 232, 240, 0.76);">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    render_graphic_pattern("Data as of: N/A")
    render_summary_status_banner(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, then rebuild position state.",
        tone="warning",
    )
    render_ownership_banner()


def main() -> None:
    st.set_page_config(layout="wide")

    apply_app_theme()
    apply_summary_ui_theme()
    render_top_nav()

    init_db()

    snapshot_df, base_state_date, latest_trade_date = get_home_snapshot_data()
    history_df = get_home_history()

    if snapshot_df.empty:
        render_empty_state()
        return

    current_date = pd.Timestamp.today().normalize()
    data_as_of_text = f"Data as of: {current_date.strftime('%B %d, %Y')}"

    metrics = _build_home_metrics(snapshot_df, history_df)

    render_graphic_pattern(data_as_of_text)

    render_system_status_section(metrics, latest_trade_date)
    render_system_health_bar(metrics)
    render_ownership_banner()


if __name__ == "__main__":
    main()
