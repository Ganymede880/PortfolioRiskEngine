"""
Factor Model page for the CMCSIF Portfolio Tracker.

This page uses a custom live four-factor model built from S&P 500
constituents and Yahoo Finance data. These are custom internal factors,
not official Fama-French factors.
"""

from __future__ import annotations

import html
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dataclasses import asdict, is_dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.analytics.exposure as exposure_module
from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices
from src.db.crud import load_position_state
from src.db.session import session_scope
from src.utils.constants import FACTOR_COLORS
from src.utils.ui import apply_app_theme, left_align_dataframe, style_plotly_figure, render_top_nav

COL_TICKER = "ticker"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"
COL_TEAM = "team"
COL_POSITION_SIDE = "position_side"
TEAM_COLORS = {
    "Consumer": "#C6D4FF",
    "E&U": "#7A82AB",
    "F&R": "#307473",
    "Healthcare": "#12664F",
    "TMT": "#2DC2BD",
    "M&I": "#3F3047",
    "Cash": "#7A82AB",
}
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


def _format_number(value, decimals: int = 3):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    formatted = f"{abs(numeric_value):,.{decimals}f}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _format_exposure_share_label(value):
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    absolute_value = abs(numeric_value)
    if absolute_value >= 0.01:
        formatted = f"{absolute_value:.2%}"
    elif absolute_value >= 0.001:
        formatted = f"{absolute_value:.3%}"
    else:
        formatted = f"{absolute_value:.4%}"
    return f"({formatted})" if numeric_value < 0 else formatted


def _render_na_reason(default_message: str, reason: str | None = None) -> None:
    if reason:
        st.info(reason)
    else:
        st.info(default_message)


def _render_collapsible_table(label: str, df: pd.DataFrame) -> None:
    with st.expander(label, expanded=False):
        st.dataframe(left_align_dataframe(df), use_container_width=True, hide_index=True)


def _get_team_color(team_name: str) -> str:
    return TEAM_COLORS.get(str(team_name).strip(), "#7A82AB")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = str(hex_color).strip().lstrip("#")
    if len(cleaned) != 6:
        return 122, 130, 171
    return tuple(int(cleaned[idx: idx + 2], 16) for idx in (0, 2, 4))


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    clamped_alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r}, {g}, {b}, {clamped_alpha:.2f})"


def _render_pod_legend(team_names: list[str]) -> None:
    unique_names = []
    seen = set()
    for team_name in team_names:
        cleaned = str(team_name).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_names.append(cleaned)
    if not unique_names:
        return

    legend_html = "".join(
        (
            "<div style='display:flex;align-items:center;gap:0.45rem;'>"
            f"<span style='width:0.9rem;height:0.9rem;border-radius:0.2rem;"
            f"display:inline-block;background:{html.escape(_get_team_color(team_name))};"
            "border:1px solid rgba(15,23,42,0.10);'></span>"
            f"<span>{html.escape(team_name)}</span>"
            "</div>"
        )
        for team_name in unique_names
    )
    st.markdown(
        (
            "<div style='display:flex;flex-wrap:wrap;gap:1rem 1.5rem;"
            "align-items:center;justify-content:center;margin:0.35rem 0 0.25rem 0;"
            "padding:0.25rem 0;'>"
            f"{legend_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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
def get_factor_model_artifacts(snapshot_df: pd.DataFrame):
    builder = getattr(
        exposure_module,
        "build_factor_analytics_platform",
        getattr(exposure_module, "build_custom_live_factor_model"),
    )
    return builder(snapshot_df)


def _artifacts_to_analytics(artifacts) -> dict:
    if isinstance(artifacts, dict):
        return artifacts
    if is_dataclass(artifacts):
        return asdict(artifacts)
    if hasattr(artifacts, "__dict__"):
        return dict(vars(artifacts))
    return {}


def render_empty_state() -> None:
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_header() -> None:
    st.markdown(
        """
        <style>
        .factor-beta-card {
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

        .factor-beta-card-label {
            color: inherit;
            opacity: 0.8;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }

        .factor-beta-card-value {
            color: inherit;
            font-size: 1.65rem;
            font-weight: 600;
            line-height: 1.2;
        }

        html[data-theme="dark"] .factor-beta-card-label,
        html[data-theme="dark"] .factor-beta-card-value,
        body[data-theme="dark"] .factor-beta-card-label,
        body[data-theme="dark"] .factor-beta-card-value,
        [data-testid="stAppViewContainer"][data-theme="dark"] .factor-beta-card-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .factor-beta-card-value {
            color: #FFFFFF !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Factor Model")


def _render_factor_beta_card(label: str, value: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="factor-beta-card">
            <div class="factor-beta-card-label">{safe_label}</div>
            <div class="factor-beta-card-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_portfolio_factor_beta_snapshot(latest_df: pd.DataFrame, summary_df: pd.DataFrame, reason: str | None = None) -> None:
    st.subheader("Portfolio Factor Betas")
    if latest_df.empty:
        _render_na_reason(
            "N/A: not enough overlapping factor and portfolio return history is available for regression yet.",
            reason,
        )
        return

    latest = latest_df.iloc[0]
    with st.container(border=True):
        row_1 = st.columns(4)
        with row_1[0]:
            _render_factor_beta_card("Market Loading", _format_number(latest.get("beta_mkt")))
        with row_1[1]:
            _render_factor_beta_card("Size Loading", _format_number(latest.get("beta_smb")))
        with row_1[2]:
            _render_factor_beta_card("Momentum Loading", _format_number(latest.get("beta_mom")))
        with row_1[3]:
            _render_factor_beta_card("Value Loading", _format_number(latest.get("beta_val")))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        row_2 = st.columns(4)
        with row_2[0]:
            _render_factor_beta_card("Portfolio Alpha", _format_number(latest.get("alpha")))
        with row_2[1]:
            _render_factor_beta_card("R-Squared", _format_number(latest.get("r_squared")))
        with row_2[2]:
            _render_factor_beta_card("Residual Volatility", _format_percent(latest.get("residual_vol")))
        with row_2[3]:
            _render_factor_beta_card("Regression Observations", _format_number(latest.get("obs_count"), decimals=0))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    st.caption("These are portfolio-level regression outputs estimated from historical portfolio returns versus factor return series.")

    if summary_df.empty:
        _render_na_reason(
            "N/A: regression detail is unavailable.",
            reason,
        )
        return

    display_df = summary_df.copy()
    if "term" in display_df.columns:
        display_df["term"] = display_df["term"].replace(
            {
                "const": "Alpha",
                "beta_mkt": "Portfolio Beta to Market",
                "beta_smb": "Portfolio Beta to Size",
                "beta_mom": "Portfolio Beta to Momentum",
                "beta_val": "Portfolio Beta to Value",
                "alpha": "Alpha",
                "r_squared": "R-Squared",
                "residual_vol": "Residual Volatility",
                "obs_count": "Regression Observations",
            }
        )
    if "coefficient" in display_df.columns:
        display_df["coefficient"] = display_df["coefficient"].map(_format_number)
    if "t_stat" in display_df.columns:
        display_df["t_stat"] = display_df["t_stat"].map(_format_number)
    if "p_value" in display_df.columns:
        display_df["p_value"] = display_df["p_value"].map(_format_number)
    display_df = display_df.rename(
        columns={
            "term": "Portfolio Regression Metric",
            "coefficient": "Value",
            "t_stat": "T-Stat",
            "p_value": "P-Value",
        }
    )
    with st.expander("Portfolio Regression Detail", expanded=False):
        st.dataframe(left_align_dataframe(display_df), use_container_width=True, hide_index=True)


def render_rolling_betas(rolling_betas_df: pd.DataFrame, reason: str | None = None) -> None:
    st.subheader("Factor Loadings")
    if rolling_betas_df.empty:
        _render_na_reason(
            "N/A: factor loadings require enough overlapping return history to estimate betas from the 1-year lookback start date.",
            reason,
        )
        return

    plot_df = rolling_betas_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date")
    start_date = plot_df["date"].max() - pd.Timedelta(days=365)
    plot_df = plot_df.loc[plot_df["date"] >= start_date].copy()
    if plot_df.empty:
        _render_na_reason(
            "N/A: factor loadings require enough overlapping return history to estimate betas from the 1-year lookback start date.",
            reason,
        )
        return

    fig = go.Figure()
    series_map = {
        "beta_mkt": "Market",
        "beta_smb": "Size",
        "beta_mom": "Momentum",
        "beta_val": "Value",
    }
    for column, label in series_map.items():
        if column not in plot_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df[column],
                mode="lines",
                name=label,
                line=dict(color=FACTOR_COLORS[label], width=2.5),
            )
        )

    fig.update_layout(
        title=dict(text="1-YEAR FACTOR LOADINGS", x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="FACTOR LOADING",
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        hovermode="x unified",
        xaxis=dict(range=[start_date, plot_df["date"].max()]),
        margin=dict(t=90, b=90),
    )
    fig = style_plotly_figure(fig, title_text="1-YEAR FACTOR LOADINGS")
    st.plotly_chart(fig, use_container_width=True)


def render_factor_return_chart(factor_returns_df: pd.DataFrame) -> None:
    st.subheader("Factor Returns")
    if factor_returns_df.empty:
        st.info("Factor return history is unavailable.")
        return

    plot_df = factor_returns_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date")

    recent_df = plot_df.loc[plot_df["date"] >= (plot_df["date"].max() - pd.Timedelta(days=365))].copy()
    if recent_df.empty:
        recent_df = plot_df.copy()

    fig = go.Figure()
    series_map = {
        "MKT": "Market",
        "SMB": "Size",
        "MOM": "Momentum",
        "VAL": "Value",
    }
    for column, label in series_map.items():
        if column not in plot_df.columns:
            continue
        cumulative = (1.0 + pd.to_numeric(recent_df[column], errors="coerce").fillna(0.0)).cumprod() - 1.0
        fig.add_trace(
            go.Scatter(
                x=recent_df["date"],
                y=cumulative,
                mode="lines",
                name=label,
                line=dict(color=FACTOR_COLORS[label], width=2.5),
            )
        )

    fig.update_layout(
        title=dict(text="1 YEAR CUMULATIVE FACTOR RETURNS", x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="CUMULATIVE RETURN",
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(t=90, b=90),
    )
    fig = style_plotly_figure(fig, title_text="1 YEAR CUMULATIVE FACTOR RETURNS")
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_return_decomposition(attribution_df: pd.DataFrame) -> None:
    st.subheader("Return Decomposition")
    if attribution_df.empty:
        st.info("Portfolio return decomposition is unavailable.")
        return

    plot_df = attribution_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if plot_df.empty:
        st.info("Portfolio return decomposition is unavailable.")
        return

    start_date = plot_df["date"].max() - pd.Timedelta(days=365)
    plot_df = plot_df.loc[plot_df["date"] >= start_date].copy()
    if len(plot_df) < 20:
        st.info("N/A: return decomposition needs more portfolio and factor observations in the last year.")
        return

    contribution_map = {
        "contribution_val": "Value",
        "contribution_mkt": "Market",
        "residual": "Idiosyncratic",
        "contribution_smb": "Size",
        "contribution_mom": "Momentum",
    }
    missing_cols = [column for column in contribution_map if column not in plot_df.columns]
    if missing_cols:
        st.info(
            "Portfolio return decomposition is unavailable because these required columns are missing: "
            + ", ".join(missing_cols)
        )
        return

    cumulative_df = plot_df[["date"]].copy()
    for column in contribution_map:
        cumulative_df[column] = pd.to_numeric(plot_df[column], errors="coerce").fillna(0.0).cumsum()

    fig = go.Figure()
    for column, label in contribution_map.items():
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
    fig = style_plotly_figure(fig, title_text="1 YEAR CUMULATIVE RETURN DECOMPOSITION")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology Note"):
        st.write(
            """
            This decomposition estimates total fund daily return as the sum of attributed
            **Market**, **Size**, **Momentum**, **Value**, and **Idiosyncratic** return
            contributions over the trailing 1-year window.

            The chart cumulatively sums each day's factor contribution so you can see how
            much of the fund's total return came from each style leg versus unexplained
            residual performance.
            """
        )


def render_drawdown_chart(factor_returns_df: pd.DataFrame, portfolio_returns_df: pd.DataFrame) -> None:
    st.subheader("Drawdown Series")
    if factor_returns_df.empty and portfolio_returns_df.empty:
        st.info("Drawdown data is unavailable.")
        return

    fig = go.Figure()
    if not portfolio_returns_df.empty:
        portfolio = portfolio_returns_df.copy().sort_values("date")
        portfolio["date"] = pd.to_datetime(portfolio["date"], errors="coerce")
        portfolio = portfolio.dropna(subset=["date"]).copy()
        portfolio = portfolio.loc[portfolio["date"] >= (portfolio["date"].max() - pd.Timedelta(days=365))].copy()
        series = pd.to_numeric(portfolio["portfolio_return"], errors="coerce").fillna(0.0)
        wealth = (1.0 + series).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        fig.add_trace(go.Scatter(x=portfolio["date"], y=drawdown, mode="lines", name="Portfolio"))

    if not factor_returns_df.empty:
        factors = factor_returns_df.copy().sort_values("date")
        factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
        factors = factors.dropna(subset=["date"]).copy()
        factors = factors.loc[factors["date"] >= (factors["date"].max() - pd.Timedelta(days=365))].copy()
        for factor in ["MKT", "SMB", "MOM", "VAL"]:
            if factor not in factors.columns:
                continue
            series = pd.to_numeric(factors[factor], errors="coerce").fillna(0.0)
            wealth = (1.0 + series).cumprod()
            drawdown = wealth / wealth.cummax() - 1.0
            fig.add_trace(go.Scatter(x=factors["date"], y=drawdown, mode="lines", name=factor))

    fig.update_layout(
        title=dict(text="PORTFOLIO AND FACTOR DRAWDOWNS", x=0.5, xanchor="center"),
        xaxis_title="DATE",
        yaxis_title="DRAWDOWN",
        legend=dict(title=None),
        hovermode="x unified",
    )
    fig = style_plotly_figure(fig, title_text="PORTFOLIO AND FACTOR DRAWDOWNS")
    st.plotly_chart(fig, use_container_width=True)


def render_factor_attribution(attribution_df: pd.DataFrame, cumulative_attribution_df: pd.DataFrame) -> None:
    st.subheader("Drawdown Attribution")
    if attribution_df.empty:
        st.info("Attribution data is unavailable.")
        return

    plot_df = attribution_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if plot_df.empty:
        st.info("Drawdown attribution data is unavailable.")
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
            "Drawdown attribution is unavailable because these required columns are missing: "
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
        st.info("Drawdown attribution is unavailable for the last year.")
        return

    if max_reconcile_error > 0.02:
        st.warning(
            "Drawdown attribution only approximately reconciles to the underwater curve. "
            f"Maximum episode gap over the full history is {max_reconcile_error:.2%}."
        )

    fig = go.Figure()
    series_map = {
        "contribution_val_since_hwm": "Value",
        "contribution_mkt_since_hwm": "Market",
        "residual_since_hwm": "Idiosyncratic",
        "contribution_smb_since_hwm": "Size",
        "contribution_mom_since_hwm": "Momentum",
    }
    for col, label in series_map.items():
        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=plot_df[col],
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
        legend=dict(title=None, orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5),
        hovermode="x unified",
        margin=dict(t=90, b=90),
    )
    fig = style_plotly_figure(fig, title_text="Drawdown Attribution Since Last High Water Mark")
    st.plotly_chart(fig, use_container_width=True)


def render_holdings_contribution(holdings_factor_contribution_df: pd.DataFrame, reason: str | None = None) -> None:
    st.subheader("Top Contributors to Total Portfolio Factor Exposure")
    if holdings_factor_contribution_df.empty:
        _render_na_reason(
            "N/A: weighted holdings signal contributions are unavailable because holdings signal inputs are missing.",
            reason,
        )
        return

    source_df = holdings_factor_contribution_df.copy()
    display_df = source_df.copy()
    display_df = display_df.rename(
        columns={
            COL_TICKER: "Ticker",
            COL_TEAM: "Pod",
            "weight": "Portfolio Weight",
            "size_contribution": "Size Exposure Contribution",
            "momentum_contribution": "Momentum Exposure Contribution",
            "value_contribution": "Value Exposure Contribution",
            "abs_total_contribution": "Total Absolute Weighted Signal",
        }
    )
    for col in ["Portfolio Weight", "Size Exposure Contribution", "Momentum Exposure Contribution", "Value Exposure Contribution", "Total Absolute Weighted Signal"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(_format_percent)
    display_cols = [
        "Ticker",
        "Pod",
        "Portfolio Weight",
        "Size Exposure Contribution",
        "Momentum Exposure Contribution",
        "Value Exposure Contribution",
        "Total Absolute Weighted Signal",
    ]
    display_cols = [col for col in display_cols if col in display_df.columns]

    chart_specs = [
        ("weight", "TOP 5 MARKET EXPOSURE CONTRIBUTORS", "Market Exposure Share", False),
        ("size_contribution", "TOP 5 SIZE EXPOSURE CONTRIBUTORS", "Share of Absolute Size Exposure", True),
        ("momentum_contribution", "TOP 5 MOMENTUM EXPOSURE CONTRIBUTORS", "Share of Absolute Momentum Exposure", True),
        ("value_contribution", "TOP 5 VALUE EXPOSURE CONTRIBUTORS", "Share of Absolute Value Exposure", True),
    ]
    for row_specs in (chart_specs[:2], chart_specs[2:]):
        columns = st.columns(2)
        for chart_col, (value_col, title_text, axis_label, use_absolute_share) in zip(columns, row_specs):
            with chart_col:
                if value_col not in source_df.columns or COL_TICKER not in source_df.columns:
                    continue
                plot_df = source_df.copy()
                plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
                plot_df = plot_df.dropna(subset=[value_col]).copy()
                if use_absolute_share:
                    total_absolute_exposure = float(plot_df[value_col].abs().sum())
                    if total_absolute_exposure < 1e-12:
                        st.info(f"{axis_label} is unavailable because total absolute factor exposure is approximately zero.")
                        continue
                    plot_df["exposure_share"] = plot_df[value_col].abs() / total_absolute_exposure
                    plot_df = plot_df.assign(raw_contribution=plot_df[value_col])
                    plot_df = plot_df.sort_values(value_col, key=lambda s: s.abs(), ascending=False).head(5)
                else:
                    plot_df = plot_df.loc[plot_df[value_col].abs() > 1e-12].copy()
                    if plot_df.empty:
                        st.info(f"{axis_label} is unavailable because all current holding contributions are approximately zero.")
                        continue
                    total_factor_exposure = float(plot_df[value_col].sum())
                    if abs(total_factor_exposure) < 1e-12:
                        st.info(f"{axis_label} is unavailable because total {axis_label.lower()} is approximately zero.")
                        continue
                    plot_df["exposure_share"] = plot_df[value_col] / total_factor_exposure
                    plot_df = plot_df.assign(raw_contribution=plot_df[value_col])
                    plot_df = plot_df.sort_values("exposure_share", ascending=False).head(5)
                if plot_df.empty:
                    continue
                plot_df = plot_df.sort_values("exposure_share", ascending=True)
                plot_df["pod_color"] = plot_df[COL_TEAM].map(_get_team_color) if COL_TEAM in plot_df.columns else "#7A82AB"
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=plot_df["exposure_share"],
                            y=plot_df[COL_TICKER],
                            orientation="h",
                            marker_color=plot_df["pod_color"],
                            text=[_format_exposure_share_label(value) for value in plot_df["exposure_share"]],
                            textposition="outside",
                            customdata=plot_df[[COL_TEAM, "raw_contribution"]].fillna("").to_numpy() if COL_TEAM in plot_df.columns else plot_df[["raw_contribution"]].to_numpy(),
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Pod: %{customdata[0]}<br>"
                                f"{axis_label}: %{{x:.4%}}<br>"
                                "Raw Contribution: %{customdata[1]:.4%}<extra></extra>"
                            ) if COL_TEAM in plot_df.columns else (
                                f"<b>%{{y}}</b><br>{axis_label}: %{{x:.4%}}<br>"
                                "Raw Contribution: %{customdata[0]:.4%}<extra></extra>"
                            ),
                        )
                    ]
                )
                fig.update_layout(
                    title=dict(text=title_text, x=0.5, xanchor="center"),
                    xaxis_title=axis_label,
                    yaxis_title="Ticker",
                    showlegend=False,
                    margin=dict(t=72, b=24, r=24),
                )
                fig.update_xaxes(tickformat=".2%")
                fig = style_plotly_figure(fig, title_text=title_text)
                st.plotly_chart(fig, use_container_width=True)
    if COL_TEAM in source_df.columns:
        _render_pod_legend(source_df[COL_TEAM].dropna().astype(str).tolist())


def render_factor_correlations(correlation_matrix_df: pd.DataFrame) -> None:
    st.subheader("Factor Correlations")
    if correlation_matrix_df.empty:
        st.info("Factor correlations are unavailable.")
        return
    fig = px.imshow(
        correlation_matrix_df,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="FACTOR CORRELATION HEATMAP",
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
        margin=dict(t=90, b=110),
    )
    fig = style_plotly_figure(fig, title_text="FACTOR CORRELATION HEATMAP")
    st.plotly_chart(fig, use_container_width=True)


def render_multi_horizon_exposures(multi_horizon_exposures_df: pd.DataFrame, reason: str | None = None) -> None:
    if multi_horizon_exposures_df.empty:
        _render_na_reason(
            "N/A: multi-horizon portfolio betas are unavailable because the regression windows do not have enough overlapping return history.",
            reason,
        )
        return


def render_holdings_factor_table(holdings_signals_df: pd.DataFrame, reason: str | None = None) -> None:
    st.subheader("Current Holdings Signals")
    if holdings_signals_df.empty:
        _render_na_reason(
            "N/A: no holdings-level signal descriptors are available.",
            reason,
        )
        return

    display_df = holdings_signals_df.copy()
    rename_map = {
        COL_TICKER: "Ticker",
        COL_TEAM: "Pod",
        COL_POSITION_SIDE: "Position",
        COL_MARKET_VALUE: "Market Value",
        COL_WEIGHT: "Portfolio Weight",
        "market_cap": "Market Cap (Raw Input)",
        "selected_pe": "Selected PE (Raw Input)",
        "earnings_yield": "Value Signal",
        "momentum_12_1": "Momentum Signal (12-1 Return)",
        "size_position": "Size Percentile",
        "value_position": "Value Percentile",
        "momentum_position": "Momentum Percentile",
    }
    display_df = display_df.rename(columns=rename_map)

    for column in ["Market Value", "Market Cap"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(_format_currency)
    for column in ["Portfolio Weight", "Value Signal", "Momentum Signal (12-1 Return)", "Size Percentile", "Value Percentile", "Momentum Percentile"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(_format_percent)
    if "Selected PE (Raw Input)" in display_df.columns:
        display_df["Selected PE (Raw Input)"] = display_df["Selected PE (Raw Input)"].map(_format_number)

    wanted_cols = [
        "Ticker",
        "Pod",
        "Position",
        "Market Value",
        "Portfolio Weight",
        "Market Cap (Raw Input)",
        "Selected PE (Raw Input)",
        "Value Signal",
        "Momentum Signal (12-1 Return)",
        "Size Percentile",
        "Value Percentile",
        "Momentum Percentile",
    ]
    wanted_cols = [column for column in wanted_cols if column in display_df.columns]


def render_notes(notes: list[str]) -> None:
    with st.expander("Methodology Notes"):
        for note in notes:
            st.write(f"- {note}")
        st.write(
            "- Factors rebalance monthly using top 10% / bottom 10% equal-weight long-short portfolios."
        )
        st.write(
            "- Momentum uses prior 12-month return excluding the most recent month."
        )
        st.write(
            "- Value uses earnings yield, defined as 1 / PE, while excluding invalid PE observations."
        )


def main() -> None:
    st.set_page_config(page_title="Factor Model", layout="wide")
    apply_app_theme()
    render_top_nav()
    render_header()


    snapshot_df = get_base_snapshot()
    if snapshot_df.empty:
        render_empty_state()
        return

    artifacts = get_factor_model_artifacts(snapshot_df)
    analytics = _artifacts_to_analytics(artifacts)
    holdings_signals = analytics.get("holdings_signals", {}) or {}
    portfolio_factor_betas = analytics.get("portfolio_factor_betas", {}) or {}

    render_portfolio_factor_beta_snapshot(
        portfolio_factor_betas.get("latest", pd.DataFrame()),
        portfolio_factor_betas.get("regression_summary", pd.DataFrame()),
        portfolio_factor_betas.get("reason"),
    )
    st.divider()

    render_rolling_betas(
        portfolio_factor_betas.get("rolling", pd.DataFrame()),
        portfolio_factor_betas.get("reason"),
    )
    st.divider()

    render_factor_return_chart(analytics.get("factor_returns", pd.DataFrame()))
    render_portfolio_return_decomposition(analytics.get("attribution", pd.DataFrame()))
    st.divider()

    render_holdings_contribution(
        holdings_signals.get("weighted_signal_contributions", pd.DataFrame()),
        holdings_signals.get("reason"),
    )
    st.divider()

    render_factor_correlations(analytics.get("factor_correlations", pd.DataFrame()))

    render_multi_horizon_exposures(
        portfolio_factor_betas.get("multi_horizon", pd.DataFrame()),
        portfolio_factor_betas.get("reason"),
    )
    render_notes(analytics.get("notes", []))


if __name__ == "__main__":
    main()
