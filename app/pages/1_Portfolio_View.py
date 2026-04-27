"""
Portfolio View dashboard page for the CMCSIF Portfolio Tracker.

This version includes:
- current top-line portfolio metrics
- performance dashboard (1Y / 1M / 1W / Daily return)
- current pod allocation table + pie chart
- weekly AUM chart for the past year (or back to oldest snapshot)
- SPY / QQQ benchmark AUM comparison, scaled to the portfolio's initial AUM

Important current assumption:
- portfolio holdings are carried forward from the latest uploaded snapshot until the
  next uploaded snapshot, unless a newer snapshot replaces them
- trade receipts are not yet applied into the carried historical series on this page
- benchmark lines use SPY and QQQ as practical proxies for S&P 500 and Nasdaq exposure
"""

from __future__ import annotations

import html
import numpy as np
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import src.analytics.exposure as exposure_module
import src.data.price_fetcher as price_fetcher_module
from src.analytics.portfolio import (
    build_current_portfolio_snapshot,
    summarize_total_portfolio,
)
from src.analytics.performance import (
    build_flow_adjusted_benchmark_series,
    compute_cumulative_return_series,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    prepare_flow_adjusted_history,
)
from src.analytics.team_benchmarks import (
    build_team_benchmark_aum_frame,
    get_team_benchmark_tickers,
)
from src.config.settings import settings
from src.data.price_fetcher import (
    fetch_latest_prices,
    fetch_multiple_price_histories,
    fetch_sp500_sector_group_weights,
)
from src.db.crud import (
    get_latest_position_state_date,
    load_all_portfolio_snapshots,
    load_cash_ledger,
    load_position_state,
    load_trade_receipts,
)
from src.db.session import session_scope
from src.analytics.ledger import apply_cash_ledger_entries_to_positions, apply_trades_to_positions
from src.utils.constants import TEAM_COLORS
from src.utils.ui import apply_app_theme, render_page_title, render_top_nav


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
MAX_RETURN_LOOKBACK_DAYS = 365
RETURN_LOOKBACK_BUFFER_DAYS = 7
EXTERNAL_FLOW_ACTIVITY_TYPES = {"SECTOR_REBALANCE", "PORTFOLIO_LIQUIDATION"}


def _get_team_color(team_name: str) -> str:
    return TEAM_COLORS.get(str(team_name).strip(), "#64748b")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    cleaned = str(hex_color).strip().lstrip("#")
    if len(cleaned) != 6:
        return 100, 116, 139
    return tuple(int(cleaned[idx: idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = [max(0, min(255, int(channel))) for channel in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


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


def apply_page_theme() -> None:
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

        .live-portfolio-card {
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

        .live-portfolio-card-label {
            color: inherit;
            opacity: 0.8;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }

        .live-portfolio-card-value {
            color: inherit;
            font-size: 1.65rem;
            font-weight: 600;
            line-height: 1.2;
        }

        .live-portfolio-card-value.positive {
            color: #15803D;
        }

        .live-portfolio-card-value.negative {
            color: #B91C1C;
        }

        .portfolio-pod-legend-item {
            display:flex;
            align-items:center;
            gap:0.45rem;
        }

        .portfolio-pod-legend-label {
            color: inherit;
            opacity: 0.88;
            font-size: 0.95rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stVerticalBlock"] div:has(> div[data-testid="stDataFrame"]) {
            gap: 0.35rem;
        }

        h3, h4 {
            letter-spacing: 0.01em;
        }

        html[data-theme="dark"] .live-portfolio-card-label,
        html[data-theme="dark"] .live-portfolio-card-value,
        body[data-theme="dark"] .live-portfolio-card-label,
        body[data-theme="dark"] .live-portfolio-card-value,
        [data-testid="stAppViewContainer"][data-theme="dark"] .live-portfolio-card-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .live-portfolio-card-value {
            color: #FFFFFF !important;
        }

        html[data-theme="dark"] .live-portfolio-card-value.positive,
        body[data-theme="dark"] .live-portfolio-card-value.positive,
        [data-testid="stAppViewContainer"][data-theme="dark"] .live-portfolio-card-value.positive {
            color: #4ADE80 !important;
        }

        html[data-theme="dark"] .live-portfolio-card-value.negative,
        body[data-theme="dark"] .live-portfolio-card-value.negative,
        [data-testid="stAppViewContainer"][data-theme="dark"] .live-portfolio-card-value.negative {
            color: #F87171 !important;
        }

        html[data-theme="dark"] .portfolio-pod-legend-label,
        body[data-theme="dark"] .portfolio-pod-legend-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .portfolio-pod-legend-label {
            color: rgba(255, 255, 255, 0.88) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_live_portfolio_card(label: str, value: str, value_class: str = "") -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_value_class = html.escape(value_class.strip())
    value_class_attr = f" {safe_value_class}" if safe_value_class else ""
    st.markdown(
        f"""
        <div class="live-portfolio-card">
            <div class="live-portfolio-card-label">{safe_label}</div>
            <div class="live-portfolio-card-value{value_class_attr}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_custom_pod_legend(team_names: list[str]) -> None:
    unique_teams: list[str] = []
    for team in team_names:
        cleaned = str(team).strip()
        if cleaned and cleaned not in unique_teams:
            unique_teams.append(cleaned)

    if not unique_teams:
        return

    legend_items = "".join(
        f"""
        <div class="portfolio-pod-legend-item">
            <span style="
                display:inline-block;
                width:0.9rem;
                height:0.9rem;
                border-radius:0.25rem;
                background:{html.escape(_get_team_color(team))};
                border:1px solid rgba(15, 23, 42, 0.12);
            "></span>
            <span class="portfolio-pod-legend-label">{html.escape(team)}</span>
        </div>
        """
        for team in unique_teams
    )

    st.markdown(
        f"""
        <div style="
            display:flex;
            flex-wrap:wrap;
            justify-content:center;
            gap:1rem 1.5rem;
            margin-top:0.5rem;
        ">
            {legend_items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_allocation_sunburst_frame(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    working = snapshot_df.copy()
    working[COL_MARKET_VALUE] = pd.to_numeric(working[COL_MARKET_VALUE], errors="coerce")
    working = working.loc[~_cash_like_mask(working)].copy()
    working = working.loc[working[COL_MARKET_VALUE].fillna(0.0) != 0.0].copy()
    if working.empty:
        return pd.DataFrame(columns=["sector", "subsector", COL_MARKET_VALUE])

    working[COL_TEAM] = working[COL_TEAM].astype(str).str.strip()
    working[COL_TICKER] = working[COL_TICKER].astype(str).str.strip().str.upper()
    working = working.rename(columns={COL_TEAM: "sector"})

    def _clean_label(value) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "n/a", "<na>"}:
            return None
        return text

    def _ticker_lookup_keys(ticker: str) -> list[str]:
        normalized = str(ticker).strip().upper()
        keys = [normalized]
        if "-" in normalized:
            base, suffix = normalized.rsplit("-", 1)
            if suffix.isalpha() and len(suffix) == 1:
                keys.append(f"{base}.{suffix}")
        if "." in normalized:
            base, suffix = normalized.rsplit(".", 1)
            if suffix.isalpha() and len(suffix) == 1:
                keys.append(f"{base}-{suffix}")
        seen: set[str] = set()
        return [key for key in keys if key and not (key in seen or seen.add(key))]

    def _build_constituent_subsector_map() -> dict[str, str]:
        constituents_getter = getattr(exposure_module, "get_sp500_constituents", None)
        constituents_df = constituents_getter() if callable(constituents_getter) else pd.DataFrame()
        if constituents_df.empty:
            return {}
        normalized = constituents_df.copy()
        normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
        normalized["gics_sub_industry"] = normalized["gics_sub_industry"].map(_clean_label)
        normalized = normalized.dropna(subset=["ticker", "gics_sub_industry"]).drop_duplicates(subset=["ticker"])
        return dict(zip(normalized["ticker"], normalized["gics_sub_industry"]))

    def _is_etf_like(profile_row: dict[str, str | None], ticker: str) -> bool:
        quote_type = str(profile_row.get("quote_type") or "").strip().upper()
        category = _clean_label(profile_row.get("category"))
        fund_family = _clean_label(profile_row.get("fund_family"))
        known_etfs = {"SPY", "QQQ", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "IGV"}
        return quote_type in {"ETF", "MUTUALFUND"} or bool(category) or bool(fund_family) or ticker in known_etfs

    def _profile_subsector_label(profile_row: dict[str, str | None], fallback_sector: str | None) -> str:
        industry = _clean_label(profile_row.get("industry"))
        category = _clean_label(profile_row.get("category"))
        sector_name = _clean_label(profile_row.get("sector"))
        quote_type = str(profile_row.get("quote_type") or "").strip().upper()

        if industry:
            return industry
        if quote_type in {"ETF", "MUTUALFUND"} and category:
            return category
        if sector_name:
            return sector_name
        if category:
            return category
        if fallback_sector:
            return f"{fallback_sector} Other"
        return "Other"

    constituent_subsector_map = _build_constituent_subsector_map()
    fallback_sector_map = (
        working[["sector", COL_TICKER]]
        .dropna()
        .drop_duplicates(subset=[COL_TICKER])
        .set_index(COL_TICKER)["sector"]
        .astype(str)
        .to_dict()
    )

    holdings_tickers = sorted(working[COL_TICKER].dropna().astype(str).str.strip().str.upper().unique().tolist())
    fetch_live_security_profiles = getattr(price_fetcher_module, "fetch_live_security_profiles", None)
    fetch_etf_top_holdings = getattr(price_fetcher_module, "fetch_etf_top_holdings", None)
    holdings_profiles_df = (
        fetch_live_security_profiles(holdings_tickers)
        if callable(fetch_live_security_profiles)
        else pd.DataFrame(columns=["ticker", "quote_type", "sector", "industry", "category", "fund_family", "long_name", "short_name"])
    )
    profile_map: dict[str, dict[str, str | None]] = {}
    if not holdings_profiles_df.empty:
        for row in holdings_profiles_df.to_dict(orient="records"):
            ticker = str(row.get("ticker", "")).strip().upper()
            if ticker:
                profile_map[ticker] = row

    def _resolve_subsector_map(
        tickers: list[str],
        sector_map: dict[str, str] | None = None,
        extra_profiles: dict[str, dict[str, str | None]] | None = None,
    ) -> dict[str, str]:
        sector_map = sector_map or {}
        combined_profiles = dict(profile_map)
        if extra_profiles:
            combined_profiles.update(extra_profiles)

        missing = sorted(
            {
                ticker
                for ticker in tickers
                if ticker
                and not any(key in constituent_subsector_map for key in _ticker_lookup_keys(ticker))
                and ticker not in combined_profiles
            }
        )
        if missing:
            fetched = (
                fetch_live_security_profiles(missing)
                if callable(fetch_live_security_profiles)
                else pd.DataFrame(columns=["ticker", "quote_type", "sector", "industry", "category", "fund_family", "long_name", "short_name"])
            )
            if not fetched.empty:
                for row in fetched.to_dict(orient="records"):
                    fetched_ticker = str(row.get("ticker", "")).strip().upper()
                    if fetched_ticker:
                        combined_profiles[fetched_ticker] = row
                        profile_map[fetched_ticker] = row

        resolved: dict[str, str] = {}
        for ticker in tickers:
            if not ticker:
                continue
            resolved_label = None
            for key in _ticker_lookup_keys(ticker):
                resolved_label = constituent_subsector_map.get(key)
                if resolved_label:
                    break
            if resolved_label:
                resolved[ticker] = resolved_label
                continue
            profile_row = next(
                (combined_profiles.get(key) for key in _ticker_lookup_keys(ticker) if combined_profiles.get(key)),
                {},
            )
            resolved[ticker] = _profile_subsector_label(profile_row or {}, sector_map.get(ticker))
        return resolved

    etf_mask = working[COL_TICKER].map(lambda ticker: _is_etf_like(profile_map.get(ticker, {}), ticker))
    direct_rows = working.loc[~etf_mask].copy()
    etf_rows = working.loc[etf_mask].copy()

    allocation_rows: list[dict[str, object]] = []
    direct_subsector_map = _resolve_subsector_map(
        direct_rows[COL_TICKER].dropna().astype(str).tolist(),
        sector_map=fallback_sector_map,
    )
    for row in direct_rows.to_dict(orient="records"):
        ticker = str(row.get(COL_TICKER, "")).strip().upper()
        allocation_rows.append(
            {
                "sector": row.get("sector"),
                "subsector": direct_subsector_map.get(ticker, f"{row.get('sector')} Other"),
                COL_MARKET_VALUE: float(pd.to_numeric(pd.Series([row.get(COL_MARKET_VALUE)]), errors="coerce").fillna(0.0).iloc[0]),
            }
        )

    etf_holdings_map: dict[str, pd.DataFrame] = {}
    all_underlying_tickers: set[str] = set()
    for etf_ticker in etf_rows[COL_TICKER].dropna().astype(str).str.strip().str.upper().unique():
        holdings_df = fetch_etf_top_holdings(etf_ticker) if callable(fetch_etf_top_holdings) else pd.DataFrame(columns=["ticker", "weight"])
        etf_holdings_map[etf_ticker] = holdings_df
        if not holdings_df.empty:
            all_underlying_tickers.update(holdings_df["ticker"].dropna().astype(str).str.strip().str.upper().tolist())

    underlying_subsector_map = _resolve_subsector_map(sorted(all_underlying_tickers))
    for row in etf_rows.to_dict(orient="records"):
        sector_name = str(row.get("sector", "")).strip()
        ticker = str(row.get(COL_TICKER, "")).strip().upper()
        market_value = float(pd.to_numeric(pd.Series([row.get(COL_MARKET_VALUE)]), errors="coerce").fillna(0.0).iloc[0])
        holdings_df = etf_holdings_map.get(ticker, pd.DataFrame())
        if holdings_df.empty:
            fallback_label = _resolve_subsector_map([ticker], sector_map={ticker: sector_name}).get(ticker, f"{sector_name} ETF Basket")
            allocation_rows.append({"sector": sector_name, "subsector": fallback_label, COL_MARKET_VALUE: market_value})
            continue

        expanded = holdings_df.copy()
        expanded["subsector"] = expanded["ticker"].map(underlying_subsector_map)
        expanded["subsector"] = expanded["subsector"].fillna(f"{sector_name} Other")
        expanded = expanded.groupby("subsector", as_index=False)["weight"].sum()
        total_weight = float(expanded["weight"].sum())
        if total_weight <= 0:
            allocation_rows.append({"sector": sector_name, "subsector": f"{sector_name} ETF Basket", COL_MARKET_VALUE: market_value})
            continue
        expanded["weight"] = expanded["weight"] / total_weight
        for slice_row in expanded.to_dict(orient="records"):
            allocation_rows.append(
                {
                    "sector": sector_name,
                    "subsector": slice_row["subsector"],
                    COL_MARKET_VALUE: market_value * float(slice_row["weight"]),
                }
            )

    allocation_df = pd.DataFrame(allocation_rows)
    if allocation_df.empty:
        return pd.DataFrame(columns=["sector", "subsector", COL_MARKET_VALUE])

    allocation_df["subsector"] = allocation_df["subsector"].map(_clean_label).fillna("Other")
    return (
        allocation_df.groupby(["sector", "subsector"], as_index=False)[COL_MARKET_VALUE]
        .sum()
        .sort_values([COL_MARKET_VALUE, "sector", "subsector"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _clean_display_label(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a", "<na>"}:
        return None
    return text


def _map_sp500_sector_to_team(gics_sector: str | None) -> str | None:
    sector_name = _clean_display_label(gics_sector)
    if sector_name is None:
        return None

    mapping = {
        "Consumer Discretionary": "Consumer",
        "Consumer Staples": "Consumer",
        "Energy": "E&U",
        "Utilities": "E&U",
        "Financials": "F&R",
        "Real Estate": "F&R",
        "Health Care": "Healthcare",
        "Information Technology": "TMT",
        "Communication Services": "TMT",
        "Industrials": "M&I",
        "Materials": "M&I",
    }
    return mapping.get(sector_name)


def _build_sp500_sunburst_frame() -> pd.DataFrame:
    constituents_getter = getattr(exposure_module, "get_sp500_constituents", None)
    constituents_df = constituents_getter() if callable(constituents_getter) else pd.DataFrame()
    if constituents_df.empty:
        return pd.DataFrame(columns=["sector", "subsector", COL_MARKET_VALUE])

    working = constituents_df.copy()
    working["sector"] = working["gics_sector"].map(_map_sp500_sector_to_team)
    working["subsector"] = working["gics_sub_industry"].map(_clean_display_label)
    working = working.dropna(subset=["sector", "subsector"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["sector", "subsector", COL_MARKET_VALUE])

    live_sp500_weights = fetch_sp500_sector_group_weights()
    grouped = (
        working.groupby(["sector", "subsector"], as_index=False)
        .size()
        .rename(columns={"size": "constituent_count"})
    )
    grouped["sector_weight"] = pd.to_numeric(grouped["sector"].map(live_sp500_weights), errors="coerce")
    grouped = grouped.loc[grouped["sector_weight"].notna() & grouped["sector_weight"].gt(0)].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["sector", "subsector", COL_MARKET_VALUE])

    sector_counts = grouped.groupby("sector")["constituent_count"].transform("sum")
    grouped[COL_MARKET_VALUE] = grouped["sector_weight"] * (grouped["constituent_count"] / sector_counts)
    return (
        grouped[["sector", "subsector", COL_MARKET_VALUE]]
        .sort_values([COL_MARKET_VALUE, "sector", "subsector"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _build_allocation_sunburst_figure(
    allocation_df: pd.DataFrame,
    value_label: str,
) -> go.Figure:
    if allocation_df.empty:
        return go.Figure()

    sector_totals_df = (
        allocation_df.groupby("sector", as_index=False)[COL_MARKET_VALUE]
        .sum()
        .sort_values(COL_MARKET_VALUE, ascending=False)
        .reset_index(drop=True)
    )
    total_value = float(pd.to_numeric(sector_totals_df[COL_MARKET_VALUE], errors="coerce").fillna(0.0).sum())
    subsector_nodes: list[dict[str, object]] = []
    for sector_name in sector_totals_df["sector"].astype(str).tolist():
        sector_slice = (
            allocation_df.loc[allocation_df["sector"].astype(str) == sector_name, ["subsector", COL_MARKET_VALUE]]
            .sort_values(COL_MARKET_VALUE, ascending=False)
            .reset_index(drop=True)
        )
        palette = _build_tonal_palette(_get_team_color(sector_name), len(sector_slice))
        for idx, row in enumerate(sector_slice.to_dict(orient="records")):
            subsector_nodes.append(
                {
                    "id": f"{sector_name}::{row['subsector']}",
                    "label": row["subsector"],
                    "parent": sector_name,
                    "value": row[COL_MARKET_VALUE],
                    "color": palette[idx],
                    "text": "",
                }
            )

    sector_nodes = [
        {
            "id": sector_name,
            "label": sector_name,
            "parent": "",
            "value": row[COL_MARKET_VALUE],
            "color": _get_team_color(sector_name),
            "text": (
                f"{float(row[COL_MARKET_VALUE]) / total_value:.0%}"
                if total_value > 0
                else ""
            ),
        }
        for sector_name, row in zip(sector_totals_df["sector"].astype(str).tolist(), sector_totals_df.to_dict(orient="records"))
    ]
    node_df = pd.DataFrame(sector_nodes + subsector_nodes)

    fig = go.Figure(
        go.Sunburst(
            ids=node_df["id"],
            labels=node_df["label"],
            parents=node_df["parent"],
            values=node_df["value"],
            branchvalues="total",
            sort=False,
            marker=dict(colors=node_df["color"], line=dict(color="rgba(0,0,0,0)", width=0)),
            text=node_df["text"],
            textinfo="text",
            insidetextorientation="horizontal",
            insidetextfont=dict(color="#FFFFFF", size=14),
            hovertemplate=f"<b>%{{label}}</b><br>{value_label}: %{{value:,.2f}}<br>Weight: %{{percentRoot:.1%}}<extra></extra>",
        )
    )
    fig.update_layout(
        height=520,
        margin=dict(t=10, b=10, l=0, r=0),
    )
    return fig


def _render_sector_tilt_heatmap(team_summary: pd.DataFrame, live_sp500_weights: dict[str, float | None]) -> None:
    heatmap_df = team_summary.copy()
    heatmap_df["sp500_weight"] = pd.to_numeric(heatmap_df[COL_TEAM].map(live_sp500_weights), errors="coerce")
    heatmap_df["sector_tilt"] = (
        pd.to_numeric(heatmap_df[COL_WEIGHT], errors="coerce").fillna(0.0)
        - heatmap_df["sp500_weight"].fillna(0.0)
    )
    pod_order = [team for team in settings.display_team_order if team in heatmap_df[COL_TEAM].astype(str).tolist()]
    plot_df = (
        heatmap_df.set_index(COL_TEAM)[["sector_tilt"]]
        .T.reindex(columns=pod_order)
        .dropna(how="all")
    )
    if plot_df.empty:
        st.info("Sector tilt heatmap is unavailable.")
        return

    numeric_values = pd.to_numeric(pd.Series(plot_df.to_numpy().ravel()), errors="coerce").dropna()
    max_abs = float(numeric_values.abs().max()) if not numeric_values.empty else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 0.01

    fig = px.imshow(
        plot_df,
        color_continuous_scale="RdBu",
        aspect="auto",
        zmin=-max_abs,
        zmax=max_abs,
    )
    fig.update_traces(
        text=plot_df.copy().map(_format_percent).to_numpy(),
        texttemplate="%{text}",
        textfont=dict(size=16),
    )
    fig.update_layout(
        height=170,
        margin=dict(t=10, b=20, l=0, r=0),
    )
    fig.update_coloraxes(showscale=False, cmid=0.0)
    fig.update_xaxes(title_text="", tickfont=dict(size=14))
    fig.update_yaxes(title_text="", showticklabels=False, automargin=False, fixedrange=True, ticks="")
    fig.update_layout(
        annotations=[
            dict(
                x=-0.02,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Sector Tilt",
                textangle=-90,
                showarrow=False,
                font=dict(size=14, color="#E2E8F0"),
                xanchor="center",
                yanchor="middle",
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)


def _prepare_master_performance_history(history_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_flow_adjusted_history(
        history_df=history_df,
        value_column="portfolio_aum",
        flow_column="net_external_flow",
    )


def _prepare_team_performance_history(history_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_flow_adjusted_history(
        history_df=history_df,
        value_column="team_aum",
        flow_column="net_external_flow",
    )


def _build_portfolio_return_series(history_df: pd.DataFrame) -> pd.Series:
    """
    Convert portfolio AUM history into a flow-adjusted daily return series.
    """
    prepared = _prepare_master_performance_history(history_df)
    if prepared.empty or "performance_return" not in prepared.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(prepared["performance_return"], errors="coerce").dropna()


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_factor_analytics(snapshot_df: pd.DataFrame):
    builder = getattr(
        exposure_module,
        "build_factor_analytics_platform",
        getattr(exposure_module, "build_custom_live_factor_model"),
    )
    return builder(snapshot_df)


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


def _compute_one_year_portfolio_turnover(history_df: pd.DataFrame) -> float | None:
    """
    Estimate 1-year turnover as replaced position value divided by average AUM.

    Replaced position value is approximated from authoritative snapshot changes:
    for each snapshot transition in the trailing year, compare investable
    position sizes and count the smaller of:
    - market value of share increases
    - market value of share decreases
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

    snapshots = snapshots.loc[~_cash_like_mask(snapshots)].copy()
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
    key_cols = [COL_TEAM, COL_TICKER, COL_POSITION_SIDE]

    for current_date in transition_dates:
        prior_dates = [dt for dt in snapshot_dates if dt < current_date]
        if not prior_dates:
            continue

        previous_date = prior_dates[-1]
        previous_df = snapshot_by_date.get(previous_date, pd.DataFrame()).copy()
        current_df = snapshot_by_date.get(current_date, pd.DataFrame()).copy()
        if previous_df.empty or current_df.empty:
            continue

        merged = previous_df[key_cols + [COL_SHARES]].merge(
            current_df[key_cols + [COL_SHARES]],
            on=key_cols,
            how="outer",
            suffixes=("_prev", "_cur"),
        )
        merged["shares_prev"] = pd.to_numeric(merged["shares_prev"], errors="coerce").fillna(0.0)
        merged["shares_cur"] = pd.to_numeric(merged["shares_cur"], errors="coerce").fillna(0.0)
        merged["delta_shares"] = merged["shares_cur"] - merged["shares_prev"]
        merged = merged.loc[merged["delta_shares"].abs() > 1e-9].copy()
        if merged.empty:
            continue

        valuation_row = None
        if current_date in price_matrix.index and len(price_matrix.columns) > 0:
            valuation_row = price_matrix.loc[current_date]
        elif previous_date in price_matrix.index and len(price_matrix.columns) > 0:
            valuation_row = price_matrix.loc[previous_date]
        if valuation_row is None:
            continue

        buy_value = 0.0
        sell_value = 0.0
        for _, pos in merged.iterrows():
            px = valuation_row.get(str(pos[COL_TICKER]).strip().upper())
            if pd.isna(px):
                continue
            delta_value = float(pos["delta_shares"]) * float(px)
            if delta_value > 0:
                buy_value += delta_value
            elif delta_value < 0:
                sell_value += abs(delta_value)

        replaced_value += min(buy_value, sell_value)

    return (replaced_value / average_aum) * 100.0


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_master_fund_snapshot() -> pd.DataFrame:
    """
    Load the latest reconstructed position state, carry it forward with any
    later trades/cash flows, and then attach live prices.
    """
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

    if position_state_df.empty:
        return pd.DataFrame()

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
        .loc[lambda s: s.ne("") & s.ne("CASH")]
        .unique()
        .tolist()
    )

    latest_prices_df, _ = fetch_latest_prices(tickers)
    snapshot_df = build_current_portfolio_snapshot(carried_positions_df, latest_prices_df)

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
) -> tuple[pd.DataFrame | None, float, float]:
    """
    Carry positions through a business date and value snapshot reconciliation
    as market-value P&L rather than external flow.
    """
    net_external_flow = 0.0
    reconciliation_pnl = 0.0

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
        if expected_positions_df is not None:
            expected_value = _apply_position_values(expected_positions_df, price_map)
            snapshot_value = _apply_position_values(snapshot_for_day, price_map)
            reconciliation_pnl = float(snapshot_value - expected_value)
        return snapshot_for_day.copy(), net_external_flow, reconciliation_pnl

    return expected_positions_df, net_external_flow, reconciliation_pnl


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


def _build_team_history(team: str, snapshots_df: pd.DataFrame) -> pd.DataFrame:
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
        if not external_cash.empty:
            cash_dates = external_cash["activity_date"].dt.normalize()
            aligned_idx = business_dates.searchsorted(cash_dates)
            valid_mask = aligned_idx < len(business_dates)
            external_cash = external_cash.loc[valid_mask].copy()
            if not external_cash.empty:
                external_cash["flow_date"] = business_dates.take(aligned_idx[valid_mask]).normalize()

    investable_tickers = (
        df[COL_TICKER]
        .loc[~_cash_like_mask(df)]
        .dropna()
        .unique()
        .tolist()
    )
    trade_tickers = trades[COL_TICKER].dropna().unique().tolist() if not trades.empty else []
    benchmark_tickers = get_team_benchmark_tickers(team)
    history_tickers = sorted(set(investable_tickers + trade_tickers + benchmark_tickers))
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
        for dt, grp in external_cash.groupby("flow_date")
    } if not external_cash.empty else {}

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

        team_aum = _apply_position_values(active_positions_df, price_map)
        rows.append(
            {
                "date": dt,
                "team_aum": float(team_aum),
                "net_external_flow": net_external_flow,
                "reconciliation_pnl": reconciliation_pnl,
            }
        )

    history_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if history_df.empty:
        return pd.DataFrame(columns=["date", "team_aum", "net_external_flow", "reconciliation_pnl", "benchmark_aum"])

    benchmark_frame = build_team_benchmark_aum_frame(
        team=team,
        price_matrix=price_matrix,
        dates=history_df["date"],
        external_flow_series=history_df["net_external_flow"],
        initial_value=float(history_df["team_aum"].iloc[0]),
    )
    if not benchmark_frame.empty:
        benchmark_columns = [column for column in benchmark_frame.columns if column != "date"]
        history_df = history_df.merge(benchmark_frame, on="date", how="left")
        for column in benchmark_columns:
            if column not in history_df.columns:
                history_df[column] = pd.NA

    return history_df


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_period_pod_relative_returns(snapshot_df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame(columns=["Pod", "Active Return", "Pure Alpha"])

    analytics = get_factor_analytics(snapshot_df)
    factor_returns_df = analytics.get("factor_returns", pd.DataFrame())
    if factor_returns_df.empty:
        return pd.DataFrame(columns=["Pod", "Active Return", "Pure Alpha"])

    factor_inputs = factor_returns_df[["date", "MKT", "SMB", "MOM", "VAL"]].copy()
    factor_inputs["date"] = pd.to_datetime(factor_inputs["date"], errors="coerce")
    for col in ["MKT", "SMB", "MOM", "VAL"]:
        factor_inputs[col] = pd.to_numeric(factor_inputs[col], errors="coerce")
    factor_inputs = factor_inputs.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if factor_inputs.empty:
        return pd.DataFrame(columns=["Pod", "Active Return", "Pure Alpha"])

    with session_scope() as session:
        snapshots_df = load_all_portfolio_snapshots(session)

    teams = [
        team
        for team in settings.display_team_order
        if team != "Cash" and team in snapshot_df[COL_TEAM].dropna().astype(str).str.strip().unique().tolist()
    ]
    rows: list[dict[str, float | str]] = []

    for team in teams:
        history_df = _build_team_history(team, snapshots_df)
        if history_df.empty:
            continue

        team_history = _prepare_team_performance_history(history_df)
        benchmark_history = prepare_flow_adjusted_history(
            history_df=history_df,
            value_column="benchmark_aum",
            flow_column="net_external_flow",
        )
        if (
            team_history.empty
            or benchmark_history.empty
            or "performance_return" not in team_history.columns
            or "performance_return" not in benchmark_history.columns
        ):
            continue

        regression_df = (
            team_history[["date", "performance_return"]]
            .rename(columns={"performance_return": "team_return"})
            .merge(
                benchmark_history[["date", "performance_return"]].rename(columns={"performance_return": "benchmark_return"}),
                on="date",
                how="inner",
            )
            .merge(factor_inputs, on="date", how="inner")
        )
        regression_df["date"] = pd.to_datetime(regression_df["date"], errors="coerce")
        for col in ["team_return", "benchmark_return", "MKT", "SMB", "MOM", "VAL"]:
            regression_df[col] = pd.to_numeric(regression_df[col], errors="coerce")
        regression_df = regression_df.dropna(subset=["date", "team_return", "benchmark_return", "MKT", "SMB", "MOM", "VAL"])
        regression_df = regression_df.sort_values("date").reset_index(drop=True)
        if regression_df.empty:
            continue

        end_date = regression_df["date"].max()
        regression_window_start = end_date - pd.Timedelta(days=365)
        trailing_regression_df = regression_df.loc[regression_df["date"] >= regression_window_start].copy()
        if len(trailing_regression_df) < 20:
            continue

        x = trailing_regression_df[["MKT", "SMB", "MOM", "VAL"]].to_numpy(dtype=float)
        y = trailing_regression_df["team_return"].to_numpy(dtype=float)
        try:
            coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        regression_df["explained_return"] = (
            coefficients[0] * regression_df["MKT"]
            + coefficients[1] * regression_df["SMB"]
            + coefficients[2] * regression_df["MOM"]
            + coefficients[3] * regression_df["VAL"]
        )
        regression_df["residual"] = regression_df["team_return"] - regression_df["explained_return"]
        regression_df["active_return"] = regression_df["team_return"] - regression_df["benchmark_return"]

        period_start = end_date - pd.Timedelta(days=lookback_days)
        period_df = regression_df.loc[regression_df["date"] > period_start, ["active_return", "residual"]].copy()
        period_df["active_return"] = pd.to_numeric(period_df["active_return"], errors="coerce")
        period_df["residual"] = pd.to_numeric(period_df["residual"], errors="coerce")
        period_df = period_df.dropna(how="all")
        if period_df.empty:
            continue

        rows.append(
            {
                "Pod": team,
                "Active Return": float(period_df["active_return"].dropna().sum()),
                "Pure Alpha": float(period_df["residual"].dropna().sum()),
            }
        )

    return pd.DataFrame(rows)


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
    Build a carried-forward daily portfolio AUM series using snapshots,
    intervening trades, and external capital flows.
    """
    with session_scope() as session:
        all_snapshots_df = load_all_portfolio_snapshots(session)
        trades_df = load_trade_receipts(session)
        cash_df = load_cash_ledger(session)

    if all_snapshots_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "sp500_aum", "nasdaq_aum"])

    df = all_snapshots_df.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df[COL_TICKER] = df[COL_TICKER].astype(str).str.strip().str.upper()
    df[COL_TEAM] = df[COL_TEAM].astype(str).str.strip()
    df[COL_POSITION_SIDE] = df[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    df[COL_SHARES] = pd.to_numeric(df[COL_SHARES], errors="coerce")

    df = df.dropna(subset=["snapshot_date", COL_TICKER, COL_SHARES]).copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "sp500_aum", "nasdaq_aum"])

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
        return pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "sp500_aum", "nasdaq_aum"])

    trades = trades_df.copy()
    if not trades.empty:
        trades["trade_date"] = pd.to_datetime(trades["trade_date"], errors="coerce")
        trades[COL_TICKER] = trades[COL_TICKER].astype(str).str.strip().str.upper().replace(ticker_normalization)
        trades = trades.dropna(subset=["trade_date", COL_TICKER]).copy()

    external_cash = cash_df.copy()
    if not external_cash.empty:
        external_cash["activity_date"] = pd.to_datetime(external_cash["activity_date"], errors="coerce")
        external_cash["activity_type"] = external_cash["activity_type"].astype(str).str.strip().str.upper()
        external_cash["team"] = external_cash["team"].astype(str).str.strip()
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

    equity_tickers = (
        df[COL_TICKER]
        .loc[~df[COL_TICKER].isin(["CASH", "EUR", "GBP"])]
        .dropna()
        .unique()
        .tolist()
    )
    trade_tickers = (
        trades[COL_TICKER].dropna().unique().tolist()
        if not trades.empty else []
    )
    benchmark_tickers = list(BENCHMARK_TICKERS.values())
    history_tickers = sorted(set(equity_tickers + trade_tickers + benchmark_tickers))

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
    trades_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in trades.groupby(trades["trade_date"].dt.normalize())
    } if not trades.empty else {}
    cash_by_date = {
        dt.normalize(): grp.copy()
        for dt, grp in external_cash.groupby("flow_date")
    } if not external_cash.empty else {}

    history_rows = []
    active_positions_df: pd.DataFrame | None = None

    for dt in business_dates:
        price_map: dict[str, float] = {}
        if dt in price_matrix.index and len(price_matrix.columns) > 0:
            price_row = price_matrix.loc[dt]
            price_map = {
                str(ticker).strip().upper(): float(price_row[ticker])
                for ticker in price_matrix.columns
                if pd.notna(price_row[ticker])
            }

        snapshot_for_day = snapshot_by_date.get(dt.normalize())
        if active_positions_df is None:
            eligible_snapshot_dates = [d for d in snapshot_dates_sorted if d <= dt]
            if not eligible_snapshot_dates:
                continue
            if snapshot_for_day is None:
                active_positions_df = snapshot_by_date[eligible_snapshot_dates[-1]].copy()

        active_positions_df, net_external_flow, reconciliation_pnl = _transition_positions_for_day(
            active_positions_df=active_positions_df,
            snapshot_for_day=snapshot_for_day,
            trades_today=trades_by_date.get(dt.normalize()),
            cash_today=cash_by_date.get(dt.normalize()),
            price_map=price_map,
        )
        if active_positions_df is None:
            continue

        portfolio_aum = _apply_position_values(active_positions_df, price_map)

        history_rows.append({
            "date": dt,
            "portfolio_aum": float(portfolio_aum),
            "net_external_flow": net_external_flow,
            "reconciliation_pnl": reconciliation_pnl,
        })

    history_df = pd.DataFrame(history_rows)
    if history_df.empty:
        return pd.DataFrame(columns=["date", "portfolio_aum", "net_external_flow", "reconciliation_pnl", "sp500_aum", "nasdaq_aum"])

    history_df = history_df.sort_values("date").reset_index(drop=True)
    history_df["net_external_flow"] = pd.to_numeric(history_df["net_external_flow"], errors="coerce").fillna(0.0)

    initial_aum = float(history_df["portfolio_aum"].iloc[0])

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

        benchmark_returns = pd.to_numeric(aligned["benchmark_price"], errors="coerce").pct_change()
        scaled_aum = pd.Series(index=history_df.index, dtype="float64")
        scaled_aum.iloc[0] = initial_aum
        if len(history_df) > 1:
            scaled_tail = build_flow_adjusted_benchmark_series(
                benchmark_return_series=benchmark_returns.iloc[1:],
                external_flow_series=history_df["net_external_flow"].iloc[1:],
                initial_value=initial_aum,
            )
            scaled_aum.iloc[1:] = scaled_tail.values

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
    Compute trailing flow-adjusted return from daily performance returns.
    """
    prepared = _prepare_master_performance_history(history_df)
    if prepared.empty:
        return None

    df = prepared.copy()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
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


def _compute_daily_return(history_df: pd.DataFrame, current_total_market_value: float) -> float | None:
    """
    Compute latest flow-adjusted daily return.
    """
    prepared = _prepare_master_performance_history(history_df)
    if prepared.empty:
        return None
    valid = pd.to_numeric(prepared["performance_return"], errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _compute_live_daily_metrics(snapshot_df: pd.DataFrame) -> tuple[float | None, float | None, pd.Timestamp | None]:
    """
    Compute mark-to-market daily P&L and return using live prices versus prior close.

    This is intentionally separate from the carried daily history series because the
    history builder uses daily bars only, which can be stale intraday.
    """
    if snapshot_df.empty or COL_MARKET_VALUE not in snapshot_df.columns:
        return None, None, None

    current_total_market_value = float(
        pd.to_numeric(snapshot_df[COL_MARKET_VALUE], errors="coerce").fillna(0.0).sum()
    )
    if current_total_market_value == 0:
        return None, None, None

    investable_df = snapshot_df.loc[~_cash_like_mask(snapshot_df)].copy()
    if investable_df.empty:
        return 0.0, 0.0, None

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
        return None, None, None

    lookback_days = 10
    history_prices_df = fetch_multiple_price_histories(tickers=tickers, lookback_days=lookback_days)
    if history_prices_df.empty:
        return None, None, None

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
        return None, None, None

    today = pd.Timestamp.today().normalize()
    prior_close_candidates = price_history.loc[price_history["date"].dt.normalize() < today, "date"]
    if prior_close_candidates.empty:
        prior_close_date = price_history["date"].max()
    else:
        prior_close_date = prior_close_candidates.max()

    if pd.isna(prior_close_date):
        return None, None, None

    prior_close_map = (
        price_history.loc[price_history["date"].dt.normalize() <= pd.Timestamp(prior_close_date).normalize()]
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")["px"]
        .to_dict()
    )
    if not prior_close_map:
        return None, None, None

    prior_close_total_market_value = _apply_position_values(snapshot_df, prior_close_map)
    if prior_close_total_market_value == 0:
        return None, None, pd.Timestamp(prior_close_date)

    daily_pnl = current_total_market_value - prior_close_total_market_value
    daily_return = daily_pnl / prior_close_total_market_value
    return float(daily_return), float(daily_pnl), pd.Timestamp(prior_close_date)


def _compute_trailing_pnl(history_df: pd.DataFrame, days: int) -> float | None:
    """
    Compute trailing performance P&L net of external flows.
    """
    prepared = _prepare_master_performance_history(history_df)
    if prepared.empty:
        return None

    df = prepared.copy()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 2:
        return None

    latest_date = df["date"].iloc[-1]
    target_date = latest_date - pd.Timedelta(days=days)
    prior_candidates = df.loc[df["date"] <= target_date].copy()
    if prior_candidates.empty:
        return None
    prior_date = prior_candidates["date"].iloc[-1]
    period_pnl = pd.to_numeric(
        df.loc[df["date"] > prior_date, "performance_pnl"],
        errors="coerce",
    ).dropna()
    if period_pnl.empty:
        return None
    return float(period_pnl.sum())


def _compute_daily_pnl(history_df: pd.DataFrame, current_total_market_value: float) -> float | None:
    """
    Compute latest flow-adjusted daily dollar P&L.
    """
    prepared = _prepare_master_performance_history(history_df)
    if prepared.empty:
        return None
    valid = pd.to_numeric(prepared["performance_pnl"], errors="coerce").dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


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
    render_page_title("Portfolio View")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def render_header_metrics(snapshot_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    """
    Top-level metrics using explicit, robust definitions.
    """
    st.subheader("Live Portfolio Snapshot")

    if snapshot_df.empty:
        total_long_exposure = 0.0
        cash_value = 0.0
        gross_exposure = 0.0
        position_count = 0
        average_position_size = 0.0
    else:
        df = snapshot_df.copy()
        team_summary_df = _build_complete_team_summary(df)

        df[COL_MARKET_VALUE] = pd.to_numeric(df[COL_MARKET_VALUE], errors="coerce")
        cash_value = float(
            pd.to_numeric(
                team_summary_df.loc[
                    team_summary_df[COL_TEAM].astype(str).str.lower().eq("cash"),
                    COL_MARKET_VALUE,
                ],
                errors="coerce",
            ).sum(skipna=True)
        ) if not team_summary_df.empty else 0.0

        cash_mask = _cash_like_mask(df)

        non_cash_df = df.loc[~cash_mask].copy()
        total_long_exposure = float(
            non_cash_df.loc[
                pd.to_numeric(non_cash_df[COL_MARKET_VALUE], errors="coerce") > 0,
                COL_MARKET_VALUE,
            ].sum(skipna=True)
        )
        gross_exposure = total_long_exposure + cash_value

        position_count = int(
            df.loc[~cash_mask, COL_TICKER].nunique(dropna=True)
        )
        average_position_size = gross_exposure / position_count if position_count else 0.0

    trailing_returns = _build_portfolio_return_series(history_df)
    one_year_sharpe = compute_sharpe_ratio(trailing_returns) if not trailing_returns.empty else None
    one_year_sortino = compute_sortino_ratio(trailing_returns) if not trailing_returns.empty else None
    one_year_turnover = _compute_one_year_portfolio_turnover(history_df)

    with st.container(border=True):
        row_1 = st.columns(4)
        with row_1[0]:
            _render_live_portfolio_card("Total Long Exposure", _format_currency(total_long_exposure))
        with row_1[1]:
            _render_live_portfolio_card("Cash", _format_currency(cash_value))
        with row_1[2]:
            _render_live_portfolio_card("Gross Exposure", _format_currency(gross_exposure))
        with row_1[3]:
            _render_live_portfolio_card("Positions", f"{position_count:,}")

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        row_2 = st.columns(4)
        with row_2[0]:
            _render_live_portfolio_card("Average Position Size", _format_currency(average_position_size))
        with row_2[1]:
            _render_live_portfolio_card("1 Year Portfolio Turnover", _format_percent(one_year_turnover / 100.0 if one_year_turnover is not None else None))
        with row_2[2]:
            _render_live_portfolio_card("1 Year Sharpe Ratio", _format_number(one_year_sharpe))
        with row_2[3]:
            _render_live_portfolio_card("1 Year Sortino Ratio", _format_number(one_year_sortino))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def render_performance_dashboard(snapshot_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    st.subheader("Performance Dashboard")

    current_total_market_value = float(
        pd.to_numeric(snapshot_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)
    ) if not snapshot_df.empty else 0.0

    ret_1y = _compute_trailing_return(history_df, 365)
    ret_1m = _compute_trailing_return(history_df, 30)
    ret_1w = _compute_trailing_return(history_df, 7)
    pnl_1y = _compute_trailing_pnl(history_df, 365)
    pnl_1m = _compute_trailing_pnl(history_df, 30)
    pnl_1w = _compute_trailing_pnl(history_df, 7)
    ret_daily, pnl_daily, prior_close_date = _compute_live_daily_metrics(snapshot_df)
    if ret_daily is None:
        ret_daily = _compute_daily_return(history_df, current_total_market_value)
    if pnl_daily is None:
        pnl_daily = _compute_daily_pnl(history_df, current_total_market_value)
    latest_business_date = _latest_history_date(history_df)

    def _value_class(value) -> str:
        if value is None or pd.isna(value):
            return ""
        return "positive" if float(value) > 0 else "negative" if float(value) < 0 else ""

    with st.container(border=True):
        row_1 = st.columns(4)
        with row_1[0]:
            _render_live_portfolio_card("1 Year Return", _format_percent(ret_1y), _value_class(ret_1y))
        with row_1[1]:
            _render_live_portfolio_card("1 Month Return", _format_percent(ret_1m), _value_class(ret_1m))
        with row_1[2]:
            _render_live_portfolio_card("1 Week Return", _format_percent(ret_1w), _value_class(ret_1w))
        with row_1[3]:
            _render_live_portfolio_card("Daily Return", _format_percent(ret_daily), _value_class(ret_daily))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        row_2 = st.columns(4)
        with row_2[0]:
            _render_live_portfolio_card("1 Year P&L", _format_currency(pnl_1y), _value_class(pnl_1y))
        with row_2[1]:
            _render_live_portfolio_card("1 Month P&L", _format_currency(pnl_1m), _value_class(pnl_1m))
        with row_2[2]:
            _render_live_portfolio_card("1 Week P&L", _format_currency(pnl_1w), _value_class(pnl_1w))
        with row_2[3]:
            _render_live_portfolio_card("Daily P&L", _format_currency(pnl_daily), _value_class(pnl_daily))

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    if prior_close_date is not None:
        st.caption(
            f"Daily return and daily P&L are mark-to-market versus the prior close on "
            f"{prior_close_date.strftime('%A, %B %d, %Y')}."
        )
    elif latest_business_date is not None:
        st.caption(
            f"Daily return and daily P&L are as of {latest_business_date.strftime('%A, %B %d, %Y')}, "
            "the latest business day in the portfolio history."
        )


def _get_ytd_lookback_days() -> int:
    today = pd.Timestamp.today().normalize()
    start_of_year = pd.Timestamp(year=today.year, month=1, day=1)
    return max((today - start_of_year).days + 7, 10)


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_period_pnl_movers(
    snapshot_df: pd.DataFrame,
    period_label: str,
    lookback_days: int,
) -> pd.DataFrame:
    """
    Compute P&L movers for the current holdings over a chosen lookback horizon.

    Important:
    - This uses CURRENT holdings/shares and backs into hypothetical contribution
      over the horizon using price change from the prior anchor date to today.
    - So Weekly and YTD movers are based on today's portfolio composition,
      not the exact historical portfolio on those earlier dates.
    """
    pnl_col = f"{period_label} P&L"

    if snapshot_df.empty:
        return pd.DataFrame(columns=["Ticker", "Pod", pnl_col])

    working = snapshot_df.copy()
    working = working.loc[~_cash_like_mask(working)].copy()
    if working.empty:
        return pd.DataFrame(columns=["Ticker", "Pod", pnl_col])

    working[COL_TICKER] = working[COL_TICKER].astype(str).str.strip().str.upper()
    working[COL_TEAM] = working[COL_TEAM].astype(str).str.strip()
    working[COL_POSITION_SIDE] = working[COL_POSITION_SIDE].astype(str).str.strip().str.upper()
    working[COL_SHARES] = pd.to_numeric(working[COL_SHARES], errors="coerce").fillna(0.0)
    working[COL_PRICE] = pd.to_numeric(working[COL_PRICE], errors="coerce")

    tickers = sorted(working[COL_TICKER].dropna().unique().tolist())
    price_history_df = fetch_multiple_price_histories(
        tickers=tickers,
        lookback_days=max(lookback_days + 10, 15),
    )
    if price_history_df.empty:
        return pd.DataFrame(columns=["Ticker", "Pod", pnl_col])

    price_history_df["date"] = pd.to_datetime(price_history_df["date"], errors="coerce")
    price_history_df["ticker"] = price_history_df["ticker"].astype(str).str.strip().str.upper()
    price_history_df["adj_close"] = pd.to_numeric(price_history_df.get("adj_close"), errors="coerce")
    price_history_df["close"] = pd.to_numeric(price_history_df.get("close"), errors="coerce")
    price_history_df["px"] = price_history_df["adj_close"]
    missing_mask = price_history_df["px"].isna()
    price_history_df.loc[missing_mask, "px"] = price_history_df.loc[missing_mask, "close"]

    price_history_df = (
        price_history_df
        .dropna(subset=["date", "ticker", "px"])
        .sort_values(["ticker", "date"])
        .copy()
    )
    if price_history_df.empty:
        return pd.DataFrame(columns=["Ticker", "Pod", pnl_col])

    latest_date = price_history_df["date"].max()
    anchor_date = latest_date - pd.Timedelta(days=lookback_days)

    prior_prices = (
        price_history_df.loc[price_history_df["date"] <= anchor_date]
        .sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "px"]]
        .rename(columns={"px": "anchor_price"})
    )

    if prior_prices.empty:
        return pd.DataFrame(columns=["Ticker", "Pod", pnl_col])

    movers_df = working.merge(
        prior_prices,
        left_on=COL_TICKER,
        right_on="ticker",
        how="left",
    )

    movers_df["price_change"] = (
        pd.to_numeric(movers_df[COL_PRICE], errors="coerce")
        - pd.to_numeric(movers_df["anchor_price"], errors="coerce")
    )
    movers_df["period_pnl"] = movers_df[COL_SHARES] * movers_df["price_change"]
    movers_df.loc[movers_df[COL_POSITION_SIDE].eq("SHORT"), "period_pnl"] *= -1.0
    movers_df = movers_df.dropna(subset=["period_pnl"]).copy()

    if movers_df.empty:
        return pd.DataFrame(columns=["Ticker", "Pod", pnl_col])

    return movers_df.rename(
        columns={
            COL_TICKER: "Ticker",
            COL_TEAM: "Pod",
            "period_pnl": pnl_col,
        }
    )[["Ticker", "Pod", pnl_col]]


def _render_pnl_mover_table(
    snapshot_df: pd.DataFrame,
    section_title: str,
    period_label: str,
    lookback_days: int,
) -> None:
    pnl_col = f"{period_label} P&L"
    movers_df = get_period_pnl_movers(
        snapshot_df=snapshot_df,
        period_label=period_label,
        lookback_days=lookback_days,
    )

    st.subheader(section_title)

    if movers_df.empty:
        st.info(f"{section_title} are unavailable.")
        return

    winners_df = movers_df.sort_values(pnl_col, ascending=False).head(5).copy()
    losers_df = movers_df.sort_values(pnl_col, ascending=True).head(5).copy()

    for df in (winners_df, losers_df):
        df[pnl_col] = df[pnl_col].map(_format_currency)

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            _render_movers_bar_chart(
                winners_df.sort_values(pnl_col, ascending=True),
                pnl_col,
                f"Top {period_label} Winners",
            )

    with right_col:
        with st.container(border=True):
            _render_movers_bar_chart(
                losers_df.sort_values(pnl_col, ascending=True),
                pnl_col,
                f"Top {period_label} Losers",
            )


def _style_movers_table(df: pd.DataFrame, pnl_col: str, bar_color: str):
    styled_df = df.copy()
    styled_df[pnl_col] = pd.to_numeric(styled_df[pnl_col], errors="coerce")

    max_abs = styled_df[pnl_col].abs().max()
    if pd.isna(max_abs) or max_abs == 0:
        max_abs = 1.0

    return (
        styled_df.style
        .format({pnl_col: _format_currency})
        .set_properties(**{
            "text-align": "left",
            "font-size": "0.95rem",
        })
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "left"), ("font-weight", "600")]},
            {"selector": "td", "props": [("padding", "0.45rem 0.6rem")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
        .bar(
            subset=[pnl_col],
            align="mid",
            color=bar_color,
            vmin=-max_abs,
            vmax=max_abs,
        )
    )

def _render_pnl_mover_table(
    snapshot_df: pd.DataFrame,
    period_label: str,
    lookback_days: int,
) -> None:
    pnl_col = f"{period_label} P&L"
    movers_df = get_period_pnl_movers(
        snapshot_df=snapshot_df,
        period_label=period_label,
        lookback_days=lookback_days,
    )

    if movers_df.empty:
        st.info(f"{period_label} P&L movers are unavailable.")
        return

    winners_df = movers_df.sort_values(pnl_col, ascending=False).head(5).copy()
    losers_df = movers_df.sort_values(pnl_col, ascending=True).head(5).copy()

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.markdown(f"#### Top Winners")
            st.dataframe(
                _style_movers_table(winners_df, pnl_col, "#d9f2e3"),
                use_container_width=True,
                hide_index=True,
            )

    with right_col:
        with st.container(border=True):
            st.markdown(f"#### Top Losers")
            st.dataframe(
                _style_movers_table(losers_df, pnl_col, "#f8d7da"),
                use_container_width=True,
                hide_index=True,
            )


def _render_period_heatmap(
    snapshot_df: pd.DataFrame,
    period_label: str,
    lookback_days: int,
) -> None:
    relative_df = get_period_pod_relative_returns(
        snapshot_df=snapshot_df,
        lookback_days=lookback_days,
    )
    if relative_df.empty:
        st.info(f"{period_label} pod return heatmap is unavailable.")
        return

    pod_order = [team for team in settings.display_team_order if team in relative_df["Pod"].tolist()]
    plot_df = (
        relative_df.set_index("Pod")[["Active Return", "Pure Alpha"]]
        .T.reindex(columns=pod_order)
        .dropna(how="all")
    )
    if plot_df.empty:
        st.info(f"{period_label} pod return heatmap is unavailable.")
        return

    numeric_values = pd.to_numeric(pd.Series(plot_df.to_numpy().ravel()), errors="coerce").dropna()
    max_abs = float(numeric_values.abs().max()) if not numeric_values.empty else 0.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 0.01
    text_df = plot_df.copy().map(_format_percent)

    fig = px.imshow(
        plot_df,
        color_continuous_scale="RdBu",
        aspect="auto",
        zmin=-max_abs,
        zmax=max_abs,
    )
    fig.update_traces(
        text=text_df.to_numpy(),
        texttemplate="%{text}",
        textfont=dict(size=16),
    )
    fig.update_layout(
        height=320,
        margin=dict(t=20, b=20, l=46, r=0),
    )
    fig.update_coloraxes(showscale=False, cmid=0.0)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(
        title_text="",
        showticklabels=False,
        automargin=False,
        fixedrange=True,
        ticks="",
    )
    fig.update_layout(
        annotations=[
            dict(
                x=-0.028,
                y=0.75,
                xref="paper",
                yref="paper",
                text="Active Return",
                textangle=-90,
                showarrow=False,
                font=dict(size=14, color="#E2E8F0"),
                xanchor="center",
                yanchor="middle",
                align="center",
                bgcolor="rgba(15, 23, 42, 0.0)",
                borderpad=0,
            ),
            dict(
                x=-0.028,
                y=0.25,
                xref="paper",
                yref="paper",
                text="Pure Alpha",
                textangle=-90,
                showarrow=False,
                font=dict(size=14, color="#E2E8F0"),
                xanchor="center",
                yanchor="middle",
                align="center",
                bgcolor="rgba(15, 23, 42, 0.0)",
                borderpad=0,
            ),
        ]
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_movers(snapshot_df: pd.DataFrame) -> None:
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    daily_tab, weekly_tab, ytd_tab, one_year_tab = st.tabs(["1D", "5D", "YTD", "1Y"])

    with daily_tab:
        _render_pnl_mover_table(
            snapshot_df=snapshot_df,
            period_label="1D",
            lookback_days=1,
        )
        _render_period_heatmap(snapshot_df, "1D", 1)

    with weekly_tab:
        _render_pnl_mover_table(
            snapshot_df=snapshot_df,
            period_label="5D",
            lookback_days=5,
        )
        _render_period_heatmap(snapshot_df, "5D", 5)

    with ytd_tab:
        _render_pnl_mover_table(
            snapshot_df=snapshot_df,
            period_label="YTD",
            lookback_days=_get_ytd_lookback_days(),
        )
        _render_period_heatmap(snapshot_df, "YTD", _get_ytd_lookback_days())

    with one_year_tab:
        _render_pnl_mover_table(
            snapshot_df=snapshot_df,
            period_label="1Y",
            lookback_days=365,
        )
        _render_period_heatmap(snapshot_df, "1Y", 365)

def _render_movers_bar_chart(df: pd.DataFrame, pnl_col: str, title: str) -> None:
    if df.empty:
        st.info(f"{title} unavailable.")
        return

    plot_df = df.copy()
    plot_df[pnl_col] = pd.to_numeric(plot_df[pnl_col], errors="coerce")
    plot_df["Label"] = plot_df["Ticker"] + " • " + plot_df["Pod"]

    fig = px.bar(
        plot_df,
        x=pnl_col,
        y="Label",
        orientation="h",
        title=title,
        text=pnl_col,
    )
    fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        xaxis_title="P&L",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_team_allocation(snapshot_df: pd.DataFrame) -> None:
    st.subheader("Live Sector Allocation")

    team_summary = _build_complete_team_summary(snapshot_df)
    if team_summary.empty:
        st.info("No pod allocation data available.")
        return

    live_sp500_weights = fetch_sp500_sector_group_weights()

    plot_df = team_summary.loc[
        pd.to_numeric(team_summary[COL_MARKET_VALUE], errors="coerce").fillna(0.0) != 0
    ].copy()

    if plot_df.empty:
        st.info("No allocation data available for charting.")
        return

    sunburst_df = _build_allocation_sunburst_frame(snapshot_df)
    benchmark_sunburst_df = _build_sp500_sunburst_frame()
    if sunburst_df.empty and benchmark_sunburst_df.empty:
        st.info("No allocation data available for charting.")
        return

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            '<div style="color:#FFFFFF;font-weight:700;font-size:1.05rem;letter-spacing:0.04em;text-align:center;"><b>CURRENT PORTFOLIO ALLOCATION</b></div>',
            unsafe_allow_html=True,
        )
        if sunburst_df.empty:
            st.info("Current portfolio allocation is unavailable.")
        else:
            st.plotly_chart(
                _build_allocation_sunburst_figure(sunburst_df, "Value"),
                use_container_width=True,
            )

    with right_col:
        st.markdown(
            '<div style="color:#FFFFFF;font-weight:700;font-size:1.05rem;letter-spacing:0.04em;text-align:center;"><b>LIVE S&amp;P 500 ALLOCATION</b></div>',
            unsafe_allow_html=True,
        )
        if benchmark_sunburst_df.empty:
            st.info("S&P 500 allocation is unavailable.")
        else:
            st.plotly_chart(
                _build_allocation_sunburst_figure(benchmark_sunburst_df, "Weight"),
                use_container_width=True,
            )

    st.markdown(
        '<div style="color:#FFFFFF;font-weight:700;font-size:1.05rem;letter-spacing:0.04em;text-align:center;"><b>SECTOR TILTS</b></div>',
        unsafe_allow_html=True,
    )
    _render_sector_tilt_heatmap(team_summary, live_sp500_weights)

    legend_teams = [
        team for team in settings.display_team_order
        if team in set(plot_df[COL_TEAM].astype(str).tolist())
        or team in set(sunburst_df["sector"].astype(str).tolist())
        or team in set(benchmark_sunburst_df["sector"].astype(str).tolist())
    ]
    _render_custom_pod_legend(legend_teams)


def render_aum_chart(history_df: pd.DataFrame) -> None:
    st.subheader("1 Year Returns vs Benchmark")

    if history_df.empty:
        st.info("No historical return data available.")
        return

    plot_df = history_df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
    plot_df = plot_df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    if plot_df.empty:
        st.info("No historical return data available.")
        return

    start_date = plot_df["date"].max() - pd.Timedelta(days=365)
    plot_df = plot_df.loc[plot_df["date"] >= start_date].copy()
    if plot_df.empty:
        st.info("No historical return data available.")
        return

    normalized_df = plot_df[["date"]].copy()
    series_specs = [
        ("portfolio_aum", "Portfolio"),
        ("sp500_aum", "S&P 500 (SPY)"),
        ("nasdaq_aum", "Nasdaq (QQQ)"),
    ]

    for source_col, label in series_specs:
        if source_col not in plot_df.columns:
            continue

        prepared = prepare_flow_adjusted_history(
            history_df=plot_df[["date", source_col, "net_external_flow"]].copy(),
            value_column=source_col,
            flow_column="net_external_flow",
        )
        if prepared.empty or "performance_return" not in prepared.columns:
            continue

        cumulative_returns = compute_cumulative_return_series(prepared["performance_return"])
        normalized_df[label] = cumulative_returns.values

    plotted_columns = [col for col in normalized_df.columns if col != "date"]
    if not plotted_columns:
        st.info("No historical return data available.")
        return

    fig = go.Figure()
    for label in plotted_columns:
        fig.add_trace(
            go.Scatter(
                x=normalized_df["date"],
                y=normalized_df[label],
                mode="lines",
                name=label,
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode="x unified",
        legend=dict(title=None),
        margin=dict(t=58),
        title=dict(text="1 YEAR RETURNS VS BENCHMARK", x=0.5, xanchor="center", font=dict(size=16)),
    )
    fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology Note"):
        st.write(
            """
            Portfolio history is reconstructed from authoritative snapshots, then carried
            forward with intervening trade receipts and any uploaded sector-rebalance
            or portfolio-liquidation cash flows.

            This chart normalizes the portfolio and benchmark series to 0 at the
            start of the trailing 1-year window to show relative return paths.
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
            COL_TEAM: "Pod",
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

    st.set_page_config(page_title="Portfolio View", layout="wide")
    apply_app_theme()
    render_top_nav()
    apply_page_theme()
    render_page_title("Portfolio View")


    snapshot_df = get_master_fund_snapshot()

    if snapshot_df.empty:
        render_empty_state()
        return

    history_df = get_master_fund_history()

    render_header_metrics(snapshot_df, history_df)
    st.divider()

    render_team_allocation(snapshot_df)
    st.divider()

    render_performance_dashboard(snapshot_df, history_df)
    render_pnl_movers(snapshot_df)
    st.divider()

    render_aum_chart(history_df)


if __name__ == "__main__":
    main()
