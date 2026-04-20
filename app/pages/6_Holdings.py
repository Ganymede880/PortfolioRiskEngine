"""
Holdings page for the CMCSIF Portfolio Tracker.

This page includes:
- current holdings detail
- pod / side / ticker filters
- top-line holdings dashboard
- holdings constellation based on return correlations
"""

from __future__ import annotations

import html
import math
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import networkx as nx
except ImportError:
    nx = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import fetch_latest_prices, fetch_multiple_price_histories
from src.db.crud import load_position_state
from src.db.session import session_scope
from src.utils.constants import TEAM_COLORS
from src.utils.ui import apply_app_theme, left_align_dataframe


COL_TEAM = "team"
COL_TICKER = "ticker"
COL_POSITION_SIDE = "position_side"
COL_SHARES = "shares"
COL_PRICE = "price"
COL_MARKET_VALUE = "market_value"
COL_WEIGHT = "weight"


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


def apply_holdings_page_theme() -> None:
    st.markdown(
        """
        <style>
        .holdings-metric-card {
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

        .holdings-metric-card-label {
            color: inherit;
            opacity: 0.8;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }

        .holdings-metric-card-value {
            color: inherit;
            font-size: 1.65rem;
            font-weight: 600;
            line-height: 1.2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_holdings_metric_card(label: str, value: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="holdings-metric-card">
            <div class="holdings-metric-card-label">{safe_label}</div>
            <div class="holdings-metric-card-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_holdings_snapshot() -> pd.DataFrame:
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
def get_constellation_price_history(snapshot_df: pd.DataFrame, lookback_days: int = 120) -> pd.DataFrame:
    if snapshot_df.empty or COL_TICKER not in snapshot_df.columns:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    eligible_df = snapshot_df.loc[~_cash_like_mask(snapshot_df)].copy()
    if eligible_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    tickers = (
        eligible_df[COL_TICKER]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda s: s.ne("")]
        .drop_duplicates()
        .tolist()
    )
    if not tickers:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close"])

    return fetch_multiple_price_histories(tickers, lookback_days=lookback_days)


def build_correlation_graph(
    snapshot_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    threshold: float = 0.50,
) -> dict[str, object] | None:
    if nx is None:
        return None
    if snapshot_df.empty or price_history_df.empty:
        return None

    eligible_df = snapshot_df.loc[~_cash_like_mask(snapshot_df)].copy()
    if eligible_df.empty:
        return None

    eligible_df[COL_TICKER] = eligible_df[COL_TICKER].astype(str).str.strip().str.upper()
    eligible_df[COL_MARKET_VALUE] = pd.to_numeric(eligible_df[COL_MARKET_VALUE], errors="coerce").fillna(0.0)
    eligible_df[COL_WEIGHT] = pd.to_numeric(eligible_df.get(COL_WEIGHT), errors="coerce").fillna(0.0)
    eligible_df["abs_market_value"] = eligible_df[COL_MARKET_VALUE].abs()
    candidate_universe_size = 40
    eligible_df = (
        eligible_df.sort_values("abs_market_value", ascending=False)
        .drop_duplicates(subset=[COL_TICKER], keep="first")
        .head(candidate_universe_size)
        .reset_index(drop=True)
    )
    if eligible_df.empty:
        return None

    returns_source = price_history_df.copy()
    price_col = "adj_close" if "adj_close" in returns_source.columns else "close"
    if price_col not in returns_source.columns:
        return None

    returns_source["date"] = pd.to_datetime(returns_source["date"], errors="coerce")
    returns_source["ticker"] = returns_source["ticker"].astype(str).str.strip().str.upper()
    returns_source[price_col] = pd.to_numeric(returns_source[price_col], errors="coerce")
    returns_source = returns_source.dropna(subset=["date", "ticker", price_col]).copy()
    returns_source = returns_source.loc[returns_source["ticker"].isin(eligible_df[COL_TICKER])]
    if returns_source.empty:
        return None

    price_matrix = (
        returns_source.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
    )
    return_matrix = price_matrix.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna(how="all")
    valid_columns = [col for col in return_matrix.columns if return_matrix[col].dropna().shape[0] >= 30]
    return_matrix = return_matrix[valid_columns]
    if return_matrix.shape[1] < 2:
        return None

    eligible_df = eligible_df.loc[eligible_df[COL_TICKER].isin(valid_columns)].copy()
    if eligible_df.shape[0] < 2:
        return None

    correlation_df = return_matrix.corr(min_periods=20)
    if correlation_df.empty:
        return None

    edge_records: list[dict[str, object]] = []
    tickers = eligible_df[COL_TICKER].tolist()
    for i, source in enumerate(tickers):
        for target in tickers[i + 1:]:
            corr_value = correlation_df.loc[source, target] if source in correlation_df.index and target in correlation_df.columns else pd.NA
            if pd.isna(corr_value) or abs(float(corr_value)) < threshold:
                continue
            corr_float = float(corr_value)
            edge_records.append(
                {
                    "source": source,
                    "target": target,
                    "correlation": corr_float,
                    "strength": abs(corr_float),
                }
            )

    if not edge_records:
        return None

    edges_df = pd.DataFrame(edge_records)
    degree_df = (
        pd.concat(
            [
                edges_df[["source"]].rename(columns={"source": "ticker"}),
                edges_df[["target"]].rename(columns={"target": "ticker"}),
            ],
            ignore_index=True,
        )
        .groupby("ticker", as_index=False)
        .size()
        .rename(columns={"size": "degree"})
    )

    nodes_df = eligible_df.merge(degree_df, on=COL_TICKER, how="left")
    nodes_df["degree"] = pd.to_numeric(nodes_df["degree"], errors="coerce").fillna(0).astype(int)
    isolated_count = int((nodes_df["degree"] == 0).sum())
    nodes_df = nodes_df.loc[nodes_df["degree"] >= 1].copy()
    if nodes_df.shape[0] < 2:
        return None

    post_threshold_cap = 30
    if nodes_df.shape[0] > post_threshold_cap:
        kept_tickers = set(
            nodes_df.sort_values("abs_market_value", ascending=False)
            .head(post_threshold_cap)[COL_TICKER]
            .tolist()
        )
        nodes_df = nodes_df.loc[nodes_df[COL_TICKER].isin(kept_tickers)].copy()
        edges_df = edges_df.loc[
            edges_df["source"].isin(kept_tickers) & edges_df["target"].isin(kept_tickers)
        ].copy()
        if edges_df.empty:
            return None
        degree_df = (
            pd.concat(
                [
                    edges_df[["source"]].rename(columns={"source": "ticker"}),
                    edges_df[["target"]].rename(columns={"target": "ticker"}),
                ],
                ignore_index=True,
            )
            .groupby("ticker", as_index=False)
            .size()
            .rename(columns={"size": "degree"})
        )
        degree_map = degree_df.set_index("ticker")["degree"].to_dict()
        nodes_df = nodes_df.drop(columns=["degree"], errors="ignore").copy()
        nodes_df["degree"] = (
            nodes_df[COL_TICKER].map(degree_map).fillna(0).astype(int)
        )
        nodes_df = nodes_df.loc[nodes_df["degree"] >= 1].copy()
        if nodes_df.shape[0] < 2:
            return None

    nodes_df = nodes_df.rename(columns={COL_TICKER: "ticker", COL_TEAM: "pod"})
    nodes_df["color"] = nodes_df["pod"].map(lambda team: TEAM_COLORS.get(team, "#94A3B8"))
    nodes_df = nodes_df[
        ["ticker", "pod", COL_MARKET_VALUE, "abs_market_value", COL_WEIGHT, "degree", "color"]
    ].reset_index(drop=True)
    edges_df = edges_df.sort_values(["strength", "source", "target"], ascending=[False, True, True]).reset_index(drop=True)

    return {
        "nodes_df": nodes_df,
        "edges_df": edges_df,
        "isolated_count": isolated_count,
        "threshold": threshold,
    }


def pack_component_layouts(component_layouts: list[dict[str, object]], gap: float = 0.35) -> dict[str, tuple[float, float]]:
    if not component_layouts:
        return {}

    ordered = sorted(component_layouts, key=lambda item: int(item["size"]), reverse=True)
    packed_positions: dict[str, tuple[float, float]] = {}

    if len(ordered) == 1:
        return dict(ordered[0]["positions"])

    center_component = ordered[0]
    for node, coords in center_component["positions"].items():
        packed_positions[node] = coords

    if len(ordered) == 2:
        offsets = [(center_component["width"] / 2 + ordered[1]["width"] / 2 + gap, 0.0)]
    else:
        offsets: list[tuple[float, float]] = []
        orbit_radius = max(center_component["width"], center_component["height"]) / 2 + gap
        for idx, component in enumerate(ordered[1:], start=1):
            angle = (idx - 1) * (math.pi / max(1, len(ordered) - 1)) - (math.pi / 2)
            offsets.append(
                (
                    orbit_radius * 1.25 * math.cos(angle),
                    orbit_radius * math.sin(angle),
                )
            )

    for component, (offset_x, offset_y) in zip(ordered[1:], offsets):
        x_shift = offset_x
        y_shift = offset_y
        for node, (x_pos, y_pos) in component["positions"].items():
            packed_positions[node] = (x_pos + x_shift, y_pos + y_shift)

    return packed_positions


def _compute_graph_positions(graph: "nx.Graph", compact_cluster_packing: bool = True) -> dict[str, tuple[float, float]]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_nodes() == 1:
        only_node = next(iter(graph.nodes()))
        return {only_node: (0.0, 0.0)}

    components = [graph.subgraph(component_nodes).copy() for component_nodes in nx.connected_components(graph)]
    if len(components) == 1:
        return nx.spring_layout(components[0], seed=42, weight="strength", k=0.95 / max(components[0].number_of_nodes(), 1))

    component_layouts: list[dict[str, object]] = []
    for component in components:
        if component.number_of_nodes() == 1:
            positions = {next(iter(component.nodes())): (0.0, 0.0)}
        elif component.number_of_nodes() == 2:
            positions = nx.spring_layout(component, seed=42, weight="strength", k=0.7)
        else:
            positions = nx.kamada_kawai_layout(component, weight="strength")

        x_values = [coords[0] for coords in positions.values()]
        y_values = [coords[1] for coords in positions.values()]
        x_center = (min(x_values) + max(x_values)) / 2
        y_center = (min(y_values) + max(y_values)) / 2
        normalized_positions = {
            node: (coords[0] - x_center, coords[1] - y_center)
            for node, coords in positions.items()
        }
        component_layouts.append(
            {
                "positions": normalized_positions,
                "width": max(x_values) - min(x_values) if x_values else 0.0,
                "height": max(y_values) - min(y_values) if y_values else 0.0,
                "size": component.number_of_nodes(),
            }
        )

    if compact_cluster_packing:
        packed_positions = pack_component_layouts(component_layouts, gap=0.55)
    else:
        packed_positions = {}
        x_cursor = 0.0
        for component in sorted(component_layouts, key=lambda item: int(item["size"]), reverse=True):
            width = float(component["width"]) or 0.5
            for node, (x_pos, y_pos) in component["positions"].items():
                packed_positions[node] = (x_pos + x_cursor, y_pos)
            x_cursor += width + 1.2

    if not packed_positions:
        return {}

    x_values = [coords[0] for coords in packed_positions.values()]
    y_values = [coords[1] for coords in packed_positions.values()]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    scale = max(x_span, y_span)

    return {
        node: ((coords[0] - x_center) / scale, (coords[1] - y_center) / scale)
        for node, coords in packed_positions.items()
    }


def render_empty_state() -> None:
    st.title("Holdings")
    st.info(
        "No reconstructed position state is available yet. Upload snapshots and/or trades, "
        "then rebuild position state."
    )


def _apply_holdings_filter_theme() -> None:
    return None


def render_filters(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    teams = (
        snapshot_df[COL_TEAM]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.ne("")]
        .unique()
        .tolist()
    )
    ordered_teams = [team for team in settings.display_team_order if team in teams]
    fallback_teams = [team for team in teams if team not in ordered_teams]
    team_options = ["All"] + ordered_teams + sorted(fallback_teams)

    side_options = ["All", "LONG", "SHORT", "CASH"]
    selected_team = st.sidebar.selectbox("Pod", options=team_options)
    selected_side = st.sidebar.selectbox("Position Side", options=side_options)
    ticker_search = st.sidebar.text_input("Search Ticker", value="").strip().upper()

    filtered = snapshot_df.copy()
    if selected_team != "All":
        filtered = filtered.loc[filtered[COL_TEAM] == selected_team].copy()
    if selected_side != "All":
        filtered = filtered.loc[filtered[COL_POSITION_SIDE] == selected_side].copy()
    if ticker_search:
        filtered = filtered.loc[
            filtered[COL_TICKER].fillna("").astype(str).str.upper().str.contains(ticker_search, na=False)
        ].copy()

    return filtered.reset_index(drop=True)


def render_constellation_controls() -> tuple[float, bool]:
    st.sidebar.header("Constellation")
    threshold = st.sidebar.slider("Correlation Threshold", min_value=0.20, max_value=0.90, value=0.50, step=0.05)
    compact_cluster_packing = st.sidebar.checkbox("Compact Cluster Packing", value=True)
    return float(threshold), bool(compact_cluster_packing)


def render_holdings_constellation(graph_payload: dict[str, object] | None, compact_cluster_packing: bool = True) -> None:
    st.subheader("Holdings Constellation")

    if nx is None:
        st.info("Install `networkx` to enable the holdings constellation on this page.")
        return
    if not graph_payload:
        st.info("Not enough non-cash holdings with recent price history to build the constellation.")
        return

    nodes_df = graph_payload["nodes_df"].copy()
    edges_df = graph_payload["edges_df"].copy()
    if nodes_df.empty or edges_df.empty:
        st.info("Not enough connected holdings remain after correlation thresholding to build the constellation.")
        return

    graph = nx.Graph()
    for row in nodes_df.itertuples(index=False):
        graph.add_node(
            row.ticker,
            pod=row.pod,
            market_value=float(getattr(row, COL_MARKET_VALUE)),
            abs_market_value=float(row.abs_market_value),
            portfolio_weight=float(getattr(row, COL_WEIGHT)),
            color=row.color,
        )
    for row in edges_df.itertuples(index=False):
        graph.add_edge(row.source, row.target, correlation=float(row.correlation), strength=float(row.strength))

    positions = _compute_graph_positions(graph, compact_cluster_packing=compact_cluster_packing)
    total_abs_mv = float(pd.to_numeric(nodes_df["abs_market_value"], errors="coerce").sum()) or 1.0

    fig = go.Figure()
    edge_widths: list[float] = []
    edge_hover_x: list[float] = []
    edge_hover_y: list[float] = []
    edge_hover_text: list[str] = []
    for source, target, attrs in graph.edges(data=True):
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        edge_width = 1.0 + 3.5 * float(attrs.get("strength", 0.0))
        edge_widths.append(edge_width)
        edge_hover_x.append((source_x + target_x) / 2)
        edge_hover_y.append((source_y + target_y) / 2)
        edge_hover_text.append(
            "<br>".join(
                [
                    f"{source} ↔ {target}",
                    f"Correlation: {float(attrs.get('correlation', 0.0)):.2f}",
                ]
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[source_x, target_x],
                y=[source_y, target_y],
                mode="lines",
                line=dict(width=edge_width, color="rgba(148, 163, 184, 0.35)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_x = []
    node_y = []
    node_sizes = []
    node_colors = []
    node_labels = []
    node_text = []
    for node, attrs in graph.nodes(data=True):
        x_pos, y_pos = positions[node]
        abs_mv = float(attrs.get("abs_market_value", 0.0))
        scaled_size = 18 + 34 * (abs_mv / total_abs_mv) ** 0.5
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_sizes.append(scaled_size)
        node_colors.append(attrs.get("color", "#94A3B8"))
        node_labels.append(node)
        node_text.append(
            "<br>".join(
                [
                    f"<b>{node}</b>",
                    f"Pod: {attrs.get('pod', 'N/A')}",
                    f"Market Value: {_format_currency(attrs.get('market_value'))}",
                    f"Portfolio Weight: {_format_percent(attrs.get('portfolio_weight'))}",
                ]
            )
        )

    edge_hover_trace = go.Scatter(
        x=edge_hover_x,
        y=edge_hover_y,
        mode="markers",
        marker=dict(size=[max(8.0, width * 2.4) for width in edge_widths], color="rgba(0,0,0,0)"),
        text=edge_hover_text,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        hovertext=node_text,
        hovertemplate="%{hovertext}<extra></extra>",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.5, color="rgba(255,255,255,0.85)"),
            opacity=0.94,
        ),
        showlegend=False,
    )

    fig.add_trace(edge_hover_trace)
    fig.add_trace(node_trace)
    fig.update_layout(
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{len(nodes_df)} holdings shown • "
        f"{len(edges_df)} correlation edges • "
        f"|corr| ≥ {graph_payload['threshold']:.2f} • "
        f"{graph_payload['isolated_count']} isolated holdings discarded"
    )


def render_summary_metrics(filtered_df: pd.DataFrame) -> None:
    total_market_value = float(pd.to_numeric(filtered_df[COL_MARKET_VALUE], errors="coerce").sum(skipna=True)) if not filtered_df.empty else 0.0
    gross_exposure = float(pd.to_numeric(filtered_df[COL_MARKET_VALUE], errors="coerce").abs().sum(skipna=True)) if not filtered_df.empty else 0.0
    unique_tickers = int(filtered_df[COL_TICKER].nunique(dropna=True)) if not filtered_df.empty else 0
    unique_teams = int(filtered_df[COL_TEAM].nunique(dropna=True)) if not filtered_df.empty else 0

    with st.container(border=True):
        row = st.columns(4)
        with row[0]:
            _render_holdings_metric_card("Filtered Market Value", _format_currency(total_market_value))
        with row[1]:
            _render_holdings_metric_card("Gross Exposure", _format_currency(gross_exposure))
        with row[2]:
            _render_holdings_metric_card("Tickers", f"{unique_tickers:,}")
        with row[3]:
            _render_holdings_metric_card("Pods", f"{unique_teams:,}")
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def render_holdings_table(filtered_df: pd.DataFrame) -> None:
    st.subheader("Holdings Detail")

    if filtered_df.empty:
        st.warning("No holdings matched the current filters.")
        return

    display_cols = [
        col
        for col in [COL_TICKER, COL_TEAM, COL_POSITION_SIDE, COL_SHARES, COL_PRICE, COL_MARKET_VALUE, COL_WEIGHT]
        if col in filtered_df.columns
    ]
    display_df = filtered_df[display_cols].copy().sort_values(by=[COL_TEAM, COL_MARKET_VALUE], ascending=[True, False])

    if COL_SHARES in display_df.columns:
        display_df[COL_SHARES] = display_df[COL_SHARES].map(_format_number)
    if COL_PRICE in display_df.columns:
        display_df[COL_PRICE] = display_df[COL_PRICE].map(_format_currency)
    if COL_MARKET_VALUE in display_df.columns:
        display_df[COL_MARKET_VALUE] = display_df[COL_MARKET_VALUE].map(_format_currency)
    if COL_WEIGHT in display_df.columns:
        display_df[COL_WEIGHT] = display_df[COL_WEIGHT].map(_format_percent)

    st.dataframe(left_align_dataframe(display_df), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Holdings", layout="wide")
    apply_app_theme()
    apply_holdings_page_theme()
    _apply_holdings_filter_theme()
    st.title("Holdings")

    snapshot_df = get_holdings_snapshot()
    if snapshot_df.empty:
        render_empty_state()
        return

    threshold, compact_cluster_packing = render_constellation_controls()
    filtered_df = render_filters(snapshot_df)
    render_summary_metrics(filtered_df)
    st.divider()

    constellation_history_df = get_constellation_price_history(filtered_df, lookback_days=120)
    graph_payload = build_correlation_graph(
        filtered_df,
        constellation_history_df,
        threshold=threshold,
    )
    render_holdings_constellation(graph_payload, compact_cluster_packing=compact_cluster_packing)
    st.divider()
    render_holdings_table(filtered_df)


if __name__ == "__main__":
    main()
