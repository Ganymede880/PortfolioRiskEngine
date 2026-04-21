from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.analytics.performance import build_flow_adjusted_benchmark_series
from src.data.price_fetcher import fetch_sp500_sector_proxy_weights


TEAM_BENCHMARK_COMPONENTS: dict[str, list[dict[str, str | None]]] = {
    "F&R": [
        {"ticker": "XLF", "sector_key": "financials"},
        {"ticker": "XLRE", "sector_key": "real_estate"},
    ],
    "M&I": [
        {"ticker": "XLB", "sector_key": "materials"},
        {"ticker": "XLI", "sector_key": "industrials"},
    ],
    "Consumer": [
        {"ticker": "XLP", "sector_key": "consumer_defensive"},
        {"ticker": "XLY", "sector_key": "consumer_cyclical"},
    ],
    "TMT": [
        {"ticker": "XLK", "sector_key": "technology"},
    ],
    "Healthcare": [
        {"ticker": "XLV", "sector_key": "healthcare"},
    ],
    "E&U": [
        {"ticker": "XLU", "sector_key": "utilities"},
        {"ticker": "XLE", "sector_key": "energy"},
    ],
}


def get_team_benchmark_spec(team: str) -> dict[str, Any]:
    normalized_team = str(team).strip()
    raw_components = TEAM_BENCHMARK_COMPONENTS.get(
        normalized_team,
        [{"ticker": "SPY", "sector_key": None}],
    )

    sector_proxy_weights = fetch_sp500_sector_proxy_weights()
    weights: list[float] = []
    for component in raw_components:
        sector_key = component.get("sector_key")
        weight = sector_proxy_weights.get(str(sector_key)) if sector_key is not None else 1.0
        numeric_weight = pd.to_numeric(pd.Series([weight]), errors="coerce").iloc[0]
        weights.append(float(numeric_weight) if pd.notna(numeric_weight) and float(numeric_weight) > 0 else np.nan)

    valid_weights = [weight for weight in weights if pd.notna(weight) and float(weight) > 0]
    if valid_weights:
        weight_sum = float(sum(valid_weights))
        normalized_weights = [
            (float(weight) / weight_sum) if pd.notna(weight) and float(weight) > 0 else 0.0
            for weight in weights
        ]
    else:
        equal_weight = 1.0 / max(len(raw_components), 1)
        normalized_weights = [equal_weight for _ in raw_components]

    components = []
    for component, weight in zip(raw_components, normalized_weights):
        components.append(
            {
                "ticker": str(component["ticker"]).strip().upper(),
                "sector_key": component.get("sector_key"),
                "weight": float(weight),
            }
        )

    return {
        "team": normalized_team,
        "label": f"{normalized_team} Benchmark",
        "components": components,
    }


def describe_team_benchmark(team: str) -> str:
    spec = get_team_benchmark_spec(team)
    components = spec.get("components", [])
    if not components:
        return "SPY"
    if len(components) == 1:
        return str(components[0]["ticker"])
    return " + ".join(
        f"{component['ticker']} ({component['weight']:.0%})"
        for component in components
        if float(component.get("weight", 0.0)) > 0
    )


def get_team_benchmark_tickers(team: str) -> list[str]:
    spec = get_team_benchmark_spec(team)
    return [
        str(component["ticker"]).strip().upper()
        for component in spec.get("components", [])
        if str(component.get("ticker", "")).strip()
    ]


def build_team_benchmark_aum_frame(
    team: str,
    price_matrix: pd.DataFrame,
    dates: pd.Series | pd.Index,
    external_flow_series: pd.Series,
    initial_value: float,
) -> pd.DataFrame:
    spec = get_team_benchmark_spec(team)
    benchmark_dates = pd.to_datetime(pd.Index(dates), errors="coerce")
    result = pd.DataFrame({"date": benchmark_dates})
    if len(result.index) == 0 or not np.isfinite(initial_value) or float(initial_value) == 0:
        result["benchmark_aum"] = pd.NA
        return result

    component_tickers = [
        str(component["ticker"]).strip().upper()
        for component in spec.get("components", [])
        if str(component.get("ticker", "")).strip().upper() in price_matrix.columns
    ]
    if not component_tickers:
        result["benchmark_aum"] = pd.NA
        return result

    price_slice = (
        price_matrix.reindex(benchmark_dates)[component_tickers]
        .apply(pd.to_numeric, errors="coerce")
        .copy()
    )
    component_returns = price_slice.pct_change()

    component_weight_map = {
        str(component["ticker"]).strip().upper(): float(component.get("weight", 0.0))
        for component in spec.get("components", [])
    }
    weight_vector = pd.Series(component_weight_map).reindex(component_tickers).fillna(0.0)

    available_weights = component_returns.notna().mul(weight_vector, axis=1)
    composite_returns = component_returns.mul(weight_vector, axis=1).sum(axis=1, min_count=1)
    available_weight_totals = available_weights.sum(axis=1)
    composite_returns = composite_returns / available_weight_totals.where(available_weight_totals.gt(0))

    benchmark_aum = pd.Series(index=result.index, dtype="float64")
    benchmark_aum.iloc[0] = float(initial_value)
    if len(result.index) > 1:
        benchmark_tail = build_flow_adjusted_benchmark_series(
            benchmark_return_series=composite_returns.iloc[1:],
            external_flow_series=pd.to_numeric(external_flow_series, errors="coerce").fillna(0.0).iloc[1:],
            initial_value=float(initial_value),
        )
        benchmark_aum.iloc[1:] = benchmark_tail.values
    result["benchmark_aum"] = benchmark_aum.values

    for ticker in component_tickers:
        component_aum = pd.Series(index=result.index, dtype="float64")
        component_aum.iloc[0] = float(initial_value)
        if len(result.index) > 1:
            component_tail = build_flow_adjusted_benchmark_series(
                benchmark_return_series=component_returns[ticker].iloc[1:],
                external_flow_series=pd.to_numeric(external_flow_series, errors="coerce").fillna(0.0).iloc[1:],
                initial_value=float(initial_value),
            )
            component_aum.iloc[1:] = component_tail.values
        result[f"benchmark_component_{ticker}_aum"] = component_aum.values

    return result
