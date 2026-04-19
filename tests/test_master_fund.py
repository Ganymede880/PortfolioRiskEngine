import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _load_master_fund_module():
    project_root = Path(__file__).resolve().parents[1]
    page_path = project_root / "app" / "pages" / "1_Total_Fund_View.py"

    fake_streamlit = types.SimpleNamespace(
        cache_data=lambda *args, **kwargs: (lambda fn: fn),
        title=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        subheader=lambda *args, **kwargs: None,
        columns=lambda *args, **kwargs: [],
        metric=lambda *args, **kwargs: None,
        dataframe=lambda *args, **kwargs: None,
    )
    fake_plotly = types.ModuleType("plotly")
    fake_plotly_express = types.ModuleType("plotly.express")
    fake_plotly_graph_objects = types.ModuleType("plotly.graph_objects")
    fake_settings_module = types.ModuleType("src.config.settings")
    fake_settings_module.settings = types.SimpleNamespace(
        price_refresh_interval_seconds=300,
        display_team_order=["Consumer", "E&U", "F&R", "Healthcare", "TMT", "M&I", "Cash"],
    )
    fake_portfolio_module = types.ModuleType("src.analytics.portfolio")
    fake_portfolio_module.build_current_portfolio_snapshot = lambda position_state_df, latest_prices_df: position_state_df
    fake_portfolio_module.summarize_total_portfolio = lambda snapshot_df: {}
    fake_price_fetcher_module = types.ModuleType("src.data.price_fetcher")
    fake_price_fetcher_module.fetch_latest_prices = lambda tickers: (pd.DataFrame(), [])
    fake_price_fetcher_module.fetch_multiple_price_histories = lambda tickers, lookback_days=None: pd.DataFrame()
    fake_crud_module = types.ModuleType("src.db.crud")
    fake_crud_module.load_all_portfolio_snapshots = lambda session: pd.DataFrame()
    fake_crud_module.load_position_state = lambda session: pd.DataFrame()
    fake_session_module = types.ModuleType("src.db.session")

    class _DummySessionScope:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_session_module.session_scope = _DummySessionScope

    previous_modules = {
        "streamlit": sys.modules.get("streamlit"),
        "plotly": sys.modules.get("plotly"),
        "plotly.express": sys.modules.get("plotly.express"),
        "plotly.graph_objects": sys.modules.get("plotly.graph_objects"),
        "src.config.settings": sys.modules.get("src.config.settings"),
        "src.analytics.portfolio": sys.modules.get("src.analytics.portfolio"),
        "src.data.price_fetcher": sys.modules.get("src.data.price_fetcher"),
        "src.db.crud": sys.modules.get("src.db.crud"),
        "src.db.session": sys.modules.get("src.db.session"),
    }

    sys.modules["streamlit"] = fake_streamlit
    sys.modules["plotly"] = fake_plotly
    sys.modules["plotly.express"] = fake_plotly_express
    sys.modules["plotly.graph_objects"] = fake_plotly_graph_objects
    sys.modules["src.config.settings"] = fake_settings_module
    sys.modules["src.analytics.portfolio"] = fake_portfolio_module
    sys.modules["src.data.price_fetcher"] = fake_price_fetcher_module
    sys.modules["src.db.crud"] = fake_crud_module
    sys.modules["src.db.session"] = fake_session_module

    try:
        spec = importlib.util.spec_from_file_location("master_fund_page_for_tests", page_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_determine_history_start_date_includes_buffer_for_trailing_lookback():
    module = _load_master_fund_module()

    today = pd.Timestamp("2026-04-18")
    oldest_snapshot_date = pd.Timestamp("2025-03-18")

    start_date = module._determine_history_start_date(oldest_snapshot_date, today)

    assert start_date == pd.Timestamp("2025-04-11")


def test_compute_trailing_return_uses_nearest_prior_business_day():
    module = _load_master_fund_module()

    history_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-16", "2025-04-18", "2026-04-17"]),
            "portfolio_aum": [100.0, 101.0, 120.0],
        }
    )

    trailing_return = module._compute_trailing_return(history_df, 365)

    assert round(trailing_return, 10) == 0.2


def test_build_price_matrix_returns_requested_index_for_empty_prices():
    module = _load_master_fund_module()

    business_dates = pd.bdate_range("2025-04-14", periods=5)

    price_matrix = module._build_price_matrix(pd.DataFrame(), business_dates)

    assert list(price_matrix.index) == list(business_dates)
    assert price_matrix.empty


def test_apply_position_values_assigns_zero_to_missing_security_prices():
    module = _load_master_fund_module()

    snapshot_df = pd.DataFrame(
        {
            "team": ["Cash", "Consumer"],
            "ticker": ["NOGXX", "DELISTED"],
            "position_side": ["LONG", "LONG"],
            "shares": [5000.0, 100.0],
        }
    )

    portfolio_aum = module._apply_position_values(snapshot_df, price_map={})

    assert portfolio_aum == 5000.0
