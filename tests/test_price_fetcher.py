import sys
import types
from pathlib import Path

import pandas as pd


def _load_price_fetcher_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "src" / "data" / "price_fetcher.py"

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None

    previous_dotenv = sys.modules.get("dotenv")
    sys.modules["dotenv"] = fake_dotenv

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("price_fetcher_for_tests", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = previous_dotenv


def test_normalize_yfinance_history_frame_falls_back_to_close_when_adj_close_missing():
    module = _load_price_fetcher_module()

    raw_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-04-16", "2026-04-17"]),
            "Close": [100.0, 101.5],
            "Volume": [1000, 1100],
        }
    )

    normalized = module._normalize_yfinance_history_frame(raw_df, "ABC")

    assert list(normalized.columns) == ["date", "ticker", "close", "adj_close"]
    assert normalized["ticker"].tolist() == ["ABC", "ABC"]
    assert normalized["close"].tolist() == [100.0, 101.5]
    assert normalized["adj_close"].tolist() == [100.0, 101.5]
