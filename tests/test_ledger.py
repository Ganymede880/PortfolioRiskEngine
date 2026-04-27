import pandas as pd

from src.analytics.ledger import apply_trades_to_positions


def test_apply_trades_to_positions_reorders_same_day_opening_trade_before_reduction():
    base_positions_df = pd.DataFrame(
        [
            {
                "team": "Consumer",
                "ticker": "CASH",
                "position_side": "CASH",
                "shares": 0.0,
                "cost_basis_per_share": 1.0,
                "total_cost_basis": 0.0,
            }
        ]
    )
    trades_df = pd.DataFrame(
        [
            {
                "id": 20,
                "trade_date": "2025-10-20",
                "settlement_date": "2025-10-20",
                "team": "Consumer",
                "ticker": "XLP",
                "trade_side": "SELL",
                "quantity": 268.0,
                "gross_price": 79.695,
                "net_cash_amount": 21348.31,
            },
            {
                "id": 19,
                "trade_date": "2025-10-20",
                "settlement_date": "2025-10-20",
                "team": "Consumer",
                "ticker": "XLP",
                "trade_side": "BUY",
                "quantity": 740.0,
                "gross_price": 77.8571,
                "net_cash_amount": -57614.25,
            },
        ]
    )

    positions_df, cash_entries_df = apply_trades_to_positions(base_positions_df, trades_df)

    xlp_row = positions_df.loc[
        positions_df["team"].eq("Consumer")
        & positions_df["ticker"].eq("XLP")
        & positions_df["position_side"].eq("LONG")
    ].iloc[0]
    cash_row = positions_df.loc[
        positions_df["team"].eq("Consumer")
        & positions_df["ticker"].eq("CASH")
        & positions_df["position_side"].eq("CASH")
    ].iloc[0]

    assert xlp_row["shares"] == 472.0
    assert round(float(cash_row["shares"]), 2) == round(21348.31 - 57614.25, 2)
    assert len(cash_entries_df) == 2


def test_apply_trades_to_positions_still_raises_for_true_negative_position():
    base_positions_df = pd.DataFrame(
        [
            {
                "team": "Consumer",
                "ticker": "CASH",
                "position_side": "CASH",
                "shares": 0.0,
                "cost_basis_per_share": 1.0,
                "total_cost_basis": 0.0,
            }
        ]
    )
    trades_df = pd.DataFrame(
        [
            {
                "trade_date": "2025-10-20",
                "settlement_date": "2025-10-20",
                "team": "Consumer",
                "ticker": "XLP",
                "trade_side": "SELL",
                "quantity": 268.0,
                "gross_price": 79.695,
                "net_cash_amount": 21348.31,
            }
        ]
    )

    try:
        apply_trades_to_positions(base_positions_df, trades_df)
    except ValueError as exc:
        assert "Trade would make position negative for Consumer-XLP-LONG" in str(exc)
    else:
        raise AssertionError("Expected a negative-position ValueError for an unrecoverable sell trade.")
