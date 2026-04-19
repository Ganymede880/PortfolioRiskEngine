"""
Ledger and portfolio-state reconstruction utilities.

This module is the core state engine for the CMCSIF portfolio tracker.

Responsibilities:
- rebuild expected positions as of a date
- apply trade receipts to prior positions
- update cost basis
- derive cash movements from trades
- compare expected positions to authoritative snapshots
- generate reconciliation events
- generate cash offsets for reconciliations

Design principles:
- snapshots are authoritative checkpoints
- trades mutate state between snapshots
- cash absorbs unmatched buys/sells, fees, withdrawals, and reconciliations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


# ============================================================================
# Constants
# ============================================================================
POSITION_KEY_COLS = ["team", "ticker", "position_side"]
CASH_LIKE_TICKERS = {"CASH", "EUR", "GBP", "NOGXX"}

TRADE_SIDE_TO_POSITION_SIDE = {
    "BUY": "LONG",
    "SELL": "LONG",
    "SHORT_SELL": "SHORT",
    "COVER": "SHORT",
}

BUY_LIKE_SIDES = {"BUY", "COVER"}
SELL_LIKE_SIDES = {"SELL", "SHORT_SELL"}


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass
class ReconciliationConfig:
    """
    Configuration for snapshot reconciliation.
    """
    default_assumed_price: float | None = None
    effective_date_fallback_to_snapshot_date: bool = True


# ============================================================================
# Basic helpers
# ============================================================================
def _standardize_position_frame(position_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a position-state-like DataFrame has canonical columns and clean types.

    Expected canonical columns:
    - team
    - ticker
    - position_side
    - shares
    - cost_basis_per_share
    - total_cost_basis
    - is_reconciled (optional)
    """
    if position_df.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "ticker",
                "position_side",
                "shares",
                "cost_basis_per_share",
                "total_cost_basis",
                "is_reconciled",
            ]
        )

    df = position_df.copy()

    for col in ["team", "ticker", "position_side"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype(str).str.strip()

    for col in ["shares", "cost_basis_per_share", "total_cost_basis"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_reconciled" not in df.columns:
        df["is_reconciled"] = False

    df = df.dropna(subset=["team", "ticker", "position_side"]).copy()
    df = df.reset_index(drop=True)

    return df


def _standardize_trade_frame(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a trade-receipt-like DataFrame has canonical columns and clean types.
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=[
                "trade_date",
                "settlement_date",
                "team",
                "ticker",
                "trade_side",
                "quantity",
                "gross_price",
                "commission",
                "fees",
                "net_cash_amount",
            ]
        )

    df = trades_df.copy()

    for col in ["team", "ticker", "trade_side"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype(str).str.strip().str.upper()

    for col in ["trade_date", "settlement_date"]:
        if col not in df.columns:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["quantity", "gross_price", "commission", "fees", "net_cash_amount"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["team", "ticker", "trade_side", "quantity", "gross_price"]).copy()
    df = df.sort_values(["trade_date", "settlement_date", "team", "ticker"]).reset_index(drop=True)

    return df


def _position_key(team: str, ticker: str, position_side: str) -> Tuple[str, str, str]:
    return (str(team).strip(), str(ticker).strip(), str(position_side).strip())


def _is_cash_like_position(team: str, ticker: str, position_side: str) -> bool:
    team_clean = str(team).strip().upper()
    ticker_clean = str(ticker).strip().upper()
    side_clean = str(position_side).strip().upper()
    return (
        ticker_clean in CASH_LIKE_TICKERS
        or team_clean == "CASH"
        or side_clean == "CASH"
    )


def _ensure_cash_position(
    positions_df: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """
    Ensure a CASH row exists for the supplied team.
    """
    df = positions_df.copy()

    cash_mask = (
        (df["team"] == team)
        & (df["ticker"] == "CASH")
        & (df["position_side"] == "CASH")
    )

    if not cash_mask.any():
        new_row = pd.DataFrame([{
            "team": team,
            "ticker": "CASH",
            "position_side": "CASH",
            "shares": 0.0,
            "cost_basis_per_share": 1.0,
            "total_cost_basis": 0.0,
            "is_reconciled": False,
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    return df


def _apply_cash_change(
    positions_df: pd.DataFrame,
    team: str,
    amount: float,
    note_reconciled: bool = False,
) -> pd.DataFrame:
    """
    Apply a cash amount to the team's CASH position.

    Convention:
    - positive amount => increase cash
    - negative amount => decrease cash
    """
    df = _ensure_cash_position(positions_df, team)

    cash_mask = (
        (df["team"] == team)
        & (df["ticker"] == "CASH")
        & (df["position_side"] == "CASH")
    )

    df.loc[cash_mask, "shares"] = pd.to_numeric(df.loc[cash_mask, "shares"], errors="coerce").fillna(0.0) + amount
    df.loc[cash_mask, "cost_basis_per_share"] = 1.0
    df.loc[cash_mask, "total_cost_basis"] = pd.to_numeric(
        df.loc[cash_mask, "shares"], errors="coerce"
    ).fillna(0.0)

    if note_reconciled:
        df.loc[cash_mask, "is_reconciled"] = True

    return df


# ============================================================================
# Trade cash math
# ============================================================================
def derive_trade_cash_amount(trade_row: pd.Series) -> float:
    """
    Derive signed trade cash amount.

    Convention:
    - BUY / COVER => negative cash
    - SELL / SHORT_SELL => positive cash

    Uses net_cash_amount if present; otherwise derives from quantity, price,
    commission, and fees.
    """
    explicit = pd.to_numeric(pd.Series([trade_row.get("net_cash_amount")]), errors="coerce").iloc[0]
    if pd.notna(explicit):
        return float(explicit)

    quantity = float(pd.to_numeric(pd.Series([trade_row.get("quantity")]), errors="coerce").iloc[0] or 0.0)
    gross_price = float(pd.to_numeric(pd.Series([trade_row.get("gross_price")]), errors="coerce").iloc[0] or 0.0)
    commission = float(pd.to_numeric(pd.Series([trade_row.get("commission")]), errors="coerce").fillna(0.0).iloc[0])
    fees = float(pd.to_numeric(pd.Series([trade_row.get("fees")]), errors="coerce").fillna(0.0).iloc[0])

    gross_notional = quantity * gross_price
    total_fees = commission + fees
    trade_side = str(trade_row.get("trade_side", "")).strip().upper()

    if trade_side in BUY_LIKE_SIDES:
        return -(gross_notional + total_fees)

    if trade_side in SELL_LIKE_SIDES:
        return gross_notional - total_fees

    return 0.0


# ============================================================================
# Position mutation logic
# ============================================================================
def _upsert_position_row(
    positions_df: pd.DataFrame,
    team: str,
    ticker: str,
    position_side: str,
) -> Tuple[pd.DataFrame, int]:
    """
    Ensure a position row exists and return (updated_df, row_index).
    """
    df = positions_df.copy()

    mask = (
        (df["team"] == team)
        & (df["ticker"] == ticker)
        & (df["position_side"] == position_side)
    )

    if mask.any():
        idx = df.index[mask][0]
        return df, int(idx)

    new_row = pd.DataFrame([{
        "team": team,
        "ticker": ticker,
        "position_side": position_side,
        "shares": 0.0,
        "cost_basis_per_share": pd.NA,
        "total_cost_basis": pd.NA,
        "is_reconciled": False,
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    return df, int(df.index[-1])


def apply_single_trade_to_positions(
    positions_df: pd.DataFrame,
    trade_row: pd.Series,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply a single normalized trade row to current position state.

    Returns:
    - updated positions DataFrame
    - cash ledger entry dict
    """
    df = _standardize_position_frame(positions_df)

    team = str(trade_row["team"]).strip()
    ticker = str(trade_row["ticker"]).strip()
    trade_side = str(trade_row["trade_side"]).strip().upper()
    quantity = float(trade_row["quantity"])
    gross_price = float(trade_row["gross_price"])
    trade_date = pd.to_datetime(trade_row["trade_date"], errors="coerce")
    activity_date = trade_date.date() if pd.notna(trade_date) else pd.to_datetime(trade_row["settlement_date"]).date()

    position_side = TRADE_SIDE_TO_POSITION_SIDE.get(trade_side)
    if position_side is None:
        raise ValueError(f"Unsupported trade_side: {trade_side}")

    df, idx = _upsert_position_row(df, team, ticker, position_side)

    current_shares = float(pd.to_numeric(pd.Series([df.at[idx, "shares"]]), errors="coerce").fillna(0.0).iloc[0])
    current_total_cost = pd.to_numeric(pd.Series([df.at[idx, "total_cost_basis"]]), errors="coerce").fillna(0.0).iloc[0]
    current_total_cost = float(current_total_cost)

    if trade_side in {"BUY", "SHORT_SELL"}:
        new_shares = current_shares + quantity
        new_total_cost = current_total_cost + (quantity * gross_price)

        df.at[idx, "shares"] = new_shares
        df.at[idx, "total_cost_basis"] = new_total_cost
        df.at[idx, "cost_basis_per_share"] = (new_total_cost / new_shares) if new_shares != 0 else pd.NA

    elif trade_side in {"SELL", "COVER"}:
        if quantity > current_shares:
            raise ValueError(
                f"Trade would make position negative for {team}-{ticker}-{position_side}. "
                f"Current shares={current_shares}, trade quantity={quantity}."
            )

        current_cost_per_share = (
            float(current_total_cost / current_shares) if current_shares != 0 else 0.0
        )

        new_shares = current_shares - quantity
        new_total_cost = current_total_cost - (quantity * current_cost_per_share)

        df.at[idx, "shares"] = new_shares
        df.at[idx, "total_cost_basis"] = new_total_cost if new_shares != 0 else 0.0
        df.at[idx, "cost_basis_per_share"] = (
            new_total_cost / new_shares if new_shares != 0 else pd.NA
        )

    cash_amount = derive_trade_cash_amount(trade_row)
    df = _apply_cash_change(df, team=team, amount=cash_amount, note_reconciled=False)

    # Drop fully closed non-cash positions
    non_cash_zero_mask = (
        (df["ticker"] != "CASH")
        & pd.to_numeric(df["shares"], errors="coerce").fillna(0.0).eq(0.0)
    )
    df = df.loc[~non_cash_zero_mask].reset_index(drop=True)

    cash_entry = {
        "activity_date": activity_date,
        "team": team,
        "amount": cash_amount,
        "activity_type": "TRADE",
        "reference_type": "TRADE_RECEIPT",
        "reference_id": None,
        "note": f"{trade_side} {quantity} {ticker} @ {gross_price}",
    }

    return df, cash_entry


def apply_trades_to_positions(
    base_positions_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply a sequence of normalized trades to a base position state.

    Returns:
    - updated position state
    - cash ledger DataFrame derived from trades
    """
    positions = _standardize_position_frame(base_positions_df)
    trades = _standardize_trade_frame(trades_df)

    cash_entries: List[dict] = []

    if trades.empty:
        return positions, pd.DataFrame(columns=[
            "activity_date", "team", "amount", "activity_type",
            "reference_type", "reference_id", "note"
        ])

    for _, trade_row in trades.iterrows():
        positions, cash_entry = apply_single_trade_to_positions(positions, trade_row)
        cash_entries.append(cash_entry)

    return positions.reset_index(drop=True), pd.DataFrame(cash_entries)


# ============================================================================
# Snapshot reconciliation
# ============================================================================
def compare_expected_positions_to_snapshot(
    expected_positions_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare expected reconstructed positions to an authoritative snapshot.

    Returns one row per mismatch with:
    - team
    - ticker
    - position_side
    - expected_shares
    - snapshot_shares
    - delta_shares
    """
    expected = _standardize_position_frame(expected_positions_df)[
        ["team", "ticker", "position_side", "shares"]
    ].copy()
    expected = expected.rename(columns={"shares": "expected_shares"})

    snapshot = _standardize_position_frame(snapshot_df)[
        ["team", "ticker", "position_side", "shares"]
    ].copy()
    snapshot = snapshot.rename(columns={"shares": "snapshot_shares"})

    merged = expected.merge(
        snapshot,
        on=POSITION_KEY_COLS,
        how="outer",
    )

    merged["expected_shares"] = pd.to_numeric(merged["expected_shares"], errors="coerce").fillna(0.0)
    merged["snapshot_shares"] = pd.to_numeric(merged["snapshot_shares"], errors="coerce").fillna(0.0)
    merged["delta_shares"] = merged["snapshot_shares"] - merged["expected_shares"]

    mismatches = merged.loc[merged["delta_shares"] != 0].copy()
    return mismatches.reset_index(drop=True)


def generate_reconciliation_events(
    expected_positions_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    snapshot_date,
    effective_date,
    assumed_price_map: Dict[Tuple[str, str, str], float] | None = None,
    config: ReconciliationConfig | None = None,
) -> pd.DataFrame:
    """
    Generate reconciliation event rows from snapshot mismatch.

    assumed_price_map keys are:
    - (team, ticker, position_side)
    """
    if config is None:
        config = ReconciliationConfig()

    mismatches = compare_expected_positions_to_snapshot(expected_positions_df, snapshot_df)
    if not mismatches.empty:
        mismatches = mismatches.loc[
            ~mismatches.apply(
                lambda row: _is_cash_like_position(
                    row["team"],
                    row["ticker"],
                    row["position_side"],
                ),
                axis=1,
            )
        ].reset_index(drop=True)

    if mismatches.empty:
        return pd.DataFrame(columns=[
            "snapshot_date",
            "effective_date",
            "team",
            "ticker",
            "position_side",
            "expected_shares",
            "snapshot_shares",
            "delta_shares",
            "assumed_price",
            "estimated_cash_impact",
            "note",
        ])

    rows: List[dict] = []

    for _, row in mismatches.iterrows():
        key = (row["team"], row["ticker"], row["position_side"])
        assumed_price = None

        if assumed_price_map is not None:
            assumed_price = assumed_price_map.get(key)

        if assumed_price is None:
            assumed_price = config.default_assumed_price

        delta_shares = float(row["delta_shares"])
        estimated_cash_impact = None

        if assumed_price is not None:
            position_side = str(row["position_side"]).strip().upper()
            if position_side == "SHORT":
                estimated_cash_impact = delta_shares * assumed_price
            else:
                # For long positions, more shares than expected implies an inferred buy.
                estimated_cash_impact = -(delta_shares * assumed_price)

        rows.append({
            "snapshot_date": pd.to_datetime(snapshot_date).date(),
            "effective_date": pd.to_datetime(effective_date).date(),
            "team": row["team"],
            "ticker": row["ticker"],
            "position_side": row["position_side"],
            "expected_shares": row["expected_shares"],
            "snapshot_shares": row["snapshot_shares"],
            "delta_shares": delta_shares,
            "assumed_price": assumed_price,
            "estimated_cash_impact": estimated_cash_impact,
            "note": "Authoritative snapshot reconciliation",
        })

    return pd.DataFrame(rows)


def generate_cash_reconciliation_entries(
    expected_positions_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
    effective_date,
) -> pd.DataFrame:
    """
    Derive residual cash adjustments by team.

    We first infer buy/sell cash from non-cash reconciliation rows, then compare
    the resulting expected cash to the authoritative snapshot cash. The
    difference is the missing-history cash residual we should log in the cash
    ledger.
    """
    expected = _standardize_position_frame(expected_positions_df)
    snapshot = _standardize_position_frame(snapshot_df)

    def _cash_value_by_team(df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}

        working = df.copy()
        working["cash_value"] = pd.to_numeric(
            working.get("total_cost_basis"),
            errors="coerce",
        )
        missing_mask = working["cash_value"].isna()
        if missing_mask.any():
            working.loc[missing_mask, "cash_value"] = (
                pd.to_numeric(working.loc[missing_mask, "shares"], errors="coerce").fillna(0.0)
                * pd.to_numeric(working.loc[missing_mask, "cost_basis_per_share"], errors="coerce").fillna(1.0)
            )
        working = working.loc[
            working.apply(
                lambda row: _is_cash_like_position(
                    row["team"],
                    row["ticker"],
                    row["position_side"],
                ),
                axis=1,
            )
        ].copy()
        if working.empty:
            return {}
        return (
            working.groupby("team", dropna=False)["cash_value"]
            .sum(min_count=1)
            .fillna(0.0)
            .to_dict()
        )

    expected_cash = _cash_value_by_team(expected)
    snapshot_cash = _cash_value_by_team(snapshot)

    inferred_security_cash: Dict[str, float] = {}
    if not reconciliation_df.empty and "estimated_cash_impact" in reconciliation_df.columns:
        inferred_security_cash = (
            reconciliation_df.assign(
                estimated_cash_impact=pd.to_numeric(
                    reconciliation_df["estimated_cash_impact"],
                    errors="coerce",
                ).fillna(0.0)
            )
            .groupby("team", dropna=False)["estimated_cash_impact"]
            .sum(min_count=1)
            .fillna(0.0)
            .to_dict()
        )

    teams = sorted(set(expected_cash) | set(snapshot_cash) | set(inferred_security_cash))
    rows: List[dict] = []

    for team in teams:
        residual = (
            float(snapshot_cash.get(team, 0.0))
            - float(expected_cash.get(team, 0.0))
            - float(inferred_security_cash.get(team, 0.0))
        )
        if abs(residual) < 1e-9:
            continue

        rows.append(
            {
                "activity_date": pd.to_datetime(effective_date).date(),
                "team": team,
                "amount": residual,
                "activity_type": "RECONCILIATION",
                "reference_type": "RECONCILIATION_RESIDUAL",
                "reference_id": None,
                "note": "Residual cash reconciliation from incomplete trade history",
            }
        )

    return pd.DataFrame(rows)


def apply_reconciliation_to_positions(
    expected_positions_df: pd.DataFrame,
    reconciliation_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply reconciliation events to reconstructed positions.

    Returns:
    - updated positions
    - cash ledger entries derived from reconciliation rows
    """
    positions = _standardize_position_frame(expected_positions_df)
    cash_entries: List[dict] = []

    if reconciliation_df.empty:
        return positions, pd.DataFrame(columns=[
            "activity_date", "team", "amount", "activity_type",
            "reference_type", "reference_id", "note"
        ])

    for _, row in reconciliation_df.iterrows():
        team = str(row["team"]).strip()
        ticker = str(row["ticker"]).strip()
        position_side = str(row["position_side"]).strip()
        snapshot_shares = float(pd.to_numeric(pd.Series([row["snapshot_shares"]]), errors="coerce").fillna(0.0).iloc[0])

        positions, idx = _upsert_position_row(positions, team, ticker, position_side)
        positions.at[idx, "shares"] = snapshot_shares
        positions.at[idx, "is_reconciled"] = True

        estimated_cash_impact = pd.to_numeric(pd.Series([row.get("estimated_cash_impact")]), errors="coerce").iloc[0]
        if pd.notna(estimated_cash_impact):
            positions = _apply_cash_change(
                positions_df=positions,
                team=team,
                amount=float(estimated_cash_impact),
                note_reconciled=True,
            )

            cash_entries.append({
                "activity_date": pd.to_datetime(row["effective_date"]).date(),
                "team": team,
                "amount": float(estimated_cash_impact),
                "activity_type": "RECONCILIATION",
                "reference_type": "RECONCILIATION_EVENT",
                "reference_id": None,
                "note": f"Snapshot reconciliation for {ticker}",
            })

    # Drop zero-share non-cash rows
    non_cash_zero_mask = (
        (positions["ticker"] != "CASH")
        & pd.to_numeric(positions["shares"], errors="coerce").fillna(0.0).eq(0.0)
    )
    positions = positions.loc[~non_cash_zero_mask].reset_index(drop=True)

    return positions, pd.DataFrame(cash_entries)


# ============================================================================
# End-to-end reconstruction helpers
# ============================================================================
def rebuild_positions_from_snapshot_and_trades(
    snapshot_df: pd.DataFrame,
    trades_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Start from an authoritative snapshot and apply subsequent trades.

    Returns:
    - reconstructed positions
    - cash ledger entries from trades
    """
    base_positions = _standardize_position_frame(snapshot_df)
    return apply_trades_to_positions(base_positions, trades_df)


def reconcile_expected_positions_to_authoritative_snapshot(
    expected_positions_df: pd.DataFrame,
    authoritative_snapshot_df: pd.DataFrame,
    snapshot_date,
    effective_date,
    assumed_price_map: Dict[Tuple[str, str, str], float] | None = None,
    config: ReconciliationConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare expected positions to a new authoritative snapshot and apply
    reconciliation.

    Returns:
    - updated positions after reconciliation
    - reconciliation events DataFrame
    - cash ledger entries DataFrame from reconciliation
    """
    recon_df = generate_reconciliation_events(
        expected_positions_df=expected_positions_df,
        snapshot_df=authoritative_snapshot_df,
        snapshot_date=snapshot_date,
        effective_date=effective_date,
        assumed_price_map=assumed_price_map,
        config=config,
    )

    updated_positions, _ = apply_reconciliation_to_positions(
        expected_positions_df=expected_positions_df,
        reconciliation_df=recon_df,
    )
    cash_entries = generate_cash_reconciliation_entries(
        expected_positions_df=expected_positions_df,
        snapshot_df=authoritative_snapshot_df,
        reconciliation_df=recon_df,
        effective_date=effective_date,
    )

    return updated_positions, recon_df, cash_entries
