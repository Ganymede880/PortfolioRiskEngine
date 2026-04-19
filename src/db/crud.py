"""
CRUD helpers for the CMCSIF portfolio tracker.

This module supports:
- authoritative portfolio snapshots
- trade receipt ingestion
- cash ledger activity
- reconciliation events
- reconstructed position state
- upload logging
- portfolio activity feed generation
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from src.db.models import (
    CashLedger,
    PortfolioSnapshot,
    PositionState,
    PriceHistory,
    ReconciliationEvent,
    TradeReceipt,
    UploadLog,
)


# ============================================================================
# Portfolio snapshots
# ============================================================================
def save_portfolio_snapshot(
    session: Session,
    snapshot_df: pd.DataFrame,
    snapshot_date: date,
    source_file: str | None = None,
    selected_sheet: str | None = None,
    replace_existing: bool = True,
) -> int:
    """
    Save an authoritative portfolio snapshot.

    Expected columns:
    - team
    - ticker
    - position_side
    - shares
    - cost_basis_per_share (optional)
    - total_cost_basis (optional)
    """
    if snapshot_df.empty:
        return 0

    if replace_existing:
        session.execute(
            delete(PortfolioSnapshot).where(
                PortfolioSnapshot.snapshot_date == snapshot_date
            )
        )

    rows_inserted = 0

    for _, row in snapshot_df.iterrows():
        team = _safe_str(row.get("team"))
        ticker = _safe_str(row.get("ticker"))
        position_side = _safe_str(row.get("position_side"))

        if not team or not ticker or not position_side:
            continue

        record = PortfolioSnapshot(
            snapshot_date=snapshot_date,
            team=team,
            ticker=ticker,
            position_side=position_side,
            shares=_safe_float(row.get("shares")) or 0.0,
            cost_basis_per_share=_safe_float(row.get("cost_basis_per_share")),
            total_cost_basis=_safe_float(row.get("total_cost_basis")),
            source_file=source_file or _safe_optional_str(row.get("source_file")),
            selected_sheet=selected_sheet or _safe_optional_str(row.get("selected_sheet")),
        )
        session.add(record)
        rows_inserted += 1

    session.flush()
    return rows_inserted


def load_portfolio_snapshot(
    session: Session,
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    """
    Load a portfolio snapshot for a given date.

    If snapshot_date is None, loads the most recent snapshot.
    """
    if snapshot_date is None:
        snapshot_date = get_latest_snapshot_date(session)

    if snapshot_date is None:
        return pd.DataFrame(
            columns=[
                "snapshot_date",
                "team",
                "ticker",
                "position_side",
                "shares",
                "cost_basis_per_share",
                "total_cost_basis",
                "source_file",
                "selected_sheet",
                "created_at",
            ]
        )

    stmt = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.snapshot_date == snapshot_date)
        .order_by(
            PortfolioSnapshot.team.asc(),
            PortfolioSnapshot.ticker.asc(),
            PortfolioSnapshot.position_side.asc(),
        )
    )
    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "snapshot_date": r.snapshot_date,
            "team": r.team,
            "ticker": r.ticker,
            "position_side": r.position_side,
            "shares": r.shares,
            "cost_basis_per_share": r.cost_basis_per_share,
            "total_cost_basis": r.total_cost_basis,
            "source_file": r.source_file,
            "selected_sheet": r.selected_sheet,
            "created_at": r.created_at,
        }
        for r in records
    ])


def load_all_portfolio_snapshots(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Load all snapshot rows, optionally filtered by date range.
    """
    stmt = select(PortfolioSnapshot)

    if start_date is not None:
        stmt = stmt.where(PortfolioSnapshot.snapshot_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(PortfolioSnapshot.snapshot_date <= end_date)

    stmt = stmt.order_by(
        PortfolioSnapshot.snapshot_date.asc(),
        PortfolioSnapshot.team.asc(),
        PortfolioSnapshot.ticker.asc(),
    )

    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "snapshot_date": r.snapshot_date,
            "team": r.team,
            "ticker": r.ticker,
            "position_side": r.position_side,
            "shares": r.shares,
            "cost_basis_per_share": r.cost_basis_per_share,
            "total_cost_basis": r.total_cost_basis,
            "source_file": r.source_file,
            "selected_sheet": r.selected_sheet,
            "created_at": r.created_at,
        }
        for r in records
    ])


def get_latest_snapshot_date(session: Session) -> Optional[date]:
    """
    Return the latest snapshot date in the database.
    """
    stmt = (
        select(PortfolioSnapshot.snapshot_date)
        .order_by(PortfolioSnapshot.snapshot_date.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


def get_latest_snapshot_before_or_on(
    session: Session,
    as_of_date: date,
) -> pd.DataFrame:
    """
    Load the most recent snapshot on or before the supplied date.
    """
    stmt = (
        select(PortfolioSnapshot.snapshot_date)
        .where(PortfolioSnapshot.snapshot_date <= as_of_date)
        .order_by(PortfolioSnapshot.snapshot_date.desc())
        .limit(1)
    )
    latest_date = session.execute(stmt).scalar_one_or_none()
    if latest_date is None:
        return pd.DataFrame()
    return load_portfolio_snapshot(session, latest_date)


# ============================================================================
# Trade receipts
# ============================================================================
def save_trade_receipts(
    session: Session,
    trades_df: pd.DataFrame,
    source_file: str | None = None,
    selected_sheet: str | None = None,
    replace_existing_for_source_file: bool = False,
) -> int:
    """
    Save uploaded trade receipt rows.

    Expected columns:
    - trade_date
    - settlement_date (optional)
    - team
    - ticker
    - trade_side
    - quantity
    - gross_price
    - commission (optional)
    - fees (optional)
    - net_cash_amount (optional)
    - raw_description (optional)
    """
    if trades_df.empty:
        return 0

    if replace_existing_for_source_file and source_file:
        session.execute(
            delete(TradeReceipt).where(TradeReceipt.source_file == source_file)
        )

    rows_inserted = 0

    for _, row in trades_df.iterrows():
        trade_date = _safe_date(row.get("trade_date"))
        team = _safe_str(row.get("team"))
        ticker = _safe_str(row.get("ticker"))
        trade_side = _safe_str(row.get("trade_side"))

        if trade_date is None or not team or not ticker or not trade_side:
            continue

        quantity = _safe_float(row.get("quantity"))
        gross_price = _safe_float(row.get("gross_price"))

        if quantity is None or gross_price is None:
            continue

        record = TradeReceipt(
            trade_date=trade_date,
            settlement_date=_safe_date(row.get("settlement_date")),
            team=team,
            ticker=ticker,
            trade_side=trade_side,
            quantity=quantity,
            gross_price=gross_price,
            commission=_safe_float(row.get("commission")),
            fees=_safe_float(row.get("fees")),
            net_cash_amount=_safe_float(row.get("net_cash_amount")),
            source_file=source_file or _safe_optional_str(row.get("source_file")),
            selected_sheet=selected_sheet or _safe_optional_str(row.get("selected_sheet")),
            raw_description=_safe_optional_str(row.get("raw_description")),
        )
        session.add(record)
        rows_inserted += 1

    session.flush()
    return rows_inserted


def load_trade_receipts(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    team: str | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    """
    Load trade receipts with optional filters.
    """
    stmt = select(TradeReceipt)

    if start_date is not None:
        stmt = stmt.where(TradeReceipt.trade_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(TradeReceipt.trade_date <= end_date)
    if team is not None:
        stmt = stmt.where(TradeReceipt.team == team)
    if ticker is not None:
        stmt = stmt.where(TradeReceipt.ticker == ticker)

    stmt = stmt.order_by(
        TradeReceipt.trade_date.asc(),
        TradeReceipt.team.asc(),
        TradeReceipt.ticker.asc(),
    )

    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "id": r.id,
            "trade_date": r.trade_date,
            "settlement_date": r.settlement_date,
            "team": r.team,
            "ticker": r.ticker,
            "trade_side": r.trade_side,
            "quantity": r.quantity,
            "gross_price": r.gross_price,
            "commission": r.commission,
            "fees": r.fees,
            "net_cash_amount": r.net_cash_amount,
            "source_file": r.source_file,
            "selected_sheet": r.selected_sheet,
            "raw_description": r.raw_description,
            "created_at": r.created_at,
        }
        for r in records
    ])


# ============================================================================
# Cash ledger
# ============================================================================
def save_cash_ledger_entries(
    session: Session,
    cash_df: pd.DataFrame,
) -> int:
    """
    Save cash ledger rows.

    Expected columns:
    - activity_date
    - team
    - amount
    - activity_type
    - reference_type (optional)
    - reference_id (optional)
    - note (optional)
    """
    if cash_df.empty:
        return 0

    rows_inserted = 0

    for _, row in cash_df.iterrows():
        activity_date = _safe_date(row.get("activity_date"))
        team = _safe_str(row.get("team"))
        amount = _safe_float(row.get("amount"))
        activity_type = _safe_str(row.get("activity_type"))

        if activity_date is None or not team or amount is None or not activity_type:
            continue

        record = CashLedger(
            activity_date=activity_date,
            team=team,
            amount=amount,
            activity_type=activity_type,
            reference_type=_safe_optional_str(row.get("reference_type")),
            reference_id=_safe_int(row.get("reference_id")),
            note=_safe_optional_str(row.get("note")),
        )
        session.add(record)
        rows_inserted += 1

    session.flush()
    return rows_inserted


def delete_cash_ledger_reconciliation_entries_for_date(
    session: Session,
    activity_date: date,
) -> int:
    """
    Delete reconciliation-derived cash ledger rows for a given effective date.
    """
    result = session.execute(
        delete(CashLedger).where(
            CashLedger.activity_date == activity_date,
            CashLedger.activity_type == "RECONCILIATION",
        )
    )
    session.flush()
    return int(result.rowcount or 0)


def load_cash_ledger(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Load cash ledger entries with optional filters.
    """
    stmt = select(CashLedger)

    if start_date is not None:
        stmt = stmt.where(CashLedger.activity_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(CashLedger.activity_date <= end_date)
    if team is not None:
        stmt = stmt.where(CashLedger.team == team)

    stmt = stmt.order_by(CashLedger.activity_date.asc(), CashLedger.id.asc())

    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "id": r.id,
            "activity_date": r.activity_date,
            "team": r.team,
            "amount": r.amount,
            "activity_type": r.activity_type,
            "reference_type": r.reference_type,
            "reference_id": r.reference_id,
            "note": r.note,
            "created_at": r.created_at,
        }
        for r in records
    ])


# ============================================================================
# Reconciliation events
# ============================================================================
def save_reconciliation_events(
    session: Session,
    reconciliation_df: pd.DataFrame,
) -> int:
    """
    Save reconciliation event rows.

    Expected columns:
    - snapshot_date
    - effective_date
    - team
    - ticker
    - position_side
    - expected_shares
    - snapshot_shares
    - delta_shares
    - assumed_price
    - estimated_cash_impact
    - note (optional)
    """
    if reconciliation_df.empty:
        return 0

    rows_inserted = 0

    for _, row in reconciliation_df.iterrows():
        snapshot_date = _safe_date(row.get("snapshot_date"))
        effective_date = _safe_date(row.get("effective_date"))
        team = _safe_str(row.get("team"))
        ticker = _safe_str(row.get("ticker"))
        position_side = _safe_str(row.get("position_side"))

        if (
            snapshot_date is None
            or effective_date is None
            or not team
            or not ticker
            or not position_side
        ):
            continue

        record = ReconciliationEvent(
            snapshot_date=snapshot_date,
            effective_date=effective_date,
            team=team,
            ticker=ticker,
            position_side=position_side,
            expected_shares=_safe_float(row.get("expected_shares")),
            snapshot_shares=_safe_float(row.get("snapshot_shares")),
            delta_shares=_safe_float(row.get("delta_shares")),
            assumed_price=_safe_float(row.get("assumed_price")),
            estimated_cash_impact=_safe_float(row.get("estimated_cash_impact")),
            note=_safe_optional_str(row.get("note")),
        )
        session.add(record)
        rows_inserted += 1

    session.flush()
    return rows_inserted


def delete_reconciliation_events_for_snapshot(
    session: Session,
    snapshot_date: date,
) -> int:
    """
    Delete reconciliation rows for a given snapshot date so recalculations can
    replace prior results cleanly.
    """
    result = session.execute(
        delete(ReconciliationEvent).where(
            ReconciliationEvent.snapshot_date == snapshot_date
        )
    )
    session.flush()
    return int(result.rowcount or 0)


def load_reconciliation_events(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Load reconciliation events with optional filters.
    """
    stmt = select(ReconciliationEvent)

    if start_date is not None:
        stmt = stmt.where(ReconciliationEvent.effective_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(ReconciliationEvent.effective_date <= end_date)
    if team is not None:
        stmt = stmt.where(ReconciliationEvent.team == team)

    stmt = stmt.order_by(
        ReconciliationEvent.effective_date.asc(),
        ReconciliationEvent.team.asc(),
        ReconciliationEvent.ticker.asc(),
    )

    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "id": r.id,
            "snapshot_date": r.snapshot_date,
            "effective_date": r.effective_date,
            "team": r.team,
            "ticker": r.ticker,
            "position_side": r.position_side,
            "expected_shares": r.expected_shares,
            "snapshot_shares": r.snapshot_shares,
            "delta_shares": r.delta_shares,
            "assumed_price": r.assumed_price,
            "estimated_cash_impact": r.estimated_cash_impact,
            "note": r.note,
            "created_at": r.created_at,
        }
        for r in records
    ])


# ============================================================================
# Position state
# ============================================================================
def replace_position_state_for_date(
    session: Session,
    as_of_date: date,
    position_state_df: pd.DataFrame,
) -> int:
    """
    Replace all reconstructed position state rows for a given date.

    Expected columns:
    - team
    - ticker
    - position_side
    - shares
    - cost_basis_per_share (optional)
    - total_cost_basis (optional)
    - is_reconciled (optional)
    """
    session.execute(
        delete(PositionState).where(PositionState.as_of_date == as_of_date)
    )

    if position_state_df.empty:
        session.flush()
        return 0

    rows_inserted = 0

    for _, row in position_state_df.iterrows():
        team = _safe_str(row.get("team"))
        ticker = _safe_str(row.get("ticker"))
        position_side = _safe_str(row.get("position_side"))
        shares = _safe_float(row.get("shares"))

        if not team or not ticker or not position_side or shares is None:
            continue

        record = PositionState(
            as_of_date=as_of_date,
            team=team,
            ticker=ticker,
            position_side=position_side,
            shares=shares,
            cost_basis_per_share=_safe_float(row.get("cost_basis_per_share")),
            total_cost_basis=_safe_float(row.get("total_cost_basis")),
            is_reconciled=bool(row.get("is_reconciled", False)),
        )
        session.add(record)
        rows_inserted += 1

    session.flush()
    return rows_inserted


def load_position_state(
    session: Session,
    as_of_date: date | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Load reconstructed position state.

    If as_of_date is None, loads the latest available position state date.
    """
    if as_of_date is None:
        as_of_date = get_latest_position_state_date(session)

    if as_of_date is None:
        return pd.DataFrame(
            columns=[
                "as_of_date",
                "team",
                "ticker",
                "position_side",
                "shares",
                "cost_basis_per_share",
                "total_cost_basis",
                "is_reconciled",
                "created_at",
            ]
        )

    stmt = select(PositionState).where(PositionState.as_of_date == as_of_date)

    if team is not None:
        stmt = stmt.where(PositionState.team == team)

    stmt = stmt.order_by(
        PositionState.team.asc(),
        PositionState.ticker.asc(),
        PositionState.position_side.asc(),
    )

    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "as_of_date": r.as_of_date,
            "team": r.team,
            "ticker": r.ticker,
            "position_side": r.position_side,
            "shares": r.shares,
            "cost_basis_per_share": r.cost_basis_per_share,
            "total_cost_basis": r.total_cost_basis,
            "is_reconciled": r.is_reconciled,
            "created_at": r.created_at,
        }
        for r in records
    ])


def get_latest_position_state_date(session: Session) -> Optional[date]:
    """
    Return the latest as_of_date present in position_state.
    """
    stmt = (
        select(PositionState.as_of_date)
        .order_by(PositionState.as_of_date.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


# ============================================================================
# Price history
# ============================================================================
def save_price_history(
    session: Session,
    price_history_df: pd.DataFrame,
    replace_existing: bool = False,
) -> int:
    """
    Save historical price rows.

    Expected columns:
    - price_date or date
    - ticker
    - close_price or close
    - adj_close_price or adj_close
    """
    if price_history_df.empty:
        return 0

    rows_inserted = 0

    for _, row in price_history_df.iterrows():
        price_date = _safe_date(row.get("price_date", row.get("date")))
        ticker = _safe_str(row.get("ticker"))

        if price_date is None or not ticker:
            continue

        if replace_existing:
            session.execute(
                delete(PriceHistory).where(
                    PriceHistory.price_date == price_date,
                    PriceHistory.ticker == ticker,
                )
            )

        record = PriceHistory(
            price_date=price_date,
            ticker=ticker,
            close_price=_safe_float(row.get("close_price", row.get("close"))),
            adj_close_price=_safe_float(row.get("adj_close_price", row.get("adj_close"))),
        )
        session.add(record)
        rows_inserted += 1

    session.flush()
    return rows_inserted


def load_price_history(
    session: Session,
    tickers: Iterable[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """
    Load historical prices with optional filters.
    """
    stmt = select(PriceHistory)

    if tickers is not None:
        ticker_list = [t for t in tickers if isinstance(t, str) and t.strip()]
        if ticker_list:
            stmt = stmt.where(PriceHistory.ticker.in_(ticker_list))

    if start_date is not None:
        stmt = stmt.where(PriceHistory.price_date >= start_date)
    if end_date is not None:
        stmt = stmt.where(PriceHistory.price_date <= end_date)

    stmt = stmt.order_by(PriceHistory.price_date.asc(), PriceHistory.ticker.asc())

    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "price_date": r.price_date,
            "ticker": r.ticker,
            "close_price": r.close_price,
            "adj_close_price": r.adj_close_price,
            "created_at": r.created_at,
        }
        for r in records
    ])


# ============================================================================
# Upload logs
# ============================================================================
def log_upload_event(
    session: Session,
    upload_type: str,
    source_file: str,
    selected_sheet: str | None = None,
    row_count: int | None = None,
    status: str = "success",
    message: str | None = None,
) -> int:
    """
    Insert an upload log row and return the new log ID.
    """
    record = UploadLog(
        upload_type=upload_type,
        source_file=source_file,
        selected_sheet=selected_sheet,
        row_count=row_count,
        status=status,
        message=message,
    )
    session.add(record)
    session.flush()
    return int(record.id)


def load_upload_logs(
    session: Session,
    limit: int = 100,
) -> pd.DataFrame:
    """
    Load recent upload logs.
    """
    stmt = (
        select(UploadLog)
        .order_by(UploadLog.upload_timestamp.desc())
        .limit(limit)
    )
    records = session.execute(stmt).scalars().all()

    return pd.DataFrame([
        {
            "id": r.id,
            "upload_timestamp": r.upload_timestamp,
            "upload_type": r.upload_type,
            "source_file": r.source_file,
            "selected_sheet": r.selected_sheet,
            "row_count": r.row_count,
            "status": r.status,
            "message": r.message,
        }
        for r in records
    ])


# ============================================================================
# Portfolio activity feed
# ============================================================================
def load_portfolio_activity(
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
    team: str | None = None,
) -> pd.DataFrame:
    """
    Build a unified activity feed from:
    - trade receipts
    - reconciliation events
    - cash ledger
    - snapshot uploads (as checkpoint events)

    Returns a DataFrame ordered by activity_date descending.
    """
    activities: list[dict] = []

    # Trades
    trades_df = load_trade_receipts(
        session=session,
        start_date=start_date,
        end_date=end_date,
        team=team,
    )
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            activities.append({
                "activity_date": row["trade_date"],
                "activity_type": "TRADE",
                "team": row["team"],
                "ticker": row["ticker"],
                "position_side": None,
                "quantity": row["quantity"],
                "price": row["gross_price"],
                "cash_impact": row["net_cash_amount"],
                "reference_id": row["id"],
                "note": row["trade_side"],
                "source_file": row["source_file"],
            })

    # Reconciliations
    recon_df = load_reconciliation_events(
        session=session,
        start_date=start_date,
        end_date=end_date,
        team=team,
    )
    if not recon_df.empty:
        for _, row in recon_df.iterrows():
            activities.append({
                "activity_date": row["effective_date"],
                "activity_type": "RECONCILIATION",
                "team": row["team"],
                "ticker": row["ticker"],
                "position_side": row["position_side"],
                "quantity": row["delta_shares"],
                "price": row["assumed_price"],
                "cash_impact": row["estimated_cash_impact"],
                "reference_id": row["id"],
                "note": row["note"],
                "source_file": None,
            })

    # Cash ledger
    cash_df = load_cash_ledger(
        session=session,
        start_date=start_date,
        end_date=end_date,
        team=team,
    )
    if not cash_df.empty:
        for _, row in cash_df.iterrows():
            activities.append({
                "activity_date": row["activity_date"],
                "activity_type": f"CASH_{row['activity_type']}",
                "team": row["team"],
                "ticker": None,
                "position_side": None,
                "quantity": None,
                "price": None,
                "cash_impact": row["amount"],
                "reference_id": row["id"],
                "note": row["note"],
                "source_file": None,
            })

    # Snapshot checkpoints
    snapshots_df = load_all_portfolio_snapshots(
        session=session,
        start_date=start_date,
        end_date=end_date,
    )
    if not snapshots_df.empty:
        checkpoint_df = (
            snapshots_df.groupby("snapshot_date", as_index=False)
            .agg(
                row_count=("ticker", "count"),
                source_file=("source_file", "first"),
            )
        )
        for _, row in checkpoint_df.iterrows():
            activities.append({
                "activity_date": row["snapshot_date"],
                "activity_type": "SNAPSHOT_CHECKPOINT",
                "team": None,
                "ticker": None,
                "position_side": None,
                "quantity": row["row_count"],
                "price": None,
                "cash_impact": None,
                "reference_id": None,
                "note": f"Uploaded snapshot with {int(row['row_count'])} rows",
                "source_file": row["source_file"],
            })

    if not activities:
        return pd.DataFrame(
            columns=[
                "activity_date",
                "activity_type",
                "team",
                "ticker",
                "position_side",
                "quantity",
                "price",
                "cash_impact",
                "reference_id",
                "note",
                "source_file",
            ]
        )

    activity_df = pd.DataFrame(activities)
    activity_df["activity_date"] = pd.to_datetime(
        activity_df["activity_date"],
        errors="coerce",
    )
    activity_df = activity_df.sort_values(
        ["activity_date", "activity_type"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return activity_df


# ============================================================================
# Internal coercion helpers
# ============================================================================
def _safe_str(value) -> str:
    """
    Coerce value to stripped string. Returns empty string for null-ish values.
    """
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _safe_optional_str(value) -> str | None:
    """
    Coerce value to stripped string, returning None for null-ish/blank values.
    """
    if value is None or pd.isna(value):
        return None
    result = str(value).strip()
    return result if result else None


def _safe_float(value) -> float | None:
    """
    Coerce value to float, returning None if conversion fails.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    """
    Coerce value to int, returning None if conversion fails.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_date(value) -> date | None:
    """
    Coerce a value to date, returning None if conversion fails.
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    if isinstance(value, datetime):
        return value.date()

    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()
    except Exception:
        return None
