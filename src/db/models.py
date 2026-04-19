"""
Core database models for the CMCSIF portfolio tracker.

This schema is designed for:
- authoritative portfolio snapshots
- trade receipt ingestion
- cash tracking
- reconciliation tracking
- historical portfolio state reconstruction
- portfolio activity feed generation
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.db.session import Base


class PortfolioSnapshot(Base):
    """
    One row per position in an uploaded portfolio snapshot.

    Snapshots are authoritative checkpoints from the endowment office.
    """

    __tablename__ = "portfolio_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_date",
            "team",
            "ticker",
            "position_side",
            name="uq_portfolio_snapshot_position",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    team: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Expected values like "LONG", "SHORT", "CASH"
    position_side: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    shares: Mapped[float] = mapped_column(Float, nullable=False)
    cost_basis_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_basis: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selected_sheet: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class TradeReceipt(Base):
    """
    One row per uploaded trade receipt line.

    These mutate the portfolio state between authoritative snapshots.
    """

    __tablename__ = "trade_receipts"
    __table_args__ = (
        UniqueConstraint(
            "trade_date",
            "settlement_date",
            "team",
            "ticker",
            "trade_side",
            "quantity",
            "gross_price",
            "source_file",
            name="uq_trade_receipt_line",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    trade_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    settlement_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)

    team: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Expected values like "BUY", "SELL", "SHORT_SELL", "COVER"
    trade_side: Mapped[str] = mapped_column(String(25), nullable=False, index=True)

    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    gross_price: Mapped[float] = mapped_column(Float, nullable=False)

    commission: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Signed total cash effect from the receipt, if available
    net_cash_amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    selected_sheet: Mapped[str | None] = mapped_column(String(255), nullable=True)
    raw_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class CashLedger(Base):
    """
    Records cash movements by team over time.

    Cash changes can come from:
    - trade receipts
    - reconciliations
    - withdrawals for expenses
    - allocator rebalances
    - manual adjustments
    """

    __tablename__ = "cash_ledger"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    activity_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    team: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Positive = cash inflow, negative = cash outflow
    amount: Mapped[float] = mapped_column(Float, nullable=False)

    # Examples: TRADE, RECONCILIATION, EXPENSE_WITHDRAWAL, REBALANCE, MANUAL
    activity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    reference_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    reference_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class ReconciliationEvent(Base):
    """
    Logged when a new authoritative snapshot does not match the system's
    reconstructed expected holdings.

    The system can assume the position change occurred at the end of the
    previous trading day and offset any residual notional through cash.
    """

    __tablename__ = "reconciliation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    effective_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    team: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    position_side: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    expected_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta_shares: Mapped[float | None] = mapped_column(Float, nullable=True)

    assumed_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    estimated_cash_impact: Mapped[float | None] = mapped_column(Float, nullable=True)

    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class PositionState(Base):
    """
    Daily reconstructed position state.

    This can be written as a cached derived table after applying trades and
    reconciliations to the latest authoritative snapshot history.
    """

    __tablename__ = "position_state"
    __table_args__ = (
        UniqueConstraint(
            "as_of_date",
            "team",
            "ticker",
            "position_side",
            name="uq_position_state_daily",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    as_of_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    team: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    position_side: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    shares: Mapped[float] = mapped_column(Float, nullable=False)

    cost_basis_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_basis: Mapped[float | None] = mapped_column(Float, nullable=True)

    is_reconciled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class PriceHistory(Base):
    """
    Historical market prices by ticker and date.
    """

    __tablename__ = "price_history"
    __table_args__ = (
        UniqueConstraint("price_date", "ticker", name="uq_price_history"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    price_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    close_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    adj_close_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class UploadLog(Base):
    """
    Audit log for all uploads and processing events.
    """

    __tablename__ = "upload_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )

    upload_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    source_file: Mapped[str] = mapped_column(String(255), nullable=False)

    selected_sheet: Mapped[str | None] = mapped_column(String(255), nullable=True)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(String(50), nullable=False, default="success")
    message: Mapped[str | None] = mapped_column(Text, nullable=True)