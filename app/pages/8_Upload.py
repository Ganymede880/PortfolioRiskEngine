"""
Upload page for the CMCSIF Portfolio Tracker (v3).

Supports two upload types:
- Portfolio Snapshot
- Trade Receipt

Snapshot logic:
- first snapshot seeds position_state
- later snapshots compare against latest expected position_state
- differences generate reconciliation events
- reconciliation cash offsets are written to cash_ledger
- latest authoritative snapshot replaces position_state

Trade logic:
- persists trade receipts
- does not yet fully rebuild position_state automatically
"""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st
from sqlalchemy import delete
from sqlalchemy import tuple_

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_uploaded_file_auto, preview_uploaded_file
from src.data.normalizers import (
    normalize_snapshot_and_tag_source,
    normalize_trade_receipt_and_tag_source,
)
from src.data.validators import validate_uploaded_dataframe
from src.db.crud import (
    get_latest_position_state_date,
    load_cash_ledger,
    load_position_state,
    load_trade_receipts,
    log_upload_event,
    replace_position_state_for_date,
    save_cash_ledger_entries,
    save_portfolio_snapshot,
    save_reconciliation_events,
    save_trade_receipts,
)
from src.db.models import CashLedger, ReconciliationEvent
from src.db.session import session_scope
from src.analytics.ledger import (
    ReconciliationConfig,
    apply_cash_ledger_entries_to_positions,
    apply_trades_to_positions,
    derive_trade_cash_amount,
    reconcile_expected_positions_to_authoritative_snapshot,
)
from src.db import crud
from src.utils.ui import apply_app_theme, left_align_dataframe


def _apply_upload_control_theme() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background: #B91C1C;
            border: 1px solid #991B1B;
            color: #FFFFFF;
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #991B1B;
            border-color: #7F1D1D;
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _save_uploaded_file_temporarily(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def _format_df_preview(df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(n).copy()


def _render_validation_block(validation_result: dict) -> None:
    if validation_result["is_valid"]:
        st.success("Validation passed.")
    else:
        st.error("Validation failed. Please fix the issues below.")

    errors = validation_result.get("errors", [])
    warnings = validation_result.get("warnings", [])

    if errors:
        st.markdown("**Errors**")
        for e in errors:
            st.write(f"- {e}")

    if warnings:
        st.markdown("**Warnings**")
        for w in warnings:
            st.write(f"- {w}")

    resolved = validation_result.get("resolved_columns", {})
    if resolved:
        with st.expander("Resolved Column Mapping"):
            st.json(resolved)


def _render_mapping_notes(normalization_result: dict) -> None:
    unmapped_teams = normalization_result.get("unmapped_teams", [])
    suspicious_tickers = normalization_result.get("suspicious_tickers", [])

    if unmapped_teams:
        st.warning(
            "Unmapped pod/sector labels detected: "
            + ", ".join(sorted(set(unmapped_teams)))
        )

    if suspicious_tickers:
        st.warning(
            "Suspicious/unmapped tickers detected: "
            + ", ".join(sorted(set(suspicious_tickers)))
            + ". Consider adding to ticker_map.csv."
        )


def _choose_snapshot_date(snapshot_df: pd.DataFrame) -> pd.Timestamp | None:
    if "snapshot_date" in snapshot_df.columns:
        series = pd.Series(
            pd.to_datetime(snapshot_df["snapshot_date"], errors="coerce")
        ).dropna()

        if not series.empty:
            detected = series.iloc[0]
            st.info(f"Detected snapshot date: {detected.date()}")
            return pd.Timestamp(detected)

    manual = st.date_input("Select snapshot date")
    return pd.Timestamp(manual)


def _choose_reconciliation_effective_date(snapshot_ts: pd.Timestamp) -> pd.Timestamp:
    """
    MVP: use previous calendar day.
    Later this should become previous trading day.
    """
    effective_date = snapshot_ts - pd.Timedelta(days=1)
    st.info(f"Reconciliation effective date: {effective_date.date()}")
    return effective_date


def _build_assumed_price_map_from_snapshot(snapshot_df: pd.DataFrame) -> dict:
    """
    Build an assumed price map from snapshot cost basis.

    Later this should use actual market close prices.
    """
    price_map = {}

    if snapshot_df.empty or "cost_basis_per_share" not in snapshot_df.columns:
        return price_map

    for _, row in snapshot_df.iterrows():
        team = str(row.get("team", "")).strip()
        ticker = str(row.get("ticker", "")).strip()
        position_side = str(row.get("position_side", "")).strip()
        cost_basis = pd.to_numeric(
            pd.Series([row.get("cost_basis_per_share")]),
            errors="coerce",
        ).iloc[0]

        if team and ticker and position_side and pd.notna(cost_basis):
            price_map[(team, ticker, position_side)] = float(cost_basis)

    return price_map


def _build_expected_positions_for_snapshot(
    session,
    snapshot_ts: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_state_date = get_latest_position_state_date(session)
    if latest_state_date is None:
        return pd.DataFrame(), pd.DataFrame()

    expected_positions_df = load_position_state(session, as_of_date=latest_state_date)
    between_trades_df = load_trade_receipts(
        session=session,
        start_date=latest_state_date + pd.Timedelta(days=1),
        end_date=snapshot_ts.date(),
    )
    between_cash_df = load_cash_ledger(
        session=session,
        start_date=latest_state_date + pd.Timedelta(days=1),
        end_date=snapshot_ts.date(),
    )

    carried_positions_df = expected_positions_df.copy()
    if not between_trades_df.empty:
        carried_positions_df, _ = apply_trades_to_positions(
            base_positions_df=carried_positions_df,
            trades_df=between_trades_df,
        )

    non_reconciliation_cash_df = between_cash_df.loc[
        between_cash_df["activity_type"].astype(str).ne("RECONCILIATION")
    ].copy() if not between_cash_df.empty else pd.DataFrame()
    if not non_reconciliation_cash_df.empty:
        carried_positions_df = apply_cash_ledger_entries_to_positions(
            positions_df=carried_positions_df,
            cash_entries_df=non_reconciliation_cash_df,
        )

    return carried_positions_df, between_trades_df


def _build_external_flow_entries_for_trade_upload(
    trades_df: pd.DataFrame,
    upload_type: str,
) -> pd.DataFrame:
    """
    Build external cash-flow ledger rows for rebalance/liquidation uploads.
    """
    if trades_df.empty or upload_type not in {"sector_rebalance", "portfolio_liquidation"}:
        return pd.DataFrame(
            columns=["activity_date", "team", "amount", "activity_type", "reference_type", "reference_id", "note"]
        )

    flow_type = "SECTOR_REBALANCE" if upload_type == "sector_rebalance" else "PORTFOLIO_LIQUIDATION"
    rows = []
    for _, row in trades_df.iterrows():
        activity_date = pd.to_datetime(
            row.get("settlement_date", row.get("trade_date")),
            errors="coerce",
        )
        if pd.isna(activity_date):
            continue

        trade_cash_amount = derive_trade_cash_amount(row)
        external_flow_amount = -float(trade_cash_amount)
        if abs(external_flow_amount) < 1e-9:
            continue

        trade_side = str(row.get("trade_side", "")).strip().upper()
        ticker = str(row.get("ticker", "")).strip().upper()
        quantity = float(pd.to_numeric(pd.Series([row.get("quantity")]), errors="coerce").fillna(0.0).iloc[0])
        gross_price = float(pd.to_numeric(pd.Series([row.get("gross_price")]), errors="coerce").fillna(0.0).iloc[0])
        rows.append(
            {
                "activity_date": activity_date.date(),
                "team": str(row.get("team", "")).strip(),
                "amount": external_flow_amount,
                "activity_type": flow_type,
                "reference_type": flow_type,
                "reference_id": None,
                "note": f"{flow_type.replace('_', ' ').title()} offset for {trade_side} {quantity:,.2f} {ticker} @ {gross_price:,.4f}",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["activity_date", "team", "amount", "activity_type", "reference_type", "reference_id", "note"]
        )

    flows_df = pd.DataFrame(rows)
    return (
        flows_df.groupby(["activity_date", "team", "activity_type", "reference_type", "note"], dropna=False, as_index=False)
        .agg(amount=("amount", "sum"))
        [["activity_date", "team", "amount", "activity_type", "reference_type", "note"]]
    )


def _delete_reconciliation_artifacts_for_snapshot(
    session,
    snapshot_date: pd.Timestamp | pd.Timestamp.date,
    effective_date: pd.Timestamp | pd.Timestamp.date,
) -> None:
    session.execute(
        delete(ReconciliationEvent).where(
            ReconciliationEvent.snapshot_date == pd.to_datetime(snapshot_date).date()
        )
    )
    session.execute(
        delete(CashLedger).where(
            CashLedger.activity_date == pd.to_datetime(effective_date).date(),
            CashLedger.activity_type == "RECONCILIATION",
        )
    )
    session.flush()


def _delete_external_flow_entries_for_upload(
    session,
    external_flow_df: pd.DataFrame,
) -> None:
    if external_flow_df.empty:
        return

    working = external_flow_df.copy()
    working["activity_date"] = pd.to_datetime(working["activity_date"], errors="coerce").dt.date
    working["team"] = working["team"].astype(str).str.strip()
    working["activity_type"] = working["activity_type"].astype(str).str.strip().str.upper()
    working = working.dropna(subset=["activity_date", "team", "activity_type"]).copy()
    if working.empty:
        return

    keys = list(
        {
            (row["activity_date"], row["team"], row["activity_type"])
            for _, row in working.iterrows()
        }
    )
    session.execute(
        delete(CashLedger).where(
            tuple_(CashLedger.activity_date, CashLedger.team, CashLedger.activity_type).in_(keys)
        )
    )
    session.flush()


def _render_reconciliation_preview(
    reconciliation_df: pd.DataFrame,
    cash_df: pd.DataFrame,
) -> None:
    st.subheader("Reconciliation Preview")

    if reconciliation_df.empty:
        st.success(
            "No reconciliation differences detected. Snapshot matches expected portfolio state."
        )
        return

    st.warning(
        f"Detected {len(reconciliation_df)} reconciliation row(s). "
        "These differences will be logged and the authoritative snapshot will replace expected state."
    )

    st.markdown("**Reconciliation Events**")
    st.dataframe(
        left_align_dataframe(_format_df_preview(reconciliation_df)),
        use_container_width=True,
        hide_index=True,
    )

    if not cash_df.empty:
        st.markdown("**Derived Cash Offsets**")
        st.dataframe(
            left_align_dataframe(_format_df_preview(cash_df)),
            use_container_width=True,
            hide_index=True,
        )


def _render_clear_uploads_action() -> None:
    with st.container(border=True):
        action_col, confirm_col = st.columns([1, 1.4])
        with action_col:
            if st.button(
                "Clear All Uploads",
                type="primary",
                use_container_width=True,
            ):
                if not st.session_state.get("confirm_clear_all_uploads", False):
                    st.warning("Check the confirmation box before clearing all uploaded portfolio data.")
                else:
                    with session_scope() as session:
                        counts = crud.clear_all_uploaded_portfolio_data(session)
                    st.session_state["confirm_clear_all_uploads"] = False
                    st.session_state["last_clear_all_uploads_counts"] = counts
                    st.rerun()
        with confirm_col:
            st.checkbox(
                "I understand this will delete all uploaded files.",
                key="confirm_clear_all_uploads",
            )

    if "last_clear_all_uploads_counts" in st.session_state:
        counts = st.session_state.pop("last_clear_all_uploads_counts")
        st.success(
            "Cleared uploaded portfolio data: "
            f"{counts.get('portfolio_snapshots', 0)} snapshot rows, "
            f"{counts.get('trade_receipts', 0)} trade rows, "
            f"{counts.get('cash_ledger', 0)} cash ledger rows, "
            f"{counts.get('reconciliation_events', 0)} reconciliation rows, "
            f"{counts.get('position_state', 0)} position-state rows, and "
            f"{counts.get('upload_logs', 0)} upload log rows."
        )


def _save_snapshot_with_reconciliation(
    session,
    snapshot_df: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    uploaded_file_name: str,
    selected_sheet: str | None,
) -> dict:
    snapshot_date = snapshot_ts.date()

    rows = save_portfolio_snapshot(
        session=session,
        snapshot_df=snapshot_df,
        snapshot_date=snapshot_date,
        source_file=uploaded_file_name,
        selected_sheet=selected_sheet,
        replace_existing=True,
    )

    latest_state_date = get_latest_position_state_date(session)

    # First snapshot: just seed
    if latest_state_date is None:
        position_state_seed_df = snapshot_df.copy()
        position_state_seed_df["as_of_date"] = snapshot_date
        position_state_seed_df["is_reconciled"] = False

        seeded_rows = replace_position_state_for_date(
            session=session,
            as_of_date=snapshot_date,
            position_state_df=position_state_seed_df,
        )

        log_upload_event(
            session=session,
            upload_type="snapshot",
            source_file=uploaded_file_name,
            selected_sheet=selected_sheet,
            row_count=rows,
            status="success",
            message=(
                f"Saved initial snapshot for {snapshot_date} with {rows} rows "
                f"and seeded {seeded_rows} position-state rows."
            ),
        )

        return {
            "rows_saved": rows,
            "position_state_rows": seeded_rows,
            "reconciliation_rows": 0,
            "cash_rows": 0,
            "mode": "initial_seed",
        }

    # Later snapshots: reconcile against expected state
    expected_positions_df, between_trades_df = _build_expected_positions_for_snapshot(
        session=session,
        snapshot_ts=snapshot_ts,
    )
    effective_ts = _choose_reconciliation_effective_date(snapshot_ts)
    assumed_price_map = _build_assumed_price_map_from_snapshot(snapshot_df)

    _, reconciliation_df, cash_df = reconcile_expected_positions_to_authoritative_snapshot(
        expected_positions_df=expected_positions_df,
        authoritative_snapshot_df=snapshot_df,
        snapshot_date=snapshot_ts,
        effective_date=effective_ts,
        assumed_price_map=assumed_price_map,
        config=ReconciliationConfig(default_assumed_price=None),
    )

    _delete_reconciliation_artifacts_for_snapshot(
        session,
        snapshot_date=snapshot_date,
        effective_date=effective_ts.date(),
    )

    recon_rows = (
        save_reconciliation_events(session, reconciliation_df)
        if not reconciliation_df.empty else 0
    )
    cash_rows = (
        save_cash_ledger_entries(session, cash_df)
        if not cash_df.empty else 0
    )

    # Authoritative snapshot becomes latest position state
    position_state_seed_df = snapshot_df.copy()
    position_state_seed_df["as_of_date"] = snapshot_date
    position_state_seed_df["is_reconciled"] = False

    if not reconciliation_df.empty:
        reconciliation_keys = set(
            zip(
                reconciliation_df["team"],
                reconciliation_df["ticker"],
                reconciliation_df["position_side"],
            )
        )

        def _is_reconciled_row(row) -> bool:
            return (
                row["team"],
                row["ticker"],
                row["position_side"],
            ) in reconciliation_keys

        position_state_seed_df["is_reconciled"] = position_state_seed_df.apply(
            _is_reconciled_row,
            axis=1,
        )

    seeded_rows = replace_position_state_for_date(
        session=session,
        as_of_date=snapshot_date,
        position_state_df=position_state_seed_df,
    )

    log_upload_event(
        session=session,
        upload_type="snapshot",
        source_file=uploaded_file_name,
        selected_sheet=selected_sheet,
        row_count=rows,
        status="success",
        message=(
                f"Saved snapshot for {snapshot_date} with {rows} rows. "
                f"Applied {len(between_trades_df)} intervening trade row(s), "
                f"generated {recon_rows} reconciliation rows, {cash_rows} cash rows, "
                f"and seeded {seeded_rows} position-state rows."
            ),
        )

    return {
        "rows_saved": rows,
        "position_state_rows": seeded_rows,
        "reconciliation_rows": recon_rows,
        "cash_rows": cash_rows,
        "trade_rows_applied": len(between_trades_df),
        "mode": "reconciled",
        "reconciliation_df": reconciliation_df,
        "cash_df": cash_df,
    }


def main() -> None:
    st.set_page_config(page_title="Upload", layout="wide")
    apply_app_theme()
    _apply_upload_control_theme()
    st.title("Upload")

    st.write(
        """
        Upload a **Portfolio Snapshot**, **Trade Receipt**, **Sector Rebalance**, or **Portfolio Liquidation** file.

        The system will:
        - auto-detect file type (or you can override)
        - validate structure
        - normalize to canonical schema
        - persist to database
        - reconcile snapshots against expected portfolio state when applicable
        """
    )

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )

    temp_path: Path | None = None

    if uploaded_file is None:
        st.info("Upload a file to begin.")
    else:
        try:
            temp_path = _save_uploaded_file_temporarily(uploaded_file)

            preview = preview_uploaded_file(temp_path)
            detected_type = preview.get("upload_type", "unknown")

            st.subheader("Detected File Structure")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("File Type", preview.get("file_type", ""))
            c2.metric("Sheet", preview.get("selected_sheet") or "CSV")
            c3.metric("Rows", int(preview.get("row_count", 0)))
            c4.metric("Detected Type", detected_type or "unknown")

            preview_df = pd.DataFrame(preview.get("preview_rows", []))
            if not preview_df.empty:
                st.dataframe(left_align_dataframe(preview_df), use_container_width=True, hide_index=True)

            st.subheader("Upload Type")
            trade_like_options = ["snapshot", "trade_receipt", "sector_rebalance", "portfolio_liquidation"]
            default_index = 0
            detected_upload_type = detected_type or "unknown"
            if detected_upload_type in trade_like_options:
                default_index = trade_like_options.index(detected_upload_type)
            upload_type = st.radio(
                "Select upload type",
                options=trade_like_options,
                index=default_index,
                horizontal=True,
            )

            raw_df, metadata = load_uploaded_file_auto(temp_path)

            st.subheader("Validation")
            validation = validate_uploaded_dataframe(raw_df, upload_type)
            _render_validation_block(validation)

            if not validation["is_valid"]:
                with session_scope() as session:
                    log_upload_event(
                        session=session,
                        upload_type=upload_type,
                        source_file=uploaded_file.name,
                        selected_sheet=metadata.get("selected_sheet"),
                        row_count=len(raw_df),
                        status="validation_failed",
                        message="; ".join(validation.get("errors", [])),
                    )
            else:
                st.subheader("Normalized Preview")

                if upload_type == "snapshot":
                    norm = normalize_snapshot_and_tag_source(
                        raw_df,
                        source_file=uploaded_file.name,
                        selected_sheet=metadata.get("selected_sheet", ""),
                    )
                    snapshot_df = norm["snapshot"]
                    _render_mapping_notes(norm)
                    st.dataframe(
                        left_align_dataframe(_format_df_preview(snapshot_df)),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("Snapshot Date")
                    snapshot_ts = _choose_snapshot_date(snapshot_df)
                    if snapshot_ts is None:
                        st.warning("Please provide a snapshot date.")
                    else:
                        with session_scope() as session:
                            latest_state_date = get_latest_position_state_date(session)

                            if latest_state_date is not None:
                                expected_positions_df, between_trades_df = _build_expected_positions_for_snapshot(
                                    session=session,
                                    snapshot_ts=snapshot_ts,
                                )
                                effective_ts = _choose_reconciliation_effective_date(snapshot_ts)
                                assumed_price_map = _build_assumed_price_map_from_snapshot(snapshot_df)

                                _, reconciliation_preview_df, cash_preview_df = reconcile_expected_positions_to_authoritative_snapshot(
                                    expected_positions_df=expected_positions_df,
                                    authoritative_snapshot_df=snapshot_df,
                                    snapshot_date=snapshot_ts,
                                    effective_date=effective_ts,
                                    assumed_price_map=assumed_price_map,
                                    config=ReconciliationConfig(default_assumed_price=None),
                                )

                                if not between_trades_df.empty:
                                    st.info(
                                        f"Carried forward {len(between_trades_df)} trade receipt row(s) into the expected portfolio before reconciliation."
                                    )

                                _render_reconciliation_preview(
                                    reconciliation_preview_df,
                                    cash_preview_df,
                                )
                            else:
                                st.info(
                                    "No prior position state exists. This snapshot will be used as the initial seed."
                                )

                        if st.button("Save Snapshot", type="primary"):
                            with session_scope() as session:
                                result = _save_snapshot_with_reconciliation(
                                    session=session,
                                    snapshot_df=snapshot_df,
                                    snapshot_ts=snapshot_ts,
                                    uploaded_file_name=uploaded_file.name,
                                    selected_sheet=metadata.get("selected_sheet"),
                                )

                            if result["mode"] == "initial_seed":
                                st.success(
                                    f"Saved {result['rows_saved']} snapshot rows for {snapshot_ts.date()}, "
                                    f"and seeded {result['position_state_rows']} position-state rows."
                                )
                            else:
                                st.success(
                                    f"Saved {result['rows_saved']} snapshot rows for {snapshot_ts.date()}, "
                                    f"applied {result['trade_rows_applied']} trade row(s), "
                                    f"generated {result['reconciliation_rows']} reconciliation row(s), "
                                    f"generated {result['cash_rows']} cash ledger row(s), "
                                    f"and seeded {result['position_state_rows']} position-state rows."
                                )

                else:
                    norm = normalize_trade_receipt_and_tag_source(
                        raw_df,
                        source_file=uploaded_file.name,
                        selected_sheet=metadata.get("selected_sheet", ""),
                    )
                    trades_df = norm["trades"]
                    _render_mapping_notes(norm)
                    st.dataframe(
                        left_align_dataframe(_format_df_preview(trades_df)),
                        use_container_width=True,
                        hide_index=True,
                    )

                    external_flow_df = _build_external_flow_entries_for_trade_upload(
                        trades_df=trades_df,
                        upload_type=upload_type,
                    )
                    if not external_flow_df.empty:
                        st.subheader("External Cash Flow Preview")
                        st.dataframe(
                            left_align_dataframe(_format_df_preview(external_flow_df)),
                            use_container_width=True,
                            hide_index=True,
                        )

                    if st.button("Save Trades", type="primary"):
                        with session_scope() as session:
                            rows = save_trade_receipts(
                                session=session,
                                trades_df=trades_df,
                                source_file=uploaded_file.name,
                                selected_sheet=metadata.get("selected_sheet"),
                                replace_existing_for_source_file=True,
                            )
                            flow_rows = 0
                            if not external_flow_df.empty:
                                _delete_external_flow_entries_for_upload(
                                    session=session,
                                    external_flow_df=external_flow_df,
                                )
                                flow_rows = save_cash_ledger_entries(
                                    session=session,
                                    cash_df=external_flow_df,
                                )
                            log_upload_event(
                                session=session,
                                upload_type=upload_type,
                                source_file=uploaded_file.name,
                                selected_sheet=metadata.get("selected_sheet"),
                                row_count=rows,
                                status="success",
                                message=f"Saved {rows} trade rows and {flow_rows} external cash-flow rows.",
                            )

                        if not external_flow_df.empty:
                            st.success(f"Saved {rows} trade rows and {flow_rows} external cash-flow rows.")
                        else:
                            st.success(f"Saved {rows} trade rows.")

        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            st.error(f"Upload failed: {exc}")

            with session_scope() as session:
                log_upload_event(
                    session=session,
                    upload_type="unknown",
                    source_file=uploaded_file.name,
                    selected_sheet=None,
                    row_count=None,
                    status="error",
                    message=str(exc),
                )

        finally:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass

    st.divider()
    _render_clear_uploads_action()


if __name__ == "__main__":
    main()
