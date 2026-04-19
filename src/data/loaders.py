"""
File loading utilities for the CMCSIF portfolio tracker.

This module supports two primary upload types:
- portfolio snapshots
- trade receipts

It is responsible for:
- reading CSV / Excel files
- selecting the correct sheet
- lightly cleaning empty rows / columns
- returning raw DataFrames plus metadata for downstream validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import settings


# ============================================================================
# Generic file helpers
# ============================================================================
def get_file_extension(file_path: str | Path) -> str:
    """
    Return the lowercase file extension for a file path.
    """
    return Path(file_path).suffix.lower()


def is_supported_file_type(file_path: str | Path) -> bool:
    """
    Return True if the file extension is supported by the application.
    """
    return get_file_extension(file_path) in settings.supported_file_extensions


def list_excel_sheets(file_path: str | Path) -> List[str]:
    """
    Return all sheet names in an Excel workbook.
    """
    excel_file = pd.ExcelFile(file_path)
    return excel_file.sheet_names


def load_csv_file(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def load_excel_sheet(file_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """
    Load a specific Excel sheet into a pandas DataFrame.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name)


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply lightweight cleaning to a raw uploaded DataFrame.

    Current behavior:
    - drop fully empty rows
    - drop fully empty columns
    - strip whitespace from string column names
    """
    if df.empty:
        return df.copy()

    working = df.copy()

    # Normalize column names
    working.columns = [str(col).strip() for col in working.columns]

    # Drop fully empty rows / columns
    working = working.dropna(axis=0, how="all")
    working = working.dropna(axis=1, how="all")

    return working.reset_index(drop=True)


def _normalized_columns(df: pd.DataFrame) -> set[str]:
    """
    Return normalized lowercase column names for matching/scoring.
    """
    return {str(col).strip().lower() for col in df.columns}


# ============================================================================
# Upload type detection / scoring
# ============================================================================
SNAPSHOT_COLUMN_HINTS = {
    "sector",
    "team",
    "date",
    "dates",
    "position",
    "ticker",
    "shares",
    "cost",
    "market value",
    "price",
}

TRADE_RECEIPT_COLUMN_HINTS = {
    "sector",
    "team",
    "trade",
    "ticker",
    "quantity",
    "gross price",
    "commission",
    "settlement date",
    "net-net consideration",
    "net consideration",
    "trade date",
}


def score_snapshot_sheet(df: pd.DataFrame) -> int:
    """
    Score a sheet for how likely it is to be a portfolio snapshot.
    """
    cols = _normalized_columns(df)
    return sum(1 for hint in SNAPSHOT_COLUMN_HINTS if hint in cols)


def score_trade_sheet(df: pd.DataFrame) -> int:
    """
    Score a sheet for how likely it is to be a trade receipt.
    """
    cols = _normalized_columns(df)
    return sum(1 for hint in TRADE_RECEIPT_COLUMN_HINTS if hint in cols)


def detect_upload_type_from_dataframe(df: pd.DataFrame) -> str:
    """
    Detect whether a DataFrame is more likely a snapshot or trade receipt.

    Returns:
    - "snapshot"
    - "trade_receipt"
    - "unknown"
    """
    snapshot_score = score_snapshot_sheet(df)
    trade_score = score_trade_sheet(df)

    if snapshot_score == 0 and trade_score == 0:
        return "unknown"

    if snapshot_score >= trade_score:
        return "snapshot"

    return "trade_receipt"


# ============================================================================
# Sheet selection
# ============================================================================
def choose_best_excel_sheet_for_upload_type(
    file_path: str | Path,
    upload_type: str,
) -> Tuple[str, pd.DataFrame]:
    """
    Choose the most likely sheet in an Excel workbook for a specific upload type.

    upload_type must be one of:
    - "snapshot"
    - "trade_receipt"
    """
    sheet_names = list_excel_sheets(file_path)

    if not sheet_names:
        raise ValueError("No sheets found in the Excel workbook.")

    best_sheet_name: Optional[str] = None
    best_sheet_df: Optional[pd.DataFrame] = None
    best_score = -1

    for sheet_name in sheet_names:
        raw_df = load_excel_sheet(file_path, sheet_name)
        df = clean_raw_dataframe(raw_df)

        if upload_type == "snapshot":
            score = score_snapshot_sheet(df)
        elif upload_type == "trade_receipt":
            score = score_trade_sheet(df)
        else:
            raise ValueError(f"Unsupported upload_type: {upload_type}")

        if score > best_score:
            best_score = score
            best_sheet_name = sheet_name
            best_sheet_df = df

    if best_sheet_name is None or best_sheet_df is None:
        raise ValueError("Could not determine the best sheet from the workbook.")

    return best_sheet_name, best_sheet_df


def choose_best_excel_sheet_auto(
    file_path: str | Path,
) -> Tuple[str, pd.DataFrame, str]:
    """
    Automatically choose the best sheet and infer upload type.

    Returns:
    - sheet_name
    - cleaned DataFrame
    - detected upload_type
    """
    sheet_names = list_excel_sheets(file_path)

    if not sheet_names:
        raise ValueError("No sheets found in the Excel workbook.")

    best_sheet_name: Optional[str] = None
    best_sheet_df: Optional[pd.DataFrame] = None
    best_type = "unknown"
    best_score = -1

    for sheet_name in sheet_names:
        raw_df = load_excel_sheet(file_path, sheet_name)
        df = clean_raw_dataframe(raw_df)

        snapshot_score = score_snapshot_sheet(df)
        trade_score = score_trade_sheet(df)

        score = max(snapshot_score, trade_score)
        upload_type = "snapshot" if snapshot_score >= trade_score else "trade_receipt"

        if score > best_score:
            best_score = score
            best_sheet_name = sheet_name
            best_sheet_df = df
            best_type = upload_type

    if best_sheet_name is None or best_sheet_df is None:
        raise ValueError("Could not determine the best sheet from the workbook.")

    return best_sheet_name, best_sheet_df, best_type


# ============================================================================
# Snapshot loaders
# ============================================================================
def load_snapshot_file(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a portfolio snapshot file and return:
    - cleaned DataFrame
    - metadata
    """
    file_path = Path(file_path)

    if not is_supported_file_type(file_path):
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types are: {settings.supported_file_extensions}"
        )

    extension = get_file_extension(file_path)

    if extension == ".csv":
        raw_df = load_csv_file(file_path)
        df = clean_raw_dataframe(raw_df)

        metadata = {
            "file_path": str(file_path),
            "file_type": "csv",
            "selected_sheet": "",
            "upload_type": "snapshot",
        }
        return df, metadata

    if extension in {".xlsx", ".xls"}:
        sheet_name, df = choose_best_excel_sheet_for_upload_type(file_path, "snapshot")
        metadata = {
            "file_path": str(file_path),
            "file_type": "excel",
            "selected_sheet": sheet_name,
            "upload_type": "snapshot",
        }
        return df, metadata

    raise ValueError(f"Unsupported file type encountered: {extension}")


# ============================================================================
# Trade receipt loaders
# ============================================================================
def load_trade_receipt_file(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a trade receipt file and return:
    - cleaned DataFrame
    - metadata
    """
    file_path = Path(file_path)

    if not is_supported_file_type(file_path):
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types are: {settings.supported_file_extensions}"
        )

    extension = get_file_extension(file_path)

    if extension == ".csv":
        raw_df = load_csv_file(file_path)
        df = clean_raw_dataframe(raw_df)

        metadata = {
            "file_path": str(file_path),
            "file_type": "csv",
            "selected_sheet": "",
            "upload_type": "trade_receipt",
        }
        return df, metadata

    if extension in {".xlsx", ".xls"}:
        sheet_name, df = choose_best_excel_sheet_for_upload_type(file_path, "trade_receipt")
        metadata = {
            "file_path": str(file_path),
            "file_type": "excel",
            "selected_sheet": sheet_name,
            "upload_type": "trade_receipt",
        }
        return df, metadata

    raise ValueError(f"Unsupported file type encountered: {extension}")


# ============================================================================
# Auto loader
# ============================================================================
def load_uploaded_file_auto(file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Automatically load an uploaded file and infer whether it is a snapshot
    or trade receipt.

    Returns:
    - cleaned DataFrame
    - metadata including upload_type
    """
    file_path = Path(file_path)

    if not is_supported_file_type(file_path):
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types are: {settings.supported_file_extensions}"
        )

    extension = get_file_extension(file_path)

    if extension == ".csv":
        raw_df = load_csv_file(file_path)
        df = clean_raw_dataframe(raw_df)
        detected_type = detect_upload_type_from_dataframe(df)

        metadata = {
            "file_path": str(file_path),
            "file_type": "csv",
            "selected_sheet": "",
            "upload_type": detected_type,
        }
        return df, metadata

    if extension in {".xlsx", ".xls"}:
        sheet_name, df, detected_type = choose_best_excel_sheet_auto(file_path)
        metadata = {
            "file_path": str(file_path),
            "file_type": "excel",
            "selected_sheet": sheet_name,
            "upload_type": detected_type,
        }
        return df, metadata

    raise ValueError(f"Unsupported file type encountered: {extension}")


# ============================================================================
# Preview helpers
# ============================================================================
def preview_uploaded_file(
    file_path: str | Path,
    upload_type: str | None = None,
    n_rows: int = 5,
) -> Dict[str, object]:
    """
    Return a lightweight preview of an uploaded file for debugging or UI display.

    If upload_type is provided, use the specific loader.
    Otherwise detect automatically.
    """
    if upload_type == "snapshot":
        df, metadata = load_snapshot_file(file_path)
    elif upload_type == "trade_receipt":
        df, metadata = load_trade_receipt_file(file_path)
    else:
        df, metadata = load_uploaded_file_auto(file_path)

    return {
        "file_path": metadata["file_path"],
        "file_type": metadata["file_type"],
        "selected_sheet": metadata["selected_sheet"],
        "upload_type": metadata["upload_type"],
        "columns": df.columns.tolist(),
        "row_count": int(len(df)),
        "preview_rows": df.head(n_rows).to_dict(orient="records"),
    }