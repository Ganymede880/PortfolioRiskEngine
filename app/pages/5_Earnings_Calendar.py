"""
Forward-looking earnings calendar for the CMCSIF Portfolio Tracker.

This page shows upcoming earnings events across:
- the current month
- next month
- the month after next

Primary source:
- Yahoo Finance public earnings calendar pages

Optional enrichment:
- yfinance analyst estimate endpoints where available
"""

from __future__ import annotations

import calendar
import html
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analytics.portfolio import build_current_portfolio_snapshot
from src.config.settings import settings
from src.data.price_fetcher import _yfinance_request_context, fetch_latest_prices
from src.db.crud import load_position_state
from src.db.session import session_scope
from src.utils.ui import apply_app_theme, render_page_title, render_top_nav


COL_TICKER = "ticker"
COL_TEAM = "team"
COL_POSITION_SIDE = "position_side"
COL_MARKET_VALUE = "market_value"
YAHOO_EARNINGS_CALENDAR_URL = "https://finance.yahoo.com/calendar/earnings"
YAHOO_PAGE_SIZE = 100
YAHOO_MAX_PAGES_PER_DAY = 20
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
CANONICAL_EVENT_COLUMNS = [
    "earnings_date",
    "ticker",
    "company_name",
    "pod",
    "earnings_session",
    "eps_estimate",
    "revenue_estimate",
    "source",
]
POD_DISPLAY_MAP = {
    "TMT": "TMT",
    "M&I": "M&I",
    "E&U": "E&U",
    "CONSUMER": "CON",
    "CON": "CON",
    "F&R": "F&R",
    "HEALTHCARE": "HC",
    "HC": "HC",
}


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)


def _render_summary_card(label: str, value: str) -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="earnings-summary-card">
            <div class="earnings-summary-card-label">{safe_label}</div>
            <div class="earnings-summary-card-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _apply_page_theme() -> None:
    st.markdown(
        """
        <style>
        .earnings-summary-card {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.14), rgba(59, 130, 246, 0.18));
            border: 1px solid rgba(14, 116, 144, 0.22);
            border-radius: 16px;
            color: inherit;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            min-height: 108px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .earnings-summary-card-label {
            color: inherit;
            opacity: 0.8;
            font-size: 0.95rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }

        .earnings-summary-card-value {
            color: inherit;
            font-size: 1.65rem;
            font-weight: 600;
            line-height: 1.2;
        }

        .earnings-calendar-dow {
            color: inherit;
            opacity: 0.78;
            font-size: 0.86rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 0 0.1rem 0.2rem;
            text-transform: uppercase;
        }

        .earnings-calendar-day-muted {
            opacity: 0.44;
        }

        .earnings-calendar-date {
            color: inherit;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }

        .earnings-calendar-events {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .earnings-calendar-event {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.10), rgba(59, 130, 246, 0.12));
            border: 1px solid rgba(59, 130, 246, 0.18);
            border-radius: 12px;
            margin-bottom: 0.55rem;
            padding: 0.55rem 0.6rem;
        }

        .earnings-calendar-event-header {
            align-items: center;
            display: flex;
            gap: 0.4rem;
            justify-content: space-between;
        }

        .earnings-calendar-ticker {
            color: inherit;
            font-size: 0.96rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }

        .earnings-calendar-pod {
            background: rgba(15, 23, 42, 0.08);
            border-radius: 999px;
            color: inherit;
            opacity: 0.88;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.12rem 0.45rem;
            white-space: nowrap;
        }

        html[data-theme="dark"] .earnings-summary-card-label,
        html[data-theme="dark"] .earnings-summary-card-value,
        body[data-theme="dark"] .earnings-summary-card-label,
        body[data-theme="dark"] .earnings-summary-card-value,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-summary-card-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-summary-card-value {
            color: #FFFFFF !important;
        }

        html[data-theme="dark"] .earnings-calendar-dow,
        html[data-theme="dark"] .earnings-calendar-date,
        html[data-theme="dark"] .earnings-calendar-ticker,
        body[data-theme="dark"] .earnings-calendar-dow,
        body[data-theme="dark"] .earnings-calendar-date,
        body[data-theme="dark"] .earnings-calendar-ticker,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-calendar-dow,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-calendar-date,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-calendar-ticker {
            color: #FFFFFF !important;
        }

        html[data-theme="dark"] .earnings-calendar-event,
        body[data-theme="dark"] .earnings-calendar-event,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-calendar-event {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.24), rgba(59, 130, 246, 0.22));
            border-color: rgba(96, 165, 250, 0.22);
        }

        html[data-theme="dark"] .earnings-calendar-pod,
        body[data-theme="dark"] .earnings-calendar-pod,
        [data-testid="stAppViewContainer"][data-theme="dark"] .earnings-calendar-pod {
            color: rgba(255, 255, 255, 0.82) !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


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


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_holdings_universe() -> pd.DataFrame:
    with session_scope() as session:
        position_state_df = load_position_state(session)

    if position_state_df.empty:
        return pd.DataFrame(columns=[COL_TICKER, COL_TEAM, COL_MARKET_VALUE])

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
    snapshot_df = build_current_portfolio_snapshot(position_state_df, latest_prices_df)
    if snapshot_df.empty:
        return pd.DataFrame(columns=[COL_TICKER, COL_TEAM, COL_MARKET_VALUE])

    snapshot_df = snapshot_df.loc[~_cash_like_mask(snapshot_df)].copy()
    if snapshot_df.empty:
        return pd.DataFrame(columns=[COL_TICKER, COL_TEAM, COL_MARKET_VALUE])

    snapshot_df[COL_TICKER] = snapshot_df[COL_TICKER].astype(str).str.strip().str.upper()
    snapshot_df[COL_TEAM] = snapshot_df[COL_TEAM].astype(str).str.strip()
    snapshot_df[COL_MARKET_VALUE] = pd.to_numeric(snapshot_df.get(COL_MARKET_VALUE), errors="coerce").fillna(0.0)
    snapshot_df = (
        snapshot_df.sort_values(COL_MARKET_VALUE, ascending=False)
        .drop_duplicates(subset=[COL_TICKER], keep="first")
        .reset_index(drop=True)
    )
    return snapshot_df[[COL_TICKER, COL_TEAM, COL_MARKET_VALUE]]


def _month_windows(reference_date: pd.Timestamp | None = None) -> list[dict[str, pd.Timestamp | str]]:
    reference = (reference_date or pd.Timestamp.today()).normalize()
    current_month_start = reference.replace(day=1)
    windows: list[dict[str, pd.Timestamp | str]] = []
    for month_offset, key in enumerate(["current_month", "next_month", "month_3"]):
        month_start = (current_month_start + pd.DateOffset(months=month_offset)).replace(day=1).normalize()
        month_end = (month_start + pd.offsets.MonthEnd(0)).normalize()
        windows.append(
            {
                "key": key,
                "label": month_start.strftime("%B %Y"),
                "start": month_start,
                "end": month_end,
            }
        )
    return windows


def _parse_numeric(value):
    if value is None:
        return pd.NA
    cleaned = str(value).strip()
    if cleaned in {"", "-", "—", "N/A", "n/a"}:
        return pd.NA
    cleaned = cleaned.replace(",", "").replace("%", "")
    try:
        return float(cleaned)
    except Exception:
        return pd.NA


def _safe_table_value(row: dict[str, str], *keys: str) -> str | None:
    normalized = {str(k).strip().lower(): v for k, v in row.items()}
    for key in keys:
        candidate = normalized.get(str(key).strip().lower())
        if candidate not in (None, ""):
            return candidate
    return None


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _fetch_yahoo_earnings_calendar_page(day_iso: str, offset: int = 0, size: int = YAHOO_PAGE_SIZE) -> str:
    params = urlencode({"day": day_iso, "offset": offset, "size": size})
    url = f"{YAHOO_EARNINGS_CALENDAR_URL}?{params}"
    request = Request(url, headers=YAHOO_HEADERS)
    try:
        with urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return ""


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _fetch_yahoo_earnings_calendar_page_debug(day_iso: str, offset: int = 0, size: int = YAHOO_PAGE_SIZE) -> dict[str, object]:
    params = urlencode({"day": day_iso, "offset": offset, "size": size})
    url = f"{YAHOO_EARNINGS_CALENDAR_URL}?{params}"
    request = Request(url, headers=YAHOO_HEADERS)
    try:
        with urlopen(request, timeout=20) as response:
            html_text = response.read().decode("utf-8", errors="ignore")
            return {
                "ok": True,
                "html_text": html_text,
                "status_code": getattr(response, "status", None),
                "final_url": response.geturl(),
                "error": None,
            }
    except Exception as exc:
        return {
            "ok": False,
            "html_text": "",
            "status_code": None,
            "final_url": url,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _extract_total_rows(html_text: str) -> int | None:
    if not html_text:
        return None
    plain_text = BeautifulSoup(html_text, "html.parser").get_text(" ", strip=True)
    match = re.search(r"(\d+)\s*-\s*(\d+)\s+of\s+(\d+)", plain_text)
    if not match:
        return None
    try:
        return int(match.group(3))
    except ValueError:
        return None


def _parse_yahoo_calendar_table(html_text: str, earnings_date: pd.Timestamp) -> pd.DataFrame:
    if not html_text:
        return pd.DataFrame(columns=[col for col in CANONICAL_EVENT_COLUMNS if col != "pod"] + ["reported_eps"])

    soup = BeautifulSoup(html_text, "html.parser")
    table = None
    headers: list[str] = []
    for candidate in soup.find_all("table"):
        header_cells = [cell.get_text(" ", strip=True) for cell in candidate.find_all("th")]
        normalized_headers = {header.strip().lower() for header in header_cells}
        if "symbol" in normalized_headers and "company" in normalized_headers:
            table = candidate
            headers = header_cells
            break

    if table is None or not headers:
        return pd.DataFrame(columns=[col for col in CANONICAL_EVENT_COLUMNS if col != "pod"] + ["reported_eps"])

    records: list[dict[str, object]] = []
    body_rows = table.find_all("tr")
    for tr in body_rows:
        cells = tr.find_all("td")
        if not cells:
            continue
        values = [cell.get_text(" ", strip=True) for cell in cells]
        if len(values) < 2:
            continue

        row_dict = dict(zip(headers, values))
        ticker = _safe_table_value(row_dict, "Symbol", "Ticker")
        if not ticker:
            continue

        ticker_clean = str(ticker).strip().upper()
        if not ticker_clean or ticker_clean == "SYMBOL":
            continue

        company_name = _safe_table_value(row_dict, "Company", "Company Name")
        earnings_session = _safe_table_value(row_dict, "Earnings Call Time", "Time", "Session")
        eps_estimate = _parse_numeric(_safe_table_value(row_dict, "EPS Estimate"))
        reported_eps = _parse_numeric(_safe_table_value(row_dict, "Reported EPS"))
        records.append(
            {
                "earnings_date": earnings_date.normalize(),
                "ticker": ticker_clean,
                "company_name": company_name if company_name else pd.NA,
                "earnings_session": earnings_session if earnings_session else pd.NA,
                "eps_estimate": eps_estimate,
                "reported_eps": reported_eps,
                "source": "Yahoo Finance earnings calendar",
            }
        )

    return pd.DataFrame(records)


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _fetch_yahoo_earnings_calendar_for_range(start_date: str, end_date: str) -> tuple[pd.DataFrame, dict[str, object]]:
    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        return _empty_events_df(), {"ok": False, "message": "Invalid earnings calendar date range.", "fallback_allowed": False}

    preflight_result = _fetch_yahoo_earnings_calendar_page_debug(
        day_iso=start_ts.strftime("%Y-%m-%d"),
        offset=0,
        size=YAHOO_PAGE_SIZE,
    )
    preflight_html = str(preflight_result.get("html_text") or "")
    if not preflight_html and preflight_result.get("error"):
        return _empty_events_df(), {
            "ok": False,
            "message": f"{start_ts.strftime('%Y-%m-%d')}: {preflight_result.get('error')}",
            "fallback_allowed": False,
        }

    all_frames: list[pd.DataFrame] = []
    failure_messages: list[str] = []
    successful_fetches = 0
    for earnings_day in pd.date_range(start_ts, end_ts, freq="D"):
        offset = 0
        total_rows = None
        page_number = 0
        seen_signatures: set[tuple[str, ...]] = set()

        while page_number < YAHOO_MAX_PAGES_PER_DAY:
            if earnings_day.normalize() == start_ts.normalize() and offset == 0:
                fetch_result = preflight_result
            else:
                fetch_result = _fetch_yahoo_earnings_calendar_page_debug(
                    day_iso=earnings_day.strftime("%Y-%m-%d"),
                    offset=offset,
                    size=YAHOO_PAGE_SIZE,
                )
            html_text = str(fetch_result.get("html_text") or "")
            if not html_text:
                error_message = fetch_result.get("error")
                if error_message:
                    failure_messages.append(
                        f"{earnings_day.strftime('%Y-%m-%d')}: {error_message}"
                    )
                break
            successful_fetches += 1

            page_df = _parse_yahoo_calendar_table(html_text, earnings_day)
            if page_df.empty and offset == 0:
                break

            if total_rows is None:
                total_rows = _extract_total_rows(html_text)

            if not page_df.empty:
                signature = tuple(page_df["ticker"].astype(str).tolist())
                if signature in seen_signatures:
                    break
                seen_signatures.add(signature)
                all_frames.append(page_df)

            row_count = len(page_df)
            if row_count < YAHOO_PAGE_SIZE:
                break

            offset += row_count
            page_number += 1
            if total_rows is not None and offset >= total_rows:
                break

    if not all_frames:
        status = {
            "ok": successful_fetches > 0,
            "message": "; ".join(failure_messages[:3]) if failure_messages else "No Yahoo Finance earnings events were returned for the selected window.",
            "fallback_allowed": successful_fetches > 0,
        }
        return _empty_events_df(), status

    combined = pd.concat(all_frames, ignore_index=True)
    combined["earnings_date"] = pd.to_datetime(combined["earnings_date"], errors="coerce")
    combined = combined.dropna(subset=["earnings_date", "ticker"]).copy()
    if combined.empty:
        return _empty_events_df(), {"ok": successful_fetches > 0, "message": "Yahoo Finance returned pages, but no parsable earnings rows were found.", "fallback_allowed": True}

    combined["reported_eps_missing"] = combined["reported_eps"].isna()
    combined = (
        combined.sort_values(
            by=["earnings_date", "ticker", "reported_eps_missing"],
            ascending=[True, True, False],
        )
        .drop_duplicates(subset=["earnings_date", "ticker"], keep="first")
        .drop(columns=["reported_eps", "reported_eps_missing"], errors="ignore")
        .reset_index(drop=True)
    )
    combined["pod"] = pd.NA
    combined["revenue_estimate"] = pd.NA
    return combined[CANONICAL_EVENT_COLUMNS], {"ok": True, "message": None, "fallback_allowed": False}


def _extract_calendar_value(calendar_obj, key: str):
    if calendar_obj is None:
        return None
    if isinstance(calendar_obj, dict):
        return calendar_obj.get(key)
    if isinstance(calendar_obj, pd.DataFrame):
        if key in calendar_obj.index and not calendar_obj.empty:
            row = calendar_obj.loc[key]
            if isinstance(row, pd.Series):
                non_null = row.dropna()
                return non_null.iloc[0] if not non_null.empty else None
            return row
        if key in calendar_obj.columns and not calendar_obj.empty:
            series = calendar_obj[key]
            if isinstance(series, pd.Series):
                non_null = series.dropna()
                return non_null.iloc[0] if not non_null.empty else None
            return series
    return None


def _normalize_calendar_date(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_calendar_date(item)
            if normalized is not None:
                return normalized
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.normalize()


def _first_non_null(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        return value
    return None


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _fetch_yfinance_calendar_for_ticker(ticker: str) -> dict[str, object]:
    result = {
        "ticker": str(ticker).strip().upper(),
        "earnings_date": None,
        "company_name": None,
        "earnings_session": None,
        "eps_estimate": None,
        "revenue_estimate": None,
        "source": "yfinance calendar fallback",
    }
    cleaned_ticker = result["ticker"]
    if not cleaned_ticker:
        return result

    try:
        with _yfinance_request_context():
            ticker_obj = yf.Ticker(cleaned_ticker)
            try:
                calendar_obj = ticker_obj.calendar
            except Exception:
                calendar_obj = None

            try:
                company_name = ticker_obj.info.get("shortName") or ticker_obj.info.get("longName")
            except Exception:
                company_name = None
    except Exception:
        return result

    earnings_date = _normalize_calendar_date(
        _first_non_null(
            _extract_calendar_value(calendar_obj, "Earnings Date"),
            _extract_calendar_value(calendar_obj, "earningsDate"),
        )
    )
    if earnings_date is not None:
        result["earnings_date"] = earnings_date

    result["company_name"] = company_name

    consensus = _fetch_yfinance_consensus_for_ticker(cleaned_ticker)
    result["eps_estimate"] = consensus.get("eps_estimate")
    result["revenue_estimate"] = consensus.get("revenue_estimate")
    return result


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _build_yfinance_calendar_fallback(
    ticker_universe: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start_date, errors="coerce")
    end_ts = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts) or not ticker_universe:
        return _empty_events_df()

    rows: list[dict[str, object]] = []
    for ticker in ticker_universe:
        row = _fetch_yfinance_calendar_for_ticker(ticker)
        earnings_date = pd.to_datetime(row.get("earnings_date"), errors="coerce")
        if pd.isna(earnings_date):
            continue
        if not (start_ts <= earnings_date <= end_ts):
            continue
        rows.append(
            {
                "earnings_date": earnings_date.normalize(),
                "ticker": row.get("ticker"),
                "company_name": row.get("company_name") or pd.NA,
                "pod": pd.NA,
                "earnings_session": row.get("earnings_session") or pd.NA,
                "eps_estimate": row.get("eps_estimate"),
                "revenue_estimate": row.get("revenue_estimate"),
                "source": row.get("source") or "yfinance calendar fallback",
            }
        )

    if not rows:
        return _empty_events_df()

    fallback_df = pd.DataFrame(rows)[CANONICAL_EVENT_COLUMNS]
    fallback_df = fallback_df.sort_values(["earnings_date", "ticker"]).drop_duplicates(
        subset=["earnings_date", "ticker"], keep="first"
    )
    return fallback_df.reset_index(drop=True)


def _extract_single_near_term_consensus(df: pd.DataFrame, value_column: str) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty or value_column not in df.columns:
        return None

    working = df.copy()
    period_series = (
        pd.Series(working.index, index=working.index)
        .astype(str)
        .str.strip()
        .str.lower()
    )
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    near_term = working.loc[period_series.isin(["0q", "+1q"]), [value_column]].dropna()
    if near_term.empty:
        return None
    if len(near_term) == 1:
        return float(near_term.iloc[0][value_column])
    return None


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _fetch_yfinance_consensus_for_ticker(ticker: str) -> dict[str, float | None]:
    result = {"eps_estimate": None, "revenue_estimate": None}
    cleaned_ticker = str(ticker).strip().upper()
    if not cleaned_ticker:
        return result

    try:
        with _yfinance_request_context():
            ticker_obj = yf.Ticker(cleaned_ticker)
            try:
                earnings_df = ticker_obj.get_earnings_estimate(as_dict=False)
                result["eps_estimate"] = _extract_single_near_term_consensus(earnings_df, "avg")
            except Exception:
                pass

            try:
                revenue_df = ticker_obj.get_revenue_estimate(as_dict=False)
                result["revenue_estimate"] = _extract_single_near_term_consensus(revenue_df, "avg")
            except Exception:
                pass
    except Exception:
        return result

    return result


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def _enrich_with_yfinance_consensus(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    enriched = events_df.copy()
    tickers = sorted(enriched["ticker"].dropna().astype(str).str.strip().str.upper().unique().tolist())
    if not tickers:
        return enriched

    enrichment_rows = []
    for ticker in tickers:
        consensus = _fetch_yfinance_consensus_for_ticker(ticker)
        enrichment_rows.append(
            {
                "ticker": ticker,
                "yf_eps_estimate": consensus.get("eps_estimate"),
                "yf_revenue_estimate": consensus.get("revenue_estimate"),
            }
        )

    enrichment_df = pd.DataFrame(enrichment_rows)
    if enrichment_df.empty:
        return enriched

    enriched = enriched.merge(enrichment_df, on="ticker", how="left")
    original_eps_missing = pd.to_numeric(enriched["eps_estimate"], errors="coerce").isna()
    original_revenue_missing = pd.to_numeric(enriched["revenue_estimate"], errors="coerce").isna()
    enriched["eps_estimate"] = pd.to_numeric(enriched["eps_estimate"], errors="coerce").where(
        ~original_eps_missing,
        pd.to_numeric(enriched["yf_eps_estimate"], errors="coerce"),
    )
    enriched["revenue_estimate"] = pd.to_numeric(enriched["revenue_estimate"], errors="coerce").where(
        ~original_revenue_missing,
        pd.to_numeric(enriched["yf_revenue_estimate"], errors="coerce"),
    )
    enrichment_used = original_eps_missing & pd.to_numeric(enriched["yf_eps_estimate"], errors="coerce").notna()
    revenue_enrichment_used = original_revenue_missing & pd.to_numeric(enriched["yf_revenue_estimate"], errors="coerce").notna()
    enriched.loc[enrichment_used | revenue_enrichment_used, "source"] = "Yahoo Finance calendar + yfinance enrichment"
    return enriched.drop(columns=["yf_eps_estimate", "yf_revenue_estimate"], errors="ignore")


@st.cache_data(ttl=settings.price_refresh_interval_seconds)
def get_earnings_calendar_data(
    snapshot_df: pd.DataFrame,
    selected_universe: str,
    custom_tickers: tuple[str, ...],
) -> pd.DataFrame:
    windows = _month_windows()
    range_start = windows[0]["start"]
    range_end = windows[-1]["end"]

    if selected_universe == "Current Holdings Only":
        ticker_universe = (
            snapshot_df[COL_TICKER].dropna().astype(str).str.strip().str.upper().unique().tolist()
            if not snapshot_df.empty else []
        )
        pod_map = (
            snapshot_df[[COL_TICKER, COL_TEAM]]
            .dropna(subset=[COL_TICKER])
            .assign(**{COL_TICKER: lambda df: df[COL_TICKER].astype(str).str.strip().str.upper()})
            .drop_duplicates(subset=[COL_TICKER], keep="first")
            .set_index(COL_TICKER)[COL_TEAM]
            .to_dict()
            if not snapshot_df.empty else {}
        )
    else:
        ticker_universe = [ticker for ticker in custom_tickers if str(ticker).strip()]
        pod_map = {}

    ticker_universe = sorted({str(ticker).strip().upper() for ticker in ticker_universe if str(ticker).strip()})
    if not ticker_universe:
        return _empty_events_df()

    yahoo_events_df, yahoo_status = _fetch_yahoo_earnings_calendar_for_range(
        start_date=pd.Timestamp(range_start).strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(range_end).strftime("%Y-%m-%d"),
    )

    filtered = _empty_events_df()
    if not yahoo_events_df.empty:
        yahoo_events_df["ticker"] = yahoo_events_df["ticker"].astype(str).str.strip().str.upper()
        filtered = yahoo_events_df.loc[
            yahoo_events_df["ticker"].isin(ticker_universe)
        ].copy()
        if not filtered.empty:
            filtered = _enrich_with_yfinance_consensus(filtered)

    if filtered.empty and bool(yahoo_status.get("fallback_allowed", True)):
        filtered = _build_yfinance_calendar_fallback(
            ticker_universe=tuple(ticker_universe),
            start_date=pd.Timestamp(range_start).strftime("%Y-%m-%d"),
            end_date=pd.Timestamp(range_end).strftime("%Y-%m-%d"),
        )

    if filtered.empty:
        return filtered

    filtered["pod"] = filtered["ticker"].map(pod_map) if pod_map else pd.NA
    filtered["earnings_date"] = pd.to_datetime(filtered["earnings_date"], errors="coerce")
    filtered = filtered.dropna(subset=["earnings_date"]).sort_values(["earnings_date", "ticker"]).reset_index(drop=True)
    return filtered


def _apply_earnings_filters(
    events_df: pd.DataFrame,
    selected_universe: str,
    selected_pods: list[str],
    ticker_search: str,
) -> pd.DataFrame:
    if events_df.empty:
        return events_df

    filtered = events_df.copy()
    if selected_universe == "Current Holdings Only" and selected_pods:
        filtered["pod"] = filtered["pod"].astype(str)
        filtered = filtered.loc[filtered["pod"].isin(selected_pods)].copy()

    search_value = str(ticker_search).strip().upper()
    if search_value:
        filtered = filtered.loc[
            filtered["ticker"].astype(str).str.upper().str.contains(search_value, na=False)
        ].copy()

    return filtered.sort_values(["earnings_date", "ticker"]).reset_index(drop=True)


def render_summary_metrics(events_df: pd.DataFrame, windows: list[dict[str, pd.Timestamp | str]]) -> None:
    total_events = len(events_df)
    counts = []
    for window in windows:
        start_ts = pd.Timestamp(window["start"])
        end_ts = pd.Timestamp(window["end"])
        count = len(
            events_df.loc[
                events_df["earnings_date"].between(start_ts, end_ts, inclusive="both")
            ]
        ) if not events_df.empty else 0
        counts.append(count)

    with st.container(border=True):
        row = st.columns(4)
        with row[0]:
            _render_summary_card("Total Upcoming Earnings Events", f"{total_events:,}")
        with row[1]:
            _render_summary_card(f"{pd.Timestamp(windows[0]['start']).strftime('%B')} Events", f"{counts[0]:,}")
        with row[2]:
            _render_summary_card(f"{pd.Timestamp(windows[1]['start']).strftime('%B')} Events", f"{counts[1]:,}")
        with row[3]:
            _render_summary_card(f"{pd.Timestamp(windows[2]['start']).strftime('%B')} Events", f"{counts[2]:,}")

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)


def _render_calendar_event_card(row: pd.Series) -> str:
    ticker = html.escape(str(row.get("ticker", "N/A")))
    pod = row.get("pod")
    pod_raw = "" if pod is None or pd.isna(pod) else str(pod).strip()
    pod_text = POD_DISPLAY_MAP.get(pod_raw.upper(), pod_raw or "Holdings")
    return f"""
        <div class="earnings-calendar-event">
            <div class="earnings-calendar-event-header">
                <div class="earnings-calendar-ticker">{ticker}</div>
                <div class="earnings-calendar-pod">{html.escape(pod_text)}</div>
            </div>
        </div>
    """


def _render_day_cell_html(current_date, is_current_month: bool, events: list[dict[str, object]], min_height_px: int) -> str:
    date_classes = "earnings-calendar-date"
    if not is_current_month:
        date_classes += " earnings-calendar-day-muted"
    events_html = "".join(_render_calendar_event_card(pd.Series(event)) for event in events)
    return f"""
        <div style="min-height:{min_height_px}px; display:flex; flex-direction:column; justify-content:flex-start; padding-bottom:0.05rem;">
            <div class="{date_classes}">{current_date.day}</div>
            {events_html}
        </div>
    """


def _group_events_by_date(month_df: pd.DataFrame) -> dict[object, list[dict[str, object]]]:
    if month_df.empty:
        return {}
    working_df = month_df.copy()
    working_df["earnings_date"] = pd.to_datetime(working_df["earnings_date"], errors="coerce").dt.date
    working_df = working_df.dropna(subset=["earnings_date"]).sort_values(["earnings_date", "ticker"])
    grouped: dict[object, list[dict[str, object]]] = {}
    for earnings_date, group in working_df.groupby("earnings_date", sort=True):
        grouped[earnings_date] = group.to_dict("records")
    return grouped


def render_month_tab(events_df: pd.DataFrame, window: dict[str, pd.Timestamp | str]) -> None:
    month_df = events_df.loc[
        events_df["earnings_date"].between(
            pd.Timestamp(window["start"]),
            pd.Timestamp(window["end"]),
            inclusive="both",
        )
    ].copy() if not events_df.empty else pd.DataFrame()

    if month_df.empty:
        st.info(f"No earnings events found for {window['label']}.")
        return

    month_df = month_df.sort_values(["earnings_date", "ticker"]).reset_index(drop=True)
    start_ts = pd.Timestamp(window["start"])
    month_calendar = calendar.Calendar(firstweekday=6)
    weeks = month_calendar.monthdatescalendar(start_ts.year, start_ts.month)
    events_by_date = _group_events_by_date(month_df)

    header_cols = st.columns(7)
    for col, day_name in zip(header_cols, ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]):
        with col:
            st.markdown(f'<div class="earnings-calendar-dow">{day_name}</div>', unsafe_allow_html=True)

    for week in weeks:
        max_events_in_week = max(len(events_by_date.get(day, [])) for day in week)
        row_min_height = 80 if max_events_in_week == 0 else 54 + (max_events_in_week * 51)
        week_cols = st.columns(7)
        for col, current_date in zip(week_cols, week):
            is_current_month = current_date.month == start_ts.month
            day_events = events_by_date.get(current_date, [])
            with col:
                with st.container(border=True):
                    st.markdown(
                        _render_day_cell_html(
                            current_date=current_date,
                            is_current_month=is_current_month,
                            events=day_events,
                            min_height_px=row_min_height,
                        ),
                        unsafe_allow_html=True,
                    )


def main() -> None:
    st.set_page_config(page_title="Earnings Calendar", layout="wide")
    apply_app_theme()
    render_top_nav()
    _apply_page_theme()
    render_page_title("Earnings Calendar")

    holdings_snapshot_df = get_holdings_universe()

    if holdings_snapshot_df.empty:
        st.info(
            "No current holdings are available yet. Upload snapshots and/or trades, then rebuild position state."
        )
        return

    events_df = get_earnings_calendar_data(
        snapshot_df=holdings_snapshot_df,
        selected_universe="Current Holdings Only",
        custom_tickers=tuple(),
    )

    windows = _month_windows()
    render_summary_metrics(events_df, windows)

    tabs = st.tabs([str(window["label"]) for window in windows])
    for tab, window in zip(tabs, windows):
        with tab:
            render_month_tab(events_df, window)

    with st.expander("Methodology Note"):
        st.write(
            """
            Future earnings events on this page come from yfinance calendar data for the
            selected holdings or custom ticker list over the current month plus the next
            two months.

            Earnings availability can still be incomplete for some tickers, so the page
            preserves partial results and fails soft when dates are unavailable.
            """
        )


if __name__ == "__main__":
    main()
