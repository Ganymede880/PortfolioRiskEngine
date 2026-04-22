from __future__ import annotations

import html
import pandas as pd


import streamlit as st

def apply_app_theme():
    st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }

    section[data-testid="stSidebar"],
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .stApp .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }

    div[data-testid="column"] {
        text-align: center;
    }

    div[data-testid="column"] > div {
        display: flex;
        justify-content: center;
        width: 100%;
    }

    a[data-testid="stPageLink"] {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-height: 44px;
        width: 100%;
        font-weight: 600;
        font-size: 0.96rem;
        color: #E2E8F0 !important;
        padding: 0.55rem 0.8rem;
        border-radius: 12px;
        text-decoration: none !important;
        border: 1px solid transparent;
        background: rgba(255, 255, 255, 0.02);
        transition: all 0.18s ease;
    }

    a[data-testid="stPageLink"]:hover {
        background: rgba(45, 194, 189, 0.10);
        border: 1px solid rgba(45, 194, 189, 0.22);
        box-shadow: 0 6px 18px rgba(45, 194, 189, 0.10);
        transform: translateY(-1px);
    }

    a[data-testid="stPageLink"] > div {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    a[data-testid="stPageLink"] p {
        color: #E2E8F0 !important;
        margin: 0 !important;
        width: 100%;
        text-align: center !important;
    }

    /* active/current page */
    a[data-testid="stPageLink"][aria-current="page"] {
        background: linear-gradient(135deg, rgba(45, 194, 189, 0.18), rgba(122, 130, 171, 0.18));
        border: 1px solid rgba(148, 163, 184, 0.26);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.16);
    }
    </style>
    """, unsafe_allow_html=True)


def render_top_nav():
    st.markdown(
        """
        <style>
        .top-nav-spacer {
            height: 0.45rem;
        }

        .top-nav-divider {
            height: 1px;
            margin: 0.45rem 0 0.85rem 0;
            background: linear-gradient(
                90deg,
                rgba(45, 194, 189, 0.00),
                rgba(45, 194, 189, 0.18),
                rgba(148, 163, 184, 0.28),
                rgba(45, 194, 189, 0.18),
                rgba(45, 194, 189, 0.00)
            );
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="top-nav-spacer"></div>', unsafe_allow_html=True)

    nav_cols = st.columns(9)

    with nav_cols[0]:
        st.page_link("Dashboard_Home.py", label="Home")
    with nav_cols[1]:
        st.page_link("pages/1_Portfolio_View.py", label="Portfolio")
    with nav_cols[2]:
        st.page_link("pages/2_Sector_View.py", label="Sectors")
    with nav_cols[3]:
        st.page_link("pages/3_Factor_Model.py", label="Factors")
    with nav_cols[4]:
        st.page_link("pages/4_Risk_Engine.py", label="Risk")
    with nav_cols[5]:
        st.page_link("pages/5_Earnings_Calendar.py", label="Earnings")
    with nav_cols[6]:
        st.page_link("pages/6_Holdings.py", label="Holdings")
    with nav_cols[7]:
        st.page_link("pages/7_Portfolio_Activity.py", label="Activity")
    with nav_cols[8]:
        st.page_link("pages/8_Upload.py", label="Upload")

    st.markdown('<div class="top-nav-divider"></div>', unsafe_allow_html=True)


def apply_summary_ui_theme() -> None:
    st.markdown(
        """
        <style>
        .summary-section-kicker {
            margin: 0.15rem 0 0.75rem 0;
            color: rgba(226, 232, 240, 0.68);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .summary-data-line {
            margin: -0.35rem 0 1.05rem 0;
            color: rgba(226, 232, 240, 0.72);
            font-size: 0.92rem;
        }

        .summary-kpi-card {
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.14), rgba(59, 130, 246, 0.18));
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            color: inherit;
            padding: 1rem 1.05rem;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.10);
            min-height: 112px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .summary-kpi-card-label {
            color: inherit;
            opacity: 0.78;
            font-size: 0.9rem;
            font-weight: 500;
            line-height: 1.3;
            margin-bottom: 0.38rem;
        }

        .summary-kpi-card-value {
            color: inherit;
            font-size: 1.62rem;
            font-weight: 600;
            line-height: 1.18;
        }

        .summary-kpi-card-value.positive {
            color: #4ADE80;
        }

        .summary-kpi-card-value.negative {
            color: #F87171;
        }

        .summary-insight-panel {
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 18px;
            padding: 0.95rem 1.1rem 0.8rem 1.1rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.42), rgba(30, 41, 59, 0.24));
        }

        .summary-insight-list {
            margin: 0.15rem 0 0 0;
            padding-left: 1.1rem;
            color: rgba(226, 232, 240, 0.9);
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .summary-insight-list li {
            margin: 0 0 0.34rem 0;
        }

        .summary-status-banner {
            margin: 0.15rem 0 0.2rem 0;
            padding: 0.7rem 0.9rem;
            border-radius: 14px;
            border: 1px solid rgba(74, 222, 128, 0.20);
            background: linear-gradient(135deg, rgba(21, 128, 61, 0.14), rgba(22, 101, 52, 0.08));
            color: rgba(220, 252, 231, 0.95);
            font-size: 0.92rem;
            line-height: 1.35;
        }

        .summary-status-banner.warning {
            border-color: rgba(251, 191, 36, 0.24);
            background: linear-gradient(135deg, rgba(146, 64, 14, 0.16), rgba(120, 53, 15, 0.08));
            color: rgba(254, 243, 199, 0.96);
        }

        .summary-next-row {
            margin: 0.15rem 0 0.4rem 0;
            padding: 0.1rem 0;
            color: rgba(226, 232, 240, 0.62);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        html[data-theme="dark"] .summary-kpi-card-label,
        html[data-theme="dark"] .summary-kpi-card-value,
        body[data-theme="dark"] .summary-kpi-card-label,
        body[data-theme="dark"] .summary-kpi-card-value,
        [data-testid="stAppViewContainer"][data-theme="dark"] .summary-kpi-card-label,
        [data-testid="stAppViewContainer"][data-theme="dark"] .summary-kpi-card-value {
            color: #FFFFFF !important;
        }

        html[data-theme="dark"] .summary-kpi-card-value.positive,
        body[data-theme="dark"] .summary-kpi-card-value.positive,
        [data-testid="stAppViewContainer"][data-theme="dark"] .summary-kpi-card-value.positive {
            color: #4ADE80 !important;
        }

        html[data-theme="dark"] .summary-kpi-card-value.negative,
        body[data-theme="dark"] .summary-kpi-card-value.negative,
        [data-testid="stAppViewContainer"][data-theme="dark"] .summary-kpi-card-value.negative {
            color: #F87171 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(label: str, value: str, value_class: str = "") -> None:
    safe_label = html.escape(label)
    safe_value = html.escape(value)
    safe_value_class = html.escape(value_class.strip())
    value_class_attr = f" {safe_value_class}" if safe_value_class else ""
    st.markdown(
        f"""
        <div class="summary-kpi-card">
            <div class="summary-kpi-card-label">{safe_label}</div>
            <div class="summary-kpi-card-value{value_class_attr}">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_status_banner(message: str, tone: str = "success") -> None:
    safe_message = html.escape(message)
    safe_tone = " warning" if tone == "warning" else ""
    st.markdown(
        f'<div class="summary-status-banner{safe_tone}">{safe_message}</div>',
        unsafe_allow_html=True,
    )


def render_page_title(title: str) -> None:
    safe_title = html.escape(title)
    st.markdown(
        f"""
        <h1 style="
            margin: 0 0 1rem 0;
            padding: 0;
            line-height: 1.15;
        ">{safe_title}</h1>
        """,
        unsafe_allow_html=True,
    )


def left_align_dataframe(df: pd.DataFrame):
    return df.style.set_properties(
        **{
            "text-align": "left",
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                ],
            }
        ]
    )


def style_plotly_figure(fig, title_text: str | None = None, center_title: bool = True):
    layout_kwargs = {}
    if title_text is not None:
        layout_kwargs["title"] = {
            "text": title_text,
            "font": {"size": 16},
            "x": 0.5 if center_title else 0.0,
            "xanchor": "center" if center_title else "left",
        }
    fig.update_layout(**layout_kwargs)
    return fig
