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
