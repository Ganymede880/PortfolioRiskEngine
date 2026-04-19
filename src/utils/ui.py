from __future__ import annotations

import pandas as pd
import streamlit as st


SITE_BG = "#07111a"
SITE_TEXT = "#f8fafc"
PANEL_BG = "rgba(15, 23, 42, 0.74)"
PANEL_BORDER = "rgba(148, 163, 184, 0.35)"


def apply_app_theme() -> None:
    st.markdown(
        f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {{
            background: {SITE_BG} !important;
            color: {SITE_TEXT} !important;
        }}

        .block-container {{
            background: {SITE_BG} !important;
            color: {SITE_TEXT} !important;
            padding-top: 1.5rem;
        }}

        h1, h2, h3, h4, h5, h6,
        p, span, label, div, li,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] {{
            color: {SITE_TEXT} !important;
        }}

        [data-testid="stMetric"] {{
            background: transparent;
            border: none;
            border-radius: 0;
            padding: 0.1rem 0.15rem;
            box-shadow: none;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {PANEL_BG};
            border-color: {PANEL_BORDER} !important;
            border-radius: 18px;
        }}

        [data-testid="stDataFrame"] table,
        [data-testid="stDataFrame"] th,
        [data-testid="stDataFrame"] td {{
            text-align: left !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def left_align_dataframe(df: pd.DataFrame):
    return df.style.set_properties(
        **{
            "text-align": "left",
            "background-color": SITE_BG,
            "color": SITE_TEXT,
        }
    ).set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("text-align", "left"),
                    ("background-color", SITE_BG),
                    ("color", SITE_TEXT),
                ],
            }
        ]
    )


def style_plotly_figure(fig, title_text: str | None = None, center_title: bool = True):
    layout_kwargs = {
        "paper_bgcolor": SITE_BG,
        "plot_bgcolor": SITE_BG,
        "font": {"color": SITE_TEXT},
        "legend": {"font": {"color": SITE_TEXT}, "title": {"font": {"color": SITE_TEXT}}},
    }
    if title_text is not None:
        layout_kwargs["title"] = {
            "text": title_text,
            "font": {"color": SITE_TEXT, "size": 16},
            "x": 0.5 if center_title else 0.0,
            "xanchor": "center" if center_title else "left",
        }
    fig.update_layout(**layout_kwargs)
    return fig
