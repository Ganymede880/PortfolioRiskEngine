from __future__ import annotations

import pandas as pd


def apply_app_theme() -> None:
    return None


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
