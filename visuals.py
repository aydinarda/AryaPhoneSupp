from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd


def render_scatter(plot_df: pd.DataFrame, *, x: str, y: str) -> None:
    """
    Renders a true x-y scatter for leaderboard submissions.

    Expected columns in plot_df:
    - Team (str)
    - Mode (str)
    - created_at (datetime/str)
    - Suppliers (str)
    - plus metric columns referenced by x and y
    """
    if plot_df is None or plot_df.empty:
        st.info("No submissions to plot yet.")
        return

    # plotly can handle x==y; this will just fall on a diagonal.
    fig = px.scatter(
        plot_df,
        x=x,
        y=y,
        color="Team",          # color by group/team
        symbol="Mode",         # show objective/mode via marker symbol
        hover_data={
            "Team": True,
            "Mode": True,
            "created_at": True,
            "Suppliers": True,
        },
    )
    fig.update_layout(
        legend_title_text="Team",
        xaxis_title=x,
        yaxis_title=y,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
