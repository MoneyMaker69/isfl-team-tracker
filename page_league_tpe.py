"""League TPE: every team's TPE, by group, over time, against age."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import ui
from page_teams import _needs_logs


def render(ctx) -> None:
    if _needs_logs(ctx):
        return
    ui.explain(
        "All 14 teams' TPE side by side. The bar chart ranks them, the scatter shows TPE against age "
        "(top-left = young and strong, bottom-right = old and about to regress), the heatmap shows which "
        "position groups each team is strong or weak in, and the timeline shows how each team's TPE has moved "
        "since S54.",
        {k: ui.GLOSSARY[k] for k in ("TPE", "Starter TPE", "Career season")}
        | {"Mean TPE": "Average over every listed player, backups included.",
           "Heatmap colour": "Green = above the league average for that position, red = below. The number is "
                             "the actual starter TPE."},
    )
    f = ctx.features
    seasons = sorted(f["season"].unique(), reverse=True)
    with st.sidebar:
        st.markdown("### League TPE")
        season = st.selectbox("Season", seasons, index=0, key="lt_season")
        metric = st.radio("Metric", ["Starter TPE", "Mean TPE"], horizontal=True)
    col = "starters_mean" if metric == "Starter TPE" else "tpe_mean"
    cur = f[f["season"] == season].merge(ctx.ratings[["season", "franchise", "wins", "srs", "elo_end", "playoff_result"]],
                                          on=["season", "franchise"], how="left")
    cur["conference"] = cur["franchise"].map(ctx.conferences)
    if season == ctx.current_season:
        cur["elo_end"] = cur["franchise"].map(ctx.elo.current)

    ui.section(f"S{season} · {metric} by team",
               "Starter TPE is the base-formation top-N per group with OL bots; mean TPE is every listed player.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(charts.ranked_bar(cur, "franchise", col, ctx.colors, "", fmt="{:,.0f}"),
                        width="stretch")
    with c2:
        st.plotly_chart(charts.scatter(cur, "age_mean", col, "franchise", ctx.colors,
                                       "TPE against age", x_label="Mean career season (older →)",
                                       y_label=metric, hover=["wins", "playoff_result"], size="elo_end",
                                       height=460), width="stretch")
        st.caption("Marker size = Elo. Top-left is young and strong; bottom-right is old and fading.")

    ui.section("Position groups", "Starter TPE per group. Colour is relative within the column.")
    mat = cur.set_index("franchise")[[f"{g}_starters" for g in config.POSITION_GROUPS]]
    mat.columns = config.POSITION_GROUPS
    mat = mat.sort_index()
    z = (mat - mat.mean()) / mat.std().replace(0, 1)
    fig = charts.heatmap(z, "", fmt="%{text:.1f}", zmid=0, colorscale=charts.diverging_scale())
    fig.update_traces(text=mat.to_numpy(), texttemplate="%{text:.0f}")
    st.plotly_chart(fig, width="stretch")

    ui.section("Timeline", "Every team's mean TPE since the portal migration.")
    highlight = st.multiselect("Highlight", ctx.cache.teams, default=[config.HOME_TEAM], key="lt_hl")
    st.plotly_chart(charts.lines(f, "season", col, "franchise", ctx.colors, "", y_label=metric,
                                 highlight=highlight or None, markers=True), width="stretch")

    ui.section("League-wide", "Distribution of TPE for the players on ISFL rosters.")
    fr = ctx.full_roster[ctx.full_roster["season"] == season]
    st.plotly_chart(charts.strip(fr, "franchise", "tpe", ctx.colors, "", hover=["name", "position"]),
                    width="stretch")
    ui.download(cur, f"league_tpe_S{season}.csv")
