"""GMs: Elo, over-performance, tenure."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import gm as gm_mod
import ui


def render(ctx) -> None:
    ui.explain(
        "Two ways of rating GMs. **GM Elo** simply gives each GM credit for how their team's rating moved while "
        "they were in charge. **Over-performance** is smarter: it asks whether the team did better or worse than "
        "its TPE said it should — which is the part a GM actually controls (depth charts, strategy, signings). "
        "Every team has two GMs, so credit is split and the numbers are noisy; treat them as a conversation "
        "starter, not a verdict.",
        {"GM Elo": "1500 is average. Each season the team's Elo change is split between its GMs.",
         "Pts/game vs roster": ui.GLOSSARY["GM factor"],
         "Tenure": "Who ran the team when, with the team's Elo on top. Dotted lines mark a change of GMs.",
         "Career records": "Straight from the portal: wins, losses, playoff record and titles."},
    )
    tabs = st.tabs(["GM Elo", "Over-performance", "Tenure", "Career records"])

    with tabs[0]:
        ui.section("GM Elo", "Each ISFL team-season's Elo change, split between its GMs and accumulated "
                   "with the same offseason reversion as team Elo. Includes every season since S1.")
        cur = ctx.gm_current
        if cur.empty:
            st.caption("No GM history.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                min_seasons = st.slider("Minimum seasons", 1, 20, 4)
                show = cur[cur["seasons"] >= min_seasons].head(25)
                st.plotly_chart(charts.ranked_bar(show, "username", "gm_elo", {}, "", fmt="{:.0f}"),
                                width="stretch")
            with c2:
                active_uids = set(gm_mod.isfl_gm_seasons(ctx.cache.gm_history)
                                  .query("season == @ctx.current_season")["uid"])
                cur2 = cur.copy()
                cur2["current"] = cur2["uid"].isin(active_uids)
                ui.table(cur2[["username", "gm_elo", "seasons", "first_season", "last_season", "teams", "current"]]
                         .rename(columns={"username": "GM", "gm_elo": "GM Elo", "seasons": "Seasons",
                                          "first_season": "First", "last_season": "Last", "teams": "Teams",
                                          "current": "Active"}),
                         formats={"GM Elo": "%.0f"}, height=600)

    with tabs[1]:
        ui.section("Over-performance", "Mean of actual minus roster-predicted SRS across a GM's seasons "
                   f"from S{config.FIRST_TPE_SEASON}. Positive means their teams beat their TPE.")
        op = ctx.gm_overperf
        if op.empty:
            st.caption("Needs the roster model (player logs).")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(charts.signed_bar(op[op["seasons"] >= 2].head(30), "username", "overperf", "",
                                                  fmt="{:+.2f}"), width="stretch")
            with c2:
                ui.table(op.rename(columns={"username": "GM", "overperf": "Pts/game vs roster",
                                            "seasons": "Seasons", "last_season": "Last", "teams": "Teams"}),
                         formats={"Pts/game vs roster": "%+.2f"}, height=600)
            cur_factor = gm_mod.team_gm_factor(ctx.cache.gm_history, op, ctx.current_season)
            if not cur_factor.empty:
                ui.section(f"S{ctx.current_season} GM factor by team",
                           "Current GMs' over-performance, shrunk toward zero for short track records. "
                           "This is the `gm_factor` the predictor uses.")
                cf = cur_factor.rename("factor").reset_index().rename(columns={"index": "franchise"})
                st.plotly_chart(charts.signed_bar(cf, "franchise", "factor", "", fmt="{:+.2f}"),
                                width="stretch")

    with tabs[2]:
        ui.section("Tenure", "Who ran each team, season by season, with the team's Elo.")
        team = st.selectbox("Franchise", ctx.cache.teams,
                            index=ctx.cache.teams.index(config.HOME_TEAM) if config.HOME_TEAM in ctx.cache.teams else 0,
                            key="gm_team")
        ten = gm_mod.tenure_table(ctx.cache.gm_history, ctx.elo.history)
        ten = ten[ten["franchise"] == team].sort_values("season")
        if ten.empty:
            st.caption("No GM history for this franchise.")
        else:
            fig = charts.lines(ten.assign(series=team), "season", "elo_end", "series",
                               {team: ctx.team_color(team)}, "", y_label="Elo", hline=config.ELO_START,
                               markers=True, height=380)
            # Mark GM changes.
            prev = None
            for r in ten.itertuples():
                if r.username != prev:
                    fig.add_vline(x=r.season - 0.5, line_dash="dot", line_color=config.COLORS["line"])
                    prev = r.username
            st.plotly_chart(fig, width="stretch")
            ui.table(ten[["season", "username", "elo_end", "change"]].sort_values("season", ascending=False)
                     .rename(columns={"season": "Season", "username": "GMs", "elo_end": "Elo", "change": "Δ Elo"}),
                     formats={"Elo": "%.0f", "Δ Elo": "%+.0f"})

    with tabs[3]:
        ui.section("Career records", "From the portal's GM records: regular season, playoffs, championships.")
        rec = ctx.cache.gm_records
        if rec.empty:
            st.caption("No records.")
        else:
            rec = rec[rec["league"] == "ISFL"].copy() if "league" in rec.columns else rec
            rec["reg_pct"] = rec["regWins"] / rec["regGames"].replace(0, pd.NA)
            ui.table(rec.sort_values("championships", ascending=False)[
                ["username", "seasons", "regWins", "regLosses", "reg_pct", "playoffWins", "playoffLosses", "championships"]]
                     .rename(columns={"username": "GM", "seasons": "Seasons", "regWins": "W", "regLosses": "L",
                                      "reg_pct": "Win%", "playoffWins": "PO W", "playoffLosses": "PO L",
                                      "championships": "Ultimus"}),
                     formats={"Win%": "%.3f"}, height=600)
