"""Projection: where every org is heading over the next few seasons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts
import config
import projection as projection_mod
import roster as roster_mod
import ui
from page_teams import _needs_logs


def run_projection(ctx, horizon: int, hazard_scale: float, window: int):
    cur = ctx.full_roster[ctx.full_roster["season"] == ctx.current_season]
    rates = roster_mod.earning_rates(ctx.timeline, ctx.current_season, window=window)
    return projection_mod.project(cur, rates, ctx.league_rate, ctx.hazard, ctx.current_season,
                                  horizon=horizon, season_left=ctx.season_left, hazard_scale=hazard_scale)


def render(ctx) -> None:
    if _needs_logs(ctx):
        return
    ui.explain(
        "Where is each team heading? For every player on a current roster we add what they usually earn per "
        "season, take away the regression they're due, retire them when the rules say so, and roll a dice for "
        "voluntary retirement based on how often players of that age actually quit. Do that a few hundred times "
        "and you get a range for each team's TPE next season, the season after, and so on. Draft picks are "
        "**not** added, so this is 'what happens to the current group', not a full forecast.",
        {"Median / p10 / p90": "The middle outcome, and the pessimistic (10th percentile) and optimistic (90th) ones. "
                               "A wide band means a lot depends on who retires.",
         "Δ": "Change in mean TPE from now to the end of the horizon. Negative for teams with many veterans.",
         "Projected roster strength": "The projected rosters scored by the What-wins model, in points per game. "
                                      "Ignores voluntary retirement, so it's the optimistic path.",
         "Seasons used for earning rate": "How many past seasons to average when estimating what a player earns. "
                                          "Short reacts to someone going quiet; long is steadier.",
         "Retirement hazard ×": "Multiply the measured retirement chances. 0 = only rule-forced retirements; "
                                "2 = twice as many quit.",
         "Regression": ui.GLOSSARY["Regression"], "Hazard": ui.GLOSSARY["Hazard"]},
    )
    with st.sidebar:
        st.markdown("### Projection")
        horizon = st.slider("Seasons ahead", 1, config.MAX_HORIZON, config.DEFAULT_HORIZON)
        window = st.slider("Seasons used for earning rate", 1, 5, config.DEFAULT_RATE_WINDOW,
                           help="Short reacts to a change in habits; long is steadier.")
        hazard_scale = st.slider("Retirement hazard ×", 0.0, 2.0, 1.0, 0.1,
                                 help="1.0 = the empirical rate by career season. 0 = only rule-forced retirements.")
        team = st.selectbox("Focus team", ctx.cache.teams,
                            index=ctx.cache.teams.index(config.HOME_TEAM) if config.HOME_TEAM in ctx.cache.teams else 0,
                            key="proj_team")

    res = run_projection(ctx, horizon, hazard_scale, window)
    cur_feat = ctx.features[ctx.features["season"] == ctx.current_season].set_index("franchise")

    ui.section("Method", "Each player: add their measured earning rate (league median for their career season if "
               "unmeasured, zero if inactive), apply the regression table, retire on the rule (13 seasons or "
               f"under {config.REGRESSION_FLOOR_TPE} TPE), then draw voluntary retirement from the empirical hazard. "
               f"{res.n_sims} draws. Rates are held fixed; no draft picks are added — a draftee needs seasons to matter.")

    # League view: change in mean TPE over the horizon.
    now = cur_feat["tpe_mean"].rename("now")
    end = res.teams[res.teams["season"] == ctx.current_season + horizon].set_index("franchise")
    league = pd.DataFrame({"now": now, "p50": end["tpe_mean_p50"], "p10": end["tpe_mean_p10"],
                           "p90": end["tpe_mean_p90"], "players": end["n_players_p50"]}).dropna()
    league["change"] = league["p50"] - league["now"]
    league = league.reset_index().rename(columns={"index": "franchise"})
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(charts.signed_bar(league, "franchise", "change",
                                          f"Change in mean TPE by S{ctx.current_season + horizon} (median)",
                                          fmt="{:+.0f}"), width="stretch")
    with c2:
        ui.table(league.sort_values("p50", ascending=False).rename(columns={
            "franchise": "Team", "now": "Now", "p50": "Median", "p10": "p10", "p90": "p90",
            "players": "Players", "change": "Δ"}),
            formats={"Now": "%.0f", "Median": "%.0f", "p10": "%.0f", "p90": "%.0f", "Players": "%.0f", "Δ": "%+.0f"})

    ui.section(f"{team} · fan chart", "Median with the 10th–90th band from retirement draws. Rule-forced retirements are certain.")
    t = res.teams[res.teams["franchise"] == team].copy()
    t0 = pd.DataFrame([{"season": ctx.current_season, "tpe_mean_p10": now.get(team), "tpe_mean_p50": now.get(team),
                        "tpe_mean_p90": now.get(team)}])
    t = pd.concat([t0, t], ignore_index=True)
    fig = charts.band(t, "season", "tpe_mean_p10", "tpe_mean_p50", "tpe_mean_p90", ctx.team_color(team), team)
    fig.update_layout(title="Mean TPE", height=380, yaxis_title="TPE")
    st.plotly_chart(fig, width="stretch")

    ui.section("Every team, median path")
    all_paths = res.teams[["franchise", "season", "tpe_mean_p50"]].copy()
    start = now.reset_index().rename(columns={"index": "franchise", "now": "tpe_mean_p50"})
    start["season"] = ctx.current_season
    all_paths = pd.concat([start, all_paths], ignore_index=True)
    st.plotly_chart(charts.lines(all_paths, "season", "tpe_mean_p50", "franchise", ctx.colors, "",
                                 y_label="Mean TPE (median)", highlight=[team], markers=True),
                    width="stretch")

    if ctx.roster_model is not None:
        ui.section("Projected roster strength", "The deterministic path (no voluntary retirements) scored by the "
                   "roster model. Points per game above average, before any draft or free agency.")
        pf = projection_mod.projected_features(res, ctx.bots, ctx.current_season)
        if not pf.empty:
            pf["strength"] = ctx.roster_model.predict(pf).values
            base = cur_feat.reset_index()
            base["strength"] = ctx.roster_model.predict(base).values
            allf = pd.concat([base[["franchise", "season", "strength"]], pf[["franchise", "season", "strength"]]])
            st.plotly_chart(charts.lines(allf, "season", "strength", "franchise", ctx.colors, "",
                                         y_label="Predicted SRS", highlight=[team], hline=0.0, markers=True),
                            width="stretch")

    ui.section(f"{team} · player paths")
    pp = res.players[res.players["franchise"] == team]
    wide = pp.pivot_table(index=["name", "pos_group"], columns="season", values="tpe").reset_index()
    alive = pp.pivot_table(index=["name", "pos_group"], columns="season", values="p_alive").reset_index()
    cur_r = ctx.full_roster[(ctx.full_roster["season"] == ctx.current_season) & (ctx.full_roster["franchise"] == team)]
    wide = wide.merge(cur_r[["name", "tpe", "career_season", "active"]], on="name", how="left")
    cols = ["name", "pos_group", "career_season", "active", "tpe"] + [c for c in wide.columns if isinstance(c, (int, float, np.integer, np.floating))]
    wide = wide[cols].rename(columns={"name": "Player", "pos_group": "Grp", "career_season": "Career S",
                                      "active": "Active", "tpe": f"S{ctx.current_season}"})
    wide.columns = [f"S{c}" if isinstance(c, (int, float, np.integer, np.floating)) else c for c in wide.columns]
    ui.table(wide.sort_values(f"S{ctx.current_season}", ascending=False),
             formats={c: "%.0f" for c in wide.columns if c.startswith("S")})
    st.caption("A zero means the rule retires the player before that season. P(still here) per season:")
    alive.columns = [f"S{c}" if isinstance(c, (int, float, np.integer, np.floating)) else c for c in alive.columns]
    ui.table(alive.rename(columns={"name": "Player", "pos_group": "Grp"}),
             formats={c: "%.0%" for c in alive.columns if c.startswith("S")})
