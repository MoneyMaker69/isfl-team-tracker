"""Teams: one franchise's roster, TPE, age and position groups."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import ui


def _needs_logs(ctx) -> bool:
    if ctx.has_logs:
        return False
    ui.empty_state("Player logs not built yet",
                   "This page needs the slow stage of the cache builder: "
                   "`python scripts/build_cache.py --stage slow`.")
    return True


def render(ctx) -> None:
    if _needs_logs(ctx):
        return
    fr = ctx.full_roster
    seasons = sorted(fr["season"].unique(), reverse=True)
    ui.explain(
        "One team's roster in one season: who is on it, how much TPE they have, how old they are in career "
        "terms, and who is about to lose TPE to regression. Pick the team and season in the sidebar. For past "
        "seasons the roster is whoever recorded a game for the team, and TPE is what they had at Week 1.",
        {k: ui.GLOSSARY[k] for k in ("TPE", "Starter TPE", "Career season", "Regression", "Inactive (IA)")}
        | {"Next regr. / Regr. TPE": "The percentage and amount this player loses at the start of next season "
                                     "if they stay.",
           "Applied / Banked": "TPE spent on attributes vs sitting unspent. Only applied TPE plays in the sim.",
           "Events": "Number of TPE-earning tasks logged that season. Under 6 means effectively inactive."},
    )
    with st.sidebar:
        st.markdown("### Team")
        team = st.selectbox("Franchise", ctx.cache.teams,
                            index=ctx.cache.teams.index(config.HOME_TEAM) if config.HOME_TEAM in ctx.cache.teams else 0)
        season = st.selectbox("Season", seasons, index=0)

    roster = fr[(fr["franchise"] == team) & (fr["season"] == season)].copy()
    bots = ctx.bots[(ctx.bots["franchise"] == team) & (ctx.bots["season"] == season)]
    feats = ctx.features[ctx.features["season"] == season].set_index("franchise")
    me = feats.loc[team] if team in feats.index else None
    is_current = season == ctx.current_season

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Players", len(roster))
    c2.metric("Mean TPE", ui.num(roster["tpe"].mean()))
    c3.metric("Starter TPE", ui.num(me["starters_mean"]) if me is not None else "—",
              help="Mean TPE of the base-formation starters (top-N per group, OL incl. bots).")
    c4.metric("Mean career season", ui.num(roster["career_season"].mean(), 1))
    c5.metric("Inactive", int((~roster["active"]).sum()))

    tab_roster, tab_groups, tab_age, tab_history = st.tabs(["Roster", "Position groups", "Age & regression", "History"])

    with tab_roster:
        tpe_note = ("Live total TPE, including this preseason's earnings."
                    if is_current else "TPE at Week 1 of that season (regression already applied).")
        ui.section(f"{team} · S{season}", tpe_note + " Players with no recorded game that season are not listed.")
        cols = ["name", "position", "archetype", "career_season", "tpe"]
        if is_current:
            cols += ["applied_tpe", "banked_tpe"]
        cols += ["next_regression_pct", "next_regression_tpe", "active", "n_events"]
        show = roster.sort_values("tpe", ascending=False)[cols].rename(columns={
            "name": "Player", "position": "Position", "archetype": "Archetype", "career_season": "Career S",
            "tpe": "TPE", "applied_tpe": "Applied", "banked_tpe": "Banked",
            "next_regression_pct": "Next regr.", "next_regression_tpe": "Regr. TPE", "active": "Active",
            "n_events": "Events"})
        ui.table(show, formats={"TPE": "%.0f", "Applied": "%.0f", "Banked": "%.0f", "Next regr.": "%.0%",
                                "Regr. TPE": "%.0f"}, height=min(800, 38 * len(show) + 40))
        if not bots.empty:
            st.caption("OL bots on this roster: " + ", ".join(f"{b.name} ({b.tpe:.0f})" for b in bots.itertuples()))
        ui.download(show, f"{team}_S{season}_roster.csv")

    with tab_groups:
        ui.section("Starter TPE by position group vs league",
                   "Top-N players per group by TPE (QB1, RB1, WR3, TE1, OL5 with bots, DL4, LB3, DB5, K1). "
                   "Short groups are padded with zeros — a missing starter should hurt.")
        rows = []
        for g in config.POSITION_GROUPS:
            col = f"{g}_starters"
            rows.append({"Group": g, team: float(me[col]) if me is not None else 0.0,
                         "League mean": float(feats[col].mean()) if col in feats else 0.0,
                         "League best": float(feats[col].max()) if col in feats else 0.0})
        gt = pd.DataFrame(rows)
        st.plotly_chart(charts.grouped_bars(gt, "Group", [team, "League mean", "League best"], "",
                                            colors=[ctx.team_color(team), config.COLORS["text_muted"], config.COLORS["blue"]],
                                            y_label="Starter TPE"), width="stretch")
        gt["Gap to mean"] = gt[team] - gt["League mean"]
        ui.table(gt, formats={team: "%.0f", "League mean": "%.0f", "League best": "%.0f", "Gap to mean": "%+.0f"})

    with tab_age:
        ui.section("TPE against career season",
                   "Regression starts after the 7th season (20%) and steepens to 60% after the 12th. "
                   "Players to the right of the line are about to lose a chunk.")
        fig = charts.scatter(roster.dropna(subset=["career_season"]), "career_season", "tpe", "name",
                             {n: ctx.team_color(team) for n in roster["name"]}, "",
                             x_label="Career season", y_label="TPE", crosshairs=False, height=520,
                             hover=["position", "next_regression_pct"])
        fig.add_vline(x=config.FIRST_REGRESSION_CAREER_SEASON - 0.5, line_dash="dot",
                      line_color=config.COLORS["red"], annotation_text="regression begins")
        st.plotly_chart(fig, width="stretch")
        upcoming = roster[roster["next_regression_pct"] > 0].sort_values("next_regression_tpe", ascending=False)
        if not upcoming.empty:
            ui.insight(f"TPE lost at the start of S{season + 1} (if nobody retires)",
                       f"{upcoming['next_regression_tpe'].sum():,.0f} across {len(upcoming)} players")
            ui.table(upcoming[["name", "position", "career_season", "tpe", "next_regression_pct", "next_regression_tpe"]]
                     .rename(columns={"name": "Player", "position": "Position", "career_season": "Career S",
                                      "tpe": "TPE", "next_regression_pct": "Regr. %", "next_regression_tpe": "Loss"}),
                     formats={"TPE": "%.0f", "Regr. %": "%.0%", "Loss": "%.0f"})
        else:
            st.caption("Nobody on this roster regresses next offseason.")

    with tab_history:
        ui.section("Team TPE and rating by season")
        hist = ctx.features[ctx.features["franchise"] == team].merge(
            ctx.ratings[["season", "franchise", "wins", "srs", "elo_end", "playoff_result"]],
            on=["season", "franchise"], how="left")
        if hist.empty:
            st.caption("No history yet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                long = hist.melt(id_vars="season", value_vars=["tpe_mean", "starters_mean"], var_name="series")
                st.plotly_chart(charts.lines(long, "season", "value", "series",
                                             {"tpe_mean": ctx.team_color(team), "starters_mean": config.COLORS["blue"]},
                                             "Mean and starter TPE", y_label="TPE", markers=True, height=380),
                                width="stretch")
            with c2:
                st.plotly_chart(charts.lines(hist.assign(series="SRS"), "season", "srs", "series",
                                             {"SRS": ctx.team_color(team)}, "SRS", y_label="points/game",
                                             hline=0.0, markers=True, height=380), width="stretch")
            ui.table(hist[["season", "n_players", "tpe_mean", "starters_mean", "age_mean", "n_inactive",
                           "wins", "srs", "elo_end", "playoff_result"]].sort_values("season", ascending=False)
                     .rename(columns={"season": "Season", "n_players": "Players", "tpe_mean": "Mean TPE",
                                      "starters_mean": "Starter TPE", "age_mean": "Mean career S",
                                      "n_inactive": "IA", "wins": "W", "srs": "SRS", "elo_end": "Elo",
                                      "playoff_result": "Playoffs"}),
                     formats={"Mean TPE": "%.0f", "Starter TPE": "%.0f", "Mean career S": "%.1f",
                              "SRS": "%+.1f", "Elo": "%.0f"})
