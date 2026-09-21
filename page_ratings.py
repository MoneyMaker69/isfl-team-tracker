"""Ratings: Elo history, SRS, Pythagorean, calibration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import elo as elo_mod
import ui


def render(ctx) -> None:
    r = ctx.ratings
    teams = ctx.cache.teams
    ui.explain(
        "Three different ways of rating how good each team has been, going back to season 1. Elo cares about "
        "who you beat, SRS cares about by how much, and Pythagorean is about whether your record matched your "
        "scoring. The calibration tab checks that when Elo says '70% to win', it really happens about 70% of the time.",
        {k: ui.GLOSSARY[k] for k in ("Elo", "SRS", "Pythagorean wins", "Luck")}
        | {"Δ Elo": "How much the rating moved during that season. Big positive = a breakout year.",
           "Tuning grid": "Tries different Elo settings and reports which predicted past games best. "
                          "Lower log-loss is better; 0.693 is a coin flip."},
    )

    with st.sidebar:
        st.markdown("### Ratings")
        from_season = st.slider("From season", 1, int(r["season"].max()), config.ENGINE_SWITCH_SEASON)
        highlight = st.multiselect("Highlight", teams, default=[config.HOME_TEAM])

    ui.section("Elo by season", f"K={ctx.elo.k:.0f}, home advantage {ctx.elo.home_adv:.0f}, "
               f"{ctx.elo.revert:.0%} reversion each offseason — tuned by log-loss on S{config.ENGINE_SWITCH_SEASON}+. "
               "Relocated franchises carry their rating (LVL→NOLA, PHI→CTC, CHI→OSK, BER→BFB).")
    hist = ctx.elo.history[(ctx.elo.history["season"] >= from_season)
                           & ctx.elo.history["franchise"].isin(teams)]
    st.plotly_chart(charts.lines(hist, "season", "elo_end", "franchise", ctx.colors, "",
                                 y_label="Elo (end of season)", highlight=highlight or None,
                                 hline=config.ELO_START),
                    width="stretch")

    tabs = st.tabs(["Season table", "SRS", "Pythagorean luck", "Calibration", "Tuning grid"])

    with tabs[0]:
        season = st.selectbox("Season", sorted(r["season"].unique(), reverse=True), key="rat_season")
        tbl = r[r["season"] == season].sort_values("elo_end", ascending=False)
        show = tbl[["franchise", "conference", "wins", "losses", "margin", "elo_start", "elo_end", "change",
                    "srs", "pyth_wins", "luck", "playoff_result"]].rename(columns={
            "franchise": "Team", "conference": "Conf", "wins": "W", "losses": "L", "margin": "Margin",
            "elo_start": "Elo start", "elo_end": "Elo end", "change": "Δ Elo", "srs": "SRS",
            "pyth_wins": "Pyth W", "luck": "Luck", "playoff_result": "Playoffs"})
        ui.table(show, formats={"Margin": "%+.1f", "Elo start": "%.0f", "Elo end": "%.0f",
                                "Δ Elo": "%+.0f", "SRS": "%+.1f", "Pyth W": "%.1f", "Luck": "%+.1f"})
        ui.download(show, f"ratings_S{season}.csv")

    with tabs[1]:
        ui.section("SRS — schedule-adjusted margin",
                   "Points per game above an average opponent, solved from season PF/PA and the opponent list. "
                   "Zero is league average.")
        srs = ctx.srs[(ctx.srs["season"] >= from_season) & ctx.srs["franchise"].isin(teams)]
        st.plotly_chart(charts.lines(srs, "season", "srs", "franchise", ctx.colors, "",
                                     y_label="SRS (points/game)", highlight=highlight or None, hline=0.0),
                        width="stretch")

    with tabs[2]:
        ui.section("Pythagorean expectation",
                   f"Expected win% = PF^k / (PF^k + PA^k), k fitted to {ctx.pyth_exponent:.2f} on S{config.ENGINE_SWITCH_SEASON}+. "
                   "'Luck' is actual wins minus Pythagorean wins. Big positive luck one season predicts a fall the next.")
        lucky = r[r["season"] >= from_season]
        nxt = lucky[["season", "franchise", "wins"]].copy()
        nxt["season"] -= 1
        j = lucky.merge(nxt.rename(columns={"wins": "next_wins"}), on=["season", "franchise"], how="inner")
        j["delta_next"] = j["next_wins"] - j["wins"]
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.scatter(j, "luck", "delta_next", "franchise", ctx.colors,
                                           "Luck this season vs change in wins next season",
                                           x_label="Wins above Pythagorean", y_label="Δ wins next season",
                                           trend=True, height=460), width="stretch")
        with c2:
            corr = j["luck"].corr(j["delta_next"])
            ui.insight("Correlation, luck → next-season change", f"{corr:+.2f} over {len(j)} team-seasons")
            top = r[r["season"] == r["season"].max()].sort_values("luck", ascending=False)
            st.plotly_chart(charts.signed_bar(top, "franchise", "luck", f"S{int(r['season'].max())} luck"),
                            width="stretch")

    with tabs[3]:
        ui.section("Calibration", "When Elo said the home team had a 70% chance, how often did it win?")
        cal = elo_mod.calibration(ctx.elo.game_log)
        st.plotly_chart(charts.calibration_chart(cal, ""), width="stretch")
        ui.insight("Regular-season log-loss (S27+)", f"{elo_mod.log_loss(ctx.elo.game_log):.4f} "
                   "(0.693 is a coin flip)")

    with tabs[4]:
        st.caption("Grid search over K, home advantage and offseason reversion. Cached per session; takes ~30 s.")
        if st.button("Run tuning grid"):
            grid = elo_mod.tune_elo(ctx.cache.games)
            st.session_state["elo_grid"] = grid
        grid = st.session_state.get("elo_grid")
        if grid is not None:
            ui.table(grid.head(15), formats={"log_loss": "%.4f"})
