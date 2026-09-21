"""Overview: where the league stands right now."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import ui


def render(ctx) -> None:
    cache = ctx.cache
    current = ctx.current_season
    ui.explain(
        "The league at a glance: last season's standings, every team's current rating, and which teams "
        "were lucky or unlucky. Use the sidebar to open the other pages — **Predict** is the one for the "
        "prediction task, **NOLA** is the home-team page.",
        {k: ui.GLOSSARY[k] for k in ("Elo", "SRS", "Pythagorean wins", "Luck")},
    )
    r = ctx.ratings
    latest_done = int(r["season"].max()) if not r.empty else current - 1

    # Current season standings if any games have been played, else last season.
    cur = cache.standings[cache.standings["season"] == current]
    played = cache.games[(cache.games["season"] == current) & (cache.games["phase"] == "regular")]
    if not cur.empty and not played.empty:
        show_season, standings_note = current, f"S{current} in progress · {len(played)} games played"
    else:
        show_season, standings_note = latest_done, f"S{latest_done} final · S{current} not started"

    c1, c2, c3, c4 = st.columns(4)
    top = r[r["season"] == latest_done].sort_values("elo_end", ascending=False)
    champ = r[(r["season"] == latest_done) & r["champion"]]
    with c1:
        st.metric("Season", f"S{current}")
    with c2:
        st.metric(f"S{latest_done} Ultimus", champ["franchise"].iloc[0] if not champ.empty else "—")
    with c3:
        st.metric("Elo #1", f"{top['franchise'].iloc[0]} · {top['elo_end'].iloc[0]:.0f}" if not top.empty else "—")
    with c4:
        home = ctx.elo.current.get(config.HOME_TEAM)
        rank = int((pd.Series(ctx.elo.current) > home).sum()) + 1 if home else None
        st.metric(f"{config.HOME_TEAM} Elo", f"{home:.0f}" if home else "—", f"#{rank} of {len(ctx.elo.current)}" if rank else None)

    left, right = st.columns([3, 2])
    with left:
        ui.section(f"Standings", standings_note)
        st_tbl = cache.standings[cache.standings["season"] == show_season].copy()
        st_tbl = st_tbl.merge(r[["season", "franchise", "elo_end", "srs", "pyth_wins", "luck"]],
                              on=["season", "franchise"], how="left")
        for conf, grp in st_tbl.groupby("conference"):
            st.markdown(f"**{conf}**")
            grp = grp.sort_values(["win_pct", "margin"], ascending=False)
            show = grp[["team", "wins", "losses", "pf", "pa", "margin", "elo_end", "srs", "playoff_result"]]
            show = show.rename(columns={"team": "Team", "wins": "W", "losses": "L", "pf": "PF", "pa": "PA",
                                        "margin": "Margin", "elo_end": "Elo", "srs": "SRS",
                                        "playoff_result": "Playoffs"})
            ui.table(show, formats={"Margin": "%+.1f", "Elo": "%.0f", "SRS": "%+.1f"})

    with right:
        ui.section("Elo right now", "After the last game played; offseason reversion not yet applied.")
        cur_elo = pd.DataFrame({"franchise": list(ctx.elo.current), "elo": list(ctx.elo.current.values())})
        cur_elo = cur_elo[cur_elo["franchise"].isin(cache.teams)]
        st.plotly_chart(charts.ranked_bar(cur_elo, "franchise", "elo", ctx.colors, "", fmt="{:.0f}"),
                        width="stretch")

    ui.section(f"S{latest_done} in one chart", "Points margin per game against Pythagorean luck. Teams high on the right were good; teams far above zero were lucky and tend to fall back.")
    last = r[r["season"] == latest_done]
    st.plotly_chart(charts.scatter(last, "margin", "luck", "franchise", ctx.colors, "",
                                   x_label="Point margin per game", y_label="Wins above Pythagorean",
                                   hover=["wins", "playoff_result"], height=460),
                    width="stretch")

    if not ctx.has_logs:
        ui.render_notices(cache.notices, only={"info", "warning"})
