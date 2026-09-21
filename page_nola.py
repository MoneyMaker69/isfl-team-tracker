"""The home-team page: everything filtered to NOLA and the ASFC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import backtest as backtest_mod
import charts
import config
import gm as gm_mod
import model as model_mod
import projection as projection_mod
import simulate
import ui

TEAM = config.HOME_TEAM


def render(ctx) -> None:
    season = ctx.current_season
    conf = ctx.conferences.get(TEAM, "ASFC")
    rivals = sorted(t for t, c in ctx.conferences.items() if c == conf and t in ctx.cache.teams)
    r = ctx.ratings
    latest = int(r["season"].max())
    ui.explain(
        f"Everything on the other pages, filtered to {TEAM} and the {conf}. **Prediction** is our expected "
        "finish; **Roster & regression** shows who loses how much TPE when; **Rivals** puts us next to the "
        "conference; **Trajectory** projects the group forward; **What would it take** turns the What-wins model "
        "into 'if we improved the OL by 100 TPE, how many wins is that?'; **Picks & history** is the archive.",
        {"Regression calendar": "Each column is a coming season; the percentage is what that player loses at its "
                                "start if they stay. 100% = forced retirement.",
         "Gap": "Difference in starter TPE between us and the conference leader in that group.",
         "Pts if matched": "Points per game the model says we'd gain by closing that gap.",
         "Strength": ui.GLOSSARY["Strength"], "Starter TPE": ui.GLOSSARY["Starter TPE"]},
    )

    # ---- Header metrics ---------------------------------------------------
    last = r[(r["season"] == latest) & (r["franchise"] == TEAM)]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Elo now", ui.num(ctx.elo.current.get(TEAM, np.nan)))
    c2.metric(f"S{latest} record", f"{int(last['wins'].iloc[0])}–{int(last['losses'].iloc[0])}" if not last.empty else "—",
              last["playoff_result"].iloc[0] if not last.empty else None)
    c3.metric(f"S{latest} SRS", ui.signed(last["srs"].iloc[0]) if not last.empty else "—")
    titles = int(r[(r["franchise"] == TEAM) & r["champion"]].shape[0])
    c4.metric("Ultimus titles", titles)
    gms = gm_mod.isfl_gm_seasons(ctx.cache.gm_history)
    cur_gms = gms[(gms["season"] == season) & (gms["franchise"] == TEAM)]["username"].tolist()
    c5.metric("GMs", " / ".join(cur_gms) if cur_gms else "—")

    tabs = st.tabs(["Prediction", "Roster & regression", conf + " rivals", "Trajectory", "What would it take",
                    "Picks & history"])

    # ---- Prediction -------------------------------------------------------
    with tabs[0]:
        pre = ctx.preseason.copy()
        if pre.empty:
            st.caption("No preseason table.")
        else:
            strength = ctx.blend.predict(pre) if ctx.blend is not None else backtest_mod.default_strength(pre)
            pre["strength"] = strength.values
            wm = ctx.win_model or simulate.WinModel(scale=10.0, hfa=config.SRS_HOME_ADVANTAGE)
            rng = np.random.default_rng(config.SIM_SEED)
            played = ctx.cache.games[(ctx.cache.games["season"] == season) & (ctx.cache.games["phase"] == "regular")]
            sched = simulate.complete_schedule(played, {t: ctx.conferences[t] for t in pre["franchise"]}, rng)
            res = simulate.simulate(pre.set_index("franchise")["strength"], sched, ctx.conferences, wm,
                                    n_sims=5000)
            row = res.table.set_index("franchise").loc[TEAM]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Expected wins", f"{row['exp_wins']:.1f}", f"{row['wins_p10']:.0f}–{row['wins_p90']:.0f} (p10–p90)")
            c2.metric("Playoffs", ui.pct(row["p_playoffs"]))
            c3.metric(f"{conf} champion", ui.pct(row["p_conf_champ"]))
            c4.metric("Ultimus", ui.pct(row["p_ultimus"]))
            left, right = st.columns([1, 1])
            with left:
                order = simulate.ranking_for_task(res)[conf]
                st.markdown(f"**Predicted {conf} order**")
                for i, t in enumerate(order, 1):
                    rr = res.table.set_index("franchise").loc[t]
                    ui.rank_row(i, t, f"{rr['exp_wins']:.1f} W · strength {rr['strength']:+.1f}")
            with right:
                mat = res.rank_matrix.loc[[TEAM]].T.reset_index()
                mat.columns = ["Finish", "p"]
                fig = charts.ranked_bar(mat, "Finish", "p", {}, f"{TEAM} finish probabilities", fmt="{:.0%}", height=320)
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")
            inp = pre[pre["franchise"].isin(rivals)][["franchise", "elo_pts", "srs_prev", "wpct_prev", "roster_pred",
                                                      "gm_factor", "strength"]]
            ui.table(inp.sort_values("strength", ascending=False).rename(columns={
                "franchise": "Team", "elo_pts": "Elo (pts)", "srs_prev": "SRS last", "wpct_prev": "Record last",
                "roster_pred": "Roster", "gm_factor": "GM", "strength": "Strength"}),
                formats={c: "%+.1f" for c in ["Elo (pts)", "SRS last", "Record last", "Roster", "GM", "Strength"]})

    if not ctx.has_logs:
        for t in tabs[1:5]:
            with t:
                st.caption("Needs player logs (`build_cache.py --stage slow`).")
    else:
        fr = ctx.full_roster
        mine = fr[(fr["franchise"] == TEAM) & (fr["season"] == season)].copy()
        feats = ctx.features[ctx.features["season"] == season].set_index("franchise")

        # ---- Roster & regression ------------------------------------------
        with tabs[1]:
            ui.section("Regression calendar", "Percentage of TPE each player loses at the start of each coming season "
                       "if they stay. Rule-forced retirement after the 13th season shows as 100%.")
            cal = projection_mod.regression_calendar(mine, season, horizon=4)
            cols = [f"S{season + h}" for h in range(1, 5)]
            show = cal.sort_values("tpe", ascending=False)[["name", "position", "career_season", "tpe"] + cols]
            show = show.rename(columns={"name": "Player", "position": "Position", "career_season": "Career S", "tpe": "TPE"})
            ui.table(show, formats={"TPE": "%.0f", **{c: "%.0%" for c in cols}}, height=min(760, 38 * len(show) + 40))
            loss = {c: float((cal["tpe"].fillna(0) * cal[c]).sum()) for c in cols}
            ui.insight("TPE lost to regression by season (nobody retires, no earning)",
                       " · ".join(f"{c}: {v:,.0f}" for c, v in loss.items()))
            ia = mine[~mine["active"]]
            if not ia.empty:
                ui.insight("Inactive on the roster", ", ".join(f"{n} ({t:.0f})" for n, t in zip(ia["name"], ia["tpe"])))

        # ---- Rivals -----------------------------------------------------------
        with tabs[2]:
            ui.section(f"{conf} starter TPE by group", "Where NOLA is ahead and behind inside the conference.")
            mat = feats.loc[[t for t in rivals if t in feats.index], [f"{g}_starters" for g in config.POSITION_GROUPS]]
            mat.columns = config.POSITION_GROUPS
            z = (mat - mat.mean()) / mat.std().replace(0, 1)
            fig = charts.heatmap(z, "", fmt="%{text:.0f}", zmid=0, colorscale=charts.diverging_scale(), height=360)
            fig.update_traces(text=mat.to_numpy(), texttemplate="%{text:.0f}")
            st.plotly_chart(fig, width="stretch")
            summary = feats.loc[[t for t in rivals if t in feats.index], ["n_players", "tpe_mean", "starters_mean", "age_mean", "n_inactive", "qb1"]]
            summary = summary.merge(pd.Series(ctx.elo.current, name="elo"), left_index=True, right_index=True, how="left")
            ui.table(summary.sort_values("starters_mean", ascending=False).reset_index().rename(columns={
                "franchise": "Team", "n_players": "Players", "tpe_mean": "Mean TPE", "starters_mean": "Starter TPE",
                "age_mean": "Career S", "n_inactive": "IA", "qb1": "QB1", "elo": "Elo"}),
                formats={"Mean TPE": "%.0f", "Starter TPE": "%.0f", "Career S": "%.1f", "QB1": "%.0f", "Elo": "%.0f"})
            hist = ctx.features[ctx.features["franchise"].isin(rivals)]
            st.plotly_chart(charts.lines(hist, "season", "starters_mean", "franchise", ctx.colors,
                                         f"{conf} starter TPE over time", y_label="Starter TPE",
                                         highlight=[TEAM], markers=True, height=400), width="stretch")

        # ---- Trajectory -----------------------------------------------------
        with tabs[3]:
            ui.section("Three seasons out", "Median mean-TPE path with retirement draws, NOLA against the conference.")
            cur = fr[fr["season"] == season]
            res_p = projection_mod.project(cur, ctx.rates, ctx.league_rate, ctx.hazard, season,
                                           horizon=3, season_left=ctx.season_left)
            t = res_p.teams[res_p.teams["franchise"].isin(rivals)]
            start = feats.loc[[x for x in rivals if x in feats.index], "tpe_mean"].rename("tpe_mean_p50").reset_index()
            start["season"] = season
            allp = pd.concat([start, t[["franchise", "season", "tpe_mean_p50"]]], ignore_index=True)
            st.plotly_chart(charts.lines(allp, "season", "tpe_mean_p50", "franchise", ctx.colors, "",
                                         y_label="Mean TPE (median)", highlight=[TEAM], markers=True, height=420),
                            width="stretch")
            mine_t = res_p.teams[res_p.teams["franchise"] == TEAM]
            fig = charts.band(pd.concat([pd.DataFrame([{"season": season, "tpe_mean_p10": start.loc[start.franchise == TEAM, "tpe_mean_p50"].iloc[0],
                                                          "tpe_mean_p50": start.loc[start.franchise == TEAM, "tpe_mean_p50"].iloc[0],
                                                          "tpe_mean_p90": start.loc[start.franchise == TEAM, "tpe_mean_p50"].iloc[0]}]), mine_t]),
                              "season", "tpe_mean_p10", "tpe_mean_p50", "tpe_mean_p90", ctx.team_color(TEAM), TEAM)
            fig.update_layout(title=f"{TEAM} band", height=340, yaxis_title="Mean TPE")
            st.plotly_chart(fig, width="stretch")

        # ---- What would it take ---------------------------------------------
        with tabs[4]:
            ui.section("What would it take", "Using the roster model's coefficients: how many points per game "
                       "a starter-TPE change in one group is worth, and what it would take to match the conference leader.")
            m = ctx.roster_model
            if m is None or TEAM not in feats.index:
                st.caption("Needs the roster model.")
            else:
                pe = model_mod.partial_effects(m).set_index("group")["effect_ppg"]
                rows = []
                for g in config.POSITION_GROUPS:
                    col = f"{g}_starters"
                    mine_v = float(feats.loc[TEAM, col])
                    leader = feats.loc[[t for t in rivals if t in feats.index], col]
                    best_team, best_v = leader.idxmax(), float(leader.max())
                    gap = best_v - mine_v
                    rows.append({"Group": g, "NOLA": mine_v, "Leader": best_team, "Leader TPE": best_v,
                                 "Gap": gap, "Pts/game per +100": float(pe.get(g, 0.0)),
                                 "Pts if matched": float(pe.get(g, 0.0)) * gap / 100.0})
                tk = pd.DataFrame(rows).sort_values("Pts if matched", ascending=False)
                ui.table(tk, formats={"NOLA": "%.0f", "Leader TPE": "%.0f", "Gap": "%+.0f",
                                      "Pts/game per +100": "%+.2f", "Pts if matched": "%+.2f"})
                g = st.selectbox("Group", config.POSITION_GROUPS, index=0)
                delta = st.slider("Starter TPE change", -300, 300, 100, 10)
                n = config.STARTERS[g]
                ppg = float(pe.get(g, 0.0)) * delta / 100.0
                # Logistic slope at a coin-flip is 1/(4·scale) per point, times 16 games.
                scale = float(ctx.win_model.scale) if ctx.win_model else 10.0
                wins = 16 * ppg / (4 * scale)
                ui.insight(f"{delta:+d} starter TPE at {g} (about {delta * n:+d} TPE across {n} starters)",
                           f"≈ {ppg:+.2f} points per game ≈ {wins:+.1f} wins over 16 games against average opponents")

    # ---- Picks & history ----------------------------------------------------
    with tabs[5]:
        dp = ctx.cache.draft_picks
        if not dp.empty:
            upcoming = dp[(dp["owning_team"] == TEAM) & (dp["season"] >= season)]
            ui.section("Draft picks owned", "Picks the portal lists NOLA as owning for this and coming seasons.")
            if upcoming.empty:
                st.caption("None listed yet.")
            else:
                ui.table(upcoming[["season", "round", "pick", "original_team", "name", "position"]].sort_values(["season", "round", "pick"])
                         .rename(columns={"season": "Season", "round": "Rd", "pick": "Pick", "original_team": "From",
                                          "name": "Selected", "position": "Position"}))
            recent = dp[(dp["owning_team"] == TEAM) & (dp["season"] < season)].sort_values("season", ascending=False).head(20)
            ui.section("Recent draft classes")
            ui.table(recent[["season", "round", "pick", "overall", "name", "position", "username"]]
                     .rename(columns={"season": "Season", "round": "Rd", "pick": "Pick", "overall": "Overall",
                                      "name": "Player", "position": "Position", "username": "User"}))
        ui.section("Season history")
        hist = r[r["franchise"] == TEAM].sort_values("season", ascending=False)
        st.plotly_chart(charts.lines(hist.assign(series=TEAM), "season", "elo_end", "series", {TEAM: ctx.team_color(TEAM)},
                                     "Elo by season", y_label="Elo", hline=config.ELO_START, height=340),
                        width="stretch")
        ten = gm_mod.tenure_table(ctx.cache.gm_history, ctx.elo.history)
        ten = ten[ten["franchise"] == TEAM][["season", "username"]]
        h = hist.merge(ten, on="season", how="left")
        ui.table(h[["season", "team", "wins", "losses", "srs", "elo_end", "playoff_result", "username"]]
                 .rename(columns={"season": "Season", "team": "Code", "wins": "W", "losses": "L", "srs": "SRS",
                                  "elo_end": "Elo", "playoff_result": "Playoffs", "username": "GMs"}),
                 formats={"SRS": "%+.1f", "Elo": "%.0f"}, height=500)
