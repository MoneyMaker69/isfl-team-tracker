"""Predict: the preseason ranking for the prediction task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import backtest as backtest_mod
import charts
import config
import simulate
import ui


def render(ctx) -> None:
    pre = ctx.preseason.copy()
    if pre.empty:
        ui.empty_state("Nothing to predict", "The preseason table needs standings for last season.")
        return
    season = ctx.current_season
    has_roster = pre["roster_pred"].notna().any()
    ui.explain(
        "**This is the page for the prediction task.** Each team gets a strength number from a blend of: its Elo "
        "coming into the season, last season's SRS and record, its current roster TPE (via the What-wins model) "
        "and its GMs' track record. Then the whole season is simulated 10,000 times — every game decided by the "
        "strength gap plus dice — and we count how often each team finishes where. The order shown is by "
        "average finish; the grey box is ready to paste into the task.",
        {"Weights: Fitted by backtest": "The blend the Backtest page found best. Default.",
         "Weights: Roster model only": "Ignore last season entirely; rank purely on this season's roster TPE. "
                                       "Use it when you think last year was a fluke.",
         "Weights: Manual": "Set the mix yourself with sliders.",
         "Exp W": "Average wins across the 10,000 simulated seasons. W p10 / p90 are the bad and good cases.",
         "Playoffs / Bye": "Chance of finishing top 3 in the conference / top 1 (first-round bye).",
         "Best record": "Chance of the best record in the whole league — the 'regular season champion' pick.",
         "Finish distribution": "For each team, the chance of finishing 1st, 2nd … 7th in its conference.",
         "Sensitivity": "If one team's strength is off by a few points, how much does its outlook change? "
                        "Shows how confident (or not) to be in a placement.",
         "Strength": ui.GLOSSARY["Strength"]},
        expanded=True,
    )

    with st.sidebar:
        st.markdown("### Predict")
        mode = st.radio("Weights", ["Fitted by backtest", "Roster model only", "Manual"], horizontal=False,
                        help="The backtest can't separate these beyond noise; roster-only has the lowest "
                             "points error and the best 14-team rank correlation.")
        manual = {}
        if mode == "Manual":
            manual["elo"] = st.slider("Elo", 0.0, 1.0, config.DEFAULT_BLEND["elo"], 0.05)
            manual["srs"] = st.slider("Last season SRS", 0.0, 1.0, config.DEFAULT_BLEND["srs"], 0.05)
            manual["wpct"] = st.slider("Last season record", 0.0, 1.0, config.DEFAULT_BLEND["wpct"], 0.05)
            manual["roster"] = st.slider("Roster TPE model", 0.0, 1.0, config.DEFAULT_BLEND["roster"], 0.05,
                                         disabled=not has_roster)
            manual["gm"] = st.slider("GM factor", 0.0, 1.0, config.DEFAULT_BLEND["gm"], 0.05)
        n_sims = st.select_slider("Simulations", [2000, 5000, 10000, 20000], value=config.SIMULATIONS)
        seed = st.number_input("Seed", 0, 10_000, config.SIM_SEED)

    # Strength.
    if mode == "Fitted by backtest" and ctx.blend is not None:
        strength = ctx.blend.predict(pre)
        w = ctx.blend.weights
        how = (f"Non-negative ridge fitted on {ctx.blend.n} preseason rows (S{ctx.blend.seasons[0]}–S{ctx.blend.seasons[-1]}). "
               "Relative weight: " + ", ".join(f"{k.replace('_pts', '').replace('_prev', ' prev').replace('_pred', '').replace('_factor', '')} {v:.0%}" for k, v in w.items()))
    elif mode == "Roster model only" and has_roster:
        strength = pre["roster_pred"].fillna(0.0)
        how = ("Roster model alone: Week 1 starter TPE by position group → predicted points per game. "
               "Ignores last season entirely.")
    else:
        strength = backtest_mod.default_strength(pre, manual or None)
        how = "Hand-weighted blend of the preseason signals (each in points per game)."
    pre["strength"] = strength.values
    pre["conference"] = pre["franchise"].map(ctx.conferences)

    # Win model and schedule.
    wm = ctx.win_model or simulate.WinModel(scale=10.0, hfa=config.SRS_HOME_ADVANTAGE)
    rng = np.random.default_rng(int(seed))
    played = ctx.cache.games[(ctx.cache.games["season"] == season) & (ctx.cache.games["phase"] == "regular")]
    schedule = simulate.complete_schedule(played, {t: ctx.conferences[t] for t in pre["franchise"]}, rng)
    n_played = int(schedule["played"].sum())

    res = simulate.simulate(pre.set_index("franchise")["strength"], schedule, ctx.conferences, wm,
                            n_sims=int(n_sims), seed=int(seed))

    ui.section(f"S{season} preseason strength", how)
    st.caption(f"Win model: scale {wm.scale:.1f} pts, home edge {wm.hfa:+.1f} pts, fitted on "
               f"{wm.n_games} games of past preseason predictions. "
               + (f"{n_played} games already played are fixed; the rest are simulated on a generated schedule."
                  if n_played else "No games played yet — the schedule is generated with the league's structure "
                                   "(every conference rival twice, four cross-conference games)."))

    # The ranking to submit.
    order = simulate.ranking_for_task(res)
    cols = st.columns(len(order))
    for col, (conf, teams) in zip(cols, sorted(order.items())):
        with col:
            st.markdown(f"#### {conf}")
            tbl = res.table.set_index("franchise")
            for i, t in enumerate(teams, 1):
                row = tbl.loc[t]
                ui.rank_row(i, t, f"{row['exp_wins']:.1f} W · playoffs {row['p_playoffs']:.0%}")
    best = res.table.sort_values("p_best_record", ascending=False).iloc[0]
    ui.insight("Regular season champion (best overall record)",
               f"{best['franchise']} — {best['p_best_record']:.0%}; next {res.table.sort_values('p_best_record', ascending=False).iloc[1]['franchise']} "
               f"{res.table.sort_values('p_best_record', ascending=False).iloc[1]['p_best_record']:.0%}")
    text = "\n".join(f"{conf}: " + ", ".join(teams) for conf, teams in sorted(order.items()))
    text += f"\nRegular season champion: {best['franchise']}"
    st.code(text, language=None)

    tabs = st.tabs(["Probabilities", "Finish distribution", "Inputs", "Sensitivity"])
    with tabs[0]:
        show = res.table[["franchise", "conference", "strength", "exp_wins", "wins_p10", "wins_p90", "p_playoffs",
                          "p_bye", "p_best_record", "p_conf_champ", "p_ultimus"]].rename(columns={
            "franchise": "Team", "conference": "Conf", "strength": "Strength", "exp_wins": "Exp W",
            "wins_p10": "W p10", "wins_p90": "W p90", "p_playoffs": "Playoffs", "p_bye": "Bye",
            "p_best_record": "Best record", "p_conf_champ": "Conf champ", "p_ultimus": "Ultimus"})
        ui.table(show, formats={"Strength": "%+.1f", "Exp W": "%.1f", "W p10": "%.0f", "W p90": "%.0f",
                                "Playoffs": "%.0%", "Bye": "%.0%", "Best record": "%.0%",
                                "Conf champ": "%.0%", "Ultimus": "%.0%"})
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.ranked_bar(res.table, "franchise", "p_ultimus", ctx.colors, "P(Ultimus)", fmt="{:.0%}"),
                            width="stretch")
        with c2:
            st.plotly_chart(charts.ranked_bar(res.table, "franchise", "exp_wins", ctx.colors, "Expected wins", fmt="{:.1f}"),
                            width="stretch")
        ui.download(show, f"prediction_S{season}.csv")

    with tabs[1]:
        ui.section("Probability of each finish within the conference")
        for conf in sorted(order):
            teams = order[conf]
            mat = res.rank_matrix.loc[teams]
            st.plotly_chart(charts.heatmap(mat, conf, fmt="%{text:.0%}", height=320), width="stretch")

    with tabs[2]:
        ui.section("What the model was given", "All in points per game above average.")
        inp = pre[["franchise", "conference", "elo_pre", "elo_pts", "srs_prev", "wpct_prev", "roster_pred",
                   "gm_factor", "strength"]].sort_values("strength", ascending=False)
        if "starters_mean" in pre.columns:
            inp = inp.merge(pre[["franchise", "starters_mean", "n_inactive"]], on="franchise", how="left")
        ui.table(inp.rename(columns={"franchise": "Team", "conference": "Conf", "elo_pre": "Elo in",
                                     "elo_pts": "Elo (pts)", "srs_prev": "SRS last", "wpct_prev": "Record last",
                                     "roster_pred": "Roster", "gm_factor": "GM", "strength": "Strength",
                                     "starters_mean": "Starter TPE", "n_inactive": "IA"}),
                 formats={"Elo in": "%.0f", "Elo (pts)": "%+.1f", "SRS last": "%+.1f", "Record last": "%+.1f",
                          "Roster": "%+.1f", "GM": "%+.1f", "Strength": "%+.1f", "Starter TPE": "%.0f"})
        if ctx.blend is not None:
            coefs = pd.DataFrame({"feature": ctx.blend.features, "std_coef": ctx.blend.coef,
                                  "share": ctx.blend.weights.values})
            st.plotly_chart(charts.signed_bar(coefs, "feature", "std_coef", "Fitted blend (standardised coefficients)",
                                              fmt="{:+.2f}", height=260), width="stretch")

    with tabs[3]:
        ui.section("Sensitivity", "How the expected wins move if one team's strength is off by ±3 points.")
        team = st.selectbox("Team", list(pre["franchise"]),
                            index=list(pre["franchise"]).index(config.HOME_TEAM) if config.HOME_TEAM in list(pre["franchise"]) else 0)
        rows = []
        for delta in (-3, -1.5, 0, 1.5, 3):
            s2 = pre.set_index("franchise")["strength"].copy()
            s2[team] += delta
            r2 = simulate.simulate(s2, schedule, ctx.conferences, wm, n_sims=3000, seed=int(seed))
            row = r2.table.set_index("franchise").loc[team]
            rows.append({"Δ strength": delta, "Exp W": row["exp_wins"], "Playoffs": row["p_playoffs"],
                         "Ultimus": row["p_ultimus"], "Modal rank": int(row["modal_rank"])})
        ui.table(pd.DataFrame(rows), formats={"Δ strength": "%+.1f", "Exp W": "%.1f", "Playoffs": "%.0%", "Ultimus": "%.0%"})
