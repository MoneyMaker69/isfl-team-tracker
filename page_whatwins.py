"""What wins: how much of the margin is TPE, and where does it need to be?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import model as model_mod
import ui
from page_teams import _needs_logs


def render(ctx) -> None:
    if _needs_logs(ctx):
        return
    ui.explain(
        "The question this page answers: **how much of winning is just having more TPE, and where does that "
        "TPE need to be?** It looks at every team-season since S54, compares each team's Week 1 TPE with how "
        "many points per game better than average it ended up, and fits a model to it.",
        {"'Starter TPE explains X%'": "If X is 63%, then 63% of the difference between good and bad teams is "
                                      "explained by starter TPE alone. The rest is scheme, depth charts, luck, "
                                      "and things the model can't see.",
         "Points per game from +100": "What one group's starters being 100 TPE better is worth. OL at +0.8 "
                                      "means: upgrade all five linemen by 100 TPE each and expect to score/allow "
                                      "about 0.8 points per game better.",
         "Champion profile": "For teams that won the Ultimus, how far above the league average was each "
                             "position group that season? Big positives = where champions were elite.",
         "Correlations": "A simpler check: does this number move with winning, on its own? +1 = perfectly, "
                         "0 = not at all.",
         "Residuals": "Teams that did much better or worse than their TPE said. Persistent over-performers "
                      "are usually well-run teams.",
         "SRS": ui.GLOSSARY["SRS"], "Starter TPE": ui.GLOSSARY["Starter TPE"]},
    )
    tbl = ctx.model_table[ctx.model_table["season"] < ctx.current_season].dropna(subset=["srs"])
    m = ctx.roster_model
    n_seasons = tbl["season"].nunique()

    ui.section("How much is TPE?",
               f"Fitted on {len(tbl)} franchise-seasons from S{int(tbl['season'].min())} to S{int(tbl['season'].max())} "
               f"({n_seasons} seasons). Target is SRS: points per game above average, schedule-adjusted. "
               "Small sample — read the direction, not the third decimal.")
    simple = model_mod.simple_tpe_model(tbl, "starters_mean")
    simple_mean = model_mod.simple_tpe_model(tbl, "tpe_mean")
    c1, c2, c3 = st.columns(3)
    c1.metric("Starter TPE explains", ui.pct(simple["r2"]), help="R² of SRS on starter TPE alone.")
    c2.metric("Mean TPE explains", ui.pct(simple_mean["r2"]))
    c3.metric("Position-group model (CV)", ui.pct(m.cv_r2) if m else "—",
              help="Leave-one-season-out R² of the ridge on nine group starter-TPEs.")
    ui.insight("+100 starter TPE across the board", f"≈ {simple['slope'] * 100:+.1f} points per game" if simple["slope"] == simple["slope"] else "—")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.scatter(tbl, "starters_mean", "srs", "franchise", ctx.colors,
                                       "Starter TPE vs SRS, every team-season",
                                       x_label="Starter TPE (Week 1)", y_label="SRS", trend=True,
                                       hover=["season", "wins", "playoff_result"], height=480),
                        width="stretch")
    with right:
        st.plotly_chart(charts.scatter(tbl, "tpe_mean", "win_pct", "franchise", ctx.colors,
                                       "Mean TPE vs win%", x_label="Mean TPE (Week 1)", y_label="Win %",
                                       trend=True, hover=["season", "playoff_result"], height=480),
                        width="stretch")

    tabs = st.tabs(["Which positions", "Champion profile", "Correlations", "Residuals"])

    with tabs[0]:
        if m is None:
            st.caption("Not enough seasons to fit the position-group model yet.")
        else:
            ui.section("Points per game from +100 starter TPE in each group",
                       f"Ridge coefficients (α={m.alpha}). Groups are correlated — good teams are good everywhere — "
                       "so the ridge spreads credit; compare against the univariate correlations tab.")
            pe = model_mod.partial_effects(m)
            st.plotly_chart(charts.signed_bar(pe, "group", "effect_ppg", "", fmt="{:+.2f}"), width="stretch")
            ui.table(pe.rename(columns={"group": "Group", "effect_ppg": "Pts/game per +100", "coef_std": "Std coef"}),
                     formats={"Pts/game per +100": "%+.2f", "Std coef": "%+.2f"})
            st.caption(f"CV RMSE {m.cv_rmse:.2f} points vs target std {m.target_std:.2f}.")

    with tabs[1]:
        ui.section("Where were the winners elite?",
                   "Mean within-season z-score of each group's starter TPE, by how the season ended.")
        prof = model_mod.champion_profile(tbl)
        prof["feature"] = prof["feature"].str.replace("_starters", "", regex=False)
        mat = prof.set_index("feature")
        st.plotly_chart(charts.heatmap(mat, "", fmt="%{text:.2f}", zmid=0, colorscale=charts.diverging_scale(),
                                       height=420), width="stretch")
        champs = tbl[tbl["playoff_result"] == "champion"].sort_values("season")
        if not champs.empty:
            z = model_mod.zscores_by_season(tbl)
            cz = z.merge(champs[["season", "franchise"]], on=["season", "franchise"])
            cz = cz.set_index(cz["season"].astype(str) + " " + cz["franchise"])
            cols = [f"{g}_starters" for g in config.POSITION_GROUPS] + ["starters_mean"]
            cz = cz[cols]
            cz.columns = config.POSITION_GROUPS + ["ALL"]
            st.plotly_chart(charts.heatmap(cz, "Each champion's roster, as z-scores", fmt="%{text:.1f}", zmid=0,
                                           colorscale=charts.diverging_scale()), width="stretch")

    with tabs[2]:
        ui.section("Univariate correlations with SRS")
        corr = model_mod.correlations(tbl)
        corr["feature"] = corr["feature"].str.replace("_starters", " starters", regex=False)
        st.plotly_chart(charts.signed_bar(corr, "feature", "pearson", "Pearson r", fmt="{:+.2f}"),
                        width="stretch")
        ui.table(corr, formats={"pearson": "%+.2f", "spearman": "%+.2f"})

    with tabs[3]:
        if ctx.residuals.empty:
            st.caption("No model, no residuals.")
        else:
            ui.section("Over- and under-performers", "Actual SRS minus roster-predicted SRS (out of sample). "
                       "Persistent positives are coaching, scheme, or luck — see the GMs page.")
            res = ctx.residuals.copy()
            res["label"] = "S" + res["season"].astype(str) + " " + res["franchise"]
            top = pd.concat([res.nlargest(8, "residual"), res.nsmallest(8, "residual")])
            st.plotly_chart(charts.signed_bar(top, "label", "residual", "", fmt="{:+.1f}"), width="stretch")
            by_team = res.groupby("franchise")["residual"].agg(["mean", "std", "size"]).reset_index()
            by_team.columns = ["Team", "Mean residual", "Std", "Seasons"]
            ui.table(by_team.sort_values("Mean residual", ascending=False),
                     formats={"Mean residual": "%+.2f", "Std": "%.2f"})
