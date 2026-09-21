"""Backtest: how well would this have predicted past seasons?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import config
import ui

METHOD_LABELS = {
    "last_season": "Copy last season",
    "elo": "Elo only",
    "srs_prev": "Last SRS only",
    "roster": "Roster TPE only",
    "blend": "Blend (used)",
}


def render(ctx) -> None:
    bt = ctx.backtest
    if bt is None or bt.per_season.empty:
        ui.empty_state("No backtest", "Needs at least two completed seasons of standings.")
        return

    ui.explain(
        "Is the prediction any good? This page pretends it is the start of each old season, makes the prediction "
        "with only what was known then, and compares it to what really happened. Every method is scored against "
        "the dumbest possible baseline — **copy last season's standings** — because if we can't beat that, the "
        "model isn't worth much. Honest answer so far: on ordering a 7-team conference, nothing beats it by more "
        "than noise; the roster model is best at predicting points.",
        {"ρ conf": "Spearman rank correlation between predicted and actual conference order, averaged over the two "
                   "conferences. 1 = perfect, 0 = random. ~0.5 is where every method lands.",
         "ρ all": "Same, but ranking all 14 teams together.",
         "SRS MAE": "Average error in points per game. Lower is better.",
         "Best record": "How often the team predicted strongest really had the best record.",
         "Predicted rank → actual rank": "Reading across a row: when we predicted 1st, how often was the team "
                                         "actually 1st, 2nd, 3rd…? A strong diagonal is good.",
         "Spearman ρ": ui.GLOSSARY["Spearman ρ"], "Walk-forward": ui.GLOSSARY["Walk-forward"]},
    )
    ui.section("Walk-forward", f"For every season from S{int(bt.per_season['season'].min())} the model is fitted on "
               "earlier seasons only, predicts the season from what was knowable before Week 1, and is scored "
               "against the final standings. The blend weights on the Predict page come from here.")
    st.caption("Spearman ρ: rank correlation between predicted and actual order (1 = perfect, 0 = random). "
               "'Conf' is averaged over the two conferences — that's the prediction task. 'Best record' is how "
               "often the top-strength team had the best record.")

    per = bt.per_season.copy()
    per["method"] = per["method"].map(METHOD_LABELS).fillna(per["method"])
    with st.sidebar:
        st.markdown("### Backtest")
        from_s = st.slider("Score from season", int(per["season"].min()), int(per["season"].max()),
                           max(int(per["season"].min()), config.FIRST_TPE_SEASON + 1),
                           help="The roster model exists only from S57 or so; compare methods on the same seasons.")
    scored = per[per["season"] >= from_s]
    summary = scored.groupby("method")[["spearman_conf", "spearman_all", "srs_mae", "best_record_hit"]].mean()
    counts = scored.groupby("method").size().rename("seasons")
    summary = summary.join(counts).reset_index().sort_values("spearman_conf", ascending=False)
    ui.table(summary.rename(columns={"method": "Method", "spearman_conf": "ρ conf", "spearman_all": "ρ all",
                                     "srs_mae": "SRS MAE", "best_record_hit": "Best record", "seasons": "Seasons"}),
             formats={"ρ conf": "%.3f", "ρ all": "%.3f", "SRS MAE": "%.2f", "Best record": "%.0%"})

    c1, c2 = st.columns(2)
    with c1:
        colors = {m: c for m, c in zip(METHOD_LABELS.values(),
                                       [config.COLORS["text_muted"], config.COLORS["blue"], config.COLORS["purple"],
                                        config.COLORS["green"], config.COLORS["primary"]])}
        st.plotly_chart(charts.lines(per, "season", "spearman_conf", "method", colors,
                                     "Spearman ρ (conference) by season", y_label="ρ", markers=True,
                                     highlight=["Blend (used)", "Copy last season"], height=420),
                        width="stretch")
    with c2:
        st.plotly_chart(charts.lines(per.dropna(subset=["srs_mae"]), "season", "srs_mae", "method", colors,
                                     "SRS mean absolute error by season", y_label="points/game", markers=True,
                                     highlight=["Blend (used)"], height=420), width="stretch")

    ui.section("Predicted vs actual", "Every team-season for the blend.")
    p = bt.predictions[(bt.predictions["method"] == "blend") & (bt.predictions["season"] >= from_s)]
    if not p.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.scatter(p, "predicted", "actual_srs", "franchise", ctx.colors,
                                           "Predicted strength vs actual SRS", x_label="Predicted (pts/game)",
                                           y_label="Actual SRS", trend=True, hover=["season", "wins"], height=460),
                            width="stretch")
        with c2:
            conf = pd.crosstab(p["pred_rank"].astype(int), p["actual_rank"].astype(int), normalize="index")
            conf.index.name = "Predicted rank"
            st.plotly_chart(charts.heatmap(conf, "Predicted rank → actual rank (row share)", fmt="%{text:.0%}",
                                           height=460), width="stretch")
        season = st.selectbox("Season detail", sorted(p["season"].unique(), reverse=True))
        d = p[p["season"] == season].sort_values(["conference", "pred_rank"])
        ui.table(d[["conference", "franchise", "pred_rank", "actual_rank", "predicted", "actual_srs", "wins"]]
                 .rename(columns={"conference": "Conf", "franchise": "Team", "pred_rank": "Pred",
                                  "actual_rank": "Actual", "predicted": "Strength", "actual_srs": "SRS", "wins": "W"}),
                 formats={"Strength": "%+.1f", "SRS": "%+.1f"})

    ui.section("Honesty notes")
    st.markdown(
        "- Rosters for past seasons are reconstructed from game logs, so deep backups who never played are missing. "
        "Starter TPE is unaffected; mean TPE is slightly overstated.\n"
        "- Historical TPE is *total*, not applied; a player sitting on 300 banked TPE looks stronger than they played.\n"
        "- The roster model has about 14 rows per season. Its coefficients move when a season is added.\n"
        "- The blend is refitted each season on all earlier preseason rows, so early seasons use a hand-weighted blend.\n"
        f"- Nothing before S{config.ENGINE_SWITCH_SEASON} (engine change) is scored."
    )
