"""
ISFL Team Tracker — application entry point.

    streamlit run app.py

Reads only the committed cache (see scripts/build_cache.py). Everything
derived — ratings, rosters, models, the backtest — is computed once per
session in `build_context` and shared by the page modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import backtest as backtest_mod
import charts
import config
import data
import elo as elo_mod
import gm as gm_mod
import model as model_mod
import projection as projection_mod
import roster as roster_mod
import simulate
import ui
from data import Cache

import page_admin
import page_backtest
import page_gms
import page_league_tpe
import page_nola
import page_overview
import page_predict
import page_projection
import page_ratings
import page_teams
import page_whatwins

st.set_page_config(page_title=config.APP_TITLE, page_icon="🏈", layout="wide",
                   initial_sidebar_state="expanded")
charts.install_template()
ui.inject_css()


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class Context:
    cache: Cache
    colors: dict[str, str]
    current_season: int
    conferences: dict[str, str]
    # ratings
    elo: elo_mod.EloResult
    srs: pd.DataFrame
    pyth_exponent: float
    ratings: pd.DataFrame                       # franchise-season, every rating
    # rosters (empty frames until the slow stage has run)
    timeline: pd.DataFrame = field(default_factory=pd.DataFrame)
    roster: pd.DataFrame = field(default_factory=pd.DataFrame)
    full_roster: pd.DataFrame = field(default_factory=pd.DataFrame)
    bots: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    roster_model: model_mod.RidgeModel | None = None
    reconciliation: dict = field(default_factory=dict)
    # GMs
    gm_per_season: pd.DataFrame = field(default_factory=pd.DataFrame)
    gm_current: pd.DataFrame = field(default_factory=pd.DataFrame)
    gm_overperf: pd.DataFrame = field(default_factory=pd.DataFrame)
    residuals: pd.DataFrame = field(default_factory=pd.DataFrame)
    # backtest and prediction
    backtest: backtest_mod.BacktestResult | None = None
    blend: backtest_mod.Blend | None = None
    win_model: simulate.WinModel | None = None
    preseason: pd.DataFrame = field(default_factory=pd.DataFrame)
    # projection inputs
    rates: pd.DataFrame = field(default_factory=pd.DataFrame)
    league_rate: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    hazard: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    season_left: float = 0.5

    @property
    def has_logs(self) -> bool:
        return not self.full_roster.empty

    def team_color(self, team: str) -> str:
        return self.colors.get(team, config.COLORS["primary"])


@st.cache_resource(show_spinner="Computing ratings, rosters and models…")
def build_context(nonce: int, built_at: str) -> Context:
    del nonce, built_at   # cache keys only
    cache = data.load_all()
    current = cache.current_season
    conferences = cache.conferences
    colors = charts.team_colors(cache.teams + list(config.FRANCHISE_MAP))

    # Ratings — from S1, on franchises.
    elo_res = elo_mod.run_elo(cache.games)
    srs = elo_mod.srs_all(cache.standings, cache.games) if not cache.standings.empty else pd.DataFrame()
    exponent = elo_mod.fit_pythagorean_exponent(cache.standings) if not cache.standings.empty else config.PYTHAGOREAN_EXPONENT
    ratings = elo_mod.season_ratings(cache.standings, elo_res, srs, exponent) if not cache.standings.empty else pd.DataFrame()

    ctx = Context(cache=cache, colors=colors, current_season=current, conferences=conferences,
                  elo=elo_res, srs=srs, pyth_exponent=exponent, ratings=ratings)

    # GM Elo needs only standings.
    ctx.gm_per_season, ctx.gm_current = gm_mod.gm_elo(cache.gm_history, elo_res.history)

    if cache.has_logs:
        ctx.timeline = roster_mod.tpe_timeline(cache)
        ctx.reconciliation = roster_mod.reconcile(ctx.timeline, cache.players, current)
        ctx.reconciliation.update(roster_mod.verify_regression_pcts(ctx.timeline, cache.players))
        ctx.season_left = projection_mod.season_remaining(cache.seasons, current)
        ctx.roster = roster_mod.rosters(cache)
        ctx.full_roster = roster_mod.roster_with_tpe(cache, ctx.timeline, ctx.roster, ctx.season_left)
        ctx.bots = roster_mod.bots_as_players(cache)
        ctx.features = roster_mod.team_features(ctx.full_roster, ctx.bots)
        ctx.features = ctx.features[ctx.features["season"] >= config.FIRST_TPE_SEASON]

        ctx.model_table = ctx.features.merge(srs[["season", "franchise", "srs"]],
                                             on=["season", "franchise"], how="left")
        ctx.model_table = ctx.model_table.merge(
            cache.standings[["season", "franchise", "wins", "win_pct", "margin", "playoff_result", "conference"]],
            on=["season", "franchise"], how="left")
        train_tbl = ctx.model_table[ctx.model_table["season"] < current]
        ctx.roster_model = model_mod.train(train_tbl)
        if ctx.roster_model is not None:
            res = ctx.roster_model.cv_predictions.copy()
            res["residual"] = res["srs"] - res["predicted"]
            ctx.residuals = res
            ctx.gm_overperf = gm_mod.over_performance(cache.gm_history, res[["season", "franchise", "residual"]])

        ctx.rates = roster_mod.earning_rates(ctx.timeline, current)
        ctx.league_rate = roster_mod.league_rate_by_career_season(ctx.timeline, cache.players, current)
        ctx.hazard = roster_mod.retirement_hazard(cache.players, cache.player_games, current)

    # Backtest and the blend for the upcoming season.
    if not cache.standings.empty:
        ctx.backtest = backtest_mod.walk_forward(cache, elo_res, srs, ctx.features)
        with_roster = ctx.roster_model is not None
        ctx.blend = backtest_mod.final_blend(ctx.backtest, with_roster)
        ctx.win_model = backtest_mod.prediction_win_model(ctx.backtest, cache.games)
        slope = backtest_mod.elo_points_slope(elo_res, srs)
        ctx.preseason = backtest_mod.preseason_table(
            current, cache, elo_res, srs, ctx.features, slope, ctx.roster_model,
            ctx.gm_overperf if not ctx.gm_overperf.empty else None)

    return ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def render_status(ctx: Context) -> None:
    meta = ctx.cache.meta
    built = meta.get("built_at")
    if built:
        try:
            when = datetime.fromisoformat(built)
            age_min = (datetime.now(timezone.utc) - when).total_seconds() / 60
            if age_min < 2:
                fresh = "just now"
            elif age_min < 90:
                fresh = f"{age_min:.0f} min ago"
            elif age_min < 48 * 60:
                fresh = f"{age_min / 60:.0f} h ago"
            else:
                fresh = f"{age_min / 1440:.0f} days ago"
            detail = f"Cache built {when:%d %b %Y %H:%M} UTC · {fresh}"
        except ValueError:
            detail = f"Cache built {built}"
    else:
        detail = "No cache metadata — run scripts/build_cache.py"
    logs = "with player logs" if ctx.has_logs else "standings only"
    ui.status_bar(f"S{ctx.current_season}", f"{detail} · {logs}")


PAGE_BLURBS = {
    "Overview": "Standings and ratings right now",
    "Ratings": "Elo and margin ratings back to S1",
    "Teams": "One roster: TPE, age, regression",
    "League TPE": "All 14 teams' TPE compared",
    "What wins": "How much TPE matters, and where",
    "GMs": "GM ratings and tenure",
    "Projection": "Where each team is heading",
    "Predict": "The prediction-task ranking",
    "Backtest": "How good the predictions are",
    config.HOME_TEAM: "Home-team page",
    "Admin": "Data status, refresh, glossary",
}

PAGES = {
    "Overview": page_overview.render,
    "Ratings": page_ratings.render,
    "Teams": page_teams.render,
    "League TPE": page_league_tpe.render,
    "What wins": page_whatwins.render,
    "GMs": page_gms.render,
    "Projection": page_projection.render,
    "Predict": page_predict.render,
    "Backtest": page_backtest.render,
    config.HOME_TEAM: page_nola.render,
    "Admin": page_admin.render,
}


def main() -> None:
    st.session_state.setdefault("nonce", 0)
    st.title(config.APP_TITLE)

    meta_built = ""
    meta_path = config.CACHE_DIR / "meta.json"
    if meta_path.exists():
        meta_built = str(meta_path.stat().st_mtime)
    ctx = build_context(st.session_state["nonce"], meta_built)

    with st.sidebar:
        st.markdown("### View")
        page = st.radio("Page", list(PAGES), label_visibility="collapsed",
                        captions=[PAGE_BLURBS.get(p, "") for p in PAGES])
        st.markdown("### Data")
        if st.button("Reload cache", width="stretch"):
            build_context.clear()
            st.session_state["nonce"] += 1
            st.rerun()
        st.caption("Reads `cache/` only. Rebuild with `python scripts/build_cache.py`.")

    render_status(ctx)
    ui.render_notices(ctx.cache.notices, only={"error"})
    if ctx.cache.standings.empty:
        ui.render_notices(ctx.cache.notices)
        ui.empty_state("No data yet", "Run `python scripts/build_cache.py --stage fast` and reload.")
        return

    PAGES[page](ctx)


if __name__ == "__main__":
    main()
