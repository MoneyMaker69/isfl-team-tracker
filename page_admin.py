"""Admin: cache state, data quality, refresh."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import ui


def render(ctx) -> None:
    cache = ctx.cache
    meta = cache.meta
    ui.explain(
        "Housekeeping. The app never talks to the league's website while you use it — it reads files in the "
        "`cache/` folder that a separate script downloads. This page shows how old those files are, whether the "
        "numbers add up, and lets you pull a quick update. The glossary at the bottom explains every term used "
        "in the app.",
        {"Light refresh": "Downloads this season's standings, the player list, bots and regressions (~30 s). "
                          "Then press 'Reload cache' in the sidebar.",
         "Slow stage": "Downloads every player's TPE history and game log. Needed after roster changes for the "
                       "TPE pages to be current. Run from a terminal; incremental, so usually a few minutes."},
    )

    ui.section("Cache", "Everything the app reads. Built by `scripts/build_cache.py`, never at page load.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Built", (meta.get("built_at") or "—")[:16].replace("T", " "))
    c2.metric("Stage", meta.get("stage", "—"))
    c3.metric("Players fetched (last slow run)", meta.get("players_fetched", "—"))
    c4.metric("Slow-stage errors", len(meta.get("slow_errors", [])))

    rows = []
    for name in sorted(config.CACHE_DIR.glob("*.csv")):
        try:
            n = sum(1 for _ in open(name, "rb")) - 1
        except OSError:
            n = -1
        rows.append({"file": name.name, "rows": n, "size_kb": round(name.stat().st_size / 1024)})
    ui.table(pd.DataFrame(rows))

    ui.section("Data quality")
    if ctx.reconciliation:
        rc = ctx.reconciliation
        st.markdown(
            f"- **Event log adds up**: the cumulative log equals today's `total_tpe` (±1) for "
            f"**{rc['rate']:.1%}** of {rc['compared']:,} players with a log (mean |Δ| {rc['mean_abs']:.1f}). "
            "Below ~95% means events are missing or a task type is misclassified.\n"
            f"- **Regression table verified**: {rc.get('matched', 0):,} of {rc.get('checked', 0):,} logged regressions "
            "equal floor(pct × TPE) for the rulebook's pct. Note: `/player/regression` stamps the *current* applied TPE "
            "on every season, so it is not used as a historical snapshot."
        )
    if ctx.has_logs:
        cur = ctx.full_roster[ctx.full_roster["season"] == ctx.current_season]
        past = ctx.full_roster[ctx.full_roster["season"] < ctx.current_season]
        st.markdown(
            f"- **Current rosters**: {len(cur)} players across {cur['franchise'].nunique()} teams "
            f"({cur['tpe'].isna().sum()} without TPE).\n"
            f"- **Historical rosters**: {len(past):,} player-seasons from S{int(past['season'].min())}; "
            f"{past.groupby(['season', 'franchise']).size().mean():.1f} players per team-season on average "
            "(ISFL minimum is 10 user players; missing rows are backups who never played).\n"
            f"- **Earning rates** measured for {ctx.rates['rate'].notna().sum():,} players."
        )
        if not ctx.hazard.empty:
            st.markdown("- **Retirement hazard** by career season: " +
                        ", ".join(f"S{int(k)} {v:.0%}" for k, v in ctx.hazard.items()))
    else:
        st.markdown("- Player logs not built.")
    if meta.get("slow_errors"):
        with st.expander(f"{len(meta['slow_errors'])} slow-stage errors"):
            st.code("\n".join(meta["slow_errors"][:50]))
    ui.render_notices(cache.notices)

    ui.section("Model state")
    m = ctx.roster_model
    st.markdown(
        f"- Elo: K={ctx.elo.k:.0f}, HFA={ctx.elo.home_adv:.0f}, revert={ctx.elo.revert:.2f}\n"
        f"- Pythagorean exponent: {ctx.pyth_exponent:.2f}\n"
        f"- Roster model: {'α=%s, n=%d, CV R²=%.2f' % (m.alpha, m.n, m.cv_r2) if m else 'not fitted'}\n"
        f"- Blend: {('fitted on %d rows' % ctx.blend.n) if ctx.blend else 'default weights'}\n"
        f"- Win model: {('scale %.1f, hfa %+.1f' % (ctx.win_model.scale, ctx.win_model.hfa)) if ctx.win_model else '—'}"
    )

    ui.section("Refresh", "Light refresh pulls this season's standings, the player table, bots and regressions "
               "(about 30 s). Player logs need the slow stage from a terminal.")
    if st.button("Light refresh from the API"):
        with st.spinner("Fetching…"):
            try:
                sys.path.insert(0, str(Path(config.ROOT / "scripts")))
                bc = importlib.import_module("build_cache")
                new_meta = bc.refresh_current()
                st.success(f"Refreshed at {new_meta['light_refresh_at']}. Press 'Reload cache' in the sidebar.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Refresh failed: {exc}")
    st.code("python scripts/build_cache.py --stage slow", language="bash")

    ui.section("Glossary", "Every term used in the app, in plain words.")
    ui.glossary()
