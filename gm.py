"""
GM ratings.

Two views, both with the caveat that a team has two GMs and credit is shared:

- GM Elo: each ISFL team-season's Elo change is credited to that season's
  GMs (divided among them), accumulated with the same offseason reversion
  as team Elo.
- Over-performance: mean of (actual SRS − roster-predicted SRS) across a
  GM's team-seasons, using out-of-sample predictions. Isolates depth-chart
  and strategy skill from roster quality. Only from FIRST_TPE_SEASON.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def isfl_gm_seasons(gm_history: pd.DataFrame) -> pd.DataFrame:
    g = gm_history[gm_history["league"] == "ISFL"].copy()
    g["season"] = pd.to_numeric(g["season"], errors="coerce")
    g = g.dropna(subset=["season"])
    g["season"] = g["season"].astype(int)
    return g[["season", "franchise", "uid", "username"]].drop_duplicates()


def gm_elo(gm_history: pd.DataFrame, elo_history: pd.DataFrame,
           revert: float = config.ELO_SEASON_REVERT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_gm_season, current_ratings).

    per_gm_season: season, franchise, uid, username, credited, gm_elo_end.
    current_ratings: uid, username, gm_elo, seasons, last_season, teams.
    """
    g = isfl_gm_seasons(gm_history)
    if g.empty or elo_history.empty:
        return pd.DataFrame(), pd.DataFrame()
    g = g.merge(elo_history[["season", "franchise", "change"]], on=["season", "franchise"], how="inner")
    g["n_gms"] = g.groupby(["season", "franchise"])["uid"].transform("size")
    g["credited"] = g["change"] / g["n_gms"]
    g = g.sort_values(["uid", "season"])

    rows = []
    ratings: dict[int, float] = {}
    last_seen: dict[int, int] = {}
    for row in g.itertuples(index=False):
        r = ratings.get(row.uid, config.ELO_START)
        gap = row.season - last_seen.get(row.uid, row.season)
        # Revert once per offseason the GM sat through, including gaps.
        for _ in range(max(0, gap)):
            r = config.ELO_START + (r - config.ELO_START) * (1 - revert)
        r += row.credited
        ratings[row.uid] = r
        last_seen[row.uid] = row.season
        rows.append({"season": row.season, "franchise": row.franchise, "uid": row.uid,
                     "username": row.username, "credited": row.credited, "gm_elo_end": r})
    per = pd.DataFrame(rows)
    current = (per.sort_values("season").groupby("uid").agg(
        username=("username", "last"), gm_elo=("gm_elo_end", "last"),
        seasons=("season", "size"), last_season=("season", "max"),
        first_season=("season", "min"),
        teams=("franchise", lambda s: ", ".join(dict.fromkeys(s))),
    ).reset_index())
    return per, current.sort_values("gm_elo", ascending=False).reset_index(drop=True)


def over_performance(gm_history: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    """
    `residuals`: season, franchise, residual (actual − predicted SRS).
    Returns uid, username, overperf (mean residual), seasons, teams.
    """
    g = isfl_gm_seasons(gm_history)
    if g.empty or residuals.empty:
        return pd.DataFrame(columns=["uid", "username", "overperf", "seasons", "teams"])
    j = g.merge(residuals, on=["season", "franchise"], how="inner")
    out = j.groupby("uid").agg(
        username=("username", "last"), overperf=("residual", "mean"),
        seasons=("season", "size"), last_season=("season", "max"),
        teams=("franchise", lambda s: ", ".join(dict.fromkeys(s))),
    ).reset_index()
    return out.sort_values("overperf", ascending=False).reset_index(drop=True)


def team_gm_factor(gm_history: pd.DataFrame, overperf: pd.DataFrame, season: int,
                   shrink_seasons: int = 4) -> pd.Series:
    """
    Per-franchise GM factor for `season`: mean over-performance of that
    season's GMs, shrunk toward 0 by experience (a one-season sample of +6
    is mostly noise). Uses only history strictly before `season`.
    """
    g = isfl_gm_seasons(gm_history)
    g = g[g["season"] == season]
    if g.empty or overperf.empty:
        return pd.Series(dtype=float)
    o = overperf.set_index("uid")
    rows = {}
    for team, grp in g.groupby("franchise"):
        vals = []
        for uid in grp["uid"]:
            if uid in o.index:
                n = float(o.loc[uid, "seasons"])
                vals.append(float(o.loc[uid, "overperf"]) * n / (n + shrink_seasons))
        rows[team] = float(np.mean(vals)) if vals else 0.0
    return pd.Series(rows)


def tenure_table(gm_history: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """Season × franchise → GM pair, with the team's Elo. For the timeline page."""
    g = isfl_gm_seasons(gm_history)
    pairs = g.groupby(["season", "franchise"])["username"].agg(" / ".join).reset_index()
    return pairs.merge(elo_history[["season", "franchise", "elo_end", "change"]],
                       on=["season", "franchise"], how="left")
