"""
Reconstruct who was on which roster with how much TPE, season by season.

Three products:

- `tpe_timeline`  — player-season: TPE at Week 1, earned during the season,
                    regression charged at its start, activity.
- `rosters`       — player-season-franchise at Week 1, from game logs for past
                    seasons and from the player table for the current one.
- `team_features` — franchise-season: position-group TPE, starters, age,
                    activity. The input to the roster-strength model.

Season-start TPE comes from the event log only. /player/regression looked
like an authoritative snapshot but its `old_tpe`/`new_tpe` are the player's
*current* applied TPE stamped on every season (verified S57–S63: identical
values, 0% rows included), so only its `pct` column is used, and only for
verification. The event log, by contrast, sums exactly to `total_tpe`;
`reconcile()` checks that and the rate is shown on the Admin page.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import config
import data
from data import Cache

_REGRESSION_LABEL = re.compile(r"^S(\d+)\s+Regression", re.IGNORECASE)


# ---------------------------------------------------------------------------
# TPE timeline
# ---------------------------------------------------------------------------


def _season_of_events(events: pd.DataFrame, starts: pd.Series) -> pd.Series:
    """
    Season each event belongs to. Regressions are labelled ("S63 Regression")
    and belong to the season they are charged at the start of, whatever their
    timestamp. Everything else is bucketed by date.
    """
    season = data.season_of_date(events["date"], starts)
    labelled = events["description"].astype(str).str.extract(_REGRESSION_LABEL)[0]
    labelled = pd.to_numeric(labelled, errors="coerce")
    mask = events["is_regression"] & labelled.notna()
    season = season.where(~mask, labelled)
    return season


def tpe_timeline(cache: Cache) -> pd.DataFrame:
    """
    One row per player-season from the first logged season to the current one.

    Columns: pid, season, tpe_start, earned, regression, other, tpe_end,
    n_events, active, source ('regression' | 'events').
    """
    ev = cache.tpe_events
    if ev.empty:
        return pd.DataFrame(columns=["pid", "season", "tpe_start", "earned", "regression",
                                     "other", "tpe_end", "n_events", "active", "source"])
    starts = data.season_starts(cache)
    ev = ev.copy()
    ev["season"] = _season_of_events(ev, starts)
    ev = ev[ev["season"].notna()]
    ev["season"] = ev["season"].astype(int)

    kind = np.select(
        [ev["is_regression"], ev["is_earning"], ev["is_snapshot"]],
        ["regression", "earned", "snapshot"], default="other",
    )
    ev["kind"] = kind
    pivot = (ev.pivot_table(index=["pid", "season"], columns="kind", values="tpe_change",
                            aggfunc="sum", fill_value=0.0)
             .reindex(columns=["earned", "regression", "snapshot", "other"], fill_value=0.0)
             .reset_index())
    counts = (ev[ev["is_earning"]].groupby(["pid", "season"]).size()
              .rename("n_events").reset_index())
    pivot = pivot.merge(counts, on=["pid", "season"], how="left").fillna({"n_events": 0})

    # Fill the gaps: a player with no events in a season still has a row, so
    # the cumulative sum carries forward.
    current = cache.current_season
    frames = []
    for pid, grp in pivot.groupby("pid"):
        first = int(grp["season"].min())
        full = pd.DataFrame({"pid": pid, "season": range(first, current + 1)})
        frames.append(full.merge(grp, on=["pid", "season"], how="left").fillna(0.0))
    tl = pd.concat(frames, ignore_index=True).sort_values(["pid", "season"])

    per_season = tl["earned"] + tl["regression"] + tl["snapshot"] + tl["other"]
    tl["tpe_end"] = per_season.groupby(tl["pid"]).cumsum()
    # Week 1 TPE: after this season's regression and snapshot, before earning.
    tl["tpe_start"] = tl["tpe_end"] - tl["earned"] - tl["other"]
    tl["active"] = tl["n_events"] >= config.INACTIVE_EVENT_THRESHOLD
    tl["source"] = "events"
    tl = tl.drop(columns=["snapshot"])
    tl["n_events"] = tl["n_events"].astype(int)
    return tl.reset_index(drop=True)


def reconcile(timeline: pd.DataFrame, players: pd.DataFrame, current_season: int,
              tolerance: int = 1) -> dict:
    """
    Does the event log add up? For every player with a log, the cumulative
    sum at the current season should equal the player table's `total_tpe`.
    Reported on the Admin page; a low rate means events are missing or a
    task type is being misclassified.
    """
    end = timeline[timeline["season"] == current_season][["pid", "tpe_end"]]
    j = end.merge(players[["pid", "total_tpe"]], on="pid", how="inner").dropna()
    if j.empty:
        return {"compared": 0, "within_tolerance": 0, "rate": float("nan"), "mean_abs": float("nan")}
    diff = (j["tpe_end"] - j["total_tpe"]).abs()
    return {
        "compared": int(len(j)),
        "within_tolerance": int((diff <= tolerance).sum()),
        "rate": float((diff <= tolerance).mean()),
        "mean_abs": float(diff.mean()),
    }


def verify_regression_pcts(timeline: pd.DataFrame, players: pd.DataFrame) -> dict:
    """
    Cross-check the rulebook table against logged regressions: for each
    player-season with a regression event, was the loss floor(pct × TPE
    before)? Uses the event log alone.
    """
    tl = timeline.merge(players[["pid", "draft_season"]], on="pid", how="left")
    tl = tl[(tl["regression"] < 0) & tl["draft_season"].notna()].copy()
    if tl.empty:
        return {"checked": 0, "matched": 0}
    finished = tl["season"] - tl["draft_season"]
    before = tl["tpe_start"] - tl["regression"]   # regression is negative
    expected = np.floor(before * finished.map(lambda f: config.regression_pct(int(f))))
    ok = (expected + tl["regression"]).abs() <= 1
    return {"checked": int(len(tl)), "matched": int(ok.sum())}


# ---------------------------------------------------------------------------
# Rosters
# ---------------------------------------------------------------------------


def rosters(cache: Cache) -> pd.DataFrame:
    """
    Player-season-franchise at Week 1.

    Past seasons: the team a player recorded their first regular-season game
    for. Current season: the player table's ISFL assignment. Players with no
    recorded game in a season are absent for that season — that is a known
    limit (deep backups), not a bug.
    """
    current = cache.current_season
    pg = cache.player_games
    rows = []
    if not pg.empty:
        reg = pg[(pg["season_state"] == "RegularSeason") & pg["season"].notna()].copy()
        reg = reg[reg["season"] < current]
        first = (reg.sort_values(["pid", "season", "week"])
                 .groupby(["pid", "season"]).agg(franchise=("franchise", "first"),
                                                 games=("week", "size")).reset_index())
        first["season"] = first["season"].astype(int)
        rows.append(first)
    cur = cache.players[(cache.players["current_league"] == "ISFL")
                        & cache.players["isfl_team"].notna()]
    if not cur.empty:
        rows.append(pd.DataFrame({
            "pid": cur["pid"].values, "season": current,
            "franchise": data.franchise(cur["isfl_team"]).values, "games": 0,
        }))
    if not rows:
        return pd.DataFrame(columns=["pid", "season", "franchise", "games"])
    out = pd.concat(rows, ignore_index=True)
    valid = set(cache.teams) | set(config.FRANCHISE_MAP.values())
    return out[out["franchise"].isin(valid)].reset_index(drop=True)


def roster_with_tpe(cache: Cache, timeline: pd.DataFrame, roster: pd.DataFrame,
                    season_left: float = 0.5) -> pd.DataFrame:
    """
    Rosters joined to TPE and player facts. One row per player-season.

    `tpe` is Week 1 TPE for past seasons — NaN where the log doesn't reach
    (before FIRST_TPE_SEASON), never back-filled from today's number. For
    the current season it's the live total from the player table, which
    includes preseason earnings.
    """
    p = cache.players[["pid", "name", "username", "draft_season", "position", "pos_group",
                       "archetype", "status", "total_tpe", "applied_tpe", "banked_tpe",
                       "weekly_activity_check", "weekly_training"]]
    out = roster.merge(p, on="pid", how="left")
    out = out.merge(timeline[["pid", "season", "tpe_start", "earned", "regression",
                              "n_events", "active"]],
                    on=["pid", "season"], how="left")
    current = cache.current_season
    is_cur = out["season"] == current
    out["tpe"] = out["tpe_start"]
    out.loc[is_cur, "tpe"] = out.loc[is_cur, "total_tpe"]
    out["career_season"] = out["season"] - out["draft_season"] + 1
    out["seasons_finished_at_end"] = out["season"] - out["draft_season"] + 1
    out["next_regression_pct"] = out["seasons_finished_at_end"].map(
        lambda s: config.regression_pct(int(s)) if pd.notna(s) else np.nan)
    out["next_regression_tpe"] = np.floor(out["tpe"].fillna(0) * out["next_regression_pct"].fillna(0))
    # Activity for the current season: the season's events are still
    # accumulating, so the bar scales with how much of it has been played
    # (about one earning event per week is the floor for "still here").
    weeks_played = (1.0 - season_left) * config.WEEKS_PER_SEASON
    bar = int(min(config.INACTIVE_EVENT_THRESHOLD, max(1, np.floor(weeks_played))))
    cur_active = (out.loc[is_cur, "status"] == "active") & (out.loc[is_cur, "n_events"].fillna(0) >= bar)
    out.loc[is_cur, "active"] = cur_active
    out["active"] = out["active"].fillna(False).astype(bool)
    out["n_events"] = out["n_events"].fillna(0).astype(int)
    return out


def bots_as_players(cache: Cache) -> pd.DataFrame:
    """OL bots as pseudo-players so they count toward the OL group."""
    b = cache.bots
    if b.empty:
        return pd.DataFrame(columns=["pid", "season", "franchise", "pos_group", "tpe", "name", "is_bot"])
    isfl = b[b["team"].notna() & (b["league"].fillna("ISFL") == "ISFL")]
    return pd.DataFrame({
        "pid": -isfl["bot_id"].astype(int),
        "season": isfl["season"].astype(int),
        "franchise": data.franchise(isfl["team"]),
        "pos_group": isfl["pos_group"].fillna("OL"),
        "tpe": isfl["tier_tpe"],
        "name": isfl["name"] + " (bot)",
        "is_bot": True,
    })


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------


def _starter_mean(values: pd.Series, n: int) -> float:
    top = values.dropna().sort_values(ascending=False).head(n)
    if top.empty:
        return np.nan
    # A short group is padded with zeros: a team with three OL fields two
    # empty slots, and that should hurt.
    return float(top.sum() / n)


def team_features(full_roster: pd.DataFrame, bots: pd.DataFrame) -> pd.DataFrame:
    """
    One row per franchise-season with the model inputs.

    Columns per group G in POSITION_GROUPS: G_mean, G_starters, G_n.
    Plus: tpe_mean, tpe_total, n_players, starters_total, age_mean,
    n_regression_years, n_inactive, qb1.
    """
    r = full_roster[["pid", "season", "franchise", "pos_group", "tpe", "career_season", "active"]].copy()
    r["is_bot"] = False
    if not bots.empty:
        bb = bots[["pid", "season", "franchise", "pos_group", "tpe"]].copy()
        bb["career_season"] = np.nan
        bb["active"] = True
        bb["is_bot"] = True
        r = pd.concat([r, bb], ignore_index=True)
    r = r[r["pos_group"].notna() & r["tpe"].notna()]

    rows = []
    for (season, team), grp in r.groupby(["season", "franchise"]):
        humans = grp[~grp["is_bot"]]
        row = {"season": int(season), "franchise": team,
               "n_players": int(len(humans)), "n_bots": int(grp["is_bot"].sum()),
               "tpe_mean": float(humans["tpe"].mean()) if len(humans) else np.nan,
               "tpe_total": float(grp["tpe"].sum()),
               "age_mean": float(humans["career_season"].mean()) if len(humans) else np.nan,
               "n_regression_years": int((humans["career_season"] >= config.FIRST_REGRESSION_CAREER_SEASON).sum()),
               "n_inactive": int((~humans["active"]).sum())}
        starters_total = 0.0
        for g in config.POSITION_GROUPS:
            gg = grp[grp["pos_group"] == g]["tpe"]
            n = config.STARTERS[g]
            row[f"{g}_mean"] = float(gg.mean()) if len(gg) else np.nan
            row[f"{g}_starters"] = _starter_mean(gg, n) if len(gg) else 0.0
            row[f"{g}_n"] = int(len(gg))
            starters_total += row[f"{g}_starters"] * n
        row["starters_total"] = starters_total
        row["starters_mean"] = starters_total / sum(config.STARTERS.values())
        qb = grp[grp["pos_group"] == "QB"]["tpe"]
        row["qb1"] = float(qb.max()) if len(qb) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["season", "franchise"]).reset_index(drop=True)


def group_table(full_roster: pd.DataFrame, bots: pd.DataFrame, season: int) -> pd.DataFrame:
    """Franchise × group starter-TPE matrix for one season (heatmap input)."""
    feats = team_features(full_roster[full_roster["season"] == season],
                          bots[bots["season"] == season] if not bots.empty else bots)
    cols = [f"{g}_starters" for g in config.POSITION_GROUPS]
    out = feats.set_index("franchise")[cols]
    out.columns = config.POSITION_GROUPS
    return out


# ---------------------------------------------------------------------------
# Earning rates
# ---------------------------------------------------------------------------


def earning_rates(timeline: pd.DataFrame, current_season: int,
                  window: int = config.DEFAULT_RATE_WINDOW) -> pd.DataFrame:
    """
    Per-player TPE earned per season, measured over the last `window` complete
    seasons, excluding the current one (partial) and each player's first
    logged season (partial too — creation or migration mid-season).

    Returns pid, rate, seasons_used. Players with nothing usable get NaN.
    """
    tl = timeline[timeline["season"] < current_season].copy()
    first = tl.groupby("pid")["season"].transform("min")
    tl = tl[tl["season"] > first]
    tl = tl.sort_values(["pid", "season"]).groupby("pid").tail(window)
    out = tl.groupby("pid").agg(rate=("earned", "mean"), seasons_used=("season", "size")).reset_index()
    return out


def league_rate_by_career_season(timeline: pd.DataFrame, players: pd.DataFrame,
                                 current_season: int) -> pd.Series:
    """Median earned TPE per season, by career season, for the fallback."""
    tl = timeline[timeline["season"] < current_season].merge(
        players[["pid", "draft_season"]], on="pid", how="left")
    first = tl.groupby("pid")["season"].transform("min")
    tl = tl[(tl["season"] > first) & tl["draft_season"].notna()]
    tl["career_season"] = tl["season"] - tl["draft_season"] + 1
    tl = tl[tl["earned"] > 0]
    return tl.groupby("career_season")["earned"].median()


def season_from_date(dates: pd.Series, starts: pd.Series) -> pd.Series:
    """
    Season a timestamp falls in. Dated seasons (S53+) are used directly;
    earlier dates are extrapolated backwards at the rulebook cadence of
    8 weeks per season from the first dated start.
    """
    known = starts.dropna().sort_values()
    if known.empty:
        return pd.Series(np.nan, index=dates.index)
    first_season, first_start = int(known.index[0]), known.iloc[0]
    out = data.season_of_date(dates, starts)
    before = dates < first_start
    weeks_back = (first_start - dates[before]).dt.days / 7.0
    out[before] = first_season - np.ceil(weeks_back / config.WEEKS_PER_SEASON)
    return out


def retirement_hazard(players: pd.DataFrame, player_games: pd.DataFrame, current_season: int) -> pd.Series:
    """
    P(retire after completing career season c), c = 1..13.

    Career length for retired players is taken from the game logs (last
    regular season with a recorded game), because `retirement_date` is
    missing or a placeholder for most of them. Numerator: retired players
    whose career lasted exactly c seasons. Denominator: retired players with
    careers >= c, plus active players who have completed more than c seasons
    (those with exactly c are censored — still deciding). Rule-forced
    retirement makes the curve hit 1 at 13.
    """
    p = players[players["draft_season"].notna()].copy()
    if player_games.empty:
        return pd.Series({c: 0.0 for c in range(1, config.RETIRE_AFTER_SEASONS)} | {config.RETIRE_AFTER_SEASONS: 1.0})
    last = (player_games[player_games["season_state"] == "RegularSeason"]
            .groupby("pid")["season"].max().rename("last_season"))
    p = p.merge(last, on="pid", how="left")
    p["length"] = np.where(p["is_retired"], p["last_season"] - p["draft_season"] + 1, np.nan)
    p["completed"] = current_season - p["draft_season"]   # seasons finished so far
    # Only players whose whole ISFL career is inside the game-log window,
    # otherwise a truncated log looks like a short career.
    min_logged = int(player_games["season"].min())
    retired = p[p["is_retired"] & p["length"].notna() & (p["length"] >= 1)
                & (p["draft_season"] >= min_logged)]
    active = p[p["is_active"]]
    rows = {}
    for c in range(1, config.RETIRE_AFTER_SEASONS + 1):
        num = int((retired["length"] == c).sum())
        den = int((retired["length"] >= c).sum()) + int((active["completed"] > c).sum())
        rows[c] = num / den if den else np.nan
    s = pd.Series(rows)
    s.loc[config.RETIRE_AFTER_SEASONS] = 1.0
    return s.clip(0, 1)
