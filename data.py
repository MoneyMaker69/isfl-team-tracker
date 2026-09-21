"""
Read the committed cache into DataFrames. No network, ever.

Every loader validates the columns it needs and returns a Notice rather than
raising, so a stale or partial cache degrades a page instead of killing the app.
`load_all()` is the single entry point the app uses; tests call the individual
loaders against a fixture directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config


@dataclass
class Notice:
    level: str            # "error" | "warning" | "info"
    title: str
    detail: str = ""


@dataclass
class Cache:
    """Everything the app reads, loaded once and cached by Streamlit."""

    seasons: pd.DataFrame
    standings: pd.DataFrame
    games: pd.DataFrame
    players: pd.DataFrame
    attributes: pd.DataFrame
    regressions: pd.DataFrame
    bots: pd.DataFrame
    gm_history: pd.DataFrame
    gm_records: pd.DataFrame
    managers: pd.DataFrame
    draft_picks: pd.DataFrame
    awards: pd.DataFrame
    tpe_events: pd.DataFrame
    player_games: pd.DataFrame
    meta: dict
    notices: list[Notice] = field(default_factory=list)

    @property
    def current_season(self) -> int:
        if self.meta.get("current_season"):
            return int(self.meta["current_season"])
        if not self.standings.empty:
            return int(self.standings["season"].max()) + 1
        return 0

    @property
    def has_logs(self) -> bool:
        return not self.tpe_events.empty and not self.player_games.empty

    @property
    def teams(self) -> list[str]:
        """Current franchises, from the latest season with standings."""
        if self.standings.empty:
            return sorted(sum(config.CONFERENCES.values(), []))
        last = self.standings[self.standings["season"] == self.standings["season"].max()]
        return sorted(last["team"].unique())

    @property
    def conferences(self) -> dict[str, str]:
        """team -> conference, from the latest standings, with config fallback."""
        out = {t: c for c, ts in config.CONFERENCES.items() for t in ts}
        if not self.standings.empty:
            last = self.standings[self.standings["season"] == self.standings["season"].max()]
            out.update(dict(zip(last["team"], last["conference"])))
        return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

REQUIRED = {
    "standings.csv": ["season", "team", "conference", "wins", "losses", "pf", "pa", "playoff_result"],
    "games.csv": ["season", "phase", "seq", "week", "home", "away", "winner"],
    "players.csv": ["pid", "name", "draft_season", "status", "position", "isfl_team", "total_tpe"],
    "regressions.csv": ["season", "pid", "old_tpe", "pct", "new_tpe"],
    "gm_history.csv": ["season", "league", "team", "uid", "username"],
    "tpe_events.csv": ["pid", "date", "task_type", "tpe_change"],
    "player_games.csv": ["pid", "season", "season_state", "week", "team"],
}

OPTIONAL = ["seasons.csv", "attributes.csv", "bots.csv", "gm_records.csv", "managers.csv",
            "draft_picks.csv", "awards.csv"]


def _read(path: Path, notices: list[Notice], required: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        if required:
            notices.append(Notice("warning", f"{path.name} is missing",
                                  "Run `python scripts/build_cache.py` to build the cache."))
        return pd.DataFrame(columns=required or [])
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 — surface anything, never crash
        notices.append(Notice("error", f"{path.name} could not be read", str(exc)))
        return pd.DataFrame(columns=required or [])
    if required:
        missing = [c for c in required if c not in frame.columns]
        if missing:
            notices.append(Notice("error", f"{path.name} has an unexpected schema",
                                  f"Missing columns: {', '.join(missing)}. Rebuild the cache."))
            return pd.DataFrame(columns=required)
    return frame


def load_all(cache_dir: Path | None = None) -> Cache:
    cache_dir = cache_dir or config.CACHE_DIR
    notices: list[Notice] = []

    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    frames = {}
    for name, cols in REQUIRED.items():
        frames[name] = _read(cache_dir / name, notices, cols)
    for name in OPTIONAL:
        frames[name] = _read(cache_dir / name, notices, None)

    cache = Cache(
        seasons=frames["seasons.csv"],
        standings=_prep_standings(frames["standings.csv"]),
        games=_prep_games(frames["games.csv"]),
        players=_prep_players(frames["players.csv"]),
        attributes=frames["attributes.csv"],
        regressions=frames["regressions.csv"],
        bots=_prep_bots(frames["bots.csv"]),
        gm_history=_prep_gm_history(frames["gm_history.csv"]),
        gm_records=frames["gm_records.csv"],
        managers=frames["managers.csv"],
        draft_picks=frames["draft_picks.csv"],
        awards=frames["awards.csv"],
        tpe_events=_prep_events(frames["tpe_events.csv"]),
        player_games=_prep_player_games(frames["player_games.csv"]),
        meta=meta,
        notices=notices,
    )

    cache.gm_history = _append_current_gms(cache.gm_history, cache.managers, cache.current_season)

    if cache.standings.empty:
        notices.append(Notice("error", "No standings in the cache",
                              "Every page needs standings. Run the fast stage of the builder."))
    if not cache.has_logs:
        notices.append(Notice("info", "Player logs not built yet",
                              "TPE, roster and projection pages need the slow stage: "
                              "`python scripts/build_cache.py --stage slow`."))
    return cache


# ---------------------------------------------------------------------------
# Preparation — add the derived columns every page needs
# ---------------------------------------------------------------------------


def franchise(team: pd.Series) -> pd.Series:
    """Map relocated abbreviations to the current franchise code."""
    return team.map(lambda t: config.FRANCHISE_MAP.get(t, t))


def _prep_standings(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["franchise"] = franchise(df["team"])
    df["games"] = df["wins"] + df["losses"] + df["ties"].fillna(0)
    df["win_pct"] = (df["wins"] + 0.5 * df["ties"].fillna(0)) / df["games"].replace(0, np.nan)
    df["margin"] = (df["pf"] - df["pa"]) / df["games"].replace(0, np.nan)
    df["made_playoffs"] = df["playoff_result"] != "missed"
    df["champion"] = df["playoff_result"] == "champion"
    return df


def _prep_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["home_f"] = franchise(df["home"])
    df["away_f"] = franchise(df["away"])
    # 1 = home win, 0 = away win, 0.5 = tie or unknown
    df["home_result"] = df["winner"].map({"home": 1.0, "away": 0.0}).fillna(0.5)
    return df.sort_values(["season", "seq"]).reset_index(drop=True)


def _prep_players(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["draft_season"] = pd.to_numeric(df["draft_season"], errors="coerce")
    df["pos_group"] = df["position"].map(config.POSITION_GROUP)
    for c in ("total_tpe", "applied_tpe", "banked_tpe", "highest_tpe"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["is_active"] = df["status"].eq("active")
    df["is_retired"] = df["status"].eq("retired")
    return df


def _prep_bots(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["tier_tpe"] = pd.to_numeric(df["tier"], errors="coerce")
    # DSFL filler bots carry their TPE in the tier column already; ISFL OL
    # bots carry a 0-4 tier index. Anything <= 4 is a tier index.
    idx = df["tier_tpe"] <= 4
    df.loc[idx, "tier_tpe"] = df.loc[idx, "tier_tpe"].map(config.BOT_TIER_TPE)
    df["pos_group"] = df["position"].map(config.POSITION_GROUP)
    return df


def _prep_gm_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["franchise"] = franchise(df["team"])
    return df


def _append_current_gms(history: pd.DataFrame, managers: pd.DataFrame, current: int) -> pd.DataFrame:
    """
    /gm-history lags a season: the current season's GMs are only in
    /manager. Add them as history rows for `current` when it has none, so
    tenure, GM Elo and the GM factor see this season's pairs.
    """
    if managers.empty or current == 0:
        return history
    if not history.empty and (history["season"] == current).any():
        return history
    cols = {"team": "team", "league": "league", "uid": "uid", "username": "username"}
    if not all(c in managers.columns for c in cols):
        return history
    rows = managers[list(cols)].copy()
    rows["season"] = current
    rows["id"] = -1
    rows["franchise"] = franchise(rows["team"])
    return pd.concat([history, rows[history.columns] if not history.empty else rows], ignore_index=True)


def _prep_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tpe_change"] = pd.to_numeric(df["tpe_change"], errors="coerce").fillna(0)
    df["task_type"] = df["task_type"].fillna("")
    df["is_earning"] = df["task_type"].isin(config.EARNING_TASK_TYPES)
    df["is_regression"] = df["task_type"].isin(config.REGRESSION_TASK_TYPES)
    df["is_snapshot"] = df["task_type"].isin(config.SNAPSHOT_TASK_TYPES)
    return df


def _prep_player_games(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["team"] = df["team"].astype(str).str.strip()
    df["franchise"] = franchise(df["team"])
    return df


# ---------------------------------------------------------------------------
# Season boundaries
# ---------------------------------------------------------------------------


def season_starts(cache: Cache) -> pd.Series:
    """
    season -> start timestamp, for the seasons where the API gives one (S53+).

    The builder fills earlier seasons with NaT. Callers that need a boundary
    before that should treat the data as unavailable, not extrapolate.
    """
    if cache.seasons.empty or "start_date" not in cache.seasons.columns:
        return pd.Series(dtype="datetime64[ns]")
    s = cache.seasons.set_index("season")["start_date"]
    return pd.to_datetime(s, errors="coerce")


def season_of_date(dates: pd.Series, starts: pd.Series) -> pd.Series:
    """Assign each timestamp to the season whose start precedes it."""
    known = starts.dropna().sort_values()
    if known.empty:
        return pd.Series(np.nan, index=dates.index)
    bins = known.values
    idx = np.searchsorted(bins, dates.values, side="right") - 1
    out = np.where(idx >= 0, known.index.values[np.clip(idx, 0, len(bins) - 1)], np.nan)
    return pd.Series(out, index=dates.index)
