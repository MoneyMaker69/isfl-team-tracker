"""
Team ratings: game-level Elo, schedule-adjusted margin (SRS) and Pythagorean
expectation.

All three are computed on franchises (relocations mapped), from S1 so the
ratings have their run-up, but only scored and tuned from
config.ENGINE_SWITCH_SEASON onward.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import config


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


@dataclass
class EloResult:
    history: pd.DataFrame      # season, franchise, elo_start, elo_end, change
    game_log: pd.DataFrame     # per game: pre-ratings, expected, result
    current: dict[str, float]  # franchise -> rating after the last game
    k: float
    home_adv: float
    revert: float


def expected(home_elo: float, away_elo: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(home_elo + home_adv - away_elo) / 400.0))


def run_elo(
    games: pd.DataFrame,
    *,
    k: float = config.ELO_K,
    home_adv: float = config.ELO_HOME_ADVANTAGE,
    revert: float = config.ELO_SEASON_REVERT,
    playoffs: bool = True,
    through_season: int | None = None,
    log: bool = True,
) -> EloResult:
    """
    Sequential Elo over every game in `games` (already sorted by season, seq).

    - New franchises enter at ELO_EXPANSION_START, except in the first season.
    - At each season boundary every rating reverts `revert` of the way to 1500.
    - `through_season` stops after that season (used by the backtest).
    """
    g = games if playoffs else games[games["phase"] == "regular"]
    if through_season is not None:
        g = g[g["season"] <= through_season]
    ratings: dict[str, float] = {}
    first_season = int(g["season"].min()) if not g.empty else 0
    hist_rows: list[dict] = []
    log_rows: list[dict] = []
    season_start: dict[str, float] = {}
    last_season = None

    for row in g.itertuples(index=False):
        season = int(row.season)
        if season != last_season:
            if last_season is not None:
                for team, r in ratings.items():
                    hist_rows.append({"season": last_season, "franchise": team,
                                      "elo_start": season_start.get(team, r), "elo_end": r})
                    ratings[team] = config.ELO_START + (r - config.ELO_START) * (1 - revert)
            season_start = dict(ratings)
            last_season = season
        h, a = row.home_f, row.away_f
        for team in (h, a):
            if team not in ratings:
                ratings[team] = config.ELO_START if season == first_season else config.ELO_EXPANSION_START
                season_start[team] = ratings[team]
        eh, ea = ratings[h], ratings[a]
        exp_home = expected(eh, ea, home_adv)
        result = float(row.home_result)
        delta = k * (result - exp_home)
        ratings[h] = eh + delta
        ratings[a] = ea - delta
        if log:
            log_rows.append({"season": season, "phase": row.phase, "week": row.week,
                             "home": h, "away": a, "home_elo": eh, "away_elo": ea,
                             "expected": exp_home, "result": result})

    if last_season is not None:
        for team, r in ratings.items():
            hist_rows.append({"season": last_season, "franchise": team,
                              "elo_start": season_start.get(team, r), "elo_end": r})

    history = pd.DataFrame(hist_rows)
    if not history.empty:
        history["change"] = history["elo_end"] - history["elo_start"]
        history = history.sort_values(["season", "elo_end"], ascending=[True, False]).reset_index(drop=True)
        history["rank"] = history.groupby("season")["elo_end"].rank(ascending=False, method="min").astype(int)
    return EloResult(history=history, game_log=pd.DataFrame(log_rows), current=ratings,
                     k=k, home_adv=home_adv, revert=revert)


def preseason_elo(result: EloResult, season: int) -> pd.Series:
    """
    Rating each franchise carries *into* `season`: last season's end, reverted.

    This is what a preseason prediction may legitimately know.
    """
    prev = result.history[result.history["season"] == season - 1]
    if prev.empty:
        return pd.Series(dtype=float)
    reverted = config.ELO_START + (prev["elo_end"] - config.ELO_START) * (1 - result.revert)
    return pd.Series(reverted.values, index=prev["franchise"].values)


def log_loss(game_log: pd.DataFrame, from_season: int = config.ENGINE_SWITCH_SEASON) -> float:
    g = game_log[(game_log["season"] >= from_season) & (game_log["phase"] == "regular")]
    p = g["expected"].clip(1e-6, 1 - 1e-6)
    y = g["result"]
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def tune_elo(games: pd.DataFrame, from_season: int = config.ENGINE_SWITCH_SEASON) -> pd.DataFrame:
    """
    Grid search K, home advantage and offseason reversion by regular-season
    log-loss from `from_season`. Returns the grid sorted best-first.
    """
    grid = []
    for k, hfa, rev in product((20, 28, 36, 44, 52, 60, 72, 84),
                               (0, 20, 35, 50, 65),
                               (0.0, 0.2, 1 / 3, 0.5, 0.65)):
        res = run_elo(games, k=k, home_adv=hfa, revert=rev)
        grid.append({"k": k, "home_adv": hfa, "revert": round(rev, 3),
                     "log_loss": log_loss(res.game_log, from_season)})
    return pd.DataFrame(grid).sort_values("log_loss").reset_index(drop=True)


def calibration(game_log: pd.DataFrame, bins: int = 10,
                from_season: int = config.ENGINE_SWITCH_SEASON) -> pd.DataFrame:
    """Expected vs observed home win rate by probability bucket."""
    g = game_log[(game_log["season"] >= from_season) & (game_log["phase"] == "regular")].copy()
    if g.empty:
        return pd.DataFrame(columns=["bucket", "expected", "observed", "games"])
    g["bucket"] = pd.cut(g["expected"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    out = g.groupby("bucket", observed=True).agg(
        expected=("expected", "mean"), observed=("result", "mean"), games=("result", "size")
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    return out


# ---------------------------------------------------------------------------
# SRS — schedule-adjusted point margin
# ---------------------------------------------------------------------------


def srs_season(standings: pd.DataFrame, games: pd.DataFrame, season: int) -> pd.Series:
    """
    Solve rating_i = margin_i + mean(rating of opponents_i) for one season.

    Uses regular-season games only. Margin is points per game from PF/PA. The
    system is singular (ratings are relative) so it's solved in least squares
    with a sum-to-zero constraint appended.
    """
    st = standings[standings["season"] == season]
    g = games[(games["season"] == season) & (games["phase"] == "regular")]
    teams = sorted(st["franchise"].unique())
    if len(teams) < 2 or g.empty:
        return pd.Series(dtype=float)
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    margin = np.zeros(n)
    played = np.zeros(n)
    opp = np.zeros((n, n))
    for row in st.itertuples(index=False):
        i = idx[row.franchise]
        margin[i] = row.margin if not np.isnan(row.margin) else 0.0
    for row in g.itertuples(index=False):
        if row.home_f in idx and row.away_f in idx:
            h, a = idx[row.home_f], idx[row.away_f]
            opp[h, a] += 1
            opp[a, h] += 1
            played[h] += 1
            played[a] += 1
    played[played == 0] = 1
    A = np.eye(n) - opp / played[:, None]
    A = np.vstack([A, np.ones(n)])
    b = np.append(margin, 0.0)
    r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pd.Series(r, index=teams)


def srs_all(standings: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(standings["season"].unique()):
        r = srs_season(standings, games, int(season))
        for team, val in r.items():
            rows.append({"season": int(season), "franchise": team, "srs": float(val)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rank"] = out.groupby("season")["srs"].rank(ascending=False, method="min").astype(int)
    return out


# ---------------------------------------------------------------------------
# Pythagorean expectation
# ---------------------------------------------------------------------------


def pythagorean(pf: pd.Series, pa: pd.Series, exponent: float) -> pd.Series:
    pf = pf.astype(float).clip(lower=1)
    pa = pa.astype(float).clip(lower=1)
    return pf ** exponent / (pf ** exponent + pa ** exponent)


def fit_pythagorean_exponent(standings: pd.DataFrame,
                             from_season: int = config.ENGINE_SWITCH_SEASON) -> float:
    st = standings[(standings["season"] >= from_season) & standings["win_pct"].notna()]
    if st.empty:
        return config.PYTHAGOREAN_EXPONENT

    def sse(k: float) -> float:
        return float(((pythagorean(st["pf"], st["pa"], k) - st["win_pct"]) ** 2).sum())

    res = minimize_scalar(sse, bounds=(0.5, 8.0), method="bounded")
    return float(res.x)


def with_pythagorean(standings: pd.DataFrame, exponent: float) -> pd.DataFrame:
    """Add pyth_pct, pyth_wins and luck (actual wins minus Pythagorean wins)."""
    out = standings.copy()
    out["pyth_pct"] = pythagorean(out["pf"], out["pa"], exponent)
    out["pyth_wins"] = out["pyth_pct"] * out["games"]
    out["luck"] = out["wins"] + 0.5 * out["ties"].fillna(0) - out["pyth_wins"]
    return out


# ---------------------------------------------------------------------------
# Combined season table
# ---------------------------------------------------------------------------


def season_ratings(standings: pd.DataFrame, elo: EloResult, srs: pd.DataFrame,
                   exponent: float) -> pd.DataFrame:
    """One row per franchise-season with every rating side by side. Seasons
    with no games played yet (the current preseason) are left out."""
    st = with_pythagorean(standings[standings["games"] > 0], exponent)
    cols = ["season", "franchise", "team", "conference", "wins", "losses", "ties", "win_pct",
            "pf", "pa", "margin", "pyth_pct", "pyth_wins", "luck", "playoff_result",
            "made_playoffs", "champion"]
    out = st[cols].merge(elo.history[["season", "franchise", "elo_start", "elo_end", "change"]],
                         on=["season", "franchise"], how="left")
    out = out.merge(srs[["season", "franchise", "srs"]], on=["season", "franchise"], how="left")
    return out.sort_values(["season", "elo_end"], ascending=[True, False]).reset_index(drop=True)
