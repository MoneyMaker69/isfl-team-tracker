"""
Where is each org heading? Per-player forward projection rolled up to teams.

Per player, per future offseason:
  1. add this season's earnings (measured rate, or the league median for the
     career season; zero for inactive players),
  2. apply the regression table for the seasons finished,
  3. retire on the rule (13 seasons, or under 150 TPE after regression),
  4. optionally retire by the empirical hazard (Monte Carlo).

Rates are held fixed across the horizon — no decay toward the mean. That is
a deliberate choice carried over from the SSL tracker, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config


@dataclass
class ProjectionResult:
    players: pd.DataFrame     # pid, franchise, name, pos_group, season, tpe (deterministic path), retired_rule
    teams: pd.DataFrame       # franchise, season, tpe_mean, tpe_total, n_players, p10/p50/p90 bands
    horizon: int
    n_sims: int


def season_remaining(seasons: pd.DataFrame, current_season: int, now: pd.Timestamp | None = None) -> float:
    """Fraction of the current season still to be played (for step-1 earnings)."""
    if seasons.empty or "start_date" not in seasons.columns:
        return 0.5
    row = seasons[seasons["season"] == current_season]
    if row.empty:
        return 0.5
    start = pd.to_datetime(row["start_date"].iloc[0], errors="coerce")
    end = pd.to_datetime(row["end_date"].iloc[0], errors="coerce")
    if pd.isna(start) or pd.isna(end) or end <= start:
        return 0.5
    now = now or pd.Timestamp.now()
    return float(np.clip((end - now) / (end - start), 0.0, 1.0))


def project(roster_now: pd.DataFrame, rates: pd.DataFrame, league_rate: pd.Series,
            hazard: pd.Series, current_season: int, *, horizon: int = config.DEFAULT_HORIZON,
            season_left: float = 0.5, hazard_scale: float = 1.0, n_sims: int = 400,
            seed: int = 7) -> ProjectionResult:
    """
    `roster_now`: current-season rows from roster.roster_with_tpe (pid,
    franchise, name, pos_group, draft_season, tpe, active).
    """
    r = roster_now[roster_now["draft_season"].notna() & roster_now["tpe"].notna()].copy()
    r = r.merge(rates, on="pid", how="left")
    career_now = current_season - r["draft_season"] + 1
    fallback = career_now.map(lambda c: league_rate.get(int(c), league_rate.median() if len(league_rate) else 150.0))
    r["rate"] = r["rate"].fillna(fallback)
    r.loc[~r["active"].astype(bool), "rate"] = 0.0

    P = len(r)
    tpe = r["tpe"].to_numpy(dtype=float)
    draft = r["draft_season"].to_numpy(dtype=float)
    rate = r["rate"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    alive_det = np.ones(P, dtype=bool)
    alive_mc = np.ones((n_sims, P), dtype=bool)

    player_rows = []
    team_rows = []
    for h in range(1, horizon + 1):
        season_played = current_season + h - 1
        frac = season_left if h == 1 else 1.0
        tpe = tpe + rate * frac
        finished = season_played - draft + 1
        pct = np.array([config.regression_pct(int(f)) if f > 0 else 0.0 for f in finished])
        tpe = np.floor(tpe * (1 - pct))
        forced = (finished >= config.RETIRE_AFTER_SEASONS) | ((pct > 0) & (tpe < config.REGRESSION_FLOOR_TPE))
        alive_det &= ~forced
        # Voluntary retirement by hazard, applied on top of the rule.
        hz = np.array([hazard.get(int(f), 0.0) if f > 0 else 0.0 for f in finished]) * hazard_scale
        hz = np.where(forced, 1.0, np.clip(hz, 0, 1))
        alive_mc &= rng.random((n_sims, P)) >= hz[None, :]

        target_season = season_played + 1
        for i in range(P):
            player_rows.append({"pid": int(r["pid"].iloc[i]), "franchise": r["franchise"].iloc[i],
                                "name": r["name"].iloc[i], "pos_group": r["pos_group"].iloc[i],
                                "season": target_season, "tpe": float(tpe[i]) if alive_det[i] else 0.0,
                                "alive": bool(alive_det[i]),
                                "p_alive": float(alive_mc[:, i].mean()),
                                "regression_pct": float(pct[i])})
        for team, idx in r.groupby("franchise").indices.items():
            t = tpe[idx]
            a = alive_mc[:, idx]
            n_alive = a.sum(axis=1)
            total = (a * t[None, :]).sum(axis=1)
            mean = np.where(n_alive > 0, total / np.maximum(n_alive, 1), 0.0)
            det = t[alive_det[idx]]
            team_rows.append({
                "franchise": team, "season": target_season,
                "tpe_mean_det": float(det.mean()) if len(det) else 0.0,
                "tpe_total_det": float(det.sum()), "n_players_det": int(len(det)),
                "tpe_mean_p10": float(np.percentile(mean, 10)),
                "tpe_mean_p50": float(np.percentile(mean, 50)),
                "tpe_mean_p90": float(np.percentile(mean, 90)),
                "tpe_total_p10": float(np.percentile(total, 10)),
                "tpe_total_p50": float(np.percentile(total, 50)),
                "tpe_total_p90": float(np.percentile(total, 90)),
                "n_players_p50": float(np.percentile(n_alive, 50)),
            })
    return ProjectionResult(players=pd.DataFrame(player_rows), teams=pd.DataFrame(team_rows),
                            horizon=horizon, n_sims=n_sims)


def projected_features(result: ProjectionResult, bots: pd.DataFrame, current_season: int) -> pd.DataFrame:
    """
    Team features for each projected season on the deterministic path, in
    the same shape as roster.team_features, so the roster model can score
    them. Bots are carried forward unchanged.
    """
    import roster as roster_mod
    p = result.players[result.players["alive"]].copy()
    p["career_season"] = np.nan
    p["active"] = True
    frames = []
    for season, grp in p.groupby("season"):
        b = bots[bots["season"] == current_season].copy() if not bots.empty else bots
        if not b.empty:
            b["season"] = season
        frames.append(roster_mod.team_features(grp, b))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def regression_calendar(roster_now: pd.DataFrame, current_season: int, horizon: int = 4) -> pd.DataFrame:
    """
    Per player: the regression % they face at the start of each of the next
    `horizon` seasons. For the NOLA page's "who loses what, when".
    """
    r = roster_now[roster_now["draft_season"].notna()].copy()
    for h in range(1, horizon + 1):
        finished = (current_season + h - 1) - r["draft_season"] + 1
        r[f"S{current_season + h}"] = finished.map(lambda f: config.regression_pct(int(f)) if f > 0 else 0.0)
    return r
