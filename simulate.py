"""
Season simulation.

Given a strength (points above average) per franchise, a schedule and a win
model, run N seasons and count finishes. The playoff format is the real one:
top three per conference, #1 seeds bye, 2 v 3, winner at 1, conference
champions meet in the Ultimus.

The win model is logistic in the strength difference: P(home) =
1 / (1 + exp(-(s_home - s_away + hfa) / scale)). `scale` and `hfa` are fitted
from history in `fit_win_model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config


# ---------------------------------------------------------------------------
# Win model
# ---------------------------------------------------------------------------


@dataclass
class WinModel:
    scale: float
    hfa: float
    log_loss: float = float("nan")
    n_games: int = 0

    def p_home(self, diff: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(diff + self.hfa) / self.scale))


def fit_win_model(games: pd.DataFrame, strength: pd.DataFrame,
                  from_season: int = config.ENGINE_SWITCH_SEASON) -> WinModel:
    """
    `strength`: season, franchise, value. Fits scale and hfa by maximum
    likelihood on regular-season results. Pass end-of-season SRS for the
    in-sample model, or backtest predictions for the honest preseason one.
    """
    g = games[(games["phase"] == "regular") & (games["season"] >= from_season)]
    s = strength.set_index(["season", "franchise"])["value"]
    idx_h = pd.MultiIndex.from_arrays([g["season"], g["home_f"]])
    idx_a = pd.MultiIndex.from_arrays([g["season"], g["away_f"]])
    sh = s.reindex(idx_h).to_numpy()
    sa = s.reindex(idx_a).to_numpy()
    ok = ~(np.isnan(sh) | np.isnan(sa))
    diff = (sh - sa)[ok]
    y = g["home_result"].to_numpy()[ok]
    if len(y) < 50:
        return WinModel(scale=10.0, hfa=config.SRS_HOME_ADVANTAGE)

    def nll(params: np.ndarray) -> float:
        scale, hfa = max(params[0], 0.5), params[1]
        p = 1.0 / (1.0 + np.exp(-(diff + hfa) / scale))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())

    res = minimize(nll, x0=np.array([8.0, 2.0]), method="Nelder-Mead")
    scale, hfa = max(res.x[0], 0.5), res.x[1]
    return WinModel(scale=float(scale), hfa=float(hfa),
                    log_loss=float(res.fun / len(y)), n_games=int(len(y)))


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def generate_schedule(conferences: dict[str, str], rng: np.random.Generator) -> pd.DataFrame:
    """
    A full 16-game schedule with the league's structure: every conference
    opponent home and away, plus four cross-conference games on a rotating
    (here random) circulant pairing so every team gets exactly four.
    """
    confs: dict[str, list[str]] = {}
    for team, conf in conferences.items():
        confs.setdefault(conf, []).append(team)
    rows = []
    for teams in confs.values():
        for a, b in combinations(sorted(teams), 2):
            rows.append({"home": a, "away": b})
            rows.append({"home": b, "away": a})
    names = list(confs)
    if len(names) == 2 and len(confs[names[0]]) == len(confs[names[1]]):
        left = list(confs[names[0]])
        right = list(confs[names[1]])
        rng.shuffle(left)
        rng.shuffle(right)
        n = len(left)
        offsets = rng.choice(n, size=min(config.CROSS_CONFERENCE_GAMES, n), replace=False)
        for k, o in enumerate(offsets):
            for i in range(n):
                a, b = left[i], right[(i + int(o)) % n]
                rows.append({"home": a, "away": b} if (k + i) % 2 == 0 else {"home": b, "away": a})
    sched = pd.DataFrame(rows)
    sched["played"] = False
    sched["home_result"] = np.nan
    return sched.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


def complete_schedule(played: pd.DataFrame, conferences: dict[str, str],
                      rng: np.random.Generator) -> pd.DataFrame:
    """
    Mid-season: keep the games already played and fill in the rest.

    Conference pairs still owe each other 2 − played games. Cross-conference
    games owed per team are 4 − played; unplayed cross pairs are matched
    randomly to satisfy those counts (greedy with retries; on failure the
    last few are filled loosely, which affects at most a game or two).
    """
    if played.empty:
        return generate_schedule(conferences, rng)
    teams = sorted(conferences)
    rows = [{"home": r.home_f, "away": r.away_f, "played": True, "home_result": r.home_result}
            for r in played.itertuples(index=False)]
    pair_count: dict[tuple[str, str], int] = {}
    cross_played: dict[str, int] = {t: 0 for t in teams}
    for r in played.itertuples(index=False):
        key = tuple(sorted((r.home_f, r.away_f)))
        pair_count[key] = pair_count.get(key, 0) + 1
        if conferences.get(r.home_f) != conferences.get(r.away_f):
            cross_played[r.home_f] += 1
            cross_played[r.away_f] += 1

    for a, b in combinations(teams, 2):
        if conferences[a] != conferences[b]:
            continue
        owed = 2 - pair_count.get((a, b), 0)
        for k in range(max(0, owed)):
            rows.append({"home": a if k == 0 else b, "away": b if k == 0 else a,
                         "played": False, "home_result": np.nan})

    need = {t: max(0, config.CROSS_CONFERENCE_GAMES - cross_played[t]) for t in teams}
    confs = sorted(set(conferences.values()))
    if len(confs) == 2:
        left = [t for t in teams if conferences[t] == confs[0]]
        right = [t for t in teams if conferences[t] == confs[1]]
        best = None
        for _ in range(200):
            n = dict(need)
            picks = []
            ok = True
            order = list(left)
            rng.shuffle(order)
            for a in order:
                cands = [b for b in right if n[b] > 0 and pair_count.get(tuple(sorted((a, b))), 0) == 0
                         and (a, b) not in picks and (b, a) not in picks]
                rng.shuffle(cands)
                cands.sort(key=lambda b: -n[b])
                while n[a] > 0 and cands:
                    b = cands.pop(0)
                    picks.append((a, b))
                    n[a] -= 1
                    n[b] -= 1
                if n[a] > 0:
                    ok = False
            short = sum(n.values())
            if best is None or short < best[0]:
                best = (short, picks)
            if ok and short == 0:
                break
        for k, (a, b) in enumerate(best[1]):
            rows.append({"home": a if k % 2 == 0 else b, "away": b if k % 2 == 0 else a,
                         "played": False, "home_result": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------


@dataclass
class SimResult:
    table: pd.DataFrame         # per franchise: probabilities and expected wins
    rank_matrix: pd.DataFrame   # franchise × finish position (within conference)
    n_sims: int
    win_model: WinModel


def simulate(strength: pd.Series, schedule: pd.DataFrame, conferences: dict[str, str],
             win_model: WinModel, n_sims: int = config.SIMULATIONS,
             seed: int = config.SIM_SEED) -> SimResult:
    rng = np.random.default_rng(seed)
    teams = sorted(strength.index)
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    s = strength.reindex(teams).fillna(0.0).to_numpy()

    sched = schedule[schedule["home"].isin(idx) & schedule["away"].isin(idx)].reset_index(drop=True)
    home_i = sched["home"].map(idx).to_numpy()
    away_i = sched["away"].map(idx).to_numpy()
    p = win_model.p_home(s[home_i] - s[away_i])
    played = sched["played"].to_numpy(dtype=bool)
    fixed = sched["home_result"].to_numpy(dtype=float)

    draws = rng.random((n_sims, len(sched))) < p
    if played.any():
        draws[:, played] = fixed[played][None, :] >= 0.5
    same_conf = np.array([conferences.get(sched.at[g, "home"]) == conferences.get(sched.at[g, "away"])
                          for g in range(len(sched))])

    wins = np.zeros((n_sims, T))
    conf_wins = np.zeros((n_sims, T))
    for g in range(len(sched)):
        hw = draws[:, g]
        wins[:, home_i[g]] += hw
        wins[:, away_i[g]] += ~hw
        if same_conf[g]:
            conf_wins[:, home_i[g]] += hw
            conf_wins[:, away_i[g]] += ~hw

    # Ranking key: wins, then conference wins, then a coin (PF is not simulated).
    key = wins * 1000 + conf_wins * 10 + rng.random((n_sims, T))
    conf_names = sorted(set(conferences.get(t, "?") for t in teams))
    conf_members = {c: np.array([idx[t] for t in teams if conferences.get(t) == c]) for c in conf_names}

    rank_counts = np.zeros((T, T))   # team × position (1-indexed at column pos-1)
    playoffs = np.zeros(T)
    byes = np.zeros(T)
    conf_champ = np.zeros(T)
    ultimus = np.zeros(T)
    seeds_by_conf: dict[str, np.ndarray] = {}
    for c, members in conf_members.items():
        order = np.argsort(-key[:, members], axis=1)          # positions within conference
        seeded = members[order]                               # team index by seed
        seeds_by_conf[c] = seeded
        for pos in range(len(members)):
            np.add.at(rank_counts[:, pos], seeded[:, pos], 1)
        k = min(config.PLAYOFF_TEAMS_PER_CONFERENCE, len(members))
        for pos in range(k):
            np.add.at(playoffs, seeded[:, pos], 1)
        np.add.at(byes, seeded[:, 0], 1)

    def game(home: np.ndarray, away: np.ndarray, neutral: bool = False) -> np.ndarray:
        d = s[home] - s[away]
        ph = 1.0 / (1.0 + np.exp(-(d + (0.0 if neutral else win_model.hfa)) / win_model.scale))
        return np.where(rng.random(len(home)) < ph, home, away)

    champs = []
    for c in conf_names:
        seeded = seeds_by_conf[c]
        if seeded.shape[1] < 3:
            champs.append(seeded[:, 0])
            continue
        w23 = game(seeded[:, 1], seeded[:, 2])
        cc = game(seeded[:, 0], w23)
        np.add.at(conf_champ, cc, 1)
        champs.append(cc)
    if len(champs) == 2:
        # Ultimus: home field to the better record, coin on a tie.
        a, b = champs
        a_home = key[np.arange(n_sims), a] >= key[np.arange(n_sims), b]
        home = np.where(a_home, a, b)
        away = np.where(a_home, b, a)
        winner = game(home, away)
        np.add.at(ultimus, winner, 1)

    best = wins.max(axis=1, keepdims=True)
    tied_best = (wins == best)
    best_prob = (tied_best / tied_best.sum(axis=1, keepdims=True)).mean(axis=0)

    table = pd.DataFrame({
        "franchise": teams,
        "conference": [conferences.get(t) for t in teams],
        "strength": s,
        "exp_wins": wins.mean(axis=0),
        "wins_p10": np.percentile(wins, 10, axis=0),
        "wins_p90": np.percentile(wins, 90, axis=0),
        "p_playoffs": playoffs / n_sims,
        "p_bye": byes / n_sims,
        "p_best_record": best_prob,
        "p_conf_champ": conf_champ / n_sims,
        "p_ultimus": ultimus / n_sims,
    })
    n_pos = max(len(m) for m in conf_members.values())
    rank_matrix = pd.DataFrame(rank_counts[:, :n_pos] / n_sims, index=teams,
                               columns=[f"{i + 1}" for i in range(n_pos)])
    table["modal_rank"] = rank_matrix.to_numpy().argmax(axis=1) + 1
    table["exp_rank"] = rank_matrix.to_numpy() @ np.arange(1, n_pos + 1)
    table = table.sort_values(["conference", "exp_rank"]).reset_index(drop=True)
    return SimResult(table=table, rank_matrix=rank_matrix, n_sims=n_sims, win_model=win_model)


def ranking_for_task(result: SimResult) -> dict[str, list[str]]:
    """The per-conference order to paste into the prediction task."""
    out = {}
    for conf, grp in result.table.groupby("conference"):
        out[str(conf)] = grp.sort_values("exp_rank")["franchise"].tolist()
    return out
