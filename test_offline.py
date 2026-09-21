"""
Offline analytics checks. No network, no Streamlit, no cache — every test
builds its own small frames.

    python test_offline.py          # or: pytest test_offline.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import backtest
import config
import data
import elo as elo_mod
import gm as gm_mod
import model as model_mod
import projection
import roster
import simulate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEAMS_N = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
TEAMS_A = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
CONF = {**{t: "NSFC" for t in TEAMS_N}, **{t: "ASFC" for t in TEAMS_A}}


def synthetic_games(seasons: int, seed: int = 1) -> pd.DataFrame:
    """Round-robin seasons where team i has strength (i-3)*3 points, plus noise."""
    rng = np.random.default_rng(seed)
    strength = {t: (i - 3) * 3.0 for i, t in enumerate(TEAMS_N)}
    strength.update({t: (i - 3) * 3.0 for i, t in enumerate(TEAMS_A)})
    rows = []
    for season in range(1, seasons + 1):
        sched = simulate.generate_schedule(CONF, rng)
        for seq, g in enumerate(sched.itertuples(index=False)):
            d = strength[g.home] - strength[g.away] + 2.0
            p = 1 / (1 + np.exp(-d / 8))
            rows.append({"season": season, "phase": "regular", "seq": seq, "week": seq // 7 + 1,
                         "home": g.home, "away": g.away, "home_score": None, "away_score": None,
                         "winner": "home" if rng.random() < p else "away"})
    return data._prep_games(pd.DataFrame(rows))


def synthetic_standings(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, g in games.groupby("season"):
        for t in TEAMS_N + TEAMS_A:
            home = g[g["home"] == t]
            away = g[g["away"] == t]
            w = int((home["winner"] == "home").sum() + (away["winner"] == "away").sum())
            l = len(home) + len(away) - w
            pf = 300 + 10 * w
            rows.append({"season": season, "team": t, "name": t, "location": t, "conference": CONF[t],
                         "wins": w, "losses": l, "ties": 0, "pct": w / 16, "pf": pf, "pa": 460 - pf,
                         "diff": 2 * pf - 460, "home_w": 0, "home_l": 0, "away_w": 0, "away_l": 0,
                         "conf_w": 0, "conf_l": 0, "playoff_result": "missed"})
    return data._prep_standings(pd.DataFrame(rows))


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "ok " if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


FAILS: list[str] = []


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def test_regression_table() -> None:
    check("no regression before 7 seasons", config.regression_pct(6) == 0.0)
    check("20% after 7", config.regression_pct(7) == 0.20)
    check("60% after 12", config.regression_pct(12) == 0.60)
    check("retired after 13", config.regression_pct(13) == 1.0)
    check("floor rounding matches API sample", int(np.floor(1443 * 0.20)) == 288)


def test_franchise_map() -> None:
    s = pd.Series(["LVL", "NOLA", "CHI", "XYZ"])
    out = data.franchise(s).tolist()
    check("relocations mapped", out == ["NOLA", "NOLA", "OSK", "XYZ"], str(out))


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


def test_elo() -> None:
    games = synthetic_games(6)
    res = elo_mod.run_elo(games, k=40, home_adv=40, revert=1 / 3)
    end = res.history[res.history["season"] == 6].set_index("franchise")["elo_end"]
    check("Elo orders synthetic strengths", end["N7"] > end["N4"] > end["N1"],
          f"{end['N7']:.0f} {end['N4']:.0f} {end['N1']:.0f}")
    check("Elo is zero-sum", abs(sum(res.current.values()) / len(res.current) - 1500) < 1e-6)
    pre = elo_mod.preseason_elo(res, 6)
    prev_end = res.history[res.history["season"] == 5].set_index("franchise")["elo_end"]
    expected = 1500 + (prev_end - 1500) * (2 / 3)
    check("preseason Elo reverts a third", np.allclose(pre.reindex(expected.index), expected))
    check("home advantage raises expectation", elo_mod.expected(1500, 1500, 40) > 0.5)
    loss = elo_mod.log_loss(res.game_log, from_season=1)
    check("Elo beats coin flip on synthetic data", loss < 0.69, f"{loss:.3f}")


def test_srs() -> None:
    # Three teams, one round: A beats B by 10, B beats C by 10, A beats C by 20.
    st = pd.DataFrame({"season": [1, 1, 1], "team": ["A", "B", "C"], "conference": ["X"] * 3,
                       "wins": [2, 1, 0], "losses": [0, 1, 2], "ties": [0, 0, 0], "pct": [1, .5, 0],
                       "pf": [60, 30, 10], "pa": [20, 30, 50], "diff": [40, 0, -40],
                       "playoff_result": ["missed"] * 3})
    st = data._prep_standings(st)
    g = pd.DataFrame({"season": [1, 1, 1], "phase": ["regular"] * 3, "seq": [0, 1, 2], "week": [1, 1, 2],
                      "home": ["A", "B", "A"], "away": ["B", "C", "C"], "home_score": None, "away_score": None,
                      "winner": ["home"] * 3})
    g = data._prep_games(g)
    r = elo_mod.srs_season(st, g, 1)
    check("SRS sums to zero", abs(r.sum()) < 1e-6)
    check("SRS order", r["A"] > r["B"] > r["C"])
    # Round robin of 3: r_A = 20 + mean(r_B, r_C) = 20 - r_A/2  →  r_A = 40/3.
    check("SRS solves the round-robin system", np.allclose(r.values, [40 / 3, 0, -40 / 3]), str(r.values))


def test_pythagorean() -> None:
    check("even scoring is .500", abs(elo_mod.pythagorean(pd.Series([300]), pd.Series([300]), 2.5).iloc[0] - 0.5) < 1e-9)
    check("more points, more wins", elo_mod.pythagorean(pd.Series([400]), pd.Series([300]), 2.5).iloc[0] > 0.5)


# ---------------------------------------------------------------------------
# TPE timeline
# ---------------------------------------------------------------------------


def _mini_cache() -> data.Cache:
    seasons = pd.DataFrame({"season": [60, 61, 62, 63],
                            "start_date": ["2026-03-21", "2026-05-16", "2026-07-11", "2026-09-05"],
                            "end_date": ["2026-05-08", "2026-07-03", "2026-08-28", "2026-10-23"], "ended": [1, 1, 1, 0]})
    events = pd.DataFrame([
        # pid 1: created S60 with 50, earns 100 in S60, 200 in S61, 150 in S62, regresses 20% at S63, earns 10 in S63
        {"pid": 1, "task_id": 1, "date": "2026-03-25", "task_type": "Create", "tpe_change": 50, "description": "Create"},
        {"pid": 1, "task_id": 2, "date": "2026-04-10", "task_type": "training", "tpe_change": 100, "description": "x"},
        {"pid": 1, "task_id": 3, "date": "2026-06-10", "task_type": "point task", "tpe_change": 200, "description": "x"},
        {"pid": 1, "task_id": 4, "date": "2026-08-01", "task_type": "activity check", "tpe_change": 150, "description": "x"},
        {"pid": 1, "task_id": 5, "date": "2026-09-05 14:31:06", "task_type": "regression", "tpe_change": -100, "description": "S63 Regression: 100 TPE"},
        {"pid": 1, "task_id": 6, "date": "2026-09-10", "task_type": "training", "tpe_change": 10, "description": "x"},
    ])
    players = pd.DataFrame([{"pid": 1, "name": "One", "username": "u1", "draft_season": 56, "status": "active",
                             "position": "Quarterback", "archetype": "x", "isfl_team": "N1", "current_league": "ISFL",
                             "total_tpe": 410, "applied_tpe": 400, "banked_tpe": 10, "highest_tpe": 500,
                             "retirement_date": None, "weekly_activity_check": 1, "weekly_training": 1}])
    empty = pd.DataFrame()
    return data.Cache(seasons=seasons, standings=empty, games=empty, players=data._prep_players(players),
                      attributes=empty, regressions=pd.DataFrame(columns=["season", "pid", "new_tpe"]),
                      bots=empty, gm_history=empty, gm_records=empty, managers=empty, draft_picks=empty,
                      awards=empty, tpe_events=data._prep_events(events),
                      player_games=pd.DataFrame(columns=["pid", "season", "season_state", "week", "team"]),
                      meta={"current_season": 63})


def test_tpe_timeline() -> None:
    c = _mini_cache()
    tl = roster.tpe_timeline(c).set_index("season")
    check("S60 start is the creation grant", tl.loc[60, "tpe_start"] == 50, str(tl.loc[60].to_dict()))
    check("S61 start = 150", tl.loc[61, "tpe_start"] == 150)
    check("S62 start = 350", tl.loc[62, "tpe_start"] == 350)
    check("regression labelled S63 lands at S63 start", tl.loc[63, "tpe_start"] == 400, str(tl.loc[63].to_dict()))
    check("S63 end equals total_tpe", tl.loc[63, "tpe_end"] == 410)
    rc = roster.reconcile(tl.reset_index(), c.players, 63)
    check("reconciliation passes", rc["rate"] == 1.0, str(rc))
    vr = roster.verify_regression_pcts(tl.reset_index(), c.players)
    check("regression pct verified (7 seasons finished → 20% of 500 = 100)", vr == {"checked": 1, "matched": 1}, str(vr))
    rates = roster.earning_rates(tl.reset_index(), 63, window=2)
    check("rate excludes first and current season (mean of 200,150)", rates["rate"].iloc[0] == 175, str(rates))


def test_team_features() -> None:
    rows = []
    for i in range(3):
        rows.append({"pid": i, "season": 63, "franchise": "N1", "pos_group": "OL", "tpe": 600 - 100 * i,
                     "career_season": 3, "active": True})
    rows.append({"pid": 9, "season": 63, "franchise": "N1", "pos_group": "QB", "tpe": 1000, "career_season": 8, "active": False})
    fr = pd.DataFrame(rows)
    bots = pd.DataFrame([{"pid": -1, "season": 63, "franchise": "N1", "pos_group": "OL", "tpe": 750, "name": "bot", "is_bot": True}])
    f = roster.team_features(fr, bots).iloc[0]
    check("OL starters pad missing slots with zero", f["OL_starters"] == (750 + 600 + 500 + 400) / 5, str(f["OL_starters"]))
    check("bots excluded from human mean", abs(f["tpe_mean"] - (600 + 500 + 400 + 1000) / 4) < 1e-9)
    check("inactive counted", f["n_inactive"] == 1)
    check("qb1", f["qb1"] == 1000)
    check("regression-years count", f["n_regression_years"] == 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_ridge_recovers_signal() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for season in range(54, 62):
        for t in range(14):
            x = {f"{g}_starters": rng.normal(700, 120) for g in config.POSITION_GROUPS}
            y = 0.02 * (x["QB_starters"] - 700) + 0.01 * (x["OL_starters"] - 700) + rng.normal(0, 1.5)
            rows.append({"season": season, "franchise": f"T{t}", "srs": y, **x})
    m = model_mod.train(pd.DataFrame(rows))
    check("model fits", m is not None)
    coef = m.coefficients.set_index("feature")["per_100_tpe"]
    check("QB effect ≈ +2 pts per 100", abs(coef["QB_starters"] - 2.0) < 0.6, f"{coef['QB_starters']:.2f}")
    check("OL effect ≈ +1 pts per 100", abs(coef["OL_starters"] - 1.0) < 0.6, f"{coef['OL_starters']:.2f}")
    check("CV R² is high on clean data", m.cv_r2 > 0.6, f"{m.cv_r2:.2f}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_schedule() -> None:
    rng = np.random.default_rng(0)
    s = simulate.generate_schedule(CONF, rng)
    counts = pd.concat([s["home"], s["away"]]).value_counts()
    check("16 games each", (counts == 16).all(), str(counts.to_dict()))
    cross = s[s["home"].map(CONF) != s["away"].map(CONF)]
    cc = pd.concat([cross["home"], cross["away"]]).value_counts()
    check("4 cross-conference games each", (cc == 4).all(), str(cc.to_dict()))
    check("112 games total", len(s) == 112)
    # Mid-season completion keeps played games and restores the counts.
    played = s.head(30).copy()
    played["home_f"], played["away_f"] = played["home"], played["away"]
    played["home_result"] = 1.0
    full = simulate.complete_schedule(played, CONF, rng)
    counts2 = pd.concat([full["home"], full["away"]]).value_counts()
    check("completed schedule has 16 each", (counts2 == 16).all(), str(counts2.to_dict()))
    check("played games kept", full["played"].sum() == 30)


def test_simulation() -> None:
    strength = pd.Series({t: (i - 3) * 3.0 for i, t in enumerate(TEAMS_N)} | {t: (i - 3) * 3.0 for i, t in enumerate(TEAMS_A)})
    rng = np.random.default_rng(1)
    sched = simulate.generate_schedule(CONF, rng)
    wm = simulate.WinModel(scale=8.0, hfa=2.0)
    res = simulate.simulate(strength, sched, CONF, wm, n_sims=3000, seed=1)
    t = res.table.set_index("franchise")
    check("strongest team has most expected wins", t["exp_wins"].idxmax() in ("N7", "A7"))
    check("rank rows sum to one", np.allclose(res.rank_matrix.sum(axis=1), 1.0))
    check("playoff probabilities sum to 6", abs(t["p_playoffs"].sum() - 6) < 1e-6)
    check("one Ultimus per sim", abs(t["p_ultimus"].sum() - 1) < 1e-6)
    check("byes sum to 2", abs(t["p_bye"].sum() - 2) < 1e-6)
    check("best record sums to 1", abs(t["p_best_record"].sum() - 1) < 1e-6)
    order = simulate.ranking_for_task(res)
    check("ranking lists both conferences", set(order) == {"NSFC", "ASFC"} and order["NSFC"][0] == "N7")


def test_win_model_fit() -> None:
    games = synthetic_games(20, seed=5)
    strength = pd.DataFrame([{"season": s, "franchise": t, "value": (i - 3) * 3.0}
                             for s in range(1, 21) for i, t in enumerate(TEAMS_N)]
                            + [{"season": s, "franchise": t, "value": (i - 3) * 3.0}
                               for s in range(1, 21) for i, t in enumerate(TEAMS_A)])
    wm = simulate.fit_win_model(games, strength, from_season=1)
    check("fitted scale near 8", abs(wm.scale - 8) < 2.5, f"{wm.scale:.2f}")
    check("fitted hfa near 2", abs(wm.hfa - 2) < 1.5, f"{wm.hfa:.2f}")


# ---------------------------------------------------------------------------
# GM and backtest plumbing
# ---------------------------------------------------------------------------


def test_gm_elo_split() -> None:
    hist = pd.DataFrame([{"season": 1, "league": "ISFL", "team": "N1", "uid": 1, "username": "a"},
                         {"season": 1, "league": "ISFL", "team": "N1", "uid": 2, "username": "b"},
                         {"season": 1, "league": "DSFL", "team": "D1", "uid": 3, "username": "c"}])
    hist = data._prep_gm_history(hist)
    elo_hist = pd.DataFrame([{"season": 1, "franchise": "N1", "change": 60.0}])
    per, cur = gm_mod.gm_elo(hist, elo_hist)
    check("credit split between two GMs", (per["credited"] == 30.0).all())
    check("DSFL GMs excluded", set(cur["uid"]) == {1, 2})


def test_backtest_score() -> None:
    actual = pd.DataFrame({"franchise": TEAMS_N, "conference": "NSFC", "wins": [2, 4, 6, 8, 10, 12, 14],
                           "win_pct": [2 / 16, 4 / 16, 6 / 16, 8 / 16, 10 / 16, 12 / 16, 14 / 16],
                           "margin": range(7), "actual_rank": [7, 6, 5, 4, 3, 2, 1], "playoff_result": "missed"})
    perfect = pd.Series(range(7), index=TEAMS_N, dtype=float)
    sc = backtest._score(perfect, actual)
    check("perfect prediction scores 1", abs(sc["spearman_conf"] - 1) < 1e-9 and sc["best_record_hit"] == 1.0)
    reversed_ = pd.Series(range(7, 0, -1), index=TEAMS_N, dtype=float)
    check("reversed prediction scores -1", abs(backtest._score(reversed_, actual)["spearman_conf"] + 1) < 1e-9)


def test_projection() -> None:
    now = pd.DataFrame([
        {"pid": 1, "franchise": "N1", "name": "Vet", "pos_group": "QB", "draft_season": 57, "tpe": 1000.0, "active": True},
        {"pid": 2, "franchise": "N1", "name": "Kid", "pos_group": "RB", "draft_season": 62, "tpe": 300.0, "active": True},
        {"pid": 3, "franchise": "N1", "name": "Old", "pos_group": "OL", "draft_season": 51, "tpe": 400.0, "active": False},
    ])
    rates = pd.DataFrame({"pid": [1, 2], "rate": [100.0, 200.0], "seasons_used": [2, 2]})
    hazard = pd.Series({c: 0.0 for c in range(1, 14)})
    res = projection.project(now, rates, pd.Series({1: 150.0}), hazard, 63, horizon=2, season_left=1.0, n_sims=10)
    p = res.players.set_index(["pid", "season"])["tpe"]
    # Vet: drafted 57, finishes S63 = 7 seasons → 20% off (1000+100)*0.8 = 880
    check("veteran regresses 20% after 7th season", p[(1, 64)] == 880.0, str(p[(1, 64)]))
    # then S64 finished = 8 seasons → 25%: floor((880+100)*0.75) = 735
    check("then 25%", p[(1, 65)] == 735.0, str(p[(1, 65)]))
    check("rookie just earns", p[(2, 64)] == 500.0 and p[(2, 65)] == 700.0)
    # Old: drafted 51, finishes S63 = 13 seasons → retired, inactive earns 0
    check("13th season forces retirement", p[(3, 64)] == 0.0)
    check("team rollup excludes retired", res.teams.set_index("season").loc[64, "n_players_det"] == 2)


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception as exc:  # noqa: BLE001
            FAILS.append(t.__name__)
            print(f"  [FAIL] raised {type(exc).__name__}: {exc}")
    print(f"\n{len(tests)} test groups, {len(FAILS)} failures" + (": " + ", ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
