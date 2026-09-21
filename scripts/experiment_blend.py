"""
Experiment: which blend target / alpha ranks best in the walk-forward?

    python scripts/experiment_blend.py

Prints Spearman ρ (conference) per method for each setting, on the seasons
before the roster model exists (so it's about Elo/SRS/record) and after.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import backtest  # noqa: E402
import config  # noqa: E402
import data  # noqa: E402
import elo as elo_mod  # noqa: E402
import roster as roster_mod  # noqa: E402


def main() -> None:
    cache = data.load_all()
    elo_res = elo_mod.run_elo(cache.games)
    srs = elo_mod.srs_all(cache.standings, cache.games)
    features = pd.DataFrame()
    if cache.has_logs:
        tl = roster_mod.tpe_timeline(cache)
        ro = roster_mod.rosters(cache)
        full = roster_mod.roster_with_tpe(cache, tl, ro)
        bots = roster_mod.bots_as_players(cache)
        features = roster_mod.team_features(full, bots)
        features = features[features["season"] >= config.FIRST_TPE_SEASON]

    rows = []
    for target in ("srs", "wpct_pts"):
        for alpha in (1.0, 3.0, 10.0, 30.0, 100.0):
            bt = backtest.walk_forward(cache, elo_res, srs, features, alpha=alpha, target=target)
            per = bt.per_season
            for label, mask in (("S28-56", per["season"] <= 56), ("S57+", per["season"] >= 57), ("all", per["season"] > 0)):
                s = per[mask].groupby("method")["spearman_conf"].mean()
                rows.append({"target": target, "alpha": alpha, "window": label,
                             **{m: round(v, 3) for m, v in s.items()}})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    # Elo variants for ranking.
    print("\nElo variants (rank by preseason Elo), S28+ spearman_conf:")
    for k in (20, 30, 40):
        for playoffs in (True, False):
            res = elo_mod.run_elo(cache.games, k=k, playoffs=playoffs)
            bt = backtest.walk_forward(cache, res, srs, pd.DataFrame())
            s = bt.per_season.groupby("method")["spearman_conf"].mean()
            print(f"  K={k} playoffs={playoffs}: elo {s.get('elo', float('nan')):.3f}  blend {s.get('blend', float('nan')):.3f}  last_season {s.get('last_season', float('nan')):.3f}")


if __name__ == "__main__":
    main()
