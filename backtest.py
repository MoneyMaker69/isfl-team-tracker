"""
Preseason strength and the walk-forward backtest.

For a target season S, a preseason prediction may know:
- Elo carried in from S-1 (reverted),
- last season's SRS,
- the Week 1 roster of S and a roster model trained on seasons < S,
- the GMs of S and their over-performance on seasons < S.

`preseason_table` assembles those; `fit_blend` learns how to weigh them on
earlier seasons' outcomes; `walk_forward` runs the whole thing season by
season and scores it against naive baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

import config
import elo as elo_mod
import gm as gm_mod
import model as model_mod
import simulate

BLEND_FEATURES = ["elo_pts", "srs_prev", "wpct_prev", "roster_pred", "gm_factor"]
BLEND_FEATURES_NO_ROSTER = ["elo_pts", "srs_prev", "wpct_prev"]


@dataclass
class Blend:
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    n: int
    seasons: list[int] = field(default_factory=list)

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        X = frame[self.features].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        return pd.Series(((X - self.mean) / self.std) @ self.coef + self.intercept, index=frame.index)

    @property
    def weights(self) -> pd.Series:
        """Share of each feature in the prediction (abs standardised coef)."""
        w = np.abs(self.coef)
        return pd.Series(w / w.sum() if w.sum() else w, index=self.features)


def elo_points_slope(elo_result: elo_mod.EloResult, srs: pd.DataFrame,
                     from_season: int = config.ENGINE_SWITCH_SEASON) -> float:
    """Points of SRS per Elo point above 1500, from preseason Elo vs season SRS."""
    rows = []
    for season in sorted(srs["season"].unique()):
        if season < from_season:
            continue
        pre = elo_mod.preseason_elo(elo_result, int(season))
        s = srs[srs["season"] == season].set_index("franchise")["srs"]
        j = pd.concat([pre.rename("elo"), s], axis=1).dropna()
        rows.append(j)
    if not rows:
        return 1 / 25.0
    j = pd.concat(rows)
    x = j["elo"] - config.ELO_START
    if x.std() == 0:
        return 1 / 25.0
    return float((x * j["srs"]).sum() / (x * x).sum())


def preseason_table(season: int, cache, elo_result: elo_mod.EloResult, srs: pd.DataFrame,
                    features: pd.DataFrame, slope: float,
                    roster_model: model_mod.RidgeModel | None,
                    overperf: pd.DataFrame | None) -> pd.DataFrame:
    """One row per franchise with what is knowable before Week 1 of `season`."""
    teams = cache.teams if season >= cache.current_season else sorted(
        cache.standings[cache.standings["season"] == season]["franchise"].unique())
    out = pd.DataFrame({"franchise": teams})
    pre = elo_mod.preseason_elo(elo_result, season)
    out["elo_pre"] = out["franchise"].map(pre)
    out["elo_pts"] = (out["elo_pre"].fillna(config.ELO_START) - config.ELO_START) * slope
    prev = srs[srs["season"] == season - 1].set_index("franchise")["srs"]
    out["srs_prev"] = out["franchise"].map(prev).fillna(0.0)
    last_st = cache.standings[cache.standings["season"] == season - 1].set_index("franchise")["win_pct"]
    # Expressed in points so the hand-weighted blend has comparable scale:
    # a 12-4 team is about +8 points/game above a 4-12 one.
    out["wpct_prev"] = (out["franchise"].map(last_st).fillna(0.5) - 0.5) * 32.0
    f = (features[features["season"] == season].set_index("franchise")
         if not features.empty else pd.DataFrame())
    if roster_model is not None and not f.empty:
        pred = roster_model.predict(f.reindex(out["franchise"]).reset_index(drop=True))
        out["roster_pred"] = pred.values
        # A franchise with no roster rows gets the league-average prediction,
        # not a prediction from a row of zeros.
        missing = ~out["franchise"].isin(f.index)
        out.loc[missing, "roster_pred"] = float(pred[~missing.values].mean()) if (~missing).any() else 0.0
        for col in ("starters_mean", "tpe_mean", "n_inactive", "age_mean"):
            out[col] = out["franchise"].map(f[col]) if col in f.columns else np.nan
    else:
        out["roster_pred"] = np.nan
    if overperf is not None and not overperf.empty:
        gf = gm_mod.team_gm_factor(cache.gm_history, overperf, season)
        out["gm_factor"] = out["franchise"].map(gf).fillna(0.0)
    else:
        out["gm_factor"] = 0.0
    out["season"] = season
    return out


def fit_blend(history: pd.DataFrame, features: list[str], alpha: float = config.BLEND_ALPHA,
              target: str = config.BLEND_TARGET) -> Blend | None:
    """
    Ridge of the season's outcome on the preseason features across past
    seasons. Target is `srs` (points) or `wpct_pts` (win% in points); the
    prediction task is a ranking by record, so the default is chosen by
    which ranks better in the walk-forward (see config.BLEND_TARGET).
    """
    h = history.dropna(subset=[target] + features)
    if len(h) < 3 * len(features) + 5:
        return None
    X = h[features].to_numpy(dtype=float)
    y = h[target].to_numpy(dtype=float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    Z = (X - mean) / std
    zm, ym = Z.mean(axis=0), y.mean()
    # Non-negative ridge: every signal points the same way as the outcome,
    # so collinearity can't flip Elo negative to offset SRS. Solved as NNLS
    # on the ridge-augmented system.
    k = Z.shape[1]
    A = np.vstack([Z - zm, np.sqrt(alpha) * np.eye(k)])
    b = np.concatenate([y - ym, np.zeros(k)])
    coef, _ = nnls(A, b)
    return Blend(features=features, mean=mean, std=std, coef=coef, intercept=float(ym - zm @ coef),
                 n=len(h), seasons=sorted(h["season"].unique().tolist()))


def default_strength(table: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.Series:
    """Hand-weighted blend for when there isn't enough history to fit one."""
    w = weights or config.DEFAULT_BLEND
    cols = {"elo": "elo_pts", "srs": "srs_prev", "wpct": "wpct_prev", "roster": "roster_pred", "gm": "gm_factor"}
    total = 0.0
    acc = pd.Series(0.0, index=table.index)
    for k, col in cols.items():
        if col in table.columns and table[col].notna().any():
            acc += w[k] * table[col].fillna(0.0)
            total += w[k]
    return acc / total if total else acc


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    per_season: pd.DataFrame       # season, method, spearman_conf, spearman_all, srs_mae, best_record_hit
    predictions: pd.DataFrame      # season, franchise, method, predicted, actual_srs, actual_rank, pred_rank
    blend_history: pd.DataFrame    # preseason tables with actuals, for fitting the final blend
    summary: pd.DataFrame          # method × metric means


def _actual(cache, season: int) -> pd.DataFrame:
    st = cache.standings[cache.standings["season"] == season].copy()
    st["actual_rank"] = st.groupby("conference")["win_pct"].rank(ascending=False, method="first")
    # tie-break by margin, then the API's own order
    st = st.sort_values(["conference", "win_pct", "margin"], ascending=[True, False, False])
    st["actual_rank"] = st.groupby("conference").cumcount() + 1
    return st[["franchise", "conference", "wins", "win_pct", "margin", "actual_rank", "playoff_result"]]


def _score(pred: pd.Series, actual: pd.DataFrame) -> dict:
    j = actual.set_index("franchise").join(pred.rename("predicted"), how="inner")
    if j["predicted"].isna().all() or len(j) < 4:
        return {"spearman_conf": np.nan, "spearman_all": np.nan, "best_record_hit": np.nan}
    j["pred_rank"] = j.groupby("conference")["predicted"].rank(ascending=False, method="first")
    confs = []
    for _, grp in j.groupby("conference"):
        if len(grp) >= 3 and grp["predicted"].std() > 0:
            confs.append(spearmanr(grp["pred_rank"], grp["actual_rank"]).correlation)
    all_rank_actual = j["win_pct"].rank(ascending=False)
    sp_all = spearmanr(j["predicted"], -all_rank_actual).correlation if j["predicted"].std() > 0 else np.nan
    best_actual = j["wins"].idxmax()
    best_pred = j["predicted"].idxmax()
    return {"spearman_conf": float(np.nanmean(confs)) if confs else np.nan,
            "spearman_all": float(sp_all), "best_record_hit": float(best_actual == best_pred)}


def walk_forward(cache, elo_result: elo_mod.EloResult, srs: pd.DataFrame, features: pd.DataFrame,
                 first_season: int | None = None, last_season: int | None = None,
                 alpha: float = config.BLEND_ALPHA, target: str = config.BLEND_TARGET) -> BacktestResult:
    """
    For every completed season from `first_season`, predict it using only
    earlier information and score the ranking. Methods:

    - last_season: rank by previous season's record (the baseline to beat)
    - elo: preseason Elo alone
    - srs_prev: last season's SRS alone
    - roster: roster model alone (from FIRST_TPE_SEASON + MIN_TRAINING_SEASONS)
    - blend: fitted combination (fitted on earlier seasons' preseason tables)
    """
    first_season = first_season or config.ENGINE_SWITCH_SEASON + 1
    completed = cache.standings[cache.standings["games"] > 0]["season"]
    if completed.empty:
        return BacktestResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    last_season = last_season or int(completed.max())
    slope = elo_points_slope(elo_result, srs)
    feat_seasons = sorted(features["season"].unique()) if not features.empty else []

    season_rows, pred_rows, hist_rows = [], [], []
    for season in range(first_season, last_season + 1):
        actual = _actual(cache, season)
        if actual.empty:
            continue
        srs_actual = srs[srs["season"] == season].set_index("franchise")["srs"]

        # Roster model on strictly earlier seasons.
        roster_model, overperf = None, None
        train_seasons = [s for s in feat_seasons if config.FIRST_TPE_SEASON <= s < season]
        if len(train_seasons) >= config.MIN_TRAINING_SEASONS and season in feat_seasons:
            tbl = features[features["season"].isin(train_seasons)].merge(
                srs[["season", "franchise", "srs"]], on=["season", "franchise"], how="inner")
            roster_model = model_mod.train(tbl)
            if roster_model is not None:
                res = roster_model.cv_predictions.copy()
                res["residual"] = res["srs"] - res["predicted"]
                overperf = gm_mod.over_performance(cache.gm_history, res[["season", "franchise", "residual"]])

        table = preseason_table(season, cache, elo_result, srs, features, slope, roster_model, overperf)
        table["srs"] = table["franchise"].map(srs_actual)
        table["wpct_pts"] = (table["franchise"].map(actual.set_index("franchise")["win_pct"]) - 0.5) * 32.0
        hist = pd.DataFrame(hist_rows)
        blend_feats = BLEND_FEATURES if roster_model is not None else BLEND_FEATURES_NO_ROSTER
        blend = fit_blend(hist, blend_feats, alpha=alpha, target=target) if not hist.empty else None
        if blend is not None:
            blended = blend.predict(table)
        else:
            blended = default_strength(table)

        methods = {
            "last_season": table["franchise"].map(
                cache.standings[cache.standings["season"] == season - 1]
                .set_index("franchise")["win_pct"]).fillna(0.0),
            "elo": table["elo_pts"],
            "srs_prev": table["srs_prev"],
            "blend": blended,
        }
        if roster_model is not None:
            methods["roster"] = table["roster_pred"]

        for name, pred in methods.items():
            pred = pd.Series(pred.values, index=table["franchise"].values)
            sc = _score(pred, actual)
            mae = float((pred - srs_actual.reindex(pred.index)).abs().mean()) if name != "last_season" else np.nan
            season_rows.append({"season": season, "method": name, "srs_mae": mae, **sc})
            j = actual.set_index("franchise").join(pred.rename("predicted"))
            j["pred_rank"] = j.groupby("conference")["predicted"].rank(ascending=False, method="first")
            for fr, r in j.iterrows():
                pred_rows.append({"season": season, "method": name, "franchise": fr,
                                  "conference": r["conference"], "predicted": r["predicted"],
                                  "actual_srs": srs_actual.get(fr, np.nan),
                                  "actual_rank": r["actual_rank"], "pred_rank": r["pred_rank"],
                                  "wins": r["wins"]})
        table["blend_pred"] = blended.values
        hist_rows.extend(table.to_dict("records"))

    per_season = pd.DataFrame(season_rows)
    summary = (per_season.groupby("method")[["spearman_conf", "spearman_all", "srs_mae", "best_record_hit"]]
               .mean().reset_index()) if not per_season.empty else pd.DataFrame()
    return BacktestResult(per_season=per_season, predictions=pd.DataFrame(pred_rows),
                          blend_history=pd.DataFrame(hist_rows), summary=summary)


def final_blend(result: BacktestResult, with_roster: bool, alpha: float = config.BLEND_ALPHA,
                target: str = config.BLEND_TARGET) -> Blend | None:
    """The blend to use for the upcoming season: fitted on every backtest season."""
    feats = BLEND_FEATURES if with_roster else BLEND_FEATURES_NO_ROSTER
    h = result.blend_history
    if h.empty:
        return None
    if with_roster:
        h = h.dropna(subset=["roster_pred"])
    return fit_blend(h, feats, alpha=alpha, target=target)


def prediction_win_model(result: BacktestResult, games: pd.DataFrame) -> simulate.WinModel:
    """
    Win model fitted on *preseason* predicted strengths, so its scale
    reflects how uncertain a preseason number really is.
    """
    p = result.predictions[result.predictions["method"] == "blend"]
    strength = p.rename(columns={"predicted": "value"})[["season", "franchise", "value"]]
    return simulate.fit_win_model(games, strength, from_season=int(strength["season"].min()))
