"""
Roster-strength model: what does Week 1 TPE, by position group, buy in points?

Ridge regression of a season's SRS (schedule-adjusted margin, points per
game) on the standardised starter-TPE of each position group. Fitted on
franchise-seasons from FIRST_TPE_SEASON; about 14 rows per season, so the
regulariser is chosen by leave-one-season-out cross-validation and the
page is honest about the sample size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config

GROUP_FEATURES = [f"{g}_starters" for g in config.POSITION_GROUPS]
CONTEXT_FEATURES = ["n_inactive", "age_mean"]
TARGET = "srs"


@dataclass
class RidgeModel:
    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray            # on standardised features
    intercept: float
    alpha: float
    n: int
    seasons: list[int]
    cv_r2: float = float("nan")
    cv_rmse: float = float("nan")
    cv_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_std: float = float("nan")

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        X = frame[self.features].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0)
        Z = (X - self.mean) / self.std
        return pd.Series(Z @ self.coef + self.intercept, index=frame.index)

    @property
    def coefficients(self) -> pd.DataFrame:
        """Standardised coefficient and its effect in raw units."""
        return pd.DataFrame({
            "feature": self.features,
            "coef_std": self.coef,
            "per_100_tpe": 100.0 * self.coef / np.where(self.std == 0, np.nan, self.std),
        })


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _fit(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Ridge with an unpenalised intercept, on already-standardised X."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(A, Xc.T @ yc)
    return coef, float(ym - xm @ coef)


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (X - mean) / std, mean, std


def train(table: pd.DataFrame, features: list[str] | None = None,
          alphas: list[float] | None = None, target: str = TARGET) -> RidgeModel | None:
    """
    `table` has one row per franchise-season with the features and `target`.
    Alpha is chosen by leave-one-season-out CV; the returned model is
    refitted on everything with that alpha.
    """
    features = features or GROUP_FEATURES
    alphas = alphas or config.RIDGE_ALPHAS
    t = table.dropna(subset=[target]).copy()
    for f in features:
        t[f] = pd.to_numeric(t[f], errors="coerce").fillna(0.0)
    seasons = sorted(t["season"].unique())
    if len(seasons) < config.MIN_TRAINING_SEASONS or len(t) < 2 * len(features):
        return None

    X_raw = t[features].to_numpy(dtype=float)
    y = t[target].to_numpy(dtype=float)
    best = None
    for alpha in alphas:
        preds = np.full(len(t), np.nan)
        for s in seasons:
            test = (t["season"] == s).to_numpy()
            Ztr, m, sd = _standardise(X_raw[~test])
            coef, b = _fit(Ztr, y[~test], alpha)
            preds[test] = ((X_raw[test] - m) / sd) @ coef + b
        sse = float(((y - preds) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, alpha, preds)
    sse, alpha, preds = best
    sst = float(((y - y.mean()) ** 2).sum())

    Z, mean, std = _standardise(X_raw)
    coef, intercept = _fit(Z, y, alpha)
    cv = t[["season", "franchise", target]].copy()
    cv["predicted"] = preds
    return RidgeModel(
        features=features, mean=mean, std=std, coef=coef, intercept=intercept,
        alpha=alpha, n=len(t), seasons=[int(s) for s in seasons],
        cv_r2=1 - sse / sst if sst else float("nan"),
        cv_rmse=float(np.sqrt(sse / len(t))),
        cv_predictions=cv, target_std=float(y.std()),
    )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------


def correlations(table: pd.DataFrame, features: list[str] | None = None,
                 target: str = TARGET) -> pd.DataFrame:
    """Univariate Pearson and Spearman correlation of each feature with the target."""
    features = features or GROUP_FEATURES + ["tpe_mean", "starters_mean", "qb1"] + CONTEXT_FEATURES
    t = table.dropna(subset=[target])
    rows = []
    for f in features:
        if f not in t.columns:
            continue
        x = pd.to_numeric(t[f], errors="coerce")
        ok = x.notna()
        if ok.sum() < 5:
            continue
        rows.append({"feature": f,
                     "pearson": float(x[ok].corr(t.loc[ok, target])),
                     "spearman": float(x[ok].corr(t.loc[ok, target], method="spearman")),
                     "n": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("pearson", ascending=False).reset_index(drop=True)


def zscores_by_season(table: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Each feature as a z-score within its season, so eras are comparable."""
    features = features or GROUP_FEATURES + ["tpe_mean", "starters_mean"]
    out = table[["season", "franchise"]].copy()
    for f in features:
        if f in table.columns:
            grp = table.groupby("season")[f]
            out[f] = (table[f] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    return out


def champion_profile(table: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """
    Mean within-season z-score of each feature for champions, runners-up,
    other playoff teams and the rest. The chart that answers "where were the
    winners elite?"
    """
    features = features or GROUP_FEATURES + ["tpe_mean", "starters_mean"]
    z = zscores_by_season(table, features)
    z["bucket"] = np.select(
        [table["playoff_result"].eq("champion"), table["playoff_result"].eq("runner_up"),
         table["playoff_result"].isin(["semifinal", "first_round"])],
        ["Champion", "Runner-up", "Other playoff"], default="Missed playoffs",
    )
    prof = z.groupby("bucket")[[f for f in features if f in z.columns]].mean().T
    prof.index.name = "feature"
    order = [c for c in ["Champion", "Runner-up", "Other playoff", "Missed playoffs"] if c in prof.columns]
    return prof[order].reset_index()


def partial_effects(model: RidgeModel, delta: float = 100.0) -> pd.DataFrame:
    """Points per game gained from +delta TPE on each group's starters."""
    c = model.coefficients
    c["effect_ppg"] = c["per_100_tpe"] * (delta / 100.0)
    c["group"] = c["feature"].str.replace("_starters", "", regex=False)
    return c[["group", "effect_ppg", "coef_std"]].sort_values("effect_ppg", ascending=False)


def simple_tpe_model(table: pd.DataFrame, feature: str = "starters_mean") -> dict:
    """
    One-variable OLS: how much of SRS does a single TPE number explain?
    The headline number for the What Wins page.
    """
    t = table.dropna(subset=[TARGET, feature])
    if len(t) < 5:
        return {"r2": float("nan"), "slope": float("nan"), "n": int(len(t))}
    x, y = t[feature].to_numpy(float), t[TARGET].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"r2": float(r2), "slope": float(slope), "intercept": float(intercept), "n": int(len(t))}
