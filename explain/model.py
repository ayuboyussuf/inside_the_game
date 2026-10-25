"""Outcome model + per-clip contributions.

Two model kinds, both with exact additive contributions in log-odds space:
  * logreg: median-impute -> standardise -> logistic regression; phi_j = w_j * z_j (linear SHAP)
  * lgbm:   LightGBM with ``pred_contrib=True`` (TreeSHAP, computed inside LightGBM)
Small data, so the model stays simple and the features stay meaningful.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from events.features import FEATURE_NAMES

warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass
class Factor:
    feature: str
    value: float
    contribution: float  # log-odds pushed toward the ACTUAL outcome (positive = explains it)
    raw_contribution: float  # log-odds toward outcome=1


class OutcomeExplainer:
    def __init__(self, kind: str = "auto", seed: int = 0, features: list[str] | None = None):
        self.kind, self.seed = kind, seed
        self.features = list(features or FEATURE_NAMES)
        self.train_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "OutcomeExplainer":
        X = df[self.features].astype(float)
        y = df["outcome"].astype(int).values
        self.train_ = df.copy()
        # drop constant columns (e.g. group_help on single-predator chases)
        keep = [c for c in self.features if X[c].nunique(dropna=True) > 1]
        self.used_ = keep or self.features[:1]
        kind = self.kind
        if kind == "auto":
            kind = "lgbm" if len(df) >= 150 else "logreg"
        self.kind_ = kind
        Xu = X[self.used_]
        if kind == "logreg":
            self.imp_ = SimpleImputer(strategy="median").fit(Xu)
            self.sc_ = StandardScaler().fit(self.imp_.transform(Xu))
            self.lr_ = LogisticRegression(C=1.0, max_iter=2000, random_state=self.seed)
            self.lr_.fit(self.sc_.transform(self.imp_.transform(Xu)), y)
        elif kind == "lgbm":
            import lightgbm as lgb
            self.gbm_ = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=8, max_depth=4,
                                           min_child_samples=8, subsample=0.9, subsample_freq=1,
                                           colsample_bytree=0.9, reg_lambda=1.0, random_state=self.seed, verbose=-1)
            self.gbm_.fit(Xu, y)
        else:
            raise ValueError(kind)
        self.medians_ = X.median()
        return self

    # ------------------------------------------------------------------
    def _X(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reindex(columns=self.features).astype(float)
        return X[self.used_]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xu = self._X(X)
        if self.kind_ == "logreg":
            return self.lr_.predict_proba(self.sc_.transform(self.imp_.transform(Xu)))[:, 1]
        return self.gbm_.predict_proba(Xu)[:, 1]

    def contributions(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """phi (n, features) in log-odds toward outcome=1, plus base values."""
        Xu = self._X(X)
        if self.kind_ == "logreg":
            Z = self.sc_.transform(self.imp_.transform(Xu))
            phi = Z * self.lr_.coef_[0]
            base = np.full(len(Xu), self.lr_.intercept_[0])
        else:
            c = self.gbm_.predict(Xu, pred_contrib=True)
            phi, base = c[:, :-1], c[:, -1]
        out = pd.DataFrame(0.0, index=X.index, columns=self.features)
        out[self.used_] = phi
        return out, base

    def explain(self, x: pd.Series, outcome: int) -> list[Factor]:
        phi, _ = self.contributions(x.to_frame().T)
        sign = 1.0 if outcome == 1 else -1.0
        fs = [Factor(f, float(x[f]), float(sign * phi.iloc[0][f]), float(phi.iloc[0][f])) for f in self.features]
        return sorted(fs, key=lambda f: f.contribution, reverse=True)

    def meaningful(self, x: pd.Series, outcome: int, k: int = 3, rel: float = 0.25, abs_min: float = 0.1) -> list[Factor]:
        """The top factor always; secondary factors only if they carry >= 25% of its weight and > 0.1 log-odds.
        Ranking near-zero contributions against each other is noise, so they are not reported."""
        fs = self.explain(x, outcome)
        top = fs[0].contribution
        return [fs[0]] + [f for f in fs[1:k] if f.contribution >= rel * top and f.contribution > abs_min]

    def top_factor(self, x: pd.Series, outcome: int) -> str:
        return self.explain(x, outcome)[0].feature

    def global_importance(self) -> pd.Series:
        phi, _ = self.contributions(self.train_[self.features])
        return phi.abs().mean().sort_values(ascending=False)

    # ------------------------------------------------------------------
    def similar_clips(self, feature: str, value: float, outcome: int, exclude: str | None = None) -> dict | None:
        """'When reaction delay < 0.55 s, escapes happened in 41 of 52 similar clips.'"""
        df = self.train_
        if df is None or feature not in df:
            return None
        d = df.drop(index=[exclude]) if exclude is not None and exclude in df.index else df
        col = d[feature].astype(float)
        ok = col.notna()
        if ok.sum() < 6 or col[ok].nunique() < 2 or not np.isfinite(value):
            return None
        tree = DecisionTreeClassifier(max_depth=1, min_samples_leaf=3, random_state=0)
        tree.fit(col[ok].values.reshape(-1, 1), d.loc[ok, "outcome"].values)
        if tree.tree_.node_count < 3:
            return None
        thr = float(tree.tree_.threshold[0])
        below = value <= thr
        side = (col[ok] <= thr) if below else (col[ok] > thr)
        n = int(side.sum())
        k = int((d.loc[ok][side]["outcome"] == outcome).sum())
        return {"feature": feature, "threshold": thr, "side": "below" if below else "above", "n": n, "k": k}

    def counterfactual(self, x: pd.Series, feature: str, outcome: int) -> dict | None:
        """Move one feature to the typical value of clips with the opposite outcome."""
        df = self.train_
        opp = df[df["outcome"] != outcome][feature].dropna()
        if len(opp) < 3:
            return None
        new_v = float(opp.median())
        x2 = x.copy()
        x2[feature] = new_v
        p0 = float(self.predict_proba(x.to_frame().T)[0])
        p1 = float(self.predict_proba(x2.to_frame().T)[0])
        return {"feature": feature, "from": float(x[feature]), "to": new_v, "p_from": p0, "p_to": p1}
