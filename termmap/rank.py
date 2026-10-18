"""Learning-to-rank over candidate features: group-relative feature derivation and the ranker itself."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config

_REL = ["d_char", "d_word", "d_dense", "m_char", "m_dense", "soft_q", "soft_d", "tsr", "ctx_dense", "jacc",
        "d_sap", "m_sap"]
_RANKED = ["d_char", "d_word", "d_dense", "m_char", "soft_q", "d_sap", "m_sap"]

DERIVED = ([f"{c}_gap" for c in _REL] + [f"{c}_grank" for c in _RANKED]
           + ["n_sources", "freq_gated", "best_evidence", "group_size"])


def derive(X: pd.DataFrame) -> pd.DataFrame:
    """Add features relative to the other candidates of the same term (a ranker decides between them)."""
    if not len(X):
        return X
    X = X.sort_values(["item", "code"], kind="stable").reset_index(drop=True)
    for c in _REL + _RANKED:  # sources that were not built (e.g. no SapBERT) contribute nothing
        if c not in X.columns:
            X[c] = 0.0
    g = X.groupby("item", sort=False)
    for c in _REL:
        X[f"{c}_gap"] = X[c] - g[c].transform("max")
    for c in _RANKED:
        X[f"{c}_grank"] = g[c].rank(ascending=False, method="min")
    X["n_sources"] = ((X.d_char > 0).astype(int) + (X.d_word > 0) + (X.d_dense > 0) + (X.m_char > 0) + (X.m_exact > 0)
                      + (X.d_sap > 0) + (X.m_sap > 0))
    X["best_evidence"] = X[["d_char", "d_dense", "m_char", "soft_q", "d_sap", "m_sap"]].max(axis=1)
    # The training-frequency prior only counts when there is textual evidence for the code.
    X["freq_gated"] = X.freq * X.best_evidence ** 2
    X["group_size"] = g.code.transform("size")
    return X


class Ranker:
    """LambdaRank (LightGBM) over candidate groups; deterministic for a fixed input."""

    def __init__(self, features: list[str], **params):
        import lightgbm as lgb

        self.features = features
        defaults = dict(objective="lambdarank", n_estimators=600, learning_rate=0.03, num_leaves=31,
                        min_child_samples=20, colsample_bytree=0.8, subsample=1.0, reg_lambda=1.0,
                        lambdarank_truncation_level=10, random_state=config.SEED,
                        n_jobs=int(os.environ.get("TERMMAP_JOBS", "4")),
                        deterministic=True, force_row_wise=True, verbose=-1)
        defaults.update(params)
        self.model = lgb.LGBMRanker(**defaults)

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        # Features of a source that was not built for this run (e.g. no SapBERT) are constant zero.
        return X.reindex(columns=self.features, fill_value=0.0)

    def fit(self, X: pd.DataFrame) -> "Ranker":
        X = X.sort_values(["item", "code"], kind="stable")
        sizes = X.groupby("item", sort=False).size().to_numpy()
        self.model.fit(self._matrix(X), X.label, group=sizes)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._matrix(X))


def softmax_by_group(X: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Turn ranking scores into per-term probabilities (confidence and margin come from these)."""
    s = pd.Series(scores, index=X.index)
    z = s - s.groupby(X.item).transform("max")
    e = np.exp(z)
    return (e / e.groupby(X.item).transform("sum")).to_numpy()
