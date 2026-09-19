"""Unsupervised, per-file anomaly-scoring baselines.

Per the modeling notes: given only 6 labeled cases, lean on unsupervised
anomaly scoring first (fit independently on each file's 8 car feature
vectors, no labels or cross-file training involved at all) and only justify
a supervised combination layer on top if it measurably improves ranking.

Each function takes the within-file z-scored feature matrix from
``features.build_feature_matrix`` (one row per car) and returns a per-car
anomaly score (higher = more suspicious); sorting descending gives the
ranking for that file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def aggregate_deviation_score(Z: pd.DataFrame) -> pd.Series:
    """Mean absolute z-score across features: 'how unusual is this car,
    averaged over every signal, regardless of direction'."""
    return Z.abs().mean(axis=1)


def mahalanobis_score(Z: pd.DataFrame) -> pd.Series:
    """Mahalanobis distance of each car's feature vector from the file's
    car-to-car distribution, using Ledoit-Wolf shrinkage since only 8 cars
    make the raw sample covariance unstable to estimate."""
    X = Z.to_numpy()
    cov = LedoitWolf().fit(X)
    diff = X - X.mean(axis=0)
    inv = cov.precision_
    d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return pd.Series(d2, index=Z.index)


def isolation_forest_score(Z: pd.DataFrame, random_state: int = 0) -> pd.Series:
    X = Z.to_numpy()
    model = IsolationForest(n_estimators=200, random_state=random_state)
    model.fit(X)
    return pd.Series(-model.score_samples(X), index=Z.index)  # higher = more anomalous


def lof_score(Z: pd.DataFrame, n_neighbors: int = 3) -> pd.Series:
    X = Z.to_numpy()
    n_neighbors = max(1, min(n_neighbors, len(X) - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors)
    model.fit_predict(X)
    return pd.Series(-model.negative_outlier_factor_, index=Z.index)  # higher = more anomalous


def pca_reconstruction_score(Z: pd.DataFrame, n_components: int = 3) -> pd.Series:
    X = Z.to_numpy()
    n_components = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X)
    reconstructed = pca.inverse_transform(transformed)
    residual = np.sum((X - reconstructed) ** 2, axis=1)
    return pd.Series(residual, index=Z.index)


def _zscore(s: pd.Series) -> pd.Series:
    """Standardize a score across the 8 cars so two differently-scaled
    anomaly scores can be averaged meaningfully."""
    std = s.std(ddof=0)
    if std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def ensemble_mahalanobis_iforest_score(Z: pd.DataFrame) -> pd.Series:
    """Average of Mahalanobis distance and Isolation Forest, each z-scored
    across the file's 8 cars first so they're on a comparable scale before
    combining. The two are tied on the 6 labeled cases (0.979 each) but
    reach that score through different mechanisms (global correlation
    structure vs. random-split isolation) — averaging them is a variance
    reduction bet: if one has an idiosyncratic blind spot on an unseen file,
    the other can compensate, at the cost of being less interpretable than
    either alone."""
    m = _zscore(mahalanobis_score(Z))
    i = _zscore(isolation_forest_score(Z))
    return (m + i) / 2.0


BASELINE_METHODS = {
    "aggregate_deviation": aggregate_deviation_score,
    "mahalanobis": mahalanobis_score,
    "isolation_forest": isolation_forest_score,
    "lof": lof_score,
    "pca_reconstruction": pca_reconstruction_score,
    "ensemble_mahalanobis_iforest": ensemble_mahalanobis_iforest_score,
}


def rank_from_score(score: pd.Series) -> list[str]:
    return list(score.sort_values(ascending=False).index)


class BaselineModel:
    """Wraps a baselines.py method behind the same ``decision_function(X)``
    interface as a fitted sklearn estimator, so predict.py can use whichever
    one wins the comparison in train.py without a branch at inference time."""

    def __init__(self, method_name: str):
        if method_name not in BASELINE_METHODS:
            raise ValueError(f"Unknown baseline method: {method_name}")
        self.method_name = method_name

    def decision_function(self, X) -> np.ndarray:
        Z = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return BASELINE_METHODS[self.method_name](Z).to_numpy()
