"""Classical anomaly scores. Each model is fitted on healthy fit cycles only.

Every scorer exposes fit(X) and score(X); higher score = more anomalous.
"""

from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.config import IFOREST_TREES, PCA_TUNED_COMPONENTS, PCA_VARIANCE, SEED


class Distance:
    """Mean squared z-score: distance from the healthy mean (data are already scaled)"""

    def fit(self, X: np.ndarray) -> "Distance":
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return (X**2).mean(axis=1)


class PCAMonitor:
    """PCA process monitoring: SPE (Q, residual) and Hotelling T2 (inside the subspace)"""

    def __init__(self, n_components: float | int = PCA_VARIANCE):
        # float in (0, 1): keep enough components for that variance share; int: fixed count
        self.pca = PCA(n_components=n_components, svd_solver="full")

    def fit(self, X: np.ndarray) -> "PCAMonitor":
        self.pca.fit(X)
        return self

    def spe(self, X: np.ndarray) -> np.ndarray:
        residual = X - self.pca.inverse_transform(self.pca.transform(X))
        return (residual**2).sum(axis=1)

    def t2(self, X: np.ndarray) -> np.ndarray:
        Z = self.pca.transform(X)
        return (Z**2 / self.pca.explained_variance_).sum(axis=1)


class SPE(PCAMonitor):
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.spe(X)


class T2(PCAMonitor):
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.t2(X)


class IForest:
    def __init__(self, n_estimators: int = IFOREST_TREES, seed: int = SEED):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, X: np.ndarray) -> "IForest":
        self.model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)  # sklearn: higher = more normal; flip it


BASELINES = {
    "distance": Distance,
    "pca_spe": SPE,
    "pca_t2": T2,
    "iforest": IForest,
    # val-tuned, reported separately from the pre-registered rows above
    "pca_t2_tuned": partial(T2, n_components=PCA_TUNED_COMPONENTS),
}
