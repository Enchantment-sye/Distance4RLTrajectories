"""Mahalanobis state distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

from .base import StateMetric


class MahalanobisDistance(StateMetric):
    """Global covariance-normalized distance."""

    def __init__(
        self,
        covariance_estimator: str = "ledoitwolf",
        implementation: str = "whitening",
        eps: float = 1e-6,
    ) -> None:
        self.covariance_estimator = covariance_estimator
        self.implementation = implementation
        self.eps = eps

    def fit(self, X: Any, y: Any = None) -> "MahalanobisDistance":
        super().fit(X, y)
        x = self.X_fit_
        dim = x.shape[1]
        eps = max(float(self.eps), 1e-12)
        if x.shape[0] < 2:
            cov = np.eye(dim, dtype=np.float64)
            estimator_name = "identity"
        else:
            key = str(self.covariance_estimator).lower()
            try:
                if key in {"empirical", "empiricalcovariance"}:
                    est = EmpiricalCovariance(store_precision=True).fit(x)
                else:
                    est = LedoitWolf(store_precision=True).fit(x)
                cov = np.asarray(est.covariance_, dtype=np.float64)
                estimator_name = key
            except Exception:
                centered = x - np.mean(x, axis=0, keepdims=True)
                cov = centered.T @ centered / max(x.shape[0] - 1, 1)
                estimator_name = "manual_empirical"
        cov = 0.5 * (cov + cov.T) + eps * np.eye(dim)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, eps, None)
        self.covariance_matrix_ = cov
        self.precision_matrix_ = (vecs * (1.0 / vals)[None, :]) @ vecs.T
        self.whitening_matrix_ = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
        self.estimator_name_ = estimator_name
        return self

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        if not hasattr(self, "whitening_matrix_"):
            self.fit(X)
        x, y = self._check_pair_inputs(X, Y)
        if str(self.implementation).lower() == "precision":
            xp = x @ self.precision_matrix_
            yp = y @ self.precision_matrix_
            x_sq = np.sum(xp * x, axis=1, keepdims=True)
            y_sq = np.sum(yp * y, axis=1, keepdims=True).T
            return np.sqrt(np.maximum(x_sq + y_sq - 2.0 * xp @ y.T, 0.0)).astype(np.float32)
        xw = x @ self.whitening_matrix_.T
        yw = y @ self.whitening_matrix_.T
        return cdist(xw, yw, metric="euclidean").astype(np.float32)

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        return (-self.pairwise_distance(X, Y)).astype(np.float32)

