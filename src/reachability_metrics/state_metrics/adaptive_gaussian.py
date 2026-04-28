"""Adaptive local-bandwidth Gaussian state distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from .base import StateMetric


class AdaptiveGaussianDistance(StateMetric):
    """Gaussian kernel with per-point bandwidth from kNN density."""

    def __init__(self, k: int = 10, eps: float = 1e-6, distance_mode: str = "one_minus_kernel") -> None:
        self.k = k
        self.eps = eps
        self.distance_mode = distance_mode

    def fit(self, X: Any, y: Any = None) -> "AdaptiveGaussianDistance":
        super().fit(X, y)
        if self.X_fit_.shape[0] == 0:
            raise ValueError("AdaptiveGaussianDistance requires at least one point")
        n = self.X_fit_.shape[0]
        if n == 1:
            self.effective_k_ = 1
            self.train_sigmas_ = np.ones(1, dtype=np.float64)
            self.knn_ = NearestNeighbors(n_neighbors=1).fit(self.X_fit_)
            return self
        self.effective_k_ = min(max(int(self.k), 1), n - 1)
        self.knn_ = NearestNeighbors(n_neighbors=self.effective_k_ + 1, metric="euclidean").fit(self.X_fit_)
        distances, _ = self.knn_.kneighbors(self.X_fit_)
        self.train_sigmas_ = np.maximum(distances[:, -1], max(float(self.eps), 1e-12))
        return self

    def estimate_sigmas(self, X: Any) -> np.ndarray:
        """Estimate local bandwidth for query states."""
        if not hasattr(self, "knn_"):
            raise RuntimeError("AdaptiveGaussianDistance must be fitted")
        x = self._check_pair_inputs(X, self.X_fit_)[0]
        neighbors = min(max(int(self.effective_k_), 1), self.X_fit_.shape[0])
        distances, _ = self.knn_.kneighbors(x, n_neighbors=neighbors)
        return np.maximum(distances[:, -1], max(float(self.eps), 1e-12)).astype(np.float64)

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        if not hasattr(self, "knn_"):
            self.fit(X)
        x, y = self._check_pair_inputs(X, Y)
        sx = self.estimate_sigmas(x)
        sy = self.estimate_sigmas(y)
        sq = cdist(x, y, metric="sqeuclidean")
        denom = np.maximum(sx[:, None] * sy[None, :], max(float(self.eps), 1e-12))
        return np.exp(-(sq / denom)).astype(np.float32)

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        k = np.asarray(self.pairwise_similarity(X, Y), dtype=np.float64)
        mode = str(self.distance_mode).lower()
        if mode in {"rkhs", "rkhs_distance"}:
            return np.sqrt(np.maximum(2.0 - 2.0 * k, 0.0)).astype(np.float32)
        if mode in {"one_minus_kernel", "1-k"}:
            return (1.0 - k).astype(np.float32)
        raise ValueError(f"Unsupported distance_mode: {self.distance_mode}")

