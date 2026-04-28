"""Base state metric interface."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from reachability_metrics.utils import as_2d_array


class StateMetric(BaseEstimator):
    """Sklearn-style state metric base class."""

    higher_similarity_is_closer = True

    def fit(self, X: Any, y: Any = None) -> "StateMetric":
        """Fit the metric."""
        self.X_fit_ = as_2d_array(X, dtype=np.float64, name="X")
        self.n_features_in_ = int(self.X_fit_.shape[1])
        return self

    def _check_pair_inputs(self, X: Any, Y: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = as_2d_array(X, dtype=np.float64, name="X")
        y = x if Y is None else as_2d_array(Y, dtype=np.float64, name="Y")
        if x.shape[1] != y.shape[1]:
            raise ValueError(f"X and Y feature dims must match, got {x.shape[1]} and {y.shape[1]}")
        return x, y

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        """Pairwise distances."""
        raise NotImplementedError

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        """Pairwise similarities."""
        return -self.pairwise_distance(X, Y)

    def kneighbors(self, X: Any, Y: Any | None = None, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest-neighbor distances and indices."""
        distances = np.asarray(self.pairwise_distance(X, Y), dtype=np.float64)
        top_k = min(int(k), distances.shape[1])
        if top_k <= 0:
            raise ValueError("k must be positive")
        idx = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
        vals = np.take_along_axis(distances, idx, axis=1)
        order = np.argsort(vals, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        return vals.astype(np.float32), idx.astype(np.int64)

