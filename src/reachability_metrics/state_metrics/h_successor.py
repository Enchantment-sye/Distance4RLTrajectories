"""H-step successor-state distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.data.windows import future_windows


class HSuccessorDistance(BaseEstimator):
    """Compare same-trajectory H-step future windows."""

    def __init__(self, horizon: int = 10, gamma: float | None = None, aggregation: str = "raw_l2") -> None:
        self.horizon = horizon
        self.gamma = gamma
        self.aggregation = aggregation

    def fit(self, trajectories: Any, y: Any = None) -> "HSuccessorDistance":
        dataset = trajectories if isinstance(trajectories, TrajectoryDataset) else TrajectoryDataset.from_arrays(trajectories)
        self.windows_, self.valid_global_indices_, self.window_episode_ids_ = future_windows(dataset, self.horizon)
        self.dataset_ = dataset
        return self

    def _weights(self) -> np.ndarray:
        h = int(self.horizon)
        if self.gamma is None:
            return np.full(h, 1.0 / h, dtype=np.float32)
        g = float(self.gamma)
        weights = np.power(g, np.arange(h, dtype=np.float32))
        return (weights / np.sum(weights)).astype(np.float32)

    def _window_distance(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mode = str(self.aggregation).lower()
        if mode == "endpoint_l2":
            diff = a[:, None, -1, :] - b[None, :, -1, :]
            return np.linalg.norm(diff, axis=-1).astype(np.float32)
        if mode == "mean_l2":
            diff = np.mean(a, axis=1)[:, None, :] - np.mean(b, axis=1)[None, :, :]
            return np.linalg.norm(diff, axis=-1).astype(np.float32)
        if mode != "raw_l2":
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        weights = self._weights()
        diff = a[:, None, :, :] - b[None, :, :, :]
        sq = np.sum(diff * diff, axis=-1)
        return np.sqrt(np.maximum(np.sum(sq * weights[None, None, :], axis=-1), 0.0)).astype(np.float32)

    def pairwise_distance(self, X: Any | None = None, Y: Any | None = None) -> np.ndarray:
        if not hasattr(self, "windows_"):
            raise RuntimeError("HSuccessorDistance must be fitted")
        a = self.windows_ if X is None else np.asarray(X, dtype=np.float32)
        b = a if Y is None else np.asarray(Y, dtype=np.float32)
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError("successor windows must have shape (N, H, D)")
        return self._window_distance(a, b)

    def pairwise_similarity(self, X: Any | None = None, Y: Any | None = None) -> np.ndarray:
        return (-self.pairwise_distance(X, Y)).astype(np.float32)

