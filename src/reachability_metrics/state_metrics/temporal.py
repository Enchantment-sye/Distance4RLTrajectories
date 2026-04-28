"""Temporal distance baselines."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.data import TrajectoryDataset
from .base import StateMetric


class TemporalDistance(StateMetric):
    """Temporal distance from trajectory identity and timestep metadata."""

    def __init__(self, mode: str = "same_trajectory_min_gap", max_window: int | None = None) -> None:
        self.mode = mode
        self.max_window = max_window

    def fit(self, X: Any, y: Any = None) -> "TemporalDistance":
        if isinstance(X, TrajectoryDataset):
            self.episode_ids_ = X.episode_ids()
            self.timesteps_ = X.timesteps()
            self.X_fit_ = X.states()
            self.n_features_in_ = int(self.X_fit_.shape[1])
            return self
        return super().fit(X, y)

    def pairwise_distance_indices(self, indices_a: np.ndarray, indices_b: np.ndarray) -> np.ndarray:
        """Distance between stacked dataset indices."""
        if not hasattr(self, "episode_ids_"):
            raise RuntimeError("TemporalDistance.fit requires a TrajectoryDataset for index distances")
        a = np.asarray(indices_a, dtype=np.int64)
        b = np.asarray(indices_b, dtype=np.int64)
        same = self.episode_ids_[a][:, None] == self.episode_ids_[b][None, :]
        gap = np.abs(self.timesteps_[a][:, None] - self.timesteps_[b][None, :]).astype(np.float32)
        dist = np.where(same, gap, np.inf).astype(np.float32)
        if self.max_window is not None:
            dist[dist > int(self.max_window)] = np.inf
        return dist

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        x, y = self._check_pair_inputs(X, Y)
        if Y is None and hasattr(self, "X_fit_") and x.shape[0] == self.X_fit_.shape[0] and np.allclose(x, self.X_fit_):
            idx = np.arange(x.shape[0], dtype=np.int64)
            return self.pairwise_distance_indices(idx, idx)
        return np.full((x.shape[0], y.shape[0]), np.inf, dtype=np.float32)

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        d = self.pairwise_distance(X, Y)
        sim = np.zeros_like(d, dtype=np.float32)
        finite = np.isfinite(d)
        sim[finite] = 1.0 / (1.0 + d[finite])
        return sim

