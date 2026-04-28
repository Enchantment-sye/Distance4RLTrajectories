"""State-to-trajectory distances."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.state_metrics import StateMetric
from reachability_metrics.trajectory_metrics.kme import KernelMeanEmbedding
from reachability_metrics.utils import as_2d_array, as_trajectory_list, softmin


class StateToTrajectoryDistance:
    """Aggregate pointwise state distances from a state to a trajectory."""

    def __init__(
        self,
        state_metric: StateMetric,
        aggregation: str = "min",
        softmin_tau: float = 1.0,
        k: int = 3,
    ) -> None:
        self.state_metric = state_metric
        self.aggregation = aggregation
        self.softmin_tau = softmin_tau
        self.k = k

    def fit(self, trajectories: Any) -> "StateToTrajectoryDistance":
        trajs = as_trajectory_list(trajectories)
        self.trajectories_ = trajs
        self.state_metric.fit(np.concatenate(trajs, axis=0))
        return self

    def _aggregate(self, distances: np.ndarray) -> np.ndarray:
        mode = str(self.aggregation).lower()
        if mode == "min":
            return np.min(distances, axis=1)
        if mode == "mean":
            return np.mean(distances, axis=1)
        if mode == "softmin":
            return softmin(distances, tau=float(self.softmin_tau), axis=1)
        if mode == "kmin_mean":
            kk = min(max(int(self.k), 1), distances.shape[1])
            part = np.partition(distances, kth=kk - 1, axis=1)[:, :kk]
            return np.mean(part, axis=1)
        raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def pairwise_distance(self, states: Any, trajectories: Any | None = None) -> np.ndarray:
        x = as_2d_array(states, dtype=np.float64, name="states")
        trajs = self.trajectories_ if trajectories is None else as_trajectory_list(trajectories)
        out = np.zeros((x.shape[0], len(trajs)), dtype=np.float32)
        for j, traj in enumerate(trajs):
            out[:, j] = self._aggregate(self.state_metric.pairwise_distance(x, traj))
        return out

    def pairwise_similarity(self, states: Any, trajectories: Any | None = None) -> np.ndarray:
        return (-self.pairwise_distance(states, trajectories)).astype(np.float32)


class StateToTrajectoryKMEDistance:
    """State-to-trajectory distance from a KME base kernel."""

    def __init__(self, base_kernel: StateMetric) -> None:
        self.base_kernel = base_kernel

    def fit(self, trajectories: Any) -> "StateToTrajectoryKMEDistance":
        self.trajectories_ = as_trajectory_list(trajectories)
        self.kme_ = KernelMeanEmbedding(self.base_kernel, normalize=True).fit(self.trajectories_)
        self.traj_embeddings_ = self.kme_.transform(self.trajectories_)
        self.traj_self_ = np.sum(self.traj_embeddings_ * self.traj_embeddings_, axis=1)
        return self

    def pairwise_similarity(self, states: Any, trajectories: Any | None = None) -> np.ndarray:
        x = as_2d_array(states, dtype=np.float64, name="states")
        fx = self.kme_._state_features(x)
        emb = self.traj_embeddings_ if trajectories is None else self.kme_.transform(trajectories)
        return (fx @ emb.T).astype(np.float32)

    def pairwise_distance(self, states: Any, trajectories: Any | None = None) -> np.ndarray:
        x = as_2d_array(states, dtype=np.float64, name="states")
        fx = self.kme_._state_features(x)
        emb = self.traj_embeddings_ if trajectories is None else self.kme_.transform(trajectories)
        traj_self = self.traj_self_ if trajectories is None else np.sum(emb * emb, axis=1)
        sx = np.sum(fx * fx, axis=1)
        sim = fx @ emb.T
        return np.sqrt(np.maximum(sx[:, None] + traj_self[None, :] - 2.0 * sim, 0.0)).astype(np.float32)

