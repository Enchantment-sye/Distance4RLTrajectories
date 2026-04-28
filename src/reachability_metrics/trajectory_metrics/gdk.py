"""Gaussian Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.state_metrics import GaussianKernelDistance
from reachability_metrics.utils import as_trajectory_list
from .base import TrajectoryMetric


class GDKTrajectoryDistance(TrajectoryMetric):
    """Trajectory distribution distance using a Gaussian state kernel."""

    def __init__(self, sigma: str = "median", sigma_value: float | None = None, random_state: int = 0) -> None:
        self.sigma = sigma
        self.sigma_value = sigma_value
        self.random_state = random_state

    def fit(self, trajectories: Any, y: Any = None) -> "GDKTrajectoryDistance":
        super().fit(trajectories, y)
        states = np.concatenate(self.trajectories_, axis=0)
        self.base_kernel_ = GaussianKernelDistance(
            sigma=self.sigma,
            sigma_value=self.sigma_value,
            distance_mode="rkhs",
            random_state=self.random_state,
        ).fit(states)
        return self

    def _self_kernel(self, traj: np.ndarray) -> float:
        return float(np.mean(self.base_kernel_.pairwise_similarity(traj, traj)))

    def _cross_kernel(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(self.base_kernel_.pairwise_similarity(a, b)))

    def pairwise_similarity(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._cross_kernel(ta, tb)
        return out

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a = as_trajectory_list(A, dtype=np.float64)
        b = a if B is None else as_trajectory_list(B, dtype=np.float64)
        k_ab = self.pairwise_similarity(a, b)
        k_aa = np.asarray([self._self_kernel(t) for t in a], dtype=np.float64)
        k_bb = k_aa if B is None else np.asarray([self._self_kernel(t) for t in b], dtype=np.float64)
        return np.sqrt(np.maximum(k_aa[:, None] + k_bb[None, :] - 2.0 * k_ab, 0.0)).astype(np.float32)

