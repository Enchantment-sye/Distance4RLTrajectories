"""Flattened/resampled Euclidean trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.utils import resample_trajectory
from .base import TrajectoryMetric


class TrajectoryEuclideanDistance(TrajectoryMetric):
    """Euclidean distance after optional linear resampling."""

    def __init__(self, target_length: int | None = None, resample: str = "linear") -> None:
        self.target_length = target_length
        self.resample = resample

    def _prepare(self, trajs: list[np.ndarray]) -> np.ndarray:
        if self.target_length is None:
            lengths = {traj.shape[0] for traj in trajs}
            if len(lengths) != 1:
                target = max(lengths)
                return np.stack([resample_trajectory(traj, target).reshape(-1) for traj in trajs])
            return np.stack([traj.reshape(-1) for traj in trajs])
        return np.stack([resample_trajectory(traj, int(self.target_length)).reshape(-1) for traj in trajs])

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        xa = self._prepare(a)
        xb = xa if B is None else self._prepare(b)
        sq = np.sum(xa * xa, axis=1, keepdims=True) + np.sum(xb * xb, axis=1, keepdims=True).T - 2.0 * xa @ xb.T
        return np.sqrt(np.maximum(sq, 0.0)).astype(np.float32)

