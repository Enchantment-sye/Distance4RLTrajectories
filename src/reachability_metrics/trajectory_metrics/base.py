"""Base trajectory metric interface."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from reachability_metrics.utils import as_trajectory_list


class TrajectoryMetric(BaseEstimator):
    """Sklearn-style trajectory metric base class."""

    def fit(self, trajectories: Any, y: Any = None) -> "TrajectoryMetric":
        self.trajectories_ = as_trajectory_list(trajectories, dtype=np.float64)
        return self

    def _check_pair_inputs(self, A: Any, B: Any | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
        a = as_trajectory_list(A, dtype=np.float64)
        b = a if B is None else as_trajectory_list(B, dtype=np.float64)
        return a, b

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        raise NotImplementedError

    def pairwise_similarity(self, A: Any, B: Any | None = None) -> np.ndarray:
        return -self.pairwise_distance(A, B)

