"""Wasserstein trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .base import TrajectoryMetric


class TrajectoryWassersteinDistance(TrajectoryMetric):
    """Uniform optimal assignment distance between trajectory point clouds."""

    def __init__(self, point_metric: str = "euclidean", p: int = 2, regularization: float | None = None) -> None:
        self.point_metric = point_metric
        self.p = p
        self.regularization = regularization

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.regularization is not None:
            try:
                import ot

                weights_a = np.full(a.shape[0], 1.0 / a.shape[0])
                weights_b = np.full(b.shape[0], 1.0 / b.shape[0])
                cost = cdist(a, b, metric=self.point_metric) ** float(self.p)
                return float(ot.sinkhorn2(weights_a, weights_b, cost, reg=float(self.regularization)) ** (1.0 / self.p))
            except Exception as exc:
                raise ModuleNotFoundError("Install reachability-metrics[optimal_transport] for regularized OT") from exc
        cost = cdist(a, b, metric=self.point_metric) ** float(self.p)
        row, col = linear_sum_assignment(cost)
        return float(np.mean(cost[row, col]) ** (1.0 / self.p))

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._distance(ta, tb)
        return out

