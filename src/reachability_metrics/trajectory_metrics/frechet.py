"""Discrete Frechet trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .base import TrajectoryMetric


class FrechetDistance(TrajectoryMetric):
    """Discrete Frechet distance."""

    def __init__(self, point_metric: str = "euclidean") -> None:
        self.point_metric = point_metric

    def _frechet(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = cdist(a, b, metric=self.point_metric)
        n, m = dist.shape
        ca = np.full((n, m), -1.0, dtype=np.float64)

        def rec(i: int, j: int) -> float:
            if ca[i, j] > -0.5:
                return float(ca[i, j])
            if i == 0 and j == 0:
                ca[i, j] = dist[0, 0]
            elif i > 0 and j == 0:
                ca[i, j] = max(rec(i - 1, 0), dist[i, 0])
            elif i == 0 and j > 0:
                ca[i, j] = max(rec(0, j - 1), dist[0, j])
            else:
                ca[i, j] = max(min(rec(i - 1, j), rec(i - 1, j - 1), rec(i, j - 1)), dist[i, j])
            return float(ca[i, j])

        return rec(n - 1, m - 1)

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._frechet(ta, tb)
        return out

