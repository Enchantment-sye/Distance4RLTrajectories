"""Dynamic Time Warping trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .base import TrajectoryMetric


class DTWDistance(TrajectoryMetric):
    """Classic DTW over pointwise Euclidean distances."""

    def __init__(self, point_metric: str = "euclidean", window: int | None = None, normalize: bool = True) -> None:
        self.point_metric = point_metric
        self.window = window
        self.normalize = normalize

    def _dtw(self, a: np.ndarray, b: np.ndarray) -> float:
        cost = cdist(a, b, metric=self.point_metric)
        n, m = cost.shape
        w = max(int(self.window), abs(n - m)) if self.window is not None else max(n, m)
        dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            j0 = max(1, i - w)
            j1 = min(m, i + w) + 1
            for j in range(j0, j1):
                dp[i, j] = cost[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
        val = float(dp[n, m])
        if self.normalize:
            val /= float(n + m)
        return val

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._dtw(ta, tb)
        return out

