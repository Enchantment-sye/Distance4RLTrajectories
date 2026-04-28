"""Hausdorff trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from .base import TrajectoryMetric


class HausdorffDistance(TrajectoryMetric):
    """Directed or symmetric Hausdorff distance over trajectory point sets."""

    def __init__(self, point_metric: str = "euclidean", directed: bool = False) -> None:
        self.point_metric = point_metric
        self.directed = directed

    def _directed(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.min(cdist(a, b, metric=self.point_metric), axis=1)))

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        out = np.zeros((len(a), len(b)), dtype=np.float32)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                d_ab = self._directed(ta, tb)
                out[i, j] = d_ab if self.directed else max(d_ab, self._directed(tb, ta))
        return out

