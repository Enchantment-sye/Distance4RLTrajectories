"""Euclidean state distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.utils import pairwise_sqeuclidean
from .base import StateMetric


class EuclideanDistance(StateMetric):
    """Plain Euclidean distance."""

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        x, y = self._check_pair_inputs(X, Y)
        return np.sqrt(pairwise_sqeuclidean(x, y)).astype(np.float32)

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        return (-self.pairwise_distance(X, Y)).astype(np.float32)

