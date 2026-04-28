"""Gaussian kernel-induced state distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.utils import pairwise_sqeuclidean
from .base import StateMetric


class GaussianKernelDistance(StateMetric):
    """Gaussian kernel and induced RKHS distance."""

    def __init__(
        self,
        sigma: str = "median",
        sigma_value: float | None = None,
        distance_mode: str = "rkhs",
        sample_size: int = 2048,
        random_state: int = 0,
    ) -> None:
        self.sigma = sigma
        self.sigma_value = sigma_value
        self.distance_mode = distance_mode
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "GaussianKernelDistance":
        super().fit(X, y)
        mode = str(self.sigma).lower()
        if mode == "fixed":
            if self.sigma_value is None or self.sigma_value <= 0:
                raise ValueError("sigma='fixed' requires sigma_value > 0")
            self.sigma_ = float(self.sigma_value)
            return self
        if mode != "median":
            raise ValueError("sigma must be 'median' or 'fixed'")
        values = self.X_fit_
        if values.shape[0] > int(self.sample_size):
            rng = np.random.default_rng(self.random_state)
            values = values[np.sort(rng.choice(values.shape[0], size=int(self.sample_size), replace=False))]
        if values.shape[0] < 2:
            self.sigma_ = 1.0
            return self
        d = np.sqrt(pairwise_sqeuclidean(values, values))
        positive = d[d > 1e-12]
        self.sigma_ = float(np.median(positive)) if positive.size else 1.0
        self.sigma_ = max(self.sigma_, 1e-12)
        return self

    def pairwise_similarity(self, X: Any, Y: Any | None = None) -> np.ndarray:
        if not hasattr(self, "sigma_"):
            self.fit(X)
        x, y = self._check_pair_inputs(X, Y)
        sq = pairwise_sqeuclidean(x, y)
        return np.exp(-(sq / (2.0 * self.sigma_ * self.sigma_))).astype(np.float32)

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        k = np.asarray(self.pairwise_similarity(X, Y), dtype=np.float64)
        mode = str(self.distance_mode).lower()
        if mode == "rkhs":
            return np.sqrt(np.maximum(2.0 - 2.0 * k, 0.0)).astype(np.float32)
        if mode in {"one_minus_kernel", "1-k"}:
            return (1.0 - k).astype(np.float32)
        raise ValueError(f"Unsupported distance_mode: {self.distance_mode}")

