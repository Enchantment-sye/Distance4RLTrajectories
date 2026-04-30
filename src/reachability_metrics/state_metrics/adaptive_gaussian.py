"""Adaptive local-bandwidth Gaussian state distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_sqeuclidean, require_torch
from .base import StateMetric


class AdaptiveGaussianDistance(StateMetric):
    """Gaussian kernel with per-point bandwidth from kNN density."""

    def __init__(
        self,
        k: int = 10,
        eps: float = 1e-6,
        distance_mode: str = "one_minus_kernel",
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 4096,
        block_size: int = 4096,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            block_size=block_size,
            return_numpy=return_numpy,
            output_format=output_format,
        )
        self.k = k
        self.eps = eps
        self.distance_mode = distance_mode

    def fit(self, X: Any, y: Any = None) -> "AdaptiveGaussianDistance":
        torch = require_torch()
        super().fit(X, y)
        if self.X_fit_.shape[0] == 0:
            raise ValueError("AdaptiveGaussianDistance requires at least one point")
        n = self.X_fit_.shape[0]
        if n == 1:
            self.effective_k_ = 1
            self.train_sigmas_ = torch.ones(1, dtype=self._dtype(), device=self._device())
            return self
        self.effective_k_ = min(max(int(self.k), 1), n - 1)
        distances = torch.sqrt(pairwise_sqeuclidean(self.X_fit_, self.X_fit_).clamp_min(0.0))
        vals = torch.topk(distances, k=self.effective_k_ + 1, largest=False, dim=1).values
        self.train_sigmas_ = vals[:, -1].clamp_min(max(float(self.eps), 1e-12))
        return self

    def estimate_sigmas(self, X: Any):
        """Estimate local bandwidth for query states."""
        torch = require_torch()
        if not hasattr(self, "X_fit_"):
            raise RuntimeError("AdaptiveGaussianDistance must be fitted")
        x = self._check_pair_inputs(X, self.X_fit_)[0]
        neighbors = min(max(int(self.effective_k_), 1), self.X_fit_.shape[0])
        distances = torch.sqrt(pairwise_sqeuclidean(x, self.X_fit_).clamp_min(0.0))
        vals = torch.topk(distances, k=neighbors, largest=False, dim=1).values
        return vals[:, -1].clamp_min(max(float(self.eps), 1e-12))

    def pairwise_similarity_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        if not hasattr(self, "X_fit_"):
            self.fit(X)
        x, y = self._check_pair_inputs(X, Y)
        sx = self.estimate_sigmas(x)
        sy = self.estimate_sigmas(y)
        sq = pairwise_sqeuclidean(x, y)
        denom = (sx[:, None] * sy[None, :]).clamp_min(max(float(self.eps), 1e-12))
        return torch.exp(-(sq / denom))

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        k = self.pairwise_similarity_tensor(X, Y)
        mode = str(self.distance_mode).lower()
        if mode in {"rkhs", "rkhs_distance"}:
            return torch.sqrt(torch.clamp(2.0 - 2.0 * k, min=0.0))
        if mode in {"one_minus_kernel", "1-k"}:
            return 1.0 - k
        raise ValueError(f"Unsupported distance_mode: {self.distance_mode}")
