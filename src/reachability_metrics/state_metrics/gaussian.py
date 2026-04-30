"""Gaussian kernel-induced state distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_sqeuclidean, require_torch
from .base import StateMetric


class GaussianKernelDistance(StateMetric):
    """Gaussian kernel and induced RKHS distance implemented with torch."""

    def __init__(
        self,
        sigma: str = "median",
        sigma_value: float | None = None,
        distance_mode: str = "rkhs",
        sample_size: int = 2048,
        random_state: int = 0,
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
        self.sigma = sigma
        self.sigma_value = sigma_value
        self.distance_mode = distance_mode
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "GaussianKernelDistance":
        torch = require_torch()
        super().fit(X, y)
        mode = str(self.sigma).lower()
        if mode == "fixed":
            if self.sigma_value is None or self.sigma_value <= 0:
                raise ValueError("sigma='fixed' requires sigma_value > 0")
            self.sigma_ = torch.tensor(float(self.sigma_value), dtype=self._dtype(), device=self._device())
            return self
        if mode != "median":
            raise ValueError("sigma must be 'median' or 'fixed'")
        values = self.X_fit_
        if values.shape[0] > int(self.sample_size):
            gen = torch.Generator(device=values.device)
            gen.manual_seed(int(self.random_state))
            idx = torch.randperm(values.shape[0], generator=gen, device=values.device)[: int(self.sample_size)]
            values = values[idx.sort().values]
        if values.shape[0] < 2:
            self.sigma_ = torch.tensor(1.0, dtype=self._dtype(), device=self._device())
            return self
        d = torch.sqrt(pairwise_sqeuclidean(values, values).clamp_min(0.0))
        positive = d[d > 1e-12]
        sigma = torch.median(positive) if positive.numel() else torch.tensor(1.0, dtype=self._dtype(), device=self._device())
        self.sigma_ = sigma.clamp_min(1e-12)
        return self

    def pairwise_similarity_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        if not hasattr(self, "sigma_"):
            self.fit(X)
        x, y = self._check_pair_inputs(X, Y)
        sq = pairwise_sqeuclidean(x, y)
        return torch.exp(-(sq / (2.0 * self.sigma_ * self.sigma_)))

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        k = self.pairwise_similarity_tensor(X, Y)
        mode = str(self.distance_mode).lower()
        if mode == "rkhs":
            return torch.sqrt(torch.clamp(2.0 - 2.0 * k, min=0.0))
        if mode in {"one_minus_kernel", "1-k"}:
            return 1.0 - k
        raise ValueError(f"Unsupported distance_mode: {self.distance_mode}")
