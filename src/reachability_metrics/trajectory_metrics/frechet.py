"""Discrete Frechet trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_euclidean, require_torch
from .base import TrajectoryMetric


class FrechetDistance(TrajectoryMetric):
    """Discrete Frechet distance."""

    def __init__(
        self,
        point_metric: str = "euclidean",
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.point_metric = point_metric

    def _frechet(self, a: Any, b: Any):
        torch = require_torch()
        if str(self.point_metric).lower() != "euclidean":
            raise ValueError("Only point_metric='euclidean' is supported for torch Frechet")
        dist = pairwise_euclidean(a, b)
        n, m = dist.shape
        ca = torch.empty((n, m), dtype=dist.dtype, device=dist.device)
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    ca[i, j] = dist[0, 0]
                elif i > 0 and j == 0:
                    ca[i, j] = torch.maximum(ca[i - 1, 0], dist[i, 0])
                elif i == 0 and j > 0:
                    ca[i, j] = torch.maximum(ca[0, j - 1], dist[0, j])
                else:
                    ca[i, j] = torch.maximum(torch.min(torch.stack([ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]])), dist[i, j])
        return ca[n - 1, m - 1]

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        torch = require_torch()
        a, b = self._check_pair_inputs(A, B)
        out = torch.zeros((len(a), len(b)), dtype=a[0].dtype, device=a[0].device)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._frechet(ta, tb)
        return out
