"""Dynamic Time Warping trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_euclidean, require_torch
from .base import TrajectoryMetric


class DTWDistance(TrajectoryMetric):
    """Classic DTW over pointwise Euclidean distances."""

    def __init__(
        self,
        point_metric: str = "euclidean",
        window: int | None = None,
        normalize: bool = True,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.point_metric = point_metric
        self.window = window
        self.normalize = normalize

    def _dtw(self, a: Any, b: Any):
        torch = require_torch()
        if str(self.point_metric).lower() != "euclidean":
            raise ValueError("Only point_metric='euclidean' is supported for torch DTW")
        cost = pairwise_euclidean(a, b)
        n, m = cost.shape
        w = max(int(self.window), abs(n - m)) if self.window is not None else max(n, m)
        dp = torch.full((n + 1, m + 1), float("inf"), dtype=cost.dtype, device=cost.device)
        dp[0, 0] = 0.0
        for i in range(1, n + 1):
            j0 = max(1, i - w)
            j1 = min(m, i + w) + 1
            for j in range(j0, j1):
                dp[i, j] = cost[i - 1, j - 1] + torch.min(torch.stack([dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]]))
        val = dp[n, m]
        if self.normalize:
            val /= float(n + m)
        return val

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        torch = require_torch()
        a, b = self._check_pair_inputs(A, B)
        out = torch.zeros((len(a), len(b)), dtype=a[0].dtype, device=a[0].device)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._dtw(ta, tb)
        return out
