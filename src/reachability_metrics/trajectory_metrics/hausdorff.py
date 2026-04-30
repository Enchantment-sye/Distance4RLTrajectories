"""Hausdorff trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.torch_utils import pairwise_euclidean, require_torch
from .base import TrajectoryMetric


class HausdorffDistance(TrajectoryMetric):
    """Directed or symmetric Hausdorff distance over trajectory point sets."""

    def __init__(
        self,
        point_metric: str = "euclidean",
        directed: bool = False,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.point_metric = point_metric
        self.directed = directed

    def _directed(self, a: Any, b: Any):
        torch = require_torch()
        if str(self.point_metric).lower() != "euclidean":
            raise ValueError("Only point_metric='euclidean' is supported for torch Hausdorff")
        return torch.max(torch.min(pairwise_euclidean(a, b), dim=1).values)

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        torch = require_torch()
        a, b = self._check_pair_inputs(A, B)
        out = torch.zeros((len(a), len(b)), dtype=a[0].dtype, device=a[0].device)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                d_ab = self._directed(ta, tb)
                out[i, j] = d_ab if self.directed else torch.maximum(d_ab, self._directed(tb, ta))
        return out
