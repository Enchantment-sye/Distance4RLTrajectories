"""Wasserstein trajectory distance."""

from __future__ import annotations

import math
from typing import Any

from scipy.optimize import linear_sum_assignment

from reachability_metrics.torch_utils import pairwise_sqeuclidean, require_torch
from .base import TrajectoryMetric


class TrajectoryWassersteinDistance(TrajectoryMetric):
    """Uniform optimal assignment distance between trajectory point clouds."""

    def __init__(
        self,
        point_metric: str = "euclidean",
        p: int = 2,
        regularization: float | None = None,
        sinkhorn_iters: int = 50,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.point_metric = point_metric
        self.p = p
        self.regularization = regularization
        self.sinkhorn_iters = sinkhorn_iters

    def _sinkhorn(self, cost: Any):
        torch = require_torch()
        reg = max(float(self.regularization), 1e-8)
        n, m = cost.shape
        log_k = -cost / reg
        log_u = torch.zeros(n, dtype=cost.dtype, device=cost.device)
        log_v = torch.zeros(m, dtype=cost.dtype, device=cost.device)
        log_mu = torch.full((n,), -math.log(float(n)), dtype=cost.dtype, device=cost.device)
        log_nu = torch.full((m,), -math.log(float(m)), dtype=cost.dtype, device=cost.device)
        for _ in range(int(self.sinkhorn_iters)):
            log_u = log_mu - torch.logsumexp(log_k + log_v[None, :], dim=1)
            log_v = log_nu - torch.logsumexp(log_k + log_u[:, None], dim=0)
        plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])
        return torch.sum(plan * cost)

    def _distance(self, a: Any, b: Any):
        if str(self.point_metric).lower() != "euclidean":
            raise ValueError("Only point_metric='euclidean' is supported for torch Wasserstein")
        cost = pairwise_sqeuclidean(a, b).clamp_min(0.0) ** (float(self.p) / 2.0)
        if self.regularization is not None:
            return self._sinkhorn(cost).clamp_min(0.0) ** (1.0 / float(self.p))
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        value = cost[row, col].mean().clamp_min(0.0) ** (1.0 / float(self.p))
        return value.to(dtype=a.dtype, device=a.device)

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        torch = require_torch()
        a, b = self._check_pair_inputs(A, B)
        out = torch.zeros((len(a), len(b)), dtype=a[0].dtype, device=a[0].device)
        for i, ta in enumerate(a):
            for j, tb in enumerate(b):
                out[i, j] = self._distance(ta, tb)
        return out
