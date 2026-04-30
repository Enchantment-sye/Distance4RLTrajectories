"""H-step successor-state distance."""

from __future__ import annotations

import math
from typing import Any

from sklearn.base import BaseEstimator

from reachability_metrics.base import PairwiseTensorMetricMixin
from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.data.windows import future_windows
from reachability_metrics.torch_utils import as_tensor, pairwise_sqeuclidean, require_torch, torch_dtype


class HSuccessorDistance(PairwiseTensorMetricMixin, BaseEstimator):
    """Compare same-trajectory H-step future windows."""

    def __init__(
        self,
        horizon: int = 10,
        gamma: float | None = None,
        aggregation: str = "raw_l2",
        kme_metric: Any | None = None,
        wasserstein_p: int = 2,
        wasserstein_regularization: float | None = None,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.horizon = horizon
        self.gamma = gamma
        self.aggregation = aggregation
        self.kme_metric = kme_metric
        self.wasserstein_p = wasserstein_p
        self.wasserstein_regularization = wasserstein_regularization
        self.device = device
        self.dtype = dtype
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectories: Any, y: Any = None) -> "HSuccessorDistance":
        dataset = trajectories if isinstance(trajectories, TrajectoryDataset) else TrajectoryDataset.from_arrays(trajectories)
        self.windows_, self.valid_global_indices_, self.window_episode_ids_ = future_windows(dataset, self.horizon)
        self.dataset_ = dataset
        if str(self.aggregation).lower() == "kme" and self.windows_.shape[0] > 0:
            if self.kme_metric is None:
                from reachability_metrics.trajectory_metrics import GDKTrajectoryDistance

                self.kme_metric_ = GDKTrajectoryDistance(device=self.device, dtype=self.dtype)
            else:
                self.kme_metric_ = self.kme_metric
            self.kme_metric_.fit([window for window in self.windows_])
        return self

    def _weights(self, *, device: Any, dtype: Any):
        torch = require_torch()
        h = int(self.horizon)
        if self.gamma is None:
            return torch.full((h,), 1.0 / h, dtype=dtype, device=device)
        g = float(self.gamma)
        weights = g ** torch.arange(h, dtype=dtype, device=device)
        return weights / torch.sum(weights)

    def _sinkhorn_distance(self, a: Any, b: Any):
        torch = require_torch()
        p = float(self.wasserstein_p)
        reg = max(float(self.wasserstein_regularization or 0.05), 1e-8)
        cost = pairwise_sqeuclidean(a, b).clamp_min(0.0) ** (p / 2.0)
        n, m = cost.shape
        log_k = -cost / reg
        log_u = torch.zeros(n, dtype=a.dtype, device=a.device)
        log_v = torch.zeros(m, dtype=a.dtype, device=a.device)
        log_mu = torch.full((n,), -math.log(float(n)), dtype=a.dtype, device=a.device)
        log_nu = torch.full((m,), -math.log(float(m)), dtype=a.dtype, device=a.device)
        for _ in range(50):
            log_u = log_mu - torch.logsumexp(log_k + log_v[None, :], dim=1)
            log_v = log_nu - torch.logsumexp(log_k + log_u[:, None], dim=0)
        plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])
        return torch.sum(plan * cost).clamp_min(0.0) ** (1.0 / p)

    def _assignment_distance(self, a: Any, b: Any):
        torch = require_torch()
        from scipy.optimize import linear_sum_assignment

        p = float(self.wasserstein_p)
        cost = (pairwise_sqeuclidean(a, b).clamp_min(0.0) ** (p / 2.0)).detach().cpu().numpy()
        row, col = linear_sum_assignment(cost)
        value = float(cost[row, col].mean() ** (1.0 / p))
        return torch.tensor(value, dtype=a.dtype, device=a.device)

    def _wasserstein_pairwise(self, a: Any, b: Any):
        torch = require_torch()
        out = torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                if self.wasserstein_regularization is None:
                    out[i, j] = self._assignment_distance(a[i], b[j])
                else:
                    out[i, j] = self._sinkhorn_distance(a[i], b[j])
        return out

    def _window_distance_tensor(self, a: Any, b: Any):
        torch = require_torch()
        mode = str(self.aggregation).lower()
        if mode == "endpoint_l2":
            diff = a[:, None, -1, :] - b[None, :, -1, :]
            return torch.linalg.norm(diff, dim=-1)
        if mode == "mean_l2":
            diff = torch.mean(a, dim=1)[:, None, :] - torch.mean(b, dim=1)[None, :, :]
            return torch.linalg.norm(diff, dim=-1)
        if mode == "kme":
            if a.shape[0] == 0 or b.shape[0] == 0:
                return torch.empty((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
            return self.kme_metric_.pairwise_distance_tensor([w for w in a], [w for w in b])
        if mode == "wasserstein":
            return self._wasserstein_pairwise(a, b)
        if mode not in {"raw_l2", "raw"}:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        weights = self._weights(device=a.device, dtype=a.dtype)
        diff = a[:, None, :, :] - b[None, :, :, :]
        sq = torch.sum(diff * diff, dim=-1)
        return torch.sqrt(torch.clamp(torch.sum(sq * weights[None, None, :], dim=-1), min=0.0))

    def pairwise_distance_tensor(self, X: Any | None = None, Y: Any | None = None):
        if not hasattr(self, "windows_"):
            raise RuntimeError("HSuccessorDistance must be fitted")
        dtype = torch_dtype(self.dtype)
        a = self.windows_.to(dtype=dtype) if X is None else as_tensor(X, dtype=dtype, device=self.windows_.device)
        b = a if Y is None else as_tensor(Y, dtype=dtype, device=a.device)
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError("successor windows must have shape (N, H, D)")
        return self._window_distance_tensor(a, b)
