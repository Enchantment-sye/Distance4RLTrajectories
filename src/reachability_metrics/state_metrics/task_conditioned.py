"""Task-conditioned state distance."""

from __future__ import annotations

from typing import Any, Callable

from reachability_metrics.torch_utils import as_2d_tensor, as_tensor, cpu_numpy, require_torch
from .base import StateMetric


class TaskConditionedStateDistance(StateMetric):
    """Add value-function disagreement to a base state distance."""

    def __init__(
        self,
        base_metric: StateMetric,
        value_fn: Callable[[Any], Any] | Any,
        gamma: float = 1.0,
        value_norm: str = "l2",
        combine: str = "add",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(return_numpy=return_numpy, output_format=output_format)
        self.base_metric = base_metric
        self.value_fn = value_fn
        self.gamma = gamma
        self.value_norm = value_norm
        self.combine = combine

    def fit(self, X: Any, y: Any = None) -> "TaskConditionedStateDistance":
        self.base_metric.fit(X, y)
        self.X_fit_ = getattr(self.base_metric, "X_fit_", None)
        return self

    def _values(self, X: Any):
        vf = self.value_fn
        if callable(vf):
            out = vf(X)
        elif hasattr(vf, "predict"):
            out = vf.predict(cpu_numpy(X))
        else:
            out = vf
            if out.shape[0] != X.shape[0]:
                raise ValueError("precomputed value array length must match query length")
        return as_2d_tensor(out, dtype=X.dtype, device=X.device, name="values").reshape(X.shape[0], -1)

    def pairwise_distance_tensor(self, X: Any, Y: Any | None = None):
        torch = require_torch()
        x, y = self._check_pair_inputs(X, Y)
        base = as_tensor(self.base_metric.pairwise_distance(x, y), dtype=x.dtype, device=x.device)
        vx = self._values(x)
        vy = self._values(y)
        if str(self.value_norm).lower() == "l1":
            dv = torch.sum(torch.abs(vx[:, None, :] - vy[None, :, :]), dim=-1)
        else:
            dv = torch.linalg.norm(vx[:, None, :] - vy[None, :, :], dim=-1)
        if str(self.combine).lower() == "multiply":
            return base * (1.0 + float(self.gamma) * dv)
        return base + float(self.gamma) * dv
