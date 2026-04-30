"""Task-conditioned trajectory distance."""

from __future__ import annotations

from typing import Any, Callable

from reachability_metrics.torch_utils import as_2d_tensor, as_tensor, cpu_numpy, require_torch
from .base import TrajectoryMetric


class TaskConditionedTrajectoryDistance(TrajectoryMetric):
    """Add value-trajectory disagreement to a base trajectory distance."""

    def __init__(
        self,
        base_trajectory_metric: TrajectoryMetric,
        value_fn: Callable[[Any], Any] | Any,
        gamma: float = 1.0,
        aggregation: str = "mean",
        beta: float = 0.99,
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(return_numpy=return_numpy, output_format=output_format)
        self.base_trajectory_metric = base_trajectory_metric
        self.value_fn = value_fn
        self.gamma = gamma
        self.aggregation = aggregation
        self.beta = beta

    def fit(self, trajectories: Any, y: Any = None) -> "TaskConditionedTrajectoryDistance":
        self.base_trajectory_metric.fit(trajectories, y)
        self.trajectories_ = self.base_trajectory_metric.trajectories_
        return self

    def _state_values(self, states: Any):
        vf = self.value_fn
        if callable(vf):
            out = vf(states)
        elif hasattr(vf, "predict"):
            out = vf.predict(cpu_numpy(states))
        else:
            out = vf
            if out.shape[0] < states.shape[0]:
                raise ValueError("precomputed value array is too short")
            out = out[: states.shape[0]]
        return as_2d_tensor(out, dtype=states.dtype, device=states.device, name="values").reshape(states.shape[0], -1)

    def _trajectory_value(self, traj: Any):
        torch = require_torch()
        vals = self._state_values(traj)
        mode = str(self.aggregation).lower()
        if mode == "endpoint":
            return vals[-1]
        if mode == "discounted_sum":
            weights = (float(self.beta) ** torch.arange(vals.shape[0], dtype=vals.dtype, device=vals.device))[:, None]
            return torch.sum(vals * weights, dim=0)
        return torch.mean(vals, dim=0)

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        torch = require_torch()
        a, b = self._check_pair_inputs(A, B)
        base = as_tensor(self.base_trajectory_metric.pairwise_distance(a, b), dtype=a[0].dtype, device=a[0].device)
        va = torch.stack([self._trajectory_value(t) for t in a], dim=0)
        vb = va if B is None else torch.stack([self._trajectory_value(t) for t in b], dim=0)
        dv = torch.linalg.norm(va[:, None, :] - vb[None, :, :], dim=-1)
        return base + float(self.gamma) * dv
