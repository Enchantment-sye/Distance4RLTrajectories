"""Task-conditioned trajectory distance."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import TrajectoryMetric


class TaskConditionedTrajectoryDistance(TrajectoryMetric):
    """Add value-trajectory disagreement to a base trajectory distance."""

    def __init__(
        self,
        base_trajectory_metric: TrajectoryMetric,
        value_fn: Callable[[np.ndarray], Any] | Any,
        gamma: float = 1.0,
        aggregation: str = "mean",
        beta: float = 0.99,
    ) -> None:
        self.base_trajectory_metric = base_trajectory_metric
        self.value_fn = value_fn
        self.gamma = gamma
        self.aggregation = aggregation
        self.beta = beta

    def fit(self, trajectories: Any, y: Any = None) -> "TaskConditionedTrajectoryDistance":
        self.base_trajectory_metric.fit(trajectories, y)
        self.trajectories_ = self.base_trajectory_metric.trajectories_
        return self

    def _state_values(self, states: np.ndarray) -> np.ndarray:
        vf = self.value_fn
        if callable(vf):
            out = vf(states)
        elif hasattr(vf, "predict"):
            out = vf.predict(states)
        else:
            out = np.asarray(vf)
            if out.shape[0] < states.shape[0]:
                raise ValueError("precomputed value array is too short")
            out = out[: states.shape[0]]
        return np.asarray(out, dtype=np.float64).reshape(states.shape[0], -1)

    def _trajectory_value(self, traj: np.ndarray) -> np.ndarray:
        vals = self._state_values(traj)
        mode = str(self.aggregation).lower()
        if mode == "endpoint":
            return vals[-1]
        if mode == "discounted_sum":
            weights = np.power(float(self.beta), np.arange(vals.shape[0], dtype=np.float64))[:, None]
            return np.sum(vals * weights, axis=0)
        return np.mean(vals, axis=0)

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        a, b = self._check_pair_inputs(A, B)
        base = np.asarray(self.base_trajectory_metric.pairwise_distance(a, b), dtype=np.float64)
        va = np.stack([self._trajectory_value(t) for t in a])
        vb = va if B is None else np.stack([self._trajectory_value(t) for t in b])
        dv = np.linalg.norm(va[:, None, :] - vb[None, :, :], axis=-1)
        return (base + float(self.gamma) * dv).astype(np.float32)

