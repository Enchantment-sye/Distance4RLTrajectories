"""Task-conditioned state distance."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .base import StateMetric


class TaskConditionedStateDistance(StateMetric):
    """Add value-function disagreement to a base state distance."""

    def __init__(
        self,
        base_metric: StateMetric,
        value_fn: Callable[[np.ndarray], Any] | Any,
        gamma: float = 1.0,
        value_norm: str = "l2",
        combine: str = "add",
    ) -> None:
        self.base_metric = base_metric
        self.value_fn = value_fn
        self.gamma = gamma
        self.value_norm = value_norm
        self.combine = combine

    def fit(self, X: Any, y: Any = None) -> "TaskConditionedStateDistance":
        self.base_metric.fit(X, y)
        self.X_fit_ = getattr(self.base_metric, "X_fit_", None)
        return self

    def _values(self, X: np.ndarray) -> np.ndarray:
        vf = self.value_fn
        if callable(vf):
            out = vf(X)
        elif hasattr(vf, "predict"):
            out = vf.predict(X)
        else:
            out = np.asarray(vf)
            if out.shape[0] != X.shape[0]:
                raise ValueError("precomputed value array length must match query length")
        return np.asarray(out, dtype=np.float64).reshape(X.shape[0], -1)

    def pairwise_distance(self, X: Any, Y: Any | None = None) -> np.ndarray:
        x, y = self._check_pair_inputs(X, Y)
        base = np.asarray(self.base_metric.pairwise_distance(x, y), dtype=np.float64)
        vx = self._values(x)
        vy = self._values(y)
        if str(self.value_norm).lower() == "l1":
            dv = np.sum(np.abs(vx[:, None, :] - vy[None, :, :]), axis=-1)
        else:
            dv = np.linalg.norm(vx[:, None, :] - vy[None, :, :], axis=-1)
        if str(self.combine).lower() == "multiply":
            return (base * (1.0 + float(self.gamma) * dv)).astype(np.float32)
        return (base + float(self.gamma) * dv).astype(np.float32)

