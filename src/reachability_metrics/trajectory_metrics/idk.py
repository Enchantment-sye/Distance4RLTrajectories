"""Isolation Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.state_metrics import IsolationKernelDistance
from .base import TrajectoryMetric
from .kme import KMETrajectoryDelegateMixin


class IDKTrajectoryDistance(KMETrajectoryDelegateMixin, TrajectoryMetric):
    """Trajectory distance from IK kernel mean embeddings."""

    def __init__(
        self,
        ensemble_size: int = 100,
        subsample_size: int = 32,
        temperature: float = 0.01,
        device: str = "auto",
        dtype: str = "float32",
        batch_size: int = 4096,
        block_size: int = 4096,
        random_state: int = 0,
        distance_mode: str = "rkhs_norm",
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
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size
        self.temperature = temperature
        self.block_size = block_size
        self.random_state = random_state
        self.distance_mode = distance_mode

    def fit(self, trajectories: Any, y: Any = None) -> "IDKTrajectoryDistance":
        super().fit(trajectories, y)
        base = IsolationKernelDistance(
            ensemble_size=self.ensemble_size,
            subsample_size=self.subsample_size,
            temperature=self.temperature,
            device=self.device,
            dtype=self.dtype,
            batch_size=self.batch_size,
            block_size=self.block_size,
            random_state=self.random_state,
        )
        self._fit_kme(
            base,
            self.trajectories_,
            normalize=False,
            distance_mode=self.distance_mode,
        )
        return self
