"""Isolation Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.state_metrics import IsolationKernelDistance
from .base import TrajectoryMetric
from .kme import KernelMeanEmbedding


class IDKTrajectoryDistance(TrajectoryMetric):
    """Trajectory distance from IK kernel mean embeddings."""

    def __init__(
        self,
        ensemble_size: int = 100,
        subsample_size: int = 32,
        temperature: float = 0.01,
        device: str = "auto",
        batch_size: int = 4096,
        random_state: int = 0,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.subsample_size = subsample_size
        self.temperature = temperature
        self.device = device
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, trajectories: Any, y: Any = None) -> "IDKTrajectoryDistance":
        super().fit(trajectories, y)
        base = IsolationKernelDistance(
            ensemble_size=self.ensemble_size,
            subsample_size=self.subsample_size,
            temperature=self.temperature,
            device=self.device,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )
        self.kme_ = KernelMeanEmbedding(base, normalize=True).fit(self.trajectories_)
        return self

    def transform(self, trajectories: Any) -> np.ndarray:
        return self.kme_.transform(trajectories)

    def pairwise_similarity(self, A: Any, B: Any | None = None) -> np.ndarray:
        return self.kme_.pairwise_kernel(A, B)

    def pairwise_distance(self, A: Any, B: Any | None = None) -> np.ndarray:
        return self.kme_.pairwise_distance(A, B)

