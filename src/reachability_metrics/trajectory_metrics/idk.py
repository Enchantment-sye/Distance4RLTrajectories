"""Isolation Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

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
        self.kme_ = KernelMeanEmbedding(
            base,
            normalize=False,
            distance_mode=self.distance_mode,
            device=self.device,
            dtype=self.dtype,
            return_numpy=self.return_numpy,
            output_format=self.output_format,
        ).fit(self.trajectories_)
        return self

    def transform_tensor(self, trajectories: Any):
        return self.kme_.transform_tensor(trajectories)

    def transform(self, trajectories: Any):
        return self._return(self.transform_tensor(trajectories))

    def pairwise_similarity_tensor(self, A: Any, B: Any | None = None):
        return self.kme_.pairwise_similarity_tensor(A, B)

    def pairwise_distance_tensor(self, A: Any, B: Any | None = None):
        return self.kme_.pairwise_distance_tensor(A, B)
