"""Gaussian Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.state_metrics import GaussianKernelDistance
from .base import TrajectoryMetric
from .kme import KernelMeanEmbedding


class GDKTrajectoryDistance(TrajectoryMetric):
    """Trajectory distribution distance using a Gaussian state kernel."""

    def __init__(
        self,
        sigma: str = "median",
        sigma_value: float | None = None,
        random_state: int = 0,
        distance_mode: str = "rkhs_norm",
        feature_approximation: str = "exact",
        num_landmarks: int = 512,
        landmark_strategy: str = "random",
        eps: float = 1e-6,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, return_numpy=return_numpy, output_format=output_format)
        self.sigma = sigma
        self.sigma_value = sigma_value
        self.random_state = random_state
        self.distance_mode = distance_mode
        self.feature_approximation = feature_approximation
        self.num_landmarks = num_landmarks
        self.landmark_strategy = landmark_strategy
        self.eps = eps

    def fit(self, trajectories: Any, y: Any = None) -> "GDKTrajectoryDistance":
        super().fit(trajectories, y)
        self.base_kernel_ = GaussianKernelDistance(
            sigma=self.sigma,
            sigma_value=self.sigma_value,
            distance_mode="rkhs",
            random_state=self.random_state,
            device=self.device,
            dtype=self.dtype,
        )
        self.kme_ = KernelMeanEmbedding(
            self.base_kernel_,
            distance_mode=self.distance_mode,
            feature_approximation=self.feature_approximation,
            num_landmarks=self.num_landmarks,
            landmark_strategy=self.landmark_strategy,
            eps=self.eps,
            random_state=self.random_state,
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
