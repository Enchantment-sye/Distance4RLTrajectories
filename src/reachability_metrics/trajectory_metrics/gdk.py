"""Gaussian Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.state_metrics import GaussianKernelDistance
from .base import TrajectoryMetric
from .kme import KMETrajectoryDelegateMixin


class GDKTrajectoryDistance(KMETrajectoryDelegateMixin, TrajectoryMetric):
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
        super().__init__(
            device=device,
            dtype=dtype,
            return_numpy=return_numpy,
            output_format=output_format,
        )
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
        base_kernel = GaussianKernelDistance(
            sigma=self.sigma,
            sigma_value=self.sigma_value,
            distance_mode="rkhs",
            random_state=self.random_state,
            device=self.device,
            dtype=self.dtype,
        )
        self._fit_kme(
            base_kernel,
            self.trajectories_,
            distance_mode=self.distance_mode,
            feature_approximation=self.feature_approximation,
            num_landmarks=self.num_landmarks,
            landmark_strategy=self.landmark_strategy,
            eps=self.eps,
            random_state=self.random_state,
        )
        return self
