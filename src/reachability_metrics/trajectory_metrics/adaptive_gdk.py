"""Adaptive Gaussian Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.state_metrics import AdaptiveGaussianDistance
from .base import TrajectoryMetric
from .gdk import GDKTrajectoryDistance


class AdaptiveGDKTrajectoryDistance(GDKTrajectoryDistance):
    """Trajectory distribution distance using adaptive Gaussian state kernel."""

    def __init__(
        self,
        k: int = 10,
        eps: float = 1e-6,
        distance_mode: str = "rkhs_norm",
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        super().__init__(
            distance_mode=distance_mode,
            device=device,
            dtype=dtype,
            return_numpy=return_numpy,
            output_format=output_format,
        )
        self.k = k
        self.eps = eps

    def fit(self, trajectories: Any, y: Any = None) -> "AdaptiveGDKTrajectoryDistance":
        import torch

        TrajectoryMetric.fit(self, trajectories, y)
        states = torch.cat(self.trajectories_, dim=0)
        base_kernel = AdaptiveGaussianDistance(
            k=self.k,
            eps=self.eps,
            distance_mode="one_minus_kernel",
            device=self.device,
            dtype=self.dtype,
        ).fit(states)

        self._fit_kme(
            base_kernel,
            self.trajectories_,
            distance_mode=self.distance_mode,
        )
        return self
