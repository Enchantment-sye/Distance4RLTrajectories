"""Adaptive Gaussian Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.state_metrics import AdaptiveGaussianDistance
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

        super(GDKTrajectoryDistance, self).fit(trajectories, y)
        states = torch.cat(self.trajectories_, dim=0)
        self.base_kernel_ = AdaptiveGaussianDistance(
            k=self.k,
            eps=self.eps,
            distance_mode="one_minus_kernel",
            device=self.device,
            dtype=self.dtype,
        ).fit(states)
        from .kme import KernelMeanEmbedding

        self.kme_ = KernelMeanEmbedding(
            self.base_kernel_,
            distance_mode=self.distance_mode,
            device=self.device,
            dtype=self.dtype,
            return_numpy=self.return_numpy,
            output_format=self.output_format,
        ).fit(self.trajectories_)
        return self
