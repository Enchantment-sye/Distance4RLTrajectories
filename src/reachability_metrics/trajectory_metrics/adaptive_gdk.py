"""Adaptive Gaussian Distributional Kernel trajectory distance."""

from __future__ import annotations

from typing import Any

from reachability_metrics.state_metrics import AdaptiveGaussianDistance
from .gdk import GDKTrajectoryDistance


class AdaptiveGDKTrajectoryDistance(GDKTrajectoryDistance):
    """Trajectory distribution distance using adaptive Gaussian state kernel."""

    def __init__(self, k: int = 10, eps: float = 1e-6) -> None:
        self.k = k
        self.eps = eps

    def fit(self, trajectories: Any, y: Any = None) -> "AdaptiveGDKTrajectoryDistance":
        from numpy import concatenate

        super(GDKTrajectoryDistance, self).fit(trajectories, y)
        states = concatenate(self.trajectories_, axis=0)
        self.base_kernel_ = AdaptiveGaussianDistance(k=self.k, eps=self.eps, distance_mode="one_minus_kernel").fit(states)
        return self

