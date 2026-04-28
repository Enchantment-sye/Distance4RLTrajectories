"""Adaptive GDK squared set distance."""

from __future__ import annotations

from reachability_metrics.trajectory_metrics import AdaptiveGDKTrajectoryDistance
from .trajectory_set import TrajectorySetDistance


class AdaptiveGDK2SetDistance(TrajectorySetDistance):
    """Two-level set distance using Adaptive-GDK trajectory embeddings."""

    def __init__(self, **kwargs):
        super().__init__(trajectory_metric=AdaptiveGDKTrajectoryDistance(**kwargs))

