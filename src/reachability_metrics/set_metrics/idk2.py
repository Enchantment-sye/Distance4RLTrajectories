"""IDK squared set distance."""

from __future__ import annotations

from reachability_metrics.trajectory_metrics import IDKTrajectoryDistance
from .trajectory_set import TrajectorySetDistance


class IDK2SetDistance(TrajectorySetDistance):
    """Two-level set distance using IDK trajectory embeddings."""

    def __init__(self, **kwargs):
        super().__init__(trajectory_metric=IDKTrajectoryDistance(**kwargs))

