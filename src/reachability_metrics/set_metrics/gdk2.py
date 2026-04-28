"""GDK squared set distance."""

from __future__ import annotations

from reachability_metrics.trajectory_metrics import GDKTrajectoryDistance
from .trajectory_set import TrajectorySetDistance


class GDK2SetDistance(TrajectorySetDistance):
    """Two-level set distance using GDK trajectory embeddings."""

    def __init__(self, **kwargs):
        super().__init__(trajectory_metric=GDKTrajectoryDistance(**kwargs))

