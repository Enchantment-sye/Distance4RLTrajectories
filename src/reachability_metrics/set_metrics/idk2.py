"""IDK squared set distance."""

from __future__ import annotations

from reachability_metrics.trajectory_metrics import IDKTrajectoryDistance
from .trajectory_set import TrajectorySetDistance, _split_set_metric_kwargs


class IDK2SetDistance(TrajectorySetDistance):
    """Two-level set distance using IDK trajectory embeddings."""

    def __init__(self, **kwargs):
        set_kwargs, metric_kwargs = _split_set_metric_kwargs(kwargs)
        super().__init__(trajectory_metric=IDKTrajectoryDistance(**metric_kwargs), **set_kwargs)
