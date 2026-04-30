"""Adaptive GDK squared set distance."""

from __future__ import annotations

from reachability_metrics.trajectory_metrics import AdaptiveGDKTrajectoryDistance
from .trajectory_set import TrajectorySetDistance


class AdaptiveGDK2SetDistance(TrajectorySetDistance):
    """Two-level set distance using Adaptive-GDK trajectory embeddings."""

    def __init__(self, **kwargs):
        set_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in {"normalize", "return_numpy", "output_format"}}
        if "set_distance_mode" in kwargs:
            set_kwargs["distance_mode"] = kwargs.pop("set_distance_mode")
        super().__init__(trajectory_metric=AdaptiveGDKTrajectoryDistance(**kwargs), **set_kwargs)
