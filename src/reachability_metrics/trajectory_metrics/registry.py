"""Factory registry for trajectory metrics."""

from __future__ import annotations

from typing import Any, Callable

from reachability_metrics.registry import MetricRegistry

from .adaptive_gdk import AdaptiveGDKTrajectoryDistance
from .dtw import DTWDistance
from .euclidean import TrajectoryEuclideanDistance
from .frechet import FrechetDistance
from .gdk import GDKTrajectoryDistance
from .hausdorff import HausdorffDistance
from .idk import IDKTrajectoryDistance
from .t2vec import T2VecDistance
from .wasserstein import TrajectoryWassersteinDistance


TRAJECTORY_METRIC_REGISTRY: dict[str, Callable[..., Any]] = {
    "euclidean": TrajectoryEuclideanDistance,
    "trajectory_euclidean": TrajectoryEuclideanDistance,
    "dtw": DTWDistance,
    "hausdorff": HausdorffDistance,
    "frechet": FrechetDistance,
    "wasserstein": TrajectoryWassersteinDistance,
    "wasserstein_w2": TrajectoryWassersteinDistance,
    "idk": IDKTrajectoryDistance,
    "gdk": GDKTrajectoryDistance,
    "adaptive_gdk": AdaptiveGDKTrajectoryDistance,
    "t2vec": T2VecDistance,
}

_TRAJECTORY_METRIC_FACTORY = MetricRegistry("trajectory metric", TRAJECTORY_METRIC_REGISTRY)


def build_trajectory_metric(method: str, **kwargs: Any) -> Any:
    """Construct a trajectory metric from a public method key."""
    return _TRAJECTORY_METRIC_FACTORY.build(method, **kwargs)
