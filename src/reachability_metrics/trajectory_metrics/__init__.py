"""Trajectory-to-trajectory metrics."""

from .base import TrajectoryMetric
from .euclidean import TrajectoryEuclideanDistance
from .dtw import DTWDistance
from .hausdorff import HausdorffDistance
from .frechet import FrechetDistance
from .wasserstein import TrajectoryWassersteinDistance
from .kme import KernelMeanEmbedding
from .idk import IDKTrajectoryDistance
from .gdk import GDKTrajectoryDistance
from .adaptive_gdk import AdaptiveGDKTrajectoryDistance
from .t2vec import T2VecDistance
from .task_conditioned import TaskConditionedTrajectoryDistance
from .registry import TRAJECTORY_METRIC_REGISTRY, build_trajectory_metric

__all__ = [
    "TrajectoryMetric",
    "TrajectoryEuclideanDistance",
    "DTWDistance",
    "HausdorffDistance",
    "FrechetDistance",
    "TrajectoryWassersteinDistance",
    "KernelMeanEmbedding",
    "IDKTrajectoryDistance",
    "GDKTrajectoryDistance",
    "AdaptiveGDKTrajectoryDistance",
    "T2VecDistance",
    "TaskConditionedTrajectoryDistance",
    "TRAJECTORY_METRIC_REGISTRY",
    "build_trajectory_metric",
]
