"""Two-level trajectory-set distance."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from reachability_metrics.trajectory_metrics import (
    AdaptiveGDKTrajectoryDistance,
    GDKTrajectoryDistance,
    IDKTrajectoryDistance,
)
from reachability_metrics.utils import as_trajectory_list, pairwise_sqeuclidean


class TrajectorySetDistance(BaseEstimator):
    """A compact two-level KME-style set metric over trajectory embeddings."""

    def __init__(self, trajectory_metric: Any | None = None, normalize: bool = True) -> None:
        self.trajectory_metric = trajectory_metric
        self.normalize = normalize

    def fit(self, trajectory_sets: list[list[Any]], y: Any = None) -> "TrajectorySetDistance":
        self.trajectory_sets_ = trajectory_sets
        flat = [traj for group in trajectory_sets for traj in group]
        metric = self.trajectory_metric or GDKTrajectoryDistance()
        metric.fit(flat)
        self.trajectory_metric_ = metric
        self._flat_ref_ = flat
        self.reference_embeddings_ = self.transform_trajectories(flat)
        return self

    def transform_trajectory(self, trajectory: Any) -> np.ndarray:
        if hasattr(self.trajectory_metric_, "transform"):
            emb = self.trajectory_metric_.transform([trajectory])[0]
        else:
            d = self.trajectory_metric_.pairwise_distance([trajectory], getattr(self, "_flat_ref_", [trajectory]))
            emb = -d.reshape(-1)
        emb = np.asarray(emb, dtype=np.float32)
        if self.normalize:
            emb = emb / max(float(np.linalg.norm(emb)), 1e-12)
        return emb

    def transform_trajectories(self, trajectories: Any) -> np.ndarray:
        trajs = as_trajectory_list(trajectories)
        if hasattr(self.trajectory_metric_, "transform"):
            emb = self.trajectory_metric_.transform(trajs)
            if self.normalize:
                emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            return emb.astype(np.float32)
        ref = getattr(self, "_flat_ref_", trajs)
        d = self.trajectory_metric_.pairwise_distance(trajs, ref)
        return (-d).astype(np.float32)

    def transform_set(self, trajectories: list[Any]) -> np.ndarray:
        emb = self.transform_trajectories(trajectories)
        return np.mean(emb, axis=0).astype(np.float32)

    def transform(self, trajectory_sets: list[list[Any]]) -> np.ndarray:
        return np.stack([self.transform_set(group) for group in trajectory_sets], axis=0)

    def pairwise_distance(self, sets_a: list[list[Any]], sets_b: list[list[Any]] | None = None) -> np.ndarray:
        a = self.transform(sets_a)
        b = a if sets_b is None else self.transform(sets_b)
        return np.sqrt(pairwise_sqeuclidean(a, b)).astype(np.float32)

    def pairwise_similarity(self, sets_a: list[list[Any]], sets_b: list[list[Any]] | None = None) -> np.ndarray:
        return (-self.pairwise_distance(sets_a, sets_b)).astype(np.float32)

    def novelty_score(self, trajectory_or_set: Any) -> np.ndarray:
        """Distance to the fitted reference trajectory-set distribution."""
        ref = self.transform(self.trajectory_sets_).mean(axis=0, keepdims=True)
        if isinstance(trajectory_or_set, list) and trajectory_or_set and not hasattr(trajectory_or_set, "states"):
            emb = self.transform_set(trajectory_or_set)[None, :]
        else:
            emb = self.transform_trajectories([trajectory_or_set])
        return np.sqrt(pairwise_sqeuclidean(emb, ref)).reshape(-1).astype(np.float32)


def build_set_metric(method: str = "gdk2", **kwargs: Any) -> TrajectorySetDistance:
    key = method.lower()
    if key == "idk2":
        traj_metric = IDKTrajectoryDistance(**kwargs)
    elif key == "adaptive_gdk2":
        traj_metric = AdaptiveGDKTrajectoryDistance(**kwargs)
    else:
        traj_metric = GDKTrajectoryDistance(**kwargs)
    return TrajectorySetDistance(traj_metric)
