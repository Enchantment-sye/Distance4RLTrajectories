"""Trajectory-to-set distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from reachability_metrics.utils import softmin


class TrajectoryToSetDistance:
    """Aggregate trajectory-to-trajectory distances over a set."""

    def __init__(self, trajectory_metric: Any, aggregation: str = "min", softmin_tau: float = 1.0) -> None:
        self.trajectory_metric = trajectory_metric
        self.aggregation = aggregation
        self.softmin_tau = softmin_tau

    def fit(self, trajectory_sets: list[list[Any]]) -> "TrajectoryToSetDistance":
        flat = [traj for group in trajectory_sets for traj in group]
        self.trajectory_metric.fit(flat)
        self.trajectory_sets_ = trajectory_sets
        return self

    def pairwise_distance(self, trajectories: Any, trajectory_sets: list[list[Any]] | None = None) -> np.ndarray:
        sets = self.trajectory_sets_ if trajectory_sets is None else trajectory_sets
        rows = []
        for group in sets:
            d = self.trajectory_metric.pairwise_distance(trajectories, group)
            if self.aggregation == "mean":
                rows.append(np.mean(d, axis=1))
            elif self.aggregation == "softmin":
                rows.append(softmin(d, tau=self.softmin_tau, axis=1))
            else:
                rows.append(np.min(d, axis=1))
        return np.stack(rows, axis=1).astype(np.float32)

