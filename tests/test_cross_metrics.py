from __future__ import annotations

import numpy as np
import torch

from reachability_metrics.cross_metrics import StateToTrajectoryDistance, TrajectoryToSetDistance
from reachability_metrics.state_metrics import EuclideanDistance
from reachability_metrics.trajectory_metrics import GDKTrajectoryDistance


def test_state_to_trajectory_min_aggregation() -> None:
    metric = StateToTrajectoryDistance(EuclideanDistance().fit(np.zeros((1, 2))), aggregation="min")
    trajectories = [
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 0.0], [6.0, 0.0]], dtype=np.float32),
    ]
    d = metric.pairwise_distance(np.array([[1.0, 0.0]], dtype=np.float32), trajectories)
    assert d.shape == (1, 2)
    torch.testing.assert_close(d[0], torch.tensor([1.0, 4.0]))


def test_trajectory_to_set_two_level_kme() -> None:
    rng = np.random.default_rng(0)
    reference = [rng.normal(scale=0.05, size=(6, 2)).astype(np.float32) for _ in range(4)]
    shifted = [(rng.normal(scale=0.05, size=(6, 2)) + 3.0).astype(np.float32) for _ in range(4)]
    metric = TrajectoryToSetDistance(GDKTrajectoryDistance(num_landmarks=8), method="two_level_kme").fit([reference, shifted])
    d = metric.pairwise_distance([reference[0], shifted[0]])
    assert d.shape == (2, 2)
    assert d[0, 0] < d[0, 1]
    assert d[1, 1] < d[1, 0]
