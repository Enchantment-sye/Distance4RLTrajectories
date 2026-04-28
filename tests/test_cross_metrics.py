from __future__ import annotations

import numpy as np

from reachability_metrics.cross_metrics import StateToTrajectoryDistance
from reachability_metrics.state_metrics import EuclideanDistance


def test_state_to_trajectory_min_aggregation() -> None:
    metric = StateToTrajectoryDistance(EuclideanDistance().fit(np.zeros((1, 2))), aggregation="min")
    trajectories = [
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 0.0], [6.0, 0.0]], dtype=np.float32),
    ]
    d = metric.pairwise_distance(np.array([[1.0, 0.0]], dtype=np.float32), trajectories)
    assert d.shape == (1, 2)
    assert np.allclose(d[0], [1.0, 4.0])

