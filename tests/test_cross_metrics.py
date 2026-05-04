from __future__ import annotations

import numpy as np
import torch

from reachability_metrics.cross_metrics import StateToTrajectoryDistance, TrajectoryToSetDistance
from reachability_metrics.set_metrics import TrajectorySetDistance
from reachability_metrics.state_metrics import EuclideanDistance
from reachability_metrics.trajectory_metrics import GDKTrajectoryDistance


class _TensorOnlyStateMetric:
    dtype = "float32"
    device = "cpu"

    def fit(self, X):
        return self

    def pairwise_distance_tensor(self, X, Y):
        return torch.cdist(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32))

    def pairwise_distance(self, X, Y):
        raise AssertionError("tensor path should be used")


class _TensorOnlyTrajectoryEmbeddingMetric:
    def fit(self, trajectories):
        return self

    def transform_tensor(self, trajectories):
        return torch.stack([torch.as_tensor(traj, dtype=torch.float32)[0] for traj in trajectories], dim=0)

    def transform(self, trajectories):
        raise AssertionError("tensor path should be used")


def test_state_to_trajectory_min_aggregation() -> None:
    metric = StateToTrajectoryDistance(EuclideanDistance().fit(np.zeros((1, 2))), aggregation="min")
    trajectories = [
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        np.array([[5.0, 0.0], [6.0, 0.0]], dtype=np.float32),
    ]
    d = metric.pairwise_distance(np.array([[1.0, 0.0]], dtype=np.float32), trajectories)
    assert d.shape == (1, 2)
    torch.testing.assert_close(d[0], torch.tensor([1.0, 4.0]))


def test_group_aggregation_prefers_pairwise_distance_tensor() -> None:
    metric = StateToTrajectoryDistance(_TensorOnlyStateMetric(), aggregation="mean").fit(
        [
            np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
            np.array([[10.0, 0.0], [12.0, 0.0]], dtype=np.float32),
        ]
    )

    d = metric.pairwise_distance_tensor(np.array([[1.0, 0.0]], dtype=np.float32))

    torch.testing.assert_close(d, torch.tensor([[1.0, 10.0]]))


def test_trajectory_set_embedding_order_prefers_transform_tensor() -> None:
    trajectory_sets = [
        [
            np.array([[1.0, 10.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[3.0, 30.0], [0.0, 0.0]], dtype=np.float32),
        ],
        [
            np.array([[100.0, 5.0], [0.0, 0.0]], dtype=np.float32),
            np.array([[200.0, 7.0], [0.0, 0.0]], dtype=np.float32),
        ],
    ]
    metric = TrajectorySetDistance(_TensorOnlyTrajectoryEmbeddingMetric(), normalize=False).fit(trajectory_sets)

    embeddings = metric.transform_tensor(trajectory_sets)

    torch.testing.assert_close(embeddings, torch.tensor([[2.0, 20.0], [150.0, 6.0]]))


def test_trajectory_to_set_two_level_kme() -> None:
    rng = np.random.default_rng(0)
    reference = [rng.normal(scale=0.05, size=(6, 2)).astype(np.float32) for _ in range(4)]
    shifted = [(rng.normal(scale=0.05, size=(6, 2)) + 3.0).astype(np.float32) for _ in range(4)]
    metric = TrajectoryToSetDistance(GDKTrajectoryDistance(num_landmarks=8), method="two_level_kme").fit([reference, shifted])
    d = metric.pairwise_distance([reference[0], shifted[0]])
    assert d.shape == (2, 2)
    assert d[0, 0] < d[0, 1]
    assert d[1, 1] < d[1, 0]
