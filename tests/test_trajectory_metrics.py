from __future__ import annotations

import numpy as np
import pytest

from reachability_metrics.trajectory_metrics import (
    DTWDistance,
    FrechetDistance,
    GDKTrajectoryDistance,
    HausdorffDistance,
    IDKTrajectoryDistance,
    TrajectoryEuclideanDistance,
    TrajectoryWassersteinDistance,
)


def _trajs() -> list[np.ndarray]:
    t = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    return [
        np.stack([t, t * 0.0], axis=1),
        np.stack([t, t + 0.1], axis=1),
        np.stack([t[:5], -t[:5]], axis=1),
    ]


def test_classical_trajectory_distances() -> None:
    trajectories = _trajs()
    metrics = [
        TrajectoryEuclideanDistance(target_length=8),
        DTWDistance(),
        HausdorffDistance(),
        FrechetDistance(),
        TrajectoryWassersteinDistance(),
    ]
    for metric in metrics:
        metric.fit(trajectories)
        d = metric.pairwise_distance(trajectories)
        assert d.shape == (3, 3)
        assert np.all(np.isfinite(d))
        assert np.allclose(np.diag(d), 0.0, atol=1e-5)


def test_idk_and_gdk_handle_variable_lengths() -> None:
    pytest.importorskip("torch")
    trajectories = _trajs()
    idk = IDKTrajectoryDistance(ensemble_size=6, subsample_size=3, temperature=0.05, device="cpu").fit(trajectories)
    gdk = GDKTrajectoryDistance().fit(trajectories)
    assert idk.pairwise_distance(trajectories).shape == (3, 3)
    assert gdk.pairwise_distance(trajectories).shape == (3, 3)

