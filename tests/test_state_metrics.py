from __future__ import annotations

import numpy as np
import pytest

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.state_metrics import (
    AdaptiveGaussianDistance,
    EuclideanDistance,
    GaussianKernelDistance,
    HSuccessorDistance,
    IsolationKernelDistance,
    MahalanobisDistance,
)


def test_euclidean_distance_shape() -> None:
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    d = EuclideanDistance().fit(x).pairwise_distance(x[:2], x[2:5])
    assert d.shape == (2, 3)
    assert np.all(d >= 0)


def test_gaussian_kernel_symmetry() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    metric = GaussianKernelDistance().fit(x)
    sim = metric.pairwise_similarity(x)
    assert sim.shape == (12, 12)
    assert np.allclose(sim, sim.T, atol=1e-6)
    assert np.allclose(np.diag(sim), 1.0, atol=1e-6)


def test_adaptive_gaussian_sigma_nonzero() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 2))
    metric = AdaptiveGaussianDistance(k=3).fit(x)
    sigmas = metric.estimate_sigmas(x[:5])
    assert sigmas.shape == (5,)
    assert np.all(sigmas > 0)


def test_mahalanobis_whitening_smoke() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 4))
    metric = MahalanobisDistance().fit(x)
    d = metric.pairwise_distance(x[:3], x[3:8])
    assert d.shape == (3, 5)
    assert np.all(np.isfinite(d))


def test_isolation_kernel_fit_transform_pairwise() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(32, 2)).astype(np.float32)
    metric = IsolationKernelDistance(
        ensemble_size=8,
        subsample_size=4,
        temperature=0.05,
        device="cpu",
        batch_size=16,
        random_state=0,
    ).fit(x)
    phi = metric.transform(x[:5])
    sim = metric.pairwise_similarity(x[:5], x[5:9])
    dist = metric.pairwise_distance(x[:5], x[5:9])
    assert phi.shape == (5, 32)
    assert sim.shape == dist.shape == (5, 4)
    assert np.all(dist >= -1e-6)


def test_h_successor_uses_valid_same_trajectory_windows() -> None:
    dataset = TrajectoryDataset.from_arrays([
        np.arange(10, dtype=np.float32).reshape(5, 2),
        np.arange(10, 22, dtype=np.float32).reshape(6, 2),
    ])
    metric = HSuccessorDistance(horizon=2).fit(dataset)
    assert metric.windows_.shape == (7, 2, 2)
    assert np.all(metric.window_episode_ids_[:3] == 0)
    assert np.all(metric.window_episode_ids_[3:] == 1)
    d = metric.pairwise_distance()
    assert d.shape == (7, 7)

