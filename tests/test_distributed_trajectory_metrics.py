from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from reachability_metrics.distributed import distributed_pairwise_distance, distributed_pairwise_similarity
from reachability_metrics.trajectory_metrics import (
    AdaptiveGDKTrajectoryDistance,
    DTWDistance,
    FrechetDistance,
    GDKTrajectoryDistance,
    HausdorffDistance,
    IDKTrajectoryDistance,
    T2VecDistance,
    TrajectoryEuclideanDistance,
    TrajectoryWassersteinDistance,
)


pytestmark = pytest.mark.distributed
torch.set_num_threads(1)


def _trajectory(length: int, *, shift: float, slope: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack([t + shift, slope * t + shift], axis=1).astype(np.float32)


def _trajectory_corpus() -> list[np.ndarray]:
    return [
        _trajectory(4, shift=0.0, slope=0.0),
        _trajectory(5, shift=0.1, slope=0.6),
        _trajectory(3, shift=-0.2, slope=-0.4),
        _trajectory(6, shift=0.5, slope=1.0),
        _trajectory(4, shift=-0.4, slope=0.2),
    ]


def _queries() -> list[np.ndarray]:
    return _trajectory_corpus()[:3]


def _references() -> list[np.ndarray]:
    return _trajectory_corpus()[2:]


def _fit_metric(metric: Any) -> Any:
    return metric.fit(_trajectory_corpus())


def _assert_exact(actual: Any, expected: Any) -> None:
    torch.testing.assert_close(
        actual.detach().cpu() if isinstance(actual, torch.Tensor) else torch.as_tensor(actual),
        expected.detach().cpu() if isinstance(expected, torch.Tensor) else torch.as_tensor(expected),
        atol=0,
        rtol=0,
    )


def _euclidean() -> TrajectoryEuclideanDistance:
    return TrajectoryEuclideanDistance(target_length=6, device="cpu")


def _dtw() -> DTWDistance:
    return DTWDistance(device="cpu")


def _hausdorff() -> HausdorffDistance:
    return HausdorffDistance(device="cpu")


def _frechet() -> FrechetDistance:
    return FrechetDistance(device="cpu")


def _wasserstein() -> TrajectoryWassersteinDistance:
    return TrajectoryWassersteinDistance(device="cpu")


def _gdk() -> GDKTrajectoryDistance:
    return GDKTrajectoryDistance(sigma="fixed", sigma_value=1.0, device="cpu")


def _idk() -> IDKTrajectoryDistance:
    return IDKTrajectoryDistance(
        ensemble_size=6,
        subsample_size=4,
        temperature=0.05,
        device="cpu",
        random_state=0,
    )


def _adaptive_gdk() -> AdaptiveGDKTrajectoryDistance:
    return AdaptiveGDKTrajectoryDistance(k=2, device="cpu")


METRIC_CASES = [
    pytest.param(_euclidean, id="trajectory_euclidean"),
    pytest.param(_hausdorff, id="hausdorff"),
    pytest.param(_gdk, id="gdk"),
    pytest.param(_idk, id="idk"),
    pytest.param(_adaptive_gdk, id="adaptive_gdk"),
    pytest.param(_dtw, marks=pytest.mark.slow, id="dtw"),
    pytest.param(_frechet, marks=pytest.mark.slow, id="frechet"),
    pytest.param(_wasserstein, marks=pytest.mark.slow, id="wasserstein"),
]


SIMILARITY_CASES = [
    pytest.param(_euclidean, id="trajectory_euclidean"),
    pytest.param(_hausdorff, id="hausdorff"),
    pytest.param(_gdk, id="gdk"),
    pytest.param(_idk, id="idk"),
    pytest.param(_adaptive_gdk, id="adaptive_gdk"),
]


@pytest.mark.parametrize("metric_factory", METRIC_CASES)
def test_world_size_one_trajectory_distance_matches_local(metric_factory: Callable[[], Any]) -> None:
    metric = _fit_metric(metric_factory())
    expected = metric.pairwise_distance(_queries(), _references())
    actual = distributed_pairwise_distance(metric, _queries(), _references(), row_block_size=1)
    _assert_exact(actual, expected)


@pytest.mark.parametrize("metric_factory", SIMILARITY_CASES)
def test_world_size_one_trajectory_similarity_matches_local(metric_factory: Callable[[], Any]) -> None:
    metric = _fit_metric(metric_factory())
    expected = metric.pairwise_similarity(_queries(), _references())
    actual = distributed_pairwise_similarity(metric, _queries(), _references(), row_block_size=1)
    _assert_exact(actual, expected)


def test_distributed_trajectory_euclidean_rejects_variable_lengths_without_target() -> None:
    metric = TrajectoryEuclideanDistance(target_length=None, device="cpu").fit(_trajectory_corpus())
    with pytest.raises(ValueError, match=r"TrajectoryEuclideanDistance.*target_length"):
        distributed_pairwise_distance(metric, _queries(), _references())


def _require_cpu_gloo(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    if hasattr(dist, "is_gloo_available") and not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is not available")
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmp_path / 'trajectory_gloo_smoke'}",
            rank=0,
            world_size=1,
        )
    except RuntimeError as exc:
        pytest.skip(f"torch.distributed gloo cannot initialize here: {exc}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_trajectory_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        metric = _fit_metric(_euclidean())
        expected = metric.pairwise_distance(_queries(), _references())
        actual = distributed_pairwise_distance(metric, _queries(), _references(), gather="all")
        _assert_exact(actual, expected)
    finally:
        dist.destroy_process_group()


def test_cpu_gloo_spawn_trajectory_distance_matches_local(tmp_path) -> None:
    _require_cpu_gloo(tmp_path)
    world_size = 2
    ctx = mp.get_context("fork")
    processes = [
        ctx.Process(
            target=_spawn_trajectory_worker,
            args=(rank, world_size, str(tmp_path / "trajectory_distance_init")),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    assert [process.exitcode for process in processes] == [0, 0]


@pytest.mark.slow
def test_world_size_one_tiny_t2vec_distance_matches_local(tmp_path) -> None:
    model_path = tmp_path / "tiny_t2vec.pt"
    metric = T2VecDistance(
        model_path=str(model_path),
        train_if_missing=True,
        normalize=True,
        embedding_dim=4,
        hidden_size=6,
        num_layers=1,
        batch_size=3,
        epochs=1,
        validation_split=0.0,
        noise_std=0.0,
        point_dropout=0.0,
        downsample_keep_prob=1.0,
        device="cpu",
        random_state=0,
    ).fit(_trajectory_corpus())
    expected = metric.pairwise_distance(_queries(), _references())
    actual = distributed_pairwise_distance(metric, _queries(), _references(), row_block_size=1)
    _assert_exact(actual, expected)
