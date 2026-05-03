from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from reachability_metrics.data import TrajectoryDataset
from reachability_metrics.state_metrics import (
    AdaptiveGaussianDistance,
    EuclideanDistance,
    GaussianKernelDistance,
    HSuccessorDistance,
    IsolationKernelDistance,
    MahalanobisDistance,
    OneStepDynamicsDistance,
    TemporalDistance,
)


pytestmark = pytest.mark.distributed

torch.set_num_threads(1)


def _distributed_api():
    return importlib.import_module("reachability_metrics.distributed")


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


def _require_cpu_gloo(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    has_gloo = not hasattr(dist, "is_gloo_available") or dist.is_gloo_available()
    if not has_gloo:
        pytest.skip("torch.distributed gloo backend is not available")
    if dist.is_initialized():
        return

    previous_ifname = os.environ.get("GLOO_SOCKET_IFNAME")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    try:
        try:
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{tmp_path / 'gloo_smoke_init'}",
                rank=0,
                world_size=1,
            )
        except RuntimeError as exc:
            pytest.skip(f"torch.distributed gloo cannot initialize here: {exc}")
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    finally:
        if previous_ifname is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = previous_ifname


def _state_train_data() -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.25],
            [0.25, 1.75],
            [2.0, -0.5],
            [2.5, 1.25],
            [-0.75, 0.5],
            [1.5, 2.5],
            [3.25, 0.75],
        ],
        dtype=torch.float32,
    )


def _state_query_data() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(
        [[0.0, 0.0], [1.0, 0.5], [2.75, 1.0], [-0.5, 1.25]],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [[0.25, 0.75], [1.75, -0.25], [2.5, 2.0], [-1.0, 0.25], [3.5, 0.5]],
        dtype=torch.float32,
    )
    return x, y


def _dynamics_case() -> tuple[OneStepDynamicsDistance, torch.Tensor, torch.Tensor]:
    trajectories = TrajectoryDataset.from_arrays(
        [
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 0.5]], dtype=np.float32),
            np.array([[0.0, 1.0], [0.0, 2.0], [0.5, 3.0], [1.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 1.0], [1.75, 1.75], [2.5, 2.5], [3.25, 3.25]], dtype=np.float32),
        ]
    )
    metric = OneStepDynamicsDistance(local_knn_m=2, device="cpu").fit(trajectories)
    x = torch.tensor([[0.2, 0.1], [1.7, 0.4], [0.25, 2.6]], dtype=torch.float32)
    y = torch.tensor([[1.0, 0.0], [0.5, 3.0], [2.5, 2.5], [3.0, 0.5]], dtype=torch.float32)
    return metric, x, y


def _state_distance_case(name: str):
    train = _state_train_data()
    x, y = _state_query_data()
    if name == "euclidean":
        return EuclideanDistance(device="cpu").fit(train), x, y
    if name == "gaussian_fixed_sigma":
        metric = GaussianKernelDistance(
            sigma="fixed",
            sigma_value=1.25,
            distance_mode="one_minus_kernel",
            device="cpu",
        ).fit(train)
        return metric, x, y
    if name == "adaptive_gaussian":
        return AdaptiveGaussianDistance(k=2, device="cpu").fit(train), x, y
    if name == "mahalanobis":
        return MahalanobisDistance(covariance_estimator="empirical", device="cpu").fit(train), x, y
    if name == "isolation_kernel":
        metric = IsolationKernelDistance(
            ensemble_size=4,
            subsample_size=3,
            temperature=0.2,
            random_state=11,
            batch_size=4,
            block_size=3,
            device="cpu",
        ).fit(train)
        return metric, x, y
    if name == "one_step_dynamics":
        return _dynamics_case()
    raise AssertionError(f"unknown state distance case: {name}")


def _successor_dataset() -> TrajectoryDataset:
    return TrajectoryDataset.from_arrays(
        [
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.5], [3.0, 1.0], [4.0, 1.0]],
                dtype=np.float32,
            ),
            np.array(
                [[0.0, 2.0], [0.5, 3.0], [1.0, 4.0], [1.5, 5.0], [2.0, 6.0]],
                dtype=np.float32,
            ),
        ]
    )


def _h_successor_case(aggregation: str):
    metric = HSuccessorDistance(horizon=2, aggregation=aggregation, device="cpu").fit(
        _successor_dataset()
    )
    return metric, metric.windows_[:3], metric.windows_[2:]


@pytest.mark.parametrize(
    "case_name",
    [
        "euclidean",
        "gaussian_fixed_sigma",
        "adaptive_gaussian",
        "mahalanobis",
        "isolation_kernel",
        "one_step_dynamics",
    ],
)
def test_world_size_one_state_distances_match_metric(case_name: str) -> None:
    api = _distributed_api()
    metric, x, y = _state_distance_case(case_name)

    actual = api.distributed_pairwise_distance(metric, x, y)
    expected = metric.pairwise_distance(x, y)

    _assert_exact(actual, expected)


@pytest.mark.parametrize(
    "case_name",
    ["euclidean", "gaussian_fixed_sigma", "adaptive_gaussian", "isolation_kernel"],
)
def test_world_size_one_state_similarities_match_metric(case_name: str) -> None:
    api = _distributed_api()
    metric, x, y = _state_distance_case(case_name)

    actual = api.distributed_pairwise_similarity(metric, x, y)
    expected = metric.pairwise_similarity(x, y)

    _assert_exact(actual, expected)


@pytest.mark.parametrize(
    "aggregation",
    [
        pytest.param("raw_l2", id="raw_l2"),
        pytest.param("endpoint_l2", id="endpoint"),
        pytest.param("mean_l2", id="mean"),
    ],
)
def test_world_size_one_h_successor_distances_match_metric(aggregation: str) -> None:
    api = _distributed_api()
    metric, x, y = _h_successor_case(aggregation)

    actual = api.distributed_pairwise_distance(metric, x, y)
    expected = metric.pairwise_distance(x, y)

    _assert_exact(actual, expected)


def test_world_size_one_temporal_default_state_distance_matches_metric() -> None:
    api = _distributed_api()
    dataset = TrajectoryDataset.from_arrays(
        [
            np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
        ]
    )
    metric = TemporalDistance(max_window=2, device="cpu").fit(dataset)
    states = dataset.states()

    actual = api.distributed_pairwise_distance(metric, states)
    expected = metric.pairwise_distance(states)

    _assert_exact(actual, expected)
    assert actual[0, 2].item() == 2.0
    assert torch.isinf(actual[0, 3])
    assert torch.isinf(actual[0, 4])


def test_world_size_one_temporal_fitted_default_uses_index_distance() -> None:
    api = _distributed_api()
    dataset = TrajectoryDataset.from_arrays(
        [
            np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            np.array([[10.0], [11.0], [12.0]], dtype=np.float32),
        ]
    )
    metric = TemporalDistance(max_window=2, device="cpu").fit(dataset)
    indices = torch.arange(dataset.states().shape[0], dtype=torch.long)

    actual = api.distributed_pairwise_distance(metric)
    expected = metric.pairwise_distance_indices(indices, indices)

    _assert_exact(actual, expected)
    assert actual[0, 1].item() == 1.0
    assert torch.isinf(actual[0, 3])
    assert actual[4, 5].item() == 1.0


def test_world_size_one_distributed_topk_matches_pairwise_distance_topk() -> None:
    api = _distributed_api()
    metric, x, y = _state_distance_case("euclidean")

    actual_values, actual_indices = api.distributed_topk(
        metric,
        x,
        y,
        k=2,
        op="distance",
        sorted=True,
    )
    expected_values, expected_indices = torch.topk(
        metric.pairwise_distance(x, y),
        k=2,
        dim=1,
        largest=False,
        sorted=True,
    )

    _assert_exact(actual_values, expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices.cpu())


def test_world_size_one_h_successor_topk_matches_pairwise_distance_topk() -> None:
    api = _distributed_api()
    metric, x, y = _h_successor_case("raw_l2")

    actual_values, actual_indices = api.distributed_topk(
        metric,
        x,
        y,
        k=2,
        op="distance",
        sorted=True,
    )
    expected_values, expected_indices = torch.topk(
        metric.pairwise_distance(x, y),
        k=2,
        dim=1,
        largest=False,
        sorted=True,
    )

    _assert_exact(actual_values, expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices.cpu())


def _spawn_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        api = _distributed_api()

        state_metric, state_x, state_y = _state_distance_case("euclidean")
        state_actual = api.distributed_pairwise_distance(state_metric, state_x, state_y)
        state_expected = state_metric.pairwise_distance(state_x, state_y)
        _assert_exact(state_actual, state_expected)

        successor_metric, successor_x, successor_y = _h_successor_case("raw_l2")
        successor_actual = api.distributed_pairwise_distance(
            successor_metric,
            successor_x,
            successor_y,
        )
        successor_expected = successor_metric.pairwise_distance(successor_x, successor_y)
        _assert_exact(successor_actual, successor_expected)

        if rank == 0:
            queue.put("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_cpu_gloo_spawn_matches_local_state_and_h_successor_distances(tmp_path) -> None:
    _require_cpu_gloo(tmp_path)

    world_size = 2
    init_file = tmp_path / "distributed_state_metrics_init"
    ctx = mp.get_context("fork")
    queue = ctx.SimpleQueue()
    previous_ifname = os.environ.get("GLOO_SOCKET_IFNAME")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")

    try:
        processes = [
            ctx.Process(target=_spawn_worker, args=(rank, world_size, str(init_file), queue))
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(30)
    finally:
        if previous_ifname is None:
            os.environ.pop("GLOO_SOCKET_IFNAME", None)
        else:
            os.environ["GLOO_SOCKET_IFNAME"] = previous_ifname

    assert [process.exitcode for process in processes] == [0, 0]
    assert queue.get() == "ok"
