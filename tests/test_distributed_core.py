from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from reachability_metrics.distributed import (
    distributed_pairwise_distance,
    distributed_pairwise_similarity,
    distributed_topk,
)
from reachability_metrics.state_metrics import EuclideanDistance


pytestmark = pytest.mark.distributed

torch.set_num_threads(1)


def _metric() -> EuclideanDistance:
    fit = np.array([[0.0], [3.0], [6.0], [9.0]], dtype=np.float32)
    return EuclideanDistance(device="cpu", dtype="float32").fit(fit)


def _queries() -> np.ndarray:
    return np.array([[0.0], [2.0], [5.0]], dtype=np.float32)


def _refs() -> np.ndarray:
    return np.array([[0.0], [3.0], [6.0], [9.0]], dtype=np.float32)


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    raise TypeError(f"Expected tensor-like output, got {type(value)!r}")


def _topk_parts(result: Any) -> tuple[Any, Any]:
    if isinstance(result, tuple):
        values, indices = result
    else:
        values, indices = result.values, result.indices
    return values, indices


def _require_cpu_gloo(tmp_path) -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    has_gloo = not hasattr(dist, "is_gloo_available")
    has_gloo = has_gloo or dist.is_gloo_available()
    if not has_gloo:
        pytest.skip("torch.distributed gloo backend is not available")
    if dist.is_initialized():
        return
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{tmp_path / 'core_gloo_smoke'}",
            rank=0,
            world_size=1,
        )
    except RuntimeError as exc:
        pytest.skip(f"torch.distributed gloo cannot initialize here: {exc}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_gather_modes_world_size_one_match_local_pairwise_distance() -> None:
    x = _queries()
    y = _refs()
    expected = _metric().pairwise_distance(x, y)

    for gather in ("all", "rank0"):
        actual = distributed_pairwise_distance(
            _metric(),
            x,
            y,
            gather=gather,
            row_block_size=2,
        )
        torch.testing.assert_close(_as_cpu_tensor(actual), expected)

    local, start, end, shape = distributed_pairwise_distance(
        _metric(),
        x,
        y,
        gather="none",
        row_block_size=2,
    )
    torch.testing.assert_close(_as_cpu_tensor(local), expected)
    assert (start, end, shape) == (0, x.shape[0], (x.shape[0], y.shape[0]))


def test_row_block_size_keeps_pairwise_distance_equivalent() -> None:
    x = np.array([[0.0], [1.0], [2.0], [4.0], [7.0]], dtype=np.float32)
    y = _refs()

    unblocked = distributed_pairwise_distance(
        _metric(),
        x,
        y,
        gather="all",
        row_block_size=None,
    )
    blocked = distributed_pairwise_distance(
        _metric(),
        x,
        y,
        gather="all",
        row_block_size=1,
    )
    similarity_unblocked = distributed_pairwise_similarity(
        _metric(),
        x,
        y,
        gather="all",
        row_block_size=None,
    )
    similarity_blocked = distributed_pairwise_similarity(
        _metric(),
        x,
        y,
        gather="all",
        row_block_size=1,
    )

    torch.testing.assert_close(_as_cpu_tensor(blocked), _as_cpu_tensor(unblocked))
    torch.testing.assert_close(
        _as_cpu_tensor(similarity_blocked),
        _as_cpu_tensor(similarity_unblocked),
    )


def test_numpy_output_returns_numpy_array_with_expected_values() -> None:
    x = _queries()
    y = _refs()
    expected = _metric().pairwise_distance(x, y).detach().cpu().numpy()

    actual = distributed_pairwise_distance(
        _metric(),
        x,
        y,
        gather="all",
        output_format="numpy",
    )

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, expected)


def test_pairwise_topk_distance_returns_values_and_indices() -> None:
    values, indices = _topk_parts(
        distributed_topk(
            _metric(),
            _queries(),
            _refs(),
            k=2,
            op="distance",
            gather="all",
        )
    )

    torch.testing.assert_close(
        _as_cpu_tensor(values),
        torch.tensor([[0.0, 3.0], [1.0, 2.0], [1.0, 2.0]], dtype=torch.float32),
    )
    assert torch.equal(
        _as_cpu_tensor(indices),
        torch.tensor([[0, 1], [1, 0], [2, 1]], dtype=torch.long),
    )


def test_pairwise_topk_similarity_returns_values_and_indices() -> None:
    values, indices = _topk_parts(
        distributed_topk(
            _metric(),
            _queries(),
            _refs(),
            k=2,
            op="similarity",
            gather="all",
        )
    )

    torch.testing.assert_close(
        _as_cpu_tensor(values),
        torch.tensor([[0.0, -3.0], [-1.0, -2.0], [-1.0, -2.0]], dtype=torch.float32),
    )
    assert torch.equal(
        _as_cpu_tensor(indices),
        torch.tensor([[0, 1], [1, 0], [2, 1]], dtype=torch.long),
    )


def _spawn_world_size_larger_than_rows_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        x = np.array([[0.0], [5.0]], dtype=np.float32)
        y = _refs()
        expected = _metric().pairwise_distance(x, y)
        actual = distributed_pairwise_distance(
            _metric(),
            x,
            y,
            gather="all",
            row_block_size=1,
        )
        assert torch.equal(_as_cpu_tensor(actual), expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
def test_spawn_world_size_larger_than_rows_matches_local_distance_exactly(tmp_path) -> None:
    _require_cpu_gloo(tmp_path)
    x = np.array([[0.0], [5.0]], dtype=np.float32)
    y = _refs()
    expected = _metric().pairwise_distance(x, y)

    world_size = 3
    ctx = mp.get_context("fork")
    processes = [
        ctx.Process(
            target=_spawn_world_size_larger_than_rows_worker,
            args=(rank, world_size, str(tmp_path / "core_distributed_init")),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    assert [process.exitcode for process in processes] == [0, 0, 0]

    actual_single = distributed_pairwise_distance(_metric(), x, y, gather="all")
    assert torch.equal(_as_cpu_tensor(actual_single), expected)
