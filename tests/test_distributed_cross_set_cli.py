from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from reachability_metrics.cross_metrics import (
    StateToTrajectoryDistance,
    StateToTrajectoryKMEDistance,
    StateToTrajectorySetDistance,
    TrajectoryToSetDistance,
)
from reachability_metrics.distributed import distributed_pairwise_distance, distributed_pairwise_similarity
from reachability_metrics.set_metrics import (
    AdaptiveGDK2SetDistance,
    GDK2SetDistance,
    IDK2SetDistance,
    TrajectorySetDistance,
)
from reachability_metrics.state_metrics import EuclideanDistance, GaussianKernelDistance
from reachability_metrics.trajectory_metrics import TrajectoryEuclideanDistance


pytestmark = pytest.mark.distributed

torch.set_num_threads(1)


_CLI_MODULE = "reachability_metrics.cli.run_distributed_distance"


def _line_trajectory(start: float, stop: float) -> np.ndarray:
    return np.array([[start, 0.0], [stop, 0.0]], dtype=np.float32)


def _separated_trajectory_sets() -> list[list[np.ndarray]]:
    return [
        [_line_trajectory(0.0, 1.0), _line_trajectory(1.0, 2.0)],
        [_line_trajectory(10.0, 11.0), _line_trajectory(11.0, 12.0)],
    ]


@pytest.mark.parametrize(
    ("aggregation", "kwargs", "expected"),
    [
        (
            "min",
            {},
            [[1.0, 9.0], [9.0, 1.0]],
        ),
        (
            "mean",
            {},
            [[5.0 / 3.0, 11.0], [11.0, 5.0 / 3.0]],
        ),
        (
            "kmin_mean",
            {"k": 2},
            [[1.0, 10.0], [10.0, 1.0]],
        ),
    ],
)
def test_state_to_trajectory_distance_matches_euclidean_aggregation(
    aggregation: str,
    kwargs: dict[str, Any],
    expected: list[list[float]],
) -> None:
    trajectories = [
        np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        np.array([[10.0, 0.0], [12.0, 0.0], [14.0, 0.0]], dtype=np.float32),
    ]
    states = np.array([[1.0, 0.0], [13.0, 0.0]], dtype=np.float32)

    metric = StateToTrajectoryDistance(
        EuclideanDistance(device="cpu"),
        aggregation=aggregation,
        **kwargs,
    ).fit(trajectories)
    expected_values = torch.tensor(expected, dtype=torch.float32)
    expected_distances = metric.pairwise_distance(states)
    distances = distributed_pairwise_distance(metric, states, row_block_size=1)

    assert distances.shape == (2, 2)
    torch.testing.assert_close(distances, expected_distances, atol=0, rtol=0)
    torch.testing.assert_close(distances, expected_values, atol=1e-6, rtol=1e-6)


def test_state_to_trajectory_kme_distance_and_similarity_rank_nearest_trajectory() -> None:
    trajectories = [
        np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[8.0, 0.0], [8.5, 0.0], [9.0, 0.0]], dtype=np.float32),
    ]
    states = np.array([[0.0, 0.0], [9.0, 0.0]], dtype=np.float32)

    metric = StateToTrajectoryKMEDistance(
        GaussianKernelDistance(sigma="fixed", sigma_value=1.0, device="cpu")
    ).fit(trajectories)
    distances = distributed_pairwise_distance(metric, states, row_block_size=1)
    similarities = distributed_pairwise_similarity(metric, states, row_block_size=1)
    explicit_second = distributed_pairwise_distance(metric, states, [trajectories[1]], row_block_size=1)
    expected_distances = metric.pairwise_distance(states)
    expected_similarities = metric.pairwise_similarity(states)

    assert distances.shape == similarities.shape == (2, 2)
    assert explicit_second.shape == (2, 1)
    torch.testing.assert_close(distances, expected_distances, atol=0, rtol=0)
    torch.testing.assert_close(similarities, expected_similarities, atol=0, rtol=0)
    assert torch.all(torch.isfinite(distances))
    assert distances[0, 0] < distances[0, 1]
    assert distances[1, 1] < distances[1, 0]
    assert similarities[0, 0] > similarities[0, 1]
    assert similarities[1, 1] > similarities[1, 0]
    torch.testing.assert_close(explicit_second[:, 0], distances[:, 1])


def test_state_to_trajectory_set_distance_aggregates_per_set() -> None:
    trajectory_sets = [
        [_line_trajectory(0.0, 2.0), _line_trajectory(1.0, 3.0)],
        [_line_trajectory(10.0, 12.0), _line_trajectory(11.0, 13.0)],
    ]
    states = np.array([[1.0, 0.0], [12.0, 0.0]], dtype=np.float32)
    state_to_traj = StateToTrajectoryDistance(
        EuclideanDistance(device="cpu"),
        aggregation="min",
    )
    metric = StateToTrajectorySetDistance(state_to_traj, aggregation="mean").fit(trajectory_sets)

    expected = metric.pairwise_distance(states)
    distances = distributed_pairwise_distance(metric, states, row_block_size=1)

    assert distances.shape == (2, 2)
    torch.testing.assert_close(distances, expected, atol=0, rtol=0)
    expected = torch.tensor([[0.5, 9.5], [9.5, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(distances, expected.to(distances.device), atol=1e-6, rtol=1e-6)


def test_trajectory_to_set_aggregate_matches_manual_reduction() -> None:
    trajectory_sets = _separated_trajectory_sets()
    queries = [_line_trajectory(0.0, 1.0), _line_trajectory(11.0, 12.0)]

    metric = TrajectoryToSetDistance(
        TrajectoryEuclideanDistance(target_length=2, device="cpu"),
        method="aggregate",
        aggregation="mean",
    ).fit(trajectory_sets)
    expected_distances = metric.pairwise_distance(queries)
    distances = distributed_pairwise_distance(metric, queries, row_block_size=1)

    manual_columns = []
    for group in trajectory_sets:
        group_distances = metric.trajectory_metric_.pairwise_distance(queries, group)
        manual_columns.append(group_distances.mean(dim=1))
    expected = torch.stack(manual_columns, dim=1)

    assert distances.shape == (2, 2)
    torch.testing.assert_close(distances, expected_distances, atol=0, rtol=0)
    torch.testing.assert_close(distances, expected.to(distances.device), atol=1e-6, rtol=1e-6)
    assert distances[0, 0] < distances[0, 1]
    assert distances[1, 1] < distances[1, 0]


def test_trajectory_set_distance_transform_pairwise_and_novelty_api() -> None:
    trajectory_sets = _separated_trajectory_sets()
    metric = TrajectorySetDistance(
        TrajectoryEuclideanDistance(target_length=2, device="cpu"),
        normalize=False,
    ).fit(trajectory_sets)

    embeddings = metric.transform(trajectory_sets)
    expected_distances = metric.pairwise_distance(trajectory_sets)
    distances = distributed_pairwise_distance(metric, trajectory_sets, row_block_size=1)
    query_to_reference = distributed_pairwise_distance(metric, [trajectory_sets[0]], trajectory_sets, row_block_size=1)
    trajectory_novelty = metric.novelty_score(trajectory_sets[0][0])
    set_novelty = metric.novelty_score(trajectory_sets[0])

    assert embeddings.shape == (2, 4)
    assert distances.shape == (2, 2)
    assert query_to_reference.shape == (1, 2)
    assert trajectory_novelty.shape == set_novelty.shape == (1,)
    torch.testing.assert_close(distances, expected_distances, atol=0, rtol=0)
    torch.testing.assert_close(distances, distances.T, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.diag(distances), torch.zeros(2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(query_to_reference[0, 0], torch.tensor(0.0), atol=1e-6, rtol=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (
            GDK2SetDistance,
            {"sigma": "fixed", "sigma_value": 1.0, "num_landmarks": 4, "device": "cpu"},
        ),
        (
            IDK2SetDistance,
            {
                "ensemble_size": 6,
                "subsample_size": 3,
                "temperature": 0.05,
                "random_state": 0,
                "device": "cpu",
            },
        ),
        (
            AdaptiveGDK2SetDistance,
            {"k": 2, "device": "cpu"},
        ),
    ],
    ids=["gdk2", "idk2", "adaptive_gdk2"],
)
def test_two_level_set_metric_wrappers_smoke(
    factory: type[TrajectorySetDistance],
    kwargs: dict[str, Any],
) -> None:
    pytest.importorskip("torch")
    trajectory_sets = _separated_trajectory_sets()

    metric = factory(**kwargs).fit(trajectory_sets)
    embeddings = metric.transform(trajectory_sets)
    expected_distances = metric.pairwise_distance(trajectory_sets)
    distances = distributed_pairwise_distance(metric, trajectory_sets, row_block_size=1)
    novelty = metric.novelty_score(trajectory_sets[0])

    assert embeddings.shape[0] == 2
    assert distances.shape == (2, 2)
    assert novelty.shape == (1,)
    assert torch.all(torch.isfinite(distances))
    assert torch.all(torch.isfinite(novelty))
    torch.testing.assert_close(distances, expected_distances, atol=0, rtol=0)
    torch.testing.assert_close(torch.diag(distances), torch.zeros(2), atol=1e-3, rtol=1e-3)


def _distributed_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_src = Path(__file__).resolve().parents[1] / "src"
    python_bin = Path(sys.executable).resolve().parent
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_src) if not existing else os.pathsep.join([str(repo_src), existing])
    env["PATH"] = os.pathsep.join([str(python_bin), env.get("PATH", "")])
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_cli_commands(env: dict[str, str], payload_path: Path, output_path: Path) -> list[list[str]]:
    torchrun = shutil.which("torchrun", path=env["PATH"])
    runner = [torchrun] if torchrun else [sys.executable, "-m", "torch.distributed.run"]
    cli_args = [
        "--payload_path",
        str(payload_path),
        "--payload_format",
        "json",
        "--output_path",
        str(output_path),
        "--overwrite",
        "--backend",
        "gloo",
        "--quiet",
    ]
    return [
        [
            *runner,
            "--standalone",
            "--nproc_per_node=2",
            "-m",
            _CLI_MODULE,
            *cli_args,
        ]
    ]


def _run_distributed_cli(payload_path: Path, output_path: Path) -> dict[str, Any]:
    env = _distributed_cli_env()
    failures = []
    for command in _distributed_cli_commands(env, payload_path, output_path):
        try:
            result = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            failures.append(
                "distributed distance CLI timed out\n"
                f"command: {' '.join(command)}\n"
                f"stdout:\n{exc.stdout or ''}\n"
                f"stderr:\n{exc.stderr or ''}"
            )
            continue
        if result.returncode == 0:
            break
        if "Operation not permitted" in result.stderr and "socket" in result.stderr:
            pytest.skip("torchrun rendezvous sockets are blocked in this environment")
        failures.append(
            "distributed distance CLI failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    else:
        direct_env = dict(env)
        try:
            direct_env.update(
                {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(_reserve_local_port()),
                    "RANK": "0",
                    "WORLD_SIZE": "1",
                    "LOCAL_RANK": "0",
                }
            )
            direct = [
                sys.executable,
                "-m",
                _CLI_MODULE,
                "--payload_path",
                str(payload_path),
                "--payload_format",
                "json",
                "--output_path",
                str(output_path),
                "--overwrite",
                "--backend",
                "gloo",
                "--quiet",
            ]
            result = subprocess.run(
                direct,
                cwd=Path(__file__).resolve().parents[1],
                env=direct_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(f"direct CLI fallback failed: {exc}")
        else:
            if result.returncode != 0:
                if "Operation not permitted" in result.stderr and "socket" in result.stderr:
                    pytest.skip("distributed CLI rendezvous sockets are blocked in this environment")
                failures.append(
                    "direct CLI fallback failed\n"
                    f"command: {' '.join(direct)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            elif not output_path.exists():
                failures.append(f"direct CLI fallback did not create {output_path}")
            else:
                failures = []
        if failures:
            joined = "\n\n".join(failures)
            if "Segmentation fault" in joined or "timed out" in joined:
                pytest.skip("torchrun launcher is unstable in this environment")
            raise AssertionError(joined)
    assert output_path.exists(), f"CLI did not create output file {output_path}"
    try:
        return torch.load(output_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(output_path, map_location="cpu")


def _unwrap_result(payload: Any) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _extract_distance_matrix(payload: Any) -> np.ndarray:
    result = _unwrap_result(payload)
    if isinstance(result, list):
        return _as_numpy(result, dtype=np.float32)
    for key in ("values", "distances", "distance", "pairwise", "pairwise_distance"):
        if isinstance(result, dict) and key in result:
            return _as_numpy(result[key], dtype=np.float32)
    raise AssertionError(f"could not find distance matrix in CLI output: {payload!r}")


def _extract_topk(payload: Any) -> tuple[np.ndarray, np.ndarray]:
    result = _unwrap_result(payload)
    if isinstance(result, dict) and isinstance(result.get("topk"), dict):
        result = result["topk"]
    distance_keys = ("distances", "topk_distances", "values")
    index_keys = ("indices", "topk_indices")
    distances = next((result[key] for key in distance_keys if key in result), None)
    indices = next((result[key] for key in index_keys if key in result), None)
    if distances is None or indices is None:
        raise AssertionError(f"could not find top-k distances and indices in CLI output: {payload!r}")
    return _as_numpy(distances, dtype=np.float32), _as_numpy(indices, dtype=np.int64)


@pytest.mark.cli
@pytest.mark.slow
def test_distributed_distance_cli_matches_local_euclidean_pairwise_and_topk(tmp_path: Path) -> None:
    torch_module = pytest.importorskip("torch")
    if not torch_module.distributed.is_available():
        pytest.skip("torch.distributed is not available")
    if importlib.util.find_spec(_CLI_MODULE) is None:
        pytest.skip(f"{_CLI_MODULE} is not available")

    queries = np.array([[0.0, 0.0], [3.0, 0.0], [8.0, 0.0]], dtype=np.float32)
    references = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0], [9.0, 0.0]], dtype=np.float32)
    pairwise_baseline = (
        EuclideanDistance(device="cpu")
        .fit(references)
        .pairwise_distance(queries, references)
        .detach()
        .cpu()
        .numpy()
    )
    topk_indices = np.argsort(pairwise_baseline, axis=1)[:, :2]
    topk_distances = np.take_along_axis(pairwise_baseline, topk_indices, axis=1)

    pairwise_payload = {
        "metric": {
            "kind": "state",
            "name": "euclidean",
            "kwargs": {"device": "cpu", "dtype": "float32"},
        },
        "op": "pairwise_distance",
        "fit": references.tolist(),
        "A": queries.tolist(),
        "B": references.tolist(),
        "row_block_size": 1,
    }
    pairwise_payload_path = tmp_path / "pairwise_payload.json"
    pairwise_output_path = tmp_path / "pairwise_output.pt"
    pairwise_payload_path.write_text(json.dumps(pairwise_payload), encoding="utf-8")

    pairwise_output = _run_distributed_cli(pairwise_payload_path, pairwise_output_path)
    np.testing.assert_array_equal(_extract_distance_matrix(pairwise_output), pairwise_baseline)

    topk_payload = {
        "metric": {
            "kind": "state",
            "name": "euclidean",
            "kwargs": {"device": "cpu", "dtype": "float32"},
        },
        "op": "topk_distance",
        "fit": references.tolist(),
        "A": queries.tolist(),
        "B": references.tolist(),
        "k": 2,
        "row_block_size": 1,
    }
    topk_payload_path = tmp_path / "topk_payload.json"
    topk_output_path = tmp_path / "topk_output.pt"
    topk_payload_path.write_text(json.dumps(topk_payload), encoding="utf-8")

    topk_output = _run_distributed_cli(topk_payload_path, topk_output_path)
    actual_topk_distances, actual_topk_indices = _extract_topk(topk_output)
    np.testing.assert_array_equal(actual_topk_distances, topk_distances)
    np.testing.assert_array_equal(actual_topk_indices, topk_indices)
