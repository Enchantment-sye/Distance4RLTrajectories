"""Shared utilities used across metrics and experiments."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from typing import Any

import numpy as np


def ensure_dir(path: str) -> None:
    """Create a directory if it is non-empty and missing."""
    if path:
        os.makedirs(path, exist_ok=True)


def dataset_slug(dataset_id: str) -> str:
    """Convert a dataset identifier into a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(dataset_id)).strip("_").lower()


def payload_hash(payload: dict[str, Any]) -> str:
    """Stable short hash for cache keys."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]


def as_2d_array(values: Any, *, dtype: np.dtype | type = np.float64, name: str = "array") -> np.ndarray:
    """Coerce a state or state batch to shape ``(N, D)``."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (D,) or (N, D), got {arr.shape}")
    return arr


def as_trajectory_array(values: Any, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    """Coerce a trajectory-like object to a 2D state sequence."""
    if hasattr(values, "states"):
        values = values.states
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"trajectory must have shape (T, D), got {arr.shape}")
    return arr


def as_trajectory_list(values: Any, *, dtype: np.dtype | type = np.float64) -> list[np.ndarray]:
    """Coerce trajectory inputs to a list of ``(T, D)`` arrays."""
    if values is None:
        raise ValueError("trajectory input cannot be None")
    if hasattr(values, "trajectories"):
        return [as_trajectory_array(traj, dtype=dtype) for traj in values.trajectories]
    if hasattr(values, "states"):
        return [as_trajectory_array(values, dtype=dtype)]
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=dtype)
        if arr.ndim == 2:
            return [arr]
        if arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        raise ValueError(f"trajectory ndarray must be 2D or 3D, got {arr.shape}")
    if isinstance(values, Iterable):
        return [as_trajectory_array(item, dtype=dtype) for item in list(values)]
    raise TypeError(f"Unsupported trajectory input type: {type(values)!r}")


def pairwise_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Memory-conscious squared Euclidean distance for two 2D arrays."""
    x = as_2d_array(x, dtype=np.float64, name="x")
    y = as_2d_array(y, dtype=np.float64, name="y")
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    y_sq = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)


def block_slices(n: int, block_size: int) -> Iterable[tuple[int, int]]:
    """Yield ``(start, end)`` slices."""
    size = max(int(block_size), 1)
    for start in range(0, int(n), size):
        yield start, min(start + size, int(n))


def resolve_device(device: str) -> str:
    """Resolve ``auto`` to cuda if torch is importable and CUDA is available."""
    if str(device) != "auto":
        return str(device)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def safe_sqrt(values: np.ndarray) -> np.ndarray:
    """Square root with negative numerical noise clipped away."""
    return np.sqrt(np.maximum(values, 0.0))


def finite_mean(values: Iterable[float], default: float = 0.0) -> float:
    """Mean of finite values."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def softmin(distances: np.ndarray, tau: float, axis: int = -1) -> np.ndarray:
    """Differentiable soft minimum implemented in NumPy."""
    tau_value = max(float(tau), 1e-12)
    scaled = -np.asarray(distances, dtype=np.float64) / tau_value
    max_scaled = np.max(scaled, axis=axis, keepdims=True)
    return -tau_value * (
        np.log(np.sum(np.exp(scaled - max_scaled), axis=axis)) + np.squeeze(max_scaled, axis=axis)
    )


def resample_trajectory(traj: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a trajectory to a fixed length."""
    values = as_trajectory_array(traj)
    target = int(target_length)
    if target <= 0:
        raise ValueError("target_length must be positive")
    if values.shape[0] == target:
        return values.copy()
    if values.shape[0] == 1:
        return np.repeat(values, target, axis=0)
    old_t = np.linspace(0.0, 1.0, values.shape[0])
    new_t = np.linspace(0.0, 1.0, target)
    out = np.empty((target, values.shape[1]), dtype=np.float64)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(new_t, old_t, values[:, dim])
    return out


def cosine_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance."""
    x = as_2d_array(x, dtype=np.float64, name="x")
    y = as_2d_array(y, dtype=np.float64, name="y")
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = np.linalg.norm(y, axis=1, keepdims=True)
    sim = (x @ y.T) / np.maximum(x_norm * y_norm.T, 1e-12)
    return 1.0 - np.clip(sim, -1.0, 1.0)
