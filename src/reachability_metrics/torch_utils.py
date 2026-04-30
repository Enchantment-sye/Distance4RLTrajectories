"""Torch-first helpers for metrics and trajectory containers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def require_torch() -> Any:
    """Import torch or raise a clear installation error."""
    try:
        import torch

        return torch
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("Torch is required for reachability-metrics torch-first APIs") from exc


def torch_dtype(dtype: str | Any = "float32") -> Any:
    """Resolve a dtype string or torch dtype."""
    torch = require_torch()
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key in {"float", "float32", "torch.float32"}:
        return torch.float32
    if key in {"float64", "double", "torch.float64"}:
        return torch.float64
    if key in {"float16", "half", "torch.float16"}:
        return torch.float16
    if key in {"bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if key in {"long", "int64", "torch.int64"}:
        return torch.long
    if key in {"int", "int32", "torch.int32"}:
        return torch.int32
    if key in {"bool", "torch.bool"}:
        return torch.bool
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def resolve_torch_device(device: str | Any = "auto") -> Any:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    torch = require_torch()
    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def as_tensor(
    values: Any,
    *,
    dtype: str | Any = "float32",
    device: str | Any = "auto",
    copy: bool = False,
) -> Any:
    """Convert arrays, tensors, or objects with ``states`` to a torch tensor."""
    torch = require_torch()
    if hasattr(values, "states"):
        values = values.states
    dev = resolve_torch_device(device)
    dt = torch_dtype(dtype)
    if isinstance(values, torch.Tensor):
        out = values.to(device=dev, dtype=dt)
        return out.clone() if copy else out
    out = torch.as_tensor(values, dtype=dt, device=dev)
    return out.clone() if copy else out


def as_2d_tensor(
    values: Any,
    *,
    dtype: str | Any = "float32",
    device: str | Any = "auto",
    name: str = "array",
) -> Any:
    """Coerce a state or state batch to a ``(N, D)`` tensor."""
    x = as_tensor(values, dtype=dtype, device=device)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (D,) or (N, D), got {tuple(x.shape)}")
    return x


def as_trajectory_tensor(
    values: Any,
    *,
    dtype: str | Any = "float32",
    device: str | Any = "auto",
) -> Any:
    """Coerce one trajectory-like object to a ``(T, D)`` tensor."""
    x = as_tensor(values, dtype=dtype, device=device)
    if x.ndim != 2:
        raise ValueError(f"trajectory must have shape (T, D), got {tuple(x.shape)}")
    return x


def as_trajectory_tensor_list(
    values: Any,
    *,
    dtype: str | Any = "float32",
    device: str | Any = "auto",
) -> list[Any]:
    """Coerce trajectory inputs to a list of torch tensors."""
    torch = require_torch()
    if values is None:
        raise ValueError("trajectory input cannot be None")
    if hasattr(values, "trajectories"):
        return [as_trajectory_tensor(traj, dtype=dtype, device=device) for traj in values.trajectories]
    if hasattr(values, "states"):
        return [as_trajectory_tensor(values, dtype=dtype, device=device)]
    if isinstance(values, torch.Tensor):
        if values.ndim == 2:
            return [as_trajectory_tensor(values, dtype=dtype, device=device)]
        if values.ndim == 3:
            return [as_trajectory_tensor(values[i], dtype=dtype, device=device) for i in range(values.shape[0])]
        raise ValueError(f"trajectory tensor must be 2D or 3D, got {tuple(values.shape)}")
    if isinstance(values, np.ndarray):
        arr = np.asarray(values)
        if arr.ndim == 2:
            return [as_trajectory_tensor(arr, dtype=dtype, device=device)]
        if arr.ndim == 3:
            return [as_trajectory_tensor(arr[i], dtype=dtype, device=device) for i in range(arr.shape[0])]
        raise ValueError(f"trajectory ndarray must be 2D or 3D, got {arr.shape}")
    if isinstance(values, Iterable):
        return [as_trajectory_tensor(item, dtype=dtype, device=device) for item in list(values)]
    raise TypeError(f"Unsupported trajectory input type: {type(values)!r}")


def resolve_output_format(output_format: str | None = None, *, return_numpy: bool = False) -> str:
    """Resolve the explicit output format, preserving ``return_numpy`` as a shim."""
    if output_format is None:
        return "numpy" if bool(return_numpy) else "torch"
    key = str(output_format).lower()
    if key in {"torch", "tensor"}:
        return "torch"
    if key in {"numpy", "np", "array"}:
        return "numpy"
    raise ValueError("output_format must be 'torch' or 'numpy'")


def maybe_numpy(
    values: Any,
    return_numpy: bool = False,
    *,
    output_format: str | None = None,
) -> Any:
    """Return torch tensors by default, or recursively convert to NumPy."""
    torch = require_torch()
    if resolve_output_format(output_format, return_numpy=return_numpy) == "torch":
        return values
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if isinstance(values, tuple):
        return tuple(maybe_numpy(v, True) for v in values)
    if isinstance(values, list):
        return [maybe_numpy(v, True) for v in values]
    return values


def cpu_numpy(values: Any) -> np.ndarray:
    """Detach a torch tensor and move it to NumPy."""
    torch = require_torch()
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def block_slices(n: int, block_size: int) -> Iterable[tuple[int, int]]:
    """Yield inclusive-exclusive slices for blockwise computation."""
    size = max(int(block_size), 1)
    for start in range(0, int(n), size):
        yield start, min(start + size, int(n))


def pairwise_sqeuclidean(x: Any, y: Any) -> Any:
    """Pairwise squared Euclidean distances for two 2D tensors."""
    torch = require_torch()
    x = as_2d_tensor(x, dtype=getattr(x, "dtype", "float32"), device=getattr(x, "device", "auto"), name="x")
    y = as_2d_tensor(y, dtype=x.dtype, device=x.device, name="y")
    x_sq = torch.sum(x * x, dim=1, keepdim=True)
    y_sq = torch.sum(y * y, dim=1, keepdim=True).T
    return torch.clamp(x_sq + y_sq - 2.0 * (x @ y.T), min=0.0)


def pairwise_euclidean(x: Any, y: Any, eps: float = 0.0) -> Any:
    """Pairwise Euclidean distances."""
    torch = require_torch()
    return torch.sqrt(torch.clamp(pairwise_sqeuclidean(x, y), min=float(eps)))


def torch_softmin(distances: Any, tau: float = 1.0, dim: int = -1) -> Any:
    """Differentiable soft minimum."""
    torch = require_torch()
    d = distances
    tau_value = max(float(tau), 1e-12)
    return -tau_value * torch.logsumexp(-d / tau_value, dim=dim)


def torch_resample_trajectory(traj: Any, target_length: int) -> Any:
    """Linearly resample a trajectory tensor to a fixed length."""
    import torch.nn.functional as F

    values = as_trajectory_tensor(traj, dtype=getattr(traj, "dtype", "float32"), device=getattr(traj, "device", "auto"))
    target = int(target_length)
    if target <= 0:
        raise ValueError("target_length must be positive")
    if values.shape[0] == target:
        return values.clone()
    if values.shape[0] == 1:
        return values.repeat(target, 1)
    x = values.T.unsqueeze(0)
    out = F.interpolate(x, size=target, mode="linear", align_corners=True)
    return out.squeeze(0).T.contiguous()


def cosine_distance_matrix(x: Any, y: Any) -> Any:
    """Pairwise cosine distance."""
    torch = require_torch()
    x = as_2d_tensor(x, dtype=getattr(x, "dtype", "float32"), device=getattr(x, "device", "auto"), name="x")
    y = as_2d_tensor(y, dtype=x.dtype, device=x.device, name="y")
    x_norm = torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    y_norm = torch.linalg.norm(y, dim=1, keepdim=True).clamp_min(1e-12)
    sim = (x @ y.T) / (x_norm @ y_norm.T)
    return 1.0 - torch.clamp(sim, -1.0, 1.0)
