"""Torch-first state preprocessing and optional temporal/window features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator

from reachability_metrics.base import TensorOutputMixin
from reachability_metrics.torch_utils import (
    as_2d_tensor,
    as_trajectory_tensor,
    as_trajectory_tensor_list,
    require_torch,
    resolve_torch_device,
    torch_dtype,
)


@dataclass
class _Stats:
    center: Any
    scale: Any
    min_: Any
    max_: Any


class StatePreprocessor(TensorOutputMixin, BaseEstimator):
    """Optional normalization, temporal features, and sliding windows for torch states."""

    def __init__(
        self,
        normalize: bool = True,
        normalization: str = "standard",
        temporal_feature: str | None = None,
        temporal_dim: int = 8,
        sliding_window: int | None = None,
        flatten_window: bool = True,
        padding: str = "repeat_first",
        eps: float = 1e-8,
        device: str = "auto",
        dtype: str = "float32",
        return_numpy: bool = False,
        output_format: str | None = None,
    ) -> None:
        self.normalize = normalize
        self.normalization = normalization
        self.temporal_feature = temporal_feature
        self.temporal_dim = temporal_dim
        self.sliding_window = sliding_window
        self.flatten_window = flatten_window
        self.padding = padding
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self._set_output_options(return_numpy=return_numpy, output_format=output_format)

    def fit(self, trajectories: Any, y: Any = None) -> "StatePreprocessor":
        """Fit normalization statistics from states or trajectories."""
        torch = require_torch()
        if hasattr(trajectories, "trajectories"):
            values = trajectories.states().to(device=resolve_torch_device(self.device), dtype=torch_dtype(self.dtype))
        else:
            try:
                values = as_2d_tensor(trajectories, dtype=self.dtype, device=self.device, name="states")
            except Exception:
                values = torch.cat(as_trajectory_tensor_list(trajectories, dtype=self.dtype, device=self.device), dim=0)
        center = torch.mean(values, dim=0)
        std = torch.std(values, dim=0, unbiased=False)
        min_ = torch.min(values, dim=0).values
        max_ = torch.max(values, dim=0).values
        median = torch.quantile(values, 0.5, dim=0)
        q75 = torch.quantile(values, 0.75, dim=0)
        q25 = torch.quantile(values, 0.25, dim=0)
        eps = float(self.eps)
        self.stats_ = _Stats(center=center, scale=std.clamp_min(eps), min_=min_, max_=max_)
        self.robust_center_ = median
        self.robust_scale_ = (q75 - q25).clamp_min(eps)
        self.n_features_in_ = int(values.shape[1])
        return self

    def _normalize(self, values: Any) -> Any:
        if not bool(self.normalize):
            return values
        if not hasattr(self, "stats_"):
            raise RuntimeError("StatePreprocessor must be fitted before transform")
        mode = str(self.normalization or "none").lower()
        if mode == "none":
            return values
        if mode == "standard":
            return (values - self.stats_.center) / self.stats_.scale
        if mode == "minmax":
            return (values - self.stats_.min_) / (self.stats_.max_ - self.stats_.min_).clamp_min(float(self.eps))
        if mode == "robust":
            return (values - self.robust_center_) / self.robust_scale_
        raise ValueError(f"Unsupported normalization: {self.normalization}")

    def _temporal_features(self, timesteps: Any, total: int) -> Any:
        torch = require_torch()
        mode = self.temporal_feature
        device = resolve_torch_device(self.device)
        dtype = torch_dtype(self.dtype)
        if mode is None or str(mode).lower() in {"none", ""}:
            return torch.empty((timesteps.shape[0], 0), dtype=dtype, device=device)
        key = str(mode).lower()
        t = timesteps.to(device=device, dtype=dtype).reshape(-1)
        norm_t = t / max(float(total - 1), 1.0)
        if key in {"learned_index", "normalized_index", "index"}:
            return norm_t[:, None]
        dim = max(int(self.temporal_dim), 2)
        if dim % 2:
            dim += 1
        frequencies = torch.exp(
            torch.arange(0, dim, 2, dtype=dtype, device=device) * (-torch.log(torch.tensor(10000.0, dtype=dtype, device=device)) / dim)
        )
        angles = t[:, None] * frequencies[None, :]
        if key == "sinusoidal":
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if key == "rope":
            return torch.stack([torch.cos(angles), torch.sin(angles)], dim=2).reshape(t.shape[0], dim)
        raise ValueError(f"Unsupported temporal_feature: {self.temporal_feature}")

    def transform_states_tensor(self, states: Any, timesteps: Any | None = None):
        """Transform a state batch as a torch tensor."""
        torch = require_torch()
        values = as_2d_tensor(states, dtype=self.dtype, device=self.device, name="states")
        transformed = self._normalize(values)
        if timesteps is None:
            timesteps = torch.arange(values.shape[0], dtype=torch.long, device=values.device)
        else:
            timesteps = torch.as_tensor(timesteps, dtype=torch.long, device=values.device)
        temporal = self._temporal_features(timesteps, total=values.shape[0])
        if temporal.shape[1] > 0:
            transformed = torch.cat([transformed, temporal], dim=1)
        return transformed

    def transform_states(self, states: Any, timesteps: Any | None = None):
        """Transform a state batch and return a torch tensor by default."""
        return self._return(self.transform_states_tensor(states, timesteps=timesteps))

    def _window_trajectory(self, states: Any) -> Any:
        torch = require_torch()
        w = self.sliding_window
        if w is None:
            return states
        window = int(w)
        if window <= 1:
            return states
        if self.padding == "drop":
            if states.shape[0] < window:
                empty = torch.empty((0, window * states.shape[1]), dtype=states.dtype, device=states.device)
                return empty
            chunks = [states[i - window + 1 : i + 1] for i in range(window - 1, states.shape[0])]
        else:
            chunks = []
            for i in range(states.shape[0]):
                start = i - window + 1
                if start < 0:
                    pad_count = -start
                    if self.padding == "repeat_first":
                        pad = states[:1].repeat(pad_count, 1)
                    elif self.padding == "zero":
                        pad = torch.zeros((pad_count, states.shape[1]), dtype=states.dtype, device=states.device)
                    else:
                        raise ValueError(f"Unsupported padding: {self.padding}")
                    chunks.append(torch.cat([pad, states[0 : i + 1]], dim=0))
                else:
                    chunks.append(states[start : i + 1])
        stacked = torch.stack(chunks, dim=0)
        if self.flatten_window:
            return stacked.reshape(stacked.shape[0], -1)
        return stacked

    def transform_trajectory(self, traj: Any):
        """Transform one trajectory and return a torch tensor by default."""
        states = as_trajectory_tensor(traj, dtype=self.dtype, device=self.device)
        timesteps = getattr(traj, "timesteps", None)
        transformed = self.transform_states_tensor(states, timesteps=timesteps)
        return self._return(self._window_trajectory(transformed))

    def transform(self, trajectories: Any):
        """Transform states or trajectories."""
        try:
            result = self.transform_states(trajectories)
            return result
        except Exception:
            result = [self.transform_trajectory(traj) for traj in as_trajectory_tensor_list(trajectories, dtype=self.dtype, device=self.device)]
            return self._return(result)

    def fit_transform(self, trajectories: Any, y: Any = None) -> Any:
        """Fit and transform."""
        return self.fit(trajectories).transform(trajectories)
