"""State preprocessing and optional temporal/window features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from reachability_metrics.utils import as_2d_array, as_trajectory_list


@dataclass
class _Stats:
    center: np.ndarray
    scale: np.ndarray
    min_: np.ndarray
    max_: np.ndarray


class StatePreprocessor(BaseEstimator):
    """Optional normalization, temporal features, and sliding windows for states."""

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
    ) -> None:
        self.normalize = normalize
        self.normalization = normalization
        self.temporal_feature = temporal_feature
        self.temporal_dim = temporal_dim
        self.sliding_window = sliding_window
        self.flatten_window = flatten_window
        self.padding = padding
        self.eps = eps

    def fit(self, trajectories: Any, y: Any = None) -> "StatePreprocessor":
        """Fit normalization statistics from states or trajectories."""
        if isinstance(trajectories, np.ndarray) and trajectories.ndim <= 2:
            values = as_2d_array(trajectories, dtype=np.float64, name="states")
        else:
            values = np.concatenate(as_trajectory_list(trajectories, dtype=np.float64), axis=0)
        center = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        min_ = np.min(values, axis=0)
        max_ = np.max(values, axis=0)
        median = np.median(values, axis=0)
        q75, q25 = np.percentile(values, [75.0, 25.0], axis=0)
        self.stats_ = _Stats(center=center, scale=np.maximum(std, self.eps), min_=min_, max_=max_)
        self.robust_center_ = median
        self.robust_scale_ = np.maximum(q75 - q25, self.eps)
        self.n_features_in_ = int(values.shape[1])
        return self

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if not bool(self.normalize):
            return values.astype(np.float32)
        if not hasattr(self, "stats_"):
            raise RuntimeError("StatePreprocessor must be fitted before transform")
        mode = str(self.normalization or "none").lower()
        if mode == "none":
            return values.astype(np.float32)
        if mode == "standard":
            return ((values - self.stats_.center) / self.stats_.scale).astype(np.float32)
        if mode == "minmax":
            return ((values - self.stats_.min_) / np.maximum(self.stats_.max_ - self.stats_.min_, self.eps)).astype(np.float32)
        if mode == "robust":
            return ((values - self.robust_center_) / self.robust_scale_).astype(np.float32)
        raise ValueError(f"Unsupported normalization: {self.normalization}")

    def _temporal_features(self, timesteps: np.ndarray, total: int) -> np.ndarray:
        mode = self.temporal_feature
        if mode is None or str(mode).lower() in {"none", ""}:
            return np.empty((timesteps.shape[0], 0), dtype=np.float32)
        key = str(mode).lower()
        t = np.asarray(timesteps, dtype=np.float32).reshape(-1)
        norm_t = t / max(float(total - 1), 1.0)
        if key in {"learned_index", "normalized_index", "index"}:
            return norm_t[:, None].astype(np.float32)
        dim = max(int(self.temporal_dim), 2)
        if dim % 2:
            dim += 1
        frequencies = np.exp(np.arange(0, dim, 2, dtype=np.float32) * (-np.log(10000.0) / dim))
        angles = t[:, None] * frequencies[None, :]
        if key == "sinusoidal":
            return np.concatenate([np.sin(angles), np.cos(angles)], axis=1).astype(np.float32)
        if key == "rope":
            return np.stack([np.cos(angles), np.sin(angles)], axis=2).reshape(t.shape[0], dim).astype(np.float32)
        raise ValueError(f"Unsupported temporal_feature: {self.temporal_feature}")

    def transform_states(self, states: Any, timesteps: np.ndarray | None = None) -> np.ndarray:
        """Transform a state batch."""
        values = as_2d_array(states, dtype=np.float64, name="states")
        transformed = self._normalize(values)
        if timesteps is None:
            timesteps = np.arange(values.shape[0], dtype=np.int64)
        temporal = self._temporal_features(np.asarray(timesteps), total=values.shape[0])
        if temporal.shape[1] > 0:
            transformed = np.concatenate([transformed, temporal], axis=1).astype(np.float32)
        return transformed

    def _window_trajectory(self, states: np.ndarray) -> np.ndarray:
        w = self.sliding_window
        if w is None:
            return states
        window = int(w)
        if window <= 1:
            return states
        if self.padding == "drop":
            if states.shape[0] < window:
                return np.empty((0, window * states.shape[1]), dtype=np.float32)
            chunks = [states[i - window + 1 : i + 1] for i in range(window - 1, states.shape[0])]
        else:
            chunks = []
            for i in range(states.shape[0]):
                start = i - window + 1
                if start < 0:
                    pad_count = -start
                    if self.padding == "repeat_first":
                        pad = np.repeat(states[:1], pad_count, axis=0)
                    elif self.padding == "zero":
                        pad = np.zeros((pad_count, states.shape[1]), dtype=states.dtype)
                    else:
                        raise ValueError(f"Unsupported padding: {self.padding}")
                    chunks.append(np.concatenate([pad, states[0 : i + 1]], axis=0))
                else:
                    chunks.append(states[start : i + 1])
        stacked = np.stack(chunks, axis=0).astype(np.float32)
        if self.flatten_window:
            return stacked.reshape(stacked.shape[0], -1)
        return stacked

    def transform_trajectory(self, traj: Any) -> np.ndarray:
        """Transform a trajectory."""
        states = np.asarray(traj.states if hasattr(traj, "states") else traj, dtype=np.float64)
        timesteps = getattr(traj, "timesteps", None)
        transformed = self.transform_states(states, timesteps=timesteps)
        return self._window_trajectory(transformed)

    def transform(self, trajectories: Any) -> list[np.ndarray] | np.ndarray:
        """Transform states or trajectories."""
        if isinstance(trajectories, np.ndarray) and trajectories.ndim <= 2:
            return self.transform_states(trajectories)
        return [self.transform_trajectory(traj) for traj in as_trajectory_list(trajectories)]

    def fit_transform(self, trajectories: Any, y: Any = None) -> Any:
        """Fit and transform."""
        return self.fit(trajectories).transform(trajectories)

